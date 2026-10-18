#!/usr/bin/env python3
"""Compare the integration options with the S1 baseline ASR, using the same model/PCM.

The default reference is S1 commit 0a10df94440fa997456620bded5d107eb3bad0d2, pinned
so committing the integration changes cannot silently turn this into a self-comparison.

No device access or training. Only Python's standard library and a host C compiler are used.
  python3 tools/asr/integration_test.py --fixture-root /path/to/kana_asr --output dist/asr-host.json
  python3 tools/asr/integration_test.py --fixture-root /path/to/kana_asr --all-recordings

The default speech fixtures are existing recordings, not copied into this repository. Synthetic
short/empty/noise/trim/restart cases are always included. --all-recordings checks raw-match
decisions for every recording in three directories; frontend/arena checks use the compact suite.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
F = C.POINTER(C.c_float)
P = C.POINTER(C.c_int16)
ALLOC = C.CFUNCTYPE(C.c_void_p, C.c_size_t, C.c_int)
# The firmware's generated kana table is authoritative, including long-vowel spelling.
PHRASES = re.findall(r'"([^"]+)"', (ROOT / 'firmware/src/phrases.h').read_text().split('kPhraseKana[] = {')[1].split('};')[0])
PHRASE_ARRAY = (C.c_char_p * len(PHRASES))(*(p.encode() for p in PHRASES))


class Core:
    def __init__(self, library, model, trial):
        self.lib = C.CDLL(str(library))
        self.trial = trial
        self.model = C.create_string_buffer(model)
        self.allocations = []
        self.fast_bytes = 0
        self.slow_bytes = 0
        libc = C.CDLL(None)
        libc.malloc.argtypes = [C.c_size_t]
        libc.malloc.restype = C.c_void_p
        libc.free.argtypes = [C.c_void_p]
        self.libc = libc

        def alloc(n, fast):
            p = libc.malloc(n)
            self.allocations.append(p)
            if fast:
                self.fast_bytes += n
            else:
                self.slow_bytes += n
            return p

        self.alloc = ALLOC(alloc)
        self.lib.asr_init.argtypes = [C.c_void_p, C.c_size_t, C.c_int, ALLOC]
        self.lib.asr_init.restype = C.c_int
        self.lib.asr_recognize.argtypes = [P, C.c_int, C.c_char_p, C.c_int, F]
        self.lib.asr_recognize.restype = C.c_int
        self.lib.asr_features.argtypes = [P, C.c_int, C.POINTER(F), C.POINTER(C.c_int)]
        self.lib.asr_logits.argtypes = self.lib.asr_features.argtypes
        self.lib.asr_match.argtypes = [C.POINTER(C.c_char_p), C.c_int, F]
        self.lib.asr_match.restype = C.c_int
        self.lib.asr_error.restype = C.c_char_p
        if trial:
            self.lib.asr_set_options.argtypes = [C.c_uint]
            self.lib.asr_frontend_begin.argtypes = [P]
            self.lib.asr_frontend_push.argtypes = [C.c_int]
            self.lib.asr_frontend_push.restype = C.c_int
            self.lib.asr_scratch_bytes.argtypes = [C.c_void_p, C.c_size_t, C.c_int]
            self.lib.asr_scratch_bytes.restype = C.c_size_t
            self.lib.asr_init_with_scratch.argtypes = [C.c_void_p, C.c_size_t, C.c_int, ALLOC, C.c_void_p, C.c_size_t]
        self.initialize()

    def check(self, rc):
        if rc < 0:
            raise AssertionError(self.lib.asr_error().decode())
        return rc

    def initialize(self, arena=False):
        for p in self.allocations:
            self.libc.free(p)
        self.allocations.clear()
        self.fast_bytes = self.slow_bytes = 0
        if arena:
            self.arena_size = self.lib.asr_scratch_bytes(self.model, len(self.model) - 1, 112000)
            assert self.arena_size > 0
            self.arena = C.create_string_buffer(self.arena_size + 32)
            C.memset(self.arena, 0xCD, len(self.arena))
            self.arena_ptr = C.addressof(self.arena) + 16
            self.check(self.lib.asr_init_with_scratch(self.model, len(self.model) - 1, 112000,
                                                     self.alloc, self.arena_ptr, self.arena_size))
            assert self.slow_bytes == 0
        else:
            self.check(self.lib.asr_init(self.model, len(self.model) - 1, 112000, self.alloc))

    def recognize(self, pcm, n, confidence=True):
        out = C.create_string_buffer(512)
        conf = C.c_float(-1)
        count = self.check(self.lib.asr_recognize(pcm, n, out, len(out), C.byref(conf) if confidence else None))
        gaps = (C.c_float * len(PHRASES))()
        best = self.lib.asr_match(PHRASE_ARRAY, len(PHRASES), gaps)
        second = min(gaps[i] for i in range(len(PHRASES)) if i != best)
        accepted = best if count and best >= 0 and gaps[best] <= 2.0 and second - gaps[best] >= 1.0 else -1
        return {'text': out.value.decode(), 'count': count, 'confidence': conf.value,
                'best': best, 'accepted': accepted, 'gaps': list(gaps)}

    def tensor(self, kind, pcm, n):
        data, frames = F(), C.c_int()
        self.check(getattr(self.lib, 'asr_' + kind)(pcm, n, C.byref(data), C.byref(frames)))
        width = 40 if kind == 'features' else self.lib.asr_vocab_size()
        return frames.value, C.string_at(data, frames.value * width * 4)

    def preload(self, pcm, n, until=None):
        self.lib.asr_frontend_begin(pcm)
        end = n if until is None else until
        # Include very short and non-hop-aligned chunks, as well as the actual 20 ms blocks.
        pos, i = 0, 0
        strides = [1, 111, 320, 517, 160, 320]
        while pos < end:
            pos = min(end, pos + strides[i % len(strides)])
            self.check(self.lib.asr_frontend_push(pos))
            i += 1

    def arena_guard(self):
        assert C.string_at(C.addressof(self.arena), 16) == b'\xcd' * 16
        assert C.string_at(self.arena_ptr + self.arena_size, 16) == b'\xcd' * 16


def pcm_buffer(samples):
    if isinstance(samples, bytes):
        n = len(samples) // 2
        return (C.c_int16 * max(1, n)).from_buffer_copy(samples or b'\0\0'), n
    return (C.c_int16 * max(1, len(samples)))(*samples), len(samples)


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (16000, 2, 1), path
        return w.readframes(min(w.getnframes(), 112000))


def compact_fixtures(root):
    rng = random.Random(42)
    for n in (0, 1, 159, 300, 399, 400, 401, 719, 720, 721, 112000):
        yield f'noise-{n}', pcm_buffer([rng.randrange(-1500, 1501) for _ in range(n)])
    yield 'silence-2s', pcm_buffer([0] * 32000)
    yield 'sweep-2s', pcm_buffer([int(7000 * math.sin(2 * math.pi * (170 + i / 200) * i / 16000)) for i in range(32000)])
    if root:
        names = [
            'recordings/20261017-195401-000.wav',
            'recordings/20261017-201508-004.wav',
            'recordings/20261017-202120-085.wav',
            'recordings_other/20261017-210026-038.wav',
            'recordings_other/20261017-210554-070.wav',
            'recordings_other/20261017-210701-077.wav',
            'recordings_replay/20261017-213232-190.wav',
            'recordings_replay/20261017-221823-604.wav',
        ]
        for name in names:
            path = root / name
            yield name, pcm_buffer(read_wav(path))
        raw = read_wav(root / names[4])
        for pause in (300, 600):
            middle = (len(raw) // 4) * 2
            derived = (raw[:middle] + b'\0' * (pause * 32) + raw[middle:])[:224000]
            yield f'artificial-midpause-{pause}ms', pcm_buffer(derived)


def build(tmp, reference_ref):
    src = ROOT / 'firmware/src'
    ref = subprocess.check_output(['git', 'show', f'{reference_ref}:firmware/src/asr_core.c'], cwd=ROOT)
    (tmp / 'reference.c').write_bytes(ref)
    extension = '.dylib' if platform.system() == 'Darwin' else '.so'
    paths = []
    for name, source in [('reference', tmp / 'reference.c'), ('trial', src / 'asr_core.c')]:
        target = tmp / (name + extension)
        cmd = ['cc', '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
               '-I', str(src), str(source), '-lm', '-o', str(target)]
        subprocess.run(cmd, check=True)
        paths.append(target)
    return paths, hashlib.sha256(ref).hexdigest()


def compare_result(a, b, label, exact_gap):
    assert (a['text'], a['count'], a['best'], a['accepted']) == (b['text'], b['count'], b['best'], b['accepted']), label
    error = max(abs(x - y) for x, y in zip(a['gaps'], b['gaps']))
    assert error <= (0 if exact_gap else 0.001), (label, error)
    return error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture-root', type=Path)
    ap.add_argument('--all-recordings', action='store_true')
    ap.add_argument('--reference-ref', default='0a10df94440fa997456620bded5d107eb3bad0d2',
                    help='ASR comparison source Git ref (default: pinned S1 commit 0a10df94440fa997456620bded5d107eb3bad0d2)')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    started = time.monotonic()
    model = (ROOT / 'models/kana_asr.bin').read_bytes()
    summary = {'reference_ref': args.reference_ref, 'model_sha256': hashlib.sha256(model).hexdigest(),
               'trial_source_sha256': hashlib.sha256((ROOT / 'firmware/src/asr_core.c').read_bytes()).hexdigest(),
               'compact': [], 'all_recordings': [], 'max_raw_gap_difference': 0.0}
    with tempfile.TemporaryDirectory(prefix='zunda-asr-integration-') as tmp:
        libraries, summary['reference_source_sha256'] = build(Path(tmp), args.reference_ref)
        ref, trial = Core(libraries[0], model, False), Core(libraries[1], model, True)
        summary['baseline_slow_scratch_bytes'] = trial.slow_bytes
        summary['baseline_fast_bytes'] = trial.fast_bytes
        fixtures = list(compact_fixtures(args.fixture_root))
        for name, (pcm, n) in fixtures:
            baseline = ref.recognize(pcm, n)
            trial.lib.asr_set_options(0)
            normal = trial.recognize(pcm, n)
            compare_result(baseline, normal, name + ':default', True)
            assert baseline['confidence'] == normal['confidence'], name
            for flag in (1, 2, 3):
                trial.lib.asr_set_options(flag)
                actual = trial.recognize(pcm, n, confidence=False)
                error = compare_result(baseline, actual, f'{name}:flags={flag}', not (flag & 2))
                summary['max_raw_gap_difference'] = max(summary['max_raw_gap_difference'], error)
            # Explicitly requested confidence must still be calculated with the skip flag enabled.
            assert trial.recognize(pcm, n)['confidence'] == baseline['confidence'], name
            trial.lib.asr_set_options(0)
            for kind in ('features', 'logits'):
                expected = ref.tensor(kind, pcm, n)
                trial.preload(pcm, n)
                actual = trial.tensor(kind, pcm, n)
                assert actual == expected, (name, kind, 'frontend tensor mismatch')
                assert trial.lib.asr_frontend_reused_frames() == (0 if n < 400 else 1 + (n - 400) // 160)
                assert trial.tensor(kind, pcm, n) == expected, (name, kind, 'second inference')
                assert trial.lib.asr_frontend_reused_frames() == 0
            # Tail trimming, including a sub-window final input, must not reuse invalid full windows.
            for final_n in sorted({min(n, 159), max(0, n - 3200)}):
                trial.preload(pcm, n)
                assert trial.tensor('logits', pcm, final_n) == ref.tensor('logits', pcm, final_n), (name, 'trim', final_n)
            # An interrupted attempt may be discarded and started again on the same buffer.
            trial.preload(pcm, n, n // 2)
            trial.lib.asr_frontend_reset()
            trial.preload(pcm, n)
            assert trial.tensor('logits', pcm, n) == ref.tensor('logits', pcm, n), (name, 'restart')
            summary['compact'].append({'fixture': name, 'samples': n, 'kana': baseline['text'],
                                       'accepted_phrase': baseline['accepted']})
            print(f'PASS {name} ({n / 16000:.2f}s)', flush=True)

        # One shared allocation gives the same outputs, survives overwrite between users,
        # and stays inside its bounds. No ASR output may be read after release.
        trial.initialize(arena=True)
        assert trial.arena_size == summary['baseline_slow_scratch_bytes']
        summary['shared_scratch_bytes'] = trial.arena_size
        trial.lib.asr_set_options(0)
        for name, (pcm, n) in fixtures:
            trial.lib.asr_release_scratch()
            assert trial.lib.asr_match(PHRASE_ARRAY, len(PHRASES), None) == -1
            C.memset(trial.arena_ptr, 0xA5, trial.arena_size)
            trial.preload(pcm, n)
            compare_result(ref.recognize(pcm, n), trial.recognize(pcm, n), name + ':arena', True)
            trial.arena_guard()

        if args.all_recordings:
            assert args.fixture_root, '--all-recordings requires --fixture-root'
            for directory in ('recordings', 'recordings_other', 'recordings_replay'):
                for path in sorted((args.fixture_root / directory).glob('*.wav')):
                    pcm, n = pcm_buffer(read_wav(path))
                    trial.lib.asr_set_options(0)
                    baseline = trial.recognize(pcm, n)
                    trial.lib.asr_set_options(3)
                    actual = trial.recognize(pcm, n, confidence=False)
                    error = compare_result(baseline, actual, str(path), False)
                    summary['max_raw_gap_difference'] = max(summary['max_raw_gap_difference'], error)
                    summary['all_recordings'].append({'fixture': str(path.relative_to(args.fixture_root)),
                                                      'accepted_phrase': baseline['accepted'], 'max_gap_difference': error})
                print(f'PASS {directory}: all recordings', flush=True)

        # API misuse fails instead of silently reusing stale data.
        pcm, n = fixtures[-1][1]
        trial.lib.asr_frontend_begin(pcm)
        trial.check(trial.lib.asr_frontend_push(n))
        assert trial.lib.asr_frontend_push(-1) < 0
        assert trial.lib.asr_frontend_push(112001) < 0
        trial.lib.asr_frontend_reset()
        assert trial.lib.asr_frontend_push(n) < 0
        assert trial.lib.asr_init_with_scratch(trial.model, len(model), 112000, trial.alloc,
                                               trial.arena_ptr, trial.arena_size - 1) < 0
        assert trial.lib.asr_init_with_scratch(trial.model, len(model), 112000, trial.alloc,
                                               trial.arena_ptr + 1, trial.arena_size) < 0
    summary['elapsed_host_seconds'] = round(time.monotonic() - started, 3)
    summary['status'] = 'PASS'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('compact', 'all_recordings')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
