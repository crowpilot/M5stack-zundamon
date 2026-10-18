#!/usr/bin/env python3
"""Exercise the integration trial's serial protocol and retain raw evidence.

This script emits audible replies in text/replay phases. It never flashes firmware.
Use PlatformIO's Python when pyserial is installed only in that environment.

  python bench.py /dev/cu.usbserial-... dist/text.jsonl --phase text \
      --profiles baseline,last-cache,phrase-cache --rounds 2 --pairs 2
  python bench.py /dev/cu.usbserial-... dist/quick.jsonl --phase text \
      --profiles baseline,tts-parallel --pairs 1 --prompts prompts.json
  python bench.py /dev/cu.usbserial-... dist/asr.jsonl --phase asr \
      --profiles baseline,asr-no-conf,asr-raw --fixture-root /path/to/kana_asr
  python bench.py /dev/cu.usbserial-... dist/hello.jsonl --phase replay \
      --fixture-root /path/to/kana_asr --fixture-names hello
  python bench.py --self-test

Port opening may reset Core2. DTR/RTS are kept false; the driver sends ?status and
waits for ready:true (boot message or status response) before starting trials.
Each operation has an absolute 180-second deadline, with
no shorter timeout caused by an idle serial line. Raw PCM is described by SHA256,
not embedded in the log. ASR/replay uploads each fixture once, then uses it for all
rounds/profiles. --prompts accepts a JSON array of strings. Existing output files
are never overwritten.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import time
import wave

PROMPTS = [
    'おはよう', 'こんにちは', '今日はいい天気ですね',
    '元気ですか', '東京は朝から晴れています', '上司が厳しくて困ってる',
]
ASR_FILES = [
    ('self-short', 'recordings/20261017-195751-000.wav'),
    ('self-normal', 'recordings/20261017-201549-013.wav'),
    ('self-long', 'recordings/20261017-202120-085.wav'),
    ('other-5s', 'recordings_other/20261017-210554-070.wav'),
    ('other-7s', 'recordings_other/20261017-210701-077.wav'),
]
REPLAY_FILES = [
    ('hello', 'recordings/20261017-195751-000.wav'),
    ('genki', 'recordings/20261017-200515-002.wav'),
    ('long', 'recordings/20261017-201549-013.wav'),
]
COMPARE_FIELDS = ('heard', 'kana', 'phrase', 'reply', 'say', 'feel', 'voice', 'pcm_hash')


class ProtocolError(RuntimeError):
    pass


class Log:
    def __init__(self, output):
        self.output = output
        self.context = {}
        self.started = time.monotonic()

    def record(self, kind, **values):
        obj = {'kind': kind, 'elapsed_s': round(time.monotonic() - self.started, 6),
               **self.context, **values}
        line = json.dumps(obj, ensure_ascii=False)
        self.output.write(line + '\n')
        self.output.flush()
        # Raw incoming lines stay in the file; progress and outcomes remain visible immediately.
        if kind not in ('rx', 'tx'):
            print(line, flush=True)
        return obj


class Protocol:
    def __init__(self, ser, log, timeout=180.0):
        self.ser = ser
        self.log = log
        self.timeout = timeout
        self.pending = bytearray()
        self.lines = deque()

    def send(self, command):
        if '\n' in command or '\r' in command:
            raise ValueError('command must be one line')
        self.log.record('tx', command=command)
        self.write_all((command + '\n').encode())

    def write_all(self, payload):
        deadline = time.monotonic() + self.timeout
        offset = 0
        while offset < len(payload):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'serial write timed out ({offset}/{len(payload)} bytes)')
            written = self.ser.write(payload[offset:offset + 4096])
            if written:
                offset += written
        self.ser.flush()

    def next_json(self, deadline, operation, allow_ready=False):
        while time.monotonic() < deadline:
            if self.lines:
                raw = self.lines.popleft().decode('utf-8', errors='replace').strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    self.log.record('diagnostic', line=raw)
                    continue
                self.log.record('rx', raw=raw, data=obj)
                if not isinstance(obj, dict):
                    continue
                if 'error' in obj:
                    raise ProtocolError(f'{operation}: device error: {obj}')
                if obj.get('ready') and obj.get('control') != 'status' and not allow_ready:
                    raise ProtocolError(f'{operation}: unexpected device restart: {obj}')
                return obj, raw
            # read() may return a partial line on its short polling timeout. Preserve it.
            block = self.ser.read(max(1, min(int(self.ser.in_waiting), 8192)))
            if not block:
                continue
            self.pending.extend(block)
            while b'\n' in self.pending:
                line, _, rest = self.pending.partition(b'\n')
                self.lines.append(bytes(line))
                self.pending = bytearray(rest)
            if len(self.pending) > 65536:
                raise ProtocolError(f'{operation}: unterminated serial line exceeds 64 KiB')
        raise TimeoutError(f'{operation}: no matching JSON before {self.timeout:g}s deadline')

    def wait(self, predicate, operation, allow_ready=False):
        deadline = time.monotonic() + self.timeout
        while True:
            obj, raw = self.next_json(deadline, operation, allow_ready=allow_ready)
            if predicate(obj):
                return obj, raw

    def ready(self):
        # A port attach does not necessarily reset the device. A status reply works in
        # that case; a boot-ready message also satisfies this wait if a reset occurred.
        self.send('?status')
        obj, raw = self.wait(lambda x: x.get('ready') is True, 'startup', allow_ready=True)
        self.log.record('ready', data=obj, raw=raw)
        return obj

    def control(self, command, name):
        self.send(command)
        obj, raw = self.wait(lambda x: x.get('control') == name, command)
        self.log.record('control', data=obj, raw=raw)
        return obj

    def mode(self, profile):
        obj = self.control('?mode ' + profile, 'mode')
        if obj.get('trial') != profile:
            raise ProtocolError(f'mode mismatch: requested {profile!r}, received {obj!r}')
        return obj

    def load(self, fixture):
        payload = fixture['pcm']
        samples = len(payload) // 2
        self.send(f'?load {samples}')
        obj, _ = self.wait(lambda x: x.get('control') == 'load_ready', 'load_ready')
        if obj.get('bytes') != len(payload):
            raise ProtocolError(f'load byte count mismatch: {obj}')
        self.log.record('payload', samples=samples, bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(), source=fixture['source'],
                        transform=fixture.get('transform'))
        self.write_all(payload)
        obj, raw = self.wait(lambda x: x.get('control') == 'loaded', 'loaded')
        if obj.get('samples') != samples:
            raise ProtocolError(f'loaded sample count mismatch: {obj}')
        self.log.record('loaded', data=obj, raw=raw)


class Comparison:
    """Record differences, including expected changes in padding/voice experiment profiles.

Difference reports are evidence, not assertions of failure: some profiles intentionally
change PCM or speaking time. The caller can distinguish those after inspecting the raw log.
"""
    def __init__(self):
        self.references = {}
        self.differences = []
        self.pending = []

    def add(self, log, data, raw, index=0):
        context = log.context
        key = (context.get('phase'), context.get('fixture'), context.get('round'),
               context.get('pair'), index)
        projection = {k: data.get(k) for k in COMPARE_FIELDS}
        log.record('response', response_index=index, data=data, raw=raw,
                   comparison_fields=projection)
        if context.get('profile') == 'baseline':
            self.references[key] = projection
        else:
            self.pending.append((key, dict(context), projection))

    def finish(self, log):
        missing = 0
        for key, context, projection in self.pending:
            reference = self.references.get(key)
            if reference is None:
                missing += 1
                continue
            changed = {field: {'baseline': reference[field], 'actual': projection[field]}
                       for field in COMPARE_FIELDS if projection[field] != reference[field]}
            if changed:
                item = {**context, 'response_index': key[-1], 'changed': changed}
                self.differences.append(item)
                log.record('difference', **item)
        return {'compared_responses': len(self.pending) - missing,
                'missing_baseline_responses': missing, 'differences': len(self.differences)}


def read_fixture(root, name, relative):
    path = root / relative
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError(f'{path}: expected uncompressed 16 kHz mono PCM16 WAV')
        if not 0 < w.getnframes() <= 112000:
            raise ValueError(f'{path}: expected 1..112000 samples; input will not be silently truncated')
        pcm = w.readframes(w.getnframes())
    return {'name': name, 'source': str(path), 'pcm': pcm}


def fixtures(phase, root):
    if phase == 'text':
        return []
    if root is None:
        raise ValueError('--fixture-root is required for asr/replay')
    selected = [read_fixture(root, name, relative) for name, relative in (ASR_FILES if phase == 'asr' else REPLAY_FILES)]
    if phase == 'replay':
        original = selected[-1]
        # Deliberately artificial pause inside the original long utterance, aligned to a 20 ms frame.
        # Keep the source recording untouched and label the transformation in the evidence.
        at = ((len(original['pcm']) // 4) // 320) * 320 * 2
        for pause_ms in (300, 600):
            selected.append({
                'name': f'long-artificial-pause-{pause_ms}ms', 'source': original['source'],
                'pcm': original['pcm'][:at] + b'\0' * (pause_ms * 32) + original['pcm'][at:],
                'transform': {'insert_silence_ms': pause_ms, 'at_sample': at // 2},
            })
        selected.append(read_fixture(root, 'limit-7s', 'recordings_other/20261017-210701-077.wav'))
    return selected


def run(ser, args, log, selected):
    protocol = Protocol(ser, log, args.timeout)
    comparison = Comparison()
    log.record('session', port=args.port, phase=args.phase, profiles=args.profiles,
               rounds=args.rounds, temperature=args.temp, seed=args.seed,
               pairs=args.pairs if args.phase == 'text' else None,
               prompts=args.prompts if args.phase == 'text' else None,
               timeout_s=args.timeout, fixture_root=str(args.fixture_root) if args.fixture_root else None,
               note='cold/warm name the first/second request of a pair; prefill_source identifies actual cache use')
    protocol.ready()
    protocol.control(f'?temp {args.temp:g}', 'temp')
    if args.phase == 'text':
        for round_no in range(1, args.rounds + 1):
            for profile in args.profiles:
                for prompt in args.prompts:
                    log.context = {'phase': 'text', 'round': round_no, 'profile': profile, 'fixture': prompt}
                    protocol.mode(profile)  # the firmware invalidates the last-prompt cache on mode selection
                    for pair in ('cold', 'warm')[:args.pairs]:
                        log.context['pair'] = pair
                        log.record('progress', action='text', prompt=prompt)
                        protocol.control(f'?seed {args.seed}', 'seed')
                        protocol.send('@' + prompt)
                        obj, raw = protocol.wait(lambda x: x.get('heard') == prompt and 'done_ms' in x, '@' + prompt)
                        comparison.add(log, obj, raw)
    else:
        for fixture in selected:
            log.context = {'phase': args.phase, 'fixture': fixture['name']}
            log.record('progress', action='load', samples=len(fixture['pcm']) // 2)
            protocol.load(fixture)
            for round_no in range(1, args.rounds + 1):
                for profile in args.profiles:
                    log.context = {'phase': args.phase, 'round': round_no, 'profile': profile, 'fixture': fixture['name']}
                    log.record('progress', action=args.phase, samples=len(fixture['pcm']) // 2)
                    protocol.mode(profile)
                    protocol.control(f'?seed {args.seed}', 'seed')
                    if args.phase == 'asr':
                        protocol.send('?asr')
                        obj, raw = protocol.wait(lambda x: x.get('control') == 'asr', '?asr')
                        comparison.add(log, obj, raw)
                    else:
                        protocol.send('?replay')
                        deadline = time.monotonic() + args.timeout
                        responses = 0
                        while True:
                            obj, raw = protocol.next_json(deadline, '?replay')
                            if obj.get('control') == 'replay_done':
                                log.record('replay_done', data=obj, raw=raw, observed_responses=responses)
                                break
                            if 'reply' in obj and 'done_ms' in obj:
                                comparison.add(log, obj, raw, responses)
                                responses += 1
    log.context = {}
    log.record('complete', status='finished', **comparison.finish(log))


class MockSerial:
    """Scripted byte transport that fragments lines and allows binary newlines in uploads."""
    def __init__(self, boot_ready=True):
        self.incoming = bytearray(b'boot diagnostic\n')
        if boot_ready:
            self.incoming.extend(b'{"ready":true,"version":"mock"}\n')
        self.commands = bytearray()
        self.command_history = []
        self.upload = bytearray()
        self.remaining = 0
        self.loaded = 0
        self.profile = 'baseline'

    @property
    def in_waiting(self):
        return min(len(self.incoming), 11)

    def read(self, count):
        block = bytes(self.incoming[:min(count, 7)])
        del self.incoming[:len(block)]
        return block

    def emit(self, obj):
        self.incoming.extend((json.dumps(obj, ensure_ascii=False) + '\n').encode())

    def write(self, data):
        # Exercise partial writes as well as fragmented reads.
        data = bytes(data[:97])
        used = len(data)
        if self.remaining:
            assert len(data) <= self.remaining
            self.upload.extend(data)
            self.remaining -= len(data)
            if not self.remaining:
                self.emit({'control': 'loaded', 'samples': self.loaded})
            return used
        self.commands.extend(data)
        while b'\n' in self.commands:
            line, _, rest = self.commands.partition(b'\n')
            self.commands = bytearray(rest)
            command = line.decode()
            self.command_history.append(command)
            if command == '?status':
                self.emit({'control': 'status', 'ready': True, 'version': 'mock'})
            elif command.startswith('?mode '):
                self.profile = command.split(' ', 1)[1]
                self.emit({'control': 'mode', 'trial': self.profile, 'flags': 0})
            elif command.startswith('?temp ') or command.startswith('?seed '):
                self.emit({'control': command[1:].split(' ', 1)[0]})
            elif command.startswith('?load '):
                self.loaded = int(command.split(' ')[1])
                self.remaining = self.loaded * 2
                self.upload.clear()
                self.emit({'control': 'load_ready', 'bytes': self.remaining})
            elif command == '?asr':
                self.emit({'control': 'asr', 'kana': 'こんにちわ', 'phrase': 'こんにちは', 'asr_ms': 100})
            elif command == '?replay':
                self.emit({'heard': 'こんにちは', 'reply': 'こんにちはなのだ', 'say': 'コンニチワナノダ',
                           'feel': 'ふつう', 'voice': 'neutral', 'pcm_hash': 42, 'done_ms': 10})
                self.emit({'control': 'replay_done', 'responses': 1})
            elif command.startswith('@'):
                self.emit({'heard': command[1:], 'reply': 'こんにちはなのだ', 'say': 'コンニチワナノダ',
                           'feel': 'ふつう', 'voice': 'neutral', 'pcm_hash': 42, 'done_ms': 10})
            else:
                raise AssertionError(command)
        return used

    def flush(self):
        pass


def self_test():
    import contextlib
    import io
    total = 0
    cases = [('text', 2, PROMPTS), ('text', 1, ['こんにちは', '別の入力']),
             ('asr', 2, PROMPTS), ('replay', 2, PROMPTS)]
    for phase, pairs, prompts in cases:
        args = argparse.Namespace(port='mock', phase=phase, profiles=['baseline', 'trial'],
                                  rounds=1 if phase == 'text' else 2, pairs=pairs, prompts=prompts,
                                  temp=0.0, seed=42, timeout=1.0, fixture_root=Path('/mock'))
        selected = [{'name': 'binary-newline-fixture', 'source': '/mock.wav', 'pcm': b'\x00\x0a\xff\x7f' * 100}]
        output = io.StringIO()
        transport = MockSerial(boot_ready=phase != 'asr')
        with contextlib.redirect_stdout(io.StringIO()):
            run(transport, args, Log(output), selected)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        ready = next(row['data'] for row in records if row['kind'] == 'ready')
        assert (ready.get('control') == 'status') == (phase == 'asr')
        assert records[-1]['kind'] == 'complete' and records[-1]['differences'] == 0
        assert records[-1]['missing_baseline_responses'] == 0
        if phase != 'text':
            assert bytes(transport.upload) == selected[0]['pcm']
            assert sum(c.startswith('?load ') for c in transport.command_history) == len(selected)
            assert sum(c.startswith('?mode ') for c in transport.command_history) == args.rounds * len(args.profiles)
            assert sum(c.startswith('?seed ') for c in transport.command_history) == args.rounds * len(args.profiles)
        responses = [r for r in records if r['kind'] == 'response']
        expected = args.rounds * len(args.profiles) * (len(prompts) * pairs if phase == 'text' else len(selected))
        assert len(responses) == expected
        assert all('raw' in row for row in responses)
        total += len(responses)
    print(json.dumps({'self_test': 'PASS', 'phases': 3, 'responses': total,
                      'checked': ['boot/attach ready', 'partial read/write', 'binary newlines', 'mode ACK',
                                  'one upload across rounds/profiles', 'pairs=1/2', 'custom prompts',
                                  'replay delimiter', 'raw log']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('port', nargs='?')
    parser.add_argument('out', nargs='?', type=Path)
    parser.add_argument('--phase', choices=('text', 'asr', 'replay'), default='text')
    parser.add_argument('--profiles', default='baseline')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--pairs', type=int, choices=(1, 2), default=2,
                        help='text requests per prompt: 1=first only, 2=first plus immediate warm repeat (default)')
    parser.add_argument('--prompts', type=Path, help='JSON file containing a nonempty array of text prompts')
    parser.add_argument('--fixture-root', type=Path)
    parser.add_argument('--fixture-names', help='comma-separated ASR/replay fixture names to include')
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.port or not args.out:
        parser.error('port and out are required unless --self-test is used')
    if args.rounds < 1 or args.timeout <= 0 or args.temp < 0 or args.seed < 0:
        parser.error('rounds/timeout must be positive, temp/seed must be nonnegative')
    args.profiles = [x.strip() for x in args.profiles.split(',') if x.strip()]
    if not args.profiles or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', x) for x in args.profiles):
        parser.error('profiles must be comma-separated named profiles')
    try:
        args.prompts = json.loads(args.prompts.read_text()) if args.prompts else list(PROMPTS)
        if not isinstance(args.prompts, list) or not args.prompts or any(
                not isinstance(p, str) or not p or '\n' in p or '\r' in p or len(p.encode('utf-8')) > 300
                for p in args.prompts):
            raise ValueError('--prompts must contain a nonempty JSON array of nonempty single-line strings (at most 300 UTF-8 bytes each)')
        if len(set(args.prompts)) != len(args.prompts):
            raise ValueError('--prompts must not contain duplicates; use --rounds for repeated measurements')
        selected = fixtures(args.phase, args.fixture_root)
        if args.fixture_names is not None:
            if args.phase == 'text':
                raise ValueError('--fixture-names is only available for asr/replay')
            names = [name.strip() for name in args.fixture_names.split(',') if name.strip()]
            if not names:
                raise ValueError('--fixture-names requires at least one fixture name')
            available = {fixture['name'] for fixture in selected}
            unknown = set(names) - available
            if unknown:
                raise ValueError('unknown fixture names: ' + ', '.join(sorted(unknown))
                                 + '; available: ' + ', '.join(sorted(available)))
            selected = [fixture for fixture in selected if fixture['name'] in names]
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    try:
        import serial
    except ImportError:
        parser.error('pyserial is unavailable; run with the PlatformIO Python environment')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        log = Log(output)
        try:
            with serial.Serial(port=None, baudrate=115200, timeout=0.2, write_timeout=args.timeout) as ser:
                ser.dtr = False
                ser.rts = False
                ser.port = args.port
                ser.open()
                run(ser, args, log, selected)
        except Exception as exc:
            log.record('failure', error=type(exc).__name__, message=str(exc), status='incomplete')
            raise


if __name__ == '__main__':
    main()
