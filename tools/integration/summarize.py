#!/usr/bin/env python3
"""Summarize bench.py JSONL evidence without mixing files or experiment profiles.

  python3 tools/integration/summarize.py dist/text.jsonl dist/asr.jsonl
  python3 tools/integration/summarize.py --self-test

JSON is written to stdout. Live logs are supported: an unfinished final JSON line
is ignored and reported separately. Malformed complete lines are reported. Every
input file retains its own groups, even when the profile/fixture names are equal.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

RESPONSE_METRICS = (
    'first_sound_ms', 'first_piece_ms', 'pcm_leading_ms',
    'asr_ms', 'match_ms', 'total_ms', 'asr_wait_ms', 'voice_to_enqueue_ms',
    'tts_ms', 'audio_ms', 'done_ms', 'llm_ms', 'frontend_us', 'frontend_frames',
    'free_psram', 'free_internal', 'tokens', 'prompt_tokens', 'cached_prompt_tokens',
    'chunks', 'preview_cancellations', 'dropped_chunks', 'samples', 'gap', 'second',
)
RESPONSE_UNIQUE = (
    'heard', 'kana', 'phrase', 'reply', 'say', 'feel', 'voice', 'pcm_hash',
    'prefill_source', 'preview_used', 'best', 'accepted', 'result',
    'tts_missing', 'yomi_skip', 'temp', 'trial',
)
REPLAY_METRICS = (
    'max_arrival_late_ms', 'responses', 'delivered_chunks',
    'preview_cancellations', 'observed_responses',
)
ESTIMATE = 'first_sound_plus_pcm_leading_ms'


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class Group:
    def __init__(self, profile, fixture, pair):
        self.profile, self.fixture, self.pair = profile, fixture, pair
        self.count = 0
        self.metrics = {}
        self.unavailable = Counter()
        self.unique = {}
        self.rounds = set()

    def metric(self, name, value):
        values = self.metrics.setdefault(name, [])
        if valid_number(value):
            values.append(value)
        else:
            self.unavailable[name] += 1

    def add(self, data, outer, names, unique_names=()):
        self.count += 1
        if isinstance(outer.get('round'), int):
            self.rounds.add(outer['round'])
        for name in names:
            if name in data:
                self.metric(name, data[name])
        for name in unique_names:
            if name in data:
                key = json.dumps(data[name], ensure_ascii=False, sort_keys=True)
                self.unique.setdefault(name, {})[key] = data[name]

    def result(self):
        metrics = {}
        for name, values in sorted(self.metrics.items()):
            metrics[name] = {
                'count': len(values),
                'median': statistics.median(values) if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'unavailable_count': self.unavailable[name],
                'missing_count': self.count - len(values) - self.unavailable[name],
            }
        return {
            'profile': self.profile, 'fixture': self.fixture, 'pair': self.pair,
            'count': self.count, 'rounds': sorted(self.rounds), 'metrics': metrics,
            'unique': {name: [values[k] for k in sorted(values)] for name, values in sorted(self.unique.items())},
        }


def group_key(record, data):
    # The outer context is the requested profile; payload trial is retained under unique.
    return (record.get('profile', data.get('trial', data.get('profile'))),
            record.get('fixture'), record.get('pair'))


def summarize(path):
    result = {
        'file': str(path.resolve()), 'records': 0, 'counts_by_kind': {},
        'ignored_incomplete_final_line': False, 'malformed_lines': [],
        'sessions': [], 'ready': [], 'errors': [], 'differences': [], 'complete': [],
        'response_groups': [], 'replay_groups': [],
    }
    try:
        content = path.read_bytes()
    except OSError as exc:
        result['read_error'] = str(exc)
        return result
    result['snapshot_bytes'] = len(content)
    counts = Counter()
    response_groups, replay_groups = {}, {}
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index == len(lines) and not line.endswith((b'\n', b'\r')):
                result['ignored_incomplete_final_line'] = True
            else:
                result['malformed_lines'].append({'line': index, 'error': str(exc)})
            continue
        if not isinstance(record, dict):
            result['malformed_lines'].append({'line': index, 'error': 'JSON record is not an object'})
            continue
        result['records'] += 1
        kind = str(record.get('kind', 'unknown'))
        counts[kind] += 1
        data = record.get('data')
        if not isinstance(data, dict):
            data = {}
        if kind == 'response':
            key = group_key(record, data)
            group = response_groups.setdefault(key, Group(*key))
            group.add(data, record, RESPONSE_METRICS, RESPONSE_UNIQUE)
            if 'first_sound_ms' in data or 'pcm_leading_ms' in data:
                first, pad = data.get('first_sound_ms'), data.get('pcm_leading_ms')
                group.metric(ESTIMATE, first + pad if valid_number(first) and valid_number(pad) else None)
        elif kind == 'replay_done':
            key = group_key(record, data)
            group = replay_groups.setdefault(key, Group(*key))
            merged = {**data}
            if 'observed_responses' in record:
                merged['observed_responses'] = record['observed_responses']
            group.add(merged, record, REPLAY_METRICS)
        elif kind == 'session':
            result['sessions'].append(record)
        elif kind == 'ready':
            result['ready'].append(data)
        elif kind == 'difference':
            result['differences'].append(record)
        elif kind == 'complete':
            result['complete'].append(record)
        if kind in ('error', 'failure') or 'error' in record or 'error' in data:
            # A device-error RX and the driver's following failure are separate log events.
            result['errors'].append({'line': index, 'kind': kind,
                                     'profile': record.get('profile'), 'fixture': record.get('fixture'),
                                     'error': record.get('error', data.get('error')),
                                     'message': record.get('message')})
    result['counts_by_kind'] = dict(sorted(counts.items()))
    order = lambda item: tuple('' if value is None else str(value) for value in item[0])
    result['response_groups'] = [group.result() for _, group in sorted(response_groups.items(), key=order)]
    result['replay_groups'] = [group.result() for _, group in sorted(replay_groups.items(), key=order)]
    result['error_events'] = len(result['errors'])
    result['difference_events'] = len(result['differences'])
    result['complete_events'] = len(result['complete'])
    return result


def self_test():
    import tempfile

    def response(profile, first, pad, pair='cold', **extra):
        return {'kind': 'response', 'profile': profile, 'fixture': 'こんにちは', 'pair': pair,
                'round': 1, 'data': {'first_sound_ms': first, 'pcm_leading_ms': pad,
                                    'reply': '返答', 'say': 'ヘントー', 'feel': 'ふつう',
                                    'voice': 'neutral', 'pcm_hash': 123, 'prefill_source': 1,
                                    'preview_used': False, **extra}}

    with tempfile.TemporaryDirectory(prefix='zunda-summary-test-') as tmp:
        a, b = Path(tmp) / 'a.jsonl', Path(tmp) / 'b.jsonl'
        records = [response('baseline', 1000, 100), response('baseline', 800, 80),
                   response('baseline', 400, 20, pair='warm'), response('trial', -1, -1),
                   {'kind': 'replay_done', 'profile': 'trial', 'fixture': '録音',
                    'data': {'max_arrival_late_ms': 12, 'responses': 1}, 'observed_responses': 1},
                   {'kind': 'difference', 'changed': {'voice': {'baseline': 'a', 'actual': 'b'}}},
                   {'kind': 'failure', 'error': 'TimeoutError', 'message': 'example'},
                   {'kind': 'complete', 'status': 'finished'}]
        a.write_bytes(('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n').encode()
                      + b'{"kind":"respon')
        b.write_text(json.dumps(response('baseline', 2000, 200), ensure_ascii=False) + '\n')
        sa, sb = summarize(a), summarize(b)
        cold = next(g for g in sa['response_groups'] if g['profile'] == 'baseline' and g['pair'] == 'cold')
        assert cold['count'] == 2 and cold['metrics']['first_sound_ms']['median'] == 900
        assert cold['metrics'][ESTIMATE]['median'] == 990
        assert cold['unique']['reply'] == ['返答'] and cold['unique']['preview_used'] == [False]
        invalid = next(g for g in sa['response_groups'] if g['profile'] == 'trial')
        assert invalid['metrics']['first_sound_ms']['count'] == 0
        assert invalid['metrics'][ESTIMATE]['unavailable_count'] == 1
        assert sa['ignored_incomplete_final_line'] and not sa['malformed_lines']
        assert sa['error_events'] == sa['difference_events'] == sa['complete_events'] == 1
        assert sa['replay_groups'][0]['metrics']['max_arrival_late_ms']['max'] == 12
        assert sb['response_groups'][0]['metrics']['first_sound_ms']['median'] == 2000
        a.write_bytes(b'{broken}\n' + a.read_bytes())
        assert len(summarize(a)['malformed_lines']) == 1
        assert 'read_error' in summarize(Path(tmp) / 'missing.jsonl')
    print(json.dumps({'self_test': 'PASS', 'checked': ['file separation', 'profile/pair separation',
          'median/min/max', 'PCM onset estimate', 'negative sentinel', 'unique values',
          'live partial tail', 'malformed complete line', 'error/difference/complete', 'replay metrics']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', type=Path, nargs='*')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.files:
        parser.error('one or more JSONL files are required')
    files = [summarize(path) for path in args.files]
    print(json.dumps({
        'schema_version': 1,
        'notes': [
            'Files are independent; no metric is pooled across files.',
            'Groups pool rounds only within the same file/profile/fixture/pair.',
            'first_sound_plus_pcm_leading_ms is an enqueue timestamp plus a PCM-derived estimate, not measured acoustic onset.',
            'Negative sentinel values and nonfinite/nonnumeric metrics are excluded and counted as unavailable.',
            'An unfinished final JSON line is ignored; complete/failure events describe the snapshot, which may still be running.',
        ],
        'files': files,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    if any('read_error' in item for item in files):
        sys.exit(1)


if __name__ == '__main__':
    main()
