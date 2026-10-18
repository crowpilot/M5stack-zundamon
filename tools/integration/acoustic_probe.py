#!/usr/bin/env python3
"""Play an existing WAV on the Mac and check the Core2 physical microphone path.

  python3 tools/integration/acoustic_probe.py /dev/cu.usbserial-... \
      --wav /path/to/recording.wav --out dist/acoustic-probe.jsonl \
      --profiles baseline,frontend,early-asr --timeout 20

Requires macOS afplay and pyserial. This plays sound, but does not record audio,
change volume, or control other applications. It is NOT a measurement recording
the acoustic onset of the Core2 reply. Response JSON arrives after its pipeline
finishes; the caller must judge whether heard/kana correspond to the played WAV.

The driver puts the microphone on hold with ?status, selects each profile and
seed, then ?listen starts calibration. Playback starts after two seconds. A missed
response is recorded as a failed trial and the next profile is still attempted.
The final ?listen restores normal listening before the serial port is closed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

try:
    from .bench import Log, Protocol, ProtocolError
except ImportError:
    from bench import Log, Protocol, ProtocolError


def timestamp(log):
    return {'utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'elapsed_s': round(time.monotonic() - log.started, 6)}


def start_player(executable, wav, log):
    started = timestamp(log)
    player = subprocess.Popen([executable, str(wav)], stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    ended = {}
    finished = threading.Event()

    def watch():
        # Observe completion while serial reception may be waiting for several seconds.
        ended['returncode'] = player.wait()
        ended.update(timestamp(log))
        finished.set()

    watcher = threading.Thread(target=watch, name='afplay-completion', daemon=True)
    watcher.start()
    log.record('playback_start', wav=str(wav), pid=player.pid, playback_start=started)
    return player, ended, finished, watcher


def finish_player(player, ended, finished, watcher, log):
    terminated = False
    if player.poll() is None:
        # Only the child afplay started by this trial is stopped. No other audio app is touched.
        terminated = True
        player.terminate()
        try:
            player.wait(timeout=2)
        except subprocess.TimeoutExpired:
            player.kill()
            player.wait(timeout=2)
    if not finished.wait(timeout=2):
        raise RuntimeError('afplay completion was not observed')
    watcher.join(timeout=2)
    stderr = player.stderr.read().decode('utf-8', errors='replace') if player.stderr else ''
    if player.stderr:
        player.stderr.close()
    log.record('playback_end', playback_end=dict(ended), terminated_by_probe=terminated,
               stderr=stderr[-2000:])
    return ended['returncode'], terminated


def run(ser, args, afplay, log):
    protocol = Protocol(ser, log, args.timeout)
    trials = []
    restored = False
    wav_hash = hashlib.sha256(args.wav.read_bytes()).hexdigest()
    log.record('session', port=args.port, phase='physical-microphone', wav=str(args.wav),
               wav_sha256=wav_hash, profiles=args.profiles, timeout_s=args.timeout,
               note='No audio recording. Received response timing is not measured acoustic reply onset; heard/kana require caller assessment.')
    try:
        protocol.ready()
        protocol.control('?status', 'status')  # hold even when boot-ready arrived before its status ACK
        for profile in args.profiles:
            log.context = {'phase': 'physical-microphone', 'profile': profile, 'fixture': str(args.wav)}
            player = None
            result = {'profile': profile, 'response_received': False, 'status': 'failed'}
            try:
                protocol.control('?status', 'status')
                protocol.mode(profile)
                protocol.control('?seed 42', 'seed')
                listen = protocol.control('?listen', 'listen')
                log.record('listen_start', data=listen, listen_start=timestamp(log), calibration_wait_s=2)
                time.sleep(2)
                player, ended, finished, watcher = start_player(afplay, args.wav, log)
                obj, raw = protocol.wait(lambda x: 'reply' in x and 'heard' in x and 'done_ms' in x,
                                         'physical microphone response')
                received = timestamp(log)
                result.update({'response_received': True, 'status': 'response-observed',
                               'heard': obj.get('heard'), 'kana': obj.get('kana'),
                               'response_received_at': received})
                log.record('response', data=obj, raw=raw, response_received_at=received,
                           corresponds_to_wav='not automatically determined')
            except (TimeoutError, ProtocolError, OSError) as exc:
                result['error'] = type(exc).__name__
                result['message'] = str(exc)
                log.record('trial_failure', error=type(exc).__name__, message=str(exc),
                           response_received=result['response_received'])
            finally:
                # This hold command may interrupt ongoing speech collection in the trial firmware.
                try:
                    protocol.control('?status', 'status')
                    result['microphone_held_after_trial'] = True
                except Exception as exc:
                    result['microphone_held_after_trial'] = False
                    result['status'] = 'failed'
                    log.record('cleanup_failure', action='microphone hold', error=type(exc).__name__, message=str(exc))
                if player is not None:
                    try:
                        code, terminated = finish_player(player, ended, finished, watcher, log)
                        result.update({'afplay_returncode': code, 'afplay_terminated': terminated,
                                       'afplay_end': dict(ended)})
                        if code != 0 and not terminated:
                            result['status'] = 'failed'
                    except Exception as exc:
                        result['status'] = 'failed'
                        log.record('cleanup_failure', action='afplay completion', error=type(exc).__name__, message=str(exc))
            trials.append(result)
            log.record('trial_complete', **result)
    finally:
        log.context = {'phase': 'physical-microphone'}
        try:
            data = protocol.control('?listen', 'listen')
            restored = True
            log.record('listening_restored', data=data, restored_at=timestamp(log))
        except Exception as exc:
            log.record('cleanup_failure', action='restore listening', error=type(exc).__name__, message=str(exc))
    log.context = {}
    failed = sum(trial['status'] == 'failed' for trial in trials)
    log.record('complete', status='finished' if restored else 'restore-failed', trials=trials,
               failed_trials=failed, listening_restored=restored,
               assessment='Responses, if present, still need caller confirmation against the source WAV.')
    return 0 if restored and not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('port')
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profiles', default='baseline,frontend,early-asr')
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args()
    args.wav = args.wav.expanduser().resolve()
    args.profiles = [part.strip() for part in args.profiles.split(',') if part.strip()]
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not args.wav.is_file():
        parser.error('--wav must be an existing file')
    if not args.profiles or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', p) for p in args.profiles):
        parser.error('--profiles must be comma-separated named profiles')
    afplay = shutil.which('afplay')
    if not afplay:
        parser.error('macOS afplay is unavailable')
    try:
        import serial
    except ImportError:
        parser.error('pyserial is unavailable; use the PlatformIO Python environment')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        log = Log(output)
        try:
            with serial.Serial(port=None, baudrate=115200, timeout=0.2, write_timeout=args.timeout) as ser:
                ser.dtr = False
                ser.rts = False
                ser.port = args.port
                ser.open()
                status = run(ser, args, afplay, log)
        except Exception as exc:
            log.record('failure', error=type(exc).__name__, message=str(exc), status='incomplete')
            raise
    sys.exit(status)


if __name__ == '__main__':
    main()
