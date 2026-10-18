#!/usr/bin/env python3
"""Measure the existing serial @text path (not microphone/acoustic latency).

Opening the serial port may reset Core2. This emits audible replies.
Requires pyserial; run with PlatformIO's Python if it is installed there.
"""
import argparse
import json
import time
from pathlib import Path

import serial

PROMPTS = [
    "おはよう", "こんにちは。", "今日はいい天気ですね",
    "上司が厳しくて困ってる", "テストで100点とったよ！",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port")
    ap.add_argument("output", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite evidence from a previous run.
    with args.output.open("x", encoding="utf-8") as out:
        def record(obj):
            line = json.dumps(obj, ensure_ascii=False)
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)

        with serial.Serial(port=None, baudrate=115200, timeout=0.2) as ser:
            ser.dtr = False
            ser.rts = False
            ser.port = args.port
            ser.open()
            # Allow initialization/noise calibration, retaining boot diagnostics.
            end = time.monotonic() + 8
            while time.monotonic() < end:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    record({"kind": "startup", "line": line})
            for round_no in range(1, args.rounds + 1):
                for prompt in PROMPTS:
                    ser.write(("@" + prompt + "\n").encode("utf-8"))
                    ser.flush()
                    end = time.monotonic() + 30
                    while time.monotonic() < end:
                        line = ser.readline().decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            record({"kind": "diagnostic", "line": line})
                            continue
                        if obj.get("heard") == prompt and "done_ms" in obj:
                            record({"kind": "response", "round": round_no, **obj})
                            break
                        record({"kind": "other", "data": obj})
                    else:
                        raise TimeoutError(f"No completed response for {prompt!r}")
                    time.sleep(0.5)


if __name__ == "__main__":
    main()
