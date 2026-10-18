#!/usr/bin/env python3
"""Exact last-prompt cache, snapshot format, and prepare-callback checks."""
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "firmware/src"
HERE = Path(__file__).resolve().parent


def main():
    phrases = re.findall(r'^  "(.*)",$', (SRC / "phrases.h").read_text().split("kPhraseKana")[0], re.M)
    phrases += ["", "こんにちは。", "Ａ：Ｂ：", "x" * 300, "あ" * 200]
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "halt_on_error=1"}
    with tempfile.TemporaryDirectory(prefix="zunda-prompt-cache-") as tmp:
        tests = ["prompt_cache_test.c"]
        if (HERE / "prompt_snapshot_test.c").exists():
            tests.append("prompt_snapshot_test.c")
        for test in tests:
            exe = str(Path(tmp) / Path(test).stem)
            subprocess.run([os.environ.get("CC", "cc"), "-O2", "-g", "-fsanitize=address,undefined",
                            "-I" + str(SRC), "-I" + str(HERE), str(HERE / test),
                            str(SRC / "llm_core.c"), str(SRC / "zunda_brain.c"), "-lm", "-o", exe], check=True)
            subprocess.run([exe, str(ROOT / "models/zunda-small-tagend-679_q8.bin"),
                            str(ROOT / "models/tok_tagend.bin")], input=("\n".join(phrases) + "\n").encode(),
                           env=env, check=True)


if __name__ == "__main__":
    main()
