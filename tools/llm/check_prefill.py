#!/usr/bin/env python3
"""Check exact equivalence to v0.0.3, plus KV reuse/invalidation boundaries."""
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "firmware/src"
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "models/zunda-small-tagend-679_q8.bin"
TOK = ROOT / "models/tok_tagend.bin"
BASE = "f7d6a0c3acb81ccc94068ff96f109d3e6bbcf317"


def build(harness, source, output, *flags):
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-g", "-fsanitize=address,undefined",
        *flags, "-I" + str(source), "-I" + str(HERE), str(HERE / harness),
        str(source / "llm_core.c"), str(source / "zunda_brain.c"), "-lm", "-o", str(output),
    ], check=True)


def main():
    # The existing API has no destructor; these short test processes intentionally
    # keep model allocations until exit. Address/undefined checks remain enabled.
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "halt_on_error=1"}
    phrases = re.findall(r'^  "(.*)",$', (SRC / "phrases.h").read_text().split("kPhraseKana")[0], re.M)
    prompts = phrases + ["こんにちは。", "上司が厳しくて困ってる", "テストで100点とったよ！", "",
                         "Ａ：Ｂ：", "x" * 300, "あ" * 200]
    # Run twice in the same process to exercise reuse across unrelated requests.
    input_bytes = ("\n".join(prompts + list(reversed(prompts))) + "\n").encode()
    with tempfile.TemporaryDirectory(prefix="zunda-prefill-") as name:
        tmp = Path(name)
        old = tmp / "baseline"
        old.mkdir()
        for filename in ("llm_core.c", "llm_core.h", "zunda_brain.c", "zunda_brain.h"):
            (old / filename).write_bytes(subprocess.check_output(
                ["git", "show", f"{BASE}:firmware/src/{filename}"], cwd=ROOT))
        build("prefill_test.c", SRC, tmp / "prefill")
        subprocess.run([str(tmp / "prefill"), str(MODEL), str(TOK)], env=env, check=True)
        build("reply_trace.c", old, tmp / "reference")
        build("reply_trace.c", SRC, tmp / "fast")
        build("reply_trace.c", SRC, tmp / "disabled", "-DZUNDA_FAST_PREFILL=0")
        for steps in (256, 64):
            for temp in (0, 0.8):
                args = [str(MODEL), str(TOK), str(steps), str(temp)]
                ref = subprocess.check_output([str(tmp / "reference"), *args], input=input_bytes, env=env)
                for variant in ("fast", "disabled"):
                    result = subprocess.check_output([str(tmp / variant), *args], input=input_bytes, env=env)
                    if result != ref:
                        raise AssertionError(f"trace differs: {variant}, steps={steps}, temp={temp}")
                print(f"reply/callback logits/RNG equal: steps={steps}, temp={temp}, "
                      f"replies={len(prompts) * 2}, sha256={hashlib.sha256(ref).hexdigest()}", flush=True)
    print("All equivalence and cache checks passed (ASan + UBSan).")


if __name__ == "__main__":
    main()
