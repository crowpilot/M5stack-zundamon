#!/usr/bin/env python3
"""Compare real-bank PCM with the pre-integration firmware on a host compiler."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "0a10df94440fa997456620bded5d107eb3bad0d2"


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    args = parser.parse_args()
    bank = args.bank.resolve(strict=True)
    cxx = os.environ.get("CXX", "c++")
    cc = os.environ.get("CC", "cc")
    shim = ROOT / "tools/tts/shim"
    with tempfile.TemporaryDirectory(prefix="zunda-tts-") as name:
        temp = Path(name)
        old = temp / "reference-src"
        old.mkdir()
        pcm = temp / "pcm"
        pcm.mkdir()
        for filename in ("tts.cpp", "tts.h", "kana.cpp", "kana.h", "yomi.c", "yomi.h", "yomi_dict.h", "moras.h"):
            content = subprocess.check_output(["git", "show", f"{BASE}:firmware/src/{filename}"], cwd=ROOT)
            (old / filename).write_bytes(content)
        for label, src, flags in (("reference", old, ["-DTTS_REFERENCE"]), ("trial", ROOT / "firmware/src", [])):
            yomi = temp / f"{label}-yomi.o"
            sanitize = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            run([cc, "-std=c99", "-O2", *sanitize, "-I", src, "-c", src / "yomi.c", "-o", yomi])
            exe = temp / label
            run([cxx, "-std=c++11", "-O2", *sanitize, *flags, "-I", shim, "-I", src,
                 ROOT / "tools/tts/host_check.cpp", src / "tts.cpp", src / "kana.cpp", yomi, "-o", exe])
            # UnitTTS has firmware-lifetime allocations and no destructor in the baseline.
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
            run([exe, bank, pcm], env=env)


if __name__ == "__main__":
    main()
