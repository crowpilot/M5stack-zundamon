#!/bin/bash
# リリース用の書き込みファイルを dist/ に作る（GitHub Releases に添付する想定）
#   tools/make_release.sh            → dist/M5stack-zundamon-v<版>-core2.bin（0x0 に書く1本もの）
#                                       dist/M5stack-zundamon-voice-bank.bin（0x490000 に書く音声バンク）
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$(sed -n 's/.*-DZUNDA_VERSION=\\"\(.*\)\\".*/\1/p' firmware/platformio.ini)
(cd firmware && pio run)
B=firmware/.pio/build/m5stack-core2
mkdir -p dist
PY="$HOME/.platformio/penv/bin/python"
[ -x "$PY" ] || PY="$(dirname "$(readlink -f "$(command -v pio)")")/python"
"$PY" "$HOME/.platformio/packages/tool-esptoolpy/esptool.py" --chip esp32 merge_bin --flash_size 16MB \
  -o "dist/M5stack-zundamon-v$VER-core2.bin" \
  0x1000 $B/bootloader.bin 0x8000 $B/partitions.bin 0x10000 $B/firmware.bin
[ -f voice/bank_mu.bin ] && cp voice/bank_mu.bin dist/M5stack-zundamon-voice-bank.bin
(cd dist && shasum -a 256 *.bin > SHA256SUMS.txt && cat SHA256SUMS.txt)
