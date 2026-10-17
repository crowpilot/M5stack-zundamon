#!/bin/bash
# 音声バンク（voice/bank_mu.bin、μ-law 約9MB）を Core2 の zbank 領域（0x490000）に書く。初回とバンクを変えたときだけ
#   tools/flash_voice.sh [ポート]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f voice/bank_mu.bin ] || { echo "voice/bank_mu.bin がありません（README の「モデルと音声データ」）"; exit 1; }
PORT="${1:-$(ls /dev/cu.usbserial* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)}"
ESPTOOL="$HOME/.platformio/packages/tool-esptoolpy/esptool.py"
PY="$HOME/.platformio/penv/bin/python"
[ -x "$PY" ] || PY="$(dirname "$(readlink -f "$(command -v pio)")")/python"
"$PY" "$ESPTOOL" --chip esp32 --port "$PORT" --baud 921600 write_flash 0x490000 voice/bank_mu.bin
