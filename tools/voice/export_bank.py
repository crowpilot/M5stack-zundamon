"""units/ キャッシュ → M5Stack用 bank.bin と moras.h を書き出す

bank.bin レイアウト（リトルエンディアン）
  header  : "ZBNK" u16 ver=1, u16 sr, u16 M, u16 n_mora, u16 n_ctx(=7), u16 n_end(=12)
  index   : n_ctx*n_mora*2*2 個の Entry  [ctx][mora][pp][cp]
  endidx  : 12 個の Entry                 [vowel(6)][p(2)]
  pcm     : int16 モノラル
  Entry   : u32 off(サンプル), u16 len, u16 sa, u16 ea, u16 reserved   (12 byte)
  ctx順   : sil a i u e o N   / 母音順: a i u e o N / ピッチ順: H=0 L=1
"""
import struct, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moras import BASE, CTX
from units import get_unit, get_end, vowel_of, SR, M

out = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
out.mkdir(parents=True, exist_ok=True)
VOW = ['a', 'i', 'u', 'e', 'o', 'N']
PIT = ['H', 'L']

pcm, idx, endidx = [], [], []
off = 0
def put(x):
    global off
    x = np.clip(np.round(x), -32767, 32767).astype('<i2')
    o = off; pcm.append(x); off += len(x); return o, len(x)

for ctx in CTX:
    for X in BASE:
        sil_ent = {}
        for pp in PIT:
            for cp in PIT:
                if ctx == 'sil' and cp in sil_ent:      # 無音文脈は先行ピッチ不問 → 共有
                    idx.append(sil_ent[cp]); continue
                x, sa, ea = get_unit(ctx, X, pp, cp)
                o, n = put(x); idx.append((o, n, sa, ea))
                if ctx == 'sil': sil_ent[cp] = idx[-1]
for v in VOW:
    for p in PIT:
        x, sa = get_end(v, p)
        o, n = put(x); endidx.append((o, n, sa, n))

with open(out/'bank.bin', 'wb') as f:
    f.write(b'ZBNK' + struct.pack('<6H', 1, SR, M, len(BASE), len(CTX), len(endidx)))
    for e in idx + endidx:
        f.write(struct.pack('<IHHHH', *e, 0))
    for x in pcm: f.write(x.tobytes())

# moras.h（デバイスのパーサ用。長い順に並べた探索順も出す）
vow = [VOW.index(vowel_of(m)) for m in BASE]
order = sorted(range(len(BASE)), key=lambda i: -len(BASE[i]))
with open(out/'moras.h', 'w') as f:
    f.write('// 自動生成: tools/export_bank.py\n#pragma once\n#include <stdint.h>\n')
    f.write(f'#define N_MORA {len(BASE)}\n')
    f.write('static const char* const MORA_TEXT[N_MORA] = {' + ','.join(f'"{m}"' for m in BASE) + '};\n')
    f.write('static const uint8_t MORA_VOWEL[N_MORA] = {' + ','.join(map(str, vow)) + '};\n')
    f.write('static const uint8_t MORA_SEARCH[N_MORA] = {' + ','.join(map(str, order)) + '};\n')
    f.write('// 母音だけのモーラ（ー の展開用）: a i u e o N\n')
    f.write('static const uint8_t VOWEL_MORA[6] = {' + ','.join(str(BASE.index(m)) for m in 'アイウエオン') + '};\n')

sec = off / SR
print(f'units={len(idx)} end={len(endidx)} total={sec:.1f}s  bank.bin={(out/"bank.bin").stat().st_size/1e6:.2f}MB')
