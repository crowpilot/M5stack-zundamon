"""export_bank.py の bank.bin（int16、18MB）→ 8bit μ-law 版（約9MB）。Core2 の16MBフラッシュの zbank 領域に書く（tools/flash_voice.sh）
   python3 make_bank_mu.py out/bank.bin ../../voice/bank_mu.bin
ヘッダと索引は同じ（off はサンプル単位のまま）。版番号だけ 1 → 2（=μ-law）にする。"""
import struct, sys
from array import array

src, dst = sys.argv[1], sys.argv[2]
b = open(src, 'rb').read()
assert b[:4] == b'ZBNK'
ver, sr, M, n_mora, n_ctx, n_end = struct.unpack('<6H', b[4:16])
assert ver == 1
hdr = 16 + (n_ctx * n_mora * 4 + n_end) * 12
pcm = array('h'); pcm.frombytes(b[hdr:])

BIAS, CLIP = 0x84, 32635
def mulaw(x):  # G.711 μ-law
    sign = 0x80 if x < 0 else 0
    x = min(abs(x), CLIP) + BIAS
    exp = x.bit_length() - 8
    mant = (x >> (exp + 3)) & 0x0F
    return ~(sign | (exp << 4) | mant) & 0xFF
tbl = bytes(mulaw(v) for v in range(-32768, 32768))
mu = bytes(tbl[v + 32768] for v in pcm)
with open(dst, 'wb') as f:
    f.write(b'ZBNK' + struct.pack('<6H', 2, sr, M, n_mora, n_ctx, n_end) + b[16:hdr] + mu)
print(f'{len(pcm)} samples ({len(pcm)/sr:.0f}s)  {len(b)/1e6:.2f}MB -> {(hdr+len(mu))/1e6:.2f}MB')
