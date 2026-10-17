"""デバイス側アルゴリズムのPython版（C++実装の仕様書を兼ねる）"""
import numpy as np, wave
from moras import BASE
from units import get_unit, get_end, vowel_of, SR, M

MORAS = sorted(BASE + ['ッ', 'ー'], key=len, reverse=True)
_vcache = {}
def V(m):
    if m not in _vcache: _vcache[m] = vowel_of(m)
    return _vcache[m]

def parse(kana):
    """AquesTalk風記法 → [(mora, 'H'|'L') | ('PAUSE',None)] 。 / で句、、で休止、' でアクセント核、？で疑問"""
    out = []
    chunks = kana.replace('？', '?').replace('、', ',/').split('/')
    for ci, chunk in enumerate(chunks):
        is_last = ci == len(chunks) - 1
        pause = chunk.endswith(',')
        chunk = chunk.rstrip(',')
        q = chunk.endswith('?'); chunk = chunk.rstrip('?')
        ms, acc, i = [], None, 0
        while i < len(chunk):
            if chunk[i] == "'": acc = len(ms); i += 1; continue
            if chunk[i] == '_': i += 1; continue          # 無声化は無視（デモ）
            for m in MORAS:
                if chunk.startswith(m, i): ms.append(m); i += len(m); break
            else: raise ValueError(f'unknown: {chunk[i:]}')
        if not ms: continue
        n = len(ms); acc = acc or n
        if acc == 1: ps = ['H'] + ['L']*(n-1)
        else: ps = ['L'] + ['H']*(acc-1) + ['L']*(n-acc)
        if q: ps[-1] = 'H'
        elif pause or is_last: ps[-1] = 'L'   # 文末・休止前の下降
        out += list(zip(ms, ps))
        if pause: out.append(('PAUSE', None))
    return out

def synth(kana):
    seq = parse(kana)
    out = np.zeros(int(0.1*SR), np.float32); pos = len(out)   # pos = 直前単位の終端アンカー
    ctx, pp = 'sil', None
    fin = np.hanning(2*M*2)[:2*M]; fout = fin[::-1]
    def close():
        nonlocal out, pos
        x, sa = get_end(ctx, pp)
        x = x.copy(); x[:2*M] *= fin
        out = out[:pos+M]; out[pos-M:pos+M] *= fout
        need = pos - sa + len(x)
        if need > len(out): out = np.concatenate([out, np.zeros(need-len(out), np.float32)])
        out[pos-sa: pos-sa+len(x)] += x; pos = len(out)
    for m, p in seq:
        if m in ('PAUSE', 'ッ'):
            if ctx != 'sil': close()
            out = np.concatenate([out, np.zeros(int((0.18 if m=='PAUSE' else 0.06)*SR), np.float32)])
            pos = len(out); ctx = 'sil'; continue
        if m == 'ー': m = {'a':'ア','i':'イ','u':'ウ','e':'エ','o':'オ','N':'ン'}[ctx if ctx!='sil' else 'a']
        x, sa, ea = get_unit(ctx, m, pp or p, p)
        x = x.copy()
        if ctx == 'sil':
            start = pos
        else:
            x[:2*M] *= fin
            out = out[:pos+M]; out[pos-M:pos+M] *= fout
            start = pos - sa
        need = start + len(x)
        if need > len(out): out = np.concatenate([out, np.zeros(need-len(out), np.float32)])
        out[start:start+len(x)] += x
        pos = start + ea; ctx, pp = V(m), p
    if ctx != 'sil': close()
    return np.concatenate([out, np.zeros(int(0.1*SR), np.float32)])

def save(path, x, sr=SR):
    x = np.clip(x, -32767, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())
