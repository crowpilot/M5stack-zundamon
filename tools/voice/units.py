"""VCV単位の生成（VOICEVOX Core使用）。ホスト側オフライン処理。"""
import io, wave, os, numpy as np
from pathlib import Path
from scipy.signal import resample_poly, butter, sosfiltfilt
from vvlib import syn
from moras import VOWEL_MORA

SR = 16000
STYLE = 3                 # ずんだもん ノーマル
PITCH = {'H': 6.03, 'L': 5.70}   # log F0（実測の15/85パーセンタイル付近）
PITCH_SHIFT = 0.0         # 赤ちゃん化するなら +0.08 程度
SPEED = 1.0
CTX_LEN = 0.10            # 文脈母音・後続母音の長さ[s]
M = int(0.012*SR)         # クロスフェード用マージン（片側）
CACHE = Path.home()/'vv/units'; CACHE.mkdir(exist_ok=True)
_lp = butter(4, 900, 'low', fs=SR, output='sos')

def _wav_to_np(b):
    with wave.open(io.BytesIO(b)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    return resample_poly(x, 2, 3)   # 24k -> 16k

def vowel_of(mora_text):
    s = syn(); q = s.create_audio_query_from_kana(mora_text+"'", STYLE)
    return q.accent_phrases[0].moras[0].vowel.lower() if q.accent_phrases[0].moras[0].vowel!='N' else 'N'

def _anchor(x, c, f0):
    """切り位置cの近傍±1周期で、低域通過波形のピークに合わせる（位相そろえ）"""
    per = int(SR/f0); lo, hi = max(0, c-per), min(len(x), c+per)
    y = sosfiltfilt(_lp, x[max(0,lo-200):hi+200])[lo-max(0,lo-200):][:hi-lo]
    return lo + int(np.argmax(y))

def _synth(kana, pitches, fix_len):
    s = syn(); q = s.create_audio_query_from_kana(kana, STYLE)
    mo = q.accent_phrases[0].moras
    assert len(mo) == len(pitches), (kana, [m.text for m in mo])
    for m, p in zip(mo, pitches):
        m.pitch = PITCH[p] + PITCH_SHIFT
    for i, L in fix_len.items():
        mo[i].vowel_length = L
    q.speed_scale = SPEED
    x = _wav_to_np(s.synthesis(q, STYLE))
    # 各モーラの時刻（16kHz サンプル）
    t = q.pre_phoneme_length; marks = []
    for m in mo:
        cs = t; t += (m.consonant_length or 0)/SPEED; vs = t; t += m.vowel_length/SPEED
        marks.append((cs, vs, t))
    return x, [(int(a*SR), int(b*SR), int(c*SR)) for a,b,c in marks]

def get_unit(ctx, X, pp, cp):
    """ctx: 'sil'|a i u e o N, X: モーラ, pp/cp: 'H'|'L'。戻り値: (float32波形, start_anchor, end_anchor)"""
    if ctx == 'sil': pp = 'L'
    f = CACHE/f"{ctx}_{X}_{pp}{cp}.npz"
    if f.exists():
        d = np.load(f); return d['x'], int(d['sa']), int(d['ea'])
    v = vowel_of(X); tail = VOWEL_MORA[v]
    if ctx == 'sil':
        x, mk = _synth(X+tail+"タ'", [cp, cp, cp], {1: CTX_LEN})
        xs = mk[0][0] - int(0.015*SR); sa = None; xi = 0
    else:
        x, mk = _synth("タ"+VOWEL_MORA[ctx]+X+tail+"タ'", [pp, pp, cp, cp, cp], {1: CTX_LEN, 3: CTX_LEN})
        c = (mk[1][1]+mk[1][2])//2
        sa = _anchor(x, c, np.exp(PITCH[pp]+PITCH_SHIFT)); xi = 2
    cs, vs, ve = mk[xi]
    vlen = ve - vs
    e = ve - min(int(0.05*SR), vlen//2)
    ea = _anchor(x, e, np.exp(PITCH[cp]+PITCH_SHIFT))
    if sa is None:
        seg = x[xs: ea+M].copy(); sa_out = 0
        seg[:80] *= np.linspace(0, 1, 80)
    else:
        seg = x[sa-M: ea+M].copy(); sa_out = M
    ea_out = len(seg) - M
    np.savez(f, x=seg, sa=sa_out, ea=ea_out)
    return seg, sa_out, ea_out

def get_end(v, p):
    """母音中央→無音への語尾単位"""
    f = CACHE/f"END_{v}_{p}.npz"
    if f.exists():
        d = np.load(f); return d['x'], int(d['sa'])
    x, mk = _synth("タ"+VOWEL_MORA[v]+VOWEL_MORA[v]+"'", [p, p, p], {1: CTX_LEN, 2: 0.08})
    c = (mk[1][1]+mk[1][2])//2
    sa = _anchor(x, c, np.exp(PITCH[p]+PITCH_SHIFT))
    seg = x[sa-M: mk[2][2] + int(0.03*SR)].copy()
    seg[-int(0.03*SR):] *= np.linspace(1, 0, int(0.03*SR))
    np.savez(f, x=seg, sa=M)
    return seg, M
