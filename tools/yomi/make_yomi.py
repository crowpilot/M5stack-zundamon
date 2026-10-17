"""LLM の返事コーパス → 読み辞書（表記 → カタカナ）を作る。pyopenjtalk が要る（pip install pyopenjtalk）
   python make_yomi.py corpus.txt tools/yomi/yomi.tsv   # corpus は tools/llm/gen_corpus.c で作る
辞書に入れるのは、カタカナにしただけでは読みが合わない語だけ:
  - 漢字を含む語（送りがなごと。例 遊び→アソビ）
  - 助詞の「は」「へ」は直前の語とつなげて登録（例 ボクは→ボクワ）。「は」単独で置き換えると「はれ」まで変わるため
同じ表記で読みが割れたら多い方。端末側は最長一致で置き換え、残りのひらがなはそのままカタカナにする。
"""
import sys, re, collections
import pyopenjtalk

KANJI = re.compile(r'[㐀-鿿々]')

def kata(s):
    return ''.join(chr(ord(c) + 0x60) if 'ぁ' <= c <= 'ゖ' else c for c in s)

def clean(p):
    return re.sub(r'[^ァ-ーー]', '', p)

cnt = collections.defaultdict(collections.Counter)
for line in open(sys.argv[1], encoding='utf-8', errors='ignore'):
    line = line.strip()
    if not line:
        continue
    words = pyopenjtalk.run_frontend(line)
    for i, w in enumerate(words):
        s, p = w['string'], clean(w['pron'])
        if not p:
            continue
        if KANJI.search(s):
            cnt[s][p] += 1
        elif w['pos'] == '助詞' and s in ('は', 'へ') and i > 0:
            prev = words[i - 1]
            ps = prev['string']
            pp = clean(prev['pron'])
            if pp and not KANJI.search(ps) and ps.strip():
                cnt[ps + s][pp + p] += 1

rows = []
for s, c in cnt.items():
    p, n = c.most_common(1)[0]
    if p != kata(s):
        rows.append((s, p, sum(c.values())))
rows.sort(key=lambda r: (-len(r[0]), r[0]))
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    for s, p, n in rows:
        f.write(f'{s}\t{p}\t{n}\n')
print(len(rows), 'entries')
