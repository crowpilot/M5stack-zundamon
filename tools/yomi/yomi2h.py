"""tools/yomi.tsv（make_yomi.py の出力）→ src/yomi_dict.h。長い表記から順に並べる（端末側は最長一致）
   python3 tools/yomi2h.py tools/yomi/yomi.tsv firmware/src/yomi_dict.h"""
import sys
# pyopenjtalk が文脈で読み違えるもの（このLLMの返事でよく出る形に合わせる）
OVERRIDE = {'空い': 'スイ', '空く': 'スク', '空き': 'スキ', '色': 'イロ'}
import re
KANJI = re.compile(r'[\u3400-\u9fff々]')
rows = {}
for line in open(sys.argv[1], encoding='utf-8'):
    s, p, n = line.rstrip('\n').split('\t')
    # コーパスには崩れた返事も混じるので、ひらがなで始まる漢字語（「かい色」など区切り違い。
    # お腹・ご飯 の お/ご は残す）は捨てる
    if KANJI.search(s) and '\u3041' <= s[0] <= '\u3096' and s[0] not in 'おご':
        continue
    rows[s] = p
rows.update(OVERRIDE)
items = sorted(rows.items(), key=lambda r: (-len(r[0].encode()), r[0]))
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write('// 自動生成: tools/yomi2h.py（元は tools/yomi/yomi.tsv）。手で編集しない\n#pragma once\n')
    f.write(f'#define YOMI_N {len(items)}\n')
    f.write('static const char* const YOMI_SURF[YOMI_N] = {\n' + ''.join(f'  "{s}",\n' for s, _ in items) + '};\n')
    f.write('static const char* const YOMI_PRON[YOMI_N] = {\n' + ''.join(f'  "{p}",\n' for _, p in items) + '};\n')
print(len(items), 'entries ->', sys.argv[2])
