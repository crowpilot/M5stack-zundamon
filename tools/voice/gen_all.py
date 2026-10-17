import time, sys
from moras import BASE, CTX
from units import get_unit, get_end
K, S = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1, 0)
t = time.time(); n = 0
for j, X in enumerate(BASE):
    if j % K != S: continue
    for ctx in CTX:
        for pp in (['L'] if ctx == 'sil' else ['H', 'L']):
            for cp in ['H', 'L']:
                get_unit(ctx, X, pp, cp); n += 1
    print(X, n, round(time.time() - t), flush=True)
if S == 0:
    for v in ['a', 'i', 'u', 'e', 'o', 'N']:
        for p in 'HL': get_end(v, p)
print('DONE', n, round(time.time() - t), flush=True)
