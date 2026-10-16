import time

import mpmath as mp
from ebcm import tmatrix

mp.mp.dps = 30
x, eps, s, nmax, nth = 3.0, 2.0, mp.mpf(1.5), 12, 80
cext = csca = 0
t0 = time.time()
for m in range(0, nmax + 1):
    T, q, rgq = tmatrix(x, eps, s, m, nmax, nth)
    K = T.rows // 2
    w = 1 if m == 0 else 2
    cext += -w * sum(mp.re(T[i, i]) for i in range(2 * K))
    csca += w * sum(abs(T[i, j]) ** 2 for i in range(2 * K) for j in range(2 * K))
    if m == 1:
        sym = max(
            max(abs(T[i, j] - T[j, i]) for i in range(K) for j in range(K)),
            max(abs(T[K + i, K + j] - T[K + j, K + i]) for i in range(K) for j in range(K)),
            max(abs(T[i, K + j] + T[K + j, i]) for i in range(K) for j in range(K)),
        )
        print(
            "m=1 reciprocity residual",
            mp.nstr(sym, 3),
            "max |T|",
            mp.nstr(max(abs(T[i, j]) for i in range(2 * K) for j in range(2 * K)), 3),
        )
cext *= 2 * mp.pi
csca *= 2 * mp.pi
print(
    "<C_ext> =",
    mp.nstr(cext, 15),
    " <C_sca> =",
    mp.nstr(csca, 15),
    " diff",
    mp.nstr(cext - csca, 3),
    f"{time.time() - t0:.0f}s",
)
