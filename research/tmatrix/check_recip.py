import mpmath as mp
from ebcm import tmatrix

mp.mp.dps = 30
for nmax, nth in [(12, 80), (12, 160), (16, 160), (20, 200)]:
    T, q, rgq = tmatrix(3.0, 2.0, mp.mpf(1.5), 1, nmax, nth)
    K = T.rows // 2
    # Compare the leading 8x8 corner (orders 1..8), where both truncations agree.
    sym = max(abs(T[i, j] - T[j, i]) for i in range(8) for j in range(8))
    print(nmax, nth, mp.nstr(sym, 3), mp.nstr(T[0, 2], 15))
