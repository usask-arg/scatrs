import mpmath as mp
from ebcm import mie_ab, tmatrix

mp.mp.dps = 30
x, s, nmax = 2.0, mp.mpc(1.5, 0.1), 6
a, b = mie_ab(x, s, nmax)
for m in (0, 1, 2):
    T, q, rgq = tmatrix(x, 1.0, s, m, nmax, 40)
    n0 = max(m, 1)
    K = nmax - n0 + 1
    for n in range(n0, min(n0 + 3, nmax + 1)):
        i = n - n0
        print(
            m,
            n,
            mp.nstr(T[i, i], 12),
            mp.nstr(-b[n - 1], 12),
            "|",
            mp.nstr(T[K + i, K + i], 12),
            mp.nstr(-a[n - 1], 12),
        )
    off = max(abs(T[i, j]) for i in range(2 * K) for j in range(2 * K) if i != j)
    print("  max off-diagonal", mp.nstr(off, 3))
