import mpmath as mp
from sources import integrate, test_rows

mp.mp.dps = 40
s = mp.mpc(1.5, 0.01)
x, eps, m, nmax, nth = 5.0, 0.5, 1, 8, 40
sd, _ = integrate(
    x, eps, s, m, nmax, nth, lambda r, ct, st: test_rows(m, [mp.mpf(0)], 8, r, ct, st), "incident"
)
print("S_D(origin) diagonal:", [mp.nstr(sd[i, i], 6) for i in range(0, 16, 3)])
print(
    "max off-diagonal:",
    mp.nstr(max(abs(sd[i, j]) for i in range(16) for j in range(16) if i != j), 3),
)
