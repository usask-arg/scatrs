"""Sphere: the amplitudes from T must be Bohren & Huffman's S₁, S₂ (any orientation)."""

import math

import mpmath as mp
from amplitude import amplitude_2x2, rotation, scattering_geometry, t_modes
from ebcm import mie_ab

x, s, nmax = 2.0, 1.5 + 0.1j, 8
tm = t_modes(x, 1.0, s, nmax, 30)
mp.mp.dps = 30
a, b = mie_ab(x, mp.mpc(s), nmax)
a = [complex(v) for v in a]
b = [complex(v) for v in b]


def bh(theta):
    mu = math.cos(theta)
    pi_prev, pi_ = 0.0, 1.0
    s1 = s2 = 0
    for n in range(1, nmax + 1):
        tau = n * mu * pi_ - (n + 1) * pi_prev
        f = (2 * n + 1) / (n * (n + 1))
        s1 += f * (a[n - 1] * pi_ + b[n - 1] * tau)
        s2 += f * (a[n - 1] * tau + b[n - 1] * pi_)
        pi_prev, pi_ = pi_, ((2 * n + 1) * mu * pi_ - (n + 1) * pi_prev) / n
    return s1, s2


for theta in (0.3, 1.2, 2.5):
    for beta, g in ((0.0, 0.0), (0.7, 1.1)):
        geo = scattering_geometry(theta, rotation(beta, g))
        smat = amplitude_2x2(tm, *geo)
        s1, s2 = bh(theta)
        print(
            f"Θ={theta} β={beta}: S(par,par)={smat[0, 0]:.6f} S(perp,perp)={smat[1, 1]:.6f} "
            f"| BH S2={s2:.6f} S1={s1:.6f} | off {abs(smat[0, 1]):.1e} {abs(smat[1, 0]):.1e}"
        )
