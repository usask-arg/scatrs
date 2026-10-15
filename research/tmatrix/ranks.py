"""Random orientation by the rank decomposition of the T-matrix, against the brute-force average.

In the helicity basis W^p = (M + pN)/√2, T^{pσ} = (T¹¹ + σT¹² + pT²¹ + pσT²²)/2. For an
axisymmetric particle (T diagonal in m) the rank-s components of T^{pσ} between orders n and n'
are t⁽ˢ⁾(n, n') = Σ_μ (−1)^{n'−μ} C^{s0}_{nμ;n'−μ} T_{nμ,n'μ}; rotating the particle by R
multiplies the q-th component of rank s by D^s_{q0}(R). The amplitude for incident helicity σ
(plane wave along ẑ) and scattered helicity p at (Θ, φ = 0) is
    A_{pσ}(Θ; R) = Σ_{s,q} D^s_{q0}(R) Φ_{sq},
    Φ_{sq} = Σ_{n,n'} far_n(q + σ, p; Θ) inc_{n'}(σ) (−1)^{n'−σ} C^{s q}_{n,q+σ;n',−σ} t⁽ˢ⁾(n, n'),
with inc the incident helicity coefficients and far the far-field factors of W^p_{nm}; the
average over orientations is ⟨A A'*⟩ = Σ_{s,q} Φ_{sq} Φ'*_{sq}/(2s + 1).

Usage: python ranks.py
"""

import math

import numpy as np
from amplitude import angular_signed, averaged_mueller, gamma, t_modes
from sympy.physics.quantum.cg import CG


def cg(j1, m1, j2, m2, j, m):
    if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j or m1 + m2 != m:
        return 0.0
    if j < abs(j1 - j2) or j > j1 + j2:
        return 0.0
    return float(CG(j1, m1, j2, m2, j, m).doit())


def helicity_blocks(tm):
    """T^{pσ}_{nm,n'm} as dict (p, σ) → dict m → (orders, K×K)."""
    out = {}
    for p in (1, -1):
        for sg in (1, -1):
            blocks = {}
            for m, (orders, t) in tm.items():
                k = len(orders)
                t11, t12, t21, t22 = t[:k, :k], t[:k, k:], t[k:, :k], t[k:, k:]
                blocks[m] = (orders, (t11 + sg * t12 + p * t21 + p * sg * t22) / 2)
            out[(p, sg)] = blocks
    return out


def incident(nmax):
    """inc[σ][n]: helicity coefficient of W^σ_{n,σ} of the plane wave (x̂ + iσŷ)/√2 along ẑ, from
    a_mn, b_mn of amplitude.py (c = (a + σ b)/√2)."""
    out = {}
    tiny = 1e-9
    for sg in (1, -1):
        e0 = np.array([1, 1j * sg]) / math.sqrt(2)
        coeffs = {}
        for n in range(1, nmax + 1):
            _, pi_, tau = angular_signed(sg, n, tiny)
            c_vec = np.array([1j * pi_[n], -tau[n]])
            b_vec = np.array([tau[n], 1j * pi_[n]])
            a = 4 * math.pi * 1j**n * gamma(n) * np.dot(np.conj(c_vec), e0)
            b = 4 * math.pi * 1j ** (n - 1) * gamma(n) * np.dot(np.conj(b_vec), e0)
            coeffs[n] = (a + sg * b) / math.sqrt(2)
        out[sg] = coeffs
    return out


def far(m, n, p, theta):
    """The helicity-p far-field factor of W^p_{nm} at (θ, φ = 0): γₙ (−i)ⁿ (π_mn + p τ_mn)."""
    _, pi_, tau = angular_signed(m, n, theta)
    return gamma(n) * (-1j) ** n * (pi_[n] + p * tau[n])


def averaged_helicity(tm, nmax, thetas):
    hb = helicity_blocks(tm)
    inc = incident(nmax)
    smax = 2 * nmax
    # t^{(s)}(n, n') per (p, σ).
    t_rank = {}
    for key, blocks in hb.items():
        for s in range(0, smax + 1):
            for n in range(1, nmax + 1):
                for n2 in range(1, nmax + 1):
                    if s < abs(n - n2) or s > n + n2:
                        continue
                    v = 0
                    for mu in range(-min(n, n2), min(n, n2) + 1):
                        if mu not in blocks:
                            continue
                        orders, t = blocks[mu]
                        if n not in orders or n2 not in orders:
                            continue
                        i, j = orders.index(n), orders.index(n2)
                        v += (-1) ** (n2 - mu) * cg(n, mu, n2, -mu, s, 0) * t[i, j]
                    t_rank[(key, s, n, n2)] = v
    out = {}
    for it, theta in enumerate(thetas):
        for p in (1, -1):
            for sg in (1, -1):
                phi = {}
                for s in range(0, smax + 1):
                    for q in range(-s, s + 1):
                        v = 0
                        for n in range(1, nmax + 1):
                            m = q + sg
                            if abs(m) > n:
                                continue
                            f = far(m, n, p, theta)
                            for n2 in range(1, nmax + 1):
                                tr = t_rank.get(((p, sg), s, n, n2))
                                if tr is None or tr == 0:
                                    continue
                                c = cg(n, m, n2, -sg, s, q)
                                if c == 0:
                                    continue
                                v += f * inc[sg][n2] * (-1) ** (n2 - sg) * c * tr
                        phi[(s, q)] = v
                out[(it, p, sg)] = phi
    avg = np.zeros((len(thetas), 2, 2, 2, 2), dtype=complex)
    hel = (1, -1)
    for it in range(len(thetas)):
        for a, p in enumerate(hel):
            for b, sg in enumerate(hel):
                for c, p2 in enumerate(hel):
                    for d, sg2 in enumerate(hel):
                        phi1, phi2 = out[(it, p, sg)], out[(it, p2, sg2)]
                        avg[it, a, b, c, d] = sum(
                            phi1[k] * np.conj(phi2[k]) / (2 * k[0] + 1) for k in phi1
                        )
    return avg


def mueller_from_products(prod):
    """⟨Z⟩ from ⟨A_{pσ} A*_{p'σ'}⟩ (helicity), via S_lin = U A U⁻¹, U = [ê₊ ê₋] in (θ̂, φ̂)."""
    u = np.array([[1, 1], [1j, -1j]]) / math.sqrt(2)
    ui = np.linalg.inv(u)
    # ⟨S_ij S*_kl⟩ = Σ U_ip Ui_σj conj(U_kp' Ui_σ'l) ⟨A_pσ A*_p'σ'⟩.
    ss = np.einsum(
        "ip,sj,kq,tl,pstq->ijkl", u, ui, np.conj(u), np.conj(ui), prod.transpose(0, 1, 3, 2)
    )
    # Careful ordering: prod[a, b, c, d] = ⟨A_{p_a σ_b} A*_{p_c σ_d}⟩.
    ss = np.einsum("ia,bj,kc,dl,abcd->ijkl", u, ui, np.conj(u), np.conj(ui), prod)
    s2s2, s1s1 = ss[0, 0, 0, 0].real, ss[1, 1, 1, 1].real
    s3s3, s4s4 = ss[0, 1, 0, 1].real, ss[1, 0, 1, 0].real
    s1s2 = ss[1, 1, 0, 0]  # S1 S2*
    s3s4 = ss[0, 1, 1, 0]  # S3 S4*
    z = np.zeros((4, 4))
    z[0, 0] = 0.5 * (s1s1 + s2s2 + s3s3 + s4s4)
    z[0, 1] = 0.5 * (s2s2 - s1s1 + s4s4 - s3s3)
    z[1, 0] = 0.5 * (s2s2 - s1s1 - s4s4 + s3s3)
    z[1, 1] = 0.5 * (s2s2 + s1s1 - s4s4 - s3s3)
    z[2, 2] = (s1s2 + s3s4).real
    z[2, 3] = (np.conj(s1s2) + np.conj(s3s4)).imag
    z[3, 2] = (s1s2 - s3s4).imag
    z[3, 3] = (s1s2 - s3s4).real
    return z


def main():
    x, eps, s, nmax = 2.0, 1.8, 1.5 + 0.02j, 6
    tm = t_modes(x, eps, s, nmax, 30, dps=25)
    thetas = [0.4, 1.3, 2.4]
    brute = averaged_mueller(tm, thetas, nbeta=2 * nmax + 4, ngamma=4 * nmax + 6)
    prods = averaged_helicity(tm, nmax, thetas)
    for it, theta in enumerate(thetas):
        z = mueller_from_products(prods[it])
        b = brute[it]
        print(f"Θ={theta}: max |Δ| {np.max(np.abs(z - b)):.2e} of {np.max(np.abs(b)):.3e}")
        print("  rank  ", np.round(z[[0, 0, 1, 2, 2, 3], [0, 1, 1, 2, 3, 3]], 6))
        print("  brute ", np.round(b[[0, 0, 1, 2, 2, 3], [0, 1, 1, 2, 3, 3]], 6))


if __name__ == "__main__":
    main()
