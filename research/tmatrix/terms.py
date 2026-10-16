"""The Q integrands as scalar terms, generated from the vector components of ebcm.vswf.

Each vector component is (radial kind, angular factor) times γ; the triple product
n̂·(A × B) = (A_θ B_φ − A_φ B_θ) + ν (A_φ B_r − A_r B_φ), ν = −r'/r = −β sinθ cosθ x² for a
spheroid (1/x² = α + β cos²θ, k = 1), expands into products of one outgoing radial factor of x,
one internal radial factor of z = s x, an angular factor and a power of x. Q's element (row type,
column type) is −i [n̂·(A₁ × B₁) + n̂·(A₂ × B₂)] with rows (M_n, N_n) | (N_n, M_n) and columns
(s N_k, M_k) | (s M_k, N_k), as in subtract.py.

Radial kinds: outgoing 'h' = h_n(x), 'h/x', 'Dh' = [x h_n]'/x; internal 'j' = j_k(z), 'j/z',
'Dj' = [z j_k]'/z.
"""

import mpmath as mp


def outgoing(n, d, pi_, tau):
    """Components (r, θ, φ) of M_n and N_n as lists of (radial kind, angular factor), with π → −π
    for the (−1)^m X_{−m n} of the EBCM."""
    p = -pi_[n]
    m_ = [[], [("h", 1j * p)], [("h", -tau[n])]]
    n_ = [[("h/x", n * (n + 1) * d[n])], [("Dh", tau[n])], [("Dh", 1j * p)]]
    return m_, n_


def internal(k, d, pi_, tau, scale):
    """Components of `scale`·M_k and `scale`·N_k (z = s x)."""
    p = pi_[k]
    m_ = [[], [("j", scale * 1j * p)], [("j", -scale * tau[k])]]
    n_ = [[("j/z", scale * k * (k + 1) * d[k])], [("Dj", scale * tau[k])], [("Dj", scale * 1j * p)]]
    return m_, n_


def cross_terms(a, b, nu_ang):
    """n̂·(A × B) as (outgoing kind, internal kind, extra power of x from ν, angular factor)."""
    out = []

    def mul(ca, cb, sign, extra, ang_extra):
        for ka, va in ca:
            for kb, vb in cb:
                out.append((ka, kb, extra, sign * va * vb * ang_extra))

    mul(a[1], b[2], 1, 0, 1)
    mul(a[2], b[1], -1, 0, 1)
    mul(a[2], b[0], 1, 2, nu_ang)
    mul(a[0], b[2], -1, 2, nu_ang)
    return out


def element_terms(n, k, row_type, col_type, s, d, pi_, tau, beta, st, ct):
    """The terms of Q_{(row_type, n), (col_type, k)} at a node, each (outgoing kind, internal
    kind, power of x including the r² of dS, coefficient), the coefficient carrying −i, the
    angular factors, γ_n γ_k and 2π (the quadrature weight is applied by the caller)."""
    xm, xn = outgoing(n, d, pi_, tau)
    rows = (xm, xn) if row_type == 0 else (xn, xm)
    if col_type == 0:
        ym, _ = internal(k, d, pi_, tau, 1)
        _, yn_s = internal(k, d, pi_, tau, s)
        cols = (yn_s, ym)
    else:
        ym_s, _ = internal(k, d, pi_, tau, s)
        _, yn = internal(k, d, pi_, tau, 1)
        cols = (ym_s, yn)
    nu_ang = -beta * st * ct
    g = mp.sqrt(mp.mpf(2 * n + 1) / (4 * mp.pi * n * (n + 1))) * mp.sqrt(
        mp.mpf(2 * k + 1) / (4 * mp.pi * k * (k + 1))
    )
    out = []
    for a, b in zip(rows, cols, strict=True):
        for ka, kb, extra, coeff in cross_terms(a, b, nu_ang):
            out.append((ka, kb, 2 + extra, -1j * 2 * mp.pi * g * coeff))
    return out
