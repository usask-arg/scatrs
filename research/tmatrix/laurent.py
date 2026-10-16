"""Laurent and Taylor series of the radial factors of the EBCM integrands.

Every Q integrand is a sum of terms (prefactor in x) × (radial factor of y_n(x)) × (radial factor
of j_k(sx)) × (angular factor), x = k r(θ), plus the regular j_n part of h_n. The y_n factors are
x^{-n-1}(series in x²) exactly (a_l = −a_{l−1}/(2l(2l − 2n − 1)), a_0 = −(2n − 1)!!), so each y-term
is x^{p₀}·(series in x²). For a spheroid 1/x² = α + β cos²θ, so a term of negative even power is a
polynomial in cos θ times the angular factor.

The integrands themselves are generated in terms.py; laurent_q.py checks which groups of terms
integrate to zero."""

import mpmath as mp


def y_series(n, terms):
    """Coefficients a_l of y_n(x) = Σ a_l x^{2l − n − 1}."""
    a = [-mp.fac2(2 * n - 1)]
    for l_ in range(1, terms):
        a.append(-a[-1] / (2 * l_ * (2 * l_ - 2 * n - 1)))
    return a


def j_series(k, terms):
    """Coefficients b_i of j_k(z) = Σ b_i z^{k + 2i}."""
    b = [1 / mp.fac2(2 * k + 1)]
    for i in range(1, terms):
        b.append(-b[-1] / (2 * i * (2 * i + 2 * k + 1)))
    return b


def radial_y(n, kind, terms):
    """(power, coeff) of y_n (kind 'y'), [x y_n]'/x ('Dy') or y_n/x ('y/x')."""
    a = y_series(n, terms)
    if kind == "y":
        return [(2 * l_ - n - 1, c) for l_, c in enumerate(a)]
    if kind == "Dy":
        return [(2 * l_ - n - 2, c * (2 * l_ - n)) for l_, c in enumerate(a)]
    return [(2 * l_ - n - 2, c) for l_, c in enumerate(a)]


def radial_j(k, kind, s, terms):
    """(power of x, coeff) of j_k(sx) ('j'), [z j_k]'/z ('Dj') or j_k/z ('j/z'), z = s x."""
    b = j_series(k, terms)
    if kind == "j":
        return [(k + 2 * i, c * s ** (k + 2 * i)) for i, c in enumerate(b)]
    if kind == "Dj":
        return [
            (k - 1 + 2 * i, c * (k + 1 + 2 * i) * s ** (k - 1 + 2 * i)) for i, c in enumerate(b)
        ]
    return [(k - 1 + 2 * i, c * s ** (k - 1 + 2 * i)) for i, c in enumerate(b)]


def product(ys, js, extra):
    out = {}
    for py, cy in ys:
        for pj, cj in js:
            p = py + pj + extra
            out[p] = out.get(p, 0) + cy * cj
    return out
