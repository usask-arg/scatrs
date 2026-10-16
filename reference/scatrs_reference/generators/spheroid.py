"""Randomly oriented spheroids: the T-matrix by the EBCM from its definition, in Arb.

For each azimuthal mode m the matrices Q and RgQ are the surface integrals of the cross
products of the vector spherical wave functions (Mishchenko, Travis & Lacis 2002, §5.8; the
conventions of ``research/tmatrix/ebcm.py``), evaluated component by component with no
simplification of the integrands, on Arb's own Gauss–Legendre rule over the whole range
-1 <= cos θ <= 1 (``arb.legendre_p_root``); the spherical Bessel functions come from
``acb.bessel_j`` and ``acb.bessel_y``. Tᵐ = −RgQ Q⁻¹ is solved by ``acb_mat.solve``. The
orientation averages are Mishchenko's (1991):
⟨C_ext⟩ = −2π Σ_m (2 − δ_m0) Σ_n Re(T¹¹_nn + T²²_nn), ⟨C_sca⟩ = 2π Σ_m (2 − δ_m0) Σ |Tᵐ|²
(k = 1), divided by πx² for the efficiencies.

Two kinds of records per case (x, ε, m):

* ``converged``: Q_ext, Q_sca and Q_abs = Q_ext − Q_sca with N and the number of nodes grown
  until both change by less than 10⁻¹⁷ relative. Labelled "high-precision": the truncation in
  N is not rigorously bounded (the ball radii bound the rounding only).
* ``fixed``: at a given N and 2n_θ nodes (the kernel's n_θ in a half range), Q_ext and Q_sca and
  a sample of Tᵐ for modes 0, 1 and N/2 (the diagonals of T¹¹ and T²², and the first column of
  each block), which isolate the solve from the convergence. Labelled "certified": for that
  truncation, the ball radii bound everything.

The working precision is ``PRECISION_BITS`` (Q's elements cancel by up to 10¹⁶, and its
conditioning reaches 10¹⁵ in the cases here), doubled for a case until every stored radius is
below 2^-``REL_BITS`` of its scale.
"""

from __future__ import annotations

from flint import acb, acb_mat, arb, ctx

from scatrs_reference.generators.mie import _map
from scatrs_reference.records import PORTABLE_BITS, nearest_double, record

GENERATOR = "spheroid"
PRECISION_BITS = 480
REL_BITS = 120
MAX_PRECISION_BITS = 1920
SETTLE = arb("1e-17")

#: (x, ε, Re m, Im m, fixed N, fixed n_θ in a half range).
CASES = (
    (0.5, 2.0, 1.5, 0.01, 6, 12),
    (1.0, 0.5, 1.33, 0.0, 8, 16),
    (2.0, 1.8, 1.5, 0.02, 10, 24),
    (2.0, 0.2, 1.53, 0.008, 14, 40),
    (3.0, 5.0, 1.31, 0.0, 14, 48),
    (3.0, 1.0, 1.5, 0.01, 12, 20),
    (4.0, 3.0, 1.53, 0.008, 16, 40),
    (4.0, 1 / 3, 1.5, 0.5, 18, 44),
    (5.0, 2.0, 3.0, 0.1, 26, 60),
    (6.0, 0.5, 1.33, 0.0, 20, 44),
)


def gauss_legendre(n: int) -> tuple[list[arb], list[arb]]:
    """Nodes and weights of the n-point rule on [-1, 1]."""
    nodes, weights = [], []
    for k in range(n):
        x, w = arb.legendre_p_root(n, k, weight=True)
        nodes.append(x)
        weights.append(w)
    return nodes, weights


def shape(x: arb, eps: arb):
    """r(θ) and dr/dθ of the spheroid of equal-volume size parameter x and aspect ratio ε."""
    a = x * eps ** (arb(1) / 3)
    c = a / eps

    def at(ct: arb, st: arb) -> tuple[arb, arb]:
        den = c * c * st * st + a * a * ct * ct
        r = a * c / den.sqrt()
        dr = -a * c * (c * c - a * a) * st * ct / (den * den.sqrt())
        return r, dr

    return at


def angular(m: int, nmax: int, ct: arb, st: arb) -> tuple[list[arb], list[arb], list[arb]]:
    """Normalized d_mn, π_mn = m d/sin θ and τ_mn = dd/dθ for n = 0..nmax (m >= 0)."""
    zero = arb(0)
    d = [zero] * (nmax + 1)
    if m <= nmax:
        am = arb(1)
        for k in range(1, m + 1):
            am *= (arb(2 * k - 1) / (2 * k)).sqrt()
        d[m] = am * st**m
        if m + 1 <= nmax:
            d[m + 1] = arb(2 * m + 1).sqrt() * ct * d[m]
        for n in range(m + 1, nmax):
            d[n + 1] = ((2 * n + 1) * ct * d[n] - arb(n * n - m * m).sqrt() * d[n - 1]) / arb(
                (n + 1) ** 2 - m * m
            ).sqrt()
    tau = [zero] * (nmax + 1)
    pi_ = [zero] * (nmax + 1)
    for n in range(max(m, 1), nmax + 1):
        prev = d[n - 1] if n - 1 >= m else zero
        tau[n] = (n * ct * d[n] - arb(n * n - m * m).sqrt() * prev) / st
        pi_[n] = m * d[n] / st
    return d, pi_, tau


def bessels(z: acb, nmax: int, outgoing: bool) -> tuple[list[acb], list[acb]]:
    """z_n(z) and [z z_n(z)]'/z for n = 0..nmax: j_n, or h_n = j_n + i y_n if ``outgoing``."""
    pre = (acb.pi() / (2 * z)).sqrt()
    half = arb(1) / 2
    zn = []
    for n in range(nmax + 1):
        v = pre * z.bessel_j(n + half)
        if outgoing:
            v += 1j * pre * z.bessel_y(n + half)
        zn.append(v)
    # z_{-1}: j_{-1} = cos z/z, y_{-1} = sin z/z.
    zm1 = (z.cos() + 1j * z.sin()) / z if outgoing else z.cos() / z
    der = [(zm1 if n == 0 else zn[n - 1]) - n * zn[n] / z for n in range(nmax + 1)]
    return zn, der


def vswf(m, nmax, z, outgoing, d, pi_, tau, flip):
    """(r, θ, φ) components of M_mn and N_mn (φ factor dropped); with ``flip``, π → −π (the
    (−1)^m X_{−mn} of the integrals)."""
    zn, der = bessels(z, nmax, outgoing)
    out_m, out_n = [None] * (nmax + 1), [None] * (nmax + 1)
    sgn = -1 if flip else 1
    for n in range(max(m, 1), nmax + 1):
        g = (arb(2 * n + 1) / (4 * arb.pi() * n * (n + 1))).sqrt()
        p = sgn * pi_[n]
        out_m[n] = (acb(0), g * zn[n] * 1j * p, -g * zn[n] * tau[n])
        out_n[n] = (
            g * n * (n + 1) * zn[n] / z * d[n],
            g * der[n] * tau[n],
            g * der[n] * 1j * p,
        )
    return out_m, out_n


def tmatrix_mode(x, eps, s, m, nmax, nodes, weights) -> list[list[acb]]:
    """Tᵐ (2K × 2K, blocks [[T¹¹, T¹²], [T²¹, T²²]], K = nmax − max(m, 1) + 1)."""
    at = shape(x, eps)
    n0 = max(m, 1)
    kk = nmax - n0 + 1
    acc = {
        (a, b, h): [[acb(0)] * kk for _ in range(kk)]
        for a in (1, 2)
        for b in (1, 2)
        for h in (True, False)
    }
    for ct, w in zip(nodes, weights, strict=True):
        st = (1 - ct * ct).sqrt()
        r, dr = at(ct, st)
        nv = -dr / r
        fac = w * r * r * 2 * arb.pi()
        d, pi_, tau = angular(m, nmax, ct, st)
        inner = vswf(m, nmax, s * r, False, d, pi_, tau, flip=False)
        for out in (True, False):
            outer = vswf(m, nmax, acb(r), out, d, pi_, tau, flip=True)
            for a, xs in ((1, outer[0]), (2, outer[1])):
                for b, ys in ((1, inner[0]), (2, inner[1])):
                    mat = acc[(a, b, out)]
                    for i in range(n0, nmax + 1):
                        A = xs[i]
                        row = mat[i - n0]
                        for j in range(n0, nmax + 1):
                            B = ys[j]
                            trip = (A[1] * B[2] - A[2] * B[1]) + nv * (A[2] * B[0] - A[0] * B[2])
                            row[j - n0] += fac * trip

    def assemble(out: bool) -> acb_mat:
        J = {(a, b): acc[(a, b, out)] for a in (1, 2) for b in (1, 2)}
        q = acb_mat(2 * kk, 2 * kk)
        mi = acb(0, -1)
        for i in range(kk):
            for j in range(kk):
                q[i, j] = mi * (s * J[1, 2][i][j] + J[2, 1][i][j])
                q[i, kk + j] = mi * (s * J[1, 1][i][j] + J[2, 2][i][j])
                q[kk + i, j] = mi * (s * J[2, 2][i][j] + J[1, 1][i][j])
                q[kk + i, kk + j] = mi * (s * J[2, 1][i][j] + J[1, 2][i][j])
        return q

    q, rgq = assemble(True), assemble(False)
    # T = −RgQ Q⁻¹: Qᵀ Tᵀ = −RgQᵀ.
    t = q.transpose().solve(-rgq.transpose(), algorithm="precond").transpose()
    return [[t[i, j] for j in range(2 * kk)] for i in range(2 * kk)]


def efficiencies(x, eps, s, nmax, n_nodes) -> tuple[arb, arb, list[list[list[acb]]]]:
    """Q_ext, Q_sca and every Tᵐ (m = 0..nmax) at N = nmax and n_nodes nodes."""
    nodes, weights = gauss_legendre(n_nodes)
    ext, sca = arb(0), arb(0)
    ts = []
    for m in range(nmax + 1):
        t = tmatrix_mode(x, eps, s, m, nmax, nodes, weights)
        ts.append(t)
        w = 1 if m == 0 else 2
        size = len(t)
        for i in range(size):
            ext -= w * t[i][i].real
            for v in t[i]:
                sca += w * (v.real * v.real + v.imag * v.imag)
    area = arb.pi() * x * x
    two_pi = 2 * arb.pi()
    return two_pi * ext / area, two_pi * sca / area, ts


def start_n(x: float, eps: float) -> int:
    """The starting N: x_max + 4 x_max^(1/3) + 3 (through Arb, for portability)."""
    with ctx.workprec(PORTABLE_BITS):
        a = arb(x) * arb(eps) ** (arb(1) / 3)
        xm = max(a, a / arb(eps), key=lambda v: v.mid())
        return max(4, int(nearest_double(xm + 4 * xm ** (arb(1) / 3))) + 3)


def converged(x, eps, s, xf: float, ef: float):
    """Q_ext and Q_sca with N, then the nodes, grown until they settle to SETTLE."""
    n = start_n(xf, ef)
    nodes = 2 * (int(n * (1 + abs(_ln(ef)))) + 8)
    e0, c0, _ = efficiencies(x, eps, s, n, nodes)
    while True:
        n2 = n + max(2, n // 6)
        nodes2 = 2 * (int(n2 * (1 + abs(_ln(ef)))) + 8)
        e1, c1, _ = efficiencies(x, eps, s, n2, nodes2)
        n, nodes = n2, nodes2
        done = abs(e1 - e0) < SETTLE * abs(e1) and abs(c1 - c0) < SETTLE * abs(c1)
        e0, c0 = e1, c1
        if done:
            break
    while True:
        nodes2 = nodes + max(8, nodes // 4)
        e1, c1, _ = efficiencies(x, eps, s, n, nodes2)
        nodes = nodes2
        done = abs(e1 - e0) < SETTLE * abs(e1) and abs(c1 - c0) < SETTLE * abs(c1)
        e0, c0 = e1, c1
        if done:
            break
    return e0, c0, n, nodes


def _ln(v: float) -> float:
    """|ln v| as a double, through Arb (portable: see ``records.nearest_pow10``)."""
    with ctx.workprec(PORTABLE_BITS):
        return nearest_double(arb(v).log())


def _relative_ok(vals: list[arb], scale: arb) -> bool:
    bound = scale * arb(2) ** (-REL_BITS)
    return all(v.rad() < bound for v in vals)


def _name(x: float, eps: float, m_re: float, m_im: float) -> str:
    return f"x={x!r},eps={eps!r},m=({m_re!r},{m_im!r})"


def case_records(x: float, eps: float, m_re: float, m_im: float, nf: int, thf: int) -> list[dict]:
    name = _name(x, eps, m_re, m_im)
    base = {"x": x, "aspect_ratio": eps, "m_re": m_re, "m_im": m_im}
    prec = PRECISION_BITS
    while True:
        with ctx.workprec(prec):
            xa, ea, s = arb(x), arb(eps), acb(m_re, m_im)
            ef, sf, ts = efficiencies(xa, ea, s, nf, 2 * thf)
            sample: dict[str, list[arb]] = {}
            for m in sorted({0, 1, nf // 2}):
                t = ts[m]
                kk = len(t) // 2
                diag11 = [t[i][i] for i in range(kk)]
                diag22 = [t[kk + i][kk + i] for i in range(kk)]
                col = [t[i][0] for i in range(2 * kk)] + [t[i][kk] for i in range(2 * kk)]
                for label, vs in (("diag11", diag11), ("diag22", diag22), ("col", col)):
                    sample[f"t{m}_{label}_re"] = [v.real for v in vs]
                    sample[f"t{m}_{label}_im"] = [v.imag for v in vs]
            qe, qs, n_conv, nodes_conv = converged(xa, ea, s, x, eps)
            scale = max(
                (abs(v.real) + abs(v.imag) for t in ts for row in t for v in row),
                key=lambda a: a.mid(),
            )
            ok = _relative_ok([ef, sf, qe, qs], qe) and _relative_ok(
                [v for vs in sample.values() for v in vs], scale
            )
            if ok or prec >= MAX_PRECISION_BITS:
                if not ok:
                    raise RuntimeError(f"{name}: radii too wide at {prec} bits")
                fixed = record(
                    GENERATOR,
                    name + f",N={nf},n_theta={thf}",
                    {**base, "n_terms": nf, "n_theta": thf},
                    {"q_ext": [ef], "q_sca": [sf], **sample},
                    label="certified",
                )
                conv = record(
                    GENERATOR,
                    name + ",converged",
                    {**base, "n_terms": n_conv, "n_theta": nodes_conv // 2},
                    {"q_ext": [qe], "q_sca": [qs], "q_abs": [qe - qs]},
                    label="high-precision",
                )
                return [fixed, conv]
        prec *= 2


def _records(case: tuple) -> list[dict]:
    return case_records(*case)


def generate() -> tuple[int, list[dict]]:
    """Every case in worker processes (see ``mie._map``), the costliest first so that they
    overlap; the records keep the order of ``CASES``."""
    # The cost grows with the fixed truncation's N·n_θ (the largest takes about 10 minutes).
    order = sorted(range(len(CASES)), key=lambda i: -CASES[i][4] * CASES[i][5])
    results = _map(_records, [CASES[i] for i in order])
    per_case = {}
    for i in order:
        per_case[i] = [r for r in results if r["case"].startswith(_name(*CASES[i][:4]) + ",")]
    return PRECISION_BITS, [r for i in range(len(CASES)) for r in per_case[i]]
