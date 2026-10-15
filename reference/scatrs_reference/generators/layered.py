"""Arb reference values for scattering by layered spheres (a core and concentric shells).

Conventions are those of `scatrs_reference.generators.mie` (Bohren & Huffman, e^(−iωt)). Layer l
(l = 0 is the core) lies between the size parameters x_{l−1} and x_l (x₋₁ = 0) and has the
refractive index m_l relative to the medium. Each value is computed at the exact double inputs.

Method. The generator does not use scatrs's algorithm (the recursion of Yang 2003 in a pole-free
form): it computes the Riccati–Bessel functions themselves and matches the fields at each
interface directly, every step rigorous.

* ψ_n(z) = ψ_{n−1}(z)·u_n(z) from ψ₀ = sin z, with u_n = ψ_n/ψ_{n−1} from the rigorous downward
  recurrences of `mie.ratios_real` and `mie.ratios_complex`; ξ_n(z) = ξ_{n−1}(z)·w_n(z) from
  ξ₀ = −i e^(iz), with w_n = ξ_n/ξ_{n−1} = (2n − 1)/z − 1/w_{n−1} from w₀ = ξ₀/ξ₋₁ = −i; and
  ψ′_n = ψ_{n−1} − nψ_n/z, ξ′_n likewise. Complex products in disk arithmetic.
* In layer l the field of order n is proportional to ψ_n(m_l k r) + c_l ξ_n(m_l k r) (c₀ = 0).
  Continuity of the tangential fields at the interface z₁ = m_l x_{l−1}, z₂ = m_l x_l makes
  the inner field's log-derivative h, times m_l/m_{l−1} for aₙ (m_{l−1}/m_l for bₙ), equal to
  the new layer's: c_l = (h ψ_n(z₁) − ψ′_n(z₁))/(ξ′_n(z₁) − h ξ_n(z₁)), and at the outer
  surface h = (ψ′_n(z₂) + c_l ξ′_n(z₂))/(ψ_n(z₂) + c_l ξ_n(z₂)).
* aₙ and bₙ are the homogeneous sphere's formulas with h/m_L and m_L h in place of D_n(mx)/m
  and m D_n(mx), at x = x_L.
* The series (Q_ext, Q_sca, Q_abs = Q_ext − Q_sca, exactly 0 when every layer is real, Q_back,
  g, S₁, S₂) are summed to ⌈x_L + 18x_L^(1/3) + 32⌉ terms, as `mie.sum_series` sums them.
* Derivatives with respect to each layer's x_l, Re m_l and Im m_l are five-point central
  differences in Arb, (8[f(p + h) − f(p − h)] − [f(p + 2h) − f(p − 2h)])/(12h) with
  h = 2^−(prec/5): rounding and the truncation O(h⁴) are both near 2^−(4·prec/5).

Each case is computed at ``PRECISION_BITS``, doubled until every stored value has a radius of at
most 2^-``REL_BITS`` of its scale (as in `mie`).

Records. For each case the file holds ``…,coefficients`` (aₙ and bₙ at the orders ``orders``)
and ``…,series`` (at the cosines ``mu``); ``layered_derivatives.jsonl`` holds the derivatives
of the coefficients (``da_dx{l}``, ``da_dm{l}``, ``db_dx{l}``, ``db_dm{l}``: the complex
derivative with respect to m_l, which is ∂/∂(Re m_l)) and of the efficiencies (``q_ext_dx{l}``,
``q_ext_dm_re{l}``, ``q_ext_dm_im{l}``, …).

Labels. Coefficient records are finite computations, so certified. The series have no tail bound
for layered spheres and the derivatives come from differences: high-precision.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from flint import acb, arb, ctx

from scatrs_reference import disk
from scatrs_reference.generators.mie import (
    MAX_PRECISION_BITS,
    PRECISION_BITS,
    _array_scales,
    _map,
    _tight,
    cosines,
    orders,
    ratios_complex,
    ratios_real,
    size_terms,
    sum_series,
)
from scatrs_reference.records import nearest_pow10, record

#: The outer size parameters of the grid.
XS = (1e-6, 1e-3, 0.1, 1.0, 5.0, 20.0, 100.0, 1000.0, 10000.0)
#: (core, shell) indices as (Re m, Im m): black carbon in sulfate, water on ice, a silica core in
#: a gold-like shell, absorbing cores in a clear shell, high contrast, near unity, an air shell.
PAIRS = (
    ((1.95, 0.79), (1.43, 1e-8)),
    ((1.31, 1e-8), (1.33, 1e-9)),
    ((1.45, 0.0), (0.2, 3.0)),
    ((1.5, 1e-8), (1.33, 0.0)),
    ((1.5, 1e-3), (1.33, 0.0)),
    ((1.5, 1.0), (1.33, 0.0)),
    ((10.0, 10.0), (1.33, 0.0)),
    ((1.01, 0.0), (1.0001, 0.0)),
    ((1.5, 0.01), (1.0, 0.0)),
)
#: Core radius / outer radius.
FRACTIONS = (1e-3, 0.3, 0.9, 0.999)
#: Three layers: fractions and indices.
THREE = ((0.3, 0.7, 1.0), ((1.5, 0.1), (2.5, 1.0), (1.33, 0.0)))
THREE_XS = (0.1, 5.0, 100.0)
#: Ten layers of a graded index, Re m from 1.58 down to 1.4.
TEN_XS = (1.0, 20.0, 200.0)
RANDOM_CASES = 40
SEED = 20261014
DERIVATIVE_CASES = (
    ((0.7, 1.3), ((1.95, 0.79), (1.43, 1e-8))),
    ((0.01, 0.02), ((10.0, 10.0), (1.33, 0.0))),
    ((3.0, 5.0, 8.0), ((1.5, 0.1), (2.5, 1.0), (1.33, 0.0))),
    ((19.0, 20.0), ((1.5, 1e-3), (1.33, 0.0))),
    ((60.0, 100.0), ((1.45, 0.0), (0.2, 3.0))),
    ((1e-6, 2e-6), ((1.5, 0.01), (1.33, 0.0))),
    ((10.0, 10.0), ((1.5, 0.01), (2.0, 0.5))),
    ((25.0, 100.0), ((1.31, 1e-8), (1.33, 1e-9))),
)

#: (interface size parameters, (Re m, Im m) per layer).
Case = tuple[tuple[float, ...], tuple[tuple[float, float], ...]]


def case_name(xs: Sequence[float], ms: Sequence[tuple[float, float]]) -> str:
    return f"x={list(xs)!r},m={[tuple(m) for m in ms]!r}"


@dataclass
class Functions:
    """ψ_n, ξ_n and their derivatives of one argument for n = 0..n_top (index n), as balls."""

    psi: list[acb]
    dpsi: list[acb]
    xi: list[acb]
    dxi: list[acb]


def functions(z: acb, real: bool, n_top: int) -> Functions:
    """ψ_n(z), ξ_n(z) and their derivatives; ``real`` takes the real recurrence for u_n."""
    if real:
        u = [acb(v) for v in ratios_real(z.real, n_top)]
    else:
        u = ratios_complex(z, n_top)
    psi = [acb(z.sin())]
    p = disk.disk(psi[0])
    xi_d = disk.disk(-acb(0, 1) * (acb(0, 1) * z).exp())
    xi = [disk.ball(xi_d)]
    w = disk.Disk(acb(0, -1), arb(0))
    for n in range(1, n_top + 1):
        p = disk.mul(p, disk.disk(u[n]))
        psi.append(disk.ball(p))
        w = disk.sub_from((2 * n - 1) / z, disk.inv(w))
        xi_d = disk.mul(xi_d, w)
        xi.append(disk.ball(xi_d))
    dpsi = [acb(z.cos())] + [psi[n - 1] - n * psi[n] / z for n in range(1, n_top + 1)]
    dxi = [(acb(0, 1) * z).exp()] + [  # ξ₀′ = e^(iz)
        xi[n - 1] - n * xi[n] / z for n in range(1, n_top + 1)
    ]
    return Functions(psi, dpsi, xi, dxi)


@dataclass
class Solution:
    """aₙ, bₙ for n = 1..n_ref (index n − 1) of a layered sphere, at the working precision."""

    x: arb
    n_ref: int
    a: list[acb]
    b: list[acb]
    real: bool


def solve(xs: Sequence[arb], ms: Sequence[acb], real: Sequence[bool], n_ref: int) -> Solution:
    """The coefficients by direct interface matching at the exact (ball) inputs."""
    layers = len(xs)
    core = functions(ms[0] * xs[0], real[0], n_ref)
    inner = [functions(ms[j] * xs[j - 1], real[j], n_ref) for j in range(1, layers)]
    outer = [functions(ms[j] * xs[j], real[j], n_ref) for j in range(1, layers)]
    x = xs[-1]
    out = functions(acb(x), True, n_ref)
    a, b = [], []
    for n in range(1, n_ref + 1):
        hs = []
        for kind in "ab":
            h = core.dpsi[n] / core.psi[n]
            for j in range(1, layers):
                f1, f2 = inner[j - 1], outer[j - 1]
                t = (ms[j] / ms[j - 1] if kind == "a" else ms[j - 1] / ms[j]) * h
                c = (t * f1.psi[n] - f1.dpsi[n]) / (f1.dxi[n] - t * f1.xi[n])
                h = (f2.dpsi[n] + c * f2.dxi[n]) / (f2.psi[n] + c * f2.xi[n])
            hs.append(h)
        m = ms[-1]
        for h, dest in ((hs[0] / m, a), (m * hs[1], b)):
            big = h + n / x
            dest.append((big * out.psi[n] - out.psi[n - 1]) / (big * out.xi[n] - out.xi[n - 1]))
    return Solution(x, n_ref, a, b, all(real))


def _inputs(case: Case) -> tuple[list[arb], list[acb], list[bool]]:
    xs, ms = case
    return [arb(v) for v in xs], [acb(re, im) for re, im in ms], [im == 0.0 for _, im in ms]


def _coefficients(s: Solution, ns: Sequence[int]) -> dict[str, list]:
    out: dict[str, list] = {}
    for name, vs in (("a", s.a), ("b", s.b)):
        out[f"{name}_re"] = [vs[n - 1].real for n in ns]
        out[f"{name}_im"] = [vs[n - 1].imag for n in ns]
    return out


def _series(s: Solution, mus: Sequence[float]) -> dict[str, list]:
    sums = sum_series(s, mus)  # uses x, a, b and n_ref
    x2 = s.x * s.x
    q_ext, q_sca = sums.q_ext, sums.q_sca
    return {
        "q_ext": [q_ext],
        "q_sca": [q_sca],
        "q_abs": [arb(0) if s.real else q_ext - q_sca],
        "q_back": [abs(sums.s_back) ** 2 / x2],
        "g": [4 * sums.g_sum / (x2 * q_sca)],
        "s1_re": [v.real for v in sums.s1],
        "s1_im": [v.imag for v in sums.s1],
        "s2_re": [v.real for v in sums.s2],
        "s2_im": [v.imag for v in sums.s2],
    }


def _series_scales(series: dict[str, list], x: arb) -> dict[str, arb]:
    q_ext, q_sca = series["q_ext"][0], series["q_sca"][0]
    s_rms = x * q_sca.sqrt() / 2
    scales = {"q_abs": abs(q_ext.mid()), "q_back": abs(q_sca.mid())}
    return scales | {k: abs(s_rms.mid()) for k in series if k.startswith("s")}


def value_records(case: Case) -> list[dict]:
    xs, ms = case
    name = case_name(xs, ms)
    ns = orders(size_terms(xs[-1], 6, 3))
    mus = cosines()
    terms = size_terms(xs[-1], 18, 32)
    prec = PRECISION_BITS
    while prec <= MAX_PRECISION_BITS:
        with ctx.workprec(prec):
            try:
                s = solve(*_inputs(case), terms)
            except disk.DiskContainsZero:
                prec *= 2
                continue
            coefficients = _coefficients(s, ns)
            series = _series(s, mus)
            if _tight(coefficients, _array_scales(coefficients)) and _tight(
                series, _series_scales(series, s.x)
            ):
                inputs = {"x": list(xs), "m_re": [m[0] for m in ms], "m_im": [m[1] for m in ms]}
                return [
                    record(
                        "layered", f"{name},coefficients", inputs | {"orders": ns}, coefficients
                    ),
                    record(
                        "layered",
                        f"{name},series",
                        inputs | {"mu": mus, "n_ref": terms},
                        series,
                        "high-precision",
                    ),
                ]
        prec *= 2
    raise ArithmeticError(f"{name} needs more than {MAX_PRECISION_BITS} bits")


def _derivative_outputs(case: Case, ns: Sequence[int], terms: int) -> dict[str, list]:
    """The five-point differences of the coefficients and efficiencies for every parameter."""
    xs, ms = _inputs(case)[:2]
    real = _inputs(case)[2]
    h = arb(2) ** -(ctx.prec // 5)
    out: dict[str, list] = {}
    for j in range(len(xs)):
        for p in ("dx", "dm_re", "dm_im"):
            sols = {}
            for k in (-2, -1, 1, 2):
                xk, mk = list(xs), list(ms)
                if p == "dx":
                    xk[j] = xk[j] + k * h
                elif p == "dm_re":
                    mk[j] = mk[j] + k * h
                else:
                    mk[j] = mk[j] + acb(0, k * h)
                # An imaginary step takes the complex recurrences even where m is real.
                rk = list(real)
                if p == "dm_im":
                    rk[j] = False
                s = solve(xk, mk, rk, terms)
                sols[k] = (s, _series(s, []))

            def diff(f):
                return (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / (12 * h)

            if p != "dm_im":
                key = "dx" if p == "dx" else "dm"
                for name, arr in (("a", "a"), ("b", "b")):
                    vals = [
                        diff(lambda k, n=n, arr=arr, s=sols: getattr(s[k][0], arr)[n - 1])
                        for n in ns
                    ]
                    out[f"d{name}_{key}{j}_re"] = [v.real for v in vals]
                    out[f"d{name}_{key}{j}_im"] = [v.imag for v in vals]
            for q in ("q_ext", "q_sca", "q_abs", "q_back", "g"):
                out[f"{q}_{p}{j}"] = [diff(lambda k, q=q, s=sols: s[k][1][q][0])]
    return out


def derivative_records(case: Case) -> list[dict]:
    xs, ms = case
    name = case_name(xs, ms)
    ns = orders(size_terms(xs[-1], 6, 3))
    terms = size_terms(xs[-1], 18, 32)
    prec = 2 * PRECISION_BITS
    while prec <= MAX_PRECISION_BITS:
        with ctx.workprec(prec):
            try:
                out = _derivative_outputs(case, ns, terms)
            except disk.DiskContainsZero:
                prec *= 2
                continue
            # A derivative of Q_abs or Q_back may be tiny: measure it against those of Q_ext and
            # Q_sca; the coefficient derivatives against their arrays. A derivative can also
            # vanish identically (with respect to the index of a shell of zero thickness):
            # every scale is at least 2^-40 of the largest of its kind in the record.
            coefficient = {k: v for k, v in out.items() if k.startswith("d")}
            scales = _array_scales(coefficient)
            floor = max(scales.values()) * arb(2) ** -40
            scales = {k: max(v, floor) for k, v in scales.items()}
            q_floor = max(abs(v[0].mid()) for k, v in out.items() if k.startswith("q"))
            q_floor *= arb(2) ** -40
            for j in range(len(xs)):
                for p in ("dx", "dm_re", "dm_im"):
                    ext, sca = out[f"q_ext_{p}{j}"][0].mid(), out[f"q_sca_{p}{j}"][0].mid()
                    scale = max(abs(ext), abs(sca), q_floor)
                    for q in ("q_ext", "q_sca", "q_abs", "q_back"):
                        scales[f"{q}_{p}{j}"] = scale
                    g = abs(out[f"g_{p}{j}"][0].mid())
                    scales[f"g_{p}{j}"] = max(g, q_floor)
            if _tight(out, scales):
                inputs = {
                    "x": list(xs),
                    "m_re": [m[0] for m in ms],
                    "m_im": [m[1] for m in ms],
                    "orders": ns,
                    "n_ref": terms,
                }
                return [record("layered_derivatives", name, inputs, out, "high-precision")]
        prec *= 2
    raise ArithmeticError(f"{name} needs more than {MAX_PRECISION_BITS} bits")


def cases() -> list[Case]:
    """The two-layer grid, three- and ten-layer particles, and seeded random cases."""
    out: list[Case] = []
    for x in XS:
        for core, shell in PAIRS:
            for f in FRACTIONS:
                out.append(((f * x, x), (core, shell)))
    fractions, indices = THREE
    for x in THREE_XS:
        out.append((tuple(f * x for f in fractions), indices))
    for x in TEN_XS:
        out.append(
            (
                tuple(j / 10 * x for j in range(1, 11)),
                tuple((1.6 - 0.02 * j, 1e-4) for j in range(1, 11)),
            )
        )
    rng = random.Random(SEED)
    for _ in range(RANDOM_CASES):
        layers = rng.randint(2, 8)
        x = nearest_pow10(rng.uniform(-6.0, 3.5))
        fs = sorted(rng.uniform(0.05, 1.0) for _ in range(layers - 1))
        ms = []
        for _ in range(layers):
            if rng.random() < 0.15:
                ms.append((rng.uniform(0.1, 1.0), rng.uniform(1.0, 6.0)))
            else:
                m_im = 0.0 if rng.random() < 0.3 else nearest_pow10(rng.uniform(-8.0, 0.3))
                ms.append((rng.uniform(1.0, 3.0), m_im))
        out.append(((*(f * x for f in fs), x), tuple(ms)))
    return out


def generate() -> tuple[int, list[dict]]:
    return PRECISION_BITS, _map(value_records, cases())


def generate_derivatives() -> tuple[int, list[dict]]:
    return 2 * PRECISION_BITS, _map(derivative_records, list(DERIVATIVE_CASES))
