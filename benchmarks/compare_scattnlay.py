"""Accuracy and speed of scatrs's layered spheres against scattnlay, both measured against Arb.

Accuracy: every case of the Arb reference suite (reference/data/layered.jsonl) inside scatrs's
validated domain, for scatrs, for scattnlay (Peña & Pal 2009, the recursion scatrs builds on,
in double precision), and for direct interface matching in double precision (the textbook
method of Aden & Kerker 1951, which Bohren & Huffman's BHCOAT implements): the coefficients aₙ
and bₙ (max over the stored orders of |error| / max |a|), Q_ext, Q_sca, Q_abs, Q_back and g
(relative errors; Q_back relative to max(Q_back, Q_sca), g to max(|g|, 10⁻³)), and S₁, S₂
(relative to max(|S|, S_rms)), each code with its own number of terms.

Speed: the efficiencies of coated spheres (black carbon in sulfate, core radius 0.5 of the
outer) at outer size parameters 1, 10, 100 and 1000, one thread, best of three; scatrs for the
values alone and with every derivative (in the derivatives' domain, x_L ≤ 100), and a sphere of
the shell's index at the same size for comparison.

Conventions: scattnlay takes the same m (Im m ≥ 0 absorbing) and returns the same aₙ, bₙ, S₁
and S₂ (e^(−iωt)), so no conjugation is needed.

Usage: uv run --directory benchmarks python compare_scattnlay.py [--quick]
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import scattnlay

from scatrs import layered, mie

DATA = Path(__file__).resolve().parent.parent / "reference" / "data" / "layered.jsonl"
CODES = ("scatrs", "scattnlay", "direct")
OUTPUTS = ("coefficients", "q_ext", "q_sca", "q_abs", "q_back", "g", "amplitudes")


def value(r: dict, name: str) -> np.ndarray:
    """The reference values of output `name`: hi + lo (double-double accuracy is plenty)."""
    o = r["outputs"][name]
    return np.array(o["hi"]) + np.array(o["lo"])


def cases() -> list[tuple[dict, dict]]:
    """(coefficients, series) record pairs."""
    records = [json.loads(line) for line in DATA.read_text().splitlines()]
    series = {
        r["case"].removesuffix(",series"): r for r in records if r["case"].endswith(",series")
    }
    return [
        (r, series[r["case"].removesuffix(",coefficients")])
        for r in records
        if r["case"].endswith(",coefficients")
    ]


def inputs(r: dict) -> tuple[np.ndarray, np.ndarray]:
    i = r["inputs"]
    return np.array(i["x"]), np.array(i["m_re"]) + 1j * np.array(i["m_im"])


def inside(x: np.ndarray, m: np.ndarray) -> bool:
    try:
        layered.efficiencies(x, m)
    except Exception:
        return False
    return True


def direct(x: np.ndarray, m: np.ndarray, n: int, mus: np.ndarray):
    """aₙ, bₙ, the efficiencies and S₁, S₂ by direct interface matching in complex128, with
    ψ and ξ from upward recurrences (the textbook method)."""

    def riccati(z: complex) -> tuple[np.ndarray, np.ndarray]:
        psi = np.zeros(n + 1, complex)
        xi = np.zeros(n + 1, complex)
        psi[0], xi[0] = np.sin(z), -1j * np.exp(1j * z)
        psi_m1, xi_m1 = np.cos(z), np.exp(1j * z)  # ψ₋₁ = cos z, ξ₋₁ = e^(iz)
        for k in range(1, n + 1):
            prev_p = psi[k - 2] if k >= 2 else psi_m1
            prev_x = xi[k - 2] if k >= 2 else xi_m1
            psi[k] = (2 * k - 1) / z * psi[k - 1] - prev_p
            xi[k] = (2 * k - 1) / z * xi[k - 1] - prev_x
        return psi, xi

    def deriv(f: np.ndarray, z: complex, prev0: complex) -> np.ndarray:
        d = np.zeros_like(f)
        d[0] = prev0
        k = np.arange(1, n + 1)
        d[1:] = f[:-1] - k * f[1:] / z
        return d

    with np.errstate(all="ignore"):
        out = []
        for kind in "ab":
            z0 = m[0] * x[0]
            p0, _ = riccati(z0)
            h = deriv(p0, z0, np.cos(z0)) / p0
            for j in range(1, len(x)):
                z1, z2 = m[j] * x[j - 1], m[j] * x[j]
                p1, x1 = riccati(z1)
                p2, x2 = riccati(z2)
                dp1, dx1 = deriv(p1, z1, np.cos(z1)), deriv(x1, z1, np.exp(1j * z1))
                dp2, dx2 = deriv(p2, z2, np.cos(z2)), deriv(x2, z2, np.exp(1j * z2))
                t = (m[j] / m[j - 1] if kind == "a" else m[j - 1] / m[j]) * h
                c = (t * p1 - dp1) / (dx1 - t * x1)
                h = (dp2 + c * dx2) / (p2 + c * x2)
            px, xx = riccati(complex(x[-1]))
            nn = np.arange(1, n + 1)
            big = (h[1:] / m[-1] if kind == "a" else m[-1] * h[1:]) + nn / x[-1]
            out.append((big * px[1:] - px[:-1]) / (big * xx[1:] - xx[:-1]))
        a, b = out
        return a, b, series(a, b, x[-1], mus)


def series(a: np.ndarray, b: np.ndarray, x: float, mus: np.ndarray) -> dict:
    """Q_ext, Q_sca, Q_abs = Q_ext − Q_sca, Q_back, g, S₁, S₂ from the coefficients."""
    n = np.arange(1, len(a) + 1)
    w = 2 * n + 1
    q_ext = 2 / x**2 * np.sum(w * (a + b).real)
    q_sca = 2 / x**2 * np.sum(w * (abs(a) ** 2 + abs(b) ** 2))
    q_back = abs(np.sum(w * (-1.0) ** n * (a - b))) ** 2 / x**2
    g = (
        4
        / (x**2 * q_sca)
        * (
            np.sum(
                n[:-1]
                * (n[:-1] + 2)
                / (n[:-1] + 1)
                * (a[:-1] * a[1:].conj() + b[:-1] * b[1:].conj()).real
            )
            + np.sum(w / (n * (n + 1)) * (a * b.conj()).real)
        )
    )
    s1, s2 = [], []
    for mu in mus:
        pi = np.zeros(len(a) + 1)
        pi[1] = 1.0
        for k in range(2, len(a) + 1):
            pi[k] = ((2 * k - 1) * mu * pi[k - 1] - k * pi[k - 2]) / (k - 1)
        tau = n * mu * pi[1:] - (n + 1) * pi[:-1]
        c = w / (n * (n + 1))
        s1.append(np.sum(c * (a * pi[1:] + b * tau)))
        s2.append(np.sum(c * (a * tau + b * pi[1:])))
    return {
        "q_ext": q_ext,
        "q_sca": q_sca,
        "q_abs": q_ext - q_sca,
        "q_back": q_back,
        "g": g,
        "s1": np.array(s1),
        "s2": np.array(s2),
    }


def run(code: str, x: np.ndarray, m: np.ndarray, mus: np.ndarray, n_ref: int):
    """(a, b, series) of one code."""
    if code == "scatrs":
        c = layered.coefficients(x, m)
        q = layered.efficiencies(x, m)
        s = layered.amplitudes(x, m, mus)
        out = {k: getattr(q, k) for k in ("q_ext", "q_sca", "q_abs", "q_back", "g")}
        return c.a, c.b, out | {"s1": s.s1, "s2": s.s2}
    if code == "scattnlay":
        _, a, b = scattnlay.scattcoeffs(x, m)
        theta = np.arccos(np.clip(mus, -1, 1))
        _, q_ext, q_sca, q_abs, q_back, _, g, _, s1, s2 = scattnlay.scattnlay(x, m, theta)
        # scattnlay's Q_bk is 4π dσ/dΩ / (π r²) at 180°, as scatrs's Q_back.
        out = {"q_ext": q_ext, "q_sca": q_sca, "q_abs": q_abs, "q_back": q_back, "g": g}
        return a, b, out | {"s1": s1, "s2": s2}
    n = int(np.ceil(x[-1] + 4.05 * x[-1] ** (1 / 3) + 2))
    a, b, out = direct(x, m, n, mus)
    return a, b, out


def errors(r: dict, s: dict, got) -> dict[str, float]:
    a, b, out = got
    orders = np.array(r["inputs"]["orders"])
    ref_a = value(r, "a_re") + 1j * value(r, "a_im")
    ref_b = value(r, "b_re") + 1j * value(r, "b_im")
    scale = max(abs(ref_a).max(), abs(ref_b).max())

    def at(c: np.ndarray) -> np.ndarray:
        c = np.ravel(c)
        return np.array([c[k - 1] if k <= len(c) else 0.0 for k in orders])

    e = {
        "coefficients": max(abs(at(a) - ref_a).max(), abs(at(b) - ref_b).max()) / scale,
    }
    q = {k: value(s, k)[0] for k in ("q_ext", "q_sca", "q_abs", "q_back", "g")}
    e["q_ext"] = abs(float(np.ravel(out["q_ext"])[0]) - q["q_ext"]) / q["q_ext"]
    e["q_sca"] = abs(float(np.ravel(out["q_sca"])[0]) - q["q_sca"]) / q["q_sca"]
    e["q_abs"] = (
        abs(float(np.ravel(out["q_abs"])[0]) - q["q_abs"]) / q["q_abs"] if q["q_abs"] > 0 else 0.0
    )
    e["q_back"] = abs(float(np.ravel(out["q_back"])[0]) - q["q_back"]) / max(
        q["q_back"], q["q_sca"]
    )
    e["g"] = abs(float(np.ravel(out["g"])[0]) - q["g"]) / max(abs(q["g"]), 1e-3)
    s_rms = r["inputs"]["x"][-1] * np.sqrt(q["q_sca"]) / 2
    worst = 0.0
    for name in ("s1", "s2"):
        ref = value(s, f"{name}_re") + 1j * value(s, f"{name}_im")
        got_s = np.ravel(out[name])
        worst = max(worst, (abs(got_s - ref) / np.maximum(abs(ref), s_rms)).max())
    e["amplitudes"] = worst
    return {k: float(v) for k, v in e.items()}


def accuracy(quick: bool) -> dict:
    pairs = cases()
    if quick:
        pairs = pairs[::20]
    per_code = {c: {o: [] for o in OUTPUTS} for c in CODES}
    worst = {c: {o: (0.0, "") for o in OUTPUTS} for c in CODES}
    used = 0
    for r, s in pairs:
        x, m = inputs(r)
        if not inside(x, m):
            continue
        used += 1
        mus = np.array(s["inputs"]["mu"])
        for code in CODES:
            try:
                e = errors(r, s, run(code, x, m, mus, s["inputs"]["n_ref"]))
            except Exception:
                e = dict.fromkeys(OUTPUTS, float("inf"))
            for o in OUTPUTS:
                v = e[o] if np.isfinite(e[o]) else float("inf")
                per_code[code][o].append(v)
                if not v <= worst[code][o][0]:
                    worst[code][o] = (v, r["case"].removesuffix(",coefficients"))
    summary = {}
    for code in CODES:
        summary[code] = {}
        for o in OUTPUTS:
            v = np.array(per_code[code][o])
            finite = v[np.isfinite(v)]
            summary[code][o] = {
                "median": float(np.median(finite)) if finite.size else None,
                "max_finite": float(finite.max()) if finite.size else None,
                "worst_case": worst[code][o][1],
                "above_1e-10": int(np.sum(~(v <= 1e-10))),
                "above_1e-6": int(np.sum(~(v <= 1e-6))),
                "failed": int(np.sum(~np.isfinite(v))),
            }
    return {"cases": used, "summary": summary}


def best_of(f, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def speed(quick: bool) -> list[dict]:
    rows = []
    m = np.array([1.95 + 0.79j, 1.43 + 1e-8j])
    for x_l in (1.0, 10.0, 100.0) if quick else (1.0, 10.0, 100.0, 1000.0):
        count = max(1, int(2000 / x_l))
        x = np.tile([0.5 * x_l, x_l], (count, 1))
        mm = np.tile(m, (count, 1))
        t_scatrs = best_of(lambda x=x, mm=mm: layered.efficiencies(x, mm)) / count
        t_deriv = (
            best_of(lambda x=x, mm=mm: layered.efficiencies(x, mm, derivatives=True)) / count
            if x_l <= 100
            else float("nan")
        )
        t_snl = (
            best_of(
                lambda x=x, mm=mm: [
                    scattnlay.scattnlay(xi, mi) for xi, mi in zip(x, mm, strict=True)
                ]
            )
            / count
        )
        xs = np.full(count, x_l)
        t_sphere = best_of(lambda xs=xs: mie.efficiencies(xs, m[1])) / count
        rows.append(
            {
                "x": x_l,
                "sphere_us": t_sphere * 1e6,
                "scatrs_us": t_scatrs * 1e6,
                "scatrs_derivatives_us": t_deriv * 1e6,
                "scattnlay_us": t_snl * 1e6,
            }
        )
        print(rows[-1], flush=True)
    return rows


def markdown(report: dict) -> str:
    lines = [
        f"{report['machine']}; scatrs {report['versions']['scatrs']}, scattnlay "
        f"{report['versions']['scattnlay']}, {report['date']}; {report['accuracy']['cases']} "
        "cases of the reference suite inside the validated domain.",
        "",
        "| output | scatrs: largest error | scattnlay: cases > 1e-10, > 1e-6, failed |"
        " direct: cases > 1e-10, > 1e-6, failed |",
        "|---|---|---|---|",
    ]
    s = report["accuracy"]["summary"]

    def counts(c: dict) -> str:
        return f"{c['above_1e-10']}, {c['above_1e-6']}, {c['failed']}"

    for o in OUTPUTS:
        lines.append(
            f"| {o} | {s['scatrs'][o]['max_finite']:.1e} | {counts(s['scattnlay'][o])} | "
            f"{counts(s['direct'][o])} |"
        )
    lines += [
        "",
        "| x_L | scatrs (µs) | with derivatives (µs) | scattnlay (µs) | a sphere of x_L (µs) |",
        "|---|---|---|---|---|",
    ]
    for r in report["speed"]:
        lines.append(
            f"| {r['x']:g} | {r['scatrs_us']:.1f} | {r['scatrs_derivatives_us']:.1f} | "
            f"{r['scattnlay_us']:.1f} | {r['sphere_us']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="fewer cases, to check the script")
    args = p.parse_args()
    report = {
        "date": datetime.date.today().isoformat(),
        "machine": platform.platform(),
        "versions": {
            "scatrs": importlib.metadata.version("scatrs"),
            "scattnlay": importlib.metadata.version("scattnlay"),
            "python": platform.python_version(),
        },
        "accuracy": accuracy(args.quick),
        "speed": speed(args.quick),
    }
    stamp = f"{report['date']}-{platform.machine()}-scattnlay"
    path = Path(__file__).parent / "results" / f"{stamp}{'-quick' if args.quick else ''}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}\n")
    print(markdown(report))


if __name__ == "__main__":
    main()
