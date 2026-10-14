"""Accuracy and speed of scatrs's ensembles against sasktran2's Mie integration.

Two workloads (docs/specs/2026-10-14-phase-2b-speed-and-sasktran2-design.md, section 7):

- W1, one broad lognormal at three wavelengths;
- W2, a database grid: 50 lognormals (10 median radii × 5 widths) at 20 wavelengths.

Both codes are compared with scatrs at rtol 1e-9 (the reference), and timed on one thread and
on all threads. sasktran2 runs as `MieDatabase` runs it (`integrate_mie_cpp`, 31-point panels,
quantile 0.99999), without its netCDF caching. Conventions: sasktran2 takes m = n − ik and
works in nm and m²; its β₁ and β₂ have the opposite sign to scatrs's; its `num_coeffs`
coefficients are the orders 0 to `num_coeffs` − 1.

Usage: uv run --locked --directory benchmarks python compare.py [--quick]
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

COEFFS = 64  # sasktran2's num_coeffs: orders 0..63
ORDER = COEFFS - 1
RTOLS = (3e-3, 1e-3, 3e-4, 1e-4)
REFERENCE_RTOL = 1e-9
REPEATS = 3
GREEK = ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")


def workloads(quick: bool) -> dict:
    """Each workload: lognormals (median radius µm, width), wavelengths µm, refractive index."""
    w2_radii = np.geomspace(0.05, 1.0, 3 if quick else 10)
    w2_widths = (1.2, 2.0) if quick else (1.2, 1.4, 1.6, 1.8, 2.0)
    return {
        "W1": {
            "description": "lognormal r_g = 1 µm, σ_g = 2, m = 1.33 + 1e-4i, λ = 0.3, 0.5, 1 µm",
            "lognormals": [(1.0, 2.0)],
            "wavelengths": [0.3, 0.5, 1.0],
            "m": complex(1.33, 1e-4),
        },
        "W2": {
            "description": (
                f"{len(w2_radii) * len(w2_widths)} lognormals (r_g {len(w2_radii)} values in "
                f"[0.05, 1] µm × σ_g in {list(w2_widths)}), {4 if quick else 20} wavelengths in "
                "[0.28, 1.02] µm, m = 1.43 + 1e-8i"
            ),
            "lognormals": [(float(r), s) for r in w2_radii for s in w2_widths],
            "wavelengths": list(np.linspace(0.28, 1.02, 4 if quick else 20)),
            "m": complex(1.43, 1e-8),
        },
    }


def best_of(f, repeats: int = REPEATS):
    """The shortest time of `repeats` calls of f, and its last result."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def run_scatrs(w: dict, rtol: float, max_panels: int = 4096):
    """scatrs's outputs as arrays (distribution, wavelength[, order])."""
    from scatrs import ensemble

    dists = [ensemble.LogNormal(r, s) for r, s in w["lognormals"]]
    r = ensemble.compute_batch(
        dists, w["wavelengths"], w["m"], greek=ORDER, rtol=rtol, max_panels=max_panels
    )
    out = {"c_ext": r.c_ext, "c_sca": r.c_sca, "converged": r.converged, "panels": r.panels}
    out.update({k: getattr(r.greek, k) for k in GREEK})
    return out


def run_sasktran2(w: dict, threads: int):
    """sasktran2's outputs in scatrs's units and conventions."""
    from sasktran2.mie.distribution import LogNormalDistribution, integrate_mie_cpp

    dists = [
        LogNormalDistribution().distribution(median_radius=1000 * r, mode_width=s)
        for r, s in w["lognormals"]
    ]
    m = w["m"]
    ds = integrate_mie_cpp(
        dists,
        lambda _: complex(m.real, -m.imag),
        1000 * np.asarray(w["wavelengths"]),
        num_coeffs=COEFFS,
        num_threads=threads,
    )
    # (wavelength, distribution, …) → (distribution, wavelength, …); m² → µm².
    out = {
        "c_ext": ds.xs_total.values.T * 1e12,
        "c_sca": ds.xs_scattering.values.T * 1e12,
    }
    for k, name in zip(GREEK, ("lm_a1", "lm_a2", "lm_a3", "lm_a4", "lm_b1", "lm_b2"), strict=True):
        sign = -1.0 if k.startswith("beta") else 1.0
        out[k] = sign * np.moveaxis(ds[name].values, 0, 1)
    return out


def metrics(out: dict, ref: dict) -> dict:
    """The largest errors: relative for the cross sections and the co-albedo, absolute for the
    single-scattering albedo and the normalized Greek coefficients (orders 0..ORDER)."""
    ssa, ssa_ref = out["c_sca"] / out["c_ext"], ref["c_sca"] / ref["c_ext"]
    m = {
        "c_ext": np.max(np.abs(out["c_ext"] / ref["c_ext"] - 1)),
        "c_sca": np.max(np.abs(out["c_sca"] / ref["c_sca"] - 1)),
        "ssa": np.max(np.abs(ssa - ssa_ref)),
        "coalbedo": np.max(np.abs((1 - ssa) / (1 - ssa_ref) - 1)),
    }
    for k in GREEK:
        m[k] = np.max(np.abs(out[k][..., : ORDER + 1] - ref[k][..., : ORDER + 1]))
    return {k: float(v) for k, v in m.items()}


# The metrics that decide matched accuracy, every one at most sasktran2's: the outputs whose
# accuracy rtol controls. ω and the co-albedo are reported, not matched: C_sca and C_ext are
# integrated on the same nodes in both codes, so their errors largely cancel in ω (sasktran2's ω
# errors in W2 are below 4e-7, by that cancellation rather than by resolving the integral), and
# for the weak absorbers of W2 (co-albedo ~1e-6) neither code resolves the co-albedo (scatrs
# refines C_abs to rtol of C_ext; sasktran2 takes C_ext − C_sca).
MATCHED = ("c_ext", "c_sca", *GREEK)


def child(args) -> None:
    """Times scatrs at every rtol in this process (its thread count is fixed at the first call)
    and saves the outputs."""
    w = workloads(args.quick)[args.workload]
    results = {}
    for rtol in RTOLS:
        t, out = best_of(lambda rtol=rtol: run_scatrs(w, rtol))
        results[str(rtol)] = {"time": t, "out": out}
    np.save(args.child, results, allow_pickle=True)


def scatrs_runs(name: str, quick: bool, threads: int | None) -> dict:
    """scatrs's times and outputs from a child process with `threads` rayon threads (all if
    None)."""
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    if threads is not None:
        env["RAYON_NUM_THREADS"] = str(threads)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.npy"
        cmd = [sys.executable, __file__, "--child", str(path), "--workload", name]
        subprocess.run([*cmd, *(["--quick"] if quick else [])], check=True, env=env)
        return np.load(path, allow_pickle=True).item()


def machine() -> str:
    if platform.system() == "Darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    else:
        brand = platform.processor() or platform.machine()
    return f"{brand}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="a smaller grid, to check the script")
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--workload", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        child(args)
        return
    report = {
        "date": datetime.date.today().isoformat(),
        "machine": machine(),
        "versions": {
            "scatrs": importlib.metadata.version("scatrs"),
            "sasktran2": importlib.metadata.version("sasktran2"),
            "python": platform.python_version(),
        },
        "orders": ORDER,
        "workloads": {},
    }
    for name, w in workloads(args.quick).items():
        print(f"{name}: {w['description']}", flush=True)
        t, ref = best_of(lambda w=w: run_scatrs(w, REFERENCE_RTOL, max_panels=400_000), 1)
        assert np.all(ref["converged"]), f"{name}: the reference did not converge"
        print(f"  reference (rtol {REFERENCE_RTOL:g}): {t:.0f} s", flush=True)
        sk = {}
        for threads, label in ((1, "one"), (0, "all")):
            sk[label], out = best_of(lambda threads=threads, w=w: run_sasktran2(w, threads))
        sk_metrics = metrics(out, ref)
        entry = {
            "description": w["description"],
            "sasktran2": {"time_one": sk["one"], "time_all": sk["all"], "errors": sk_metrics},
            "scatrs": {},
        }
        runs = {label: scatrs_runs(name, args.quick, n) for label, n in (("one", 1), ("all", None))}
        for rtol in RTOLS:
            out = runs["all"][str(rtol)]["out"]
            entry["scatrs"][str(rtol)] = {
                "time_one": runs["one"][str(rtol)]["time"],
                "time_all": runs["all"][str(rtol)]["time"],
                "panels": int(np.max(out["panels"])),
                "converged": bool(np.all(out["converged"])),
                "errors": metrics(out, ref),
            }
        matched = [
            r
            for r in RTOLS
            if all(entry["scatrs"][str(r)]["errors"][k] <= sk_metrics[k] for k in MATCHED)
        ]
        entry["matched_rtol"] = max(matched) if matched else None
        report["workloads"][name] = entry
    stamp = f"{report['date']}-{platform.machine()}"
    path = Path(__file__).parent / "results" / f"{stamp}{'-quick' if args.quick else ''}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}\n")
    print(markdown(report))


def markdown(report: dict) -> str:
    """The tables of docs/validation/sasktran2.md."""
    lines = [
        f"{report['machine']}; scatrs {report['versions']['scatrs']}, sasktran2 "
        f"{report['versions']['sasktran2']}, {report['date']}.",
        "",
    ]
    cols = ("c_ext", "c_sca", "ssa", "coalbedo", "alpha1", "beta1")
    for name, e in report["workloads"].items():
        lines += [f"**{name}**: {e['description']}.", ""]
        lines.append("| | one thread | all threads | C_ext | C_sca | ω | co-albedo | α₁ | β₁ |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        s = e["sasktran2"]
        err = " | ".join(f"{s['errors'][c]:.1e}" for c in cols)
        lines.append(f"| sasktran2 | {s['time_one']:.3g} s | {s['time_all']:.3g} s | {err} |")
        for rtol, r in e["scatrs"].items():
            err = " | ".join(f"{r['errors'][c]:.1e}" for c in cols)
            mark = " (matched)" if float(rtol) == e["matched_rtol"] else ""
            lines.append(
                f"| scatrs rtol {float(rtol):g}{mark} | {r['time_one']:.3g} s | "
                f"{r['time_all']:.3g} s | {err} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
