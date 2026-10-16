"""scatrs.tmatrix against Mishchenko's T-matrix code for randomly oriented spheroids.

Mishchenko's extended-precision code (``tmq.lp.f``, ``lpq.f``, ``tmq.par.f``, REAL*16) is not
redistributed here: it is fetched from its page at NASA GISS into ``tmatd/src`` (not committed),
and built with gfortran after two patches, which change no computation:

* the input block reads the particle from standard input: a monodisperse spheroid as the code's
  own notes describe (NDISTR = 4, NKMAX = -1, R1 and R2 within 10⁻⁷ of the radius), with
  RAT = 1 (equal-volume radius), LAM = 2π so that the radius is the size parameter, and the
  convergence tolerance DDELT;
* the output formats print 16 digits instead of 6 or 7, and the sample output that opens the
  file is dropped;
* the vendor-specific quad-precision intrinsics (``QCONJG``, ``QSQRT``, ``QFLOAT``, …), which
  gfortran does not have, become their generic equivalents at kind 16.

For each case it compares Q_ext, Q_sca, g and the normalized Greek coefficients with scatrs at
``rtol`` = 10⁻¹⁰, and times both (tmatd at DDELT = 10⁻⁶ and 10⁻⁹, scatrs at 10⁻⁶ and 10⁻⁸).
tmatd's β₁ and β₂ are compared with their signs reversed: it expands in the generalized
spherical functions P⁰²ₗ = −d⁰²ₗ (Mishchenko, Travis & Lacis 2002, Appendix B), scatrs in the
Wigner d-functions.

Usage: uv run --directory benchmarks python compare_tmatd.py [--quick] [--threads N]

scatrs runs on one thread by default, as tmatd does.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

# scatrs on one thread unless asked (`--threads N`), as tmatd runs: set before rayon starts.
_THREADS = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1"
os.environ["RAYON_NUM_THREADS"] = _THREADS

import scatrs  # noqa: E402
from scatrs import tmatrix  # noqa: E402

HERE = Path(__file__).parent
SRC = HERE / "tmatd" / "src"
BUILD = HERE / "tmatd" / "build"
URL = "https://www.giss.nasa.gov/staff/mmishchenko/ftpcode/"
FILES = {
    "tmq.lp.f": "612718d332fc3405a0fc50f717f543c8948637162bd994cdd633a5ee6955a66c",
    "lpq.f": "5de1eb42504ea3058b3afe10e43d64ef75ea84d999379c213854cb29cfcd6d65",
    "tmq.par.f": "9ecb43b79b985d182b0c794194d092669be522f88970ca7553f60a980a4138fb",
}

# The input block, from its first assignment to its last (its lines end in spaces).
INPUT_BLOCK = re.compile(r"^      RAT=0\.5D0\n.*?^      NDGS=2 *\n", re.MULTILINE | re.DOTALL)
NEW_INPUT = """      RAT=1D0
      NDISTR=4
      NPNAX=1
      B=1D-1
      GAM=0.5D0
      NKMAX=-1
      NP=-1
      NPNA=19
      LAM=2D0*DACOS(-1D0)
      READ (5,*) AXMAX,EPS,MRR,MRI,DDELT,NDGS
"""
# The radii of the size distribution, which the notes say to replace for one particle.
RADII = re.compile(
    r"^         R1=0\.89031D0\*AXI *\n^         R2=1\.56538D0\*AXI *\n", re.MULTILINE
)
NEW_RADII = """         R1=0.9999999D0*AXI
         R2=1.0000001D0*AXI
"""


#: Vendor quad-precision intrinsics and their generic gfortran forms.
QUAD_INTRINSICS = [
    (r"\bQCONJG\b", "CONJG"),
    (r"\bQFLOAT\(", "REAL(KIND=16,A="),
    (r"\bQSQRT\b", "SQRT"),
    (r"\bQCOS\b", "COS"),
    (r"\bQSIN\b", "SIN"),
    (r"\bQABS\b", "ABS"),
    (r"\bQMAX1\b", "MAX"),
    (r"\bQARCOS\b", "ACOS"),
    (r"\bQSINH\b", "SINH"),
    (r"\bQCOSH\b", "COSH"),
    (r"\bQIMAG\b", "AIMAG"),
    (r"\bQATAN\b", "ATAN"),
    (r"QCMPLX\(QR\(I,J\),QI\(I,J\)\)", "CMPLX(QR(I,J),QI(I,J),KIND=16)"),
    # The Q edit descriptor of formats.
    (r",Q(\d+\.\d+)\)", r",E\1)"),
]


def generic(text: str) -> str:
    for pattern, new in QUAD_INTRINSICS:
        text = re.sub(pattern, new, text, flags=re.IGNORECASE)
    return text


def fetch() -> None:
    import hashlib

    SRC.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = SRC / name
        if not path.exists():
            with urllib.request.urlopen(URL + name, timeout=60) as r:
                path.write_bytes(r.read())
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            raise RuntimeError(f"{name}: SHA-256 {got} differs from the one compared ({digest})")


def build() -> Path:
    exe = BUILD / "tmq"
    if exe.exists():
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    text = (SRC / "tmq.lp.f").read_text()
    # The file opens with a sample of the code's output: the Fortran starts at its notes.
    text = text[text.index("C   New release including the LAPACK") :]
    text, count = INPUT_BLOCK.subn(NEW_INPUT, text)
    text, count2 = RADII.subn(NEW_RADII, text)
    if count != 1 or count2 != 1:
        raise RuntimeError("tmq.lp.f differs from the version the patches were written for")
    text = text.replace(
        "9100    FORMAT('CEXT=',D12.6,2X,'CSCA=',D12.6,2X,",
        "9100    FORMAT('CEXT=',D24.16,2X,'CSCA=',D24.16,2X,",
    )
    text = text.replace(
        "     &          2X,'W=',D12.6,2X,'<COS>=',D12.6)",
        "     &          2X,'W=',D24.16,2X,'<COS>=',D24.16)",
    )
    text = text.replace("  575    FORMAT(6D14.7)", "  575    FORMAT(6D25.16)")
    (BUILD / "tmq.lp.f").write_text(generic(text))
    (BUILD / "lpq.f").write_text(generic((SRC / "lpq.f").read_text()))
    shutil.copy(SRC / "tmq.par.f", BUILD / "tmq.par.f")
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", "-o", "tmq", "tmq.lp.f", "lpq.f"],
        cwd=BUILD,
        check=True,
    )
    return exe


def run_tmatd(exe: Path, x: float, eps: float, m: complex, ddelt: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        subprocess.run(
            [str(exe)],
            input=f"{x!r} {eps!r} {m.real!r} {m.imag!r} {ddelt!r} 2\n",
            text=True,
            cwd=tmp,
            check=True,
            capture_output=True,
            timeout=3600,
        )
        elapsed = time.perf_counter() - t0
        out = (Path(tmp) / "test").read_text()
        coeffs = (Path(tmp) / "tmatr.write").read_text().split("\n")
    m_ = re.search(r"CEXT=\s*(\S+)\s+CSCA=\s*(\S+)\s+W=\s*(\S+)\s+<COS>=\s*(\S+)", out)
    if m_ is None:
        raise RuntimeError(f"tmatd failed:\n{out[-2000:]}")
    num = [float(v.replace("D", "E")) for v in m_.groups()]
    area = math.pi * x * x
    rows = [
        [float(v.replace("D", "E")) for v in line.split()] for line in coeffs[1:] if line.strip()
    ]
    greek = np.array(rows).T  # alpha1..4, beta1, beta2 by order
    # tmatd expands F12 and F34 in the generalized spherical functions P⁰²ₗ = −d⁰²ₗ, scatrs in
    # the Wigner d-functions: the same phase matrix has β₁ and β₂ of opposite signs.
    greek[4:] *= -1
    return {
        "q_ext": num[0] / area,
        "q_sca": num[1] / area,
        "g": num[3],
        "greek": greek,
        "seconds": elapsed,
        "converged": "WARNING" not in out,
    }


def run_scatrs(x: float, eps: float, m: complex, rtol: float) -> dict:
    t0 = time.perf_counter()
    t = tmatrix.compute(x, m, eps, rtol=rtol, allow_unvalidated=True)
    q, g, _ = t.random_orientation()
    elapsed = time.perf_counter() - t0
    return {
        "q_ext": float(q.q_ext),
        "q_sca": float(q.q_sca),
        "g": float(q.g),
        "greek": np.array([g.alpha1, g.alpha2, g.alpha3, g.alpha4, g.beta1, g.beta2]),
        "seconds": elapsed,
        "estimate": t.error_estimate,
    }


CASES = [
    (x, eps, m)
    for eps in (0.2, 1 / 3, 0.5, 2.0, 3.0, 5.0)
    for x in (1.0, 3.0, 6.0, 10.0)
    for m in (1.33 + 0j, 1.53 + 0.008j, 1.5 + 0.5j)
]


def main() -> None:
    quick = "--quick" in sys.argv
    fetch()
    exe = build()
    cases = CASES[::7] if quick else CASES
    results = []
    for x, eps, m in cases:
        ref = run_scatrs(x, eps, m, 1e-10)
        fast = run_scatrs(x, eps, m, 1e-6)
        mid = run_scatrs(x, eps, m, 1e-8)
        row = {
            "x": x,
            "aspect_ratio": eps,
            "m": [m.real, m.imag],
            "scatrs_estimate": ref["estimate"],
        }
        for ddelt in (1e-6, 1e-9):
            try:
                tm = run_tmatd(exe, x, eps, m, ddelt)
            except (RuntimeError, subprocess.SubprocessError) as e:
                row[f"tmatd_{ddelt:g}"] = {"error": str(e)[:300]}
                continue
            n = min(tm["greek"].shape[1], ref["greek"].shape[1])
            big = np.abs(ref["greek"][0]).max()
            greek_diff = float(np.abs(tm["greek"][:, :n] - ref["greek"][:, :n]).max() / big)
            tail = (
                float(np.abs(tm["greek"][:, n:]).max() / big) if tm["greek"].shape[1] > n else 0.0
            )
            row[f"tmatd_{ddelt:g}"] = {
                "q_ext": abs(tm["q_ext"] / ref["q_ext"] - 1),
                "q_sca": abs(tm["q_sca"] / ref["q_sca"] - 1),
                "g": abs(tm["g"] - ref["g"]),
                "greek": max(greek_diff, tail),
                "seconds": tm["seconds"],
                "converged": tm["converged"],
            }
        row["scatrs_seconds"] = {"1e-6": fast["seconds"], "1e-8": mid["seconds"]}
        row["scatrs_1e-6_q_ext"] = abs(fast["q_ext"] / ref["q_ext"] - 1)
        results.append(row)
        print(json.dumps(row), flush=True)
    out = HERE / "results" / f"{time.strftime('%Y-%m-%d')}-{platform.machine()}-tmatd.json"
    meta = {
        "scatrs": scatrs.__version__,
        "tmatd": {k: v for k, v in FILES.items()},
        "machine": platform.machine(),
        "scatrs_threads": int(_THREADS),
        "processor": platform.processor(),
        "gfortran": subprocess.run(
            ["gfortran", "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
    }
    out.write_text(json.dumps({"meta": meta, "cases": results}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
