"""Reference records: certified double-double splits of Arb balls, and JSON Lines I/O.

Every stored value v is represented three ways:

* ``hi``/``lo``: two doubles with ``hi == fl(hi + lo)`` whose exact sum approximates v;
* ``dec``: v to 25 significant digits, for humans (tests use ``hi``/``lo``);
* ``err``: a rigorous upper bound on ``|v - (hi + lo)|``, including the Arb ball radius.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import flint
from flint import arb

FORMAT_VERSION = 1
DECIMAL_DIGITS = 25
LABELS = ("certified", "high-precision")


def nearest_double(v: arb) -> float:
    """The double nearest to the midpoint of ``v``."""
    mid = v.mid()
    f = float(mid)
    for cand in (math.nextafter(f, -math.inf), math.nextafter(f, math.inf)):
        if abs(mid - arb(cand)) < abs(mid - arb(f)):
            f = cand
    return f


def upper_double(x: arb) -> float:
    """The smallest double certainly >= every point of ``|x|``."""
    u = x.abs_upper()
    f = float(u)
    while not arb(f) >= u:
        f = math.nextafter(f, math.inf)
    return f


def split(v: arb) -> tuple[float, float, float]:
    """Split ``v`` into ``(hi, lo, err)`` with ``|v - (hi + lo)| <= err`` rigorously."""
    hi = nearest_double(v)
    lo = nearest_double(v - arb(hi))
    s = hi + lo
    if s != hi:  # renormalise (Fast2Sum) so that hi == fl(hi + lo)
        lo = lo - (s - hi)
        hi = s
    err = upper_double(v - arb(hi) - arb(lo))
    return hi, lo, err


def values(vs: Iterable[arb]) -> dict[str, list]:
    """Encode Arb balls as the parallel arrays stored in a record."""
    out: dict[str, list] = {"hi": [], "lo": [], "dec": [], "err": []}
    for v in vs:
        hi, lo, err = split(v)
        out["hi"].append(hi)
        out["lo"].append(lo)
        out["dec"].append(v.str(DECIMAL_DIGITS, radius=False))
        out["err"].append(err)
    return out


def record(
    generator: str,
    case: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, Iterable[arb]],
    label: str = "certified",
) -> dict:
    """Build one reference record."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
    return {
        "generator": generator,
        "case": case,
        "label": label,
        "inputs": dict(inputs),
        "outputs": {name: values(vs) for name, vs in outputs.items()},
    }


def write_jsonl(path: Path, records: Sequence[dict]) -> str:
    """Write records one per line with sorted keys and LF endings; return the SHA-256."""
    text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def update_manifest(
    data_dir: Path,
    filename: str,
    generator: str,
    precision_bits: int,
    n_records: int,
    sha256: str,
) -> None:
    """Insert or replace the manifest entry for one data file."""
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["format_version"] = FORMAT_VERSION
    manifest["python_flint"] = flint.__version__
    manifest["flint"] = flint.__FLINT_VERSION__
    files = manifest.get("files", {})
    files[filename] = {
        "generator": generator,
        "precision_bits": precision_bits,
        "records": n_records,
        "sha256": sha256,
    }
    manifest["files"] = dict(sorted(files.items()))
    path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
