"""Reference records: certified double-double splits of Arb balls, and JSON Lines I/O.

Every stored value v is represented three ways:

* ``hi``/``lo``/``lo2``: three doubles whose exact sum approximates v to about 160 bits.
  ``hi`` is the double nearest to the ball midpoint and ``hi == fl(hi + lo)``, so
  ``(hi, lo)`` is a normalised double-double; ``lo2`` is the double nearest to the
  remainder, which lets tests measure double-double errors almost exactly;
* ``dec``: v as printed by Arb with up to 25 significant digits, or fewer when the ball is
  too wide to justify 25 (for humans; tests use the doubles);
* ``err``: a rigorous upper bound on ``|v - (hi + lo + lo2)|``, including the Arb ball
  radius.

Values that are not finite, or whose double representation overflows, are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import flint
from flint import arb

FORMAT_VERSION = 2  # 2: added lo2
DECIMAL_DIGITS = 25
LABELS = ("certified", "high-precision")


def nearest_double(v: arb) -> float:
    """The double nearest to the midpoint of ``v``.

    Raises ``ValueError`` if the midpoint is NaN, and ``OverflowError`` if it is infinite
    or too large in magnitude to round to a finite double.
    """
    mid = v.mid()
    f = float(mid)
    if math.isnan(f):
        raise ValueError(f"midpoint of {v.str(15)} is NaN")
    if math.isinf(f):
        raise OverflowError(f"midpoint of {v.str(15)} does not fit in a finite double")
    for cand in (math.nextafter(f, -math.inf), math.nextafter(f, math.inf)):
        if abs(mid - arb(cand)) < abs(mid - arb(f)):
            f = cand
    return f


def upper_double(x: arb) -> float:
    """The smallest double certainly >= every point of ``|x|``.

    Raises ``ValueError`` if the bound is NaN, and ``OverflowError`` if it is infinite or
    exceeds the largest finite double.
    """
    u = x.abs_upper()
    f = float(u)
    if math.isnan(f):
        raise ValueError(f"upper bound of |{x.str(15)}| is NaN")
    if math.isinf(f):
        raise OverflowError(f"upper bound of |{x.str(15)}| does not fit in a finite double")
    while not arb(f) >= u:
        f = math.nextafter(f, math.inf)
    if math.isinf(f):  # u lies just above the largest finite double
        raise OverflowError(f"upper bound of |{x.str(15)}| does not fit in a finite double")
    return f


def split(v: arb) -> tuple[float, float, float, float]:
    """Split ``v`` into ``(hi, lo, lo2, err)`` with ``|v - (hi + lo + lo2)| <= err`` rigorously.

    ``hi`` is the double nearest to the midpoint of ``v`` and ``hi + lo == hi`` in
    floating point, so ``(hi, lo)`` is a normalised double-double. ``lo2`` is the double
    nearest to the midpoint of ``v - hi - lo``: ``|lo2| <= ulp(lo)/2``, or ``<= 2 ulp(lo)``
    when ``lo`` was moved towards zero to keep ``hi + lo == hi``. It may underflow; ``err``
    covers that. Raises ``ValueError`` if ``v`` is not finite (NaN or infinite midpoint or
    radius), and ``OverflowError`` if ``hi``, ``lo``, ``lo2`` or ``err`` does not fit in a
    finite double.
    """
    if not v.is_finite():
        raise ValueError(f"cannot split non-finite value {v.str(15)}")
    hi = nearest_double(v)
    lo = nearest_double(v - arb(hi))
    if hi + lo != hi:  # |lo| is exactly half an ulp of hi and rounds away from hi
        lo = math.nextafter(lo, 0.0)
    rest = v - arb(hi) - arb(lo)
    lo2 = nearest_double(rest)
    err = upper_double(rest - arb(lo2))
    if not all(map(math.isfinite, (hi, lo, lo2, err))):  # unreachable; keep the contract
        raise OverflowError(f"{v.str(15)} does not split into finite doubles")
    return hi, lo, lo2, err


def values(vs: Iterable[arb]) -> dict[str, list]:
    """Encode Arb balls as the parallel arrays stored in a record."""
    out: dict[str, list] = {"hi": [], "lo": [], "lo2": [], "dec": [], "err": []}
    for v in vs:
        hi, lo, lo2, err = split(v)
        out["hi"].append(hi)
        out["lo"].append(lo)
        out["lo2"].append(lo2)
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
    """Insert or replace the manifest entry for one data file.

    Each entry records the python-flint and FLINT versions that produced that file, so
    regenerating a subset of files never misattributes the others.
    """
    path = data_dir / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    files = old.get("files", {})
    files[filename] = {
        "flint": flint.__FLINT_VERSION__,
        "generator": generator,
        "precision_bits": precision_bits,
        "python_flint": flint.__version__,
        "records": n_records,
        "sha256": sha256,
    }
    manifest = {"files": dict(sorted(files.items())), "format_version": FORMAT_VERSION}
    path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
