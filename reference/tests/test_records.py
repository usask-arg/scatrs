import json
import math
import sys

import flint
import pytest
from flint import arb, ctx

from scatrs_reference.records import (
    nearest_double,
    record,
    split,
    update_manifest,
    upper_double,
    write_jsonl,
)


@pytest.fixture(autouse=True)
def _precision():
    previous = ctx.prec
    ctx.prec = 400
    yield
    ctx.prec = previous


def _near_tie():
    # Just below the midpoint between 1 + 2**-52 and 1 + 2**-51: the nearest double is
    # 1 + 2**-52, but RN(v - hi) == 2**-53 would make fl(hi + lo) round up to 1 + 2**-51.
    return arb(1) + arb(2) ** -52 + arb(2) ** -53 - arb(2) ** -200


def _pi_ball(radius):
    return arb(arb.pi().mid(), radius)


CASES = {
    "minus_pi": lambda: -arb.pi(),
    "pi_plus_minus_1e-20": lambda: _pi_ball(1e-20),
    "zero_ball_1e-50": lambda: arb(0, 1e-50),
    "subnormal_3.7": lambda: arb(3.7) * arb(2) ** -1074,
    "min_normal_over_3": lambda: arb(2) ** -1022 / 3,
    "near_tie": _near_tie,
}


def _lo2_bound(v, hi, lo):
    """|lo2| <= ulp(lo)/2, or 2 ulp(lo) when lo was moved towards zero (the tie case)."""
    moved = lo != nearest_double(v - arb(hi))
    return (2.0 if moved else 0.5) * math.ulp(lo)


def test_split_is_normalised_and_certified():
    v = arb.pi()
    hi, lo, lo2, err = split(v)
    assert hi == math.pi
    assert hi + lo == hi
    assert abs(lo2) <= math.ulp(lo) / 2
    assert lo2 != 0.0
    assert err <= 2.0**-158 * abs(hi)  # about 160 bits, not just a double-double
    assert abs(v - arb(hi) - arb(lo) - arb(lo2)) <= arb(err)


@pytest.mark.parametrize("name", CASES)
def test_split_is_certified_normalised_and_nearest(name):
    v = CASES[name]()
    hi, lo, lo2, err = split(v)
    assert abs(v - arb(hi) - arb(lo) - arb(lo2)) <= arb(err)
    assert hi + lo == hi
    assert hi == nearest_double(v)
    assert abs(lo2) <= _lo2_bound(v, hi, lo)
    assert lo2 == nearest_double(v - arb(hi) - arb(lo))


@pytest.mark.parametrize("radius", [1e-20, 1e-50])
def test_split_error_covers_the_ball_radius(radius):
    _, _, _, err = split(_pi_ball(radius))
    assert err >= radius


def test_split_keeps_hi_nearest_on_a_near_tie():
    hi, lo, lo2, _ = split(_near_tie())
    assert hi == 1.0 + 2.0**-52
    assert lo == math.nextafter(2.0**-53, 0.0)
    # v - hi - lo = 2**-106 - 2**-200: one ulp of lo, which the third word recovers.
    assert lo2 == 2.0**-106 == math.ulp(lo)


def test_split_of_a_double_is_exact():
    assert split(arb(0.1)) == (0.1, 0.0, 0.0, 0.0)


def test_split_hi_lo_do_not_depend_on_the_third_word():
    for make in CASES.values():
        v = make()
        hi, lo, _, _ = split(v)
        expected_lo = nearest_double(v - arb(hi))
        if hi + expected_lo != hi:
            expected_lo = math.nextafter(expected_lo, 0.0)
        assert (hi, lo) == (nearest_double(v), expected_lo)


@pytest.mark.parametrize(
    "make",
    [
        lambda: arb("nan"),
        lambda: arb(1) / arb(0),
        lambda: arb("inf"),
        lambda: arb("-inf"),
    ],
    ids=["nan", "division_by_zero", "inf", "minus_inf"],
)
def test_split_rejects_non_finite_values(make):
    with pytest.raises(ValueError, match="non-finite"):
        split(make())


MAX = sys.float_info.max


def just_above_max():
    """Exactly MAX + 2**960: it rounds to MAX, yet no finite double is >= it."""
    return arb(MAX) + arb(2) ** 960  # evaluated under the 400-bit test precision, so exact


@pytest.mark.parametrize(
    "make",
    [
        lambda: arb(2) ** 1024,
        lambda: -(arb(2) ** 1024),
        lambda: arb(10) ** 400,
        lambda: arb(0, arb(2) ** 1024),
    ],
    ids=["2^1024", "-2^1024", "10^400", "huge_radius"],
)
def test_split_rejects_values_that_overflow_a_double(make):
    with pytest.raises(OverflowError):
        split(make())


def test_split_accepts_the_largest_finite_double():
    assert split(arb(MAX)) == (MAX, 0.0, 0.0, 0.0)
    # Representable as the pair (MAX, 2**960) even though no double is >= the sum.
    assert split(just_above_max()) == (MAX, 2.0**960, 0.0, 0.0)


def test_nearest_double_exceptions():
    with pytest.raises(ValueError):
        nearest_double(arb("nan"))
    with pytest.raises(ValueError):
        nearest_double(arb(1) / arb(0))  # NaN midpoint, infinite radius
    for make in (lambda: arb("inf"), lambda: arb("-inf"), lambda: arb(2) ** 1024):
        with pytest.raises(OverflowError):
            nearest_double(make())
    assert nearest_double(just_above_max()) == MAX  # rounds to the largest finite double


def test_upper_double_exceptions():
    with pytest.raises(ValueError):
        upper_double(arb("nan"))
    for make in (lambda: arb("inf"), lambda: arb(2) ** 1024, just_above_max):
        with pytest.raises(OverflowError):
            upper_double(make())
    assert upper_double(arb(MAX)) == MAX


def test_nearest_double_and_upper_double():
    third = arb(1) / 3
    assert nearest_double(third) == 1 / 3
    u = upper_double(third)
    assert arb(u) >= third
    assert u == math.nextafter(1 / 3, math.inf)


def test_record_rejects_unknown_label():
    with pytest.raises(ValueError):
        record("g", "c", {}, {}, label="bogus")


def test_record_layout():
    r = record("g", "c", {"n": 3}, {"value": [arb(1) / 3, arb(2)]})
    assert r["label"] == "certified"
    v = r["outputs"]["value"]
    assert set(v) == {"hi", "lo", "lo2", "dec", "err"}
    assert v["hi"] == [1 / 3, 2.0]
    assert v["dec"][1].startswith("2")


def test_write_jsonl_is_deterministic_and_updates_manifest(tmp_path):
    recs = [record("g", "c", {}, {"value": [arb(1) / 7]})]
    sha1 = write_jsonl(tmp_path / "g.jsonl", recs)
    sha2 = write_jsonl(tmp_path / "g.jsonl", recs)
    assert sha1 == sha2
    assert (tmp_path / "g.jsonl").read_bytes().count(b"\r") == 0
    update_manifest(tmp_path, "g.jsonl", "g", 400, 1, sha1)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"files", "format_version"}
    assert manifest["files"]["g.jsonl"] == {
        "flint": flint.__FLINT_VERSION__,
        "generator": "g",
        "precision_bits": 400,
        "python_flint": "0.9.0",
        "records": 1,
        "sha256": sha1,
    }


def test_manifest_versions_are_per_file(tmp_path):
    update_manifest(tmp_path, "a.jsonl", "a", 400, 1, "aa")
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["a.jsonl"]["python_flint"] = "0.0.0-older"
    manifest["python_flint"] = "0.0.0-legacy"  # a legacy top-level field is dropped on update
    path.write_text(json.dumps(manifest), encoding="utf-8")
    update_manifest(tmp_path, "b.jsonl", "b", 400, 2, "bb")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifest) == {"files", "format_version"}
    assert list(manifest["files"]) == ["a.jsonl", "b.jsonl"]
    assert manifest["files"]["a.jsonl"]["python_flint"] == "0.0.0-older"
    assert manifest["files"]["b.jsonl"]["python_flint"] == flint.__version__
