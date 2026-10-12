import json
import math

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
    ctx.prec = 400


def test_split_is_normalised_and_certified():
    v = arb.pi()
    hi, lo, err = split(v)
    assert hi == math.pi
    assert hi + lo == hi
    assert err <= 2.0**-105 * abs(hi)
    assert abs(v - arb(hi) - arb(lo)) <= arb(err)


def test_split_of_a_double_is_exact():
    assert split(arb(0.1)) == (0.1, 0.0, 0.0)


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
    assert set(v) == {"hi", "lo", "dec", "err"}
    assert v["hi"] == [1 / 3, 2.0]
    assert v["dec"][1].startswith("2")


def test_write_jsonl_is_deterministic_and_updates_manifest(tmp_path):
    recs = [record("g", "c", {}, {"value": [arb(1) / 7]})]
    sha1 = write_jsonl(tmp_path / "g.jsonl", recs)
    sha2 = write_jsonl(tmp_path / "g.jsonl", recs)
    assert sha1 == sha2
    assert (tmp_path / "g.jsonl").read_bytes().count(b"\r") == 0
    update_manifest(tmp_path, "g.jsonl", "g", 400, 1, sha1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["files"]["g.jsonl"] == {
        "generator": "g",
        "precision_bits": 400,
        "records": 1,
        "sha256": sha1,
    }
    assert manifest["python_flint"] == "0.9.0"
