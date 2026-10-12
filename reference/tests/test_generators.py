import hashlib
import json
from pathlib import Path

import pytest
from flint import ctx

from scatrs_reference.__main__ import main
from scatrs_reference.generators import GENERATORS, constants
from scatrs_reference.records import write_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"


def _manifest():
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_lists_exactly_the_registered_generators():
    assert set(_manifest()["files"]) == {f"{n}.jsonl" for n in GENERATORS}
    assert {p.name for p in DATA.glob("*.jsonl")} == {f"{n}.jsonl" for n in GENERATORS}


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_committed_file_matches_manifest(name):
    manifest = _manifest()
    entry = manifest["files"][f"{name}.jsonl"]
    content = (DATA / f"{name}.jsonl").read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    lines = content.decode().splitlines()
    assert len(lines) == entry["records"]
    for line in lines:
        rec = json.loads(line)
        assert rec["label"] in ("certified", "high-precision")
        for out in rec["outputs"].values():
            assert len(out["hi"]) == len(out["lo"]) == len(out["dec"]) == len(out["err"])


def test_constants_regenerate_identically(tmp_path):
    _, recs = constants.generate()
    sha = write_jsonl(tmp_path / "constants.jsonl", recs)
    assert (tmp_path / "constants.jsonl").read_bytes() == (DATA / "constants.jsonl").read_bytes()
    assert sha == _manifest()["files"]["constants.jsonl"]["sha256"]


def test_generate_does_not_leak_the_working_precision():
    before = ctx.prec
    constants.generate()
    assert ctx.prec == before


def test_cli_generates_the_committed_data_from_a_dirty_context(tmp_path):
    with ctx.workprec(10):  # main() must reset the context before running a generator
        assert main(["generate", "--out", str(tmp_path), "--only", "constants"]) == 0
    assert (tmp_path / "constants.jsonl").read_bytes() == (DATA / "constants.jsonl").read_bytes()
    assert set(json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]) == {
        "constants.jsonl"
    }


def test_constants_values():
    _, recs = constants.generate()
    by_case = {r["case"]: r["outputs"]["value"] for r in recs}
    assert by_case["PI"]["hi"] == [3.141592653589793]
    assert by_case["INV_FACT_3"]["hi"] == [1 / 6]
    assert by_case["FRAC_PI_2_TAIL"]["hi"] == [-1.4973849048591698e-33]
    assert by_case["LN_2_TAIL"]["hi"] == [5.707708438416212e-34]
    assert by_case["POW10_2POW5"]["lo"] == [-5366162204393472.0]
