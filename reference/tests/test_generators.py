import hashlib
import json
from pathlib import Path

import pytest

from scatrs_reference.generators import GENERATORS, constants
from scatrs_reference.records import write_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_committed_file_matches_manifest(name):
    manifest = json.loads((DATA / "manifest.json").read_text())
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
    assert sha == json.loads((DATA / "manifest.json").read_text())["files"]["constants.jsonl"]["sha256"]


def test_constants_values():
    _, recs = constants.generate()
    by_case = {r["case"]: r["outputs"]["value"] for r in recs}
    assert by_case["PI"]["hi"] == [3.141592653589793]
    assert by_case["INV_FACT_3"]["hi"] == [1 / 6]
    assert by_case["FRAC_PI_2_TAIL"]["hi"] == [-1.4973849048591698e-33]
    assert by_case["LN_2_TAIL"]["hi"] == [5.707708438416212e-34]
    assert by_case["POW10_2POW5"]["lo"] == [-5366162204393472.0]
