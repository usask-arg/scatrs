import hashlib
import json
from pathlib import Path

import pytest
from flint import arb, ctx

from scatrs_reference.__main__ import main
from scatrs_reference.generators import GENERATORS, constants
from scatrs_reference.records import record, write_jsonl

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
            n = len(out["hi"])
            assert all(len(out[k]) == n for k in ("lo", "lo2", "dec", "err"))


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
    with ctx.workprec(10):
        assert main(["generate", "--out", str(tmp_path), "--only", "constants"]) == 0
    assert (tmp_path / "constants.jsonl").read_bytes() == (DATA / "constants.jsonl").read_bytes()
    assert set(json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]) == {
        "constants.jsonl"
    }


def test_cli_resets_the_flint_context_before_each_generator(tmp_path, monkeypatch):
    seen = {}

    def make(name, leaked_prec):
        def generate():
            seen[name] = ctx.prec
            ctx.prec = leaked_prec  # a careless generator that leaks its precision
            return ctx.prec, [record(name, "c", {}, {"value": [arb(1)]})]

        return generate

    monkeypatch.setitem(GENERATORS, "dummy_a", make("dummy_a", 77))
    monkeypatch.setitem(GENERATORS, "dummy_b", make("dummy_b", 91))
    with ctx.workprec(10):  # the CLI is entered with a dirty context
        assert main(["generate", "--out", str(tmp_path), "--only", "dummy_a", "dummy_b"]) == 0
    assert seen == {"dummy_a": 53, "dummy_b": 53}
    assert (tmp_path / "dummy_a.jsonl").exists() and (tmp_path / "dummy_b.jsonl").exists()
