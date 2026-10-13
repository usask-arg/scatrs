import hashlib
import json
import math
from pathlib import Path

import pytest
from flint import arb, ctx

from scatrs_reference.__main__ import main
from scatrs_reference.generators import GENERATORS, constants, dd_trig, wigner_d
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


@pytest.mark.parametrize("e", range(0, 17))
def test_hardest_reduction_cases_match_brute_force(e):
    # The lattice search must find the double of [2^e, 2^(e+1)) closest to a multiple of
    # π/2; for small binades every multiple k·π/2 can be checked directly.
    with ctx.workprec(dd_trig.SEARCH_BITS + 64):
        half_pi = arb.pi() / 2
        k_lo = int((arb(2.0**e) / half_pi).floor().unique_fmpz())
        k_hi = int((arb(2.0 ** (e + 1)) / half_pi).floor().unique_fmpz()) + 1
        best = None
        for k in range(max(1, k_lo), k_hi + 1):
            v = k * half_pi
            x = float(v.mid())
            for cand in (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf)):
                if 2.0**e <= cand < 2.0 ** (e + 1):
                    d = abs(arb(cand) - v)
                    if best is None or d < best[0]:
                        best = (d, cand, k)
    x, k = dd_trig.closest_to_multiple_of_half_pi(e)
    assert (x, k) == (best[1], best[2])


def test_gauss_legendre_nodes_ascend_and_are_symmetric():
    lines = (DATA / "gauss_legendre.jsonl").read_text().splitlines()
    for line in lines:
        rec = json.loads(line)
        hi = rec["outputs"]["nodes"]["hi"]
        assert hi == sorted(hi)
        assert hi == [-x for x in reversed(hi)]


@pytest.mark.parametrize("m,n", [(0, 0), (0, 2)])
@pytest.mark.parametrize("x", [-0.95, 0.3, 0.9999])
def test_wigner_d_from_legendre_matches_the_explicit_sum(m, n, x):
    with ctx.workprec(400):
        for l in range(2, 40):
            a, b = wigner_d.d_legendre(l, m, n, x), wigner_d.d_ref(l, m, n, x)
            assert (a - b).abs_upper() <= arb(2) ** -300, (l, a, b)
