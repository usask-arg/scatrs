import json
from pathlib import Path

from flint import arb, ctx

from scatrs_reference.generators import mie_greek, wigner_d

DATA = Path(__file__).resolve().parent.parent / "data"


def test_wigner_columns_match_the_explicit_sum():
    with ctx.workprec(200):
        for m, n in ((0, 0), (2, 2), (2, -2), (0, 2)):
            for x in (-0.95, -0.3, 0.0, 0.4, 0.999):
                col = mie_greek.wigner_column(m, n, arb(x), 30)
                for l in range(31):
                    assert abs(col[l] - wigner_d.d_ref(l, m, n, x)) < arb(2) ** -150, (m, n, x, l)


def _value(out, i):
    return out["hi"][i] + out["lo"][i] + out["lo2"][i]


def test_records_satisfy_the_identities():
    records = [json.loads(line) for line in (DATA / "mie_greek.jsonl").read_text().splitlines()]
    greek = {r["case"].removesuffix(",greek"): r for r in records if r["case"].endswith(",greek")}
    phase = {r["case"].removesuffix(",phase"): r for r in records if r["case"].endswith(",phase")}
    assert greek.keys() == phase.keys() and len(greek) == len(mie_greek.CASES)
    for case, g in greek.items():
        a1 = g["outputs"]["alpha1"]
        assert _value(a1, 0) == 1.0, case
        assert g["label"] == phase[case]["label"] == "certified"
        assert len(a1["hi"]) == g["inputs"]["order"] + 1
        if g["inputs"]["order"] == 2 * g["inputs"]["n_terms"]:
            # All orders: F11(1) is the sum of alpha1, since d00_l(1) = 1.
            f11 = phase[case]["outputs"]["f11"]
            assert phase[case]["inputs"]["mu"][0] == 1.0
            total = sum(_value(a1, i) for i in range(len(a1["hi"])))
            assert abs(total - _value(f11, 0)) <= 1e-12 * _value(f11, 0), case
