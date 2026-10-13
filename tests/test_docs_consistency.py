"""The contracts quoted in the documentation pages must match the rustdoc and docstrings.

The pages under ``docs/`` repeat accuracy bounds, limits and costs from the Rust sources (marked
``<!-- synced with ... -->``). These tests look for each figure in every file that states it, so
a change in one place fails until the others follow. Texts are compared after normalising
Unicode superscripts, LaTeX and Python operators to one form (see ``normalise``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "docs" / "theory").is_dir():  # e.g. tests run from an installed sdist
    pytest.skip("documentation sources not available", allow_module_level=True)

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def normalise(text: str) -> str:
    """One spelling for 2⁻⁵², 2^{-52} and 2**-52; for ·, \\cdot and *; and no spacing."""
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^" + m.group().translate(_SUPERSCRIPTS), text)
    text = text.replace("**", "^")
    for latex in (r"\,", r"\cdot", r"\text", r"\mathrm", r"\left", r"\right"):
        text = text.replace(latex, "")
    text = text.replace(r"\le", "≤")
    return re.sub(r"[\s{}$`*·\\]", "", text)


def read(path: str) -> str:
    return normalise((ROOT / path).read_text(encoding="utf-8"))


GL_RS = "scatrs/src/special/gauss_legendre.rs"
WIGNER_RS = "scatrs/src/special/wigner.rs"
SPECIAL_PY = "python/scatrs/special.py"
BINDINGS_RS = "scatrs-py/src/special.rs"
DD_RS = "scatrs/src/scalar/dd/mod.rs"
TRIG_RS = "scatrs/src/scalar/dd/trig.rs"
DUAL_RS = "scatrs/src/scalar/dual/mod.rs"
THEORY = "docs/theory/special-functions.md"
SCALARS = "docs/theory/scalar-types.md"
LIMITATIONS = "docs/limitations.md"
WIGNER_EXAMPLE = "docs/examples/wigner-d.md"
HG_EXAMPLE = "docs/examples/henyey-greenstein.md"
REFERENCE_SUITE = "docs/validation/reference-suite.md"
DOMAIN_RS = "scatrs/src/mie/domain.rs"
MIE_THEORY = "docs/theory/mie.md"
MIE_RS = "scatrs/src/mie/mod.rs"
MIE_PY = "python/scatrs/mie.py"
MIE_BINDINGS_RS = "scatrs-py/src/mie.rs"
INSTALLATION = "docs/installation.md"
CONTRIBUTING = "CONTRIBUTING.md"

CONTRACTS = [
    # (what, normalised text, files that must state it)
    ("Wigner f64 bound", "16(l+1)2^-52", [WIGNER_RS, SPECIAL_PY, THEORY]),
    ("Wigner double-double bound", "(l+1)2^-97", [WIGNER_RS, THEORY]),
    ("Gauss-Legendre cap", "2^17", [GL_RS, SPECIAL_PY, THEORY, LIMITATIONS]),
    ("Gauss-Legendre cap", "131072", [GL_RS, SPECIAL_PY, THEORY]),
    ("Gauss-Legendre cost model", "28nsn^2", [GL_RS, SPECIAL_PY, THEORY]),
    ("Gauss-Legendre f64 node contract", "nodeabsoluteerror≤2^-52", [GL_RS]),
    ("Gauss-Legendre f64 weight contract", "weightrelativeerror≤2^-52", [GL_RS]),
    ("Gauss-Legendre f64 contract", "|f64|≤2^-52|≤2^-52|", [THEORY]),
    ("Gauss-Legendre double-double weight bound", "2^-100(4+n^2)", [GL_RS, THEORY]),
    ("Python Wigner output limit", "2^28", [SPECIAL_PY, LIMITATIONS]),
    ("Double-double full precision", "2^-969", [DD_RS, SCALARS]),
    ("Double-double trigonometric domain", "2^50", [TRIG_RS, SCALARS, LIMITATIONS]),
    ("Dual<f64> derivative accuracy", "5.1u", [DUAL_RS, SCALARS]),
    # The executed examples check against the same contract.
    ("Wigner f64 bound in the examples", "16(l+1)eps", [WIGNER_EXAMPLE, HG_EXAMPLE]),
    ("Mie f64 truncation", "x+6x^(1/3)+3", [MIE_RS, MIE_PY, MIE_THEORY, LIMITATIONS]),
    ("Mie double-double truncation", "x+12x^(1/3)+6", [MIE_RS, MIE_THEORY]),
    ("Mie order cap", "2^20", [MIE_RS, MIE_PY, LIMITATIONS]),
    ("Mie output cap", "2^27", [MIE_BINDINGS_RS, MIE_PY, LIMITATIONS]),
    ("Mie tolerance", "+cond)2^-52", [DOMAIN_RS, MIE_THEORY]),
    ("Mie validated x range", "10^-6≤x≤2", [MIE_THEORY, LIMITATIONS]),
    ("MSRV", "1.85", [INSTALLATION, CONTRIBUTING]),
    ("Python floor", "3.11", [INSTALLATION, CONTRIBUTING]),
    ("numpy floor", "1.26", [INSTALLATION, CONTRIBUTING]),
]


@pytest.mark.parametrize(("what", "text", "files"), CONTRACTS)
def test_contract_is_stated_everywhere(what: str, text: str, files: list[str]) -> None:
    missing = [f for f in files if text not in read(f)]
    assert not missing, f"{what} ({text!r}) is not stated in {missing}"


def test_limits_in_code_match_the_documented_values() -> None:
    gl = (ROOT / GL_RS).read_text(encoding="utf-8")
    assert re.search(r"pub const MAX_GAUSS_LEGENDRE_POINTS: usize = 1 << 17;", gl)
    bindings = (ROOT / BINDINGS_RS).read_text(encoding="utf-8")
    assert re.search(r"pub const MAX_WIGNER_D_VALUES: usize = 1 << 28;", bindings)
    mie = (ROOT / MIE_RS).read_text(encoding="utf-8")
    assert re.search(r"pub const MAX_MIE_ORDER: usize = 1 << 20;", mie)
    assert re.search(r"terms\(x, 6\.0, 3\.0\)", mie) and re.search(r"terms\(x, 12\.0, 6\.0\)", mie)
    mie_bindings = (ROOT / MIE_BINDINGS_RS).read_text(encoding="utf-8")
    assert re.search(r"pub const MAX_MIE_VALUES: usize = 1 << 27;", mie_bindings)


def test_version_floors_match_the_build_files() -> None:
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert 'rust-version = "1.85"' in cargo
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11"' in pyproject
    assert '"numpy>=1.26"' in pyproject


def _declared_mie_constants() -> dict[str, float]:
    """The constants of `ValidatedDomain::constant`, by the `MieOutput` variants of each arm."""
    source = (ROOT / DOMAIN_RS).read_text(encoding="utf-8")
    body = source[source.index("pub fn constant(") :]
    body = body[: body.index("\n    }\n")]
    constants = {}
    for arm in re.finditer(r"((?:\s*\|?\s*MieOutput::\w+)+)\s*=>\s*([\d.]+)", body):
        for variant in re.findall(r"MieOutput::(\w+)", arm.group(1)):
            constants[variant] = float(arm.group(2))
    return constants


def test_mie_constants_match_the_accuracy_map() -> None:
    import json

    columns = {
        "coefficients": "Coefficients",
        "q_ext": "Extinction",
        "q_sca": "Scattering",
        "q_abs": "Absorption",
        "q_back": "Backscattering",
        "g": "Asymmetry",
        "amplitudes": "Amplitudes",
        "coefficient_derivatives": "CoefficientDerivatives",
        "efficiency_jacobian": "EfficiencyJacobian",
        "amplitude_derivatives": "AmplitudeDerivatives",
    }
    constants = _declared_mie_constants()
    assert set(constants) == set(columns.values())
    data = json.loads((ROOT / "docs/validation/data/mie-accuracy-map.json").read_text())
    for column, variant in columns.items():
        assert data["declared"][column] == constants[variant], column
        # The documented margin: every constant at least 4 times the worst C of the map.
        assert 4 * max(max(row) for row in data["c"][column]) <= constants[variant], column


# The `MieOutput` variants that share a constant, as the theory chapter and the rustdoc table of
# `ValidatedDomain::constant` name them.
MIE_CONSTANT_GROUPS = [
    (
        ("Coefficients", "Extinction", "Scattering", "Absorption", "Backscattering", "Asymmetry"),
        "the coefficients, efficiencies and $g$",
        "coefficients, efficiencies, g",
    ),
    (
        ("Amplitudes", "AmplitudeDerivatives"),
        "$S_1$, $S_2$ and their derivatives",
        "S₁, S₂ and their derivatives",
    ),
    (("EfficiencyJacobian",), "the efficiency Jacobian", "efficiency Jacobian"),
    (("CoefficientDerivatives",), "the coefficient derivatives", "coefficient derivatives"),
]


def test_mie_constants_are_stated_in_the_theory_and_the_rustdoc() -> None:
    constants = _declared_mie_constants()
    assert {v for variants, _, _ in MIE_CONSTANT_GROUPS for v in variants} == set(constants)
    phrases, rows = [], []
    for variants, in_theory, in_table in MIE_CONSTANT_GROUPS:
        values = {constants[v] for v in variants}
        assert len(values) == 1, f"{variants} no longer share a constant"
        value = int(values.pop())
        phrases.append(f"{value} for {in_theory}")
        rows.append(f"/// | {in_table} | {value} |")
    assert normalise("; ".join(phrases)) in read(MIE_THEORY)
    source = (ROOT / DOMAIN_RS).read_text(encoding="utf-8")
    missing = [row for row in rows if row not in source]
    assert not missing, f"rows missing from the table of ValidatedDomain::constant: {missing}"


def test_reference_suite_page_matches_the_manifest() -> None:
    import json

    manifest = json.loads((ROOT / "reference/data/manifest.json").read_text(encoding="utf-8"))
    page = (ROOT / REFERENCE_SUITE).read_text(encoding="utf-8")
    for name, entry in manifest["files"].items():
        row = re.search(rf"^\| `{re.escape(name)}` \| (\d+) \|", page, re.MULTILINE)
        assert row, f"{name} is missing from the table"
        assert int(row.group(1)) == entry["records"], name
    total = sum(p.stat().st_size for p in (ROOT / "reference/data").glob("*.jsonl")) / 1e6
    stated = re.search(r"The committed data take about ([\d.]+) MB", page)
    assert stated and abs(float(stated.group(1)) - total) <= 0.1, (stated, total)


def test_synced_sections_name_existing_files() -> None:
    markers = 0
    for page in (ROOT / "docs").rglob("*.md"):
        if {"plans", "specs", "_build"} & set(page.parts):
            continue
        for source in re.findall(r"<!-- synced with ([\w./-]+)", page.read_text(encoding="utf-8")):
            markers += 1
            assert (ROOT / source).exists(), f"{page.name}: {source} does not exist"
    assert markers > 0
