"""The figures quoted in the documentation must match their sources.

The pages under ``docs/`` repeat accuracy bounds, limits, costs and version floors from the
Rust sources, the docstrings, the build files, the reference manifest and the accuracy map
(marked ``<!-- synced with ... -->``). These tests look for each figure in every file that
states it, so a change in one place fails until the others follow. Texts are compared after
normalising Unicode superscripts, LaTeX and Python operators to one form (see ``normalise``).
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "docs" / "theory").is_dir():  # e.g. tests run from an installed sdist
    pytest.skip("documentation sources not available", allow_module_level=True)

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def normalise(text: str) -> str:
    """One spelling for 2⁻⁵², 2^{-52} and 2**-52; for ·, \\cdot and *; for ≤, ≥, | and their LaTeX;
    and no spacing."""
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^" + m.group().translate(_SUPERSCRIPTS), text)
    text = text.replace("**", "^")
    for latex in (r"\,", r"\cdot", r"\text", r"\mathrm", r"\operatorname", r"\left", r"\right"):
        text = text.replace(latex, "")
    text = text.replace(r"\le", "≤").replace(r"\ge", "≥")
    text = text.replace(r"\lvert", "|").replace(r"\rvert", "|")
    return re.sub(r"[\s{}$`*·\\]", "", text)


def read(path: str) -> str:
    return normalise((ROOT / path).read_text(encoding="utf-8"))


def states(page: str, text: str) -> bool:
    """Whether the normalised ``page`` contains ``text``, not followed by more digits when it ends
    in one: "2^20" is not stated by "2^200", nor "x+6x^(1/3)+3" by "...+30". (Normalising
    removes spaces, so a digit may legitimately precede ``text``.)"""
    after = r"(?!\.?\d)" if text[-1].isdigit() else ""
    return re.search(re.escape(text) + after, page) is not None


def states_version(page: str, version: str) -> bool:
    """Whether the normalised ``page`` states ``version`` as a whole version number, possibly
    followed by ".0" parts ("1.85.0" states 1.85; "11.85" and "1.85.1" do not)."""
    pattern = r"(?<![\d.])" + re.escape(version) + r"(?:\.0)*(?![.\d])"
    return re.search(pattern, page) is not None


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
QUADRATURE = "docs/examples/quadrature.md"
REFERENCE_SUITE = "docs/validation/reference-suite.md"
DOMAIN_RS = "scatrs/src/mie/domain.rs"
MIE_THEORY = "docs/theory/mie.md"
MIE_RS = "scatrs/src/mie/mod.rs"
MIE_PY = "python/scatrs/mie.py"
MIE_BINDINGS_RS = "scatrs-py/src/mie.rs"
GREEK_RS = "scatrs/src/mie/greek.rs"
INSTALLATION = "docs/installation.md"
CONTRIBUTING = "CONTRIBUTING.md"
ENSEMBLE_THEORY = "docs/theory/ensembles.md"
ENSEMBLE_RS = "scatrs/src/ensemble/mod.rs"
ENSEMBLE_TESTS = "scatrs/tests/ensemble.rs"
ACCURACY = "docs/validation/accuracy.md"
FOLLOWUPS_2A = "docs/plans/phase-2a-followups.md"

CONTRACTS = [
    # (what, normalised text, files that must state it)
    ("Wigner f64 bound", "16(l+1)2^-52", [WIGNER_RS, SPECIAL_PY, THEORY]),
    ("Wigner double-double bound", "(l+1)2^-97", [WIGNER_RS, THEORY]),
    ("Gauss-Legendre cap", "2^17", [GL_RS, SPECIAL_PY, THEORY, LIMITATIONS]),
    ("Gauss-Legendre cap", "131072", [GL_RS, SPECIAL_PY, THEORY]),
    ("Gauss-Legendre double-double cost model", "28nsn^2", [GL_RS, THEORY]),
    ("Gauss-Legendre double-double timing", "0.72s", [GL_RS, THEORY]),
    ("Gauss-Legendre double-double timing", "11s", [GL_RS, THEORY, LIMITATIONS]),
    ("Gauss-Legendre f64 timing at the cap", "40ms", [GL_RS, SPECIAL_PY, THEORY]),
    (
        "Gauss-Legendre asymptotic bound",
        "322^-52",
        [GL_RS, SPECIAL_PY, THEORY, LIMITATIONS, QUADRATURE],
    ),
    ("Gauss-Legendre asymptotic measured", "2.82^-52", [GL_RS, THEORY]),
    ("Gauss-Legendre asymptotic measured", "4.02^-52", [GL_RS, THEORY]),
    (
        "Gauss-Legendre asymptotic measured",
        "2.8and4.0unitsof2^-52",
        [SPECIAL_PY, LIMITATIONS, "CHANGELOG.md"],
    ),
    ("Wigner cost", "2.6ns", [WIGNER_RS, THEORY]),
    ("Gauss-Legendre f64 node contract", "nodeabsoluteerror≤2^-52", [GL_RS]),
    ("Gauss-Legendre f64 weight contract", "weightrelativeerror≤2^-52", [GL_RS]),
    ("Gauss-Legendre f64 contract", "|f64,n≤100|≤2^-52|≤2^-52|", [THEORY]),
    ("Gauss-Legendre asymptotic contract", "|f64,n>100|≤322^-52|≤322^-52|", [THEORY]),
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
    ("Mie validated domain", "10^-6≤x≤210^4,0<Rem≤10,0≤Imm≤10,|m|≥0.1", [MIE_THEORY, LIMITATIONS]),
    ("Mie validated domain", "x∈[10^-6,210^4],0<Rem≤10,0≤Imm≤10,|m|≥0.1", [DOMAIN_RS]),
    ("Mie validated domain", "1e-6<=x<=2e4,0<Rem<=10,0<=Imm<=10,|m|>=0.1", [MIE_PY]),
    ("Mie accepted x", "xin[10^-30,10^6]", [LIMITATIONS]),
    ("Mie accepted |m|", "|m|in[10^-6,10^3]", [LIMITATIONS]),
    ("Mie accepted x", "xoutside[1e-30,1e6]", [MIE_PY]),
    ("Mie accepted |m|", "|m|outside[1e-6,1e3]", [MIE_PY]),
    (
        "Greek all-orders timing",
        "7.6s",
        [MIE_THEORY, LIMITATIONS, MIE_PY, GREEK_RS, "docs/plans/phase-1b-followups.md"],
    ),
    (
        "Greek all-orders timing",
        "0.09s",
        [MIE_THEORY, MIE_PY, GREEK_RS, "docs/plans/phase-1b-followups.md"],
    ),
    ("Gauss-Legendre f64 timing", "6ms", [GL_RS, THEORY, MIE_THEORY]),
    ("Greek node limit", "6.510^4", [GREEK_RS, LIMITATIONS]),
    (
        "Greek quadrature term",
        "((L+1)^2+x^2)/8",
        [DOMAIN_RS, MIE_THEORY, "docs/plans/phase-1b-followups.md"],
    ),
    (
        "Greek node rounding",
        "L^22^-54",
        [DOMAIN_RS, MIE_THEORY, "docs/plans/phase-1b-followups.md"],
    ),
    (
        "Greek node rounding",
        "x^22^-54",
        [DOMAIN_RS, MIE_THEORY, "docs/plans/phase-1b-followups.md"],
    ),
    (
        "Greek C without the term",
        "C=85atx=100(allorders)",
        [MIE_THEORY, "docs/validation/accuracy.md"],
    ),
    (
        "Greek C without the term",
        "C=100atx=210^4(toorder64)",
        [MIE_THEORY, "docs/validation/accuracy.md"],
    ),
    ("Greek worst C", "Cisatmost1.7", [MIE_THEORY, "docs/validation/accuracy.md"]),
    ("Greek node-rounding error", "210^-9", [LIMITATIONS, "docs/plans/phase-1b-followups.md"]),
    ("Greek node-rounding error", "210^-11", [LIMITATIONS, "docs/plans/phase-1b-followups.md"]),
    # Ensembles of spheres: the measured cost (lognormal r_g = 1, σ_g = 2, m = 1.33 + 1e-4i).
    *(
        ("Ensemble cost", t, [ENSEMBLE_THEORY, LIMITATIONS])
        for t in ("0.037s", "0.11s", "0.20s", "0.093s", "0.31s", "0.90s")
    ),
    *(
        ("Ensemble cost before the matrix products", t, [ENSEMBLE_THEORY, FOLLOWUPS_2A])
        for t in ("0.62s", "2.9s", "10.4s")
    ),
    ("Ensemble max_panels default", "default4096", [ENSEMBLE_RS, ENSEMBLE_THEORY, LIMITATIONS]),
    ("Ensemble calibration grid", "ninedistributions", [ACCURACY, LIMITATIONS]),
    ("Ensemble calibration, refined outputs", "0.003", [ENSEMBLE_TESTS, ACCURACY]),
    ("Ensemble calibration, other outputs", "1.210", [ENSEMBLE_TESTS, ACCURACY]),
    ("Ensemble non-absorbing estimate", "estimateofabout10^-5of", [ACCURACY, LIMITATIONS]),
]


@pytest.mark.parametrize(("what", "text", "files"), CONTRACTS)
def test_contract_is_stated_everywhere(what: str, text: str, files: list[str]) -> None:
    missing = [f for f in files if not states(read(f), text)]
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
    for limit in (
        r"MIN_SIZE_PARAMETER: f64 = 1e-30;",
        r"MAX_SIZE_PARAMETER: f64 = 1e6;",
        r"MIN_REFRACTIVE_INDEX: f64 = 1e-6;",
        r"MAX_REFRACTIVE_INDEX: f64 = 1e3;",
    ):
        assert f"pub const {limit}" in mie, limit
    domain = (ROOT / DOMAIN_RS).read_text(encoding="utf-8")
    assert re.search(
        r"VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain \{\s*x_min: 1e-6,\s*x_max: 2e4,"
        r"\s*m_re_max: 10\.0,\s*m_im_max: 10\.0,\s*abs_m_min: 0\.1,\s*\};",
        domain,
    ), "VALIDATED_DOMAIN differs from the documented domain"


def _version_floors() -> dict[str, str]:
    """The MSRV, the Python floor and the numpy floor, from the build files."""
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    python = re.fullmatch(r">=\s*([\d.]+)", project["requires-python"])
    numpy = [re.fullmatch(r"numpy\s*>=\s*([\d.]+)", d) for d in project["dependencies"]]
    numpy = [m for m in numpy if m]
    assert python and len(numpy) == 1, (project["requires-python"], project["dependencies"])
    return {
        "MSRV": cargo["workspace"]["package"]["rust-version"],
        "Python floor": python.group(1),
        "numpy floor": numpy[0].group(1),
    }


@pytest.mark.parametrize("what", ["MSRV", "Python floor", "numpy floor"])
def test_version_floors_are_stated_everywhere(what: str) -> None:
    floor = _version_floors()[what]
    missing = [f for f in (INSTALLATION, CONTRIBUTING) if not states_version(read(f), floor)]
    assert not missing, f"{what} {floor} (from the build files) is not stated in {missing}"


def _declared_mie_constants() -> dict[str, float]:
    """The constants of `ValidatedDomain::constant`, by the `MieOutput` variants of each arm."""
    source = (ROOT / DOMAIN_RS).read_text(encoding="utf-8")
    body = source[source.index("pub fn constant(") :]
    body = body[: body.index("\n    }\n")]
    constants = {}
    arm_variant = r"MieOutput::\w+(?:\s*\{\s*\.\.\s*\})?"
    for arm in re.finditer(rf"((?:\s*\|?\s*{arm_variant})+)\s*=>\s*([\d.]+)", body):
        for variant in re.findall(r"MieOutput::(\w+)", arm.group(1)):
            constants[variant] = float(arm.group(2))
    return constants


# The columns of each accuracy map, by the `MieOutput` variant each measures.
ACCURACY_MAPS = {
    "mie-accuracy-map.json": {
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
    },
    "greek-accuracy-map.json": {
        "phase_matrix": "PhaseMatrix",
        "phase_matrix_derivatives": "PhaseMatrixDerivatives",
        "greek": "GreekCoefficients",
        "greek_derivatives": "GreekDerivatives",
    },
}


@pytest.mark.parametrize("name", sorted(ACCURACY_MAPS))
def test_mie_constants_match_the_accuracy_map(name: str) -> None:
    columns = ACCURACY_MAPS[name]
    constants = _declared_mie_constants()
    measured = {v for c in ACCURACY_MAPS.values() for v in c.values()}
    assert set(constants) == measured, "a constant that no map measures, or the reverse"
    data = json.loads((ROOT / "docs/validation/data" / name).read_text())
    assert set(data["declared"]) == set(data["c"]) == set(columns)
    for column, variant in columns.items():
        assert data["declared"][column] == constants[variant], column
        # The documented margin: every constant at least 4 times the worst C of the map. A null
        # (a non-finite C) counts as infinite.
        worst = max(math.inf if c is None else c for row in data["c"][column] for c in row)
        assert 4 * worst <= constants[variant], column


# The `MieOutput` variants that share a constant, as the theory chapter and the rustdoc table of
# `ValidatedDomain::constant` name them.
MIE_CONSTANT_GROUPS = [
    (
        ("Coefficients", "Extinction", "Scattering", "Absorption", "Backscattering", "Asymmetry"),
        "the coefficients, efficiencies and $g$",
        "coefficients, efficiencies, g",
    ),
    (("Amplitudes",), "$S_1$, $S_2$", "S₁, S₂"),
    (("AmplitudeDerivatives",), "their derivatives", "derivatives of S₁, S₂"),
    (("EfficiencyJacobian",), "the efficiency Jacobian", "efficiency Jacobian"),
    (("CoefficientDerivatives",), "the coefficient derivatives", "coefficient derivatives"),
    (("PhaseMatrix",), "the phase matrix", "phase matrix"),
    (("PhaseMatrixDerivatives",), "its derivatives", "derivatives of the phase matrix"),
    (("GreekCoefficients",), "the Greek coefficients", "Greek coefficients"),
    (("GreekDerivatives",), "their derivatives", "derivatives of the Greek coefficients"),
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
    manifest = json.loads((ROOT / "reference/data/manifest.json").read_text(encoding="utf-8"))
    page = (ROOT / REFERENCE_SUITE).read_text(encoding="utf-8")
    rows = dict(re.findall(r"^\| `([\w.]+\.jsonl)` \| (\d+) \|", page, re.MULTILINE))
    assert set(rows) == set(manifest["files"]), "the table and the manifest list different files"
    for name, entry in manifest["files"].items():
        assert int(rows[name]) == entry["records"], name
    total = sum(p.stat().st_size for p in (ROOT / "reference/data").glob("*.jsonl")) / 1e6
    stated = re.search(r"The committed data take about\s+([\d.]+)\s+MB", page)
    assert stated, "size sentence not found"
    assert abs(float(stated.group(1)) - total) <= 0.1, (stated.group(1), total)


def test_synced_sections_name_existing_files() -> None:
    markers = 0
    for page in (ROOT / "docs").rglob("*.md"):
        if {"plans", "specs", "_build"} & set(page.parts):
            continue
        for source in re.findall(r"<!-- synced with ([\w./-]+)", page.read_text(encoding="utf-8")):
            markers += 1
            assert (ROOT / source).exists(), f"{page.name}: {source} does not exist"
    assert markers > 0


def test_the_sasktran2_page_states_the_published_results() -> None:
    """docs/validation/sasktran2.md states every time, error and matched rtol of the results
    file it names, formatted as `benchmarks/compare.py` prints them."""
    page = (ROOT / "docs/validation/sasktran2.md").read_text(encoding="utf-8")
    name = re.search(r"benchmarks/results/([\w.-]+\.json)", page).group(1)
    results = json.loads((ROOT / "benchmarks/results" / name).read_text(encoding="utf-8"))
    cols = ("c_ext", "c_sca", "ssa", "coalbedo", "alpha1", "beta1")
    missing = []
    for w, e in results["workloads"].items():
        rows = [("sasktran2", e["sasktran2"])]
        rows += [(f"scatrs rtol {float(r):g}", v) for r, v in e["scatrs"].items()]
        for label, r in rows:
            mark = " (matched)" if label.endswith(f"{e['matched_rtol']:g}") else ""
            row = (
                f"| {label}{mark} | {r['time_one']:.3g} s | {r['time_all']:.3g} s | "
                + " | ".join(f"{r['errors'][c]:.1e}" for c in cols)
                + " |"
            )
            if row not in page:
                missing.append(f"{w}: {row}")
    assert not missing, "\n".join(missing)
    assert results["versions"]["sasktran2"] in page and results["date"] in page
