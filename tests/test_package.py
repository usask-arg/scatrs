import importlib
import importlib.metadata
import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version

import scatrs
from scatrs import _scatrs, ensemble, mie, particles, refractive, size, special

ROOT = Path(__file__).parents[1]


def test_version_matches_distribution_metadata():
    assert scatrs.__version__ == version("scatrs")


def test_version_matches_cargo_workspace():
    # The single version source: [workspace.package] version in the root Cargo.toml. It is
    # SemVer; maturin writes it in normalised PEP 440 form (0.1.0-alpha.1 becomes 0.1.0a1).
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text("utf-8"))
    assert scatrs.__version__ == str(Version(cargo["workspace"]["package"]["version"]))


def test_extension_is_built_from_the_installed_version():
    # The extension reports the Cargo version it was compiled from (SemVer form).
    assert Version(_scatrs.__version__) == Version(scatrs.__version__)


def test_version_falls_back_to_the_extension_without_metadata(monkeypatch):
    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        assert importlib.reload(scatrs).__version__ == _scatrs.__version__
    finally:
        monkeypatch.undo()
        importlib.reload(scatrs)
    assert scatrs.__version__ == version("scatrs")


def test_crate_license_is_the_repository_license():
    # scatrs/LICENSE is a copy, so that the published crate ships the license text.
    assert (ROOT / "scatrs" / "LICENSE").read_text("utf-8") == (ROOT / "LICENSE").read_text("utf-8")


def test_public_names():
    for name in (
        "Ensemble",
        "Mixture",
        "Sphere",
        "LogNormal",
        "Tabulated",
        "EnsembleResult",
        "ensemble",
        "mie",
        "particles",
        "refractive",
        "size",
        "special",
        "ScatrsError",
        "InvalidInputError",
        "NotConvergedError",
        "OutsideValidatedDomainError",
        "__version__",
    ):
        assert hasattr(scatrs, name)
    for module in (scatrs, special, mie, ensemble, particles, refractive, size):
        for name in module.__all__:
            assert hasattr(module, name)
    # The classes re-exported at the top level are those of their modules.
    assert scatrs.LogNormal is size.LogNormal and scatrs.Ensemble is ensemble.Ensemble


def test_limits_are_exposed():
    assert special.MAX_GAUSS_LEGENDRE_POINTS == _scatrs.MAX_GAUSS_LEGENDRE_POINTS == 2**17
    assert special.MAX_WIGNER_D_VALUES == _scatrs.MAX_WIGNER_D_VALUES == 2**28
    assert mie.MAX_MIE_ORDER == _scatrs.MAX_MIE_ORDER == 2**20
    assert mie.MAX_MIE_VALUES == _scatrs.MAX_MIE_VALUES == 2**27
    assert mie.MAX_GREEK_ORDER == _scatrs.MAX_GREEK_ORDER == 2**21


def test_test_hooks_live_outside_the_extension_namespace():
    assert set(_scatrs.__all__) == {
        "MAX_GAUSS_LEGENDRE_POINTS",
        "MAX_GREEK_ORDER",
        "MAX_LAYERS",
        "MAX_MIE_ORDER",
        "MAX_MIE_VALUES",
        "MAX_TMATRIX_ORDER",
        "MAX_WIGNER_D_VALUES",
        "TMatrixObject",
        "__version__",
        "ensemble_compute",
        "ensemble_compute_batch",
        "ensemble_parameters",
        "gauss_legendre",
        "layered_amplitudes",
        "layered_coefficients",
        "layered_efficiencies",
        "layered_greek",
        "layered_phase_matrix",
        "mie_amplitudes",
        "mie_coefficients",
        "mie_default_n_max",
        "mie_efficiencies",
        "mie_greek",
        "mie_phase_matrix",
        "tmatrix_random",
        "wigner_d",
    }
    assert not hasattr(_scatrs, "_raise_core_error")
    testing = importlib.import_module("scatrs._scatrs._testing")
    assert testing is _scatrs._testing
    assert set(testing.__all__) == {"raise_core_error", "with_write_borrow"}
