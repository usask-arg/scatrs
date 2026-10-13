import importlib
import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version

import scatrs
from scatrs import _scatrs, special

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


def test_crate_license_is_the_repository_license():
    # scatrs/LICENSE is a copy, so that the published crate ships the license text.
    assert (ROOT / "scatrs" / "LICENSE").read_text("utf-8") == (ROOT / "LICENSE").read_text("utf-8")


def test_public_names():
    for name in (
        "special",
        "ScatrsError",
        "InvalidInputError",
        "NotConvergedError",
        "OutsideValidatedDomainError",
        "__version__",
    ):
        assert hasattr(scatrs, name)
    for name in special.__all__:
        assert hasattr(special, name)


def test_limits_are_exposed():
    assert special.MAX_GAUSS_LEGENDRE_POINTS == _scatrs.MAX_GAUSS_LEGENDRE_POINTS == 2**17
    assert special.MAX_WIGNER_D_VALUES == _scatrs.MAX_WIGNER_D_VALUES == 2**28


def test_test_hooks_live_outside_the_extension_namespace():
    assert set(_scatrs.__all__) == {
        "MAX_GAUSS_LEGENDRE_POINTS",
        "MAX_WIGNER_D_VALUES",
        "__version__",
        "gauss_legendre",
        "wigner_d",
    }
    assert not hasattr(_scatrs, "_raise_core_error")
    testing = importlib.import_module("scatrs._scatrs._testing")
    assert testing is _scatrs._testing
    assert set(testing.__all__) == {"raise_core_error", "with_write_borrow"}
