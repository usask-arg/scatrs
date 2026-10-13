import tomllib
from importlib.metadata import version
from pathlib import Path

import scatrs
from scatrs import _scatrs, special


def test_version_matches_distribution_metadata():
    assert scatrs.__version__ == version("scatrs")


def test_version_matches_cargo_workspace():
    # The single version source: [workspace.package] version in the root Cargo.toml.
    cargo = tomllib.loads((Path(__file__).parents[1] / "Cargo.toml").read_text("utf-8"))
    assert scatrs.__version__ == cargo["workspace"]["package"]["version"]


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
    assert special.MAX_WIGNER_D_VALUES == _scatrs.MAX_WIGNER_D_VALUES == 2**31
