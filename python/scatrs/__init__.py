"""scatrs: fast, linearized, rigorously validated single-scattering calculations."""

from importlib.metadata import version as _version

from scatrs import special
from scatrs.errors import (
    InvalidInputError,
    NotConvergedError,
    OutsideValidatedDomainError,
    ScatrsError,
)

__all__ = [
    "InvalidInputError",
    "NotConvergedError",
    "OutsideValidatedDomainError",
    "ScatrsError",
    "__version__",
    "special",
]

# The distribution's version, in PEP 440 form: maturin derives it from the Cargo workspace
# version, which the extension reports as `scatrs._scatrs.__version__` in SemVer form (a
# pre-release 0.1.0-alpha.1 there is 0.1.0a1 here).
__version__: str = _version("scatrs")
