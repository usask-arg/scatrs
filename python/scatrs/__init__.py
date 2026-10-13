"""scatrs: fast, linearized, rigorously validated single-scattering calculations."""

from scatrs import special
from scatrs._scatrs import __version__
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
