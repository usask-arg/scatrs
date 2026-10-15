"""scatrs: fast, linearized, rigorously validated single-scattering calculations."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from scatrs import (
    _scatrs,
    ensemble,
    layered,
    mie,
    particles,
    refractive,
    size,
    special,
    tmatrix,
)
from scatrs.ensemble import (
    Ensemble,
    EnsembleGreek,
    EnsembleOutputs,
    EnsemblePhase,
    EnsembleResult,
    Mixture,
)
from scatrs.errors import (
    InvalidInputError,
    NotConvergedError,
    OutsideValidatedDomainError,
    ScatrsError,
)
from scatrs.particles import LayeredSphere, Sphere
from scatrs.size import (
    Gamma,
    LogNormal,
    ModifiedGamma,
    Monodisperse,
    MultiModal,
    PowerLaw,
    Tabulated,
)

__all__ = [
    "Ensemble",
    "EnsembleGreek",
    "EnsembleOutputs",
    "EnsemblePhase",
    "EnsembleResult",
    "Gamma",
    "InvalidInputError",
    "LayeredSphere",
    "LogNormal",
    "Mixture",
    "ModifiedGamma",
    "Monodisperse",
    "MultiModal",
    "NotConvergedError",
    "OutsideValidatedDomainError",
    "PowerLaw",
    "ScatrsError",
    "Sphere",
    "Tabulated",
    "__version__",
    "ensemble",
    "layered",
    "mie",
    "particles",
    "refractive",
    "size",
    "special",
    "tmatrix",
]

# The distribution's version, in PEP 440 form: maturin derives it from the Cargo workspace
# version, which the extension reports as `scatrs._scatrs.__version__` in SemVer form (a
# pre-release 0.1.0-alpha.1 there is 0.1.0a1 here). Without distribution metadata (the package
# imported from a source tree with a built extension), the extension's version is used.
try:
    __version__: str = _version("scatrs")
except _PackageNotFoundError:
    __version__ = _scatrs.__version__
