# Stubs for the extension module scatrs._scatrs (a stub package, for its _testing submodule).
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from . import _testing as _testing

__all__ = [
    "MAX_GAUSS_LEGENDRE_POINTS",
    "MAX_GREEK_ORDER",
    "MAX_MIE_ORDER",
    "MAX_MIE_VALUES",
    "MAX_WIGNER_D_VALUES",
    "__version__",
    "ensemble_compute",
    "ensemble_compute_batch",
    "ensemble_parameters",
    "gauss_legendre",
    "mie_amplitudes",
    "mie_coefficients",
    "mie_default_n_max",
    "mie_efficiencies",
    "mie_greek",
    "mie_phase_matrix",
    "wigner_d",
]

__version__: Final[str]
MAX_GAUSS_LEGENDRE_POINTS: Final[int]
MAX_WIGNER_D_VALUES: Final[int]
MAX_MIE_ORDER: Final[int]
MAX_MIE_VALUES: Final[int]
MAX_GREEK_ORDER: Final[int]

def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wigner_d(m: int, n: int, l_max: int, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
def mie_default_n_max(x: NDArray[np.float64]) -> NDArray[np.uint64]: ...
def mie_efficiencies(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
def mie_coefficients(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128] | None,
]: ...
def mie_amplitudes(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    mu: NDArray[np.float64],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]: ...
def mie_phase_matrix(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    mu: NDArray[np.float64],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
    normalized: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
def mie_greek(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    order: int | None,
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
    normalized: bool,
) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64] | None]: ...
def ensemble_parameters(items: list[tuple[Any, Any, float]], mixture: bool) -> list[str]: ...
def ensemble_compute(
    items: list[tuple[Any, Any, float]],
    wavelengths: NDArray[np.float64],
    mixture: bool,
    greek: int | None,
    all_orders: bool,
    mu: NDArray[np.float64],
    normalized: bool,
    derivatives: list[int],
    rtol: float,
    max_panels: int,
    allow_unvalidated: bool,
    refine_backscatter: bool,
) -> dict[str, Any]: ...
def ensemble_compute_batch(
    specs: list[Any],
    particle_spec: Any,
    wavelengths: NDArray[np.float64],
    greek: int | None,
    all_orders: bool,
    mu: NDArray[np.float64],
    normalized: bool,
    rtol: float,
    max_panels: int,
    allow_unvalidated: bool,
    refine_backscatter: bool,
) -> dict[str, Any]: ...
