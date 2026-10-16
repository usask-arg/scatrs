# Stubs for the extension module scatrs._scatrs (a stub package, for its _testing submodule).
from typing import Any, Final, final

import numpy as np
from numpy.typing import NDArray

from . import _testing as _testing

__all__ = [
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

MAX_LAYERS: Final[int]

def layered_efficiencies(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
def layered_coefficients(
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
def layered_amplitudes(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    mu: NDArray[np.float64],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]: ...
def layered_phase_matrix(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    mu: NDArray[np.float64],
    n_max: int | None,
    derivatives: bool,
    allow_unvalidated: bool,
    normalized: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
def layered_greek(
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

MAX_TMATRIX_ORDER: Final[int]

def tmatrix_random(
    x: NDArray[np.float64],
    m: NDArray[np.complex128],
    eps: NDArray[np.float64],
    rtol: float,
    precision: str,
    n_terms: int | None,
    n_theta: int | None,
    allow_unconverged: bool,
    allow_unvalidated: bool,
    derivatives: bool,
    greek: bool,
    order: int | None,
    mu: NDArray[np.float64] | None,
    normalized: bool,
) -> dict[str, NDArray[Any]]: ...

@final
class TMatrixObject:
    def __new__(
        cls,
        x: float,
        m_re: float,
        m_im: float,
        eps: float,
        rtol: float,
        precision: str,
        n_terms: int | None,
        n_theta: int | None,
        allow_unconverged: bool,
        allow_unvalidated: bool,
        derivatives: bool,
    ) -> TMatrixObject: ...
    @property
    def n_terms(self) -> int: ...
    @property
    def n_theta(self) -> int: ...
    @property
    def n_modes(self) -> int: ...
    @property
    def error_estimate(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def precision_paths(self) -> list[str]: ...
    def block(
        self, m: int
    ) -> (
        tuple[
            NDArray[np.int64],
            list[NDArray[np.complex128]],
        ]
        | None
    ): ...
    def efficiencies(self) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
    def greek(
        self, order: int | None, normalized: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
    def phase_matrix(
        self, mu: NDArray[np.float64], normalized: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
