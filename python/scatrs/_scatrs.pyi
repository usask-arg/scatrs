from typing import NoReturn

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MAX_GAUSS_LEGENDRE_POINTS",
    "MAX_WIGNER_D_VALUES",
    "__version__",
    "_raise_core_error",
    "gauss_legendre",
    "wigner_d",
]

__version__: str
MAX_GAUSS_LEGENDRE_POINTS: int
MAX_WIGNER_D_VALUES: int

def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wigner_d(m: int, n: int, l_max: int, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
def _raise_core_error(
    variant: str, message: str = "", achieved: float = 0.0, requested: float = 0.0
) -> NoReturn: ...
