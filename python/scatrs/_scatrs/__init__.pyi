# Stubs for the extension module scatrs._scatrs (a stub package, for its _testing submodule).
from typing import Final

import numpy as np
from numpy.typing import NDArray

from . import _testing as _testing

__all__ = [
    "MAX_GAUSS_LEGENDRE_POINTS",
    "MAX_WIGNER_D_VALUES",
    "__version__",
    "gauss_legendre",
    "wigner_d",
]

__version__: Final[str]
MAX_GAUSS_LEGENDRE_POINTS: Final[int]
MAX_WIGNER_D_VALUES: Final[int]

def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wigner_d(m: int, n: int, l_max: int, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
