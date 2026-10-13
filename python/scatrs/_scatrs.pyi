from collections.abc import Callable
from typing import Final, NoReturn, TypeVar

import numpy as np
from numpy.typing import NDArray

_T = TypeVar("_T")

__all__ = [
    "MAX_GAUSS_LEGENDRE_POINTS",
    "MAX_WIGNER_D_VALUES",
    "__version__",
    "_raise_core_error",
    "_with_write_borrow",
    "gauss_legendre",
    "wigner_d",
]

__version__: Final[str]
MAX_GAUSS_LEGENDRE_POINTS: Final[int]
MAX_WIGNER_D_VALUES: Final[int]

def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wigner_d(m: int, n: int, l_max: int, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

# For tests only: raises the exception a core scatrs::Error variant maps to.
def _raise_core_error(
    variant: str, message: str = "", achieved: float = 0.0, requested: float = 0.0
) -> NoReturn: ...

# For tests only: calls f() while holding rust-numpy's writeable borrow of x.
def _with_write_borrow(x: NDArray[np.float64], f: Callable[[], _T]) -> _T: ...
