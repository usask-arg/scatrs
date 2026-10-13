# Stubs for scatrs._scatrs._testing: hooks for the tests only, not part of the scatrs API.
from collections.abc import Callable
from typing import NoReturn, TypeVar

import numpy as np
from numpy.typing import NDArray

_T = TypeVar("_T")

__all__ = ["raise_core_error", "with_write_borrow"]

def raise_core_error(
    variant: str, message: str = "", achieved: float = 0.0, requested: float = 0.0
) -> NoReturn: ...
def with_write_borrow(x: NDArray[np.float64], f: Callable[[], _T]) -> _T: ...
