"""Argument conversion shared by the wrappers around ``scatrs._scatrs`` (private).

The rule: Python converts types and shapes; the Rust core validates values and words their
errors, so each check lives in one place. Python checks a value only where converting it to
the Rust parameter type would otherwise fail, as for an integer that does not fit a ``usize``.

Arrays handed to Rust should be fresh views (``xs.reshape(-1)``, say), never the caller's own
array object; see :func:`real_array`.
"""

from __future__ import annotations

import operator
from typing import Final, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs.errors import InvalidInputError

_BOOLS: Final = (bool, np.bool_)
_USIZE_MAX: Final = int(np.iinfo(np.uintp).max)
_INT32_MIN: Final = int(np.iinfo(np.int32).min)
_INT32_MAX: Final = int(np.iinfo(np.int32).max)


def integer(value: SupportsIndex, name: str) -> int:
    """``value`` as an int: Python and numpy integers are accepted.

    Raises
    ------
    TypeError
        For bools (Python or numpy) and anything without ``__index__``, such as floats.
    """
    if isinstance(value, _BOOLS):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}") from None


def size(value: SupportsIndex, name: str) -> int:
    """An integer for a Rust ``usize`` parameter; Rust checks its value further.

    Raises
    ------
    TypeError
        As :func:`integer`.
    InvalidInputError
        If ``value`` is negative or above the largest ``usize``.
    """
    v = integer(value, name)
    if v < 0:
        raise InvalidInputError(f"invalid input: {name} must not be negative, got {v}")
    if v > _USIZE_MAX:
        raise InvalidInputError(f"invalid input: {name} must be at most {_USIZE_MAX}, got {v}")
    return v


def flag(value: bool, name: str) -> bool:
    """``value`` as a bool: Python and numpy bools only.

    Raises
    ------
    TypeError
        For anything else, such as ``0``, ``"no"`` or ``None``.
    """
    if not isinstance(value, _BOOLS):
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return bool(value)


def int32(value: SupportsIndex, name: str) -> int:
    """An integer for a Rust ``i32`` parameter.

    Raises
    ------
    TypeError
        As :func:`integer`.
    OverflowError
        If ``value`` is outside the 32-bit signed range.
    """
    v = integer(value, name)
    if not _INT32_MIN <= v <= _INT32_MAX:
        raise OverflowError(f"{name} = {v} is outside the 32-bit signed range")
    return v


def real_array(x: ArrayLike, name: str) -> NDArray[np.float64]:
    """``x`` as an aligned, C-contiguous, native-endian float64 array of the same shape.

    Such an array is returned as it is, so Rust reads it in place; other real (integer or
    float) arrays, lists and scalars are converted, copying once. Hand Rust a fresh view of the
    result rather than the result itself: rust-numpy keys its borrow on the array object it is
    given, and if that object's data pointer changed during the call (``x.resize(refcheck=False)``
    from another thread), releasing the borrow would abort the interpreter. A view's data pointer
    cannot change.

    Raises
    ------
    TypeError
        If ``x`` is not real: complex, boolean, string, object, datetime or timedelta.
    ValueError
        Raised by numpy for a ragged nested sequence.
    """
    xs = np.asarray(x)
    if xs.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numbers, not an array of dtype {xs.dtype}")
    # `==` also compares byte order, so a byte-swapped float64 array is converted.
    if xs.dtype == np.float64 and xs.flags.c_contiguous and xs.flags.aligned:
        return xs
    return np.require(xs, dtype=np.float64, requirements=["C", "A"])


def complex_array(x: ArrayLike, name: str) -> NDArray[np.complex128]:
    """``x`` as an aligned, C-contiguous, native-endian complex128 array of the same shape.

    As :func:`real_array`, for complex numbers: such an array is returned as it is, and other
    real or complex arrays, lists and scalars are converted, copying once.

    Raises
    ------
    TypeError
        If ``x`` is not numeric: boolean, string, object, datetime or timedelta.
    ValueError
        Raised by numpy for a ragged nested sequence.
    """
    xs = np.asarray(x)
    if xs.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be numbers, not an array of dtype {xs.dtype}")
    if xs.dtype == np.complex128 and xs.flags.c_contiguous and xs.flags.aligned:
        return xs
    return np.require(xs, dtype=np.complex128, requirements=["C", "A"])
