"""Exception hierarchy raised by scatrs.

Every scatrs-specific failure is a :class:`ScatrsError`:

* :class:`InvalidInputError` (also a :class:`ValueError`): an argument has the right type but
  a value outside the function's domain, including the documented size limits, such as
  ``n > MAX_GAUSS_LEGENDRE_POINTS`` or a NaN where a number in [-1, 1] is needed.
* :class:`NotConvergedError` (also an :class:`ArithmeticError`): a computation did not reach
  the accuracy it promises.
* :class:`OutsideValidatedDomainError` (also a :class:`ValueError`): the inputs are valid, but
  outside the domain where the accuracy has been validated.

Standard exceptions are used where Python itself would raise them:

* :class:`TypeError` for an argument of the wrong type: a ``float`` or ``bool`` (Python or
  numpy) where an integer is needed, or non-real values (complex, strings, objects) where
  real numbers are needed;
* :class:`ValueError`, raised by numpy, for a ragged nested sequence where an array is needed;
* :class:`OverflowError` for an integer that does not fit the type the Rust core uses, such as
  a Wigner order outside the 32-bit signed range;
* :class:`BufferError` when another extension built on rust-numpy holds a writeable borrow of
  an input array;
* :class:`MemoryError` when an output cannot be allocated.
"""

from __future__ import annotations

__all__ = [
    "InvalidInputError",
    "NotConvergedError",
    "OutsideValidatedDomainError",
    "ScatrsError",
]


class ScatrsError(Exception):
    """Base class for every error raised by scatrs."""


class InvalidInputError(ScatrsError, ValueError):
    """An input is outside its valid range.

    That is its mathematical domain (a NaN, or ``x`` outside [-1, 1] for ``x = cos(theta)``) or
    a documented limit, such as ``MAX_GAUSS_LEGENDRE_POINTS`` or ``MAX_WIGNER_D_VALUES``.
    """


class NotConvergedError(ScatrsError, ArithmeticError):
    """A computation did not reach the requested accuracy.

    Parameters
    ----------
    message : str
        Description of the failure.
    achieved : float, optional
        Accuracy actually reached (may be ``inf``).
    requested : float, optional
        Accuracy that was requested.
    """

    def __init__(
        self, message: str, achieved: float | None = None, requested: float | None = None
    ) -> None:
        super().__init__(message)
        self.achieved = achieved
        self.requested = requested


class OutsideValidatedDomainError(ScatrsError, ValueError):
    """Inputs are valid but outside the domain where accuracy has been validated."""
