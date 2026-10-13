"""Special functions and quadrature rules (thin wrappers over the Rust core).

Arguments are converted and checked here; the computation runs in Rust with the GIL released.
See :mod:`scatrs.errors` for which exception each kind of bad argument raises.
"""

from __future__ import annotations

import operator
from typing import Final, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _scatrs
from scatrs.errors import InvalidInputError

__all__ = ["MAX_GAUSS_LEGENDRE_POINTS", "MAX_WIGNER_D_VALUES", "gauss_legendre", "wigner_d"]

MAX_GAUSS_LEGENDRE_POINTS: Final[int] = _scatrs.MAX_GAUSS_LEGENDRE_POINTS
"""The largest ``n`` that :func:`gauss_legendre` accepts: ``2**17 = 131072``."""

MAX_WIGNER_D_VALUES: Final[int] = _scatrs.MAX_WIGNER_D_VALUES
"""The most values one :func:`wigner_d` call returns: ``2**28``, 2 GiB of float64.

The bound applies to ``max(x.size, 1) * (l_max + 1)``, so ``l_max`` is bounded even for an
empty ``x``. It keeps a request from being accepted by an overcommitting allocator and then
killed for lack of memory; it may be raised in a later release.
"""

_INT32_MIN: Final = -(2**31)
_INT32_MAX: Final = 2**31 - 1


def _integer(value: SupportsIndex, name: str) -> int:
    """``value`` as an int; anything without ``__index__``, and bools, raise TypeError."""
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}") from None


def gauss_legendre(n: SupportsIndex) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss–Legendre nodes and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of nodes, ``1 <= n <= MAX_GAUSS_LEGENDRE_POINTS`` (``2**17``). The rule
        integrates polynomials of degree up to ``2 * n - 1`` exactly. Python and numpy
        integers are accepted.

    Returns
    -------
    nodes : numpy.ndarray
        Nodes in ascending order, shape ``(n,)``.
    weights : numpy.ndarray
        Weights, shape ``(n,)``.

    Raises
    ------
    InvalidInputError
        If ``n < 1`` or ``n > MAX_GAUSS_LEGENDRE_POINTS``.
    TypeError
        If ``n`` is not an integer; ``float`` and ``bool`` values are rejected.
    NotConvergedError
        If a Newton iteration fails to converge. No ``n`` where this happens is known.

    Notes
    -----
    The rule is computed in double-double arithmetic and rounded once, so nodes and weights
    are accurate to about half an ulp, and the rule is exactly symmetric.

    The cost is O(n**2) time and O(n) memory: about ``28 ns * n**2`` on an Apple M4 Pro, which
    is 32 ms at n = 1000, 0.72 s at n = 5000, 11 s at n = 20000 and about 8 minutes at the cap;
    x86-64 builds without FMA are roughly ten times slower. The call releases the GIL, so other
    Python threads keep running, but it cannot be interrupted: Ctrl+C takes effect only once it
    returns. Compute a rule once and reuse it.
    """
    n = _integer(n, "n")
    if n < 1:
        raise InvalidInputError(f"invalid input: Gauss-Legendre rule needs n >= 1, got {n}")
    if n > MAX_GAUSS_LEGENDRE_POINTS:
        raise InvalidInputError(
            f"invalid input: Gauss-Legendre rule needs n <= {MAX_GAUSS_LEGENDRE_POINTS}, got {n}"
        )
    return _scatrs.gauss_legendre(n)


def wigner_d(
    m: SupportsIndex, n: SupportsIndex, l_max: SupportsIndex, x: ArrayLike
) -> NDArray[np.float64]:
    r"""Wigner d-functions :math:`d^l_{mn}(\theta)` for :math:`l = 0, \ldots, l_{max}`.

    Uses the convention of Mishchenko, Travis & Lacis (2002), Appendix B, for example
    :math:`d^1_{01}(\theta) = \sin\theta / \sqrt{2}`.

    Parameters
    ----------
    m, n : int
        Orders, any 32-bit signed integers.
    l_max : int
        Largest degree (at least 0).
    x : array_like
        :math:`\cos\theta`, with every element in [-1, 1]. An aligned, C-contiguous,
        native-endian float64 array is read in place; anything else real (integer or other
        float dtypes, other memory layouts, lists, scalars) is converted to one first.

    Returns
    -------
    numpy.ndarray
        Shape ``np.shape(x) + (l_max + 1,)``. Entries with ``l < max(|m|, |n|)`` are zero.

    Raises
    ------
    InvalidInputError
        If ``l_max < 0``, if an element of ``x`` is NaN or outside [-1, 1], or if
        ``max(x.size, 1) * (l_max + 1) > MAX_WIGNER_D_VALUES`` (``2**28``).
    TypeError
        If ``m``, ``n`` or ``l_max`` is not an integer (``float`` and ``bool`` values are
        rejected), or if ``x`` is not real: complex, boolean, string, object, datetime or
        timedelta.
    OverflowError
        If ``m`` or ``n`` is outside the 32-bit signed range.
    MemoryError
        If the output cannot be allocated.

    Notes
    -----
    The absolute error is within ``16 * (l + 1) * 2**-52``, a bound measured against Arb (not
    proven): for degrees up to 4000 with general orders, and up to 2.5e5 for
    ``(m, n) = (0, 0)`` and ``(0, 2)``.

    The cost is O(max(|m|, |n|)) per element of ``x`` for the start value, then about 3 ns per
    degree (2.6 to 3.7 ns on an Apple M4 Pro); the output takes ``8 * x.size * (l_max + 1)``
    bytes. The call releases the GIL but cannot be interrupted.
    """
    m, n, l_max = _integer(m, "m"), _integer(n, "n"), _integer(l_max, "l_max")
    for name, value in (("m", m), ("n", n)):
        if not _INT32_MIN <= value <= _INT32_MAX:
            raise OverflowError(f"{name} = {value} is outside the 32-bit signed range")
    if l_max < 0:
        raise InvalidInputError(f"invalid input: wigner_d needs l_max >= 0, got {l_max}")
    xs = np.asarray(x)
    if xs.dtype.kind not in "iuf":
        raise TypeError(f"x must be real numbers, not an array of dtype {xs.dtype}")
    if max(xs.size, 1) * (l_max + 1) > MAX_WIGNER_D_VALUES:
        raise InvalidInputError(
            "invalid input: wigner_d needs max(x.size, 1) * (l_max + 1) <= "
            f"{MAX_WIGNER_D_VALUES}, got x.size = {xs.size}, l_max = {l_max}"
        )
    # Aligned, contiguous, native-endian float64: what the extension reads in place.
    xs = np.require(xs, dtype=np.float64, requirements=["C", "A"])
    out = _scatrs.wigner_d(m, n, l_max, xs.reshape(-1))
    return out.reshape((*xs.shape, l_max + 1))
