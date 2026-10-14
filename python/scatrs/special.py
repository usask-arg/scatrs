"""Special functions and quadrature rules (thin wrappers over the Rust core).

Arguments are converted here (see ``scatrs._args``), their values are checked by the Rust
core, and the computation runs in Rust with the GIL released. See :mod:`scatrs.errors` for
which exception each kind of bad argument raises.
"""

from __future__ import annotations

from typing import Final, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs

__all__ = ["MAX_GAUSS_LEGENDRE_POINTS", "MAX_WIGNER_D_VALUES", "gauss_legendre", "wigner_d"]

MAX_GAUSS_LEGENDRE_POINTS: Final[int] = _scatrs.MAX_GAUSS_LEGENDRE_POINTS
"""The largest ``n`` that :func:`gauss_legendre` accepts: ``2**17 = 131072``."""

MAX_WIGNER_D_VALUES: Final[int] = _scatrs.MAX_WIGNER_D_VALUES
"""The most values one :func:`wigner_d` call returns: ``2**28``, 2 GiB of float64.

The bound applies to ``max(x.size, 1) * (l_max + 1)``, so ``l_max`` is bounded even for an
empty ``x``. It keeps a request from being accepted by an overcommitting allocator and then
killed for lack of memory; it may be raised in a later release.
"""


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
    For ``n <= 100`` the rule is computed in double-double arithmetic and rounded once, so
    nodes and weights are accurate to about half an ulp. Above, it comes from Bogaert's (2014)
    asymptotic expansions, with node absolute errors and weight relative errors of at most
    ``16 * 2**-52`` (measured: 2.4 and 3.6 units of ``2**-52``). Either way the weights next to
    +-1 are as accurate as the others, and the rule is exactly symmetric.

    The cost is ``O(n)``: about 1 ms at n = 1000 and 40 ms at the cap on an Apple M4 Pro. The
    call releases the GIL, so other Python threads keep running.
    """
    return _scatrs.gauss_legendre(_args.size(n, "n"))


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
        float dtypes, other memory layouts, lists, scalars) is converted to one first. Do not
        modify or resize ``x`` from another thread during the call.

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
    ValueError
        Raised by numpy if ``x`` is a ragged nested sequence.
    OverflowError
        If ``m`` or ``n`` is outside the 32-bit signed range.
    BufferError
        If another extension built on rust-numpy holds a writeable borrow of ``x``.
    MemoryError
        If the output cannot be allocated.

    Notes
    -----
    The absolute error is within ``16 * (l + 1) * 2**-52``, a bound measured against Arb (not
    proven): up to degree 2.5e5 for ``(m, n) = (0, 0)`` and ``(0, 2)``, and for general orders
    at sampled degrees up to 2000 (up to 4000 in two cases next to the poles).

    The cost is ``O(max(|m|, |n|))`` per element of ``x`` for the start value, then about 3 ns
    per degree (2.6 to 3.7 ns on an Apple M4 Pro); the output takes ``8 * x.size * (l_max + 1)``
    bytes. The call releases the GIL but cannot be interrupted.
    """
    m, n = _args.int32(m, "m"), _args.int32(n, "n")
    l_max = _args.size(l_max, "l_max")
    xs = _args.real_array(x, "x")
    shape = xs.shape
    # A fresh view, not the caller's array: see _args.real_array.
    out = _scatrs.wigner_d(m, n, l_max, xs.reshape(-1))
    return out.reshape((*shape, l_max + 1))
