"""Lorenz–Mie scattering by homogeneous spheres (vectorized wrappers over the Rust core).

The size parameters ``x`` and refractive indices ``m`` broadcast against each other to a shape
``S``; every result has that shape, extended by the orders or the angles. Conventions are those
of Bohren & Huffman (1983): time dependence e^(−iωt), so absorbing media have ``Im m >= 0``, and
``x = 2 pi r / wavelength``. See :doc:`/theory/mie` for the method and
:doc:`/validation/accuracy` for the validated domain and the accuracy.

Each function runs entirely in Rust, with the GIL released, over all elements in turn.
``derivatives`` is ``True`` (all of ``"x"``, ``"m_real"``, ``"m_imag"``) or a sequence of those
names; the result's ``d`` method (for example :meth:`MieEfficiencies.d`) then returns the
derivative with respect to one of them, in the same structure. Derivatives cost little: all
three together take 1.3 to 1.8 times as long as the values alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs

__all__ = [
    "MAX_MIE_ORDER",
    "MAX_MIE_VALUES",
    "PARAMETERS",
    "MieAmplitudes",
    "MieCoefficients",
    "MieEfficiencies",
    "amplitudes",
    "coefficients",
    "default_n_max",
    "efficiencies",
]

MAX_MIE_ORDER: Final[int] = _scatrs.MAX_MIE_ORDER
"""The largest number of terms accepted: ``2**20``."""

MAX_MIE_VALUES: Final[int] = _scatrs.MAX_MIE_VALUES
"""The most complex values one call returns, counting derivatives: ``2**27`` (2 GiB)."""

PARAMETERS: Final = ("x", "m_real", "m_imag")
"""The parameters ``derivatives`` may name."""


def _parameters(derivatives: bool | Sequence[str]) -> tuple[str, ...]:
    if derivatives is True:
        return PARAMETERS
    if derivatives is False:
        return ()
    if isinstance(derivatives, str):
        raise TypeError("derivatives must be a bool or a sequence of parameter names")
    names = tuple(derivatives)
    for name in names:
        if name not in PARAMETERS:
            raise ValueError(f"unknown parameter {name!r}; expected one of {PARAMETERS}")
    return names


def _inputs(
    x: ArrayLike, m: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.complex128], tuple[int, ...]]:
    """``x`` and ``m`` broadcast to one shape, as fresh views for Rust (see ``scatrs._args``)."""
    xs = _args.real_array(x, "x")
    ms = _args.complex_array(m, "m")
    xb, mb = np.broadcast_arrays(xs, ms)
    return xb, mb, xb.shape


def _n_max(n_max: SupportsIndex | None) -> int | None:
    return None if n_max is None else _args.size(n_max, "n_max")


def _check(params: tuple[str, ...], param: str) -> None:
    if param not in params and not (param == "m" and {"m_real", "m_imag"} & set(params)):
        raise ValueError(f"no derivative with respect to {param!r}; computed: {params}")


def default_n_max(x: ArrayLike) -> NDArray[np.int64]:
    """The default number of terms, ``ceil(x + 6 x**(1/3) + 3)``, for each size parameter.

    Parameters
    ----------
    x : array_like
        Size parameters (real).

    Returns
    -------
    numpy.ndarray
        int64, the shape of ``x``.
    """
    xs = _args.real_array(x, "x")
    return _scatrs.mie_default_n_max(xs.reshape(xs.shape)).astype(np.int64).reshape(xs.shape)


@dataclass(frozen=True)
class MieEfficiencies:
    """Efficiencies and asymmetry parameter of spheres, each an array of shape ``S``.

    Attributes
    ----------
    q_ext, q_sca, q_abs, q_back : numpy.ndarray
        Extinction, scattering, absorption and backscattering efficiencies (cross sections
        divided by pi r**2). ``q_abs`` is computed without the cancellation of
        ``q_ext - q_sca``, and is exactly 0 for real ``m``; ``q_ext = q_sca + q_abs``.
    g : numpy.ndarray
        Asymmetry parameter.
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    q_ext: NDArray[np.float64]
    q_sca: NDArray[np.float64]
    q_abs: NDArray[np.float64]
    q_back: NDArray[np.float64]
    g: NDArray[np.float64]
    params: tuple[str, ...] = ()
    _jacobian: NDArray[np.float64] | None = field(default=None, repr=False)

    def d(self, param: str) -> MieEfficiencies:
        """The derivatives with respect to ``param`` (``"x"``, ``"m_real"`` or ``"m_imag"``).

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        _check(self.params, param)
        if param == "m":
            raise ValueError("efficiencies are real: use 'm_real' or 'm_imag'")
        assert self._jacobian is not None
        rows = self._jacobian[PARAMETERS.index(param)]
        return MieEfficiencies(*rows)


@dataclass(frozen=True)
class MieCoefficients:
    """Mie coefficients aₙ, bₙ of spheres, arrays of shape ``S + (N,)``, with ``N`` the largest
    number of terms of any sphere; ``a[..., n - 1]`` is aₙ, zero beyond each sphere's
    ``n_terms``.

    Attributes
    ----------
    a, b : numpy.ndarray
        complex128.
    n_terms : numpy.ndarray
        int64, shape ``S``: the number of terms of each sphere.
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    a: NDArray[np.complex128]
    b: NDArray[np.complex128]
    n_terms: NDArray[np.int64]
    params: tuple[str, ...] = ()
    _derivatives: NDArray[np.complex128] | None = field(default=None, repr=False)

    def d(self, param: str) -> MieCoefficients:
        """The derivatives with respect to ``param``: ``"x"``, ``"m_real"``, ``"m_imag"``, or
        ``"m"`` for the complex derivative d/dm (which equals ∂/∂(Re m); ∂/∂(Im m) is i d/dm).

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        _check(self.params, param)
        assert self._derivatives is not None
        da_dx, da_dm, db_dx, db_dm = self._derivatives
        if param == "x":
            return MieCoefficients(da_dx, db_dx, self.n_terms)
        if param == "m_imag":
            return MieCoefficients(1j * da_dm, 1j * db_dm, self.n_terms)
        return MieCoefficients(da_dm, db_dm, self.n_terms)


@dataclass(frozen=True)
class MieAmplitudes:
    """Amplitude scattering functions S₁(μ), S₂(μ), arrays of shape ``S + mu.shape``.

    Attributes
    ----------
    s1, s2 : numpy.ndarray
        complex128 (Bohren & Huffman 1983, Eq. 4.74).
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    s1: NDArray[np.complex128]
    s2: NDArray[np.complex128]
    params: tuple[str, ...] = ()
    _derivatives: NDArray[np.complex128] | None = field(default=None, repr=False)

    def d(self, param: str) -> MieAmplitudes:
        """The derivatives with respect to ``param``: ``"x"``, ``"m_real"``, ``"m_imag"``, or
        ``"m"`` for the complex derivative d/dm.

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        _check(self.params, param)
        assert self._derivatives is not None
        ds1_dx, ds2_dx, ds1_dm, ds2_dm = self._derivatives
        if param == "x":
            return MieAmplitudes(ds1_dx, ds2_dx)
        if param == "m_imag":
            return MieAmplitudes(1j * ds1_dm, 1j * ds2_dm)
        return MieAmplitudes(ds1_dm, ds2_dm)


_SHARED_DOC = """
    Parameters
    ----------
    x : array_like
        Size parameters, 2 pi r / wavelength (real).
    m : array_like
        Refractive indices relative to the medium (real or complex), broadcast against ``x``.
    {extra}n_max : int, optional
        The number of terms; by default ``ceil(x + 6 x**(1/3) + 3)``. Fewer terms than the
        default are outside the validated domain.
    derivatives : bool or sequence of str, optional
        Also compute derivatives with respect to these of ``"x"``, ``"m_real"`` and
        ``"m_imag"`` (``True`` for all three); read them with the result's ``d`` method.
    allow_unvalidated : bool, optional
        Compute outside the validated domain (``1e-6 <= x <= 2e4``, ``0 < Re m <= 10``,
        ``0 <= Im m <= 10``) instead of raising
        :class:`~scatrs.errors.OutsideValidatedDomainError`.

    Raises
    ------
    InvalidInputError
        For ``x`` outside ``[1e-30, 1e6]``, non-finite ``m``, ``Re m < 0`` or ``Im m < 0``
        (gain media), ``|m|`` outside ``[1e-6, 1e3]``, ``n_max`` outside
        ``[1, MAX_MIE_ORDER]``{extra_raise}, or an output above ``MAX_MIE_VALUES``. The
        message names the first failing element (in C order).
    OutsideValidatedDomainError
        For valid inputs outside the validated domain, unless ``allow_unvalidated``.
    TypeError
        For non-numeric ``x`` or ``m`` (complex ``x``), or a ``derivatives`` string.
    ValueError
        For an unknown parameter name, or (from numpy) shapes that do not broadcast.
    BufferError
        If another rust-numpy extension holds a writeable borrow of an input.
    MemoryError
        If the outputs cannot be allocated.
"""


def efficiencies(
    x: ArrayLike,
    m: ArrayLike,
    *,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> MieEfficiencies:
    params = _parameters(derivatives)
    xb, mb, shape = _inputs(x, m)
    values, jacobian = _scatrs.mie_efficiencies(
        xb, mb, _n_max(n_max), bool(params), bool(allow_unvalidated)
    )
    fields = [v.reshape(shape) for v in values]
    jac = None if jacobian is None else jacobian.reshape((3, 5, *shape))
    return MieEfficiencies(*fields, params=params, _jacobian=jac)


efficiencies.__doc__ = """Efficiencies and asymmetry parameter of spheres.

    Returns
    -------
    MieEfficiencies
        Each field has the broadcast shape of ``x`` and ``m``.
""" + _SHARED_DOC.format(extra="", extra_raise="")


def coefficients(
    x: ArrayLike,
    m: ArrayLike,
    *,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> MieCoefficients:
    params = _parameters(derivatives)
    xb, mb, shape = _inputs(x, m)
    terms, a, b, d = _scatrs.mie_coefficients(
        xb, mb, _n_max(n_max), bool(params), bool(allow_unvalidated)
    )
    width = a.shape[1]
    n_terms = terms.astype(np.int64).reshape(shape)
    dd = None if d is None else d.reshape((4, *shape, width))
    return MieCoefficients(
        a.reshape((*shape, width)), b.reshape((*shape, width)), n_terms, params, dd
    )


coefficients.__doc__ = """Mie coefficients aₙ and bₙ of spheres.

    Returns
    -------
    MieCoefficients
        Shape ``S + (N,)``, zero-padded to the largest number of terms ``N``.
""" + _SHARED_DOC.format(extra="", extra_raise="")


def amplitudes(
    x: ArrayLike,
    m: ArrayLike,
    mu: ArrayLike,
    *,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> MieAmplitudes:
    params = _parameters(derivatives)
    xb, mb, shape = _inputs(x, m)
    mus = _args.real_array(mu, "mu")
    s, d = _scatrs.mie_amplitudes(
        xb, mb, mus.reshape(mus.shape), _n_max(n_max), bool(params), bool(allow_unvalidated)
    )
    out = (*shape, *mus.shape)
    dd = None if d is None else d.reshape((4, *out))
    return MieAmplitudes(s[0].reshape(out), s[1].reshape(out), params, dd)


amplitudes.__doc__ = """Amplitude scattering functions S₁ and S₂ of spheres.

    Returns
    -------
    MieAmplitudes
        Shape ``S + mu.shape``.
""" + _SHARED_DOC.format(
    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n    ",
    extra_raise=", ``mu`` outside [-1, 1] or NaN",
)
