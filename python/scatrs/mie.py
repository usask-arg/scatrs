"""Lorenz–Mie scattering by homogeneous spheres (vectorized wrappers over the Rust core).

The size parameters ``x`` and refractive indices ``m`` broadcast against each other to a shape
``S``; every result has that shape, extended by the orders or the angles. Conventions are those
of Bohren & Huffman (1983): time dependence e^(−iωt), so absorbing media have ``Im m >= 0``, and
``x = 2 pi r / wavelength``. See :doc:`/theory/mie` for the method and
:doc:`/validation/accuracy` for the validated domain and the accuracy.

Each function runs entirely in Rust, with the GIL released, over all elements in turn.
``derivatives`` is ``True`` (all of ``"x"``, ``"m_real"``, ``"m_imag"``) or a sequence of those
names; the result's ``d`` method (for example :meth:`MieEfficiencies.d`) then returns the
derivative with respect to one of them, in the same structure. Derivatives cost less than
finite differences would: all three together take 1.2 to 2.3 times as long as the values
alone.

Results are frozen dataclasses of read-only arrays (copy an array to modify it); they compare
by identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, SupportsIndex, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs

__all__ = [
    "MAX_GREEK_ORDER",
    "MAX_MIE_ORDER",
    "MAX_MIE_VALUES",
    "PARAMETERS",
    "MieAmplitudes",
    "MieCoefficients",
    "MieEfficiencies",
    "MieGreek",
    "MiePhaseMatrix",
    "amplitudes",
    "coefficients",
    "default_n_max",
    "efficiencies",
    "greek",
    "phase_matrix",
]

MAX_MIE_ORDER: Final[int] = _scatrs.MAX_MIE_ORDER
"""The largest number of terms accepted: ``2**20``."""

MAX_MIE_VALUES: Final[int] = _scatrs.MAX_MIE_VALUES
"""The most complex values one call returns, counting derivatives: ``2**27`` (2 GiB); twice as
many real values for :func:`efficiencies`, :func:`phase_matrix` and :func:`greek`."""

MAX_GREEK_ORDER: Final[int] = _scatrs.MAX_GREEK_ORDER
"""The largest ``order`` :func:`greek` accepts: ``2**21``, twice :data:`MAX_MIE_ORDER`."""

PARAMETERS: Final = ("x", "m_real", "m_imag")
"""The parameters ``derivatives`` may name."""

_A = TypeVar("_A")


def _parameters(derivatives: bool | Sequence[str]) -> tuple[str, ...]:
    if isinstance(derivatives, (bool, np.bool_)):
        return PARAMETERS if derivatives else ()
    if isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
        raise TypeError(
            "derivatives must be a bool or a sequence of parameter names, "
            f"not {type(derivatives).__name__}"
        )
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
    # broadcast_arrays returns an input of the broadcast shape as it is: take views.
    xb, mb = np.broadcast_arrays(xs, ms)
    return xb.view(), mb.view(), xb.shape


def _n_max(n_max: SupportsIndex | None) -> int | None:
    return None if n_max is None else _args.size(n_max, "n_max")


def _frozen(a: _A) -> _A:
    """A read-only view of ``a``. An array that owns its data can be made writeable again; a view
    of a read-only array cannot. (Numpy scalars, which ``d`` gives for scalar inputs, are
    immutable already.)"""
    if not isinstance(a, np.ndarray):
        return a
    a.flags.writeable = False
    return a.view()


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

    Raises
    ------
    InvalidInputError
        For ``x`` outside ``[1e-30, 1e6]`` (or NaN); the message names the first failing
        element (in C order).
    TypeError
        For non-real ``x``.
    """
    xs = _args.real_array(x, "x")
    return _scatrs.mie_default_n_max(xs.reshape(xs.shape)).astype(np.int64).reshape(xs.shape)


@dataclass(frozen=True, eq=False)
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
        p = _check_real(self.params, param, "efficiencies")
        assert self._jacobian is not None
        rows = self._jacobian[p]
        return MieEfficiencies(*rows)


@dataclass(frozen=True, eq=False)
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
            return MieCoefficients(_frozen(1j * da_dm), _frozen(1j * db_dm), self.n_terms)
        return MieCoefficients(da_dm, db_dm, self.n_terms)


@dataclass(frozen=True, eq=False)
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
            return MieAmplitudes(_frozen(1j * ds1_dm), _frozen(1j * ds2_dm))
        return MieAmplitudes(ds1_dm, ds2_dm)


def _check_real(params: tuple[str, ...], param: str, what: str) -> int:
    _check(params, param)
    if param == "m":
        raise ValueError(f"{what} are real: use 'm_real' or 'm_imag'")
    return PARAMETERS.index(param)


@dataclass(frozen=True, eq=False)
class MiePhaseMatrix:
    """The phase matrix of spheres, arrays of shape ``S + mu.shape`` (Bohren & Huffman 1983,
    §4.4). For a sphere ``F22 = F11`` and ``F44 = F33``; the other elements are 0.

    Attributes
    ----------
    f11, f12, f33, f34 : numpy.ndarray
        ``(|S1|**2 + |S2|**2)/2``, ``(|S2|**2 - |S1|**2)/2``, ``Re(S2 conj(S1))`` and
        ``Im(S2 conj(S1))``, times ``4/(x**2 q_sca)`` (normalized: ``F11`` averages to 1 over
        the sphere of directions) or ``4/x**2`` (unnormalized: per unit geometric cross
        section, which sums over a size distribution).
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    f11: NDArray[np.float64]
    f12: NDArray[np.float64]
    f33: NDArray[np.float64]
    f34: NDArray[np.float64]
    params: tuple[str, ...] = ()
    _derivatives: NDArray[np.float64] | None = field(default=None, repr=False)

    def d(self, param: str) -> MiePhaseMatrix:
        """The derivatives with respect to ``param`` (``"x"``, ``"m_real"`` or ``"m_imag"``);
        normalized derivatives include that of ``q_sca``.

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        p = _check_real(self.params, param, "phase-matrix elements")
        assert self._derivatives is not None
        return MiePhaseMatrix(*self._derivatives[p])


@dataclass(frozen=True, eq=False)
class MieGreek:
    """Greek (generalized-spherical-function) expansion coefficients of the phase matrix of
    spheres, arrays of shape ``S + (L + 1,)``; ``alpha1[..., l]`` is the coefficient of order
    ``l`` (Mishchenko, Travis & Lacis 2002, Eqs. 4.75–4.80):
    ``F11 = sum alpha1[l] d00_l``, ``F22 + F33 = sum (alpha2 + alpha3)[l] d22_l``,
    ``F22 - F33 = sum (alpha2 - alpha3)[l] d2-2_l``, ``F44 = sum alpha4[l] d00_l``,
    ``F12 = sum beta1[l] d02_l``, ``F34 = sum beta2[l] d02_l``, with ``dmn_l`` the Wigner
    d-functions of :func:`scatrs.special.wigner_d` at the cosine of the scattering angle.

    Attributes
    ----------
    alpha1, alpha2, alpha3, alpha4, beta1, beta2 : numpy.ndarray
        float64. Normalized, ``alpha1[..., 0]`` is 1 and ``alpha1[..., 1]`` is ``3 g``;
        unnormalized, they are those times ``q_sca``. Orders above a sphere's ``2 n_terms``
        are exactly 0.
    n_terms : numpy.ndarray
        int64, shape ``S``: the number of terms of each sphere.
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    alpha1: NDArray[np.float64]
    alpha2: NDArray[np.float64]
    alpha3: NDArray[np.float64]
    alpha4: NDArray[np.float64]
    beta1: NDArray[np.float64]
    beta2: NDArray[np.float64]
    n_terms: NDArray[np.int64]
    params: tuple[str, ...] = ()
    _derivatives: NDArray[np.float64] | None = field(default=None, repr=False)

    def d(self, param: str) -> MieGreek:
        """The derivatives with respect to ``param`` (``"x"``, ``"m_real"`` or ``"m_imag"``).

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        p = _check_real(self.params, param, "Greek coefficients")
        assert self._derivatives is not None
        return MieGreek(*self._derivatives[p], self.n_terms)


_SHARED_DOC = """
    Parameters
    ----------
    x : array_like
        Size parameters, 2 pi r / wavelength (real).
    m : array_like
        Refractive indices relative to the medium (real or complex), broadcast against ``x``.
        Aligned, C-contiguous, native-endian float64 ``x`` and complex128 ``m`` are read in
        place; anything else is converted first. Do not modify or resize them from another
        thread during the call.
    {extra}n_max : int, optional
        The number of terms; by default ``ceil(x + 6 x**(1/3) + 3)``. Fewer terms than the
        default are outside the validated domain.
    derivatives : bool or sequence of str, optional
        Also compute derivatives with respect to these of ``"x"``, ``"m_real"`` and
        ``"m_imag"`` (``True`` for all three); read them with the result's ``d`` method.
    allow_unvalidated : bool, optional
        Compute outside the validated domain (``1e-6 <= x <= 2e4``, ``0 < Re m <= 10``,
        ``0 <= Im m <= 10``, ``|m| >= 0.1``) instead of raising
        :class:`~scatrs.errors.OutsideValidatedDomainError`.

    Raises
    ------
    InvalidInputError
        For ``x`` outside ``[1e-30, 1e6]``, non-finite ``m``, ``Re m < 0`` or ``Im m < 0``
        (gain media), ``|m|`` outside ``[1e-6, 1e3]``, ``n_max`` outside
        ``[1, MAX_MIE_ORDER]``{extra_raise}, or an output above ``MAX_MIE_VALUES`` complex
        values (``2 * MAX_MIE_VALUES`` real values). The message names the first failing
        element (in C order).
    OutsideValidatedDomainError
        For valid inputs outside the validated domain, unless ``allow_unvalidated``.
    TypeError
        For non-numeric ``x`` or ``m`` (complex ``x``), a non-integer ``n_max`` (``float``
        and ``bool`` values are rejected), ``derivatives`` that is neither a bool nor a
        sequence of names, or a non-bool ``allow_unvalidated``{extra_type}.
    ValueError
        For an unknown parameter name, or (from numpy) shapes that do not broadcast.
    BufferError
        If another rust-numpy extension holds a writeable borrow of an input.
    MemoryError
        If the outputs cannot be allocated.

    Notes
    -----
    The cost per sphere grows like ``x + |m| x`` (see :doc:`/theory/mie`): the largest
    accepted ``|m| x`` take seconds per sphere. The call releases the GIL, so other Python
    threads keep running, but it cannot be interrupted: Ctrl+C takes effect only once it
    returns.
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
        xb, mb, _n_max(n_max), bool(params), _args.flag(allow_unvalidated, "allow_unvalidated")
    )
    values, jacobian = _frozen(values), _frozen(jacobian)
    fields = [v.reshape(shape) for v in values]
    jac = None if jacobian is None else jacobian.reshape((3, 5, *shape))
    return MieEfficiencies(*fields, params=params, _jacobian=jac)


efficiencies.__doc__ = """Efficiencies and asymmetry parameter of spheres.

    Returns
    -------
    MieEfficiencies
        Each field has the broadcast shape of ``x`` and ``m``.
""" + _SHARED_DOC.format(extra="", extra_raise="", extra_type="")


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
        xb, mb, _n_max(n_max), bool(params), _args.flag(allow_unvalidated, "allow_unvalidated")
    )
    a, b, d = _frozen(a), _frozen(b), _frozen(d)
    width = a.shape[1]
    n_terms = _frozen(terms.astype(np.int64)).reshape(shape)
    dd = None if d is None else d.reshape((4, *shape, width))
    return MieCoefficients(
        a.reshape((*shape, width)), b.reshape((*shape, width)), n_terms, params, dd
    )


coefficients.__doc__ = """Mie coefficients aₙ and bₙ of spheres.

    Returns
    -------
    MieCoefficients
        Shape ``S + (N,)``, zero-padded to the largest number of terms ``N``.
""" + _SHARED_DOC.format(extra="", extra_raise="", extra_type="")


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
        xb,
        mb,
        mus.reshape(mus.shape),
        _n_max(n_max),
        bool(params),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
    )
    s, d = _frozen(s), _frozen(d)
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
    extra_raise=",\n        ``mu`` outside [-1, 1] or NaN",
    extra_type="",
)


def phase_matrix(
    x: ArrayLike,
    m: ArrayLike,
    mu: ArrayLike,
    *,
    normalized: bool = True,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> MiePhaseMatrix:
    params = _parameters(derivatives)
    xb, mb, shape = _inputs(x, m)
    mus = _args.real_array(mu, "mu")
    f, d = _scatrs.mie_phase_matrix(
        xb,
        mb,
        mus.reshape(mus.shape),
        _n_max(n_max),
        bool(params),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
        _args.flag(normalized, "normalized"),
    )
    f, d = _frozen(f), _frozen(d)
    out = (*shape, *mus.shape)
    dd = None if d is None else d.reshape((3, 4, *out))
    return MiePhaseMatrix(*(v.reshape(out) for v in f), params=params, _derivatives=dd)


phase_matrix.__doc__ = """The phase matrix F11, F12, F33, F34 of spheres.

    Returns
    -------
    MiePhaseMatrix
        Shape ``S + mu.shape``.
""" + _SHARED_DOC.format(
    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n"
    "    normalized : bool, optional\n        Divide by ``q_sca`` (the default), so that F11"
    " averages to 1.\n    ",
    extra_raise=",\n        ``mu`` outside [-1, 1] or NaN",
    extra_type=" or ``normalized``",
)


def greek(
    x: ArrayLike,
    m: ArrayLike,
    order: SupportsIndex | None = None,
    *,
    normalized: bool = True,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> MieGreek:
    params = _parameters(derivatives)
    xb, mb, shape = _inputs(x, m)
    terms, g, d = _scatrs.mie_greek(
        xb,
        mb,
        None if order is None else _args.size(order, "order"),
        _n_max(n_max),
        bool(params),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
        _args.flag(normalized, "normalized"),
    )
    g, d = _frozen(g), _frozen(d)
    width = g.shape[-1]
    out = (*shape, width)
    n_terms = _frozen(terms.astype(np.int64)).reshape(shape)
    dd = None if d is None else d.reshape((3, 6, *out))
    return MieGreek(*(v.reshape(out) for v in g), n_terms, params, dd)


greek.__doc__ = (
    """Greek expansion coefficients of the phase matrix of spheres.

    Exact up to rounding: the phase matrix at the nodes of a Gauss–Legendre rule with
    ``n_terms + ceil((L + 1)/2)`` points, projected onto the Wigner d-functions.

    Returns
    -------
    MieGreek
        Shape ``S + (L + 1,)``.
"""
    + _SHARED_DOC.format(
        extra="order : int, optional\n        The highest order L; by default ``2 n_terms`` of the"
        " largest sphere, which is\n        all of them. Orders above ``2 n_terms`` are exactly 0."
        "\n    normalized : bool, optional\n        Normalize ``alpha1[..., 0]`` to 1 (the"
        " default), or to ``q_sca``.\n    ",
        extra_raise=",\n        a negative ``order`` or one above ``MAX_GREEK_ORDER``, a sphere"
        " whose rule would\n        need more than ``2**17`` nodes",
        extra_type=" or ``normalized``,\n        or a non-integer ``order``",
    )
    + """
    Projecting adds about ``n_terms * L`` operations per sphere to the amplitudes at the
    ``n_terms + ceil((L + 1)/2)`` nodes: all orders take about 0.09 s at x = 1e3 and
    7.6 s at x = 1e4.
"""
)
