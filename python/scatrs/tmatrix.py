"""Scattering by randomly oriented spheroids, by the T-matrix method (vectorized wrappers over
the Rust core).

A spheroid has semi-axes ``a, a, c``, ``c`` along its symmetry axis, and aspect ratio
``aspect_ratio = a / c``: oblate above 1, prolate below 1, a sphere at 1. Its size is the size
parameter ``x = 2 pi r_v / wavelength`` of the sphere of equal volume (``r_v**3 = a**2 c``),
which also normalizes the efficiencies (cross sections divided by ``pi r_v**2``) and the
unnormalized phase matrix and Greek coefficients. ``x``, ``m`` and ``aspect_ratio`` broadcast
against each other to a shape ``S``; every result has that shape, extended by the orders or the
angles. Conventions are those of :mod:`scatrs.mie`, averaged uniformly over orientations. See
:doc:`/theory/spheroids` for the method and :doc:`/validation/accuracy` for the validated domain.

Each spheroid's T-matrix is converged to the relative tolerance ``rtol`` in its number of
orders and quadrature nodes; the results carry ``n_terms``, ``n_theta`` and an ``error``
estimate per spheroid. ``derivatives`` is ``True`` (all of ``"x"``, ``"aspect_ratio"``,
``"m_real"``, ``"m_imag"``) or a sequence of those names; the result's ``d`` method returns the
derivative with respect to one of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs
from scatrs.mie import MieEfficiencies, MieGreek, _frozen

__all__ = [
    "MAX_TMATRIX_ORDER",
    "PARAMETERS",
    "SpheroidEfficiencies",
    "SpheroidGreek",
    "SpheroidPhaseMatrix",
    "TMatrix",
    "compute",
    "efficiencies",
    "greek",
    "phase_matrix",
]

MAX_TMATRIX_ORDER: Final[int] = _scatrs.MAX_TMATRIX_ORDER
"""The largest number of orders N accepted: 500."""

PARAMETERS: Final = ("x", "aspect_ratio", "m_real", "m_imag")
"""The parameters ``derivatives`` may name."""

Precision = Literal["auto", "f64", "double-double"]


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


def _index(params: tuple[str, ...], param: str) -> int:
    if param not in params:
        if param == "m":
            raise ValueError("these outputs are real: use 'm_real' or 'm_imag'")
        raise ValueError(f"no derivative with respect to {param!r}; computed: {params}")
    return PARAMETERS.index(param)


def _inputs(
    x: ArrayLike, m: ArrayLike, aspect_ratio: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64], tuple[int, ...]]:
    xs = _args.real_array(x, "x")
    ms = _args.complex_array(m, "m")
    es = _args.real_array(aspect_ratio, "aspect_ratio")
    xb, mb, eb = np.broadcast_arrays(xs, ms, es)
    shape = xb.shape
    return (
        np.ascontiguousarray(xb).reshape(-1),
        np.ascontiguousarray(mb).reshape(-1),
        np.ascontiguousarray(eb).reshape(-1),
        shape,
    )


def _precision(precision: str) -> str:
    if precision not in ("auto", "f64", "double-double"):
        raise ValueError(f"precision must be 'auto', 'f64' or 'double-double', got {precision!r}")
    return precision


def _optional_size(value: SupportsIndex | None, name: str) -> int | None:
    return None if value is None else _args.size(value, name)


@dataclass(frozen=True, eq=False)
class SpheroidEfficiencies(MieEfficiencies):
    """Orientation-averaged efficiencies of spheroids (:class:`~scatrs.mie.MieEfficiencies`,
    per unit geometric cross section of the equal-volume sphere; ``q_abs = q_ext - q_sca``),
    with the convergence information of each spheroid.

    Attributes
    ----------
    n_terms, n_theta : numpy.ndarray
        int64, shape ``S``: the number of orders and of quadrature nodes (in a half range of
        the polar angle) each T-matrix converged at.
    error : numpy.ndarray
        float64, shape ``S``: each spheroid's estimated relative error.
    """

    n_terms: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    n_theta: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    error: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    def d(self, param: str) -> MieEfficiencies:
        """The derivatives with respect to ``param`` (one of :data:`PARAMETERS`)."""
        p = _index(self.params, param)
        assert self._jacobian is not None
        return MieEfficiencies(*self._jacobian[p])


@dataclass(frozen=True, eq=False)
class SpheroidGreek(MieGreek):
    """Greek expansion coefficients of the orientation-averaged phase matrix of spheroids
    (:class:`~scatrs.mie.MieGreek`, with ``F22`` and ``F44`` independent); orders above a
    spheroid's ``2 n_terms`` are exactly 0."""

    def d(self, param: str) -> MieGreek:
        """The derivatives with respect to ``param`` (one of :data:`PARAMETERS`)."""
        p = _index(self.params, param)
        assert self._derivatives is not None
        return MieGreek(*self._derivatives[p], self.n_terms)


@dataclass(frozen=True, eq=False)
class SpheroidPhaseMatrix:
    """The orientation-averaged phase matrix of spheroids, arrays of shape ``S + mu.shape``:
    its six independent elements (the others are 0), normalized (``F11`` averages to 1 over the
    sphere of directions) or unnormalized (times ``q_sca``). For a sphere ``F22 = F11`` and
    ``F44 = F33``.

    Attributes
    ----------
    f11, f12, f22, f33, f34, f44 : numpy.ndarray
        The elements.
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    f11: NDArray[np.float64]
    f12: NDArray[np.float64]
    f22: NDArray[np.float64]
    f33: NDArray[np.float64]
    f34: NDArray[np.float64]
    f44: NDArray[np.float64]
    params: tuple[str, ...] = ()
    _derivatives: NDArray[np.float64] | None = field(default=None, repr=False)

    def d(self, param: str) -> SpheroidPhaseMatrix:
        """The derivatives with respect to ``param`` (one of :data:`PARAMETERS`); normalized
        derivatives include that of ``q_sca``."""
        p = _index(self.params, param)
        assert self._derivatives is not None
        return SpheroidPhaseMatrix(*self._derivatives[p])


def _run(
    x: ArrayLike,
    m: ArrayLike,
    aspect_ratio: ArrayLike,
    *,
    rtol: float,
    precision: str,
    n_terms: SupportsIndex | None,
    n_theta: SupportsIndex | None,
    allow_unconverged: bool,
    allow_unvalidated: bool,
    derivatives: bool | Sequence[str],
    greek: bool = False,
    order: SupportsIndex | None = None,
    mu: ArrayLike | None = None,
    normalized: bool = True,
) -> tuple[dict[str, NDArray], tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    params = _parameters(derivatives)
    xb, mb, eb, shape = _inputs(x, m, aspect_ratio)
    mus = None if mu is None else _args.real_array(mu, "mu")
    out = _scatrs.tmatrix_random(
        xb,
        mb,
        eb,
        float(rtol),
        _precision(precision),
        _optional_size(n_terms, "n_terms"),
        _optional_size(n_theta, "n_theta"),
        _args.flag(allow_unconverged, "allow_unconverged"),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
        bool(params),
        greek,
        _optional_size(order, "order"),
        None if mus is None else mus.reshape(-1),
        _args.flag(normalized, "normalized"),
    )
    return out, shape, params, () if mus is None else mus.shape


def _info(out: dict[str, NDArray], shape: tuple[int, ...]) -> dict[str, NDArray]:
    return {
        "n_terms": _frozen(out["n_terms"].astype(np.int64)).reshape(shape),
        "n_theta": _frozen(out["n_theta"].astype(np.int64)).reshape(shape),
        "error": _frozen(out["error"]).reshape(shape),
    }


def efficiencies(
    x: ArrayLike,
    m: ArrayLike,
    aspect_ratio: ArrayLike,
    *,
    rtol: float = 1e-8,
    precision: Precision = "auto",
    n_terms: SupportsIndex | None = None,
    n_theta: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unconverged: bool = False,
    allow_unvalidated: bool = False,
) -> SpheroidEfficiencies:
    out, shape, params, _ = _run(
        x,
        m,
        aspect_ratio,
        rtol=rtol,
        precision=precision,
        n_terms=n_terms,
        n_theta=n_theta,
        allow_unconverged=allow_unconverged,
        allow_unvalidated=allow_unvalidated,
        derivatives=derivatives,
    )
    q = _frozen(out["efficiencies"])
    jac = out.get("efficiencies_d")
    jac = None if jac is None else _frozen(jac).reshape((4, 5, *shape))
    return SpheroidEfficiencies(*(v.reshape(shape) for v in q), params, jac, **_info(out, shape))


def greek(
    x: ArrayLike,
    m: ArrayLike,
    aspect_ratio: ArrayLike,
    order: SupportsIndex | None = None,
    *,
    normalized: bool = True,
    rtol: float = 1e-8,
    precision: Precision = "auto",
    n_terms: SupportsIndex | None = None,
    n_theta: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unconverged: bool = False,
    allow_unvalidated: bool = False,
) -> SpheroidGreek:
    out, shape, params, _ = _run(
        x,
        m,
        aspect_ratio,
        rtol=rtol,
        precision=precision,
        n_terms=n_terms,
        n_theta=n_theta,
        allow_unconverged=allow_unconverged,
        allow_unvalidated=allow_unvalidated,
        derivatives=derivatives,
        greek=True,
        order=order,
        normalized=normalized,
    )
    g = _frozen(out["greek"])
    width = g.shape[-1]
    shaped = (*shape, width)
    d = out.get("greek_d")
    d = None if d is None else _frozen(d).reshape((4, 6, *shaped))
    info = _info(out, shape)
    return SpheroidGreek(*(v.reshape(shaped) for v in g), info["n_terms"], params, d)


def phase_matrix(
    x: ArrayLike,
    m: ArrayLike,
    aspect_ratio: ArrayLike,
    mu: ArrayLike,
    *,
    normalized: bool = True,
    rtol: float = 1e-8,
    precision: Precision = "auto",
    n_terms: SupportsIndex | None = None,
    n_theta: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unconverged: bool = False,
    allow_unvalidated: bool = False,
) -> SpheroidPhaseMatrix:
    out, shape, params, mu_shape = _run(
        x,
        m,
        aspect_ratio,
        rtol=rtol,
        precision=precision,
        n_terms=n_terms,
        n_theta=n_theta,
        allow_unconverged=allow_unconverged,
        allow_unvalidated=allow_unvalidated,
        derivatives=derivatives,
        mu=mu,
        normalized=normalized,
    )
    f = _frozen(out["phase"])
    shaped = (*shape, *mu_shape)
    d = out.get("phase_d")
    d = None if d is None else _frozen(d).reshape((4, 6, *shaped))
    return SpheroidPhaseMatrix(*(v.reshape(shaped) for v in f), params=params, _derivatives=d)


_SHARED_DOC = """
    Parameters
    ----------
    x : array_like
        Equal-volume size parameters, 2 pi r_v / wavelength (real).
    m : array_like
        Refractive indices relative to the medium (real or complex).
    aspect_ratio : array_like
        ``a / c``: oblate above 1, prolate below 1 (real, positive).
    {extra}rtol : float, optional
        The relative tolerance of each T-matrix's convergence (default ``1e-8``).
    precision : {{"auto", "f64", "double-double"}}, optional
        The precision paths allowed: ``"auto"`` refines in mixed precision the modes whose
        error estimate requires it; ``"f64"`` never does (fastest); ``"double-double"`` computes
        everything in double-double (slowest, for checks).
    n_terms, n_theta : int, optional
        Fix the number of orders or of quadrature nodes instead of converging them.
    derivatives : bool or sequence of str, optional
        Also compute derivatives with respect to these of :data:`PARAMETERS` (``True`` for all
        four); read them with the result's ``d`` method.
    allow_unconverged : bool, optional
        Return the last step of a convergence that reached its cap, with its error estimate,
        instead of raising :class:`~scatrs.errors.NotConvergedError`.
    allow_unvalidated : bool, optional
        Compute outside the validated domain instead of raising
        :class:`~scatrs.errors.OutsideValidatedDomainError`.

    Raises
    ------
    InvalidInputError
        For non-positive or non-finite ``x`` or ``aspect_ratio``, non-finite ``m``,
        ``Re m <= 0`` or ``Im m < 0``, ``rtol`` outside (0, 1){extra_raise}. The message names
        the first failing element (in C order).
    OutsideValidatedDomainError
        For valid inputs outside the validated domain, unless ``allow_unvalidated``.
    NotConvergedError
        If a T-matrix reaches the cap of its orders or nodes, unless ``allow_unconverged``.
    ValueError
        For an unknown parameter name or precision, or shapes that do not broadcast.

    Notes
    -----
    The cost per spheroid grows like the fourth power of ``|m| x_max``, with ``x_max`` the size
    parameter of the longest semi-axis, and steeply for extreme aspect ratios, where the
    refinement engages; the call releases the GIL but cannot be interrupted.
"""

efficiencies.__doc__ = """Orientation-averaged efficiencies of spheroids.

    Returns
    -------
    SpheroidEfficiencies
        Shape ``S``.
""" + _SHARED_DOC.format(extra="", extra_raise="")

greek.__doc__ = """Greek expansion coefficients of the orientation-averaged phase matrix of
    spheroids.

    Returns
    -------
    SpheroidGreek
        Shape ``S + (L + 1,)``, with ``L`` = ``order``, by default the largest ``2 n_terms``.
""" + _SHARED_DOC.format(
    extra="order : int, optional\n        The highest order L.\n"
    "    normalized : bool, optional\n        Normalize so that ``alpha1[..., 0]`` is 1"
    " (the default); otherwise times ``q_sca``.\n    ",
    extra_raise=", ``order`` above ``MAX_GREEK_ORDER``",
)

phase_matrix.__doc__ = """The six-element orientation-averaged phase matrix of spheroids.

    Returns
    -------
    SpheroidPhaseMatrix
        Shape ``S + mu.shape``.
""" + _SHARED_DOC.format(
    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n"
    "    normalized : bool, optional\n        Normalize so that F11 averages to 1 (the"
    " default); otherwise times ``q_sca``.\n    ",
    extra_raise=", ``mu`` outside [-1, 1]",
)


@dataclass(frozen=True, eq=False)
class TMatrixBlock:
    """The T-matrix of one azimuthal mode ``m >= 0`` for orders ``n, k = max(m, 1), ..., N``.

    Attributes
    ----------
    orders : numpy.ndarray
        int64, the orders of the rows and columns.
    t11, t12, t21, t22 : numpy.ndarray
        complex128, square: T11 (M from M), T12 (M from N), T21 (N from M), T22 (N from N).
        Mode ``-m`` has ``t11`` and ``t22`` equal, ``t12`` and ``t21`` negated.
    """

    orders: NDArray[np.int64]
    t11: NDArray[np.complex128]
    t12: NDArray[np.complex128]
    t21: NDArray[np.complex128]
    t22: NDArray[np.complex128]


class TMatrix:
    """The converged T-matrix of one spheroid, from :func:`compute`.

    Attributes
    ----------
    n_terms, n_theta : int
        The number of orders and of quadrature nodes.
    error_estimate : float
        The estimated relative error of the outputs.
    converged : bool
        False only for a result returned at a cap with ``allow_unconverged``.
    precision_paths : tuple of str
        How each mode ``m = 0, ..., N`` was solved: ``"f64"``, ``"refined"`` or
        ``"double-double"``.
    params : tuple of str
        The parameters of the derivatives, if computed.
    """

    __slots__ = ("_obj", "params")

    def __init__(self, obj: _scatrs.TMatrixObject, params: tuple[str, ...]) -> None:
        self._obj = obj
        self.params = params

    @property
    def n_terms(self) -> int:
        return int(self._obj.n_terms)

    @property
    def n_theta(self) -> int:
        return int(self._obj.n_theta)

    @property
    def error_estimate(self) -> float:
        return float(self._obj.error_estimate)

    @property
    def converged(self) -> bool:
        return bool(self._obj.converged)

    @property
    def precision_paths(self) -> tuple[str, ...]:
        return tuple(self._obj.precision_paths)

    def block(self, m: SupportsIndex) -> TMatrixBlock:
        """The blocks of mode ``m`` (0 to ``n_terms``).

        Raises
        ------
        IndexError
            For ``m`` outside 0 to ``n_terms``.
        """
        mm = _args.integer(m, "m")
        b = None if mm < 0 else self._obj.block(mm)
        if b is None:
            raise IndexError(f"mode {mm} is outside 0..={self.n_terms}")
        orders, (t11, t12, t21, t22) = b
        return TMatrixBlock(*(_frozen(v) for v in (orders, t11, t12, t21, t22)))

    def efficiencies(self) -> SpheroidEfficiencies:
        """The orientation-averaged efficiencies (scalars), with their derivatives if
        computed."""
        q, d = self._obj.efficiencies()
        jac = None if d is None else _frozen(d)
        info = {
            "n_terms": np.int64(self.n_terms),
            "n_theta": np.int64(self.n_theta),
            "error": np.float64(self.error_estimate),
        }
        return SpheroidEfficiencies(*q, self.params, jac, **info)

    def greek(
        self, order: SupportsIndex | None = None, *, normalized: bool = True
    ) -> SpheroidGreek:
        """The Greek coefficients up to ``order`` (default ``2 n_terms``)."""
        g, d = self._obj.greek(_optional_size(order, "order"), _args.flag(normalized, "normalized"))
        return SpheroidGreek(
            *_frozen(g),
            np.int64(self.n_terms),
            self.params,
            None if d is None else _frozen(d),
        )

    def phase_matrix(self, mu: ArrayLike, *, normalized: bool = True) -> SpheroidPhaseMatrix:
        """The six-element phase matrix at the cosines ``mu``."""
        mus = _args.real_array(mu, "mu")
        f, d = self._obj.phase_matrix(mus.reshape(-1), _args.flag(normalized, "normalized"))
        f = _frozen(f).reshape((6, *mus.shape))
        d = None if d is None else _frozen(d).reshape((4, 6, *mus.shape))
        return SpheroidPhaseMatrix(*f, params=self.params, _derivatives=d)

    def random_orientation(
        self,
        mu: ArrayLike | None = None,
        order: SupportsIndex | None = None,
        *,
        normalized: bool = True,
    ) -> tuple[SpheroidEfficiencies, SpheroidGreek, SpheroidPhaseMatrix | None]:
        """The orientation-averaged outputs: efficiencies, Greek coefficients up to ``order``
        and, with cosines ``mu``, the phase matrix."""
        return (
            self.efficiencies(),
            self.greek(order, normalized=normalized),
            None if mu is None else self.phase_matrix(mu, normalized=normalized),
        )

    def __repr__(self) -> str:
        return (
            f"TMatrix(n_terms={self.n_terms}, n_theta={self.n_theta}, "
            f"error_estimate={self.error_estimate:.1e})"
        )


def compute(
    x: float,
    m: complex,
    aspect_ratio: float,
    *,
    rtol: float = 1e-8,
    precision: Precision = "auto",
    n_terms: SupportsIndex | None = None,
    n_theta: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unconverged: bool = False,
    allow_unvalidated: bool = False,
) -> TMatrix:
    params = _parameters(derivatives)
    mc = complex(m)
    obj = _scatrs.TMatrixObject(
        float(x),
        mc.real,
        mc.imag,
        float(aspect_ratio),
        float(rtol),
        _precision(precision),
        _optional_size(n_terms, "n_terms"),
        _optional_size(n_theta, "n_theta"),
        _args.flag(allow_unconverged, "allow_unconverged"),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
        bool(params),
    )
    return TMatrix(obj, params)


compute.__doc__ = """The converged T-matrix of one spheroid, with its convergence information and
    orientation-averaged outputs.

    Returns
    -------
    TMatrix
""" + _SHARED_DOC.format(extra="", extra_raise="").replace(
    "x : array_like\n        Equal-volume size parameters",
    "x : float\n        The equal-volume size parameter",
)
