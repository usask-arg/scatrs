"""Scattering by layered spheres: a core and concentric shells (vectorized wrappers over the
Rust core).

``x`` and ``m`` have a trailing axis of layers, core first: ``x[..., l]`` is the size parameter
2 pi r_l / wavelength of the outer surface of layer ``l`` (nondecreasing along the axis, the
last the particle's) and ``m[..., l]`` the refractive index of layer ``l`` relative to the
medium. Their leading axes broadcast against each other to a shape ``S``, and every result has
that shape, extended as in :mod:`scatrs.mie`. A single layer is a sphere: its results are
:mod:`scatrs.mie`'s, bitwise. Conventions are those of :mod:`scatrs.mie`; the outer size
parameter normalizes the efficiencies and the phase matrix. See :doc:`/theory/layered` for the
method and :doc:`/validation/accuracy` for the validated domain.

``derivatives`` is ``True`` (every layer's ``"layer{l}.x"``, ``"layer{l}.m_real"`` and
``"layer{l}.m_imag"``) or a sequence of those names; the result's ``d`` method returns the
derivative with respect to one of them, or ``"layer{l}.m"`` (the complex derivative) for the
complex outputs. The derivatives are validated up to an outer size parameter of 100.

The results are those of :mod:`scatrs.mie` (:class:`~scatrs.mie.MieEfficiencies`, …), with
``d`` accepting the layers' parameter names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs
from scatrs.mie import (
    MieAmplitudes,
    MieCoefficients,
    MieEfficiencies,
    MieGreek,
    MiePhaseMatrix,
    _frozen,
    _n_max,
)
from scatrs.mie import default_n_max as _sphere_default_n_max

__all__ = [
    "MAX_LAYERS",
    "LayeredAmplitudes",
    "LayeredCoefficients",
    "LayeredEfficiencies",
    "LayeredGreek",
    "LayeredPhaseMatrix",
    "amplitudes",
    "coefficients",
    "default_n_max",
    "efficiencies",
    "greek",
    "parameters",
    "phase_matrix",
]

MAX_LAYERS: Final[int] = _scatrs.MAX_LAYERS
"""The largest number of layers accepted: ``2**10``."""

_PARTS: Final = ("x", "m_real", "m_imag")
_NAME: Final = re.compile(r"layer(\d+)\.(x|m_real|m_imag|m)")


def parameters(layers: int) -> tuple[str, ...]:
    """The parameter names of a particle of ``layers`` layers, in the order of the results."""
    return tuple(f"layer{j}.{p}" for j in range(layers) for p in _PARTS)


def default_n_max(x: ArrayLike) -> NDArray[np.int64]:
    """The default number of terms: the sphere's (:func:`scatrs.mie.default_n_max`) at the outer
    size parameter ``x[..., -1]``."""
    xs = _args.real_array(x, "x")
    if xs.ndim == 0 or xs.shape[-1] == 0:
        raise ValueError("x needs a trailing axis of at least one layer")
    return _sphere_default_n_max(xs[..., -1])


def _inputs(
    x: ArrayLike, m: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.complex128], tuple[int, ...]]:
    """``x`` and ``m`` broadcast to one shape ``S + (L,)``, as C-contiguous (P, L) arrays."""
    xs = _args.real_array(x, "x")
    ms = _args.complex_array(m, "m")
    if xs.ndim == 0 or ms.ndim == 0:
        raise ValueError("x and m need a trailing axis of layers (one value per layer)")
    if xs.shape[-1] != ms.shape[-1]:
        raise ValueError(
            f"x and m must have one value per layer: their last axes have {xs.shape[-1]} and "
            f"{ms.shape[-1]}"
        )
    if xs.shape[-1] == 0:
        raise ValueError("a layered sphere needs at least one layer")
    xb, mb = np.broadcast_arrays(xs, ms)
    layers = xb.shape[-1]
    shape = xb.shape[:-1]
    return (
        np.ascontiguousarray(xb).reshape(-1, layers),
        np.ascontiguousarray(mb).reshape(-1, layers),
        shape,
    )


def _parameters(derivatives: bool | Sequence[str], layers: int) -> tuple[str, ...]:
    names = parameters(layers)
    if isinstance(derivatives, (bool, np.bool_)):
        return names if derivatives else ()
    if isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
        raise TypeError(
            "derivatives must be a bool or a sequence of parameter names, "
            f"not {type(derivatives).__name__}"
        )
    out = tuple(derivatives)
    for name in out:
        if name not in names:
            raise ValueError(f"unknown parameter {name!r}; expected one of {names}")
    return out


def _index(params: tuple[str, ...], param: str, real: str | None = None) -> tuple[int, int]:
    """(layer, part) of ``param`` (part 0, 1, 2 for x, Re m, Im m; 3 for the complex d/dm),
    if it was requested; ``real`` names real outputs, which have no complex derivative."""
    match = _NAME.fullmatch(param)
    if match is not None:
        layer, part = int(match[1]), match[2]
        requested = param in params or (
            part == "m" and {f"layer{layer}.m_real", f"layer{layer}.m_imag"} & set(params)
        )
        if requested:
            if part == "m" and real is not None:
                raise ValueError(f"{real} are real: use 'layer{layer}.m_real' or '.m_imag'")
            return layer, ("x", "m_real", "m_imag", "m").index(part)
    raise ValueError(f"no derivative with respect to {param!r}; computed: {params}")


@dataclass(frozen=True, eq=False)
class LayeredEfficiencies(MieEfficiencies):
    """:class:`~scatrs.mie.MieEfficiencies` of layered spheres, per unit geometric cross section
    of the outer surface; ``d`` takes the layers' parameter names."""

    def d(self, param: str) -> MieEfficiencies:
        layer, p = _index(self.params, param, "efficiencies")
        assert self._jacobian is not None
        return MieEfficiencies(*self._jacobian[layer, p])


@dataclass(frozen=True, eq=False)
class LayeredCoefficients(MieCoefficients):
    """:class:`~scatrs.mie.MieCoefficients` of layered spheres; ``d`` takes the layers'
    parameter names (``"layer{l}.m"``: the complex derivative d/dm_l)."""

    def d(self, param: str) -> MieCoefficients:
        layer, p = _index(self.params, param)
        assert self._derivatives is not None
        da_dx, da_dm, db_dx, db_dm = self._derivatives[layer]
        if p == 0:
            return MieCoefficients(da_dx, db_dx, self.n_terms)
        if p == 2:
            return MieCoefficients(_frozen(1j * da_dm), _frozen(1j * db_dm), self.n_terms)
        return MieCoefficients(da_dm, db_dm, self.n_terms)


@dataclass(frozen=True, eq=False)
class LayeredAmplitudes(MieAmplitudes):
    """:class:`~scatrs.mie.MieAmplitudes` of layered spheres; ``d`` takes the layers'
    parameter names (``"layer{l}.m"``: the complex derivative d/dm_l)."""

    def d(self, param: str) -> MieAmplitudes:
        layer, p = _index(self.params, param)
        assert self._derivatives is not None
        ds1_dx, ds2_dx, ds1_dm, ds2_dm = self._derivatives[layer]
        if p == 0:
            return MieAmplitudes(ds1_dx, ds2_dx)
        if p == 2:
            return MieAmplitudes(_frozen(1j * ds1_dm), _frozen(1j * ds2_dm))
        return MieAmplitudes(ds1_dm, ds2_dm)


@dataclass(frozen=True, eq=False)
class LayeredPhaseMatrix(MiePhaseMatrix):
    """:class:`~scatrs.mie.MiePhaseMatrix` of layered spheres; ``d`` takes the layers'
    parameter names."""

    def d(self, param: str) -> MiePhaseMatrix:
        layer, p = _index(self.params, param, "phase-matrix elements")
        assert self._derivatives is not None
        return MiePhaseMatrix(*self._derivatives[layer, p])


@dataclass(frozen=True, eq=False)
class LayeredGreek(MieGreek):
    """:class:`~scatrs.mie.MieGreek` of layered spheres; ``d`` takes the layers' parameter
    names."""

    def d(self, param: str) -> MieGreek:
        layer, p = _index(self.params, param, "Greek coefficients")
        assert self._derivatives is not None
        return MieGreek(*self._derivatives[layer, p], self.n_terms)


_SHARED_DOC = """
    Parameters
    ----------
    x : array_like
        Size parameters of the interfaces, 2 pi r_l / wavelength, along a trailing axis of
        layers (core first, nondecreasing).
    m : array_like
        Refractive indices of the layers relative to the medium (real or complex), along the
        same trailing axis; the leading axes broadcast against those of ``x``.
    {extra}n_max : int, optional
        The number of terms; by default the sphere's at the outer size parameter. Fewer terms
        than the default are outside the validated domain.
    derivatives : bool or sequence of str, optional
        Also compute derivatives with respect to these of ``"layer{{l}}.x"``,
        ``"layer{{l}}.m_real"`` and ``"layer{{l}}.m_imag"`` (``True`` for all); read them with the
        result's ``d`` method. Validated up to an outer size parameter of 100.
    allow_unvalidated : bool, optional
        Compute outside the validated domain instead of raising
        :class:`~scatrs.errors.OutsideValidatedDomainError` (see
        :doc:`/validation/accuracy`).

    Raises
    ------
    InvalidInputError
        For invalid inputs (as :mod:`scatrs.mie`'s, for every layer), size parameters that
        decrease from the core out, or more than ``MAX_LAYERS`` layers{extra_raise}. The
        message names the first failing particle (in C order).
    OutsideValidatedDomainError
        For valid inputs outside the validated domain, unless ``allow_unvalidated``.
    TypeError
        For non-numeric ``x`` or ``m``, or arguments of the wrong type{extra_type}.
    ValueError
        For an unknown parameter name, no layer axis, last axes of different lengths, or
        (from numpy) leading shapes that do not broadcast.
"""


def efficiencies(
    x: ArrayLike,
    m: ArrayLike,
    *,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> LayeredEfficiencies:
    xb, mb, shape = _inputs(x, m)
    params = _parameters(derivatives, xb.shape[1])
    values, jacobian = _scatrs.layered_efficiencies(
        xb, mb, _n_max(n_max), bool(params), _args.flag(allow_unvalidated, "allow_unvalidated")
    )
    values, jacobian = _frozen(values), _frozen(jacobian)
    fields = [v.reshape(shape) for v in values]
    jac = None if jacobian is None else jacobian.reshape((*jacobian.shape[:3], *shape))
    return LayeredEfficiencies(*fields, params=params, _jacobian=jac)


efficiencies.__doc__ = """Efficiencies and asymmetry parameter of layered spheres.

    Returns
    -------
    LayeredEfficiencies
        Each field has the broadcast leading shape ``S``.
""" + _SHARED_DOC.format(extra="", extra_raise="", extra_type="")


def coefficients(
    x: ArrayLike,
    m: ArrayLike,
    *,
    n_max: SupportsIndex | None = None,
    derivatives: bool | Sequence[str] = False,
    allow_unvalidated: bool = False,
) -> LayeredCoefficients:
    xb, mb, shape = _inputs(x, m)
    params = _parameters(derivatives, xb.shape[1])
    terms, a, b, d = _scatrs.layered_coefficients(
        xb, mb, _n_max(n_max), bool(params), _args.flag(allow_unvalidated, "allow_unvalidated")
    )
    a, b, d = _frozen(a), _frozen(b), _frozen(d)
    width = a.shape[1]
    n_terms = _frozen(terms.astype(np.int64)).reshape(shape)
    dd = None if d is None else d.reshape((*d.shape[:2], *shape, width))
    return LayeredCoefficients(
        a.reshape((*shape, width)), b.reshape((*shape, width)), n_terms, params, dd
    )


coefficients.__doc__ = """Mie coefficients aₙ and bₙ of layered spheres.

    Returns
    -------
    LayeredCoefficients
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
) -> LayeredAmplitudes:
    xb, mb, shape = _inputs(x, m)
    params = _parameters(derivatives, xb.shape[1])
    mus = _args.real_array(mu, "mu")
    s, d = _scatrs.layered_amplitudes(
        xb,
        mb,
        mus.reshape(mus.shape),
        _n_max(n_max),
        bool(params),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
    )
    s, d = _frozen(s), _frozen(d)
    out = (*shape, *mus.shape)
    dd = None if d is None else d.reshape((*d.shape[:2], *out))
    return LayeredAmplitudes(s[0].reshape(out), s[1].reshape(out), params, dd)


amplitudes.__doc__ = """Amplitude scattering functions S₁ and S₂ of layered spheres.

    Returns
    -------
    LayeredAmplitudes
        Shape ``S + mu.shape``.
""" + _SHARED_DOC.format(
    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n    ",
    extra_raise=", ``mu`` outside [-1, 1] or NaN",
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
) -> LayeredPhaseMatrix:
    xb, mb, shape = _inputs(x, m)
    params = _parameters(derivatives, xb.shape[1])
    mus = _args.real_array(mu, "mu")
    f, d = _scatrs.layered_phase_matrix(
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
    dd = None if d is None else d.reshape((*d.shape[:3], *out))
    return LayeredPhaseMatrix(*(v.reshape(out) for v in f), params=params, _derivatives=dd)


phase_matrix.__doc__ = """The phase matrix F11, F12, F33, F34 of layered spheres.

    Returns
    -------
    LayeredPhaseMatrix
        Shape ``S + mu.shape``.
""" + _SHARED_DOC.format(
    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n"
    "    normalized : bool, optional\n        Divide by ``q_sca`` (the default), so that F11"
    " averages to 1.\n    ",
    extra_raise=", ``mu`` outside [-1, 1] or NaN",
    extra_type="",
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
) -> LayeredGreek:
    xb, mb, shape = _inputs(x, m)
    params = _parameters(derivatives, xb.shape[1])
    terms, g, d = _scatrs.layered_greek(
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
    dd = None if d is None else d.reshape((*d.shape[:3], *out))
    return LayeredGreek(*(v.reshape(out) for v in g), n_terms, params, dd)


greek.__doc__ = """Greek expansion coefficients of the phase matrix of layered spheres, exact up to
    rounding, as :func:`scatrs.mie.greek`.

    Returns
    -------
    LayeredGreek
        Shape ``S + (L + 1,)``.
""" + _SHARED_DOC.format(
    extra="order : int, optional\n        The highest order L; by default ``2 n_terms`` of the"
    " largest particle.\n    normalized : bool, optional\n        Normalize ``alpha1[..., 0]`` to"
    " 1 (the default), or to ``q_sca``.\n    ",
    extra_raise=", a negative ``order`` or one above ``MAX_GREEK_ORDER``",
    extra_type="",
)
