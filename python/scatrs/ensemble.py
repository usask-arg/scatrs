"""Ensembles of particles: size distributions, adaptive size integration and mixtures.

An :class:`Ensemble` is a particle (:class:`~scatrs.Sphere`) and a size distribution
(:mod:`scatrs.size`); a :class:`Mixture` is several ensembles with relative number
concentrations. Their ``compute`` methods average the optical properties over the sizes at
each wavelength. Radii and wavelengths are in the same (any) unit; cross sections are per
particle, averaged over the number distribution, in that unit squared. See
:doc:`/theory/ensembles` for the method and the error estimates.

The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the cross
sections, g and the distribution moments, and optionally the backscattering, meet ``rtol``).
Every output comes with an error estimate (:attr:`~scatrs.EnsembleResult.error`) and, with
``derivatives``, its derivatives with respect to the distribution parameters, ``"m_real"``,
``"m_imag"`` and the mixture weights (:meth:`~scatrs.EnsembleResult.d`).

A distribution whose parameters are arrays is a grid of distributions, computed at once with
shared size nodes (as an optics table needs): the results have the grid's axes first.

Each call runs in Rust with the GIL released, in parallel over wavelengths and size panels;
results are bitwise the same for any number of threads. Results are frozen dataclasses of
read-only arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs
from scatrs.particles import Sphere
from scatrs.size import Distribution, _check, _grid, _scalar_spec, _specs

__all__ = [
    "SCALARS",
    "Ensemble",
    "EnsembleGreek",
    "EnsembleOutputs",
    "EnsemblePhase",
    "EnsembleResult",
    "Mixture",
]

SCALARS = (
    "c_ext",
    "c_sca",
    "c_abs",
    "c_back",
    "ssa",
    "g",
    "lidar_ratio",
    "depolarization",
    "effective_radius",
    "mean_area",
    "mean_volume",
)
"""The scalar outputs of :class:`EnsembleOutputs`, in order."""


def _frozen(a: np.ndarray) -> np.ndarray:
    """A read-only view of ``a``: an array that owns its data can be made writeable again, a
    view of a read-only array cannot."""
    a.flags.writeable = False
    return a.view()


class _ReadOnly:
    """Unpickles and deep-copies a result read-only: numpy restores arrays writeable."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            frozen = _frozen(value) if isinstance(value, np.ndarray) else value
            object.__setattr__(self, name, frozen)


class _Grid(Mapping[str, NDArray[np.float64]]):
    """A read-only mapping of names to read-only copies of arrays (which, unlike
    :class:`types.MappingProxyType`, pickles)."""

    def __init__(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        self._arrays = {n: _frozen(np.array(a)) for n, a in arrays.items()}

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        return self._arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __reduce__(self) -> tuple:
        return (_Grid, (self._arrays,))

    def __repr__(self) -> str:
        return f"Grid({', '.join(f'{n}={a.shape}' for n, a in self._arrays.items())})"


@dataclass(frozen=True, eq=False)
class EnsembleGreek(_ReadOnly):
    """Greek coefficients per wavelength, arrays of shape ``wavelength.shape + (L + 1,)``
    (after the grid's axes), zero-padded to the largest order (see
    :class:`scatrs.mie.MieGreek` for the expansion)."""

    alpha1: NDArray[np.float64]
    alpha2: NDArray[np.float64]
    alpha3: NDArray[np.float64]
    alpha4: NDArray[np.float64]
    beta1: NDArray[np.float64]
    beta2: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class EnsemblePhase(_ReadOnly):
    """The phase matrix per wavelength at the cosines ``mu``, arrays of shape
    ``wavelength.shape + mu.shape`` (after the grid's axes)."""

    f11: NDArray[np.float64]
    f12: NDArray[np.float64]
    f33: NDArray[np.float64]
    f34: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class EnsembleOutputs(_ReadOnly):
    """The outputs of an ensemble (or their error estimates, or their derivatives with
    respect to one parameter), each an array of shape ``grid + wavelength.shape``.

    Attributes
    ----------
    c_ext, c_sca, c_abs, c_back : numpy.ndarray
        Number-averaged cross sections per particle; ``c_back`` is Bohren & Huffman's
        backscattering cross section (4π times dσ/dΩ at 180°).
    ssa, g : numpy.ndarray
        Single-scattering albedo and asymmetry parameter.
    lidar_ratio, depolarization : numpy.ndarray
        4π ``c_ext``/``c_back``, and the linear depolarization ratio (exactly 0 for spheres).
    effective_radius, mean_area, mean_volume : numpy.ndarray
        ⟨r³⟩/⟨r²⟩, ⟨πr²⟩ and ⟨4πr³/3⟩.
    greek : EnsembleGreek or None
        The Greek coefficients, if requested.
    phase : EnsemblePhase or None
        The phase matrix, if requested.
    """

    c_ext: NDArray[np.float64]
    c_sca: NDArray[np.float64]
    c_abs: NDArray[np.float64]
    c_back: NDArray[np.float64]
    ssa: NDArray[np.float64]
    g: NDArray[np.float64]
    lidar_ratio: NDArray[np.float64]
    depolarization: NDArray[np.float64]
    effective_radius: NDArray[np.float64]
    mean_area: NDArray[np.float64]
    mean_volume: NDArray[np.float64]
    greek: EnsembleGreek | None = None
    phase: EnsemblePhase | None = None


@dataclass(frozen=True, eq=False, repr=False)
class EnsembleResult(EnsembleOutputs):
    """:class:`EnsembleOutputs` with their error estimates and derivatives.

    Attributes
    ----------
    error : EnsembleOutputs
        Non-negative error estimates of every output (see :doc:`/validation/accuracy`).
    converged : numpy.ndarray
        bool per point: whether every refined quantity met ``rtol`` within ``max_panels``
        panels.
    panels : numpy.ndarray
        int64 per point: the number of size panels (shared by the points of a grid).
    params : tuple of str
        The parameters :meth:`d` accepts.
    wavelength : numpy.ndarray
        The wavelengths, as given.
    refractive_index : numpy.ndarray
        m at the wavelengths: of the wavelengths' shape, or ``(components,) +
        wavelength.shape`` for a :class:`Mixture`.
    grid : Mapping[str, numpy.ndarray]
        Each array-valued distribution parameter, by name (as in :attr:`params`), broadcast
        to the grid's shape; empty if there is no grid.
    """

    error: EnsembleOutputs = field(kw_only=True)
    converged: NDArray[np.bool_] = field(kw_only=True)
    panels: NDArray[np.int64] = field(kw_only=True)
    wavelength: NDArray[np.float64] = field(kw_only=True)
    refractive_index: NDArray[np.complex128] = field(kw_only=True)
    grid: Mapping[str, NDArray[np.float64]] = field(kw_only=True)
    params: tuple[str, ...] = field(default=(), kw_only=True)
    _d: tuple[EnsembleOutputs, ...] = field(default=(), kw_only=True)
    _d_errors: tuple[EnsembleOutputs, ...] = field(default=(), kw_only=True)

    def _index(self, param: str) -> int:
        if param not in self.params:
            raise ValueError(f"no derivative with respect to {param!r}; computed: {self.params}")
        return self.params.index(param)

    def d(self, param: str) -> EnsembleOutputs:
        """The derivatives with respect to ``param``.

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        return self._d[self._index(param)]

    def d_error(self, param: str) -> EnsembleOutputs:
        """The error estimates of the derivatives with respect to ``param``.

        Raises
        ------
        ValueError
            If the derivative was not requested.
        """
        return self._d_errors[self._index(param)]

    def __repr__(self) -> str:
        parts = [f"shape={self.c_ext.shape}"]
        if self.greek is not None:
            parts.append(f"greek={self.greek.alpha1.shape}")
        if self.phase is not None:
            parts.append(f"phase={self.phase.f11.shape}")
        parts.append(f"params={self.params}")
        if self.grid:
            parts.append(f"grid={tuple(self.grid)}")
        parts.append(f"converged={int(self.converged.sum())}/{self.converged.size}")
        return f"EnsembleResult({', '.join(parts)})"


def _outputs(
    raw: dict, which: str, k: int | None, shape: tuple[int, ...], mu_shape: tuple[int, ...]
) -> EnsembleOutputs:
    """The outputs of kind ``which`` ("" for values, "_errors", "_d" with derivative ``k``,
    "_d_errors")."""
    key = {"": "values", "_errors": "errors", "_d": "d", "_d_errors": "d_errors"}[which]
    rows = raw[key] if k is None else raw[key][k]
    # The arrays of `raw` are read-only already, so these views cannot be made writeable.
    scalars = [r.reshape(shape) for r in rows]
    greek = phase = None
    if "greek" in raw:
        g = raw["greek" + which]
        g = g if k is None else g[k]
        greek = EnsembleGreek(*(a.reshape((*shape, a.shape[-1])) for a in g))
    if "phase" in raw:
        p = raw["phase" + which]
        p = p if k is None else p[k]
        phase = EnsemblePhase(*(a.reshape((*shape, *mu_shape)) for a in p))
    return EnsembleOutputs(*scalars, greek=greek, phase=phase)


def _greek_order(greek: SupportsIndex | Literal["all"] | None) -> tuple[int | None, bool]:
    """The order and whether every order is wanted."""
    if greek is None:
        return None, False
    if isinstance(greek, str):
        if greek != "all":
            raise ValueError(f"greek must be an order, 'all' or None, not {greek!r}")
        return None, True
    return _args.size(greek, "greek"), False


def _parameters(names: tuple[str, ...], derivatives: bool | Sequence[str]) -> tuple[str, ...]:
    """The requested parameters, checked against ``names``."""
    if isinstance(derivatives, (bool, np.bool_)):
        return names if derivatives else ()
    if isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
        raise TypeError(
            "derivatives must be a bool or a sequence of parameter names, "
            f"not {type(derivatives).__name__}"
        )
    params = tuple(derivatives)
    for p in params:
        if p not in names:
            raise ValueError(f"unknown parameter {p!r}; expected one of {names}")
    if len(set(params)) != len(params):
        raise ValueError(f"a parameter is requested twice: {params}")
    return params


@dataclass(frozen=True, eq=False)
class _Options:
    """The options shared by every ``compute``, converted for the engine."""

    order: int | None
    all_orders: bool
    mu: NDArray[np.float64] | None
    normalized: bool
    rtol: float
    max_panels: int
    allow_unvalidated: bool
    refine_backscatter: bool

    @classmethod
    def of(
        cls,
        greek: SupportsIndex | Literal["all"] | None,
        mu: ArrayLike | None,
        normalized: bool,
        rtol: float,
        max_panels: SupportsIndex,
        allow_unvalidated: bool,
        refine_backscatter: bool,
    ) -> _Options:
        order, all_orders = _greek_order(greek)
        return cls(
            order,
            all_orders,
            None if mu is None else _args.real_array(mu, "mu"),
            _args.flag(normalized, "normalized"),
            float(rtol),
            _args.size(max_panels, "max_panels"),
            _args.flag(allow_unvalidated, "allow_unvalidated"),
            _args.flag(refine_backscatter, "refine_backscatter"),
        )

    def flat_mu(self) -> NDArray[np.float64]:
        return np.ascontiguousarray(np.zeros(0) if self.mu is None else self.mu.reshape(-1))

    def mu_shape(self) -> tuple[int, ...]:
        return () if self.mu is None else self.mu.shape


def _result(
    raw: dict,
    shape: tuple[int, ...],
    o: _Options,
    wavelength: NDArray[np.float64],
    m: NDArray[np.complex128],
    grid: dict[str, NDArray[np.float64]],
    params: tuple[str, ...] = (),
) -> EnsembleResult:
    """The result from the engine's dict ``raw``, every array of shape ``shape`` (+ trailing)."""
    raw["panels"] = raw["panels"].astype(np.int64)
    for a in raw.values():
        a.flags.writeable = False
    mu_shape = o.mu_shape()
    values = _outputs(raw, "", None, shape, mu_shape)
    return EnsembleResult(
        **{f: getattr(values, f) for f in (*SCALARS, "greek", "phase")},
        error=_outputs(raw, "_errors", None, shape, mu_shape),
        converged=raw["converged"].reshape(shape),
        panels=raw["panels"].reshape(shape),
        wavelength=_frozen(wavelength.copy()),
        refractive_index=_frozen(m),
        grid=_Grid(grid),
        params=params,
        _d=tuple(_outputs(raw, "_d", k, shape, mu_shape) for k in range(len(params))),
        _d_errors=tuple(_outputs(raw, "_d_errors", k, shape, mu_shape) for k in range(len(params))),
    )


def _single(
    items: list,
    mixture: bool,
    wl: NDArray[np.float64],
    derivatives: bool | Sequence[str],
    o: _Options,
) -> tuple[dict, tuple[str, ...]]:
    """The engine's single-distribution or mixture path."""
    names = tuple(_scatrs.ensemble_parameters(items, mixture))
    params = _parameters(names, derivatives)
    raw = _scatrs.ensemble_compute(
        items,
        np.ascontiguousarray(wl.reshape(-1)),
        mixture,
        o.order,
        o.all_orders,
        o.flat_mu(),
        o.normalized,
        [names.index(p) for p in params],
        o.rtol,
        o.max_panels,
        o.allow_unvalidated,
        o.refine_backscatter,
    )
    return raw, params


def _weight(w: ArrayLike, what: str) -> float:
    """``w`` as a float.

    Raises
    ------
    TypeError
        If ``w`` is not a real number: a bool, a string or an array, say.
    """
    a = np.asarray(w)
    if a.ndim != 0:
        raise TypeError(f"{what} must be a real number, not an array of shape {a.shape}")
    if a.dtype.kind not in "iuf":
        given = a.dtype if isinstance(w, np.ndarray) else type(w).__name__
        raise TypeError(f"{what} must be a real number, not {given}")
    return float(a)


def _sphere(particle: object, what: str) -> Sphere:
    if not isinstance(particle, Sphere):
        raise TypeError(f"{what} must be a Sphere, not {type(particle).__name__}")
    return particle


_COMPUTE_DOC = """{summary}

    Parameters
    ----------
    wavelength : array_like
        Wavelengths, in the unit of the radii (real, scalar or any shape).
    greek : int, "all" or None, optional
        The Greek coefficients to compute: up to this order, every order, or none (the
        default).
    mu : array_like, optional
        Cosines of the scattering angle at which to compute the phase matrix.
    normalized : bool, optional
        Normalize the phase matrix and the Greek coefficients (``alpha1[..., 0]`` = 1, the
        default) or give them per particle (``alpha1[..., 0]`` = ``c_sca``).
    derivatives : bool or sequence of str, optional
        The parameters to differentiate with respect to (``True`` for all): {derivatives}
    rtol : float, optional
        Relative tolerance of the size integration, in [1e-13, 0.1].
    max_panels : int, optional
        The most size panels per {per}; reaching it sets ``converged`` to False.
    allow_unvalidated : bool, optional
        Compute sizes outside the validated domain of the Mie solution instead of raising.
    refine_backscatter : bool, optional
        Also refine on ``c_back`` (to ``rtol`` of the larger of ``c_back`` and ``c_sca``),
        which makes its error estimate, and the lidar ratio's, a calibrated bound. Off by
        default: the backscattering of weakly absorbing spheres is a sum of narrow glory
        ripples that can take far more size panels than the cross sections.

    Returns
    -------
    EnsembleResult
        {returns}

    Raises
    ------
    InvalidInputError
        For invalid distribution parameters, wavelengths, refractive indices (including a
        model with no valid value at a wavelength) or options, or outputs that would exceed
        ``2 * MAX_MIE_VALUES`` real values (the bound of :mod:`scatrs.mie`; with
        ``greek="all"``, known only after the computation).
    OutsideValidatedDomainError
        If a size node is outside the validated domain, unless ``allow_unvalidated``; the
        message names the wavelength and the radius.
    TypeError, ValueError
        For arguments of the wrong type, or unknown or repeated parameter names.{raises}
    BufferError
        If another rust-numpy extension holds a writeable borrow of an input.
    MemoryError
        If the outputs cannot be allocated.

    Notes
    -----
    The cost grows with the size parameters and with the order of the Greek coefficients
    (see :doc:`/limitations`). The call releases the GIL, so other Python threads keep
    running, but it cannot be interrupted: Ctrl+C takes effect only once it returns.
"""


@dataclass(frozen=True)
class Ensemble:
    """Particles of one kind over a size distribution.

    Parameters
    ----------
    particle : Sphere
        The particle, with its refractive-index model.
    size : LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal or list
        The number size distribution. If its parameters are arrays, the ensemble is a grid
        of distributions, one per point of their broadcast shape. A list (or tuple) of
        distributions with scalar parameters, of any kinds, is a batch with one axis, one
        point per distribution; it is stored as a tuple.

    Raises
    ------
    TypeError
        If ``particle`` is not a :class:`~scatrs.Sphere`, or ``size`` or an element of a
        list not a distribution.
    ValueError
        If the distribution's array-valued parameters do not broadcast together, or an
        element of a list is a grid.
    """

    particle: Sphere
    size: Distribution | tuple[Distribution, ...]

    def __post_init__(self) -> None:
        _sphere(self.particle, "particle")
        if isinstance(self.size, (list, tuple)):
            sizes = tuple(_check(d, f"size[{i}]") for i, d in enumerate(self.size))
            for i, d in enumerate(sizes):
                shape = _grid(d)[0]
                if shape != ():
                    raise ValueError(
                        f"size[{i}] is a grid of shape {shape}; a list takes distributions "
                        "with scalar parameters"
                    )
            object.__setattr__(self, "size", sizes)
        else:
            _check(self.size, "size")
            _grid(self.size)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """The shape of the grid of distributions: ``()`` if no parameter is an array, and
        ``(len(size),)`` for a list."""
        if isinstance(self.size, tuple):
            return (len(self.size),)
        return _grid(self.size)[0]

    def _layout(self) -> tuple[tuple[int, ...], dict[str, NDArray[np.float64]], list[tuple]]:
        """The grid shape, the array-valued parameters broadcast to it, and the engine spec of
        each point in C order (one for a single distribution)."""
        if isinstance(self.size, tuple):
            return (len(self.size),), {}, [_scalar_spec(d) for d in self.size]
        shape, grid = _grid(self.size)
        if shape == ():
            return shape, grid, [_scalar_spec(self.size)]
        return shape, grid, _specs(self.size, (shape, grid))[1]

    def compute(
        self,
        wavelength: ArrayLike,
        *,
        greek: SupportsIndex | Literal["all"] | None = None,
        mu: ArrayLike | None = None,
        normalized: bool = True,
        derivatives: bool | Sequence[str] = False,
        rtol: float = 1e-6,
        max_panels: SupportsIndex = 4096,
        allow_unvalidated: bool = False,
        refine_backscatter: bool = False,
    ) -> EnsembleResult:
        wl = _args.real_array(wavelength, "wavelength")
        m = self.particle.refractive_index.evaluate(wl)
        o = _Options.of(
            greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
        )
        shape, grid, specs = self._layout()
        flat_m = np.ascontiguousarray(m.reshape(-1))
        if shape == ():
            items = [(specs[0], flat_m, 1.0)]
            raw, params = _single(items, False, wl, derivatives, o)
            return _result(raw, wl.shape, o, wl, m, grid, params)
        if not (isinstance(derivatives, (bool, np.bool_)) and not derivatives):
            raise NotImplementedError(
                "derivatives of a grid or list of distributions are not supported; compute "
                "each point as its own Ensemble"
            )
        raw = _scatrs.ensemble_compute_batch(
            specs,
            flat_m,
            np.ascontiguousarray(wl.reshape(-1)),
            o.order,
            o.all_orders,
            o.flat_mu(),
            o.normalized,
            o.rtol,
            o.max_panels,
            o.allow_unvalidated,
            o.refine_backscatter,
        )
        return _result(raw, (*shape, *wl.shape), o, wl, m, grid)


Ensemble.compute.__doc__ = _COMPUTE_DOC.format(
    summary="Averages over the ensemble at each wavelength.",
    derivatives='the distribution\'s (``"median_radius"``, …; ``"mode{j}.<name>"`` and '
    '``"mode{j}.weight"`` for a multimodal one), ``"m_real"`` and ``"m_imag"`` (Re m and Im m '
    "at each wavelength). Not available for a grid or a list.",
    per="wavelength (shared by the points of a grid or a list)",
    returns="Every array of shape ``grid + wavelength.shape`` (``grid`` is "
    ":attr:`grid_shape`). The points of a grid or a list share their size nodes and each "
    "meets its own ``rtol``: being at least as refined as on its own, a point's values differ "
    "from those of its own ensemble within the error estimates.",
    raises="\n    NotImplementedError\n        For ``derivatives`` of a grid or a list.",
)


@dataclass(frozen=True)
class Mixture:
    """An external mixture: ensembles with relative number concentrations.

    Cross sections and moments are averaged with the number fractions wᵢ/Σw, phase matrices
    with wᵢ times the ``c_sca`` of ensemble i.

    Parameters
    ----------
    components : sequence of (float, Ensemble)
        The weights wᵢ ≥ 0 (finite, not all zero) and the ensembles, each of one
        distribution (not a grid or a list).

    Raises
    ------
    TypeError
        If a component is not a (weight, Ensemble) pair, or a weight not a real number.
    ValueError
        If an ensemble is a grid or a list.
    """

    components: Sequence[tuple[float, Ensemble]]

    def __post_init__(self) -> None:
        out = []
        for i, c in enumerate(self.components):
            try:
                w, e = c
            except (TypeError, ValueError):
                raise TypeError(
                    f"component {i} must be a (weight, Ensemble) pair, not {type(c).__name__}"
                ) from None
            if not isinstance(e, Ensemble):
                raise TypeError(
                    f"component {i} must be a (weight, Ensemble) pair, not "
                    f"({type(w).__name__}, {type(e).__name__})"
                )
            if e.grid_shape != ():
                raise ValueError(
                    f"the size of component {i} is a grid or list of shape {e.grid_shape}; "
                    "a mixture needs one distribution with scalar parameters per component"
                )
            out.append((_weight(w, f"the weight of component {i}"), e))
        object.__setattr__(self, "components", tuple(out))

    def compute(
        self,
        wavelength: ArrayLike,
        *,
        greek: SupportsIndex | Literal["all"] | None = None,
        mu: ArrayLike | None = None,
        normalized: bool = True,
        derivatives: bool | Sequence[str] = False,
        rtol: float = 1e-6,
        max_panels: SupportsIndex = 4096,
        allow_unvalidated: bool = False,
        refine_backscatter: bool = False,
    ) -> EnsembleResult:
        wl = _args.real_array(wavelength, "wavelength")
        ms = [e.particle.refractive_index.evaluate(wl) for _, e in self.components]
        o = _Options.of(
            greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
        )
        items = [
            (_scalar_spec(e.size), np.ascontiguousarray(m.reshape(-1)), w)
            for (w, e), m in zip(self.components, ms, strict=True)
        ]
        raw, params = _single(items, True, wl, derivatives, o)
        return _result(raw, wl.shape, o, wl, np.stack(ms), {}, params)


Mixture.compute.__doc__ = _COMPUTE_DOC.format(
    summary="Averages over the mixture at each wavelength.",
    derivatives='each component\'s distribution parameters, ``"m_real"`` and ``"m_imag"``, '
    'prefixed by ``"c{i}."``, and the weights ``"c{i}.weight"``.',
    per="component and wavelength",
    returns="Every array of the wavelengths' shape.",
    raises="",
)
