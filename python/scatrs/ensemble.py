"""Ensembles of spheres: size distributions, adaptive size integration and mixtures.

:func:`compute` averages the optical properties of spheres over a size distribution at each
wavelength; :func:`compute_mixture` averages over several :class:`Component` s. Radii and
wavelengths are in the same (any) unit; cross sections are per particle, averaged over the
number distribution, in that unit squared. See :doc:`/theory/ensembles` for the method and
the error estimates.

The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the cross
sections, g and the distribution moments, and optionally the backscattering, meet ``rtol``).
Every output comes with an error estimate (:attr:`EnsembleResult.error`) and, with
``derivatives``, its derivatives with respect to the distribution parameters, ``"m_real"``,
``"m_imag"`` and the mixture weights (:meth:`EnsembleResult.d`).

Each call runs in Rust with the GIL released, in parallel over wavelengths and size panels;
results are bitwise the same for any number of threads. Results are frozen dataclasses of
read-only arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, SupportsIndex, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args, _scatrs

__all__ = [
    "SCALARS",
    "Component",
    "EnsembleGreek",
    "EnsembleOutputs",
    "EnsemblePhase",
    "EnsembleResult",
    "Gamma",
    "LogNormal",
    "ModifiedGamma",
    "Monodisperse",
    "MultiModal",
    "PowerLaw",
    "Tabulated",
    "compute",
    "compute_mixture",
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


@dataclass(frozen=True)
class LogNormal:
    """n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: ``median_radius`` r_g > 0, ``width``
    σ_g > 1."""

    median_radius: float
    width: float

    def _spec(self) -> tuple:
        return ("lognormal", float(self.median_radius), float(self.width))


@dataclass(frozen=True)
class Gamma:
    """Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
    ``effective_radius`` a > 0, ``effective_variance`` 0 < b < 1/2."""

    effective_radius: float
    effective_variance: float

    def _spec(self) -> tuple:
        return ("gamma", float(self.effective_radius), float(self.effective_variance))


@dataclass(frozen=True)
class ModifiedGamma:
    """Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): ``alpha`` > −1,
    ``b`` > 0, ``gamma`` > 0."""

    alpha: float
    b: float
    gamma: float

    def _spec(self) -> tuple:
        return ("modified_gamma", float(self.alpha), float(self.b), float(self.gamma))


@dataclass(frozen=True)
class PowerLaw:
    """n(r) ∝ r^(−α) on [``min_radius``, ``max_radius``]: ``exponent`` α (the limits are not
    differentiable)."""

    exponent: float
    min_radius: float
    max_radius: float

    def _spec(self) -> tuple:
        return (
            "power_law",
            float(self.exponent),
            float(self.min_radius),
            float(self.max_radius),
        )


@dataclass(frozen=True)
class Monodisperse:
    """Spheres of one ``radius``."""

    radius: float

    def _spec(self) -> tuple:
        return ("monodisperse", float(self.radius))


@dataclass(frozen=True)
class Tabulated:
    """n(r) linear between the points (``radius[i]``, ``density[i]``) and 0 outside: radii
    increasing, densities ≥ 0. No parameters."""

    radius: Sequence[float]
    density: Sequence[float]

    def __post_init__(self) -> None:
        # Tuples of floats, so that the distribution compares and hashes by value.
        for name in ("radius", "density"):
            values = np.asarray(getattr(self, name), dtype=float).ravel()
            object.__setattr__(self, name, tuple(float(v) for v in values))

    def _spec(self) -> tuple:
        return ("tabulated", list(self.radius), list(self.density))


@dataclass(frozen=True)
class MultiModal:
    """Σⱼ wⱼ nⱼ(r), each mode normalized: ``modes`` is a sequence of (weight wⱼ > 0, mode).
    The modes' parameters are named ``"mode{j}.<name>"``, their weights ``"mode{j}.weight"``.
    """

    modes: Sequence[tuple[float, Distribution]]

    def __post_init__(self) -> None:
        modes = tuple((float(w), _check(d, f"mode {j}")) for j, (w, d) in enumerate(self.modes))
        object.__setattr__(self, "modes", modes)

    def _spec(self) -> tuple:
        return ("multimodal", [(w, d._spec()) for w, d in self.modes])


Distribution: TypeAlias = (
    LogNormal | Gamma | ModifiedGamma | PowerLaw | Monodisperse | Tabulated | MultiModal
)

_DISTRIBUTIONS = (LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal)


def _check(d: object, what: str) -> Distribution:
    if not isinstance(d, _DISTRIBUTIONS):
        names = ", ".join(c.__name__ for c in _DISTRIBUTIONS)
        raise TypeError(f"{what} must be one of {names}, not {type(d).__name__}")
    return d


@dataclass(frozen=True, eq=False)
class Component:
    """One component of an external mixture: a size distribution, its refractive index
    (broadcast to the wavelengths' shape), and its relative number concentration ``weight``
    ≥ 0. Components compare by identity."""

    distribution: Distribution
    refractive_index: ArrayLike
    weight: float = 1.0


def _frozen(a: np.ndarray) -> np.ndarray:
    """A read-only view of ``a``: an array that owns its data can be made writeable again, a
    view of a read-only array cannot."""
    a.flags.writeable = False
    return a.view()


@dataclass(frozen=True, eq=False)
class EnsembleGreek:
    """Greek coefficients per wavelength, arrays of shape ``wavelength.shape + (L + 1,)``,
    zero-padded to the largest order (see :class:`scatrs.mie.MieGreek` for the expansion)."""

    alpha1: NDArray[np.float64]
    alpha2: NDArray[np.float64]
    alpha3: NDArray[np.float64]
    alpha4: NDArray[np.float64]
    beta1: NDArray[np.float64]
    beta2: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class EnsemblePhase:
    """The phase matrix per wavelength at the cosines ``mu``, arrays of shape
    ``wavelength.shape + mu.shape``."""

    f11: NDArray[np.float64]
    f12: NDArray[np.float64]
    f33: NDArray[np.float64]
    f34: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class EnsembleOutputs:
    """The outputs of an ensemble (or their error estimates, or their derivatives with
    respect to one parameter), each an array of the wavelengths' shape.

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


@dataclass(frozen=True, eq=False)
class EnsembleResult(EnsembleOutputs):
    """:class:`EnsembleOutputs` with their error estimates and derivatives.

    Attributes
    ----------
    error : EnsembleOutputs
        Non-negative error estimates of every output (see :doc:`/validation/accuracy`).
    converged : numpy.ndarray
        bool per wavelength: whether every refined quantity met ``rtol`` within
        ``max_panels`` panels.
    panels : numpy.ndarray
        int64 per wavelength: the number of size panels.
    params : tuple of str
        The parameters :meth:`d` accepts.
    """

    error: EnsembleOutputs = field(kw_only=True)
    converged: NDArray[np.bool_] = field(kw_only=True)
    panels: NDArray[np.int64] = field(kw_only=True)
    params: tuple[str, ...] = field(default=(), kw_only=True)
    _d: tuple[EnsembleOutputs, ...] = field(default=(), repr=False, kw_only=True)
    _d_errors: tuple[EnsembleOutputs, ...] = field(default=(), repr=False, kw_only=True)

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


def _outputs(
    raw: dict, which: str, k: int | None, shape: tuple[int, ...], mu_shape
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


def _specs(components: Sequence[Component], shape: tuple[int, ...]) -> list:
    out = []
    for i, c in enumerate(components):
        if not isinstance(c, Component):
            raise TypeError(f"component {i} must be a Component, not {type(c).__name__}")
        spec = _check(c.distribution, f"the distribution of component {i}")._spec()
        m = _args.complex_array(c.refractive_index, "refractive_index")
        try:
            m = np.broadcast_to(m, shape)
        except ValueError:
            raise ValueError(
                f"refractive_index of shape {m.shape} does not broadcast to the wavelengths' "
                f"shape {shape}"
            ) from None
        out.append((spec, np.ascontiguousarray(m.reshape(-1)), float(c.weight)))
    return out


def _greek_order(greek: SupportsIndex | Literal["all"] | None) -> tuple[int | None, bool]:
    """The order and whether every order is wanted."""
    if greek is None:
        return None, False
    if isinstance(greek, str):
        if greek != "all":
            raise ValueError(f"greek must be an order, 'all' or None, not {greek!r}")
        return None, True
    return _args.size(greek, "greek"), False


def _run(
    items: list,
    wavelength: ArrayLike,
    mixture: bool,
    greek: SupportsIndex | Literal["all"] | None,
    mu: ArrayLike | None,
    normalized: bool,
    derivatives: bool | Sequence[str],
    rtol: float,
    max_panels: SupportsIndex,
    allow_unvalidated: bool,
    refine_backscatter: bool,
) -> EnsembleResult:
    wl = _args.real_array(wavelength, "wavelength")
    shape = wl.shape
    names = tuple(_scatrs.ensemble_parameters(items, mixture))
    if isinstance(derivatives, (bool, np.bool_)):
        params = names if derivatives else ()
    elif isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
        raise TypeError(
            "derivatives must be a bool or a sequence of parameter names, "
            f"not {type(derivatives).__name__}"
        )
    else:
        params = tuple(derivatives)
        for p in params:
            if p not in names:
                raise ValueError(f"unknown parameter {p!r}; expected one of {names}")
        if len(set(params)) != len(params):
            raise ValueError(f"a parameter is requested twice: {params}")
    mus = None if mu is None else _args.real_array(mu, "mu")
    order, all_orders = _greek_order(greek)
    raw = _scatrs.ensemble_compute(
        items,
        np.ascontiguousarray(wl.reshape(-1)),
        mixture,
        order,
        all_orders,
        np.ascontiguousarray(np.zeros(0) if mus is None else mus.reshape(-1)),
        _args.flag(normalized, "normalized"),
        [names.index(p) for p in params],
        float(rtol),
        _args.size(max_panels, "max_panels"),
        _args.flag(allow_unvalidated, "allow_unvalidated"),
        _args.flag(refine_backscatter, "refine_backscatter"),
    )
    raw["panels"] = raw["panels"].astype(np.int64)
    for a in raw.values():
        a.flags.writeable = False
    mu_shape = () if mus is None else mus.shape
    values = _outputs(raw, "", None, shape, mu_shape)
    return EnsembleResult(
        **{f: getattr(values, f) for f in (*SCALARS, "greek", "phase")},
        error=_outputs(raw, "_errors", None, shape, mu_shape),
        converged=raw["converged"].reshape(shape),
        panels=raw["panels"].reshape(shape),
        params=params,
        _d=tuple(_outputs(raw, "_d", k, shape, mu_shape) for k in range(len(params))),
        _d_errors=tuple(_outputs(raw, "_d_errors", k, shape, mu_shape) for k in range(len(params))),
    )


_SHARED_DOC = """
    Parameters
    ----------
    {first}wavelength : array_like
        Wavelengths, in the unit of the radii (real, scalar or any shape).
    {second}
    greek : int, "all" or None, optional
        The Greek coefficients to compute: up to this order, every order, or none (the
        default).
    mu : array_like, optional
        Cosines of the scattering angle at which to compute the phase matrix.
    normalized : bool, optional
        Normalize the phase matrix and the Greek coefficients (``alpha1[..., 0]`` = 1, the
        default) or give them per particle (``alpha1[..., 0]`` = ``c_sca``).
    derivatives : bool or sequence of str, optional
        The parameters to differentiate with respect to (``True`` for all): the
        distribution's (``"median_radius"``, …), ``"m_real"``,
        ``"m_imag"``{weights}.
    rtol : float, optional
        Relative tolerance of the size integration, in [1e-13, 0.1].
    max_panels : int, optional
        The most size panels per component and wavelength; reaching it sets ``converged``
        to False.
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

    Raises
    ------
    InvalidInputError
        For invalid distribution parameters, wavelengths, refractive indices or options.
    OutsideValidatedDomainError
        If a size node is outside the validated domain, unless ``allow_unvalidated``; the
        message names the wavelength and the radius.
    TypeError, ValueError
        For arguments of the wrong type, refractive indices that do not broadcast to the
        wavelengths, or unknown or repeated parameter names.
    BufferError
        If another rust-numpy extension holds a writeable borrow of an input.
    MemoryError
        If the outputs cannot be allocated. They are bounded like those of
        :mod:`scatrs.mie`: at most ``2 * MAX_MIE_VALUES`` real values.

    Notes
    -----
    The cost grows with the size parameters and with the order of the Greek coefficients
    (see :doc:`/limitations`). The call releases the GIL, so other Python threads keep
    running, but it cannot be interrupted: Ctrl+C takes effect only once it returns.
"""


def compute(
    distribution: Distribution,
    wavelength: ArrayLike,
    refractive_index: ArrayLike,
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
    items = _specs([Component(distribution, refractive_index)], wl.shape)
    return _run(
        items,
        wl,
        False,
        greek,
        mu,
        normalized,
        derivatives,
        rtol,
        max_panels,
        allow_unvalidated,
        refine_backscatter,
    )


compute.__doc__ = """Averages over spheres of one size distribution.
""" + _SHARED_DOC.format(
    first="distribution : LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated "
    "or MultiModal\n        The number size distribution.\n    ",
    second="refractive_index : array_like\n        The refractive index, broadcast to the "
    "wavelengths' shape.\n    ",
    weights=', and ``"mode{j}.weight"`` for multimodal distributions',
)


def compute_mixture(
    components: Sequence[Component],
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
    items = _specs(list(components), wl.shape)
    return _run(
        items,
        wl,
        True,
        greek,
        mu,
        normalized,
        derivatives,
        rtol,
        max_panels,
        allow_unvalidated,
        refine_backscatter,
    )


compute_mixture.__doc__ = """Averages over an external mixture: cross sections and moments
    weighted by wᵢ/Σw, phase matrices by wᵢ C_sca,ᵢ.
""" + _SHARED_DOC.format(
    first="components : sequence of Component\n        The components.\n    ",
    second="",
    weights=' (each prefixed by ``"c{i}."``), and the weights ``"c{i}.weight"``',
)
