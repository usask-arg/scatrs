"""Number size distributions of particles.

Radii are in any length unit, the unit of the wavelengths they are computed with. Every
numeric parameter may be an array: the parameters of a distribution broadcast together into a
grid of distributions, which :meth:`scatrs.Ensemble.compute` computes at once (see
:doc:`/theory/ensembles`). Arrays are copied and stored read-only, so distributions are
immutable values that compare and hash by their parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args

__all__ = [
    "Distribution",
    "Gamma",
    "LogNormal",
    "ModifiedGamma",
    "Monodisperse",
    "MultiModal",
    "PowerLaw",
    "Tabulated",
]

Param: TypeAlias = float | NDArray[np.float64]
"""A parameter: a float, or a read-only float64 array of at least one dimension."""

_Pick: TypeAlias = Callable[[str, Param], float]
"""The value of a parameter at one point of a grid, given its name and value."""


def _param(value: ArrayLike, name: str) -> Param:
    """``value`` as a float, or as a read-only copy of an array of dimension ≥ 1."""
    a = _args.real_array(value, name)
    if a.ndim == 0:
        return float(a)
    a = a.copy()
    a.flags.writeable = False
    return a


def _key(value: Param) -> Any:
    """``value`` in a form that compares and hashes by value."""
    if isinstance(value, float):
        return value
    return (value.shape, tuple(value.ravel().tolist()))


class _Distribution:
    """The parameters, value semantics and engine specs shared by the distributions."""

    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
        """The (name, value) of every parameter, nested ones prefixed."""
        raise NotImplementedError

    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
        """The engine's tagged tuple, with each parameter's value given by ``pick``."""
        raise NotImplementedError

    def _identity(self) -> tuple:
        return (type(self).__name__, *((n, _key(v)) for n, v in self._leaves()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Distribution):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


class _Simple(_Distribution):
    """A distribution whose parameters are its fields, in the engine's order."""

    _KIND: ClassVar[str]

    def __post_init__(self) -> None:
        for f in fields(self):  # type: ignore[arg-type]
            object.__setattr__(self, f.name, _param(getattr(self, f.name), f.name))

    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
        return [(prefix + f.name, getattr(self, f.name)) for f in fields(self)]  # type: ignore[arg-type]

    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
        return (self._KIND, *(pick(n, v) for n, v in self._leaves(prefix)))


@dataclass(frozen=True, eq=False)
class LogNormal(_Simple):
    """n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: ``median_radius`` r_g > 0, ``width``
    σ_g > 1."""

    _KIND = "lognormal"
    median_radius: Param
    width: Param


@dataclass(frozen=True, eq=False)
class Gamma(_Simple):
    """Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
    ``effective_radius`` a > 0, ``effective_variance`` 0 < b < 1/2."""

    _KIND = "gamma"
    effective_radius: Param
    effective_variance: Param


@dataclass(frozen=True, eq=False)
class ModifiedGamma(_Simple):
    """Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): ``alpha`` > −1,
    ``b`` > 0, ``gamma`` > 0."""

    _KIND = "modified_gamma"
    alpha: Param
    b: Param
    gamma: Param


@dataclass(frozen=True, eq=False)
class PowerLaw(_Simple):
    """n(r) ∝ r^(−α) on [``min_radius``, ``max_radius``]: ``exponent`` α (the limits are not
    differentiable)."""

    _KIND = "power_law"
    exponent: Param
    min_radius: Param
    max_radius: Param


@dataclass(frozen=True, eq=False)
class Monodisperse(_Simple):
    """Spheres of one ``radius``."""

    _KIND = "monodisperse"
    radius: Param


@dataclass(frozen=True, eq=False)
class Tabulated(_Distribution):
    """n(r) linear between the points (``radius[i]``, ``density[i]``) and 0 outside: radii
    increasing, densities ≥ 0. The table is data, not parameters: it is neither
    differentiated nor broadcast into a grid."""

    radius: Sequence[float]
    density: Sequence[float]

    def __post_init__(self) -> None:
        # Tuples of floats, so that the distribution compares and hashes by value.
        for name in ("radius", "density"):
            values = np.asarray(getattr(self, name), dtype=float).ravel()
            object.__setattr__(self, name, tuple(float(v) for v in values))

    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
        return []

    def _identity(self) -> tuple:
        return ("Tabulated", self.radius, self.density)

    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
        return ("tabulated", list(self.radius), list(self.density))


@dataclass(frozen=True, eq=False)
class MultiModal(_Distribution):
    """Σⱼ wⱼ nⱼ(r), each mode normalized: ``modes`` is a sequence of (weight wⱼ > 0, mode).
    The modes' parameters are named ``"mode{j}.<name>"``, their weights ``"mode{j}.weight"``.
    """

    modes: Sequence[tuple[Param, Distribution]]

    def __post_init__(self) -> None:
        modes = tuple(
            (_param(w, f"the weight of mode {j}"), _check(d, f"mode {j}"))
            for j, (w, d) in enumerate(self.modes)
        )
        object.__setattr__(self, "modes", modes)

    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
        out = []
        for j, (w, d) in enumerate(self.modes):
            out += d._leaves(f"{prefix}mode{j}.")
            out.append((f"{prefix}mode{j}.weight", w))
        return out

    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
        return (
            "multimodal",
            [
                (pick(f"{prefix}mode{j}.weight", w), d._spec(pick, f"{prefix}mode{j}."))
                for j, (w, d) in enumerate(self.modes)
            ],
        )


Distribution: TypeAlias = (
    LogNormal | Gamma | ModifiedGamma | PowerLaw | Monodisperse | Tabulated | MultiModal
)

_DISTRIBUTIONS = (LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal)


def _check(d: object, what: str) -> Distribution:
    """``d``, if it is a distribution.

    Raises
    ------
    TypeError
        Otherwise.
    """
    if not isinstance(d, _DISTRIBUTIONS):
        names = ", ".join(c.__name__ for c in _DISTRIBUTIONS)
        raise TypeError(f"{what} must be one of {names}, not {type(d).__name__}")
    return d


def _grid(d: Distribution) -> tuple[tuple[int, ...], dict[str, NDArray[np.float64]]]:
    """The grid shape of ``d``'s parameters, and the array-valued ones broadcast to it.

    Raises
    ------
    ValueError
        If the array-valued parameters do not broadcast together.
    """
    arrays = {n: v for n, v in d._leaves() if not isinstance(v, float)}
    try:
        shape = np.broadcast_shapes(*(a.shape for a in arrays.values()))
    except ValueError:
        shapes = ", ".join(f"{n} {a.shape}" for n, a in arrays.items())
        raise ValueError(
            f"the distribution's parameters do not broadcast together: {shapes}"
        ) from None
    return shape, {n: np.broadcast_to(a, shape) for n, a in arrays.items()}


def _specs(d: Distribution) -> tuple[tuple[int, ...], list[tuple]]:
    """The grid shape of ``d``, and the engine spec of each of its points in C order (one for
    a scalar distribution)."""
    shape, arrays = _grid(d)
    flat = {n: a.reshape(-1) for n, a in arrays.items()}
    count = int(np.prod(shape))

    def at(k: int) -> _Pick:
        return lambda n, v: float(v) if isinstance(v, float) else float(flat[n][k])

    return shape, [d._spec(at(k)) for k in range(count)]
