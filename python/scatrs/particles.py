"""Particles: what an :class:`~scatrs.Ensemble` is made of."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scatrs.refractive import Constant, Function, RefractiveIndex

__all__ = ["LayeredSphere", "Sphere", "Spheroid"]


def _model(value: object) -> RefractiveIndex:
    """``value`` as a refractive-index model."""
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind != "O":
        value = value[()]  # a 0-d array is its numpy scalar, as for a number
    if isinstance(value, RefractiveIndex):
        return value
    if isinstance(value, type):  # callable, but not a model
        raise TypeError(
            "refractive_index must be a number, a model or a callable, not the class "
            f"{value.__name__}; instantiate it, as in scatrs.refractive.Constant(1.5)"
        )
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            "refractive_index must be a number, a scatrs.refractive model or a callable, not a bool"
        )
    if isinstance(value, numbers.Number):
        return Constant(value)  # type: ignore[arg-type]
    if callable(value):
        return Function(value)
    hint = (
        "; for values at given wavelengths use scatrs.refractive.Table"
        if isinstance(value, (np.ndarray, list, tuple))
        else ""
    )
    raise TypeError(
        "refractive_index must be a number, a scatrs.refractive model or a callable, not "
        f"{type(value).__name__}{hint}"
    )


@dataclass(frozen=True)
class Sphere:
    """A homogeneous sphere (Lorenz–Mie).

    ``refractive_index`` is a :mod:`scatrs.refractive` model; a number becomes
    :class:`~scatrs.refractive.Constant` and any other callable
    :class:`~scatrs.refractive.Function`.

    Raises
    ------
    TypeError
        For anything else, including arrays: an array of m means something only together
        with its wavelengths, which :class:`~scatrs.refractive.Table` holds.
    """

    refractive_index: RefractiveIndex

    def __post_init__(self) -> None:
        object.__setattr__(self, "refractive_index", _model(self.refractive_index))

    def _engine(self, wavelength: NDArray[np.float64]) -> tuple[tuple, NDArray[np.complex128]]:
        """The engine's tagged tuple for the particle at ``wavelength``, and its refractive
        index there (the wavelengths' shape), which the result reports."""
        m = self.refractive_index.evaluate(wavelength)
        return ("sphere", np.ascontiguousarray(m.reshape(-1))), m


def _fractions(value: object, layers: int) -> tuple[float, ...]:
    """``value`` as the radius fractions of ``layers`` layers.

    Raises
    ------
    TypeError
        If ``value`` is not a sequence of real numbers.
    ValueError
        If there are not ``layers - 1`` of them, or they are not nondecreasing in (0, 1].
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        raise TypeError(
            f"radius_fraction must be a sequence of real numbers, not {type(value).__name__}"
        )
    out: list[float] = []
    for i, v in enumerate(value):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            raise TypeError(f"radius_fraction[{i}] must be a real number, not {type(v).__name__}")
        f = float(v)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"radius_fraction[{i}] must be in (0, 1], got {f}")
        if out and f < out[-1]:
            raise ValueError(
                f"radius_fraction must be nondecreasing from the core out, but "
                f"radius_fraction[{i}] = {f} < {out[-1]}"
            )
        out.append(f)
    if len(out) != layers - 1:
        raise ValueError(
            f"a layered sphere of {layers} layers needs {layers - 1} radius fractions, "
            f"got {len(out)}"
        )
    return tuple(out)


@dataclass(frozen=True)
class LayeredSphere:
    """A layered sphere: a core and concentric shells (see :mod:`scatrs.layered`).

    In an :class:`~scatrs.Ensemble` the size distribution gives the outer radius, and each
    interface stays at a fixed fraction of it. The core's volume fraction is the cube of its
    radius fraction: a core of a quarter of the particle's volume has
    ``radius_fraction=[0.25 ** (1 / 3)]``.

    Parameters
    ----------
    refractive_index : sequence
        Each layer's refractive index, the core first: :mod:`scatrs.refractive` models, numbers
        (:class:`~scatrs.refractive.Constant`) or other callables
        (:class:`~scatrs.refractive.Function`), as for :class:`Sphere`. Stored as a tuple.
    radius_fraction : sequence of float
        The outer radius of each layer but the last, as a fraction of the particle's radius:
        L − 1 values, nondecreasing in (0, 1] (empty for one layer). Stored as a tuple.

    Raises
    ------
    TypeError
        If ``refractive_index`` is not a sequence of models, numbers or callables, or
        ``radius_fraction`` not a sequence of real numbers.
    ValueError
        If there is no layer, or the fractions are not L − 1 values nondecreasing in (0, 1].
    """

    refractive_index: tuple[RefractiveIndex, ...]
    radius_fraction: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        m = self.refractive_index
        if isinstance(m, (str, bytes)) or not isinstance(m, (Sequence, np.ndarray)):
            raise TypeError(
                "refractive_index must be a sequence with one model, number or callable per "
                f"layer, not {type(m).__name__}; for a homogeneous sphere use Sphere"
            )
        if len(m) == 0:
            raise ValueError("a layered sphere needs at least one layer")
        object.__setattr__(self, "refractive_index", tuple(_model(v) for v in m))
        object.__setattr__(
            self, "radius_fraction", _fractions(self.radius_fraction, len(self.refractive_index))
        )

    @classmethod
    def coated(cls, core: object, shell: object, *, core_fraction: float) -> LayeredSphere:
        """A core of refractive index ``core`` in a shell of ``shell``, the core's radius
        ``core_fraction`` of the particle's."""
        return cls((_model(core), _model(shell)), (core_fraction,))

    @property
    def layers(self) -> int:
        """The number of layers L."""
        return len(self.refractive_index)

    def _engine(self, wavelength: NDArray[np.float64]) -> tuple[tuple, NDArray[np.complex128]]:
        """The engine's tagged tuple for the particle at ``wavelength``, and its refractive
        indices there (shape ``(L,) + wavelength.shape``), which the result reports."""
        m = np.stack([v.evaluate(wavelength) for v in self.refractive_index])
        fractions = np.array(self.radius_fraction, dtype=np.float64)
        return ("layered", fractions, np.ascontiguousarray(m.reshape(self.layers, -1))), m


@dataclass(frozen=True)
class Spheroid:
    """Randomly oriented spheroids of one aspect ratio (see :mod:`scatrs.tmatrix`).

    A spheroid has semi-axes ``a, a, c``, ``c`` along its axis of symmetry, and aspect ratio
    ``aspect_ratio = a / c``: oblate above 1, prolate below 1, a sphere at 1. In an
    :class:`~scatrs.Ensemble` the size distribution gives the radius of the sphere of equal
    volume, and its cross sections are per particle, as for spheres. Each size node solves a
    T-matrix, which costs milliseconds to seconds rather than the microseconds of a sphere
    (see :doc:`/theory/ensembles`).

    Parameters
    ----------
    refractive_index
        A :mod:`scatrs.refractive` model; a number becomes
        :class:`~scatrs.refractive.Constant` and any other callable
        :class:`~scatrs.refractive.Function`, as for :class:`Sphere`.
    aspect_ratio : float
        a/c, finite and positive.

    Raises
    ------
    TypeError
        If ``refractive_index`` is not a model, number or callable, or ``aspect_ratio`` not a
        real number.
    ValueError
        If ``aspect_ratio`` is not finite and positive.
    """

    refractive_index: RefractiveIndex
    aspect_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "refractive_index", _model(self.refractive_index))
        v = self.aspect_ratio
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            raise TypeError(f"aspect_ratio must be a real number, not {type(v).__name__}")
        f = float(v)
        if not (np.isfinite(f) and f > 0.0):
            raise ValueError(f"aspect_ratio must be finite and positive, got {f}")
        object.__setattr__(self, "aspect_ratio", f)

    def _engine(self, wavelength: NDArray[np.float64]) -> tuple[tuple, NDArray[np.complex128]]:
        """The engine's tagged tuple for the particle at ``wavelength``, and its refractive
        index there (the wavelengths' shape), which the result reports."""
        m = self.refractive_index.evaluate(wavelength)
        return ("spheroid", self.aspect_ratio, np.ascontiguousarray(m.reshape(-1))), m
