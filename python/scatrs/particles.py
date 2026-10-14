"""Particles: what an :class:`~scatrs.Ensemble` is made of."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from scatrs.refractive import Constant, Function, RefractiveIndex

__all__ = ["Sphere"]


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
