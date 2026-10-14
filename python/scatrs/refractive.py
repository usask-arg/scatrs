"""Refractive-index models: m(λ) = n + ik relative to the surrounding medium.

Im m ≥ 0 means absorbing (see :doc:`/theory/conventions`); codes that write m = n − ik need the
complex conjugate. Wavelengths are in the unit of the particle's radii, and so are the
coefficients of :class:`Cauchy` and :class:`Sellmeier`: published coefficients usually assume
micrometres. A model evaluated at a wavelength where it has no valid value raises
:class:`~scatrs.errors.InvalidInputError` naming that wavelength; there is no extrapolation.

Every model names its differentiable parameters (:attr:`RefractiveIndex.parameters`) and gives
∂m/∂θ for each (:meth:`RefractiveIndex.derivatives`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _args
from scatrs.errors import InvalidInputError

__all__ = ["Cauchy", "Constant", "Function", "RefractiveIndex", "Sellmeier", "Table"]

# Accepted by float() and complex() but not numbers: rejected with TypeError.
_NOT_NUMBERS: Final = (bool, np.bool_, str, bytes)


def _finite(value: object, what: str) -> float:
    """``value`` as a finite float; ``what`` names it in errors ("Cauchy: A", say)."""
    try:
        if isinstance(value, _NOT_NUMBERS):
            raise TypeError
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(f"{what} must be a real number, not {type(value).__name__}") from None
    if not np.isfinite(v):
        raise InvalidInputError(f"invalid input: {what} must be finite, got {v}")
    return v


def _absorption(k: object, what: str) -> float:
    v = _finite(k, what)
    if v < 0:
        raise InvalidInputError(f"invalid input: {what} must not be negative, got {v}")
    return v


def _finite_sequence(values: Sequence[float], what: str) -> tuple[float, ...]:
    """A one-dimensional sequence of finite floats, as :func:`_finite` for each element."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{what} must be a sequence of real numbers, not {type(values).__name__}")
    try:
        shape = np.shape(values)
    except ValueError:  # ragged
        shape = None
    if shape is None or len(shape) != 1:
        got = "a ragged sequence" if shape is None else f"shape {shape}"
        raise InvalidInputError(
            f"invalid input: {what} must be a one-dimensional sequence, got {got}"
        )
    # The elements themselves, not np.asarray's conversion of them, so bools and strings
    # among numbers are caught.
    return tuple(_finite(v, f"{what}[{i}]") for i, v in enumerate(values))


class RefractiveIndex(ABC):
    """The interface of the models: subclasses implement ``_m(wavelength)`` (m, unchecked)
    and, if they have parameters, :attr:`parameters` and ``_dm(wavelength)`` (∂m/∂θ)."""

    @property
    def parameters(self) -> tuple[str, ...]:
        """The names of the differentiable parameters, in the order of
        :meth:`~scatrs.refractive.RefractiveIndex.derivatives`."""
        return ()

    @abstractmethod
    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
        """m at ``wavelength`` (unchecked)."""

    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
        """∂m/∂θ, of shape ``(len(parameters),) + wavelength.shape``."""
        return np.zeros((0, *wavelength.shape), dtype=complex)

    def evaluate(self, wavelength: ArrayLike) -> NDArray[np.complex128]:
        """m at ``wavelength`` (real, any shape), of the wavelengths' shape.

        Raises
        ------
        InvalidInputError
            If m is not a finite complex array of the wavelengths' shape, or Im m < 0, naming
            the first wavelength where it fails.
        """
        wl = _args.real_array(wavelength, "wavelength")
        m = np.asarray(self._m(wl))
        if m.dtype.kind not in "iufc":
            raise InvalidInputError(
                f"invalid input: {self!r} returned an array of dtype {m.dtype}, not numbers"
            )
        if m.shape != wl.shape:
            raise InvalidInputError(
                f"invalid input: {self!r} returned shape {m.shape} for wavelengths of shape "
                f"{wl.shape}"
            )
        m = m.astype(complex)
        bad = ~np.isfinite(m) | (m.imag < 0)
        if bad.any():
            i = np.unravel_index(np.argmax(bad), bad.shape)
            raise InvalidInputError(
                f"invalid input: {self!r} gives m = {m[i]} at wavelength {wl[i]}; m must be "
                "finite with Im m ≥ 0"
            )
        return m

    def derivatives(self, wavelength: ArrayLike) -> NDArray[np.complex128]:
        """∂m/∂θ at ``wavelength`` for each parameter θ of :attr:`parameters`, of shape
        ``(len(parameters),) + wavelength.shape``."""
        wl = _args.real_array(wavelength, "wavelength")
        return np.asarray(self._dm(wl), dtype=complex)


@dataclass(frozen=True)
class Constant(RefractiveIndex):
    """m at every wavelength. Parameters ``"n"`` = Re m and ``"k"`` = Im m."""

    m: complex

    def __post_init__(self) -> None:
        try:
            if isinstance(self.m, _NOT_NUMBERS):
                raise TypeError
            m = complex(self.m)
        except (TypeError, ValueError):
            raise TypeError(f"Constant: m must be a number, not {type(self.m).__name__}") from None
        if not (np.isfinite(m.real) and np.isfinite(m.imag)) or m.imag < 0:
            raise InvalidInputError(
                f"invalid input: Constant: m must be finite with Im m ≥ 0, got {m}"
            )
        object.__setattr__(self, "m", m)

    @property
    def parameters(self) -> tuple[str, ...]:
        return ("n", "k")

    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
        return np.full(wavelength.shape, self.m)

    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
        out = np.empty((2, *wavelength.shape), dtype=complex)
        out[0], out[1] = 1.0, 1j
        return out


@dataclass(frozen=True)
class Table(RefractiveIndex):
    """m tabulated at increasing ``wavelength`` (at least two), interpolated between them.

    ``interpolation="linear"`` interpolates Re m and Im m linearly in λ; ``"log_k"``
    interpolates ln Im m instead, for absorption spanning decades, and needs Im m > 0 at every
    node. The table is data: there are no parameters.
    """

    wavelength: Sequence[float]
    m: Sequence[complex]
    interpolation: Literal["linear", "log_k"] = "linear"

    def __post_init__(self) -> None:
        wl = _args.real_array(self.wavelength, "wavelength")
        m = _args.complex_array(self.m, "m")
        if wl.ndim != 1 or m.shape != wl.shape or wl.size < 2:
            raise InvalidInputError(
                "invalid input: a Table needs one-dimensional wavelength and m of the same "
                f"length, at least 2; got shapes {wl.shape} and {m.shape}"
            )
        if not (np.all(np.isfinite(wl)) and np.all(np.isfinite(m))):
            raise InvalidInputError("invalid input: a Table's wavelengths and m must be finite")
        if not np.all(np.diff(wl) > 0):
            raise InvalidInputError(
                "invalid input: a Table's wavelengths must be strictly increasing"
            )
        if self.interpolation not in ("linear", "log_k"):
            raise InvalidInputError(
                "invalid input: a Table's interpolation must be 'linear' or 'log_k', not "
                f"{self.interpolation!r}"
            )
        floor = 0.0 if self.interpolation == "linear" else np.nextafter(0.0, 1.0)
        if np.any(m.imag < floor):
            need = "≥ 0" if self.interpolation == "linear" else "> 0 for log_k interpolation"
            raise InvalidInputError(f"invalid input: a Table's Im m must be {need}")
        # Tuples, so that the table compares and hashes by value.
        object.__setattr__(self, "wavelength", tuple(wl.tolist()))
        object.__setattr__(self, "m", tuple(complex(v) for v in m))

    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
        lo, hi = self.wavelength[0], self.wavelength[-1]
        outside = ~((wavelength >= lo) & (wavelength <= hi))
        if outside.any():
            i = np.unravel_index(np.argmax(outside), outside.shape)
            raise InvalidInputError(
                f"invalid input: wavelength {wavelength[i]} is outside the Table's range "
                f"[{lo}, {hi}]"
            )
        nodes = np.array(self.m)
        n = np.interp(wavelength, self.wavelength, nodes.real)
        if self.interpolation == "linear":
            k = np.interp(wavelength, self.wavelength, nodes.imag)
        else:
            k = np.exp(np.interp(wavelength, self.wavelength, np.log(nodes.imag)))
        return n + 1j * k


@dataclass(frozen=True)
class Cauchy(RefractiveIndex):
    """n = A + B/λ² + C/λ⁴ and Im m = k. Parameters ``"A"``, ``"B"``, ``"C"``, ``"k"``."""

    A: float
    B: float = 0.0
    C: float = 0.0
    k: float = 0.0

    def __post_init__(self) -> None:
        for name in ("A", "B", "C"):
            object.__setattr__(self, name, _finite(getattr(self, name), f"Cauchy: {name}"))
        object.__setattr__(self, "k", _absorption(self.k, "Cauchy: k"))

    @property
    def parameters(self) -> tuple[str, ...]:
        return ("A", "B", "C", "k")

    # At λ = 0 these are not finite; evaluate() reports that, so numpy need not warn.
    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s = wavelength**-2.0
            return self.A + self.B * s + self.C * s * s + 1j * self.k

    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s = wavelength**-2.0
            out = np.empty((4, *wavelength.shape), dtype=complex)
            out[0], out[1], out[2], out[3] = 1.0, s, s * s, 1j
        return out


@dataclass(frozen=True)
class Sellmeier(RefractiveIndex):
    """n² = 1 + Σᵢ Bᵢλ²/(λ² − Cᵢ) and Im m = k, for sequences ``B`` and ``C`` of the same
    length. Parameters ``"B0"``, …, ``"C0"``, …, ``"k"``. Evaluating it where n² ≤ 0 (at or
    between poles, where the law does not describe the material) raises."""

    B: Sequence[float]
    C: Sequence[float]
    k: float = 0.0

    def __post_init__(self) -> None:
        b = _finite_sequence(self.B, "Sellmeier: B")
        c = _finite_sequence(self.C, "Sellmeier: C")
        if len(b) != len(c) or not b:
            raise InvalidInputError(
                "invalid input: Sellmeier: B and C must have the same length, at least 1; got "
                f"{len(b)} and {len(c)}"
            )
        object.__setattr__(self, "B", b)
        object.__setattr__(self, "C", c)
        object.__setattr__(self, "k", _absorption(self.k, "Sellmeier: k"))

    @property
    def parameters(self) -> tuple[str, ...]:
        terms = range(len(self.B))
        return (*(f"B{i}" for i in terms), *(f"C{i}" for i in terms), "k")

    def _n(self, wavelength: NDArray[np.float64]) -> NDArray[np.float64]:
        l2 = wavelength * wavelength
        n2 = np.ones_like(l2)
        with np.errstate(divide="ignore", invalid="ignore"):
            for b, c in zip(self.B, self.C, strict=True):
                n2 += b * l2 / (l2 - c)
        bad = ~(n2 > 0) | ~np.isfinite(n2)
        if np.any(bad):
            i = np.unravel_index(np.argmax(bad), bad.shape)
            raise InvalidInputError(
                f"invalid input: {self!r} gives n² = {n2[i]} at wavelength {wavelength[i]}; "
                "n² must be positive and finite"
            )
        return np.sqrt(n2)

    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
        return self._n(wavelength) + 1j * self.k

    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
        n = self._n(wavelength)
        l2 = wavelength * wavelength
        terms = len(self.B)
        out = np.empty((2 * terms + 1, *wavelength.shape), dtype=complex)
        for i, (b, c) in enumerate(zip(self.B, self.C, strict=True)):
            out[i] = l2 / (l2 - c) / (2 * n)  # ∂n/∂Bᵢ = (∂n²/∂Bᵢ)/(2n)
            out[terms + i] = b * l2 / (l2 - c) ** 2 / (2 * n)
        out[-1] = 1j
        return out


@dataclass(frozen=True, eq=False)
class Function(RefractiveIndex):
    """m = ``f(wavelength)`` for any callable ``f`` that takes a float64 array and returns m of
    its shape. No parameters; compares by identity."""

    f: Callable[[NDArray[np.float64]], ArrayLike]

    def __post_init__(self) -> None:
        if not callable(self.f):
            raise TypeError(f"f must be callable, not {type(self.f).__name__}")

    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
        return self.f(wavelength.copy())
