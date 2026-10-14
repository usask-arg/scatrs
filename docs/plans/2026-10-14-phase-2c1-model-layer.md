# Phase 2c-1: The Model Layer of the Python API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the function API of `scatrs.ensemble` with immutable `Sphere`/`Ensemble`/`Mixture` objects, refractive-index models over wavelength, and array-valued distribution parameters (grids) in place of `compute_batch`.

**Architecture:** Python only. `scatrs.refractive` models map wavelengths to m and name their parameters; `scatrs.Sphere` holds one; `scatrs.size` distributions store scalar or read-only array parameters and flatten their broadcast grid into the engine's tagged tuples; `Ensemble.compute` / `Mixture.compute` evaluate the model, then call the unchanged private bindings (`_scatrs.ensemble_compute` for a scalar distribution or a mixture, `_scatrs.ensemble_compute_batch` for a grid) exactly as 0.1.0's functions did, so results are bitwise those of 0.1.0.

**Tech Stack:** Python ≥ 3.11, numpy ≥ 1.26, the existing pyo3 extension (no Rust changes), pytest, ruff, Sphinx 9 + myst-nb (executed examples, `-W`).

**Spec:** `docs/specs/2026-10-14-phase-2c1-model-layer-design.md`.

## Global Constraints

- No change to the Rust core crate or to the private bindings `_scatrs.ensemble_parameters`, `ensemble_compute`, `ensemble_compute_batch`.
- Breaking the 0.1.0 `scatrs.ensemble` API is allowed (user's choice); the CHANGELOG lists each removed call with its replacement.
- Errors: `InvalidInputError` for invalid values (including a model with no valid value at a wavelength, naming it), `TypeError` for wrong types, `ValueError` for shapes that do not broadcast and mixtures of grids, `NotImplementedError` for derivatives of grids.
- Docs are user-facing: no phases or development history. `sphinx-build -E -W --keep-going` must pass (it executes the examples).
- `uv run --locked --only-group dev ruff check .` and `ruff format --check` clean; `uv run pytest -q` green at every commit.
- Never send personal data in network requests. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## How this plan was made

As in phases 1b and 2a, both tasks were prototyped and verified before this plan was written; their code is the exact diff of their commits on branch `phase-2c` (task 1: `d472d0c`, task 2: `92845e9`). Verified at each commit: ruff check and format, `uv run pytest -q` (325 then 333 passed, 1 skipped), and for task 2 the docs build with `-W` and mypy on the new modules (`ensemble.py` keeps the two `EnsembleOutputs(**…, greek=…)` errors it has on `main`). The documentation is folded into task 2 because the docs build executes the examples, which use the API that task 2 replaces.

## Measured results the plan relies on

| What | Measured |
|---|---|
| `benchmarks/compare.py` scatrs results (quick W1, W2 at rtol 1e-3), old `compute_batch` vs ported grid | bitwise equal |
| Grid of 12 r_g in [0.02, 1] µm × 3 σ_g, m = 1.43 + 1e-8i, λ = 0.35 µm | 7 of 36 points unconverged at the shared 4096-panel cap (resonance ripples, a known limitation); the example uses r_g ≤ 0.5 µm, all converged |
| Python test suite | 1.5 s |

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `python/scatrs/refractive.py` | 1 | `RefractiveIndex` (ABC: `evaluate`, `parameters`, `derivatives`), `Constant`, `Table`, `Cauchy`, `Sellmeier`, `Function` |
| `python/scatrs/particles.py` | 1 | `Sphere` and the conversion of its `refractive_index` |
| `tests/test_refractive.py` | 1 | models, derivatives vs central differences, errors, `Sphere` |
| `python/scatrs/size.py` | 2 | the seven distributions with array-valued parameters; `_check`, `_grid`, `_specs` |
| `python/scatrs/ensemble.py` | 2 | `Ensemble`, `Mixture`, result classes, engine calls |
| `python/scatrs/__init__.py` | 1, 2 | top-level exports |
| `tests/test_ensemble.py`, `tests/test_size.py`, `tests/test_ensemble_engine_calls.py`, `tests/test_package.py` | 2 | ported and new tests; bitwise equivalence with the engine calls |
| `benchmarks/compare.py` | 2 | ported to a grid |
| `README.md`, `CHANGELOG.md`, `docs/…`, `.gitignore` | 2 | docs on the new API; new `docs/theory/refractive-index.md` |

---

### Task 1: Refractive-index models and `Sphere`

**Files:**
- Create: `python/scatrs/refractive.py`, `python/scatrs/particles.py`, `tests/test_refractive.py`
- Modify: `python/scatrs/__init__.py`

**Interfaces:**
- Consumes: `scatrs._args.real_array`, `scatrs._args.complex_array`, `scatrs.errors.InvalidInputError`.
- Produces:
  - `scatrs.refractive.RefractiveIndex` with `evaluate(wavelength: ArrayLike) -> NDArray[complex128]` (wavelengths' shape; raises `InvalidInputError`), `parameters -> tuple[str, ...]`, `derivatives(wavelength) -> NDArray[complex128]` of shape `(len(parameters),) + wavelength.shape`;
  - `Constant(m)` (`"n"`, `"k"`), `Table(wavelength, m, interpolation="linear" | "log_k")` (no parameters), `Cauchy(A, B=0.0, C=0.0, k=0.0)` (`"A"`, `"B"`, `"C"`, `"k"`), `Sellmeier(B, C, k=0.0)` (`"B{i}"`, `"C{i}"`, `"k"`), `Function(f)` (no parameters, identity equality);
  - `scatrs.Sphere(refractive_index)`, field `refractive_index: RefractiveIndex`.

- [ ] **Step 1: Write the failing tests** — `tests/test_refractive.py` as in the diff below.
- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q tests/test_refractive.py`
Expected: FAIL at import (`cannot import name 'Sphere' from 'scatrs'`).

- [ ] **Step 3: Implement** — `python/scatrs/refractive.py`, `python/scatrs/particles.py` and the `__init__.py` exports, as in the diff below.
- [ ] **Step 4: Run the tests and the linters**

Run: `uv run pytest -q && uv run --locked --only-group dev ruff check . && uv run --locked --only-group dev ruff format --check python tests`
Expected: `325 passed, 1 skipped`; ruff clean.

- [ ] **Step 5: Commit** — `git add python/scatrs/refractive.py python/scatrs/particles.py python/scatrs/__init__.py tests/test_refractive.py && git commit -m "feat(python): refractive-index models and Sphere"` (body and `Co-Authored-By` as in `d472d0c`).

The diff (commit `d472d0c`):

````diff
diff --git a/python/scatrs/__init__.py b/python/scatrs/__init__.py
index 99c2dc6..9bd9cf7 100644
--- a/python/scatrs/__init__.py
+++ b/python/scatrs/__init__.py
@@ -3,22 +3,26 @@
 from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
 from importlib.metadata import version as _version
 
-from scatrs import _scatrs, ensemble, mie, special
+from scatrs import _scatrs, ensemble, mie, particles, refractive, special
 from scatrs.errors import (
     InvalidInputError,
     NotConvergedError,
     OutsideValidatedDomainError,
     ScatrsError,
 )
+from scatrs.particles import Sphere
 
 __all__ = [
     "InvalidInputError",
     "NotConvergedError",
     "OutsideValidatedDomainError",
     "ScatrsError",
+    "Sphere",
     "__version__",
     "ensemble",
     "mie",
+    "particles",
+    "refractive",
     "special",
 ]
 
diff --git a/python/scatrs/particles.py b/python/scatrs/particles.py
new file mode 100644
index 0000000..f774dd2
--- /dev/null
+++ b/python/scatrs/particles.py
@@ -0,0 +1,54 @@
+"""Particles: what an :class:`~scatrs.Ensemble` is made of."""
+
+from __future__ import annotations
+
+import numbers
+from dataclasses import dataclass
+
+import numpy as np
+
+from scatrs.refractive import Constant, Function, RefractiveIndex
+
+__all__ = ["Sphere"]
+
+
+def _model(value: object) -> RefractiveIndex:
+    """``value`` as a refractive-index model."""
+    if isinstance(value, RefractiveIndex):
+        return value
+    if isinstance(value, (bool, np.bool_)):
+        raise TypeError("refractive_index must be a number or a model, not a bool")
+    if isinstance(value, numbers.Number):
+        return Constant(value)  # type: ignore[arg-type]
+    if callable(value):
+        return Function(value)
+    hint = (
+        "; for values at given wavelengths use scatrs.refractive.Table"
+        if isinstance(value, (np.ndarray, list, tuple))
+        else ""
+    )
+    raise TypeError(
+        "refractive_index must be a number, a scatrs.refractive model or a callable, not "
+        f"{type(value).__name__}{hint}"
+    )
+
+
+@dataclass(frozen=True)
+class Sphere:
+    """A homogeneous sphere (Lorenz–Mie).
+
+    ``refractive_index`` is a :mod:`scatrs.refractive` model; a number becomes
+    :class:`~scatrs.refractive.Constant` and any other callable
+    :class:`~scatrs.refractive.Function`.
+
+    Raises
+    ------
+    TypeError
+        For anything else, including arrays: an array of m means something only together
+        with its wavelengths, which :class:`~scatrs.refractive.Table` holds.
+    """
+
+    refractive_index: RefractiveIndex
+
+    def __post_init__(self) -> None:
+        object.__setattr__(self, "refractive_index", _model(self.refractive_index))
diff --git a/python/scatrs/refractive.py b/python/scatrs/refractive.py
new file mode 100644
index 0000000..1d76fa4
--- /dev/null
+++ b/python/scatrs/refractive.py
@@ -0,0 +1,284 @@
+"""Refractive-index models: m(λ) = n + ik relative to the surrounding medium.
+
+Im m ≥ 0 means absorbing (see :doc:`/theory/conventions`); codes that write m = n − ik need the
+complex conjugate. Wavelengths are in the unit of the particle's radii, and so are the
+coefficients of :class:`Cauchy` and :class:`Sellmeier`: published coefficients usually assume
+micrometres. A model evaluated at a wavelength where it has no valid value raises
+:class:`~scatrs.errors.InvalidInputError` naming that wavelength; there is no extrapolation.
+
+Every model names its differentiable parameters (:attr:`RefractiveIndex.parameters`) and gives
+∂m/∂θ for each (:meth:`RefractiveIndex.derivatives`).
+"""
+
+from __future__ import annotations
+
+from abc import ABC, abstractmethod
+from collections.abc import Callable, Sequence
+from dataclasses import dataclass
+from typing import Literal
+
+import numpy as np
+from numpy.typing import ArrayLike, NDArray
+
+from scatrs import _args
+from scatrs.errors import InvalidInputError
+
+__all__ = ["Cauchy", "Constant", "Function", "RefractiveIndex", "Sellmeier", "Table"]
+
+
+def _finite(value: object, what: str) -> float:
+    try:
+        v = float(value)  # type: ignore[arg-type]
+    except (TypeError, ValueError):
+        raise TypeError(f"{what} must be a real number, not {type(value).__name__}") from None
+    if not np.isfinite(v):
+        raise InvalidInputError(f"invalid input: {what} must be finite, got {v}")
+    return v
+
+
+def _absorption(k: object, what: str) -> float:
+    v = _finite(k, what)
+    if v < 0:
+        raise InvalidInputError(f"invalid input: {what} must not be negative, got {v}")
+    return v
+
+
+class RefractiveIndex(ABC):
+    """The interface of the models: subclasses implement ``_m(wavelength)`` (m, unchecked)
+    and, if they have parameters, :attr:`parameters` and ``_dm(wavelength)`` (∂m/∂θ)."""
+
+    @property
+    def parameters(self) -> tuple[str, ...]:
+        """The names of the differentiable parameters, in the order of
+        :meth:`~scatrs.refractive.RefractiveIndex.derivatives`."""
+        return ()
+
+    @abstractmethod
+    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
+        """m at ``wavelength`` (unchecked)."""
+
+    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
+        """∂m/∂θ, of shape ``(len(parameters),) + wavelength.shape``."""
+        return np.zeros((0, *wavelength.shape), dtype=complex)
+
+    def evaluate(self, wavelength: ArrayLike) -> NDArray[np.complex128]:
+        """m at ``wavelength`` (real, any shape), of the wavelengths' shape.
+
+        Raises
+        ------
+        InvalidInputError
+            If m is not a finite complex array of the wavelengths' shape, or Im m < 0, naming
+            the first wavelength where it fails.
+        """
+        wl = _args.real_array(wavelength, "wavelength")
+        m = np.asarray(self._m(wl))
+        if m.dtype.kind not in "iufc":
+            raise InvalidInputError(
+                f"invalid input: {self!r} returned an array of dtype {m.dtype}, not numbers"
+            )
+        if m.shape != wl.shape:
+            raise InvalidInputError(
+                f"invalid input: {self!r} returned shape {m.shape} for wavelengths of shape "
+                f"{wl.shape}"
+            )
+        m = m.astype(complex)
+        bad = ~np.isfinite(m) | (m.imag < 0)
+        if bad.any():
+            i = np.unravel_index(np.argmax(bad), bad.shape)
+            raise InvalidInputError(
+                f"invalid input: {self!r} gives m = {m[i]} at wavelength {wl[i]}; m must be "
+                "finite with Im m ≥ 0"
+            )
+        return m
+
+    def derivatives(self, wavelength: ArrayLike) -> NDArray[np.complex128]:
+        """∂m/∂θ at ``wavelength`` for each parameter θ of :attr:`parameters`, of shape
+        ``(len(parameters),) + wavelength.shape``."""
+        wl = _args.real_array(wavelength, "wavelength")
+        return np.asarray(self._dm(wl), dtype=complex)
+
+
+@dataclass(frozen=True)
+class Constant(RefractiveIndex):
+    """m at every wavelength. Parameters ``"n"`` = Re m and ``"k"`` = Im m."""
+
+    m: complex
+
+    def __post_init__(self) -> None:
+        try:
+            m = complex(self.m)
+        except (TypeError, ValueError):
+            raise TypeError(f"m must be a number, not {type(self.m).__name__}") from None
+        if not (np.isfinite(m.real) and np.isfinite(m.imag)) or m.imag < 0:
+            raise InvalidInputError(f"invalid input: m must be finite with Im m ≥ 0, got {m}")
+        object.__setattr__(self, "m", m)
+
+    @property
+    def parameters(self) -> tuple[str, ...]:
+        return ("n", "k")
+
+    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
+        return np.full(wavelength.shape, self.m)
+
+    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
+        out = np.empty((2, *wavelength.shape), dtype=complex)
+        out[0], out[1] = 1.0, 1j
+        return out
+
+
+@dataclass(frozen=True)
+class Table(RefractiveIndex):
+    """m tabulated at increasing ``wavelength`` (at least two), interpolated between them.
+
+    ``interpolation="linear"`` interpolates Re m and Im m linearly in λ; ``"log_k"``
+    interpolates ln Im m instead, for absorption spanning decades, and needs Im m > 0 at every
+    node. The table is data: there are no parameters.
+    """
+
+    wavelength: Sequence[float]
+    m: Sequence[complex]
+    interpolation: Literal["linear", "log_k"] = "linear"
+
+    def __post_init__(self) -> None:
+        wl = _args.real_array(self.wavelength, "wavelength")
+        m = _args.complex_array(self.m, "m")
+        if wl.ndim != 1 or m.shape != wl.shape or wl.size < 2:
+            raise InvalidInputError(
+                "invalid input: a Table needs one-dimensional wavelength and m of the same "
+                f"length, at least 2; got shapes {wl.shape} and {m.shape}"
+            )
+        if not (np.all(np.isfinite(wl)) and np.all(np.isfinite(m))):
+            raise InvalidInputError("invalid input: a Table's wavelengths and m must be finite")
+        if not np.all(np.diff(wl) > 0):
+            raise InvalidInputError(
+                "invalid input: a Table's wavelengths must be strictly increasing"
+            )
+        if self.interpolation not in ("linear", "log_k"):
+            raise ValueError(
+                f"interpolation must be 'linear' or 'log_k', not {self.interpolation!r}"
+            )
+        floor = 0.0 if self.interpolation == "linear" else np.nextafter(0.0, 1.0)
+        if np.any(m.imag < floor):
+            need = "≥ 0" if self.interpolation == "linear" else "> 0 for log_k interpolation"
+            raise InvalidInputError(f"invalid input: a Table's Im m must be {need}")
+        # Tuples, so that the table compares and hashes by value.
+        object.__setattr__(self, "wavelength", tuple(wl.tolist()))
+        object.__setattr__(self, "m", tuple(complex(v) for v in m))
+
+    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
+        lo, hi = self.wavelength[0], self.wavelength[-1]
+        outside = ~((wavelength >= lo) & (wavelength <= hi))
+        if outside.any():
+            i = np.unravel_index(np.argmax(outside), outside.shape)
+            raise InvalidInputError(
+                f"invalid input: wavelength {wavelength[i]} is outside the Table's range "
+                f"[{lo}, {hi}]"
+            )
+        nodes = np.array(self.m)
+        n = np.interp(wavelength, self.wavelength, nodes.real)
+        if self.interpolation == "linear":
+            k = np.interp(wavelength, self.wavelength, nodes.imag)
+        else:
+            k = np.exp(np.interp(wavelength, self.wavelength, np.log(nodes.imag)))
+        return n + 1j * k
+
+
+@dataclass(frozen=True)
+class Cauchy(RefractiveIndex):
+    """n = A + B/λ² + C/λ⁴ and Im m = k. Parameters ``"A"``, ``"B"``, ``"C"``, ``"k"``."""
+
+    A: float
+    B: float = 0.0
+    C: float = 0.0
+    k: float = 0.0
+
+    def __post_init__(self) -> None:
+        for name in ("A", "B", "C"):
+            object.__setattr__(self, name, _finite(getattr(self, name), name))
+        object.__setattr__(self, "k", _absorption(self.k, "k"))
+
+    @property
+    def parameters(self) -> tuple[str, ...]:
+        return ("A", "B", "C", "k")
+
+    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
+        s = wavelength**-2.0
+        return self.A + self.B * s + self.C * s * s + 1j * self.k
+
+    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
+        s = wavelength**-2.0
+        out = np.empty((4, *wavelength.shape), dtype=complex)
+        out[0], out[1], out[2], out[3] = 1.0, s, s * s, 1j
+        return out
+
+
+@dataclass(frozen=True)
+class Sellmeier(RefractiveIndex):
+    """n² = 1 + Σᵢ Bᵢλ²/(λ² − Cᵢ) and Im m = k, for sequences ``B`` and ``C`` of the same
+    length. Parameters ``"B0"``, …, ``"C0"``, …, ``"k"``. Evaluating it where n² ≤ 0 (at or
+    between poles, where the law does not describe the material) raises."""
+
+    B: Sequence[float]
+    C: Sequence[float]
+    k: float = 0.0
+
+    def __post_init__(self) -> None:
+        b = tuple(_finite(v, f"B[{i}]") for i, v in enumerate(np.ravel(self.B)))
+        c = tuple(_finite(v, f"C[{i}]") for i, v in enumerate(np.ravel(self.C)))
+        if len(b) != len(c) or not b:
+            raise InvalidInputError(
+                f"invalid input: B and C must have the same length, at least 1; got {len(b)} "
+                f"and {len(c)}"
+            )
+        object.__setattr__(self, "B", b)
+        object.__setattr__(self, "C", c)
+        object.__setattr__(self, "k", _absorption(self.k, "k"))
+
+    @property
+    def parameters(self) -> tuple[str, ...]:
+        terms = range(len(self.B))
+        return (*(f"B{i}" for i in terms), *(f"C{i}" for i in terms), "k")
+
+    def _n(self, wavelength: NDArray[np.float64]) -> NDArray[np.float64]:
+        l2 = wavelength * wavelength
+        n2 = np.ones_like(l2)
+        with np.errstate(divide="ignore", invalid="ignore"):
+            for b, c in zip(self.B, self.C, strict=True):
+                n2 += b * l2 / (l2 - c)
+        bad = ~(n2 > 0) | ~np.isfinite(n2)
+        if np.any(bad):
+            i = np.unravel_index(np.argmax(bad), bad.shape)
+            raise InvalidInputError(
+                f"invalid input: {self!r} gives n² = {n2[i]} at wavelength {wavelength[i]}; "
+                "n² must be positive and finite"
+            )
+        return np.sqrt(n2)
+
+    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
+        return self._n(wavelength) + 1j * self.k
+
+    def _dm(self, wavelength: NDArray[np.float64]) -> NDArray[np.complex128]:
+        n = self._n(wavelength)
+        l2 = wavelength * wavelength
+        terms = len(self.B)
+        out = np.empty((2 * terms + 1, *wavelength.shape), dtype=complex)
+        for i, (b, c) in enumerate(zip(self.B, self.C, strict=True)):
+            out[i] = l2 / (l2 - c) / (2 * n)  # ∂n/∂Bᵢ = (∂n²/∂Bᵢ)/(2n)
+            out[terms + i] = b * l2 / (l2 - c) ** 2 / (2 * n)
+        out[-1] = 1j
+        return out
+
+
+@dataclass(frozen=True, eq=False)
+class Function(RefractiveIndex):
+    """m = ``f(wavelength)`` for any callable ``f`` that takes a float64 array and returns m of
+    its shape. No parameters; compares by identity."""
+
+    f: Callable[[NDArray[np.float64]], ArrayLike]
+
+    def __post_init__(self) -> None:
+        if not callable(self.f):
+            raise TypeError(f"f must be callable, not {type(self.f).__name__}")
+
+    def _m(self, wavelength: NDArray[np.float64]) -> ArrayLike:
+        return self.f(wavelength.copy())
diff --git a/tests/test_refractive.py b/tests/test_refractive.py
new file mode 100644
index 0000000..653c187
--- /dev/null
+++ b/tests/test_refractive.py
@@ -0,0 +1,175 @@
+"""scatrs.refractive models and scatrs.Sphere."""
+
+import numpy as np
+import pytest
+
+from scatrs import InvalidInputError, Sphere, refractive
+from scatrs.refractive import Cauchy, Constant, Function, Sellmeier, Table
+
+WL = np.array([[0.35, 0.5, 0.65], [0.8, 1.0, 1.6]])
+
+
+def test_constant():
+    c = Constant(1.5 + 1e-3j)
+    np.testing.assert_array_equal(c.evaluate(WL), np.full(WL.shape, 1.5 + 1e-3j))
+    assert c.evaluate(0.5).shape == ()
+    assert c.parameters == ("n", "k")
+    d = c.derivatives(WL)
+    assert d.shape == (2, *WL.shape)
+    np.testing.assert_array_equal(d[0], 1.0)
+    np.testing.assert_array_equal(d[1], 1j)
+    assert Constant(1.5) == Constant(1.5 + 0j) and hash(Constant(1.5)) == hash(Constant(1.5 + 0j))
+
+
+def test_table_interpolates_linearly_and_is_exact_at_its_nodes():
+    wl, m = [0.4, 0.6, 1.0], [1.50 + 1e-3j, 1.46 + 3e-3j, 1.44 + 1e-2j]
+    t = Table(wl, m)
+    np.testing.assert_array_equal(t.evaluate(wl), m)
+    np.testing.assert_allclose(t.evaluate(0.5), 1.48 + 2e-3j, rtol=1e-15)
+    np.testing.assert_allclose(t.evaluate(0.8), 1.45 + 6.5e-3j, rtol=1e-15)
+    assert t.parameters == () and t.derivatives(WL[:, 1:]).shape == (0, 2, 2)
+    assert t == Table(np.array(wl), tuple(m)) and hash(t) == hash(Table(np.array(wl), tuple(m)))
+
+
+def test_table_log_k_interpolates_the_logarithm_of_k():
+    t = Table([0.4, 0.8], [1.5 + 1e-6j, 1.5 + 1e-2j], interpolation="log_k")
+    np.testing.assert_allclose(t.evaluate(0.6), 1.5 + 1e-4j, rtol=1e-14)
+    np.testing.assert_allclose(t.evaluate([0.4, 0.8]), [1.5 + 1e-6j, 1.5 + 1e-2j], rtol=1e-15)
+
+
+def test_table_errors():
+    t = Table([0.4, 0.8], [1.5, 1.4])
+    with pytest.raises(InvalidInputError, match=r"wavelength 0\.3 is outside the Table's range"):
+        t.evaluate([0.5, 0.3, 0.9])
+    with pytest.raises(InvalidInputError, match="outside"):
+        t.evaluate(np.nan)
+    with pytest.raises(InvalidInputError, match="at least 2"):
+        Table([0.4], [1.5])
+    with pytest.raises(InvalidInputError, match="same length"):
+        Table([0.4, 0.5], [1.5, 1.4, 1.3])
+    with pytest.raises(InvalidInputError, match="strictly increasing"):
+        Table([0.5, 0.4], [1.5, 1.4])
+    with pytest.raises(InvalidInputError, match="finite"):
+        Table([0.4, np.inf], [1.5, 1.4])
+    with pytest.raises(InvalidInputError, match="≥ 0"):
+        Table([0.4, 0.5], [1.5, 1.4 - 1e-3j])
+    with pytest.raises(InvalidInputError, match="> 0 for log_k"):
+        Table([0.4, 0.5], [1.5, 1.4 + 1e-3j], interpolation="log_k")
+    with pytest.raises(ValueError, match="interpolation"):
+        Table([0.4, 0.5], [1.5, 1.4], interpolation="cubic")
+
+
+def test_cauchy():
+    c = Cauchy(1.45, 0.004, 1e-4, k=1e-5)
+    np.testing.assert_allclose(
+        c.evaluate(WL), 1.45 + 0.004 / WL**2 + 1e-4 / WL**4 + 1e-5j, rtol=1e-15
+    )
+    assert c.parameters == ("A", "B", "C", "k")
+    with pytest.raises(InvalidInputError, match="k must not be negative"):
+        Cauchy(1.45, k=-1e-3)
+    with pytest.raises(InvalidInputError, match="finite"):
+        Cauchy(np.nan)
+
+
+def test_sellmeier():
+    # Fused silica (Malitson 1965), λ in µm.
+    b, c = (0.6961663, 0.4079426, 0.8974794), (0.0684043**2, 0.1162414**2, 9.896161**2)
+    s = Sellmeier(b, c)
+    np.testing.assert_allclose(s.evaluate(0.5893), 1.4584, atol=1e-4)
+    l2 = WL**2
+    n = np.sqrt(1 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c, strict=True)))
+    np.testing.assert_allclose(s.evaluate(WL), n, rtol=1e-15)
+    assert s.parameters == ("B0", "B1", "B2", "C0", "C1", "C2", "k")
+    with pytest.raises(InvalidInputError, match="n² must be positive"):
+        Sellmeier([1.0], [0.25]).evaluate([0.6, 0.4])  # n² < 0 just above the pole at λ = 0.5
+    with pytest.raises(InvalidInputError, match="n² must be positive"):
+        Sellmeier([1.0], [0.25]).evaluate(0.5)  # at the pole
+    with pytest.raises(InvalidInputError, match="same length"):
+        Sellmeier([1.0, 2.0], [0.25])
+
+
+@pytest.mark.parametrize(
+    "model",
+    [
+        Constant(1.5 + 1e-3j),
+        Cauchy(1.45, 0.004, 1e-4, k=1e-5),
+        Sellmeier([0.6961663, 0.4079426], [0.0047, 0.0135], k=2e-4),
+    ],
+    ids=lambda m: type(m).__name__,
+)
+def test_derivatives_match_central_differences(model):
+    from dataclasses import replace
+
+    d = model.derivatives(WL)
+    assert d.shape == (len(model.parameters), *WL.shape)
+    for i, p in enumerate(model.parameters):
+        h = 1e-6
+
+        def at(e, p=p):
+            if isinstance(model, Constant):
+                return replace(model, m=model.m + (e if p == "n" else 1j * e))
+            if isinstance(model, Sellmeier) and p != "k":
+                seq, j = p[0], int(p[1:])
+                values = list(getattr(model, seq))
+                values[j] += e
+                return replace(model, **{seq: values})
+            return replace(model, **{p: getattr(model, p) + e})
+
+        fd = (at(h).evaluate(WL) - at(-h).evaluate(WL)) / (2 * h)
+        np.testing.assert_allclose(d[i], fd, rtol=1e-7, atol=1e-9, err_msg=p)
+
+
+def test_function_wraps_a_callable_and_checks_its_values():
+    f = Function(lambda wl: 1.33 + 1e-3 * wl * 1j)
+    np.testing.assert_allclose(f.evaluate(WL), 1.33 + 1e-3j * WL, rtol=1e-15)
+    assert f.parameters == () and f.derivatives(WL).shape == (0, *WL.shape)
+    assert f == f and f != Function(f.f)
+    with pytest.raises(InvalidInputError, match=r"m = \(1\.5-0\.001j\) at wavelength 0\.5"):
+        Function(lambda wl: np.full(wl.shape, 1.5 - 1e-3j)).evaluate([0.5, 0.6])
+    with pytest.raises(InvalidInputError, match="returned shape"):
+        Function(lambda wl: 1.5).evaluate([0.5, 0.6])
+    with pytest.raises(InvalidInputError, match="not numbers"):
+        Function(lambda wl: np.full(wl.shape, "x")).evaluate([0.5])
+    with pytest.raises(InvalidInputError, match="finite"):
+        Function(lambda wl: wl * np.nan).evaluate([0.5])
+    with pytest.raises(TypeError, match="callable"):
+        Function(1.5)
+    # The callable gets a copy: changing it changes nothing.
+    wl = np.array([0.5, 0.6])
+    Function(lambda w: w.__setitem__(..., 0.0) or np.full(w.shape, 1.5)).evaluate(wl)
+    np.testing.assert_array_equal(wl, [0.5, 0.6])
+
+
+def test_sphere_converts_its_refractive_index():
+    assert Sphere(1.5).refractive_index == Constant(1.5)
+    assert Sphere(np.complex128(1.5 + 1e-3j)).refractive_index == Constant(1.5 + 1e-3j)
+    assert Sphere(2).refractive_index == Constant(2.0)
+    t = Table([0.4, 0.5], [1.5, 1.4])
+    assert Sphere(t).refractive_index is t
+    s = Sphere(lambda wl: np.full(wl.shape, 1.5))
+    assert isinstance(s.refractive_index, Function)
+    assert Sphere(1.5) == Sphere(1.5 + 0j) and hash(Sphere(1.5)) == hash(Sphere(1.5 + 0j))
+    with pytest.raises(TypeError, match=r"refractive\.Table"):
+        Sphere(np.array([1.5, 1.4]))
+    with pytest.raises(TypeError, match=r"refractive\.Table"):
+        Sphere([1.5, 1.4])
+    with pytest.raises(TypeError, match="bool"):
+        Sphere(True)
+    with pytest.raises(TypeError, match="str"):
+        Sphere("1.5")
+    with pytest.raises(InvalidInputError, match="Im m ≥ 0"):
+        Sphere(1.5 - 1e-3j)
+
+
+def test_models_are_immutable():
+    from dataclasses import FrozenInstanceError
+
+    for model in (
+        Constant(1.5),
+        Table([0.4, 0.5], [1.5, 1.4]),
+        Cauchy(1.4),
+        Sellmeier([1.0], [0.01]),
+    ):
+        with pytest.raises(FrozenInstanceError):
+            model.k = 0.1
+    assert isinstance(Constant(1.5), refractive.RefractiveIndex)
````

### Task 2: `Ensemble`, `Mixture` and grids, with the docs

**Files:**
- Create: `python/scatrs/size.py`, `tests/test_size.py`, `tests/test_ensemble_engine_calls.py`, `docs/theory/refractive-index.md`
- Modify: `python/scatrs/ensemble.py` (rewrite), `python/scatrs/__init__.py`, `tests/test_ensemble.py` (port), `tests/test_package.py`, `benchmarks/compare.py`, `README.md`, `CHANGELOG.md`, `.gitignore`, `docs/api.md`, `docs/examples/ensemble.md`, `docs/index.md`, `docs/limitations.md`, `docs/quickstart.md`, `docs/rust-api.md`, `docs/theory/ensembles.md`, `docs/validation/sasktran2.md`

**Interfaces:**
- Consumes: task 1's `Sphere`, `RefractiveIndex.evaluate`, `scatrs.refractive` models.
- Produces:
  - `scatrs.size`: `LogNormal`, `Gamma`, `ModifiedGamma`, `PowerLaw`, `Monodisperse`, `Tabulated`, `MultiModal` (numeric fields `float | read-only NDArray[float64]`; value equality and hashing); `Distribution` alias; private `_check(d, what)`, `_grid(d) -> (shape, {name: broadcast array})`, `_specs(d) -> (shape, [engine tuple per point, C order])`; parameter names as the engine's (`"mode{j}.<name>"`, `"mode{j}.weight"`);
  - `scatrs.Ensemble(particle: Sphere, size: Distribution)` with `grid_shape` and `compute(wavelength, *, greek=None, mu=None, normalized=True, derivatives=False, rtol=1e-6, max_panels=4096, allow_unvalidated=False, refine_backscatter=False) -> EnsembleResult`;
  - `scatrs.Mixture(components: Sequence[tuple[float, Ensemble]])` with the same `compute`;
  - `EnsembleResult` adds `wavelength`, `refractive_index` (`(components,) + wavelength.shape` for a mixture), `grid: Mapping[str, NDArray]` and a `__repr__` without arrays;
  - removed: `scatrs.ensemble.compute`, `compute_mixture`, `compute_batch`, `Component`, and the distributions in `scatrs.ensemble`.

- [ ] **Step 1: Write the failing tests** — `tests/test_size.py`, `tests/test_ensemble_engine_calls.py`, the ported `tests/test_ensemble.py` and `tests/test_package.py`, as in the diff below.
- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest -q tests/test_ensemble.py tests/test_size.py tests/test_ensemble_engine_calls.py`
Expected: FAIL at import (`cannot import name 'Ensemble' from 'scatrs'`).

- [ ] **Step 3: Implement** — `python/scatrs/size.py`, the rewritten `python/scatrs/ensemble.py`, the `__init__.py` exports and `benchmarks/compare.py`, as in the diff below.
- [ ] **Step 4: Run the tests and the linters**

Run: `uv run pytest -q && uv run --locked --only-group dev ruff check . && uv run --locked --only-group dev ruff format --check python tests benchmarks`
Expected: `333 passed, 1 skipped`; ruff clean.

- [ ] **Step 5: Check the benchmark's scatrs results are bitwise unchanged** — with 0.1.0's module saved from `main` (`git show main:python/scatrs/ensemble.py > $SCRATCH/old_ensemble.py`), run in `benchmarks/` with `PYTHONPATH=$SCRATCH:.`:

```python
import numpy as np
import compare, old_ensemble
for name, w in compare.workloads(True).items():
    new = compare.run_scatrs(w, 1e-3)
    dists = [old_ensemble.LogNormal(r, s) for r, s in w["lognormals"]]
    r = old_ensemble.compute_batch(dists, w["wavelengths"], w["m"], greek=compare.ORDER, rtol=1e-3, max_panels=4096)
    old = {"c_ext": r.c_ext, "c_sca": r.c_sca, "converged": r.converged, "panels": r.panels}
    old.update({k: getattr(r.greek, k) for k in compare.GREEK})
    print(name, all(np.array_equal(old[k], new[k]) for k in old))
```

Expected: `W1 True`, `W2 True`.

- [ ] **Step 6: Update the docs** — every file under "Modify" outside `python/`, `tests/` and `benchmarks/`, and the new `docs/theory/refractive-index.md`, as in the diff below.
- [ ] **Step 7: Build the docs**

Run: `uv run --locked --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html`
Expected: `build succeeded.`

- [ ] **Step 8: Commit** — `git add -A python tests benchmarks docs README.md CHANGELOG.md .gitignore && git commit -m "feat(python)!: Ensemble and Mixture objects with parameter grids"` (body and `Co-Authored-By` as in `92845e9`).

The diff (commit `92845e9`):

````diff
diff --git a/.gitignore b/.gitignore
index 0094c38..87e2763 100644
--- a/.gitignore
+++ b/.gitignore
@@ -11,3 +11,4 @@
 .mypy_cache/
 *.egg-info/
 .DS_Store
+/docs/build/
diff --git a/CHANGELOG.md b/CHANGELOG.md
index 1a29c6a..8219a32 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -6,6 +6,37 @@ All notable changes are documented here. The format follows
 
 ## [Unreleased]
 
+### Breaking
+
+The functions of `scatrs.ensemble` are replaced by objects. Each removed call and its
+replacement:
+
+- `ensemble.compute(dist, wavelength, m, ...)` →
+  `scatrs.Ensemble(scatrs.Sphere(m), dist).compute(wavelength, ...)`.
+- `ensemble.compute_mixture([ensemble.Component(dist, m, w), ...], wavelength, ...)` →
+  `scatrs.Mixture([(w, scatrs.Ensemble(scatrs.Sphere(m), dist)), ...]).compute(wavelength, ...)`.
+- `ensemble.compute_batch([LogNormal(r1, s1), LogNormal(r2, s2), ...], wavelength, m, ...)` →
+  `scatrs.Ensemble(scatrs.Sphere(m), scatrs.LogNormal([r1, r2, ...], [s1, s2, ...])).compute(wavelength, ...)`:
+  array-valued distribution parameters broadcast into a grid of distributions.
+- `ensemble.Component` is removed; the size distributions move from `scatrs.ensemble` to
+  `scatrs.size` and the top level (`scatrs.LogNormal`, …).
+- A refractive index given as an array of values per wavelength is no longer accepted: use
+  `scatrs.refractive.Table` (or `scatrs.refractive.Function`).
+
+The keyword options, the outputs, their values and the parameter names of `derivatives` are
+unchanged.
+
+### Added
+
+- `scatrs.Sphere`, `scatrs.Ensemble` and `scatrs.Mixture`: immutable, hashable descriptions of
+  particles, ensembles and mixtures, each ensemble and mixture with a `compute` method.
+- `scatrs.refractive`: refractive-index models `Constant`, `Table` (linear, or log-linear in
+  k), `Cauchy`, `Sellmeier` and `Function`, with the names of their parameters and ∂m/∂θ.
+- Grids of distributions: any numeric distribution parameter can be an array; the grid is
+  computed at once with shared size nodes, and the results have the grid's axes first.
+- `EnsembleResult.wavelength`, `.refractive_index` (the m used) and `.grid` (the array-valued
+  parameters, broadcast), and a short `repr`.
+
 ## [0.1.0] - Unreleased
 
 The first release: a standalone library built out of the scattering core of the
diff --git a/README.md b/README.md
index 7a6c562..76ddd24 100644
--- a/README.md
+++ b/README.md
@@ -32,15 +32,19 @@ cargo add scatrs
 
 ```python
 import numpy as np
-from scatrs import ensemble, mie
+import scatrs
+from scatrs import mie
 
 # One water droplet: efficiencies, asymmetry parameter and the phase matrix.
 q = mie.efficiencies(10.0, 1.33 + 1e-8j)  # size parameter, refractive index n + ik
 f = mie.phase_matrix(10.0, 1.33 + 1e-8j, np.cos(np.radians([0, 90, 180])))
 
 # A lognormal aerosol at three wavelengths (µm), with Greek coefficients to order 32.
-sulfate = ensemble.LogNormal(median_radius=0.08, width=1.6)
-r = ensemble.compute(sulfate, [0.35, 0.55, 1.0], 1.43 + 1e-8j, greek=32)
+sulfate = scatrs.Ensemble(
+    scatrs.Sphere(refractive_index=1.43 + 1e-8j),
+    scatrs.LogNormal(median_radius=0.08, width=1.6),
+)
+r = sulfate.compute([0.35, 0.55, 1.0], greek=32)
 print(r.c_ext, r.ssa, r.g, r.greek.alpha1[:, :4])
 ```
 
diff --git a/benchmarks/compare.py b/benchmarks/compare.py
index 70a812f..412245a 100644
--- a/benchmarks/compare.py
+++ b/benchmarks/compare.py
@@ -74,12 +74,11 @@ def best_of(f, repeats: int = REPEATS):
 
 def run_scatrs(w: dict, rtol: float, max_panels: int = 4096):
     """scatrs's outputs as arrays (distribution, wavelength[, order])."""
-    from scatrs import ensemble
+    import scatrs
 
-    dists = [ensemble.LogNormal(r, s) for r, s in w["lognormals"]]
-    r = ensemble.compute_batch(
-        dists, w["wavelengths"], w["m"], greek=ORDER, rtol=rtol, max_panels=max_panels
-    )
+    radii, widths = np.array(w["lognormals"]).T
+    grid = scatrs.Ensemble(scatrs.Sphere(w["m"]), scatrs.LogNormal(radii, widths))
+    r = grid.compute(w["wavelengths"], greek=ORDER, rtol=rtol, max_panels=max_panels)
     out = {"c_ext": r.c_ext, "c_sca": r.c_sca, "converged": r.converged, "panels": r.panels}
     out.update({k: getattr(r.greek, k) for k in GREEK})
     return out
diff --git a/docs/api.md b/docs/api.md
index d2511f9..1591892 100644
--- a/docs/api.md
+++ b/docs/api.md
@@ -17,19 +17,70 @@ available from Rust ({doc}`rust-api`). Both APIs follow the {doc}`/theory/conven
 ```
 
 The package also re-exports the exceptions of {mod}`scatrs.errors` and the modules
-{mod}`scatrs.mie`, {mod}`scatrs.ensemble` and {mod}`scatrs.special`.
+{mod}`scatrs.mie`, {mod}`scatrs.ensemble`, {mod}`scatrs.refractive`, {mod}`scatrs.size` and
+{mod}`scatrs.special`. The classes below are defined in `scatrs.particles`,
+{mod}`scatrs.size` and {mod}`scatrs.ensemble`, and used from the top level.
 
-## `scatrs.mie`
+### Particles, ensembles and mixtures
 
 ```{eval-rst}
-.. automodule:: scatrs.mie
-   :members:
+.. autoclass:: scatrs.Sphere
+
+.. autoclass:: scatrs.Ensemble
+   :members: compute, grid_shape
+
+.. autoclass:: scatrs.Mixture
+   :members: compute
+```
+
+### Size distributions
+
+```{eval-rst}
+.. automodule:: scatrs.size
+   :no-members:
+
+.. autoclass:: scatrs.LogNormal
+.. autoclass:: scatrs.Gamma
+.. autoclass:: scatrs.ModifiedGamma
+.. autoclass:: scatrs.PowerLaw
+.. autoclass:: scatrs.Monodisperse
+.. autoclass:: scatrs.Tabulated
+.. autoclass:: scatrs.MultiModal
+```
+
+### Results
+
+```{eval-rst}
+.. autoclass:: scatrs.EnsembleResult
+   :members: d, d_error
+
+.. autoclass:: scatrs.EnsembleOutputs
+
+.. autoclass:: scatrs.EnsembleGreek
+
+.. autoclass:: scatrs.EnsemblePhase
+
+.. autodata:: scatrs.ensemble.SCALARS
 ```
 
 ## `scatrs.ensemble`
 
 ```{eval-rst}
 .. automodule:: scatrs.ensemble
+   :no-members:
+```
+
+## `scatrs.refractive`
+
+```{eval-rst}
+.. automodule:: scatrs.refractive
+   :members:
+```
+
+## `scatrs.mie`
+
+```{eval-rst}
+.. automodule:: scatrs.mie
    :members:
 ```
 
diff --git a/docs/examples/ensemble.md b/docs/examples/ensemble.md
index 73f4b47..abf0cdd 100644
--- a/docs/examples/ensemble.md
+++ b/docs/examples/ensemble.md
@@ -8,8 +8,8 @@ kernelspec:
 
 # Example: an aerosol size distribution
 
-{mod}`scatrs.ensemble` averages the optical properties of spheres over a size distribution, at
-many wavelengths, with an error estimate for every output and, on request, derivatives. See
+An {class}`~scatrs.Ensemble` averages the optical properties of spheres over a size
+distribution, at many wavelengths, with an error estimate for every output and, on request, derivatives. See
 {doc}`/theory/ensembles` for the method. This page runs whenever the documentation is built;
 its checks are `assert`s, so a failing check fails the build.
 
@@ -23,12 +23,12 @@ wavelengths share a unit (µm here), so the cross sections are in µm² per part
 import matplotlib.pyplot as plt
 import numpy as np
 
-from scatrs import ensemble
+import scatrs
 
-sulfate = ensemble.LogNormal(median_radius=0.08, width=1.6)
 m = 1.43 + 1e-8j
+sulfate = scatrs.Ensemble(scatrs.Sphere(m), scatrs.LogNormal(median_radius=0.08, width=1.6))
 wavelength = np.array([0.35, 0.40, 0.45, 0.50, 0.55, 0.67, 0.87, 1.0])
-r = ensemble.compute(sulfate, wavelength, m)
+r = sulfate.compute(wavelength)
 print(f"{'λ (µm)':>7}{'C_ext (µm²)':>13}{'1 - ω':>10}{'g':>8}{'panels':>8}")
 for row in zip(wavelength, r.c_ext, 1 - r.ssa, r.g, r.panels, strict=True):
     print("{:7.2f}{:13.4e}{:10.1e}{:8.4f}{:8d}".format(*row))
@@ -56,7 +56,7 @@ and more, so the larger ones dominate the optics.
 
 ```{code-cell} python
 lam = np.geomspace(0.3, 2.0, 60)
-fine_grid = ensemble.compute(sulfate, lam, m)
+fine_grid = sulfate.compute(lam)
 
 radius = np.geomspace(0.005, 1.5, 400)
 number = np.exp(-np.log(radius / 0.08) ** 2 / (2 * np.log(1.6) ** 2))  # dN/d ln r, unnormalized
@@ -102,7 +102,7 @@ its estimate is only indicative ({ref}`ensemble-accuracy`); `refine_backscatter=
 it too.
 
 ```{code-cell} python
-tight = ensemble.compute(sulfate, wavelength, m, rtol=1e-11, max_panels=50_000)
+tight = sulfate.compute(wavelength, rtol=1e-11, max_panels=50_000)
 print(f"{'output':<10}{'largest error/estimate':>24}{'largest relative error':>24}")
 for name in ("c_ext", "c_sca", "g", "ssa", "c_back", "lidar_ratio"):
     value, exact, estimate = getattr(r, name), getattr(tight, name), getattr(r.error, name)
@@ -142,7 +142,7 @@ first few at 0.55 µm. Normalized, $\alpha_1^0 = 1$ and $\alpha_1^1 = 3g$, with
 asymmetry parameters of the spheres.
 
 ```{code-cell} python
-g = ensemble.compute(sulfate, 0.55, m, greek=32)
+g = sulfate.compute(0.55, greek=32)
 names = ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
 print(f"{'l':>3}" + "".join(f"{name:>11}" for name in names))
 for l in range(6):
@@ -159,7 +159,7 @@ scattering is forward-peaked.
 ```{code-cell} python
 angles = np.linspace(0, 180, 361)
 mu = np.cos(np.radians(angles))
-pm = ensemble.compute(sulfate, [0.35, 0.55, 1.0], m, mu=mu)
+pm = sulfate.compute([0.35, 0.55, 1.0], mu=mu)
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
 for i, lam in enumerate([0.35, 0.55, 1.0]):
     ax1.semilogy(angles, pm.phase.f11[i], label=f"λ = {lam} µm")
@@ -180,8 +180,8 @@ C_\text{sca}\rangle$ itself, and its estimate is less than twice that of $\langl
 C_\text{sca}\rangle$.
 
 ```{code-cell} python
-u = ensemble.compute(sulfate, 0.55, m, greek=32, normalized=False)
-exact = ensemble.compute(sulfate, 0.55, m, greek=32, normalized=False, rtol=1e-11, max_panels=50_000)
+u = sulfate.compute(0.55, greek=32, normalized=False)
+exact = sulfate.compute(0.55, greek=32, normalized=False, rtol=1e-11, max_panels=50_000)
 print(f"alpha1[0] = {u.greek.alpha1[0]:.12e}, c_sca = {u.c_sca:.12e}")
 print(f"estimates: alpha1[0] {u.error.greek.alpha1[0]:.1e}, c_sca {u.error.c_sca:.1e}; "
       f"true error {abs(u.greek.alpha1[0] - exact.greek.alpha1[0]):.1e}")
@@ -201,9 +201,8 @@ does not swamp the difference.
 
 ```{code-cell} python
 def run(median_radius, **kwargs):
-    return ensemble.compute(
-        ensemble.LogNormal(median_radius, 1.6), 0.55, m, rtol=1e-11, max_panels=50_000, **kwargs
-    )
+    ens = scatrs.Ensemble(scatrs.Sphere(m), scatrs.LogNormal(median_radius, 1.6))
+    return ens.compute(0.55, rtol=1e-11, max_panels=50_000, **kwargs)
 
 
 d = run(0.08, derivatives=["median_radius"])
@@ -219,24 +218,21 @@ for name in ("c_ext", "g"):
 
 ## A mixture with a coarse mode
 
-{func}`~scatrs.ensemble.compute_mixture` averages over an external mixture of components, each
-a distribution, a refractive index and a weight, its relative number concentration. Here the
+A {class}`~scatrs.Mixture` averages over an external mixture of ensembles, each with a weight,
+its relative number concentration. Here the
 sulfate is mixed with one coarse, weakly absorbing particle per thousand ($r_g = 0.4$ µm,
 $\sigma_g = 1.8$, $m = 1.50 + 10^{-3}i$). Cross sections average with the number fractions
 $w_i/\sum w$; the phase matrix, and so the Greek coefficients, with $w_i\langle
 C_\text{sca}\rangle_i$.
 
 ```{code-cell} python
-coarse = ensemble.LogNormal(median_radius=0.4, width=1.8)
+coarse = scatrs.Ensemble(scatrs.Sphere(1.50 + 1e-3j), scatrs.LogNormal(median_radius=0.4, width=1.8))
 weights = (1.0, 1e-3)
-components = [
-    ensemble.Component(sulfate, m, weights[0]),
-    ensemble.Component(coarse, 1.50 + 1e-3j, weights[1]),
-]
+mixture = scatrs.Mixture([(weights[0], sulfate), (weights[1], coarse)])
 lam = np.array([0.35, 0.55, 1.0])
-mix = ensemble.compute_mixture(components, lam, greek=16, derivatives=["c1.weight"])
-fine = ensemble.compute(sulfate, lam, m, greek=16)
-big = ensemble.compute(coarse, lam, 1.50 + 1e-3j, greek=16)
+mix = mixture.compute(lam, greek=16, derivatives=["c1.weight"])
+fine = sulfate.compute(lam, greek=16)
+big = coarse.compute(lam, greek=16)
 f = np.array(weights) / sum(weights)
 print(f"{'λ (µm)':>7}{'C_ext (µm²)':>13}{'ω':>9}{'g':>8}{'coarse share of C_ext':>23}")
 for i in range(len(lam)):
@@ -257,14 +253,14 @@ Across the spectrum the coarse mode takes over the extinction as the wavelength
 
 ```{code-cell} python
 lam = np.geomspace(0.3, 2.0, 40)
-mix_l = ensemble.compute_mixture(components, lam)
-fine_l = ensemble.compute(sulfate, lam, m)
-coarse_l = ensemble.compute(coarse, lam, 1.50 + 1e-3j)
+mix_l = mixture.compute(lam)
+fine_l = sulfate.compute(lam)
+coarse_l = coarse.compute(lam)
 at_550 = dict(mu=mu)
 phases = {
-    "sulfate": ensemble.compute(sulfate, 0.55, m, **at_550).phase.f11,
-    "coarse mode": ensemble.compute(coarse, 0.55, 1.50 + 1e-3j, **at_550).phase.f11,
-    "mixture": ensemble.compute_mixture(components, 0.55, **at_550).phase.f11,
+    "sulfate": sulfate.compute(0.55, **at_550).phase.f11,
+    "coarse mode": coarse.compute(0.55, **at_550).phase.f11,
+    "mixture": mixture.compute(0.55, **at_550).phase.f11,
 }
 
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
@@ -295,3 +291,37 @@ print("d C_ext / d c1.weight:", dw, "relative to C_ext:", dw / mix.c_ext)
 assert np.all((dw / mix.c_ext > 39) & (dw / mix.c_ext < 340))
 assert np.allclose(dw, (big.c_ext - mix.c_ext) / sum(weights), rtol=1e-12, atol=0)
 ```
+
+## An optics table
+
+Distribution parameters given as arrays make a grid of distributions, as an optics table
+needs; they broadcast together like NumPy arrays, and the results have the grid's axes before
+the wavelengths'. The grid's distributions share their size nodes, so the whole table costs
+little more than its broadest distribution alone ({doc}`/theory/ensembles`). Here, sulfate
+lognormals over 12 median radii and 3 widths at four wavelengths.
+
+```{code-cell} python
+radii, widths = np.geomspace(0.02, 0.5, 12), np.array([1.4, 1.6, 1.8])
+table = scatrs.Ensemble(scatrs.Sphere(m), scatrs.LogNormal(radii[:, None], widths))
+t = table.compute([0.35, 0.55, 0.87, 1.6], greek=16)
+print(t)
+
+fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
+for j, width in enumerate(widths):
+    ax.loglog(radii, t.c_ext[:, j, 1] / (np.pi * t.grid["median_radius"][:, j] ** 2),
+              label=f"$\\sigma_g$ = {width}")
+ax.set(xlabel="median radius $r_g$ (µm)", ylabel="$\\langle C_\\mathrm{ext}\\rangle/(\\pi r_g^2)$ at 0.55 µm")
+ax.legend()
+plt.show()
+assert t.c_ext.shape == (12, 3, 4) and t.greek.alpha1.shape == (12, 3, 4, 17)
+assert t.converged.all()
+```
+
+Each point of the grid meets its own tolerance and differs from its own ensemble within the
+error estimates. A grid computes values only: derivatives come from an ensemble of one
+distribution.
+
+```{code-cell} python
+one = scatrs.Ensemble(scatrs.Sphere(m), scatrs.LogNormal(radii[5], widths[1])).compute([0.35, 0.55, 0.87, 1.6])
+assert np.all(abs(t.c_ext[5, 1] - one.c_ext) <= 2 * (t.error.c_ext[5, 1] + one.error.c_ext))
+```
diff --git a/docs/index.md b/docs/index.md
index 98212a3..ffe3868 100644
--- a/docs/index.md
+++ b/docs/index.md
@@ -11,10 +11,11 @@ library for the optical properties of particles. Install it with `pip install sc
 - **Mie scattering by homogeneous spheres** ({mod}`scatrs.mie`): Mie coefficients,
   efficiencies, asymmetry parameter, amplitudes, phase matrix and Greek coefficients to every
   order, vectorized over size parameter and refractive index.
-- **Ensembles of spheres** ({mod}`scatrs.ensemble`): averages over seven kinds of size
-  distribution and over external mixtures, at many wavelengths in parallel, with an error
-  estimate for every output. Batches of distributions share their Mie solutions, for building
-  optics databases.
+- **Ensembles of spheres** ({class}`scatrs.Ensemble`, {class}`scatrs.Mixture`): averages over
+  seven kinds of size distribution and over external mixtures, at many wavelengths in
+  parallel, with an error estimate for every output, and refractive indices that vary with
+  wavelength ({mod}`scatrs.refractive`). Grids of distributions share their Mie solutions, for
+  building optics tables.
 - **Derivatives** of every output, in closed form: with respect to the size parameter and the
   refractive index, and for ensembles, the distribution parameters and the mixture weights.
 - **Validated accuracy.** Every output has a declared accuracy, checked against independent
@@ -61,6 +62,7 @@ limitations
 :caption: Theory
 
 theory/conventions
+theory/refractive-index
 theory/mie
 theory/ensembles
 theory/special-functions
diff --git a/docs/limitations.md b/docs/limitations.md
index 9efeba7..454529a 100644
--- a/docs/limitations.md
+++ b/docs/limitations.md
@@ -99,14 +99,15 @@ describes what the accuracy figures rest on.
   moments take 0.037 s, 0.11 s and 0.20 s at $\lambda$ = 1, 0.5 and 0.3 µm; with the Greek
   coefficients to order 32, 0.093 s, 0.31 s and 0.90 s. See {doc}`theory/ensembles` and
   {doc}`validation/sasktran2`.
-- **Batches compute values only.** `compute_batch` shares the size nodes of many
-  distributions; their derivatives come from `compute`, one distribution at a time.
-- **Memory of batches.** Each distribution of a batch holds sums over every band's tables:
+- **Grids compute values only.** A grid of distributions (array-valued parameters) shares its
+  size nodes; derivatives come from an {class}`~scatrs.Ensemble` of one distribution at a time.
+  A {class}`~scatrs.Mixture` takes ensembles of one distribution each, not grids.
+- **Memory of grids.** Each distribution of a grid holds sums over every band's tables:
   about 2.5 MB per distribution and wavelength in flight at $\lambda = 0.28$ µm for the
   database grid of {doc}`validation/sasktran2` (2 GB for 800 distributions on one thread; more
-  wavelengths run at once on more threads). Split very large batches.
+  wavelengths run at once on more threads). Split very large grids.
 - **Tables can exceed `max_panels`.** Every radius of a `Tabulated` distribution is a panel
-  edge, so a table of more radii than `max_panels`, or a batch of many tables, takes more
+  edge, so a table of more radii than `max_panels`, or a grid with many tables, takes more
   panels than `max_panels` (still converging).
 - **Distributions over size only.** The particles are spheres, so the distributions are over
   their radius only.
diff --git a/docs/quickstart.md b/docs/quickstart.md
index ef752c9..de392cb 100644
--- a/docs/quickstart.md
+++ b/docs/quickstart.md
@@ -9,7 +9,8 @@ kernelspec:
 # Quick start
 
 scatrs computes the optical properties of spherical particles: single spheres with
-{mod}`scatrs.mie`, and size distributions and mixtures of them with {mod}`scatrs.ensemble`.
+{mod}`scatrs.mie`, and size distributions and mixtures of them with {class}`~scatrs.Ensemble` and
+{class}`~scatrs.Mixture`.
 This page shows the main calls and the conventions to know before using the results. See
 {doc}`installation` to install the package.
 
@@ -21,8 +22,8 @@ in {doc}`/theory/conventions`.
 | Quantity | Convention |
 |---|---|
 | Refractive index | $m = n + ik$, with $k \ge 0$ for absorbing particles (Bohren & Huffman). Codes that write $m = n - ik$, such as MIEV0 and sasktran2, need the complex conjugate. A negative imaginary part raises {class}`~scatrs.errors.InvalidInputError`. |
-| Size | {mod}`scatrs.mie` takes the size parameter $x = 2\pi r/\lambda$. {mod}`scatrs.ensemble` takes radii and wavelengths in any unit, the same unit for both. $\lambda$ and $m$ are relative to the surrounding medium. |
-| Cross sections | {mod}`scatrs.mie` returns efficiencies $Q = C/(\pi r^2)$. {mod}`scatrs.ensemble` returns cross sections per particle, in the square of the length unit. |
+| Size | {mod}`scatrs.mie` takes the size parameter $x = 2\pi r/\lambda$. Ensembles take radii and wavelengths in any unit, the same unit for both. $\lambda$ and $m$ are relative to the surrounding medium. |
+| Cross sections | {mod}`scatrs.mie` returns efficiencies $Q = C/(\pi r^2)$. Ensembles return cross sections per particle, in the square of the length unit. |
 | Angles | Functions of angle take $\mu = \cos\Theta$, with $\Theta$ the scattering angle. |
 | Phase matrix | Normalized so that $\tfrac12\int_{-1}^{1} F_{11}\,d\mu = 1$. The degree of linear polarization is $-F_{12}/F_{11}$. |
 | Greek coefficients | The expansion of Mishchenko, Travis & Lacis (2002), normalized so that $\alpha_1^0 = 1$, and then $\alpha_1^1 = 3g$. $\beta_1$ and $\beta_2$ have the opposite sign to sasktran2's. |
@@ -86,16 +87,20 @@ assert g.alpha1[0] == 1 and abs(g.alpha1[1] - 3 * q.g) < 1e-12
 
 ## A size distribution
 
-{func}`scatrs.ensemble.compute` averages over a size distribution at each wavelength. Every
-output comes with an error estimate, and `converged` says whether the size integration met
-its tolerance (`rtol`, by default $10^{-6}$).
+An {class}`~scatrs.Ensemble` is a particle and a size distribution;
+{meth}`~scatrs.Ensemble.compute` averages over the sizes at each wavelength. Every output comes
+with an error estimate, and `converged` says whether the size integration met its tolerance
+(`rtol`, by default $10^{-6}$).
 
 ```{code-cell} python
-from scatrs import ensemble
+import scatrs
 
-sulfate = ensemble.LogNormal(median_radius=0.08, width=1.6)  # µm
+sulfate = scatrs.Ensemble(
+    scatrs.Sphere(refractive_index=1.43 + 1e-8j),
+    scatrs.LogNormal(median_radius=0.08, width=1.6),  # µm
+)
 wavelengths = np.array([0.35, 0.55, 1.0])  # µm
-r = ensemble.compute(sulfate, wavelengths, 1.43 + 1e-8j, greek=32)
+r = sulfate.compute(wavelengths, greek=32)
 
 for i, lam in enumerate(wavelengths):
     print(f"λ = {lam} µm: C_ext = {r.c_ext[i]:.4e} µm² (± {r.error.c_ext[i]:.0e}), "
@@ -104,9 +109,20 @@ print("converged:", r.converged)
 print("alpha1 at 0.55 µm:", r.greek.alpha1[1, :4])
 ```
 
-Mixtures of several components, each a distribution with its own refractive index, use
-{func}`~scatrs.ensemble.compute_mixture`; many distributions of one material (an optics
-database) use {func}`~scatrs.ensemble.compute_batch`.
+A refractive index that varies with wavelength is a model from {mod}`scatrs.refractive`, such
+as a table of measured values ({doc}`theory/refractive-index`). A {class}`~scatrs.Mixture` combines
+ensembles of different materials. Distribution parameters given as arrays make a grid of
+distributions, computed at once, as an optics table needs:
+
+```{code-cell} python
+table = scatrs.Ensemble(
+    scatrs.Sphere(1.43 + 1e-8j),
+    scatrs.LogNormal(median_radius=np.geomspace(0.05, 0.5, 4)[:, None], width=[1.4, 1.6, 1.8]),
+)
+t = table.compute(wavelengths)
+print(t)
+print("C_ext of r_g = 0.05 µm, σ_g = 1.6:", t.c_ext[0, 1])
+```
 
 ## Derivatives
 
@@ -119,7 +135,7 @@ q = mie.efficiencies(x, m, derivatives=True)
 print("dQ_ext/dx       =", q.d("x").q_ext)
 print("dQ_ext/d(Re m)  =", q.d("m_real").q_ext)
 
-r = ensemble.compute(sulfate, 0.55, 1.43 + 1e-8j, derivatives=["median_radius"])
+r = sulfate.compute(0.55, derivatives=["median_radius"])
 print("dC_ext/dr_g =", r.d("median_radius").c_ext, "µm")
 ```
 
diff --git a/docs/rust-api.md b/docs/rust-api.md
index 58a5124..6353731 100644
--- a/docs/rust-api.md
+++ b/docs/rust-api.md
@@ -13,7 +13,7 @@ Add it with `cargo add scatrs`. The same reference is on [docs.rs](https://docs.
 | Rust | Python | Contents |
 |---|---|---|
 | {rust}`scatrs::mie` | {mod}`scatrs.mie` | One sphere: coefficients, efficiencies, amplitudes, phase matrix, Greek coefficients and their derivatives. |
-| {rust}`scatrs::ensemble` | {mod}`scatrs.ensemble` | Averages over size distributions, mixtures and batches, with error estimates and derivatives. |
+| {rust}`scatrs::ensemble` | {class}`scatrs.Ensemble`, {class}`scatrs.Mixture` | Averages over size distributions, mixtures and batches (Python: grids), with error estimates and derivatives. |
 | {rust}`scatrs::special` | {mod}`scatrs.special` | Gauss–Legendre quadrature and Wigner d-functions. |
 | {rust}`scatrs::scalar` | | The {rust}`Real <scatrs::scalar::Real>` trait and the double-double and dual-number types. Rust only. |
 | {rust}`scatrs::error` | {mod}`scatrs.errors` | The error type, and the exceptions that mirror it. |
diff --git a/docs/theory/ensembles.md b/docs/theory/ensembles.md
index 2048795..d59bbea 100644
--- a/docs/theory/ensembles.md
+++ b/docs/theory/ensembles.md
@@ -1,6 +1,7 @@
 # Ensembles of spheres
 
-This page describes the methods behind {rust}`scatrs::ensemble` (Python: {mod}`scatrs.ensemble`): the
+This page describes the methods behind {rust}`scatrs::ensemble` (Python: {class}`scatrs.Ensemble`
+and {class}`scatrs.Mixture`): the
 size distributions, their cut-offs, the adaptive size integration, the error estimates, the
 Jacobians and the mixtures. The rustdoc of {rust}`scatrs::ensemble` is the authoritative statement of
 the interface; the single-sphere quantities come from {doc}`mie`.
@@ -304,9 +305,10 @@ $\mathrm{Re}\,m$ of a mixture's second component, a mode weight and a mode's wid
 
 ## Mixtures
 
-`compute_mixture` averages over external mixtures: components, each a distribution, its
-refractive indices and a weight $w_i \ge 0$ (a relative number concentration; their sum finite
-and positive). `compute` is a mixture of one component, without the weight parameter. Cross
+`compute_mixture` (Python {class}`scatrs.Mixture`) averages over external mixtures: components,
+each a distribution, its refractive indices and a weight $w_i \ge 0$ (a relative number
+concentration; their sum finite and positive). `compute` is a mixture of one component, without
+the weight parameter. Cross
 sections and moments are averaged with the number fractions $w_i/\sum w$. The phase-matrix
 tables, in $C_\text{sca}$ units, are averaged with the same fractions, so the phase matrices
 are averaged with $w_i\langle C_\text{sca}\rangle_i$, before one projection of all components.
@@ -314,9 +316,11 @@ A multimodal distribution is a mixture of its modes with one refractive index.
 
 ## Batches
 
-`compute_batch` (Python {func}`scatrs.ensemble.compute_batch`) averages over each of many
-distributions of one material, as an optics database needs: a grid of median radii and widths,
-say. The distributions share their size nodes: one set of panels per wavelength covers the
+`compute_batch` averages over each of many distributions of one material, as an optics
+database needs: a grid of median radii and widths, say. In Python, an {class}`scatrs.Ensemble`
+whose distribution has array-valued parameters is such a batch: its parameters broadcast
+together into a grid, flattened in C order, and the results have the grid's axes before the
+wavelengths'. The distributions share their size nodes: one set of panels per wavelength covers the
 union of their Mie ranges, with the ends of finite supports and `Tabulated` radii as panel
 edges, and is refined until every distribution meets its own targets of pass 1 (a panel splits
 if it exceeds its share of any distribution's target) and extended until every distribution's
diff --git a/docs/theory/refractive-index.md b/docs/theory/refractive-index.md
new file mode 100644
index 0000000..2969568
--- /dev/null
+++ b/docs/theory/refractive-index.md
@@ -0,0 +1,81 @@
+---
+file_format: mystnb
+kernelspec:
+  name: python3
+  display_name: Python 3
+  language: python
+---
+
+# Refractive-index models
+
+A {class}`~scatrs.Sphere` takes its refractive index as a model of the wavelength, from
+{mod}`scatrs.refractive`. {meth}`Ensemble.compute <scatrs.Ensemble.compute>` evaluates the
+model once at the requested wavelengths; the values used are in the result's
+`refractive_index`. scatrs has no database of materials: the data or coefficients come from
+the user.
+
+## Conventions
+
+- $m = n + ik$ relative to the surrounding medium, with $k \ge 0$ for absorbing particles
+  ({doc}`conventions`). Codes and tables that write $m = n - ik$ need the complex conjugate.
+- Wavelengths are in the unit of the radii, and the coefficients of {class}`~scatrs.refractive.Cauchy`
+  and {class}`~scatrs.refractive.Sellmeier` are in that unit too. Published coefficients
+  usually assume micrometres: with radii in nanometres, convert them (Cauchy's $B$ scales by
+  $10^6$ and $C$ by $10^{12}$; Sellmeier's $C_i$ by $10^6$).
+- There is no extrapolation. A model evaluated where it has no valid value (a wavelength
+  outside a table, a Sellmeier law with $n^2 \le 0$, a callable returning $k < 0$ or a value
+  that is not finite) raises {class}`~scatrs.errors.InvalidInputError` naming the wavelength.
+
+## Models
+
+| Model | $m(\lambda)$ | Parameters |
+|---|---|---|
+| `Constant(m)` | $m$ | `n`, `k` |
+| `Table(wavelength, m, interpolation="linear")` | $n$ and $k$ linear in $\lambda$ between the nodes; `"log_k"` interpolates $\ln k$ instead | none |
+| `Cauchy(A, B=0, C=0, k=0)` | $n = A + B/\lambda^2 + C/\lambda^4$, constant $k$ | `A`, `B`, `C`, `k` |
+| `Sellmeier(B, C, k=0)` | $n^2 = 1 + \sum_i B_i\lambda^2/(\lambda^2 - C_i)$, constant $k$ | `B0`, …, `C0`, …, `k` |
+| `Function(f)` | `f(wavelength)` | none |
+
+A number given to {class}`~scatrs.Sphere` becomes a `Constant`, and any other callable a
+`Function`. An array is not accepted: values of $m$ mean something only together with their
+wavelengths, which a `Table` holds.
+
+A `Table` needs at least two nodes, at strictly increasing wavelengths. Linear interpolation
+suits smooth $n$ and $k$; absorption that changes by decades between nodes is better
+interpolated in $\ln k$ (`"log_k"`, which needs $k > 0$ at every node). The table is data: it
+has no parameters.
+
+Each model lists its parameters in `parameters`, and `derivatives(wavelength)` gives
+$\partial m/\partial\theta$ for each. The derivatives of an ensemble with respect to
+`"m_real"` and `"m_imag"` are those with respect to $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$ at
+each wavelength, whatever model produced $m$.
+
+## Example
+
+Fused silica from its Sellmeier coefficients (Malitson, 1965, $\lambda$ in µm), and an
+absorbing material from a short table, interpolated in $\ln k$:
+
+```{code-cell} python
+import numpy as np
+
+import scatrs
+from scatrs import refractive
+
+silica = refractive.Sellmeier(
+    B=[0.6961663, 0.4079426, 0.8974794],
+    C=[0.0684043**2, 0.1162414**2, 9.896161**2],
+)
+dust = refractive.Table(
+    [0.3, 0.5, 1.0, 2.0],
+    [1.55 + 5e-3j, 1.53 + 1.5e-3j, 1.52 + 8e-4j, 1.50 + 2e-3j],
+    interpolation="log_k",
+)
+wavelength = np.array([0.35, 0.55, 1.6])
+print("silica:", silica.evaluate(wavelength).real)
+print("dust:  ", dust.evaluate(wavelength))
+assert abs(silica.evaluate(0.5893) - 1.4584) < 1e-4  # the sodium D line
+
+r = scatrs.Ensemble(scatrs.Sphere(dust), scatrs.LogNormal(0.5, 1.8)).compute(wavelength)
+print("C_ext:", r.c_ext, "µm²; m used:", r.refractive_index)
+assert np.array_equal(r.refractive_index, dust.evaluate(wavelength))
+```
diff --git a/docs/validation/sasktran2.md b/docs/validation/sasktran2.md
index c7f908c..9a071da 100644
--- a/docs/validation/sasktran2.md
+++ b/docs/validation/sasktran2.md
@@ -3,7 +3,7 @@
 scatrs is built out of the scattering core of the radiative transfer model
 [sasktran2](https://github.com/usask-arg/sasktran2), whose own Mie code, through its
 `MieDatabase`, integrates the optical properties of spheres over size distributions. It is a
-separate implementation of the same averages as {mod}`scatrs.ensemble`, so it serves both as
+separate implementation of the same averages as {class}`scatrs.Ensemble`, so it serves both as
 an external check of scatrs's results and as a benchmark of its speed. This page compares the two for accuracy and speed on the same machine.
 The script `benchmarks/compare.py` produces every number here, and the results it wrote are in
 `benchmarks/results/2026-10-13-arm64.json`.
@@ -17,7 +17,8 @@ The script `benchmarks/compare.py` produces every number here, and the results i
   coefficients, without the database's netCDF caching. It builds one radius grid, from QUADPACK's
   panels for an extinction integrand at the shortest wavelength, for every wavelength and
   distribution, and projects the phase matrix on a split Gauss–Legendre grid of 128 cosines.
-- **scatrs** `ensemble.compute_batch` with the Greek coefficients to order 63 (the same 64
+- **scatrs** {class}`~scatrs.Ensemble` with the 50 lognormals as one grid (array-valued
+  parameters), with the Greek coefficients to order 63 (the same 64
   coefficients) at rtol $3\cdot10^{-3}$, $10^{-3}$, $3\cdot10^{-4}$ and $10^{-4}$.
 - **Reference:** scatrs at rtol $10^{-9}$ (converged everywhere).
 - **Conventions:** sasktran2 takes $m = n - ik$, and works in nm and m²; its $\beta_1$ and
@@ -92,7 +93,7 @@ results come without estimates.
 - **What the time buys.** sasktran2's grid is fixed by its settings; scatrs's panels follow the
   requested rtol, so the tables also give its accuracy-against-time curve. scatrs's default rtol,
   $10^{-6}$, is far tighter than these: ask for the accuracy the application needs.
-- **Batches.** W2 uses `compute_batch`, which shares the size nodes of the 50 distributions;
+- **Grids.** W2 computes the 50 distributions as one grid, which shares their size nodes;
   computing them one by one cost about 2.5 times as much on this grid (measured separately, one
   thread, rtol $10^{-3}$).
 - **Reproduce:** `uv run --locked --directory benchmarks python compare.py` (several minutes
diff --git a/python/scatrs/__init__.py b/python/scatrs/__init__.py
index 9bd9cf7..8635be3 100644
--- a/python/scatrs/__init__.py
+++ b/python/scatrs/__init__.py
@@ -3,7 +3,15 @@
 from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
 from importlib.metadata import version as _version
 
-from scatrs import _scatrs, ensemble, mie, particles, refractive, special
+from scatrs import _scatrs, ensemble, mie, particles, refractive, size, special
+from scatrs.ensemble import (
+    Ensemble,
+    EnsembleGreek,
+    EnsembleOutputs,
+    EnsemblePhase,
+    EnsembleResult,
+    Mixture,
+)
 from scatrs.errors import (
     InvalidInputError,
     NotConvergedError,
@@ -11,18 +19,41 @@ from scatrs.errors import (
     ScatrsError,
 )
 from scatrs.particles import Sphere
+from scatrs.size import (
+    Gamma,
+    LogNormal,
+    ModifiedGamma,
+    Monodisperse,
+    MultiModal,
+    PowerLaw,
+    Tabulated,
+)
 
 __all__ = [
+    "Ensemble",
+    "EnsembleGreek",
+    "EnsembleOutputs",
+    "EnsemblePhase",
+    "EnsembleResult",
+    "Gamma",
     "InvalidInputError",
+    "LogNormal",
+    "Mixture",
+    "ModifiedGamma",
+    "Monodisperse",
+    "MultiModal",
     "NotConvergedError",
     "OutsideValidatedDomainError",
+    "PowerLaw",
     "ScatrsError",
     "Sphere",
+    "Tabulated",
     "__version__",
     "ensemble",
     "mie",
     "particles",
     "refractive",
+    "size",
     "special",
 ]
 
diff --git a/python/scatrs/ensemble.py b/python/scatrs/ensemble.py
index 4b8da7f..0269fe6 100644
--- a/python/scatrs/ensemble.py
+++ b/python/scatrs/ensemble.py
@@ -1,16 +1,20 @@
-"""Ensembles of spheres: size distributions, adaptive size integration and mixtures.
+"""Ensembles of particles: size distributions, adaptive size integration and mixtures.
 
-:func:`compute` averages the optical properties of spheres over a size distribution at each
-wavelength; :func:`compute_mixture` averages over several :class:`Component` s. Radii and
-wavelengths are in the same (any) unit; cross sections are per particle, averaged over the
-number distribution, in that unit squared. See :doc:`/theory/ensembles` for the method and
-the error estimates.
+An :class:`Ensemble` is a particle (:class:`~scatrs.Sphere`) and a size distribution
+(:mod:`scatrs.size`); a :class:`Mixture` is several ensembles with relative number
+concentrations. Their ``compute`` methods average the optical properties over the sizes at
+each wavelength. Radii and wavelengths are in the same (any) unit; cross sections are per
+particle, averaged over the number distribution, in that unit squared. See
+:doc:`/theory/ensembles` for the method and the error estimates.
 
 The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the cross
 sections, g and the distribution moments, and optionally the backscattering, meet ``rtol``).
-Every output comes with an error estimate (:attr:`EnsembleResult.error`) and, with
+Every output comes with an error estimate (:attr:`~scatrs.EnsembleResult.error`) and, with
 ``derivatives``, its derivatives with respect to the distribution parameters, ``"m_real"``,
-``"m_imag"`` and the mixture weights (:meth:`EnsembleResult.d`).
+``"m_imag"`` and the mixture weights (:meth:`~scatrs.EnsembleResult.d`).
+
+A distribution whose parameters are arrays is a grid of distributions, computed at once with
+shared size nodes (as an optics table needs): the results have the grid's axes first.
 
 Each call runs in Rust with the GIL released, in parallel over wavelengths and size panels;
 results are bitwise the same for any number of threads. Results are frozen dataclasses of
@@ -19,32 +23,26 @@ read-only arrays.
 
 from __future__ import annotations
 
-from collections.abc import Iterable, Sequence
+from collections.abc import Iterable, Mapping, Sequence
 from dataclasses import dataclass, field
-from typing import Literal, SupportsIndex, TypeAlias
+from types import MappingProxyType
+from typing import Literal, SupportsIndex
 
 import numpy as np
 from numpy.typing import ArrayLike, NDArray
 
 from scatrs import _args, _scatrs
+from scatrs.particles import Sphere
+from scatrs.size import Distribution, _check, _grid, _specs
 
 __all__ = [
     "SCALARS",
-    "Component",
+    "Ensemble",
     "EnsembleGreek",
     "EnsembleOutputs",
     "EnsemblePhase",
     "EnsembleResult",
-    "Gamma",
-    "LogNormal",
-    "ModifiedGamma",
-    "Monodisperse",
-    "MultiModal",
-    "PowerLaw",
-    "Tabulated",
-    "compute",
-    "compute_batch",
-    "compute_mixture",
+    "Mixture",
 ]
 
 SCALARS = (
@@ -63,130 +61,6 @@ SCALARS = (
 """The scalar outputs of :class:`EnsembleOutputs`, in order."""
 
 
-@dataclass(frozen=True)
-class LogNormal:
-    """n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: ``median_radius`` r_g > 0, ``width``
-    σ_g > 1."""
-
-    median_radius: float
-    width: float
-
-    def _spec(self) -> tuple:
-        return ("lognormal", float(self.median_radius), float(self.width))
-
-
-@dataclass(frozen=True)
-class Gamma:
-    """Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
-    ``effective_radius`` a > 0, ``effective_variance`` 0 < b < 1/2."""
-
-    effective_radius: float
-    effective_variance: float
-
-    def _spec(self) -> tuple:
-        return ("gamma", float(self.effective_radius), float(self.effective_variance))
-
-
-@dataclass(frozen=True)
-class ModifiedGamma:
-    """Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): ``alpha`` > −1,
-    ``b`` > 0, ``gamma`` > 0."""
-
-    alpha: float
-    b: float
-    gamma: float
-
-    def _spec(self) -> tuple:
-        return ("modified_gamma", float(self.alpha), float(self.b), float(self.gamma))
-
-
-@dataclass(frozen=True)
-class PowerLaw:
-    """n(r) ∝ r^(−α) on [``min_radius``, ``max_radius``]: ``exponent`` α (the limits are not
-    differentiable)."""
-
-    exponent: float
-    min_radius: float
-    max_radius: float
-
-    def _spec(self) -> tuple:
-        return (
-            "power_law",
-            float(self.exponent),
-            float(self.min_radius),
-            float(self.max_radius),
-        )
-
-
-@dataclass(frozen=True)
-class Monodisperse:
-    """Spheres of one ``radius``."""
-
-    radius: float
-
-    def _spec(self) -> tuple:
-        return ("monodisperse", float(self.radius))
-
-
-@dataclass(frozen=True)
-class Tabulated:
-    """n(r) linear between the points (``radius[i]``, ``density[i]``) and 0 outside: radii
-    increasing, densities ≥ 0. No parameters."""
-
-    radius: Sequence[float]
-    density: Sequence[float]
-
-    def __post_init__(self) -> None:
-        # Tuples of floats, so that the distribution compares and hashes by value.
-        for name in ("radius", "density"):
-            values = np.asarray(getattr(self, name), dtype=float).ravel()
-            object.__setattr__(self, name, tuple(float(v) for v in values))
-
-    def _spec(self) -> tuple:
-        return ("tabulated", list(self.radius), list(self.density))
-
-
-@dataclass(frozen=True)
-class MultiModal:
-    """Σⱼ wⱼ nⱼ(r), each mode normalized: ``modes`` is a sequence of (weight wⱼ > 0, mode).
-    The modes' parameters are named ``"mode{j}.<name>"``, their weights ``"mode{j}.weight"``.
-    """
-
-    modes: Sequence[tuple[float, Distribution]]
-
-    def __post_init__(self) -> None:
-        modes = tuple((float(w), _check(d, f"mode {j}")) for j, (w, d) in enumerate(self.modes))
-        object.__setattr__(self, "modes", modes)
-
-    def _spec(self) -> tuple:
-        return ("multimodal", [(w, d._spec()) for w, d in self.modes])
-
-
-Distribution: TypeAlias = (
-    LogNormal | Gamma | ModifiedGamma | PowerLaw | Monodisperse | Tabulated | MultiModal
-)
-
-_DISTRIBUTIONS = (LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal)
-
-
-def _check(d: object, what: str) -> Distribution:
-    if not isinstance(d, _DISTRIBUTIONS):
-        names = ", ".join(c.__name__ for c in _DISTRIBUTIONS)
-        raise TypeError(f"{what} must be one of {names}, not {type(d).__name__}")
-    return d
-
-
-@dataclass(frozen=True, eq=False)
-class Component:
-    """One component of an external mixture: a size distribution, its refractive index
-    (broadcast to the wavelengths' shape), and its relative number concentration ``weight``
-    ≥ 0. Components compare by identity."""
-
-    distribution: Distribution
-    refractive_index: ArrayLike
-    weight: float = 1.0
-
-
 def _frozen(a: np.ndarray) -> np.ndarray:
     """A read-only view of ``a``: an array that owns its data can be made writeable again, a
     view of a read-only array cannot."""
@@ -196,8 +70,9 @@ def _frozen(a: np.ndarray) -> np.ndarray:
 
 @dataclass(frozen=True, eq=False)
 class EnsembleGreek:
-    """Greek coefficients per wavelength, arrays of shape ``wavelength.shape + (L + 1,)``,
-    zero-padded to the largest order (see :class:`scatrs.mie.MieGreek` for the expansion)."""
+    """Greek coefficients per wavelength, arrays of shape ``wavelength.shape + (L + 1,)``
+    (after the grid's axes), zero-padded to the largest order (see
+    :class:`scatrs.mie.MieGreek` for the expansion)."""
 
     alpha1: NDArray[np.float64]
     alpha2: NDArray[np.float64]
@@ -210,7 +85,7 @@ class EnsembleGreek:
 @dataclass(frozen=True, eq=False)
 class EnsemblePhase:
     """The phase matrix per wavelength at the cosines ``mu``, arrays of shape
-    ``wavelength.shape + mu.shape``."""
+    ``wavelength.shape + mu.shape`` (after the grid's axes)."""
 
     f11: NDArray[np.float64]
     f12: NDArray[np.float64]
@@ -221,7 +96,7 @@ class EnsemblePhase:
 @dataclass(frozen=True, eq=False)
 class EnsembleOutputs:
     """The outputs of an ensemble (or their error estimates, or their derivatives with
-    respect to one parameter), each an array of the wavelengths' shape.
+    respect to one parameter), each an array of shape ``grid + wavelength.shape``.
 
     Attributes
     ----------
@@ -255,7 +130,7 @@ class EnsembleOutputs:
     phase: EnsemblePhase | None = None
 
 
-@dataclass(frozen=True, eq=False)
+@dataclass(frozen=True, eq=False, repr=False)
 class EnsembleResult(EnsembleOutputs):
     """:class:`EnsembleOutputs` with their error estimates and derivatives.
 
@@ -264,20 +139,31 @@ class EnsembleResult(EnsembleOutputs):
     error : EnsembleOutputs
         Non-negative error estimates of every output (see :doc:`/validation/accuracy`).
     converged : numpy.ndarray
-        bool per wavelength: whether every refined quantity met ``rtol`` within
-        ``max_panels`` panels.
+        bool per point: whether every refined quantity met ``rtol`` within ``max_panels``
+        panels.
     panels : numpy.ndarray
-        int64 per wavelength: the number of size panels.
+        int64 per point: the number of size panels (shared by the points of a grid).
     params : tuple of str
         The parameters :meth:`d` accepts.
+    wavelength : numpy.ndarray
+        The wavelengths, as given.
+    refractive_index : numpy.ndarray
+        m at the wavelengths: of the wavelengths' shape, or ``(components,) +
+        wavelength.shape`` for a :class:`Mixture`.
+    grid : Mapping[str, numpy.ndarray]
+        Each array-valued distribution parameter, by name (as in :attr:`params`), broadcast
+        to the grid's shape; empty if there is no grid.
     """
 
     error: EnsembleOutputs = field(kw_only=True)
     converged: NDArray[np.bool_] = field(kw_only=True)
     panels: NDArray[np.int64] = field(kw_only=True)
+    wavelength: NDArray[np.float64] = field(kw_only=True)
+    refractive_index: NDArray[np.complex128] = field(kw_only=True)
+    grid: Mapping[str, NDArray[np.float64]] = field(kw_only=True)
     params: tuple[str, ...] = field(default=(), kw_only=True)
-    _d: tuple[EnsembleOutputs, ...] = field(default=(), repr=False, kw_only=True)
-    _d_errors: tuple[EnsembleOutputs, ...] = field(default=(), repr=False, kw_only=True)
+    _d: tuple[EnsembleOutputs, ...] = field(default=(), kw_only=True)
+    _d_errors: tuple[EnsembleOutputs, ...] = field(default=(), kw_only=True)
 
     def _index(self, param: str) -> int:
         if param not in self.params:
@@ -304,6 +190,18 @@ class EnsembleResult(EnsembleOutputs):
         """
         return self._d_errors[self._index(param)]
 
+    def __repr__(self) -> str:
+        parts = [f"shape={self.c_ext.shape}"]
+        if self.greek is not None:
+            parts.append(f"greek={self.greek.alpha1.shape}")
+        if self.phase is not None:
+            parts.append(f"phase={self.phase.f11.shape}")
+        parts.append(f"params={self.params}")
+        if self.grid:
+            parts.append(f"grid={tuple(self.grid)}")
+        parts.append(f"converged={int(self.converged.sum())}/{self.converged.size}")
+        return f"EnsembleResult({', '.join(parts)})"
+
 
 def _outputs(
     raw: dict, which: str, k: int | None, shape: tuple[int, ...], mu_shape
@@ -326,29 +224,6 @@ def _outputs(
     return EnsembleOutputs(*scalars, greek=greek, phase=phase)
 
 
-def _refractive_index(refractive_index: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
-    """The refractive index broadcast to the wavelengths' ``shape``, flattened."""
-    m = _args.complex_array(refractive_index, "refractive_index")
-    try:
-        m = np.broadcast_to(m, shape)
-    except ValueError:
-        raise ValueError(
-            f"refractive_index of shape {m.shape} does not broadcast to the wavelengths' "
-            f"shape {shape}"
-        ) from None
-    return np.ascontiguousarray(m.reshape(-1))
-
-
-def _specs(components: Sequence[Component], shape: tuple[int, ...]) -> list:
-    out = []
-    for i, c in enumerate(components):
-        if not isinstance(c, Component):
-            raise TypeError(f"component {i} must be a Component, not {type(c).__name__}")
-        spec = _check(c.distribution, f"the distribution of component {i}")._spec()
-        out.append((spec, _refractive_index(c.refractive_index, shape), float(c.weight)))
-    return out
-
-
 def _greek_order(greek: SupportsIndex | Literal["all"] | None) -> tuple[int | None, bool]:
     """The order and whether every order is wanted."""
     if greek is None:
@@ -360,74 +235,131 @@ def _greek_order(greek: SupportsIndex | Literal["all"] | None) -> tuple[int | No
     return _args.size(greek, "greek"), False
 
 
-def _run(
-    items: list,
-    wavelength: ArrayLike,
-    mixture: bool,
-    greek: SupportsIndex | Literal["all"] | None,
-    mu: ArrayLike | None,
-    normalized: bool,
-    derivatives: bool | Sequence[str],
-    rtol: float,
-    max_panels: SupportsIndex,
-    allow_unvalidated: bool,
-    refine_backscatter: bool,
-) -> EnsembleResult:
-    wl = _args.real_array(wavelength, "wavelength")
-    shape = wl.shape
-    names = tuple(_scatrs.ensemble_parameters(items, mixture))
+def _parameters(names: tuple[str, ...], derivatives: bool | Sequence[str]) -> tuple[str, ...]:
+    """The requested parameters, checked against ``names``."""
     if isinstance(derivatives, (bool, np.bool_)):
-        params = names if derivatives else ()
-    elif isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
+        return names if derivatives else ()
+    if isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
         raise TypeError(
             "derivatives must be a bool or a sequence of parameter names, "
             f"not {type(derivatives).__name__}"
         )
-    else:
-        params = tuple(derivatives)
-        for p in params:
-            if p not in names:
-                raise ValueError(f"unknown parameter {p!r}; expected one of {names}")
-        if len(set(params)) != len(params):
-            raise ValueError(f"a parameter is requested twice: {params}")
-    mus = None if mu is None else _args.real_array(mu, "mu")
-    order, all_orders = _greek_order(greek)
-    raw = _scatrs.ensemble_compute(
-        items,
-        np.ascontiguousarray(wl.reshape(-1)),
-        mixture,
-        order,
-        all_orders,
-        np.ascontiguousarray(np.zeros(0) if mus is None else mus.reshape(-1)),
-        _args.flag(normalized, "normalized"),
-        [names.index(p) for p in params],
-        float(rtol),
-        _args.size(max_panels, "max_panels"),
-        _args.flag(allow_unvalidated, "allow_unvalidated"),
-        _args.flag(refine_backscatter, "refine_backscatter"),
-    )
+    params = tuple(derivatives)
+    for p in params:
+        if p not in names:
+            raise ValueError(f"unknown parameter {p!r}; expected one of {names}")
+    if len(set(params)) != len(params):
+        raise ValueError(f"a parameter is requested twice: {params}")
+    return params
+
+
+@dataclass(frozen=True, eq=False)
+class _Options:
+    """The options shared by every ``compute``, converted for the engine."""
+
+    order: int | None
+    all_orders: bool
+    mu: NDArray[np.float64] | None
+    normalized: bool
+    rtol: float
+    max_panels: int
+    allow_unvalidated: bool
+    refine_backscatter: bool
+
+    @classmethod
+    def of(
+        cls,
+        greek: SupportsIndex | Literal["all"] | None,
+        mu: ArrayLike | None,
+        normalized: bool,
+        rtol: float,
+        max_panels: SupportsIndex,
+        allow_unvalidated: bool,
+        refine_backscatter: bool,
+    ) -> _Options:
+        order, all_orders = _greek_order(greek)
+        return cls(
+            order,
+            all_orders,
+            None if mu is None else _args.real_array(mu, "mu"),
+            _args.flag(normalized, "normalized"),
+            float(rtol),
+            _args.size(max_panels, "max_panels"),
+            _args.flag(allow_unvalidated, "allow_unvalidated"),
+            _args.flag(refine_backscatter, "refine_backscatter"),
+        )
+
+    def flat_mu(self) -> NDArray[np.float64]:
+        return np.ascontiguousarray(np.zeros(0) if self.mu is None else self.mu.reshape(-1))
+
+    def mu_shape(self) -> tuple[int, ...]:
+        return () if self.mu is None else self.mu.shape
+
+
+def _result(
+    raw: dict,
+    shape: tuple[int, ...],
+    o: _Options,
+    wavelength: NDArray[np.float64],
+    m: NDArray[np.complex128],
+    grid: dict[str, NDArray[np.float64]],
+    params: tuple[str, ...] = (),
+) -> EnsembleResult:
+    """The result from the engine's dict ``raw``, every array of shape ``shape`` (+ trailing)."""
     raw["panels"] = raw["panels"].astype(np.int64)
     for a in raw.values():
         a.flags.writeable = False
-    mu_shape = () if mus is None else mus.shape
+    mu_shape = o.mu_shape()
     values = _outputs(raw, "", None, shape, mu_shape)
     return EnsembleResult(
         **{f: getattr(values, f) for f in (*SCALARS, "greek", "phase")},
         error=_outputs(raw, "_errors", None, shape, mu_shape),
         converged=raw["converged"].reshape(shape),
         panels=raw["panels"].reshape(shape),
+        wavelength=_frozen(wavelength.copy()),
+        refractive_index=_frozen(m),
+        grid=MappingProxyType({n: _frozen(a.copy()) for n, a in grid.items()}),
         params=params,
         _d=tuple(_outputs(raw, "_d", k, shape, mu_shape) for k in range(len(params))),
         _d_errors=tuple(_outputs(raw, "_d_errors", k, shape, mu_shape) for k in range(len(params))),
     )
 
 
-_SHARED_DOC = """
+def _single(
+    items: list, mixture: bool, wl: NDArray[np.float64], derivatives, o: _Options
+) -> tuple[dict, tuple[str, ...]]:
+    """The engine's single-distribution or mixture path."""
+    names = tuple(_scatrs.ensemble_parameters(items, mixture))
+    params = _parameters(names, derivatives)
+    raw = _scatrs.ensemble_compute(
+        items,
+        np.ascontiguousarray(wl.reshape(-1)),
+        mixture,
+        o.order,
+        o.all_orders,
+        o.flat_mu(),
+        o.normalized,
+        [names.index(p) for p in params],
+        o.rtol,
+        o.max_panels,
+        o.allow_unvalidated,
+        o.refine_backscatter,
+    )
+    return raw, params
+
+
+def _sphere(particle: object, what: str) -> Sphere:
+    if not isinstance(particle, Sphere):
+        raise TypeError(f"{what} must be a Sphere, not {type(particle).__name__}")
+    return particle
+
+
+_COMPUTE_DOC = """{summary}
+
     Parameters
     ----------
-    {first}wavelength : array_like
+    wavelength : array_like
         Wavelengths, in the unit of the radii (real, scalar or any shape).
-    {second}
     greek : int, "all" or None, optional
         The Greek coefficients to compute: up to this order, every order, or none (the
         default).
@@ -437,14 +369,11 @@ _SHARED_DOC = """
         Normalize the phase matrix and the Greek coefficients (``alpha1[..., 0]`` = 1, the
         default) or give them per particle (``alpha1[..., 0]`` = ``c_sca``).
     derivatives : bool or sequence of str, optional
-        The parameters to differentiate with respect to (``True`` for all): the
-        distribution's (``"median_radius"``, …), ``"m_real"``,
-        ``"m_imag"``{weights}.
+        The parameters to differentiate with respect to (``True`` for all): {derivatives}
     rtol : float, optional
         Relative tolerance of the size integration, in [1e-13, 0.1].
     max_panels : int, optional
-        The most size panels per component and wavelength; reaching it sets ``converged``
-        to False.
+        The most size panels per {per}; reaching it sets ``converged`` to False.
     allow_unvalidated : bool, optional
         Compute sizes outside the validated domain of the Mie solution instead of raising.
     refine_backscatter : bool, optional
@@ -456,19 +385,20 @@ _SHARED_DOC = """
     Returns
     -------
     EnsembleResult
+        {returns}
 
     Raises
     ------
     InvalidInputError
-        For invalid distribution parameters, wavelengths, refractive indices or options, or
-        outputs that would exceed ``2 * MAX_MIE_VALUES`` real values (the bound of
-        :mod:`scatrs.mie`; with ``greek="all"``, known only after the computation).
+        For invalid distribution parameters, wavelengths, refractive indices (including a
+        model with no valid value at a wavelength) or options, or outputs that would exceed
+        ``2 * MAX_MIE_VALUES`` real values (the bound of :mod:`scatrs.mie`; with
+        ``greek="all"``, known only after the computation).
     OutsideValidatedDomainError
         If a size node is outside the validated domain, unless ``allow_unvalidated``; the
         message names the wavelength and the radius.
     TypeError, ValueError
-        For arguments of the wrong type, refractive indices that do not broadcast to the
-        wavelengths, or unknown or repeated parameter names.
+        For arguments of the wrong type, or unknown or repeated parameter names.{raises}
     BufferError
         If another rust-numpy extension holds a writeable borrow of an input.
     MemoryError
@@ -482,170 +412,171 @@ _SHARED_DOC = """
 """
 
 
-def compute(
-    distribution: Distribution,
-    wavelength: ArrayLike,
-    refractive_index: ArrayLike,
-    *,
-    greek: SupportsIndex | Literal["all"] | None = None,
-    mu: ArrayLike | None = None,
-    normalized: bool = True,
-    derivatives: bool | Sequence[str] = False,
-    rtol: float = 1e-6,
-    max_panels: SupportsIndex = 4096,
-    allow_unvalidated: bool = False,
-    refine_backscatter: bool = False,
-) -> EnsembleResult:
-    wl = _args.real_array(wavelength, "wavelength")
-    items = _specs([Component(distribution, refractive_index)], wl.shape)
-    return _run(
-        items,
-        wl,
-        False,
-        greek,
-        mu,
-        normalized,
-        derivatives,
-        rtol,
-        max_panels,
-        allow_unvalidated,
-        refine_backscatter,
-    )
-
-
-compute.__doc__ = """Averages over spheres of one size distribution.
-""" + _SHARED_DOC.format(
-    first="distribution : LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated "
-    "or MultiModal\n        The number size distribution.\n    ",
-    second="refractive_index : array_like\n        The refractive index, broadcast to the "
-    "wavelengths' shape.\n    ",
-    weights=', and ``"mode{j}.weight"`` for multimodal distributions',
-)
+@dataclass(frozen=True)
+class Ensemble:
+    """Particles of one kind over a size distribution.
 
+    Parameters
+    ----------
+    particle : Sphere
+        The particle, with its refractive-index model.
+    size : LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated or MultiModal
+        The number size distribution. If its parameters are arrays, the ensemble is a grid
+        of distributions, one per point of their broadcast shape.
 
-def compute_mixture(
-    components: Sequence[Component],
-    wavelength: ArrayLike,
-    *,
-    greek: SupportsIndex | Literal["all"] | None = None,
-    mu: ArrayLike | None = None,
-    normalized: bool = True,
-    derivatives: bool | Sequence[str] = False,
-    rtol: float = 1e-6,
-    max_panels: SupportsIndex = 4096,
-    allow_unvalidated: bool = False,
-    refine_backscatter: bool = False,
-) -> EnsembleResult:
-    wl = _args.real_array(wavelength, "wavelength")
-    items = _specs(list(components), wl.shape)
-    return _run(
-        items,
-        wl,
-        True,
-        greek,
-        mu,
-        normalized,
-        derivatives,
-        rtol,
-        max_panels,
-        allow_unvalidated,
-        refine_backscatter,
-    )
+    Raises
+    ------
+    TypeError
+        If ``particle`` is not a :class:`~scatrs.Sphere` or ``size`` not a distribution.
+    ValueError
+        If the distribution's array-valued parameters do not broadcast together.
+    """
 
+    particle: Sphere
+    size: Distribution
 
-compute_mixture.__doc__ = """Averages over an external mixture: cross sections and moments
-    weighted by wᵢ/Σw, phase matrices by wᵢ C_sca,ᵢ.
-""" + _SHARED_DOC.format(
-    first="components : sequence of Component\n        The components.\n    ",
-    second="",
-    weights=' (each prefixed by ``"c{i}."``), and the weights ``"c{i}.weight"``',
+    def __post_init__(self) -> None:
+        _sphere(self.particle, "particle")
+        _check(self.size, "size")
+        _grid(self.size)
+
+    @property
+    def grid_shape(self) -> tuple[int, ...]:
+        """The shape of the grid of distributions: ``()`` if no parameter is an array."""
+        return _grid(self.size)[0]
+
+    def compute(
+        self,
+        wavelength: ArrayLike,
+        *,
+        greek: SupportsIndex | Literal["all"] | None = None,
+        mu: ArrayLike | None = None,
+        normalized: bool = True,
+        derivatives: bool | Sequence[str] = False,
+        rtol: float = 1e-6,
+        max_panels: SupportsIndex = 4096,
+        allow_unvalidated: bool = False,
+        refine_backscatter: bool = False,
+    ) -> EnsembleResult:
+        wl = _args.real_array(wavelength, "wavelength")
+        m = self.particle.refractive_index.evaluate(wl)
+        o = _Options.of(
+            greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
+        )
+        shape, grid = _grid(self.size)
+        flat_m = np.ascontiguousarray(m.reshape(-1))
+        if shape == ():
+            items = [(_specs(self.size)[1][0], flat_m, 1.0)]
+            raw, params = _single(items, False, wl, derivatives, o)
+            return _result(raw, wl.shape, o, wl, m, grid, params)
+        if not (isinstance(derivatives, (bool, np.bool_)) and not derivatives):
+            raise NotImplementedError(
+                "derivatives of a grid of distributions are not available yet; compute each "
+                "point as its own Ensemble"
+            )
+        raw = _scatrs.ensemble_compute_batch(
+            _specs(self.size)[1],
+            flat_m,
+            np.ascontiguousarray(wl.reshape(-1)),
+            o.order,
+            o.all_orders,
+            o.flat_mu(),
+            o.normalized,
+            o.rtol,
+            o.max_panels,
+            o.allow_unvalidated,
+            o.refine_backscatter,
+        )
+        return _result(raw, (*shape, *wl.shape), o, wl, m, grid)
+
+
+Ensemble.compute.__doc__ = _COMPUTE_DOC.format(
+    summary="Averages over the ensemble at each wavelength.",
+    derivatives='the distribution\'s (``"median_radius"``, …; ``"mode{j}.<name>"`` and '
+    '``"mode{j}.weight"`` for a multimodal one), ``"m_real"`` and ``"m_imag"`` (Re m and Im m '
+    "at each wavelength). Not available for a grid.",
+    per="wavelength (shared by a grid's points)",
+    returns="Every array of shape ``grid + wavelength.shape``. A grid's points share their "
+    "size nodes and each meets its own ``rtol``: being at least as refined as on its own, a "
+    "point's values differ from those of its own ensemble within the error estimates.",
+    raises="\n    NotImplementedError\n        For ``derivatives`` of a grid.",
 )
 
 
-def compute_batch(
-    distributions: Sequence[Distribution],
-    wavelength: ArrayLike,
-    refractive_index: ArrayLike,
-    *,
-    greek: SupportsIndex | Literal["all"] | None = None,
-    mu: ArrayLike | None = None,
-    normalized: bool = True,
-    rtol: float = 1e-6,
-    max_panels: SupportsIndex = 4096,
-    allow_unvalidated: bool = False,
-    refine_backscatter: bool = False,
-) -> EnsembleResult:
-    """Averages over spheres of each of many size distributions of one material, sharing the
-    size nodes.
+@dataclass(frozen=True)
+class Mixture:
+    """An external mixture: ensembles with relative number concentrations.
 
-    One set of size panels per wavelength covers every distribution, refined until each meets
-    its own targets, so every Mie solution and phase-matrix table serves all of them: for a
-    grid of distributions (an optics database) this costs about as much as the broadest
-    distribution alone. Each distribution's result has its own error estimates and
-    ``converged`` flag; being at least as refined as on its own, its values differ from
-    :func:`compute`'s within the error estimates. No derivatives: use :func:`compute`.
+    Cross sections and moments are averaged with the number fractions wᵢ/Σw, phase matrices
+    with wᵢ times the ``c_sca`` of ensemble i.
 
     Parameters
     ----------
-    distributions : sequence of LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, \
-Tabulated or MultiModal
-        The number size distributions.
-    wavelength : array_like
-        Wavelengths, in the unit of the radii.
-    refractive_index : array_like
-        The refractive index, broadcast to the wavelengths' shape.
-    greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
-        As for :func:`compute`.
-
-    Returns
-    -------
-    EnsembleResult
-        Every array with a leading axis of the distributions, then the wavelengths' shape;
-        ``panels`` counts the shared panels.
+    components : sequence of (float, Ensemble)
+        The weights wᵢ ≥ 0 (finite, not all zero) and the ensembles, whose distributions
+        must not be grids.
 
     Raises
     ------
-    InvalidInputError
-        For no distributions, invalid parameters, wavelengths, refractive indices or options,
-        or outputs that would exceed ``2 * MAX_MIE_VALUES`` real values.
-    OutsideValidatedDomainError
-        If a size node is outside the validated domain, unless ``allow_unvalidated``.
-    TypeError, ValueError
-        For arguments of the wrong type, or refractive indices that do not broadcast to the
-        wavelengths.
+    TypeError
+        If a component is not a (weight, Ensemble) pair.
+    ValueError
+        If an ensemble is a grid.
     """
-    wl = _args.real_array(wavelength, "wavelength")
-    if isinstance(distributions, (str, bytes)) or not isinstance(distributions, Iterable):
-        raise TypeError(
-            f"distributions must be a sequence of distributions, not {type(distributions).__name__}"
+
+    components: Sequence[tuple[float, Ensemble]]
+
+    def __post_init__(self) -> None:
+        out = []
+        for i, c in enumerate(self.components):
+            try:
+                w, e = c
+            except (TypeError, ValueError):
+                raise TypeError(
+                    f"component {i} must be a (weight, Ensemble) pair, not {type(c).__name__}"
+                ) from None
+            if not isinstance(e, Ensemble):
+                raise TypeError(f"component {i} must be a (weight, Ensemble) pair")
+            if e.grid_shape != ():
+                raise ValueError(
+                    f"the distribution of component {i} is a grid of shape {e.grid_shape}; "
+                    "a mixture needs scalar parameters"
+                )
+            out.append((float(w), e))
+        object.__setattr__(self, "components", tuple(out))
+
+    def compute(
+        self,
+        wavelength: ArrayLike,
+        *,
+        greek: SupportsIndex | Literal["all"] | None = None,
+        mu: ArrayLike | None = None,
+        normalized: bool = True,
+        derivatives: bool | Sequence[str] = False,
+        rtol: float = 1e-6,
+        max_panels: SupportsIndex = 4096,
+        allow_unvalidated: bool = False,
+        refine_backscatter: bool = False,
+    ) -> EnsembleResult:
+        wl = _args.real_array(wavelength, "wavelength")
+        ms = [e.particle.refractive_index.evaluate(wl) for _, e in self.components]
+        o = _Options.of(
+            greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
         )
-    dists = list(distributions)
-    specs = [_check(d, f"distribution {i}")._spec() for i, d in enumerate(dists)]
-    m = _refractive_index(refractive_index, wl.shape)
-    mus = None if mu is None else _args.real_array(mu, "mu")
-    order, all_orders = _greek_order(greek)
-    raw = _scatrs.ensemble_compute_batch(
-        specs,
-        m,
-        np.ascontiguousarray(wl.reshape(-1)),
-        order,
-        all_orders,
-        np.ascontiguousarray(np.zeros(0) if mus is None else mus.reshape(-1)),
-        _args.flag(normalized, "normalized"),
-        float(rtol),
-        _args.size(max_panels, "max_panels"),
-        _args.flag(allow_unvalidated, "allow_unvalidated"),
-        _args.flag(refine_backscatter, "refine_backscatter"),
-    )
-    raw["panels"] = raw["panels"].astype(np.int64)
-    for a in raw.values():
-        a.flags.writeable = False
-    shape = (len(dists), *wl.shape)
-    mu_shape = () if mus is None else mus.shape
-    values = _outputs(raw, "", None, shape, mu_shape)
-    return EnsembleResult(
-        **{f: getattr(values, f) for f in (*SCALARS, "greek", "phase")},
-        error=_outputs(raw, "_errors", None, shape, mu_shape),
-        converged=raw["converged"].reshape(shape),
-        panels=raw["panels"].reshape(shape),
-    )
+        items = [
+            (_specs(e.size)[1][0], np.ascontiguousarray(m.reshape(-1)), w)
+            for (w, e), m in zip(self.components, ms, strict=True)
+        ]
+        raw, params = _single(items, True, wl, derivatives, o)
+        m = np.stack(ms) if ms else np.zeros((0, *wl.shape), dtype=complex)
+        return _result(raw, wl.shape, o, wl, m, {}, params)
+
+
+Mixture.compute.__doc__ = _COMPUTE_DOC.format(
+    summary="Averages over the mixture at each wavelength.",
+    derivatives='each component\'s distribution parameters, ``"m_real"`` and ``"m_imag"``, '
+    'prefixed by ``"c{i}."``, and the weights ``"c{i}.weight"``.',
+    per="component and wavelength",
+    returns="Every array of the wavelengths' shape.",
+    raises="",
+)
diff --git a/python/scatrs/size.py b/python/scatrs/size.py
new file mode 100644
index 0000000..dd75e1d
--- /dev/null
+++ b/python/scatrs/size.py
@@ -0,0 +1,252 @@
+"""Number size distributions of particles.
+
+Radii are in any length unit, the unit of the wavelengths they are computed with. Every
+numeric parameter may be an array: the parameters of a distribution broadcast together into a
+grid of distributions, which :meth:`scatrs.Ensemble.compute` computes at once (see
+:doc:`/theory/ensembles`). Arrays are copied and stored read-only, so distributions are
+immutable values that compare and hash by their parameters.
+"""
+
+from __future__ import annotations
+
+from collections.abc import Callable, Sequence
+from dataclasses import dataclass, fields
+from typing import Any, ClassVar, TypeAlias
+
+import numpy as np
+from numpy.typing import ArrayLike, NDArray
+
+from scatrs import _args
+
+__all__ = [
+    "Distribution",
+    "Gamma",
+    "LogNormal",
+    "ModifiedGamma",
+    "Monodisperse",
+    "MultiModal",
+    "PowerLaw",
+    "Tabulated",
+]
+
+Param: TypeAlias = float | NDArray[np.float64]
+"""A parameter: a float, or a read-only float64 array of at least one dimension."""
+
+_Pick: TypeAlias = Callable[[str, Param], float]
+"""The value of a parameter at one point of a grid, given its name and value."""
+
+
+def _param(value: ArrayLike, name: str) -> Param:
+    """``value`` as a float, or as a read-only copy of an array of dimension ≥ 1."""
+    a = _args.real_array(value, name)
+    if a.ndim == 0:
+        return float(a)
+    a = a.copy()
+    a.flags.writeable = False
+    return a
+
+
+def _key(value: Param) -> Any:
+    """``value`` in a form that compares and hashes by value."""
+    if isinstance(value, float):
+        return value
+    return (value.shape, tuple(value.ravel().tolist()))
+
+
+class _Distribution:
+    """The parameters, value semantics and engine specs shared by the distributions."""
+
+    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
+        """The (name, value) of every parameter, nested ones prefixed."""
+        raise NotImplementedError
+
+    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
+        """The engine's tagged tuple, with each parameter's value given by ``pick``."""
+        raise NotImplementedError
+
+    def _identity(self) -> tuple:
+        return (type(self).__name__, *((n, _key(v)) for n, v in self._leaves()))
+
+    def __eq__(self, other: object) -> bool:
+        if not isinstance(other, _Distribution):
+            return NotImplemented
+        return self._identity() == other._identity()
+
+    def __hash__(self) -> int:
+        return hash(self._identity())
+
+
+class _Simple(_Distribution):
+    """A distribution whose parameters are its fields, in the engine's order."""
+
+    _KIND: ClassVar[str]
+
+    def __post_init__(self) -> None:
+        for f in fields(self):  # type: ignore[arg-type]
+            object.__setattr__(self, f.name, _param(getattr(self, f.name), f.name))
+
+    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
+        return [(prefix + f.name, getattr(self, f.name)) for f in fields(self)]  # type: ignore[arg-type]
+
+    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
+        return (self._KIND, *(pick(n, v) for n, v in self._leaves(prefix)))
+
+
+@dataclass(frozen=True, eq=False)
+class LogNormal(_Simple):
+    """n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: ``median_radius`` r_g > 0, ``width``
+    σ_g > 1."""
+
+    _KIND = "lognormal"
+    median_radius: Param
+    width: Param
+
+
+@dataclass(frozen=True, eq=False)
+class Gamma(_Simple):
+    """Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
+    ``effective_radius`` a > 0, ``effective_variance`` 0 < b < 1/2."""
+
+    _KIND = "gamma"
+    effective_radius: Param
+    effective_variance: Param
+
+
+@dataclass(frozen=True, eq=False)
+class ModifiedGamma(_Simple):
+    """Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): ``alpha`` > −1,
+    ``b`` > 0, ``gamma`` > 0."""
+
+    _KIND = "modified_gamma"
+    alpha: Param
+    b: Param
+    gamma: Param
+
+
+@dataclass(frozen=True, eq=False)
+class PowerLaw(_Simple):
+    """n(r) ∝ r^(−α) on [``min_radius``, ``max_radius``]: ``exponent`` α (the limits are not
+    differentiable)."""
+
+    _KIND = "power_law"
+    exponent: Param
+    min_radius: Param
+    max_radius: Param
+
+
+@dataclass(frozen=True, eq=False)
+class Monodisperse(_Simple):
+    """Spheres of one ``radius``."""
+
+    _KIND = "monodisperse"
+    radius: Param
+
+
+@dataclass(frozen=True, eq=False)
+class Tabulated(_Distribution):
+    """n(r) linear between the points (``radius[i]``, ``density[i]``) and 0 outside: radii
+    increasing, densities ≥ 0. The table is data, not parameters: it is neither
+    differentiated nor broadcast into a grid."""
+
+    radius: Sequence[float]
+    density: Sequence[float]
+
+    def __post_init__(self) -> None:
+        # Tuples of floats, so that the distribution compares and hashes by value.
+        for name in ("radius", "density"):
+            values = np.asarray(getattr(self, name), dtype=float).ravel()
+            object.__setattr__(self, name, tuple(float(v) for v in values))
+
+    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
+        return []
+
+    def _identity(self) -> tuple:
+        return ("Tabulated", self.radius, self.density)
+
+    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
+        return ("tabulated", list(self.radius), list(self.density))
+
+
+@dataclass(frozen=True, eq=False)
+class MultiModal(_Distribution):
+    """Σⱼ wⱼ nⱼ(r), each mode normalized: ``modes`` is a sequence of (weight wⱼ > 0, mode).
+    The modes' parameters are named ``"mode{j}.<name>"``, their weights ``"mode{j}.weight"``.
+    """
+
+    modes: Sequence[tuple[Param, Distribution]]
+
+    def __post_init__(self) -> None:
+        modes = tuple(
+            (_param(w, f"the weight of mode {j}"), _check(d, f"mode {j}"))
+            for j, (w, d) in enumerate(self.modes)
+        )
+        object.__setattr__(self, "modes", modes)
+
+    def _leaves(self, prefix: str = "") -> list[tuple[str, Param]]:
+        out = []
+        for j, (w, d) in enumerate(self.modes):
+            out += d._leaves(f"{prefix}mode{j}.")
+            out.append((f"{prefix}mode{j}.weight", w))
+        return out
+
+    def _spec(self, pick: _Pick, prefix: str = "") -> tuple:
+        return (
+            "multimodal",
+            [
+                (pick(f"{prefix}mode{j}.weight", w), d._spec(pick, f"{prefix}mode{j}."))
+                for j, (w, d) in enumerate(self.modes)
+            ],
+        )
+
+
+Distribution: TypeAlias = (
+    LogNormal | Gamma | ModifiedGamma | PowerLaw | Monodisperse | Tabulated | MultiModal
+)
+
+_DISTRIBUTIONS = (LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal)
+
+
+def _check(d: object, what: str) -> Distribution:
+    """``d``, if it is a distribution.
+
+    Raises
+    ------
+    TypeError
+        Otherwise.
+    """
+    if not isinstance(d, _DISTRIBUTIONS):
+        names = ", ".join(c.__name__ for c in _DISTRIBUTIONS)
+        raise TypeError(f"{what} must be one of {names}, not {type(d).__name__}")
+    return d
+
+
+def _grid(d: Distribution) -> tuple[tuple[int, ...], dict[str, NDArray[np.float64]]]:
+    """The grid shape of ``d``'s parameters, and the array-valued ones broadcast to it.
+
+    Raises
+    ------
+    ValueError
+        If the array-valued parameters do not broadcast together.
+    """
+    arrays = {n: v for n, v in d._leaves() if not isinstance(v, float)}
+    try:
+        shape = np.broadcast_shapes(*(a.shape for a in arrays.values()))
+    except ValueError:
+        shapes = ", ".join(f"{n} {a.shape}" for n, a in arrays.items())
+        raise ValueError(
+            f"the distribution's parameters do not broadcast together: {shapes}"
+        ) from None
+    return shape, {n: np.broadcast_to(a, shape) for n, a in arrays.items()}
+
+
+def _specs(d: Distribution) -> tuple[tuple[int, ...], list[tuple]]:
+    """The grid shape of ``d``, and the engine spec of each of its points in C order (one for
+    a scalar distribution)."""
+    shape, arrays = _grid(d)
+    flat = {n: a.reshape(-1) for n, a in arrays.items()}
+    count = int(np.prod(shape))
+
+    def at(k: int) -> _Pick:
+        return lambda n, v: float(v) if isinstance(v, float) else float(flat[n][k])
+
+    return shape, [d._spec(at(k)) for k in range(count)]
diff --git a/tests/test_ensemble.py b/tests/test_ensemble.py
index 3c2a897..3e5809e 100644
--- a/tests/test_ensemble.py
+++ b/tests/test_ensemble.py
@@ -1,4 +1,4 @@
-"""scatrs.ensemble: values, shapes, derivatives, mixtures and errors."""
+"""Ensemble and Mixture: values, shapes, derivatives, grids, mixtures and errors."""
 
 import threading
 import time
@@ -6,14 +6,34 @@ import time
 import numpy as np
 import pytest
 
-from scatrs import InvalidInputError, OutsideValidatedDomainError, ensemble, mie
+from scatrs import (
+    Ensemble,
+    Gamma,
+    InvalidInputError,
+    LogNormal,
+    Mixture,
+    ModifiedGamma,
+    Monodisperse,
+    MultiModal,
+    OutsideValidatedDomainError,
+    PowerLaw,
+    Sphere,
+    Tabulated,
+    mie,
+    refractive,
+)
 
-LOGNORMAL = ensemble.LogNormal(0.08, 1.6)
+LOGNORMAL = LogNormal(0.08, 1.6)
+
+
+def compute(distribution, wavelength, m, **kw):
+    """Spheres of the constant refractive index ``m`` over ``distribution``."""
+    return Ensemble(Sphere(m), distribution).compute(wavelength, **kw)
 
 
 def test_monodisperse_is_the_single_sphere():
     r, wl, m = 0.7, np.array([0.45, 0.55]), 1.5 + 0.01j
-    e = ensemble.compute(ensemble.Monodisperse(r), wl, m, greek="all", mu=[1.0, -1.0])
+    e = compute(Monodisperse(r), wl, m, greek="all", mu=[1.0, -1.0])
     q = mie.efficiencies(2 * np.pi * r / wl, m)
     np.testing.assert_allclose(e.c_ext, np.pi * r**2 * q.q_ext, rtol=1e-15)
     np.testing.assert_allclose(e.g, q.g, rtol=1e-14)
@@ -24,7 +44,7 @@ def test_monodisperse_is_the_single_sphere():
 
 def test_shapes_and_frozen_results():
     wl = np.array([[0.35, 0.5], [0.75, 1.0]])
-    r = ensemble.compute(LOGNORMAL, wl, 1.43, greek=8, mu=np.linspace(-1, 1, 5), derivatives=True)
+    r = compute(LOGNORMAL, wl, 1.43, greek=8, mu=np.linspace(-1, 1, 5), derivatives=True)
     assert r.c_ext.shape == (2, 2) and r.converged.dtype == bool
     assert r.greek.alpha1.shape == (2, 2, 9) and r.phase.f34.shape == (2, 2, 5)
     assert r.error.greek.beta2.shape == (2, 2, 9)
@@ -33,15 +53,22 @@ def test_shapes_and_frozen_results():
     for a in (r.c_ext, r.greek.alpha1, r.d("width").g, r.error.ssa, r.converged):
         with pytest.raises(ValueError, match="read-only"):
             a[...] = 0
-    s = ensemble.compute(LOGNORMAL, 0.5, 1.43)
+    s = compute(LOGNORMAL, 0.5, 1.43)
     assert s.c_ext.shape == () and s.greek is None and s.phase is None
 
 
-def test_refractive_index_broadcasts_to_the_wavelengths():
-    wl = [0.4, 0.6]
-    both = ensemble.compute(LOGNORMAL, wl, [1.45, 1.43 + 1e-3j])
-    one = ensemble.compute(LOGNORMAL, wl[1], 1.43 + 1e-3j)
-    assert both.c_ext[1] == one.c_ext
+def test_the_model_is_evaluated_at_each_wavelength():
+    wl = np.array([[0.4, 0.5], [0.6, 0.7]])
+    table = refractive.Table([0.4, 0.7], [1.45, 1.40 + 3e-3j])
+    r = Ensemble(Sphere(table), LOGNORMAL).compute(wl)
+    np.testing.assert_array_equal(r.refractive_index, table.evaluate(wl))
+    np.testing.assert_array_equal(r.wavelength, wl)
+    for i in range(2):
+        for j in range(2):
+            m = complex(table.evaluate(wl[i, j]))
+            assert r.c_ext[i, j] == compute(LOGNORMAL, wl[i, j], m).c_ext
+    with pytest.raises(InvalidInputError, match="outside the Table's range"):
+        Ensemble(Sphere(table), LOGNORMAL).compute([0.5, 0.8])
 
 
 def _fd(f, h):
@@ -51,87 +78,97 @@ def _fd(f, h):
 def test_derivatives_match_finite_differences():
     wl, m = 0.55, 1.45 + 0.02j
     kw = {"rtol": 1e-11, "max_panels": 50_000, "greek": 4}
-    r = ensemble.compute(ensemble.LogNormal(0.2, 1.6), wl, m, derivatives=True, **kw)
+    r = compute(LogNormal(0.2, 1.6), wl, m, derivatives=True, **kw)
     steps = {
-        "median_radius": lambda e: (ensemble.LogNormal(0.2 + e, 1.6), m),
-        "width": lambda e: (ensemble.LogNormal(0.2, 1.6 + e), m),
-        "m_real": lambda e: (ensemble.LogNormal(0.2, 1.6), m + e),
-        "m_imag": lambda e: (ensemble.LogNormal(0.2, 1.6), m + 1j * e),
+        "median_radius": lambda e: (LogNormal(0.2 + e, 1.6), m),
+        "width": lambda e: (LogNormal(0.2, 1.6 + e), m),
+        "m_real": lambda e: (LogNormal(0.2, 1.6), m + e),
+        "m_imag": lambda e: (LogNormal(0.2, 1.6), m + 1j * e),
     }
     for p, at in steps.items():
         h = 1e-4
         for name in ("c_ext", "ssa", "g", "c_back"):
             fd = _fd(
-                lambda e, at=at, name=name: getattr(
-                    ensemble.compute(at(e)[0], wl, at(e)[1], **kw), name
-                ),
+                lambda e, at=at, name=name: getattr(compute(at(e)[0], wl, at(e)[1], **kw), name),
                 h,
             )
             assert abs(getattr(r.d(p), name) - fd) <= 1e-5 * abs(fd) + 1e-12, (p, name)
-        fd = _fd(lambda e, at=at: ensemble.compute(at(e)[0], wl, at(e)[1], **kw).greek.alpha1[2], h)
+        fd = _fd(lambda e, at=at: compute(at(e)[0], wl, at(e)[1], **kw).greek.alpha1[2], h)
         assert abs(r.d(p).greek.alpha1[2] - fd) <= 1e-5 * abs(fd) + 1e-10, p
 
 
 def test_mixtures_and_their_weights():
-    a = ensemble.Component(ensemble.LogNormal(0.1, 1.5), 1.45, 2.0)
-    b = ensemble.Component(ensemble.Gamma(1.0, 0.1), 1.53 + 0.01j, 0.5)
-    m = ensemble.compute_mixture([a, b], 0.6, derivatives=["c0.weight"])
-    ra = ensemble.compute(a.distribution, 0.6, a.refractive_index)
-    rb = ensemble.compute(b.distribution, 0.6, b.refractive_index)
+    a = Ensemble(Sphere(1.45), LogNormal(0.1, 1.5))
+    b = Ensemble(Sphere(1.53 + 0.01j), Gamma(1.0, 0.1))
+    m = Mixture([(2.0, a), (0.5, b)]).compute(0.6, derivatives=["c0.weight"])
+    ra, rb = a.compute(0.6), b.compute(0.6)
     np.testing.assert_allclose(m.c_ext, (2.0 * ra.c_ext + 0.5 * rb.c_ext) / 2.5, rtol=1e-13)
     # ∂/∂w₀ of (w₀ A + w₁ B)/(w₀ + w₁) = (A − mix)/(w₀ + w₁).
     np.testing.assert_allclose(m.d("c0.weight").c_ext, (ra.c_ext - m.c_ext) / 2.5, rtol=1e-12)
-    names = ensemble.compute_mixture([a, b], 0.6, derivatives=True).params
+    names = Mixture([(2.0, a), (0.5, b)]).compute(0.6, derivatives=True).params
     assert names[:3] == ("c0.median_radius", "c0.width", "c0.m_real") and names[-1] == "c1.weight"
-    bi = ensemble.MultiModal(
-        [(2.0, ensemble.LogNormal(0.1, 1.5)), (0.5, ensemble.Monodisperse(1.0))]
-    )
-    assert ensemble.compute(bi, 0.5, 1.5, derivatives=True).params[:3] == (
+    bi = MultiModal([(2.0, LogNormal(0.1, 1.5)), (0.5, Monodisperse(1.0))])
+    assert compute(bi, 0.5, 1.5, derivatives=True).params[:3] == (
         "mode0.median_radius",
         "mode0.width",
         "mode0.weight",
     )
 
 
+def test_mixture_results_carry_each_components_refractive_index():
+    wl = [[0.4, 0.5, 0.6]]
+    a = Ensemble(Sphere(1.45), LOGNORMAL)
+    b = Ensemble(Sphere(refractive.Cauchy(1.5, 0.004)), LOGNORMAL)
+    r = Mixture([(1.0, a), (1.0, b)]).compute(wl)
+    assert r.refractive_index.shape == (2, 1, 3) and r.c_ext.shape == (1, 3)
+    np.testing.assert_array_equal(r.refractive_index[1], b.particle.refractive_index.evaluate(wl))
+    assert dict(r.grid) == {}
+
+
 def test_tabulated_and_power_law_agree():
-    t = ensemble.compute(ensemble.Tabulated([0.1, 0.5], [1.0, 1.0]), 0.5, 1.5, rtol=1e-9)
-    p = ensemble.compute(ensemble.PowerLaw(0.0, 0.1, 0.5), 0.5, 1.5, rtol=1e-9)
+    t = compute(Tabulated([0.1, 0.5], [1.0, 1.0]), 0.5, 1.5, rtol=1e-9)
+    p = compute(PowerLaw(0.0, 0.1, 0.5), 0.5, 1.5, rtol=1e-9)
     np.testing.assert_allclose(t.c_ext, p.c_ext, rtol=1e-8)
 
 
 def test_errors():
     with pytest.raises(InvalidInputError, match="width"):
-        ensemble.compute(ensemble.LogNormal(0.1, 0.5), 0.5, 1.5)
+        compute(LogNormal(0.1, 0.5), 0.5, 1.5)
     with pytest.raises(InvalidInputError, match="rtol"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5, rtol=1.0)
+        compute(LOGNORMAL, 0.5, 1.5, rtol=1.0)
     with pytest.raises(InvalidInputError, match="wavelength"):
-        ensemble.compute(LOGNORMAL, -0.5, 1.5)
+        compute(LOGNORMAL, -0.5, 1.5)
     with pytest.raises(OutsideValidatedDomainError, match="radius"):
-        ensemble.compute(ensemble.Monodisperse(2000.0), 0.5, 1.5)
+        compute(Monodisperse(2000.0), 0.5, 1.5)
     with pytest.raises(ValueError, match="unknown parameter"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives=["radius"])
+        compute(LOGNORMAL, 0.5, 1.5, derivatives=["radius"])
     with pytest.raises(ValueError, match="no derivative"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5).d("width")
+        compute(LOGNORMAL, 0.5, 1.5).d("width")
     with pytest.raises(ValueError, match="greek must be"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5, greek="some")
-    with pytest.raises(TypeError, match="Component"):
-        ensemble.compute_mixture([LOGNORMAL], 0.5)
+        compute(LOGNORMAL, 0.5, 1.5, greek="some")
     with pytest.raises(TypeError, match="derivatives must be"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives="width")
+        compute(LOGNORMAL, 0.5, 1.5, derivatives="width")
     with pytest.raises(InvalidInputError, match="weights"):
-        ensemble.compute_mixture([ensemble.Component(LOGNORMAL, 1.5, 0.0)], 0.5)
-    with pytest.raises(ValueError):
-        ensemble.compute(LOGNORMAL, [0.4, 0.5], [1.5, 1.5, 1.5])
+        Mixture([(0.0, Ensemble(Sphere(1.5), LOGNORMAL))]).compute(0.5)
+    with pytest.raises(InvalidInputError, match="needs a component"):
+        Mixture([]).compute(0.5)
+    with pytest.raises(TypeError, match="pair"):
+        Mixture([LOGNORMAL])
+    with pytest.raises(TypeError, match="pair"):
+        Mixture([(1.0, LOGNORMAL)])
+    with pytest.raises(TypeError, match="must be a Sphere"):
+        Ensemble(1.5, LOGNORMAL)
+    with pytest.raises(TypeError, match="must be one of"):
+        Ensemble(Sphere(1.5), "lognormal")
 
 
 def test_gil_is_released_while_computing():
     span = {}
+    ens = Ensemble(Sphere(1.45 + 1e-3j), LogNormal(0.3, 1.6))
 
     def worker():
         span["start"] = time.perf_counter()
-        ensemble.compute(
-            ensemble.LogNormal(0.3, 1.6), np.linspace(0.3, 1.0, 64), 1.45 + 1e-3j, greek=32
-        )
+        ens.compute(np.linspace(0.3, 1.0, 64), greek=32)
         span["end"] = time.perf_counter()
 
     stamps = []
@@ -148,7 +185,8 @@ def test_gil_is_released_while_computing():
 
 
 def test_results_cannot_be_made_writeable():
-    r = ensemble.compute(LOGNORMAL, [0.4, 0.5], 1.43, greek=4, mu=[0.0], derivatives=["width"])
+    r = compute(LOGNORMAL, [0.4, 0.5], 1.43, greek=4, mu=[0.0], derivatives=["width"])
+    g = compute(LogNormal([0.1, 0.2], 1.5), 0.5, 1.43)
     arrays = [
         r.c_ext,
         r.converged,
@@ -158,6 +196,9 @@ def test_results_cannot_be_made_writeable():
         r.error.greek.beta1,
         r.d("width").greek.alpha2,
         r.d_error("width").phase.f11,
+        r.wavelength,
+        r.refractive_index,
+        g.grid["median_radius"],
     ]
     for a in arrays:
         with pytest.raises(ValueError):
@@ -169,104 +210,107 @@ def test_buffer_error_when_an_input_is_borrowed_for_writing():
 
     wl = np.array([0.4, 0.5])
     with pytest.raises(BufferError):
-        _testing.with_write_borrow(wl, lambda: ensemble.compute(LOGNORMAL, wl, 1.5))
-
-
-def test_refractive_indices_broadcast_to_the_wavelengths_shape():
-    wl = np.array([[0.4, 0.5], [0.6, 0.7]])
-    m = np.array([1.45, 1.40 + 1e-3j])  # along the last axis
-    r = ensemble.compute(LOGNORMAL, wl, m)
-    for i in range(2):
-        for j in range(2):
-            assert r.c_ext[i, j] == ensemble.compute(LOGNORMAL, wl[i, j], m[j]).c_ext
-    with pytest.raises(ValueError, match="refractive_index of shape"):
-        ensemble.compute(LOGNORMAL, wl, [1.4, 1.5, 1.6])
-    with pytest.raises(ValueError, match="refractive_index of shape"):
-        ensemble.compute(LOGNORMAL, [0.4, 0.5, 0.6, 0.7], m.reshape(2, 1) * np.ones((2, 2)))
-
-
-def test_distributions_are_values():
-    t1 = ensemble.Tabulated(np.array([0.1, 0.5]), [1.0, 2.0])
-    t2 = ensemble.Tabulated([0.1, 0.5], (1.0, 2.0))
-    assert t1 == t2 and hash(t1) == hash(t2)
-    b = ensemble.MultiModal([(1.0, LOGNORMAL), (2.0, t1)])
-    assert hash(b) == hash(ensemble.MultiModal([(1, LOGNORMAL), (2, t2)]))
-    c = ensemble.Component(LOGNORMAL, np.array([1.5]))
-    assert c == c and c != ensemble.Component(LOGNORMAL, np.array([1.5]))
-    with pytest.raises(TypeError, match="must be one of"):
-        ensemble.compute("lognormal", 0.5, 1.5)
-    with pytest.raises(TypeError, match="must be one of"):
-        ensemble.MultiModal([(1.0, "x")])
+        _testing.with_write_borrow(wl, lambda: compute(LOGNORMAL, wl, 1.5))
 
 
 def test_greek_orders_and_parameter_names():
-    empty = ensemble.compute(LOGNORMAL, np.zeros(0), 1.5, greek=7)
+    empty = compute(LOGNORMAL, np.zeros(0), 1.5, greek=7)
     assert empty.greek.alpha1.shape == (0, 8) and empty.c_ext.shape == (0,)
     with pytest.raises(InvalidInputError, match="Greek order"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5, greek=2**63)
-    a = ensemble.compute(ensemble.Monodisperse(0.3), [0.3, 3.0], 1.5, greek="all")
+        compute(LOGNORMAL, 0.5, 1.5, greek=2**63)
+    a = compute(Monodisperse(0.3), [0.3, 3.0], 1.5, greek="all")
     n = mie.default_n_max(2 * np.pi * 0.3 / np.array([0.3, 3.0]))
     assert a.greek.alpha1.shape == (2, 2 * n[0] + 1)
     assert np.all(a.greek.alpha1[1, 2 * n[1] + 1 :] == 0)
     for d, names in [
-        (ensemble.Gamma(1.0, 0.1), ("effective_radius", "effective_variance")),
-        (ensemble.ModifiedGamma(2.0, 15.0, 0.5), ("alpha", "b", "gamma")),
-        (ensemble.PowerLaw(3.0, 0.1, 1.0), ("exponent",)),
-        (ensemble.Tabulated([0.1, 1.0], [1.0, 1.0]), ()),
+        (Gamma(1.0, 0.1), ("effective_radius", "effective_variance")),
+        (ModifiedGamma(2.0, 15.0, 0.5), ("alpha", "b", "gamma")),
+        (PowerLaw(3.0, 0.1, 1.0), ("exponent",)),
+        (Tabulated([0.1, 1.0], [1.0, 1.0]), ()),
     ]:
-        assert ensemble.compute(d, 0.5, 1.5, derivatives=True).params == (
-            *names,
-            "m_real",
-            "m_imag",
-        )
+        assert compute(d, 0.5, 1.5, derivatives=True).params == (*names, "m_real", "m_imag")
     with pytest.raises(ValueError, match="twice"):
-        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives=["width", "width"])
+        compute(LOGNORMAL, 0.5, 1.5, derivatives=["width", "width"])
 
 
 def test_error_estimates_of_derivatives_and_convergence_flags():
-    r = ensemble.compute(LOGNORMAL, 0.5, 1.43, derivatives=["median_radius"])
+    r = compute(LOGNORMAL, 0.5, 1.43, derivatives=["median_radius"])
     e = r.d_error("median_radius")
     assert e.c_ext >= 0 and e.c_ext < 1e-4 * abs(r.d("median_radius").c_ext)
-    capped = ensemble.compute(ensemble.LogNormal(1.0, 2.0), 0.5, 1.33, max_panels=1)
+    capped = compute(LogNormal(1.0, 2.0), 0.5, 1.33, max_panels=1)
     assert not capped.converged and capped.panels > 0
     with pytest.raises(InvalidInputError, match="MAX_MIE_VALUES"):
-        ensemble.compute(LOGNORMAL, np.full(2**14, 0.5), 1.5, greek=2**15)
+        compute(LOGNORMAL, np.full(2**14, 0.5), 1.5, greek=2**15)
 
 
 def test_a_single_distribution_has_one_component():
     from scatrs import _scatrs
 
-    spec = (LOGNORMAL._spec(), np.array([1.5 + 0j]), 1.0)
+    spec = (("lognormal", 0.08, 1.6), np.array([1.5 + 0j]), 1.0)
     with pytest.raises(InvalidInputError, match="exactly one component"):
         _scatrs.ensemble_parameters([spec, spec], False)
 
 
-def test_batch_shares_nodes_and_matches_each_distribution():
-    dists = [ensemble.LogNormal(0.1, 1.5), ensemble.Gamma(0.4, 0.1), ensemble.Monodisperse(0.3)]
+def test_a_grid_shares_nodes_and_matches_each_distribution():
+    size = LogNormal(np.array([0.1, 0.2, 0.3])[:, None], [1.5, 1.7])
     wl = np.array([[0.4, 0.6, 0.8]])
-    r = ensemble.compute_batch(dists, wl, 1.45 + 1e-3j, greek=8, mu=[0.5, -0.5])
-    assert r.c_ext.shape == (3, 1, 3) and r.greek.alpha1.shape == (3, 1, 3, 9)
-    assert r.phase.f11.shape == (3, 1, 3, 2) and r.converged.all()
+    ens = Ensemble(Sphere(1.45 + 1e-3j), size)
+    assert ens.grid_shape == (3, 2)
+    r = ens.compute(wl, greek=8, mu=[0.5, -0.5])
+    assert r.c_ext.shape == (3, 2, 1, 3) and r.greek.alpha1.shape == (3, 2, 1, 3, 9)
+    assert r.phase.f11.shape == (3, 2, 1, 3, 2) and r.converged.all()
     assert r.params == () and not r.c_ext.flags.writeable
-    # Shared panels for the distributions with a range; none for the single radius.
-    assert np.all(r.panels[0] == r.panels[1]) and np.all(r.panels[2] == 0)
-    for i, d in enumerate(dists):
-        a = ensemble.compute(d, wl, 1.45 + 1e-3j, greek=8)
-        np.testing.assert_array_less(
-            np.abs(r.c_ext[i] - a.c_ext), 2 * (r.error.c_ext[i] + a.error.c_ext) + 1e-15
-        )
-        np.testing.assert_array_less(
-            np.abs(r.greek.alpha1[i] - a.greek.alpha1),
-            2 * (r.error.greek.alpha1[i] + a.error.greek.alpha1) + 1e-13,
-        )
-    # The refractive index broadcasts to the wavelengths.
-    m = ensemble.compute_batch(dists[:1], [0.4, 0.6], [1.45, 1.5])
-    assert m.c_ext.shape == (1, 2)
-    with pytest.raises(InvalidInputError):
-        ensemble.compute_batch([], 0.5, 1.45)
-    with pytest.raises(TypeError):
-        ensemble.compute_batch(ensemble.LogNormal(0.1, 1.5), 0.5, 1.45)
-    with pytest.raises(TypeError):
-        ensemble.compute_batch([1.0], 0.5, 1.45)
-    with pytest.raises(ValueError):
-        ensemble.compute_batch(dists, [0.4, 0.6], [1.45, 1.5, 1.6])
+    assert set(r.grid) == {"median_radius", "width"} and r.grid["width"].shape == (3, 2)
+    np.testing.assert_array_equal(r.grid["median_radius"][:, 1], [0.1, 0.2, 0.3])
+    assert np.all(r.panels == r.panels[0, 0])  # one set of panels per wavelength
+    for i, rg in enumerate([0.1, 0.2, 0.3]):
+        for j, sg in enumerate([1.5, 1.7]):
+            a = compute(LogNormal(rg, sg), wl, 1.45 + 1e-3j, greek=8)
+            np.testing.assert_array_less(
+                np.abs(r.c_ext[i, j] - a.c_ext), 2 * (r.error.c_ext[i, j] + a.error.c_ext) + 1e-15
+            )
+            np.testing.assert_array_less(
+                np.abs(r.greek.alpha1[i, j] - a.greek.alpha1),
+                2 * (r.error.greek.alpha1[i, j] + a.error.greek.alpha1) + 1e-13,
+            )
+
+
+def test_grids_of_multimodal_distributions_and_power_law_limits():
+    bi = MultiModal(
+        [([1.0, 2.0, 3.0], LogNormal(0.1, 1.5)), (1.0, LogNormal(0.5, [1.4, 1.6, 1.8]))]
+    )
+    r = Ensemble(Sphere(1.5), bi).compute(0.5)
+    assert r.c_ext.shape == (3,)
+    assert set(r.grid) == {"mode0.weight", "mode1.width"}
+    one = compute(MultiModal([(2.0, LogNormal(0.1, 1.5)), (1.0, LogNormal(0.5, 1.6))]), 0.5, 1.5)
+    assert abs(r.c_ext[1] - one.c_ext) <= 2 * (r.error.c_ext[1] + one.error.c_ext)
+    p = Ensemble(Sphere(1.5), PowerLaw(3.0, 0.1, [0.5, 1.0])).compute([0.4, 0.5])
+    assert p.c_ext.shape == (2, 2) and np.all(p.c_ext[1] > p.c_ext[0])
+    # A table is data: Tabulated never makes a grid.
+    assert Ensemble(Sphere(1.5), Tabulated([0.1, 0.2, 0.3], [1.0, 2.0, 1.0])).grid_shape == ()
+
+
+def test_grid_errors():
+    with pytest.raises(ValueError, match="do not broadcast together"):
+        Ensemble(Sphere(1.5), LogNormal([0.1, 0.2], [1.5, 1.6, 1.7]))
+    grid = Ensemble(Sphere(1.5), LogNormal([0.1, 0.2], 1.5))
+    with pytest.raises(NotImplementedError, match="grid"):
+        grid.compute(0.5, derivatives=True)
+    with pytest.raises(NotImplementedError, match="grid"):
+        grid.compute(0.5, derivatives=["width"])
+    with pytest.raises(ValueError, match="a mixture needs scalar parameters"):
+        Mixture([(1.0, grid)])
+    with pytest.raises(InvalidInputError, match="needs a distribution"):
+        Ensemble(Sphere(1.5), LogNormal(np.zeros(0), 1.5)).compute(0.5)
+
+
+def test_repr_prints_no_arrays():
+    r = compute(LOGNORMAL, np.linspace(0.4, 0.8, 50), 1.5, greek=8, derivatives=["width"])
+    text = repr(r)
+    assert text == (
+        "EnsembleResult(shape=(50,), greek=(50, 9), params=('width',), converged=50/50)"
+    )
+    g = compute(LogNormal([0.1, 0.2], 1.5), 0.5, 1.5)
+    assert (
+        repr(g) == "EnsembleResult(shape=(2,), params=(), grid=('median_radius',), converged=2/2)"
+    )
diff --git a/tests/test_ensemble_engine_calls.py b/tests/test_ensemble_engine_calls.py
new file mode 100644
index 0000000..f4fb0ce
--- /dev/null
+++ b/tests/test_ensemble_engine_calls.py
@@ -0,0 +1,109 @@
+"""Ensemble and Mixture call the engine exactly as the functions they replaced did.
+
+The private bindings are unchanged, so these results are bitwise those of 0.1.0's
+``scatrs.ensemble.compute``, ``compute_batch`` and ``compute_mixture`` for the same inputs.
+"""
+
+import numpy as np
+
+from scatrs import Ensemble, Gamma, LogNormal, Mixture, MultiModal, Sphere, _scatrs
+
+WL = np.array([[0.35, 0.5], [0.75, 1.0]])
+MU = np.linspace(-1, 1, 7)
+OPTIONS = dict(rtol=1e-6, max_panels=4096, allow_unvalidated=False, refine_backscatter=False)
+
+
+def _engine_options():
+    return [
+        8,  # greek order
+        False,  # all orders
+        np.ascontiguousarray(MU),
+        True,  # normalized
+    ]
+
+
+def _equal(raw, result, shape, params=()):
+    names = (
+        "c_ext",
+        "c_sca",
+        "c_abs",
+        "c_back",
+        "ssa",
+        "g",
+        "lidar_ratio",
+        "depolarization",
+        "effective_radius",
+        "mean_area",
+        "mean_volume",
+    )
+    for key, outputs in [("values", result), ("errors", result.error)]:
+        for row, name in zip(raw[key], names, strict=True):
+            np.testing.assert_array_equal(getattr(outputs, name), row.reshape(shape))
+    for row, name in zip(
+        raw["greek"], ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2"), strict=True
+    ):
+        np.testing.assert_array_equal(getattr(result.greek, name), row.reshape((*shape, 9)))
+    for row, name in zip(raw["phase_errors"], ("f11", "f12", "f33", "f34"), strict=True):
+        np.testing.assert_array_equal(getattr(result.error.phase, name), row.reshape((*shape, 7)))
+    for k, p in enumerate(params):
+        for row, name in zip(raw["d"][k], names, strict=True):
+            np.testing.assert_array_equal(getattr(result.d(p), name), row.reshape(shape))
+        np.testing.assert_array_equal(
+            result.d(p).greek.beta2, raw["greek_d"][k][5].reshape((*shape, 9))
+        )
+    np.testing.assert_array_equal(result.converged, raw["converged"].reshape(shape))
+    np.testing.assert_array_equal(result.panels, raw["panels"].reshape(shape))
+
+
+def test_a_scalar_ensemble_is_the_single_distribution_call():
+    m = 1.45 + 2e-3j
+    size = MultiModal([(2.0, LogNormal(0.1, 1.5)), (1.0, Gamma(0.5, 0.1))])
+    r = Ensemble(Sphere(m), size).compute(
+        WL, greek=8, mu=MU, derivatives=["mode0.width", "m_imag"], **OPTIONS
+    )
+    items = [(size._spec(lambda n, v: float(v)), np.full(4, m), 1.0)]
+    names = _scatrs.ensemble_parameters(items, False)
+    raw = _scatrs.ensemble_compute(
+        items,
+        WL.reshape(-1),
+        False,
+        *_engine_options(),
+        [names.index("mode0.width"), names.index("m_imag")],
+        *OPTIONS.values(),
+    )
+    _equal(raw, r, WL.shape, ("mode0.width", "m_imag"))
+
+
+def test_a_grid_is_the_batch_call():
+    m = 1.45 + 2e-3j
+    radii, widths = np.array([0.1, 0.2, 0.3]), np.array([1.5, 1.7])
+    r = Ensemble(Sphere(m), LogNormal(radii[:, None], widths)).compute(
+        WL, greek=8, mu=MU, **OPTIONS
+    )
+    specs = [("lognormal", float(a), float(b)) for a in radii for b in widths]
+    raw = _scatrs.ensemble_compute_batch(
+        specs, np.full(4, m), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
+    )
+    _equal(raw, r, (3, 2, *WL.shape))
+
+
+def test_a_mixture_is_the_mixture_call():
+    a = Ensemble(Sphere(1.45), LogNormal(0.1, 1.5))
+    b = Ensemble(Sphere(1.53 + 0.01j), Gamma(1.0, 0.1))
+    r = Mixture([(2.0, a), (0.5, b)]).compute(
+        WL, greek=8, mu=MU, derivatives=["c1.weight"], **OPTIONS
+    )
+    items = [
+        (("lognormal", 0.1, 1.5), np.full(4, 1.45 + 0j), 2.0),
+        (("gamma", 1.0, 0.1), np.full(4, 1.53 + 0.01j), 0.5),
+    ]
+    names = _scatrs.ensemble_parameters(items, True)
+    raw = _scatrs.ensemble_compute(
+        items,
+        WL.reshape(-1),
+        True,
+        *_engine_options(),
+        [names.index("c1.weight")],
+        *OPTIONS.values(),
+    )
+    _equal(raw, r, WL.shape, ("c1.weight",))
diff --git a/tests/test_package.py b/tests/test_package.py
index 0b1883b..b7849f4 100644
--- a/tests/test_package.py
+++ b/tests/test_package.py
@@ -7,7 +7,7 @@ from pathlib import Path
 from packaging.version import Version
 
 import scatrs
-from scatrs import _scatrs, ensemble, mie, special
+from scatrs import _scatrs, ensemble, mie, particles, refractive, size, special
 
 ROOT = Path(__file__).parents[1]
 
@@ -48,8 +48,17 @@ def test_crate_license_is_the_repository_license():
 
 def test_public_names():
     for name in (
+        "Ensemble",
+        "Mixture",
+        "Sphere",
+        "LogNormal",
+        "Tabulated",
+        "EnsembleResult",
         "ensemble",
         "mie",
+        "particles",
+        "refractive",
+        "size",
         "special",
         "ScatrsError",
         "InvalidInputError",
@@ -58,9 +67,11 @@ def test_public_names():
         "__version__",
     ):
         assert hasattr(scatrs, name)
-    for module in (special, mie, ensemble):
+    for module in (scatrs, special, mie, ensemble, particles, refractive, size):
         for name in module.__all__:
             assert hasattr(module, name)
+    # The classes re-exported at the top level are those of their modules.
+    assert scatrs.LogNormal is size.LogNormal and scatrs.Ensemble is ensemble.Ensemble
 
 
 def test_limits_are_exposed():
diff --git a/tests/test_size.py b/tests/test_size.py
new file mode 100644
index 0000000..f092c42
--- /dev/null
+++ b/tests/test_size.py
@@ -0,0 +1,53 @@
+"""scatrs.size: distributions as values, array-valued parameters, grids."""
+
+from dataclasses import FrozenInstanceError
+
+import numpy as np
+import pytest
+
+from scatrs import LogNormal, MultiModal, PowerLaw, Tabulated, size
+
+
+def test_scalar_parameters_are_floats_and_arrays_read_only_copies():
+    d = LogNormal(np.float32(0.1), 2)
+    assert type(d.median_radius) is float and d.width == 2.0
+    radii = np.array([0.1, 0.2])
+    g = LogNormal(radii, 1.5)
+    radii[0] = 9.0
+    assert g.median_radius[0] == 0.1 and not g.median_radius.flags.writeable
+    with pytest.raises(FrozenInstanceError):
+        g.width = 1.6
+    with pytest.raises(TypeError, match="real numbers"):
+        LogNormal(0.1 + 1j, 1.5)
+
+
+def test_distributions_are_values():
+    t1 = Tabulated(np.array([0.1, 0.5]), [1.0, 2.0])
+    t2 = Tabulated([0.1, 0.5], (1.0, 2.0))
+    assert t1 == t2 and hash(t1) == hash(t2)
+    b = MultiModal([(1.0, LogNormal(0.08, 1.6)), (2.0, t1)])
+    assert hash(b) == hash(MultiModal([(1, LogNormal(0.08, 1.6)), (2, t2)]))
+    g1, g2 = LogNormal([0.1, 0.2], 1.5), LogNormal(np.array([0.1, 0.2]), 1.5)
+    assert g1 == g2 and hash(g1) == hash(g2)
+    assert g1 != LogNormal([0.1, 0.3], 1.5) and g1 != LogNormal([[0.1, 0.2]], 1.5)
+    assert LogNormal(0.1, 1.5) != PowerLaw(0.1, 1.5, 2.0)
+    with pytest.raises(TypeError, match="must be one of"):
+        MultiModal([(1.0, "x")])
+
+
+def test_grids_broadcast_and_flatten_in_c_order():
+    d = PowerLaw([[3.0], [4.0]], 0.1, [0.5, 1.0, 2.0])
+    shape, specs = size._specs(d)
+    assert shape == (2, 3) and len(specs) == 6
+    assert specs[1] == ("power_law", 3.0, 0.1, 1.0)
+    assert specs[3] == ("power_law", 4.0, 0.1, 0.5)
+    assert size._specs(LogNormal(0.1, 1.5)) == ((), [("lognormal", 0.1, 1.5)])
+    m = MultiModal([([1.0, 2.0], LogNormal(0.1, 1.5)), (1.0, LogNormal(0.5, 1.6))])
+    shape, specs = size._specs(m)
+    assert shape == (2,)
+    assert specs[1] == (
+        "multimodal",
+        [(2.0, ("lognormal", 0.1, 1.5)), (1.0, ("lognormal", 0.5, 1.6))],
+    )
+    shape, arrays = size._grid(m)
+    assert set(arrays) == {"mode0.weight"}
````

## Spec coverage

| Spec section | Task |
|---|---|
| §2 modules, §3.1 models, §3.2 `Sphere` | 1 |
| §3.3 distributions and grids, §3.4 `Ensemble`/`Mixture`, §4 computation, §5 results, §6 parameter names | 2 |
| §7 testing | 1 (models, `Sphere`), 2 (equivalence, grids, errors, objects, ports) |
| §8 docs and migration, §1 housekeeping (`.gitignore`) | 2 |
