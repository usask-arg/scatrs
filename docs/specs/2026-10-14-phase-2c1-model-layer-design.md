# Phase 2c-1 design: the model layer of the Python API

Status: approved 2026-10-14. Builds on phase 2a/2b (`docs/specs/2026-10-14-phase-2a-ensembles-design.md`,
`docs/specs/2026-10-14-phase-2b-speed-and-sasktran2-design.md`) and the main spec
(`docs/specs/2026-10-12-scatrs-design.md`, section 9.2).

Phase 2c (the high-level Python API) serves four goals: room for the particles of phases 3 and
4, retrievals (Jacobians), optics tables, and ease of use. It is split into three sub-phases,
each with its own spec and plan:

- **2c-1 (this spec):** particles, refractive-index models, `Ensemble`/`Mixture` objects, and
  parameter grids in place of `compute_batch`.
- **2c-2:** Jacobians: a stacked `.jacobian`, Jacobians of grids, derivatives with respect to
  refractive-index model parameters.
- **2c-3:** tables: `.to_xarray()` (netCDF through xarray) and thread control
  (`set_num_threads`, `num_threads`).

## 1. Scope

In:

- `scatrs.Sphere`, the module `scatrs.refractive` (`Constant`, `Table`, `Cauchy`, `Sellmeier`,
  `Function`), `scatrs.Ensemble`, `scatrs.Mixture`;
- the size distributions at the top level of `scatrs`, with array-valued parameters (grids);
- `Ensemble.compute` and `Mixture.compute`, replacing `scatrs.ensemble.compute`,
  `compute_mixture`, `compute_batch` and `Component`, which are removed;
- additions to `EnsembleResult` (section 5);
- tests, docs, the CHANGELOG, and the port of `benchmarks/compare.py`.

Out: everything in 2c-2 and 2c-3; grids of mixtures; grids over the refractive index (one
material per call, as the batch engine requires); a refractive-index database (main spec
section 1.2); any change to `scatrs.mie`; a release.

Compatibility: the 0.1.0 `scatrs.ensemble` API changes incompatibly (the user's choice: the
package is 0.x). The CHANGELOG lists each removed call with its replacement.

The work is in the Python layer only. The Rust core crate and the private binding functions
`_scatrs.ensemble_parameters`, `ensemble_compute` and `ensemble_compute_batch` do not change,
which is what makes the equivalence tests of section 7 bitwise.

Housekeeping: `/docs/build/` is added to `.gitignore` (a local Sphinx build directory that
shows up as untracked).

## 2. Modules

| File | Contents |
|---|---|
| `python/scatrs/size.py` | `LogNormal`, `Gamma`, `ModifiedGamma`, `PowerLaw`, `Monodisperse`, `Tabulated`, `MultiModal` |
| `python/scatrs/refractive.py` | `RefractiveIndex` (the protocol), `Constant`, `Table`, `Cauchy`, `Sellmeier`, `Function` |
| `python/scatrs/particles.py` | `Sphere` |
| `python/scatrs/ensemble.py` | `Ensemble`, `Mixture`, `EnsembleResult`, `EnsembleOutputs`, `EnsembleGreek`, `EnsemblePhase`, `SCALARS` |

`scatrs` re-exports the distributions, `Sphere`, `Ensemble`, `Mixture` and the result classes,
and exposes `scatrs.refractive` as a module (its `Table` would otherwise clash with the size
distribution `Tabulated` in readers' minds, and the models read well qualified:
`refractive.Table`). `scatrs.mie` and `scatrs.special` are unchanged.

## 3. Objects

All are frozen dataclasses, hashable and compared by value, except where noted.

### 3.1 Refractive-index models

A model maps wavelengths to m = n + ik (Im m ≥ 0, main spec section 3) and names its
parameters:

- `evaluate(wavelength) -> complex ndarray` of the wavelengths' shape;
- `parameters -> tuple[str, ...]`, the names of its differentiable parameters (possibly empty);
- `derivatives(wavelength) -> complex ndarray` of shape `(len(parameters),) + wavelength.shape`,
  ∂m/∂θ for each parameter.

2c-1 defines and tests `parameters` and `derivatives`; 2c-2 uses them for Jacobians.

| Model | m(λ) | Parameters |
|---|---|---|
| `Constant(m)` | m | `n`, `k` |
| `Table(wavelength, m, interpolation="linear")` | Re m and Im m linear in λ between the nodes; `"log_k"` interpolates ln Im m linearly instead (Im m > 0 required at every node) | none (the table is data) |
| `Cauchy(A, B=0.0, C=0.0, k=0.0)` | n = A + B/λ² + C/λ⁴, Im m = k | `A`, `B`, `C`, `k` |
| `Sellmeier(B, C, k=0.0)` | n² = 1 + Σᵢ Bᵢλ²/(λ² − Cᵢ), Im m = k; `B`, `C` sequences of equal length | `B{i}`, `C{i}`, `k` |
| `Function(f)` | `f(wavelength)` | none; compares by identity |

- `Table` requires at least two nodes, strictly increasing wavelengths, and stores its arrays as
  tuples (as `Tabulated` does) so that it hashes by value. A wavelength outside
  [λ₀, λₙ] raises `InvalidInputError` naming it: no extrapolation.
- Coefficients of `Cauchy` and `Sellmeier` are in the user's length unit (λ in the unit of the
  radii). The docs point out that published coefficients usually assume µm.
- Construction validates what it can (finite values, k ≥ 0, `B` and `C` of equal length,
  table shape and order); evaluation checks the result: complex, finite, of the wavelengths'
  shape, Im m ≥ 0. Failures raise `InvalidInputError` naming the model and, for evaluation,
  the first offending wavelength. `Sellmeier` raises if n² ≤ 0 at a wavelength (a pole or
  an absorption band the law does not describe).

### 3.2 Particles

`Sphere(refractive_index)`:

- a Python or NumPy complex/real scalar becomes `Constant`;
- a `RefractiveIndex` model is kept;
- any other callable becomes `Function`;
- anything else, including arrays, raises `TypeError`; for an array the message suggests
  `refractive.Table`.

Later phases add `LayeredSphere`, `Spheroid`, … next to it; `Ensemble` accepts any particle
type the engine supports (only `Sphere` in 2c-1).

### 3.3 Size distributions

The seven distributions keep their names, fields and meanings (2a spec section 3). Every
numeric field may be a scalar or array_like, including PowerLaw's limits and MultiModal's mode
weights and nested mode parameters; `Tabulated`'s `radius` and `density` are a table, not
parameters, and are never broadcast. Arrays are stored read-only so that the objects stay
immutable; distributions with array fields hash and compare by their values.

### 3.4 Ensembles and mixtures

- `Ensemble(particle, size)`.
- `Mixture(components)`: a sequence of `(weight, Ensemble)` pairs, weights the relative number
  concentrations (≥ 0, not all zero), as `Component.weight` was. Every ensemble of a mixture
  must have scalar parameters; a grid raises `ValueError`.

## 4. Computation

`Ensemble.compute(wavelength, *, greek=None, mu=None, normalized=True, derivatives=False,
rtol=1e-6, max_panels=4096, allow_unvalidated=False, refine_backscatter=False)`, and
`Mixture.compute` with the same signature. The arguments have the meanings and defaults of
the 0.1.0 functions.

1. The particle's model is evaluated at `wavelength` once (section 3.1 checks).
2. The distribution's fields broadcast together to the grid shape `G` (a `ValueError` names
   the fields that do not broadcast).
3. If `G == ()`: the single-distribution path (`_scatrs.ensemble_compute`), with derivatives.
4. If `G != ()`: the grid is flattened in C order into one distribution per point and computed
   by the shared-node batch engine (`_scatrs.ensemble_compute_batch`). Each point meets its
   own `rtol`; being at least as refined as on its own, its values differ from a scalar
   computation of that point within the error estimates (the docstring says so).
   `derivatives` other than `False` raise `NotImplementedError` (Jacobians of grids are planned).
5. Every result array has shape `G + wavelength.shape`, followed by `(L + 1,)` for Greek
   coefficients or `mu.shape` for the phase matrix.

`Mixture.compute` uses the mixture path (`_scatrs.ensemble_compute`, mixture mode).

## 5. Results

`EnsembleResult`, `EnsembleOutputs`, `EnsembleGreek` and `EnsemblePhase` keep their fields and
meanings. Added to `EnsembleResult`:

- `wavelength`: the wavelengths as computed (read-only, their original shape);
- `refractive_index`: m at those wavelengths; for a mixture, shape `(n_components,) +
  wavelength.shape`;
- `grid`: a mapping from each array-valued parameter's name (as in section 6) to its values
  broadcast to `G`; empty for `G == ()`;
- a `__repr__` that lists the outputs present with their shapes, `params`, `grid` names and
  whether every point converged, instead of printing arrays.

`d(param)` and `d_error(param)` are unchanged.

## 6. Parameter names

Unchanged from 0.1.0:

- distribution fields by name (`"median_radius"`, `"width"`, …); `MultiModal` modes as
  `"mode{j}.<name>"` and `"mode{j}.weight"`;
- `"m_real"`, `"m_imag"`: derivatives with respect to Re m and Im m at each wavelength,
  whatever model produced m;
- for `Mixture`, every name prefixed by `"c{i}."`, plus `"c{i}.weight"`.

Model parameters (`"index.<name>"`) and `.jacobian` come in 2c-2.

## 7. Testing

Tests are written first (`tests/`).

- **Equivalence, bitwise:**
  - a scalar `Ensemble` against `_scatrs.ensemble_compute` called as 0.1.0's `compute` did;
  - a grid against `_scatrs.ensemble_compute_batch` as `compute_batch` did;
  - a `Mixture` against `_scatrs.ensemble_compute` in mixture mode as `compute_mixture` did;
  - each with Greek coefficients, the phase matrix and (scalar and mixture) derivatives.
- **Refractive-index models:**
  - `Table` exact at its nodes, linear (or log-linear in k) between them;
  - out-of-range wavelengths raise;
  - `Cauchy` and `Sellmeier` against their formulas;
  - `Function` wrapping;
  - `derivatives` against central finite differences for every model with parameters;
  - each construction and evaluation error of section 3.1.
- **Grids:**
  - broadcast shapes, including MultiModal weights and nested parameters;
  - `Tabulated` tables are not broadcast;
  - each point of a small grid against a scalar computation within the error estimates;
  - `res.grid` contents.
- **Errors:**
  - a grid with derivatives raises `NotImplementedError`;
  - a mixture of grids raises `ValueError`;
  - an array as refractive index raises `TypeError`;
  - fields that do not broadcast raise `ValueError`.
- **Objects:** immutability, hashing, equality; `__repr__` prints no arrays.
- **Existing tests:** `tests/test_ensemble.py` and `tests/test_package.py` ported to the new
  API with their physics checks unchanged.

## 8. Docs and migration

- Updated to the new API: README, `docs/index.md`, quick start, `docs/examples/ensemble.md`,
  `docs/api.md`, `docs/theory/ensembles.md`, `docs/limitations.md`,
  `docs/validation/sasktran2.md`, and the type stub `python/scatrs/_scatrs/__init__.pyi` if
  its docstrings refer to the removed functions.
- A new page on refractive-index models: the models, interpolation, units of the
  coefficients, and the sign convention reminder (codes that write m = n − ik need the
  conjugate).
- `benchmarks/compare.py` ported; its scatrs results are checked to be bitwise unchanged.
- The docs stay user-facing: no phases or development history.
- CHANGELOG: an Unreleased section with **Breaking** (each removed call and its replacement)
  and **Added**.
- `tests/test_docs_consistency.py` keeps passing; extended if it checks API names.

## 9. Decisions

- Redesign rather than wrap the 0.1.0 API (user, 2026-10-14).
- Objects with `.compute()` plus array-valued parameters in place of `compute_batch` (user's
  choice of approaches A + C).
- All four kinds of refractive-index model (user).
- A bare array is not a refractive index: it means something only with a wavelength grid.
- Derivatives of grids raise instead of looping point by point; 2c-2 implements them.

## 10. Changes after approval

- **Lists of distributions (user, 2026-10-14).** The reviews found that grids cannot express
  0.1.0's batches of distributions of different kinds (or many `Tabulated` tables). `size=`
  also accepts a list (or tuple) of distributions with scalar parameters, stored as a tuple: a
  batch with one leading axis, computed by `_scatrs.ensemble_compute_batch` like a grid. As for
  grids: values only (`NotImplementedError` for derivatives), not allowed in a `Mixture`, an
  empty list is the engine's `InvalidInputError`, and `grid` is empty.
- **Changelog from GitHub releases (user, 2026-10-14).** As in sasktran2, the release notes of
  the GitHub releases are the changelog: `CHANGELOG.md` is removed, the docs gain a changelog
  page (sphinx-github-changelog), and `CONTRIBUTING.md`'s release steps say to write the notes
  on the release. The breaking changes of this work go in the next release's notes.
