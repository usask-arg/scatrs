# Draft notes for the next release

The GitHub release notes are the changelog (see "Releasing" in `CONTRIBUTING.md`). Paste this
into the next release's notes, above GitHub's generated list, then delete this file.

## Breaking changes

The functions of `scatrs.ensemble` are replaced by objects. Each removed call and its
replacement:

- `ensemble.compute(dist, wavelength, m, ...)` →
  `scatrs.Ensemble(scatrs.Sphere(m), dist).compute(wavelength, ...)`
- `ensemble.compute_mixture([ensemble.Component(dist, m, w), ...], wavelength, ...)` →
  `scatrs.Mixture([(w, scatrs.Ensemble(scatrs.Sphere(m), dist)), ...]).compute(wavelength, ...)`
- `ensemble.compute_batch([d1, d2, ...], wavelength, m, ...)` →
  `scatrs.Ensemble(scatrs.Sphere(m), [d1, d2, ...]).compute(wavelength, ...)`; for one family's
  parameters a grid also works: `scatrs.LogNormal([r1, r2, ...], [s1, s2, ...])`.
- `ensemble.Component` is removed; the size distributions move to `scatrs.size` and the top
  level (`scatrs.LogNormal`, …).
- A refractive index given as an array of values per wavelength is no longer accepted: use
  `scatrs.refractive.Table` (or `scatrs.refractive.Function`).

The keyword options, the outputs, their values and the parameter names of `derivatives` are
unchanged.

## Added

- `scatrs.Sphere`, `scatrs.Ensemble` and `scatrs.Mixture`: immutable, hashable, picklable
  descriptions of particles, ensembles and mixtures, each ensemble and mixture with a
  `compute` method.
- `scatrs.refractive`: refractive-index models `Constant`, `Table` (linear, or log-linear in
  k), `Cauchy`, `Sellmeier` and `Function`, with the names of their parameters and ∂m/∂θ.
- Grids and lists of distributions: array-valued distribution parameters broadcast into a grid,
  and a list of distributions of any kinds is a batch; both are computed at once with shared
  size nodes, the results with the grid's or list's axes first.
- `EnsembleResult.wavelength`, `.refractive_index` (the m used) and `.grid` (the array-valued
  parameters, broadcast), and a short `repr`.
- A changelog page in the documentation, built from these release notes.
- Rust: `scatrs::mie::AngularTable` and the `MieSolution::*_from_table` methods (amplitudes,
  phase matrix, and their derivatives) evaluate the angular functions once for many spheres at
  the same angles, bitwise as the methods that take the cosines; amplitudes at 100 angles are
  about 3.7 times faster for x ≥ 100. Python's amplitude and phase-matrix functions use one
  for several spheres.

## Changed

- The Rust crate no longer depends on faer (54 crates): the ensembles' matrix products use
  `matrixmultiply` (2 crates), at the same speed on the machines measured. The crate's lint is
  `#![deny(unsafe_code)]` rather than `forbid`, for the one call into `matrixmultiply`.
  Ensemble results change at the rounding level.
