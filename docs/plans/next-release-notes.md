# Draft notes for the next release

The GitHub release notes are the changelog (see "Releasing" in `CONTRIBUTING.md`). Paste this
into the next release's notes, above GitHub's generated list, then delete this file.

## 0.3.0: layered spheres

### Breaking changes

- Rust: `scatrs::ensemble::Component`'s public field `refractive_index` is replaced by
  `particle`, a `scatrs::ensemble::Particle` (`Particle::Sphere { refractive_index }` or
  `Particle::LayeredSphere { radius_fractions, refractive_index }`). `Component::new(distribution,
  refractive_index, weight)` still builds a component of spheres; `Component::with_particle`
  takes any particle.
- Python: the `refractive_index` of a `Mixture`'s result is a tuple with each component's array
  (it was one stacked array, which components of different shapes no longer allow). Indexing
  it by component works as before.

### Added

- Layered spheres: a core and any number of concentric shells (coated spheres among them).
  - `scatrs.layered` (`coefficients`, `efficiencies`, `amplitudes`, `phase_matrix`, `greek`,
    `default_n_max`, `parameters`) and, in Rust, `scatrs::layered::Layered`, with every output
    of a sphere and closed-form derivatives with respect to each layer's size parameter and
    refractive index.
  - `scatrs.LayeredSphere(refractive_index=[...], radius_fraction=[...])` and
    `LayeredSphere.coated(core, shell, core_fraction=...)`, in `Ensemble`, `Mixture` and grids
    of distributions. The size distribution gives the outer radius, each interface at a fixed
    fraction of it. Derivatives: `layer{l}.m_real`, `layer{l}.m_imag` and
    `layer{l}.radius_fraction`.
  - Rust: `Particle::layered_sphere`, `compute_with_particle`, `compute_batch_with_particle`,
    `parameters_with_particle`, and `ParamKind::{LayerRefrReal, LayerRefrImag,
    RadiusFraction}`.
  - Validated against independent Arb references (direct interface matching) and dense
    double-double accuracy maps: outer size parameters 10⁻⁶ to 10⁴ with |m|·x ≤ 3·10⁴, up to 16
    layers, derivatives up to x = 1000. Compared with scattnlay.
  - Documentation: the theory chapter "Layered spheres", the validation pages, and the example
    "Coated black carbon".

## 0.2.0 (the notes of v0.2.0: delete this section once they are on its GitHub release)

### Breaking changes

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

### Added

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

### Changed

- The Rust crate no longer depends on faer (54 crates): the ensembles' matrix products use
  `matrixmultiply` (2 crates), at the same speed on the machines measured. The crate's lint is
  `#![deny(unsafe_code)]` rather than `forbid`, for the one call into `matrixmultiply`.
  Ensemble results change at the rounding level.
