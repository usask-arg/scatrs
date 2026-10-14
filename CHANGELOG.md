# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking

The functions of `scatrs.ensemble` are replaced by objects. Each removed call and its
replacement:

- `ensemble.compute(dist, wavelength, m, ...)` →
  `scatrs.Ensemble(scatrs.Sphere(m), dist).compute(wavelength, ...)`.
- `ensemble.compute_mixture([ensemble.Component(dist, m, w), ...], wavelength, ...)` →
  `scatrs.Mixture([(w, scatrs.Ensemble(scatrs.Sphere(m), dist)), ...]).compute(wavelength, ...)`.
- `ensemble.compute_batch([LogNormal(r1, s1), LogNormal(r2, s2), ...], wavelength, m, ...)` →
  `scatrs.Ensemble(scatrs.Sphere(m), scatrs.LogNormal([r1, r2, ...], [s1, s2, ...])).compute(wavelength, ...)`:
  array-valued distribution parameters broadcast into a grid of distributions.
- `ensemble.Component` is removed; the size distributions move from `scatrs.ensemble` to
  `scatrs.size` and the top level (`scatrs.LogNormal`, …).
- A refractive index given as an array of values per wavelength is no longer accepted: use
  `scatrs.refractive.Table` (or `scatrs.refractive.Function`).

The keyword options, the outputs, their values and the parameter names of `derivatives` are
unchanged.

### Added

- `scatrs.Sphere`, `scatrs.Ensemble` and `scatrs.Mixture`: immutable, hashable descriptions of
  particles, ensembles and mixtures, each ensemble and mixture with a `compute` method.
- `scatrs.refractive`: refractive-index models `Constant`, `Table` (linear, or log-linear in
  k), `Cauchy`, `Sellmeier` and `Function`, with the names of their parameters and ∂m/∂θ.
- Grids of distributions: any numeric distribution parameter can be an array; the grid is
  computed at once with shared size nodes, and the results have the grid's axes first.
- `EnsembleResult.wavelength`, `.refractive_index` (the m used) and `.grid` (the array-valued
  parameters, broadcast), and a short `repr`.

## [0.1.0] - Unreleased

The first release: a standalone library built out of the scattering core of the
[sasktran2](https://github.com/usask-arg/sasktran2) radiative transfer model, published as the
Python package `scatrs` (PyPI) and the Rust crate `scatrs` (crates.io).

### Added

- **Mie scattering by homogeneous spheres** (`scatrs::mie`, Python `scatrs.mie`): the Mie
  coefficients, efficiencies, asymmetry parameter, amplitudes, phase matrix (F₁₁, F₁₂, F₃₃,
  F₃₄; normalized or per unit geometric cross section) and Greek coefficients, exact up to
  rounding and up to every order. Closed-form derivatives of every output with respect to the
  size parameter and the real and imaginary parts of the refractive index. Absorption is
  computed without the cancellation of Q_ext − Q_sca, so weakly absorbing spheres keep full
  precision. `GreekProjector` projects any phase-matrix tables given at its Gauss–Legendre
  nodes onto Greek coefficients, with `project_bounds` for bounds on their errors.
- **Ensembles of spheres** (`scatrs::ensemble`, Python `scatrs.ensemble`): seven size
  distributions (lognormal, gamma, modified gamma, power law, monodisperse, tabulated,
  multimodal) and external mixtures, averaged at many wavelengths by adaptive Gauss–Kronrod
  integration, with an error estimate for every output and derivatives with respect to the
  distribution parameters, the refractive indices and the mixture weights. Outputs: cross
  sections, single-scattering albedo, asymmetry parameter, backscattering, lidar and
  depolarization ratios, moments, the phase matrix at given angles and the Greek
  coefficients. `compute_batch` averages over many distributions of one material on shared
  size nodes, for optics databases. Parallel over wavelengths and size panels, with results
  bitwise identical for any number of threads.
- **Declared accuracy.** Every Mie output has a declared tolerance over a validated domain
  (10⁻⁶ ≤ x ≤ 2·10⁴, 0 < Re m ≤ 10, 0 ≤ Im m ≤ 10, |m| ≥ 0.1), enforced in CI against
  arbitrary-precision (Arb) reference values and a dense double-double accuracy map; the
  ensembles' error estimates are calibrated in CI against tight-tolerance integrals. Inputs
  outside the validated domain raise `OutsideValidatedDomain` unless explicitly allowed.
- **Special functions** (`scatrs::special`, Python `scatrs.special`): Gauss–Legendre rules
  (O(n) asymptotic expansions above 100 points in `f64`) and Wigner d-functions.
- **Scalar types** (Rust): the `Real` trait that the numerical kernels are generic over,
  `DoubleDouble` (about 32 significant digits) and forward-mode `Dual<T, N>` numbers.
- **Documentation**: a quick start, examples, the validation of the results (including
  Wiscombe's MIEV0 test cases and a comparison with sasktran2), the theory behind every method,
  and the Python and Rust API references.
- Wheels for Linux, macOS and Windows (abi3, Python 3.11 and later).

[Unreleased]: https://github.com/usask-arg/scatrs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/usask-arg/scatrs/releases/tag/v0.1.0
