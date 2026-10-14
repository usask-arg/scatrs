# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
