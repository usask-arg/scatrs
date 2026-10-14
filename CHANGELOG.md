# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Workspace, `Real` scalar trait and error type.
- `DoubleDouble` (~32 digits) and forward-mode `Dual<T, N>` scalar types implementing `Real`.
- Gauss–Legendre quadrature and Wigner d-functions.
- Arb reference-value generator and certified reference data.
- Python package `scatrs` with `special.gauss_legendre` and `special.wigner_d`.
- Documentation site, CI on Linux/macOS/Windows, wheel release workflow.
- Mie scattering by homogeneous spheres (`scatrs::mie`, Python `scatrs.mie`): coefficients,
  efficiencies, asymmetry parameter and amplitudes, with closed-form derivatives with respect
  to the size parameter and the refractive index; absorption without cancellation; a declared
  accuracy over a validated domain, enforced in CI against Arb references and a dense
  double-double accuracy map.
- Arb reference data for Mie scattering, certified with a proven series-tail bound (the series
  of derivatives and the largest |m|x cases are high-precision).
- O(n) Gauss–Legendre rules for `f64` precision above 100 points (Bogaert's asymptotic
  expansions), and `gauss_legendre_into`, which writes a rule into caller-supplied buffers.
- The phase matrix of homogeneous spheres (F₁₁, F₁₂, F₃₃, F₃₄; normalized or per
  unit geometric cross section), with derivatives with respect to the size parameter and
  the refractive index.
- Greek expansion coefficients of the sphere phase matrix, exact up to rounding and up to every
  order, with derivatives, and `GreekProjector`, which projects any phase-matrix tables given
  at its Gauss–Legendre nodes; Arb references and an accuracy map for both.
- Python `scatrs.mie.phase_matrix` and `scatrs.mie.greek`.

### Changed
- Gauss–Legendre rules of `f64` precision above 100 points come from the asymptotic
  expansions, within 32·2⁻⁵² (measured 2.8 and 4.1 units of 2⁻⁵² for nodes and weights),
  instead of the double-double Newton rule rounded to within half an ulp. The Newton rule
  remains up to 100 points and for double-double precision.
- `GaussLegendre` is `#[non_exhaustive]`: code outside the crate can no longer build one with a
  struct literal.
