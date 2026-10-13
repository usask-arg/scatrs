# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Workspace, `Real` scalar trait and error type.
- `DoubleDouble` (~32 digits) and forward-mode `Dual<T, N>` scalar types implementing `Real`.
- Gauss–Legendre quadrature (computed in double-double) and Wigner d-functions.
- Arb reference-value generator and certified reference data.
- Python package `scatrs` with `special.gauss_legendre` and `special.wigner_d`.
- Documentation site, CI on Linux/macOS/Windows, wheel release workflow.
- Mie scattering by homogeneous spheres (`scatrs::mie`, Python `scatrs.mie`): coefficients,
  efficiencies, asymmetry parameter and amplitudes, with closed-form derivatives with respect
  to the size parameter and the refractive index; absorption without cancellation; a declared
  accuracy over a validated domain, enforced in CI against Arb references and a dense
  double-double accuracy map.
- Arb reference data for Mie scattering, certified with a proven series-tail bound.
