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
