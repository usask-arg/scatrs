# scatrs

Fast, linearized, rigorously validated single-scattering calculations: Lorenz–Mie, layered
spheres and axisymmetric T-matrix, with a Python interface. The name reads as both
"scatterers" and "scat-rs".

**Status:** pre-alpha. Phase 1: Lorenz–Mie scattering by homogeneous spheres, with
closed-form derivatives and a validated, CI-enforced accuracy, including the phase matrix and
exact Greek coefficients (phase 1b); size distributions and mixtures of spheres, with error
estimates and derivatives (phase 2a). Layered spheres and the T-matrix are still to come.

- Design: [`docs/specs/2026-10-12-scatrs-design.md`](https://github.com/usask-arg/scatrs/blob/main/docs/specs/2026-10-12-scatrs-design.md)
- Development: see [`CONTRIBUTING.md`](https://github.com/usask-arg/scatrs/blob/main/CONTRIBUTING.md)

License: MIT.
