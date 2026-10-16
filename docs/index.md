# scatrs

Fast, linearized, rigorously validated single-scattering calculations for spherical and
spheroidal particles, written in Rust with a Python interface.

scatrs is built out of the scattering core of the
[sasktran2](https://github.com/usask-arg/sasktran2) radiative transfer model, as a standalone
library for the optical properties of particles. Install it with `pip install scatrs` or
`cargo add scatrs` ({doc}`installation`).

- **Mie scattering by homogeneous spheres** ({mod}`scatrs.mie`): Mie coefficients,
  efficiencies, asymmetry parameter, amplitudes, phase matrix and Greek coefficients to every
  order, vectorized over size parameter and refractive index.
- **Layered spheres** ({mod}`scatrs.layered`): a core and any number of concentric shells,
  coated spheres among them, with the same outputs and derivatives with respect to every
  layer.
- **Randomly oriented spheroids** ({mod}`scatrs.tmatrix`): efficiencies, Greek coefficients
  and the six-element phase matrix of oblate and prolate spheroids, by the T-matrix method,
  converged to a tolerance with an error estimate, and their derivatives.
- **Ensembles** of spheres, layered spheres and spheroids ({class}`scatrs.Ensemble`,
  {class}`scatrs.Mixture`): averages over seven kinds of size distribution and over external
  mixtures, at many wavelengths in parallel, with an error estimate for every output, and refractive indices that vary with
  wavelength ({mod}`scatrs.refractive`). Grids of distributions share their Mie solutions, for
  building optics tables.
- **Derivatives** of every output, in closed form: with respect to a sphere's size parameter
  and refractive index, to every layer's of a layered sphere, and for ensembles, the
  distribution parameters, a layered sphere's radius fractions and the mixture weights.
- **Validated accuracy.** Every output has a declared accuracy, checked against independent
  reference values with rigorous error bounds across the whole validated domain, and against
  published results. See {doc}`validation/overview`.

To get started, read {doc}`quickstart`, then the examples. The validation pages explain why
the results can be trusted, and the theory pages describe the methods in full.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Examples

examples/mie
examples/greek
examples/ensemble
examples/distributions
examples/coated
examples/spheroids
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Validation

validation/overview
validation/accuracy
validation/external
validation/sasktran2
limitations
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Theory

theory/conventions
theory/refractive-index
theory/mie
theory/layered
theory/spheroids
theory/ensembles
theory/special-functions
theory/scalar-types
theory/reference-suite
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Internal utilities

utilities/quadrature
utilities/wigner-d
utilities/legendre-moments
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api
rust-api
references
changelog
```
