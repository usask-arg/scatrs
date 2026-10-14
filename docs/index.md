# scatrs

Fast, linearized, rigorously validated single-scattering calculations for spherical
particles, written in Rust with a Python interface.

```{note}
scatrs is in early development and not yet published on PyPI or crates.io; see
{doc}`installation` to build it.
```

- **Mie scattering by homogeneous spheres** ({mod}`scatrs.mie`): Mie coefficients,
  efficiencies, asymmetry parameter, amplitudes, phase matrix and Greek coefficients to every
  order, vectorized over size parameter and refractive index.
- **Ensembles of spheres** ({mod}`scatrs.ensemble`): averages over seven kinds of size
  distribution and over external mixtures, at many wavelengths in parallel, with an error
  estimate for every output. Batches of distributions share their Mie solutions, for building
  optics databases.
- **Derivatives** of every output, in closed form: with respect to the size parameter and the
  refractive index, and for ensembles, the distribution parameters and the mixture weights.
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
theory/mie
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
```
