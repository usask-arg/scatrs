# scatrs

Fast, linearized, rigorously validated single-scattering calculations: Lorenz–Mie, layered
spheres and axisymmetric T-matrix, written in Rust with a Python interface.

```{warning}
Pre-alpha. Phase 0 provides the numerical foundations: the scalar types, Gauss–Legendre
quadrature, Wigner d-functions and the Arb reference suite. There are no scattering
calculations yet; they arrive in phase 1. Nothing is published on PyPI or crates.io yet.
```

What exists today:

- **The Rust crate `scatrs`.** The `Real` trait, which every numerical kernel is generic over,
  and its three implementations: `f64`, a double-double type (`DoubleDouble`, about 32
  significant digits) and forward-mode dual numbers (`Dual<T, N>`). Also two special
  functions, `gauss_legendre` and `wigner_d`. See {doc}`theory/scalar-types` and
  {doc}`theory/special-functions`.
- **The Python package `scatrs`.** {func}`scatrs.special.gauss_legendre` and
  {func}`scatrs.special.wigner_d`, and the exceptions in {mod}`scatrs.errors`. See
  {doc}`api`.
- **The reference suite.** Values computed independently with Arb ball arithmetic, each with
  a rigorous error bound. The tests check the accuracy of `DoubleDouble` and of the special
  functions against them. See {doc}`validation/reference-suite`.

Every accuracy figure in these pages says what backs it: a proof or error analysis, a bound
that the tests enforce (against the reference values or other high-precision results), or a
measurement. {doc}`limitations` lists what is known
not to work yet.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
conventions
examples/quadrature
examples/wigner-d
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/special-functions
theory/scalar-types
```

```{toctree}
:maxdepth: 2
:caption: Validation

validation/reference-suite
limitations
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
