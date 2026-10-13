# scatrs

Fast, linearized, rigorously validated single-scattering calculations: Lorenz–Mie, layered
spheres and axisymmetric T-matrix, written in Rust with a Python interface.

```{warning}
Pre-alpha. Phase 0 provided the numerical foundations: the scalar types, Gauss–Legendre
quadrature, Wigner d-functions and the Arb reference suite. Phase 1 adds Mie scattering by
homogeneous spheres, with derivatives. Nothing is published on PyPI or crates.io yet.
```

What exists today:

- **The Rust crate `scatrs`.** The `Real` trait, which every numerical kernel is generic over,
  and its three implementations: `f64`, a double-double type (`DoubleDouble`, about 32
  significant digits) and forward-mode dual numbers (`Dual<T, N>`). Also two special
  functions, `gauss_legendre` and `wigner_d`. See {doc}`theory/scalar-types` and
  {doc}`theory/special-functions`.
- **Mie scattering.** `scatrs::mie` (Python {mod}`scatrs.mie`): coefficients, efficiencies,
  asymmetry parameter and amplitudes of homogeneous spheres, with closed-form derivatives with
  respect to the size parameter and refractive index, and a declared, tested accuracy over a
  validated domain. See {doc}`theory/mie` and {doc}`validation/accuracy`.
- **The Python package `scatrs`.** {mod}`scatrs.mie`, {func}`scatrs.special.gauss_legendre`
  and {func}`scatrs.special.wigner_d`, and the exceptions in {mod}`scatrs.errors`. See
  {doc}`api`.
- **The reference suite.** Values computed independently with Arb ball arithmetic, each with
  a rigorous error bound. The tests check the accuracy of `DoubleDouble` and of the special
  functions against them. See {doc}`validation/reference-suite`.

Every accuracy figure in these pages says what backs it. In short, the double-double basic
operations have proved bounds; the bounds of the double-double functions, Gauss–Legendre and
the Wigner d-functions are tested against Arb (and the Wigner ones are measured, not derived);
other figures are measurements. The Mie tolerances are measured and enforced in CI against Arb
and a dense double-double map. Python exposes only `f64` results. See
{ref}`what-backs-the-numbers` and the rest of {doc}`limitations`.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
conventions
examples/mie
examples/quadrature
examples/wigner-d
examples/henyey-greenstein
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/mie
theory/special-functions
theory/scalar-types
```

```{toctree}
:maxdepth: 2
:caption: Validation

validation/reference-suite
validation/accuracy
limitations
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
references
```
