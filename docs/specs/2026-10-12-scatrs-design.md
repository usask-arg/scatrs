# scatrs — design specification

- **Date:** 2026-10-12
- **Status:** approved design, pre-implementation
- **Scope of this spec:** milestone 1 (core, homogeneous Mie, layered spheres, axisymmetric T-matrix)

## 1. Purpose

scatrs is a Rust library with a Python interface for single-scattering properties of particles.
Its name reads as both "scatterers" and "scat-rs".

Its distinguishing features are:

1. **First-class linearization.** Every output has a Jacobian with respect to particle, material, shape, orientation-distribution and size-distribution parameters.
2. **Speed as a primary requirement.**
3. **Validation against reference values with proven error bounds.**

It will eventually replace the Mie code in sasktran2. It is intended to be general enough for any application that needs these calculations: remote sensing retrievals, radar and lidar, climate-model optics tables, lab instruments, holography, nanophotonics, astrophysics.

### 1.1 Milestone 1 includes

- Shared core: scalar types, special functions, scattering representations, Jacobians, ensembles.
- Homogeneous spheres (Lorenz–Mie) with hand-derived analytic derivatives.
- Layered spheres (any number of layers).
- Axisymmetric T-matrix (EBCM) with random orientation, fixed orientation and orientation distributions.
- Size and shape distributions, adaptive integration, external mixtures.
- Python package: low-level vectorized kernels and a high-level declarative layer.
- Validation infrastructure: Arb reference suite, accuracy maps, invariant tests, derivative checks, cross-code comparisons, fuzzing.
- Benchmarks, CI, wheels, docs, release 0.1.0.

### 1.2 Non-goals for milestone 1

- **Integration with sasktran2.** The API must make it possible (section 11); the linking work happens later.
- **Geometric optics.** This is the next milestone, with its own spec.
- **Non-axisymmetric particles** (hexagonal prisms, triaxial ellipsoids, meshes), aggregates, DDA.
- **Delta-M or other phase-function truncation.** This belongs in RT solvers.
- **A refractive-index database.** Users supply m(λ).
- **GPU execution.**
- **Runtime interval-arithmetic error bounds.**

## 2. Naming

| Item | Name |
|---|---|
| Rust core crate (crates.io) | `scatrs` |
| Python distribution (PyPI) | `scatrs` |
| Python import | `scatrs` |
| Python extension module | `scatrs._scatrs` |
| GitHub repository | `usask-arg/scatrs` |
| Local folder | `scat-rs` (unchanged; nothing depends on it) |

`scatrs` was unregistered on both crates.io and PyPI on 2026-10-12.

## 3. Conventions

- **Refractive index.** m = n + ik, where Im m ≥ 0 means absorbing (the Bohren–Huffman and Mishchenko convention). Negative Im m is rejected with `InvalidInput`; it is never silently flipped.
- **Lengths.** Radii and wavelength may be in any unit, provided both use the same one. Cross sections are returned in that unit squared.
- **Phase-matrix normalization.** ½ ∫ F₁₁(μ) dμ = 1.
- **Greek coefficients.** F₁₁(Θ) = Σₗ α₁,ₗ Pₗ(cos Θ), with α₁,₀ = 1 and α₁,₁ = 3g. The remaining elements follow Hovenier, van der Mee & Domke (2004):
  - α₂ ± α₃ against d^l₂,±₂;
  - α₄ against Pₗ;
  - β₁ and β₂ against d^l₀,₂.

  The theory chapter of the docs gives the exact formulas and sign conventions, and a test checks the round trip from angular values to Greek coefficients and back.
- **Random-orientation phase matrix.** Six independent elements: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄. This assumes the ensemble is macroscopically isotropic and mirror-symmetric.
- **Fixed-orientation geometry.** Euler angles (α, β, γ) in the z-y-z convention. Incidence and scattering directions (θ, φ) are in the laboratory frame. Both follow Mishchenko's conventions.
- **Spheroid aspect ratio.** ε is the equatorial semi-axis divided by the polar semi-axis: ε > 1 is oblate, ε < 1 is prolate.
- **Cylinder aspect ratio.** Diameter divided by length.
- **Chebyshev particle.** r(θ) = r₀(1 + ε cos nθ), with integer n ≥ 2 and |ε| < 1.
- **Size of non-spherical particles.** The radius of the sphere with the same volume, or the same surface area, as the user selects.

## 4. Architecture

### 4.1 Workspace layout

```
scat-rs/                    virtual Cargo workspace (edition 2024)
  scatrs/                   crate `scatrs`: core library, published, no pyo3
    src/scalar/             Real trait; f64, DoubleDouble, Dual<T, N>
    src/special/            Riccati–Bessel, logarithmic derivative, Wigner d/D,
                            associated Legendre, Gauss–Legendre / Gauss–Kronrod
    src/mie/                homogeneous and layered spheres
    src/tmatrix/            EBCM: shapes, Q/RgQ, solve, convergence, orientation
    src/scattering/         amplitude → Mueller/phase matrix, Greek expansion,
                            cross sections, lidar quantities
    src/ensemble/           distributions, adaptive integration, mixtures
    src/jacobian.rs         parameter identifiers, Jacobian block layout
    benches/                criterion and instruction-count benchmarks
  scatrs-py/                pyo3 cdylib → scatrs._scatrs (not published to crates.io)
  python/scatrs/            Python package (maturin python-source = "python")
  reference/                Arb reference generator (python-flint) + committed data
  validation/               cross-code comparison scripts and stored fixtures
  benchmarks/               cross-code performance comparison (compare.py)
  docs/                     Sphinx site; docs/specs/ holds design documents
```

### 4.2 Scalar abstraction

- Every numerical kernel is generic over a `Real` trait. The trait is compatible with `num_traits::Float`, so `num_complex::Complex<T>` works for every `T: Real`.
- There are three implementations:
  - `f64`;
  - `DoubleDouble`, implemented inside scatrs: about 32 significant digits, with error-free transformations using FMA, and `sqrt`/`exp`/`ln`/`sin`/`cos` validated against Arb;
  - `Dual<T, const N: usize>`, forward-mode dual numbers over any `T: Real`.
- The same kernel therefore yields values, values with derivatives, extended-precision values, or extended-precision values with derivatives.

### 4.3 Derivative strategy

Derivative computation for each method goes through a strategy trait, selected per method at compile time:

- **`Analytic`:** hand-derived derivatives. This is the default for homogeneous Mie.
- **`Dual`:** forward-mode automatic differentiation through the generic kernel. This is the default for layered spheres and the T-matrix.

A hand-derived implementation (in the style of Spurr et al. 2012) can replace any `Dual` path later without changing the API. When it does, the `Dual` implementation stays as its test reference.

### 4.4 Core data types

- **Particles:** `Sphere`, `LayeredSphere`, `Spheroid`, `Cylinder`, `Chebyshev`, `AxisymmetricProfile` (a sampled r(θ) interpolated by spline).
- **Single-particle outputs:**
  - Mie coefficients aₙ, bₙ;
  - T-matrix blocks Tᵐ;
  - amplitude matrix (S₁–S₄);
  - 4×4 Mueller matrix;
  - 4×4 extinction matrix;
  - C_ext, C_sca, C_abs.
- **Ensemble outputs:** listed in section 7.4.
- **Jacobians.** Every output type has a matching Jacobian type, stored as one contiguous block per requested parameter.
- **Parameter identifiers.** Parameters are identified by a `Param` enum, for example:
  - `Radius`, `LayerRadius(i)`;
  - `RefrReal(layer)`, `RefrImag(layer)`;
  - `EquivRadius`, `AspectRatio`, `ChebyshevEps`;
  - `Orient(..)`, `Dist(..)`, `MixWeight(i)`.

### 4.5 API rules for the core crate

- **Inputs and outputs.** Kernels take plain slices and scatrs types, and write into buffers the caller supplies, described by explicit shape and stride. Convenience functions that allocate their own output are layered on top.
- **No ndarray or pyo3 in the core public API.** Downstream crates (sasktran2) can then use any ndarray or pyo3 version.
- **No global thread pool.** Callers may pass a `rayon::ThreadPool`; otherwise the current pool is used.
- **Determinism.** Results are bitwise identical for any thread count. Reductions happen in a fixed order.
- **Errors.** One `scatrs::Error` (via `thiserror`) with variants:
  - `InvalidInput`;
  - `NotConverged { achieved, requested }`;
  - `OutsideValidatedDomain`.

  Computing outside the validated domain is an error unless the caller sets `allow_unvalidated`.
- **No panics.** No finite input may cause a panic or a silent NaN. Every input either gives a result or returns a typed error.
- **Dependencies.** The core depends only on `num-complex`, `num-traits`, `rayon` and `thiserror`, and declares `#![forbid(unsafe_code)]`.
  - Linear algebra is a generic LU with partial pivoting, implemented inside scatrs.
  - `faer` may be added as an f64 fast path, but only if benchmarks justify it.
  - A SIMD crate with a safe API (such as `pulp`) may be added under the same condition.

## 5. Methods

### 5.1 Homogeneous spheres

Amended by the phase 1 design (`docs/specs/2026-10-13-phase-1-mie-design.md`, sections 11
and 12): no Lentz continued fraction and no small-particle expansion.

- **Baseline:** Wiscombe's MIEV0 scheme.
  - Truncation: N = x + 4.05 x^⅓ + 2.
  - Dₙ(mx) by downward recurrence, seeded with a Lentz continued fraction. Upward recurrence is used where Wiscombe's criterion allows it.
  - ψₙ and ξₙ by upward recurrence.
  - The small-particle expansion for |m|x < 0.1.
- **Changes driven by the accuracy maps.** Each element of the baseline is kept only if the accuracy maps (section 8.2) confirm it. Wherever it loses digits, a more stable formulation replaces it — for example, a downward ratio recurrence for ψₙ when n > x.
- **Derivatives** are analytic closed forms built from quantities the recurrences already compute. Examples:
  - D′ₙ(z) = n(n+1)/z² − 1 − Dₙ²;
  - ψ′ₙ = ψₙ₋₁ − nψₙ/x.
- **Initial validated domain:** x ∈ [10⁻⁶, 2×10⁴], Re m ∈ (0, 10], Im m ∈ [0, 10]. The final domain is whatever the accuracy maps support.

### 5.2 Layered spheres

- **Algorithm:** the stable recursion of Yang (2003) and Peña & Pal (2009), the one scattnlay uses. Any number of layers is supported.
- **Derivatives:** with respect to each layer's radius and refractive index, via `Dual`.
- **Required identities:**
  - one layer equals homogeneous Mie;
  - layers with identical m equal a homogeneous sphere with the outer radius.

### 5.3 Axisymmetric T-matrix (EBCM)

- **Construction.** For each azimuthal mode m, Qᵐ and RgQᵐ come from Gauss quadrature in θ, halved when the particle is mirror-symmetric. Then Tᵐ = −RgQᵐ (Qᵐ)⁻¹.
- **Convergence.**
  - First, increase N_max until C_ext and C_sca converge to the requested tolerance (default 10⁻⁶ relative).
  - Then, increase the number of θ quadrature points until they converge again.
  - The achieved accuracy is always reported.
  - If convergence fails, the result is `NotConverged`; an unconverged result is never returned silently.
- **Precision.**
  - The f64 version is tried first.
  - If ill-conditioning is detected (convergence stalls or the condition estimate exceeds a threshold), the calculation is retried automatically in `DoubleDouble`.
  - The user can force either precision.
- **Derivatives:** `Dual` is propagated through the Qᵐ and RgQᵐ integrands, then dTᵐ = −(dRgQᵐ + Tᵐ dQᵐ)(Qᵐ)⁻¹, reusing one LU factorization per m for values and all derivatives.
- **Orientation.**
  - **Random:** Mishchenko's analytic orientation averaging. Cross sections and Greek coefficients come out directly.
  - **Fixed:** the T-matrix is rotated with Wigner D-functions to give the amplitude, Mueller and extinction matrices for given Euler angles and directions.
  - **Distribution:** quadrature over Euler angles weighted by a parametric pdf (for example Gaussian canting in β), including forward and backward amplitudes for radar quantities.
- **Required identity:** a sphere's T-matrix is diagonal and equals the Mie coefficients (up to the convention's sign).

### 5.4 Exact Greek coefficients for spheres

- S₁ and S₂ are polynomials of degree ≤ N in μ.
- Every phase-matrix element is therefore a polynomial of degree ≤ 2N.
- Every generalized spherical function used in the expansion is a polynomial of degree ≤ l.
- Gauss–Legendre quadrature with n_q ≥ N + ⌈(L+1)/2⌉ nodes is therefore exact (apart from rounding) for all expansion coefficients up to order L.
- scatrs chooses n_q from this bound automatically.

## 6. Linearization

### 6.1 Differentiable parameters

| Particle or component | Parameters |
|---|---|
| Sphere | r (or x), Re m, Im m |
| Layered sphere | rᵢ, Re mᵢ, Im mᵢ for each layer |
| T-matrix shapes | equivalent radius; continuous shape parameters (aspect ratio, Chebyshev ε); Re m, Im m |
| Orientation distributions | their parameters (e.g. canting width σ) |
| Size and shape distributions | declared by each type (section 7.1) |
| External mixtures | relative number concentrations wᵢ |

- **Not differentiable:** integer parameters such as Chebyshev n.
- **Wavelength** is not a parameter. The low-level API exposes ∂/∂x and ∂/∂m so that users can apply the chain rule themselves.

### 6.2 Outputs

- Every output has a Jacobian.
- The phase matrix and Greek coefficients are available in two forms, each with its own Jacobian:
  - normalized (α₁,₀ = 1);
  - unnormalized (C_sca × coefficient).

### 6.3 Mechanics

- **Requested parameters only.** Callers request a subset of parameters, and cost scales with the number requested.
- **Dual sizes.** `Dual` uses N ∈ {1, 2, 4, 8}, chosen by runtime dispatch to the smallest size that fits. Larger requests are processed in chunks of 8 (as in ForwardDiff.jl).
- **Distribution parameters.** ∂/∂θ ∫ f(r) p(r; θ) dr = ∫ f(r) ∂p/∂θ dr, evaluated at the same quadrature nodes, so no single-particle derivatives are needed.
  - Cut-off limits are held fixed within a call.
  - Renormalization of the distribution over the cut-off range is included in the derivative.
- **Refractive-index and shape parameters** are integrated from the single-particle Jacobians, using the same quadrature weights.

## 7. Ensembles

### 7.1 Distributions

- **Size distributions:**
  - `LogNormal` (r_g, σ_g);
  - `Gamma` (r_eff, v_eff);
  - `ModifiedGamma`;
  - `PowerLaw` (explicit limits required);
  - `Monodisperse`;
  - `Tabulated`;
  - `MultiModal` (a weighted sum of size distributions).
- **What each provides:**
  - its pdf;
  - ∂pdf/∂θ for its parameters;
  - cut-off limits;
  - analytic moments, where they exist (these are also used as test checks).
- **Cut-off limits** come from the analytic tail of the area-weighted (r²) distribution, so that the omitted extinction is below the tolerance.
  - For lognormal and gamma distributions this is available in closed form.
  - `Tabulated` uses the table's range.
  - `PowerLaw` uses its required limits.
- **Distributions over other parameters.** Any continuous particle parameter may have a distribution, not only size (for example aspect ratio, as in Dubovik-style spheroid dust models).
  - Size is integrated adaptively.
  - Other parameters use fixed quadrature or tabulated weights.
  - The joint distribution is their product.

### 7.2 External mixtures

- A `Mixture` combines `Ensemble`s, possibly of different particle types and methods, for example spheres plus spheroids.
- Each component has a relative number concentration wᵢ ≥ 0.
- Per-particle cross sections are averaged with weights wᵢ / Σw.
- Phase matrices are averaged with weights wᵢ C_sca,ᵢ.

### 7.3 Size integration

Integration is per wavelength, entirely in Rust.

1. **Pass 1 (grid).** Gauss–Kronrod (G7–K15) panels in ln x, refined adaptively until C_ext, C_sca and g, and their ∂p/∂θ-weighted versions, meet `rtol` (default 10⁻⁶).
   - This pass needs only aₙ and bₙ, at O(N) cost per radius.
   - Optional extra refinement criteria can be added: backscatter, or chosen Greek orders.
2. **Pass 2 (outputs).** The expensive outputs (angular values or exact Greek coefficients, O(N·n_q) per radius) are evaluated only at the final nodes.

Further rules:

- **Error estimate.** The achieved error estimate is always returned.
- **Refinement cap.** A maximum refinement depth caps the work, and hitting it sets a flag in the result. This matters because, without absorption, narrow resonance ripples cannot all be resolved; they average out in the integral.
- **Parallelism.** Work is parallel over wavelength × panel, with a deterministic reduction.

### 7.4 Ensemble outputs

All outputs are per wavelength, with cross sections per particle averaged over the number distribution:

- ⟨C_ext⟩, ⟨C_sca⟩, ⟨C_abs⟩, single-scattering albedo ω, asymmetry parameter g;
- Greek coefficients α₁–α₄, β₁, β₂ up to a requested order L, or all orders;
- phase-matrix elements at requested angles;
- lidar quantities: backscatter, lidar ratio, linear depolarization ratio δ;
- distribution moments: r_eff, mean area, mean volume;
- Jacobians for all of the above;
- integration error estimates and convergence flags.

## 8. Validation

### 8.1 Arb reference suite (`reference/`)

- **Generator.** Python using `python-flint` (Arb ball arithmetic). It computes:
  - Mie aₙ, bₙ, cross sections, S₁/S₂ and Greek coefficients;
  - layered-sphere quantities;
  - small-particle T-matrix cases, using Arb's rigorous numerical integration and matrix solve.
- **Truncation.** The generator truncates the series at a far larger N than scatrs uses. The series-tail bound is derived analytically and documented in the theory chapter.
- **Labels.** Each value is labelled:
  - **certified** — both the ball-arithmetic error and the series tail are rigorously bounded;
  - **high-precision** — no rigorous tail bound yet.

  Labels are never overstated.
- **Case grid:**
  - log-spaced x crossed with a set of m values;
  - targeted hard regimes: large x, large |m|, |m − 1| → 0, strong absorption, near-resonance;
  - seeded random samples.
- **Storage.**
  - Values are stored as 25-significant-digit decimal strings in JSON Lines files, plus a manifest recording the python-flint/Arb version, working precision and truncation.
  - Committed data stays under about 20 MB. Larger sweeps are generated only by the scheduled job.
- **Use in CI.** CI uses the committed files only and does not need Arb. A scheduled job regenerates a sample and checks it reproduces.

### 8.2 Accuracy maps

- A sweep tool produces maximum-relative-error heatmaps for each output and region. They are published in the docs.
- The validated domain and per-output tolerances declared in the code are taken from these maps.
- A CI test enforces that every reference point inside the declared domain meets its declared tolerance.

### 8.3 Invariant and identity tests (`proptest`)

- **Physical bounds:**
  - Q_ext = Q_sca when Im m = 0;
  - Q_abs ≥ 0;
  - ω ∈ [0, 1];
  - |g| ≤ 1;
  - the optical theorem.
- **Mueller realizability:**
  - single spheres: F₁₂² + F₃₃² + F₃₄² = F₁₁²;
  - ensembles: the Cloude coherency matrix is positive semi-definite.
- **Greek coefficients:** α₁,₀ = 1, and the angular values agree with those reconstructed from the Greek coefficients.
- **T-matrix:**
  - unitarity and reciprocity when the particle is lossless;
  - block structure for mirror-symmetric shapes.
- **Agreement between methods:**
  - T-matrix sphere = Mie;
  - one-layer sphere = Mie;
  - identical layers = homogeneous sphere;
  - aspect ratio 1 = sphere;
  - Rayleigh and Rayleigh–Gans limits;
  - monodisperse distribution = single particle.
- **Thread determinism:** results are bitwise identical on 1, 2 and 8 threads.

### 8.4 Derivative verification

- `Analytic` and `Dual` must agree to about 10⁻¹² relative.
- Both are checked against Arb central differences with Richardson extrapolation at ≥ 100 digits, on a subset of cases.
- Quick f64 central-difference checks cover every parameter type, including distribution, orientation and mixture parameters.

### 8.5 Comparison with other codes (`validation/`)

- **Codes compared:**
  - MIEV0 published tables, BHMIE, miepython;
  - scattnlay;
  - Mishchenko's `ampld` and `tmd`;
  - pytmatrix;
  - SMARTIES;
  - current sasktran2.
- **How it runs:** on demand and in the weekly scheduled job, not on every PR.
- **Licensing:** outputs are stored as fixtures; no external code is vendored. Each code's licence is checked before its fixtures are committed (scattnlay, for example, is GPL).

### 8.6 Robustness and Python tests

- `cargo-fuzz` checks the no-panic, no-silent-NaN rule (section 4.5).
- pytest covers:
  - broadcasting and zero-copy behaviour;
  - concurrent calls with the GIL released;
  - error types;
  - `xarray` conversion;
  - executed documentation notebooks.
- `mypy.stubtest` checks the `.pyi` stubs against the compiled module.

## 9. Python layer

### 9.1 Low-level API

- **Modules.** `scatrs.mie`, `scatrs.layered` and `scatrs.tmatrix` provide vectorized kernels such as `coefficients`, `amplitudes`, `efficiencies` and `greek`.
- **Derivatives.** Kernels take `derivatives=True` or a list of parameters.
- **Broadcasting.**
  - Inputs follow numpy broadcasting rules.
  - The output shape is the broadcast shape plus angle or order dimensions.
  - Python only computes shapes; all loops run in Rust.
- **T-matrix objects.** `scatrs.tmatrix.compute(shape, m, wavelength)` returns a `TMatrix` object with `.amplitude(orientation, angles)` and `.random_orientation(...)`.

### 9.2 High-level API

- **Classes:**
  - particles: `Sphere`, `LayeredSphere`, `Spheroid`, `Cylinder`, `Chebyshev`, `AxisymmetricProfile`;
  - distributions: `LogNormal`, `Gamma`, `ModifiedGamma`, `PowerLaw`, `Monodisperse`, `Tabulated`, `MultiModal`;
  - orientation: `Random`, `Fixed`, `OrientationDistribution`;
  - combining: `Ensemble`, `Mixture`.
- **Computation.** `Ensemble.compute(wavelength=..., num_moments=..., angles=..., derivatives=[...])` and `Mixture.compute(...)` return a result object.
- **Result objects:**
  - numpy array attributes, handed over without copying;
  - `.d(param)` returns the same structure for one parameter;
  - `.jacobian` gives the stacked arrays plus a `params` list;
  - `.to_xarray()` requires the `xarray` extra.

Example:

```python
aer = scatrs.Ensemble(
    particle=scatrs.Sphere(refractive_index=1.43 + 1e-8j),
    size=scatrs.LogNormal(median_radius=0.08, width=1.6),
)
res = aer.compute(wavelength=[0.35, 0.5, 0.75], num_moments=64,
                  derivatives=["median_radius", "width", "n", "k"])
res.ext_xs, res.ssa, res.greek.a1
res.d("median_radius").greek.a1
```

### 9.3 Runtime behaviour

- **Inputs.** Contiguous `float64` and `complex128` arrays are passed zero-copy through `PyReadonlyArray`. Anything else is converted once.
- **GIL.** The GIL is released for all computation.
- **Threads.** The extension owns a rayon pool, configured with `scatrs.set_num_threads(n)` or a per-call `num_threads`.
- **Exceptions.** `ScatrsError` is the base class, with subclasses `NotConvergedError` and `OutsideValidatedDomainError`. `allow_unvalidated=True` opts out of the domain check.

### 9.4 Packaging

- Bindings use the current pyo3 and numpy crates (0.29 at time of writing, matching pratmo).
- Wheels are abi3 for Python ≥ 3.11, with numpy ≥ 1.26 (numpy 2.x supported).
- Typing uses hand-written `.pyi` stubs plus a `py.typed` marker.

## 10. Performance

### 10.1 Targets

- **Single-threaded speed.** Match or beat the fastest existing code on each benchmark:
  - Mie against sasktran2, scattnlay and miepython;
  - T-matrix against Mishchenko's Fortran compiled with `-O3`, and against pytmatrix.
- **Scaling.** Near-linear with core count.
- **Jacobian overhead.**
  - Mie with the full (r, n, k) Jacobian: at most 2× the value-only cost.
  - `Dual<N>` paths: at most about (1 + N)× the value-only cost.
- **End-to-end ensembles.** Building a lognormal ensemble over many wavelengths must be substantially faster than sasktran2's current Python-driven path. The ratio is measured and published.

### 10.2 Algorithmic savings

- Exact-quadrature Greek coefficients (section 5.4).
- The two-pass size integration (section 7.3).
- πₙ/τₙ tables shared across sizes per wavelength, and Wigner-d tables cached per (L, n_q).
- T-matrix:
  - N_max warm-started from neighbouring sizes;
  - mirror-symmetry halving of the θ quadrature;
  - per-m block structure;
  - one LU factorization reused for values and derivatives.

### 10.3 Implementation techniques

- Structure-of-arrays layouts.
- Per-thread workspaces, so inner loops never allocate.
- No complex division in hot loops.
- Autovectorization first, with the generated assembly inspected.
- Explicit SIMD only where profiling justifies it, using runtime CPU-feature dispatch (AVX2/FMA, NEON) so that wheels stay portable. The vector direction is across radii.
- Release profile: `lto = "fat"`, `codegen-units = 1`, default panic unwinding (so pyo3 can convert panics into Python exceptions).

### 10.4 Benchmark infrastructure

- **Criterion benchmarks** for local wall-clock timing:
  - each kernel across x;
  - Jacobian overhead;
  - T-matrix across size and aspect ratio;
  - full ensemble builds.
- **Instruction-count benchmarks** (Callgrind-based, e.g. iai-callgrind) in CI. A PR fails if it regresses beyond a set threshold.
- **`benchmarks/compare.py`** runs the cross-code comparisons. Results are published in the docs.

## 11. Constraints for future sasktran2 integration

There is no integration work in this milestone. The design must keep the following true:

- **Dependency.** sasktran2 will depend on the `scatrs` crate as a Rust dependency, not through Python.
- **Version independence.** The core public API exposes no ndarray or pyo3 types. (sasktran2 currently uses ndarray 0.16 and pyo3 0.27.)
- **Sign convention.** sasktran2 uses m = n − ik. Its adapter will conjugate m and map derivatives with respect to Im m accordingly.
- **Greek coefficients.** α₁,₀ = 1 and α₁,₁ = 3g already match sasktran2. The adapter handles any β sign-convention differences, and a test documents them.
- **Threads.** sasktran2 must be able to pass in its own thread pool.
- **Jacobian mapping.** Jacobians must map onto sasktran2's per-parameter derivative storage: d(cross section), d(scattering cross section), d(stacked Legendre coefficients).
- **Units.** sasktran2 uses nm and converts cross sections to m²; the adapter does the conversion.

## 12. Build, CI, release, docs

### 12.1 Workflows

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | push, PR | **Rust:** `cargo fmt --check`, `clippy -D warnings`, `cargo test` on Linux, macOS and Windows; minimum supported Rust version (MSRV) job. **Python:** wheel build + pytest on Linux, macOS and Windows for Python 3.11 and latest; `stubtest`. **Also:** strict Sphinx build (`-W`), reference-suite tests, instruction-count benchmark gate. |
| `release.yml` | `v*` tags (build-only on PRs) | maturin-action wheel matrix as in pratmo (manylinux, musllinux, Windows, macOS) and sdist; PyPI Trusted Publishing; `cargo publish` for `scatrs` |
| `docs.yml` | push to main | Sphinx to GitHub Pages |
| `scheduled.yml` | weekly | Arb reference regeneration sample; time-boxed `cargo-fuzz`; cross-code validation |

### 12.2 Tooling

- **Single version source.** `pyproject.toml` declares `dynamic = ["version"]` so maturin reads the version from `Cargo.toml`. This removes pratmo's three-place manual version sync.
- **Rust toolchain.** `rust-toolchain.toml` pins stable. The MSRV is set in phase 0 to the oldest Rust the dependency set supports, recorded in `rust-version` and tested in CI.
- **Formatting and linting.** `rustfmt.toml`; ruff for Python; pre-commit.
- **Dependencies.** uv for the development environment, with `uv.lock` committed; dependabot.
- **Release notes.** A CHANGELOG.
- **License.** MIT.

### 12.3 Docs

- **Tooling.** Sphinx with furo and MyST-NB; notebooks are executed at build time. Rust API docs are on docs.rs.
- **Sections:**
  - user guide;
  - theory (algorithms, derivative formulas, the truncation-tail bound, conventions);
  - accuracy maps;
  - benchmarks;
  - API reference.

## 13. Build order

Each phase gets its own implementation plan.

| Phase | Contents | Exit criterion |
|---|---|---|
| 0 | Workspace, CI, `Real` / `DoubleDouble` / `Dual`, special functions, reference-generator skeleton | CI green on 3 OSes; special functions match Arb references |
| 1 | Homogeneous Mie with analytic derivatives; low-level Python API | Accuracy map published; declared validated domain enforced in CI |
| 2 | Distributions, adaptive integration, Greek coefficients, Jacobians, mixtures, high-level Python API | Cross sections agree with sasktran2 `MieDatabase` outputs within sasktran2's integration tolerance; Greek-coefficient differences are quantified and attributed (section 5.4 and Appendix A); benchmark against sasktran2 published |
| 3 | Layered spheres (`Dual` derivatives) | Matches Arb references and scattnlay |
| 4 | T-matrix: shapes, convergence, `DoubleDouble` fallback, random, fixed and distributed orientation, derivatives | Matches Arb references, Mishchenko and pytmatrix; sphere matches Mie |
| 5 | Performance pass, documentation, release 0.1.0 | Section 10.1 targets met |

## 14. Research track

These run alongside the phases. An item is adopted only if the accuracy maps and benchmarks show it is better, and each outcome is written up in the docs.

1. **Stable spheroid EBCM.** Somerville, Auguié & Le Ru (2013) and SMARTIES (2016) remove the cancellations in the spheroid Q-matrix integrals. This may make `DoubleDouble` unnecessary for spheroids.
2. **Mie ψₙ stability.** A downward ratio recurrence for ψₙ where upward recurrence loses digits.
3. **Series-tail bound.** A rigorous analytic bound on the Mie truncation tail, needed for the "certified" label.
4. **Size quadrature.** Resonance-aware quadrature, and comparison against Gauss–Hermite in ln r for lognormal distributions.
5. **Random-orientation T-matrix Greek coefficients.** Mishchenko's Clebsch–Gordan approach versus alternatives.
6. **Hand-derived derivatives** (approach 2) for any `Dual` path that misses the section 10.1 overhead target.

## Appendix A. Observations about current sasktran2 Mie code

These were recorded during the design survey (2026-10-12). They are relevant to validation and to the future replacement.

- **Phase-matrix quadrature grid.** `integrate_mie_cpp` uses a split Gauss–Legendre grid with 2 × `num_coeffs` nodes. When N ≫ `num_coeffs` this is not exact, so Greek coefficients alias. Phase 2 quantifies the effect.
- **Bug in dead code.** `rust/sasktran2-rs/src/optical/mie/worker.rs:84` is missing parentheses in the πₙ recurrence: it should be `((2n−1)μπₙ₋₁ − nπₙ₋₂)/(n−1)`. The struct is used only by its own tests.
- **Missing method.** `src/sasktran2/database/mie.py:123` calls `_generate_sasktran2_rust()`, which does not exist, so `backend="sasktran2_rust"` raises `AttributeError`.
- **No analytic derivatives.** Today's aerosol derivatives come from interpolation weights over the database parameter grid.
