# Phase 4 design: randomly oriented spheroids

- **Date:** 2026-10-14
- **Status:** draft for review.
- **Parent spec:** `docs/specs/2026-10-12-scatrs-design.md` (cited as "main spec"). This document refines main spec §5.3, §6.1, §9 and §13 row 4, and lists its amendments in section 11. It builds on the phase 3 design (`docs/specs/2026-10-14-phase-3-layered-spheres-design.md`, "phase 3 spec"), whose section 15 prepared the engine and the public API for this phase.
- **Evidence:** the precision study in `research/tmatrix/` (its README, cited as "the study"). It is an EBCM built from the definition in mpmath, used to find where and why the method loses precision.

The user's decisions of 2026-10-14:

- **Random orientation first.** Randomly oriented particles are the priority; fixed orientation and other shapes may come later.
- **Spheroids only** in this phase.
- **A fixed aspect ratio per ensemble component.** Shape mixtures are built from `Mixture` components.
- **Full Jacobians:** with respect to size, aspect ratio, Re m and Im m, for single particles and in ensembles.
- **The validated domain as large as possible,** with a double-double fallback.
- **The method:** a numerically stable spheroid EBCM, generic over the precision, with Mishchenko's analytic orientation averaging.
- **Theory before code.** The user asked for the precision study before the algorithm was settled. Its outcome (section 2.4) changes the solve.

## 1. Scope and split

One spec, two plans, each with its own exit criterion and review.

### 1.1 Phase 4a: the spheroid T-matrix

- **Kernel and outputs.**
  - The kernel (section 2), with its convergence loop, precision paths and derivatives.
  - The random-orientation outputs (section 3).
- **APIs.** The Rust API `scatrs::tmatrix` (section 5.1) and the low-level Python API `scatrs.tmatrix` (section 5.2).
- **Validation and performance.**
  - The reference generator, the accuracy maps and the declared validated domain (sections 6 and 7).
  - The identities, the external comparisons and the benchmarks (sections 8 to 10).
- **Docs:** a theory chapter and the validation sections.

**Exit criterion:**

- Every reference in the declared domain meets its tolerance in CI, and points outside the domain raise `OutsideValidatedDomain`.
- Aspect ratio 1 gives `scatrs.mie`'s results to rounding.
- The identities of section 8 hold.
- The comparisons with Mishchenko's `tmatd` and pytmatrix are published, with every disagreement attributed.

### 1.2 Phase 4b: spheroids in ensembles

- **The engine's direct path:** per-node cross sections and Greek coefficients instead of aₙ and bₙ (section 4).
- **F₂₂ and F₄₄** in the result types.
- **`Particle::Spheroid` and `scatrs.Spheroid`,** in `Ensemble`, `Mixture` and grids, with Jacobians.
- **An example notebook** on mineral dust, and the release notes.

**Exit criterion:**

- An ensemble of spheroids of aspect ratio 1 agrees with the sphere ensemble within the error estimates.
- Ensemble Jacobians match central differences.
- Mixtures of spheres and spheroids work.
- Spheres, layered spheres and their ensembles are bitwise unchanged.

### 1.3 Not in phase 4

| Item | Where it goes |
|---|---|
| Fixed orientation (amplitude, Mueller and extinction matrices at given Euler angles) | A later phase. `TMatrixSolution::block` and Python's `TMatrix` object are where it attaches. |
| Distributed orientation (canting) | A later phase, after fixed orientation. |
| Cylinders, Chebyshev particles, sampled profiles | Later. They need the generic EBCM: the symbolic cancellation removal of section 2.3 is spheroid-specific. `Shape` is non-exhaustive. |
| Aspect-ratio distributions | Later, as a distribution over a real direction of the particle. |
| A spheroidal-basis method (separation of variables) | A research track (section 2.4). |
| A release | The user decides. |

## 2. Algorithm

### 2.1 Geometry and conventions

- **Shape.** A spheroid has equal-volume radius r_v and aspect ratio ε = a/c, the equatorial over the polar semi-axis (main spec §3): ε > 1 is oblate and ε < 1 prolate.
  - The semi-axes are a = r_v ε^{1/3} and c = r_v ε^{−2/3}.
  - The surface is r(θ) = a c / √(c² sin²θ + a² cos²θ).
- **Size and index.** The size parameter is x = k r_v. The relative index is s = m, with Im m ≥ 0 and e^{−iωt}.
- **T-matrix convention.** The T-matrix follows Mishchenko, Travis & Lacis (2002). For a sphere it is diagonal, with T¹¹ₙ = −bₙ and T²²ₙ = −aₙ; the study checks this to 30 digits.
- **Outputs.** They use scatrs's existing conventions (Bohren & Huffman's S₁, S₂, the phase-matrix normalizations and the Greek coefficients of phase 1b). The mapping from Mishchenko's amplitude matrix is tested on spheres, to rounding.

### 2.2 The T-matrix per azimuthal mode

- **Integrals.** For each mode m = 0…N, Q and P = RgQ are surface integrals of the vector spherical wave functions. With the reciprocity form B(X, Y) = ∫_S n̂·(X × ∇×Y − Y × ∇×X), every element is Q_nk = −i B(Xₙ^out, Y_k), where X is an outgoing wave at k and Y a regular wave at k₁ = s k. P_nk is the same with X regular.
- **Solve.** Tᵐ = −Pᵐ (Qᵐ)⁻¹.
- **Quadrature and symmetry.** The integrals use Gauss–Legendre quadrature in cos θ. Mirror symmetry splits each mode into even and odd (n + k) blocks, halving the integrals and the solves.
- **Reciprocity.** It is not imposed. The truncated EBCM T-matrix is symmetric only in the limit, and the study's reciprocity residual (10⁻¹⁸ at N = 20) is a convergence check.

### 2.3 Cancellation-free integrals

**Cause** (the study, findings 1–3):
- In f64 the textbook integrals lose up to 10⁸ (x = 10, ε = 2) to cancellation.
- For a spheroid, 1/r(θ)² is a polynomial in cos²θ. So every negative-power Laurent term of the product hₙ(kr) j_k(s k r) multiplies a polynomial in cos θ of degree below |n − k|, and integrates to exactly zero against the Legendre functions.
- That cancellation amplifies any pointwise error, in the outgoing or in the internal functions. Rounding errors alone in the internal functions' samples give T errors of 3·10⁻¹¹ at x = 20, ε = 2, and of order 1 at x = 15, ε = 4.

**Removal:**
- The vanishing terms are removed symbolically, per pair (n, k), as Somerville, Auguié & Le Ru (2013) do: Bessel-function products with their non-contributing terms dropped.
- 4a derives the formulas independently and checks them in the study's lab, element by element, against the definition at high precision, before the Rust kernel.

**What this buys.** Each element is then accurate to a few units of its own rounding, with errors independent between elements.

**An exact identity, for later shapes** (the study, finding 6):
- Q = Q̃ + diag(α) Q⁽ᵏ¹⁾ holds for any axisymmetric shape.
- Q̃ uses the test functions Xₙ(kr) − αₙ Xₙ(k₁r), and Q⁽ᵏ¹⁾ is surface-independent and diagonal in the order.
- It cuts the cancellation only 2–6× (with α fitted), so it is not used for spheroids. It is recorded for the generic shapes.

### 2.4 The solve: f64, refinement, double-double

**The f64 floor.** With cancellation-free elements, the error of an f64 solve is about Skeel(Q)·2⁻⁵³, where Skeel is the componentwise condition number ‖ |Q⁻¹| |Q| ‖.
- An f64 LU factorization perturbs every element by about 2⁻⁵³|Q| (backward stability), so the floor binds the factorization as well as the elements.
- Measured on mode 1, s = 1.5 + 0.01i (the study, finding 4):

  | ε | floor ≤ 10⁻¹¹ up to | floor about 10⁻⁸ at |
  |---|---|---|
  | 2 | x = 30 | x = 45–50 |
  | 3 | x = 20 | x = 30 |
  | 1/3 | x = 15 | x ≈ 20 (10⁻⁶ at x = 25) |
  | 5 | x = 10 | x = 15–20 |

- That is about 10⁻⁸ near |s|·x_max ≈ 60–95, with x_max = k max(a, c).

**Where it lives** (findings 5 and 7):
- The floor is carried by the high outgoing-order rows.
- Rows n ≥ N/4 exact, with the rest rounded, give rounding-level T in all three hard cases.
- On a non-spherical surface those rows are the evanescent multipoles r^{−n−1}Pₙ. They concentrate at r_min and are nearly dependent.

**Mixed-precision iterative refinement** (finding 10):
1. Factorize Qᵐ in f64.
2. Evaluate the residual of X Qᵐ = −Pᵐ with double-double elements and double-double arithmetic. This costs O(N²) per sweep.
3. Correct with the f64 factors.

- It converges to double-double accuracy when Skeel·2⁻⁵³ < 1, i.e. Skeel below about 10¹⁶; the grid's worst case is 5·10¹¹.
- Double-double is needed only for the integrals of rows n ≥ N/4, beside the residuals. The O(N³) factorization stays in f64.
- Measured, as f64 floor → refined:

  | case | rows n ≥ N/4 in double-double, 2 sweeps | all rows in double-double |
  |---|---|---|
  | x = 25, ε = 1/3 | 1.2·10⁻⁶ → 7·10⁻¹⁷ | 1.2·10⁻²² |
  | x = 15, ε = 5 | 3.7·10⁻¹⁰ → 4·10⁻¹⁶ | 4·10⁻²⁶ |
  | x = 50, ε = 2 | 5.9·10⁻⁸ → 4·10⁻¹⁷ | 4·10⁻²⁴ |

**Precision paths,** chosen per mode by a Skeel estimate from the LU factors (Higham's estimator, O(N²)). The maps set the thresholds:
1. **f64:** the estimate × 2⁻⁵³ is below `rtol`/10.
2. **Refined:** otherwise, while the estimate is below 10¹⁴. Double-double integrals cover the rows n ≥ N/4 (the maps may move the cut) plus the residuals, and the solve is refined.
3. **Double-double:** the whole kernel at `T = DoubleDouble`, refined in quad-double residuals only if the maps show it is needed. The kernel is generic over `Real`, as the layered kernel is.

**Other routes the study tried:**
- A change of internal basis removes the conditioning in exact arithmetic (Skeel 4·10⁴ → 22). It cannot be exploited in f64 with spherical waves, because the cancellation of section 2.3 is defined per spherical pair (finding 5).
- Distributed sources (NFM-DS) cut the cancellation, but raise Skeel 10⁵–10⁹× (finding 8).
- Test functions that are both cancellation-free and well-conditioned are the outgoing spheroidal waves. That is the separation-of-variables method: a research track, not part of this phase.

### 2.5 Convergence

1. **N** starts from an estimate based on x_max and the index, and grows by steps until the changes in C_ext and C_sca fall below `rtol` (default 10⁻⁸).
2. **n_θ** then grows until they settle again. It starts at about 2N, and more for large or small ε.
3. **A final check** also requires the Greek coefficients up to the requested order to settle, since ensembles use them.
4. **Caps.** N and n_θ have caps (options). Reaching a cap raises `NotConverged`, unless the caller asks for the unconverged result with its estimate. A result is never returned silently unconverged.
5. **Reporting.** The result reports N, n_θ, the precision path of each mode and the error estimate (section 6).

### 2.6 Derivatives

- **Parameters:** x, ε, Re m and Im m.
- **Re m and Im m.** The coefficients are holomorphic in m, so the complex derivative d/dm gives both, as for spheres.
- **x and ε.** These move r(θ), so their integrands are differentiated through the shape and through the symbolic removal of section 2.3.
- **The solve.** dTᵐ = −(dPᵐ + Tᵐ dQᵐ)(Qᵐ)⁻¹, reusing the factorization of each mode for all four directions. On the refined path the derivative solves are refined too.
- **Through the averaging.** The derivatives are chained through the averaging formulas of section 3.
- **Method choice.** 4a prototypes closed-form differentiated integrands against `Dual` inside them, and adopts the one that is accurate and cheaper (target in section 10).

## 3. Random-orientation outputs

Mishchenko (1991) and Mishchenko, Travis & Lacis (2002), chapter 5:

- **Cross sections:**
  - ⟨C_ext⟩ = −(2π/k²) Σ_{n,m} (2 − δ_m0) Re(T¹¹_{nn|m} + T²²_{nn|m});
  - ⟨C_sca⟩ = (2π/k²) Σ_{n,k,m} (2 − δ_m0) Σ_{ij} |Tⁱʲ_{nk|m}|²;
  - ⟨C_abs⟩ is their difference, computed without cancellation where possible. 4a checks the remaining loss on the maps, as phase 1 did for spheres.
- **Asymmetry and backscattering:** g, and ⟨C_back⟩ from the Greek expansion at μ = −1.
- **Greek coefficients** α₁–α₄ and β₁, β₂, up to the requested order (exact up to 2N), by the Clebsch–Gordan expansion: O(N³) per particle. Main spec §14 item 5 keeps the alternatives as research.
- **The phase matrix** at any cosines, from the Greek expansion. It has six independent elements: F₁₁, F₂₂, F₃₃, F₄₄, F₁₂ and F₃₄.
- **Normalization:** per unit geometric cross section of the equal-volume sphere (efficiencies), and normalized or unnormalized phase matrix and Greek coefficients, as for spheres.

## 4. Engine: the direct path (4b)

Phase 3 spec section 15 prepared this.

- **The solver.** For a spheroid it returns, per size node, the cross sections and the Greek coefficients up to the order the call needs, and their derivatives along the particle's directions. It does not return aₙ and bₙ.
- **Pass 1** refines on the same quantities as for spheres: the cross sections, C_sca·g, the optional Greek coefficients, and the derivatives along the particle's directions.
- **Pass 2** integrates the Greek coefficients linearly over the sizes. The spheres' band tables and `PanelProducts` are not used. The phase matrix at the requested cosines comes from the averaged Greek coefficients, so a call asking for cosines and not for `greek` computes them internally.
- **Per-node tolerance.** The T-matrix of each node converges to a tenth of the ensemble's `rtol`, and at most 10⁻⁸, so that its error stays below the size integration's.
- **Directions.** For a spheroid: Re m and Im m (one group), then the aspect ratio as a real direction, the mechanism of the radius fractions. The radius derivative of a monodisperse ensemble uses ∂/∂x.
- **Spheres stay on their path.** Spheres and layered spheres keep the coefficient path unchanged, and mixtures may combine both paths: the Greek coefficients of the components are mixed with their C_sca weights, as the tables are.

## 5. APIs

### 5.1 Rust: `scatrs::tmatrix`

```rust
#[non_exhaustive]
pub enum Shape<T> { Spheroid { aspect_ratio: T } }

pub struct TMatrix<T: Real> { /* a reusable workspace, one per thread */ }
impl<T: Real> TMatrix<T> {
    /// `x`: the size parameter of the equal-volume sphere.
    pub fn compute(&mut self, shape: Shape<T>, x: T, m: Complex<T>, opts: &TMatrixOptions)
        -> Result<TMatrixSolution<'_, T>>;
    pub fn compute_with_derivatives(&mut self, shape: Shape<T>, x: T, m: Complex<T>,
        opts: &TMatrixOptions) -> Result<TMatrixSolution<'_, T>>;
}
```

- **`TMatrixOptions`:**
  - `rtol` (default 10⁻⁸);
  - `precision`: `Auto`, `F64` or `DoubleDouble`;
  - optional caps or fixed values for N and n_θ;
  - `allow_unvalidated`.
- **`TMatrixSolution`:**
  - **Convergence diagnostics:** `n_terms()`, `n_theta()`, `precision_paths()` and `error_estimate()`.
  - **The T-matrix:** `block(m)`, the blocks of Tᵐ.
  - **Outputs:** `efficiencies()`, `greek(order, normalization)` and `phase_matrix(mu, normalization, buffers)`, with six elements.
  - **Derivatives:** `efficiencies_jacobian()` and the `*_with_derivatives` forms, with respect to x, ε, Re m and Im m.
- **Phase-matrix types.** `PhaseBuffers`, `PhaseValues` and the ensemble phase gain `f22` and `f44`, equal to F₁₁ and F₃₃ for spheres. The structs are `non_exhaustive`, so this is additive.
- **`tmatrix::VALIDATED_DOMAIN`** is public, with the same accuracy-model API as `mie` and `layered`.

### 5.2 Python: `scatrs.tmatrix`

- **Functions:** `efficiencies(x, m, aspect_ratio, …)`, `greek(…)` and `phase_matrix(…)`. They are vectorized, and their leading axes broadcast as in `scatrs.mie`.
- **Options:**
  - `rtol`, `precision`, `allow_unvalidated`;
  - `derivatives=` (`True`, or names among `"x"`, `"aspect_ratio"`, `"m_real"`, `"m_imag"`; for complex outputs `"m"`).
- **Results:** they reuse `scatrs.mie`'s result classes. The phase-matrix result gains `f22` and `f44`.
- **`scatrs.tmatrix.compute(x, m, aspect_ratio, …)`** returns a `TMatrix` object for one particle: its blocks, its convergence information and `.random_orientation(...)`. Fixed orientation (`.amplitude(euler, angles)`, main spec §9) attaches there later.

### 5.3 Python high-level (4b)

- **`scatrs.Spheroid(refractive_index, aspect_ratio)`.** The index is converted as `Sphere`'s is. The object is frozen and hashable.
- **Ensembles.** It is accepted by `Ensemble`, `Mixture` and grids.
- **Size convention.** The size distribution describes the equal-volume radius. The docstrings say so, as phase 3 spec section 15 decided.
- **Derivative names:** `"aspect_ratio"`, `"m_real"` and `"m_imag"`, with the distribution's and the `c{i}.` and `mode{j}.` prefixes.
- **Results.** `EnsemblePhase` gains `f22` and `f44`. `depolarization` becomes (F₁₁ − F₂₂)/(F₁₁ + F₂₂) at 180°, which is 0 for spheres.
- **Rust:** `Particle::Spheroid { aspect_ratio, refractive_index }` and `ParamKind::AspectRatio`.

### 5.4 Errors

- `InvalidInput`: for ε ≤ 0 or a non-finite ε, for the inputs `mie` rejects, and for a requested derivative the particle does not have.
- `OutsideValidatedDomain`: outside the maps' domain, unless `allow_unvalidated` is set.
- `NotConverged`: when N or n_θ reaches its cap (section 2.5).

## 6. Accuracy model and validated domain

- **Three sources of error:** truncation (N), quadrature (n_θ), and rounding amplified by Q's conditioning.
- **The error estimate:** the last convergence change, plus Skeel·u per mode, with u = 2⁻⁵³, or 2⁻¹⁰⁶ after refinement. It comes with every result.
- **What is declared.** Inside `tmatrix::VALIDATED_DOMAIN`, every output is within C·max(`rtol`, estimate) of the exact value.
  - C is per output, as in phases 1 and 3, measured on the maps and at the references, and at least 4 times the worst case found.
  - The derivatives get their own constants.
- **Domain target.**
  - Aspect ratios 1/5 to 5. |s|·x_max up to about 100 or beyond, every path included.
  - Indices in the Mie domain, where the maps support them.
  - The maps are run past the target to find the real edge. The published domain is what they support.

## 7. References (`reference/scatrs_reference/generators/spheroid.py`)

- **Independence.** The generator uses the study's EBCM built from the definition: the surface integrals of the vector spherical wave functions, with none of section 2.3's removals and no refinement. It runs at 50 or more digits, with N and n_θ grown until every stored output settles to 10⁻²⁰. It shares no code or algorithm with the kernel.
- **Labels.** "High-precision", not certified, since the truncation in N is not rigorously bounded (main spec §8.4).
- **Cases:**
  - x up to about 30 (mpmath cost);
  - ε ∈ {1/5, 1/3, 1/2, 2, 3, 5} and 1;
  - indices: dust (1.53 + 0.008i), ice (1.31), water (1.33), an absorber (1.5 + 0.5i) and a high index (3 + 0.1i);
  - derivatives by Richardson-extrapolated differences.
- **Budget.** The data is stored at log-spaced orders, as in phase 1. It keeps within about 3 MB, as the total is limited.

## 8. Tests beyond the references

- **Aspect ratio 1:** the T-matrix is diagonal and equals −bₙ, −aₙ. Every random-orientation output equals `scatrs.mie`'s to rounding.
- **Lossless particles:** ⟨C_ext⟩ = ⟨C_sca⟩. Reciprocity converges with N.
- **The Rayleigh limit:** at small x, ⟨C_ext⟩ and ⟨C_sca⟩ follow the spheroid's electrostatic polarizability tensor, from its depolarization factors (Bohren & Huffman 1983, §5.3).
- **Symmetry:** the even and odd block structure of mirror-symmetric spheroids.
- **Oblate against prolate:** ε and 1/ε at the same equal-volume size give different results, and each matches its own references. This guards against exchanging a and c.
- **Derivatives:** against central differences in `DoubleDouble`.
- **Precision paths:** each path gives the same result within its estimate on overlapping cases, and the refined path matches full double-double.
- **Thread independence:** ensembles of spheroids are bitwise the same on 1, 2 and 8 threads.
- **Python:** broadcasting, derivative names, errors, the `TMatrix` object, and immutability and hashing of `Spheroid`.

## 9. External validation

- **Mishchenko's `tmatd`** (random orientation, the extended-precision build), compiled in the benchmark environment: cross sections and expansion coefficients over the case grid. Every disagreement beyond `tmatd`'s own tolerance is attributed.
- **pytmatrix:** the same comparison where it applies.
- **Published tables:** Mishchenko's and SMARTIES's published values, as check points.
- **The external-validation page** shows the comparisons and the Rayleigh limit.

## 10. Performance

- **Speed.** On one thread, scatrs is at least as fast as `tmatd` for random-orientation cross sections and Greek coefficients at equal accuracy, over the case grid.
- **Derivatives.** All four together cost at most 3 times the values.
- **Precision paths.** The refined path costs at most 4 times plain f64. The double-double path is measured (10–20 times expected). The maps report where each path engages.
- **Ensembles** (4b): a dust lognormal, with the Greek coefficients to order 63, per wavelength. The cost is published; there is no target for the first such path.
- The measured ratios are published.

## 11. Amendments to the main spec

- **§5.3:**
  - The cancellation-free spheroid integrals (section 2.3).
  - The precision fallback becomes the three paths of section 2.4, chosen by a Skeel estimate, with mixed-precision iterative refinement in place of a plain retry in double-double.
  - Fixed and distributed orientation move to a later phase.
- **§5.3 shapes:** spheroids only in phase 4. Cylinders, Chebyshev particles and sampled profiles come later, with the generic EBCM.
- **§13 row 4:** the exit criterion is that of section 1. The `tmatd` comparison replaces "Mishchenko" for random orientation; fixed orientation's comparisons move with it.
- **§14 item 1:** answered by the study. The stable spheroid integrals remove the integral cancellation, but not the conditioning floor, which refinement removes.

## 12. Decisions

- Random orientation, spheroids, fixed aspect ratio per component, full Jacobians, and the largest domain reachable (user, 2026-10-14).
- The stable spheroid EBCM generic over `Real`, over the plain EBCM with double-double and over IITM (user).
- The precision study first (user). Its outcome:
  - symbolic cancellation removal;
  - f64 where the conditioning allows;
  - mixed-precision refinement beyond;
  - full double-double last.

  Separation of variables in spheroidal coordinates is recorded as the research track that could extend f64 further.
- The study's lab becomes the reference generator's core.
