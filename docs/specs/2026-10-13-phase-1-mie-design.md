# Phase 1 design — homogeneous-sphere Mie

- **Date:** 2026-10-13
- **Status:** approved; revised during planning (section 12)
- **Parent spec:** `docs/specs/2026-10-12-scatrs-design.md` (cited as "main spec"). This document refines main spec §5.1, §5.4, §6, §8, §9.1 and §13 row 1 for phase 1, and lists the amendments it makes (section 11).
- **Inputs:** `docs/plans/phase-0-followups.md`, especially "Phase 1 plan must include".

## 1. Scope and split

Phase 1 is about as large as phase 0, so it is delivered as two plans, each with its own exit criterion.

### 1.1 Phase 1a — Mie kernel and validation

- A scatrs-owned complex reciprocal (section 2.8).
- Log-derivative and Riccati–Bessel machinery (sections 2.3–2.4).
- Mie coefficients aₙ, bₙ with closed-form derivatives ∂/∂x and ∂/∂m.
- Efficiencies Q_ext, Q_sca, Q_abs, Q_back and asymmetry parameter g, with Jacobians.
- Amplitudes S₁(μ), S₂(μ) with derivatives.
- Arb reference generator for Mie, including the series-tail bound for "certified" labels.
- Accuracy maps, the declared validated domain, `allow_unvalidated`, and the CI test that enforces the domain.
- Python: `scatrs.mie.coefficients`, `efficiencies`, `amplitudes`, `default_n_max`.
- Criterion benchmarks for the Mie kernel. The Jacobian-overhead target (§8) is checked here.
- Docs: Mie theory chapter, accuracy-map page, a Mie example notebook.
- The small phase 0 carry-overs (section 10).

**Exit criterion:** the accuracy maps for coefficients, efficiencies and amplitudes are published; the declared validated domain is enforced in CI; Analytic and `Dual` derivatives agree.

### 1.2 Phase 1b — Greek coefficients and performance infrastructure

Outlined in section 9. It gets a short design update before its plan is written.

**Exit criterion:** the Greek kernel is exact by the §5.4 bound and validated against Arb; results are bitwise identical on 1, 2 and 8 threads; the instruction-count gate runs in CI; the comparison against sasktran2 and miepython is published.

### 1.3 Not in phase 1

| Item | Reason / where it goes |
|---|---|
| Kahan `csqrt`, `ln` near 1, complex division | Mie needs only a reciprocal. The others are added when layered spheres or the T-matrix need them. |
| Derivative strategy trait (main spec §4.3) | Its first consumer is the ensemble layer: phase 2. Phase 1 has analytic Jacobian entry points, plus generic kernels that also run on `Dual`. |
| `DoubleDouble`/`Dual` in Python | No user need yet. They are used inside Rust: DD for accuracy maps, `Dual` as the derivative oracle. |
| x86_64 no-FMA decision | Before the first public release. The f64 Mie path does not use DD. The only DD-heavy step left, the Gauss–Legendre rule, moves to an O(n) f64 method in 1b. |
| First real CI run | Needs the GitHub repository (user action). It does not gate phase 1 work; the 1a merge notes whether it has happened. |

## 2. Algorithm

### 2.1 Notation

Bohren & Huffman conventions throughout:

- ψₙ(ρ) = ρ jₙ(ρ), χₙ(ρ) = −ρ yₙ(ρ), ξₙ = ψₙ − iχₙ.
- Dₙ(z) = ψₙ′(z)/ψₙ(z), z = mx.
- aₙ = [(Dₙ(z)/m + n/x)ψₙ − ψₙ₋₁] / Δₐ, with Δₐ = (Dₙ(z)/m + n/x)ξₙ − ξₙ₋₁.
- bₙ = [(mDₙ(z) + n/x)ψₙ − ψₙ₋₁] / Δ_b, with Δ_b = (mDₙ(z) + n/x)ξₙ − ξₙ₋₁.
- Two identities are used below. Both hold exactly for all n and real x:
  - Wronskian: ψₙξₙ′ − ψₙ′ξₙ = i;
  - Casoratian: ψₙχₙ₋₁ − ψₙ₋₁χₙ = −1.

### 2.2 Truncation

- **Default:** N = ⌈x + 6x^⅓ + 3⌉ in `f64`, ⌈x + 12x^⅓ + 6⌉ in `DoubleDouble`. `n_max` overrides it. Wiscombe's ⌈x + 4.05x^⅓ + 2⌉ left truncation errors up to 10⁻¹⁰ in Q_ext and 10⁻⁶ in Q_back against Arb; the new constants were chosen from the same measurements.
- **Contract.** The accuracy contract holds only for `n_max ≥ N`. A smaller `n_max` is outside the validated domain, so it raises `OutsideValidatedDomain` unless the caller sets `allow_unvalidated`.
- **Upper cap.** `n_max ≤ MAX_MIE_ORDER` = 2²⁰, bounding workspace memory to about 200 MB. Larger values raise `InvalidInput`.
- **Accepted inputs.** x ∈ [10⁻³⁰, 10⁶] and |m| ∈ [10⁻⁶, 10³], Re m ≥ 0, Im m ≥ 0. Outside, `InvalidInput`; inside, every output is finite (a property test checks this).
- **Known limitation, documented on the limitations page with a worked case.** Spheres with Re m > 1 and small Im m have whispering-gallery resonances at orders x < n < Re(m)·x. Resonances above N are exponentially narrow, so the truncation misses them only for x within a tiny window. The rule is therefore validated by the accuracy maps, not proven. `n_max` lets a user include such resonances.

### 2.3 Log-derivatives

- **Dₙ(mx).** Downward recurrence of uₖ = ψₖ/ψₖ₋₁ = 1/((2k + 1)/z − uₖ₊₁), with Dₙ = (n + 1)/z − uₙ₊₁, started from uₖ₊₁ = 0 at k = max(N + 1, ⌈1.5|z|⌉) plus a margin of half the type's bits. Above 1.5|z| each step contracts errors at least 6.25-fold (section 2.3 of the theory chapter), so the start is forgotten to rounding level.
  - No continued fraction is evaluated, so there is no convergence test and no `NotConverged`.
  - Downward recurrence is always used. Wiscombe's upward option only saves the seed cost and needs its own stability criterion. The 1b benchmarks may reinstate it as an optimisation, but correctness never depends on it.
  - For real m the real recurrence is used: twice as fast, and for m = 1 it reproduces uₖ(x) bit for bit.
- **Dₙ(x)** for real x, by the same method in real arithmetic. It is needed only for n > x, where it has no poles (section 2.4).

### 2.4 ψₙ and ξₙ: a hybrid scheme

The baseline's upward ψ recurrence starts with ψ₁ = sin x/x − cos x. That cancels catastrophically for small x: the relative error is about ε/x², so 2·10⁻⁴ at x = 10⁻⁶. MIEV0 avoids this with a small-particle expansion, which is only accurate to about 10⁻⁸. scatrs replaces both with one scheme that is accurate for every x:

- **For n ≤ nₛ = ⌊x⌋ (only when x ≥ 1):**
  - ψₙ and χₙ by upward recurrence from ψ₀ = sin x, ψ₋₁ = cos x, χ₀ = cos x, χ₋₁ = −sin x;
  - aₙ, bₙ from the formulas in section 2.1.
- **For n > nₛ: ratio form.**
  - Rₙ = ξₙ₋₁/ξₙ by the upward recurrence 1/Rₙ = (2n − 1)/x − Rₙ₋₁. This is stable because |ξₙ| increases with n.
  - ψₙ₋₁/ψₙ = Dₙ(x) + n/x, which is positive and finite because ψₙ(x) has no zeros for n > x.
  - Tₙ = ψₙ/ξₙ = Tₙ₋₁ Rₙ / (Dₙ(x) + n/x).
  - 1/ξₙ = Rₙ/ξₙ₋₁.
  - aₙ = Tₙ Nₐ / (Dₙ(z)/m + n/x − Rₙ), with Nₐ = (n + 1)(1/m² − 1)/x + uₙ₊₁(x) − uₙ₊₁(z)/m.
  - bₙ = Tₙ N_b / (mDₙ(z) + n/x − Rₙ), with N_b = uₙ₊₁(x) − m uₙ₊₁(z).
  - Nₐ and N_b equal Dₙ(z)/m − Dₙ(x) and mDₙ(z) − Dₙ(x) with the cancelling n/x terms removed exactly; without this, bₙ lost all its digits at x = 10⁻⁶.
  - Tₙ and 1/ξₙ underflow gracefully to zero, so no `n_max` can overflow or produce NaN.
- **Switch point.** nₛ = ⌊x⌋ is the starting value; the accuracy maps tune it.
- **Small-particle expansion.** Dropped: the ratio form is exact at every x. Main spec §5.1 kept each baseline element only if the maps confirm it, so this change must be confirmed by them too.
- **Comparison.** sasktran2's MIEV0-style code is the baseline data point on the accuracy-map page.

### 2.5 Absorption without cancellation

Computing Q_abs = Q_ext − Q_sca loses about log₁₀(Q_ext/Q_abs) digits. For k = 10⁻⁸ that is about 8 digits of the co-albedo 1 − ω, which radiative transfer depends on.

From the Casoratian there is an exact per-term identity:

- Re aₙ − |aₙ|² = −Im(Dₙ(z)/m) / |Δₐ|²
- Re bₙ − |bₙ|² = −Im(mDₙ(z)) / |Δ_b|²

So Q_abs = (2/x²) Σ (2n+1) [−Im(Dₙ/m)/|Δₐ|² − Im(mDₙ)/|Δ_b|²].

- Each term is computed directly and is exactly zero when Im m = 0.
- In the ratio region, |Δ|² = |ξₙ|² · |…|², computed through 1/ξₙ.
- Q_sca keeps its usual sum, and **Q_ext = Q_sca + Q_abs**. The usual Q_ext sum of Re(aₙ + bₙ) is inaccurate for small non-absorbing spheres, where Re aₙ = |aₙ|² ≪ |aₙ| (relative error 1 at x = 10⁻⁶). The optical theorem becomes the invariant test.
- The identity was checked against Arb at four (n, x, m) points during design; the agreement was 10⁻⁵⁶ or better.

### 2.6 Other outputs

- Q_ext = (2/x²) Σ (2n+1) Re(aₙ + bₙ).
- Q_sca = (2/x²) Σ (2n+1)(|aₙ|² + |bₙ|²).
- Q_back = (1/x²) |Σ (2n+1)(−1)ⁿ(aₙ − bₙ)|².
- g = (4/(x² Q_sca)) Σ [n(n+2)/(n+1) Re(aₙa*ₙ₊₁ + bₙb*ₙ₊₁) + (2n+1)/(n(n+1)) Re(aₙb*ₙ)].
- S₁, S₂ at caller-supplied μ = cos Θ, using the upward πₙ/τₙ recurrences, with many μ per call.

### 2.7 Closed-form derivatives

Let D = Dₙ(z) and D′ = n(n+1)/z² − 1 − D². Then:

| Derivative | Formula |
|---|---|
| ∂aₙ/∂m | i (D − zD′) / (m² Δₐ²) |
| ∂bₙ/∂m | −i (D + zD′) / Δ_b² |
| ∂aₙ/∂x | i (m² − 1)(n(n+1)/x² + D²) / (m² Δₐ²) |
| ∂bₙ/∂x | i (m² − 1) / Δ_b² |

- **Derivation.** Each follows from the Wronskian. Proofs go in the theory chapter. All four matched Arb finite differences to better than 10⁻²⁴ during design.
- **Cost.** Each derivative costs about one complex multiply on top of quantities already computed, so the full Jacobian should cost well under the 2× target.
- **Holomorphy.** aₙ and bₙ are holomorphic in m, so ∂/∂(Re m) = d/dm and ∂/∂(Im m) = i·d/dm. Complex outputs (coefficients, S₁, S₂) store d/dx and d/dm. Real outputs (efficiencies, g) store ∂/∂x, ∂/∂Re m and ∂/∂Im m.
- **Real outputs.** Their Jacobians follow by the chain rule. Q_abs is differentiated through its direct formula, using dDₙ/dz = D′ and dΔ/dp built from the same quantities.
- **Small z.** D + zD′ = O(z) while D ≈ (n + 1)/z, so the closed form cancels catastrophically. It is evaluated as (2n + 1)uₙ₊₁ − z(1 + uₙ₊₁²), with D′ = (n + 1)(2uₙ₊₁/z − 1/z²) − (1 + uₙ₊₁²) and D − zD′ = 2D − (D + zD′).
- **Radius.** The low-level API differentiates with respect to x; users apply ∂x/∂r = 2π/λ.

### 2.8 Scalars and complex arithmetic

- **Generic kernels.** Every kernel is generic over `T: Real`:
  - `f64` is the production type;
  - `DoubleDouble` drives the dense accuracy maps;
  - `Dual<T, N>` is the derivative oracle.
- **Complex reciprocal.** A scatrs-owned Smith reciprocal replaces num-complex's in the recurrences. It does not overflow or underflow for |w| near the range limits, which also fixes the DD hazard (b) in the follow-ups, and is validated against Arb.

### 2.9 Input rules

| Error | When |
|---|---|
| `InvalidInput` | x outside [10⁻³⁰, 10⁶]; m not finite; Re m < 0 or Im m < 0; \|m\| outside [10⁻⁶, 10³]; `n_max` = 0 or > `MAX_MIE_ORDER`; μ NaN or outside [−1, 1]; buffer lengths that differ from the cosines |
| `OutsideValidatedDomain` | valid inputs outside the declared domain, or `n_max` < N, unless `allow_unvalidated` is set |

## 3. Rust API (`scatrs::mie`)

```rust
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct MieOptions { pub n_max: Option<usize>, pub allow_unvalidated: bool }
// plus builder methods, because #[non_exhaustive] blocks struct literals outside the crate

pub const MAX_MIE_ORDER: usize = 1 << 20;
pub fn default_n_max(x: f64) -> usize;

/// Reusable per-thread workspace; it does not allocate once warm.
pub struct Mie<T: Real> { /* buffers */ }
impl<T: Real> Mie<T> {
    pub fn new() -> Self;
    pub fn compute(&mut self, x: T, m: Complex<T>, opts: &MieOptions) -> Result<MieSolution<'_, T>>;
    pub fn compute_with_derivatives(&mut self, x: T, m: Complex<T>, opts: &MieOptions)
        -> Result<MieSolution<'_, T>>;
}

/// Borrowed view of one solved sphere. Index 0 of each slice is order n = 1.
impl<T: Real> MieSolution<'_, T> {
    pub fn n_terms(&self) -> usize;
    pub fn a(&self) -> &[Complex<T>];
    pub fn b(&self) -> &[Complex<T>];
    pub fn derivatives(&self) -> Option<CoefficientDerivatives<'_, T>>; // da_dx, da_dm, db_dx, db_dm
    pub fn efficiencies(&self) -> Efficiencies<T>;          // q_ext, q_sca, q_abs, q_back, g
    pub fn efficiencies_jacobian(&self) -> Option<EfficiencyJacobian<T>>; // .x, .m_real, .m_imag
    pub fn amplitudes(&self, mu: &[T], out: AmplitudesOut<'_, T>) -> Result<()>;
    pub fn amplitudes_with_derivatives(&self, mu: &[T], out: AmplitudesJacobianOut<'_, T>) -> Result<()>;
}
```

- **Output buffers.** `AmplitudesOut` and `AmplitudesJacobianOut` are small structs of `&mut` slices supplied by the caller (main spec §4.5).
- **Exhaustiveness.** The output structs are `#[non_exhaustive]`, so new outputs can be added without a breaking change.
- **Batching.** Parallel batch functions over many (x, m) arrive in 1b with rayon.

## 4. Python API (`scatrs.mie`)

```python
scatrs.mie.default_n_max(x) -> int ndarray
scatrs.mie.coefficients(x, m, *, n_max=None, derivatives=False, allow_unvalidated=False) -> MieCoefficients
scatrs.mie.efficiencies(x, m, *, n_max=None, derivatives=False, allow_unvalidated=False) -> MieEfficiencies
scatrs.mie.amplitudes(x, m, mu, *, n_max=None, derivatives=False, allow_unvalidated=False) -> MieAmplitudes
```

- **Shapes.** x and m broadcast to a shape S.
  - Efficiencies have shape S.
  - Amplitudes have shape S + (len(mu),), where mu is one-dimensional.
  - Coefficients have shape S + (N_max,), zero-padded beyond each element's own order. `n_terms` (shape S) records that order.
- **Results** are frozen dataclasses holding numpy arrays.
- **Derivatives.** `derivatives` is `True`, `False`, or a subset of `("x", "m_real", "m_imag")`. When requested:
  - `result.d(param)` returns the same structure holding derivatives;
  - `result.params` lists the parameters;
  - complex outputs also accept `d("m")`, the complex derivative.

  This mirrors the high-level `.d()` of main spec §9.2.
- **Binding rules.** Phase 0 patterns apply:
  - `Bound<PyArray…>` + `try_readonly()`;
  - broadcasting through strided views, not materialised copies, with an explicit alignment check;
  - the GIL released for the whole loop.
- **Threading.** One `Mie<f64>` workspace per call; the loop is serial until 1b.

## 5. Validated domain and accuracy maps

- **Declaration.** The domain lives in code as `mie::VALIDATED_DOMAIN`: a short list of (x band × m region) cells, each with a tolerance per output.
  - The starting target is main spec §5.1: x ∈ [10⁻⁶, 2×10⁴], Re m ∈ (0, 10], Im m ∈ [0, 10].
  - The published domain is whatever the maps support.
- **Error metrics.**

  | Output | Metric |
  |---|---|
  | Q_ext, Q_sca, Q_abs, Q_back, g | relative error |
  | aₙ, bₙ | max over n of \|Δ\|, divided by max over n of \|ref\| |
  | S₁, S₂ | \|ΔS(μ)\| / max(\|S(μ)\|, S_rms), where S_rms = x√Q_sca / 2 is the RMS amplitude, so deep interference minima are not over-penalised |
  | derivatives | same metric as the corresponding output |

- **Conditioning near m = 1.** aₙ ∝ (m − 1) as m → 1, so its relative condition number with respect to m is about |m|/|m − 1|. Cells near m = 1 declare a tolerance with that factor, and the maps show it explicitly.
- **Sweep tool.** A Rust example, `accuracy_map`, compares f64 against `DoubleDouble` on a dense grid.
  - The grid is about 60 log-spaced x values against an (Re m, Im m) grid that includes the hard regimes.
  - It writes per-cell maxima to a small JSON file under `docs/validation/data/`. A docs notebook plots the heatmaps.
  - DD is a trustworthy reference here because it is checked against Arb at every reference point, and its own algorithmic error is ε_DD/ε_f64 ≈ 10⁻¹⁶ times the f64 error.
- **CI enforcement.** A reference-backed Rust test checks that:
  - every Arb point inside the declared domain meets its tolerance;
  - points outside the domain raise `OutsideValidatedDomain`;
  - `allow_unvalidated` computes them anyway.
- **Scheduled job.** It reruns the dense DD sweep and fails if any cell exceeds its declared tolerance.

## 6. Arb reference data (`reference/generators/mie.py`)

### 6.1 Rigorous coefficients

- Dₙ(mx) is seeded at the top order from Arb Bessel-function ratios (rigorous), then recurred downward in ball arithmetic.
- ψₙ and χₙ are computed in ball arithmetic, with working precision raised until the balls are tight.
- No step is heuristic, so aₙ, bₙ and their closed-form derivatives are **certified**.

### 6.2 Series outputs (Q, g, S)

These are certified only where the generator sums to N_ref ≥ n₀(x, m) and a proven tail bound covers n > N_ref.

- **Why n₀ ∝ |m|x.** Below about Re(m)·x, resonances can make individual terms O(1) (section 2.2), so the bound only becomes useful beyond n₀ ≈ c·|m|x.
- **The bound.** It comes from continued-fraction bounds on spherical-Bessel ratios and gives a geometric tail.
- **Phase 1a task.**
  - Derive the bound, prove it in the theory chapter, implement it in the generator, and test that it dominates the computed terms.
  - The task is time-boxed. If it does not finish, series outputs stay "high-precision" and the gap is recorded.
  - Cases where N_ref ≥ n₀ is too expensive (large |m|x) are labelled high-precision.
  - Labels are never overstated.

### 6.3 Reference semantics

- References give the infinite-series value. scatrs' truncation error is therefore part of the measured error, which validates the truncation rule as well.

### 6.4 Derivative references

- Values come from the closed forms in ball arithmetic.
- Independence comes from a generator self-test: on every case, the closed forms are compared against Arb central differences with Richardson extrapolation at ≥ 100 digits (main spec §8.4).

### 6.5 Case grid, about 330 cases

- x ∈ {10⁻⁶, 10⁻⁴, 10⁻², 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 10³, 2·10³, 5·10³, 10⁴, 2·10⁴}.
- Fourteen m values covering:
  - near-unity (1.0001, 1.01);
  - water and aerosol (1.33, 1.33 + 10⁻⁸i, 1.5 + 10⁻³i, 1.5 + 0.1i, 1.75 + 0.44i);
  - high index (2.5 + i, 4 + 0.01i, 10, 10 + 10i);
  - strong absorption (1.5 + 5i);
  - metal-like (0.2 + 3i, 0.05 + 4i).
- About 50 seeded random cases.

### 6.6 Stored per case

- All efficiencies and g, with their Jacobians.
- S₁ and S₂ at 13 fixed angles, including 0° and 180°, with derivatives.
- aₙ and bₙ at every order when N ≤ 64; otherwise about 40 log-spaced orders plus the last 8.
- Inputs are made portable with the phase 0 Arb-rounding rules.

### 6.7 Data budget

- The Mie data aims for ≤ 6 MB.
- The phase 0 size test (≤ 20 MB in total) is added first (section 10).

## 7. Tests beyond the references

- **Derivative oracle.** The kernel run with T = `Dual<f64, 3>` (seeds x, Re m, Im m) agrees with the analytic Jacobian to ≤ 10⁻¹² relative across the reference grid.
- **Invariants (proptest):**
  - Q_abs = 0 exactly when Im m = 0;
  - Q_abs ≥ 0, ω ∈ [0, 1], |g| ≤ 1;
  - |aₙ|² ≤ Re aₙ, and likewise for bₙ;
  - Q_ext = Q_sca + Q_abs to rounding;
  - optical theorem Q_ext = (4/x²) Re S(0);
  - S₁ = S₂ at μ = 1 and S₁ = −S₂ at μ = −1;
  - Rayleigh limit;
  - D-recurrence and ratio-form agreement at the switch point.
- **Published tables.** Wiscombe's MIEV0 test cases, the published table values (as in sasktran2's tests), are used as Rust fixtures with attribution.
- **DD versus f64 smoke test** at a few points in CI.
- **Python tests:**
  - broadcasting, padding and `n_terms`;
  - zero-copy inputs;
  - error types;
  - `.d()` against finite differences;
  - concurrent calls with the GIL released.

## 8. Performance

- **Targets** (main spec §10.1):
  - value-only is faster per (x, m) than sasktran2's `mie_f.rs` and miepython, across x ∈ [0.1, 10⁴];
  - the full (x, Re m, Im m) Jacobian costs ≤ 2× value-only.
- **Phase 1a** adds criterion benchmarks: values, Jacobian, and amplitudes at 100 angles, across x. These justify the always-downward Dₙ and the hybrid ψ scheme.
- **Phase 1b** adds the instruction-count CI gate, the batch kernels and the cross-code comparison.

## 9. Phase 1b outline

- **Gauss–Legendre.**
  - Bogaert's O(n) method above a threshold, cross-checked against the DD Newton rule and Arb.
  - API rework: `#[non_exhaustive]`, caller-supplied buffers, and caller-owned rule caching (an `Arc`, never a global).
  - Cross-platform golden digests.
- **Batched generalised spherical functions** at many nodes.
- **Phase matrix and Greek coefficients.**
  - Sphere phase-matrix elements F₁₁, F₁₂, F₃₃, F₃₄ with derivatives.
  - Greek α₁…α₄, β₁, β₂ by exact quadrature with n_q = N + ⌈(L+1)/2⌉, in normalized and unnormalized forms (main spec §6.2), with Jacobians.
  - `scatrs.mie.greek`, plus its Arb references and accuracy map.
- **Parallelism.** rayon batch kernels with a caller-supplied pool; bitwise determinism on 1, 2 and 8 threads; Python `set_num_threads` / `num_threads=`.
- **Benchmarks.** iai-callgrind gate on Linux; `benchmarks/compare.py` against sasktran2 and miepython, with results in the docs.
- **Fuzzing.** cargo-fuzz targets for the Mie inputs, GL and Wigner, run in the scheduled job.
- **Wheels.** Install-and-test musllinux wheels.

## 10. Phase 0 carry-overs folded into the first 1a task

- A `RUSTDOCFLAGS=-D warnings cargo doc` CI step.
- `# Examples` doctests for `gauss_legendre`, `wigner_d`, `DoubleDouble` and `Dual`.
- Fix the intra-doc link at `dd/traits.rs:2`.
- Extend the docs-consistency test: record counts, data size, MSRV, Python and numpy floors, GL timings, Wigner ranges.
- The reference data-budget size test.
- Declare `packaging` in the `test` extra.

## 11. Amendments to the main spec

- **§5.1:**
  - the small-particle expansion and the upward ψ recurrence are replaced by the hybrid scheme of section 2.4;
  - Dₙ is always computed downward;
  - Q_abs uses the direct formula;
  - the closed-form derivatives are those of section 2.7;
  - the truncation limitation of section 2.2 is documented.
- **§4.3:** the strategy trait is deferred to phase 2.
- **§8.1:** coefficients are certified directly; series outputs need the tail bound, with N_ref ≥ n₀(x, m).
- **§9.3:** add `InvalidInputError`, and replace `PyReadonlyArray` parameters with `Bound<PyArray…>` + `try_readonly()`.
- **§13:** phase 1 is split into 1a and 1b, with the exit criteria of section 1.

## 12. Decisions made while planning (2026-10-13)

The plan's code was prototyped and measured before the plan was written. These results changed
the design above, or settle what it left open:

- **Truncation** constants (section 2.2): measured against Arb.
- **No continued fraction** (section 2.3): the downward recurrence from a contracting start is
  simpler, as cheap, and cannot fail.
- **Cancellation-free numerators and D + zD′** (sections 2.4 and 2.7), and
  **Q_ext = Q_sca + Q_abs** (section 2.5): found by the reference comparisons.
- **Accuracy model** (section 5). A tolerance C·κ·2⁻⁵² fails near the sharp resonances of
  weakly absorbing and lossless spheres: on the dense grid the needed C reached 360 for the
  coefficients and 7600 for the efficiency Jacobian, and grows as the grid is refined (and g of
  a small strongly absorbing sphere needs 98). A tolerance from the output's own condition
  number alone fails where the series cancel: metallic backscattering needs C ≈ 6·10⁵. The
  declared tolerance is C·(κ′ + cond)·2⁻⁵², with cond the relative condition number of the
  output in its error measure; every output then stays within a small C (≤ 10 for values,
  ≤ 70 for derivatives). Amended during execution (Task 5 review): cond is componentwise in
  Re m and Im m for the real outputs (a normwise cond made the tolerances of Q_ext and Q_abs
  vacuous at small Im m) and normwise for the holomorphic ones; κ′ is κ for the values,
  κ·max(1, |m|/|m − 1|) for the efficiency Jacobian and 1 + |m|x for the coefficient and
  amplitude derivatives. Declared constants: 16 (coefficients, efficiencies, g), 64 (S₁, S₂
  and their derivatives), 256 (efficiency Jacobian), 512 (coefficient derivatives).
  `DoubleDouble` is checked against Arb on every output, within 16 times the declared
  constants in units of 2⁻¹⁰⁴ (section 5 expected it to be far better than that relative to
  f64; at x = 2·10⁴ its amplitude derivatives are not, a phase 1b follow-up).
- **The accuracy map is a CI test** (`scatrs/tests/mie_accuracy_map.rs`, about 20 s with all
  cores), not only a scheduled sweep: f64 against DoubleDouble on 104 × 24 points, with
  decimal-valued size parameters so the grid is the same on every platform.
- **Reference generator** (section 6). Arb's Bessel functions are unusable at large orders (5 of
  320 bits at order 2000), so the generator uses only the recurrences, seeded by the
  unit-disk enclosure, with disk arithmetic for complex recurrences (Arb's rectangular balls
  grew 1.5× per step). π_n, τ_n come from Arb's Legendre polynomials. The tail bound needs
  2n_ref + 3 ≥ 2|m|x and n_ref ≥ 1.5x − ½; it certifies 297 of 306 series records (the rest
  would need more than 60 000 terms). Series of derivatives are high-precision. Nine angles;
  the data take 5.0 MB. Cases run in parallel processes (about 30 s on 10 cores).
- **Q_abs Jacobian measure**: ∂Q_abs/∂x and ∂Q_abs/∂(Re m) of weakly absorbing spheres are
  accurate relative to Q_ext, not Q_abs (relative error up to 4·10⁻⁹ at k = 10⁻⁸ near a
  resonance; `Dual` gives the same values, so it is the conditioning of the computed f64
  path). ∂Q_abs/∂(Im m) stays accurate relative to itself.
- **Measured cost** (Apple M4 Pro): 0.55 µs at x = 0.1, 36 µs at x = 1000, 0.34 ms at
  x = 10⁴ (m = 1.5 + 0.01i); the full Jacobian costs 1.3–1.8× the values there (target ≤ 2×).
  Amended during execution (Task 7 review): at real m the values take the real recurrence and
  cost about 25% less, so the Jacobian reaches 2.0× from x = 1000 on, at the limit of the
  target (a phase 1b item). The benchmarks do not compare alternative algorithms, so they do
  not by themselves justify the downward Dₙ and the hybrid ψ scheme (section 8); the accuracy
  measurements do.
- **Python amplitude shape** is S + mu.shape (any shape of mu), and `derivatives` accepts
  `True`, `False` or a sequence of names.
- **Rust API names** follow the plan: output buffers are `AmplitudeBuffers { s1, s2 }` for the
  values and for each derivative; the input limits are `MIN_SIZE_PARAMETER`,
  `MAX_SIZE_PARAMETER`, `MIN_REFRACTIVE_INDEX`, `MAX_REFRACTIVE_INDEX` and `MAX_MIE_ORDER`;
  `ValidatedDomain::tolerance(output, x, m, condition)` takes the condition number.
- **Python output bounds and validation** (section 4; amended during execution, Task 6
  review). Broadcasting makes outputs far larger than inputs, so every Python call bounds its
  output by `MAX_MIE_VALUES` (2²⁷ complex values, or twice as many real values for the
  efficiencies) before allocating anything, and allocates without the GIL. Rust gains
  `scatrs::mie::n_terms(x, m, &opts)`, which validates as `compute` does and returns the number
  of terms without solving; the binding sizes the coefficients with it. Results are frozen
  dataclasses of read-only arrays; `default_n_max` raises `InvalidInputError` for x outside
  the accepted range.
