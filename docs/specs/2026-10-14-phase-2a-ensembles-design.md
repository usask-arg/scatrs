# Phase 2a design: ensembles of spheres (size distributions, adaptive integration, mixtures)

Status: approved 2026-10-14. Builds on phase 1b (`docs/specs/2026-10-13-phase-1b-greek-design.md`)
and the main spec (`docs/specs/2026-10-12-scatrs-design.md`, sections 6, 7 and 9.1).

Phase 2 of the main spec is split into three sub-phases, in this order:

- **2a (this spec):** the Rust ensemble engine for spheres, with a low-level Python module.
- **2b:** the comparison with sasktran2's `MieDatabase` and the benchmark (the phase-2 exit
  criterion of main spec section 13).
- **2c:** the high-level declarative Python API (main spec section 9.2).

## 1. Scope

In:

- the seven size distributions of main spec section 7.1, with cut-offs and analytic moments;
- two-pass adaptive Gauss–Kronrod size integration with error estimates for every output;
- every ensemble output of main spec section 7.4 for spheres;
- Jacobians with respect to distribution parameters, Re m, Im m and mixture weights;
- external mixtures of sphere ensembles;
- rayon parallelism, bitwise deterministic for any thread count;
- the Python module `scatrs.ensemble`;
- validation (identities, reference integrals, error-estimate calibration) and docs.

Out: distributions over parameters other than size (they need the T-matrix, phase 4), the
sasktran2 comparison (2b), the declarative API, refractive-index models and xarray (2c),
`set_num_threads` (2c), shared πₙ/τₙ tables and other speed work (phase 5).

Housekeeping: the branch starts with the clippy 1.99 fix of `phase.rs` (CI runs Rust 1.99).

## 2. Units and inputs

- Radii and wavelengths in the same (any) unit; x = 2πr/λ. Cross sections are per particle,
  in that unit squared, averaged over the number distribution.
- One refractive index per wavelength (per component, for mixtures).
- `f64` only: the ensemble engine is not generic over `Real` (the reference integrals of the
  validation use `f64` at a tight tolerance instead of double-double nodes, since the Mie
  rounding errors at the grid's sizes are far below the tolerances tested).

## 3. Distributions (`scatrs::ensemble::SizeDistribution`)

An enum (`#[non_exhaustive]`), each variant with named, differentiable parameters:

| Variant | Number density n(r) ∝ | Parameters |
|---|---|---|
| `LogNormal` | exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r | `median_radius` r_g, `width` σ_g > 1 |
| `Gamma` (Hansen & Travis) | r^((1−3b)/b) exp(−r/(ab)) | `effective_radius` a, `effective_variance` b ∈ (0, ½) |
| `ModifiedGamma` (Deirmendjian) | r^α exp(−b r^γ) | `alpha` > −1, `b` > 0, `gamma` > 0 |
| `PowerLaw` | r^(−α) on [r_min, r_max] | `exponent` α (limits fixed) |
| `Monodisperse` | δ(r − r₀) | `radius` r₀ |
| `Tabulated` | piecewise linear through (rᵢ, nᵢ) | none |
| `MultiModal` | Σ wⱼ nⱼ(r), each mode normalized | each mode's parameters, and `weight` wⱼ |

- **Normalization is numeric.** The engine integrates the unnormalized density and its
  parameter gradient over the integration range, and divides by their integral. This makes
  every distribution normalized over its cut-off range (the renormalization of main spec
  section 6.3) and needs no Γ functions. Analytic moments exist for every variant except
  `Tabulated` and are used by the tests.
- **Cut-offs.** Finite-support variants use their limits. For the others, the range starts
  at a characteristic radius and marches outward in ln r in steps of ½ until the remaining
  area-weighted (r²) mass, bounded by the geometric decay of the last steps, is below rtol/10
  of the total on each side. After pass 1 (section 4), the range is extended by a further
  step at any end where the Mie integrand is not negligible (above rtol/10 of the total per
  unit ln r): the r² tail is not conservative when the bulk of the distribution is in the
  Rayleigh regime and the tail is not.
- **Invalid parameters** (non-finite, σ_g ≤ 1, b outside (0, ½), r_min ≥ r_max, a table that is
  not increasing or has negative densities, mode weights not positive) are `InvalidInput`.

## 4. Size integration

Per wavelength, in t = ln r, with ∫ C p dr = ∫ C(r) p(r) r dt.

- **Rule.** Gauss–Kronrod G7–K15 panels (QUADPACK's nodes and weights). The G7 nodes are a
  subset of the K15 nodes, so every output gets |K15 − G7| per panel at no extra cost.
- **Initial panels.** Width at most ½ in t and at most Δx = 20 in size parameter (the broad
  interference structure of the efficiencies has a period of about π/|m − 1|), with
  `Tabulated` breakpoints as panel edges.
- **Pass 1 (refinement).** Only the efficiencies are evaluated (O(N) per radius). The refined
  quantities are ⟨C_ext⟩, ⟨C_sca⟩ and ⟨C_sca·g⟩, plus, for each requested distribution
  parameter θ, the integrals weighted by ∂p/∂θ. Each has a target: rtol·⟨C⟩ for the cross
  sections, rtol·⟨C_sca⟩ for ⟨C_sca·g⟩, and rtol·∫|C·∂p/∂θ|dr for the derivative
  integrals. In rounds, every panel whose estimate exceeds its share (target/number of
  panels) of any quantity is halved, until all totals meet their targets or the panel count
  reaches `max_panels` (default 4096), which sets `converged = false` (not an error: without
  absorption, narrow resonances cannot all be resolved and average out). Rounds are
  evaluated in parallel; decisions depend only on values, so the panels are deterministic.
  Optionally, Greek orders up to `refine_order` join the refined quantities.
- **Pass 2 (outputs).** At the final K15 nodes, the expensive outputs (section 5) are computed
  once, with derivatives if requested.

## 5. Outputs (per wavelength)

- ⟨C_ext⟩, ⟨C_sca⟩, ⟨C_abs⟩ (from Q_abs, without cancellation), ω = ⟨C_sca⟩/⟨C_ext⟩,
  g = ⟨C_sca g⟩/⟨C_sca⟩.
- **Lidar:** ⟨C_back⟩ (Bohren & Huffman's σ_b = 4π dσ/dΩ at 180°), the lidar ratio
  4π⟨C_ext⟩/⟨C_back⟩, and the linear depolarization ratio δ, exactly 0 for spheres.
- **Moments** over the integration range: r_eff = ⟨r³⟩/⟨r²⟩, ⟨πr²⟩, ⟨4πr³/3⟩.
- **Greek coefficients** up to order L, or all orders (L = 2N_max), normalized (α₁⁰ = 1) or
  unnormalized (α₁⁰ = ⟨C_sca⟩), with a shared projector: n_q = N_max + ⌈(L + 1)/2⌉ nodes,
  where N_max is the largest number of terms at any final node, which is exact for every
  radius. The tables πr²·F (C_sca times the normalized phase matrix) are accumulated at those
  nodes with the integration weights and projected once (phase 1b's `GreekProjector`).
- **Phase matrix** F₁₁, F₁₂, F₃₃, F₃₄ at given cosines, normalized or unnormalized.
- **Error estimates** for every output, of the same shape: Σ over panels of |K15 − G7| for
  the integrals; for the Greek coefficients, (2l + 1)/2·Σₖ wₖ|ΔT(μₖ)||d_l(μₖ)|, the projection
  with absolute values of the per-node sums of the panels' |K15 − G7| tables (a bound by the
  triangle inequality); ratios (ω, g, normalized forms) by first-order propagation.
- **Flags:** `converged`, the number of panels, and the integration range.

## 6. Jacobians

Requested parameters only; every output above has one.

- **Distribution parameters:** ∂/∂θ ∫ C p dr through ∂p/∂θ at the same nodes, including the
  numeric renormalization (quotient rule with ∫∂p/∂θ). Cut-offs are held fixed within a call.
- **Monodisperse radius:** through the single-sphere ∂/∂x (times 2π/λ) and ∂(πr²)/∂r.
- **Re m, Im m** (of each component, at each wavelength): from the single-sphere Jacobians
  and phase-matrix derivative tables, accumulated with the same weights.
- **Mixture weights** wᵢ.
- **Normalized and ratio outputs:** quotient rule.

## 7. Mixtures

`compute_mixture(components, wavelengths, options)`: each component is a distribution, its
refractive indices and a weight wᵢ ≥ 0 (not all zero). Cross sections and moments are averaged
with wᵢ/Σw; phase-matrix tables are combined in C_sca units with the same weights (so phase
matrices are averaged with wᵢ·C_sca,ᵢ), before one projection over all components.
`compute(distribution, …)` is a mixture of one component without the weight parameter.

## 8. Errors, domain and determinism

- Every node's (x, m) must be in phase 1a's validated domain unless `allow_unvalidated`;
  otherwise `OutsideValidatedDomain`, naming the wavelength and radius. (Tails of broad
  distributions can push x past 2·10⁴.)
- Invalid inputs (lengths that do not match, non-finite or non-positive wavelengths, invalid
  m, invalid distribution parameters, rtol outside [1e-13, 0.1], cosines outside [−1, 1]) are
  `InvalidInput`; nothing panics and nothing returns a silent NaN.
- Parallel over wavelengths and over panels on the current rayon pool (the caller may
  `install` its own); reductions in a fixed order, so results are bitwise identical for any
  thread count. Each task owns its Mie workspace.
- New dependency of the core: `rayon` (allowed by main spec section 4.5).

## 9. Python: `scatrs.ensemble`

- Distributions are frozen dataclasses with the parameter names of section 3;
  `Component(distribution, refractive_index, weight=1.0)`.
- `compute(distribution, wavelength, refractive_index, *, greek=None, mu=None,
  normalized=True, derivatives=False, rtol=1e-6, max_panels=4096, allow_unvalidated=False)`
  and `compute_mixture(components, wavelength, *, …)`. `wavelength` is 1-D; refractive
  indices broadcast to it. `greek` is an order, `"all"`, or `None`.
- A frozen `EnsembleResult` with arrays over wavelength (Greek coefficients zero-padded to
  the largest order), `.error` (the same structure), `.d(param)`, `converged`, `panels`.
  Parameter names: the distribution's (`"median_radius"`, …), `"m_real"`, `"m_imag"`; for
  mixtures and multimodal distributions prefixed by component and mode
  (`"c0.median_radius"`, `"c1.weight"`, `"mode0.width"`).
- GIL released; outputs bounded like the other modules.

## 10. Validation

- **Identities:** monodisperse equals the single sphere (cross sections, Greek coefficients,
  phase matrix); integrated moments equal the analytic ones to rtol; Rayleigh-limit ensembles
  against the closed form with ⟨r⁶⟩; mixture linearity and weight derivatives; the
  normalized phase function integrates to 1; ω, g, α₁¹ = 3g.
- **Jacobians:** against central differences of the whole computation with the integration
  range and panels frozen (an internal entry point), so the difference differentiates the
  same discretization.
- **Reference integrals and calibration:** on a grid of distributions × refractive indices ×
  wavelengths, the result at rtol 1e-6 against the same at rtol 1e-11: when `converged`, the
  true error of every refined quantity is within rtol of its scale, and the true error of
  every output is within its reported estimate. A CI test, and a table in the docs.
- **Determinism:** thread pools of 1 and 4 give bitwise identical results.
- Python: shapes, broadcasting, errors, `.d`, frozen results, GIL release.

## 11. Docs

A theory page (distributions, cut-offs, the two passes, the shared projector, error
estimates, Jacobians), an executed example (a lognormal sulfate-like aerosol across
wavelengths: cross sections, ω, g, Greek coefficients, derivatives), the accuracy page
(calibration), limitations (cost of pass 2 for broad distributions of large spheres;
unresolvable ripples without absorption), CHANGELOG.

## 12. Changes during implementation

Recorded 2026-10-14, after prototyping and measuring, and amended after the engine review
(marked *review*):

- **Cut-offs.** The Mie integration range bounds only the area (r²) tail, at rtol/10 (as in
  section 3); the cross sections of small spheres fall faster than r². The normalization
  Z = ∫u dr, ⟨r²⟩ and ⟨r³⟩ are integrated separately, without Mie solutions, on a moment range
  that bounds the number (r⁰), area and volume (r³) tails at max(10⁻³·rtol, 10⁻¹⁶), and every
  average is S/Z. The volume tail is not used for the Mie range (for σ_g = 2 it pushed the range
  to x ≈ 5000). *Review:* the Mie range first bounded the number tail too, at rtol/10, since
  renormalizing over it moves every average by the omitted number fraction; with Z from the
  moment integration the tiny spheres of a long number tail need no Mie solutions, and the
  smaller tail of the moment range covers the derivatives, which weight the tails by about
  (∂ ln u/∂θ)². The cut-offs move each average by at most about 2·rtol/10, which every error
  estimate includes.
- **Peak width** (*review*). The march of the cut-offs and the initial panels step by at most
  min(½, ½·w) in ln r, w the width of the density's peak in ln r: ln σ_g (lognormal),
  √(b/(1 − 2b)) (gamma), 1/√(αγ) (modified gamma with α > 0, else 1/γ), ∞ for finite support.
  Steps of ½ skipped over narrow peaks.
- **Range extension** (*review*). After pass 1 meets its targets, the range is extended at an
  end where any refined integrand (values and every derivative block) per unit ln r at the end
  node exceeds a tenth of its target, not only the extinction integrand: the cross sections of
  small absorbing spheres and the derivatives can have heavier tails than the area. The θ
  blocks are compared with rtol·max(|their integral|, 10⁻³·their absolute integral), since they
  cancel. Each extension is ½ in ln r, or w for narrower peaks, with the end criterion scaled by
  the extension over ½; at most 100 per side; an extension beyond `max_panels` returns the
  result unconverged.
- **Refined quantities.** Pass 1 refines on 1, ⟨C_ext⟩, ⟨C_sca⟩ and ⟨C_sca g⟩, the requested
  distribution-parameter integrals (to rtol·∫|C·∂p/∂θ|dr, with no |θ| factor), and, when
  derivatives with respect to m are requested, the derivatives of the cross sections with
  respect to m (to rtol of their absolute integrals): without that, their estimates fell short
  by a factor of 3. ⟨C_back⟩ is refined only with `refine_backscatter` (to rtol of
  max(⟨C_back⟩, ⟨C_sca⟩), as phase 1a measures Q_back): its glory ripples made a broad weakly
  absorbing case unconverged at 4096 panels; without refinement its estimate is indicative (it
  fell short by up to a factor of 7.6 on a derivative, before the review). Panels are split only
  for quantities still above their targets. *Review:* ⟨C_abs⟩ is refined too, to rtol of
  ⟨C_ext⟩ (the accuracy of C_ext − C_sca), and its derivatives against the matching C_ext
  integrals: relative to C_abs itself, the absorption resonances of weakly absorbing large
  spheres took the whole panel budget (r_g = 1, σ_g = 2, m = 1.33 + 10⁻⁴i at λ = 0.5: 4096
  panels, unconverged, against 2634 converged). `refine_order` is at most `MAX_REFINE_ORDER`
  = 1024.
- **Projector bands.** One shared projector (n_q = N_max + ⌈(L + 1)/2⌉) made every small sphere
  pay for the largest: 20 s per wavelength for a broad distribution. Bands of the number of
  terms (growing by a factor 1.25) each have their own exact projector, to order
  L_b = min(L, 2·N_band) on N_band + ⌈(L_b + 1)/2⌉ nodes. *Review:* a band is chosen per panel,
  by the largest number of terms at its nodes, not per node: per-node bands made each band's
  table a step function within a panel, and the Greek error estimates loose by orders of
  magnitude.
- **Initial panels** are capped at `max_panels` (about `max_panels` equal panels in ln r beyond
  it, at least one between `Tabulated` breakpoints).
- **Power law** (*review*). The reference radius of the unnormalized density is r_min for
  α ≥ 0 and r_max otherwise, where the density is largest.
- **Errors** (*review*). `InvalidInput` also for a derivative parameter requested twice,
  mixture or mode weights with a non-finite sum, a range that reaches size parameters above
  `MAX_SIZE_PARAMETER` (before any panel), and a zero or non-finite ⟨C_ext⟩, ⟨C_sca⟩ or
  ⟨C_back⟩ (m = 1). Errors are deterministic: the first in wavelength and part order, whatever
  the thread count.
- **Validation.** Jacobians are checked against central differences at rtol 1e-11 rather than
  with frozen panels; the reference integrals are `f64` at rtol 1e-11. On the calibration grid
  (nine distributions, with backscattering refined; the review added a multimodal fine + coarse
  mode) the refined outputs are within their estimates and the others within twice theirs
  (|K15 − G7| is an estimate, not a bound): at most 0.003 of the estimate for the refined
  outputs and 1.210 for the others.
- **Measured cost** (Apple M4 Pro, 12 cores, per wavelength, best of 2, lognormal r_g = 1,
  σ_g = 2, m = 1.33 + 10⁻⁴i, rtol 1e-6): cross sections, g and moments 0.035 s at λ = 1
  (1465 panels), 0.091 s at λ = 0.5 (2634), 0.21 s at λ = 0.3 (4096), all converged; with Greek
  coefficients to order 32, 0.62 s, 2.9 s and 10.4 s. The exact Greek coefficients need each
  node's amplitudes at about N + L/2 cosines, O(N²) for large spheres. Recorded as a follow-up
  for the performance phase: Clebsch–Gordan Greek coefficients (O(N·L) per sphere) and shared
  πₙ/τₙ tables per band.
- **Non-absorbing spheres.** Derivatives with respect to Im m at Im m = 0 have a spike at every
  resonance and do not converge (flagged); the values do.
