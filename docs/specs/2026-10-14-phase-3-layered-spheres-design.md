# Phase 3 design: layered spheres

- **Date:** 2026-10-14
- **Status:** draft for review.
- **Parent spec:** `docs/specs/2026-10-12-scatrs-design.md` (cited as "main spec"). This document
  refines main spec §5.2, §6.1, §9 and §13 row 3, and lists its amendments in section 12. It
  builds on phase 1 (`docs/specs/2026-10-13-phase-1-mie-design.md`, cited as "phase 1 spec"), the
  ensemble engine of phases 2a/2b, and the model layer of 2c-1.

The user's decisions of 2026-10-14:

- Any number of layers (coated spheres are L = 2), not a separate core–shell solver.
- In ensembles, the size distribution gives the outer radius and each interface is at a fixed
  fraction of it.
- The full phase: kernel, low-level API, and layered spheres in `Ensemble`/`Mixture` with
  Jacobians.
- The engine refactor comes first, as its own reviewed step.

## 1. Scope and split

One spec, three plans. Each plan has its own exit criterion and review.

### 1.1 Phase 3a: coefficient layer and particle abstraction (refactor)

- The outputs that need only aₙ and bₙ are computed on a shared view of the coefficients rather
  than on `MieSolution`. These are the efficiencies, amplitudes, phase matrix, Greek
  coefficients, the angular-table methods and their derivatives (section 3.1).
- The ensemble engine takes a particle per component rather than a refractive index. It
  accepts derivative directions in any number (section 3.2).
- The binding and the Python layer pass a particle description to the engine. This is the
  "particle-side hook" of `docs/plans/phase-2c1-followups.md`.

**Exit criterion:**

- Every existing test passes, with no change to any test's expected values.
- Ensemble results, with and without derivatives, are bitwise unchanged on the test suite and on
  `benchmarks/compare.py`.
- The criterion benchmarks show no regression beyond noise.

### 1.2 Phase 3b: the layered kernel and its validation

- The kernel (section 2), forward-mode derivatives (section 2.7), the Rust API (section 4) and
  the low-level Python API `scatrs.layered` (section 5.1).
- The Arb generator (section 7), the accuracy maps and the declared validated domain
  (section 6).
- The identity tests (section 8), the scattnlay comparison (section 9) and the benchmarks
  (section 10).
- Docs: the theory chapter and the validation sections (section 11).

**Exit criterion:**

- Every Arb reference in the declared domain meets its tolerance in CI, and points outside the
  domain raise `OutsideValidatedDomain`.
- The identities of section 8 hold.
- The scattnlay comparison is published, with every disagreement attributed.

### 1.3 Phase 3c: layered spheres in ensembles

- `scatrs.LayeredSphere` (section 5.2) in `Ensemble`, in `Mixture` and in grids of
  distributions.
- Derivatives with respect to each layer's refractive index and each radius fraction.
- An example notebook on coated aerosol, and the docs updates.

**Exit criterion:**

- A one-layer `LayeredSphere` ensemble is bitwise equal to the `Sphere` ensemble.
- Ensemble Jacobians match central differences.
- Mixtures of spheres and layered spheres work.

### 1.4 Not in phase 3

| Item | Where it goes |
|---|---|
| Fractions or layer indices as grid axes (`radius_fraction=[...]` arrays) | Later. The batch engine shares one particle per call, as it shares one material today. |
| Shell thickness fixed in absolute units across a distribution | Later, if asked for. It needs per-node fractions in the engine. |
| Refractive-index model parameters of each layer (`layer{i}.index.<name>`) | Phase 2c-2, with those of `Sphere`. |
| Near fields and fields inside the particle | Not planned (main spec §1.2). |
| Non-concentric layers, inclusions | T-matrix territory (phase 4 and later). |
| A release | The user decides. The Rust `Component` change is breaking (section 3.2), so the next release is 0.3.0. |

## 2. Algorithm

### 2.1 Geometry and notation

- There are L ≥ 1 layers, counted from the core outward (l = 1 is the core). Layer l occupies
  r_{l−1} < r < r_l, with r₀ = 0 and r_L = r, the particle radius. The APIs number layers from
  0 (`layer0` is the core), like the other indices of the API.
- x_l = 2πr_l/λ, with 0 < x₁ ≤ x₂ ≤ … ≤ x_L. Interfaces of zero thickness are allowed.
- m_l is the refractive index of layer l relative to the medium, with Re m_l ≥ 0 and
  Im m_l ≥ 0, as in phase 1.
- The conventions are Bohren & Huffman with e^(−iωt), as in phase 1: ψₙ, χₙ, ξₙ = ψₙ − iχₙ.
  Also:
  - D⁽¹⁾ₙ = ψₙ′/ψₙ and D⁽³⁾ₙ = ξₙ′/ξₙ;
  - Πₙ(z) = ψₙ(z)ξₙ(z);
  - uₙ = ψₙ/ψₙ₋₁ (phase 1's ratio) and wₙ = ξₙ/ξₙ₋₁.
- The Wronskian ψξ′ − ψ′ξ = i gives D⁽³⁾ₙ − D⁽¹⁾ₙ = i/Πₙ.

### 2.2 The recursion (Yang 2003; Peña & Pal 2009)

For each order n, the effective log-derivatives at the outer surface of each layer,
Hₐ⁽ˡ⁾ and H_b⁽ˡ⁾, are found from the core outward:

- Hₐ⁽¹⁾ = H_b⁽¹⁾ = D⁽¹⁾ₙ(m₁x₁).
- For l = 2, …, L, with z₁ = m_l x_{l−1} and z₂ = m_l x_l:
  - G₁ᵃ = m_l Hₐ⁽ˡ⁻¹⁾ − m_{l−1} D⁽¹⁾ₙ(z₁) and G₂ᵃ = m_l Hₐ⁽ˡ⁻¹⁾ − m_{l−1} D⁽³⁾ₙ(z₁);
  - G₁ᵇ = m_{l−1} H_b⁽ˡ⁻¹⁾ − m_l D⁽¹⁾ₙ(z₁) and G₂ᵇ = m_{l−1} H_b⁽ˡ⁻¹⁾ − m_l D⁽³⁾ₙ(z₁);
  - with Qₙ = [ψₙ/ξₙ](z₁) / [ψₙ/ξₙ](z₂), Yang's form is
    H⁽ˡ⁾ = [G₂ D⁽¹⁾ₙ(z₂) − Qₙ G₁ D⁽³⁾ₙ(z₂)] / [G₂ − Qₙ G₁].
- Then aₙ and bₙ are the homogeneous formulas at the outer surface, with Hₐ⁽ᴸ⁾/m_L in place of
  Dₙ(mx)/m and m_L H_b⁽ᴸ⁾ in place of m Dₙ(mx), at x = x_L.

**Pole-free form.** Qₙ and D⁽¹⁾ₙ(z₂) have poles at the real zeros of ψₙ(z₂), but H⁽ˡ⁾ does
not. scatrs therefore uses

- H⁽ˡ⁾ = D⁽³⁾ₙ(z₂) − i G₂ / (Πₙ(z₂) G₂ − Pₙ G₁),

where Pₙ = Qₙ Πₙ(z₂) = [ψₙ/ξₙ](z₁)·ξₙ(z₂)². D⁽³⁾ₙ, Πₙ and Pₙ are finite for Im z ≥ 0: ξₙ has
no zeros there. H⁽ˡ⁾ keeps its genuine poles, where Πₙ(z₂)G₂ = PₙG₁. These carry into the next
layer as large G₁ ≈ G₂, which is harmless, and an exact zero denominator is nudged by a rounding
error, as phase 1's recurrence does.

Both forms, and the split of section 2.4, were checked during design against a direct
boundary-matching solve in 60-digit mpmath: three particles (L = 2 and 3, including a black
carbon core in sulfate and |m|x = 0.03 with m₁ = 10 + 10i) at orders 1, 2, 5 and 9. They agreed to 10⁻⁴⁰
or better.

### 2.3 Auxiliary functions

Each is computed per order n, for every argument z of each layer:

- **D⁽¹⁾ₙ(z)** comes from phase 1's downward ratio recurrence for uₖ(z), with the same start
  rule and margin. There are 2L − 1 complex arguments: m_l x_l for every l, and m_l x_{l−1} for
  l ≥ 2. Each costs about max(N, 1.5|z|) steps. For real m_l the real recurrence is used, as in
  phase 1.
- **Πₙ(z)** is computed by the upward recurrence Πₙ = Πₙ₋₁·uₙ·wₙ from Π₀(z) = (1 − e^(2iz))/2.
  Then D⁽³⁾ₙ = D⁽¹⁾ₙ + i/Πₙ, and wₙ₊₁ = (n + 1)/z − D⁽³⁾ₙ.
- **Pₙ** is computed by the upward recurrence Pₙ = Pₙ₋₁·[uₙ(z₁)/wₙ(z₁)]·wₙ(z₂)², from
  P₀ = e^(2i(z₂−z₁))(1 − e^(2iz₁))/2.
  - |e^(2i(z₂−z₁))| = e^(−2 Im m_l (x_l − x_{l−1})) ≤ 1, so P is bounded where Q would
    overflow.
  - A thick absorbing layer underflows P to 0: the layers inside it become invisible, which is
    correct.
- **1 − e^(2iz)** at small |z| needs a complex `expm1`. It is the first of the Kahan-style
  complex functions that the phase 1 spec (§1.3) deferred "until layered spheres need them". It
  is validated against Arb like the reciprocal.

### 2.4 The outer step: reusing the homogeneous kernel

The outer step is phase 1's kernel with Dₙ(mx) replaced by H⁽ᴸ⁾. That kernel has two regions,
split at n_s = ⌊x_L⌋.

- **Upward region, n ≤ n_s.** Hₐ⁽ᴸ⁾ and H_b⁽ᴸ⁾ are computed in the pole-free form, and aₙ, bₙ
  come from the direct formulas, as in phase 1.
- **Ratio region, n > n_s.** Phase 1 removes the cancelling n/x terms from the numerators by
  writing Dₙ(mx)/m − Dₙ(x) as Nₐ (phase 1 spec §2.4). Without this, bₙ lost every digit at
  x = 10⁻⁶. The layered numerators keep the same property through the split
  H⁽ᴸ⁾ = D⁽¹⁾ₙ(z₂) + δ, with
  - δ = −i Pₙ G₁ / [Πₙ(z₂)(Πₙ(z₂) G₂ − Pₙ G₁)], computed directly;
  - Nₐ → Nₐ + δₐ/m_L and N_b → N_b + m_L δ_b.

  For a single layer, or for layers of identical m, G₁ = 0 exactly, so δ = 0 and the result
  is phase 1's numerator.
- **The poles of D⁽¹⁾ₙ(z₂).** In the ratio region, D⁽¹⁾ₙ(z₂) can still have poles when
  x_L < n < |m_L|x_L. Near them, D⁽¹⁾ₙ(z₂) and δ are both large and cancel. The split is
  therefore used only for n > max(n_s, ⌊|m_L|x_L⌋), where ψₙ(z₂) has no real zeros. Orders in
  between use the pole-free H⁽ᴸ⁾ with the numerator H⁽ᴸ⁾/m_L − Dₙ(x_L) formed directly.
  - The cancellation there is at most about a factor |m_L|, and the band is thin when
    m_L ≈ 1, where it would be worst.
  - The accuracy maps confirm or move this switch, as they did n_s (phase 1 spec §2.4).
- **One layer.** For L = 1 the layered entry points call phase 1's kernel itself, so the results
  are bitwise those of `Mie`.

### 2.5 Absorption

- Phase 1's identity holds for any complex A: Re aₙ − |aₙ|² = −Im(A)/|Δₐ|². This was checked
  in mpmath with A = Hₐ/m_L + n/x_L. So Q_abs keeps its per-term form, with Im(Hₐ⁽ᴸ⁾/m_L) and
  Im(m_L H_b⁽ᴸ⁾), and Q_ext = Q_sca + Q_abs as in phase 1.
- **Lossless particles.** For a lossless particle (Im m_l = 0 in every layer), H⁽ᴸ⁾ is real in
  exact arithmetic, but complex intermediates (ξ, P) give it a rounding-level imaginary part.
  Q_abs is therefore set to exactly 0 when every layer is real. This is exact, and phase 1's
  Q_abs is exactly 0 for real m too.
- **Partly absorbing particles.** Only Im(δ) carries an absorbing core's absorption through a
  lossless shell, and its relative accuracy is limited by rounding in the complex intermediates.
  The accuracy maps measure Q_abs relative to itself, as for spheres. If they show losses for
  weakly absorbing cores (for example Im m₁ = 10⁻⁸), the fallback is to use the real second
  solution χ in place of ξ for real layers, so that a real layer contributes no imaginary
  rounding. This is decided from the maps, not in advance.

### 2.6 Truncation

- N = `default_n_max(x_L)`, the homogeneous rule at the outer size parameter (⌈x + 12x^⅓ + 6⌉
  in `DoubleDouble`). The Arb references contain the infinite series, so the maps validate the
  rule for layered spheres too. This includes high-index cores, where |m₁|x₁ > x_L is possible.
- `n_max` overrides it, and fewer terms than the default are outside the validated domain, as
  in phase 1.

### 2.7 Derivatives

- **Parameters.** The parameters are, for every layer, x_l, Re m_l and Im m_l: 3L in total.
  aₙ and bₙ are holomorphic in each m_l, so the coefficients need ∂/∂x_l and d/dm_l, and real
  outputs need all three.
- **Method: forward-mode `Dual` through the generic kernel**, as main spec §5.2 plans. Unlike
  phase 1, there are no hand-written closed forms. L is known only at run time, so:
  - **L ≤ 4:** one pass with `Dual<f64, 3L>`, monomorphized for each L;
  - **L > 4:** one pass per layer with `Dual<f64, 3>`, seeded in that layer's (x_l, Re m_l,
    Im m_l).
  - The values are bitwise those of `f64` in every pass, since `Dual` guarantees this.
  - Q_abs's derivatives come from the `Dual` absorption sum, including ∂/∂(Im m_l). Seeding
    Im m_l skips the real-recurrence shortcut, since `Dual`'s `is_zero` is strict.
- **Target.** At most (1 + 3L)× the value-only cost (main spec §10.1). If it is missed, the
  fallback is closed forms by the chain rule through the recursion (main spec §14 item 6). Those
  are O(L) per order with an adjoint sweep, built from D′ = n(n + 1)/z² − 1 − D² and
  d ln(ψ/ξ)/dz = −i/Π.
- **One layer.** For L = 1, phase 1's closed forms are used, so the derivatives are bitwise
  those of `Mie`.
- **Ensembles.** With the interfaces at fixed fractions f_l = x_l/x_L:
  - ∂/∂x_L at fixed fractions is Σ_l f_l ∂/∂x_l;
  - ∂/∂f_l = x_L ∂/∂x_l for l < L.

### 2.8 Input rules

| Error | When |
|---|---|
| `InvalidInput` | Any rule of phase 1 for x_L and each m_l; L = 0; x not nondecreasing; x₁ ≤ 0; L above `MAX_LAYERS` (2¹⁰, which bounds the workspace); x and m of different lengths |
| `OutsideValidatedDomain` | Valid inputs outside `layered::VALIDATED_DOMAIN` (section 6), or `n_max` below the default, unless `allow_unvalidated` is set |

## 3. Phase 3a: the refactor

### 3.1 Coefficient layer (`scatrs::mie`, crate-internal)

- A crate-internal view `Coefficients<'a, T>` holds:
  - x (the outer size parameter);
  - aₙ and bₙ;
  - the absorption sum;
  - a list of derivative directions. Each direction carries ∂aₙ/∂p, ∂bₙ/∂p and the absorption
    sum's derivative, and is either *real* (one real parameter) or *holomorphic* (a complex
    parameter giving its real and imaginary parts).
- Efficiencies, amplitudes (cosine and `AngularTable` forms), phase matrix and Greek
  coefficients, with their derivatives, are implemented once on this view.
- `MieSolution` keeps its public API and delegates. Its directions are (x: real) and
  (m: holomorphic), in that order, so `EfficiencyJacobian { x, m_re, m_im }` is unchanged.
- **Check:** every existing Mie and Greek test unchanged, plus a bitwise comparison of all
  outputs before and after on the Arb case grid.

### 3.2 Engine

- `Component { distribution, particle, weight }`, where `Particle` is a non-exhaustive enum:
  - `Sphere { refractive_index: Vec<Complex<f64>> }`;
  - `LayeredSphere { radius_fractions: Vec<f64>, refractive_index: Vec<Vec<Complex<f64>>> }`, the
    latter per layer and per wavelength (added in 3c).
- `Component::new(distribution, refractive_index, weight)` keeps its meaning: a sphere. This
  keeps the common case source-compatible for sasktran2. `Component::with_particle` is added.
  The public field `refractive_index` becomes `particle`, a breaking change that the release
  notes state.
- The node evaluations (`quantities1`, `quantities2`, the panel products of kernel A) take the
  particle's coefficient view and its directions rather than `Mie<f64>` and m.
  - `PanelProducts` holds 4 + 4·(number of directions) columns per sphere rather than 4 or 8.
  - The derivative groups (`M_GROUP` per refractive-index part) become groups per direction.
- **Heuristics that read m.** The engine reads m in the initial panel width (MAX_DX, from the
  period π/|m − 1|), in the number of terms per band, and in the domain checks. For layered
  particles each takes the layer that bounds it:
  - the largest |m_l − 1| for the period;
  - x_L for the terms;
  - every layer for the domain.
- `ParamKind` gains `LayerRefrReal(l)`, `LayerRefrImag(l)` and `RadiusFraction(l)` (3c).
  `RefrReal` and `RefrImag` keep meaning the sphere's index.
- **The binding.** `_scatrs.ensemble_compute`, `ensemble_compute_batch` and
  `ensemble_parameters` take items of `(distribution spec, particle spec, weight)`. The
  particle spec is `("sphere", m)` or `("layered", fractions, m[L, n_λ])`. `Ensemble` and
  `Mixture` build it through a private method on the particle class (the 2c-1 follow-up).

## 4. Rust API (`scatrs::layered`)

```rust
/// A reusable workspace, one per thread (as `Mie`).
pub struct Layered<T: Real> { /* … */ }

impl<T: Real> Layered<T> {
    pub fn new() -> Self;
    /// `x`: the size parameters of the interfaces, core first, nondecreasing; `m`: each layer's index.
    pub fn compute(&mut self, x: &[T], m: &[Complex<T>], opts: &MieOptions)
        -> Result<LayeredSolution<'_, T>>;
    pub fn compute_with_derivatives(&mut self, x: &[T], m: &[Complex<T>], opts: &MieOptions)
        -> Result<LayeredSolution<'_, T>>;
}
```

- `LayeredSolution` has the same methods as `MieSolution`:
  - `n_terms`, `a`, `b`, `efficiencies`;
  - amplitudes (cosines and `AngularTable`);
  - `phase_matrix`, `greek`.
- Derivatives are per layer: `derivatives(l)` returns phase 1's `CoefficientDerivatives` for
  layer l (∂/∂x_l, d/dm_l), and `efficiencies_jacobian()` returns one `EfficiencyJacobian` per
  layer. The phase-matrix and Greek derivatives take one buffer set per layer and parameter.
- `layered::VALIDATED_DOMAIN` and `layered::MAX_LAYERS` are public. `MieOptions` is shared.
- A module of its own rather than `mie::Layered`, matching the Python module (main spec §9).
  This amends the main spec's source layout (section 12).

## 5. Python API

### 5.1 Low-level: `scatrs.layered`

- **Functions.** `coefficients`, `efficiencies`, `amplitudes`, `phase_matrix`, `greek` and
  `default_n_max`, with `scatrs.mie`'s signatures and options.
  - `x` and `m` have a trailing layer axis of length L. Their leading axes broadcast as in
    `scatrs.mie`.
  - `x[..., l]` is the size parameter of layer l's outer surface, and `x[..., -1]` that of the
    particle.
- **Results.** The result classes of `scatrs.mie` are reused. Derivative names are:
  - `"layer{l}.x"`, `"layer{l}.m_real"` and `"layer{l}.m_imag"`;
  - for complex outputs, `"layer{l}.m"`, the complex derivative, as `"m"` is in `scatrs.mie`.

  `derivatives=True` means all of them, and a sequence selects some.
- **Errors.** A single layer is allowed, and equals `scatrs.mie` bitwise. A ragged or zero
  layer axis raises `ValueError`. Inputs outside the rules of section 2.8 raise
  `InvalidInputError`.

### 5.2 High-level: `scatrs.LayeredSphere`

```python
LayeredSphere(refractive_index=[m_core, ..., m_outer], radius_fraction=[f_1, ..., f_{L-1}])
LayeredSphere.coated(core, shell, *, core_fraction)   # L = 2
```

- **Refractive indices.** Each layer's `refractive_index` is converted as `Sphere`'s is: a number
  becomes `Constant`, a model is kept, a callable becomes `Function`. The sequence is stored as
  a tuple, so the object stays frozen and hashable.
- **Fractions.** `radius_fraction` gives the outer radius of each inner layer as a fraction of
  the particle radius. They must be nondecreasing in (0, 1], with length L − 1. A single layer
  takes `radius_fraction=()`.
  - The docs give the core volume fraction as f³ and show the conversion.
  - The fractions are scalars (section 1.4).
- **Size distributions** describe the outer radius. `effective_radius`, `mean_area` and
  `mean_volume` refer to it, as the docstring says.
- **Derivative names** (with the `c{i}.` and `mode{j}.` prefixes as in 2c-1):
  - `"layer{l}.m_real"` and `"layer{l}.m_imag"` for l = 0, …, L − 1;
  - `"layer{l}.radius_fraction"` for l = 0, …, L − 2.
- **Results.** `EnsembleResult.refractive_index` has shape `(L,) + wavelength.shape` for a
  layered sphere. For a mixture it is a tuple, one array per component, since the components'
  shapes differ.

## 6. Validated domain and accuracy maps

- **Form.** `layered::VALIDATED_DOMAIN` declares C·(κ′ + cond)·2⁻⁵² as in phase 1, with the same
  `MieOutput` measures. The starting form of κ′ is
  - κ′ = (1 + max_l max(1, |m_l|)·x_L)·max(1, max_l |m_l| / max_l |m_l − 1|).

  The maps may add factors, for example for thin strongly absorbing layers, and each one is
  documented with the evidence for it.
- **Starting target:**
  - x_L ∈ [10⁻⁶, 10⁴];
  - every m_l in the Mie domain;
  - L ≤ 16;
  - any fractions, including zero thickness.

  The published domain is whatever the maps support.
- **Sweep tool.** The `accuracy_map` example gains a layered mode, comparing f64 against
  `DoubleDouble` on dense grids. Its axes are x_L, the core fraction (10⁻³ to 1 − 10⁻⁶), and
  pairs of indices chosen to hit the hard regimes:
  - absorbing cores in clear shells;
  - metallic shells;
  - high contrast;
  - near-unity indices;
  - thin, strongly absorbing coatings at large x;
  - many thin layers approximating a gradient.

  The results go to `docs/validation/data/` as for spheres.
- **CI** enforces the domain against the Arb references. The scheduled job reruns the dense DD
  sweep.

## 7. Arb references (`reference/scatrs_reference/generators/layered.py`)

- **Independence.** The generator does not use Yang's recursion. It matches the fields directly
  at each interface, as the mpmath check of section 2.2 did:
  - ψₙ(z) and ξₙ(z) of complex z are computed in disk arithmetic, from phase 1's rigorous ratios
    and an upward ξ recurrence;
  - each interface is a 2×2 transfer;
  - the precision is doubled until every stored value meets the generator's `REL_BITS`.

  So the references check the algorithm as well as its arithmetic.
- **Labels.** Coefficients are certified, since they are finite computations. Series are
  high-precision: phase 1's tail bound is for homogeneous spheres, and extending it is a
  research item rather than an exit criterion.
- **Derivatives** are Arb central differences with Richardson extrapolation at ≥ 100 digits,
  labelled high-precision (main spec §8.4).
- **Cases:**
  - **x_L:** {10⁻⁶, 10⁻³, 0.1, 1, 5, 20, 100, 10³, 10⁴}.
  - **Index pairs:**
    - black carbon in sulfate: (1.95 + 0.79i, 1.43 + 10⁻⁸i);
    - water on ice;
    - a silica core in a gold-like shell: (1.45, 0.2 + 3i);
    - an absorbing core in a clear shell, at Im m₁ ∈ {10⁻⁸, 10⁻³, 1};
    - high contrast: (10 + 10i, 1.33);
    - near-unity: (1.01, 1.0001);
    - an air shell: (1.5 + 0.01i, 1).
  - **Core fractions:** {10⁻³, 0.1, 0.5, 0.9, 0.999}.
  - **Three-layer and ten-layer cases**, including a graded index.
  - **About 40 seeded random cases** with L ∈ [2, 8].
- **Budget.** The reference data is 13 MB of the 20 MB total, so the layered data aims for
  ≤ 5 MB. The coefficients are stored at log-spaced orders as in phase 1.

## 8. Tests beyond the references

- **One layer** equals `Mie` bitwise: coefficients, every output and every derivative.
- **Identical indices.** Layers with identical m equal the homogeneous sphere of radius x_L.
  This is exact by construction (G₁ = 0), and the test asserts bitwise equality.
- **Zero thickness.** A layer of zero thickness changes nothing, to rounding: H⁽ˡ⁾ =
  (m_l/m_{l−1})H⁽ˡ⁻¹⁾ for a, and likewise for b.
- **Air shell.** An outer layer with m = 1 gives the coefficients of the bare core (x_{L−1},
  m_{L−1}), to rounding. The cross sections are equal and the efficiencies differ by the area
  ratio. This was checked in mpmath to 10⁻⁶¹.
- **Rayleigh limit.** At small x, a₁ → −(2i x_L³/3)·α̃, where α̃ is the coated sphere's
  electrostatic polarizability ratio (Bohren & Huffman 1983, chapter 5):
  - α̃ = [(ε₂ − 1)(ε₁ + 2ε₂) + f(2ε₂ + 1)(ε₁ − ε₂)] / [(ε₂ + 2)(ε₁ + 2ε₂) + 2f(ε₂ − 1)(ε₁ − ε₂)];
  - ε = m² and f = (x₁/x₂)³.

  The mpmath check gave relative agreement of 3·10⁻⁹ at x = 2·10⁻⁴, which is the expected
  O(x²) correction.
- **Lossless particles.** Q_abs = 0 exactly, and Q_ext = Q_sca.
- **Optical theorem.** At moderate x, Q_ext from the per-term absorption agrees with
  (2/x²)Σ(2n + 1)Re(aₙ + bₙ).
- **Derivatives.** `Dual` agrees with central differences in `DoubleDouble`. The two `Dual`
  strategies of section 2.7 agree bitwise at L ≤ 4.
- **Thread independence.** Ensembles of layered spheres are bitwise the same on 1, 2 and 8
  threads.
- **Python.** Broadcasting, derivative names, error cases, the `coated` constructor,
  immutability and hashing of `LayeredSphere`, and 1-layer ensemble equivalence.

## 9. External validation

- **scattnlay.** A script `benchmarks/compare_scattnlay.py` compares the coefficients,
  efficiencies and S₁, S₂ on the Arb case grid and a random sample. The results go on the
  external-validation page.
  - Every disagreement beyond scattnlay's own accuracy is attributed: to scattnlay, to scatrs,
    or to conventions (S₁ and S₂ conjugates, or sign conventions).
  - scattnlay is a dependency of the benchmark environment only.
- **Bohren & Huffman's BHCOAT** (1983, appendix B), ported to Python in the benchmark
  environment, on the case grid. Its known loss of accuracy for large absorbing cores is shown
  rather than hidden.
- **Analytic limits.** The Rayleigh limit and the identities of section 8 are summarized for
  users on the validation page.

## 10. Performance

- **Criterion benchmarks:**
  - the layered kernel at L ∈ {2, 3, 10} across x_L;
  - the derivative overhead;
  - an ensemble of coated spheres.
- **Targets:**
  - single-threaded speed at least that of scattnlay on coated spheres (main spec §10.1);
  - derivatives within (1 + 3L)× the value-only cost;
  - the cost of one coated sphere within about 2.5× that of a homogeneous sphere of the same
    x_L. Each extra layer adds two D⁽¹⁾ recurrences and one P recurrence.

  The measured ratios are published.
- **Ensembles of layered spheres** use kernel A unchanged, given the coefficients. No new fast
  path.

## 11. Docs

The docs stay user-facing: no phases or development history.

- **Theory chapter "Layered spheres":**
  - the recursion and the pole-free form, with proofs;
  - the cancellation-free split;
  - absorption;
  - the derivative method;
  - truncation;
  - Πₙ and Pₙ, and why they stay bounded.
- **Validation:** the layered accuracy map on the internal page, and the scattnlay comparison,
  BHCOAT and the limits on the external page.
- **Example notebook:** black carbon coated in sulfate. It covers the absorption enhancement
  with coating thickness, a size distribution of coated particles, and the Jacobian with
  respect to the coating fraction.
- **Updated pages:**
  - `docs/api.md` and the rustdoc overview;
  - `docs/limitations.md`: concentric layers only, and fractions fixed across a distribution;
  - the quick start: one line showing `LayeredSphere.coated`;
  - `docs/references.md`: Yang 2003, Peña & Pal 2009, Toon & Ackerman 1981.
- **Release notes:** drafted in `docs/plans/next-release-notes.md`, including the breaking
  `Component` field change.

## 12. Amendments to the main spec

- **§5.2.** Adds the pole-free form, the cancellation-free split, the absorption treatment and
  the derivative strategy for runtime L (this spec, section 2).
- **Source layout (§4).** Layered spheres get `src/layered/` and the public module
  `scatrs::layered`, rather than living in `src/mie/`. The shared coefficient layer stays in
  `mie`.
- **§13 row 3.** The exit criterion adds layered spheres in ensembles with Jacobians, and the
  bitwise one-layer and identical-layer identities, to "matches Arb references and scattnlay".

## 13. Decisions

- Any number of layers rather than a core–shell solver (user, 2026-10-14).
- Fixed radius fractions in ensembles (user).
- The full phase, with the engine refactor first as its own reviewed step (user).
- **`Dual` derivatives first, closed forms only if the overhead target is missed.** The user
  agreed to this in discussion. Because L is known only at run time, this spec dispatches
  `Dual<f64, 3L>` for L ≤ 4 and per-layer `Dual<f64, 3>` passes above.
- **Yang's recursion in a pole-free form, with phase 1's cancellation-free numerators.** This
  was verified in mpmath during design. The accuracy maps decide the switch points and whether
  real layers need the χ form for Q_abs.
- **References by direct interface matching, independent of the algorithm.**

## 14. Changes while planning 3a

- **Groups rather than general directions.** The coefficient view's derivatives come in
  *groups*. Each group holds the derivatives with respect to one real size parameter and one
  refractive index, in which the coefficients are holomorphic. This replaces section 3.1's list
  of real or holomorphic directions.
  - A sphere has one group, and a layered sphere one per layer (x_l, m_l), so every existing
    output's derivative code serves both unchanged.
  - The engine's *directions* are Re m and Im m of each group, in order. Phase 3c appends the
    real directions of the radius fractions.
- **One flag deferred to 3b.** Only the group of the outer size parameter carries the
  −(2/x)·Q term of the efficiencies and the scale of the phase matrix. That flag is added in
  3b, with its first use.
- **Correction to section 3.2.** The engine reads m only in the solves and in the number of
  terms per band. `MAX_DX` is a constant and does not read m, despite its comment's reference
  to the period π/|m − 1|.
- **Batch binding.** Until 3c, the batch binding accepts a sphere's particle spec only.
- **Release notes.** The release-note entry for the `Component` field change is written in
  3c, with the rest of the release notes.

## 15. Looking ahead to phase 4 (spheroids, T-matrix)

Spheroids and the T-matrix come next (user, 2026-10-14). Phase 3's public API is chosen so that
phase 4 extends it rather than changes it. The internal engine interface will need one new
path.

**Already compatible:**

- `Particle` and `ParamKind` are non-exhaustive, so `Particle::Spheroid { … }` and its
  parameters can be added. In Python, `Ensemble(particle, size)` and `Mixture` take any particle
  class with an `_engine` hook, including mixtures of spheres and spheroids (main spec §7.2).
- Phase 3c's real directions (radius fractions) are the mechanism for shape parameters such as
  the aspect ratio. Their derivatives come from `Dual` through the EBCM as main spec §5.3
  plans.
- `scatrs.layered` stands next to a future `scatrs.tmatrix`, as main spec §9 plans.
- `Outputs.depolarization` and the separate α₂ and α₃ already exist. For spheres they are 0
  and α₂ = α₃.

**Decided now:**

- **Variant names.** The Rust variant for layered spheres is `Particle::LayeredSphere`
  (section 3.2; first drafted as `Layered`), matching Python and leaving room for `Spheroid`,
  `Cylinder` and the other shapes.
- **Size convention.** A size distribution describes the radius that the particle type
  defines: the radius for `Sphere`, the outer radius for `LayeredSphere`, and for T-matrix
  shapes the equal-volume-sphere radius unless phase 4 decides otherwise. The `Ensemble`
  docstring says this generally rather than for spheres alone.

**For phase 4, not phase 3:**

- **Phase matrix elements.** Randomly oriented non-spherical particles have six independent
  phase-matrix elements. Phase 4 adds F₂₂ and F₄₄ to `PhaseValues` (non-exhaustive, so this
  is additive), to `PhaseBuffers`'s ensemble counterpart, and to `EnsemblePhase`, equal to F₁₁
  and F₃₃ for spheres.
- **A direct path in the engine.** Spheres and layered spheres give aₙ and bₙ, and the engine
  builds their phase matrix and Greek coefficients from them (kernel A and the projection
  bands). A randomly oriented T-matrix particle gives its cross sections and Greek coefficients
  directly, by Mishchenko's analytic averaging. The engine's `Solver` will therefore return
  either coefficients or per-node outputs:
  - Greek coefficients integrate linearly over sizes, beside the band tables.
  - The phase matrix at the requested cosines comes from the Greek expansion.

  The crate-internal `Optics`/`Solver`/`Solved` interface of 3a is where this goes. Phase 3
  keeps the coefficient path as is and adds no assumption to the public API that only it can
  meet.
- **Distributions over shape parameters**, such as aspect ratio (main spec §7.1). These are a
  property of the ensemble, not of the particle: phase 3 keeps every `Particle` field a single
  value per wavelength.

## 16. Changes while implementing 3b

Each change was forced or decided by a measurement. The plan, `docs/plans/2026-10-14-phase-3b-layered-kernel.md`, gives the numbers.

**Algorithm (section 2):**

- **The form is chosen per order.** The δ form is used unless D_n(z₂) and δ cancel by more than a factor 4, and the pole-free form otherwise. This replaces the fixed switch at ⌊|m_L|x_L⌋ of section 2.4. Every layer uses it, not only the outer one.
- **D^ζ comes from the upward ratio recurrence of ζ** (w_{n+1} = (2n + 1)/z − 1/w_n). Computing it as D_n + W/Π_n subtracts two huge terms near the poles of D_n.
- **The second solution is chosen per layer.** χ is used when Im m_l·x_l < ½, and ξ otherwise; the formulas hold for any ζ, with its Wronskian W. With ξ only, the error of Q_abs grew like 2⁻⁵²/Im m (10⁻⁵ at k = 10⁻¹²). The χ fallback of section 2.5 is thus used for every optically thin layer, real or not.
- **Two more cancellations are removed.**
  - The (n + 1)/x terms of G₁ are removed exactly. b_n lost four digits at x₀ = 10⁻⁹.
  - In the outer step's upward region, δ·ψ_n is added to the outer layer's own numerator, which is exactly 0 for m_L = 1. An air shell lost a₁ to seven digits.
- **The derivatives seed x_l and Re m_l only: 2L, not 3L** (section 2.7). Seeding Im m switched off the real recurrence and the exact Q_abs = 0 of lossless particles, so the values were not those of `compute`. ∂/∂(Im m_l) of the absorption sum now follows from the coefficients. Passes carry 2L derivatives for L ≤ 4, and 2 per layer above.

**Domain and accuracy (section 6):**

- **The values are validated up to |m_l|·x_L ≤ 3·10⁴.** Thin lossless shells of index 10 at x_L = 10⁴ lose 10⁹ ulps in either basis.
- **The derivatives are validated up to x_L = 100.** Above it, the amplitude derivatives reach C ≈ 6000. `compute_with_derivatives` refuses larger particles unless `allow_unvalidated` is set. One layer keeps the sphere's domain.
- **κ′ changes.**
  - Its near-unity factor is the largest |m_l|/|m_l − 1| over the layers whose index is not exactly 1. It was the largest |m| over the largest |m − 1|, which understated a near-unity shell around another core.
  - The coefficient and amplitude derivatives take κ, near-unity factor included.
- **The layered constants are measured** (`layered::ValidatedDomain::constant`). They are 32, 16, 128, 64, 128, 1024, 256 and 64/256, against the sphere's 16 to 512.

**APIs (sections 4 and 5):**

- **Rust.** The per-layer derivative methods take a `layer` argument; there is no buffer per layer. `layered::n_terms` is added, and so are `ValidatedDomain::{max_phase, derivative_x_max, contains_derivatives}`.
- **Python.** The results are subclasses of `scatrs.mie`'s classes. `scatrs.layered.parameters(L)` lists the names.

**Validation and comparison (sections 7 to 10):**

- **References.** Core fractions are {10⁻³, 0.3, 0.9, 0.999}, four not five, for the 5 MB budget: 3.9 MB went to values and 0.2 MB to derivatives. The derivative case at x_L = 200 moved to 100.
- **BHCOAT.** It is represented by the textbook method it implements, direct interface matching in double precision, written in the comparison script, not by a port.
- **Speed targets.**
  - Two layers cost 2.5–2.65× a sphere, against a target of about 2.5×.
  - scatrs is 4.3–8.4× faster than scattnlay.
  - Derivatives meet (1 + 3L)× up to x_L ≈ 10. At x_L = 100 they cost 7.4× for two layers (target 7) and 32× for ten (target 31). The fallback is section 2.7's closed forms (follow-up).
