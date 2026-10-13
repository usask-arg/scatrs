# Phase 1b design: sphere phase matrix and exact Greek coefficients

Status: approved 2026-10-13. Builds on phase 1a (`docs/specs/2026-10-13-phase-1-mie-design.md`)
and the main spec (`docs/specs/2026-10-12-scatrs-design.md`, sections 5.4, 6.2 and 7.4).

## 1. Scope

In:

- an O(n) Gauss–Legendre rule (Bogaert 2014) for large n, and reusable, caller-owned rules;
- the sphere phase matrix F₁₁, F₁₂, F₃₃, F₃₄ at any cosine, normalized and unnormalized, with
  Jacobians with respect to x, Re m and Im m;
- a Greek projector: Gauss–Legendre nodes and weights for an order L, turning tables of
  phase-matrix elements at those nodes into α₁…α₄, β₁, β₂ for l = 0…L, exactly (up to
  rounding) for polynomials of degree ≤ 2n_q − 1;
- single-sphere Greek coefficients `greek(x, m, L)` with Jacobians, L up to 2N (all orders);
- Python `scatrs.mie.phase_matrix` and `scatrs.mie.greek`;
- Arb references, an accuracy map, theory and example pages.

Out (phase 2 or later): size distributions and mixtures, the sasktran2 comparison, rayon and
thread pools, the instruction-count gate, fuzzing, the 1a follow-ups (Jacobian cost, |m| floor).

## 2. Gauss–Legendre

- **Method.** Below `BOGAERT_THRESHOLD` (about 100, chosen by measurement) the existing
  double-double Newton rule, rounded to `f64`. Above it, Bogaert's asymptotic expansions of the
  nodes θₖ and weights (interior and boundary regions), evaluated in `f64`; O(n) time.
- **Accuracy.** Node absolute error and weight relative error a few units of 2⁻⁵², measured
  against the double-double rule (where it is affordable) and against Arb, and declared with the
  same measured-constant convention as phase 0. Symmetry is exact (nodes computed for one half and
  reflected).
- **API.** `GaussLegendre<T>` gains `#[non_exhaustive]` and accessors; rules are plain values a
  caller computes once and shares (`Arc` if needed); no global cache. A variant writes into
  caller-supplied buffers. `DoubleDouble` rules keep the Newton method at every n.
- **Limit.** `MAX_GAUSS_LEGENDRE_POINTS` stays 2¹⁷; the full expansion at x = 2·10⁴ needs about
  4·10⁴ nodes.

## 3. Phase matrix

- From the amplitudes (Bohren & Huffman 1983, §4.4, time dependence e^(−iωt)):
  F₁₁ = (|S₁|² + |S₂|²)/2, F₁₂ = (|S₂|² − |S₁|²)/2, F₃₃ = Re(S₂S₁*), F₃₄ = Im(S₂S₁*);
  F₂₂ = F₁₁ and F₄₄ = F₃₃ for spheres.
- **Normalized:** divided by x²Q_sca/4, so that ½∫F₁₁ dμ = 1. **Unnormalized:** the normalized
  form times C_sca/(πr²) = Q_sca (per unit geometric cross section), the quantity phase 2 sums
  over sizes. Both have Jacobians; the normalized one includes the derivative of Q_sca.
- Jacobians come from the existing amplitude derivatives (∂/∂x, d/dm; ∂/∂(Im m) = i·d/dm applied
  to S, then the real formulas).

## 4. Greek projector

- **Expansion** (Hovenier, van der Mee & Domke 2004, main spec): F₁₁ = Σ α₁ˡ d⁰⁰ₗ;
  F₂₂ + F₃₃ = Σ (α₂ˡ + α₃ˡ) d²²ₗ; F₂₂ − F₃₃ = Σ (α₂ˡ − α₃ˡ) d²⁻²ₗ; F₄₄ = Σ α₄ˡ d⁰⁰ₗ;
  F₁₂ = Σ β₁ˡ d⁰²ₗ; F₃₄ = Σ β₂ˡ d⁰²ₗ, with d^l_{mn}(cos Θ) as in `wigner_d`. Coefficients follow from
  orthogonality: c_l = (2l + 1)/2 · ∫ F d_l dμ.
- **Quadrature.** n_q = N + ⌈(L + 1)/2⌉ nodes make every integrand (degree ≤ 2N + L) exact. The
  projector is built for (L, n_q) and holds the rule; `project` takes F tables at its nodes.
- **Streaming.** For each node the d-function recurrences in l run once and accumulate into the
  coefficient arrays: O(n_q·L) time, O(L) memory. Storing d at all nodes would need about 25 GB at
  x = 10⁴.
- **Linearity.** Derivative tables project with the same call, giving the Greek Jacobians.
- **Sign conventions.** A table in the theory page relates Bohren & Huffman's elements, Hovenier's
  and sasktran2's β signs; a test pins it.

## 5. Single-sphere Greek

- `greek(x, m, L, options)` with L defaulting to 2N (all orders; higher orders are exactly 0).
  Requesting L > 2N is `InvalidInput`, not silent padding.
- Amplitudes at the projector's nodes, then one projection. Normalized (α₁⁰ = 1, α₁¹ = 3g) and
  unnormalized forms, each with Jacobians.
- Cost at x = 10⁴, full expansion: seconds (n_q·L ≈ 8·10⁸ recurrence steps per element);
  documented in limitations.

## 6. Python

- `scatrs.mie.phase_matrix(x, m, mu, normalized=True, derivatives=…, …)` and
  `scatrs.mie.greek(x, m, order=None, normalized=True, derivatives=…, …)`.
- Same conventions as 1a: broadcasting x and m, fresh views, bounded outputs (`MAX_MIE_VALUES`),
  GIL released, frozen read-only results with `.d(param)`.
- `scatrs.special.gauss_legendre` uses the new rule above the threshold; its docstring states the
  method and cost.

## 7. Accuracy and validation

- **Model.** C·(κ′ + cond)·2⁻⁵² as in 1a, constants measured with a 4× margin. Phase-matrix
  elements are measured like the amplitudes (relative to max(|F|, F_rms)); Greek coefficients
  relative to the largest coefficient of the same element (α₁⁰ = 1 when normalized).
- **Arb references.** A new generator computes phase-matrix elements and Greek coefficients for a
  few dozen spheres and several L (including all orders) with Arb's own Legendre roots as
  quadrature nodes; Gauss–Legendre references for n up to 10⁵ at sampled indices.
- **Accuracy map.** Greek coefficients and phase-matrix elements over a grid like 1a's (smaller in
  x if cost requires).
- **Identities.** α₁⁰ = 1, α₁¹ = 3g, F₁₁(1) = Σα₁ˡ, ½∫F₁₁ = 1, projection of known polynomials
  exact; Jacobians against `Dual` and finite differences.
- **Docs-consistency.** Contracts for the new figures (GL accuracy and threshold, Greek
  constants).

## 8. Docs

Theory sections (Bogaert's method, exactness of the projection, the recurrences, conventions and
signs), an example page (phase function, polarization, Greek coefficients), the accuracy page
extended, and a limitations entry for the cost of large spheres.
