# Phase 2b: speed of ensembles, and the sasktran2 comparison

Status: approved 2026-10-14. Builds on phase 2a (`docs/specs/2026-10-14-phase-2a-ensembles-design.md`).

## 1. Goal

Make ensembles with Greek coefficients fast, then compare scatrs with sasktran2's `MieDatabase`
for accuracy and speed, and publish the results (main spec section 13, phase-2 exit criterion).

Decisions, made by the user:

- **Workload:** sasktran2-style database grids first: many distributions (a grid of r_g, σ) ×
  many wavelengths, Greek coefficients to about order 64.
- **Success:** on a database grid, scatrs is faster than sasktran2 at matched accuracy, single
  threaded and multi-threaded, with exact Greek coefficients. An accuracy-against-time curve over
  rtol is published too.
- **Comparison:** a script with published results (`benchmarks/compare.py` in its own uv project,
  sasktran2 pinned); CI does not install sasktran2.
- **Scope:** kernel A (fast exact projection), kernel B (products and Clebsch–Gordan
  coefficients) with automatic selection, and `compute_batch` (shared size nodes).

## 2. Baseline (measured 2026-10-14, Apple M4 Pro, 12 cores)

Lognormal r_g = 1 µm, σ_g = 2, m = 1.33 + 10⁻⁴i, one wavelength:

| | λ = 1 | λ = 0.5 | λ = 0.3 | accuracy (C_ext; α₁¹) |
|---|---|---|---|---|
| sasktran2 2026.10.1, 64 coefficients, 1 thread | 0.35 s | 0.43 s | 0.51 s | 7·10⁻⁵; 3·10⁻⁴ |
| sasktran2, 32 coefficients, 1 thread | 0.32 s | 0.37 s | 0.42 s | 7·10⁻⁵; 7·10⁻³ |
| scatrs rtol 10⁻⁴, cross sections, 1 thread | 0.03 s | 0.11 s | 0.25 s | ~10⁻⁵ |
| scatrs rtol 10⁻⁴, Greek to 32, 1 thread | 0.43 s | 2.7 s | 10.3 s | ~10⁻⁵ |
| scatrs rtol 10⁻⁴, Greek to 32, 12 threads | 0.11 s | 0.44 s | 1.5 s | ~10⁻⁵ |

The Greek tables are about 98% of the time: each sphere's S₁, S₂ at n_q ≈ N + L/2 cosines,
O(N²), about 1.3 ms per sphere at x ≈ 600. sasktran2 uses 2·`num_coeffs` cosines split at a
forward-peak point (not exact: its α₁¹ moves by 0.7% between 32 and 64 coefficients at 300 nm),
and one fixed radius grid (QUADPACK subintervals of a Q_ext integrand at the shortest
wavelength, 31 Gauss points each) for all wavelengths and all distributions of a database.

## 3. Kernel A: fast exact projection

Same projector bands, same exactness, same error bounds; a faster evaluation of S₁, S₂ at a
band's Gauss nodes.

- **Band tables.** Per band, wₙπₙ(μ) and wₙτₙ(μ) (wₙ = (2n + 1)/(n(n + 1))) at the band's
  non-negative nodes, split by the parity of n: G_E = [π̃ for odd n; τ̃ for even n] and
  G_O = [π̃ for even n; τ̃ for odd n]. Built once per call and shared by every node,
  wavelength and distribution (a cache keyed by band).
- **Parity.** πₙ(−μ) = (−1)ⁿ⁺¹πₙ(μ), τₙ(−μ) = (−1)ⁿτₙ(μ), so with
  E₁ = G_Eᵀ[a_odd; b_even], O₁ = G_Oᵀ[a_even; b_odd], E₂ = G_Eᵀ[b_odd; a_even],
  O₂ = G_Oᵀ[b_even; a_odd]: S(±μ) = E ± O. Half the cosines, no recurrence per sphere.
- **Matrix products.** The coefficients of a panel's 15 nodes (zero-padded to the band's N)
  form the right-hand sides, so S₁, S₂ at every node come from two real matrix products
  (h × N)·(N × 4·15), h the number of non-negative nodes; m-derivatives add the columns of
  daₙ/dm, dbₙ/dm. The GEMM is measured first: the `matrixmultiply` crate (after a dependency
  audit) against a hand-written kernel; `faer` only if both fall short.
- The monodisperse path (one node, with ∂/∂r) keeps the existing per-sphere evaluation.

## 4. Kernel B: products and Clebsch–Gordan coefficients

As in de Rooij & van der Stap (1984) and Mishchenko's `spher.f`: the Greek coefficients of a
size-averaged phase matrix are linear in the size-averaged products of Mie coefficients.

- With c±ₙ = aₙ ± bₙ, accumulate over the size nodes, for |n − n′| ≤ L, the products
  c⁺ₙc⁺*ₙ′ and c⁻ₙc⁻*ₙ′ (Hermitian: n′ ≥ n) and c⁺ₙc⁻*ₙ′, with the weight of each node and the
  per-sphere scale of the unnormalized phase matrix (πr²·4/x²·(2n + 1)(2n′ + 1)/4): about
  4·N·L complex products per sphere, O(N·L).
- Once per wavelength and part, contract with Clebsch–Gordan coefficients, A = C(n 1 n′ −1 | l 0)
  and B = C(n 1 n′ 1 | l 2), by recurrence: α₁ and α₄ from A² and the parity (−1)ⁿ⁺ⁿ′⁺ˡ,
  α₂ ± α₃ from B², β₁ and β₂ from A·B and the mixed products. O(N·L²), not per sphere. The
  exact signs and factors are fixed against the projector, the test oracle.
- θ blocks accumulate the same products with ∂u/∂θ weights; m-derivatives accumulate c′ₙc*ₙ′
  (∂/∂Re m = c′c* + c c′*, ∂/∂Im m = i(c′c* − c c′*)). Error estimates: the per-panel
  |K15 − G7| of the products, summed, contracted with |weights| (bounds, as `project_bounds`).
- `GreekOrder::All` always uses kernel A.

## 5. Choosing the kernel

`EnsembleOptions::greek_method`: `GreekMethod::{Auto, Projection, Products}` (Python:
`greek_method="auto" | "projection" | "products"`). `Auto` chooses per wavelength and part
group from a cost model whose constants come from the benchmark:

- Projection: c_A·N·n_q per node + D·4·n_q per accumulation;
- Products: c_B·4·N·L per node + D·(1 + n_θ + n_m)·4·N·L per accumulation,

with D the number of distributions sharing the nodes. The choice depends on the inputs only,
so results stay bitwise identical for any thread count. Both kernels are exact up to rounding;
results of the two agree to rounding, not bitwise.

## 6. `compute_batch`: shared size nodes

`compute_batch(distributions, wavelengths, m, options)` (Rust and Python) averages over many
distributions of one material.

- One panel set per wavelength over the union of the distributions' Mie ranges, refined until
  every distribution meets its own targets (each distribution's pass-1 sums and targets; a
  panel splits when it exceeds its share for any of them). Extensions as in 2a, per
  distribution at the union's ends. Normalization and moments per distribution (Mie-free).
- Each node's Mie solution and tables are computed once and accumulated into each
  distribution's sums. Monodisperse distributions are computed on their own.
- Results per distribution and wavelength, with each distribution's error estimates and
  `converged` flag. A distribution's result in a batch is at least as refined as alone, but
  not bitwise equal to `compute`'s.
- Values only: `derivatives` must be empty (`InvalidInput` otherwise); Jacobians stay on
  `compute`. Mixtures stay on `compute_mixture`.
- Python: `ensemble.compute_batch(distributions, wavelength, refractive_index, *, greek, mu,
  normalized, rtol, max_panels, allow_unvalidated, refine_backscatter, greek_method)` returns an
  `EnsembleResult` whose arrays have a leading distribution axis.
- **Bounded memory, determinism.** Per-chunk sums grow with D, so chunks are reduced in order in
  fixed-size groups of panels rather than held until the end; the partition never depends on
  the thread count.

## 7. The comparison (`benchmarks/`)

- `benchmarks/pyproject.toml`: its own uv project; sasktran2 pinned (2026.10.1), scatrs from
  the repository.
- `benchmarks/compare.py` runs:
  - **W1:** the lognormal of section 2 at λ = 0.3, 0.5, 1 µm, L = 64;
  - **W2 (the database grid):** lognormals, r_g at 10 values log-spaced in [0.05, 1] µm ×
    σ_g ∈ {1.2, 1.4, 1.6, 1.8, 2.0}, 20 wavelengths in [0.28, 1.02] µm, m = 1.43 + 10⁻⁸i,
    L = 64.
- **Conventions:** sasktran2 m = n − ik (scatrs n + ik), nm and m² (scatrs µm and µm²); its
  Greek coefficients' normalization and signs are compared with scatrs's and documented.
- **Accuracy:** both against a scatrs reference at rtol 10⁻⁹. Metrics: the largest relative
  errors of C_ext, C_sca and the co-albedo, and the largest absolute error of the normalized
  Greek coefficients up to L.
- **Matched accuracy:** the loosest rtol in {10⁻³, 3·10⁻⁴, 10⁻⁴, 3·10⁻⁵, 10⁻⁵} at which every
  scatrs metric is at most sasktran2's.
- **Timing:** best of 3; one thread (`RAYON_NUM_THREADS=1`; sasktran2 `num_threads=1`) and all
  threads (sasktran2 `num_threads=0`); sasktran2 timed through `integrate_mie_cpp` with
  `MieDatabase`'s settings (31 points, quantile 0.99999), without its netCDF caching.
- Results go to `benchmarks/results/*.json` and are published in `docs/validation/sasktran2.md`
  with the machine, the versions and the date; docs-consistency contracts keep the numbers in
  sync.

## 8. Testing

- Kernel A and kernel B against the existing projector: every Greek coefficient, value,
  derivative and error bound, on the calibration grid; agreement to 10⁻¹² of each
  coefficient's scale. The existing ensemble tests pass with each `greek_method`.
- Kernel B against the Arb single-sphere Greek references, through nearly monodisperse
  distributions (a monodisperse part keeps the per-sphere path).
- `compute_batch`: each distribution against `compute` within the combined error estimates;
  per-distribution `converged`; determinism over thread counts; bad inputs.
- The calibration test (error estimates against rtol 10⁻¹¹ references) runs for each method.
- A criterion benchmark of pass 2 with Greek coefficients (`benches/ensemble.rs`).

## 9. Order of work

1. Measure: the GEMM options, and a timing harness for the reference case.
2. Kernel A.
3. `compute_batch` (Rust, then Python).
4. Kernel B and `Auto`.
5. The comparison, the published results, the docs (theory, accuracy, limitations, CHANGELOG,
   follow-ups).

## 10. Changes during implementation

Recorded 2026-10-14, after measuring:

- **GEMM.** faer 0.24 (MIT, MSRV 1.84, 46 crates without default features, safe API) rather
  than `matrixmultiply`, whose `dgemm` is an unsafe raw-pointer API that the core's
  `#![forbid(unsafe_code)]` rules out. On the kernel's shape (316 × 600 × 60) faer reached
  55 GFLOP/s, `matrixmultiply` 52, a safe hand-blocked loop 20 (Apple M4 Pro).
- **Kernel A** as in section 3, restricted to each panel's own number of terms. The band tables
  are built per wavelength (a fraction of a millisecond), not cached across wavelengths.
- **Kernel B is not adopted, and there is no `greek_method` option.** A numpy prototype
  confirmed an exact variant without Clebsch–Gordan coefficients: the averaged products
  ⟨c±ₙc±*ₙ′⟩ (|n − n′| ≤ L) evaluated as banded quadratic forms in wₙ(τₙ ± πₙ) at a band's
  Gauss nodes, then projected (pairs with |n − n′| > L contribute nothing to orders ≤ L), which
  matched `mie.greek` to 10⁻¹⁴. But measured per panel of 15 spheres, even with the products
  as complex GEMMs over tiles of the band, kernel B cost 0.37–1.7 times kernel A (0.37 only at
  N = 600, L = 32; 1.0 at N = 600, L = 64; 1.7 at N = 150), before doubling for the Kronrod and
  Gauss sums its error estimates need; and per distribution in a batch it needs N·L
  accumulators where kernel A needs 4·n_q. It is recorded in `docs/plans/phase-2b-followups.md`.
- **Pass 2 sums and the projection as matrix products.** Profiling the database grid showed the
  per-node accumulation into each distribution's sums, then the per-distribution projections
  (each recomputing the generalized spherical functions), dominating. A panel's sums are now
  (weights) × (nodes' values), and each band projects every distribution's tables at once with
  matrices (2l + 1)/2·wₖ·dₗ(μₖ): 13 s → 3.2 s for the grid on one thread.
- **A greedy split rule was tried and rejected.** Splitting, per unmet target, the panels that
  hold the excess error took fewer panels (306 against 477 on the grid) but lost more accuracy
  than time: at rtol 10⁻³, C_ext's error reached 1.3·10⁻³, and the calibration's worst ratios
  rose to 0.021 and 1.9. The share rule stays.
- **Matched accuracy** is decided on C_ext, C_sca and the Greek coefficients (section 7 named
  the co-albedo too). ω and the co-albedo are reported but not matched: both codes integrate
  C_sca and C_ext on the same nodes, so their errors largely cancel in ω (sasktran2's ω errors on
  W2 are below 4·10⁻⁷, from that cancellation), and for W2's weak absorbers (co-albedo about
  10⁻⁶) neither code resolves the co-albedo. Computing ω as 1 − ⟨C_abs⟩/⟨C_ext⟩ (equal up to
  rounding) would give a tight estimate, but ⟨C_abs⟩'s own estimate fell up to 1.6 times short of
  its true error on W2, so ω keeps the conservative quotient estimate (follow-up).
- **The rtol of the comparison** are 3·10⁻³, 10⁻³, 3·10⁻⁴ and 10⁻⁴ (section 7 named 10⁻³ to
  10⁻⁵): the matched rtol fell at 3·10⁻⁴ (W1) and 10⁻³ (W2), and 3·10⁻³ brackets them.
- **Review fixes** (*review*). In a group, the ends of finite supports are panel edges (a power
  law beside a broader lognormal exceeded its estimate, ratio 1.05); each part's largest step
  applies within its own range, and an extension takes the width and steps of the parts that
  want it (a narrow gamma beside a broad lognormal took 1248 panels against 61, and a nearly
  monodisperse member left a broad one unconverged); the window of chunk sums is sized from the
  inputs, at most 32 chunks and about 64 MB (peak memory for 800 distributions: 3.2 → 2.0 GB).
