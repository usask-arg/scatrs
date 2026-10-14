# Phase 2b follow-ups

These items came out of phase 2b (fast ensembles and the sasktran2 comparison) and were
deliberately deferred. Each belongs to the phase or task named in its heading.

## Performance (phase 5)
- **Kernel B, the products of Mie coefficients.**
  - **What:** accumulate the size-averaged products ⟨c±ₙc±*ₙ′⟩ (c± = aₙ ± bₙ, |n − n′| ≤ L),
    O(N·L) per sphere, and evaluate them once per wavelength as banded quadratic forms in
    wₙ(τₙ ± πₙ) at a band's Gauss nodes before the exact projection. A numpy prototype matched
    `mie.greek` to 10⁻¹⁴.
  - **Why deferred:** measured per panel of 15 spheres (Apple M4 Pro), with the products as
    complex matrix products over tiles of the band, it cost 0.37, 1.0 and 1.7 times kernel A at
    (N, L) = (600, 32), (600, 64) and (150, 32), before doubling for the sums its error
    estimates need; in a batch it needs N·L accumulators per distribution.
  - **Then:** only for single distributions of spheres of thousands of terms at L ≲ 32,
    chosen per panel by a cost model; validate against kernel A.
- **Pass 2 recomputes pass 1's Mie solutions.** Pass 2 solves every final node again: pass 1
  evaluated those nodes, and the panels split before them, so roughly a third of a call's Mie
  solutions are repeats (an estimate, not measured). Keeping pass 1's coefficients needs about
  N·15·32 bytes per panel, hundreds of megabytes for broad distributions of large spheres at
  many wavelengths in parallel.
- **Panels shared within mixtures.** The modes of a multimodal distribution share m and could
  share panels as `compute_batch` does, but sharing changes `compute`'s results and needs its own
  calibration.

## Accuracy of the error estimates (phase 5)
- **⟨C_abs⟩ of weak absorbers.** ⟨C_abs⟩ is refined to rtol of ⟨C_ext⟩; for the weak absorbers
  of the database grid of `docs/validation/sasktran2.md` (m = 1.43 + 10⁻⁸i, co-albedo about
  10⁻⁶) it is barely resolved, and its |K15 − G7| estimate fell up to 1.6 times short of its
  true error. ω keeps the conservative estimate of ⟨C_sca⟩/⟨C_ext⟩; a tight one from
  1 − ⟨C_abs⟩/⟨C_ext⟩ needs a reliable ⟨C_abs⟩ estimate first.

## API (phase 2c)
- **Jacobians of batches.** `compute_batch` computes values only; the derivatives with respect
  to each distribution's parameters would need per-distribution θ blocks, already supported by
  the engine's groups, and a result layout for them.
