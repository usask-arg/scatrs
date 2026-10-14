# Phase 2a follow-ups

These items came out of phase 2a (ensembles of spheres: size distributions, adaptive size
integration and mixtures) and were deliberately deferred. Each belongs to the phase or task named
in its heading.

## Performance (phase 5)
- **Resolved in phase 2b** (`docs/specs/2026-10-14-phase-2b-speed-and-sasktran2-design.md`):
  - **Shared πₙ/τₙ tables per band**, now with matrix products for a panel's spheres, which
    made the Greek coefficients of the broad lognormal of `docs/theory/ensembles.md` 7–12 times
    faster (to order 32 at λ = 1, 0.5 and 0.3: 0.62 s, 2.9 s and 10.4 s before).
  - **Clebsch–Gordan Greek coefficients** (products of Mie coefficients, O(N·L) per sphere) were
    measured and not adopted; see `docs/plans/phase-2b-followups.md`.
- **The phase-1b item "Caching projectors across spheres"** was resolved differently: one
  projector per band of the number of terms (bands growing by a factor 1.25, chosen per panel),
  since one shared projector at the largest n_q made every small sphere pay for the largest.

## Accuracy of the error estimates (phase 5)
- **Unresolved ripples of non-absorbing spheres.** Found while updating the documentation: for
  the lognormal r_g = 1, σ_g = 2, m = 1.33 at λ = 0.5 with `max_panels` raised, the converged
  result's estimate of ⟨C_ext⟩ (9.7·10⁻⁷ of it) falls short of its true error (1.1·10⁻⁶), and
  by 5–11× at rtol 1e-8 to 1e-10: K15 and G7 both miss resonance ripples narrower than a panel.
  Candidates: refining on a ripple-sensitive quantity, or a minimum node density in x for
  Im m ≈ 0.

## Cross-code comparison and benchmark (phase 2b)
- **Done in phase 2b:** `benchmarks/compare.py` and `docs/validation/sasktran2.md`.

## Declarative Python API (phase 2c)
- **The high-level API of main spec section 9.2:** refractive-index models over wavelength,
  xarray outputs, and `set_num_threads` (Python now runs on rayon's global pool).

## Other scattering methods (phase 4)
- **Distributions over parameters other than size** (shape, aspect ratio) need the T-matrix and
  arrive with it.
