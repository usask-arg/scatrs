# Phase 1b follow-ups

These items came out of phase 1b (the sphere phase matrix, the Greek coefficients and the O(n)
Gauss–Legendre rule) and were deliberately deferred. Each belongs to the phase or task named in
its heading.

## Cross-code comparison (with the sasktran2 comparison)
- **Greek coefficients against sasktran2.** Phase 2b compared size-averaged coefficients
  (`docs/validation/sasktran2.md`): the same α₁ to α₄ and normalization, β₁ and β₂ of opposite
  sign, now in `docs/theory/mie.md`. The per-sphere comparison and the aliasing remain.
  - **What:** compare scatrs' Greek coefficients with sasktran2's for the same spheres, and
    record sasktran2's conventions (the signs of β₁ and β₂, the d-function convention and the
    normalization) in the sign-convention table of `docs/theory/mie.md`.
  - **Why deferred:** the table states only scatrs' convention, pinned by the Rayleigh limit
    (β₁² = −√6/2, test `rayleigh_limit_pins_the_sign_conventions`); no other code's convention
    is claimed until it has been checked.
  - **Expect differences from quadrature:** sasktran2 projects on a split Gauss–Legendre grid
    with 2 × `num_coeffs` nodes, which is not exact when N ≫ `num_coeffs`, so its coefficients
    alias (main spec, Appendix A). Compare at orders where both are converged, and
    quantify the aliasing separately.
- **Speed against sasktran2.** Time the Greek coefficients (all orders, and the orders a
  radiative-transfer calculation uses) against sasktran2's Mie and Greek code on the same
  machine, for x from 1 to 10⁴. Measured so far for scatrs only: 0.09 s at x = 10³ and 7.6 s at
  x = 10⁴, all orders, `f64`, Apple M4 Pro.

## Accuracy (phase 2 or later)
- **Remove the node-rounding term.** The `f64` Gauss–Legendre nodes differ from the exact
  ones by about half an ulp next to μ = 1 (rounded double-double and asymptotic rules alike),
  so the Greek coefficients are those of a rule with nodes moved that much. Next to μ = 1 that changes d_L by about L²·2⁻⁵⁴ and the forward peak by about x²·2⁻⁵⁴
  of its size, hence the term ((L + 1)² + x²)/8 in κ′ (measured: about 2·10⁻⁹ of max α₁ at
  x = 2·10⁴ with L = 64, and 2·10⁻¹¹ at x = 10³ for all orders).
  - **Fix:** evaluate the amplitudes (π_n, τ_n) and the Wigner d-functions in terms of 1 − μ
    (or of the angle θ, which the asymptotic rule computes first) rather than of the rounded μ,
    so that the nodes near μ = 1 carry their full relative accuracy.
  - **Then:** drop the quadrature term from the tolerance, re-measure the constants, and widen
    the map beyond x = 10³ if the double-double cost allows.

## Performance (phase 2, size distributions)
- **Caching projectors across spheres.** A `GreekProjector` holds its rule and order; a size
  distribution needs one projector at the largest n_q of the ensemble (a rule exact to degree D
  is exact for every lower degree), with the summed phase matrices of all spheres tabulated at
  its nodes and projected once. Build that in the ensemble layer, with a cache owned by the
  caller or an `Arc`, not a hidden global (main spec §4.5). Round n_q up to buckets if
  projectors are cached across ensembles.
