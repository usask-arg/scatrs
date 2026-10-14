# Phase 2a follow-ups

These items came out of phase 2a (ensembles of spheres: size distributions, adaptive size
integration and mixtures) and were deliberately deferred. Each belongs to the phase or task named
in its heading.

## Performance (phase 5)
- **Clebsch–Gordan Greek coefficients.**
  - **What:** compute each sphere's Greek coefficients from its Mie coefficients with
    Clebsch–Gordan (Wigner 3j) sums, O(N·L) per sphere, instead of projecting its amplitudes at
    about N + L/2 Gauss–Legendre cosines, O(N²) for large spheres.
  - **Why deferred:** the projection is exact and validated (phase 1b), and pass 2 with Greek
    coefficients is the cost of broad distributions of large spheres. Measured on an Apple M4
    Pro, one wavelength, lognormal r_g = 1, σ_g = 2, m = 1.33 + 10⁻⁴i, rtol 1e-6: 1.1 s, 5.9 s
    and 22.6 s at λ = 1, 0.5 and 0.3 to order 32, against 0.04 s, 0.11 s and 0.23 s for the
    cross sections alone.
  - **Then:** validate against the projection and the Arb references, and re-measure the table
    in `docs/theory/ensembles.md` and `docs/limitations.md`.
- **Shared πₙ/τₙ tables per band.** The nodes of a projector band share its cosines, and πₙ(μ)
  and τₙ(μ) do not depend on the sphere: tabulate them once per band and wavelength instead of
  once per size node.
- **A faster pass 2 for low orders.** Radiative transfer often asks for tens of orders of
  spheres with thousands of terms; the exact projection still needs about N + L/2 cosines per
  sphere whatever L, so pass 2 costs about the same for L = 32 as for far higher orders. The two
  items above are the candidates.
- **The phase-1b item "Caching projectors across spheres"** was resolved differently: one
  projector per band of the number of terms (bands growing by a factor 1.25), since one shared
  projector at the largest n_q took 20 s per wavelength for a broad distribution.

## Accuracy of the error estimates (phase 5)
- **Tighter Greek error estimates.** Found while writing the documentation: the estimates of the
  Greek coefficients are bounds but loose by orders of magnitude. For the lognormal
  r_g = 0.08, σ_g = 1.6, m = 1.43 + 10⁻⁸i at λ = 0.55, the estimate of the unnormalized α₁⁰
  (which equals ⟨C_sca⟩) is 6.8·10⁻⁴ against 1.8·10⁻⁸ for ⟨C_sca⟩, with a true error of about
  10⁻⁹. Within a panel the band of a node changes with its radius, so each band's table is a
  step function of r, and its |K15 − G7| is of the order of the panel's whole contribution.
  Projection is linear, so the difference of the K15 and G7 coefficients summed over the bands
  would estimate the error of the continuous integrand instead (per panel it would need a
  projection per panel; over all panels it loses the triangle inequality across panels).

## Cross-code comparison and benchmark (phase 2b)
- **sasktran2's `MieDatabase`.** Compare the ensemble outputs (cross sections, ω, g, Greek
  coefficients) with sasktran2's for the same distributions and wavelengths, and record any
  differences of convention; then benchmark both on the same machine. This is the phase-2 exit
  criterion of the main spec (section 13). The phase-1b follow-ups on sasktran2's Greek
  conventions and aliasing apply here too.

## Declarative Python API (phase 2c)
- **The high-level API of main spec section 9.2:** refractive-index models over wavelength,
  xarray outputs, and `set_num_threads` (Python now runs on rayon's global pool).

## Other scattering methods (phase 4)
- **Distributions over parameters other than size** (shape, aspect ratio) need the T-matrix and
  arrive with it.
