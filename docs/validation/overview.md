# How scatrs is validated

A scattering code can go wrong in two different ways, and scatrs checks for each separately.

- **The numerics.** The Mie series is summed with recurrences that can lose accuracy to
  rounding, cancellation, instability or early truncation. Such errors grow in ways that are
  hard to predict: one size parameter may be fine while a nearby one, at a resonance or with a
  refractive index near 1, is not. A few spot checks cannot find them.
- **The physics and the conventions.** A formula with a wrong sign, a phase matrix with the
  wrong normalization, or a refractive index of the opposite convention gives results that
  are precise but wrong. These errors need a comparison with independent codes and with known
  physics.

scatrs addresses the first with **internal validation**: every output is compared with
reference values computed independently at very high precision, over the whole domain where
scatrs claims accuracy. It addresses the second with **external validation**: comparisons
with the published results of other codes and with analytic limits. Both run automatically,
on Linux, macOS and Windows, for every change to the code.

## Internal validation: is every result as accurate as claimed?

Each output of {mod}`scatrs.mie` has a declared accuracy, a few units of double-precision
rounding, times a factor that grows with the size of the sphere and with how sensitive the
output is to its inputs (near a sharp resonance, even rounding $x$ and $m$ to double precision
changes the answer). The tests check it two ways:

- **Against rigorous reference values.** An independent program computes the Mie series in
  [Arb](https://arblib.org) ball arithmetic, which carries a guaranteed error bound with every
  number, at hundreds of bits of precision. It never uses scatrs's code. Its values cover
  size parameters from $10^{-6}$ to $2\cdot10^4$ and refractive indices from near 1 to
  metal-like: 306 spheres for the efficiencies and amplitudes, 80 for their derivatives, and
  18 for the phase matrix and Greek coefficients.
- **On a dense grid.** The same algorithms also run in double-double arithmetic, with about
  32 significant digits (itself checked against Arb). The double-precision results are
  compared with these on a grid of thousands of spheres, so that no region of the domain goes
  unchecked.

The declared accuracy of each output is at least four times the worst error found, and a test
fails if any point exceeds it. For the averages over size distributions, a further test
checks that the error estimate of every output covers its true error, measured against the
same integral at a far tighter tolerance.

{doc}`accuracy` shows the results.

## External validation: does scatrs agree with established results?

The reference values are computed from the same equations as scatrs. They would not catch a
mistake in an equation, or a convention that differs from everyone else's. So scatrs is also
compared with results it shares nothing with:

- **Wiscombe's MIEV0 test cases.** MIEV0 is the most widely used Mie code, and its published
  test cases are the standard benchmark for new ones. scatrs reproduces all fifteen cases,
  from $x = 0.055$ to $10^4$ and from $m = 0.75$ to $10 + 10i$, to within about one unit in
  the last of the seven digits published.
- **Analytic limits.** Small spheres must follow Rayleigh scattering, in their cross sections
  and in the sign and normalization of every element of the phase matrix and every Greek
  coefficient. Large spheres must remove twice their geometric cross section, and every
  sphere must satisfy the optical theorem.
- **sasktran2.** For size distributions, scatrs is compared with the Mie integration of the
  sasktran2 radiative transfer model, on a single distribution and on an optics-database grid.

{doc}`external` runs the first two on every build of this documentation, and
{doc}`sasktran2` shows the third.

## Summary

| Check | What it would catch | Reference | Runs |
|---|---|---|---|
| Arb reference values | numerical error in any output, derivatives included | independent arbitrary-precision computation with rigorous error bounds | every change |
| Double-double accuracy map | numerical error anywhere in the validated domain | the same algorithms with about 32 digits | every change |
| Ensemble error estimates | estimates that understate the error of a size average | the same average at a far tighter tolerance | every change |
| MIEV0 test cases | wrong equations or conventions | Wiscombe's published values | every change |
| Analytic limits | wrong normalization or sign conventions | Rayleigh scattering, the optical theorem, the extinction paradox | every change |
| sasktran2 | disagreement with an established distribution code | sasktran2's Mie integration | when the comparison is rerun |

(what-backs-the-numbers)=
## How strong is each accuracy figure?

Every accuracy figure in these pages says what supports it:

- **Measured and enforced:** the accuracy of the Mie outputs, $C\,(\kappa' + \mathrm{cond})
  \,2^{-52}$ with measured constants $C$, checked against the Arb references and the
  double-double map. These are not proofs: a point that no test samples could in principle
  exceed its tolerance, but the dense map and the margin of four make that unlikely.
- **Calibrated:** the error estimates of the ensembles are estimates, not proven bounds. A
  test checks them against the same integrals at a tight tolerance, on a grid of nine
  distributions.
- **Tested against Arb:** the accuracy of the special functions (Gauss–Legendre quadrature and
  the Wigner d-functions) and of the double-double arithmetic used by the Rust API.
- **Measured:** worst-case errors and timings quoted from the tests and benchmarks, or from
  one-off measurements on an Apple M4 Pro, which are marked as such.

{doc}`/limitations` lists what is not covered: inputs outside the validated domain, and the
behaviour of the outputs near sharp resonances.
