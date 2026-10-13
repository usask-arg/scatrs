# Limitations and known issues

<!-- synced with docs/plans/phase-0-followups.md -->

scatrs is pre-alpha. This page lists what users of phase 0 should know: missing pieces, costs,
and behaviour at the edges. The repository tracks each item, with the phase that is to address
it, in `docs/plans/phase-0-followups.md`.

(what-backs-the-numbers)=
## What backs the accuracy figures

- **Proved:** the error bounds of the double-double basic operations (`+`, `−`, `×`, `÷`), from
  Joldes, Muller & Popescu (2017).
- **Tested against Arb:** the bounds of the double-double elementary functions, of
  Gauss–Legendre and of the Wigner d-functions. They hold on every reference value, but they
  are not proved for all inputs, and the Wigner bounds are measured, not derived.
- **Measured:** worst-case errors and timings quoted from test output, and from probes run
  during code review that are not in the repository (marked as such).

## Scope and availability

- There are no scattering calculations yet: Mie, layered spheres and the T-matrix arrive from
  phase 1 on. Phase 0 provides the scalar types, two special functions and the reference suite.
- Python exposes only `f64` results in phase 0. `DoubleDouble` and the `Dual` derivatives are
  available from Rust only.
- Neither wheels nor the Rust crate are published, and the Rust API is not on docs.rs. See
  {doc}`installation`.
- Continuous integration runs the Rust and Python tests on Linux (x86_64), macOS (arm64) and
  Windows (x86_64) for every change. The slow tests marked `#[ignore]` run only weekly, on the
  same three systems: the 20 000-point Gauss–Legendre rule against Arb, and the root checks for
  every rule up to 3000 points. The release workflow also builds wheels for platforms that no
  tests run on (32-bit x86 and ARM, aarch64 Linux and Windows, s390x, ppc64le, musl Linux and
  Intel macOS).

## Gauss–Legendre quadrature

- **Cost.** The method is $O(n^2)$. On an Apple M4 Pro it takes about 11 s at $n$ = 20 000,
  and an extrapolated 8 minutes at the cap of $2^{17}$ points. Mie quadrature at size parameter
  $2\cdot10^4$ needs about $4\cdot10^4$ points, which takes about 45 s.
- **Interruption.** A call cannot be interrupted. In Python, Ctrl+C takes effect only once it
  returns. Compute a rule once and reuse it.
- **Planned speed-ups.** Two tunings of the present method measured about 5× faster together
  during code review (not committed). The real fix is an $O(n)$ method (Bogaert 2014). Faster
  rules are a priority for phase 1, and the cap will be raised once the $O(n)$ method lands.

## Builds without FMA

On x86_64 the default target has no fused multiply-add, so every double-double operation calls
a software `mul_add`. That makes double-double code, Gauss–Legendre included, about ten times
slower. Build with `RUSTFLAGS="-C target-cpu=native"` where the CPU has FMA. How the published
wheels will handle this is still to be decided.

## Wigner d-functions

- **Bounds.** The accuracy bounds are measured, not proven. The tests validate them up to
  $l = 2.5\cdot10^5$ for $(m, n) = (0, 0)$ and $(0, 2)$. For other pairs they validate them at
  sampled degrees up to $l = 2000$, up to $l = 3000$ in two cases with underflowing start
  values, and up to $l = 4000$ only for $(35, 34)$ and $(0, 0)$ next to the poles. Probes
  during code review (not committed) went to $l$ = 20 000. See {ref}`wigner-d`.
- **Second derivatives at the poles.** With nested duals, second derivatives at $x = 1$ are
  silently zero when $\lvert m - n\rvert$ is 3 or 4, and at $x = -1$ when $\lvert m + n\rvert$
  is 3 or 4. The true second derivative is infinite for 3, and finite but nonzero for 4.
- **Output size.** One Python call returns at most `MAX_WIGNER_D_VALUES` = $2^{28}$ values
  (2 GiB of float64).

## Double-double arithmetic

- **Trigonometric domain.** `sin`, `cos` and `tan` return NaN for $\lvert x\rvert > 2^{50}$,
  where `f64` still gives finite values. Kernels that take angles from users must check them
  first.
- **Near multiples of $\pi/2$.** Beyond $\pi/4$ the contract of `sin` and `cos` is only
  $2^{-100}$ absolute. The tests also check a relative bound with an absolute floor,
  $2^{-100}\lvert f\rvert + \lvert k\rvert\cdot2^{-161}$ for `f64` arguments. So relative
  accuracy is lost only within about $\lvert k\rvert\cdot2^{-60}$ of a nonzero multiple
  $k\pi/2$. See {ref}`double-double`.
- **Full precision.** It needs magnitudes of at least $2^{-969}$.
- **Platform-dependent last bits.** `ln`, `cbrt` and `atan2` (and functions built on them) start
  from the platform's `f64` functions. Their last bits can differ between platforms, within
  the accuracy bounds, so bitwise golden tests across operating systems would fail.
- **`mul_add` is not fused.** See {ref}`generic-code-notes`.
- **Speed.** `exp` takes about 200 ns and `sin` about 170 ns. Table-driven versions, measured at
  about 3× faster for `exp`, are planned.

## Complex arithmetic

`num_complex::Complex<T>` uses textbook formulas, which are accurate only in norm, for
`Complex<f64>` as well as `Complex<DoubleDouble>`:

- division overflows in `f64` for components beyond about $10^{154}$, and has no componentwise
  accuracy;
- `ln` near $z = 1$ loses relative accuracy.

For `Complex<DoubleDouble>` specifically:

- **`sqrt` next to the negative real axis.** It can return the wrong branch.
  `sqrt(-9 + 1e-300i)` gives $\operatorname{Re} \approx -4.5\cdot10^{-33}$ instead of
  $+1.7\cdot10^{-301}$.
- **Division by tiny divisors.** It loses precision for $\lvert w\rvert \lesssim 10^{-146}$.
- **NaN where `f64` is finite.** `exp`, `sin`, `cos`, `powc` and related functions return NaN
  when a component exceeds $2^{50}$, the double-double trigonometric domain.

Phase 1 is to add scatrs-owned complex division, `ln`, `sqrt` and `abs`, validated against
Arb, before the Mie kernels.

## Dual numbers

- **Roots of squares.** `sqrt(p * p)` at $p = 0$ gives derivative 0, not $\pm p'$. Use `abs`
  or `hypot`. See {ref}`strong-zeros`.
- **Points where a function is not differentiable.** These have fixed conventions. For
  example, `hypot` has derivative 0 at the origin, while `atan2` has NaN derivatives there. See
  {ref}`dual-conventions`.
- **Apple targets.** `Dual<f64, N>` `sin` and `cos` take their values from the fused `sincos`.
  For about 1 in 13 000 arguments its sine is an ulp off a lone `f64::sin`. See
  {ref}`dual-sin-cos`.
- **`Complex<Dual>` values.** Strict zero tests change some values of `Complex<Dual>`
  functions from those of `Complex<f64>`, by a few ulps or in special cases. See
  {ref}`dual-zero-tests`.
- **libm accuracy.** The tests of `f64` derivatives assume each libm call is within 2 ulps.
  That has been measured only on macOS (arm64); CI also runs these tests with glibc and the
  Windows C runtime, and weekly on Intel macOS. There Apple's `sin` and `cos` reach 2.4 units
  of $2^{-53}$ (1.1 on arm64, 1.0 with glibc on x86_64), so the test of the values `Dual`
  takes from them allows 3 units instead of 2 on that target only.
- **Vectorisation.** A loop over many `Dual`s is not vectorised across elements.

## Python bindings

- **Shared inputs.** Calls release the GIL and read input arrays in place. Do not modify or
  resize an input array from another thread during a call.
- **Per-call overhead.** Measured during code review on an Apple M4 Pro (not committed), the
  extension call costs about 180 ns and the Python wrapper about 0.84 µs. Prefer one call on a
  large array to many small calls.

## Reference suite

- **Recorded precision.** The manifest records the starting precision of the `wigner_d`
  generator, 800 bits, although it doubles that as needed, to at most 51 200 bits.
