# Limitations and known issues

<!-- synced with docs/plans/phase-0-followups.md -->
<!-- synced with docs/plans/phase-1b-followups.md -->
<!-- synced with docs/plans/phase-2a-followups.md -->

This page lists what users should know: what scatrs does not cover yet, what calculations
cost, and how results behave at the edges of the validated domain. {doc}`validation/overview`
describes what the accuracy figures rest on.

## Scope and availability

- scatrs computes homogeneous spheres and layered spheres (a core and concentric shells),
  single or averaged over size distributions and external mixtures, and randomly oriented
  spheroids, single.
- Python exposes only `f64` results. Double-double precision and the `Dual` derivatives of
  the Rust core are available from Rust only.
- The tests run on Linux (x86_64), macOS (arm64) and Windows (x86_64) for every change. The
  slowest tests run weekly, on the same three systems: the 20 000-point double-double
  Gauss–Legendre rule against Arb, every asymptotic rule from 101 to 3000 points against
  double-double, and the phase matrix and Greek coefficients of spheres at
  $x = 2\cdot10^3$, $10^4$ and $2\cdot10^4$. The released wheels are also tested on aarch64
  Linux and Windows and on Intel macOS; the s390x, ppc64le and musl Linux wheels are built
  but not tested. 32-bit platforms are not supported.

## Layered spheres

- **Concentric layers only.** Inclusions off the centre, and non-spherical layers, need the
  T-matrix method, which is planned.
- **Validated domain.** $10^{-6} \le x_L \le 10^4$ with $|m_l|\,x_L \le 3\cdot10^4$ for every
  layer, every index in the sphere's domain, and at most 16 layers. Beyond
  $|m|\,x_L = 3\cdot10^4$ the narrow resonances of thin lossless shells of high index cost the
  coefficients up to $10^9$ ulps. A single layer is a sphere, with the sphere's domain.
- **Derivatives up to $x_L = 1000$.** Above it the derivatives lose accuracy faster than the
  values for thin lossless shells (the forward amplitude's derivative with respect to the index
  of a thin air shell around a large core reached errors of $2\cdot10^{-8}$ relative at
  $x_L = 10^4$). Above $x_L = 1000$, `compute_with_derivatives` and `derivatives=` raise
  `OutsideValidatedDomain` unless `allow_unvalidated` is set.
- **Cost.** About 2.5 times a sphere of the outer size for a coated sphere, and roughly
  1.5 times more per further layer ({doc}`theory/layered`).
- **Fixed fractions in ensembles.** A size distribution of layered spheres scales the whole
  particle: each interface stays at a fixed fraction of the outer radius. A coating of fixed
  thickness across the sizes, and fractions or layer indices as grid axes, are not available.
- **Ensemble derivatives up to $x_L = 1000$.** The derivatives of an ensemble of layered
  spheres with respect to their indices and fractions, and the radius derivative of a
  monodisperse one, need every size node within the derivatives' domain: a distribution that
  reaches a larger outer size parameter raises `OutsideValidatedDomain` unless
  `allow_unvalidated` is set. Its values, and its derivatives with respect to the
  distribution's own parameters, are validated up to $10^4$.

## Spheroids

- **Random orientation only.** Fixed orientations, other axisymmetric shapes (cylinders,
  Chebyshev particles) and spheroids in size distributions and mixtures are not yet
  available.
- **Validated domain.** Aspect ratios from 1/5 to 5, equal-volume $x \ge 0.3$,
  $|m|\,x_{\max} \le 60$ with $x_{\max}$ the size parameter of the longest semi-axis,
  $1.3 \le \mathrm{Re}\,m \le 3$, $\mathrm{Im}\,m \le 0.5$, and `rtol` down to $10^{-12}$
  ({doc}`validation/accuracy`). The edge in $|m|\,x_{\max}$ is set by cost, not accuracy.
- **Cost.** It grows like $x_{\max}^4$, and much faster where the modes need mixed-precision
  refinement, which is most often for prolate spheroids and at the largest $|m|\,x_{\max}$.
  At `rtol = 1e-8` on one core an oblate spheroid with $\varepsilon = 2$ takes about 22 ms at
  $x = 5$ and 2 s at $x = 15$; $\varepsilon = 1/5$ at $x = 12$ takes about a minute. The
  quadrature runs in parallel on several cores. `precision="f64"` skips the
  refinement: it is fastest, and the error estimate still reports what it costs in accuracy.
- **Derivatives**, all four together, cost 4 to 8 times the values.

## Mie scattering

- **Validated domain.** $10^{-6} \le x \le 2\cdot10^4$, $0 < \operatorname{Re} m \le 10$,
  $0 \le \operatorname{Im} m \le 10$, $\lvert m\rvert \ge 0.1$ (below 0.1, $Q_\text{abs}$ and
  the Jacobian of nearly real $m$ lose accuracy). Outside it, calls raise `OutsideValidatedDomain`
  unless `allow_unvalidated` is set. Accepted inputs extend to $x \in [10^{-30}, 10^6]$ and
  $\lvert m\rvert \in [10^{-6}, 10^3]$, where the outputs are finite but their accuracy is not
  validated. The number of terms is at most `MAX_MIE_ORDER` = $2^{20}$.
- **Accuracy near resonances.** The declared accuracy is relative to the conditioning of each
  output. Near the sharp resonances of weakly absorbing and lossless spheres the outputs are
  very sensitive to $x$ and $m$, and their absolute errors grow accordingly. See
  {doc}`validation/accuracy`.
- **Truncation and narrow resonances.** The default truncation, `ceil(x + 6 x**(1/3) + 3)`
  terms, can miss an exponentially narrow whispering-gallery resonance of a high-index,
  weakly absorbing sphere above that order. The size parameter must fall in an exponentially
  small window to meet one. Pass a larger `n_max` to include higher orders. See
  {doc}`theory/mie`.
- **Absorption derivatives.** $\partial Q_\text{abs}/\partial x$ and
  $\partial Q_\text{abs}/\partial(\operatorname{Re} m)$ of weakly absorbing spheres are
  accurate relative to $Q_\text{ext}$, not to $Q_\text{abs}$ itself: near a resonance with
  $k = 10^{-8}$ their relative error reached $4\cdot10^{-9}$.
- **m = 1 exactly.** The coefficients vanish, but for $x \ge 1$ the lowest orders are computed
  as rounding noise rather than exact zeros, so $g$ and its derivatives are meaningless there
  (the efficiencies are at rounding level).
- **Cost.** $O(x + \lvert m\rvert x)$ per sphere; the Python functions of {mod}`scatrs.mie`
  loop over the spheres on one thread. The largest accepted $\lvert m\rvert x$ take seconds
  per sphere, and a call cannot be interrupted.
- **Cost of the Greek coefficients.** All orders of one sphere cost $O(x^2)$: 7.6 s at
  $x = 10^4$ in `f64` on an Apple M4 Pro. Ask for fewer orders (`order=`) where a calculation
  does not need them all. All orders also need $2N + 1$ quadrature nodes, at most $2^{17}$, so
  `greek` rejects spheres with $N \ge 2^{16}$ ($x$ above about $6.5\cdot10^4$, outside the
  validated domain) unless asked for fewer orders.
- **Node rounding in the Greek coefficients.** The `f64` Gauss–Legendre nodes are rounded, and
  the error this causes grows like $L^2 + x^2$ for order $L$: measured, about $2\cdot10^{-9}$ of
  $\max_l\lvert\alpha_1^l\rvert$ at $x = 2\cdot10^4$ with $L = 64$, and $2\cdot10^{-11}$ at
  $x = 10^3$ for all orders. The declared accuracy includes it. See {ref}`mie-accuracy-model`.
- **Validation of large spheres.** The accuracy map of the phase matrix and the Greek
  coefficients stops at $x = 10^3$; larger spheres are checked weekly at three sizes,
  $x = 2\cdot10^3$, $10^4$ and $2\cdot10^4$, the last two only to order 64. See
  {ref}`greek-accuracy`.

## Ensembles of spheres

- **Check `converged`.** A size integration that cannot meet `rtol` within `max_panels` panels
  returns its values with `converged = false` rather than raising.
- **Non-absorbing spheres.** Without absorption the efficiencies of large spheres have narrow
  resonance ripples that the size integration may not resolve within `max_panels` (default
  4096). For a lognormal with $r_g = 1$ µm, $\sigma_g = 2$ and $m = 1.33$ at
  $\lambda = 0.5$ µm, the result has `converged = false` at 4096 panels and an estimate of
  about $10^{-5}$ of $\langle C_\text{ext}\rangle$. A little absorption, or a larger
  `max_panels`, helps, though with `max_panels` raised the estimate of the converged result,
  $9.7\cdot10^{-7}$ of $\langle C_\text{ext}\rangle$, fell short of its true error,
  $1.1\cdot10^{-6}$. See {ref}`ensemble-accuracy`.
- **Derivatives with respect to $\mathrm{Im}\,m$ at $\mathrm{Im}\,m = 0$** have a spike at every
  resonance and do not converge (`converged = false`); the values do.
- **Backscattering is not refined by default.** The estimates of $\langle C_\text{back}\rangle$
  and the lidar ratio are indicative unless `refine_backscatter` is set, which can cost many
  more panels for weakly absorbing spheres. See {ref}`ensemble-accuracy`.
- **⟨C_abs⟩ of weak absorbers.** ⟨C_abs⟩ is refined to rtol of ⟨C_ext⟩, so for weak absorbers
  its relative accuracy, and that of the co-albedo, is far worse than rtol, and its estimate is
  indicative: for the database grid of {doc}`validation/sasktran2` ($m = 1.43 + 10^{-8}i$) it
  fell up to 1.6 times short of the true error. The single-scattering albedo itself is accurate
  to about rtol, and its estimate is conservative.
- **Validated domain.** Every size node must be in the validated domain of the Mie solution;
  the tails of broad distributions can reach $x > 2\cdot10^4$. Outside it, calls raise
  `OutsideValidatedDomain` (naming the wavelength and the radius) unless `allow_unvalidated` is
  set. A range that reaches $x > 10^6$, beyond any Mie solution, raises `InvalidInput`.
- **No scattering, no ratios.** A zero or non-finite $\langle C_\text{ext}\rangle$,
  $\langle C_\text{sca}\rangle$ or $\langle C_\text{back}\rangle$ (for $m = 1$, say) raises
  `InvalidInput`: the albedo, the asymmetry parameter and the lidar ratio are undefined.
- **Cost of the Greek coefficients.** The exact Greek coefficients need each size node's
  amplitudes at about $N + L/2$ cosines, $O(N^2)$ for large spheres, whatever the order $L$.
  Measured on an Apple M4 Pro (12 cores), per wavelength, a lognormal with $r_g = 1$ µm,
  $\sigma_g = 2$ and $m = 1.33 + 10^{-4}i$ at rtol $10^{-6}$: the cross sections, $g$ and the
  moments take 0.037 s, 0.11 s and 0.20 s at $\lambda$ = 1, 0.5 and 0.3 µm; with the Greek
  coefficients to order 32, 0.093 s, 0.31 s and 0.90 s. See {doc}`theory/ensembles` and
  {doc}`validation/sasktran2`.
- **Grids and lists compute values only.** A grid of distributions (array-valued parameters)
  or a list of distributions shares its size nodes; derivatives come from an
  {class}`~scatrs.Ensemble` of one distribution at a time. A {class}`~scatrs.Mixture` takes
  ensembles of one distribution each, not grids or lists.
- **Memory of grids.** Each distribution of a grid holds sums over every band's tables:
  about 2.5 MB per distribution and wavelength in flight at $\lambda = 0.28$ µm for the
  database grid of {doc}`validation/sasktran2` (2 GB for 800 distributions on one thread; more
  wavelengths run at once on more threads). Split very large grids.
- **Tables can exceed `max_panels`.** Every radius of a `Tabulated` distribution is a panel
  edge, so a table of more radii than `max_panels` still takes more panels than `max_panels`
  (and converges).
- **Distributions over size only.** The particles are spheres, so the distributions are over
  their radius only.
- **Threads.** The computation runs on all cores, through rayon's global thread pool. Python
  has no call to set the number of threads; set the environment variable `RAYON_NUM_THREADS`
  before the first call (in Rust, a caller can `install` its own pool).
- **`f64` only.** The ensemble engine is not generic over `Real`: no double-double or `Dual`
  ensembles. Its derivatives come from the closed-form single-sphere derivatives instead.

## Python bindings

- **Shared inputs.** Calls release the GIL and read input arrays in place. Do not modify or
  resize an input array from another thread during a call.
- **Output size.** One Mie call returns at most `MAX_MIE_VALUES` = $2^{27}$ complex values,
  counting derivatives (2 GiB), or twice as many real values for the efficiencies, the phase
  matrix and the Greek coefficients. The bound applies to the broadcast output, which small
  inputs can make large: a call above it raises `InvalidInputError` before allocating
  anything.
- **Per-call overhead.** About 1 µs per call (a one-off measurement on an Apple M4 Pro): prefer
  one call on a large array to many small calls.

## Rust API

The rest of this page concerns the numerical building blocks of the Rust crate, and the
special functions of {mod}`scatrs.special`.

### Gauss–Legendre quadrature

- **Accuracy above 100 points.** `f64` rules above 100 points come from asymptotic expansions,
  within $32\cdot2^{-52}$ (measured: 2.8 and 4.0 units of $2^{-52}$ for nodes and weights)
  rather than the half ulp of the double-double rule rounded once.
- **Double-double cost.** Double-double rules use the $O(n^2)$ Newton method at every $n$. On
  an Apple M4 Pro that takes about 11 s at $n$ = 20 000, and an extrapolated 8 minutes at the
  cap of $2^{17}$ points. A call cannot be interrupted; compute a double-double rule once and
  reuse it.

### Builds without FMA

On x86_64 the default target has no fused multiply-add, so every double-double operation calls
a software `mul_add`. That makes double-double code, Gauss–Legendre included, about ten times
slower. Build with `RUSTFLAGS="-C target-cpu=native"` where the CPU has FMA.

### Wigner d-functions

- **Bounds.** The accuracy bounds are measured, not proven. The tests validate them up to
  $l = 2.5\cdot10^5$ for $(m, n) = (0, 0)$ and $(0, 2)$. For other pairs they validate them at
  sampled degrees up to $l = 2000$, up to $l = 3000$ in two cases with underflowing start
  values, and up to $l = 4000$ only for $(35, 34)$ and $(0, 0)$ next to the poles. One-off
  measurements went to $l$ = 20 000. See {ref}`wigner-d`.
- **Second derivatives at the poles.** With nested duals, second derivatives at $x = 1$ are
  silently zero when $\lvert m - n\rvert$ is 3 or 4, and at $x = -1$ when $\lvert m + n\rvert$
  is 3 or 4. The true second derivative is infinite for 3, and finite but nonzero for 4.
- **Output size.** One Python call returns at most `MAX_WIGNER_D_VALUES` = $2^{28}$ values
  (2 GiB of float64).

### Double-double arithmetic

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
- **Speed.** `exp` takes about 200 ns and `sin` about 170 ns.

### Complex arithmetic

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

The Mie kernels use their own complex reciprocal (the plain formula where it can neither overflow
nor underflow, Smith's algorithm elsewhere), validated against Arb, in place of num-complex's.

### Dual numbers

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
  That has been measured only on macOS (arm64); the tests also run with glibc and the Windows C
  runtime, and weekly on Intel macOS. There Apple's `sin` and `cos` reach 2.4 units of
  $2^{-53}$ (1.1 on arm64, 1.0 with glibc on x86_64), so the test of the values `Dual` takes
  from them allows 3 units instead of 2 on that target only.
- **Vectorisation.** A loop over many `Dual`s is not vectorised across elements.
