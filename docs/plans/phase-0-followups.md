# Phase 0 follow-ups

These items came out of the Phase 0 reviews and were deliberately deferred. Each belongs to the phase or task named in its heading.

## Performance (Phase 1, with the criterion and instruction-count benchmarks)
- **Table-driven `DoubleDouble::exp`.**
  - **Method:** 2^e · 2^(j/64) · e^r with a 1 KB table and |r| ≤ ln2/128. The polynomial is double-double up to r⁶ and f64 from r⁷ onward.
  - **Measured on a prototype in the Task 4 review:** 60 ns versus 203 ns, with identical accuracy (≤ 1.9 u²).
  - **Knock-on effect:** it speeds up everything that is built on `exp`.
- **Table-driven `DoubleDouble::ln` that doesn't go through `exp`.**
  - **Expected:** about 40–50 ns, versus about 235 ns now.
  - **Side benefits:** it removes the dependence on the platform's `f64::ln` seed, and the ×4.5 error amplification near |ln a| ≈ 0.22.
- **`ln_1p_series` (about 200 ns).** Use reciprocal tables, or an f64 tail for terms below 2⁻⁵³.
- **Gauss–Legendre is O(n²) in DD and too slow for large Mie rules. This is a priority for Phase 1.**
  - **Current cost:** about 28 ns·n² on an M4 Pro (n = 1000 takes 32 ms, n = 5000 takes 0.72 s, n = 20000 takes 11 s). Mie at x = 2·10⁴ needs about 4·10⁴ nodes, about 45 s.
  - **Cap:** `MAX_GAUSS_LEGENDRE_POINTS` = 2¹⁷, a controller decision in Task 9. Raise it when the O(n) method lands.
  - **Speed-ups measured in the Task 9 review, without changing the algorithm:**
    1. **Drop the polish step.** The remaining error after |Δx| ≤ 2⁻⁹⁰ is ≤ 2⁻¹⁴², even at n = 2²⁰. This was measured before the cap was lowered to 2¹⁷. Replace the final Pₙ′ evaluation with the ODE correction P′(x−δ) ≈ P′(x)(1 − 2xδ/(1−x²)), stopping at |Δx| ≤ 2⁻⁶⁰/n. This needs 2.0–2.2× fewer evaluations, with f64 output bit-identical on 31 random n ≤ 6000.
    2. **Interleave K roots in lockstep.** K = 2 gives 1.9×; K = 4 or 8 gives 2.7× (saturated). Results are bitwise identical per root.
    - Combined, about 5×: n = 4·10⁴ goes from about 45 s to about 8 s.
  - **Real fix: Bogaert (2014),** an iteration-free O(n) method that works in θ. It is about ε-accurate and also removes the DD weight loss near ±1, about log₂(n²/10) bits. Cross-check it against the DD rule and Arb.
  - **Caching (Phase 1 design note):** a rule exact to degree D is exact for every lower degree, so an ensemble needs only the rule at its largest n_q. Round n_q up to buckets if caching. A hidden global cache conflicts with §4.5, so use a cache owned by the caller or an `Arc`.
- **CI for `#[ignore]` tests:** done. The weekly `scheduled.yml` `ignored-tests` job runs them.
- **x86_64 builds without FMA.**
  - **Problem:** `f64::mul_add` becomes a libm call, which makes every DD operation about 10× slower in default wheels.
  - **Fix:** select Dekker's split `two_prod` when `cfg(not(target_feature = "fma"))`, and consider shipping x86-64-v3 wheels as well. Decide this before the first public release (CONTRIBUTING says the same). A non-FMA build also has to change `mul_dd` and `mul_f64`, which use FMA in their non-exact steps. Those operations then need the non-FMA error bounds from Joldes, Muller & Popescu (2017), not just a different `two_prod`. FMA and non-FMA builds will then give bitwise different results. The spec allows this, since it only requires determinism across thread counts.
- **`mul_pow2` zero inputs:** these deliberately take the cold `renormalize_tiny` path, which turns a −0 low word into +0. Do not optimise this away.
- **`DoubleDouble::rem`** allocates a `Vec` on every call. The expansion length bound is unproven; 6 components is the most observed. Switch to a stack buffer if a bound can be proved.

## Correctness and robustness (low priority, outside the documented domain)
- **DD `sin`/`cos`/`tan` return NaN for |x| > 2⁵⁰.**
  - **Decision (controller, Task 5):** this is a documented domain limit, like `sqrt(-1)`, so it is not a silent failure. Kernels never pass such arguments and must validate their own domains.
  - **If ever needed:** Payne–Hanek reduction, with a table of about 1200 bits of 2/π.
- **DD trig accuracy near nonzero multiples of π/2.** The contract only promises absolute accuracy there, but the actual bounds are better. These were derived in the Task 5 follow-up, checked on 30k Arb inputs, and are now documented in `trig.rs`:
  - The absolute error floor is |k|·2⁻¹⁶¹ for f64 arguments and |k|·2⁻¹⁵⁸ for DD arguments.
  - Relative error stays ≤ 1.5·2⁻¹⁰⁰ whenever |x − kπ/2| ≥ |k|·2⁻⁶⁰ (f64) or |k|·2⁻⁵⁷ (DD).
  - Only arguments closer than that to a multiple need callers to reduce them first.
- **Platform dependence of the DD `atan2` seed.** atan2, atan, asin and acos start from the platform `f64::atan2`.
  - A seed that differs by one ulp changes the final DD bits in about 26% of cases, by up to 2⁻¹⁰³·⁷ relative.
  - Accuracy is unaffected: the Newton step tolerates seeds up to 2¹⁶ ulps off.
  - Bitwise golden tests across operating systems would fail, however.
  - Fix: a Rust port of fdlibm atan2 as the seed, or a table-driven DD atan that needs no seed. The same applies to the `ln`/`cbrt` seeds.
- **Kernels that take user-supplied angles** must validate them before converting to DD, because DD sin/cos return NaN for |x| > 2⁵⁰ where f64 gives finite values.
- **DD trig speed.** sin/cos take about 170 ns, atan2 about 370 ns and asin/acos about 400 ns, against 2.3 ns for f64 sin. Measured options from the Task 5 review:
  1. **f64 Horner tail for orders above 17.** sin goes from 148 to 82 ns (1.8×) with an identical result to 2⁻¹⁰⁶. For cos, start the tail at order 20.
  2. **cos as √((1−s)(1+s)) inside sin_cos.** 117 ns versus 302 ns (2.6×), with a difference of 2⁻¹⁰⁴·⁵. This breaks the documented bitwise guarantee `sin_cos == (sin, cos)`, so it needs either an internal-only variant for atan2 or a relaxed guarantee.
  3. **sin/cos(j/64) tables.** An estimated 3–4× on sin_cos.
  4. **A table-driven DD atan.** This avoids sin_cos and the libm seed.

## Packaging and CI
- **Done in Task 13:**
  - `LICENSE` ships in the crate.
  - README links are absolute.
  - Reference-backed tests are excluded from the crate, or skip unless `SCATRS_REQUIRE_REFERENCE_DATA=1`.
  - The release has a preflight job, tests every wheel with `test-wheels` and smoke-tests the sdist.
  - Actions are SHA-pinned.
  - Reference generators no longer depend on the platform: regeneration is byte-identical on macOS, Linux x86_64 and Linux aarch64.
- **Open:**
  - musllinux wheels are built but never installed and tested. Test them inside a `python:3.x-alpine` container on `ubuntu-latest`.
- **First real CI run:**
  - Windows numerics (UCRT sin/cos/pow/fma) are the main unknown; the logs record the measured worst sin/cos error.
  - The `intel-macos` 3u sin/cos bound was measured under Rosetta (2.40u), so the weekly job confirms it on real Intel hardware.

## Complex arithmetic (Phase 1, before the Mie kernels)
- **Accuracy of num-complex.** num-complex's formulas give only norm-wise accuracy. This was found in the Task 6 review and holds for `Complex<f64>` as well as `Complex<DoubleDouble>`:
  - **Division** uses `(ac + bd)/(c² + d²)`, which overflows for |c|, |d| > ~1e154 in f64 and has no componentwise accuracy.
  - **`ln` near z = 1** loses relative accuracy.
  - **Componentwise accuracy fails** near cancellation.

  Mie and T-matrix recurrences can reach large |ξₙ| with absorbing particles. Phase 1 should add scatrs-owned complex helpers for these operations and validate them against Arb:
  - Smith or Baudin–Smith division;
  - `ln` via `ln_1p` near 1;
  - a stable `hypot`/`abs`.
- **`Complex<DoubleDouble>` hazards** (found in the Task 6 review):
  - **(a) `sqrt` breaks the principal branch next to the negative real axis.** `sqrt(-9 + 1e-300i)` gives Re = −4.49e-33, but the true value is +1.7e-301. The DD π sits about 3e-33 above π, so θ/2 overshoots −π/2. Kernels that choose a branch by the sign of Re would flip the root. Fix: a scatrs-owned Kahan- or Hull-style `csqrt`.
  - **(b) DD complex division loses precision for |w| ≲ 2⁻⁴⁸⁴ (~1e-146).** This happens because `norm_sqr` falls below 2⁻⁹⁶⁹. Measured errors: 3.5e5 u² at 2⁻⁴⁹⁰ and 1.8e11 u² at 2⁻⁵⁰⁰. f64 is still exact there. Fix: scaled Smith division.
  - **(c) NaN where f64 is finite.** `exp`, `cis`, `from_polar`, `sin`, `cos` and `powc` return NaN when a component exceeds 2⁵⁰, the DD trig domain. Example: `exp(i·2⁵¹)`.
- **Fused `mul_add`.** `DoubleDouble::mul_add` is not fused (error ≤ 4u²·|x·a|). Add a fused version only if a kernel needs it.

## Dual numbers (Phase 1)
- **Performance on other architectures.** Measure `Dual` division and `Dual / f64` on x86_64 before settling the formulation. x86 `divsd` throughput is about 1 per 4–5 cycles, versus ≥ 1 per cycle on Apple M4, so the correctly rounded second division will cost more there.
- **No cross-element vectorisation.** Loops over many `Dual`s are not vectorised across elements, unlike the same loops over `f64`. Within a single `Dual` at N ≥ 4, standalone `*` and `/` do use two-lane NEON across the derivative components. Results are codegen-sensitive: the plan's N=8 multiply chain varied between 10 and 33 cycles across harnesses. Real-kernel Jacobian overhead (§10.1 target ≤ (1+N)×) must be judged on phase 1 kernels. A structure-of-arrays layout across sizes may be needed.
- **N=2 throughput regression in one harness.** The one-branch division check (b609ea7) cost N=2 its NEON packing in the Task 7 reviewer's element-wise harness, 1.33 → 1.71 ns per element (+28%). N=1 improved by 10%, and N=4, N=8 and the chains were unchanged. The trade-off depends on codegen, so re-check it in the phase 1 kernels.
- **Strong-zero caveat.** sqrt(p²) or (p²)^¼ at p = 0 give derivative 0, not inf/NaN. Kernels must use `abs`/`hypot` rather than sqrt of squares.

- **Conventions at non-differentiable points** (controller decision, Task 8):
  - `hypot` at the origin returns derivative 0. That is a valid subgradient, since hypot is convex.
  - `atan2` at the origin returns NaN, because the function jumps there.
  - `log` with base ±0 returns a base derivative of −sign(ln x)·∞, and 0 at x = 1 (changed in the Task 8 fixes).
  - All three are documented in `dual/mod.rs`.
- **Libm accuracy assumptions in tests.** The f64 derivative bounds in `dual_functions.rs` assume each libm call is within 2 ulps, which has only been checked on macOS. If CI on glibc or the Windows CRT fails, measure that platform's libm and adjust the per-platform bound with a written justification.
- **`Dual<DoubleDouble>` sin speed.** The derivative needs DD cos, so `Dual<Dd>` sin costs about 2× DD sin. It benefits directly from the DD sin_cos speedups above.

- **Indicative `Dual<f64, N>` costs, throughput relative to plain f64.** Measured on an M4 after the Task 8 sin/cos revert (2a35d9e). Replace these with the phase 1 criterion benchmarks.

  | Function | Cost |
  |---|---|
  | sin and cos | 1.6–1.8× (one fused libm call) |
  | `Complex::exp` | 1.2× (N=1), 1.8× (N=4) |
  | exp | about 1.15× |
  | sqrt | about 4.4× (plain f64 sqrt vectorises across elements) |
  | tanh, \|x\| > 0.55 | about 2× |
  | DD sin | about 2× |

- **Apple fused sincos.** On Apple targets, `Dual::sin`/`cos` return the fused `__sincos_stret` values. Any f64 kernel computing both sin and cos of an argument gets the same values, so value-only and Jacobian calls agree bitwise. A lone `f64::sin` can differ by 1 ulp in about 1 in 13,000 arguments. glibc and MSVC are unaffected. This was a controller decision in Task 8.
- **Cross-platform CI is essential.** The Task 8 review found a Linux-only failure: macOS passed only because Apple's `pow` happened to cancel a test-reference error. The CI added in Task 13 runs the full test suite on Linux, macOS and Windows. The first real run, especially Windows, is still pending.
- **`Dual<Dd>::log` cost.** The value must come bitwise from `T::log`, so it computes DD `ln` twice, about 2× DD `log`.

## Wigner d-functions (Phase 1)
- **Method change from the plan** (controller-approved in Task 10). The planned MTL B.22 three-term recurrence has error ∝ l² near the poles (16–266·(l+1)·eps measured). So for x ≥ 0.9, or m = ±n on the |d| → 1 side, Reinsch difference forms are used instead. Everywhere else the results are bitwise equal to the plan.
- **Reciprocal speed-up** in the three-term step: 1.42× for f64 (3.75 → 2.63 ns per l), but as a generic change it slows DD by 13–26% and changes the bits. Defer it, perhaps behind a per-type hint.
- **Batched API for the Greek projection.** Evaluate many nodes at once, sharing the per-l coefficients, so the latency-bound recurrence becomes throughput-bound. This is likely a bigger gain than per-(m, n, L) coefficient tables.
- **Python bindings bound `l_max`** via `MAX_WIGNER_D_VALUES` = 2²⁸ (resolved in Task 11).
- **The manifest `precision_bits` understates generator precision** for `wigner_d`: 800 is the starting value, and it doubles up to at most 51 200 bits. Consider recording per-record precision.

## Python binding patterns (from the Task 11 review; apply in Phase 1)
- **Never use `PyReadonlyArray`/`PyReadwriteArray` as `#[pyfunction]` parameter types.** Their extractor unwraps the borrow check and panics, raising `PanicException`, a `BaseException`. Take `Bound<PyArray…>` and call `try_readonly()`/`try_readwrite()`, mapping failure to `BufferError`. This becomes essential once kernels take `out=` arguments, because `f(x, out=x)` would otherwise panic.
- **Python converts types and shapes; Rust validates values.** Shared helpers live in `python/scatrs/_args.py`.
- **Broadcasting in multi-input kernels.** Don't copy the `np.require(C)` + `as_slice` pattern, because it materialises broadcast copies. Use `np.broadcast_shapes` plus `PyReadonlyArrayDyn::as_array()`, which handles strides including zero strides. rust-numpy's safe `as_array()` doesn't check alignment (only `as_slice` does), so the extension must check `is_aligned()` itself.
- **Thread safety.** With the GIL released, another thread can mutate the input arrays. Document this rather than copying: copying doesn't close the hole, and the spec requires zero-copy. Pass a fresh view to the extension, because `resize(refcheck=False)` on the borrowed object can abort through rust-numpy's borrow tracking.
- **Per-call overhead,** M4 Pro, from the Task 11 review: the extension costs about 180 ns per call and the wrapper about 0.84 µs after the Task 11 fixes.

## Platform notes
- On aarch64, `f64::max`/`f64::min` return NaN for a signaling NaN, contrary to Rust's documentation. `DoubleDouble` follows the documented rule. No DD operation produces a signaling NaN.

## Spec amendments and decisions to carry into Phase 1 (from the final phase 0 review)
- **Validated-domain rule (spec §4.5), controller decision.**
  - The `OutsideValidatedDomain` / `allow_unvalidated` rule applies to the **scattering outputs**: Mie, layered spheres and T-matrix, whose validated domains come from the accuracy maps (§8.2).
  - Special functions (`gauss_legendre`, `wigner_d`) are building blocks. They document their tested ranges in their rustdoc rather than refusing inputs.
  - Phase 1 designs the `allow_unvalidated` mechanism together with the first accuracy map, and its exit criterion ("declared validated domain enforced in CI") applies to the Mie outputs.
- **Spec §9.3 lags the implementation:**
  - the exception hierarchy also has `InvalidInputError(ScatrsError, ValueError)`;
  - zero-copy input now uses `Bound<PyArray…>` + `try_readonly()`, not `PyReadonlyArray` parameters (see "Python binding patterns" above).
- **Other documented deviations from the spec or plan:**
  - Reinsch difference forms in Wigner;
  - the 2¹⁷ GL cap;
  - the 2²⁸ cap on Python `wigner_d` output;
  - strict `is_zero` for `Dual`;
  - Apple's fused sincos;
  - DD trig is NaN beyond 2⁵⁰;
  - reference format v2 (with `lo2`).

## Phase 1 plan must include (from the final phase 0 review)
- **Before the Mie kernels:**
  - scatrs-owned complex helpers validated against Arb (Smith/Baudin division, Kahan csqrt, ln near 1, stable hypot);
  - a green first CI run, with Windows libm results checked.
- **Infrastructure deferred from phase 0:**
  - rayon with a caller-supplied thread pool;
  - criterion and instruction-count benchmarks, plus the CI gate (§12.1);
  - cargo-fuzz targets;
  - Riccati–Bessel and log-derivative functions;
  - the `allow_unvalidated` mechanism;
  - a Mie Arb generator plus the series-tail bound needed for "certified" labels;
  - testing musllinux wheels.
- **Data budget:**
  - phase 0 uses 7.8 of 20 MB (`dd_arith.jsonl` alone is 3.3 MB);
  - add a size test in `reference/tests`;
  - consider trimming `dd_arith`.
- **Docs and rustdoc:**
  - a `RUSTDOCFLAGS=-D warnings cargo doc -p scatrs --no-deps` CI step;
  - `# Examples` doctests for `gauss_legendre`, `wigner_d`, `DoubleDouble` and `Dual`;
  - fix the broken intra-doc link at `dd/traits.rs:2`, which only shows with `--document-private-items`.
- **Docs-consistency test:** extend it to check manifest-driven record counts and data size, MSRV 1.85, Python 3.11, numpy 1.26, the GL timing figures, and the Wigner ranges and costs.
- **API shape before publishing the crate:**
  - `GaussLegendre<T>` has public fields and no `#[non_exhaustive]`;
  - offer caller-supplied-buffer variants (§4.5);
  - settle both together with the rule-caching design.
- **Cross-platform golden digests** of GL rule bits (f64 and DD, a few n) and some Wigner outputs, so 3-OS CI verifies the claimed bitwise identity.
- **Packaging nits:**
  - declare `packaging` in the `test` extra;
  - the `xarray` extra is unused until the high-level API arrives.

## Phase 1a (homogeneous-sphere Mie): decisions and deferrals

Resolved from the lists above: the complex reciprocal (Smith; complex division, `csqrt` and
`ln` near 1 wait for a consumer), Riccati–Bessel and log-derivative functions (inside
`scatrs::mie`), the `allow_unvalidated` mechanism with the first accuracy map, the Mie Arb
generator with a proven tail bound, the data-budget test, rustdoc in CI, the four doctests,
the `dd/traits.rs` link, the docs-consistency extensions and `packaging` in the `test` extra.

Decisions (spec `docs/specs/2026-10-13-phase-1-mie-design.md`, section 12):
- Truncation ⌈x + 6x^(1/3) + 3⌉ (f64) and ⌈x + 12x^(1/3) + 6⌉ (DoubleDouble), measured.
- Dₙ(mx) by a contracting downward recurrence: no Lentz continued fraction, no `NotConverged`.
- Q_ext = Q_sca + Q_abs, with Q_abs from the per-term identity.
- Tolerance C·(κ′ + cond)·2⁻⁵², enforced by `mie_reference.rs` and `mie_accuracy_map.rs`.
- The reference generator avoids Arb's Bessel functions and uses disk arithmetic.

Phase 1b must include (besides spec section 9):
- **Performance.** The Jacobian costs 1.8× the values at large x: profile the per-order
  derivative work (three complex reciprocals per order in the ratio region). Measure Mie on
  x86_64 with and without FMA (the f64 kernel uses no `mul_add`, but `DoubleDouble` maps do).
- **Derivative series tail bound.** Certify the series of derivatives, for example with
  Cauchy's estimate in m (a disk of m in the tail bound of Proposition 3).
- **Larger certified cases.** Series records with |m|x > 6·10⁴ are high-precision (9 of 306);
  a faster generator (fewer Python-level operations per order) would certify them.
- **Accuracy-map cost.** The map takes about 200 CPU-seconds in the test profile; if CI time
  matters, sample derivative conditions on every second point.
- **∂Q_abs/∂x and ∂Q_abs/∂(Re m)** of weakly absorbing spheres are accurate only relative to
  Q_ext (relative error up to 4·10⁻⁹ at k = 10⁻⁸ near a resonance): investigate a formulation
  accurate relative to Q_abs.
- **First real CI run.** The Mie accuracy map and reference tests run on Windows for the first
  time there; libm sin/cos differences change last digits only, but check the printed worst
  constants.
