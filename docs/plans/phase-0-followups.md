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
    1. **Drop the polish step.** The remaining error after |Δx| ≤ 2⁻⁹⁰ is ≤ 2⁻¹⁴² at 2²⁰. Replace the final Pₙ′ evaluation with the ODE correction P′(x−δ) ≈ P′(x)(1 − 2xδ/(1−x²)), stopping at |Δx| ≤ 2⁻⁶⁰/n. This needs 2.0–2.2× fewer evaluations, with f64 output bit-identical on 31 random n ≤ 6000.
    2. **Interleave K roots in lockstep.** K = 2 gives 1.9×; K = 4 or 8 gives 2.7× (saturated). Results are bitwise identical per root.
    - Combined, about 5×: n = 4·10⁴ goes from about 45 s to about 8 s.
  - **Real fix: Bogaert (2014),** an iteration-free O(n) method that works in θ. It is about ε-accurate and also removes the DD weight loss near ±1, about log₂(n²/10) bits. Cross-check it against the DD rule and Arb.
  - **Caching (Phase 1 design note):** a rule exact to degree D is exact for every lower degree, so an ensemble needs only the rule at its largest n_q. Round n_q up to buckets if caching. A hidden global cache conflicts with §4.5, so use a cache owned by the caller or an `Arc`.
- **CI for `#[ignore]` tests.** These tests (e.g. GL at n = 20000) must run on a schedule in Task 13, or the claims they back are never re-checked.
- **x86_64 builds without FMA.**
  - **Problem:** `f64::mul_add` becomes a libm call, which makes every DD operation about 10× slower in default wheels.
  - **Fix:** select Dekker's split `two_prod` when `cfg(not(target_feature = "fma"))`, and consider shipping x86-64-v3 wheels as well. Decide this before Task 13's wheels are published. A non-FMA build also has to change `mul_dd` and `mul_f64`, which use FMA in their non-exact steps. Those operations then need the non-FMA error bounds from Joldes, Muller & Popescu (2017), not just a different `two_prod`. FMA and non-FMA builds will then give bitwise different results. The spec allows this, since it only requires determinism across thread counts.
- **`mul_pow2` zero inputs:** these deliberately take the cold `renormalize_tiny` path, which turns a −0 low word into +0. Do not optimise this away.
- **`DoubleDouble::rem`** allocates a `Vec` on every call. The expansion length bound is unproven; 6 components is the most observed. Switch to a stack buffer if a bound can be proved.

## Correctness and robustness (low priority, outside the documented domain)
- **DD `sin`/`cos`/`tan` return NaN for |x| > 2⁵⁰.**
  - **Decision (controller, Task 5):** this is a documented domain limit, like `sqrt(-1)`, so it is not a silent failure. Kernels never pass such arguments and must validate their own domains.
  - **If ever needed:** Payne–Hanek reduction, with a table of about 1200 bits of 2/π.
- **DD trig accuracy near nonzero multiples of π/2.** The contract only promises absolute accuracy there, but the actual bounds are better. These were derived in the Task 5 follow-up, checked on 30k Arb inputs, and are now documented in `trig.rs`:
  - The absolute error floor is |k|·2⁻¹⁶¹ for f64 arguments and |k|·2⁻¹⁵⁸ for DD arguments.
  - Relative error stays ≤ 2⁻¹⁰⁰ whenever |x − kπ/2| ≥ |k|·2⁻⁶⁰ (f64) or |k|·2⁻⁵⁷ (DD).
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

## Packaging (Task 13)
- Ship `LICENSE` inside the `scatrs` crate package, for example by copying it to `scatrs/LICENSE`.
- Use absolute GitHub URLs in `README.md`, since crates.io breaks relative links.
- `scatrs/tests/common/mod.rs` reads `../reference/data`. Exclude the reference-backed tests from the published crate, or make them skip when the data is absent.

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
  - `log` with base ±0 returns a NaN base derivative.
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
- **Cross-platform CI is essential.** The Task 8 review found a Linux-only failure: macOS passed only because Apple's `pow` happened to cancel a test-reference error. Run the full test suite on Linux and Windows in CI before trusting libm-dependent bounds (Task 13).
- **`Dual<Dd>::log` cost.** The value must come bitwise from `T::log`, so it computes DD `ln` twice, about 2× DD `log`.

## Platform notes
- On aarch64, `f64::max`/`f64::min` return NaN for a signaling NaN, contrary to Rust's documentation. `DoubleDouble` follows the documented rule. No DD operation produces a signaling NaN.
