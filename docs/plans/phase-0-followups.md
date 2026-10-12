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
