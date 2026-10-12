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
- `ldexp` can double-round subnormal results, i.e. results below 2⁻⁹⁶⁹. This is outside the documented full-precision domain.

## Packaging (Task 13)
- Ship `LICENSE` inside the `scatrs` crate package, for example by copying it to `scatrs/LICENSE`.
- Use absolute GitHub URLs in `README.md`, since crates.io breaks relative links.
- `scatrs/tests/common/mod.rs` reads `../reference/data`. Exclude the reference-backed tests from the published crate, or make them skip when the data is absent.
