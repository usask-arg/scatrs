# Reference suite

The tests take their ground truth from an independent generator in `reference/`. It computes
every value with [Arb](https://arblib.org) ball arithmetic through
[python-flint](https://python-flint.readthedocs.io), and never imports scatrs. The values are
committed as JSON Lines files in `reference/data/`. The Rust tests read those files, so
running the tests does not need Arb.

## Record format

<!-- synced with reference/scatrs_reference/records.py docstrings -->

Each line of a data file is one record, a JSON object with these keys:

| Key | Contents |
|---|---|
| `generator` | the generator that wrote it (also the file name) |
| `case` | a name, unique within the file |
| `label` | `certified` or `high-precision` (see below) |
| `inputs` | the arguments, as exact doubles, integers or strings |
| `outputs` | named sets of values, each as the parallel arrays below |

The manifest calls this layout format version 2. Every stored value $v$ is represented in
three ways:

- **`hi`, `lo`, `lo2`: three doubles.** Their exact sum approximates $v$ to about 160 bits.
  `hi` is the double nearest to the midpoint of the Arb ball, and `hi == fl(hi + lo)`, so
  `(hi, lo)` is a normalised double-double. `lo2` is the double nearest to the rest.
- **`dec`: a decimal string.** $v$ as Arb prints it, with up to 25 significant digits, or fewer
  when the ball is too wide to justify 25. It is for people to read; the tests use the
  doubles.
- **`err`: an error bound.** A rigorous upper bound on $\lvert v - (\text{hi} + \text{lo} +
  \text{lo2})\rvert$, which includes the radius of the Arb ball.

The third word lets the tests measure the error of a double-double result almost exactly.
`RefValues::dd_error` in `scatrs/tests/common/mod.rs` returns
$\lvert \text{got} - (\text{hi} + \text{lo} + \text{lo2})\rvert + \text{err}$. It forms the
difference with error-free transformations and compensated summation, so the result is within
about $2^{-150}\lvert v\rvert$ of a rigorous bound on $\lvert \text{got} - v\rvert$. That is
negligible against double-double errors of order $2^{-106}$.

## Labels

A record is **certified** when every source of error is rigorously bounded, including any
truncation of a series. It is **high-precision** when some error, such as a series tail, is
not yet rigorously bounded. Labels are never overstated.

All 358 records in phase 0 are certified:

- the arithmetic, constants and decimal values are exact rationals or Arb balls;
- the elementary and complex functions are Arb's rigorous implementations;
- the Gauss–Legendre roots and weights come from Arb's certified Legendre root finder;
- the Wigner values come from the finite explicit sum, or, at very large degrees, from Arb's
  Legendre functions.

The `high-precision` label is for later phases: for example, values of a scattering series
whose truncation does not yet have a rigorous tail bound.

## Files

<!-- synced with reference/data/manifest.json and the generator docstrings -->

`reference/data/manifest.json` lists every data file. Each entry gives:

- the python-flint and FLINT versions that produced it (0.9.0 and 3.6.0 for every file now);
- the generator's working precision in bits;
- the number of records;
- the SHA-256 of the file.

| File | Records | Contents |
|---|---|---|
| `constants.jsonl` | 62 | the `DoubleDouble` constants: the 19 `FloatConst` values ($\pi$, $\ln 2$, …) and $3\pi/4$, the tables $1/k!$ for $k \le 30$ and $10^{2^i}$ for $i \le 8$, and the third words of $\pi/2$ and $\ln 2$ used by the argument reductions |
| `dd_arith.jsonl` | 9 | true results of double-double `+`, `−`, `×`, `÷`, the `f64` forms of `+`, `×` and `÷`, division by huge divisors, and the remainder `%`: 2000 random and adversarial inputs each |
| `dd_complex.jsonl` | 4 | `exp`, `ln`, `sqrt` and division of `Complex<DoubleDouble>`, at 200 random arguments each, plus arguments on num-complex's branches and next to the branch cut |
| `dd_elementary.jsonl` | 38 | 19 functions (`sqrt`, `exp`, `exp_m1`, `ln`, `ln_1p`, `log2`, `log10`, `cbrt`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp2`, `hypot`, `powf`, `powi` and `log(x, base)`), each with 200 random inputs and a record of edge cases |
| `dd_parse.jsonl` | 2 | exact values of 300 random decimal literals and of hand-picked edge cases |
| `dd_trig.jsonl` | 14 | `sin`, `cos`, `tan`, `atan`, `asin`, `acos` and `atan2`, each with 200 random inputs and a record of edge cases |
| `gauss_legendre.jsonl` | 17 | see below |
| `wigner_d.jsonl` | 212 | see below |

The edge cases sit at the edges of each function's domain and of its algorithm's branches:

- overflow and underflow thresholds, and subnormal inputs;
- arguments next to 1;
- the switch points between series, reductions and fast and wide-range paths;
- for `sin` and `cos`, the double closest to a multiple of $\pi/2$ in every binade below
  $2^{50}$ (for `tan`, below $2^{21}$).

**`gauss_legendre.jsonl`** holds 17 rules, nodes and weights:

- full rules for $n$ = 1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 128, 257, 500 and 1000;
- partial rules for $n$ = 5000 and 20 000, holding the 8 nodes nearest each of $\pm1$ and the 4
  nearest 0;
- the same 20 nodes of the largest rule scatrs accepts, $n = 2^{17}$.

**`wigner_d.jsonl`** holds 212 records:

| Records | Contents |
|---|---|
| 170 | every degree $l \le 100$ of 17 pairs $(m, n)$ at 10 values of $x$, including $\pm1$ and doubles a few ulps from them |
| 1 | $(0, 2)$ at $x = 0.3$, every degree up to 300 |
| 25 | sampled degrees up to $l = 2000$, for $(0, 0)$, $(2, \pm2)$, $(0, 2)$, $(50, -37)$, $(-37, -50)$ and $(7, 150)$ |
| 2 | sampled degrees up to $l = 4000$, next to the poles |
| 7 | five degrees from $10^5$ to $2.5\cdot10^5$, for $(0, 0)$ and $(0, 2)$, from Arb's Legendre functions |
| 7 | start values below the normal double range, with later degrees that are normal doubles |

The committed data take about 7.8 MB, within the 20 MB budget.

## How the tests use it

- **Tolerances and output.** Each accuracy test compares scatrs with every value of the
  relevant records and asserts the documented bound. It also prints its worst measured error
  (run `cargo test -- --nocapture` to see it).
- **Loading.** The loader rejects malformed records, duplicate case names and output arrays of
  unequal length.
- **Slow cases.** Records that take too long to recompute are checked by ignored tests, run on
  demand with `cargo test -p scatrs --release -- --ignored`. Today that is the 20 000-point
  Gauss–Legendre rule.
- **The largest rule.** The rule with $2^{17}$ points is checked root by root, by a unit test.

**Tolerance policy.** A test tolerance is never loosened without a written justification in the
test.

## Regenerating the data

```bash
uv run --directory reference python -m scatrs_reference generate             # every file
uv run --directory reference python -m scatrs_reference generate --only wigner_d
uv run --directory reference python -m pytest
```

python-flint is pinned exactly (0.9.0), so that regeneration can be checked byte for byte
against the SHA-256 in the manifest. The generator's own tests check this for
`constants.jsonl`, and they check every committed file against its manifest hash. A scheduled
CI job that regenerates every file and fails on any difference is planned with the CI
workflows.

The manifest's precision for `wigner_d.jsonl`, 800 bits, is only the starting precision. The
generator doubles it as needed, to at most 51 200 bits, until each ball is narrow enough: a
radius of at most $2^{-200}$ relative or $2^{-1300}$ absolute. The absolute criterion is meant
for values that are zero.
