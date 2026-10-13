# Scalar types (Rust)

Every numerical kernel in the Rust crate is generic over the trait `scatrs::Real`. One kernel
therefore gives:

- values, with `f64`;
- extended-precision values, with `DoubleDouble`;
- values with derivatives, with `Dual<f64, N>`;
- extended-precision derivatives, with `Dual<DoubleDouble, N>`;
- second derivatives, with nested duals `Dual<Dual<T, M>, N>`.

The example `scatrs/examples/scalar_types.rs` (run it with
`cargo run --example scalar_types`) evaluates one generic kernel, $\int_{-1}^{1} e^{ax}\,dx$
by Gauss–Legendre quadrature, with each type:

```{literalinclude} ../../scatrs/examples/scalar_types.rs
:language: rust
```

`Dual::variables` seeds independent variables: each gets derivative 1 in its own direction. For
a second derivative, nest two duals and seed both levels in the same direction. The value then
sits at `.re().re()`, the first derivative at both `.re().eps()[0]` and `.eps()[0].re()`, and
the second derivative at `.eps()[0].eps()[0]`.

This page summarises the contracts that the rustdoc states for each type. Until the crate is
published on [docs.rs](https://docs.rs/scatrs) (it is not yet), read the full rustdoc with
`cargo doc -p scatrs --no-deps --open`.

Throughout, $u = 2^{-53}$ (the unit roundoff of `f64`) and $u^2 = 2^{-106}$.

## The `Real` trait

<!-- synced with scatrs/src/scalar/mod.rs rustdoc -->

`Real` requires `num_traits::Float` and `FloatConst`, `Debug + Send + Sync + 'static`, the
compound assignment operators, and arithmetic with an `f64` on the right (`x * 2.0`, not
`2.0 * x`). Because it requires `Float`, `num_complex::Complex<T>` works for every `T: Real`.
It adds three methods:

| Method | Meaning |
|---|---|
| `from_f64(x)` | converts an `f64` exactly |
| `from_f64_pair(hi, lo)` | the value of the unevaluated sum `hi + lo`: exact for `DoubleDouble`, rounded once for `f64` |
| `approx_f64()` | the nearest `f64` to the value, ignoring any derivatives |

(generic-code-notes)=
### Notes for generic code

`DoubleDouble` and `Dual` follow `f64` closely, but generic code must not assume more than
`Float` promises:

- **`mul_add` is fused for `f64` but not for `DoubleDouble`.** The double-double version
  rounds the product before adding. Error-free transformations must not rely on it: Kahan's
  idiom `e = b.mul_add(c, -(b * c))` gives exactly 0 for `DoubleDouble`, not the rounding error
  of `b * c`.
- **`epsilon()` is a unit, not an error bound.** It is $2^{-52}$ for `f64` and $2^{-104}$ for
  `DoubleDouble`. One `DoubleDouble` division can err by up to about 2.5·`epsilon()`.
- **`NumCast::from` keeps only the leading word.** Converting from another `Real` this way
  goes through `to_f64`. Use the value itself, or `from_f64_pair`, to keep full precision.
- **`Dual` compares values only, but tests zero strictly.** `==` and `<` compare values only,
  while `is_zero`, `is_one`, `is_nan` and `is_finite` also inspect the derivatives. Branch on
  comparisons of values, as in `x == T::zero()`, where the derivatives should not matter.
- **`Dual` takes `sin` and `cos` from `T::sin_cos`.** They equal `T::sin` and `T::cos`
  wherever `sin_cos` does, which is always the case for `DoubleDouble`. For `f64` on Apple
  targets they can differ by an ulp; see [below](dual-sin-cos).

### Writing a generic kernel

A checklist, following from the notes above and the rules below:

- Write constants as `T::from_f64(0.5)`, or with `FloatConst` (`T::PI()`), and combine `T`
  with `f64` with the `f64` on the right. A constant that needs more than a double takes
  `T::from_f64_pair(hi, lo)`.
- Branch on comparisons of values (`x < T::zero()`), not on `is_zero` or `is_one`, unless the
  derivatives should decide.
- Do not round-trip through `f64` (`approx_f64`, `to_f64`, `NumCast`) inside the computation.
  That drops the double-double precision and the derivatives. Use `approx_f64` only for
  decisions, such as choosing a branch or a number of terms.
- Do not use `mul_add` as an error-free fused multiply-add.
- Write `abs` or `hypot`, not the square root of a square (see [strong zeros](strong-zeros)).
- Check inputs and return `Error::InvalidInput` rather than relying on NaN. Angles, for
  example, must stay within $\lvert x\rvert \le 2^{50}$ for the double-double trigonometric
  functions.
- Test with `f64`, `DoubleDouble` and `Dual`. The scatrs tests compare `Dual` derivatives with
  double-double finite differences, and `DoubleDouble` values with the Arb reference.

(double-double)=
## `DoubleDouble`

A `DoubleDouble` is the unevaluated sum `hi + lo` of two `f64`s, normalised so that
`hi == fl(hi + lo)`. That gives about 106 significant bits, roughly 32 decimal digits.

- **Algorithms.** The basic operations are those of Joldes, Muller & Popescu, "Tight and
  rigorous error bounds for basic building blocks of double-word arithmetic", *ACM TOMS*
  44(2), 2017. They rely on a correctly rounded fused multiply-add, which `f64::mul_add`
  provides on every platform, in software where the hardware lacks it.
- **Range.** Full relative precision needs magnitudes of at least $2^{-969}$
  ($\approx 2\cdot10^{-292}$), which is `min_positive_value()`. Below that the low word is
  subnormal and precision degrades gradually. Within about an ulp of `f64::MAX` a result can
  overflow early.
- **Special values.** Non-finite results are `(hi, 0)`, so infinities do not turn into NaN. A
  zero result is `(±0, +0)`, with the IEEE 754 sign of the `f64` operation on the leading
  words. Special values and signed zeros otherwise follow `f64`.
- **Constants.** The `FloatConst` values, and the internal tables of the elementary
  functions, are the certified splits stored in the reference file `constants.jsonl`.

### Accuracy

<!-- synced with scatrs/src/scalar/dd/mod.rs, elementary.rs, trig.rs and parse.rs rustdoc -->

These are the bounds stated in the rustdoc of each operation. They hold for results of
magnitude at least $2^{-969}$. Unless stated otherwise they are relative, and $2^{-100}$ is
$64u^2$.

| Operations | Bound | Basis |
|---|---|---|
| `a + b`, `a - b` | $3u^2$ | proved (Joldes, Muller & Popescu), to leading order |
| `a + y`, `a - y`, `a * y`, with `y` an `f64` | $2u^2$ | proved |
| `a * b` | $4u^2$ | proved |
| `a / b` | $10u^2$ | proved |
| `a / y`, with `y` an `f64` | $3u^2$ | proved |
| `a % b` | within $2u^2\,\lvert b\rvert$ absolute | exact remainder, rounded once |
| `mul_add(x, a, b)` (not fused) | $4u^2\lvert x a\rvert + 3u^2\lvert x a + b\rvert$ absolute | from the bounds above |
| `to_degrees`, `to_radians` | $7u^2$, $8u^2$ | from the bounds above |
| `from_decimal_str`, `str::parse` | $2^{-100}$ for results in $[2^{-969}, \text{MAX}]$; exact for integers below $2^{106}$ | first-order error analysis |
| `sqrt`, `cbrt`, `exp`, `exp_m1`, `exp2`, `ln`, `ln_1p`, `log2`, `log10`, `hypot`, `tanh`, `asinh`, `acosh`, `atanh` | $2^{-100}$ | tested against Arb |
| `sinh`, `cosh`, `log(x, base)` | $2^{-99}$ | tested against Arb |
| `powf(x, y)` | $2^{-99}\max(1, \lvert y \ln\lvert x\rvert\rvert)$ | tested against Arb |
| `powi(x, n)` | $16(\lvert n\rvert + 1)\,u^2$ | tested against Arb |
| `sin`, `cos`, `sin_cos` | $2^{-100}$ for $\lvert x\rvert \le \pi/4$; $2^{-100}$ absolute for $\lvert x\rvert \le 2^{50}$ | tested against Arb |
| `tan` | $2^{-99}$ for $\lvert x\rvert \le \pi/4$; $2^{-99}(1 + \lvert\tan x\rvert)/\lvert\cos x\rvert$ absolute | tested against Arb |
| `atan`, `atan2`, `asin`, `acos` | $2^{-99}$ | tested against Arb |

What the basis column means:

- **Proved.** The bound is the leading term of a theorem of the 2017 paper; terms of order
  $u^3$ are dropped. The tests also check it against Arb on 2000 random and adversarial inputs
  per operation.
- **Tested against Arb.** The bound holds on every reference input, random and hand-picked at
  the edges of each algorithm's branches. It is not proved for all inputs.

The measured errors are well inside the bounds. For the arithmetic, the worst measured errors
are:

- $2.0u^2$ for `+`, $1.4u^2$ for `-`, $3.5u^2$ for `*` and $4.3u^2$ for `/`;
- $1.0u^2$ for `+ f64` and `* f64`, and $2.0u^2$ for `/ f64`.

For the functions tested against Arb, the output of the accuracy tests
(`cargo test -p scatrs --release -- --nocapture`, in `tests/dd_elementary.rs` and
`tests/dd_trig.rs`) gave these worst errors when these pages were written:

- at most 0.11 of the bounds in the table, the largest being 0.102 for the `powi` edge cases
  (0.047 for `tan`, the largest of the trigonometric functions);
- 0.43 and 0.54 of the tighter bound with an absolute floor for `sin` and `cos` next to
  multiples of $\pi/2$ (described below), at the edge cases.

Further details from the rustdoc:

- **Domain of `exp`.** `exp` is validated for $x \in [-670, 709.78]$. It overflows and
  underflows as `f64::exp` does.
- **Domain of the trigonometric functions.** `sin`, `cos` and `tan` return NaN for
  $\lvert x\rvert > 2^{50}$. This is a documented domain limit, like `sqrt(-1)`, not a silent
  failure.
- **Absolute floor near multiples of $\pi/2$.** Next to a nonzero multiple $k\pi/2$ the argument
  reduction leaves an absolute error floor. It is at most $\lvert k\rvert\cdot2^{-161}$ for
  `f64` arguments and $\lvert k\rvert\cdot2^{-158}$ for double-double ones. The tests check
  that the error is within $2^{-100}\lvert f\rvert$ plus that floor, so the relative error is
  at most $1.5\cdot2^{-100}$ wherever $\lvert x - k\pi/2\rvert \ge \lvert k\rvert\cdot2^{-60}$
  (`f64` arguments) or $\lvert k\rvert\cdot2^{-57}$ (double-double arguments). Some doubles lie
  closer to such multiples than that. At three of them the measured relative errors are
  $2^{-100}$ at $x = 45.553093477052$ (the closest of all below $2^{50}$), $2^{-81}$ at
  $x = 14461176.67027838$ and $2^{-61}$ at $x = 410195257422896.8$.
- **`sin_cos`.** It is bitwise equal to `(sin, cos)`.
- **Ranges.** `atan2` clamps its result to $[-\pi, \pi]$, using the double-double $\pi$.
- **Seeds from the platform.** `ln`, `cbrt` and `atan2` start from the platform's `f64::ln`,
  `f64::cbrt` and `f64::atan2`. The results meet the bounds above whatever the last bits of
  those starting values. Their own last bits, though, may differ between platforms.

### Use as `num_traits::Float`

<!-- synced with scatrs/src/scalar/dd/mod.rs rustdoc -->

`DoubleDouble` implements `Float`, `FloatConst`, `NumCast`, `ToPrimitive` and `Real`, so
generic kernels and `Complex<DoubleDouble>` work. Arithmetic and the elementary functions are
the inherent methods above, with their accuracy. Some methods act on the whole double-double:

- rounding: `floor`, `ceil`, `round`, `trunc` and `fract`;
- `abs`, `signum` and `copysign`;
- `max`, `min`, `clamp` and `abs_sub`.

Classification (`classify`, `is_normal`, …), the sign tests and `integer_decode` describe the
leading word. Where `DoubleDouble` differs from `f64`:

- `epsilon()` is $2^{-104}$, a unit rather than an error bound.
- `min_positive_value()` is $2^{-969}$, the smallest magnitude with full precision. `classify`
  and `is_normal` still call values in $[2^{-1022}, 2^{-969})$ normal.
- `mul_add` is not fused (see the table).
- `max` and `min` ignore a NaN argument, even a signaling one. For two zeros they follow IEEE
  754-2019 maximum and minimum: $\max(-0, +0) = +0$ and $\min(-0, +0) = -0$, in either order.
- `clamp` returns NaN instead of panicking when `min > max` or a bound is NaN.
- Conversion is exact where it can be:
  - `ToPrimitive` truncates towards zero exactly, and returns `None` outside the target range.
  - `NumCast::from` is exact for `f32`, `f64` and integers below $2^{106}$ in magnitude.
- Parsing differs in its handling of whitespace:
  - `DoubleDouble::from_decimal_str` ignores surrounding whitespace.
  - `str::parse` accepts exactly the strings that `f64` accepts, so it does not.

### Speed

`f64::mul_add` is one instruction where FMA is enabled. On x86_64 builds without the `fma`
target feature, which is the default, it is a much slower software call; see
{doc}`../installation`. Indicative timings measured during code review (not committed):
`exp` takes about 200 ns and `sin` about 170 ns, against 2.3 ns for an `f64` `sin`. Faster
table-driven versions are planned.

(dual)=
## `Dual<T, N>`

<!-- synced with scatrs/src/scalar/dual/mod.rs rustdoc -->

A `Dual<T, N>` is a value with `N` first-order directional derivatives,
$\text{re} + \sum_i \text{eps}_i\,\varepsilon_i$. `Dual::variables([...])` seeds independent
variables, and `Dual::constant(v)` makes a value with zero derivatives. `re()` and `eps()`
read the parts.

The value of every operation is `T`'s operation on the values, so rounding, special values and
signed zeros follow `T` bit for bit. For example, `(a / b).re() == a.re() / b.re()`. The one
exception is `sin` and `cos`; see [below](dual-sin-cos). The derivatives come from the chain
rule, evaluated in `T`.

(dual-zero-tests)=
### Comparisons and zero tests

`==`, `<` and the other comparisons look only at the value, so a `Dual` orders like its value.

`is_zero` and `is_one` are **strict**: they also require every derivative to be zero. Generic
code uses them as shortcuts, and a value-only test would silently drop derivatives there.
num-complex 0.4.6, for example, does two things:

- `Complex::sqrt` and `cbrt` return early when a component `is_zero()`.
- `powf` and `powc` return `one()` when the exponent `is_zero()`.

Take a refractive index $m = n + 0i$ with $\partial/\partial(\operatorname{Im} m)$ seeded. A
value-only `is_zero` would make `m.sqrt()` lose that derivative. ForwardDiff.jl made the same
choice (v1.0, PR #481).

`is_nan` and `is_finite` also inspect the derivatives, so a NaN derivative is never silent.
The other tests describe the value alone:

- `is_infinite`, `is_normal` and `is_subnormal`;
- `classify`, the sign tests and `integer_decode`.

A finite value with an infinite derivative is therefore neither NaN, nor infinite, nor finite.

In `Complex<Dual<T, N>>` the strict tests change some values. Where a component is zero in
value but carries a derivative, num-complex skips its shortcut and takes the general polar
path. Its derivatives are correct, but its values are not always bitwise those of
`Complex<T>`:

- `sqrt` on the imaginary axis differs by up to 2 ulps, and `cbrt` there always differs
  (within 2 ulps in the tests).
- `sqrt(-4 + 0i)` gives $\operatorname{Re} = 2\cos(\pi/2) \approx 1.2\cdot10^{-16}$ instead of
  0.
- `powf` with an exponent that is zero in value returns $1 - 0i$ when $\arg z < 0$, and NaN
  for a NaN base, instead of $1 + 0i$.
- `powc` with such an exponent returns NaN at $z = 0$ and $z = \infty$ instead of 1.
- `ToPrimitive` returns `None` when the imaginary part is zero in value but has a derivative.
- Since `is_finite` sees derivatives, `Complex::exp` with $\operatorname{Re} z = +\infty$
  takes its corner case when $\operatorname{Im} z$ has an infinite or NaN derivative. It then
  returns $(+\infty, \operatorname{Im} z)$.

(strong-zeros)=
### Strong zeros

A derivative component that is exactly zero is a *strong zero*: it contributes zero even when
multiplied by an infinite or NaN factor. So a direction on which no operand depends keeps
derivative 0 whatever the values are. `Dual::constant(2.0) * Dual::constant(f64::INFINITY)`
has zero derivatives, not NaN. A derivative never depends on `N` or on the other directions.

In arithmetic with finite operands, a derivative is NaN only in two cases:

- the value is NaN too (`0 / 0`, `x % 0`);
- two terms of its formula overflow with opposite signs ($\infty - \infty$).

Otherwise IEEE 754 rules apply: `x / 0` has derivative $x'/0 = \pm\infty$, for example, and a
nonzero derivative times an infinite value is infinite. The strong-zero rule applies to zero
derivatives only: an infinite or NaN derivative times a zero value is NaN, as in `T`. The sign
of a zero derivative is unspecified, and carries no information.

```{caution}
Strong zeros cannot tell a direction the operands do not depend on from one in which they are
stationary. For `p` seeded at 0, `p * p` rightly has derivative 0. So `sqrt(p * p)`, or
`(p * p).powf(0.25)`, gets derivative 0 at the singular point of the root, where $\lvert p\rvert$
has no derivative. The plain chain rule would give $0\cdot\infty = \text{NaN}$ there. Write
`p.abs()` rather than the root of a square if $\pm p'$ is wanted.
```

(dual-conventions)=
### Points where a function is not differentiable

At these points scatrs makes a fixed choice. For a nonzero derivative $x'$:

| Case | Derivative |
|---|---|
| `floor`, `ceil`, `round`, `trunc`, `signum` | 0, also at the jumps |
| `abs`, `copysign` | $-x'$ where the sign changes (for `abs`, where the sign bit is set, −0 included), else $x'$ |
| `max`, `min`, `clamp` | those of the operand whose value is returned; at a tie, the one whose sign the result has |
| `sqrt`, `cbrt`, `ln`, `log2`, `log10` at $\pm0$; `ln_1p` at −1; `asin`, `atanh` at ±1; `acosh` at 1 | $+\infty$, the one-sided limit |
| `acos` at ±1 | $-\infty$ |
| `hypot` at the origin | 0 (a subgradient, since `hypot` is convex) |
| `atan2` at the origin | NaN (the function jumps there) |
| `atan2` with an infinite argument | 0 |
| `log(x, b)` at $x = \pm0$ | the limit from the right in $x$, $1/(x\ln b) \to \pm\infty$ |
| `log(x, b)` at base $b = \pm0$ | $-\operatorname{sign}(\ln x)\cdot\infty$ in $b$; 0 for $x = 1$ |
| `powf`: $x^0$ | 0 in $x$ everywhere, also at $x = 0$, $\infty$ or NaN |
| `powf` at $x = 0$ | $y\,x^{y-1}$ is $+\infty$, 1 or 0 for $0 < y < 1$, $y = 1$ and $y > 1$ |
| `powf`, derivative in $y$ | 0 where $x^y = 0$ with $x \ge 0$; NaN for $x < 0$ |
| at $\pm\infty$ | the limits: 0 for `sqrt`, `ln`, `atan`, `tanh`, `asinh`, …; $\pm\infty$ for `exp`, `sinh`, `cosh`, … |
| outside a function's domain (`ln(-1)`, `asin(2)`, …), or a NaN argument | NaN, except that the piecewise-constant functions keep 0, and `abs`, `copysign`, `max`, `min` and `clamp` keep the derivatives of the operand they return |

- **`hypot` versus `atan2` at the origin.** Because of these two choices, `Complex::ln` at
  $z = 0$ gets derivative 0 in its real part, $\ln\lvert z\rvert$, silently. NaN in the
  imaginary part flags the point.
- **Derivatives that overflow.** The chain rule multiplies $x'$ by $f'(x)$ evaluated in `T`.
  Where $f'(x)$ overflows or underflows, the derivative is $\pm\infty$ or 0 even if the true
  value is representable. For example, `atan` has derivative 0 beyond
  $\lvert x\rvert \approx 1.3\cdot10^{154}$, where $1/(1 + x^2)$ underflows.
- **Nested duals.** The limits at the singular points of `sqrt`, `ln`, `log2` and `log10`
  carry inner derivatives. The other special cases return constants, with zero inner
  derivatives.

(dual-sin-cos)=
### `sin` and `cos`

`Dual::sin` and `Dual::cos` take their values and derivatives from one `T::sin_cos`, so that
value-only and derivative calls of a kernel agree. For `DoubleDouble`, `sin_cos` is bitwise
`(sin, cos)`.

For `f64` on Apple targets, LLVM fuses a `sin` and a `cos` of the same argument into one call
to `__sincos_stret`. Its sine can be an ulp off a lone `f64::sin`, for about 1 in 13 000
arguments (0.768039326588954 and −0.3820992815488752, for example). The `Dual` value is then
the one that any `f64` code computing both functions gets, such as `Complex::exp`. `f64` code
computing only `sin x` gets the lone sine, which `Dual::sin` may miss by an ulp. On glibc the
fused and separate values agree.

### Accuracy of derivatives

**Arithmetic.** Each derivative formula is evaluated in `T`, so it errs by a few units in the
last place of its largest term. Cancellation between the terms is not compensated. An
intermediate can also overflow although the derivative is finite. For `a / b` with
`a = 1e300`, `b = 10` and `b′ = 1e10`, for example, $q\,b'$ overflows and the derivative is
$-\infty$ where the true value is $-10^{308}$.

**Elementary functions.** The tests (`tests/dual_functions.rs`) measured each function over
its whole domain against high-precision derivatives:

- **`Dual<f64, N>`.** Within $5.1u$ of the exact derivative at the argument, with Apple's libm.
  The worst case is `tan`. The tests bound each derivative by the measured errors of the libm
  results it is computed from, times their amplification, plus $1u$ per rounding.
- **`Dual<DoubleDouble, N>`.** Within $2^{-96}$ relative. For powers whose `DoubleDouble`
  values are only that accurate, the bound is $2^{-96}\max(1, \lvert y\ln x\rvert)$. For
  $\partial/\partial x$ the bound uses $\lvert (y - 1)\ln x\rvert$ instead.
- **Small arguments.** Derivatives computed from arguments or intermediates below $2^{-969}$
  lose precision, as such `DoubleDouble` values do.

### Cost

Operations are inlined. The derivatives are fixed-length arrays, which LLVM unrolls; for
`N = 4` and `N = 8` it packs them into SIMD lanes. Multiplication, division and the chain rule
check the values for special cases first. These checks are branches, so unlike the same loop
over `f64`, a loop over many `Dual`s is not vectorised across elements. Division needs two
divisions: `a/b` for the value and `1/b` for the derivatives.

Indicative throughput of `Dual<f64, N>` relative to plain `f64`, measured during code review
on an Apple M4 (not committed):

| Function | Cost |
|---|---|
| `exp` | about 1.15× |
| `sin`, `cos` | 1.6–1.8× (one fused libm call) |
| `Complex::exp` | 1.2× ($N = 1$), 1.8× ($N = 4$) |
| `tanh`, $\lvert x\rvert > 0.55$ | about 2× |
| `sqrt` | about 4.4× (plain `f64` `sqrt` vectorises across elements) |

These figures will be replaced by benchmarks in phase 1.
