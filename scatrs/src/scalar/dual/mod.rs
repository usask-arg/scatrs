//! Forward-mode dual numbers carrying `N` first-order directional derivatives.

use core::array;
use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

use crate::scalar::Real;

mod traits;

/// A value with `N` first-order derivatives: `re + Σᵢ eps[i]·εᵢ`.
///
/// # Comparisons and zero tests
///
/// `==`, `<` and the other comparisons look only at the value `re`, so a `Dual` orders like its
/// value. [`Zero::is_zero`] and [`One::is_one`] are strict: they also require every derivative
/// to be zero. Generic code uses them as shortcuts, where a value-only test would silently drop
/// derivatives: num-complex 0.4.6 returns early from `Complex::sqrt` and `cbrt` when a component
/// `is_zero()`, and returns `one()` from `powf`/`powc` when the exponent `is_zero()`. So for a
/// refractive index `m = n + 0i` with ∂/∂(Im m) seeded, `m.sqrt()` would lose that derivative.
/// ForwardDiff.jl made the same choice (v1.0, PR #481). `Float::is_nan` and `Float::is_finite`
/// also inspect the derivatives, so a NaN derivative is never silent; `is_infinite`,
/// `is_normal`, `is_subnormal`, `classify`, the sign tests and `integer_decode` describe the
/// value alone. A finite value with an infinite derivative is therefore neither NaN, infinite
/// nor finite.
///
/// In `num_complex::Complex<Dual<T, N>>`, strict zero tests change some values. Where a
/// component is zero in value but carries a derivative, num-complex skips its shortcut and takes
/// the general polar path, whose values are not always bitwise those of `Complex<T>` (the
/// derivatives on these paths are correct):
///
/// * `sqrt` on the imaginary axis differs by up to 2 ulps, and `cbrt` there always differs
///   (within 2 ulps in the tests);
/// * `sqrt(−4 + 0i)` gives `Re = 2·cos(π/2) ≈ 1.2e-16` instead of 0;
/// * `powf` with an exponent that is zero in value returns `1 − 0i` when arg z < 0, and NaN for
///   a NaN base, instead of `1 + 0i`;
/// * `powc` with such an exponent returns NaN at z = 0 and z = ∞ instead of 1;
/// * `ToPrimitive` returns `None` when the imaginary part is zero in value but has a
///   derivative.
///
/// And since `is_finite` sees derivatives, `Complex::exp` takes its corner case for `Re z = +∞`
/// when `Im z` has an infinite or NaN derivative, returning `(+∞, Im z)` instead of
/// `(∞·cos(Im z), ∞·sin(Im z))`.
///
/// # Arithmetic
///
/// The value of every operation is `T`'s operation on the values, so its rounding, special
/// values and signed zeros follow `T` bit for bit: `(a / b).re() == a.re() / b.re()`. With `a′`
/// and `b′` the derivatives (componentwise), `y` an `f64`, and `q` and `r` the computed quotient
/// and remainder, the derivatives are:
///
/// | Operation | Derivatives |
/// |---|---|
/// | `a + b`, `a - b`, `-a` | `a′ + b′`, `a′ − b′`, `−a′` |
/// | `a * b` | `a′·b + a·b′` |
/// | `a / b` | `(a′ − q·b′)·(1/b)`, or `(a′ − q·b′)/b` in the special cases below |
/// | `a % b` | `a′ − t·b′`, where `t` is the integer with `a = t·b + r` |
/// | `a + y`, `a - y` | `a′` |
/// | `a * y`, `a / y` | `a′·y`, `a′/y` |
///
/// Each formula is evaluated in `T`, so a derivative errs by a few units in the last place of
/// its largest term; cancellation between the terms is not compensated. An intermediate can
/// overflow although the derivative is finite, as with the same formula in `T`: for `a / b`
/// with `a = 1e300`, `b = 10` and `b′ = 1e10`, `q·b′` overflows and the derivative is −∞ where
/// the true value is −1e308.
///
/// `%` jumps where `a/b` is an integer; there the derivative is that of the branch that gave
/// the value (`t = a/b`, `r = ±0`). `t` is recovered from the exact remainder as
/// `round((a − r)/b)`, not as `trunc(a/b)`, which rounding can move to the next integer:
/// `1.0 / 0.1 == 10.0`, yet `1.0 % 0.1 == 1.0 − 9·0.1`. With `f64` the computed quotient is
/// within (2u + u²)·|t| of `t` (u = 2⁻⁵³), so the recovered `t` is exact for |t| < 2⁵¹ and
/// within (2u + u²)·|t| + ½ beyond; with `DoubleDouble` it is exact for |t| below about 2¹⁰⁰.
///
/// # Elementary functions
///
/// `Dual` implements `num_traits::Float`, `FloatConst` and [`Real`], so generic kernels and
/// `num_complex::Complex<Dual<T, N>>` carry derivatives, and `Dual<Dual<T, M>, N>` gives second
/// derivatives. The value of every method is `T`'s method on the value, bit for bit, including
/// special values and signed zeros. The exceptions are `sin` and `cos`, which take their values
/// and derivatives from one `T::sin_cos`, so that value-only and derivative calls of a kernel
/// agree. On Apple targets LLVM fuses an `f64` `sin` and `cos` of one argument into
/// `__sincos_stret`, whose sine can be an ulp off a lone `f64::sin` (about 1 in 13,000
/// arguments, e.g. 0.768039326588954 and −0.3820992815488752). The value is then the one every
/// `f64` code computing both functions of the argument gets, such as `Complex::exp`. `f64` code
/// computing only `sin x` gets the lone sine instead, which `Dual::sin` (which also needs `cos x`)
/// may then miss by an ulp. On glibc the fused and separate values agree. Constants (`FloatConst`,
/// `Real::from_f64`, `nan()`, `epsilon()`, …) have zero derivatives. With `x′` and `y′` the
/// derivatives of `self` and of the second argument:
///
/// | Methods | Derivatives |
/// |---|---|
/// | `floor`, `ceil`, `round`, `trunc`, `signum` | 0, also at the jumps |
/// | `fract` | `x′` |
/// | `abs`, `copysign` | `−x′` where the sign changes (for `abs`, a set sign bit, −0 too), else `x′` |
/// | `max`, `min`, `clamp` | those of the operand whose value is returned (see below) |
/// | `abs_sub` | `x′ − y′`, or 0 where `x <= y` |
/// | `mul_add(a, b)` | `x′·a + x·a′ + b′` |
/// | `recip` | `−(x′·r)·r` with `r = 1/x` |
/// | `powi(n)` | `n·xⁿ⁻¹·x′` (0 for `n = 0`) |
/// | `powf(y)` | `y·xʸ⁻¹·x′ + xʸ·ln x·y′`, the second term only if some `y′` is nonzero |
/// | `sqrt`, `cbrt` | `x′/(2√x)`, `x′/(3·∛x²)` |
/// | `exp`, `exp2`, `exp_m1` | `eˣ·x′`, `2ˣ·ln 2·x′`, `eˣ·x′` |
/// | `ln`, `log2`, `log10`, `ln_1p` | `x′/x`, `log₂e·x′/x`, `log₁₀e·x′/x`, `x′/(1 + x)` |
/// | `log(b)` | `(1/ln b)·x′/x − (log_b x/ln b)·b′/b` |
/// | `hypot(y)` | `x′·x/h + y′·y/h` |
/// | `sin`, `cos`, `tan` | `cos x·x′`, `−sin x·x′`, `(1 + tan²x)·x′` |
/// | `asin`, `acos`, `atan` | `±x′/√((1 − x)(1 + x))`, `x′/(1 + x²)` |
/// | `atan2(x)` (`self` is y) | `(x·y′ − y·x′)/(x² + y²)` |
/// | `sinh`, `cosh`, `tanh` | `cosh x·x′`, `sinh x·x′`, `x′/cosh²x` |
/// | `asinh`, `acosh`, `atanh` | `x′/√(x² + 1)`, `x′/√((x − 1)(x + 1))`, `x′/((1 − x)(1 + x))` |
/// | `to_degrees`, `to_radians` | `x′` times `T::one().to_degrees()` or `.to_radians()` |
///
/// Each f′(x) is evaluated in `T` from a form that neither cancels nor needlessly over- or
/// underflows:
///
/// * `asin`, `acos` and `atanh` use (1 − x)(1 + x); `exp_m1` uses (eˣ − 1) + 1 only where
///   eˣ − 1 ≥ −½, and eˣ below; `tanh` uses (1 − t)(1 + t) only where |t| ≤ ½, and 4e/(1 + e)²
///   with e = e^(−2|x|) beyond; `sinh` uses √(1 + sinh²x); and `sinh`, `asinh` and `acosh`
///   replace √(x² ± 1) by |x| beyond 2⁵⁴.
/// * `hypot` and `atan2` rescale arguments whose squares leave the normal range.
/// * `powf` uses y·(xʸ/x) while xʸ and xʸ/x are normal, and (y·xʸ)/x for |y| < 1 where xʸ/x
///   overflows. Otherwise it uses y·p·p with p = |x|^((y−1)/2), or |x|^(y/2)/√|x| where y − 1
///   is not exact, and p·ln x·p with p = x^(y/2), so that no subnormal or infinite power, or
///   rounded exponent, enters a normal result.
/// * `log` computes its two partials directly, not through `/`.
///
/// Accuracy was measured over each whole domain against high-precision derivatives
/// (`tests/dual_functions.rs`), down to the bottom of the normal range for `powf`, `hypot` and
/// `atan2`.
///
/// * The derivatives of `Dual<f64, N>` are within 5.1u (u = 2⁻⁵³) of the exact derivative at the
///   argument with Apple's libm, the worst being `tan`.
/// * The tests bound each one by the measured errors of the libm results it is computed from,
///   times their amplification, plus 1u per rounding, so the bounds hold for any libm.
/// * `Dual<DoubleDouble, N>` derivatives are within 2⁻⁹⁶ relative. For powers whose
///   `DoubleDouble` values are only that accurate, the bound is 2⁻⁹⁶ times max(1, |y ln x|) and,
///   for ∂/∂x, |(y − 1) ln x|.
/// * `DoubleDouble` has full precision only from 2⁻⁹⁶⁹ (`min_positive_value()`). Derivatives
///   computed from smaller arguments or intermediates lose precision, as such values do.
///
/// Special values, for a nonzero derivative `x′`:
///
/// * Outside a function's domain, where `T` returns NaN (`ln(−1)`, `asin(2)`, `acosh(½)`,
///   `sin(∞)`, `fract(∞)`, `log(−1, b)`), and for a NaN argument, the derivatives are NaN. Some
///   methods are exceptions:
///   * the piecewise-constant methods keep zero derivatives;
///   * `abs`, `copysign`, `max`, `min` and `clamp` keep those of the operand they return;
///   * `mul_add` and `abs_sub` follow the arithmetic rules (as `x·a + b` and `x − y`), where a
///     NaN value makes a derivative NaN only through a product.
/// * At a singular point of f′ on the edge of the domain the derivative is the one-sided limit:
///   +∞ for `sqrt`, `cbrt`, `ln`, `log2` and `log10` at either zero, for `ln_1p` at −1, for
///   `asin` and `atanh` at ±1 and for `acosh` at 1; −∞ for `acos` at ±1. `recip` and negative
///   powers at ±0 follow IEEE 754 (`recip` gives −∞).
/// * At ±∞ the derivatives are the limits: 0 for `sqrt`, `ln`, `atan`, `tanh`, `asinh`, …, and
///   ±∞ for `exp`, `sinh`, `cosh`, ….
/// * `powf`:
///   * x⁰ = 1 has derivative 0 in x everywhere (also at x = 0, ∞ or NaN);
///   * at x = 0, y·x^(y−1) is +∞, 1 or 0 for 0 < y < 1, y = 1 and y > 1;
///   * where x^(±∞) = 0, the derivative in x is the limit 0;
///   * the derivative in y, xʸ·ln x, is 0 where xʸ = 0 with x ≥ 0, and NaN for x < 0, where xʸ
///     is real only at integers y.
/// * `exp2` keeps a finite derivative where 2ˣ overflows but 2ˣ·ln 2 does not (x < 1024.53).
/// * `hypot` at the origin, where |z| has no derivative, has zero derivatives (a subgradient,
///   as for `sqrt(p * p + q * q)`). So `Complex::ln` at z = 0 gets derivative 0 in its real
///   part, ln|z|, silently, while `atan2`'s NaN flags the imaginary part.
/// * `atan2` jumps at the origin and has NaN derivatives there; with an infinite argument its
///   derivatives are 0.
/// * `log` at x = ±0 has the limit from the right in x, 1/(x·ln b) → ±∞ (as `ln`); at base ±0
///   it has derivative −sign(ln x)·∞ in the base (0 for x = 1, where the function is 0 for
///   every base); and at base 1 the IEEE 754 values of 1/(x·ln b).
/// * `max`, `min` and `clamp` take the derivatives of the operand whose value `T` returns: the
///   larger (or smaller), the one that is not NaN, and at a tie the one whose sign the result
///   has (`self` if both have it). `clamp` takes a bound's derivatives outside the interval and
///   `self`'s inside and at its ends; where `f64::clamp` would panic (a NaN bound, or
///   `min > max`) it returns NaN, as `DoubleDouble` does.
///
/// The chain rule multiplies `x′` by f′(x) evaluated in `T`. Where f′(x) itself overflows or
/// underflows, `x′·f′(x)` is therefore ±∞ or 0 even if it would be representable, as for an
/// intermediate of `/`. For example `atan` beyond |x| ≈ 1.3·10¹⁵⁴, where 1/(1 + x²)
/// underflows, has derivative 0 even for a huge `x′`.
///
/// In nested duals, the limits at singular points of `sqrt`, `ln`, `log2` and `log10` carry
/// inner derivatives (their second derivatives at 0 are −∞). The other special cases return
/// constants, and those have zero inner derivatives: the origin of `hypot` and `atan2`, the
/// limits of `powf`, and `log` at base 0.
///
/// # Special values and signed zeros
///
/// A derivative component that is exactly zero is a *strong zero*: it contributes zero even
/// when multiplied by an infinite or NaN factor. So a direction on which no operand depends
/// keeps derivative 0 whatever the values: `Dual::constant(2.0) * Dual::constant(f64::INFINITY)`
/// has zero derivatives, not NaN, and a derivative never depends on `N` or on the other
/// directions. In arithmetic with finite operands, a derivative is NaN only if the value is NaN
/// too (`0 / 0`, `x % 0`) or two terms of its formula overflow with opposite signs (`∞ − ∞`)
/// (for the elementary functions, see above). Otherwise
/// IEEE 754 rules apply: `x / 0` has derivative `x′/0 = ±∞`, `a / b` at `b = ±0` with `b′ ≠ 0`
/// is dominated by the `a·b′/b²` term, and a nonzero derivative times an infinite value is
/// infinite. An infinite or NaN derivative times a zero value is NaN, as in `T`.
///
/// The rule cannot tell a direction the operands do not depend on from one in which they are
/// stationary. For `p` seeded at 0, `p * p` rightly has derivative 0, so a square root of it
/// (`sqrt(p * p)`, or `(p * p).powf(0.25)`) gets derivative 0 at the singular point of the root,
/// where `|p|` and `|p|^½` have no derivative (the plain chain rule would give 0·∞ = NaN). Write
/// `abs` rather than the root of a square, if ±p′ is wanted there (`hypot` at the origin also
/// gives zero derivatives, by choice; see "Elementary functions").
///
/// The sign of a zero derivative is unspecified (it may be `+0` or `−0`). It carries no
/// information, and no operation lets it change a nonzero result.
///
/// # Speed
///
/// Operations are `#[inline]`, and the derivatives are fixed-length arrays that LLVM unrolls;
/// for `N = 4` and `N = 8` it packs them into SIMD lanes (two-lane NEON on aarch64), for
/// `N = 2` only in some code. `*`, `/`, `* f64`, `/ f64`, the chain rule and the formulas of
/// `recip`, `powf`, `hypot`, `atan2`, `log` and `mul_add` first check the values for the
/// special cases above (for `/`, three floating-point comparisons) and hand those to `#[cold]`
/// out-of-line code. These checks are branches, so unlike the same loop over `f64`, a loop over
/// many `Dual`s is not vectorised across elements. The correctly rounded value of `/` costs a
/// second division: `a/b` for the value and `1/b` for the derivatives.
///
/// An elementary function adds to `T`'s cost that of f′(x) and one multiplication per
/// derivative:
///
/// * almost nothing for `exp`, whose derivative is its value, or for `sin` and `cos`, which get
///   both from one `sin_cos`;
/// * a division for `sqrt`, `ln` and the inverse functions;
/// * a second function call for `cosh`, `tanh` (beyond |t| = ½) and `exp_m1` (below
///   eˣ − 1 = −½);
/// * `ln x` for `powf` with a varying exponent;
/// * for `DoubleDouble`, whose functions dominate, the second function is a second series.
///
/// # Examples
///
/// ```
/// use num_traits::Float;
/// use scatrs::scalar::Dual;
///
/// // d/dx (x sin x) = sin x + x cos x, at x = 2.
/// let x = Dual::<f64, 1>::new(2.0, [1.0]);
/// let y = x * x.sin();
/// assert_eq!(y.re(), 2.0 * 2f64.sin());
/// assert!((y.eps()[0] - (2f64.sin() + 2.0 * 2f64.cos())).abs() < 1e-15);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Dual<T, const N: usize> {
    re: T,
    eps: [T; N],
}

impl<T: Real, const N: usize> Dual<T, N> {
    /// A dual number with the given value and derivatives.
    #[inline]
    pub fn new(re: T, eps: [T; N]) -> Self {
        Self { re, eps }
    }

    /// A constant: all derivatives zero.
    #[inline]
    pub fn constant(re: T) -> Self {
        Self {
            re,
            eps: [T::zero(); N],
        }
    }

    /// Seeds `N` independent variables: element `i` has value `values[i]` and derivative 1
    /// in direction `i` only.
    #[inline]
    pub fn variables(values: [T; N]) -> [Self; N] {
        array::from_fn(|i| {
            let mut eps = [T::zero(); N];
            eps[i] = T::one();
            Self { re: values[i], eps }
        })
    }

    /// The value.
    #[inline]
    pub fn re(&self) -> T {
        self.re
    }

    /// The derivatives.
    #[inline]
    pub fn eps(&self) -> &[T; N] {
        &self.eps
    }

    /// Chain rule: value `f`, derivatives scaled by `df` = f′(re).
    ///
    /// Zero derivatives stay zero when `df` is infinite or NaN (a singular point such as
    /// `sqrt` at 0), so a constant stays a constant. A NaN value `f` (an argument outside the
    /// domain, or NaN) makes every nonzero derivative NaN, even where the formula for `df` is
    /// finite there (`1/x` for `ln` at x < 0).
    #[inline]
    pub(crate) fn chain(self, f: T, df: T) -> Self {
        self.chain_or(f, df, |_, _, df| df)
    }

    /// [`chain`](Self::chain), where the cold path first replaces `df` by `fix(re, f, df)`: the
    /// derivative at a point where the fast formula for `df` is not finite but has the wrong
    /// sign or is not the limit (`1/x` at `x = −0` for `ln`). `fix` runs out of line and only
    /// when `df` is not finite and `f` is not NaN.
    #[inline]
    pub(crate) fn chain_or<F: FnOnce(T, T, T) -> T>(self, f: T, df: T, fix: F) -> Self {
        // Combined without short-circuiting: one branch.
        let eps = if df.is_finite() & !f.is_nan() {
            self.eps.map(|e| e * df)
        } else {
            chain_special(self, f, df, fix).eps
        };
        Self { re: f, eps }
    }
}

// Special-value paths. Each checked operation calls one of the `*_special` functions below as
// `special(..).eps`. Measured on `Dual<f64, N>` (Apple M4, cycle counts):
// * out of line and `#[cold]`, a special path keeps the check a real, predicted branch; inline,
//   LLVM turned small ones into selects that put the check on the critical path (a multiply
//   chain with N = 1 took 2.2× the cycles, a division chain 1.9×);
// * returning the whole `Dual` keeps the merge in floating-point registers (a `[f64; 1]` is
//   returned in an integer register), and taking `.eps` rather than returning early keeps the
//   fast path's result out of the return slot (an early return cost ~14% throughput at N = 4);
// * operands go by value; by reference measured the same or slower;
// * branch-free strong zeros everywhere, instead of a check, cost ~25% throughput at N = 4.

/// The strong-zero product of a derivative component `e` and a factor `v`: `+0` whenever
/// `e.is_zero()` (`e` is `±0`; for a nested `Dual`, zero in every part), whatever `v` is,
/// including ±∞ and NaN; otherwise exactly `e * v`, so a nonzero `e` times ±∞ or NaN still gives
/// ±∞ or NaN.
#[inline]
fn strong_mul<T: Real>(e: T, v: T) -> T {
    if e.is_zero() { T::zero() } else { e * v }
}

/// The strong-zero quotient of a derivative component `e` by `v`: `+0` whenever `e.is_zero()`,
/// whatever `v` is, including ±0 and NaN; otherwise exactly `e / v`, so a nonzero `e` divided
/// by ±0 gives ±∞.
#[inline]
fn strong_div<T: Real>(e: T, v: T) -> T {
    if e.is_zero() { T::zero() } else { e / v }
}

/// Value `f`, derivatives `x′·df` with strong zeros (`df` infinite or NaN, or `f` NaN), where
/// `df` is first replaced by NaN if `f` is NaN and by `fix(x, f, df)` otherwise.
#[cold]
#[inline(never)]
fn chain_special<T: Real, const N: usize, F: FnOnce(T, T, T) -> T>(
    x: Dual<T, N>,
    f: T,
    df: T,
    fix: F,
) -> Dual<T, N> {
    let df = if f.is_nan() {
        T::nan()
    } else {
        fix(x.re, f, df)
    };
    Dual {
        re: f,
        eps: x.eps.map(|e| strong_mul(e, df)),
    }
}

/// `a * b` with strong zeros (a value is infinite or NaN, or the product overflowed).
#[cold]
#[inline(never)]
fn mul_special<T: Real, const N: usize>(a: Dual<T, N>, b: Dual<T, N>, re: T) -> Dual<T, N> {
    Dual {
        re,
        eps: array::from_fn(|i| strong_mul(a.eps[i], b.re) + strong_mul(b.eps[i], a.re)),
    }
}

/// `a * y` with strong zeros (`y` infinite or NaN).
#[cold]
#[inline(never)]
fn mul_f64_special<T: Real, const N: usize>(a: Dual<T, N>, y: f64) -> Dual<T, N> {
    Dual {
        re: a.re * y,
        eps: a.eps.map(|e| if e.is_zero() { T::zero() } else { e * y }),
    }
}

/// `a / y` with strong zeros (`y` zero or NaN).
#[cold]
#[inline(never)]
fn div_f64_special<T: Real, const N: usize>(a: Dual<T, N>, y: f64) -> Dual<T, N> {
    Dual {
        re: a.re / y,
        eps: a.eps.map(|e| if e.is_zero() { T::zero() } else { e / y }),
    }
}

/// `a / b` with quotient `q`, when the fast path's `(a′ − q·b′)·(1/b)` could meet 0·∞ or a
/// reciprocal without full precision: `(a′ − q·b′)/b`, divided directly, with strong zeros.
#[cold]
#[inline(never)]
fn div_special<T: Real, const N: usize>(a: Dual<T, N>, b: Dual<T, N>, q: T) -> Dual<T, N> {
    Dual {
        re: q,
        eps: array::from_fn(|i| strong_div(a.eps[i] - strong_mul(b.eps[i], q), b.re)),
    }
}

impl<T: Real, const N: usize> Neg for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            eps: self.eps.map(|e| -e),
        }
    }
}

impl<T: Real, const N: usize> Add for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            eps: array::from_fn(|i| self.eps[i] + o.eps[i]),
        }
    }
}

impl<T: Real, const N: usize> Sub for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            eps: array::from_fn(|i| self.eps[i] - o.eps[i]),
        }
    }
}

impl<T: Real, const N: usize> Mul for Dual<T, N> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)] // the product rule adds two products
    fn mul(self, o: Self) -> Self {
        let re = self.re * o.re;
        // A finite product has finite factors, so the product rule meets no 0·∞.
        let eps = if re.is_finite() {
            array::from_fn(|i| self.eps[i] * o.re + self.re * o.eps[i])
        } else {
            mul_special(self, o, re).eps
        };
        Self { re, eps }
    }
}

impl<T: Real, const N: usize> Div for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let q = self.re / o.re;
        let inv = T::one() / o.re;
        // A finite quotient has a finite dividend and a nonzero divisor, so no 0·∞ arises, and
        // a finite 1/b of at least `min_positive_value()` has full precision. Otherwise b is
        // ±0, ±∞ or NaN, |b| is subnormal or near MAX, the dividend is not finite, or the
        // quotient overflowed. The three comparisons are combined without short-circuiting, so
        // they cost one branch (NaN fails each of them).
        let (max, m) = (T::max_value(), inv.abs());
        let fast = (q.abs() <= max) & (m <= max) & (m >= T::min_positive_value());
        let eps = if fast {
            array::from_fn(|i| (self.eps[i] - q * o.eps[i]) * inv)
        } else {
            div_special(self, o, q).eps
        };
        Self { re: q, eps }
    }
}

impl<T: Real, const N: usize> Rem for Dual<T, N> {
    type Output = Self;
    fn rem(self, o: Self) -> Self {
        let r = self.re % o.re;
        // The integer quotient, from the exact remainder; infinite if it overflows (then a
        // nonzero b′ gives an infinite derivative and a zero one contributes nothing).
        let t = ((self.re - r) / o.re).round();
        Self {
            re: r,
            eps: array::from_fn(|i| self.eps[i] - strong_mul(o.eps[i], t)),
        }
    }
}

impl<T: Real, const N: usize> Add<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, y: f64) -> Self {
        Self {
            re: self.re + y,
            eps: self.eps,
        }
    }
}

impl<T: Real, const N: usize> Sub<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, y: f64) -> Self {
        Self {
            re: self.re - y,
            eps: self.eps,
        }
    }
}

impl<T: Real, const N: usize> Mul<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, y: f64) -> Self {
        let eps = if y.is_finite() {
            self.eps.map(|e| e * y)
        } else {
            mul_f64_special(self, y).eps
        };
        Self {
            re: self.re * y,
            eps,
        }
    }
}

impl<T: Real, const N: usize> Div<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, y: f64) -> Self {
        let eps = if y == 0.0 || y.is_nan() {
            div_f64_special(self, y).eps
        } else {
            self.eps.map(|e| e / y)
        };
        Self {
            re: self.re / y,
            eps,
        }
    }
}

macro_rules! assign_ops {
    ($($trait:ident, $method:ident, $op:tt);* $(;)?) => {$(
        impl<T: Real, const N: usize> $trait for Dual<T, N> {
            #[inline]
            fn $method(&mut self, o: Self) {
                *self = *self $op o;
            }
        }
        impl<T: Real, const N: usize> $trait<f64> for Dual<T, N> {
            #[inline]
            fn $method(&mut self, o: f64) {
                *self = *self $op o;
            }
        }
    )*};
}

assign_ops!(
    AddAssign, add_assign, +;
    SubAssign, sub_assign, -;
    MulAssign, mul_assign, *;
    DivAssign, div_assign, /;
);

impl<T: Real, const N: usize> RemAssign for Dual<T, N> {
    #[inline]
    fn rem_assign(&mut self, o: Self) {
        *self = *self % o;
    }
}

/// Compares the values only.
impl<T: Real, const N: usize> PartialEq for Dual<T, N> {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        self.re == o.re
    }
}

/// Compares the values only.
impl<T: Real, const N: usize> PartialOrd for Dual<T, N> {
    #[inline]
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&o.re)
    }
}

impl<T: Real, const N: usize> Zero for Dual<T, N> {
    #[inline]
    fn zero() -> Self {
        Self::constant(T::zero())
    }

    /// Whether the value and every derivative are zero (of either sign). Strict, so that
    /// generic shortcuts on zero keep derivatives (see "Comparisons and zero tests" on [`Dual`]).
    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps.iter().all(Zero::is_zero)
    }
}

impl<T: Real, const N: usize> One for Dual<T, N> {
    #[inline]
    fn one() -> Self {
        Self::constant(T::one())
    }

    /// Whether the value is one and every derivative zero (see [`Zero::is_zero`] for why).
    #[inline]
    fn is_one(&self) -> bool {
        self.re.is_one() && self.eps.iter().all(Zero::is_zero)
    }
}

impl<T: Real, const N: usize> Num for Dual<T, N> {
    type FromStrRadixErr = T::FromStrRadixErr;

    /// A constant parsed by `T::from_str_radix`.
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(Self::constant)
    }
}

impl<T: Real, const N: usize> NumCast for Dual<T, N> {
    /// A constant converted by `T`'s `NumCast::from`, which reads `n` through `ToPrimitive`.
    /// From another `Dual` that is its value alone: the derivatives are dropped.
    fn from<U: ToPrimitive>(n: U) -> Option<Self> {
        <T as NumCast>::from(n).map(Self::constant)
    }
}

/// Converts the value with `T`'s conversions, ignoring the derivatives.
impl<T: Real, const N: usize> ToPrimitive for Dual<T, N> {
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.re.to_isize()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.re.to_i8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.re.to_i16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.re.to_i32()
    }
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.re.to_i64()
    }
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.re.to_i128()
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.re.to_usize()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.re.to_u8()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.re.to_u16()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.re.to_u32()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.re.to_u64()
    }
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.re.to_u128()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.re.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.re.to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_scales_and_keeps_zero_derivatives_zero() {
        let x = Dual::new(4.0_f64, [1.0, 0.0, -2.0]);
        let s = x.chain(2.0, 0.25); // sqrt at 4
        assert_eq!((s.re(), s.eps()), (2.0, &[0.25, 0.0, -0.5]));
        // sqrt at 0: f′ = +∞. A zero derivative must not become 0·∞ = NaN.
        let z = Dual::new(0.0_f64, [1.0, 0.0, -2.0]).chain(0.0, f64::INFINITY);
        assert_eq!(z.eps(), &[f64::INFINITY, 0.0, f64::NEG_INFINITY]);
        let c = Dual::<f64, 2>::constant(-1.0).chain(f64::NAN, f64::NAN); // sqrt(-1)
        assert!(c.re().is_nan());
        assert_eq!(c.eps(), &[0.0, 0.0]);
        let n = Dual::new(-1.0_f64, [1.0]).chain(f64::NAN, f64::NAN);
        assert!(n.eps()[0].is_nan());
    }
}
