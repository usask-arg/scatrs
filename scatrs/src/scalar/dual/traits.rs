//! `Float`, `FloatConst` and `Real` for [`Dual`]: the chain rule for every function. The
//! semantics are documented on [`Dual`] ("Elementary functions").

use core::array;
use core::num::FpCategory;

use num_traits::{Float, FloatConst};

use super::{Dual, strong_div, strong_mul};
use crate::scalar::Real;

/// 2⁵⁴: beyond it √(x² ± 1) is |x| to within 2⁻¹⁰⁹ relative, better than `DoubleDouble`
/// precision, and |x| avoids the overflow of x² beyond about 1.3·10¹⁵⁴.
const TWO_POW_54: f64 = f64::from_bits((1023 + 54) << 52);
/// 2⁶⁰⁰ and 2⁻⁶⁰⁰: rescale arguments whose squares leave the normal range.
const TWO_POW_600: f64 = f64::from_bits((1023 + 600) << 52);
const TWO_POW_M600: f64 = f64::from_bits((1023 - 600) << 52);

/// `fix` for `chain_or` where f′ is 1/x up to a factor: +∞ at either zero (the limit from the
/// side where f is defined; the formula gives −∞ at −0).
#[inline]
fn plus_infinity_at_zero<T: Real>(x: T, _f: T, df: T) -> T {
    if x == T::zero() { T::infinity() } else { df }
}

impl<T: Real, const N: usize> Float for Dual<T, N> {
    #[inline]
    fn nan() -> Self {
        Self::constant(T::nan())
    }

    #[inline]
    fn infinity() -> Self {
        Self::constant(T::infinity())
    }

    #[inline]
    fn neg_infinity() -> Self {
        Self::constant(T::neg_infinity())
    }

    #[inline]
    fn neg_zero() -> Self {
        Self::constant(T::neg_zero())
    }

    #[inline]
    fn min_value() -> Self {
        Self::constant(T::min_value())
    }

    #[inline]
    fn min_positive_value() -> Self {
        Self::constant(T::min_positive_value())
    }

    #[inline]
    fn epsilon() -> Self {
        Self::constant(T::epsilon())
    }

    #[inline]
    fn max_value() -> Self {
        Self::constant(T::max_value())
    }

    /// Whether the value or any derivative is NaN.
    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.eps.iter().any(|e| e.is_nan())
    }

    /// Whether the value is infinite (the derivatives are not inspected).
    #[inline]
    fn is_infinite(self) -> bool {
        self.re.is_infinite()
    }

    /// Whether the value and every derivative are finite.
    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.eps.iter().all(|e| e.is_finite())
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.re.is_normal()
    }

    #[inline]
    fn is_subnormal(self) -> bool {
        self.re.is_subnormal()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.re.classify()
    }

    #[inline]
    fn floor(self) -> Self {
        Self::constant(self.re.floor())
    }

    #[inline]
    fn ceil(self) -> Self {
        Self::constant(self.re.ceil())
    }

    #[inline]
    fn round(self) -> Self {
        Self::constant(self.re.round())
    }

    #[inline]
    fn trunc(self) -> Self {
        Self::constant(self.re.trunc())
    }

    #[inline]
    fn fract(self) -> Self {
        self.chain(self.re.fract(), T::one())
    }

    #[inline]
    fn abs(self) -> Self {
        let eps = if self.re.is_sign_negative() {
            self.eps.map(|e| -e)
        } else {
            self.eps
        };
        Self {
            re: self.re.abs(),
            eps,
        }
    }

    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.re.signum())
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        self.re.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        self.re.is_sign_negative()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let re = self.re.mul_add(a.re, b.re);
        // A finite result has finite operands, so the product rule meets no 0·∞.
        let eps = if re.is_finite() {
            array::from_fn(|i| self.eps[i] * a.re + self.re * a.eps[i] + b.eps[i])
        } else {
            mul_add_special(self, a, b, re).eps
        };
        Self { re, eps }
    }

    #[inline]
    fn recip(self) -> Self {
        let r = self.re.recip();
        // −x′/x² as −(x′·r)·r, while r is finite and normal (as for `/`).
        let (max, m) = (T::max_value(), r.abs());
        let eps = if (m <= max) & (m >= T::min_positive_value()) {
            self.eps.map(|e| -(e * r) * r)
        } else {
            recip_special(self, r).eps
        };
        Self { re: r, eps }
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let x = self.re;
        let f = x.powi(n);
        if n == 0 {
            return Self::constant(f); // x⁰ = 1 for every x, NaN included
        }
        let df = match n.checked_sub(1) {
            Some(m) => x.powi(m) * f64::from(n),
            None => f / x * f64::from(n), // n = i32::MIN: x^(n−1) = xⁿ/x
        };
        self.chain(f, df)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        let (x, y) = (self.re, n.re);
        let f = x.powf(y);
        // ∂/∂x = y·x^(y−1) as y·(f/x), valid for a normal f (then x is finite and nonzero).
        let dx = y * (f / x);
        // ∂/∂y = f·ln x, needed only when the exponent varies.
        let varies = n.eps.iter().any(|e| !e.is_zero());
        let dy = if varies { f * x.ln() } else { T::zero() };
        let fast = dx.is_finite() & dy.is_finite() & (f.abs() >= T::min_positive_value());
        let eps = if fast {
            array::from_fn(|i| self.eps[i] * dx + n.eps[i] * dy)
        } else {
            powf_special(self, n, f, varies).eps
        };
        Self { re: f, eps }
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        // sqrt(−0) = −0, but the derivative at either zero is +∞.
        self.chain_or(s, (s + s).recip(), plus_infinity_at_zero)
    }

    #[inline]
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }

    #[inline]
    fn exp2(self) -> Self {
        let e = self.re.exp2();
        // Where 2ˣ overflows, 2ˣ·ln 2 may not (x < 1024.53): use 2^(x−1)·2 ln 2.
        self.chain_or(e, e * T::LN_2(), |x, f, df| {
            if f.is_infinite() && x.is_finite() {
                (x - 1.0).exp2() * T::LN_2() * 2.0
            } else {
                df
            }
        })
    }

    #[inline]
    fn ln(self) -> Self {
        self.chain_or(self.re.ln(), self.re.recip(), plus_infinity_at_zero)
    }

    /// `ln(self)/ln(base)` differentiated as a quotient; the value is `T::log`.
    #[inline]
    fn log(self, base: Self) -> Self {
        let q = self.ln() / base.ln();
        Self {
            re: self.re.log(base.re),
            eps: q.eps,
        }
    }

    #[inline]
    fn log2(self) -> Self {
        let x = self.re;
        self.chain_or(x.log2(), T::LOG2_E() / x, plus_infinity_at_zero)
    }

    #[inline]
    fn log10(self) -> Self {
        let x = self.re;
        self.chain_or(x.log10(), T::LOG10_E() / x, plus_infinity_at_zero)
    }

    #[inline]
    fn to_degrees(self) -> Self {
        self.chain(self.re.to_degrees(), T::one().to_degrees())
    }

    #[inline]
    fn to_radians(self) -> Self {
        self.chain(self.re.to_radians(), T::one().to_radians())
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        let (a, b) = (self.re, other.re);
        let re = a.max(b);
        // The operand `T::max` returned: the larger, the one that is not NaN, or at a tie the one
        // with the sign of the result (`self` if both have it).
        let take_other =
            a.is_nan() || b > a || (b == a && re.is_sign_negative() != a.is_sign_negative());
        let eps = if take_other { other.eps } else { self.eps };
        Self { re, eps }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let (a, b) = (self.re, other.re);
        let re = a.min(b);
        let take_other =
            a.is_nan() || b < a || (b == a && re.is_sign_negative() != a.is_sign_negative());
        let eps = if take_other { other.eps } else { self.eps };
        Self { re, eps }
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if min.re.is_nan() || max.re.is_nan() || min.re > max.re {
            return Self::nan(); // as `DoubleDouble`; `f64::clamp` panics
        }
        let re = self.re.clamp(min.re, max.re);
        let eps = if self.re < min.re {
            min.eps
        } else if self.re > max.re {
            max.eps
        } else {
            self.eps
        };
        Self { re, eps }
    }

    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        let re = self.re.abs_sub(other.re);
        let eps = if self.re <= other.re {
            [T::zero(); N]
        } else {
            array::from_fn(|i| self.eps[i] - other.eps[i])
        };
        Self { re, eps }
    }

    #[inline]
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        self.chain(c, (c * c * 3.0).recip())
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        // x′·x/h + y′·y/h, while h and 1/h are normal (a computed h below that has lost bits).
        let (inv, min) = (h.recip(), T::min_positive_value());
        let eps = if (h >= min) & (inv >= min) {
            let (a, b) = (self.re * inv, other.re * inv);
            array::from_fn(|i| self.eps[i] * a + other.eps[i] * b)
        } else {
            hypot_special(self, other, h).eps
        };
        Self { re: h, eps }
    }

    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(s, c)
    }

    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(c, -s)
    }

    #[inline]
    fn tan(self) -> Self {
        let t = self.re.tan();
        self.chain(t, t * t + 1.0)
    }

    #[inline]
    fn asin(self) -> Self {
        let x = self.re;
        let d = ((T::one() - x) * (T::one() + x)).sqrt();
        self.chain(x.asin(), d.recip())
    }

    #[inline]
    fn acos(self) -> Self {
        let x = self.re;
        let d = ((T::one() - x) * (T::one() + x)).sqrt();
        self.chain(x.acos(), -d.recip())
    }

    #[inline]
    fn atan(self) -> Self {
        let x = self.re;
        self.chain(x.atan(), (x * x + 1.0).recip())
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let (y, x) = (self.re, other.re);
        let re = y.atan2(x);
        // (y′x − x′y)/(x² + y²), while x² + y² and its reciprocal are normal (a computed x² + y²
        // below that has lost bits).
        let r2 = x * x + y * y;
        let (inv, min) = (r2.recip(), T::min_positive_value());
        let eps = if (r2 >= min) & (inv >= min) {
            let (a, b) = (x * inv, -(y * inv));
            array::from_fn(|i| self.eps[i] * a + other.eps[i] * b)
        } else {
            atan2_special(self, other, re).eps
        };
        Self { re, eps }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (self.chain(s, c), self.chain(c, -s))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        let x = self.re;
        let m = x.exp_m1();
        // eˣ as (eˣ − 1) + 1 where that does not cancel; eˣ itself below.
        let d = if m >= T::from_f64(-0.5) {
            m + 1.0
        } else {
            x.exp()
        };
        self.chain(m, d)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        let x = self.re;
        self.chain(x.ln_1p(), (x + 1.0).recip())
    }

    #[inline]
    fn sinh(self) -> Self {
        let s = self.re.sinh();
        // cosh x = √(1 + sinh²x), which is |sinh x| where sinh² could overflow.
        let a = s.abs();
        let c = if a < T::from_f64(TWO_POW_54) {
            (s * s + 1.0).sqrt()
        } else {
            a
        };
        self.chain(s, c)
    }

    #[inline]
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }

    #[inline]
    fn tanh(self) -> Self {
        let t = self.re.tanh();
        // 1 − tanh²x as (1 − t)(1 + t) for |t| <= ½, and as 1/cosh²x beyond, where 1 − |t| would
        // cancel.
        let d = if t.abs() <= T::from_f64(0.5) {
            (T::one() - t) * (T::one() + t)
        } else {
            let r = self.re.cosh().recip();
            r * r
        };
        self.chain(t, d)
    }

    #[inline]
    fn asinh(self) -> Self {
        let x = self.re;
        let a = x.abs();
        let d = if a < T::from_f64(TWO_POW_54) {
            (x * x + 1.0).sqrt()
        } else {
            a
        };
        self.chain(x.asinh(), d.recip())
    }

    #[inline]
    fn acosh(self) -> Self {
        let x = self.re;
        let d = if x < T::from_f64(TWO_POW_54) {
            ((x - 1.0) * (x + 1.0)).sqrt()
        } else {
            x
        };
        self.chain(x.acosh(), d.recip())
    }

    #[inline]
    fn atanh(self) -> Self {
        let x = self.re;
        self.chain(x.atanh(), ((T::one() - x) * (T::one() + x)).recip())
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.re.integer_decode()
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let eps = if self.re.is_sign_negative() != sign.re.is_sign_negative() {
            self.eps.map(|e| -e)
        } else {
            self.eps
        };
        Self {
            re: self.re.copysign(sign.re),
            eps,
        }
    }
}

/// `x·a + b` with strong zeros (a value is infinite or NaN, or the result overflowed).
#[cold]
#[inline(never)]
fn mul_add_special<T: Real, const N: usize>(
    x: Dual<T, N>,
    a: Dual<T, N>,
    b: Dual<T, N>,
    re: T,
) -> Dual<T, N> {
    Dual {
        re,
        eps: array::from_fn(|i| strong_mul(x.eps[i], a.re) + strong_mul(a.eps[i], x.re) + b.eps[i]),
    }
}

/// `1/x` when `r = 1/x` is not finite and normal: −(x′/x)/x, divided directly, with strong zeros.
#[cold]
#[inline(never)]
fn recip_special<T: Real, const N: usize>(x: Dual<T, N>, r: T) -> Dual<T, N> {
    Dual {
        re: r,
        eps: x.eps.map(|e| -strong_div(strong_div(e, x.re), x.re)),
    }
}

/// `x^y` when the fast formulas fail: `f` is zero, subnormal, infinite or NaN, or a partial
/// derivative is not finite.
#[cold]
#[inline(never)]
fn powf_special<T: Real, const N: usize>(
    x: Dual<T, N>,
    y: Dual<T, N>,
    f: T,
    varies: bool,
) -> Dual<T, N> {
    let (a, b) = (x.re, y.re);
    let (dx, dy) = if f.is_nan() {
        (T::nan(), T::nan())
    } else {
        // y·x^(y−1). x⁰ = 1 for every x, so its derivative is 0 even where x^(−1) is infinite;
        // and where x^(±∞) = 0, the limit of y·x^(y−1) is 0 (not ∞·0).
        let m = f.abs();
        let normal = m >= T::min_positive_value() && m <= T::max_value();
        let dx = if b == T::zero() || (b.is_infinite() && f == T::zero()) {
            T::zero()
        } else if normal {
            b * (f / a)
        } else {
            b * a.powf(b - T::one())
        };
        // f·ln x, which is 0 where f is: at x = ±0 with y > 0, at x = +∞ with y < 0, and on
        // underflow; but not for x < 0, where xʸ has no derivative in y.
        let dy = if !varies || (f == T::zero() && a >= T::zero()) {
            T::zero()
        } else {
            f * a.ln()
        };
        (dx, dy)
    };
    Dual {
        re: f,
        eps: array::from_fn(|i| strong_mul(x.eps[i], dx) + strong_mul(y.eps[i], dy)),
    }
}

/// `hypot(x, y)` when `h` or `1/h` is not normal.
#[cold]
#[inline(never)]
fn hypot_special<T: Real, const N: usize>(x: Dual<T, N>, y: Dual<T, N>, h: T) -> Dual<T, N> {
    let (a, b) = hypot_partials(x.re, y.re, h);
    Dual {
        re: h,
        eps: array::from_fn(|i| strong_mul(x.eps[i], a) + strong_mul(y.eps[i], b)),
    }
}

/// (x/h, y/h) for `h = hypot(x, y)` outside the range where `h` and `1/h` are normal.
fn hypot_partials<T: Real>(x: T, y: T, h: T) -> (T, T) {
    if h == T::zero() {
        return (T::zero(), T::zero()); // the origin: |z| has no derivative; 0 is a subgradient
    }
    if h.is_nan() {
        return (T::nan(), T::nan());
    }
    if h.is_infinite() {
        match (x.is_finite(), y.is_finite()) {
            (false, true) if x.is_infinite() => return (x.signum(), T::zero()),
            (true, false) if y.is_infinite() => return (T::zero(), y.signum()),
            (true, true) => {}                // overflow: rescale below
            _ => return (T::nan(), T::nan()), // both infinite, or one NaN: no direction
        }
    }
    // h is tiny or huge: rescale by a power of two, exactly where it matters.
    let s = if h < T::one() {
        TWO_POW_600
    } else {
        TWO_POW_M600
    };
    let (xs, ys) = (x * s, y * s);
    let hs = xs.hypot(ys);
    (xs / hs, ys / hs)
}

/// `atan2(y, x)` when `x² + y²` or its reciprocal is not normal.
#[cold]
#[inline(never)]
fn atan2_special<T: Real, const N: usize>(y: Dual<T, N>, x: Dual<T, N>, re: T) -> Dual<T, N> {
    let (a, b) = atan2_partials(y.re, x.re);
    Dual {
        re,
        eps: array::from_fn(|i| strong_mul(y.eps[i], a) + strong_mul(x.eps[i], b)),
    }
}

/// (∂/∂y, ∂/∂x) of atan2(y, x), (x, −y)/(x² + y²), where x² + y² overflows or underflows.
fn atan2_partials<T: Real>(y: T, x: T) -> (T, T) {
    if y.is_nan() || x.is_nan() {
        return (T::nan(), T::nan());
    }
    if y.is_infinite() || x.is_infinite() {
        return (T::zero(), T::zero()); // both tend to 0 as |x| + |y| → ∞
    }
    let m = y.abs().max(x.abs());
    if m == T::zero() {
        return (T::nan(), T::nan()); // the origin, where atan2 jumps
    }
    let (ys, xs) = (y / m, x / m);
    let r2 = xs * xs + ys * ys; // in [1, 2]
    ((xs / r2) / m, -((ys / r2) / m))
}

macro_rules! float_consts {
    ($($name:ident),* $(,)?) => {
        /// Constants, with zero derivatives.
        impl<T: Real, const N: usize> FloatConst for Dual<T, N> {
            $(
                #[inline]
                fn $name() -> Self {
                    Self::constant(T::$name())
                }
            )*
        }
    };
}

float_consts!(
    E,
    FRAC_1_PI,
    FRAC_1_SQRT_2,
    FRAC_2_PI,
    FRAC_2_SQRT_PI,
    FRAC_PI_2,
    FRAC_PI_3,
    FRAC_PI_4,
    FRAC_PI_6,
    FRAC_PI_8,
    LN_10,
    LN_2,
    LOG10_E,
    LOG2_E,
    PI,
    SQRT_2,
    TAU,
    LOG10_2,
    LOG2_10,
);

impl<T: Real, const N: usize> Real for Dual<T, N> {
    /// A constant.
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::constant(T::from_f64(x))
    }

    /// A constant.
    #[inline]
    fn from_f64_pair(hi: f64, lo: f64) -> Self {
        Self::constant(T::from_f64_pair(hi, lo))
    }

    /// The value's nearest `f64`; the derivatives are ignored.
    #[inline]
    fn approx_f64(self) -> f64 {
        self.re.approx_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescaled_partials_at_the_ends_of_the_range() {
        // hypot and atan2 partials where x² + y² overflows or underflows.
        let (a, b) = hypot_partials(3e300, 4e300, f64::INFINITY);
        assert!((a - 0.6).abs() <= 1e-16 && (b - 0.8).abs() <= 1e-16);
        let (a, b) = hypot_partials(3e-320, -4e-320, 5e-320);
        assert!((a - 0.6).abs() <= 1e-15 && (b + 0.8).abs() <= 1e-15);
        // ∂/∂y = x/(x² + y²) and ∂/∂x = −y/(x² + y²) with x² + y² = 25e400 or 25e-400.
        let (dy, dx) = atan2_partials(4e200, 3e200);
        assert!((dy - 1.2e-201).abs() <= 1e-216 && (dx + 1.6e-201).abs() <= 1e-216);
        let (dy, dx) = atan2_partials(4e-200, 3e-200);
        assert!((dy - 1.2e199).abs() <= 1e184 && (dx + 1.6e199).abs() <= 1e184);
    }
}
