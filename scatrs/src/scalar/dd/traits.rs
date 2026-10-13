//! num-traits and `Real` implementations for `DoubleDouble`; their semantics are documented
//! on [`DoubleDouble`](super::DoubleDouble) ("Use as `num_traits::Float`").

use core::cmp::Ordering;
use core::num::FpCategory;

use num_traits::{Float, FloatConst, Num, NumCast, One, ToPrimitive, Zero};

use super::{
    DoubleDouble as Dd, ParseDoubleDoubleError, TWO_POW_128, consts, from_u128, pow2,
    scale_down_tiny,
};
use crate::scalar::Real;

/// 2⁻¹⁰⁴: `Float::epsilon`.
const EPSILON: f64 = f64::from_bits((1023 - 104) << 52);
/// 2⁻⁹⁶⁹: below this the trailing word becomes subnormal and precision is lost.
const MIN_POSITIVE_FULL_PRECISION: f64 = f64::from_bits((1023 - 969) << 52);
/// 2⁵³: doubles of at least this magnitude are integers.
const TWO_POW_53: f64 = f64::from_bits((1023 + 53) << 52);
/// 2¹²⁷.
const TWO_POW_127: f64 = f64::from_bits((1023 + 127) << 52);
/// `mul_add` adds the product directly when its leading word lies in [2⁻⁹⁶⁸, MAX], where the
/// product has full precision.
const MUL_ADD_DIRECT_MIN: f64 = f64::from_bits((1023 - 968) << 52);
/// With a product below 2⁻⁹⁶⁸, whose lost bits are below 2⁻¹⁰⁷⁴, an addend of at least 2⁻⁸⁰⁰
/// makes them negligible; smaller addends are summed at scale 2^`MUL_ADD_SCALE`.
const MUL_ADD_SMALL_ADDEND: f64 = f64::from_bits((1023 - 800) << 52);
/// Binary exponent of the scale for tiny products with tiny addends: the scaled smaller factor
/// stays below 2¹¹⁶ and the scaled addend below 2⁻²⁰⁰.
const MUL_ADD_SCALE: i32 = 600;

/// `n` as a double-double (see [`from_u128`]).
fn from_i128(n: i128) -> Dd {
    let m = from_u128(n.unsigned_abs());
    if n < 0 { -m } else { m }
}

/// `x·a + b`: the product rounded to a double-double, then added.
#[inline]
fn mul_add(x: Dd, a: Dd, b: Dd) -> Dd {
    let p = x * a;
    if (MUL_ADD_DIRECT_MIN..=f64::MAX).contains(&p.hi.abs()) {
        return p + b;
    }
    mul_add_wide(x, a, b, p)
}

/// `mul_add` when the product `p` is zero, below 2⁻⁹⁶⁸, infinite or NaN, following
/// `f64::mul_add`: special values; no overflow of the product alone; and the signs of zero and
/// the rounding of subnormal results that a tiny product affects.
#[cold]
#[inline(never)]
fn mul_add_wide(x: Dd, a: Dd, b: Dd, p: Dd) -> Dd {
    if !x.hi.is_finite() || !a.hi.is_finite() {
        return p + b; // ∞·0 and ∞ − ∞ give NaN; NaN propagates
    }
    if !b.hi.is_finite() {
        return b; // a finite product leaves ±∞ and NaN unchanged
    }
    if p.hi.is_infinite() {
        // Only the product overflowed. |x| >= 1, so x/4 is exact, and the sum overflows at
        // quarter scale only if |x·a| >= 4·MAX, when it overflows anyway.
        return (x.mul_pow2(0.25) * a + b.mul_pow2(0.25)).mul_pow2(4.0);
    }
    if (x.hi == 0.0 || a.hi == 0.0) && b.hi == 0.0 {
        return p + b; // an exact zero product: the IEEE sign rule for a sum of zeros
    }
    if b.hi.abs() >= MUL_ADD_SMALL_ADDEND {
        return p + b; // the product's lost bits are negligible against b
    }
    // A nonzero product below 2^-968 with an addend below 2^-800: scale the smaller factor
    // and the addend up, sum there with full precision, and round once on the way back (with
    // the sign of the exact result when that underflows to zero). A product that underflows
    // even at this scale (to a zero of its sign) is negligible against a nonzero addend; a
    // zero addend is not added, as it could change the sign of such a zero.
    let (small, large) = if x.hi.abs() <= a.hi.abs() {
        (x, a)
    } else {
        (a, x)
    };
    let scale = pow2(MUL_ADD_SCALE);
    let product = small.mul_pow2(scale) * large;
    let r = if b.hi == 0.0 {
        product
    } else {
        product + b.mul_pow2(scale)
    };
    scale_down_tiny(r, MUL_ADD_SCALE)
}

impl Zero for Dd {
    #[inline]
    fn zero() -> Self {
        Dd::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.hi == 0.0
    }
}

impl One for Dd {
    #[inline]
    fn one() -> Self {
        Dd::ONE
    }
}

impl Num for Dd {
    type FromStrRadixErr = ParseDoubleDoubleError;

    /// [`DoubleDouble::from_decimal_str`](Dd::from_decimal_str), which ignores surrounding
    /// whitespace (unlike `f64`'s, and unlike `str::parse`); only radix 10 is supported.
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err(ParseDoubleDoubleError::new());
        }
        Dd::from_decimal_str(s)
    }
}

impl ToPrimitive for Dd {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.to_i128().and_then(|n| i64::try_from(n).ok())
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.to_u128().and_then(|n| u64::try_from(n).ok())
    }

    /// The truncated value. Both words of a truncated double-double are integers, and the
    /// trailing word is at most 2⁷⁴ in magnitude here.
    fn to_i128(&self) -> Option<i128> {
        let t = self.trunc();
        if t.hi.is_nan() || t.hi.abs() > TWO_POW_127 {
            return None; // NaN, infinite or out of range
        }
        let lo = t.lo as i128;
        if t.hi == TWO_POW_127 {
            return i128::MAX.checked_add(lo + 1); // 2^127 itself does not fit
        }
        (t.hi as i128).checked_add(lo)
    }

    /// The truncated value (see [`to_i128`](Self::to_i128)).
    fn to_u128(&self) -> Option<u128> {
        let t = self.trunc();
        if t.hi.is_nan() || t.hi < 0.0 || t.hi > TWO_POW_128 {
            return None; // NaN, negative, infinite or out of range (-0 is fine)
        }
        let lo = t.lo as i128;
        if t.hi == TWO_POW_128 {
            return u128::MAX.checked_add_signed(lo + 1); // 2^128 itself does not fit
        }
        (t.hi as u128).checked_add_signed(lo)
    }

    /// The leading word: the correctly rounded double.
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(self.hi)
    }

    /// Rounded once: the value is first rounded to odd at double precision (moving the leading
    /// word one step towards the trailing word when it is even and the trailing word nonzero),
    /// which a further rounding to the 24 bits of an `f32` does not disturb. Overflow gives ±∞.
    fn to_f32(&self) -> Option<f32> {
        let (hi, lo) = (self.hi, self.lo);
        let odd = if lo == 0.0 || hi.to_bits() & 1 == 1 {
            hi
        } else if (lo > 0.0) == (hi > 0.0) {
            f64::from_bits(hi.to_bits() + 1) // away from zero; MAX is odd, so this is finite
        } else {
            f64::from_bits(hi.to_bits() - 1) // towards zero
        };
        Some(odd as f32)
    }
}

impl NumCast for Dd {
    /// Exact for `f32`, `f64` and primitive integers below 2¹⁰⁶ in magnitude, and within u²·|n|
    /// (u = 2⁻⁵³) of wider integers. Any other `ToPrimitive` type, including `DoubleDouble`
    /// itself, goes through `to_f64`, or `to_i128` or `to_u128` at or beyond 2⁵³ in magnitude:
    /// a `DoubleDouble` keeps only its leading word (or its integer part beyond 2⁵³).
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let f = n.to_f64()?;
        // to_f64 rounds integers beyond 2^53: take them exactly instead.
        if f.is_finite() && f.abs() >= TWO_POW_53 {
            if let Some(i) = n.to_i128() {
                return Some(from_i128(i));
            }
            if let Some(u) = n.to_u128() {
                return Some(from_u128(u));
            }
        }
        Some(Dd::from_f64(f))
    }
}

impl Float for Dd {
    #[inline]
    fn nan() -> Self {
        Dd::NAN
    }

    #[inline]
    fn infinity() -> Self {
        Dd::INFINITY
    }

    #[inline]
    fn neg_infinity() -> Self {
        Dd::NEG_INFINITY
    }

    #[inline]
    fn neg_zero() -> Self {
        Dd::from_f64(-0.0)
    }

    #[inline]
    fn min_value() -> Self {
        Dd::from_f64(f64::MIN)
    }

    /// 2⁻⁹⁶⁹, the smallest magnitude with full double-double precision.
    #[inline]
    fn min_positive_value() -> Self {
        Dd::from_f64(MIN_POSITIVE_FULL_PRECISION)
    }

    /// 2⁻¹⁰⁴.
    #[inline]
    fn epsilon() -> Self {
        Dd::from_f64(EPSILON)
    }

    #[inline]
    fn max_value() -> Self {
        Dd::from_f64(f64::MAX)
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.hi.is_nan()
    }

    #[inline]
    fn is_infinite(self) -> bool {
        self.hi.is_infinite()
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.hi.is_finite()
    }

    /// Describes the leading word only: values in [2⁻¹⁰²², 2⁻⁹⁶⁹) are `Normal`, yet below
    /// `min_positive_value()`, the smallest magnitude with full precision.
    #[inline]
    fn is_normal(self) -> bool {
        self.hi.is_normal()
    }

    /// Describes the leading word only: values in [2⁻¹⁰²², 2⁻⁹⁶⁹) are `Normal`, yet below
    /// `min_positive_value()`, the smallest magnitude with full precision.
    #[inline]
    fn is_subnormal(self) -> bool {
        self.hi.is_subnormal()
    }

    /// Describes the leading word only: values in [2⁻¹⁰²², 2⁻⁹⁶⁹) are `Normal`, yet below
    /// `min_positive_value()`, the smallest magnitude with full precision.
    #[inline]
    fn classify(self) -> FpCategory {
        self.hi.classify()
    }

    #[inline]
    fn floor(self) -> Self {
        Dd::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        Dd::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        Dd::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        Dd::trunc(self)
    }

    /// `self − self.trunc()`, exactly; as for `f64`, +0 for integers and NaN for infinities.
    #[inline]
    fn fract(self) -> Self {
        self - Dd::trunc(self)
    }

    #[inline]
    fn abs(self) -> Self {
        Dd::abs(self)
    }

    #[inline]
    fn signum(self) -> Self {
        Dd::from_f64(self.hi.signum())
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        self.hi.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        self.hi.is_sign_negative()
    }

    /// `self·a + b`, with the product rounded to a double-double before the addition: error
    /// ≤ 4u²|self·a| + 3u²|self·a + b| (u = 2⁻⁵³), fused for `f64` arguments. Special values,
    /// signed zeros, overflow and subnormal results follow `f64::mul_add` (the product alone
    /// never overflows).
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        mul_add(self, a, b)
    }

    #[inline]
    fn recip(self) -> Self {
        Dd::recip(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Dd::powi(self, n)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Dd::powf(self, n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Dd::sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        Dd::exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        Dd::exp2(self)
    }

    #[inline]
    fn ln(self) -> Self {
        Dd::ln(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        Dd::log(self, base)
    }

    #[inline]
    fn log2(self) -> Self {
        Dd::log2(self)
    }

    #[inline]
    fn log10(self) -> Self {
        Dd::log10(self)
    }

    /// `self·(1/π)·180`: relative error ≤ 7u², overflowing only when the result does.
    #[inline]
    fn to_degrees(self) -> Self {
        self * consts::FRAC_1_PI * 180.0
    }

    /// `(self/180)·π`: relative error ≤ 8u², never overflowing.
    #[inline]
    fn to_radians(self) -> Self {
        self / 180.0 * consts::PI
    }

    /// The larger value; a NaN argument is ignored, and of two zeros the result is +0 unless both
    /// are −0 (IEEE 754-2019 maximum), whatever their order.
    #[inline]
    fn max(self, other: Self) -> Self {
        match self.partial_cmp(&other) {
            Some(Ordering::Less) => other,
            Some(Ordering::Equal) if self.hi.is_sign_negative() => other, // −0 vs ±0
            Some(Ordering::Greater | Ordering::Equal) => self,
            None if self.hi.is_nan() => other,
            None => self,
        }
    }

    /// The smaller value; a NaN argument is ignored, and of two zeros the result is −0 unless both
    /// are +0 (IEEE 754-2019 minimum), whatever their order.
    #[inline]
    fn min(self, other: Self) -> Self {
        match self.partial_cmp(&other) {
            Some(Ordering::Greater) => other,
            Some(Ordering::Equal) if self.hi.is_sign_positive() => other, // +0 vs ±0
            Some(Ordering::Less | Ordering::Equal) => self,
            None if self.hi.is_nan() => other,
            None => self,
        }
    }

    /// `self` restricted to [min, max], as `f64::clamp`; NaN (instead of a panic) when
    /// `min > max` or either bound is NaN.
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if min.hi.is_nan() || max.hi.is_nan() || min > max {
            return Dd::NAN;
        }
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    /// `self − other` if `self > other`, otherwise +0 (C's `fdim`); NaN if either is NaN.
    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        if self <= other {
            Dd::ZERO
        } else {
            self - other
        }
    }

    #[inline]
    fn cbrt(self) -> Self {
        Dd::cbrt(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Dd::hypot(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        Dd::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        Dd::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        Dd::tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        Dd::asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        Dd::acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        Dd::atan(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Dd::atan2(self, other)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        Dd::sin_cos(self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Dd::exp_m1(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Dd::ln_1p(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        Dd::sinh(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        Dd::cosh(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        Dd::tanh(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        Dd::asinh(self)
    }

    #[inline]
    fn acosh(self) -> Self {
        Dd::acosh(self)
    }

    #[inline]
    fn atanh(self) -> Self {
        Dd::atanh(self)
    }

    /// The decomposition of the leading word.
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self.hi)
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if self.hi.is_sign_negative() == sign.hi.is_sign_negative() {
            self
        } else {
            -self
        }
    }
}

impl FloatConst for Dd {
    #[inline]
    fn E() -> Self {
        consts::E
    }

    #[inline]
    fn FRAC_1_PI() -> Self {
        consts::FRAC_1_PI
    }

    #[inline]
    fn FRAC_1_SQRT_2() -> Self {
        consts::FRAC_1_SQRT_2
    }

    #[inline]
    fn FRAC_2_PI() -> Self {
        consts::FRAC_2_PI
    }

    #[inline]
    fn FRAC_2_SQRT_PI() -> Self {
        consts::FRAC_2_SQRT_PI
    }

    #[inline]
    fn FRAC_PI_2() -> Self {
        consts::FRAC_PI_2
    }

    #[inline]
    fn FRAC_PI_3() -> Self {
        consts::FRAC_PI_3
    }

    #[inline]
    fn FRAC_PI_4() -> Self {
        consts::FRAC_PI_4
    }

    #[inline]
    fn FRAC_PI_6() -> Self {
        consts::FRAC_PI_6
    }

    #[inline]
    fn FRAC_PI_8() -> Self {
        consts::FRAC_PI_8
    }

    #[inline]
    fn LN_10() -> Self {
        consts::LN_10
    }

    #[inline]
    fn LN_2() -> Self {
        consts::LN_2
    }

    #[inline]
    fn LOG10_E() -> Self {
        consts::LOG10_E
    }

    #[inline]
    fn LOG2_E() -> Self {
        consts::LOG2_E
    }

    #[inline]
    fn PI() -> Self {
        consts::PI
    }

    #[inline]
    fn SQRT_2() -> Self {
        consts::SQRT_2
    }

    #[inline]
    fn TAU() -> Self {
        consts::TAU
    }

    #[inline]
    fn LOG10_2() -> Self {
        consts::LOG10_2
    }

    #[inline]
    fn LOG2_10() -> Self {
        consts::LOG2_10
    }
}

impl Real for Dd {
    #[inline]
    fn from_f64(x: f64) -> Self {
        Dd::from_f64(x)
    }

    #[inline]
    fn from_f64_pair(hi: f64, lo: f64) -> Self {
        Dd::new(hi, lo)
    }

    #[inline]
    fn approx_f64(self) -> f64 {
        self.hi
    }
}
