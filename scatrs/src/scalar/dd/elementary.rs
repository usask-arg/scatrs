//! Square root, exponential and logarithm family for `DoubleDouble`.
//!
//! * `exp`: Cody–Waite reduction x = m·ln 2 + r by a three-word ln 2, then eʳ − 1 from a
//!   degree-10 Taylor polynomial at r/2ᵏ (k ≤ 9 halvings bring it below 2⁻¹⁰), followed by
//!   k squarings in the form (1 + s)² − 1 = 2s + s². `exp2` splits off the nearest integer
//!   instead, so `exp2` of an integer is exact.
//! * `ln`: one Newton step from the `f64` logarithm x₀. With t = a·e^(−x₀) − 1,
//!   ln a = x₀ + ln(1 + t) ≈ x₀ + t − t²/2; the t²/2 term removes the quadratic error of the
//!   plain step x₀ + t, which reaches several 2⁻¹⁰⁰ relative when |ln a| is in the hundreds.
//!   Near 1 it uses the series ln(1 + x) = 2 atanh(x / (2 + x)) on the exact x − 1.
//! * `sqrt`: Karp & Markstein, √a ≈ y + (a − y²)·(1/√a)/2 with y² computed exactly.
//! * `cbrt`: one correction of the `f64` cube root y with its second-order term,
//!   ∛a ≈ y + d − d²/y where d = (a − y³)/(3y²) is formed in `f64`. The truncation error is
//!   cubic in the error of y, and rounding d adds about 4k·u² for a start k ulps off, so a
//!   platform `cbrt` a few ulps off still meets the bound below.
//!
//! Range: `sqrt`, `cbrt`, `ln` and `powi` rescale by exact powers of two, so subnormal and huge
//! arguments lose no precision; `exp`, `exp2`, `sinh`, `cosh` and `powf` overflow exactly
//! where the rounded result exceeds `f64::MAX`.
//!
//! Accuracy (relative error, verified against Arb on exact double-double inputs whose results
//! have magnitude at least 2⁻⁹⁶⁹; below that the low word is subnormal):
//!
//! | Functions | Bound |
//! |---|---|
//! | `sqrt`, `exp`, `exp_m1`, `ln`, `ln_1p`, `log2`, `log10`, `cbrt`, `tanh`, `asinh`, `acosh`, `atanh`, `hypot` | 2⁻¹⁰⁰ |
//! | `sinh`, `cosh` | 2⁻⁹⁹ |
//! | `exp2(x)` | 2⁻¹⁰⁰·max(1, \|x\|) |
//! | `powf(x, y)` | 2⁻⁹⁹·max(1, \|y ln \|x\|\|) |
//! | `powi(x, n)` | 16(\|n\| + 1)u², u = 2⁻⁵³ |
//!
//! `exp` is validated for x ∈ [−670, 709.78]. Special values (zeros, infinities, NaN,
//! out-of-domain arguments) follow `f64`, including the sign of zero: for example
//! `sinh(-0) = -0`, `exp_m1(-inf) = -1`, `ln_1p(-1) = -inf` and `atanh(-1) = -inf`, and `powf`
//! follows the C99 rules of `f64::powf` (`pow(-8, 1/3)` is NaN, `pow(-inf, 0.5)` is `inf`).
//! Zero results are `(±0, +0)`.
//!
//! `ln` and `cbrt` start from the platform's `f64::ln` and `f64::cbrt`; the result is within
//! the bounds above whatever their last-bit behaviour, but its last bits may differ between
//! platforms.

use super::consts::{INV_FACT, LN_2, LN_2_TAIL, LOG2_E, LOG10_E};
use super::{DoubleDouble as Dd, pow2, two_prod};

/// |x| below which ln(1 + x) uses the atanh series (and ln(a) uses it for |a − 1| below this).
const LN_1P_SERIES_LIMIT: f64 = 0.25;
/// ln(2)/2: below this |x|, exp_m1 avoids the cancellation in exp(x) − 1.
const HALF_LN_2: f64 = 0.34657359027997264;
/// For x in [EXP_FAST_MIN, EXP_FAST_MAX], eˣ = 2ᵐ·eʳ with −967 <= m <= 1021: scaling by 2ᵐ
/// is exact and keeps the result normalised (it is at least 2⁻⁹⁶⁸, so the low word cannot
/// round to a subnormal ulp(hi)/2). This is the validated domain of `exp`.
const EXP_FAST_MIN: f64 = -670.0;
/// Upper end of the fast range of `exp`.
const EXP_FAST_MAX: f64 = 708.0;
/// For x in [EXP2_FAST_MIN, EXP2_FAST_MAX], 2ˣ needs no over- or underflow handling (as for exp).
const EXP2_FAST_MIN: f64 = -966.0;
/// Upper end of the fast range of `exp2`.
const EXP2_FAST_MAX: f64 = 1022.0;
/// Beyond ±746, eˣ/2 overflows and eˣ underflows to zero.
const EXP_WIDE_LIMIT: f64 = 746.0;
/// Above this |x|, e^(−|x|) < 2⁻¹¹⁵·e^|x|, so sinh and cosh are e^|x|/2 to working precision.
const HYPERBOLIC_EXP_ONLY: f64 = 40.0;
/// ln uses its Newton step directly for leading words in this range (|x₀| <= 622).
const LN_NEWTON_MIN: f64 = 1e-270;
/// Upper end of the direct Newton range of ln.
const LN_NEWTON_MAX: f64 = 1e270;
/// Above this, asinh(x) = ln(2x) and acosh(x) = ln(2x) to working precision (and x² overflows).
const LN_2X_ABOVE: f64 = 1e150;
/// `sqrt` and `cbrt` need no rescaling for leading words in [2⁻⁹⁰⁰, 2¹⁰⁰⁰]: y² and y³ and
/// their rounding errors are then normal and finite.
const ROOT_MIN: f64 = two_pow(-900);
/// Upper end of the unscaled range of `sqrt` and `cbrt`.
const ROOT_MAX: f64 = two_pow(1000);
/// Rescaling exponent for `sqrt` and `cbrt` outside [ROOT_MIN, ROOT_MAX] (divisible by 2 and 3).
const ROOT_RESCALE: i32 = 600;
/// `powi` uses plain binary powering when all intermediate powers lie in [2⁻⁹⁶⁰, 2⁹⁶⁰].
const POWI_PLAIN_EXPONENT: i64 = 960;
/// `powf` of a negative base with an integer exponent of at most this magnitude uses `powi`
/// (exact for small powers, and 16(|n| + 1)u² <= 2⁻⁹⁹ here); larger ones use exp and ln.
const POWF_POWI_LIMIT: f64 = 7.0;

/// 2ᵉ for −1022 <= e <= 1023.
const fn two_pow(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The unbiased binary exponent field of `x`: ⌊log₂|x|⌋ for normal `x`, −1023 for zero and
/// subnormals, 1024 for infinities and NaN.
#[inline]
fn exponent(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// x·2ᵉ for any `e`, normalised, with a zero result returned as (±0, +0).
fn scale(x: Dd, e: i64) -> Dd {
    let r = x.ldexp(e.clamp(-2200, 2200) as i32);
    if r.hi == 0.0 {
        return Dd { hi: r.hi, lo: 0.0 };
    }
    // Below 2⁻⁹⁶⁸ the low word may round to exactly ulp(hi)/2 with the tie away from hi.
    Dd::new(r.hi, r.lo)
}

/// eʳ − 1 for |r| < 2⁻¹⁰ (truncation error below 2⁻¹²⁵ relative).
#[inline]
fn expm1_kernel(r: Dd) -> Dd {
    let mut p = INV_FACT[10];
    for k in (2..10).rev() {
        p = p * r + INV_FACT[k];
    }
    (p * r + 1.0) * r
}

/// eˣ − 1 for |x| < 1/2: the kernel at x/2ᵏ, then k squarings of 1 + s.
#[inline]
fn expm1_small(x: Dd) -> Dd {
    // |x| < 2^(e + 1), so k = e + 11 halvings bring it below 2^-10; k <= 9 as |x| < 1/2.
    let k = (exponent(x.hi) + 11).clamp(0, 9);
    let mut s = expm1_kernel(x.mul_pow2(two_pow(-k)));
    for _ in 0..k {
        s = s.mul_pow2(2.0) + s * s;
    }
    s
}

/// eˣ = 2ᵐ·p for finite |x| <= 746, with p = eʳ, |r| <= ln(2)/2 + tiny.
#[inline]
fn exp_reduced(x: Dd) -> (Dd, i32) {
    let m = (x.hi * LOG2_E.hi()).round();
    let (a, b) = two_prod(m, LN_2.hi());
    let (c, d) = two_prod(m, LN_2.lo());
    let r = x - Dd::from_parts_unchecked(a, b) - Dd::from_parts_unchecked(c, d) - m * LN_2_TAIL;
    (expm1_small(r) + 1.0, m as i32)
}

/// eˣ·2ᵏ (k = 0 or −1) outside the fast range of `exp` (or NaN): overflows and underflows
/// exactly where the rounded result does.
#[cold]
#[inline(never)]
fn exp_wide(x: Dd, k: i32) -> Dd {
    let h = x.hi;
    if h > EXP_WIDE_LIMIT {
        return Dd::INFINITY;
    }
    if h < -EXP_WIDE_LIMIT {
        return Dd::ZERO;
    }
    if h.is_nan() {
        return Dd::NAN;
    }
    let (p, m) = exp_reduced(x);
    scale(p, i64::from(m + k))
}

/// e^a / 2 for a > 40 or NaN, finite up to ln(2·MAX).
#[inline]
fn exp_half(a: Dd) -> Dd {
    if a.hi <= EXP_FAST_MAX {
        let (p, m) = exp_reduced(a);
        return p.mul_pow2(pow2(m - 1));
    }
    exp_wide(a, -1)
}

/// ln(1 + x) for |x| < LN_1P_SERIES_LIMIT (or x = ±0): with z = x/(2 + x),
/// 2 atanh(z) = 2z·(1 + z²/3 + z⁴/5 + …), where 2z is formed directly at the scale of x.
fn ln_1p_series(x: Dd) -> Dd {
    let w = x.mul_pow2(2.0) / (x + 2.0); // 2z
    let z2 = (w * w).mul_pow2(0.25); // z² <= (1/7)²
    let mut power = z2;
    let mut sum = Dd::ONE;
    let mut k = 3.0;
    loop {
        let t = power / k;
        sum += t;
        // t >= 0 and sum >= 1; the cap is never reached for |z| < 1/7 (about 20 terms).
        if t.hi <= 1e-34 || k > 121.0 {
            break;
        }
        power *= z2;
        k += 2.0;
    }
    w * sum
}

/// ln a for leading words in [LN_NEWTON_MIN, LN_NEWTON_MAX], from x₀ ≈ ln a.
#[inline]
fn ln_newton(a: Dd, x0: f64) -> Dd {
    // t = a·e^(−x₀) − 1 = e^δ − 1 with δ = ln a − x₀, a few ulps of x₀ (|δ| < 2⁻⁴⁰).
    let t = a * Dd::from_f64(-x0).exp() - 1.0;
    // ln a = x₀ + ln(1 + t) = x₀ + t − t²/2 + O(t³); t² is needed only to f64 precision.
    t.add_f64(-0.5 * t.hi * t.hi) + x0
}

/// ln a outside the direct Newton range: rescales tiny and huge arguments; special values.
#[cold]
#[inline(never)]
fn ln_wide(a: Dd) -> Dd {
    let x = a.hi;
    if x > 0.0 && x < f64::INFINITY {
        let k = if x < 1.0 { 1000 } else { -1000 };
        return a.ldexp(k).ln() - LN_2 * f64::from(k);
    }
    if x == 0.0 {
        return Dd::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return Dd::INFINITY;
    }
    Dd::NAN
}

/// √x for leading words in [ROOT_MIN, ROOT_MAX].
#[inline]
fn sqrt_kernel(x: Dd) -> Dd {
    let inv = 1.0 / x.hi.sqrt();
    let y = x.hi * inv;
    let (p, e) = two_prod(y, y);
    let resid = (x - Dd::from_parts_unchecked(p, e)).hi;
    Dd::new(y, resid * inv * 0.5)
}

/// √x outside [ROOT_MIN, ROOT_MAX]: special values, and exact rescaling of tiny and huge x.
#[cold]
#[inline(never)]
fn sqrt_wide(x: Dd) -> Dd {
    let h = x.hi;
    if h == 0.0 {
        return Dd::from_f64(h); // ±0
    }
    if h.is_nan() || h < 0.0 {
        return Dd::NAN;
    }
    if h == f64::INFINITY {
        return Dd::INFINITY;
    }
    let k = if h < 1.0 { ROOT_RESCALE } else { -ROOT_RESCALE };
    sqrt_kernel(x.ldexp(k)).ldexp(-k / 2)
}

/// ∛x for |leading word| in [ROOT_MIN, ROOT_MAX], from y ≈ ∛x.
#[inline]
fn cbrt_kernel(x: Dd, y: f64) -> Dd {
    // y³ = (y·y exactly)·y with relative error <= 2u², so r = x − y³ is accurate to 2u²|x|.
    let r = (x - Dd::from_f64(y) * y * y).hi;
    let d = r / (3.0 * y * y);
    // ∛(y³ + r) = y·(1 + δ)^(1/3) = y + d − d²/y + O(δ³), δ = r/y³ = 3d/y.
    Dd::new(y, d - d * d / y)
}

/// ∛x outside [ROOT_MIN, ROOT_MAX]: special values, and exact rescaling of tiny and huge x.
#[cold]
#[inline(never)]
fn cbrt_wide(x: Dd) -> Dd {
    let h = x.hi;
    if h == 0.0 || !h.is_finite() {
        return Dd::from_f64(h.cbrt()); // ±0, ±inf, NaN
    }
    let k = if h.abs() < 1.0 {
        ROOT_RESCALE
    } else {
        -ROOT_RESCALE
    };
    let x = x.ldexp(k);
    cbrt_kernel(x, x.hi.cbrt()).ldexp(-k / 3)
}

/// Binary powering, for arguments whose intermediate powers cannot over- or underflow.
#[inline]
fn powi_plain(x: Dd, n: i32) -> Dd {
    let mut result = Dd::ONE;
    let mut base = x;
    let mut e = n.unsigned_abs();
    while e > 0 {
        if e & 1 == 1 {
            result *= base;
        }
        e >>= 1;
        if e > 0 {
            base *= base;
        }
    }
    if n < 0 { result.recip() } else { result }
}

/// (m, e) with x = m·2ᵉ and 1 <= |m.hi| < 2, for finite nonzero x.
fn frexp(x: Dd) -> (Dd, i64) {
    let (x, offset) = if x.hi.abs() < f64::MIN_POSITIVE {
        (x.mul_pow2(two_pow(54)), -54) // subnormal (its low word is zero)
    } else {
        (x, 0)
    };
    let e = exponent(x.hi);
    (x.ldexp(-e), i64::from(e + offset))
}

/// Moves the exponent of m (1 <= |m.hi| < 4) into e, leaving 1 <= |m.hi| < 2.
#[inline]
fn renormalise(m: Dd, e: i64) -> (Dd, i64) {
    let k = exponent(m.hi); // 0 or 1
    (m.mul_pow2(two_pow(-k)), e + i64::from(k))
}

/// Binary powering on significands with the binary exponent carried separately, so that no
/// intermediate power over- or underflows; special values as `f64::powi`.
#[cold]
#[inline(never)]
fn powi_wide(x: Dd, n: i32) -> Dd {
    let h = x.hi;
    if h == 0.0 || !h.is_finite() {
        return Dd::from_f64(h.powi(n)); // ±0, ±inf, NaN (n != 0 here)
    }
    let (mut base, mut base_exp) = frexp(x);
    let (mut result, mut result_exp) = (Dd::ONE, 0_i64);
    let mut e = n.unsigned_abs();
    loop {
        if e & 1 == 1 {
            (result, result_exp) = renormalise(result * base, result_exp + base_exp);
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        (base, base_exp) = renormalise(base * base, 2 * base_exp);
    }
    if n < 0 {
        result = result.recip();
        result_exp = -result_exp;
    }
    scale(result, result_exp)
}

/// Whether the integer-valued `y` is odd. Both words of an integer double-double are integers.
fn is_odd(y: Dd) -> bool {
    (y.hi % 2.0 != 0.0) != (y.lo % 2.0 != 0.0)
}

/// `x.powf(y)` unless x is positive and finite and y finite: the C99 special cases of
/// `f64::powf`, and negative bases with integer exponents.
#[cold]
#[inline(never)]
fn powf_special(x: Dd, y: Dd) -> Dd {
    if y.hi == 0.0 || x == Dd::ONE {
        return Dd::ONE; // even for a NaN base or exponent, as for f64
    }
    if x.hi.is_nan() || y.hi.is_nan() {
        return Dd::NAN;
    }
    let ax = x.abs();
    if y.hi.is_infinite() {
        if ax == Dd::ONE {
            return Dd::ONE;
        }
        return if (ax > Dd::ONE) == (y.hi > 0.0) {
            Dd::INFINITY
        } else {
            Dd::ZERO
        };
    }
    // Now y is finite and nonzero, and x is +inf, ±0, -inf or negative.
    let integer = y == y.trunc();
    let magnitude = if x.hi == 0.0 || x.hi.is_infinite() {
        if (x.hi == 0.0) == (y.hi < 0.0) {
            Dd::INFINITY
        } else {
            Dd::ZERO
        }
    } else if !integer {
        return Dd::NAN;
    } else if y.hi.abs() <= POWF_POWI_LIMIT {
        return x.powi(y.hi as i32);
    } else {
        ax.powf(y)
    };
    if x.hi.is_sign_negative() && integer && is_odd(y) {
        -magnitude
    } else {
        magnitude
    }
}

impl Dd {
    /// Square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        if (ROOT_MIN..=ROOT_MAX).contains(&self.hi) {
            return sqrt_kernel(self);
        }
        sqrt_wide(self)
    }

    /// eˣ.
    #[inline]
    pub fn exp(self) -> Self {
        if (EXP_FAST_MIN..=EXP_FAST_MAX).contains(&self.hi) {
            let (p, m) = exp_reduced(self);
            return p.mul_pow2(pow2(m));
        }
        exp_wide(self, 0)
    }

    /// eˣ − 1, accurate near zero.
    pub fn exp_m1(self) -> Self {
        if self.hi.abs() < HALF_LN_2 {
            return expm1_small(self);
        }
        self.exp() - 1.0
    }

    /// Natural logarithm.
    pub fn ln(self) -> Self {
        let x = self.hi;
        if (x - 1.0).abs() < LN_1P_SERIES_LIMIT {
            return ln_1p_series(self - 1.0);
        }
        if (LN_NEWTON_MIN..=LN_NEWTON_MAX).contains(&x) {
            return ln_newton(self, x.ln());
        }
        ln_wide(self)
    }

    /// ln(1 + x), accurate near zero.
    pub fn ln_1p(self) -> Self {
        if self.hi.abs() < LN_1P_SERIES_LIMIT {
            return ln_1p_series(self);
        }
        (self + 1.0).ln()
    }

    /// 2ˣ (exact for integer x).
    pub fn exp2(self) -> Self {
        let x = self.hi;
        if (EXP2_FAST_MIN..=EXP2_FAST_MAX).contains(&x) {
            let n = x.round();
            let p = expm1_small((self - n) * LN_2) + 1.0;
            return p.mul_pow2(pow2(n as i32));
        }
        exp2_wide(self)
    }

    /// Base-2 logarithm.
    pub fn log2(self) -> Self {
        self.ln() * LOG2_E
    }

    /// Base-10 logarithm.
    pub fn log10(self) -> Self {
        self.ln() * LOG10_E
    }

    /// Logarithm in an arbitrary base.
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    /// Integer power by binary powering.
    ///
    /// Intermediate powers carry their binary exponent separately when they would leave the
    /// range, so the result is accurate wherever it is (unlike `f64::powi`, which gives
    /// `2.0.powi(-1074) == 0`). Zeros, infinities and NaN behave as for `f64::powi`.
    pub fn powi(self, n: i32) -> Self {
        // |x| < 2^k with k = |exponent| + 1; plain powering keeps every power within 2^±960.
        let k = i64::from(exponent(self.hi).abs()) + 1;
        if k * i64::from(n.unsigned_abs()) <= POWI_PLAIN_EXPONENT {
            return powi_plain(self, n);
        }
        powi_wide(self, n)
    }

    /// `self` raised to a real power (negative bases only for integer exponents).
    pub fn powf(self, y: Self) -> Self {
        if self.hi > 0.0 && self.hi < f64::INFINITY && y.hi.is_finite() {
            return (y * self.ln()).exp();
        }
        powf_special(self, y)
    }

    /// Cube root.
    pub fn cbrt(self) -> Self {
        if (ROOT_MIN..=ROOT_MAX).contains(&self.hi.abs()) {
            return cbrt_kernel(self, self.hi.cbrt());
        }
        cbrt_wide(self)
    }

    /// √(self² + other²) without intermediate overflow.
    pub fn hypot(self, other: Self) -> Self {
        let (a, b) = (self.abs(), other.abs());
        if a.hi.is_infinite() || b.hi.is_infinite() {
            return Self::INFINITY;
        }
        if a.hi.is_nan() || b.hi.is_nan() {
            return Self::NAN;
        }
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        if big.hi == 0.0 {
            return Self::ZERO;
        }
        let r = small / big;
        big * (r * r + 1.0).sqrt()
    }

    /// Hyperbolic sine.
    pub fn sinh(self) -> Self {
        let a = self.hi.abs();
        if a < 0.5 {
            let e = self.exp_m1();
            // `* 0.5` (not mul_pow2) renormalises results below 2⁻⁹⁶⁸ whose low word rounds.
            return e * (e + 2.0) / (e + 1.0) * 0.5;
        }
        if a <= HYPERBOLIC_EXP_ONLY {
            let ex = self.exp();
            return (ex - ex.recip()).mul_pow2(0.5);
        }
        let r = exp_half(self.abs());
        if self.hi < 0.0 { -r } else { r }
    }

    /// Hyperbolic cosine.
    pub fn cosh(self) -> Self {
        let a = self.abs();
        if a.hi <= HYPERBOLIC_EXP_ONLY {
            let ex = a.exp();
            return (ex + ex.recip()).mul_pow2(0.5);
        }
        exp_half(a)
    }

    /// Hyperbolic tangent.
    pub fn tanh(self) -> Self {
        if self.hi.abs() <= HYPERBOLIC_EXP_ONLY {
            let e = self.mul_pow2(2.0).exp_m1();
            return e / (e + 2.0);
        }
        if self.hi.is_nan() {
            return Self::NAN;
        }
        // 1 − |tanh x| = 2/(e^(2|x|) + 1) < 2⁻¹¹⁴.
        Dd::from_f64(1.0_f64.copysign(self.hi))
    }

    /// Inverse hyperbolic sine.
    pub fn asinh(self) -> Self {
        if !self.hi.is_finite() {
            return self;
        }
        let a = self.abs();
        let r = if a.hi > LN_2X_ABOVE {
            a.ln() + LN_2
        } else {
            let a2 = a * a;
            (a + a2 / ((a2 + 1.0).sqrt() + 1.0)).ln_1p()
        };
        if self.hi.is_sign_negative() { -r } else { r }
    }

    /// Inverse hyperbolic cosine (`self >= 1`).
    pub fn acosh(self) -> Self {
        if self.hi.is_nan() || self < Self::ONE {
            return Self::NAN;
        }
        if self.hi > LN_2X_ABOVE {
            return self.ln() + LN_2; // also +inf
        }
        let t = self - 1.0;
        (t + (t * (self + 1.0)).sqrt()).ln_1p()
    }

    /// Inverse hyperbolic tangent (`|self| <= 1`).
    pub fn atanh(self) -> Self {
        let a = self.abs();
        if a.hi.is_nan() || a >= Self::ONE {
            if a == Self::ONE {
                return Dd::from_f64(f64::INFINITY.copysign(self.hi));
            }
            return Self::NAN;
        }
        // atanh is odd; on |x| the argument 2|x|/(1 − |x|) of ln_1p never cancels against 1.
        let r = (a.mul_pow2(2.0) / (Self::ONE - a)).ln_1p() * 0.5; // `* 0.5`: see sinh
        if self.hi.is_sign_negative() { -r } else { r }
    }
}

/// 2ˣ outside the fast range (or NaN): overflows and underflows exactly where the rounded
/// result does.
#[cold]
#[inline(never)]
fn exp2_wide(x: Dd) -> Dd {
    let h = x.hi;
    if h > 1025.0 {
        return Dd::INFINITY;
    }
    if h < -1076.0 {
        return Dd::ZERO;
    }
    if h.is_nan() {
        return Dd::NAN;
    }
    let n = h.round();
    let p = expm1_small((x - n) * LN_2) + 1.0;
    scale(p, n as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ln` and `cbrt` start from the platform's `f64::ln` and `f64::cbrt`. Starting values
    /// several ulps off must still meet the 2⁻¹⁰⁰ bound with margin. Compared with the result
    /// from the platform's starting value: `ln` within 2⁻¹⁰² for starts 16 ulps off, `cbrt`
    /// within 2⁻¹⁰¹ for 4 ulps (its `f64` correction adds about 4k·u² for a start k ulps off).
    #[test]
    fn refinements_tolerate_inaccurate_starting_values() {
        let xs = [
            Dd::new(3.0, 1.1e-16),
            Dd::new(0.6, -2.3e-17),
            Dd::new(1.3, 4.1e-17),
            Dd::new(1e-200, 3.7e-217),
            Dd::new(7.5e250, -2.9e234),
            Dd::new(1e-269, 1.3e-286),
            Dd::new(9.99e269, 2.1e253),
        ];
        let (ln_tol, cbrt_tol) = (two_pow(-102), two_pow(-101));
        for x in xs {
            let (ln, cbrt) = (x.ln(), x.cbrt());
            for ulps in [-16.0, -4.0, -1.0, 1.0, 4.0, 16.0] {
                let x0 = x.hi.ln() * (1.0 + ulps * f64::EPSILON);
                let got = ln_newton(x, x0);
                let err = ((got - ln) / ln).abs().hi;
                assert!(err <= ln_tol, "ln {x:?} from {x0}: {got:?}");
                if ulps.abs() <= 4.0 {
                    let y0 = x.hi.cbrt() * (1.0 + ulps * f64::EPSILON);
                    let got = cbrt_kernel(x, y0);
                    let err = ((got - cbrt) / cbrt).abs().hi;
                    assert!(err <= cbrt_tol, "cbrt {x:?} from {y0}: {got:?}");
                }
            }
        }
    }
}
