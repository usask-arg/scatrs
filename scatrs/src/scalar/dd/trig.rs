//! Trigonometric functions for `DoubleDouble`.
//!
//! * `sin`, `cos`, `sin_cos`, `tan`: Cody–Waite reduction x = k·π/2 + r with a three-word π/2
//!   (the products k·P₁ and k·P₂ are exact), for |x| ≤ 2⁵⁰; then Taylor series in r to order 29
//!   (sin) and 28 (cos). Arguments with |x.hi| ≤ π/4 are not reduced. Elsewhere k is rounded
//!   from the `f64` product x.hi·(2/π), so |r| ≤ 1.09 (π/4 + 0.3 at worst, for |x| near 2⁵⁰),
//!   where both series are truncated below 2⁻¹⁰⁴. `tan` is sin/cos.
//! * `atan2`: one Newton step from the `f64` angle z = atan2(y.hi, x.hi). With R = √(x² + y²)
//!   and θ the true angle, y·cos z − x·sin z = R sin(θ − z) and x·cos z + y·sin z = R cos(θ − z),
//!   so θ ≈ z + (y·cos z − x·sin z)/(x·cos z + y·sin z): the quotient is tan(θ − z), whose
//!   error (θ − z)³/3 is negligible for a start many ulps off. The same formula serves every
//!   octant (there is no switch at |y| = |x|), and the denominator never cancels. Arguments
//!   outside [2⁻⁴⁵⁰, 2⁴⁵⁰] are first scaled by a power of two, which leaves the angle unchanged.
//!   `atan(x)` is `atan2(x, 1)`; `asin(x)` and `acos(x)` are `atan2(x, w)` and `atan2(w, x)` with
//!   w = √((1 − |x|)(1 + |x|)), where 1 − |x| is exact next to 1.
//!
//! Accuracy (verified against Arb on exact double-double inputs whose results have magnitude
//! at least 2⁻⁹⁶⁹; below that the low word is subnormal):
//!
//! | Functions | Bound |
//! |---|---|
//! | `sin`, `cos`, `sin_cos` | 2⁻¹⁰⁰ relative for \|x\| ≤ π/4; 2⁻¹⁰⁰ absolute for \|x\| ≤ 2⁵⁰ |
//! | `tan` | 2⁻⁹⁹ relative for \|x\| ≤ π/4; 2⁻⁹⁹(1 + \|tan x\|)/\|cos x\| absolute |
//! | `atan`, `atan2`, `asin`, `acos` | 2⁻⁹⁹ relative |
//!
//! Next to a nonzero multiple kπ/2 the reduction leaves an absolute error floor: at most
//! |k|·2⁻¹⁶¹ for `f64` arguments (from rounding k times the third word of π/2, and the fourth
//! word), and |k|·2⁻¹⁵⁸ for double-double ones (the first subtraction, x − k·P₁ − k·P₂, adds
//! 3u²·|k·P₂|). sin and cos are therefore within 2⁻¹⁰⁰ relative wherever |x − kπ/2| ≥ |k|·2⁻⁶⁰
//! (`f64`) or |k|·2⁻⁵⁷ (double-double), and within the floor closer in. At the doubles closest
//! to such multiples the measured relative error is 2⁻¹⁰⁰ at x = 45.553093477052
//! (|cos x| = 6.2·10⁻¹⁹, k = 29, the closest of all below 2⁵⁰), 2⁻⁸¹ at x = 14461176.67027838
//! and 2⁻⁶¹ at x = 410195257422896.8.
//!
//! Ranges: |sin x| ≤ 1 and |cos x| ≤ 1; `atan` and `asin` lie in [−π/2, π/2], `acos` in [0, π]
//! and `atan2` in [−π, π] (and in [−π/2, π/2] for x > 0), where π/2 and π are the
//! double-double constants, which lie above the true values. `atan2` clamps its result to these
//! limits, so it never passes them whatever the platform's starting value.
//!
//! Special values follow `f64`, including the sign of zero: `sin(-0) = -0`, `tan(-0) = -0`,
//! `atan(-0) = -0`, `asin(-0) = -0`, `acos(1) = +0`, `atan(±inf) = ±π/2`, and `atan2` for
//! every combination of signed zeros and infinities (giving 0, ±π/4, ±π/2, ±3π/4 or ±π as
//! double-double constants). `sin`, `cos` and `tan` are NaN for NaN, ±∞ and |x| > 2⁵⁰ (the
//! double-double value, so 2⁵⁰ + 0.1 is outside); `asin` and `acos` are NaN for |x| > 1.
//! `sin` and `tan` are odd and `cos` is even, bitwise up to the sign of a zero low word; so are
//! `atan`, `asin` and `atan2` (in y) wherever the platform's `f64::atan2` is odd in y.
//!
//! `atan2` starts from the platform's `f64::atan2`; the result is within the bounds above
//! whatever its last-bit behaviour, but its last bits may differ between platforms.

use super::consts::{FRAC_2_PI, FRAC_3PI_4, FRAC_PI_2, FRAC_PI_2_TAIL, FRAC_PI_4, INV_FACT, PI};
use super::{DoubleDouble as Dd, two_prod};

/// Largest |x| accepted by the argument reduction (2⁵⁰).
const MAX_REDUCIBLE: f64 = 1_125_899_906_842_624.0;
/// `atan2` uses its Newton step directly when |y.hi| and |x.hi| both lie in [2⁻⁴⁵⁰, 2⁴⁵⁰]. The
/// numerator's products y·cos z and x·sin z are then about min(|x|, |y|) ≥ 2⁻⁴⁵⁰ and the
/// result at least 2⁻⁹⁰¹, inside the full-precision range; in the denominator the larger
/// product is about max(|x|, |y|) ≤ 2⁴⁵⁰, and the smaller matters only when it is not tiny.
const ATAN2_FAST_MIN: f64 = f64::from_bits((1023 - 450) << 52);
/// Upper end of the direct range of `atan2`.
const ATAN2_FAST_MAX: f64 = f64::from_bits((1023 + 450) << 52);
/// After the larger argument of `atan2` is scaled into [1, 2), a smaller one below 2⁻⁹⁶⁸ puts
/// the angle within 2⁻⁹⁶⁷ of 0, ±π/2 or ±π: below the full-precision range near 0, and far
/// below the last bit of the double-double π/2 and π.
const ATAN2_TINY_RATIO: f64 = f64::from_bits((1023 - 968) << 52);

/// Returns (k, r) with x = k·π/2 + r, or `None` for NaN, ±∞ and |x| > 2⁵⁰.
#[inline]
fn reduce_pio2(x: Dd) -> Option<(i64, Dd)> {
    let a = x.hi.abs();
    if a <= FRAC_PI_4.hi() {
        return Some((0, x));
    }
    if a < MAX_REDUCIBLE || within_reduction_limit(x) {
        let k = (x.hi * FRAC_2_PI.hi()).round();
        let (a, b) = two_prod(k, FRAC_PI_2.hi());
        let (c, d) = two_prod(k, FRAC_PI_2.lo());
        let r = x
            - Dd::from_parts_unchecked(a, b)
            - Dd::from_parts_unchecked(c, d)
            - k * FRAC_PI_2_TAIL;
        return Some((k as i64, r));
    }
    None
}

/// Whether |x| ≤ 2⁵⁰ when |x.hi| ≥ 2⁵⁰ (or x is NaN): only for x.hi = ±2⁵⁰ with a low word
/// that is zero or points towards zero.
#[cold]
#[inline(never)]
fn within_reduction_limit(x: Dd) -> bool {
    x.hi.abs() == MAX_REDUCIBLE && x.lo * x.hi <= 0.0
}

/// sin r for |r| ≤ 1.09: r − r·r²·(1/3! − r²(1/5! − … r²/29!)).
fn sin_taylor(r: Dd) -> Dd {
    if r.hi == 0.0 {
        return r; // ±0: the series would give r − r³p = +0 for r = -0
    }
    let r2 = r * r;
    let mut p = INV_FACT[29];
    for k in (3..=27).rev().step_by(2) {
        p = INV_FACT[k] - r2 * p;
    }
    r - r * r2 * p
}

/// cos r for |r| ≤ 1.09: 1 − r²·(1/2! − r²(1/4! − … r²/28!)), never above 1.
fn cos_taylor(r: Dd) -> Dd {
    let r2 = r * r;
    let mut p = INV_FACT[28];
    for k in (2..=26).rev().step_by(2) {
        p = INV_FACT[k] - r2 * p;
    }
    Dd::ONE - r2 * p
}

/// IEEE special cases of atan2 (a zero or infinite argument), mapped to exact constants.
fn special_atan2(y: f64, x: f64) -> Dd {
    let a = y.atan2(x);
    let exact = [FRAC_PI_4, FRAC_PI_2, FRAC_3PI_4, PI]
        .into_iter()
        .find(|c| c.hi() == a.abs());
    match exact {
        Some(c) if a < 0.0 => -c,
        Some(c) => c,
        None => Dd::from_f64(a), // ±0 (or NaN)
    }
}

/// `limit` with the sign of `y`.
#[inline]
fn signed_as(limit: Dd, y: f64) -> Dd {
    if y.is_sign_negative() { -limit } else { limit }
}

/// atan2(y, x) by one Newton step from the `f64` angle, for finite nonzero arguments whose
/// products y·cos z, x·sin z, … stay inside the full-precision range.
#[inline]
fn atan2_newton(y: Dd, x: Dd) -> Dd {
    atan2_from(y, x, y.hi.atan2(x.hi))
}

/// One Newton step for atan2(y, x) from the starting angle z.
#[inline]
fn atan2_from(y: Dd, x: Dd, z: f64) -> Dd {
    let (s, c) = Dd::from_f64(z).sin_cos();
    // tan(θ − z) = R sin(θ − z) / R cos(θ − z), and tan δ = δ + δ³/3 + … for δ = θ − z.
    let t = (y * c - x * s) / (x * c + y * s);
    let r = t + z;
    // The double-double π/2 and π lie above the true values (x > 0 gives |θ| < π/2). From a
    // start a few ulps off, t is accurate far beyond that gap, so this acts only for starts
    // far below |θ|, whose step error δ³/3 is positive: it keeps the range whatever the start.
    let limit = if x.hi > 0.0 { FRAC_PI_2 } else { PI };
    if r.abs() > limit {
        return signed_as(limit, y.hi);
    }
    r
}

/// ⌊log₂ |v|⌋ for finite nonzero `v`, including subnormals.
fn exponent(v: f64) -> i32 {
    let bits = |v: f64| ((v.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if v.abs() < f64::MIN_POSITIVE {
        bits(v * f64::from_bits((1023 + 64) << 52)) - 64
    } else {
        bits(v)
    }
}

/// atan2 outside the direct range: NaN, zeros, infinities, and huge, tiny or extremely
/// unequal arguments.
#[cold]
#[inline(never)]
fn atan2_wide(y: Dd, x: Dd) -> Dd {
    if y.hi.is_nan() || x.hi.is_nan() {
        return Dd::NAN;
    }
    if y.hi == 0.0 || x.hi == 0.0 || y.hi.is_infinite() || x.hi.is_infinite() {
        return special_atan2(y.hi, x.hi);
    }
    // Scale the larger argument into [1, 2): exact, except that the smaller may lose bits
    // when it falls below 2⁻⁹⁶⁹, where only the branch below uses it.
    let e = exponent(y.hi.abs().max(x.hi.abs()));
    let (y, x) = (y.ldexp(-e), x.ldexp(-e));
    if y.hi.abs().min(x.hi.abs()) >= ATAN2_TINY_RATIO {
        return atan2_newton(y, x);
    }
    // The smaller is below 2⁻⁹⁶⁸ (possibly zero after scaling).
    if y.hi.abs() < x.hi.abs() {
        if x.hi > 0.0 {
            return y / x; // atan(y/x) = y/x to within (y/x)³/3; keeps the sign of zero
        }
        return signed_as(PI, y.hi);
    }
    signed_as(FRAC_PI_2, y.hi)
}

/// √(1 − a²) for 0 ≤ a ≤ 1, as √((1 − a)(1 + a)): 1 − a is exact for a ≥ 1/2.
#[inline]
fn sqrt_one_minus_square(a: Dd) -> Dd {
    ((Dd::ONE - a) * (Dd::ONE + a)).sqrt()
}

impl Dd {
    /// Sine.
    ///
    /// Relative error ≤ 2⁻¹⁰⁰ for |x| ≤ π/4, absolute ≤ 2⁻¹⁰⁰ for |x| ≤ 2⁵⁰; NaN for |x| > 2⁵⁰.
    pub fn sin(self) -> Self {
        let Some((k, r)) = reduce_pio2(self) else {
            return Self::NAN;
        };
        match k.rem_euclid(4) {
            0 => sin_taylor(r),
            1 => cos_taylor(r),
            2 => -sin_taylor(r),
            _ => -cos_taylor(r),
        }
    }

    /// Cosine.
    ///
    /// Relative error ≤ 2⁻¹⁰⁰ for |x| ≤ π/4, absolute ≤ 2⁻¹⁰⁰ for |x| ≤ 2⁵⁰; NaN for |x| > 2⁵⁰.
    pub fn cos(self) -> Self {
        let Some((k, r)) = reduce_pio2(self) else {
            return Self::NAN;
        };
        match k.rem_euclid(4) {
            0 => cos_taylor(r),
            1 => -sin_taylor(r),
            2 => -cos_taylor(r),
            _ => sin_taylor(r),
        }
    }

    /// `(sin x, cos x)`, bitwise equal to `(x.sin(), x.cos())`.
    ///
    /// Relative error ≤ 2⁻¹⁰⁰ for |x| ≤ π/4, absolute ≤ 2⁻¹⁰⁰ for |x| ≤ 2⁵⁰; NaN for |x| > 2⁵⁰.
    pub fn sin_cos(self) -> (Self, Self) {
        let Some((k, r)) = reduce_pio2(self) else {
            return (Self::NAN, Self::NAN);
        };
        let (s, c) = (sin_taylor(r), cos_taylor(r));
        match k.rem_euclid(4) {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }

    /// Tangent.
    ///
    /// Relative error ≤ 2⁻⁹⁹ (|x| ≤ π/4), absolute ≤ 2⁻⁹⁹(1 + |tan x|)/|cos x|; NaN for |x| > 2⁵⁰.
    pub fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s / c
    }

    /// Four-quadrant arctangent of `self / x` (`self` is y), in [−π, π].
    ///
    /// Relative error ≤ 2⁻⁹⁹; signed zeros and infinities as for `f64::atan2`.
    pub fn atan2(self, x: Self) -> Self {
        let direct = |v: f64| (ATAN2_FAST_MIN..=ATAN2_FAST_MAX).contains(&v.abs());
        if direct(self.hi) && direct(x.hi) {
            return atan2_newton(self, x);
        }
        atan2_wide(self, x)
    }

    /// Arctangent, in [−π/2, π/2].
    ///
    /// Relative error ≤ 2⁻⁹⁹.
    pub fn atan(self) -> Self {
        self.atan2(Self::ONE)
    }

    /// Arcsine, in [−π/2, π/2].
    ///
    /// Relative error ≤ 2⁻⁹⁹; NaN for |x| > 1.
    pub fn asin(self) -> Self {
        let a = self.abs();
        if self.hi.is_nan() || a > Self::ONE {
            return Self::NAN;
        }
        self.atan2(sqrt_one_minus_square(a))
    }

    /// Arccosine, in [0, π].
    ///
    /// Relative error ≤ 2⁻⁹⁹; NaN for |x| > 1.
    pub fn acos(self) -> Self {
        let a = self.abs();
        if self.hi.is_nan() || a > Self::ONE {
            return Self::NAN;
        }
        sqrt_one_minus_square(a).atan2(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2^e for normal powers.
    fn two_pow(e: i32) -> f64 {
        f64::from_bits(((e + 1023) as u64) << 52)
    }

    /// `atan2` starts from the platform's `f64::atan2`, and its Newton step converges
    /// cubically: from starts up to 1024 ulps off (δ ≤ 2⁻⁴²|z|, error δ³/3) the result must
    /// agree with the one from the platform's start to within 2⁻¹⁰¹ relative (each is
    /// within a few u² of the true value), and stay within the range.
    #[test]
    fn newton_step_tolerates_inaccurate_starting_values() {
        let d = Dd::from_f64;
        let cases = [
            (d(1.0), d(1.0)),
            (d(1e-3), d(1.0)),
            (d(1.0), d(1e-3)),
            (d(-0.3), d(-2.0)),
            (d(2.5), d(-1e-20)),
            (d(1e-30), d(-1.0)),
            (d(-1e-30), d(-1.0)),
            (d(1.0), d(1e-30)),
            (d(-1.0), d(1e-30)),
            (Dd::new(0.7, 1e-17), Dd::new(-0.2, 3e-18)),
            (Dd::new(1e-200, -3e-217), Dd::new(4e-201, 1e-217)),
        ];
        for (y, x) in cases {
            let want = y.atan2(x);
            let limit = if x.hi > 0.0 { FRAC_PI_2 } else { PI };
            let z0 = y.hi.atan2(x.hi);
            for ulps in [-1024.0, -16.0, -1.0, 1.0, 16.0, 1024.0] {
                let z = z0 * (1.0 + ulps * f64::EPSILON);
                let got = atan2_from(y, x, z);
                let err = ((got - want) / want).abs().hi;
                assert!(
                    err <= two_pow(-101),
                    "atan2({y:?}, {x:?}) from {z}: {got:?}"
                );
                assert!(got.abs() <= limit, "atan2({y:?}, {x:?}) from {z}: {got:?}");
            }
        }
    }

    /// A start far below |θ| next to ±π/2 or ±π leaves the step's positive error δ³/3, which
    /// would carry the result past the limit: the clamp returns the limit instead.
    #[test]
    fn newton_step_never_passes_the_range_limits() {
        let d = Dd::from_f64;
        for s in [1.0, -1.0] {
            for (y, x, limit) in [(d(s * 1e-30), d(-1.0), PI), (d(s), d(1e-30), FRAC_PI_2)] {
                let z = y.hi.atan2(x.hi) * (1.0 - two_pow(-20));
                let unclamped = {
                    let (sz, cz) = Dd::from_f64(z).sin_cos();
                    (y * cz - x * sz) / (x * cz + y * sz) + z
                };
                assert!(unclamped.abs() > limit, "{unclamped:?}");
                assert_eq!(atan2_from(y, x, z), signed_as(limit, s));
            }
        }
    }
}
