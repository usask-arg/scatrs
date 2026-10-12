//! Double-double arithmetic: a value is the unevaluated sum `hi + lo` of two `f64`s,
//! normalised so that `hi == fl(hi + lo)`, giving about 106 significant bits.
//!
//! Basic operations are the algorithms of Joldes, Muller & Popescu, "Tight and rigorous
//! error bounds for basic building blocks of double-word arithmetic", ACM TOMS 44(2), 2017.
//! With u = 2⁻⁵³ the relative error bounds are: `+`/`-` 3u², `+ f64` 2u², `*` 4u²,
//! `* f64` 2u², `/` 10u², `/ f64` 3u². They rely on a correctly rounded fused multiply-add,
//! which `f64::mul_add` provides on every platform (in software where hardware lacks FMA).
//!
//! Non-finite results are returned as `(hi, 0.0)` so that infinities do not become NaN, and a
//! zero result is `(±0.0, +0.0)`, with the IEEE 754 sign of the `f64` operation on the
//! leading words.
//! Full relative precision requires magnitudes above about 2⁻⁹⁶⁹ ≈ 2·10⁻²⁹².
//! A result within about an ulp of `f64::MAX` may overflow early: for example `add_dd` adds
//! the leading words first, so it overflows when `hi + hi` reaches the overflow threshold
//! (`MAX` plus half an ulp, rounded to infinity) even if the trailing words would bring the
//! sum back below it.
//!
//! Speed: `f64::mul_add` is a single instruction where FMA is enabled, but on x86_64 builds
//! without the `fma` target feature (the default) it is a much slower libm call; build with
//! `-C target-feature=+fma` or `-C target-cpu=native` where the hardware has FMA.

use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

mod consts;
mod elementary;
mod parse;
mod rem;
mod traits;
mod trig;

pub use parse::ParseDoubleDoubleError;

/// A double-double number: the exact sum `hi + lo` with `hi == fl(hi + lo)`.
///
/// The error bounds stated on its methods hold for results of magnitude at least 2⁻⁹⁶⁹; below
/// that the low word is subnormal and precision degrades gradually.
///
/// # Use as `num_traits::Float`
///
/// `DoubleDouble` implements `num_traits::Float`, `FloatConst`, `NumCast`, `ToPrimitive` and
/// [`Real`](crate::Real), so generic kernels and `num_complex::Complex<DoubleDouble>` work.
/// `Float` delegates arithmetic and the elementary functions to the inherent methods, with their
/// documented accuracy. Rounding (`floor`, `ceil`, `round`, `trunc`, `fract`), `abs`, `signum`,
/// `copysign`, `max`, `min`, `clamp` and `abs_sub` act on the whole double-double;
/// classification (`classify`, `is_normal`, `is_subnormal`, …), the sign tests and
/// `integer_decode` describe the leading word. Special values and signed zeros follow `f64`
/// (the tests compare these methods with `f64` on `f64` arguments, bit for bit where the `f64`
/// result is exact). Where a double-double differs from `f64`:
///
/// * `epsilon()` is 2⁻¹⁰⁴, a unit rather than a bound on the error of one operation (a division
///   errs by up to 10u² ≈ 2.5·epsilon, u = 2⁻⁵³), and `min_positive_value()` is 2⁻⁹⁶⁹, the
///   smallest magnitude with full precision; `classify` and `is_normal` still call values in
///   [2⁻¹⁰²², 2⁻⁹⁶⁹) normal.
/// * `mul_add` is not fused: the product is rounded to a double-double before the addition, so
///   the error is up to 4u²|x·a| + 3u²|x·a + b|. The product of two doubles is exact, so for
///   `f64` arguments it is the fused result to double-double precision. Special values, signed
///   zeros, overflow (never of the product alone) and subnormal results follow `f64::mul_add`.
/// * `max` and `min` ignore a NaN argument (even a signaling one, for which aarch64's `f64::max`
///   returns NaN) and, of two zeros, follow IEEE 754-2019 maximum and minimum:
///   max(−0, +0) = +0 and min(−0, +0) = −0 in either order (`f64` leaves this unspecified).
/// * `clamp` returns NaN instead of panicking when `min > max` or a bound is NaN.
/// * `integer_decode` decodes the leading word only.
///
/// `ToPrimitive` truncates towards zero exactly and returns `None` outside the target range, as
/// for `f64`; `to_f64` is the leading word (the correctly rounded double) and `to_f32` also
/// rounds once. `NumCast::from` is exact for `f32`, `f64` and primitive integers below 2¹⁰⁶ in
/// magnitude, and within u²·|n| of wider integers; from another `DoubleDouble` it keeps only the
/// leading word (or the integer part beyond 2⁵³).
///
/// Parsing: [`from_decimal_str`](Self::from_decimal_str) and `Num::from_str_radix` (radix 10)
/// ignore surrounding whitespace; `str::parse` (`FromStr`) does not, accepting exactly the
/// strings that `f64` accepts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoubleDouble {
    hi: f64,
    lo: f64,
}

/// Error-free sum: `s = fl(a + b)` and `s + e = a + b` exactly (Knuth's 2Sum), unless an
/// intermediate overflows.
///
/// The intermediate `fl(s - a)` is `b + δ` rounded, with `|δ| <= ulp(s)/2 <= 2^970`, so it
/// can overflow (giving `e = NaN` although `s` is finite) only when `|b| = MAX`, e.g.
/// `a = 8.2e307`, `b = -MAX`; that is also the only failure seen in millions of tests near
/// `MAX`. Use [`two_sum_safe`] when `b` may be `±MAX`; the hot paths of `add_dd` and
/// `add_f64` instead check the error word, which they need to inspect anyway.
#[inline]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// [`two_sum`] for arbitrary operands, including `±MAX`: exact whenever `fl(a + b)` is finite.
#[inline]
pub(crate) fn two_sum_safe(a: f64, b: f64) -> (f64, f64) {
    let (s, e) = two_sum(a, b);
    if !e.is_finite() && s.is_finite() {
        return two_sum_ordered(a, b);
    }
    (s, e)
}

/// The rare case where 2Sum's intermediate overflowed although `fl(a + b)` is finite: Fast2Sum
/// with the larger-magnitude operand first is exact and cannot overflow then.
#[cold]
#[inline(never)]
fn two_sum_ordered(a: f64, b: f64) -> (f64, f64) {
    if a.abs() >= b.abs() {
        fast_two_sum(a, b)
    } else {
        fast_two_sum(b, a)
    }
}

/// Error-free sum, valid when `|a| >= |b|` or `a == 0`, and more generally whenever `a` is an
/// integer multiple of `ulp(b)` (the grid condition).
///
/// `div_dd` relies on the general form: for normal `y.hi`, `rh = 1 - y.hi·th` is exact and a
/// multiple of `ulp(y.hi)·ulp(th) >= 2^-105`, while `|rl| <= 2^-53` gives `ulp(rl) <= 2^-105`,
/// so `rh` lies on the grid of `rl` even when `|rh| < |rl|`.
#[inline]
pub(crate) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Error-free product: `p = fl(a * b)` and `p + e = a * b` exactly.
#[inline]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Above this magnitude a divisor's reciprocal, and the correction terms of `div_dd`, are
/// close enough to the subnormal range to lose accuracy, so `div_dd` rescales first. The
/// rounding errors of subnormal intermediates are at most 2⁻¹⁰⁷⁵ absolute, below
/// 2⁻¹¹⁵ = u²/512 relative to `1/y` for `|y| <= 2^960`.
const DIV_RESCALE_ABOVE: f64 = f64::from_bits((960 + 1023) << 52);

/// 2⁻⁹⁶⁸: `mul_pow2` renormalises scaled-down results below this magnitude.
const RENORMALIZE_BELOW: f64 = f64::from_bits((1023 - 968) << 52);

/// `2^e` for `-1022 <= e <= 1023`.
#[inline]
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// 2¹²⁸.
const TWO_POW_128: f64 = f64::from_bits((1023 + 128) << 52);

/// `n` as a double-double: exact below 2¹⁰⁶, and within u²·n above (u = 2⁻⁵³).
///
/// The leading word is the nearest double and the residual is rounded once, then the pair is
/// renormalised. Next to a tie of the leading word that can move it: n = 2¹²⁶ + 2⁷³ + 1 rounds to
/// the double 2¹²⁶ + 2⁷⁴, the residual −(2⁷³ − 1) to −2⁷³, and the pair renormalises to
/// (2¹²⁶, 2⁷³), which is off by 1 (and is the nearest double-double).
pub(crate) fn from_u128(n: u128) -> DoubleDouble {
    if n >> 53 == 0 {
        return DoubleDouble::from_f64(n as u64 as f64);
    }
    let hi = n as f64; // rounded to nearest, at most 2^128
    let residual = if hi == TWO_POW_128 {
        -((u128::MAX - n + 1) as f64) // n − 2^128, at most 2^75 in magnitude
    } else {
        let h = hi as u128; // exact
        if n >= h {
            (n - h) as f64
        } else {
            -((h - n) as f64)
        }
    };
    DoubleDouble::new(hi, residual)
}

/// v·2⁻ˢ for |v| < 2¹⁰²³ and 54 <= s <= 1000, keeping the leading word the correctly rounded
/// double of the value (`mul_pow2` instead rounds the trailing word to the subnormal grid and
/// renormalises, which can move the leading word when that rounding reaches half an ulp of it).
///
/// A subnormal result is rounded once by [`round_to_subnormal`]. Otherwise the leading word
/// scales exactly and the trailing word is rounded to the subnormal grid, then stepped once
/// towards zero if it reached half an ulp of an odd leading word (or a whole ulp, in the
/// lowest binade, where it becomes zero); the trailing word's error is then at most 2⁻¹⁰⁷⁴.
pub(crate) fn scale_down_tiny(v: DoubleDouble, s: i32) -> DoubleDouble {
    if v.hi.abs() < pow2(s - 1022) {
        return DoubleDouble::from_f64(round_to_subnormal(v, s));
    }
    let hi = v.hi * pow2(-s);
    let mut lo = v.lo * pow2(-s);
    if hi + lo != hi {
        lo -= f64::from_bits(1).copysign(lo);
    }
    DoubleDouble::from_parts_unchecked(hi, if lo == 0.0 { 0.0 } else { lo })
}

/// (v.hi + v.lo)·2⁻ˢ rounded once to the nearest subnormal (or ±2⁻¹⁰²²), ties to even, for
/// |v| < 2^(s − 1022) and 54 <= s <= 1000; the sign of v is kept, also when the result is zero.
///
/// Scaling the leading word alone rounds once, but can land on the wrong side when the trailing
/// word decides a tie (or near-tie) of the leading word. With q the scaled leading word,
/// d = v.hi − q·2ˢ is that rounding's error, exact (both terms are multiples of ulp(v.hi) and
/// |d| is at most half the scaled subnormal spacing G = 2^(s − 1074)); the true residual
/// d + v.lo is then compared with G/2. As ulp(v.hi) ≤ G, |v.lo| ≤ G/2 and the residual is below
/// G, so q moves by at most one step.
fn round_to_subnormal(v: DoubleDouble, s: i32) -> f64 {
    const STEP: f64 = f64::from_bits(1); // 2^-1074
    if v.hi.is_sign_negative() {
        return -round_to_subnormal(-v, s);
    }
    let q = v.hi * pow2(-s);
    let d = v.hi - q * pow2(s);
    let (r, r_lo) = two_sum(d, v.lo); // the residual r + r_lo, exactly
    let half = pow2(s - 1075);
    let above = r > half || (r == half && r_lo > 0.0);
    let below = r < -half || (r == -half && r_lo < 0.0);
    let tie = r.abs() == half && r_lo == 0.0 && q.to_bits() & 1 == 1;
    if above || (tie && r > 0.0) {
        q + STEP
    } else if below || (tie && r < 0.0) {
        q - STEP
    } else {
        q
    }
}

impl DoubleDouble {
    /// Zero.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    /// One.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    /// Not a number.
    pub const NAN: Self = Self {
        hi: f64::NAN,
        lo: 0.0,
    };
    /// Positive infinity.
    pub const INFINITY: Self = Self {
        hi: f64::INFINITY,
        lo: 0.0,
    };
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self {
        hi: f64::NEG_INFINITY,
        lo: 0.0,
    };

    /// Converts an `f64` exactly.
    #[inline]
    pub const fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// The exact value `hi + lo` of any two `f64`s, renormalised.
    #[inline]
    pub fn new(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum_safe(hi, lo);
        if !s.is_finite() {
            return Self { hi: s, lo: 0.0 };
        }
        Self { hi: s, lo: e }
    }

    /// Builds from parts already known to be normalised (constant tables).
    #[inline]
    pub(crate) const fn from_parts_unchecked(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// The leading word: the value rounded to `f64`.
    #[inline]
    pub const fn hi(self) -> f64 {
        self.hi
    }

    /// The trailing word.
    #[inline]
    pub const fn lo(self) -> f64 {
        self.lo
    }

    /// Fast2Sum of a result's words; an overflowing sum gives `(±inf, 0)`.
    #[inline]
    fn renormalize(hi: f64, lo: f64) -> Self {
        let (h, l) = fast_two_sum(hi, lo);
        if !h.is_finite() {
            return Self { hi: h, lo: 0.0 };
        }
        Self { hi: h, lo: l }
    }

    /// Gives a zero result the sign of `lead`, the `f64` operation on the leading words, as
    /// the IEEE 754 sign rules require (`-0 + -0 = -0`, `0 * -1 = -0`, `x - x = +0`).
    #[inline]
    fn zero_signed_as(self, lead: f64) -> Self {
        if self.hi == 0.0 {
            Self {
                hi: 0.0_f64.copysign(lead),
                lo: 0.0,
            }
        } else {
            self
        }
    }

    /// `self + y` (relative error ≤ 2u²).
    #[inline]
    pub fn add_f64(self, y: f64) -> Self {
        let (mut sh, mut sl) = two_sum(self.hi, y);
        // One branch covers both overflows: a non-finite sh always makes sl non-finite too.
        if !sl.is_finite() {
            if !sh.is_finite() {
                return Self { hi: sh, lo: 0.0 };
            }
            (sh, sl) = two_sum_ordered(self.hi, y); // 2Sum's intermediate overflowed (|y| = MAX)
        }
        Self::renormalize(sh, self.lo + sl).zero_signed_as(sh)
    }

    /// `self + y` (relative error ≤ 3u²).
    #[inline]
    pub fn add_dd(self, y: Self) -> Self {
        let (mut sh, mut sl) = two_sum(self.hi, y.hi);
        // One branch covers both overflows: a non-finite sh always makes sl non-finite too.
        if !sl.is_finite() {
            if !sh.is_finite() {
                return Self { hi: sh, lo: 0.0 };
            }
            (sh, sl) = two_sum_ordered(self.hi, y.hi); // 2Sum's intermediate overflowed
        }
        // No guard needed here: low words are at most ulp(hi)/2 <= 2^970 in magnitude, so no
        // step of this 2Sum can come near the overflow threshold.
        let (th, tl) = two_sum(self.lo, y.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        if !vh.is_finite() {
            return Self { hi: vh, lo: 0.0 };
        }
        let w = tl + vl;
        Self::renormalize(vh, w).zero_signed_as(sh)
    }

    /// `self * y` (relative error ≤ 2u²).
    #[inline]
    pub fn mul_f64(self, y: f64) -> Self {
        let (ch, cl1) = two_prod(self.hi, y);
        if !ch.is_finite() {
            return Self { hi: ch, lo: 0.0 };
        }
        let cl3 = self.lo.mul_add(y, cl1);
        Self::renormalize(ch, cl3).zero_signed_as(ch)
    }

    /// `self * y` (relative error ≤ 4u²).
    #[inline]
    pub fn mul_dd(self, y: Self) -> Self {
        let (ch, cl1) = two_prod(self.hi, y.hi);
        if !ch.is_finite() {
            return Self { hi: ch, lo: 0.0 };
        }
        let tl0 = self.lo * y.lo;
        let tl1 = self.hi.mul_add(y.lo, tl0);
        let cl2 = self.lo.mul_add(y.hi, tl1);
        let cl3 = cl1 + cl2;
        Self::renormalize(ch, cl3).zero_signed_as(ch)
    }

    /// `self / y` (relative error ≤ 3u²).
    #[inline]
    pub fn div_f64(self, y: f64) -> Self {
        let th = self.hi / y;
        if !th.is_finite() || !y.is_finite() {
            return Self { hi: th, lo: 0.0 };
        }
        let (ph, pl) = two_prod(th, y);
        if !ph.is_finite() {
            // fl(th·y) overflowed although th is finite (|self| near MAX): halve, divide, double.
            return self.mul_pow2(0.5).div_f64(y).mul_pow2(2.0);
        }
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / y;
        Self::renormalize(th, tl).zero_signed_as(th)
    }

    /// `self / y` (relative error ≤ 10u²).
    #[inline]
    pub fn div_dd(self, y: Self) -> Self {
        let q = self.hi / y.hi;
        if !q.is_finite() || !y.hi.is_finite() {
            return Self { hi: q, lo: 0.0 };
        }
        let th = 1.0 / y.hi;
        if !th.is_finite() {
            // |y.hi| <= 2^-1024, so |self| < 1 since q is finite: scaling both by 2^106 is exact.
            return self.mul_pow2(pow2(106)).div_dd(y.mul_pow2(pow2(106)));
        }
        if y.hi.abs() > DIV_RESCALE_ABOVE {
            // th would be near-subnormal. Scaling both by 2^-106 is exact unless |self| is
            // below about 2^-860, and then the quotient underflows anyway.
            return self.mul_pow2(pow2(-106)).div_dd(y.mul_pow2(pow2(-106)));
        }
        let rh = (-y.hi).mul_add(th, 1.0);
        let rl = -y.lo * th;
        let (eh, el) = fast_two_sum(rh, rl);
        let m = Self { hi: eh, lo: el }.mul_f64(th).add_f64(th);
        let z = self.mul_dd(m);
        if z.hi.is_infinite() {
            // fl(self.hi·m.hi) overflowed although q is finite: halve, divide, double.
            return self.mul_pow2(0.5).div_dd(y).mul_pow2(2.0);
        }
        z
    }

    /// Multiplies by `p`: exactly (barring overflow and underflow) when `p` is `±0`, `±inf` or
    /// a normal power of two; any other `p` is handled by [`mul_f64`](Self::mul_f64).
    ///
    /// The result is always normalised. Scaling down to below 2⁻⁹⁶⁸ can round the low word
    /// to a subnormal equal to ulp(hi)/2 with the tie away from `hi` (or round `hi` itself), so
    /// such results are renormalised; a zero result is `(±0, +0)`.
    #[inline]
    pub fn mul_pow2(self, p: f64) -> Self {
        // A zero significand field means ±0, ±inf or a normal power of two.
        if p.to_bits() & ((1 << 52) - 1) != 0 {
            return self.mul_f64(p);
        }
        let hi = self.hi * p;
        if !hi.is_finite() {
            return Self { hi, lo: 0.0 };
        }
        let lo = self.lo * p;
        // Above 2⁻⁹⁶⁸, ulp(hi)/2 exceeds every subnormal, so a rounded low word stays below it.
        if p < 1.0 && hi.abs() < RENORMALIZE_BELOW {
            return Self::renormalize_tiny(hi, lo);
        }
        Self { hi, lo }
    }

    /// Renormalises a scaled-down result below 2⁻⁹⁶⁸ (see [`mul_pow2`](Self::mul_pow2)).
    #[cold]
    #[inline(never)]
    fn renormalize_tiny(hi: f64, lo: f64) -> Self {
        if hi == 0.0 {
            return Self { hi, lo: 0.0 };
        }
        let (h, l) = fast_two_sum(hi, lo);
        Self { hi: h, lo: l }
    }

    /// `self * 2^e`.
    pub fn ldexp(self, e: i32) -> Self {
        let mut x = self;
        // Beyond ±2200 every finite nonzero value overflows or underflows anyway.
        let mut e = e.clamp(-2200, 2200);
        while e > 1000 {
            x = x.mul_pow2(pow2(1000));
            e -= 1000;
        }
        while e < -1000 {
            x = x.mul_pow2(pow2(-1000));
            e += 1000;
        }
        x.mul_pow2(pow2(e))
    }

    /// `1 / self`.
    #[inline]
    pub fn recip(self) -> Self {
        Self::ONE.div_dd(self)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        if self.hi.is_sign_negative() {
            -self
        } else {
            self
        }
    }

    /// Largest integer not greater than `self`.
    pub fn floor(self) -> Self {
        if !self.hi.is_finite() {
            return self;
        }
        let f = self.hi.floor();
        if f != self.hi {
            return Self { hi: f, lo: 0.0 };
        }
        if self.lo == 0.0 {
            return Self { hi: f, lo: 0.0 }; // already an integer; keeps the sign of -0
        }
        let (h, l) = two_sum(f, self.lo.floor()); // |floor(lo)| <= 2^970: no overflow
        Self { hi: h, lo: l }
    }

    /// Smallest integer not less than `self`.
    pub fn ceil(self) -> Self {
        -(-self).floor()
    }

    /// Integer part (rounds towards zero).
    pub fn trunc(self) -> Self {
        if self.hi.is_sign_negative() {
            self.ceil()
        } else {
            self.floor()
        }
    }

    /// Nearest integer, rounding half-way cases away from zero (like `f64::round`).
    pub fn round(self) -> Self {
        let t = self.trunc();
        if (self - t).abs() >= Self::from_f64(0.5) {
            if self.hi.is_sign_negative() {
                t - 1.0
            } else {
                t + 1.0
            }
        } else {
            t
        }
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: 0.0 - self.lo, // exactly -lo, except that a zero low word stays +0
        }
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, y: Self) -> Self {
        self.add_dd(y)
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, y: Self) -> Self {
        self.add_dd(-y)
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, y: Self) -> Self {
        self.mul_dd(y)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, y: Self) -> Self {
        self.div_dd(y)
    }
}

/// The truncated remainder `self - y * trunc(self / y)`, like `f64`'s `%`: it has the sign of
/// `self` (including a zero result), magnitude below `|y|`, `x % ±inf == x` for finite `x`,
/// and NaN when `self` is infinite or `y` is zero.
///
/// Accuracy: the remainder is computed exactly, by long division on floating-point
/// expansions (any quotient, up to 2²⁰⁹⁸), and then rounded once, so the result is within
/// `2u²·|y|` of the true remainder (u = 2⁻⁵³). The error is at most about `u²·|result|`,
/// except when the true remainder lies within that distance below `|y|` and rounding would
/// reach `|y|`; then a double-double one step of the low word below `|y|` is returned. The
/// remainder of two `f64`s, which is itself an `f64`, is returned exactly. As for the other
/// operations, full precision requires magnitudes above about 2⁻⁹⁶⁹.
impl Rem for DoubleDouble {
    type Output = Self;
    fn rem(self, y: Self) -> Self {
        rem::rem(self, y)
    }
}

impl Add<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, y: f64) -> Self {
        self.add_f64(y)
    }
}

impl Sub<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, y: f64) -> Self {
        self.add_f64(-y)
    }
}

impl Mul<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, y: f64) -> Self {
        self.mul_f64(y)
    }
}

impl Div<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, y: f64) -> Self {
        self.div_f64(y)
    }
}

impl Add<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn add(self, y: DoubleDouble) -> DoubleDouble {
        y.add_f64(self)
    }
}

impl Sub<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn sub(self, y: DoubleDouble) -> DoubleDouble {
        (-y).add_f64(self)
    }
}

impl Mul<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn mul(self, y: DoubleDouble) -> DoubleDouble {
        y.mul_f64(self)
    }
}

impl Div<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn div(self, y: DoubleDouble) -> DoubleDouble {
        DoubleDouble::from_f64(self).div_dd(y)
    }
}

macro_rules! assign_ops {
    ($($trait:ident, $method:ident, $op:tt);* $(;)?) => {$(
        impl $trait for DoubleDouble {
            #[inline]
            fn $method(&mut self, y: Self) {
                *self = *self $op y;
            }
        }
        impl $trait<f64> for DoubleDouble {
            #[inline]
            fn $method(&mut self, y: f64) {
                *self = *self $op y;
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

impl RemAssign for DoubleDouble {
    #[inline]
    fn rem_assign(&mut self, y: Self) {
        *self = *self % y;
    }
}

impl From<f64> for DoubleDouble {
    #[inline]
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl PartialOrd for DoubleDouble {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&other.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&other.lo),
            ord => Some(ord),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subnormal_rounding_uses_the_trailing_word() {
        let s = 128;
        let g = pow2(s - 1074); // the scaled subnormal spacing
        // 2.5 steps: a tie, decided by the trailing word.
        let v = DoubleDouble::from_f64(2.5 * g);
        assert_eq!(round_to_subnormal(v, s), f64::from_bits(2)); // ties to even
        assert_eq!(
            round_to_subnormal(DoubleDouble::new(2.5 * g, 1e-310), s),
            f64::from_bits(3)
        );
        assert_eq!(
            round_to_subnormal(DoubleDouble::new(3.5 * g, -1e-310), s),
            f64::from_bits(3)
        );
        assert_eq!(
            round_to_subnormal(DoubleDouble::from_f64(3.5 * g), s),
            f64::from_bits(4)
        );
        assert_eq!(
            round_to_subnormal(DoubleDouble::new(0.5 * g, 1e-310), s),
            f64::from_bits(1)
        );
        assert_eq!(round_to_subnormal(DoubleDouble::from_f64(0.5 * g), s), 0.0);
        assert_eq!(round_to_subnormal(DoubleDouble::ZERO, s).to_bits(), 0);
        assert_eq!(
            round_to_subnormal(-DoubleDouble::ZERO, s).to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            round_to_subnormal(DoubleDouble::new(-2.5 * g, -1e-310), s),
            -f64::from_bits(3)
        );
        assert_eq!(
            round_to_subnormal(DoubleDouble::from_f64(-0.25 * g), s).to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn from_u128_is_within_u2_next_to_ties() {
        // 2^126 + 2^73 + 1 is just above a tie of the leading word.
        let n: u128 = (1 << 126) + (1 << 73) + 1;
        let x = from_u128(n);
        assert_eq!((x.hi, x.lo), (pow2(126), pow2(73)));
        assert_ne!(x.hi, n as f64); // the renormalisation moved the leading word
        let below = from_u128((1 << 126) + (1 << 73) - 1);
        assert_eq!((below.hi, below.lo), (pow2(126), pow2(73)));
        assert_eq!(from_u128(u128::MAX).hi, pow2(127) * 2.0);
        assert_eq!(from_u128((1 << 106) - 1).lo, -1.0);
    }
}
