//! Double-double arithmetic: a value is the unevaluated sum `hi + lo` of two `f64`s,
//! normalised so that `hi == fl(hi + lo)`, giving about 106 significant bits.
//!
//! Basic operations are the algorithms of Joldes, Muller & Popescu, "Tight and rigorous
//! error bounds for basic building blocks of double-word arithmetic", ACM TOMS 44(2), 2017.
//! With u = 2⁻⁵³ the relative error bounds are: `+`/`-` 3u², `+ f64` 2u², `*` 4u²,
//! `* f64` 2u², `/` 10u², `/ f64` 3u². They rely on a correctly rounded fused multiply-add,
//! which `f64::mul_add` provides on every platform (in software where hardware lacks FMA).
//!
//! Non-finite results are returned as `(hi, 0.0)` so that infinities do not become NaN.
//! Full relative precision requires magnitudes above about 2⁻⁹⁶⁹ ≈ 2·10⁻²⁹².

use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

mod consts;

/// A double-double number: the exact sum `hi + lo` with `hi == fl(hi + lo)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoubleDouble {
    hi: f64,
    lo: f64,
}

/// Error-free sum: `s = fl(a + b)` and `s + e = a + b` exactly.
#[inline]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// Error-free sum, valid when `|a| >= |b|` or `a == 0`.
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

/// `2^e` for `-1022 <= e <= 1023`.
#[inline]
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
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
        let (s, e) = two_sum(hi, lo);
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

    #[inline]
    fn renormalize(hi: f64, lo: f64) -> Self {
        let (h, l) = fast_two_sum(hi, lo);
        Self { hi: h, lo: l }
    }

    /// `self + y` (relative error ≤ 2u²).
    #[inline]
    pub fn add_f64(self, y: f64) -> Self {
        let (sh, sl) = two_sum(self.hi, y);
        if !sh.is_finite() {
            return Self { hi: sh, lo: 0.0 };
        }
        Self::renormalize(sh, self.lo + sl)
    }

    /// `self + y` (relative error ≤ 3u²).
    #[inline]
    pub fn add_dd(self, y: Self) -> Self {
        let (sh, sl) = two_sum(self.hi, y.hi);
        if !sh.is_finite() {
            return Self { hi: sh, lo: 0.0 };
        }
        let (th, tl) = two_sum(self.lo, y.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        let w = tl + vl;
        Self::renormalize(vh, w)
    }

    /// `self * y` (relative error ≤ 2u²).
    #[inline]
    pub fn mul_f64(self, y: f64) -> Self {
        let (ch, cl1) = two_prod(self.hi, y);
        if !ch.is_finite() {
            return Self { hi: ch, lo: 0.0 };
        }
        let cl3 = self.lo.mul_add(y, cl1);
        Self::renormalize(ch, cl3)
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
        Self::renormalize(ch, cl3)
    }

    /// `self / y` (relative error ≤ 3u²).
    #[inline]
    pub fn div_f64(self, y: f64) -> Self {
        let th = self.hi / y;
        if !th.is_finite() || !y.is_finite() {
            return Self { hi: th, lo: 0.0 };
        }
        let (ph, pl) = two_prod(th, y);
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / y;
        Self::renormalize(th, tl)
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
            // |y.hi| < 2^-1024, so |self| < 2 since q is finite: scaling both by 2^106 is exact.
            return self.mul_pow2(pow2(106)).div_dd(y.mul_pow2(pow2(106)));
        }
        let rh = (-y.hi).mul_add(th, 1.0);
        let rl = -y.lo * th;
        let (eh, el) = fast_two_sum(rh, rl);
        let m = Self { hi: eh, lo: el }.mul_f64(th).add_f64(th);
        self.mul_dd(m)
    }

    /// Multiplies by `p`, which must be a power of two (exact unless over/underflow).
    #[inline]
    pub fn mul_pow2(self, p: f64) -> Self {
        let hi = self.hi * p;
        if !hi.is_finite() {
            return Self { hi, lo: 0.0 };
        }
        Self {
            hi,
            lo: self.lo * p,
        }
    }

    /// `self * 2^e`.
    pub fn ldexp(self, e: i32) -> Self {
        let mut x = self;
        let mut e = e;
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
        if f == self.hi {
            let (h, l) = two_sum(f, self.lo.floor());
            Self { hi: h, lo: l }
        } else {
            Self { hi: f, lo: 0.0 }
        }
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
            lo: -self.lo,
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

impl Rem for DoubleDouble {
    type Output = Self;
    /// `self - y * trunc(self / y)`, with the sign of `self` (like `f64`'s `%`).
    fn rem(self, y: Self) -> Self {
        if y.hi.is_infinite() && self.hi.is_finite() {
            return self; // as for f64; the formula below would give inf * 0 = NaN
        }
        self - y * (self / y).trunc()
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
