//! Forward-mode dual numbers carrying `N` first-order directional derivatives.

use core::array;
use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

use crate::scalar::Real;

/// A value with `N` first-order derivatives: `re + Σᵢ eps[i]·εᵢ`.
///
/// Comparisons and `is_zero` look only at the value `re`. `Float::is_nan` and
/// `Float::is_finite` also inspect the derivatives, so a NaN derivative is never silent.
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
/// | `a / b` | `(a′ − q·b′)·(1/b)` |
/// | `a % b` | `a′ − t·b′`, where `t` is the integer with `a = t·b + r` |
/// | `a + y`, `a - y` | `a′` |
/// | `a * y`, `a / y` | `a′·y`, `a′/y` |
///
/// Each formula is evaluated in `T`, so a derivative errs by a few units in the last place of
/// its largest term; cancellation between the terms is not compensated.
///
/// `%` jumps where `a/b` is an integer; there the derivative is that of the branch that gave
/// the value (`t = a/b`, `r = ±0`). `t` is recovered from the exact remainder as
/// `round((a − r)/b)`, not as `trunc(a/b)`, which rounding can move to the next integer:
/// `1.0 / 0.1 == 10.0`, yet `1.0 % 0.1 == 1.0 − 9·0.1`. It is exact for |t| < 2⁵⁰ with `f64`
/// (about 2⁹⁸ with `DoubleDouble`), and within about 2u·|t| beyond (u the unit roundoff of `T`).
///
/// # Special values and signed zeros
///
/// A derivative component that is exactly zero is a *strong zero*: it contributes zero even
/// when multiplied by an infinite or NaN factor. So a direction on which no operand depends
/// keeps derivative 0 whatever the values: `Dual::constant(2.0) * Dual::constant(f64::INFINITY)`
/// has zero derivatives, not NaN. With finite operands, a derivative is NaN only if the value is
/// NaN too (`0 / 0`, `x % 0`) or two terms of its formula overflow with opposite signs
/// (`∞ − ∞`, as `T` would give for the same formula). Otherwise IEEE 754 rules apply: `x / 0`
/// has derivative `x′/0 = ±∞`, `a / b` at `b = ±0` with `b′ ≠ 0` is dominated by the `a·b′/b²`
/// term, and a nonzero derivative times an infinite value is infinite. An infinite or NaN
/// derivative times a zero value is NaN, as in `T`.
///
/// The sign of a zero derivative is unspecified (it may be `+0` or `−0`). It carries no
/// information, and no operation lets it change a nonzero result.
///
/// # Speed
///
/// Operations are `#[inline]` and loop over the `N` derivatives with fixed trip counts; for
/// `Dual<f64, N>` they compile to straight-line vectorised code. `*`, `/`, `* f64` and `/ f64`
/// first check, with one predictable branch, that the plain rule cannot meet a non-finite factor
/// (or, for `/`, a reciprocal outside the full-precision range); only the special values take
/// the slower path. The correctly rounded value of `/` costs a second division: `a/b` for the
/// value and `1/b` for the derivatives.
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
    /// `sqrt` at 0), so a constant stays a constant.
    #[allow(dead_code)] // used by the Float impl in Task 8
    #[inline]
    pub(crate) fn chain(self, f: T, df: T) -> Self {
        let eps = if df.is_finite() {
            self.eps.map(|e| e * df)
        } else {
            scale_special(&self, f, df).eps
        };
        Self { re: f, eps }
    }
}

// Special-value paths. `*`, `* f64`, `/ f64` and `chain` check that the plain rule cannot meet
// a non-finite factor and otherwise call one of these, as `special(..).eps` (`/` keeps its
// special path inline, see there). Measured on `Dual<f64, N>`:
// * out of line and `#[cold]`, they keep a real, predicted branch; inline, LLVM turns small ones
//   into selects that put the check on the critical path (a multiply chain with N = 1 ran at
//   2.2× the cycles);
// * returning the whole `Dual` keeps the merge in floating-point registers (a `[f64; 1]` is
//   returned in an integer register), and taking `.eps` rather than returning early keeps the
//   fast path's result out of the return slot (an early return cost ~14% throughput at N = 4);
// * branch-free strong zeros everywhere, instead of a check, cost ~25% throughput at N = 4.

/// `e·v`, but zero when the derivative component `e` is zero, even if `v` is infinite or NaN.
#[inline]
fn strong_mul<T: Real>(e: T, v: T) -> T {
    if e.is_zero() { T::zero() } else { e * v }
}

/// `e/v`, but zero when `e` is zero, even if `v` is zero or NaN.
#[inline]
fn strong_div<T: Real>(e: T, v: T) -> T {
    if e.is_zero() { T::zero() } else { e / v }
}

/// Value `f`, derivatives `x′·df` with strong zeros (`df` infinite or NaN).
#[cold]
#[inline(never)]
fn scale_special<T: Real, const N: usize>(x: &Dual<T, N>, f: T, df: T) -> Dual<T, N> {
    Dual {
        re: f,
        eps: x.eps.map(|e| strong_mul(e, df)),
    }
}

/// `a * b` with strong zeros (a value is infinite or NaN, or the product overflowed).
#[cold]
#[inline(never)]
fn mul_special<T: Real, const N: usize>(a: &Dual<T, N>, b: &Dual<T, N>, re: T) -> Dual<T, N> {
    Dual {
        re,
        eps: array::from_fn(|i| strong_mul(a.eps[i], b.re) + strong_mul(b.eps[i], a.re)),
    }
}

/// `a * y` with strong zeros (`y` infinite or NaN).
#[cold]
#[inline(never)]
fn mul_f64_special<T: Real, const N: usize>(a: &Dual<T, N>, y: f64) -> Dual<T, N> {
    Dual {
        re: a.re * y,
        eps: a.eps.map(|e| if e.is_zero() { T::zero() } else { e * y }),
    }
}

/// `a / y` with strong zeros (`y` zero or NaN).
#[cold]
#[inline(never)]
fn div_f64_special<T: Real, const N: usize>(a: &Dual<T, N>, y: f64) -> Dual<T, N> {
    Dual {
        re: a.re / y,
        eps: a.eps.map(|e| if e.is_zero() { T::zero() } else { e / y }),
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
            mul_special(&self, &o, re).eps
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
        // a finite 1/b of at least `min_positive_value()` has full precision. Otherwise (b is
        // ±0, ±∞ or NaN, |b| is subnormal or near MAX, the dividend is not finite, or the
        // quotient overflowed) the derivatives are (a′ − q·b′)/b, divided directly, with strong
        // zeros. That path stays inline: the division's latency hides the check, and inline
        // measured faster than out of line here.
        let eps = if q.is_finite() && inv.is_finite() && inv.abs() >= T::min_positive_value() {
            array::from_fn(|i| (self.eps[i] - q * o.eps[i]) * inv)
        } else {
            array::from_fn(|i| strong_div(self.eps[i] - strong_mul(o.eps[i], q), o.re))
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
            mul_f64_special(&self, y).eps
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
            div_f64_special(&self, y).eps
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

    /// Whether the value is zero (of either sign), whatever the derivatives.
    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero()
    }
}

impl<T: Real, const N: usize> One for Dual<T, N> {
    #[inline]
    fn one() -> Self {
        Self::constant(T::one())
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
    /// A constant converted by `T`'s `NumCast::from`.
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
