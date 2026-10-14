//! Scalar types that every numerical kernel is generic over.
//!
//! The numerical kernels (the Mie solution and the special functions) are generic over
//! [`Real`], so one implementation gives:
//!
//! * values, with `f64`;
//! * extended-precision values, with [`DoubleDouble`] (about 32 significant digits);
//! * values with N first derivatives, with [`Dual<f64, N>`](Dual);
//! * extended-precision derivatives, with `Dual<DoubleDouble, N>`;
//! * second derivatives, with nested duals `Dual<Dual<T, M>, N>`.
//!
//! ```
//! use scatrs::Real;
//! use scatrs::scalar::{DoubleDouble, Dual};
//!
//! // A generic function, evaluated in double-double and with a derivative.
//! fn f<T: Real>(x: T) -> T {
//!     (x * 2.0).sin() * x.exp()
//! }
//! let value = f(DoubleDouble::from_f64(0.5));
//! assert!((value.approx_f64() - (1.0_f64).sin() * (0.5_f64).exp()).abs() < 1e-15);
//!
//! let [x] = Dual::<f64, 1>::variables([0.5]);
//! let y = f(x); // y.eps()[0] = d/dx [sin 2x · eˣ] = (2 cos 2x + sin 2x) eˣ
//! let exact = (2.0 * (1.0_f64).cos() + (1.0_f64).sin()) * (0.5_f64).exp();
//! assert!((y.eps()[0] - exact).abs() < 1e-14);
//! ```
//!
//! [`Real`] lists what generic code may rely on, and [`DoubleDouble`] and [`Dual`] state their
//! accuracy and conventions.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{Float, FloatConst};

mod dd;
mod dual;

pub use dd::{DoubleDouble, ParseDoubleDoubleError};
pub use dual::Dual;

/// A real scalar type for generic kernels.
///
/// Implemented for `f64`, `DoubleDouble` (about 32 significant digits) and `Dual<T, N>`
/// (forward-mode derivatives). The `num_traits::Float` supertrait makes
/// `num_complex::Complex<T>` usable for every `T: Real`.
/// Mixed operations with `f64` take the `f64` on the right (`x * 2.0`, not `2.0 * x`);
/// for `Complex<T>` use `z.scale(T::from_f64(2.0))`.
///
/// # Notes for generic code
///
/// `DoubleDouble` follows `f64` closely, but generic code must not assume more than `Float`
/// promises (see [`DoubleDouble`]'s "Use as `num_traits::Float`"):
///
/// * `mul_add` is fused for `f64` but not for `DoubleDouble`, which rounds the product before
///   adding. Error-free transformations must not rely on it: Kahan's idiom
///   `e = b.mul_add(c, -(b * c))` gives exactly 0 for `DoubleDouble`, not the rounding error
///   of `b * c`.
/// * `epsilon()` is a unit (2⁻⁵² for `f64`, 2⁻¹⁰⁴ for `DoubleDouble`), not a bound on the
///   error of one operation: a `DoubleDouble` division errs by up to about 2.5·epsilon.
/// * `NumCast::from` another `Real` keeps only its leading word (`to_f64`); use the value
///   itself, or [`from_f64_pair`](Real::from_f64_pair), to keep full precision.
/// * For [`Dual`], `==` and `<` compare values only, while `is_zero`, `is_one`, `is_nan` and
///   `is_finite` also inspect the derivatives (see its "Comparisons and zero tests"): branch on
///   comparisons of values, as `x == T::zero()`, where the derivatives should not matter.
/// * `Dual<T, N>` takes the values of `sin` and `cos` from `T::sin_cos`. They equal `T::sin` and
///   `T::cos` where `sin_cos` does: always for `DoubleDouble`. For `f64` on Apple targets, LLVM
///   fuses `sin` and `cos` of one argument into one call whose sine can be an ulp off a lone
///   `sin`, in `Dual` as in any `f64` code computing both.
pub trait Real:
    Float
    + FloatConst
    + Debug
    + Send
    + Sync
    + 'static
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + AddAssign<f64>
    + SubAssign<f64>
    + MulAssign<f64>
    + DivAssign<f64>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// Converts an `f64` exactly.
    fn from_f64(x: f64) -> Self;

    /// Builds the value of the unevaluated sum `hi + lo`.
    ///
    /// Exact for `DoubleDouble`; for `f64` the sum is rounded once.
    fn from_f64_pair(hi: f64, lo: f64) -> Self;

    /// The nearest `f64` to the value, ignoring any derivative parts.
    fn approx_f64(self) -> f64;
}

impl Real for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }

    #[inline]
    fn from_f64_pair(hi: f64, lo: f64) -> Self {
        hi + lo
    }

    #[inline]
    fn approx_f64(self) -> f64 {
        self
    }
}
