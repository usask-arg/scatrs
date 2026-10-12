//! Scalar types that every numerical kernel is generic over.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{Float, FloatConst};

/// A real scalar type for generic kernels.
///
/// Implemented for `f64`, `DoubleDouble` (about 32 significant digits) and `Dual<T, N>`
/// (forward-mode derivatives). The `num_traits::Float` supertrait makes
/// `num_complex::Complex<T>` usable for every `T: Real`.
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
