//! The validated domain of the Mie outputs.

use core::fmt;

use num_complex::Complex;

/// The region of inputs whose accuracy has been validated.
///
/// [`Mie::compute`](super::Mie::compute) returns
/// [`Error::OutsideValidatedDomain`](crate::Error::OutsideValidatedDomain) outside it unless
/// the caller sets `allow_unvalidated`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ValidatedDomain {
    /// The smallest validated size parameter.
    pub x_min: f64,
    /// The largest validated size parameter.
    pub x_max: f64,
    /// The largest validated Re m (the smallest is any value above 0).
    pub m_re_max: f64,
    /// The largest validated Im m (the smallest is 0).
    pub m_im_max: f64,
}

/// The validated domain: x ∈ [10⁻⁶, 2·10⁴], 0 < Re m ≤ 10, 0 ≤ Im m ≤ 10.
pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
    x_min: 1e-6,
    x_max: 2e4,
    m_re_max: 10.0,
    m_im_max: 10.0,
};

impl ValidatedDomain {
    /// Whether (x, m) is inside the domain.
    pub fn contains(&self, x: f64, m: Complex<f64>) -> bool {
        (self.x_min..=self.x_max).contains(&x)
            && m.re > 0.0
            && m.re <= self.m_re_max
            && (0.0..=self.m_im_max).contains(&m.im)
    }

    /// κ(x, m) = (1 + |m|x)·max(1, |m|/|m − 1|): infinite for m = 1, where the coefficients
    /// vanish.
    ///
    /// The outputs depend on the phase mx, so rounding x or m changes them by about |m|x times
    /// the rounding; and aₙ, bₙ ∝ m − 1 as m → 1, so a relative change in m changes them by
    /// |m|/|m − 1| times as much.
    pub fn condition_factor(x: f64, m: Complex<f64>) -> f64 {
        (1.0 + m.norm() * x) * near_one(m)
    }
}

/// max(1, |m|/|m − 1|).
fn near_one(m: Complex<f64>) -> f64 {
    (m.norm() / (m - Complex::new(1.0, 0.0)).norm()).max(1.0)
}

impl fmt::Display for ValidatedDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "x in [{:e}, {:e}], 0 < Re m <= {}, 0 <= Im m <= {}",
            self.x_min, self.x_max, self.m_re_max, self.m_im_max
        )
    }
}
