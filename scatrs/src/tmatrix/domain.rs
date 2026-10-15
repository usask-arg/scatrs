//! The validated domain of randomly oriented spheroids and their declared accuracy.

use num_complex::Complex;

use super::driver::x_max;
use crate::mie::MieOutput;

/// The region of inputs whose accuracy has been validated, and the declared tolerances there.
///
/// Inside it, every output of [`TMatrix::compute`](super::TMatrix::compute), and of its solution's
/// methods, is within
///
/// C·max(`rtol`, ε̂)
///
/// of the exact value, in the measure of its [`MieOutput`], with ε̂ the solution's
/// [`error_estimate`](super::TMatrixSolution::error_estimate). The constants are measured, at
/// least 4 times the worst case found at every high-precision reference value in the domain and
/// on the accuracy maps.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ValidatedDomain {
    /// The smallest validated aspect ratio ε (prolate below 1).
    pub aspect_ratio_min: f64,
    /// The largest validated aspect ratio ε (oblate above 1).
    pub aspect_ratio_max: f64,
    /// The smallest validated equal-volume size parameter.
    pub x_min: f64,
    /// The largest validated |m|·x_max, with x_max = k·max(a, c) the size parameter of the
    /// longest semi-axis.
    pub max_phase: f64,
    /// The largest validated Re m (the smallest is `abs_m_min`).
    pub m_re_max: f64,
    /// The largest validated Im m (the smallest is 0).
    pub m_im_max: f64,
    /// The smallest validated |m|.
    pub abs_m_min: f64,
    /// The smallest validated `rtol`.
    pub rtol_min: f64,
}

/// The validated domain: aspect ratios 1/5 to 5, equal-volume x ≥ 10⁻³, |m|·x_max ≤ 60,
/// 1 ≤ Re m ≤ 3, Im m ≤ 1, |m| ≥ 1, `rtol` ≥ 10⁻¹².
pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
    aspect_ratio_min: 0.2,
    aspect_ratio_max: 5.0,
    x_min: 1e-3,
    max_phase: 60.0,
    m_re_max: 3.0,
    m_im_max: 1.0,
    abs_m_min: 1.0,
    rtol_min: 1e-12,
};

impl ValidatedDomain {
    /// Whether the spheroid of aspect ratio `eps`, equal-volume size parameter `x` and index `m`
    /// is inside the domain (for any `rtol` ≥ `rtol_min`).
    pub fn contains(&self, eps: f64, x: f64, m: Complex<f64>) -> bool {
        (self.aspect_ratio_min..=self.aspect_ratio_max).contains(&eps)
            && x >= self.x_min
            && m.norm() * x_max(x, eps) <= self.max_phase
            && m.re <= self.m_re_max
            && (0.0..=self.m_im_max).contains(&m.im)
            && m.norm() >= self.abs_m_min
    }

    /// The declared bound on the error of `output`: C·max(`rtol`, `estimate`).
    pub fn tolerance(&self, output: MieOutput, rtol: f64, estimate: f64) -> f64 {
        Self::constant(output) * rtol.max(estimate)
    }

    /// The constant C of [`tolerance`](Self::tolerance), measured on the accuracy maps (at least
    /// 4 times the worst case found there and at the references).
    ///
    /// | output | C |
    /// |---|---|
    /// | Q_ext, Q_sca, g | 4 |
    /// | Q_abs, Q_back, phase matrix, Greek coefficients | 16 |
    pub fn constant(output: MieOutput) -> f64 {
        match output {
            MieOutput::Extinction | MieOutput::Scattering | MieOutput::Asymmetry => 4.0,
            _ => 16.0,
        }
    }
}
