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
    /// The smallest validated Re m.
    pub m_re_min: f64,
    /// The largest validated Re m.
    pub m_re_max: f64,
    /// The largest validated Im m (the smallest is 0).
    pub m_im_max: f64,
    /// The smallest validated `rtol`.
    pub rtol_min: f64,
}

/// The validated domain: aspect ratios 1/5 to 5, equal-volume x ≥ 10⁻⁶, |m|·x_max ≤ 60,
/// 1.3 ≤ Re m ≤ 3, Im m ≤ 0.5, `rtol` ≥ 10⁻¹².
///
/// The accuracy map covers 349 spheroids: 8 aspect ratios from 1/5 to 5, x from 10⁻⁶ to 14 and
/// four indices from 1.33 to 2.5 + 0.1i and 1.5 + 0.5i, up to |m|·x_max = 70. Its worst ratio of
/// error to max(`rtol`, estimate) inside the domain is 1.04 (Q_back, ε = 1/3, x = 4), and 0.57
/// for the Greek coefficients; the cross sections and g stay below 0.04. Past |m|·x_max = 60 the
/// results are as accurate, but the refined modes make a spheroid cost minutes: the edge is set
/// by cost.
pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
    aspect_ratio_min: 0.2,
    aspect_ratio_max: 5.0,
    x_min: 1e-6,
    max_phase: 60.0,
    m_re_min: 1.3,
    m_re_max: 3.0,
    m_im_max: 0.5,
    rtol_min: 1e-12,
};

impl ValidatedDomain {
    /// Whether the spheroid of aspect ratio `eps`, equal-volume size parameter `x` and index `m`
    /// is inside the domain (for any `rtol` ≥ `rtol_min`).
    pub fn contains(&self, eps: f64, x: f64, m: Complex<f64>) -> bool {
        (self.aspect_ratio_min..=self.aspect_ratio_max).contains(&eps)
            && x >= self.x_min
            && m.norm() * x_max(x, eps) <= self.max_phase
            && (self.m_re_min..=self.m_re_max).contains(&m.re)
            && (0.0..=self.m_im_max).contains(&m.im)
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
    /// | Q_ext, Q_sca, g | 4 (worst measured: 0.03) |
    /// | Q_abs, Q_back, phase matrix, Greek coefficients | 16 (worst measured: 1.04) |
    pub fn constant(output: MieOutput) -> f64 {
        match output {
            MieOutput::Extinction | MieOutput::Scattering | MieOutput::Asymmetry => 4.0,
            _ => 16.0,
        }
    }
}
