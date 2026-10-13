//! The validated domain of the Mie outputs and their declared accuracy.

use core::fmt;

use num_complex::Complex;

/// The outputs whose accuracy [`ValidatedDomain::tolerance`] declares, with the error measure
/// each tolerance bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MieOutput {
    /// aₙ and bₙ: max over n of |error|, divided by max over n of |aₙ| (and likewise bₙ).
    Coefficients,
    /// Each of ∂aₙ/∂x, daₙ/dm, ∂bₙ/∂x and dbₙ/dm, measured as [`Coefficients`](Self::Coefficients).
    CoefficientDerivatives,
    /// Q_ext, relative error.
    Extinction,
    /// Q_sca, relative error.
    Scattering,
    /// Q_abs, relative error (exactly 0 for real m).
    Absorption,
    /// Q_back: |error| / max(Q_back, Q_sca). Backscattering by large weakly refracting spheres
    /// is many orders of magnitude below Q_sca, and its relative accuracy is limited by the
    /// cancellation in its series.
    Backscattering,
    /// g: |error| / max(|g|, 10⁻³ min(1, x²)).
    Asymmetry,
    /// S₁ and S₂: |error| / max(|S|, S_rms), where S_rms = x√Q_sca/2 is the RMS amplitude, so
    /// the deep minima of the angular pattern are measured against its typical size.
    Amplitudes,
    /// Each ∂Q/∂p of [`EfficiencyJacobian`](super::EfficiencyJacobian): |error| /
    /// max(|∂Q/∂p|, Q′·(1 + |m|x)/|p|), with |p| = x for x and |m| for Re m and Im m, and the
    /// size Q′ of Q in the measure of its value (Q_ext for Q_abs: relative to Q_abs itself,
    /// ∂Q_abs/∂x and ∂Q_abs/∂(Re m) of weakly absorbing spheres can be less accurate, by up to
    /// Q_ext/Q_abs). Q·(1 + |m|x)/|p| is the natural size of a derivative: the outputs vary
    /// with the phase |m|x.
    EfficiencyJacobian,
    /// ∂S/∂x and dS/dm: max over μ of |error| / max over μ of
    /// max(|∂S/∂p|, S_rms·(1 + |m|x)/|p|).
    AmplitudeDerivatives,
}

/// The region of inputs whose accuracy has been validated, and the declared tolerances there.
///
/// Inside it, every output of [`Mie::compute`](super::Mie::compute) in `f64`, with the default
/// truncation or more terms, is within
///
/// C·(κ′(x, m) + cond)·2⁻⁵²
///
/// of the exact value for the exact inputs, in the measure of its [`MieOutput`]
/// ([`tolerance`](Self::tolerance)). Here
///
/// * cond is the output's relative condition number with respect to x and m in that measure:
///   (x·|∂y/∂x| + |m|·|∇ₘy|)/s, where s is the denominator of the measure, |∇ₘy| is
///   √((∂y/∂Re m)² + (∂y/∂Im m)²) (= |dy/dm| for the holomorphic aₙ, bₙ, S₁, S₂), and for
///   the measures over an array the numerator is the largest over the array. It is large near
///   the sharp resonances of weakly absorbing spheres, where no algorithm can do better, since
///   rounding x and m to `f64` already perturbs the outputs by cond·2⁻⁵³.
/// * κ′(x, m) is [`condition_factor`](Self::condition_factor) for the values, and that times
///   max(1, |m|/|m − 1|) for the derivatives. It covers the errors that cond does not: the
///   cancellation in the series (in Q_back, the alternating sum is up to x times smaller than
///   its terms) and the orders' separate losses near m = 1.
/// * C is the output's constant ([`constant`](Self::constant)).
///
/// The constants are measured, at least 4 times the worst case found: at every Arb reference
/// value in the domain, and against double-double on a dense grid (the accuracy map in the
/// documentation). Tests enforce both in CI.
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
    pub fn condition_factor(x: f64, m: Complex<f64>) -> f64 {
        (1.0 + m.norm() * x) * near_one(m)
    }

    /// The declared bound on the error of `output` at (x, m): C·(κ′ + `condition`)·2⁻⁵², in
    /// the measure of [`MieOutput`] (see the type documentation for κ′ and the condition
    /// number). With `condition` = 0 this is the a-priori part alone, which the accuracy map
    /// shows to suffice away from sharp resonances.
    pub fn tolerance(&self, output: MieOutput, x: f64, m: Complex<f64>, condition: f64) -> f64 {
        let derivative = matches!(
            output,
            MieOutput::CoefficientDerivatives
                | MieOutput::EfficiencyJacobian
                | MieOutput::AmplitudeDerivatives
        );
        let extra = if derivative { near_one(m) } else { 1.0 };
        let prior = Self::condition_factor(x, m) * extra;
        Self::constant(output) * (prior + condition) * f64::EPSILON
    }

    /// The constant C of [`tolerance`](Self::tolerance).
    ///
    /// | output | C |
    /// |---|---|
    /// | coefficients, efficiencies, g | 16 |
    /// | derivatives of S₁, S₂ | 32 |
    /// | S₁, S₂ | 64 |
    /// | efficiency Jacobian | 128 |
    /// | coefficient derivatives | 256 |
    pub fn constant(output: MieOutput) -> f64 {
        match output {
            MieOutput::Coefficients
            | MieOutput::Extinction
            | MieOutput::Scattering
            | MieOutput::Absorption
            | MieOutput::Backscattering
            | MieOutput::Asymmetry => 16.0,
            MieOutput::AmplitudeDerivatives => 32.0,
            MieOutput::Amplitudes => 64.0,
            MieOutput::EfficiencyJacobian => 128.0,
            MieOutput::CoefficientDerivatives => 256.0,
        }
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
