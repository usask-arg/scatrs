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
    /// F₁₁, F₁₂, F₃₃, F₃₄ at each cosine: |error| / max(F₁₁, F̄), where F̄ = ½∫F₁₁ dμ is 1
    /// normalized and Q_sca unnormalized, so the deep minima of the angular pattern are measured
    /// against its mean.
    PhaseMatrix,
    /// Each ∂F/∂p of the phase matrix: max over μ of |error| / max over μ of
    /// max(|∂F/∂p|, max(F₁₁, F̄)·(1 + |m|x)/|p|), with |p| as for
    /// [`EfficiencyJacobian`](Self::EfficiencyJacobian).
    PhaseMatrixDerivatives,
    /// Each of α₁, α₂, α₃, α₄, β₁, β₂ up to `order` L: max over l of |error| / max over l of
    /// |α₁|.
    GreekCoefficients {
        /// The highest order L.
        order: usize,
    },
    /// Each derivative of α₁, …, β₂ with respect to p: max over l of |error| / max over l of
    /// max(|∂c/∂p|, A·(1 + |m|x)/|p|), with A = max over l of |α₁| and |p| as for
    /// [`EfficiencyJacobian`](Self::EfficiencyJacobian).
    GreekDerivatives {
        /// The highest order L.
        order: usize,
    },
}

/// The region of inputs whose accuracy has been validated, and the declared tolerances there.
///
/// Inside it, every output of [`Mie::compute`](super::Mie::compute) in `f64`, with the default
/// truncation or more terms, and of the solution's methods (amplitudes, phase matrix, Greek
/// coefficients, in either [`Normalization`](super::Normalization)), is within
///
/// C·(κ′(x, m) + cond)·2⁻⁵²
///
/// of the exact value for the exact inputs, in the measure of its [`MieOutput`]
/// ([`tolerance`](Self::tolerance)). Here
///
/// * cond is the output's relative condition number with respect to x and m in that measure,
///   with s the denominator of the measure (for the measures over an array, the numerator is
///   the largest over the array):
///   * for the holomorphic aₙ, bₙ, S₁, S₂ and their derivatives, (x·|∂y/∂x| + |m|·|dy/dm|)/s;
///   * for the real outputs (the efficiencies, g, the phase matrix, the Greek coefficients
///     and their derivatives), componentwise:
///     (x·|∂y/∂x| + |Re m|·|∂y/∂(Re m)| + |Im m|·|∂y/∂(Im m)|)/s.
///
///   Rounding x and m to `f64` (each part of m to within 2⁻⁵³ of itself) already perturbs the
///   output by up to about cond·2⁻⁵³, so no algorithm can avoid it; it is large near the sharp
///   resonances of weakly absorbing spheres.
/// * κ′(x, m) is an a-priori factor for the errors that cond does not cover: the cancellation
///   in the series (in Q_back, the alternating sum is up to x times smaller than its terms),
///   and the separate losses of each order near m = 1, where aₙ, bₙ ∝ m − 1. It is
///   [`condition_factor`](Self::condition_factor) κ for the values, κ·max(1, |m|/|m − 1|) for
///   the efficiency Jacobian (the derivatives of g lose a second factor |m|/|m − 1| near
///   m = 1), and 1 + max(1, |m|)·x for the derivatives of the coefficients and amplitudes,
///   which lose nothing near m = 1: m − 1 enters the x-derivatives as an exact factor of
///   m² − 1, and the m-derivatives have no factor that cancels. The phase matrix takes κ and
///   its derivatives κ·max(1, |m|/|m − 1|), like the efficiencies and their Jacobian (the
///   normalized phase matrix divides by Q_sca). The Greek coefficients up to order L add
///   ((L + 1)² + x²)/8 to those: the coefficients are those of a rule whose `f64` nodes differ
///   from the exact ones, by about half an ulp next to μ = 1, where it matters; there such a
///   move changes d_L by about L²·2⁻⁵⁴ and the forward peak of the phase matrix, whose
///   slope in μ grows like x⁴ while the weights there shrink like 1/x², by about x²·2⁻⁵⁴ of
///   its size.
/// * C is the output's constant ([`constant`](Self::constant)).
///
/// Neither term alone is a bound: the accuracy map in the documentation shows outputs that
/// need each. The constants are measured, at least 4 times the worst case found: at every Arb
/// reference value in the domain, and against double-double on a dense grid (the accuracy
/// map). Tests enforce both in CI.
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
    /// The smallest validated |m|. Below it, Q_abs and the Jacobian of nearly real m lose
    /// accuracy to cancellation in Im(m·Dₙ(mx)).
    pub abs_m_min: f64,
}

/// The validated domain: x ∈ [10⁻⁶, 2·10⁴], 0 < Re m ≤ 10, 0 ≤ Im m ≤ 10, |m| ≥ 0.1.
pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
    x_min: 1e-6,
    x_max: 2e4,
    m_re_max: 10.0,
    m_im_max: 10.0,
    abs_m_min: 0.1,
};

impl ValidatedDomain {
    /// Whether (x, m) is inside the domain.
    pub fn contains(&self, x: f64, m: Complex<f64>) -> bool {
        (self.x_min..=self.x_max).contains(&x)
            && m.re > 0.0
            && m.re <= self.m_re_max
            && (0.0..=self.m_im_max).contains(&m.im)
            && m.norm() >= self.abs_m_min
    }

    /// κ(x, m) = (1 + max(1, |m|)·x)·max(1, |m|/|m − 1|): infinite for m = 1, where the
    /// coefficients vanish. The phase of the series grows like |m|x inside the sphere and like
    /// x outside it (and the number of terms like x), hence max(1, |m|)·x.
    pub fn condition_factor(x: f64, m: Complex<f64>) -> f64 {
        (1.0 + m.norm().max(1.0) * x) * near_one(m)
    }

    /// The declared bound on the error of `output` at (x, m): C·(κ′ + `condition`)·2⁻⁵², in
    /// the measure of [`MieOutput`], where `condition` is the output's condition number (see
    /// the type documentation for it and for κ′).
    pub fn tolerance(&self, output: MieOutput, x: f64, m: Complex<f64>, condition: f64) -> f64 {
        let kappa = Self::condition_factor(x, m);
        let quadrature = |order: usize| ((order as f64 + 1.0).powi(2) + x * x) / 8.0;
        let prior = match output {
            MieOutput::CoefficientDerivatives | MieOutput::AmplitudeDerivatives => {
                1.0 + m.norm().max(1.0) * x
            }
            MieOutput::EfficiencyJacobian | MieOutput::PhaseMatrixDerivatives => {
                kappa * near_one(m)
            }
            MieOutput::GreekCoefficients { order } => kappa + quadrature(order),
            MieOutput::GreekDerivatives { order } => kappa * near_one(m) + quadrature(order),
            _ => kappa,
        };
        Self::constant(output) * (prior + condition) * f64::EPSILON
    }

    /// The constant C of [`tolerance`](Self::tolerance).
    ///
    /// | output | C |
    /// |---|---|
    /// | coefficients, efficiencies, g | 16 |
    /// | S₁, S₂ | 128 |
    /// | derivatives of S₁, S₂ | 64 |
    /// | efficiency Jacobian | 256 |
    /// | coefficient derivatives | 512 |
    /// | phase matrix | 32 |
    /// | derivatives of the phase matrix | 128 |
    /// | Greek coefficients | 8 |
    /// | derivatives of the Greek coefficients | 16 |
    pub fn constant(output: MieOutput) -> f64 {
        match output {
            MieOutput::Coefficients
            | MieOutput::Extinction
            | MieOutput::Scattering
            | MieOutput::Absorption
            | MieOutput::Backscattering
            | MieOutput::Asymmetry => 16.0,
            MieOutput::Amplitudes => 128.0,
            MieOutput::AmplitudeDerivatives => 64.0,
            MieOutput::EfficiencyJacobian => 256.0,
            MieOutput::CoefficientDerivatives => 512.0,
            MieOutput::PhaseMatrix => 32.0,
            MieOutput::PhaseMatrixDerivatives => 128.0,
            MieOutput::GreekCoefficients { .. } => 8.0,
            MieOutput::GreekDerivatives { .. } => 16.0,
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
            "x in [{:e}, {:e}], 0 < Re m <= {}, 0 <= Im m <= {}, |m| >= {}",
            self.x_min, self.x_max, self.m_re_max, self.m_im_max, self.abs_m_min
        )
    }
}
