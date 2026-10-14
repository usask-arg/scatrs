//! The validated domain of layered spheres and their declared accuracy.

use core::fmt;

use num_complex::Complex;

use crate::mie::MieOutput;

/// The region of inputs whose accuracy has been validated, and the declared tolerances there.
///
/// Inside it, every output of [`Layered::compute`](super::Layered::compute) in `f64`, with the
/// default truncation or more terms, and of its solution's methods, is within
///
/// C·(κ′(x, m) + cond)·2⁻⁵²
///
/// of the exact value for the exact inputs, in the measure of its [`MieOutput`], with cond the
/// output's relative condition number with respect to every layer's x_l, Re m_l and Im m_l
/// (as [`mie::ValidatedDomain`](crate::mie::ValidatedDomain) defines it for one layer) and κ′
/// the a-priori factor of [`condition_factor`](Self::condition_factor). The constants are
/// measured, at least 4 times the worst case found at every Arb reference value in the domain
/// and against double-double on a dense grid (the accuracy map).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ValidatedDomain {
    /// The smallest validated size parameter of the outer surface.
    pub x_min: f64,
    /// The largest validated size parameter of the outer surface.
    pub x_max: f64,
    /// The largest validated Re m of any layer (the smallest is any value above 0).
    pub m_re_max: f64,
    /// The largest validated Im m of any layer (the smallest is 0).
    pub m_im_max: f64,
    /// The smallest validated |m| of any layer.
    pub abs_m_min: f64,
    /// The largest validated number of layers.
    pub max_layers: usize,
}

/// The validated domain: x_L ∈ [10⁻⁶, 10⁴], every layer's m with 0 < Re m ≤ 10, 0 ≤ Im m ≤ 10
/// and |m| ≥ 0.1, at most 16 layers, any radii.
pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
    x_min: 1e-6,
    x_max: 1e4,
    m_re_max: 10.0,
    m_im_max: 10.0,
    abs_m_min: 0.1,
    max_layers: 16,
};

impl ValidatedDomain {
    /// Whether the layered sphere of interface size parameters `x` (the outer one last) and
    /// indices `m` is inside the domain.
    pub fn contains(&self, x: &[f64], m: &[Complex<f64>]) -> bool {
        let Some(&outer) = x.last() else {
            return false;
        };
        x.len() <= self.max_layers
            && (self.x_min..=self.x_max).contains(&outer)
            && m.iter().all(|m| {
                m.re > 0.0
                    && m.re <= self.m_re_max
                    && (0.0..=self.m_im_max).contains(&m.im)
                    && m.norm() >= self.abs_m_min
            })
    }

    /// κ(x_L, m) = (1 + max_l max(1, |m_l|)·x_L)·max(1, max_l |m_l|/|m_l − 1|): as for a sphere,
    /// with the largest index and the largest near-unity factor of any layer whose index is not
    /// exactly 1 (an index near 1 loses accuracy in that layer's differences, as for a sphere;
    /// an index of exactly 1 makes them exact). Infinite when every layer has m = 1.
    pub fn condition_factor(x: f64, m: &[Complex<f64>]) -> f64 {
        let big = m.iter().map(|m| m.norm()).fold(1.0, f64::max);
        let one = Complex::new(1.0, 0.0);
        let near_one = m
            .iter()
            .filter(|&&m| m != one)
            .map(|m| (m.norm() / (m - one).norm()).max(1.0))
            .fold(f64::NEG_INFINITY, f64::max);
        let near_one = if near_one.is_finite() {
            near_one
        } else {
            f64::INFINITY
        };
        (1.0 + big * x) * near_one
    }

    /// The declared bound on the error of `output`: C·(κ′ + `condition`)·2⁻⁵², with κ′ as for
    /// a sphere ([`mie::ValidatedDomain::tolerance`](crate::mie::ValidatedDomain::tolerance))
    /// with [`condition_factor`](Self::condition_factor) for κ.
    pub fn tolerance(&self, output: MieOutput, x: f64, m: &[Complex<f64>], condition: f64) -> f64 {
        let kappa = Self::condition_factor(x, m);
        let big = m.iter().map(|m| m.norm()).fold(1.0, f64::max);
        let near_one = (kappa / (1.0 + big * x)).max(1.0);
        let quadrature = |order: usize| ((order as f64 + 1.0).powi(2) + x * x) / 8.0;
        let prior = match output {
            MieOutput::CoefficientDerivatives | MieOutput::AmplitudeDerivatives => 1.0 + big * x,
            MieOutput::EfficiencyJacobian | MieOutput::PhaseMatrixDerivatives => kappa * near_one,
            MieOutput::GreekCoefficients { order } => kappa + quadrature(order),
            MieOutput::GreekDerivatives { order } => kappa * near_one + quadrature(order),
            _ => kappa,
        };
        Self::constant(output) * (prior + condition) * f64::EPSILON
    }

    /// The constant C of [`tolerance`](Self::tolerance).
    pub fn constant(output: MieOutput) -> f64 {
        crate::mie::ValidatedDomain::constant(output)
    }
}

impl fmt::Display for ValidatedDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "x_L in [{:e}, {:e}], every layer 0 < Re m <= {}, 0 <= Im m <= {}, |m| >= {}, at \
             most {} layers",
            self.x_min, self.x_max, self.m_re_max, self.m_im_max, self.abs_m_min, self.max_layers
        )
    }
}
