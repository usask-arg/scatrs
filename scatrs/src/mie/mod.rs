//! Lorenz–Mie scattering by homogeneous spheres.
//!
//! [`Mie`] is a reusable workspace: [`Mie::compute`] (or [`Mie::compute_with_derivatives`])
//! solves one sphere and returns a [`MieSolution`], from which the coefficients aₙ, bₙ, the
//! [`Efficiencies`], their [`EfficiencyJacobian`] and the amplitudes S₁, S₂ are read.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::mie::{Mie, MieOptions};
//!
//! let mut mie = Mie::<f64>::new();
//! let sphere = mie.compute(10.0, Complex::new(1.33, 1e-8), &MieOptions::new())?;
//! let q = sphere.efficiencies();
//! assert!((q.q_ext - (q.q_sca + q.q_abs)).abs() <= 1e-15 * q.q_ext);
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! # Conventions
//!
//! Bohren & Huffman (1983), with time dependence e^(−iωt): the refractive index m = n + ik has
//! k ≥ 0 for absorbing spheres, x = 2πr/λ is the size parameter, and
//!
//! aₙ = [(Dₙ(mx)/m + n/x) ψₙ(x) − ψₙ₋₁(x)] / [(Dₙ(mx)/m + n/x) ξₙ(x) − ξₙ₋₁(x)],
//! bₙ = [(m Dₙ(mx) + n/x) ψₙ(x) − ψₙ₋₁(x)] / [(m Dₙ(mx) + n/x) ξₙ(x) − ξₙ₋₁(x)],
//!
//! with ψₙ(ρ) = ρ jₙ(ρ), ξₙ = ψₙ − iχₙ, χₙ(ρ) = −ρ yₙ(ρ) and Dₙ = ψₙ′/ψₙ. Codes using
//! e^(+iωt), such as Wiscombe's MIEV0, report the complex conjugates of aₙ, bₙ, S₁ and S₂.
//!
//! # Method
//!
//! See the theory chapter of the documentation, "Homogeneous spheres", for the derivations.
//!
//! * **Truncation.** N = ⌈x + 6x^(1/3) + 3⌉ terms in `f64` ([`default_n_max`]), and
//!   ⌈x + 12x^(1/3) + 6⌉ in `DoubleDouble`, beyond Wiscombe's ⌈x + 4.05x^(1/3) + 2⌉, which
//!   leaves truncation errors up to 10⁻⁶ in backscattering.
//! * **Dₙ(mx)** by the downward recurrence of uₖ = ψₖ/ψₖ₋₁ = 1/((2k + 1)/(mx) − uₖ₊₁), from
//!   uₖ₊₁ = 0 at k = max(N + 1, ⌈1.5|mx|⌉) plus a margin of about half the bits of precision.
//!   Above 1.5|mx| every step shrinks an error at least sixfold, so the start value is
//!   forgotten to rounding level. Then Dₙ = (n + 1)/(mx) − uₙ₊₁. No continued fraction is
//!   evaluated, and nothing can fail to converge.
//! * **ψₙ and ξₙ.** Up to n = ⌊x⌋ (for x ≥ 1) by upward recurrence; above, through the ratios
//!   Tₙ = ψₙ/ξₙ and 1/ξₙ, built from uₙ(x) (downward, as for Dₙ) and Rₙ = ξₙ₋₁/ξₙ (upward).
//!   The upward ψ recurrence loses about log₁₀(1/x²) digits at small x (ψ₁ = sin x/x − cos x
//!   cancels), and ψₙ underflows while ξₙ overflows as n grows beyond x; the ratios do
//!   neither, so no small-particle expansion is needed and any `n_max` is safe.
//! * **Numerators without cancellation.** In the ratio region, Dₙ(mx)/m − Dₙ(x) =
//!   (n + 1)(1/m² − 1)/x + uₙ₊₁(x) − uₙ₊₁(mx)/m and m·Dₙ(mx) − Dₙ(x) = uₙ₊₁(x) − m·uₙ₊₁(mx):
//!   the terms of size n/x that cancel are removed exactly.
//! * **Absorption.** Re aₙ − |aₙ|² = −Im(Dₙ(mx)/m)/|Δₐ|² and Re bₙ − |bₙ|² =
//!   −Im(m Dₙ(mx))/|Δ_b|², where Δ are the denominators above; Q_abs is summed from these
//!   terms, without the cancellation of Q_ext − Q_sca, and Q_ext = Q_sca + Q_abs.
//! * **Derivatives** in closed form (each a few operations per order):
//!   ∂aₙ/∂m = i(D − zD′)/(m²Δₐ²), ∂bₙ/∂m = −i(D + zD′)/Δ_b²,
//!   ∂aₙ/∂x = i(m² − 1)(n(n + 1)/x² + D²)/(m²Δₐ²), ∂bₙ/∂x = i(m² − 1)/Δ_b²,
//!   with z = mx, D = Dₙ(z) and D′ = n(n + 1)/z² − 1 − D². aₙ and bₙ are holomorphic in m,
//!   so ∂/∂(Re m) = d/dm and ∂/∂(Im m) = i·d/dm.
//!
//! # Accuracy
//!
//! [`VALIDATED_DOMAIN`] is the domain where the accuracy is validated.
//!
//! # Generic scalars
//!
//! Every method is generic over [`Real`]: `f64`, `DoubleDouble` and `Dual<T, N>`. With
//! `Dual`, the derivatives are those of the computed values with respect to whatever the
//! caller seeds in x and m, and the values are bitwise those of the underlying type: the
//! derivative oracle for the closed forms.

mod amplitudes;
mod domain;
mod efficiencies;
mod kernel;

use num_complex::Complex;

use crate::scalar::Real;
use crate::{Error, Result};

pub use amplitudes::AmplitudeBuffers;
pub use domain::{VALIDATED_DOMAIN, ValidatedDomain};
pub use efficiencies::{Efficiencies, EfficiencyJacobian};

/// The largest number of terms [`Mie::compute`] accepts, 2²⁰.
///
/// It bounds the workspace to about 200 MB with derivatives. Larger `n_max`, or a size
/// parameter whose default truncation exceeds it, raise [`Error::InvalidInput`].
pub const MAX_MIE_ORDER: usize = 1 << 20;
/// The smallest size parameter accepted. Below it the coefficients (∝ x³) and efficiencies
/// (∝ x⁴) leave the normal `f64` range.
pub const MIN_SIZE_PARAMETER: f64 = 1e-30;
/// The largest size parameter accepted (its default truncation is below [`MAX_MIE_ORDER`]).
pub const MAX_SIZE_PARAMETER: f64 = 1e6;
/// The smallest |m| accepted.
pub const MIN_REFRACTIVE_INDEX: f64 = 1e-6;
/// The largest |m| accepted. The cost of Dₙ(mx) grows like |m|x.
pub const MAX_REFRACTIVE_INDEX: f64 = 1e3;

/// The default number of terms for `f64`: N = ⌈x + 6x^(1/3) + 3⌉.
///
/// Wiscombe's ⌈x + 4.05x^(1/3) + 2⌉ (MIEV0) leaves truncation errors up to about 10⁻¹⁰ in
/// Q_ext and 10⁻⁶ in Q_back (measured against Arb). This rule was chosen from the same
/// measurements so that truncation errors are below rounding errors in `f64`.
pub fn default_n_max(x: f64) -> usize {
    terms(x, 6.0, 3.0)
}

/// ⌈x + c·x^(1/3) + d⌉ (`usize::MAX` if not finite, which every limit rejects).
fn terms(x: f64, c: f64, d: f64) -> usize {
    let n = (x + c * x.cbrt() + d).ceil();
    if n.is_finite() && n < 2f64.powi(53) {
        n as usize
    } else {
        usize::MAX
    }
}

/// The default truncation for scalar type `T`: [`default_n_max`] for `f64` (and `Dual<f64>`),
/// ⌈x + 12x^(1/3) + 6⌉ for `DoubleDouble`.
fn default_terms<T: Real>(x: f64) -> usize {
    if T::epsilon().approx_f64() < 2f64.powi(-60) {
        terms(x, 12.0, 6.0)
    } else {
        default_n_max(x)
    }
}

/// Options for [`Mie::compute`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct MieOptions {
    /// The number of terms; `None` for the default (see [`default_n_max`]). Fewer terms than
    /// the default are outside the validated domain.
    pub n_max: Option<usize>,
    /// Compute even outside [`VALIDATED_DOMAIN`] instead of returning
    /// [`Error::OutsideValidatedDomain`].
    pub allow_unvalidated: bool,
}

impl MieOptions {
    /// The defaults: the default truncation, and only inside the validated domain.
    pub fn new() -> Self {
        Self::default()
    }

    /// With `n_max` terms.
    #[must_use]
    pub fn with_n_max(mut self, n_max: usize) -> Self {
        self.n_max = Some(n_max);
        self
    }

    /// Allowing (or not) inputs outside the validated domain.
    #[must_use]
    pub fn with_allow_unvalidated(mut self, allow: bool) -> Self {
        self.allow_unvalidated = allow;
        self
    }
}

/// A reusable workspace for Mie calculations, one per thread.
///
/// Its buffers grow to the largest number of terms used and are then reused, so repeated
/// calls do not allocate.
#[derive(Clone, Debug)]
pub struct Mie<T: Real> {
    state: kernel::State<T>,
}

impl<T: Real> Default for Mie<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> Mie<T> {
    /// An empty workspace.
    pub fn new() -> Self {
        Self {
            state: kernel::State::new(),
        }
    }

    /// Solves the sphere of size parameter `x` and refractive index `m`: the coefficients and
    /// everything derived from them, without derivatives.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] if x is not in [[`MIN_SIZE_PARAMETER`],
    ///   [`MAX_SIZE_PARAMETER`]], if m is not finite, if Re m < 0 or Im m < 0 (gain media), if
    ///   |m| is not in [[`MIN_REFRACTIVE_INDEX`], [`MAX_REFRACTIVE_INDEX`]], or if the number of
    ///   terms is 0 or above [`MAX_MIE_ORDER`].
    /// * [`Error::OutsideValidatedDomain`] if (x, m) is outside [`VALIDATED_DOMAIN`] or `n_max`
    ///   is below the default, unless `allow_unvalidated` is set.
    ///
    /// For `Dual` inputs the checks apply to the values.
    pub fn compute(
        &mut self,
        x: T,
        m: Complex<T>,
        opts: &MieOptions,
    ) -> Result<MieSolution<'_, T>> {
        self.solve(x, m, opts, false)
    }

    /// As [`compute`](Self::compute), also computing the derivatives of the coefficients
    /// with respect to x and m, which [`MieSolution::derivatives`],
    /// [`MieSolution::efficiencies_jacobian`] and [`MieSolution::amplitudes_with_derivatives`]
    /// need.
    ///
    /// # Errors
    ///
    /// As [`compute`](Self::compute).
    pub fn compute_with_derivatives(
        &mut self,
        x: T,
        m: Complex<T>,
        opts: &MieOptions,
    ) -> Result<MieSolution<'_, T>> {
        self.solve(x, m, opts, true)
    }

    fn solve(
        &mut self,
        x: T,
        m: Complex<T>,
        opts: &MieOptions,
        derivatives: bool,
    ) -> Result<MieSolution<'_, T>> {
        let n = check_inputs::<T>(x.approx_f64(), m.re.approx_f64(), m.im.approx_f64(), opts)?;
        self.state.solve(x, m, n, derivatives);
        Ok(MieSolution { state: &self.state })
    }
}

/// Validates the inputs and returns the number of terms.
fn check_inputs<T: Real>(x: f64, m_re: f64, m_im: f64, opts: &MieOptions) -> Result<usize> {
    if !(MIN_SIZE_PARAMETER..=MAX_SIZE_PARAMETER).contains(&x) {
        return Err(Error::InvalidInput(format!(
            "the size parameter x must be in [{MIN_SIZE_PARAMETER:e}, {MAX_SIZE_PARAMETER:e}], \
             got {x:e}"
        )));
    }
    if !(m_re.is_finite() && m_im.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "the refractive index must be finite, got {m_re:e} + {m_im:e}i"
        )));
    }
    if m_re < 0.0 || m_im < 0.0 {
        return Err(Error::InvalidInput(format!(
            "the refractive index must have Re m >= 0 and Im m >= 0 (Im m < 0 is a gain medium), \
             got {m_re:e} + {m_im:e}i"
        )));
    }
    let abs_m = m_re.hypot(m_im);
    if !(MIN_REFRACTIVE_INDEX..=MAX_REFRACTIVE_INDEX).contains(&abs_m) {
        return Err(Error::InvalidInput(format!(
            "|m| must be in [{MIN_REFRACTIVE_INDEX:e}, {MAX_REFRACTIVE_INDEX:e}], got {abs_m:e}"
        )));
    }
    let default = default_terms::<T>(x);
    let n = opts.n_max.unwrap_or(default);
    if n == 0 || n > MAX_MIE_ORDER {
        return Err(Error::InvalidInput(format!(
            "the number of terms must be in [1, {MAX_MIE_ORDER}], got {n}"
        )));
    }
    if !opts.allow_unvalidated {
        let m = Complex::new(m_re, m_im);
        if !VALIDATED_DOMAIN.contains(x, m) {
            return Err(Error::OutsideValidatedDomain(format!(
                "x = {x:e}, m = {m_re:e} + {m_im:e}i is outside the validated domain {}",
                VALIDATED_DOMAIN
            )));
        }
        if n < default {
            return Err(Error::OutsideValidatedDomain(format!(
                "n_max = {n} is below the default truncation {default} for x = {x:e}"
            )));
        }
    }
    Ok(n)
}

/// The solution for one sphere: a view of the workspace that computed it.
#[derive(Clone, Copy, Debug)]
pub struct MieSolution<'a, T: Real> {
    state: &'a kernel::State<T>,
}

/// The derivatives of the coefficients, indexed like [`MieSolution::a`].
///
/// d/dm is the complex derivative: ∂/∂(Re m) = d/dm and ∂/∂(Im m) = i·d/dm.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CoefficientDerivatives<'a, T> {
    /// ∂aₙ/∂x.
    pub da_dx: &'a [Complex<T>],
    /// daₙ/dm.
    pub da_dm: &'a [Complex<T>],
    /// ∂bₙ/∂x.
    pub db_dx: &'a [Complex<T>],
    /// dbₙ/dm.
    pub db_dm: &'a [Complex<T>],
}

impl<'a, T: Real> MieSolution<'a, T> {
    /// The size parameter.
    pub fn x(&self) -> T {
        self.state.x
    }

    /// The refractive index.
    pub fn m(&self) -> Complex<T> {
        self.state.m
    }

    /// The number of terms N.
    pub fn n_terms(&self) -> usize {
        self.state.a.len()
    }

    /// aₙ for n = 1, …, N (index n − 1).
    pub fn a(&self) -> &'a [Complex<T>] {
        &self.state.a
    }

    /// bₙ for n = 1, …, N (index n − 1).
    pub fn b(&self) -> &'a [Complex<T>] {
        &self.state.b
    }

    /// The derivatives of the coefficients, if computed by
    /// [`Mie::compute_with_derivatives`].
    pub fn derivatives(&self) -> Option<CoefficientDerivatives<'a, T>> {
        self.state.derivatives.then(|| CoefficientDerivatives {
            da_dx: &self.state.da_dx,
            da_dm: &self.state.da_dm,
            db_dx: &self.state.db_dx,
            db_dm: &self.state.db_dm,
        })
    }
}
