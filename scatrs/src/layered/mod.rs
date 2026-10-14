//! Scattering by layered spheres: a core and any number of concentric shells.
//!
//! [`Layered`] is a reusable workspace, as [`Mie`](crate::mie::Mie) is: [`Layered::compute`]
//! (or [`Layered::compute_with_derivatives`]) solves one layered sphere and returns a
//! [`LayeredSolution`], with the same outputs as a sphere's [`MieSolution`](crate::mie::MieSolution):
//! the coefficients aₙ, bₙ, the efficiencies, amplitudes, phase matrix and Greek coefficients,
//! and their derivatives with respect to each layer's size parameter and refractive index.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::layered::Layered;
//! use scatrs::mie::MieOptions;
//!
//! // Black carbon coated in sulfate: core and outer size parameters, then their indices.
//! let x = [0.7, 1.3];
//! let m = [Complex::new(1.95, 0.79), Complex::new(1.43, 1e-8)];
//! let mut layered = Layered::<f64>::new();
//! let particle = layered.compute(&x, &m, &MieOptions::new())?;
//! let q = particle.efficiencies();
//! assert!(q.q_abs > 0.0 && (q.q_ext - (q.q_sca + q.q_abs)).abs() <= 1e-15 * q.q_ext);
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! # Conventions
//!
//! Those of [`crate::mie`]. Layers are numbered from the core (layer 0) outward: layer l lies
//! between the radii r_{l−1} and r_l (r₋₁ = 0), `x[l]` = 2πr_l/λ is the size parameter of its
//! outer surface and `m[l]` its refractive index relative to the medium. The outer size
//! parameter x_L = `x[L − 1]` normalizes the efficiencies and the phase matrix, and sets the
//! number of terms.
//!
//! # Method
//!
//! See the theory chapter of the documentation, "Layered spheres". The effective
//! log-derivatives Hₐ and H_b at the outer surface of each layer follow from the core outward
//! (Yang 2003; Peña & Pal 2009) in a form without the poles of D_n(m_l x_l), and the outer step
//! is the homogeneous sphere's with Hₐ and H_b in place of D_n(mx), keeping its numerators free
//! of cancellation. A single layer is a sphere: it gives [`crate::mie`]'s results bitwise.
//! The derivatives with respect to every layer's x and m are in closed form: within a layer, H
//! solves a Riccati equation, whose sensitivities to its two surfaces and to its starting value
//! are chained from the outer layer inward at each order (for one layer, the sphere's).
//!
//! # Accuracy
//!
//! See [`VALIDATED_DOMAIN`].

mod domain;
mod kernel;

use num_complex::Complex;

pub use domain::{VALIDATED_DOMAIN, ValidatedDomain};

use crate::mie::{
    AmplitudeBuffers, AngularTable, CoefficientDerivatives, Coefficients, Efficiencies,
    EfficiencyJacobian, GreekCoefficients, Group, MAX_MIE_ORDER, MAX_REFRACTIVE_INDEX,
    MAX_SIZE_PARAMETER, MIN_REFRACTIVE_INDEX, MIN_SIZE_PARAMETER, MieOptions, Normalization,
    PhaseBuffers, default_terms,
};
use crate::scalar::Real;
use crate::{Error, Result};

/// The largest number of layers [`Layered::compute`] accepts, 2¹⁰.
pub const MAX_LAYERS: usize = 1 << 10;

/// The derivatives of one layer's coefficients and of the absorption sum.
#[derive(Clone, Debug)]
struct LayerDerivatives<T> {
    da_dx: Vec<Complex<T>>,
    da_dm: Vec<Complex<T>>,
    db_dx: Vec<Complex<T>>,
    db_dm: Vec<Complex<T>>,
    abs_sum: [T; 3],
}

impl<T: Real> LayerDerivatives<T> {
    fn new() -> Self {
        Self {
            da_dx: Vec::new(),
            da_dm: Vec::new(),
            db_dx: Vec::new(),
            db_dm: Vec::new(),
            abs_sum: [T::zero(); 3],
        }
    }
}

/// A reusable workspace for layered spheres, one per thread.
///
/// Its buffers grow to the largest number of terms and layers used and are then reused.
#[derive(Clone, Debug)]
pub struct Layered<T: Real> {
    kernel: kernel::Kernel<T>,
    x: T,
    a: Vec<Complex<T>>,
    b: Vec<Complex<T>>,
    abs_sum: T,
    derivatives: bool,
    layers: Vec<LayerDerivatives<T>>,
}

impl<T: Real> Default for Layered<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> Layered<T> {
    /// An empty workspace.
    pub fn new() -> Self {
        Self {
            kernel: kernel::Kernel::new(),
            x: T::zero(),
            a: Vec::new(),
            b: Vec::new(),
            abs_sum: T::zero(),
            derivatives: false,
            layers: Vec::new(),
        }
    }

    /// Solves the layered sphere of interface size parameters `x` (the core's first, the outer
    /// surface's last) and layer refractive indices `m`: the coefficients and everything
    /// derived from them, without derivatives.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] if `x` and `m` are empty or of different lengths, if there are
    ///   more than [`MAX_LAYERS`] layers, if `x` is not nondecreasing, if x₀ is below
    ///   [`MIN_SIZE_PARAMETER`] or x_L above [`MAX_SIZE_PARAMETER`], if any m is not finite, has
    ///   Re m < 0 or Im m < 0, or |m| outside [[`MIN_REFRACTIVE_INDEX`],
    ///   [`MAX_REFRACTIVE_INDEX`]], or if the number of terms is 0 or above [`MAX_MIE_ORDER`].
    /// * [`Error::OutsideValidatedDomain`] if the inputs are outside [`VALIDATED_DOMAIN`] or
    ///   `n_max` is below the default, unless `allow_unvalidated` is set.
    ///
    /// For `Dual` inputs the checks apply to the values.
    pub fn compute(
        &mut self,
        x: &[T],
        m: &[Complex<T>],
        opts: &MieOptions,
    ) -> Result<LayeredSolution<'_, T>> {
        let n = check_inputs(x, m, opts)?;
        let last = x.len() - 1;
        if last == 0 {
            self.kernel.mie.solve(x[0], m[0], n, false);
        } else {
            self.kernel.solve(x, m, n, false);
        }
        let s = &self.kernel.mie;
        self.x = x[last];
        self.a.clone_from(&s.a);
        self.b.clone_from(&s.b);
        self.abs_sum = s.abs_sum;
        self.derivatives = false;
        Ok(LayeredSolution { s: self })
    }

    /// As [`compute`](Self::compute), also computing the derivatives of the coefficients with
    /// respect to each layer's size parameter and refractive index, in closed form: the
    /// sensitivities of each layer's logarithmic derivative (a Riccati equation's), combined by
    /// an adjoint sweep over the layers at every order, and the sphere's closed forms at the
    /// outer surface (for one layer, the sphere's own). The values are bitwise those of
    /// [`compute`](Self::compute).
    ///
    /// # Errors
    ///
    /// As [`compute`](Self::compute), and [`Error::OutsideValidatedDomain`] above the
    /// derivatives' validated x_L ([`ValidatedDomain::derivative_x_max`]) unless
    /// `allow_unvalidated` is set.
    pub fn compute_with_derivatives(
        &mut self,
        x: &[T],
        m: &[Complex<T>],
        opts: &MieOptions,
    ) -> Result<LayeredSolution<'_, T>> {
        let n = check_inputs(x, m, opts)?;
        let outer = x[x.len() - 1].approx_f64();
        if !opts.allow_unvalidated && x.len() > 1 && outer > VALIDATED_DOMAIN.derivative_x_max {
            return Err(Error::OutsideValidatedDomain(format!(
                "the derivatives are validated up to x_L = {:e}, got {outer:e}",
                VALIDATED_DOMAIN.derivative_x_max
            )));
        }
        let layers = x.len();
        let last = layers - 1;
        self.x = x[last];
        self.layers.resize_with(layers, LayerDerivatives::new);
        self.layers.truncate(layers);
        if last == 0 {
            self.kernel.mie.solve(x[0], m[0], n, true);
        } else {
            self.kernel.solve(x, m, n, true);
            for (d, g) in self.layers.iter_mut().zip(&self.kernel.grads) {
                d.da_dx.clone_from(&g.da_dx);
                d.da_dm.clone_from(&g.da_dm);
                d.db_dx.clone_from(&g.db_dx);
                d.db_dm.clone_from(&g.db_dm);
                d.abs_sum = g.abs_sum;
            }
        }
        let s = &self.kernel.mie;
        self.a.clone_from(&s.a);
        self.b.clone_from(&s.b);
        self.abs_sum = s.abs_sum;
        let d = &mut self.layers[last];
        d.da_dx.clone_from(&s.da_dx);
        d.da_dm.clone_from(&s.da_dm);
        d.db_dx.clone_from(&s.db_dx);
        d.db_dm.clone_from(&s.db_dm);
        d.abs_sum = s.abs_sum_jacobian;
        self.derivatives = true;
        Ok(LayeredSolution { s: self })
    }
}

/// The number of terms [`Layered::compute`] would use for `f64` at (`x`, `m`) with `opts`
/// (the sphere's default at x_L unless `opts.n_max` is set), after the same checks, without
/// solving: for sizing outputs before computing them.
///
/// # Errors
///
/// As [`Layered::compute`].
pub fn n_terms(x: &[f64], m: &[Complex<f64>], opts: &MieOptions) -> Result<usize> {
    check_inputs(x, m, opts)
}

/// Validates the inputs and returns the number of terms.
fn check_inputs<T: Real>(x: &[T], m: &[Complex<T>], opts: &MieOptions) -> Result<usize> {
    let invalid = |s: String| Err(Error::InvalidInput(s));
    let layers = x.len();
    if layers == 0 || m.len() != layers {
        return invalid(format!(
            "a layered sphere needs one size parameter and one refractive index per layer, got \
             {} and {}",
            layers,
            m.len()
        ));
    }
    if layers > MAX_LAYERS {
        return invalid(format!(
            "at most {MAX_LAYERS} layers are accepted, got {layers}"
        ));
    }
    let xs: Vec<f64> = x.iter().map(|v| v.approx_f64()).collect();
    let ms: Vec<Complex<f64>> = m
        .iter()
        .map(|v| Complex::new(v.re.approx_f64(), v.im.approx_f64()))
        .collect();
    if let Some((l, v)) = xs.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return invalid(format!(
            "the size parameter of layer {l} must be finite, got {v:e}"
        ));
    }
    if let Some(l) = (1..layers).find(|&l| xs[l] < xs[l - 1]) {
        return invalid(format!(
            "the size parameters must be nondecreasing from the core out, but layer {l} has \
             {:e} < {:e}",
            xs[l],
            xs[l - 1]
        ));
    }
    let outer = xs[layers - 1];
    if xs[0] < MIN_SIZE_PARAMETER || outer > MAX_SIZE_PARAMETER {
        return invalid(format!(
            "the size parameters must be in [{MIN_SIZE_PARAMETER:e}, {MAX_SIZE_PARAMETER:e}], \
             got {:e} to {outer:e}",
            xs[0]
        ));
    }
    for (l, v) in ms.iter().enumerate() {
        if !(v.re.is_finite() && v.im.is_finite()) {
            return invalid(format!(
                "the refractive index of layer {l} must be finite, got {:e} + {:e}i",
                v.re, v.im
            ));
        }
        if v.re < 0.0 || v.im < 0.0 {
            return invalid(format!(
                "the refractive index of layer {l} must have Re m >= 0 and Im m >= 0 (Im m < 0 \
                 is a gain medium), got {:e} + {:e}i",
                v.re, v.im
            ));
        }
        if !(MIN_REFRACTIVE_INDEX..=MAX_REFRACTIVE_INDEX).contains(&v.norm()) {
            return invalid(format!(
                "|m| of layer {l} must be in [{MIN_REFRACTIVE_INDEX:e}, \
                 {MAX_REFRACTIVE_INDEX:e}], got {:e}",
                v.norm()
            ));
        }
    }
    let default = default_terms::<T>(outer);
    let n = opts.n_max.unwrap_or(default);
    if n == 0 || n > MAX_MIE_ORDER {
        return invalid(format!(
            "the number of terms must be in [1, {MAX_MIE_ORDER}], got {n}"
        ));
    }
    if !opts.allow_unvalidated {
        if !VALIDATED_DOMAIN.contains(&xs, &ms) {
            return Err(Error::OutsideValidatedDomain(format!(
                "a layered sphere of {layers} layers with x_L = {outer:e} and m = {ms:?} is \
                 outside the validated domain {VALIDATED_DOMAIN}"
            )));
        }
        if n < default {
            return Err(Error::OutsideValidatedDomain(format!(
                "n_max = {n} is below the default truncation {default} for x_L = {outer:e}"
            )));
        }
    }
    // The values are checked above; for `Dual` this also rejects non-finite derivative parts.
    if !(x.iter().all(|v| v.is_finite()) && m.iter().all(|v| v.re.is_finite() && v.im.is_finite()))
    {
        return invalid("the derivative parts of x and m must be finite".into());
    }
    Ok(n)
}

/// The solution for one layered sphere: a view of the workspace that computed it.
#[derive(Clone, Copy, Debug)]
pub struct LayeredSolution<'a, T: Real> {
    s: &'a Layered<T>,
}

/// The error for derivatives of a solution computed without them.
fn no_derivatives(what: &str) -> Error {
    Error::InvalidInput(format!(
        "{what} derivatives need a solution from Layered::compute_with_derivatives"
    ))
}

impl<'a, T: Real> LayeredSolution<'a, T> {
    /// The size parameter of the outer surface, x_L.
    pub fn x(&self) -> T {
        self.s.x
    }

    /// The number of terms N.
    pub fn n_terms(&self) -> usize {
        self.s.a.len()
    }

    /// aₙ for n = 1, …, N (index n − 1).
    pub fn a(&self) -> &'a [Complex<T>] {
        &self.s.a
    }

    /// bₙ for n = 1, …, N (index n − 1).
    pub fn b(&self) -> &'a [Complex<T>] {
        &self.s.b
    }

    /// The number of layers whose derivatives were computed (0 without derivatives).
    fn layers(&self) -> usize {
        if self.s.derivatives {
            self.s.layers.len()
        } else {
            0
        }
    }

    /// The derivatives of the coefficients with respect to the size parameter x_l and the
    /// refractive index m_l of `layer` (the complex derivative: ∂/∂(Re m_l) = d/dm_l and
    /// ∂/∂(Im m_l) = i·d/dm_l), if computed by
    /// [`Layered::compute_with_derivatives`] and `layer` exists.
    pub fn derivatives(&self, layer: usize) -> Option<CoefficientDerivatives<'a, T>> {
        (layer < self.layers()).then(|| {
            let d = &self.s.layers[layer];
            CoefficientDerivatives {
                da_dx: &d.da_dx,
                da_dm: &d.da_dm,
                db_dx: &d.db_dx,
                db_dm: &d.db_dm,
            }
        })
    }

    pub(crate) fn coefficients(&self) -> Coefficients<'a, T> {
        Coefficients {
            x: self.s.x,
            a: &self.s.a,
            b: &self.s.b,
            abs_sum: self.s.abs_sum,
        }
    }

    /// The derivative group of `layer`, if computed.
    pub(crate) fn group(&self, layer: usize) -> Option<Group<'a, T>> {
        self.derivatives(layer).map(|d| Group {
            d,
            abs_sum: self.s.layers[layer].abs_sum,
            outer: layer + 1 == self.s.layers.len(),
        })
    }

    fn group_or(&self, layer: usize, what: &str) -> Result<Group<'a, T>> {
        if !self.s.derivatives {
            return Err(no_derivatives(what));
        }
        self.group(layer).ok_or_else(|| {
            Error::InvalidInput(format!(
                "no layer {layer}: the particle has {} layers",
                self.s.layers.len()
            ))
        })
    }

    /// The efficiencies and the asymmetry parameter, per unit geometric cross section of the
    /// outer surface.
    pub fn efficiencies(&self) -> Efficiencies<T> {
        self.coefficients().efficiencies()
    }

    /// The derivatives of the efficiencies with respect to each layer's x_l, Re m_l and Im m_l
    /// (one [`EfficiencyJacobian`] per layer), if computed.
    pub fn efficiencies_jacobian(&self) -> Option<Vec<EfficiencyJacobian<T>>> {
        if !self.s.derivatives {
            return None;
        }
        let c = self.coefficients();
        Some(
            (0..self.layers())
                .map(|l| c.efficiencies_jacobian(&self.group(l).expect("a layer")))
                .collect(),
        )
    }

    /// S₁ and S₂ at the cosines `mu`, as [`MieSolution::amplitudes`](crate::mie::MieSolution::amplitudes).
    ///
    /// # Errors
    ///
    /// As [`MieSolution::amplitudes`](crate::mie::MieSolution::amplitudes).
    pub fn amplitudes(&self, mu: &[T], out: AmplitudeBuffers<'_, T>) -> Result<()> {
        self.coefficients().amplitudes(mu, out)
    }

    /// S₁ and S₂ at the cosines of `table`, bitwise as [`amplitudes`](Self::amplitudes).
    ///
    /// # Errors
    ///
    /// As [`MieSolution::amplitudes_from_table`](crate::mie::MieSolution::amplitudes_from_table).
    pub fn amplitudes_from_table(
        &self,
        table: &AngularTable<T>,
        out: AmplitudeBuffers<'_, T>,
    ) -> Result<()> {
        self.coefficients().amplitudes_from_table(table, out)
    }

    /// S₁, S₂ and their derivatives with respect to `layer`'s x_l (`d_dx`) and m_l (`d_dm`,
    /// the complex derivative).
    ///
    /// # Errors
    ///
    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] without derivatives or
    /// for a layer that does not exist.
    pub fn amplitudes_with_derivatives(
        &self,
        mu: &[T],
        layer: usize,
        values: AmplitudeBuffers<'_, T>,
        d_dx: AmplitudeBuffers<'_, T>,
        d_dm: AmplitudeBuffers<'_, T>,
    ) -> Result<()> {
        let g = self.group_or(layer, "amplitude")?;
        self.coefficients()
            .amplitudes_with_derivatives(mu, &g, values, d_dx, d_dm)
    }

    /// As [`amplitudes_with_derivatives`](Self::amplitudes_with_derivatives) at the cosines of
    /// `table` (bitwise).
    ///
    /// # Errors
    ///
    /// As [`amplitudes_from_table`](Self::amplitudes_from_table), and [`Error::InvalidInput`]
    /// without derivatives or for a layer that does not exist.
    pub fn amplitudes_with_derivatives_from_table(
        &self,
        table: &AngularTable<T>,
        layer: usize,
        values: AmplitudeBuffers<'_, T>,
        d_dx: AmplitudeBuffers<'_, T>,
        d_dm: AmplitudeBuffers<'_, T>,
    ) -> Result<()> {
        let g = self.group_or(layer, "amplitude")?;
        self.coefficients()
            .amplitudes_with_derivatives_from_table(table, &g, values, d_dx, d_dm)
    }

    /// The phase matrix at the cosines `mu`, as
    /// [`MieSolution::phase_matrix`](crate::mie::MieSolution::phase_matrix).
    ///
    /// # Errors
    ///
    /// As [`MieSolution::phase_matrix`](crate::mie::MieSolution::phase_matrix).
    pub fn phase_matrix(
        &self,
        mu: &[T],
        normalization: Normalization,
        out: PhaseBuffers<'_, T>,
    ) -> Result<()> {
        self.coefficients().phase_matrix(mu, normalization, out)
    }

    /// The phase matrix at the cosines of `table`, bitwise as
    /// [`phase_matrix`](Self::phase_matrix).
    ///
    /// # Errors
    ///
    /// As [`amplitudes_from_table`](Self::amplitudes_from_table).
    pub fn phase_matrix_from_table(
        &self,
        table: &AngularTable<T>,
        normalization: Normalization,
        out: PhaseBuffers<'_, T>,
    ) -> Result<()> {
        self.coefficients()
            .phase_matrix_from_table(table, normalization, out)
    }

    /// The phase matrix and its derivatives with respect to `layer`'s x_l, Re m_l and Im m_l, in
    /// `d[0]`, `d[1]` and `d[2]`.
    ///
    /// # Errors
    ///
    /// As [`phase_matrix`](Self::phase_matrix), and [`Error::InvalidInput`] without derivatives
    /// or for a layer that does not exist.
    pub fn phase_matrix_with_derivatives(
        &self,
        mu: &[T],
        layer: usize,
        normalization: Normalization,
        values: PhaseBuffers<'_, T>,
        d: [PhaseBuffers<'_, T>; 3],
    ) -> Result<()> {
        let g = self.group_or(layer, "phase-matrix")?;
        self.coefficients()
            .phase_matrix_with_derivatives(mu, &g, normalization, values, d)
    }

    /// As [`phase_matrix_with_derivatives`](Self::phase_matrix_with_derivatives) at the cosines
    /// of `table` (bitwise).
    ///
    /// # Errors
    ///
    /// As [`phase_matrix_from_table`](Self::phase_matrix_from_table), and
    /// [`Error::InvalidInput`] without derivatives or for a layer that does not exist.
    pub fn phase_matrix_with_derivatives_from_table(
        &self,
        table: &AngularTable<T>,
        layer: usize,
        normalization: Normalization,
        values: PhaseBuffers<'_, T>,
        d: [PhaseBuffers<'_, T>; 3],
    ) -> Result<()> {
        let g = self.group_or(layer, "phase-matrix")?;
        self.coefficients()
            .phase_matrix_with_derivatives_from_table(table, &g, normalization, values, d)
    }

    /// The Greek coefficients, as [`MieSolution::greek`](crate::mie::MieSolution::greek).
    ///
    /// # Errors
    ///
    /// As [`MieSolution::greek`](crate::mie::MieSolution::greek).
    pub fn greek(
        &self,
        order: Option<usize>,
        normalization: Normalization,
    ) -> Result<GreekCoefficients<T>> {
        self.coefficients().greek(order, normalization)
    }

    /// The Greek coefficients and their derivatives with respect to `layer`'s x_l, Re m_l and
    /// Im m_l.
    ///
    /// # Errors
    ///
    /// As [`greek`](Self::greek), and [`Error::InvalidInput`] without derivatives or for a
    /// layer that does not exist.
    pub fn greek_with_derivatives(
        &self,
        order: Option<usize>,
        layer: usize,
        normalization: Normalization,
    ) -> Result<(GreekCoefficients<T>, [GreekCoefficients<T>; 3])> {
        let g = self.group_or(layer, "Greek")?;
        self.coefficients()
            .greek_with_derivatives(order, &g, normalization)
    }
}
