//! Ensembles of particles: size distributions, adaptive size integration and external
//! mixtures.
//!
//! [`compute`] averages the optical properties of spheres over a [`SizeDistribution`] at each
//! wavelength, and [`compute_with_particle`] those of any [`Particle`] (a layered sphere's size
//! is its outer radius, with each interface at a fixed fraction of it; a spheroid's the radius
//! of the sphere of equal volume, its T-matrix solved at every size node); [`compute_mixture`]
//! averages over several [`Component`]s, and [`compute_batch`] over each of many distributions
//! of one material (an optics database), sharing their solutions. The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the
//! cross sections and g meet `rtol`; the distribution's moments and normalization are
//! integrated separately), and every output comes with an error estimate and, on request,
//! derivatives with respect to the distribution parameters, the refractive indices and the
//! mixture weights. See the theory chapter of the documentation.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::ensemble::{EnsembleOptions, GreekOrder, SizeDistribution, compute};
//!
//! let aerosol = SizeDistribution::LogNormal { median_radius: 0.08, width: 1.6 };
//! let options = EnsembleOptions::new().with_greek(GreekOrder::Order(16));
//! let result = compute(&aerosol, &[0.5], &[Complex::new(1.43, 1e-8)], &options)?;
//! let r = &result[0];
//! assert!(r.converged && r.values.ssa > 0.99);
//! assert_eq!(r.values.greek.as_ref().unwrap().alpha1[0], 1.0);
//! # Ok::<(), scatrs::Error>(())
//! ```

mod distribution;
mod engine;
mod fast;
mod kronrod;
mod lin;
mod matrix;
mod moments;
mod particle;
mod range;

use num_complex::Complex;

pub use distribution::SizeDistribution;

use crate::mie::{GreekCoefficients, MAX_GREEK_ORDER, Normalization};
use crate::{Error, Result};

/// The particles of a component, given at each wavelength. A size distribution describes the
/// radius the particle defines: the radius of a sphere, the outer radius of a layered sphere,
/// the radius of the sphere of equal volume of a spheroid.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Particle {
    /// Homogeneous spheres.
    Sphere {
        /// The refractive index at each wavelength.
        refractive_index: Vec<Complex<f64>>,
    },
    /// Layered spheres: a core and concentric shells ([`crate::layered`]), with the outer radius
    /// of each layer but the last a fixed fraction of the particle's.
    LayeredSphere {
        /// The outer radius of layers 0, …, L − 2 (the core first) as fractions of the
        /// particle's radius, nondecreasing in (0, 1]: L − 1 values.
        radius_fractions: Vec<f64>,
        /// The refractive index of each layer (the core first) at each wavelength:
        /// `refractive_index[l][i]` is layer l's at wavelength i.
        refractive_index: Vec<Vec<Complex<f64>>>,
    },
    /// Randomly oriented spheroids ([`crate::tmatrix`]) of one aspect ratio a/c (oblate above
    /// 1, prolate below 1), the size distribution describing the radius of the sphere of equal
    /// volume.
    Spheroid {
        /// The aspect ratio a/c, > 0.
        aspect_ratio: f64,
        /// The refractive index at each wavelength.
        refractive_index: Vec<Complex<f64>>,
    },
}

impl Particle {
    /// Homogeneous spheres of refractive index `refractive_index[i]` at wavelength i.
    pub fn sphere(refractive_index: Vec<Complex<f64>>) -> Self {
        Particle::Sphere { refractive_index }
    }

    /// Layered spheres of L layers: layer l's outer radius is `radius_fractions[l]` times the
    /// particle's (l < L − 1), and its refractive index `refractive_index[l][i]` at wavelength
    /// i.
    ///
    /// ```
    /// use num_complex::Complex;
    /// use scatrs::ensemble::Particle;
    ///
    /// // Black carbon (core radius 0.6 of the particle's) coated in sulfate, at one wavelength.
    /// let coated = Particle::layered_sphere(
    ///     vec![0.6],
    ///     vec![vec![Complex::new(1.95, 0.79)], vec![Complex::new(1.43, 1e-8)]],
    /// );
    /// ```
    pub fn layered_sphere(
        radius_fractions: Vec<f64>,
        refractive_index: Vec<Vec<Complex<f64>>>,
    ) -> Self {
        Particle::LayeredSphere {
            radius_fractions,
            refractive_index,
        }
    }

    /// Randomly oriented spheroids of aspect ratio `aspect_ratio` = a/c and refractive index
    /// `refractive_index[i]` at wavelength i.
    ///
    /// ```
    /// use num_complex::Complex;
    /// use scatrs::ensemble::Particle;
    ///
    /// // Oblate dust spheroids, twice as wide as tall, at one wavelength.
    /// let dust = Particle::spheroid(2.0, vec![Complex::new(1.53, 0.008)]);
    /// ```
    pub fn spheroid(aspect_ratio: f64, refractive_index: Vec<Complex<f64>>) -> Self {
        Particle::Spheroid {
            aspect_ratio,
            refractive_index,
        }
    }
}

/// One component of an external mixture: particles of one size distribution, in relative
/// number concentration `weight`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Component {
    /// The size distribution.
    pub distribution: SizeDistribution,
    /// The particles.
    pub particle: Particle,
    /// The relative number concentration, ≥ 0.
    pub weight: f64,
}

impl Component {
    /// A component of spheres of refractive index `refractive_index[i]` at wavelength i, of
    /// weight `weight`.
    pub fn new(
        distribution: SizeDistribution,
        refractive_index: Vec<Complex<f64>>,
        weight: f64,
    ) -> Self {
        Self::with_particle(distribution, Particle::sphere(refractive_index), weight)
    }

    /// A component of `particle`s, of weight `weight`.
    ///
    /// ```
    /// use num_complex::Complex;
    /// use scatrs::ensemble::{Component, Particle, SizeDistribution};
    ///
    /// let size = SizeDistribution::LogNormal { median_radius: 0.1, width: 1.5 };
    /// let m = vec![Complex::new(1.45, 1e-3)];
    /// let c = Component::with_particle(size.clone(), Particle::sphere(m.clone()), 2.0);
    /// assert_eq!(c, Component::new(size, m, 2.0));
    /// ```
    pub fn with_particle(distribution: SizeDistribution, particle: Particle, weight: f64) -> Self {
        Self {
            distribution,
            particle,
            weight,
        }
    }
}

/// The orders of the Greek coefficients to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GreekOrder {
    /// Every order, up to 2N for the largest number of terms N at any size node.
    All,
    /// Orders 0 to L (exactly 0 above 2N).
    Order(usize),
}

/// What a derivative is taken with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ParamKind {
    /// A parameter of the size distribution, by its name in
    /// [`SizeDistribution::parameter_names`].
    Distribution(&'static str),
    /// The real part of the component's refractive index (at each wavelength).
    RefrReal,
    /// The imaginary part of the component's refractive index (at each wavelength).
    RefrImag,
    /// The real part of layer l's refractive index (at each wavelength) of a layered sphere.
    LayerRefrReal(usize),
    /// The imaginary part of layer l's refractive index (at each wavelength) of a layered
    /// sphere.
    LayerRefrImag(usize),
    /// The radius fraction of layer l's outer surface of a layered sphere (l < L − 1), at a
    /// fixed outer radius.
    RadiusFraction(usize),
    /// The aspect ratio a/c of a spheroid, at a fixed equal-volume radius.
    AspectRatio,
    /// The weight of a mixture component (`mode` = `None`) or of a mode of a multimodal
    /// distribution.
    Weight,
}

/// A parameter the outputs can be differentiated with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct EnsembleParam {
    /// The mixture component (0 for [`compute`]).
    pub component: usize,
    /// The mode of a multimodal distribution, if the parameter belongs to one.
    pub mode: Option<usize>,
    /// The parameter.
    pub kind: ParamKind,
}

impl EnsembleParam {
    /// A parameter of `component` (and `mode`).
    pub fn new(component: usize, mode: Option<usize>, kind: ParamKind) -> Self {
        Self {
            component,
            mode,
            kind,
        }
    }

    /// Its name: `median_radius`, `m_real`, `m_imag`, `layer{l}.m_real`, `layer{l}.m_imag`,
    /// `layer{l}.radius_fraction`, `aspect_ratio`, `weight`, …, prefixed by `mode{j}.` for a
    /// mode and, with `mixture`, by `c{i}.` for the component.
    pub fn name(&self, mixture: bool) -> String {
        let base = match self.kind {
            ParamKind::Distribution(name) => name.to_owned(),
            ParamKind::RefrReal => "m_real".to_owned(),
            ParamKind::RefrImag => "m_imag".to_owned(),
            ParamKind::LayerRefrReal(l) => format!("layer{l}.m_real"),
            ParamKind::LayerRefrImag(l) => format!("layer{l}.m_imag"),
            ParamKind::RadiusFraction(l) => format!("layer{l}.radius_fraction"),
            ParamKind::AspectRatio => "aspect_ratio".to_owned(),
            ParamKind::Weight => "weight".to_owned(),
        };
        let mut out = String::new();
        if mixture {
            out += &format!("c{}.", self.component);
        }
        if let Some(j) = self.mode {
            out += &format!("mode{j}.");
        }
        out + &base
    }
}

/// The differentiable parameters of `distribution` as component `index`, with the component
/// weight if `weight`.
fn component_parameters(
    index: usize,
    distribution: &SizeDistribution,
    particle: &Particle,
    weight: bool,
) -> Vec<EnsembleParam> {
    let mut out = Vec::new();
    if let SizeDistribution::MultiModal { modes } = distribution {
        for (j, (_, mode)) in modes.iter().enumerate() {
            for name in mode.parameter_names() {
                out.push(EnsembleParam::new(
                    index,
                    Some(j),
                    ParamKind::Distribution(name),
                ));
            }
            out.push(EnsembleParam::new(index, Some(j), ParamKind::Weight));
        }
    } else {
        for name in distribution.parameter_names() {
            out.push(EnsembleParam::new(
                index,
                None,
                ParamKind::Distribution(name),
            ));
        }
    }
    match particle {
        Particle::Sphere { .. } => {
            out.push(EnsembleParam::new(index, None, ParamKind::RefrReal));
            out.push(EnsembleParam::new(index, None, ParamKind::RefrImag));
        }
        Particle::Spheroid { .. } => {
            out.push(EnsembleParam::new(index, None, ParamKind::RefrReal));
            out.push(EnsembleParam::new(index, None, ParamKind::RefrImag));
            out.push(EnsembleParam::new(index, None, ParamKind::AspectRatio));
        }
        Particle::LayeredSphere {
            radius_fractions,
            refractive_index,
        } => {
            for l in 0..refractive_index.len() {
                out.push(EnsembleParam::new(index, None, ParamKind::LayerRefrReal(l)));
                out.push(EnsembleParam::new(index, None, ParamKind::LayerRefrImag(l)));
            }
            for l in 0..radius_fractions.len() {
                out.push(EnsembleParam::new(
                    index,
                    None,
                    ParamKind::RadiusFraction(l),
                ));
            }
        }
    }
    if weight {
        out.push(EnsembleParam::new(index, None, ParamKind::Weight));
    }
    out
}

/// Every parameter of a single distribution of spheres, for [`compute`].
pub fn parameters(distribution: &SizeDistribution) -> Vec<EnsembleParam> {
    parameters_with_particle(distribution, &Particle::sphere(Vec::new()))
}

/// Every parameter of a single distribution of `particle`s, for [`compute_with_particle`]: the
/// distribution's, then the particle's (`m_real`, `m_imag` for a sphere; for a layered sphere
/// `layer{l}.m_real` and `layer{l}.m_imag` of each layer, then `layer{l}.radius_fraction` of
/// each but the last; `m_real`, `m_imag` and `aspect_ratio` for a spheroid).
pub fn parameters_with_particle(
    distribution: &SizeDistribution,
    particle: &Particle,
) -> Vec<EnsembleParam> {
    component_parameters(0, distribution, particle, false)
}

/// Every parameter of a mixture, for [`compute_mixture`].
pub fn mixture_parameters(components: &[Component]) -> Vec<EnsembleParam> {
    components
        .iter()
        .enumerate()
        .flat_map(|(i, c)| component_parameters(i, &c.distribution, &c.particle, true))
        .collect()
}

/// Options for [`compute`] and [`compute_mixture`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EnsembleOptions {
    /// The relative tolerance of the size integration, in [1e-13, 0.1]; default 1e-6.
    pub rtol: f64,
    /// The most panels per component (or mode) and wavelength; default 4096. Reaching it sets
    /// [`EnsembleResult::converged`] to false.
    pub max_panels: usize,
    /// The Greek coefficients to compute, if any.
    pub greek: Option<GreekOrder>,
    /// Cosines of the scattering angle at which to compute the phase matrix.
    pub mu: Vec<f64>,
    /// The form of the phase matrix and the Greek coefficients.
    pub normalization: Normalization,
    /// The parameters to differentiate with respect to.
    pub derivatives: Vec<EnsembleParam>,
    /// Compute at sizes outside the validated domain of the Mie solution instead of
    /// returning [`Error::OutsideValidatedDomain`].
    pub allow_unvalidated: bool,
    /// Also refine the size integration on the Greek coefficients up to this order.
    pub refine_order: Option<usize>,
    /// Also refine on ⟨C_back⟩ (to rtol of the larger of ⟨C_back⟩ and ⟨C_sca⟩), which makes
    /// its error estimate (and the lidar ratio's) a calibrated bound. Off by default: the
    /// backscattering of weakly absorbing spheres is a sum of narrow glory ripples that can
    /// take far more panels than the cross sections; without it the estimate is indicative.
    pub refine_backscatter: bool,
}

impl Default for EnsembleOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            max_panels: 4096,
            greek: None,
            mu: Vec::new(),
            normalization: Normalization::Normalized,
            derivatives: Vec::new(),
            allow_unvalidated: false,
            refine_order: None,
            refine_backscatter: false,
        }
    }
}

impl EnsembleOptions {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// With relative tolerance `rtol`.
    #[must_use]
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// With at most `max_panels` panels per component and wavelength.
    #[must_use]
    pub fn with_max_panels(mut self, max_panels: usize) -> Self {
        self.max_panels = max_panels;
        self
    }

    /// With the Greek coefficients of `order`.
    #[must_use]
    pub fn with_greek(mut self, order: GreekOrder) -> Self {
        self.greek = Some(order);
        self
    }

    /// With the phase matrix at the cosines `mu`.
    #[must_use]
    pub fn with_mu(mut self, mu: Vec<f64>) -> Self {
        self.mu = mu;
        self
    }

    /// With the phase matrix and Greek coefficients in `normalization`.
    #[must_use]
    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// With the derivatives with respect to `params`.
    #[must_use]
    pub fn with_derivatives(mut self, params: Vec<EnsembleParam>) -> Self {
        self.derivatives = params;
        self
    }

    /// Allowing (or not) sizes outside the validated domain.
    #[must_use]
    pub fn with_allow_unvalidated(mut self, allow: bool) -> Self {
        self.allow_unvalidated = allow;
        self
    }

    /// Also refining on the Greek coefficients up to `order`.
    #[must_use]
    pub fn with_refine_order(mut self, order: usize) -> Self {
        self.refine_order = Some(order);
        self
    }

    /// Also refining (or not) on ⟨C_back⟩.
    #[must_use]
    pub fn with_refine_backscatter(mut self, refine: bool) -> Self {
        self.refine_backscatter = refine;
        self
    }
}

/// The phase-matrix elements at each requested cosine.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PhaseValues {
    /// F₁₁.
    pub f11: Vec<f64>,
    /// F₁₂.
    pub f12: Vec<f64>,
    /// F₂₂ (= F₁₁ for spheres).
    pub f22: Vec<f64>,
    /// F₃₃.
    pub f33: Vec<f64>,
    /// F₃₄.
    pub f34: Vec<f64>,
    /// F₄₄ (= F₃₃ for spheres).
    pub f44: Vec<f64>,
}

/// The outputs of an ensemble at one wavelength (or their error estimates, or their
/// derivatives with respect to one parameter). Cross sections are per particle, averaged
/// over the number distribution, in the square of the length unit of the radii.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Outputs {
    /// ⟨C_ext⟩.
    pub c_ext: f64,
    /// ⟨C_sca⟩.
    pub c_sca: f64,
    /// ⟨C_abs⟩.
    pub c_abs: f64,
    /// ⟨C_back⟩, Bohren & Huffman's backscattering cross section (4π dσ/dΩ at 180°).
    pub c_back: f64,
    /// The single-scattering albedo ⟨C_sca⟩/⟨C_ext⟩.
    pub ssa: f64,
    /// The asymmetry parameter ⟨C_sca g⟩/⟨C_sca⟩.
    pub g: f64,
    /// The lidar ratio 4π⟨C_ext⟩/⟨C_back⟩.
    pub lidar_ratio: f64,
    /// The linear depolarization ratio at 180°, (F₁₁ − F₂₂)/(F₁₁ + F₂₂): exactly 0 for spheres.
    pub depolarization: f64,
    /// The effective radius ⟨r³⟩/⟨r²⟩.
    pub effective_radius: f64,
    /// The mean geometric cross section ⟨πr²⟩.
    pub mean_area: f64,
    /// The mean volume ⟨4πr³/3⟩.
    pub mean_volume: f64,
    /// The Greek coefficients, if requested: normalized (α₁⁰ = 1) or unnormalized
    /// (α₁⁰ = ⟨C_sca⟩), as in [`EnsembleOptions::normalization`].
    pub greek: Option<GreekCoefficients<f64>>,
    /// The phase matrix at [`EnsembleOptions::mu`], if requested: normalized (½∫F₁₁ dμ = 1)
    /// or unnormalized (⟨C_sca F⟩).
    pub phase: Option<PhaseValues>,
}

/// The result at one wavelength.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EnsembleResult {
    /// The outputs.
    pub values: Outputs,
    /// Their error estimates (non-negative; see the theory chapter).
    pub errors: Outputs,
    /// The derivatives with respect to each of [`params`](Self::params).
    pub derivatives: Vec<Outputs>,
    /// The error estimates of the derivatives.
    pub derivative_errors: Vec<Outputs>,
    /// The parameters of `derivatives`, as requested.
    pub params: Vec<EnsembleParam>,
    /// Whether every refined quantity met `rtol` within `max_panels` panels.
    pub converged: bool,
    /// The number of panels, summed over the components (and modes).
    pub panels: usize,
    /// The integration range [r_lo, r_hi] of each component (and mode), in order (a single
    /// radius r gives (r, r)).
    pub ranges: Vec<(f64, f64)>,
}

/// The highest [`EnsembleOptions::refine_order`]: pass 1 integrates 6·(L + 1) Greek
/// coefficients at every size node, for every requested parameter.
pub const MAX_REFINE_ORDER: usize = 1024;

fn invalid(msg: String) -> Error {
    Error::InvalidInput(msg)
}

/// Averages over `distribution` of spheres of refractive index `refractive_index[i]` at
/// `wavelengths[i]` (in the unit of the radii).
///
/// # Errors
///
/// [`Error::InvalidInput`] for invalid distribution parameters, wavelengths, refractive
/// indices or options, or a derivative parameter the distribution does not have;
/// [`Error::OutsideValidatedDomain`] if a size node of the integration is outside the Mie
/// solution's validated domain, unless [`EnsembleOptions::allow_unvalidated`].
pub fn compute(
    distribution: &SizeDistribution,
    wavelengths: &[f64],
    refractive_index: &[Complex<f64>],
    options: &EnsembleOptions,
) -> Result<Vec<EnsembleResult>> {
    let particle = Particle::sphere(refractive_index.to_vec());
    compute_with_particle(distribution, wavelengths, &particle, options)
}

/// Averages over `distribution` of `particle`s at `wavelengths` (in the unit of the radii), as
/// [`compute`] does for spheres. For a layered sphere the distribution gives the outer radius.
///
/// ```
/// use num_complex::Complex;
/// use scatrs::ensemble::{
///     EnsembleOptions, Particle, SizeDistribution, compute_with_particle,
///     parameters_with_particle,
/// };
///
/// let size = SizeDistribution::LogNormal { median_radius: 0.1, width: 1.5 };
/// let coated = Particle::layered_sphere(
///     vec![0.6],
///     vec![vec![Complex::new(1.95, 0.79)], vec![Complex::new(1.43, 1e-8)]],
/// );
/// // ∂/∂(core fraction), the last parameter.
/// let fraction = *parameters_with_particle(&size, &coated).last().unwrap();
/// let options = EnsembleOptions::new().with_derivatives(vec![fraction]);
/// let r = &compute_with_particle(&size, &[0.55], &coated, &options)?[0];
/// assert!(r.derivatives[0].c_abs > 0.0); // a larger core absorbs more
/// # Ok::<(), scatrs::Error>(())
/// ```
///
/// # Errors
///
/// As [`compute`], and [`Error::InvalidInput`] for a layered sphere without a layer, with
/// more than [`crate::layered::MAX_LAYERS`], with fractions that are not nondecreasing in
/// (0, 1] or not one fewer than the layers, or with a layer's refractive indices not one per
/// wavelength. With derivatives along the particle's parameters, a layered sphere's size nodes
/// must be within its derivatives' validated domain
/// ([`crate::layered::ValidatedDomain::derivative_x_max`]) unless `allow_unvalidated`.
pub fn compute_with_particle(
    distribution: &SizeDistribution,
    wavelengths: &[f64],
    particle: &Particle,
    options: &EnsembleOptions,
) -> Result<Vec<EnsembleResult>> {
    let component = Component::with_particle(distribution.clone(), particle.clone(), 1.0);
    let allowed = parameters_with_particle(distribution, particle);
    validate(
        core::slice::from_ref(&component),
        wavelengths,
        options,
        &allowed,
    )?;
    engine::run(&[component], wavelengths, options)
}

/// Averages over the external mixture `components`: cross sections and moments weighted by
/// wᵢ/Σw, phase matrices by wᵢ C_sca,ᵢ.
///
/// # Errors
///
/// As [`compute`], and [`Error::InvalidInput`] for an empty mixture, a negative weight or
/// weights that are all 0.
pub fn compute_mixture(
    components: &[Component],
    wavelengths: &[f64],
    options: &EnsembleOptions,
) -> Result<Vec<EnsembleResult>> {
    if components.is_empty() {
        return Err(invalid("a mixture needs a component".into()));
    }
    let allowed = mixture_parameters(components);
    validate(components, wavelengths, options, &allowed)?;
    engine::run(components, wavelengths, options)
}

/// Averages over each of many `distributions` of one material, sharing the size nodes: one
/// set of panels per wavelength over the union of their ranges, refined until every
/// distribution meets its own targets, so each Mie solution and phase-matrix table serves
/// every distribution. For grids of distributions (an optics database) this costs about as
/// much as the broadest distribution alone. Results by distribution, then wavelength; each
/// has its own error estimates and `converged` flag (`panels` counts the shared panels). A
/// distribution in a batch is at least as refined as on its own, so its values differ from
/// [`compute`]'s within the error estimates.
///
/// # Errors
///
/// As [`compute`], and [`Error::InvalidInput`] for no distributions or any requested
/// derivative (Jacobians come from [`compute`]).
pub fn compute_batch(
    distributions: &[SizeDistribution],
    wavelengths: &[f64],
    refractive_index: &[Complex<f64>],
    options: &EnsembleOptions,
) -> Result<Vec<Vec<EnsembleResult>>> {
    let particle = Particle::sphere(refractive_index.to_vec());
    compute_batch_with_particle(distributions, wavelengths, &particle, options)
}

/// As [`compute_batch`], for `particle`s.
///
/// # Errors
///
/// As [`compute_batch`] and [`compute_with_particle`].
pub fn compute_batch_with_particle(
    distributions: &[SizeDistribution],
    wavelengths: &[f64],
    particle: &Particle,
    options: &EnsembleOptions,
) -> Result<Vec<Vec<EnsembleResult>>> {
    if distributions.is_empty() {
        return Err(invalid("a batch needs a distribution".into()));
    }
    if !options.derivatives.is_empty() {
        return Err(invalid(
            "a batch computes values only: request derivatives from compute".into(),
        ));
    }
    particle.validate().map_err(invalid)?;
    if particle.wavelengths() != wavelengths.len() {
        return Err(invalid(format!(
            "{} refractive indices for {} wavelengths",
            particle.wavelengths(),
            wavelengths.len()
        )));
    }
    for (i, d) in distributions.iter().enumerate() {
        d.validate()
            .map_err(|e| invalid(format!("distribution {i}: {e}")))?;
    }
    let components: Vec<Component> = distributions
        .iter()
        .map(|d| Component::with_particle(d.clone(), particle.clone(), 1.0))
        .collect();
    validate(&components, wavelengths, options, &[])?;
    engine::run_batch(&components, wavelengths, options)
}

/// Checks the inputs common to [`compute`] and [`compute_mixture`].
fn validate(
    components: &[Component],
    wavelengths: &[f64],
    options: &EnsembleOptions,
    allowed: &[EnsembleParam],
) -> Result<()> {
    if !(1e-13..=0.1).contains(&options.rtol) {
        return Err(invalid(format!(
            "rtol must be in [1e-13, 0.1], got {}",
            options.rtol
        )));
    }
    if options.max_panels == 0 {
        return Err(invalid("max_panels must be at least 1".into()));
    }
    let greek_order = match options.greek {
        Some(GreekOrder::Order(l)) => Some(l),
        _ => None,
    };
    if let Some(l) = greek_order.filter(|&l| l > MAX_GREEK_ORDER) {
        return Err(invalid(format!(
            "the Greek order must be at most {MAX_GREEK_ORDER}, got {l}"
        )));
    }
    if let Some(l) = options.refine_order.filter(|&l| l > MAX_REFINE_ORDER) {
        return Err(invalid(format!(
            "refine_order must be at most {MAX_REFINE_ORDER}, got {l}"
        )));
    }
    if let Some((j, v)) = options
        .mu
        .iter()
        .enumerate()
        .find(|&(_, v)| !(-1.0..=1.0).contains(v))
    {
        return Err(invalid(format!("cosine {j} must be in [-1, 1], got {v}")));
    }
    if let Some((i, w)) = wavelengths
        .iter()
        .enumerate()
        .find(|&(_, w)| !(w.is_finite() && *w > 0.0))
    {
        return Err(invalid(format!(
            "wavelength {i} must be finite and > 0, got {w}"
        )));
    }
    let mut total = 0.0;
    for (i, c) in components.iter().enumerate() {
        c.distribution
            .validate()
            .map_err(|e| invalid(format!("component {i}: {e}")))?;
        if let SizeDistribution::MultiModal { modes } = &c.distribution {
            if modes
                .iter()
                .any(|(_, m)| matches!(m, SizeDistribution::MultiModal { .. }))
            {
                return Err(invalid(format!(
                    "component {i}: the modes of a multimodal distribution cannot be multimodal"
                )));
            }
        }
        c.particle
            .validate()
            .map_err(|e| invalid(format!("component {i}: {e}")))?;
        if c.particle.wavelengths() != wavelengths.len() {
            return Err(invalid(format!(
                "component {i} has {} refractive indices for {} wavelengths",
                c.particle.wavelengths(),
                wavelengths.len()
            )));
        }
        if !(c.weight.is_finite() && c.weight >= 0.0) {
            return Err(invalid(format!(
                "the weight of component {i} must be finite and >= 0, got {}",
                c.weight
            )));
        }
        if let SizeDistribution::MultiModal { modes } = &c.distribution {
            let sum: f64 = modes.iter().map(|(w, _)| w).sum();
            if !sum.is_finite() {
                return Err(invalid(format!(
                    "component {i}: the mode weights must have a finite sum"
                )));
            }
        }
        total += c.weight;
    }
    if !(total.is_finite() && total > 0.0) {
        return Err(invalid(format!(
            "the mixture weights must have a finite, positive sum, got {total}"
        )));
    }
    let mixture = components.len() > 1
        || allowed
            .iter()
            .any(|p| p.kind == ParamKind::Weight && p.mode.is_none());
    for (k, p) in options.derivatives.iter().enumerate() {
        if options.derivatives[..k].contains(p) {
            return Err(invalid(format!(
                "parameter {} requested twice",
                p.name(mixture)
            )));
        }
        if !allowed.contains(p) {
            return Err(invalid(format!(
                "no parameter {} here; the parameters are {:?}",
                p.name(mixture),
                allowed.iter().map(|q| q.name(mixture)).collect::<Vec<_>>()
            )));
        }
    }
    Ok(())
}
