//! Scattering by randomly oriented spheroids, by the T-matrix method.
//!
//! [`TMatrix`] is a reusable workspace: [`TMatrix::compute`] solves one spheroid, converging the
//! T-matrix to a relative tolerance, and returns a [`TMatrixSolution`] with the
//! orientation-averaged efficiencies, Greek coefficients and phase matrix (six independent
//! elements: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄), the T-matrix itself, and how it was obtained.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::tmatrix::{Shape, TMatrix, TMatrixOptions};
//!
//! // An oblate dust spheroid, twice as wide as tall, of equal-volume size parameter 5.
//! let shape = Shape::Spheroid { aspect_ratio: 2.0 };
//! let mut tm = TMatrix::new();
//! let s = tm.compute(shape, 5.0, Complex::new(1.53, 0.008), &TMatrixOptions::new())?;
//! let q = s.efficiencies();
//! assert!(q.q_abs > 0.0 && q.q_ext > q.q_sca);
//! assert!(s.error_estimate() < 1e-8);
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! # Conventions
//!
//! A spheroid has semi-axes a, a, c, with c along its symmetry axis, and aspect ratio
//! ε = a/c: oblate for ε > 1, prolate for ε < 1, a sphere for ε = 1. Its size is the size
//! parameter x = 2πr_v/λ of the sphere of equal volume (r_v³ = a²c), which also normalizes the
//! efficiencies (Q = C/(πr_v²)) and the unnormalized phase matrix and Greek coefficients. The
//! refractive index m is relative to the medium. The phase matrix is that of
//! [`crate::mie`] (Bohren & Huffman's), averaged uniformly over orientations, and its Greek
//! coefficients are those of [`crate::mie::GreekCoefficients`] with F₂₂ and F₄₄ independent.
//!
//! # Method
//!
//! See the theory chapter of the documentation, "Spheroids". The extended boundary condition
//! method gives the T-matrix of each azimuthal mode from surface integrals. Their outgoing-wave
//! parts are summed in a form free of the electrostatic cancellation that limits the classical
//! method. Each mode is solved in `f64` with an estimate of its error, and the modes that need
//! it are refined in mixed precision (double-double residuals, the high outgoing orders
//! integrated in double-double). The orientation average follows from the rank decomposition of
//! T (Mishchenko 1991), and the Greek coefficients from its exact projection. The number of
//! orders N and of quadrature nodes n_θ grow until the cross sections (and the Greek
//! coefficients, if asked for) settle to `rtol`.
//!
//! # Accuracy
//!
//! See [`VALIDATED_DOMAIN`].

mod domain;
mod driver;
pub(crate) mod functions;
pub(crate) mod kernel;
pub(crate) mod random;
pub(crate) mod regular;
pub(crate) mod solve;

use std::sync::OnceLock;

use num_complex::Complex;

pub use domain::{VALIDATED_DOMAIN, ValidatedDomain};
pub use driver::{Precision, PrecisionPath};

use crate::scalar::Real;
use kernel::{D4, TMatrixModes};

use crate::mie::{Efficiencies, GreekCoefficients, MAX_GREEK_ORDER, Normalization};
use crate::special::wigner_d;
use crate::{Error, Result};

/// The shape of a particle.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Shape {
    /// A spheroid of aspect ratio ε = a/c (horizontal over polar semi-axis): oblate for ε > 1,
    /// prolate for ε < 1.
    Spheroid {
        /// ε = a/c.
        aspect_ratio: f64,
    },
}

/// Options for [`TMatrix::compute`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct TMatrixOptions {
    /// The relative tolerance of the convergence in N and n_θ (default 10⁻⁸).
    pub rtol: f64,
    /// Which precision paths may be used (default [`Precision::Auto`]).
    pub precision: Precision,
    /// A fixed number of orders N instead of the convergence loop's.
    pub n_terms: Option<usize>,
    /// A fixed number of quadrature nodes in 0 < θ < π/2 instead of the loop's.
    pub n_theta: Option<usize>,
    /// The cap on N (default: twice the starting estimate, plus 20).
    pub n_terms_max: Option<usize>,
    /// The cap on n_θ (default: 4 times N's starting estimate (1 + |ln ε|), plus 40).
    pub n_theta_max: Option<usize>,
    /// The Greek order that must settle too (default: none, only the cross sections).
    pub greek_order: Option<usize>,
    /// At a cap, return the last step with its estimate instead of [`Error::NotConverged`].
    pub allow_unconverged: bool,
    /// Compute even outside [`VALIDATED_DOMAIN`] instead of returning
    /// [`Error::OutsideValidatedDomain`].
    pub allow_unvalidated: bool,
}

impl Default for TMatrixOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            precision: Precision::Auto,
            n_terms: None,
            n_theta: None,
            n_terms_max: None,
            n_theta_max: None,
            greek_order: None,
            allow_unconverged: false,
            allow_unvalidated: false,
        }
    }
}

impl TMatrixOptions {
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

    /// With the precision paths `precision`.
    #[must_use]
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// With N fixed at `n`.
    #[must_use]
    pub fn with_n_terms(mut self, n: usize) -> Self {
        self.n_terms = Some(n);
        self
    }

    /// With n_θ fixed at `n_theta`.
    #[must_use]
    pub fn with_n_theta(mut self, n_theta: usize) -> Self {
        self.n_theta = Some(n_theta);
        self
    }

    /// Requiring the Greek coefficients up to `order` to settle.
    #[must_use]
    pub fn with_greek_order(mut self, order: usize) -> Self {
        self.greek_order = Some(order);
        self
    }

    /// Allowing (or not) an unconverged result at a cap.
    #[must_use]
    pub fn with_allow_unconverged(mut self, allow: bool) -> Self {
        self.allow_unconverged = allow;
        self
    }

    /// Allowing (or not) inputs outside the validated domain.
    #[must_use]
    pub fn with_allow_unvalidated(mut self, allow: bool) -> Self {
        self.allow_unvalidated = allow;
        self
    }
}

/// The largest N [`TMatrix::compute`] accepts, 500.
pub const MAX_TMATRIX_ORDER: usize = 500;

/// A workspace for T-matrix calculations, one per thread.
#[derive(Debug, Default)]
pub struct TMatrix {
    state: Option<State>,
}

/// T with its derivatives, and their Greek coefficients on first use.
type WithDerivatives = (TMatrixModes<D4>, OnceLock<Result<GreekCoefficients<D4>>>);

#[derive(Debug)]
struct State {
    x: f64,
    eps: f64,
    m: Complex<f64>,
    rtol: f64,
    converged: driver::Converged,
    /// The Greek coefficients up to 2N (unnormalized, 1/k²), computed on first use.
    greek: OnceLock<Result<GreekCoefficients<f64>>>,
    /// T with its derivatives, if asked for, and their Greek coefficients on first use.
    derivatives: Option<WithDerivatives>,
}

impl TMatrix {
    /// An empty workspace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves the particle of `shape`, equal-volume size parameter `x` and refractive index `m`.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] if ε or x is not finite and positive, if m is not finite or
    ///   has Re m ≤ 0 or Im m < 0, if `rtol` is not in (0, 1), or if N would exceed
    ///   [`MAX_TMATRIX_ORDER`].
    /// * [`Error::OutsideValidatedDomain`] outside [`VALIDATED_DOMAIN`], unless
    ///   `allow_unvalidated` is set.
    /// * [`Error::NotConverged`] if N or n_θ reaches its cap first, unless `allow_unconverged`
    ///   is set.
    pub fn compute(
        &mut self,
        shape: Shape,
        x: f64,
        m: Complex<f64>,
        opts: &TMatrixOptions,
    ) -> Result<TMatrixSolution<'_>> {
        self.run(shape, x, m, opts, false)
    }

    /// As [`compute`](Self::compute), also computing the derivatives of every output with
    /// respect to x, the aspect ratio, Re m and Im m, at the N and n_θ the values converged at:
    /// the elements in dual numbers (their values are those of `f64`), and
    /// dT = −(dP + T dQ) Q⁻¹ with the factors of the solve, refined as T is.
    ///
    /// # Errors
    ///
    /// As [`compute`](Self::compute).
    pub fn compute_with_derivatives(
        &mut self,
        shape: Shape,
        x: f64,
        m: Complex<f64>,
        opts: &TMatrixOptions,
    ) -> Result<TMatrixSolution<'_>> {
        self.run(shape, x, m, opts, true)
    }

    /// The last solution, if any.
    pub fn solution(&self) -> Option<TMatrixSolution<'_>> {
        self.state.as_ref().map(|s| TMatrixSolution { s })
    }

    fn run(
        &mut self,
        shape: Shape,
        x: f64,
        m: Complex<f64>,
        opts: &TMatrixOptions,
        with_derivatives: bool,
    ) -> Result<TMatrixSolution<'_>> {
        let Shape::Spheroid { aspect_ratio: eps } = shape;
        let invalid = |s: String| Err(Error::InvalidInput(s));
        if !(eps.is_finite() && eps > 0.0) {
            return invalid(format!(
                "the aspect ratio must be finite and positive, got {eps}"
            ));
        }
        if !(x.is_finite() && x > 0.0) {
            return invalid(format!("x must be finite and positive, got {x}"));
        }
        if !(m.re.is_finite() && m.im.is_finite() && m.re > 0.0 && m.im >= 0.0) {
            return invalid(format!(
                "m must be finite with Re m > 0 and Im m ≥ 0, got {m}"
            ));
        }
        if !(opts.rtol > 0.0 && opts.rtol < 1.0) {
            return invalid(format!("rtol must be in (0, 1), got {}", opts.rtol));
        }
        if !opts.allow_unvalidated {
            if !VALIDATED_DOMAIN.contains(eps, x, m) {
                return Err(Error::OutsideValidatedDomain(format!(
                    "aspect ratio {eps}, x = {x}, m = {m}"
                )));
            }
            if opts.rtol < VALIDATED_DOMAIN.rtol_min {
                return Err(Error::OutsideValidatedDomain(format!(
                    "rtol {:e} is below {:e}",
                    opts.rtol, VALIDATED_DOMAIN.rtol_min
                )));
            }
        }
        let start = driver::start_n(x, eps);
        let n_cap = opts
            .n_terms_max
            .unwrap_or(2 * start + 20)
            .min(MAX_TMATRIX_ORDER);
        let theta_cap = opts
            .n_theta_max
            .unwrap_or(4 * driver::theta_of(start, eps) + 40);
        if let Some(n) = opts.n_terms {
            if n == 0 || n > MAX_TMATRIX_ORDER {
                return invalid(format!("N must be in 1..={MAX_TMATRIX_ORDER}, got {n}"));
            }
        }
        if opts.n_theta == Some(0) {
            return invalid("n_theta must be at least 1".into());
        }
        if let Some(order) = opts.greek_order {
            if order > MAX_GREEK_ORDER {
                return invalid(format!("the Greek order must be at most {MAX_GREEK_ORDER}"));
            }
        }
        let converged = driver::converge(
            x,
            eps,
            m,
            &driver::Loop {
                rtol: opts.rtol,
                precision: opts.precision,
                n_fixed: opts.n_terms,
                n_cap,
                n_theta_fixed: opts.n_theta,
                n_theta_cap: theta_cap,
                greek_order: opts.greek_order,
                allow_unconverged: opts.allow_unconverged,
            },
        )?;
        let mut converged = converged;
        let derivatives = if with_derivatives {
            let solved = &mut converged.solved;
            let (n, nth) = (solved.t.n, solved.n_theta);
            let d = driver::solve_derivatives(x, eps, m, n, nth, opts.precision, opts.rtol)?;
            // The values are the dual T's, so that they match the derivatives.
            solved.t = d.0.values();
            solved.paths = d.1;
            solved.error = d.2;
            Some((d.0, OnceLock::new()))
        } else {
            None
        };
        self.state = Some(State {
            x,
            eps,
            m,
            rtol: opts.rtol,
            converged,
            greek: OnceLock::new(),
            derivatives,
        });
        Ok(TMatrixSolution {
            s: self.state.as_ref().expect("just set"),
        })
    }
}

/// The T-matrix blocks of one azimuthal mode m ≥ 0, for orders n, k = max(m, 1), …, N: row-major
/// K×K matrices, T¹¹ (M from M), T¹² (M from N), T²¹ (N from M), T²² (N from N), in the
/// convention of [Mishchenko, Travis & Lacis 2002, §5.2] without the (−1)^m of their vector
/// spherical wave functions. Mode −m has T¹¹ and T²² equal, and T¹² and T²¹ negated.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ModeBlocks {
    /// The orders n of the rows and columns.
    pub orders: Vec<usize>,
    /// T¹¹.
    pub t11: Vec<Complex<f64>>,
    /// T¹².
    pub t12: Vec<Complex<f64>>,
    /// T²¹.
    pub t21: Vec<Complex<f64>>,
    /// T²².
    pub t22: Vec<Complex<f64>>,
}

/// Output buffers for the six independent elements of the orientation-averaged phase matrix,
/// one value per cosine.
#[derive(Debug)]
pub struct PhaseBuffers<'b> {
    /// F₁₁.
    pub f11: &'b mut [f64],
    /// F₁₂.
    pub f12: &'b mut [f64],
    /// F₂₂.
    pub f22: &'b mut [f64],
    /// F₃₃.
    pub f33: &'b mut [f64],
    /// F₃₄.
    pub f34: &'b mut [f64],
    /// F₄₄.
    pub f44: &'b mut [f64],
}

/// A solved spheroid: borrowed from its [`TMatrix`] until the next computation.
#[derive(Clone, Copy, Debug)]
pub struct TMatrixSolution<'a> {
    s: &'a State,
}

/// The derivatives of an output with respect to the spheroid's parameters.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Derivatives<V> {
    /// ∂/∂x (equal-volume size parameter).
    pub x: V,
    /// ∂/∂ε (aspect ratio).
    pub aspect_ratio: V,
    /// ∂/∂(Re m).
    pub m_re: V,
    /// ∂/∂(Im m).
    pub m_im: V,
}

impl<V> Derivatives<V> {
    /// The derivatives with respect to x, ε, Re m and Im m (output buffers, for example).
    pub fn new(x: V, aspect_ratio: V, m_re: V, m_im: V) -> Self {
        Self {
            x,
            aspect_ratio,
            m_re,
            m_im,
        }
    }

    fn from_fn(mut f: impl FnMut(usize) -> V) -> Self {
        Self {
            x: f(0),
            aspect_ratio: f(1),
            m_re: f(2),
            m_im: f(3),
        }
    }
}

/// The efficiencies from T, its raw Greek coefficients and x (Q = C/(πx²) in 1/k² units).
fn efficiencies_of<T: Real>(
    t: &kernel::TMatrixModes<T>,
    raw: &GreekCoefficients<T>,
    x: T,
) -> Efficiencies<T> {
    let (ext, sca) = t.cross_sections();
    let area = T::PI() * x * x;
    let a0 = raw.alpha1[0];
    let g = if a0 > T::zero() {
        raw.alpha1.get(1).copied().unwrap_or(T::zero()) / (T::from_f64(3.0) * a0)
    } else {
        T::zero()
    };
    // F₁₁(−1) = Σ α₁ₗ (−1)ˡ.
    let mut f11 = T::zero();
    for (l, &v) in raw.alpha1.iter().enumerate() {
        f11 = if l % 2 == 0 { f11 + v } else { f11 - v };
    }
    Efficiencies {
        q_ext: ext / area,
        q_sca: sca / area,
        q_abs: (ext - sca) / area,
        q_back: T::from_f64(4.0) * T::PI() * f11 / area,
        g,
    }
}

/// The Greek coefficients up to `order` from the raw ones, normalized or unnormalized.
fn greek_of<T: Real>(
    raw: &GreekCoefficients<T>,
    x: T,
    order: usize,
    normalization: Normalization,
) -> GreekCoefficients<T> {
    let mut g = raw.clone();
    let len = (order + 1).min(g.alpha1.len());
    for v in [
        &mut g.alpha1,
        &mut g.alpha2,
        &mut g.alpha3,
        &mut g.alpha4,
        &mut g.beta1,
        &mut g.beta2,
    ] {
        v.truncate(len);
    }
    let scale = match normalization {
        Normalization::Normalized => g.alpha1[0].recip(),
        // α₁(0) = C_sca/(4π) in 1/k²: Q_sca/α₁(0) = 4π/(πx²).
        Normalization::Unnormalized => T::from_f64(4.0) / (x * x),
    };
    g.map(|v| v * scale);
    g.pad(order + 1);
    g
}

/// F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ at cosine `mu` from Greek coefficients.
fn phase_of<T: Real>(g: &GreekCoefficients<T>, d: &[Vec<f64>; 4]) -> [T; 6] {
    let [d00, d22, d2m2, d02] = d;
    let dot = |a: &[T], b: &[f64]| {
        a.iter()
            .zip(b)
            .fold(T::zero(), |acc, (&x, &y)| acc + x * T::from_f64(y))
    };
    let mut sum = T::zero();
    let mut diff = T::zero();
    for l in 0..g.alpha2.len() {
        sum += (g.alpha2[l] + g.alpha3[l]) * T::from_f64(d22[l]);
        diff += (g.alpha2[l] - g.alpha3[l]) * T::from_f64(d2m2[l]);
    }
    let half = T::from_f64(0.5);
    [
        dot(&g.alpha1, d00),
        dot(&g.beta1, d02),
        half * (sum + diff),
        half * (sum - diff),
        dot(&g.beta2, d02),
        dot(&g.alpha4, d00),
    ]
}

fn wigner_tables(mu: f64, len: usize, d: &mut [Vec<f64>; 4]) -> Result<()> {
    for v in d.iter_mut() {
        v.resize(len, 0.0);
    }
    let [d00, d22, d2m2, d02] = d;
    wigner_d(0, 0, mu, d00)?;
    wigner_d(2, 2, mu, d22)?;
    wigner_d(2, -2, mu, d2m2)?;
    wigner_d(0, 2, mu, d02)?;
    Ok(())
}

fn check_buffers(mu: &[f64], out: &PhaseBuffers<'_>) -> Result<()> {
    let lens = [
        out.f11.len(),
        out.f12.len(),
        out.f22.len(),
        out.f33.len(),
        out.f34.len(),
        out.f44.len(),
    ];
    if lens.iter().any(|&l| l != mu.len()) {
        return Err(Error::InvalidInput(format!(
            "phase-matrix buffers must have one value per cosine ({}), got {lens:?}",
            mu.len()
        )));
    }
    Ok(())
}

fn store(out: &mut PhaseBuffers<'_>, j: usize, v: [f64; 6]) {
    out.f11[j] = v[0];
    out.f12[j] = v[1];
    out.f22[j] = v[2];
    out.f33[j] = v[3];
    out.f34[j] = v[4];
    out.f44[j] = v[5];
}

fn no_derivatives() -> Error {
    Error::InvalidInput(
        "the solution was computed without derivatives (use compute_with_derivatives)".into(),
    )
}

fn check_order(order: usize) -> Result<()> {
    if order > MAX_GREEK_ORDER {
        return Err(Error::InvalidInput(format!(
            "the Greek order must be at most {MAX_GREEK_ORDER}"
        )));
    }
    Ok(())
}

impl TMatrixSolution<'_> {
    fn solved(&self) -> &driver::Solved {
        &self.s.converged.solved
    }

    /// The number of orders N.
    pub fn n_terms(&self) -> usize {
        self.solved().t.n
    }

    /// The number of quadrature nodes in 0 < θ < π/2.
    pub fn n_theta(&self) -> usize {
        self.solved().n_theta
    }

    /// How each mode m = 0, …, N was solved.
    pub fn precision_paths(&self) -> &[PrecisionPath] {
        &self.solved().paths
    }

    /// Whether the loop converged to `rtol` (it always has, unless `allow_unconverged` let it
    /// stop at a cap).
    pub fn converged(&self) -> bool {
        self.s.converged.converged
    }

    /// The estimated relative error of the outputs: the last change of the convergence loop
    /// and the solve's estimate for the largest element of T, relative to it.
    pub fn error_estimate(&self) -> f64 {
        let solved = self.solved();
        let big = solved
            .t
            .modes
            .iter()
            .flat_map(|(ts, _)| ts.iter().flatten())
            .fold(0.0f64, |a, v| a.max(v.norm()));
        let solve = if big > 0.0 { solved.error / big } else { 0.0 };
        self.s.converged.change.max(solve)
    }

    /// The aspect ratio, size parameter, index and tolerance this solution was computed for.
    pub fn inputs(&self) -> (f64, f64, Complex<f64>, f64) {
        (self.s.eps, self.s.x, self.s.m, self.s.rtol)
    }

    /// The T-matrix blocks of mode `m` (`None` above N).
    pub fn block(&self, m: usize) -> Option<ModeBlocks> {
        let t = &self.solved().t;
        if m > t.n {
            return None;
        }
        let (ts, index) = &t.modes[m];
        let orders: Vec<usize> = (m.max(1)..=t.n).collect();
        let k = orders.len();
        let mut blocks: [Vec<Complex<f64>>; 4] =
            core::array::from_fn(|_| vec![Complex::new(0.0, 0.0); k * k]);
        let n0 = m.max(1);
        for c in 0..2 {
            let kc = index[c].len();
            for (i, &(ti, ni)) in index[c].iter().enumerate() {
                for (j, &(tj, nj)) in index[c].iter().enumerate() {
                    blocks[ti * 2 + tj][(ni - n0) * k + (nj - n0)] = ts[c][i * kc + j];
                }
            }
        }
        let [t11, t12, t21, t22] = blocks;
        Some(ModeBlocks {
            orders,
            t11,
            t12,
            t21,
            t22,
        })
    }

    /// The raw Greek coefficients up to 2N (1/k² units, α₁ at l = 0 is ⟨C_sca⟩/(4π)).
    fn greek_raw(&self) -> Result<&GreekCoefficients<f64>> {
        let t = &self.solved().t;
        self.s
            .greek
            .get_or_init(|| t.greek_raw(2 * t.n))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The same with derivatives.
    fn greek_raw_dual(&self) -> Result<&GreekCoefficients<D4>> {
        let (t, cell) = self.s.derivatives.as_ref().ok_or_else(no_derivatives)?;
        cell.get_or_init(|| t.greek_raw(2 * t.n))
            .as_ref()
            .map_err(Clone::clone)
    }

    fn x_dual(&self) -> D4 {
        D4::new(self.s.x, [1.0, 0.0, 0.0, 0.0])
    }

    /// The orientation-averaged efficiencies, per unit geometric cross section of the
    /// equal-volume sphere: Q_ext and Q_sca from T (Mishchenko 1991), Q_abs = Q_ext − Q_sca,
    /// g = α₁(1)/3, and Q_back = 4π F₁₁(180°)/(πx²) from the Greek expansion.
    pub fn efficiencies(&self) -> Efficiencies<f64> {
        match self.greek_raw() {
            Ok(raw) => efficiencies_of(&self.solved().t, raw, self.s.x),
            Err(_) => {
                let (ext, sca) = self.solved().t.cross_sections();
                let area = core::f64::consts::PI * self.s.x * self.s.x;
                Efficiencies {
                    q_ext: ext / area,
                    q_sca: sca / area,
                    q_abs: (ext - sca) / area,
                    q_back: f64::NAN,
                    g: f64::NAN,
                }
            }
        }
    }

    /// The derivatives of the efficiencies (`None` without derivatives).
    pub fn efficiencies_jacobian(&self) -> Option<Derivatives<Efficiencies<f64>>> {
        let (t, _) = self.s.derivatives.as_ref()?;
        let raw = self.greek_raw_dual().ok()?;
        let q = efficiencies_of(t, raw, self.x_dual());
        Some(Derivatives::from_fn(|p| Efficiencies {
            q_ext: q.q_ext.eps()[p],
            q_sca: q.q_sca.eps()[p],
            q_abs: q.q_abs.eps()[p],
            q_back: q.q_back.eps()[p],
            g: q.g.eps()[p],
        }))
    }

    /// The Greek coefficients up to `order` (default 2N; zero above 2N), normalized (α₁ at
    /// l = 0 is 1) or unnormalized (times Q_sca).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`].
    pub fn greek(
        &self,
        order: Option<usize>,
        normalization: Normalization,
    ) -> Result<GreekCoefficients<f64>> {
        let order = order.unwrap_or(2 * self.n_terms());
        check_order(order)?;
        let mut g = greek_of(self.greek_raw()?, self.s.x, order, normalization);
        if normalization == Normalization::Normalized {
            g.alpha1[0] = 1.0;
        }
        Ok(g)
    }

    /// The Greek coefficients and their derivatives.
    ///
    /// # Errors
    ///
    /// As [`greek`](Self::greek), and [`Error::InvalidInput`] without derivatives.
    pub fn greek_with_derivatives(
        &self,
        order: Option<usize>,
        normalization: Normalization,
    ) -> Result<(GreekCoefficients<f64>, Derivatives<GreekCoefficients<f64>>)> {
        let order = order.unwrap_or(2 * self.n_terms());
        check_order(order)?;
        let g = greek_of(self.greek_raw_dual()?, self.x_dual(), order, normalization);
        let split = |f: &dyn Fn(D4) -> f64| {
            let mut out = GreekCoefficients::zeros(order + 1);
            for (o, i) in [
                (&mut out.alpha1, &g.alpha1),
                (&mut out.alpha2, &g.alpha2),
                (&mut out.alpha3, &g.alpha3),
                (&mut out.alpha4, &g.alpha4),
                (&mut out.beta1, &g.beta1),
                (&mut out.beta2, &g.beta2),
            ] {
                for (a, &b) in o.iter_mut().zip(i) {
                    *a = f(b);
                }
            }
            out
        };
        let mut values = split(&|v| v.re());
        let mut d = Derivatives::from_fn(|p| split(&|v| v.eps()[p]));
        if normalization == Normalization::Normalized {
            values.alpha1[0] = 1.0;
            for v in [&mut d.x, &mut d.aspect_ratio, &mut d.m_re, &mut d.m_im] {
                v.alpha1[0] = 0.0;
            }
        }
        Ok((values, d))
    }

    /// The orientation-averaged phase matrix at the cosines `mu` of the scattering angle, from
    /// the Greek expansion (exact up to 2N), normalized (½∫F₁₁ dμ = 1) or unnormalized (times
    /// Q_sca).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if a buffer's length differs from `mu`'s or a cosine is outside
    /// [−1, 1].
    pub fn phase_matrix(
        &self,
        mu: &[f64],
        normalization: Normalization,
        mut out: PhaseBuffers<'_>,
    ) -> Result<()> {
        check_buffers(mu, &out)?;
        let g = self.greek(None, normalization)?;
        let mut d: [Vec<f64>; 4] = Default::default();
        for (j, &c) in mu.iter().enumerate() {
            wigner_tables(c, g.alpha1.len(), &mut d)?;
            store(&mut out, j, phase_of(&g, &d));
        }
        Ok(())
    }

    /// The phase matrix and its derivatives, into `out` and `d`.
    ///
    /// # Errors
    ///
    /// As [`phase_matrix`](Self::phase_matrix), and [`Error::InvalidInput`] without
    /// derivatives.
    pub fn phase_matrix_with_derivatives(
        &self,
        mu: &[f64],
        normalization: Normalization,
        mut out: PhaseBuffers<'_>,
        d: Derivatives<PhaseBuffers<'_>>,
    ) -> Result<()> {
        check_buffers(mu, &out)?;
        let Derivatives {
            x: mut dx,
            aspect_ratio: mut de,
            m_re: mut dr,
            m_im: mut di,
        } = d;
        for b in [&dx, &de, &dr, &di] {
            check_buffers(mu, b)?;
        }
        let n = self.n_terms();
        let g = greek_of(self.greek_raw_dual()?, self.x_dual(), 2 * n, normalization);
        let mut w: [Vec<f64>; 4] = Default::default();
        for (j, &c) in mu.iter().enumerate() {
            wigner_tables(c, g.alpha1.len(), &mut w)?;
            let v = phase_of(&g, &w);
            store(&mut out, j, v.map(|x| x.re()));
            for (p, b) in [&mut dx, &mut de, &mut dr, &mut di].into_iter().enumerate() {
                store(b, j, v.map(|x| x.eps()[p]));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mie::{Mie, MieOptions, PhaseBuffers as MiePhase};

    fn spheroid(eps: f64) -> Shape {
        Shape::Spheroid { aspect_ratio: eps }
    }

    /// Aspect ratio 1 gives the sphere's efficiencies, Greek coefficients and phase matrix.
    #[test]
    fn a_sphere_matches_mie() {
        let (x, m) = (4.0, Complex::new(1.5, 0.01));
        let mut tm = TMatrix::new();
        let s = tm
            .compute(spheroid(1.0), x, m, &TMatrixOptions::new().with_rtol(1e-10))
            .unwrap();
        let q = s.efficiencies();
        let mut mie = Mie::<f64>::new();
        let ms = mie.compute(x, m, &MieOptions::new()).unwrap();
        let qm = ms.efficiencies();
        for (a, b, what) in [
            (q.q_ext, qm.q_ext, "q_ext"),
            (q.q_sca, qm.q_sca, "q_sca"),
            (q.q_abs, qm.q_abs, "q_abs"),
            (q.q_back, qm.q_back, "q_back"),
            (q.g, qm.g, "g"),
        ] {
            assert!((a - b).abs() <= 1e-9 * b.abs(), "{what}: {a} vs {b}");
        }
        let order = 2 * ms.n_terms();
        let g = s.greek(Some(order), Normalization::Normalized).unwrap();
        let gm = ms.greek(Some(order), Normalization::Normalized).unwrap();
        let big = gm.alpha1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for (u, v) in [
            (&g.alpha1, &gm.alpha1),
            (&g.alpha2, &gm.alpha2),
            (&g.alpha3, &gm.alpha3),
            (&g.alpha4, &gm.alpha4),
            (&g.beta1, &gm.beta1),
            (&g.beta2, &gm.beta2),
        ] {
            for (a, b) in u.iter().zip(v) {
                assert!((a - b).abs() <= 1e-9 * big, "{a} vs {b}");
            }
        }
        let mu = [-1.0, -0.3, 0.0, 0.5, 1.0];
        let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; mu.len()]);
        let [f11, f12, f22, f33, f34, f44] = &mut f;
        s.phase_matrix(
            &mu,
            Normalization::Unnormalized,
            PhaseBuffers {
                f11,
                f12,
                f22,
                f33,
                f34,
                f44,
            },
        )
        .unwrap();
        let mut fm: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; mu.len()]);
        let [m11, m12, m33, m34] = &mut fm;
        ms.phase_matrix(
            &mu,
            Normalization::Unnormalized,
            MiePhase {
                f11: m11,
                f12: m12,
                f33: m33,
                f34: m34,
            },
        )
        .unwrap();
        let big = fm[0].iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for (a, b) in [
            (&f[0], &fm[0]),
            (&f[1], &fm[1]),
            (&f[2], &fm[0]),
            (&f[3], &fm[2]),
            (&f[4], &fm[3]),
            (&f[5], &fm[2]),
        ] {
            for (u, v) in a.iter().zip(b) {
                assert!((u - v).abs() <= 1e-9 * big, "{u} vs {v}");
            }
        }
    }

    /// A lossless spheroid: Q_ext = Q_sca; unnormalized α₁(0) = Q_sca; ½∫F₁₁ dμ = 1 normalized.
    #[test]
    fn lossless_and_normalizations() {
        let mut tm = TMatrix::new();
        let s = tm
            .compute(
                spheroid(0.5),
                3.0,
                Complex::new(1.33, 0.0),
                &TMatrixOptions::new(),
            )
            .unwrap();
        let q = s.efficiencies();
        assert!((q.q_ext - q.q_sca).abs() <= 1e-9 * q.q_ext, "{q:?}");
        let gu = s.greek(None, Normalization::Unnormalized).unwrap();
        assert!((gu.alpha1[0] - q.q_sca).abs() <= 1e-12 * q.q_sca);
        let rule = crate::special::gauss_legendre::<f64>(80).unwrap();
        let k = rule.nodes.len();
        let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; k]);
        let [f11, f12, f22, f33, f34, f44] = &mut f;
        s.phase_matrix(
            &rule.nodes,
            Normalization::Normalized,
            PhaseBuffers {
                f11,
                f12,
                f22,
                f33,
                f34,
                f44,
            },
        )
        .unwrap();
        let half: f64 = 0.5
            * rule
                .weights
                .iter()
                .zip(&f[0])
                .map(|(w, v)| w * v)
                .sum::<f64>();
        assert!((half - 1.0).abs() < 1e-12, "{half}");
        assert!(s.converged() && s.error_estimate() < 1e-8);
    }

    /// The derivatives of the efficiencies, Greek coefficients and phase matrix against central
    /// differences at fixed N and n_θ; the values equal `compute`'s.
    #[test]
    fn derivatives_against_differences() {
        let (eps, x, m) = (0.6, 3.0, Complex::new(1.45, 0.03));
        let base = TMatrixOptions::new().with_rtol(1e-10);
        let mut tm = TMatrix::new();
        let s = tm.compute(spheroid(eps), x, m, &base).unwrap();
        let (n, nth) = (s.n_terms(), s.n_theta());
        let q0 = s.efficiencies();
        let fixed = base.with_n_terms(n).with_n_theta(nth);
        let mut td = TMatrix::new();
        let d = td
            .compute_with_derivatives(spheroid(eps), x, m, &fixed)
            .unwrap();
        let qd = d.efficiencies();
        assert!((qd.q_ext - q0.q_ext).abs() <= 1e-14 * q0.q_ext);
        let jac = d.efficiencies_jacobian().unwrap();
        let (gv, gd) = d
            .greek_with_derivatives(Some(8), Normalization::Unnormalized)
            .unwrap();
        let mu = [-1.0, -0.2, 0.7];
        let mk = || -> [Vec<f64>; 6] { core::array::from_fn(|_| vec![0.0; mu.len()]) };
        let (mut f, mut fx, mut fe, mut fr, mut fi) = (mk(), mk(), mk(), mk(), mk());
        fn buf(b: &mut [Vec<f64>; 6]) -> PhaseBuffers<'_> {
            let [f11, f12, f22, f33, f34, f44] = b;
            PhaseBuffers {
                f11,
                f12,
                f22,
                f33,
                f34,
                f44,
            }
        }
        d.phase_matrix_with_derivatives(
            &mu,
            Normalization::Normalized,
            buf(&mut f),
            Derivatives {
                x: buf(&mut fx),
                aspect_ratio: buf(&mut fe),
                m_re: buf(&mut fr),
                m_im: buf(&mut fi),
            },
        )
        .unwrap();
        let h = 1e-5;
        let eval = |eps: f64, x: f64, m: Complex<f64>| {
            let mut t = TMatrix::new();
            let s = t.compute(spheroid(eps), x, m, &fixed).unwrap();
            let mut p = mk();
            s.phase_matrix(&mu, Normalization::Normalized, buf(&mut p))
                .unwrap();
            (
                s.efficiencies(),
                s.greek(Some(8), Normalization::Unnormalized).unwrap(),
                p,
            )
        };
        let dirs = [
            (0.0, h * x, Complex::new(0.0, 0.0), &jac.x, &gd.x, &fx, x),
            (
                h * eps,
                0.0,
                Complex::new(0.0, 0.0),
                &jac.aspect_ratio,
                &gd.aspect_ratio,
                &fe,
                eps,
            ),
            (
                0.0,
                0.0,
                Complex::new(h, 0.0),
                &jac.m_re,
                &gd.m_re,
                &fr,
                m.norm(),
            ),
            (
                0.0,
                0.0,
                Complex::new(0.0, h),
                &jac.m_im,
                &gd.m_im,
                &fi,
                m.norm(),
            ),
        ];
        for (de, dx, dm, jq, jg, jf, size) in dirs {
            let step = de + dx + dm.re + dm.im;
            let (qp, gp, pp) = eval(eps + de, x + dx, m + dm);
            let (qm, gm, pm) = eval(eps - de, x - dx, m - dm);
            let fd = |a: f64, b: f64| (a - b) / (2.0 * step);
            let ok = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-6 * scale / size;
            for (a, p, mm, v) in [
                (jq.q_ext, qp.q_ext, qm.q_ext, q0.q_ext),
                (jq.q_sca, qp.q_sca, qm.q_sca, q0.q_sca),
                (jq.q_abs, qp.q_abs, qm.q_abs, q0.q_abs),
                (jq.q_back, qp.q_back, qm.q_back, q0.q_back),
                (jq.g, qp.g, qm.g, q0.g),
            ] {
                assert!(ok(a, fd(p, mm), v.abs()), "{a} vs {}", fd(p, mm));
            }
            let big = gv.alpha1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            for l in 0..=8 {
                assert!(ok(jg.alpha1[l], fd(gp.alpha1[l], gm.alpha1[l]), big));
                assert!(ok(jg.beta1[l], fd(gp.beta1[l], gm.beta1[l]), big));
                assert!(ok(jg.alpha3[l], fd(gp.alpha3[l], gm.alpha3[l]), big));
            }
            let fbig = f[0].iter().fold(0.0f64, |a, v| a.max(v.abs()));
            for k in 0..6 {
                for j in 0..mu.len() {
                    assert!(ok(jf[k][j], fd(pp[k][j], pm[k][j]), fbig), "phase {k} {j}");
                }
            }
        }
    }

    /// A cap reached first raises `NotConverged`, unless an unconverged result is allowed.
    #[test]
    fn caps() {
        let mut tm = TMatrix::new();
        let start = driver::start_n(6.0, 2.0);
        let mut o = TMatrixOptions::new().with_rtol(1e-12);
        o.n_terms_max = Some(start + 1);
        let m = Complex::new(1.5, 0.01);
        assert!(matches!(
            tm.compute(spheroid(2.0), 6.0, m, &o),
            Err(Error::NotConverged { .. })
        ));
        let s = tm
            .compute(spheroid(2.0), 6.0, m, &o.with_allow_unconverged(true))
            .unwrap();
        assert!(!s.converged() && s.error_estimate() > 1e-12);
    }

    /// The Rayleigh limit: at x ≪ 1, ⟨C_abs⟩ = k Im ᾱ and ⟨C_sca⟩ = k⁴/(6π) mean |αⱼ|², with
    /// αⱼ = V(m² − 1)/(1 + Lⱼ(m² − 1)) from the depolarization factors Lⱼ (Bohren & Huffman
    /// 1983, §5.3), for oblate and prolate spheroids, to O(x²).
    #[test]
    fn the_rayleigh_limit() {
        let (x, m) = (0.01, Complex::new(1.5, 0.1));
        for eps in [0.25, 0.5, 2.0, 4.0] {
            let a = x * f64::cbrt(eps);
            let c = a / eps;
            // Prolate: L along the (long) symmetry axis c; oblate: L along the two equal axes a.
            let (l_a, l_c) = if c > a {
                let e = (1.0 - a * a / (c * c)).sqrt();
                let l_c =
                    (1.0 - e * e) / (e * e) * (-1.0 + ((1.0 + e) / (1.0 - e)).ln() / (2.0 * e));
                ((1.0 - l_c) / 2.0, l_c)
            } else {
                let e2 = 1.0 - c * c / (a * a);
                let g = ((1.0 - e2) / e2).sqrt();
                let l_a = g / (2.0 * e2) * (core::f64::consts::FRAC_PI_2 - g.atan()) - g * g / 2.0;
                (l_a, 1.0 - 2.0 * l_a)
            };
            let v = 4.0 / 3.0 * core::f64::consts::PI * x * x * x;
            let em1 = m * m - 1.0;
            let alpha = |l: f64| em1 * v / (1.0 + em1 * l);
            let al = [alpha(l_a), alpha(l_a), alpha(l_c)];
            let mean: Complex<f64> = al.iter().sum::<Complex<f64>>() / 3.0;
            let area = core::f64::consts::PI * x * x;
            let q_abs = mean.im / area;
            let q_sca = al.iter().map(|a| a.norm_sqr()).sum::<f64>()
                / 3.0
                / (6.0 * core::f64::consts::PI)
                / area;
            let mut tm = TMatrix::new();
            let s = tm
                .compute(
                    spheroid(eps),
                    x,
                    m,
                    &TMatrixOptions::new()
                        .with_rtol(1e-12)
                        .with_allow_unvalidated(true),
                )
                .unwrap();
            let q = s.efficiencies();
            assert!(
                (q.q_abs - q_abs).abs() < 1e-3 * q_abs,
                "eps {eps}: {} vs {q_abs}",
                q.q_abs
            );
            assert!(
                (q.q_sca - q_sca).abs() < 1e-3 * q_sca,
                "eps {eps}: {} vs {q_sca}",
                q.q_sca
            );
        }
    }

    #[test]
    fn invalid_inputs_and_domain() {
        let mut tm = TMatrix::new();
        let m = Complex::new(1.5, 0.0);
        let o = TMatrixOptions::new();
        for (eps, x, mm) in [
            (0.0, 1.0, m),
            (f64::NAN, 1.0, m),
            (2.0, -1.0, m),
            (2.0, 1.0, Complex::new(1.5, -0.1)),
        ] {
            assert!(matches!(
                tm.compute(spheroid(eps), x, mm, &o),
                Err(Error::InvalidInput(_))
            ));
        }
        assert!(matches!(
            tm.compute(spheroid(8.0), 1.0, m, &o),
            Err(Error::OutsideValidatedDomain(_))
        ));
        assert!(
            tm.compute(spheroid(2.0), 3.0, m, &o.with_n_theta(3).with_n_terms(4))
                .is_ok()
        );
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    /// The convergence loop on harder cases: run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn loop_costs() {
        for &(eps, x, m) in &[
            (2.0, 5.0, Complex::new(1.53, 0.008)),
            (2.0, 15.0, Complex::new(1.53, 0.008)),
            (0.5, 15.0, Complex::new(1.53, 0.008)),
            (4.0, 10.0, Complex::new(1.33, 0.0)),
            (0.25, 10.0, Complex::new(1.33, 0.0)),
            (0.2, 12.0, Complex::new(1.53, 0.008)),
            (5.0, 12.0, Complex::new(1.53, 0.008)),
        ] {
            let mut tm = TMatrix::new();
            let t0 = std::time::Instant::now();
            let r = tm.compute(
                Shape::Spheroid { aspect_ratio: eps },
                x,
                m,
                &TMatrixOptions::new().with_allow_unvalidated(true),
            );
            match r {
                Ok(s) => {
                    let refined = s
                        .precision_paths()
                        .iter()
                        .filter(|p| **p != PrecisionPath::F64)
                        .count();
                    let q = s.efficiencies();
                    println!(
                        "eps {eps} x {x}: {:.2?} N {} nθ {} refined {refined} est {:.1e} Qext {:.10} Qsca {:.10}",
                        t0.elapsed(),
                        s.n_terms(),
                        s.n_theta(),
                        s.error_estimate(),
                        q.q_ext,
                        q.q_sca
                    );
                }
                Err(e) => println!("eps {eps} x {x}: {:.2?} {e}", t0.elapsed()),
            }
        }
    }
}
