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

#[derive(Debug)]
struct State {
    x: f64,
    eps: f64,
    m: Complex<f64>,
    rtol: f64,
    converged: driver::Converged,
    /// The Greek coefficients up to 2N (unnormalized, 1/k²), computed on first use.
    greek: OnceLock<Result<GreekCoefficients<f64>>>,
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
        self.state = Some(State {
            x,
            eps,
            m,
            rtol: opts.rtol,
            converged,
            greek: OnceLock::new(),
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

    /// πx², the geometric cross section of the equal-volume sphere in 1/k².
    fn area(&self) -> f64 {
        core::f64::consts::PI * self.s.x * self.s.x
    }

    /// The orientation-averaged efficiencies, per unit geometric cross section of the
    /// equal-volume sphere: Q_ext and Q_sca from T (Mishchenko 1991), Q_abs = Q_ext − Q_sca,
    /// g = α₁(1)/3, and Q_back = 4π F₁₁(180°)/(πx²) from the Greek expansion.
    pub fn efficiencies(&self) -> Efficiencies<f64> {
        let (ext, sca) = self.solved().t.cross_sections();
        let area = self.area();
        let (g, back) = match self.greek_raw() {
            Ok(gr) => {
                let a0 = gr.alpha1[0];
                let g = if a0 > 0.0 {
                    gr.alpha1.get(1).copied().unwrap_or(0.0) / (3.0 * a0)
                } else {
                    0.0
                };
                // F₁₁(−1) = Σ α₁ₗ (−1)ˡ.
                let f11: f64 = gr
                    .alpha1
                    .iter()
                    .enumerate()
                    .map(|(l, &v)| if l % 2 == 0 { v } else { -v })
                    .sum();
                (g, 4.0 * core::f64::consts::PI * f11 / area)
            }
            Err(_) => (f64::NAN, f64::NAN),
        };
        Efficiencies {
            q_ext: ext / area,
            q_sca: sca / area,
            q_abs: (ext - sca) / area,
            q_back: back,
            g,
        }
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
        let n = self.n_terms();
        let order = order.unwrap_or(2 * n);
        if order > MAX_GREEK_ORDER {
            return Err(Error::InvalidInput(format!(
                "the Greek order must be at most {MAX_GREEK_ORDER}"
            )));
        }
        let mut g = self.greek_raw()?.clone();
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
        let a0 = g.alpha1[0];
        let scale = match normalization {
            Normalization::Normalized => 1.0 / a0,
            // α₁(0) = C_sca/(4π) in 1/k²: Q_sca/α₁(0) = 4π/(πx²).
            Normalization::Unnormalized => 4.0 * core::f64::consts::PI / self.area(),
        };
        g.map(|v| v * scale);
        if normalization == Normalization::Normalized {
            g.alpha1[0] = 1.0;
        }
        g.pad(order + 1);
        Ok(g)
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
        out: PhaseBuffers<'_>,
    ) -> Result<()> {
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
        let g = self.greek(None, normalization)?;
        let len = g.alpha1.len();
        let mut d: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; len]);
        for (j, &c) in mu.iter().enumerate() {
            let [d00, d22, d2m2, d02] = &mut d;
            wigner_d(0, 0, c, d00)?;
            wigner_d(2, 2, c, d22)?;
            wigner_d(2, -2, c, d2m2)?;
            wigner_d(0, 2, c, d02)?;
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            let plus: Vec<f64> = g.alpha2.iter().zip(&g.alpha3).map(|(a, b)| a + b).collect();
            let minus: Vec<f64> = g.alpha2.iter().zip(&g.alpha3).map(|(a, b)| a - b).collect();
            let (sum, diff) = (dot(&plus, d22), dot(&minus, d2m2));
            out.f11[j] = dot(&g.alpha1, d00);
            out.f44[j] = dot(&g.alpha4, d00);
            out.f22[j] = 0.5 * (sum + diff);
            out.f33[j] = 0.5 * (sum - diff);
            out.f12[j] = dot(&g.beta1, d02);
            out.f34[j] = dot(&g.beta2, d02);
        }
        Ok(())
    }

    /// The aspect ratio, size parameter, index and tolerance this solution was computed for.
    pub fn inputs(&self) -> (f64, f64, Complex<f64>, f64) {
        (self.s.eps, self.s.x, self.s.m, self.s.rtol)
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
