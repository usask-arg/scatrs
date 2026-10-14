//! Greek (generalized-spherical-function) expansion coefficients of the phase matrix.

use super::{MAX_MIE_ORDER, MieSolution, Normalization, PhaseBuffers};
use crate::scalar::Real;
use crate::special::{GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre, wigner_d};
use crate::{Error, Result};

/// The expansion coefficients of a sphere's phase matrix for l = 0, …, L (Mishchenko, Travis &
/// Lacis 2002, Eqs. 4.75–4.80, with `wigner_d`'s convention):
///
/// F₁₁ = Σ α₁ d⁰⁰, F₂₂ + F₃₃ = Σ (α₂ + α₃) d²², F₂₂ − F₃₃ = Σ (α₂ − α₃) d²⁻², F₄₄ = Σ α₄ d⁰⁰,
/// F₁₂ = Σ β₁ d⁰², F₃₄ = Σ β₂ d⁰², each dₗ at cos Θ.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GreekCoefficients<T> {
    /// α₁ (of F₁₁); α₁ at l = 0 is 1 and at l = 1 is 3g when normalized.
    pub alpha1: Vec<T>,
    /// α₂.
    pub alpha2: Vec<T>,
    /// α₃.
    pub alpha3: Vec<T>,
    /// α₄ (of F₄₄).
    pub alpha4: Vec<T>,
    /// β₁ (of F₁₂).
    pub beta1: Vec<T>,
    /// β₂ (of F₃₄).
    pub beta2: Vec<T>,
}

impl<T: Real> GreekCoefficients<T> {
    fn zeros(len: usize) -> Self {
        let z = || vec![T::zero(); len];
        Self {
            alpha1: z(),
            alpha2: z(),
            alpha3: z(),
            alpha4: z(),
            beta1: z(),
            beta2: z(),
        }
    }

    /// The highest order L (0 if the arrays have been emptied; this crate never returns empty
    /// arrays).
    pub fn order(&self) -> usize {
        self.alpha1.len().saturating_sub(1)
    }

    fn arrays_mut(&mut self) -> [&mut Vec<T>; 6] {
        [
            &mut self.alpha1,
            &mut self.alpha2,
            &mut self.alpha3,
            &mut self.alpha4,
            &mut self.beta1,
            &mut self.beta2,
        ]
    }

    fn arrays(&self) -> [&Vec<T>; 6] {
        [
            &self.alpha1,
            &self.alpha2,
            &self.alpha3,
            &self.alpha4,
            &self.beta1,
            &self.beta2,
        ]
    }

    /// Extends every array with zeros to `len` values.
    fn pad(&mut self, len: usize) {
        for v in self.arrays_mut() {
            v.resize(len, T::zero());
        }
    }

    /// Applies `f` to every coefficient.
    fn map(&mut self, f: impl Fn(T) -> T) {
        for v in self.arrays_mut() {
            for c in v.iter_mut() {
                *c = f(*c);
            }
        }
    }

    /// Replaces every coefficient c by f(c, the matching coefficient of `other`).
    fn zip_map(&mut self, other: &Self, f: impl Fn(T, T) -> T) {
        for (v, w) in self.arrays_mut().into_iter().zip(other.arrays()) {
            for (c, &o) in v.iter_mut().zip(w) {
                *c = f(*c, o);
            }
        }
    }
}

/// The phase-matrix elements of one table at the projector's nodes: F₁₁, F₁₂, F₃₃, F₃₄ (for a
/// sphere F₂₂ = F₁₁ and F₄₄ = F₃₃).
#[derive(Clone, Copy, Debug)]
pub struct PhaseTables<'a, T> {
    /// F₁₁.
    pub f11: &'a [T],
    /// F₁₂.
    pub f12: &'a [T],
    /// F₃₃.
    pub f33: &'a [T],
    /// F₃₄.
    pub f34: &'a [T],
}

/// Gauss–Legendre quadrature that projects phase-matrix tables onto the Greek coefficients up
/// to an order L.
///
/// With n nodes the projection is exact (up to rounding) for elements that are polynomials of
/// degree ≤ 2n − 1 − L in μ; a sphere's elements have degree 2N, so n = N + ⌈(L + 1)/2⌉
/// suffices ([`MieSolution::greek`] chooses it). Build a projector once and reuse it: for
/// example, sum the elements of many spheres at its nodes and project once.
#[derive(Clone, Debug)]
pub struct GreekProjector<T> {
    order: usize,
    rule: GaussLegendre<T>,
}

impl<T: Real> GreekProjector<T> {
    /// A projector for orders 0, …, `order` with an `n_nodes`-point Gauss–Legendre rule.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`]; as [`gauss_legendre`].
    pub fn new(order: usize, n_nodes: usize) -> Result<Self> {
        check_order(order)?;
        Ok(Self {
            order,
            rule: gauss_legendre(n_nodes)?,
        })
    }

    /// The highest order L.
    pub fn order(&self) -> usize {
        self.order
    }

    /// The cosines at which to tabulate the phase matrix (ascending).
    pub fn nodes(&self) -> &[T] {
        &self.rule.nodes
    }

    /// The Greek coefficients of each table: c = (2l + 1)/2 · Σ wₖ F(μₖ) dₗ(μₖ). The tables share
    /// the generalized spherical functions, which dominate the cost: O(nodes·L) for any number of
    /// tables.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if a table's length differs from the number of nodes.
    pub fn project(&self, tables: &[PhaseTables<'_, T>]) -> Result<Vec<GreekCoefficients<T>>> {
        let n = self.rule.nodes.len();
        for t in tables {
            let lens = [t.f11.len(), t.f12.len(), t.f33.len(), t.f34.len()];
            if lens.iter().any(|&l| l != n) {
                return Err(Error::InvalidInput(format!(
                    "phase-matrix tables must have one value per node ({n}), got {lens:?}"
                )));
            }
        }
        if tables.is_empty() {
            return Ok(Vec::new());
        }
        let len = self.order + 1;
        let mut out: Vec<_> = tables
            .iter()
            .map(|_| GreekCoefficients::zeros(len))
            .collect();
        let mut d = [
            vec![T::zero(); len],
            vec![T::zero(); len],
            vec![T::zero(); len],
            vec![T::zero(); len],
        ];
        for (k, (&mu, &w)) in self.rule.nodes.iter().zip(&self.rule.weights).enumerate() {
            let [d00, d22, d2m2, d02] = &mut d;
            wigner_d(0, 0, mu, d00)?;
            wigner_d(2, 2, mu, d22)?;
            wigner_d(2, -2, mu, d2m2)?;
            wigner_d(0, 2, mu, d02)?;
            for (t, c) in tables.iter().zip(&mut out) {
                let f11 = w * t.f11[k];
                let f33 = w * t.f33[k];
                let (sum, diff) = (f11 + f33, f11 - f33);
                let (f12, f34) = (w * t.f12[k], w * t.f34[k]);
                for l in 0..len {
                    c.alpha1[l] += f11 * d00[l];
                    c.alpha4[l] += f33 * d00[l];
                    c.alpha2[l] += sum * d22[l]; // α₂ + α₃ for now
                    c.alpha3[l] += diff * d2m2[l]; // α₂ − α₃ for now
                    c.beta1[l] += f12 * d02[l];
                    c.beta2[l] += f34 * d02[l];
                }
            }
        }
        let half = T::from_f64(0.5);
        for c in &mut out {
            for l in 0..len {
                let s = T::from_f64((2 * l + 1) as f64) * half;
                let (plus, minus) = (c.alpha2[l], c.alpha3[l]);
                c.alpha1[l] *= s;
                c.alpha4[l] *= s;
                c.alpha2[l] = s * half * (plus + minus);
                c.alpha3[l] = s * half * (plus - minus);
                c.beta1[l] *= s;
                c.beta2[l] *= s;
            }
        }
        Ok(out)
    }
}

/// The highest order [`MieSolution::greek`] accepts, 2²¹: twice [`MAX_MIE_ORDER`], since the
/// coefficients of an N-term sphere vanish above order 2N.
///
/// Two other limits apply: all orders of a sphere need N + N + 1 ≤ 2¹⁷ quadrature nodes
/// ([`MAX_GAUSS_LEGENDRE_POINTS`]), so N ≤ 2¹⁶ − 1 (x up to about 6.5·10⁴), and every order
/// asked for is stored, so the largest order costs 6·(2²¹ + 1) values per coefficient set
/// even for a tiny sphere (100 MB in `f64`, four sets with derivatives).
pub const MAX_GREEK_ORDER: usize = 2 * MAX_MIE_ORDER;

fn check_order(order: usize) -> Result<()> {
    if order > MAX_GREEK_ORDER {
        return Err(Error::InvalidInput(format!(
            "the Greek order must be at most {MAX_GREEK_ORDER}, got {order}"
        )));
    }
    Ok(())
}

/// Phase-matrix tables over four vectors.
fn tables<T>(f: &[Vec<T>; 4]) -> PhaseTables<'_, T> {
    PhaseTables {
        f11: &f[0],
        f12: &f[1],
        f33: &f[2],
        f34: &f[3],
    }
}

/// Phase-matrix buffers over four vectors.
fn buffers<T>(b: &mut [Vec<T>; 4]) -> PhaseBuffers<'_, T> {
    let [f11, f12, f33, f34] = b;
    PhaseBuffers { f11, f12, f33, f34 }
}

impl<T: Real> MieSolution<'_, T> {
    /// The order L asked for (default 2N) and the projector for orders up to min(L, 2N): the
    /// coefficients of an N-term sphere vanish above 2N.
    fn greek_projector(&self, order: Option<usize>) -> Result<(usize, GreekProjector<T>)> {
        let n = self.n_terms();
        let order = order.unwrap_or(2 * n);
        check_order(order)?;
        let projected = order.min(2 * n);
        let nodes = n + (projected + 1).div_ceil(2);
        if nodes > MAX_GAUSS_LEGENDRE_POINTS {
            return Err(Error::InvalidInput(format!(
                "the Greek coefficients of a sphere with {n} terms up to order {projected} need \
                 {nodes} quadrature nodes, more than {MAX_GAUSS_LEGENDRE_POINTS}"
            )));
        }
        Ok((order, GreekProjector::new(projected, nodes)?))
    }

    /// The Greek coefficients up to `order` (default: all, 2N), exact up to rounding: the phase
    /// matrix at the nodes of an (N + ⌈(L + 1)/2⌉)-point Gauss–Legendre rule, projected. Orders
    /// above 2N are exactly 0. Normalized coefficients are divided by the projected α₁ at l = 0,
    /// so that it is exactly 1; unnormalized ones are those times Q_sca.
    ///
    /// The cost is the amplitudes at the N + ⌈(L + 1)/2⌉ nodes, O(N²) for L ≈ 2N, and four
    /// Wigner recurrences up to L at each, O(N·L): for all orders, about 0.1 s at x = 10³ and
    /// 8 s at x = 10⁴ in `f64` on an Apple M4 Pro.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`], or if the rule would need more
    /// than [`MAX_GAUSS_LEGENDRE_POINTS`] nodes (N + ⌈(min(L, 2N) + 1)/2⌉ > 2¹⁷, so N above
    /// about 4.4·10⁴ for all orders).
    pub fn greek(
        &self,
        order: Option<usize>,
        normalization: Normalization,
    ) -> Result<GreekCoefficients<T>> {
        let (order, projector) = self.greek_projector(order)?;
        let n = projector.nodes().len();
        let mut f: [Vec<T>; 4] = core::array::from_fn(|_| vec![T::zero(); n]);
        self.phase_matrix(
            projector.nodes(),
            Normalization::Unnormalized,
            buffers(&mut f),
        )?;
        let mut g = projector.project(&[tables(&f)])?.remove(0);
        self.normalize(&mut g, normalization);
        g.pad(order + 1);
        Ok(g)
    }

    /// The Greek coefficients and their derivatives with respect to x, Re m and Im m.
    ///
    /// # Errors
    ///
    /// As [`greek`](Self::greek), and [`Error::InvalidInput`] if the solution was computed
    /// without derivatives.
    pub fn greek_with_derivatives(
        &self,
        order: Option<usize>,
        normalization: Normalization,
    ) -> Result<(GreekCoefficients<T>, [GreekCoefficients<T>; 3])> {
        let Some(jac) = self.efficiencies_jacobian() else {
            return Err(Error::InvalidInput(
                "Greek derivatives need a solution from Mie::compute_with_derivatives".into(),
            ));
        };
        let (order, projector) = self.greek_projector(order)?;
        let n = projector.nodes().len();
        let mut f: [[Vec<T>; 4]; 4] =
            core::array::from_fn(|_| core::array::from_fn(|_| vec![T::zero(); n]));
        {
            let [v, dx, dr, di] = &mut f;
            self.phase_matrix_with_derivatives(
                projector.nodes(),
                Normalization::Unnormalized,
                buffers(v),
                [buffers(dx), buffers(dr), buffers(di)],
            )?;
        }
        let projected =
            projector.project(&[tables(&f[0]), tables(&f[1]), tables(&f[2]), tables(&f[3])])?;
        let [r, dr_x, dr_re, dr_im]: [GreekCoefficients<T>; 4] =
            projected.try_into().expect("four tables");
        // Normalized g = r/r₀: dg = (dr − g·dr₀)/r₀. Unnormalized: Q·g, d = dQ·g + Q·dg.
        let r0 = r.alpha1[0];
        let mut g = r;
        self.normalize(&mut g, Normalization::Normalized);
        let q = self.efficiencies().q_sca;
        let dq = [jac.x.q_sca, jac.m_re.q_sca, jac.m_im.q_sca];
        let mut d = [dr_x, dr_re, dr_im];
        for (p, dp) in d.iter_mut().enumerate() {
            let dr0 = dp.alpha1[0];
            dp.zip_map(&g, |dv, gv| (dv - gv * dr0) / r0);
            if normalization == Normalization::Unnormalized {
                dp.zip_map(&g, |dv, gv| dq[p] * gv + q * dv);
            }
        }
        if normalization == Normalization::Unnormalized {
            g.map(|v| v * q);
        }
        g.pad(order + 1);
        for dp in &mut d {
            dp.pad(order + 1);
        }
        Ok((g, d))
    }

    /// Divides projected coefficients by their α₁ at l = 0, which is then exactly 1, and
    /// multiplies them by Q_sca if unnormalized.
    fn normalize(&self, g: &mut GreekCoefficients<T>, normalization: Normalization) {
        let r0 = g.alpha1[0];
        g.map(|v| v / r0);
        g.alpha1[0] = T::one();
        if normalization == Normalization::Unnormalized {
            let q = self.efficiencies().q_sca;
            g.map(|v| v * q);
        }
    }
}
