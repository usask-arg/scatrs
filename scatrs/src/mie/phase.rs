//! The phase matrix of a sphere.

use num_complex::Complex;

use super::{AmplitudeBuffers, AngularTable, MieSolution};
use crate::scalar::Real;
use crate::{Error, Result};

/// How the phase matrix (and the Greek coefficients) are scaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Normalization {
    /// ½∫F₁₁ dμ = 1 (so that the Greek coefficient α₁ at l = 0 is 1).
    Normalized,
    /// The normalized form times Q_sca: the phase matrix per unit geometric cross section, the
    /// quantity that sums over a size distribution.
    Unnormalized,
}

/// Output buffers for the independent elements of a sphere's phase matrix, or for their
/// derivatives, one value per cosine. For a sphere F₂₂ = F₁₁ and F₄₄ = F₃₃.
#[derive(Debug)]
pub struct PhaseBuffers<'b, T> {
    /// F₁₁ = (|S₁|² + |S₂|²)/2, scaled.
    pub f11: &'b mut [T],
    /// F₁₂ = (|S₂|² − |S₁|²)/2, scaled.
    pub f12: &'b mut [T],
    /// F₃₃ = Re(S₂S₁*), scaled.
    pub f33: &'b mut [T],
    /// F₃₄ = Im(S₂S₁*), scaled.
    pub f34: &'b mut [T],
}

impl<T> PhaseBuffers<'_, T> {
    fn check(&self, len: usize, what: &str) -> Result<()> {
        let lens = [
            self.f11.len(),
            self.f12.len(),
            self.f33.len(),
            self.f34.len(),
        ];
        if lens.iter().any(|&l| l != len) {
            return Err(Error::InvalidInput(format!(
                "the {what} buffers must have one value per cosine ({len}), got {lens:?}"
            )));
        }
        Ok(())
    }
}

/// (|S₁|² + |S₂|²)/2, (|S₂|² − |S₁|²)/2, Re(S₂S₁*), Im(S₂S₁*).
pub(crate) fn elements<T: Real>(s1: Complex<T>, s2: Complex<T>) -> [T; 4] {
    let half = T::from_f64(0.5);
    let (n1, n2) = (s1.norm_sqr(), s2.norm_sqr());
    let p = s2 * s1.conj();
    [half * (n1 + n2), half * (n2 - n1), p.re, p.im]
}

/// The derivatives of [`elements`] for amplitude derivatives (d1, d2).
pub(crate) fn element_derivatives<T: Real>(
    s1: Complex<T>,
    s2: Complex<T>,
    d1: Complex<T>,
    d2: Complex<T>,
) -> [T; 4] {
    // d|S|² = 2 Re(S* dS); d(S₂S₁*) = dS₂ S₁* + S₂ dS₁*.
    let r1 = (s1.conj() * d1).re;
    let r2 = (s2.conj() * d2).re;
    let p = d2 * s1.conj() + s2 * d1.conj();
    [r1 + r2, r2 - r1, p.re, p.im]
}

impl<T: Real> MieSolution<'_, T> {
    /// The scale of the phase matrix: 4/(x²Q_sca) for [`Normalization::Normalized`], 4/x² for
    /// [`Normalization::Unnormalized`].
    fn phase_scale(&self, normalization: Normalization) -> T {
        let x = self.x();
        let four_x2 = T::from_f64(4.0) / (x * x);
        match normalization {
            Normalization::Normalized => four_x2 / self.efficiencies().q_sca,
            Normalization::Unnormalized => four_x2,
        }
    }

    /// The phase matrix F₁₁, F₁₂, F₃₃, F₃₄ at the cosines `mu` (Bohren & Huffman 1983, §4.4),
    /// scaled by `normalization`.
    ///
    /// # Errors
    ///
    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] if a buffer's length
    /// differs from `mu.len()`. Nothing is written then.
    pub fn phase_matrix(
        &self,
        mu: &[T],
        normalization: Normalization,
        out: PhaseBuffers<'_, T>,
    ) -> Result<()> {
        out.check(mu.len(), "phase-matrix")?;
        let zero = Complex::new(T::zero(), T::zero());
        let (mut s1, mut s2) = (vec![zero; mu.len()], vec![zero; mu.len()]);
        self.amplitudes(
            mu,
            AmplitudeBuffers {
                s1: &mut s1,
                s2: &mut s2,
            },
        )?;
        self.phase_from_amplitudes(&s1, &s2, normalization, out);
        Ok(())
    }

    /// The phase matrix at the cosines of `table`, as [`phase_matrix`](Self::phase_matrix)
    /// gives it (bitwise).
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
        let len = table.mu().len();
        out.check(len, "phase-matrix")?;
        let zero = Complex::new(T::zero(), T::zero());
        let (mut s1, mut s2) = (vec![zero; len], vec![zero; len]);
        self.amplitudes_from_table(
            table,
            AmplitudeBuffers {
                s1: &mut s1,
                s2: &mut s2,
            },
        )?;
        self.phase_from_amplitudes(&s1, &s2, normalization, out);
        Ok(())
    }

    /// The phase matrix from S₁ and S₂.
    fn phase_from_amplitudes(
        &self,
        s1: &[Complex<T>],
        s2: &[Complex<T>],
        normalization: Normalization,
        out: PhaseBuffers<'_, T>,
    ) {
        let c = self.phase_scale(normalization);
        for j in 0..s1.len() {
            let f = elements(s1[j], s2[j]);
            out.f11[j] = c * f[0];
            out.f12[j] = c * f[1];
            out.f33[j] = c * f[2];
            out.f34[j] = c * f[3];
        }
    }

    /// The phase matrix and its derivatives with respect to x, Re m and Im m, in `d[0]`, `d[1]`
    /// and `d[2]`. For [`Normalization::Normalized`] the derivatives include those of Q_sca.
    ///
    /// # Errors
    ///
    /// As [`phase_matrix`](Self::phase_matrix), and [`Error::InvalidInput`] if the solution was
    /// computed without derivatives.
    pub fn phase_matrix_with_derivatives(
        &self,
        mu: &[T],
        normalization: Normalization,
        values: PhaseBuffers<'_, T>,
        d: [PhaseBuffers<'_, T>; 3],
    ) -> Result<()> {
        let s = self.phase_derivative_amplitudes(mu.len(), &values, &d, |s| {
            let [s1, s2, dx1, dx2, dm1, dm2] = s;
            self.amplitudes_with_derivatives(
                mu,
                AmplitudeBuffers { s1, s2 },
                AmplitudeBuffers { s1: dx1, s2: dx2 },
                AmplitudeBuffers { s1: dm1, s2: dm2 },
            )
        })?;
        self.phase_derivatives_from_amplitudes(&s, normalization, values, d);
        Ok(())
    }

    /// The phase matrix and its derivatives at the cosines of `table`, as
    /// [`phase_matrix_with_derivatives`](Self::phase_matrix_with_derivatives) gives them
    /// (bitwise).
    ///
    /// # Errors
    ///
    /// As [`phase_matrix_from_table`](Self::phase_matrix_from_table), and
    /// [`Error::InvalidInput`] if the solution was computed without derivatives.
    pub fn phase_matrix_with_derivatives_from_table(
        &self,
        table: &AngularTable<T>,
        normalization: Normalization,
        values: PhaseBuffers<'_, T>,
        d: [PhaseBuffers<'_, T>; 3],
    ) -> Result<()> {
        let s = self.phase_derivative_amplitudes(table.mu().len(), &values, &d, |s| {
            let [s1, s2, dx1, dx2, dm1, dm2] = s;
            self.amplitudes_with_derivatives_from_table(
                table,
                AmplitudeBuffers { s1, s2 },
                AmplitudeBuffers { s1: dx1, s2: dx2 },
                AmplitudeBuffers { s1: dm1, s2: dm2 },
            )
        })?;
        self.phase_derivatives_from_amplitudes(&s, normalization, values, d);
        Ok(())
    }

    /// After checking that the solution has derivatives and the buffers' lengths are `len`,
    /// S₁, S₂, ∂S₁/∂x, ∂S₂/∂x, dS₁/dm, dS₂/dm from `amplitudes`.
    fn phase_derivative_amplitudes(
        &self,
        len: usize,
        values: &PhaseBuffers<'_, T>,
        d: &[PhaseBuffers<'_, T>; 3],
        amplitudes: impl FnOnce([&mut [Complex<T>]; 6]) -> Result<()>,
    ) -> Result<[Vec<Complex<T>>; 6]> {
        if self.derivatives().is_none() {
            return Err(Error::InvalidInput(
                "phase-matrix derivatives need a solution from Mie::compute_with_derivatives"
                    .into(),
            ));
        }
        values.check(len, "phase-matrix")?;
        for buffers in d {
            buffers.check(len, "derivative")?;
        }
        let zero = Complex::new(T::zero(), T::zero());
        let mut s: [Vec<Complex<T>>; 6] = core::array::from_fn(|_| vec![zero; len]);
        amplitudes(s.each_mut().map(|v| v.as_mut_slice()))?;
        Ok(s)
    }

    /// The phase matrix and its derivatives from the amplitudes and their derivatives (as
    /// [`phase_derivative_amplitudes`](Self::phase_derivative_amplitudes) orders them).
    fn phase_derivatives_from_amplitudes(
        &self,
        s: &[Vec<Complex<T>>; 6],
        normalization: Normalization,
        values: PhaseBuffers<'_, T>,
        d: [PhaseBuffers<'_, T>; 3],
    ) {
        let jac = self
            .efficiencies_jacobian()
            .expect("a solution with derivatives");
        // The scale c and its derivatives: c = 4/(x²Q) or 4/x².
        let x = self.x();
        let c = self.phase_scale(normalization);
        let two = T::from_f64(2.0);
        let dc = match normalization {
            Normalization::Normalized => {
                let q = self.efficiencies().q_sca;
                [
                    -c * (two / x + jac.x.q_sca / q),
                    -c * jac.m_re.q_sca / q,
                    -c * jac.m_im.q_sca / q,
                ]
            }
            Normalization::Unnormalized => [-two * c / x, T::zero(), T::zero()],
        };
        let i1 = Complex::new(T::zero(), T::one());
        let v = values;
        let [mut dx, mut dr, mut di] = d;
        let [s1s, s2s, dx1, dx2, dm1, dm2] = s;
        for (j, (&s1, &s2)) in s1s.iter().zip(s2s).enumerate() {
            let f = elements(s1, s2);
            // ∂S/∂x, ∂S/∂(Re m) = dS/dm, ∂S/∂(Im m) = i·dS/dm.
            let df = [
                element_derivatives(s1, s2, dx1[j], dx2[j]),
                element_derivatives(s1, s2, dm1[j], dm2[j]),
                element_derivatives(s1, s2, i1 * dm1[j], i1 * dm2[j]),
            ];
            v.f11[j] = c * f[0];
            v.f12[j] = c * f[1];
            v.f33[j] = c * f[2];
            v.f34[j] = c * f[3];
            for (p, out) in [&mut dx, &mut dr, &mut di].into_iter().enumerate() {
                let g = |k: usize| c * df[p][k] + dc[p] * f[k];
                out.f11[j] = g(0);
                out.f12[j] = g(1);
                out.f33[j] = g(2);
                out.f34[j] = g(3);
            }
        }
    }
}
