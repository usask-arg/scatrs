//! The amplitude scattering functions S₁ and S₂.

use num_complex::Complex;

use super::MieSolution;
use crate::scalar::Real;
use crate::{Error, Result};

/// Output buffers for S₁ and S₂, or for their derivatives, one value per cosine.
#[derive(Debug)]
pub struct AmplitudeBuffers<'b, T> {
    /// S₁ (perpendicular polarization).
    pub s1: &'b mut [Complex<T>],
    /// S₂ (parallel polarization).
    pub s2: &'b mut [Complex<T>],
}

impl<T> AmplitudeBuffers<'_, T> {
    fn check(&self, len: usize, what: &str) -> Result<()> {
        if self.s1.len() != len || self.s2.len() != len {
            return Err(Error::InvalidInput(format!(
                "the {what} buffers must have one value per cosine ({len}), got {} and {}",
                self.s1.len(),
                self.s2.len()
            )));
        }
        Ok(())
    }
}

/// Every cosine in [−1, 1] (for `Dual`, its value) and finite (for `Dual`, its derivative
/// parts too).
fn check_cosines<T: Real>(mu: &[T]) -> Result<()> {
    match mu
        .iter()
        .find(|v| !(v.is_finite() && (-1.0..=1.0).contains(&v.approx_f64())))
    {
        Some(v) => Err(Error::InvalidInput(format!(
            "a cosine of the scattering angle must be finite and in [-1, 1], and for dual \
             numbers have finite derivative parts, got {}",
            v.approx_f64()
        ))),
        None => Ok(()),
    }
}

/// Calls `f(n, wπₙ, wτₙ)` for n = 1..=n_max, with w = (2n + 1)/(n(n + 1)).
///
/// πₙ and τₙ by the upward recurrences (Bohren & Huffman 1983, Eq. 4.47):
/// πₙ = ((2n − 1)/(n − 1)) μ πₙ₋₁ − (n/(n − 1)) πₙ₋₂ from π₀ = 0, π₁ = 1, and
/// τₙ = n μ πₙ − (n + 1) πₙ₋₁.
fn for_each_order<T: Real>(mu: T, n_max: usize, mut f: impl FnMut(usize, T, T)) {
    let mut pi_prev = T::zero();
    let mut pi = T::one();
    for n in 1..=n_max {
        let nf = T::from_f64(n as f64);
        let n1 = nf + T::one();
        let tau = nf * mu * pi - n1 * pi_prev;
        let w = T::from_f64((2 * n + 1) as f64) / (nf * n1);
        f(n, w * pi, w * tau);
        let pi_next = (T::from_f64((2 * n + 1) as f64) * mu * pi - n1 * pi_prev) / nf;
        pi_prev = pi;
        pi = pi_next;
    }
}

impl<T: Real> MieSolution<'_, T> {
    /// S₁(μ) and S₂(μ) at the cosines `mu` of the scattering angle (Bohren & Huffman 1983,
    /// Eq. 4.74): S₁ = Σ (2n + 1)/(n(n + 1)) (aₙπₙ + bₙτₙ) and
    /// S₂ = Σ (2n + 1)/(n(n + 1)) (aₙτₙ + bₙπₙ).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if a cosine is NaN or outside [−1, 1] (for `Dual`, its value),
    /// or a buffer's length differs from `mu.len()`. Nothing is written then.
    pub fn amplitudes(&self, mu: &[T], out: AmplitudeBuffers<'_, T>) -> Result<()> {
        out.check(mu.len(), "amplitude")?;
        check_cosines(mu)?;
        let (a, b) = (&self.state.a, &self.state.b);
        for (j, &muj) in mu.iter().enumerate() {
            let zero = Complex::new(T::zero(), T::zero());
            let (mut s1, mut s2) = (zero, zero);
            for_each_order(muj, a.len(), |n, pi, tau| {
                let (an, bn) = (a[n - 1], b[n - 1]);
                s1 = s1 + an.scale(pi) + bn.scale(tau);
                s2 = s2 + an.scale(tau) + bn.scale(pi);
            });
            out.s1[j] = s1;
            out.s2[j] = s2;
        }
        Ok(())
    }

    /// S₁, S₂ and their derivatives: `d_dx` receives ∂S/∂x and `d_dm` the complex
    /// derivative dS/dm (∂/∂(Re m) = d/dm, ∂/∂(Im m) = i·d/dm).
    ///
    /// # Errors
    ///
    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] if the solution was
    /// computed without derivatives.
    pub fn amplitudes_with_derivatives(
        &self,
        mu: &[T],
        values: AmplitudeBuffers<'_, T>,
        d_dx: AmplitudeBuffers<'_, T>,
        d_dm: AmplitudeBuffers<'_, T>,
    ) -> Result<()> {
        let Some(d) = self.derivatives() else {
            return Err(Error::InvalidInput(
                "amplitude derivatives need a solution from Mie::compute_with_derivatives".into(),
            ));
        };
        values.check(mu.len(), "amplitude")?;
        d_dx.check(mu.len(), "derivative")?;
        d_dm.check(mu.len(), "derivative")?;
        check_cosines(mu)?;
        let (a, b) = (&self.state.a, &self.state.b);
        for (j, &muj) in mu.iter().enumerate() {
            let zero = Complex::new(T::zero(), T::zero());
            let mut s = [zero; 6]; // S₁, S₂, ∂S₁/∂x, ∂S₂/∂x, dS₁/dm, dS₂/dm
            for_each_order(muj, a.len(), |n, pi, tau| {
                let k = n - 1;
                let pairs = [
                    (a[k], b[k]),
                    (d.da_dx[k], d.db_dx[k]),
                    (d.da_dm[k], d.db_dm[k]),
                ];
                for (i, (u, v)) in pairs.into_iter().enumerate() {
                    s[2 * i] = s[2 * i] + u.scale(pi) + v.scale(tau);
                    s[2 * i + 1] = s[2 * i + 1] + u.scale(tau) + v.scale(pi);
                }
            });
            (values.s1[j], values.s2[j]) = (s[0], s[1]);
            (d_dx.s1[j], d_dx.s2[j]) = (s[2], s[3]);
            (d_dm.s1[j], d_dm.s2[j]) = (s[4], s[5]);
        }
        Ok(())
    }
}
