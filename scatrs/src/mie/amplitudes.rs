//! The amplitude scattering functions S₁ and S₂.

use num_complex::Complex;

use super::{MAX_MIE_ORDER, MieSolution};
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
pub(crate) fn for_each_order<T: Real>(mu: T, n_max: usize, mut f: impl FnMut(usize, T, T)) {
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

/// The angular functions of the amplitudes at fixed cosines of the scattering angle, for
/// evaluating the amplitudes or the phase matrix of many spheres at the same angles: wπₙ(μ) and
/// wτₙ(μ), w = (2n + 1)/(n(n + 1)), for n = 1, …, `n_max` at each cosine.
///
/// [`MieSolution::amplitudes`] evaluates πₙ and τₙ by their recurrences at every call, which
/// costs about as much as the sums over n themselves. A table evaluates them once, and the
/// `*_from_table` methods of [`MieSolution`] read them: the results are bitwise those of the
/// methods that take the cosines. The table holds 2·`n_max` values per cosine (32 bytes per
/// order and cosine for `f64`); `n_max` must be at least the number of terms of every
/// solution it is used with (see [`n_terms`](super::n_terms) to size it before solving).
///
/// ```
/// use num_complex::Complex;
/// use scatrs::mie::{AmplitudeBuffers, AngularTable, Mie, MieOptions, n_terms};
///
/// let mu: Vec<f64> = (0..181).map(|k| f64::from(k).to_radians().cos()).collect();
/// let (spheres, m, opts) = ([1.0, 5.0, 25.0], Complex::new(1.5, 0.01), MieOptions::new());
/// let n_max = n_terms(25.0, m, &opts)?; // the largest sphere has the most terms
/// let table = AngularTable::new(&mu, n_max)?;
/// let mut mie = Mie::<f64>::new();
/// let (mut s1, mut s2) = (vec![Complex::default(); mu.len()], vec![Complex::default(); mu.len()]);
/// for x in spheres {
///     let sphere = mie.compute(x, m, &opts)?;
///     sphere.amplitudes_from_table(&table, AmplitudeBuffers { s1: &mut s1, s2: &mut s2 })?;
/// }
/// # Ok::<(), scatrs::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct AngularTable<T> {
    mu: Vec<T>,
    n_max: usize,
    /// wπₙ(μⱼ) and wτₙ(μⱼ) at (n − 1)·mu.len() + j: every cosine of an order together.
    pi: Vec<T>,
    tau: Vec<T>,
}

impl<T: Real> AngularTable<T> {
    /// The table of orders 1, …, `n_max` at the cosines `mu`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if a cosine is NaN or outside [−1, 1] (for `Dual`, its value),
    /// if `n_max` is not in [1, [`MAX_MIE_ORDER`]], or if the table cannot be allocated.
    pub fn new(mu: &[T], n_max: usize) -> Result<Self> {
        check_cosines(mu)?;
        if !(1..=MAX_MIE_ORDER).contains(&n_max) {
            return Err(Error::InvalidInput(format!(
                "the number of orders must be in [1, {MAX_MIE_ORDER}], got {n_max}"
            )));
        }
        let len = mu.len();
        let size = n_max.checked_mul(len);
        let alloc = || -> Option<Vec<T>> {
            let mut v = Vec::new();
            v.try_reserve_exact(size?).ok()?;
            v.resize(size?, T::zero());
            Some(v)
        };
        let (Some(mut pi_w), Some(mut tau_w)) = (alloc(), alloc()) else {
            return Err(Error::InvalidInput(format!(
                "an angular table of {n_max} orders at {len} cosines is too large to allocate"
            )));
        };
        // The recurrences of `for_each_order`, every cosine at once, in the same operations.
        let mut pi_prev = vec![T::zero(); len];
        let mut pi = vec![T::one(); len];
        for n in 1..=n_max {
            let nf = T::from_f64(n as f64);
            let n1 = nf + T::one();
            let c = T::from_f64((2 * n + 1) as f64);
            let w = c / (nf * n1);
            let row = (n - 1) * len;
            for j in 0..len {
                let (p, pp, m) = (pi[j], pi_prev[j], mu[j]);
                let tau = nf * m * p - n1 * pp;
                pi_w[row + j] = w * p;
                tau_w[row + j] = w * tau;
                pi_prev[j] = p;
                pi[j] = (c * m * p - n1 * pp) / nf;
            }
        }
        Ok(Self {
            mu: mu.to_vec(),
            n_max,
            pi: pi_w,
            tau: tau_w,
        })
    }

    /// The cosines.
    pub fn mu(&self) -> &[T] {
        &self.mu
    }

    /// The number of orders.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// (wπₙ, wτₙ) at every cosine, for n = 1, …, `n` (at most `n_max`; nothing without
    /// cosines).
    fn orders(&self, n: usize) -> impl Iterator<Item = (&[T], &[T])> {
        let len = self.mu.len().max(1);
        self.pi
            .chunks_exact(len)
            .zip(self.tau.chunks_exact(len))
            .take(n)
    }
}

impl<T: Real> MieSolution<'_, T> {
    /// The number of terms, if `table` has as many orders.
    fn table_terms(&self, table: &AngularTable<T>) -> Result<usize> {
        let n = self.n_terms();
        if n > table.n_max {
            return Err(Error::InvalidInput(format!(
                "the solution has {n} terms, more than the {} orders of the angular table",
                table.n_max
            )));
        }
        Ok(n)
    }

    /// S₁ and S₂ at the cosines of `table`, as [`amplitudes`](Self::amplitudes) gives them
    /// (bitwise).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if a buffer's length differs from the number of cosines, or the
    /// solution has more terms than the table has orders. Nothing is written then.
    pub fn amplitudes_from_table(
        &self,
        table: &AngularTable<T>,
        out: AmplitudeBuffers<'_, T>,
    ) -> Result<()> {
        out.check(table.mu.len(), "amplitude")?;
        let n = self.table_terms(table)?;
        let zero = Complex::new(T::zero(), T::zero());
        out.s1.fill(zero);
        out.s2.fill(zero);
        let (a, b) = (&self.state.a, &self.state.b);
        for (k, (pi, tau)) in table.orders(n).enumerate() {
            let (an, bn) = (a[k], b[k]);
            let s = out.s1.iter_mut().zip(out.s2.iter_mut());
            for ((s1, s2), (&p, &t)) in s.zip(pi.iter().zip(tau)) {
                *s1 = *s1 + an.scale(p) + bn.scale(t);
                *s2 = *s2 + an.scale(t) + bn.scale(p);
            }
        }
        Ok(())
    }

    /// S₁, S₂ and their derivatives at the cosines of `table`, as
    /// [`amplitudes_with_derivatives`](Self::amplitudes_with_derivatives) gives them
    /// (bitwise).
    ///
    /// # Errors
    ///
    /// As [`amplitudes_from_table`](Self::amplitudes_from_table), and
    /// [`Error::InvalidInput`] if the solution was computed without derivatives.
    pub fn amplitudes_with_derivatives_from_table(
        &self,
        table: &AngularTable<T>,
        values: AmplitudeBuffers<'_, T>,
        d_dx: AmplitudeBuffers<'_, T>,
        d_dm: AmplitudeBuffers<'_, T>,
    ) -> Result<()> {
        let Some(d) = self.derivatives() else {
            return Err(Error::InvalidInput(
                "amplitude derivatives need a solution from Mie::compute_with_derivatives".into(),
            ));
        };
        let len = table.mu.len();
        values.check(len, "amplitude")?;
        d_dx.check(len, "derivative")?;
        d_dm.check(len, "derivative")?;
        let n = self.table_terms(table)?;
        // S₁, S₂, ∂S₁/∂x, ∂S₂/∂x, dS₁/dm, dS₂/dm.
        let mut s = [values.s1, values.s2, d_dx.s1, d_dx.s2, d_dm.s1, d_dm.s2];
        let zero = Complex::new(T::zero(), T::zero());
        for buffer in &mut s {
            buffer.fill(zero);
        }
        let (a, b) = (&self.state.a, &self.state.b);
        for (k, (pi, tau)) in table.orders(n).enumerate() {
            let pairs = [
                (a[k], b[k]),
                (d.da_dx[k], d.db_dx[k]),
                (d.da_dm[k], d.db_dm[k]),
            ];
            for (j, (&p, &t)) in pi.iter().zip(tau).enumerate() {
                for (i, &(u, v)) in pairs.iter().enumerate() {
                    s[2 * i][j] = s[2 * i][j] + u.scale(p) + v.scale(t);
                    s[2 * i + 1][j] = s[2 * i + 1][j] + u.scale(t) + v.scale(p);
                }
            }
        }
        Ok(())
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
