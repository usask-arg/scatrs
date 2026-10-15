//! Efficiencies, the asymmetry parameter and their Jacobian.

use num_complex::Complex;

use super::{CoefficientDerivatives, Coefficients, Group, MieSolution};
use crate::complex::times_i;
use crate::scalar::Real;

/// Efficiencies (cross sections divided by the geometric cross section πr²) and the
/// asymmetry parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Efficiencies<T> {
    /// Extinction, Q_sca + Q_abs. It equals (2/x²) Σ (2n + 1) Re(aₙ + bₙ) (the optical
    /// theorem), which is not used to compute it: for small non-absorbing spheres
    /// Re aₙ ≈ |aₙ|² ≪ |aₙ| would leave few correct digits.
    pub q_ext: T,
    /// Scattering, (2/x²) Σ (2n + 1)(|aₙ|² + |bₙ|²).
    pub q_sca: T,
    /// Absorption, (2/x²) Σ (2n + 1)(Re aₙ − |aₙ|² + Re bₙ − |bₙ|²), each term from the
    /// identity Re aₙ − |aₙ|² = −Im(Dₙ(mx)/m)/|Δₐ|² (and likewise for bₙ): accurate relative
    /// to itself even when Q_abs ≪ Q_ext, and exactly 0 for real m.
    pub q_abs: T,
    /// Backscattering, |Σ (2n + 1)(−1)ⁿ(aₙ − bₙ)|²/x² = 4|S₁(π)|²/x².
    pub q_back: T,
    /// Asymmetry parameter g = ⟨cos Θ⟩, (4/(x² Q_sca)) Σ [n(n + 2)/(n + 1)
    /// Re(aₙ aₙ₊₁* + bₙ bₙ₊₁*) + (2n + 1)/(n(n + 1)) Re(aₙ bₙ*)]; 0 when Q_sca = 0.
    ///
    /// At m = 1 the coefficients vanish, and for x < 1 they are computed exactly 0, so g = 0.
    /// For x ≥ 1 they are rounding errors (the conditioning κ is infinite there), and g and its
    /// derivatives are meaningless.
    pub g: T,
}

/// The derivatives of each efficiency with respect to x, Re m and Im m.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct EfficiencyJacobian<T> {
    /// ∂/∂x.
    pub x: Efficiencies<T>,
    /// ∂/∂(Re m).
    pub m_re: Efficiencies<T>,
    /// ∂/∂(Im m).
    pub m_im: Efficiencies<T>,
}

/// Re(u v*).
fn re_mul_conj<T: Real>(u: Complex<T>, v: Complex<T>) -> T {
    u.re * v.re + u.im * v.im
}

fn int<T: Real>(k: usize) -> T {
    T::from_f64(k as f64)
}

/// The sums behind Q_sca (`sca`, with Q_sca = 2·sca/x²), Q_back (`back`, with
/// Q_back = |back|²/x²) and g (`g`, with g = 2·g/sca), or their derivatives.
struct Sums<T> {
    sca: T,
    back: Complex<T>,
    g: T,
}

fn value_sums<T: Real>(a: &[Complex<T>], b: &[Complex<T>]) -> Sums<T> {
    let n = a.len();
    let mut sca = T::zero();
    let mut back = Complex::new(T::zero(), T::zero());
    let mut g = T::zero();
    for k in 1..=n {
        let (ak, bk) = (a[k - 1], b[k - 1]);
        let w: T = int(2 * k + 1);
        sca += w * (ak.norm_sqr() + bk.norm_sqr());
        let t = (ak - bk).scale(w);
        back = if k % 2 == 0 { back + t } else { back - t };
        let kf: T = int(k);
        if k < n {
            let (a1, b1) = (a[k], b[k]);
            g += kf * (kf + T::from_f64(2.0)) / (kf + T::one())
                * (re_mul_conj(ak, a1) + re_mul_conj(bk, b1));
        }
        g += w / (kf * (kf + T::one())) * re_mul_conj(ak, bk);
    }
    Sums { sca, back, g }
}

/// The derivatives of the sums of [`value_sums`] with respect to the size parameter, Re m and
/// Im m (∂/∂(Im m) = i·d/dm), in one pass over the coefficients and their derivatives; the
/// same operations, in the same order, as three separate passes would take.
fn derivative_sums<T: Real>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    d: &CoefficientDerivatives<'_, T>,
) -> [Sums<T>; 3] {
    let n = a.len();
    let (b, da_dx, da_dm, db_dx, db_dm) = (
        &b[..n],
        &d.da_dx[..n],
        &d.da_dm[..n],
        &d.db_dx[..n],
        &d.db_dm[..n],
    );
    let two = T::from_f64(2.0);
    let zero = Complex::new(T::zero(), T::zero());
    let mut sca = [T::zero(); 3];
    let mut back = [zero; 3];
    let mut g = [T::zero(); 3];
    // The derivatives at index i, with respect to x, Re m and Im m.
    let derivs = |i: usize| {
        (
            [da_dx[i], da_dm[i], times_i(da_dm[i])],
            [db_dx[i], db_dm[i], times_i(db_dm[i])],
        )
    };
    let mut current = derivs(0);
    for k in 1..=n {
        let (ak, bk) = (a[k - 1], b[k - 1]);
        let (dak, dbk) = current;
        let w: T = int(2 * k + 1);
        let kf: T = int(k);
        let next = (k < n).then(|| derivs(k));
        let pair = kf * (kf + two) / (kf + T::one());
        let own = w / (kf * (kf + T::one()));
        for p in 0..3 {
            sca[p] += w * two * (re_mul_conj(dak[p], ak) + re_mul_conj(dbk[p], bk));
            let t = (dak[p] - dbk[p]).scale(w);
            back[p] = if k % 2 == 0 { back[p] + t } else { back[p] - t };
            if let Some((da1, db1)) = next {
                let (a1, b1) = (a[k], b[k]);
                g[p] += pair
                    * (re_mul_conj(dak[p], a1)
                        + re_mul_conj(ak, da1[p])
                        + re_mul_conj(dbk[p], b1)
                        + re_mul_conj(bk, db1[p]));
            }
            g[p] += own * (re_mul_conj(dak[p], bk) + re_mul_conj(ak, dbk[p]));
        }
        if let Some(nx) = next {
            current = nx;
        }
    }
    [0, 1, 2].map(|p| Sums {
        sca: sca[p],
        back: back[p],
        g: g[p],
    })
}

impl<T: Real> MieSolution<'_, T> {
    /// The efficiencies and the asymmetry parameter.
    pub fn efficiencies(&self) -> Efficiencies<T> {
        self.coefficients().efficiencies()
    }

    /// The derivatives of the efficiencies with respect to x, Re m and Im m, if the solution
    /// was computed by [`Mie::compute_with_derivatives`](super::Mie::compute_with_derivatives).
    pub fn efficiencies_jacobian(&self) -> Option<EfficiencyJacobian<T>> {
        let g = self.group()?;
        Some(self.coefficients().efficiencies_jacobian(&g))
    }
}

impl<T: Real> Coefficients<'_, T> {
    /// The efficiencies and the asymmetry parameter.
    pub(crate) fn efficiencies(&self) -> Efficiencies<T> {
        self.efficiencies_from(&value_sums(self.a, self.b))
    }

    /// The efficiencies from the value sums of [`value_sums`].
    fn efficiencies_from(&self, sums: &Sums<T>) -> Efficiencies<T> {
        let inv_x = self.x.recip();
        let inv_x2 = inv_x * inv_x;
        let two_x2 = T::from_f64(2.0) * inv_x2;
        let q_sca = two_x2 * sums.sca;
        let q_abs = two_x2 * self.abs_sum;
        let g = if sums.sca == T::zero() {
            T::zero()
        } else {
            T::from_f64(2.0) * sums.g / sums.sca
        };
        Efficiencies {
            q_ext: q_sca + q_abs,
            q_sca,
            q_abs,
            q_back: sums.back.norm_sqr() * inv_x2,
            g,
        }
    }

    /// The derivatives of the efficiencies with respect to the size parameter and the real and
    /// imaginary parts of the refractive index of `group` (Q = (2/x²)·sum: only the outer size
    /// parameter's derivative has the term of 2/x²).
    pub(crate) fn efficiencies_jacobian(&self, group: &Group<'_, T>) -> EfficiencyJacobian<T> {
        let (a, b, d) = (self.a, self.b, &group.d);
        let two = T::from_f64(2.0);
        let inv_x = self.x.recip();
        let inv_x2 = inv_x * inv_x;
        let two_x2 = two * inv_x2;
        let values = value_sums(a, b);
        let q = self.efficiencies_from(&values);
        let all = derivative_sums(a, b, d);
        let mut out = [q; 3];
        for (p, (o, sums)) in out.iter_mut().zip(&all).enumerate() {
            // Q = (2/x²)·sum: ∂Q/∂x = (2/x²)·∂sum/∂x − (2/x)·Q.
            let x_term = |v: T| {
                if p == 0 && group.outer {
                    two * inv_x * v
                } else {
                    T::zero()
                }
            };
            let q_sca = two_x2 * sums.sca - x_term(q.q_sca);
            let q_abs = two_x2 * group.abs_sum[p] - x_term(q.q_abs);
            let q_back = two * re_mul_conj(sums.back, values.back) * inv_x2 - x_term(q.q_back);
            // g = 2G/S with S the scattering sum: ∂g = (2∂G − g ∂S)/S (x cancels).
            let g = if values.sca == T::zero() {
                T::zero()
            } else {
                (two * sums.g - q.g * sums.sca) / values.sca
            };
            *o = Efficiencies {
                q_ext: q_sca + q_abs,
                q_sca,
                q_abs,
                q_back,
                g,
            };
        }
        let [x, m_re, m_im] = out;
        EfficiencyJacobian { x, m_re, m_im }
    }
}
