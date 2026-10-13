//! Efficiencies, the asymmetry parameter and their Jacobian.

use num_complex::Complex;

use super::MieSolution;
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

/// The derivatives of the sums of [`value_sums`], given the coefficients' derivatives.
fn derivative_sums<T: Real>(
    a: &[Complex<T>],
    b: &[Complex<T>],
    da: impl Fn(usize) -> Complex<T>,
    db: impl Fn(usize) -> Complex<T>,
) -> Sums<T> {
    let n = a.len();
    let two = T::from_f64(2.0);
    let mut sca = T::zero();
    let mut back = Complex::new(T::zero(), T::zero());
    let mut g = T::zero();
    for k in 1..=n {
        let (ak, bk, dak, dbk) = (a[k - 1], b[k - 1], da(k - 1), db(k - 1));
        let w: T = int(2 * k + 1);
        sca += w * two * (re_mul_conj(dak, ak) + re_mul_conj(dbk, bk));
        let t = (dak - dbk).scale(w);
        back = if k % 2 == 0 { back + t } else { back - t };
        let kf: T = int(k);
        if k < n {
            let (a1, b1, da1, db1) = (a[k], b[k], da(k), db(k));
            g += kf * (kf + two) / (kf + T::one())
                * (re_mul_conj(dak, a1)
                    + re_mul_conj(ak, da1)
                    + re_mul_conj(dbk, b1)
                    + re_mul_conj(bk, db1));
        }
        g += w / (kf * (kf + T::one())) * (re_mul_conj(dak, bk) + re_mul_conj(ak, dbk));
    }
    Sums { sca, back, g }
}

impl<T: Real> MieSolution<'_, T> {
    /// The efficiencies and the asymmetry parameter.
    pub fn efficiencies(&self) -> Efficiencies<T> {
        self.efficiencies_from(&value_sums(&self.state.a, &self.state.b))
    }

    /// The efficiencies from the value sums of [`value_sums`].
    fn efficiencies_from(&self, sums: &Sums<T>) -> Efficiencies<T> {
        let s = self.state;
        let inv_x = s.x.recip();
        let inv_x2 = inv_x * inv_x;
        let two_x2 = T::from_f64(2.0) * inv_x2;
        let q_sca = two_x2 * sums.sca;
        let q_abs = two_x2 * s.abs_sum;
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

    /// The derivatives of the efficiencies with respect to x, Re m and Im m, if the solution
    /// was computed by [`Mie::compute_with_derivatives`](super::Mie::compute_with_derivatives).
    pub fn efficiencies_jacobian(&self) -> Option<EfficiencyJacobian<T>> {
        let s = self.state;
        if !s.derivatives {
            return None;
        }
        let i1 = Complex::new(T::zero(), T::one());
        let two = T::from_f64(2.0);
        let inv_x = s.x.recip();
        let inv_x2 = inv_x * inv_x;
        let two_x2 = two * inv_x2;
        let values = value_sums(&s.a, &s.b);
        let q = self.efficiencies_from(&values);
        let mut out = [q; 3];
        for (p, o) in out.iter_mut().enumerate() {
            let sums = match p {
                0 => derivative_sums(&s.a, &s.b, |k| s.da_dx[k], |k| s.db_dx[k]),
                1 => derivative_sums(&s.a, &s.b, |k| s.da_dm[k], |k| s.db_dm[k]),
                _ => derivative_sums(&s.a, &s.b, |k| i1 * s.da_dm[k], |k| i1 * s.db_dm[k]),
            };
            // Q = (2/x²)·sum: ∂Q/∂x = (2/x²)·∂sum/∂x − (2/x)·Q.
            let x_term = |v: T| if p == 0 { two * inv_x * v } else { T::zero() };
            let q_sca = two_x2 * sums.sca - x_term(q.q_sca);
            let q_abs = two_x2 * s.abs_sum_jacobian[p] - x_term(q.q_abs);
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
        Some(EfficiencyJacobian { x, m_re, m_im })
    }
}
