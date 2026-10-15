//! The phase matrix of a randomly oriented axisymmetric particle from its T-matrix, by the rank
//! decomposition of T (verified against a brute-force average over orientations to 10⁻¹⁵ in
//! research/tmatrix/ranks.py).
//!
//! In the helicity basis W^p = (M + pN)/√2, T^{pσ} = (T¹¹ + σT¹² + pT²¹ + pσT²²)/2, and
//! T^{pσ}_{−m} = T^{−p,−σ}_m. The rank-s components of T^{pσ} between orders n and n' are
//! t⁽ˢ⁾(n, n') = Σ_μ (−1)^{n'−μ} C^{s0}_{nμ;n'−μ} T^{pσ}_{μ}(n, n') (T is diagonal in the mode for
//! an axisymmetric particle); rotating the particle by R multiplies the q-th component by
//! D^s_{q0}(R). For incident helicity σ along ẑ and scattered helicity p at (Θ, φ = 0),
//!
//!   A_{pσ}(Θ; R) = Σ_{s,q} D^s_{q0}(R) Φ_{sq},
//!   Φ_{sq} = Σ_n farₙ(q + σ, p; Θ) w_n,  w_n = Σ_{n'} inc_{n'} (−1)^{n'−σ} C^{sq}_{n,q+σ;n',−σ} t⁽ˢ⁾(n, n'),
//!
//! with inc_{n'} = 2 i^{n'−1} √(π(2n'+1)) the plane wave's helicity coefficients and
//! farₙ(m, p; Θ) = γₙ (−i)ⁿ (π_{mn} + p τ_{mn})(Θ) the far field of W^p_{nm}. Over orientations
//! ⟨D^s_{q0} D^{s'*}_{q'0}⟩ = δ_{ss'} δ_{qq'}/(2s + 1), so ⟨A_{pσ} A*_{p'σ'}⟩ = Σ_{s,q} Φ Φ'*/(2s + 1).
//! The linear amplitude matrix is S = U A U⁻¹, U = [ê₊ ê₋] = [[1, 1], [i, −i]]/√2 in (θ̂, φ̂),
//! with S = [[S₂, S₃], [S₄, S₁]], and the phase matrix follows from ⟨S_{ij} S*_{kl}⟩.
//!
//! Cost: O(N⁴) for N orders and N + L/2 cosines.

use num_complex::Complex;

use super::functions::Angular;
use super::kernel::TMatrixModes;
use crate::mie::kernel::real;
use crate::scalar::Real;
use crate::special::clebsch::clebsch_gordan;

/// The orientation-averaged phase matrix ⟨Z⟩ at cosines μ: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ in units
/// of 1/k² per steradian (∫ F₁₁ dΩ = ⟨C_sca⟩).
#[derive(Clone, Debug)]
pub(crate) struct AveragedPhase<T> {
    pub(crate) f11: Vec<T>,
    pub(crate) f12: Vec<T>,
    pub(crate) f22: Vec<T>,
    pub(crate) f33: Vec<T>,
    pub(crate) f34: Vec<T>,
    pub(crate) f44: Vec<T>,
}

fn sgn<T: Real>(k: i64) -> T {
    if k.rem_euclid(2) == 0 {
        T::one()
    } else {
        -T::one()
    }
}

/// i^k.
fn i_pow<T: Real>(k: i64) -> Complex<T> {
    match k.rem_euclid(4) {
        0 => real(T::one()),
        1 => Complex::new(T::zero(), T::one()),
        2 => real(-T::one()),
        _ => Complex::new(T::zero(), -T::one()),
    }
}

impl<T: Real> TMatrixModes<T> {
    /// T^{ij}_m(n, n') of mode m ≥ 0 as four dense (N+1)×(N+1) blocks (index by order).
    fn dense(&self, m: usize) -> [Vec<Complex<T>>; 4] {
        let n = self.n;
        let size = (n + 1) * (n + 1);
        let mut out: [Vec<Complex<T>>; 4] = core::array::from_fn(|_| vec![real(T::zero()); size]);
        let (ts, index) = &self.modes[m];
        for c in 0..2 {
            let kc = index[c].len();
            for (i, &(ti, ni)) in index[c].iter().enumerate() {
                for (j, &(tj, nj)) in index[c].iter().enumerate() {
                    out[ti * 2 + tj][ni * (n + 1) + nj] = ts[c][i * kc + j];
                }
            }
        }
        out
    }

    /// ⟨Z⟩ at the cosines `mu`.
    pub(crate) fn random_orientation(&self, mu: &[T]) -> AveragedPhase<T> {
        let n = self.n;
        let n1 = n + 1;
        let zero = real(T::zero());
        let half = T::from_f64(0.5);
        // Helicity blocks for μ = −N..N: hel[(p, σ)][μ + N][n * n1 + n'].
        let mut hel: Vec<Vec<Vec<Complex<T>>>> = vec![vec![Vec::new(); 2 * n + 1]; 4];
        for m in 0..=n {
            let [t11, t12, t21, t22] = self.dense(m);
            for (pi, p) in [1i64, -1].into_iter().enumerate() {
                for (si, s) in [1i64, -1].into_iter().enumerate() {
                    let block = |p: i64, s: i64| -> Vec<Complex<T>> {
                        (0..n1 * n1)
                            .map(|k| {
                                (t11[k]
                                    + t12[k].scale(T::from_f64(s as f64))
                                    + t21[k].scale(T::from_f64(p as f64))
                                    + t22[k].scale(T::from_f64((p * s) as f64)))
                                .scale(half)
                            })
                            .collect()
                    };
                    hel[pi * 2 + si][n + m] = block(p, s);
                    if m > 0 {
                        // T^{pσ}_{−m} = T^{−p,−σ}_m.
                        hel[pi * 2 + si][n - m] = block(-p, -s);
                    }
                }
            }
        }
        // Rank components t[(p,σ)][n][n'] for s = |n − n'|, …, n + n'.
        let mut cg = Vec::new();
        let mut t_rank: Vec<Vec<Vec<Complex<T>>>> = vec![vec![Vec::new(); n1 * n1]; 4];
        for nn in 1..=n {
            for np in 1..=n {
                let s_min = nn.abs_diff(np);
                let len = nn + np - s_min + 1;
                let mut acc: [Vec<Complex<T>>; 4] = core::array::from_fn(|_| vec![zero; len]);
                let mu_max = nn.min(np) as i64;
                for mu in -mu_max..=mu_max {
                    let j0 = clebsch_gordan::<T>(nn, mu, np, -mu, &mut cg);
                    let ph: T = sgn((np as i64) - mu);
                    for (k, a) in acc.iter_mut().enumerate() {
                        let tv = hel[k][(n as i64 + mu) as usize][nn * n1 + np];
                        if tv == zero {
                            continue;
                        }
                        for (q, &c) in cg.iter().enumerate() {
                            let s = j0 + q;
                            a[s - s_min] = a[s - s_min] + tv.scale(ph * c);
                        }
                    }
                }
                for k in 0..4 {
                    t_rank[k][nn * n1 + np] = core::mem::take(&mut acc[k]);
                }
            }
        }
        let inc: Vec<Complex<T>> = (0..=n)
            .map(|k| {
                if k == 0 {
                    zero
                } else {
                    i_pow::<T>(k as i64 - 1).scale(
                        T::from_f64(2.0) * (T::PI() * T::from_f64((2 * k + 1) as f64)).sqrt(),
                    )
                }
            })
            .collect();
        let gam: Vec<T> = (0..=n)
            .map(|k| {
                if k == 0 {
                    T::zero()
                } else {
                    (T::from_f64((2 * k + 1) as f64)
                        / (T::PI() * T::from_f64((4 * k * (k + 1)) as f64)))
                    .sqrt()
                }
            })
            .collect();
        let nq = mu.len();
        let s_max = 2 * n;
        // prod[a][b][c][d][j] = ⟨A_{p_a σ_b} A*_{p_c σ_d}⟩ at node j.
        let mut prod = vec![zero; 16 * nq];
        let mut ang = Angular::new();
        // far[p][n][j] for the current m, and Φ[(p,σ)][s][j] for the current q.
        let mut phi: Vec<Vec<Complex<T>>> = vec![vec![zero; (s_max + 1) * nq]; 4];
        let mut active = [false; 4];
        let mut w = vec![zero; (s_max + 1) * n1];
        let qmax = n as i64 + 1;
        for q in -qmax..=qmax {
            for v in &mut phi {
                v.iter_mut().for_each(|x| *x = zero);
            }
            active = [false; 4];
            for (si, sigma) in [1i64, -1].into_iter().enumerate() {
                let m = q + sigma;
                if m.unsigned_abs() as usize > n {
                    continue;
                }
                let am = m.unsigned_abs() as usize;
                // far_n(m, p; Θ_j) = γₙ (−i)ⁿ (π + p τ), the signed-m functions from |m|:
                // d_{−m} = (−1)^m d_m, τ likewise, π_{−m} = (−m) d_{−m}/sinθ.
                let mut far = vec![zero; 2 * n1 * nq];
                for (j, &c) in mu.iter().enumerate() {
                    let st = (T::one() - c * c).sqrt();
                    ang.compute(am, n, c, st);
                    let flip: T = if m < 0 { sgn(am as i64) } else { T::one() };
                    for k in am.max(1)..=n {
                        let tau = ang.tau[k] * flip;
                        let pi_ = if m < 0 {
                            -(ang.pi[k] * flip)
                        } else {
                            ang.pi[k]
                        };
                        let f = i_pow::<T>(-(k as i64)).scale(gam[k]);
                        far[k * nq + j] = f.scale(pi_ + tau);
                        far[(n1 + k) * nq + j] = f.scale(pi_ - tau);
                    }
                }
                for (pi, _p) in [1i64, -1].into_iter().enumerate() {
                    let key = pi * 2 + si;
                    w.iter_mut().for_each(|x| *x = zero);
                    for nn in am.max(1)..=n {
                        for np in 1..=n {
                            let tr = &t_rank[key][nn * n1 + np];
                            if tr.is_empty() {
                                continue;
                            }
                            let s_min = nn.abs_diff(np);
                            let j0 = clebsch_gordan::<T>(nn, m, np, -sigma, &mut cg);
                            let ph: T = sgn(np as i64 - sigma);
                            let pre = inc[np].scale(ph);
                            for (k, &c) in cg.iter().enumerate() {
                                let s = j0 + k;
                                if s < s_min || s - s_min >= tr.len() {
                                    continue;
                                }
                                w[s * n1 + nn] = w[s * n1 + nn] + pre * tr[s - s_min].scale(c);
                            }
                        }
                    }
                    let fr = &far[pi * n1 * nq..(pi + 1) * n1 * nq];
                    for s in 0..=s_max {
                        for nn in am.max(1)..=n {
                            let wv = w[s * n1 + nn];
                            if wv == zero {
                                continue;
                            }
                            for j in 0..nq {
                                phi[key][s * nq + j] = phi[key][s * nq + j] + wv * fr[nn * nq + j];
                            }
                        }
                    }
                    active[key] = true;
                }
            }
            for a in 0..4 {
                if !active[a] {
                    continue;
                }
                for b in 0..4 {
                    if !active[b] {
                        continue;
                    }
                    let (pa, sa) = (a / 2, a % 2);
                    let (pb, sb) = (b / 2, b % 2);
                    let slot = ((pa * 2 + sa) * 2 + pb) * 2 + sb;
                    for s in 0..=s_max {
                        let wt = T::from_f64((2 * s + 1) as f64).recip();
                        for j in 0..nq {
                            let v = phi[a][s * nq + j] * phi[b][s * nq + j].conj();
                            prod[slot * nq + j] = prod[slot * nq + j] + v.scale(wt);
                        }
                    }
                }
            }
        }
        let _ = active;
        mueller(&prod, nq)
    }
}

/// The phase matrix from the helicity products: ⟨S_{ij} S*_{kl}⟩ = Σ U_{ia} U⁻¹_{bj} conj(U_{kc}
/// U⁻¹_{dl}) ⟨A_{ab} A*_{cd}⟩.
fn mueller<T: Real>(prod: &[Complex<T>], nq: usize) -> AveragedPhase<T> {
    let r = T::from_f64(0.5).sqrt();
    let one = real(r);
    let i1 = Complex::new(T::zero(), r);
    let u = [[one, one], [i1, -i1]];
    let ui = [[one, -i1], [one, i1]];
    let mut out = AveragedPhase {
        f11: Vec::with_capacity(nq),
        f12: Vec::with_capacity(nq),
        f22: Vec::with_capacity(nq),
        f33: Vec::with_capacity(nq),
        f34: Vec::with_capacity(nq),
        f44: Vec::with_capacity(nq),
    };
    let half = T::from_f64(0.5);
    for j in 0..nq {
        let ss = |i: usize, jj: usize, k: usize, l: usize| -> Complex<T> {
            let mut acc = real(T::zero());
            for a in 0..2 {
                for b in 0..2 {
                    for c in 0..2 {
                        for d in 0..2 {
                            let slot = ((a * 2 + b) * 2 + c) * 2 + d;
                            let f = u[i][a] * ui[b][jj] * (u[k][c] * ui[d][l]).conj();
                            acc = acc + f * prod[slot * nq + j];
                        }
                    }
                }
            }
            acc
        };
        let s2s2 = ss(0, 0, 0, 0).re;
        let s1s1 = ss(1, 1, 1, 1).re;
        let s3s3 = ss(0, 1, 0, 1).re;
        let s4s4 = ss(1, 0, 1, 0).re;
        let s1s2 = ss(1, 1, 0, 0);
        let s3s4 = ss(0, 1, 1, 0);
        out.f11.push(half * (s1s1 + s2s2 + s3s3 + s4s4));
        out.f12.push(half * (s2s2 - s1s1 + s4s4 - s3s3));
        out.f22.push(half * (s2s2 + s1s1 - s4s4 - s3s3));
        out.f33.push(s1s2.re + s3s4.re);
        out.f34.push(-s1s2.im - s3s4.im);
        out.f44.push(s1s2.re - s3s4.re);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tmatrix::kernel::tmatrix;

    /// research/tmatrix/ranks.py: x = 2, ε = 1.8, s = 1.5 + 0.02i, N = 6, 30 nodes over [−1, 1]
    /// (15 here), at Θ = 0.4: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ (brute-force average, 10⁻¹⁵).
    #[test]
    fn matches_the_brute_force_average() {
        let t = tmatrix(2.0, 1.8, Complex::new(1.5, 0.02), 6, 15).unwrap();
        let mu = [0.4f64.cos()];
        let f = t.random_orientation(&mu);
        let got = [f.f11[0], f.f12[0], f.f22[0], f.f33[0], f.f34[0], f.f44[0]];
        let want = [6.980506, -0.33857, 6.912394, 6.902738, 0.118319, 6.84077];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 2e-6, "{got:?} vs {want:?}");
        }
    }

    /// Aspect ratio 1: the average is the sphere's phase matrix, (4/x²)⟨Z⟩ = Mie's unnormalized
    /// one, with F₂₂ = F₁₁ and F₄₄ = F₃₃.
    #[test]
    fn a_sphere_gives_the_mie_phase_matrix() {
        use crate::mie::{Mie, MieOptions, Normalization, PhaseBuffers};
        let (x, m, n) = (3.0, Complex::new(1.5, 0.01), 14);
        let t = tmatrix(x, 1.0, m, n, 30).unwrap();
        let mu: Vec<f64> = (0..7).map(|k| -0.95 + 0.3 * f64::from(k)).collect();
        let f = t.random_orientation(&mu);
        let mut mie = Mie::<f64>::new();
        let opts = MieOptions::new().with_n_max(n).with_allow_unvalidated(true);
        let s = mie.compute(x, m, &opts).unwrap();
        let (mut f11, mut f12, mut f33, mut f34) =
            (vec![0.0; 7], vec![0.0; 7], vec![0.0; 7], vec![0.0; 7]);
        s.phase_matrix(
            &mu,
            Normalization::Unnormalized,
            PhaseBuffers {
                f11: &mut f11,
                f12: &mut f12,
                f33: &mut f33,
                f34: &mut f34,
            },
        )
        .unwrap();
        let k = 4.0 / (x * x);
        for j in 0..7 {
            let scale = f11[j].abs();
            for (got, want) in [
                (k * f.f11[j], f11[j]),
                (k * f.f22[j], f11[j]),
                (k * f.f12[j], f12[j]),
                (k * f.f33[j], f33[j]),
                (k * f.f44[j], f33[j]),
                (k * f.f34[j], f34[j]),
            ] {
                assert!(
                    (got - want).abs() < 1e-12 * scale,
                    "μ {}: {got} vs {want}",
                    mu[j]
                );
            }
        }
    }

    /// 2π ∫ F₁₁ dμ = ⟨C_sca⟩ from T, for a spheroid (Gauss–Legendre, exact for the polynomial).
    #[test]
    fn the_phase_function_integrates_to_the_scattering_cross_section() {
        let n = 10;
        let t = tmatrix(3.0, 0.5, Complex::new(1.5, 0.02), n, 30).unwrap();
        let rule = crate::special::gauss_legendre::<f64>(2 * n + 2).unwrap();
        let f = t.random_orientation(&rule.nodes);
        let integral: f64 = f
            .f11
            .iter()
            .zip(&rule.weights)
            .map(|(v, w)| v * w)
            .sum::<f64>()
            * 2.0
            * core::f64::consts::PI;
        let (_, sca) = t.cross_sections();
        assert!((integral - sca).abs() < 1e-12 * sca, "{integral} vs {sca}");
    }
}
