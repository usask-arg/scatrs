//! The EBCM for a spheroid: per azimuthal mode m, the matrices Q and P = RgQ from surface
//! integrals of the vector spherical wave functions, and Tᵐ = −P Q⁻¹ (Mishchenko, Travis & Lacis
//! 2002, §5.8; Waterman 1971).
//!
//! Every element is a value of the reciprocity form B(X, Y) = ∫_S n̂·(X × ∇×Y − Y × ∇×X):
//! Q_{nk} = −i B(Xₙ, Y_k) with X an outgoing wave of the medium and Y a regular wave of the
//! particle (P: X regular). With k = 1, x = r(θ), z = s x, ν = −r'/r and the components
//!
//! Mₙ = γₙ (0, i hₙ pₙ, −hₙ τₙ),  Nₙ = γₙ (n(n+1) hₙ/x dₙ, Dhₙ τₙ, i Dhₙ pₙ),
//!
//! (pₙ = −πₙ for the outgoing wave, the (−1)^m X_{−mn} of the integrals; Dh = [x h]'/x), the
//! element of row type a and column type b is −i 2π γₙ γ_k Σ w x² [T(A₁, B₁) + T(A₂, B₂)], with
//! T(A, B) = A_θ B_φ − A_φ B_θ + ν (A_φ B_r − A_r B_φ), rows (A₁, A₂) = (Mₙ, Nₙ) or (Nₙ, Mₙ) and
//! columns (B₁, B₂) = (s N_k, M_k) or (s M_k, N_k). A sphere gives T¹¹ = −bₙ, T²² = −aₙ.
//!
//! A spheroid is mirror-symmetric: the diagonal blocks vanish unless n + k is even and the
//! off-diagonal ones unless it is odd, so each mode splits into two systems by the class
//! (n + type) mod 2, and the integrands of the non-zero elements are even in cos θ: the
//! quadrature runs over 0 < θ < π/2 and doubles.

use num_complex::Complex;

use super::functions::{Angular, Radial, Scales};
use super::regular::{Series, angular_factors, l1};
use super::solve::{
    F64Solve, derivative_f64, error_estimate, refine, refine_derivative, solve_f64, solve_t,
};
use crate::Result;
use crate::mie::kernel::{int, real};
use crate::scalar::DoubleDouble as Dd;
use crate::scalar::Dual;
use crate::scalar::Real;
use crate::special::gauss_legendre;

/// A spheroid: semi-axes a (equatorial) and c (polar) in units of 1/k, from the equal-volume
/// size parameter x and the aspect ratio ε = a/c.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Spheroid<T> {
    pub(crate) a: T,
    pub(crate) c: T,
}

impl<T: Real> Spheroid<T> {
    pub(crate) fn new(x: T, eps: T) -> Self {
        let a = x * eps.cbrt();
        Self { a, c: a / eps }
    }

    /// r(θ) and ν = −r'/r at cos θ = `ct`, sin θ = `st`.
    fn at(&self, ct: T, st: T) -> (T, T) {
        let (a2, c2) = (self.a * self.a, self.c * self.c);
        let den = c2 * st * st + a2 * ct * ct;
        let r = self.a * self.c / den.sqrt();
        // r'/r = −(c² − a²) sin θ cos θ / den.
        let nu = (c2 - a2) * st * ct / den;
        (r, nu)
    }

    fn r_min(&self) -> T {
        self.a.min(self.c)
    }

    fn r_max(&self) -> T {
        self.a.max(self.c)
    }
}

/// One mode's matrices, scaled (row n by σₙ, column k by c_k; see `super::functions`), in the
/// two parity classes: `q[c]`, `p[c]` are K_c × K_c, row-major, indexed by `index[c]`.
pub(crate) struct ModeMatrices<T> {
    pub(crate) q: [Vec<Complex<T>>; 2],
    pub(crate) p: [Vec<Complex<T>>; 2],
    /// A bound on the rounding of each element of Q: the magnitudes it was summed from (Y's in the
    /// kept form, and |P|), to be multiplied by the unit roundoff.
    pub(crate) q_abs: [Vec<T>; 2],
    /// The same for P.
    pub(crate) p_abs: [Vec<T>; 2],
    /// (type, order) of each index of a class.
    pub(crate) index: [Vec<(usize, usize)>; 2],
}

/// γₙ = √((2n + 1)/(4π n(n + 1))).
fn gamma<T: Real>(n: usize) -> T {
    (int::<T>(2 * n + 1) / (T::PI() * int::<T>(4 * n * (n + 1)))).sqrt()
}

type Vec3<T> = [Complex<T>; 3];

/// n̂·(A × B) for n̂ ∝ (1, ν, 0).
fn triple<T: Real>(nu: T, a: &Vec3<T>, b: &Vec3<T>) -> Complex<T> {
    a[1] * b[2] - a[2] * b[1] + (a[2] * b[0] - a[0] * b[2]).scale(nu)
}

/// Q and P of every mode m = 0, …, n, on n_theta nodes in 0 < θ < π/2, nodes outside and modes
/// inside so that each node's radial functions and regularized products serve every mode. With
/// `regularize`, each element of Q keeps the cleanest of its three exact forms (`super::regular`).
pub(crate) fn all_modes<T: Real>(
    shape: &Spheroid<T>,
    s: Complex<T>,
    n: usize,
    n_theta: usize,
    regularize: bool,
    rows: &[Option<usize>],
) -> Result<Vec<ModeMatrices<T>>> {
    // The lowest row order integrated in any mode (rows below are left zero).
    let low = rows.iter().flatten().copied().min().unwrap_or(n + 1);
    let zero = real(T::zero());
    let mut modes: Vec<ModeMatrices<T>> = Vec::with_capacity(n + 1);
    // Per mode and class: Y's three forms and their magnitude sums.
    let mut yv: Vec<[[Vec<Complex<T>>; 3]; 2]> = Vec::with_capacity(n + 1);
    let mut ya: Vec<[[Vec<T>; 3]; 2]> = Vec::with_capacity(n + 1);
    for m in 0..=n {
        let n0 = m.max(1);
        let mut index: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
        for t in 0..2 {
            for k in n0..=n {
                index[(k + t) % 2].push((t, k));
            }
        }
        let size = [index[0].len(), index[1].len()];
        let mk = |c: usize| vec![zero; size[c] * size[c]];
        yv.push(core::array::from_fn(|c| core::array::from_fn(|_| mk(c))));
        ya.push(core::array::from_fn(|c| {
            core::array::from_fn(|_| vec![T::zero(); size[c] * size[c]])
        }));
        modes.push(ModeMatrices {
            q: core::array::from_fn(mk),
            p: core::array::from_fn(mk),
            q_abs: core::array::from_fn(|c| vec![T::zero(); size[c] * size[c]]),
            p_abs: core::array::from_fn(|c| vec![T::zero(); size[c] * size[c]]),
            index,
        });
    }
    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
    let series = Series::new(shape.r_min(), shape.r_max(), s, n, &scales);
    let rule = gauss_legendre::<T>(2 * n_theta)?;
    let mut radial = Radial::new();
    let mut ang = Angular::new();
    let gam: Vec<T> = (0..=n)
        .map(|k| if k == 0 { T::zero() } else { gamma(k) })
        .collect();
    let i1 = Complex::new(T::zero(), T::one());
    let zero3 = [zero; 3];
    let mut reg_m = vec![zero3; n + 1];
    let mut reg_n = vec![zero3; n + 1];
    let mut in_m = vec![zero3; n + 1];
    let mut in_n = vec![zero3; n + 1];
    let mut in_ms = vec![zero3; n + 1];
    let mut in_ns = vec![zero3; n + 1];
    let n1 = n + 1;
    // forms[(t * n1 + n) * n1 + k] at the node.
    let mut forms = Vec::new();
    let mut full = vec![zero; 8 * n1 * n1];
    let (a2, c2) = (shape.a * shape.a, shape.c * shape.c);
    let beta = c2.recip() - a2.recip();
    for (&t, &w) in rule.nodes.iter().zip(&rule.weights) {
        if t <= T::zero() {
            continue;
        }
        let (ct, st) = (t, (T::one() - t * t).sqrt());
        let (r, nu) = shape.at(ct, st);
        radial.compute(r, s, n, &scales);
        let r2 = r * r;
        let x_e = [r2, r2 * r2];
        // The y-part radial factors (scaled yₙ = Im of the scaled hₙ for real x) and the
        // particle's, per kind: plain, [x f]'/x, f/x.
        let yk = |kind: usize, nn: usize| match kind {
            0 => radial.h[nn].im,
            1 => radial.dh[nn].im,
            _ => radial.h_x[nn].im,
        };
        let jk = |kind: usize, kk: usize| match kind {
            0 => radial.jz[kk],
            1 => radial.djz[kk],
            _ => radial.jz_z[kk],
        };
        const KINDS: [(usize, usize, usize); 8] = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 2, 1),
            (1, 2, 1),
            (2, 0, 1),
            (2, 1, 1),
        ];
        for (ti, &(yki, jki, ei)) in KINDS.iter().enumerate() {
            for nn in 1..=n {
                for kk in 1..=n {
                    full[(ti * n1 + nn) * n1 + kk] = jk(jki, kk).scale(yk(yki, nn) * x_e[ei]);
                }
            }
        }
        if regularize {
            let ns = series.at(r / shape.r_min());
            forms.clear();
            for ti in 0..8 {
                for nn in 0..=n {
                    for kk in 0..=n {
                        if nn == 0 || kk == 0 || nn == kk || nn < low {
                            forms.push(None);
                            continue;
                        }
                        let f = full[(ti * n1 + nn) * n1 + kk];
                        forms.push(Some(ns.forms(ti, nn, kk, f, x_e)));
                    }
                }
            }
        }
        // 2 for the two halves of the range, 2π for φ.
        let g0 = w * T::PI() * T::from_f64(4.0);
        let fac = g0 * r2;
        let nu_prime = -(beta * st * ct);
        for (m, mm) in modes.iter_mut().enumerate() {
            let Some(from) = rows[m] else { continue };
            let n0 = m.max(1);
            ang.compute(m, n, ct, st);
            for k in n0..=n {
                let (d, pi, tau) = (ang.d[k], ang.pi[k], ang.tau[k]);
                let kk = int::<T>(k * (k + 1));
                let field = |h: Complex<T>, dh: Complex<T>, h_x: Complex<T>, p: Complex<T>| {
                    (
                        [real(T::zero()), i1 * h * p, -(h.scale(tau))],
                        [h_x.scale(kk * d), dh.scale(tau), i1 * dh * p],
                    )
                };
                (reg_m[k], reg_n[k]) = field(radial.j[k], radial.dj[k], radial.j_x[k], real(-pi));
                (in_m[k], in_n[k]) = field(radial.jz[k], radial.djz[k], radial.jz_z[k], real(pi));
                in_ms[k] = in_m[k].map(|v| v * s);
                in_ns[k] = in_n[k].map(|v| v * s);
            }
            for c in 0..2 {
                let idx = &mm.index[c];
                let kc = idx.len();
                for (i, &(rt, nn)) in idx.iter().enumerate() {
                    if nn < from {
                        continue;
                    }
                    let (g1, g2) = if rt == 0 {
                        (&reg_m[nn], &reg_n[nn])
                    } else {
                        (&reg_n[nn], &reg_m[nn])
                    };
                    let an = (ang.d[nn], ang.pi[nn], ang.tau[nn]);
                    for (j, &(ctp, kk_)) in idx.iter().enumerate() {
                        let (b1, b2) = if ctp == 0 {
                            (&in_ns[kk_], &in_m[kk_])
                        } else {
                            (&in_ms[kk_], &in_n[kk_])
                        };
                        let gg = gam[nn] * gam[kk_];
                        let pv = triple(nu, g1, b1) + triple(nu, g2, b2);
                        let slot = i * kc + j;
                        mm.p[c][slot] = mm.p[c][slot] + Complex::new(pv.im, -pv.re).scale(fac * gg);
                        mm.p_abs[c][slot] += l1(pv) * (fac * gg).abs();
                        // Y: Σ_t a_t R_t in its three forms.
                        let ak = (ang.d[kk_], ang.pi[kk_], ang.tau[kk_]);
                        let a = angular_factors(rt, ctp, s, nu_prime, nn, kk_, an, ak);
                        let ge = g0 * gg;
                        let (mut va, mut aa) = (zero, T::zero());
                        let (mut vb, mut ab) = (zero, T::zero());
                        let (mut vc, mut ac) = (zero, T::zero());
                        for &(ti, at) in &a {
                            let fi = (ti * n1 + nn) * n1 + kk_;
                            let rf = full[fi];
                            let mag = l1(at);
                            va = va + at * rf;
                            aa += mag * l1(rf);
                            if let Some(Some(f)) = forms.get(fi) {
                                vc = vc + at * f.le0;
                                ac += mag * f.le0_abs;
                                vb = vb + at * f.lt0;
                                ab += mag * f.lt0_abs;
                            }
                        }
                        let ys = &mut yv[m][c];
                        let ys_abs = &mut ya[m][c];
                        ys[0][slot] = ys[0][slot] + va.scale(ge);
                        ys_abs[0][slot] += aa * ge.abs();
                        ys[1][slot] = ys[1][slot] + vb.scale(ge);
                        ys_abs[1][slot] += ab * ge.abs();
                        ys[2][slot] = ys[2][slot] + vc.scale(ge);
                        ys_abs[2][slot] += ac * ge.abs();
                    }
                }
            }
        }
    }
    // Q = P + Y in the cleanest exact form of each element: the plain one, minus the powers < 0
    // (|n − k| ≥ 3), or minus the powers ≤ 0 (n ≠ k).
    for (m, mm) in modes.iter_mut().enumerate() {
        for c in 0..2 {
            let idx = mm.index[c].clone();
            let kc = idx.len();
            for (i, &(_, nn)) in idx.iter().enumerate() {
                for (j, &(_, kk_)) in idx.iter().enumerate() {
                    let slot = i * kc + j;
                    let mut best = 0;
                    if regularize && nn != kk_ {
                        if ya[m][c][2][slot] < ya[m][c][best][slot] {
                            best = 2;
                        }
                        if nn.abs_diff(kk_) >= 3 && ya[m][c][1][slot] < ya[m][c][best][slot] {
                            best = 1;
                        }
                    }
                    mm.q[c][slot] = mm.p[c][slot] + yv[m][c][best][slot];
                    mm.q_abs[c][slot] = ya[m][c][best][slot] + mm.p_abs[c][slot];
                }
            }
        }
    }
    Ok(modes)
}

/// The unscaling factor of T's element (row order n, column order k): σ_k/σₙ =
/// |hₙ(x_min)|/|h_k(x_min)|, from the cumulative logarithms of the scale ratios.
pub(crate) fn log_h(scales: &Scales<impl Real>) -> Vec<f64> {
    let mut out = Vec::with_capacity(scales.h_ratio.len());
    let mut acc = 0.0;
    for v in &scales.h_ratio {
        acc += v.approx_f64().ln();
        out.push(acc);
    }
    out
}

/// One mode's T: the two parity classes' matrices (unscaled, row-major) and index sets.
pub(crate) type ModeT<T> = ([Vec<Complex<T>>; 2], [Vec<(usize, usize)>; 2]);

/// The T-matrix of a spheroid, mode by mode: for each m = 0, …, N the two classes' T (unscaled)
/// with their index sets.
#[derive(Clone, Debug)]
pub(crate) struct TMatrixModes<T> {
    pub(crate) n: usize,
    pub(crate) modes: Vec<ModeT<T>>,
}

/// T of every mode for orders up to n and n_theta nodes in 0 < θ < π/2, with or without the
/// regularized integrals.
pub(crate) fn tmatrix<T: Real>(
    x: T,
    eps: T,
    s: Complex<T>,
    n: usize,
    n_theta: usize,
    regularize: bool,
) -> Result<TMatrixModes<T>> {
    let shape = Spheroid::new(x, eps);
    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
    let lh = log_h(&scales);
    let all = all_modes(&shape, s, n, n_theta, regularize, &vec![Some(0); n + 1])?;
    let mut modes = Vec::with_capacity(n + 1);
    for mm in all {
        let mut ts: [Vec<Complex<T>>; 2] = [Vec::new(), Vec::new()];
        for ((slot, index), (q, p)) in ts.iter_mut().zip(&mm.index).zip(mm.q.iter().zip(&mm.p)) {
            let kc = index.len();
            if kc == 0 {
                continue;
            }
            let mut t = solve_t(q, p, kc)?;
            unscale(&mut t, index, &lh);
            *slot = t;
        }
        modes.push((ts, mm.index));
    }
    Ok(TMatrixModes { n, modes })
}

/// Undoes the scaling of one class's T: T = Σ⁻¹ T' Σ, T_{nk} = T'_{nk} σ_k/σₙ =
/// T' |hₙ|/|h_k| (x_min).
fn unscale<T: Real>(t: &mut [Complex<T>], index: &[(usize, usize)], lh: &[f64]) {
    let kc = index.len();
    for (i, &(_, nn)) in index.iter().enumerate() {
        for (j, &(_, kk_)) in index.iter().enumerate() {
            let f = T::from_f64((lh[nn] - lh[kk_]).exp());
            t[i * kc + j] = t[i * kc + j].scale(f);
        }
    }
}

/// How each mode's classes were solved: the estimated largest error of T (unscaled, absolute)
/// and the refinement sweeps (0: plain f64).
#[derive(Clone, Debug)]
pub(crate) struct SolveReport {
    pub(crate) error: Vec<[f64; 2]>,
    pub(crate) sweeps: Vec<[usize; 2]>,
}

/// The lowest row order integrated in double-double for refinement in mode m: the upper three
/// quarters of the orders (findings 5 and 7 of the study: the f64 floor lives there).
fn refine_from(m: usize, n: usize) -> usize {
    let n0 = m.max(1);
    n0 + (n + 1 - n0) / 4
}

/// T of the modes m ≤ `m_max` (the others are left empty) in f64, refined (mixed precision) in
/// the modes whose error estimate exceeds
/// `tol` times the largest |T|: their rows of order ≥ [`refine_from`] are integrated in
/// double-double, and T is corrected with double-double residuals.
pub(crate) fn tmatrix_refined(
    x: f64,
    eps: f64,
    s: Complex<f64>,
    n: usize,
    n_theta: usize,
    tol: f64,
    m_max: usize,
) -> Result<(TMatrixModes<f64>, SolveReport)> {
    // The elements' rounding: u Σ|terms| times 2N for the accuracy of the function values (the
    // largest true error measured is about N u Σ|terms|).
    let u = f64::EPSILON / 2.0 * (2 * n) as f64;
    let u_dd = u * f64::EPSILON / 2.0;
    let shape = Spheroid::new(x, eps);
    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
    let lh = log_h(&scales);
    let modes_in: Vec<Option<usize>> = (0..=n).map(|m| (m <= m_max).then_some(0)).collect();
    let all = all_modes(&shape, s, n, n_theta, true, &modes_in)?;
    let mut report = SolveReport {
        error: vec![[0.0; 2]; n + 1],
        sweeps: vec![[0; 2]; n + 1],
    };
    let mut ts: Vec<[Vec<Complex<f64>>; 2]> = Vec::with_capacity(n + 1);
    let mut big = 0.0f64;
    for (m, mm) in all.iter().enumerate() {
        let mut pair: [Vec<Complex<f64>>; 2] = [Vec::new(), Vec::new()];
        for (c, slot) in pair.iter_mut().enumerate() {
            let index = &mm.index[c];
            let kc = index.len();
            if kc == 0 || m > m_max {
                continue;
            }
            let f = solve_f64(&mm.q[c], &mm.p[c], kc)?;
            let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
            let eq: Vec<f64> = mm.q_abs[c].iter().map(|&v| u * v).collect();
            let ep: Vec<f64> = mm.p_abs[c].iter().map(|&v| u * v).collect();
            report.error[m][c] = error_estimate(&f, &f.t, &eq, &ep, &lh_c, kc);
            let mut t = f.t;
            unscale(&mut t, index, &lh);
            big = t.iter().fold(big, |a, v| a.max(v.norm()));
            *slot = t;
        }
        ts.push(pair);
    }
    let rows: Vec<Option<usize>> = (0..=n)
        .map(|m| {
            let worst = report.error[m][0].max(report.error[m][1]);
            (m <= m_max && worst > tol * big).then(|| refine_from(m, n))
        })
        .collect();
    if rows.iter().any(Option::is_some) {
        let shape_dd = Spheroid::new(Dd::from(x), Dd::from(eps));
        let s_dd = Complex::new(Dd::from(s.re), Dd::from(s.im));
        let exact = all_modes(&shape_dd, s_dd, n, n_theta, true, &rows)?;
        for (m, from) in rows.iter().enumerate() {
            let Some(from) = *from else { continue };
            let mm = &all[m];
            for (c, slot) in ts[m].iter_mut().enumerate() {
                let index = &mm.index[c];
                let kc = index.len();
                if kc == 0 {
                    continue;
                }
                let flags: Vec<bool> = index.iter().map(|&(_, nn)| nn >= from).collect();
                let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
                let refined = refine(
                    &mm.q[c],
                    &mm.p[c],
                    &exact[m].q[c],
                    &exact[m].p[c],
                    &flags,
                    &lh_c,
                    kc,
                    4,
                )?;
                let (mut t, solved, sweeps, last) =
                    (refined.t, &refined.solved, refined.sweeps, refined.last);
                // The f64 rows' rounding remains, beside the double-double rows' and the last
                // correction.
                let bound = |v: &[f64]| -> Vec<f64> {
                    v.iter()
                        .enumerate()
                        .map(|(ij, &a)| a * if flags[ij / kc] { u_dd } else { u })
                        .collect()
                };
                let est = error_estimate(
                    solved,
                    &t,
                    &bound(&mm.q_abs[c]),
                    &bound(&mm.p_abs[c]),
                    &lh_c,
                    kc,
                );
                unscale(&mut t, index, &lh);
                // T is returned in f64: its own rounding.
                let rounding = t.iter().fold(0.0f64, |a, v| a.max(v.norm())) * f64::EPSILON;
                report.sweeps[m][c] = sweeps;
                report.error[m][c] = est + last + rounding;
                *slot = t;
            }
        }
    }
    let modes = ts
        .into_iter()
        .zip(all)
        .map(|(t, mm)| (t, mm.index))
        .collect();
    Ok((TMatrixModes { n, modes }, report))
}

/// One class's T' and its four derivatives (scaled), with the factors of its solve.
type ClassParts = ([Vec<Complex<f64>>; 5], F64Solve);

/// One direction's dual numbers: x, ε, Re m, Im m.
pub(crate) type D4 = Dual<f64, 4>;

/// The cumulative logarithms of the scale ratios, as [`log_h`], in `T` (with derivatives).
fn log_h_t<T: Real>(scales: &Scales<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(scales.h_ratio.len());
    let mut acc = T::zero();
    for &v in &scales.h_ratio {
        acc += v.ln();
        out.push(acc);
    }
    out
}

/// As [`tmatrix_refined`] for all modes, with the derivatives of T with respect to x, ε, Re m
/// and Im m (the dual parts, in that order). The elements are computed in `Dual<f64, 4>`, whose
/// values are bitwise those of `f64`; the derivatives follow from dT' = −(dP + T' dQ) Q⁻¹ with
/// the same factors, and in the refined modes from double-double residuals as T' is refined.
pub(crate) fn tmatrix_derivatives(
    x: f64,
    eps: f64,
    s: Complex<f64>,
    n: usize,
    n_theta: usize,
    tol: f64,
) -> Result<(TMatrixModes<D4>, SolveReport)> {
    let u = f64::EPSILON / 2.0 * (2 * n) as f64;
    let u_dd = u * f64::EPSILON / 2.0;
    let [xd, ed, sr, si] = D4::variables([x, eps, s.re, s.im]);
    let sd = Complex::new(sr, si);
    let shape = Spheroid::new(xd, ed);
    let scales = Scales::new(shape.r_min(), sd.scale(shape.r_max()), n);
    let lh_d = log_h_t(&scales);
    let lh: Vec<f64> = lh_d.iter().map(|v| v.re()).collect();
    let all = all_modes(&shape, sd, n, n_theta, true, &vec![Some(0); n + 1])?;
    let value = |v: &[Complex<D4>]| -> Vec<Complex<f64>> {
        v.iter()
            .map(|c| Complex::new(c.re.re(), c.im.re()))
            .collect()
    };
    let part = |v: &[Complex<D4>], p: usize| -> Vec<Complex<f64>> {
        v.iter()
            .map(|c| Complex::new(c.re.eps()[p], c.im.eps()[p]))
            .collect()
    };
    let mut report = SolveReport {
        error: vec![[0.0; 2]; n + 1],
        sweeps: vec![[0; 2]; n + 1],
    };
    // Per mode and class: T' and its four derivatives (scaled), and the factors.
    let mut ts: Vec<[Option<ClassParts>; 2]> = Vec::with_capacity(n + 1);
    let mut big = 0.0f64;
    for (m, mm) in all.iter().enumerate() {
        let mut pair: [Option<ClassParts>; 2] = [None, None];
        for (c, slot) in pair.iter_mut().enumerate() {
            let index = &mm.index[c];
            let kc = index.len();
            if kc == 0 {
                continue;
            }
            let (qv, pv) = (value(&mm.q[c]), value(&mm.p[c]));
            let f = solve_f64(&qv, &pv, kc)?;
            let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
            let eq: Vec<f64> = mm.q_abs[c].iter().map(|v| u * v.re()).collect();
            let ep: Vec<f64> = mm.p_abs[c].iter().map(|v| u * v.re()).collect();
            report.error[m][c] = error_estimate(&f, &f.t, &eq, &ep, &lh_c, kc);
            for (i, &(_, a)) in index.iter().enumerate() {
                for (j, &(_, b)) in index.iter().enumerate() {
                    big = big.max(f.t[i * kc + j].norm() * (lh[a] - lh[b]).exp());
                }
            }
            let d: [Vec<Complex<f64>>; 4] = core::array::from_fn(|p| {
                derivative_f64(&f, &f.t, &part(&mm.q[c], p), &part(&mm.p[c], p), kc)
            });
            let [d0, d1, d2, d3] = d;
            *slot = Some(([f.t.clone(), d0, d1, d2, d3], f));
        }
        ts.push(pair);
    }
    let rows: Vec<Option<usize>> = (0..=n)
        .map(|m| {
            let worst = report.error[m][0].max(report.error[m][1]);
            (worst > tol * big).then(|| refine_from(m, n))
        })
        .collect();
    if rows.iter().any(Option::is_some) {
        type Dd4 = Dual<Dd, 4>;
        let [xd, ed, sr, si] =
            Dd4::variables([Dd::from(x), Dd::from(eps), Dd::from(s.re), Dd::from(s.im)]);
        let exact = all_modes(
            &Spheroid::new(xd, ed),
            Complex::new(sr, si),
            n,
            n_theta,
            true,
            &rows,
        )?;
        let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
        for (m, from) in rows.iter().enumerate() {
            let Some(from) = *from else { continue };
            let mm = &all[m];
            for (c, slot) in ts[m].iter_mut().enumerate() {
                let index = &mm.index[c];
                let kc = index.len();
                let Some((parts, _)) = slot else {
                    continue;
                };
                let flags: Vec<bool> = index.iter().map(|&(_, nn)| nn >= from).collect();
                let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
                let ex_val = |v: &[Complex<Dd4>]| -> Vec<Complex<Dd>> {
                    v.iter()
                        .map(|c| Complex::new(c.re.re(), c.im.re()))
                        .collect()
                };
                let ex_part = |v: &[Complex<Dd4>], p: usize| -> Vec<Complex<Dd>> {
                    v.iter()
                        .map(|c| Complex::new(c.re.eps()[p], c.im.eps()[p]))
                        .collect()
                };
                let refined = refine(
                    &value(&mm.q[c]),
                    &value(&mm.p[c]),
                    &ex_val(&exact[m].q[c]),
                    &ex_val(&exact[m].p[c]),
                    &flags,
                    &lh_c,
                    kc,
                    4,
                )?;
                let mixed =
                    |exact_v: Vec<Complex<Dd>>, f64_v: Vec<Complex<f64>>| -> Vec<Complex<Dd>> {
                        (0..kc * kc)
                            .map(|ij| {
                                if flags[ij / kc] {
                                    exact_v[ij]
                                } else {
                                    dd(f64_v[ij])
                                }
                            })
                            .collect()
                    };
                for p in 0..4 {
                    let dq = mixed(ex_part(&exact[m].q[c], p), part(&mm.q[c], p));
                    let dp = mixed(ex_part(&exact[m].p[c], p), part(&mm.p[c], p));
                    parts[p + 1] = refine_derivative(&refined, &dq, &dp, kc, 3);
                }
                let bound = |v: &[D4]| -> Vec<f64> {
                    v.iter()
                        .enumerate()
                        .map(|(ij, a)| a.re() * if flags[ij / kc] { u_dd } else { u })
                        .collect()
                };
                let est = error_estimate(
                    &refined.solved,
                    &refined.t,
                    &bound(&mm.q_abs[c]),
                    &bound(&mm.p_abs[c]),
                    &lh_c,
                    kc,
                );
                let largest = refined
                    .t
                    .iter()
                    .zip(
                        index
                            .iter()
                            .flat_map(|&(_, a)| index.iter().map(move |&(_, b)| (a, b))),
                    )
                    .fold(0.0f64, |acc, (v, (a, b))| {
                        acc.max(v.norm() * (lh[a] - lh[b]).exp())
                    });
                report.sweeps[m][c] = refined.sweeps;
                report.error[m][c] = est + refined.last + largest * f64::EPSILON;
                parts[0] = refined.t;
            }
        }
    }
    // Assemble the dual T' and unscale it with the dual factors σ_k/σₙ.
    let mut modes = Vec::with_capacity(n + 1);
    for (pair, mm) in ts.into_iter().zip(all) {
        let mut out: [Vec<Complex<D4>>; 2] = [Vec::new(), Vec::new()];
        for (c, slot) in pair.into_iter().enumerate() {
            let Some((parts, _)) = slot else { continue };
            let index = &mm.index[c];
            let kc = index.len();
            let mut t: Vec<Complex<D4>> = (0..kc * kc)
                .map(|ij| {
                    let re = D4::new(
                        parts[0][ij].re,
                        core::array::from_fn(|p| parts[p + 1][ij].re),
                    );
                    let im = D4::new(
                        parts[0][ij].im,
                        core::array::from_fn(|p| parts[p + 1][ij].im),
                    );
                    Complex::new(re, im)
                })
                .collect();
            for (i, &(_, a)) in index.iter().enumerate() {
                for (j, &(_, b)) in index.iter().enumerate() {
                    let f = num_traits::Float::exp(lh_d[a] - lh_d[b]);
                    t[i * kc + j] = t[i * kc + j].scale(f);
                }
            }
            out[c] = t;
        }
        modes.push((out, mm.index));
    }
    Ok((TMatrixModes { n, modes }, report))
}

impl<T: Real> TMatrixModes<T> {
    /// The orientation-averaged ⟨C_ext⟩ and ⟨C_sca⟩ in units of 1/k² (Mishchenko 1991):
    /// −2π Σ (2 − δ_{m0}) Re(T¹¹ₙₙ + T²²ₙₙ) and 2π Σ (2 − δ_{m0}) |T|².
    pub(crate) fn cross_sections(&self) -> (T, T) {
        let (mut ext, mut sca) = (T::zero(), T::zero());
        for (m, (ts, index)) in self.modes.iter().enumerate() {
            let w = if m == 0 { T::one() } else { T::from_f64(2.0) };
            for (t, idx) in ts.iter().zip(index) {
                let kc = idx.len();
                if t.is_empty() {
                    continue;
                }
                for i in 0..kc {
                    ext -= w * t[i * kc + i].re;
                    for v in &t[i * kc..(i + 1) * kc] {
                        sca += w * v.norm_sqr();
                    }
                }
            }
        }
        let two_pi = T::PI() * T::from_f64(2.0);
        (two_pi * ext, two_pi * sca)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dual derivatives of ⟨C_ext⟩, ⟨C_sca⟩ and the Greek coefficients match central
    /// differences of the refined values, with and without refinement; the values are those of
    /// [`tmatrix_refined`].
    #[test]
    fn derivatives_match_differences() {
        let (x, eps, n, nth) = (6.0, 3.0, 20, 40);
        let s = Complex::new(1.5, 0.02);
        for tol in [1.0, 0.0] {
            let (d, rep) = tmatrix_derivatives(x, eps, s, n, nth, tol).unwrap();
            assert_eq!(rep.sweeps.iter().flatten().any(|&k| k > 0), tol == 0.0);
            let (ext, sca) = d.cross_sections();
            let g = d.greek_raw(12).unwrap();
            let at = |x: f64, eps: f64, s: Complex<f64>| {
                let (t, _) = tmatrix_refined(x, eps, s, n, nth, 0.0, n).unwrap();
                let (e, c) = t.cross_sections();
                (e, c, t.greek_raw(12).unwrap())
            };
            let (e0, c0, g0) = at(x, eps, s);
            assert!((ext.re() - e0).abs() < 1e-13 * e0 && (sca.re() - c0).abs() < 1e-13 * c0);
            let h = 1e-5;
            let steps = [
                (h * x, 0.0, Complex::new(0.0, 0.0)),
                (0.0, h * eps, Complex::new(0.0, 0.0)),
                (0.0, 0.0, Complex::new(h, 0.0)),
                (0.0, 0.0, Complex::new(0.0, h)),
            ];
            for (p, &(dx, de, ds)) in steps.iter().enumerate() {
                let (ep, cp, gp) = at(x + dx, eps + de, s + ds);
                let (em, cm, gm) = at(x - dx, eps - de, s - ds);
                let step = dx + de + ds.re + ds.im;
                let fd = |a: f64, b: f64| (a - b) / (2.0 * step);
                // Measured against the natural size of a derivative, value/|parameter|.
                let param = [x, eps, s.norm(), s.norm()][p];
                let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-7 * scale / param;
                assert!(
                    close(ext.eps()[p], fd(ep, em), e0),
                    "ext p {p}: {} vs {}",
                    ext.eps()[p],
                    fd(ep, em)
                );
                assert!(close(sca.eps()[p], fd(cp, cm), c0), "sca p {p}");
                let big = g0.alpha1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                for l in 0..g.alpha1.len() {
                    for (u, a, b) in [
                        (&g.alpha1, &gp.alpha1, &gm.alpha1),
                        (&g.alpha2, &gp.alpha2, &gm.alpha2),
                        (&g.beta2, &gp.beta2, &gm.beta2),
                    ] {
                        assert!(
                            close(u[l].eps()[p], fd(a[l], b[l]), big),
                            "greek l {l} p {p}"
                        );
                    }
                }
            }
        }
    }

    /// The refined path matches full double-double, its error estimate bounds its error, and the
    /// f64 path's estimate bounds the f64 error.
    #[test]
    fn refinement_reaches_double_double() {
        use crate::scalar::DoubleDouble as Dd;
        let (x, eps, n, nth) = (7.0, 3.0, 20, 40);
        let s = Complex::new(1.5, 0.01);
        let (f, f_rep) = tmatrix_refined(x, eps, s, n, nth, 1.0, n).unwrap();
        let (r, r_rep) = tmatrix_refined(x, eps, s, n, nth, 0.0, n).unwrap();
        let sd = Complex::new(Dd::from(s.re), Dd::from(s.im));
        let d = tmatrix(Dd::from(x), Dd::from(eps), sd, n, nth, true).unwrap();
        for (u, rep) in [(&f, &f_rep), (&r, &r_rep)] {
            for (m, (mu, md)) in u.modes.iter().zip(&d.modes).enumerate() {
                for c in 0..2 {
                    let err = mu.0[c]
                        .iter()
                        .zip(&md.0[c])
                        .map(|(a, b)| (a - Complex::new(b.re.hi(), b.im.hi())).norm())
                        .fold(0.0, f64::max);
                    assert!(
                        err <= rep.error[m][c].max(1e-300),
                        "m {m} c {c}: {err:e} est {:e} sweeps {}",
                        rep.error[m][c],
                        rep.sweeps[m][c]
                    );
                }
            }
        }
        assert!(f_rep.sweeps.iter().flatten().all(|&s| s == 0));
        assert!(r_rep.sweeps.iter().flatten().all(|&s| s > 0));
        let worst = r_rep.error.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
        assert!(worst < 1e-12, "{worst:e}");
        let err = r
            .modes
            .iter()
            .zip(&d.modes)
            .flat_map(|(a, b)| a.0.iter().flatten().zip(b.0.iter().flatten()))
            .map(|(a, b)| (a - Complex::new(b.re.hi(), b.im.hi())).norm())
            .fold(0.0, f64::max);
        assert!(err < 1e-13, "{err:e}");
    }
    use crate::mie::{Mie, MieOptions};
    use crate::scalar::DoubleDouble as Dd;

    /// Aspect ratio 1: T is diagonal with T¹¹ = −bₙ, T²² = −aₙ, in every mode.
    #[test]
    fn a_sphere_gives_the_mie_coefficients() {
        let (x, m) = (3.0, Complex::new(1.5, 0.01));
        let n = 12;
        let t = tmatrix(x, 1.0, m, n, 30, true).unwrap();
        let mut mie = Mie::<f64>::new();
        let opts = MieOptions::new().with_n_max(n).with_allow_unvalidated(true);
        let s = mie.compute(x, m, &opts).unwrap();
        let (a, b) = (s.a(), s.b());
        let mut worst = 0.0f64;
        for (mm, (ts, index)) in t.modes.iter().enumerate() {
            for c in 0..2 {
                let kc = index[c].len();
                for (i, &(ty, k)) in index[c].iter().enumerate() {
                    let want = if ty == 0 { -b[k - 1] } else { -a[k - 1] };
                    worst = worst.max((ts[c][i * kc + i] - want).norm() / want.norm().max(1e-300));
                    for j in 0..kc {
                        if j != i {
                            assert!(ts[c][i * kc + j].norm() < 1e-13, "mode {mm} off-diagonal");
                        }
                    }
                }
            }
        }
        assert!(worst < 1e-12, "worst relative error {worst:e}");
    }

    /// A lossless spheroid conserves energy: ⟨C_ext⟩ = ⟨C_sca⟩.
    #[test]
    fn lossless_spheroids_conserve_energy() {
        for (x, eps) in [(3.0, 2.0), (3.0, 0.5), (5.0, 1.5)] {
            let t = tmatrix(x, eps, Complex::new(1.5, 0.0), 16, 40, true).unwrap();
            let (ext, sca) = t.cross_sections();
            assert!(
                (ext - sca).abs() < 1e-11 * ext,
                "x {x} eps {eps}: {ext} vs {sca}"
            );
        }
    }

    /// The study's lab (research/tmatrix, mpmath at 30 digits): x = 3, ε = 2, s = 1.5, N = 12,
    /// 80 nodes over [−1, 1] give ⟨C_ext⟩ = 91.9311274132168 (in units of 1/k²).
    #[test]
    fn matches_the_reference_ebcm() {
        let t = tmatrix(
            Dd::from(3.0),
            Dd::from(2.0),
            Complex::new(Dd::from(1.5), Dd::from(0.0)),
            12,
            40,
            true,
        )
        .unwrap();
        let (ext, _) = t.cross_sections();
        assert!(
            (ext.hi() - 91.931_127_413_216_8).abs() < 1e-12,
            "{}",
            ext.hi()
        );
    }
}

#[cfg(test)]
mod diagnostics {
    use super::*;
    use crate::scalar::DoubleDouble as Dd;

    fn dd(c: Complex<f64>) -> Complex<Dd> {
        Complex::new(Dd::from(c.re), Dd::from(c.im))
    }

    fn tref_big(
        f: &[ModeMatrices<f64>],
        d: &[ModeMatrices<Dd>],
        m: usize,
        c: usize,
        lh: &[f64],
    ) -> f64 {
        let kc = f[m].index[c].len();
        let mut t = solve_t(&d[m].q[c], &d[m].p[c], kc).unwrap();
        unscale(&mut t, &f[m].index[c], lh);
        t.iter().map(|v| v.norm().hi()).fold(0.0, f64::max)
    }

    /// Mode 1, prolate: f64 solve of the double-double elements rounded, against the f64
    /// elements.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn rounded_elements() {
        let s = Complex::new(1.5, 0.01);
        let (x, eps, n, nth) = (15.0, 0.333_333_333_333, 42, 90);
        let rows = vec![Some(0); n + 1];
        let f = all_modes(&Spheroid::new(x, eps), s, n, nth, true, &rows).unwrap();
        let d = all_modes(
            &Spheroid::new(Dd::from(x), Dd::from(eps)),
            dd(s),
            n,
            nth,
            true,
            &rows,
        )
        .unwrap();
        let lh = log_h(&Scales::new(
            Spheroid::new(x, eps).r_min(),
            s.scale(Spheroid::new(x, eps).r_max()),
            n,
        ));
        let hi = |v: &[Complex<Dd>]| {
            v.iter()
                .map(|c| Complex::new(c.re.hi(), c.im.hi()))
                .collect::<Vec<_>>()
        };
        for m in [0usize, 1, 2, 8] {
            for c in 0..2 {
                let kc = f[m].index[c].len();
                let tref = {
                    let mut t = solve_t(&d[m].q[c], &d[m].p[c], kc).unwrap();
                    unscale(&mut t, &f[m].index[c], &lh);
                    hi(&t)
                };
                let run = |q: &[Complex<f64>], p: &[Complex<f64>]| {
                    let sol = super::super::solve::solve_f64(q, p, kc).unwrap();
                    let mut t = sol.t.clone();
                    unscale(&mut t, &f[m].index[c], &lh);
                    let big = tref.iter().map(|v| v.norm()).fold(0.0, f64::max);
                    let e = t
                        .iter()
                        .zip(&tref)
                        .map(|(a, b)| (a - b).norm())
                        .fold(0.0, f64::max);
                    e / big
                };
                let e_r = run(&hi(&d[m].q[c]), &hi(&d[m].p[c]));
                let e_f = run(&f[m].q[c], &f[m].p[c]);
                // The estimator fed with the true element errors, and with the bounds.
                let lh_c: Vec<f64> = f[m].index[c].iter().map(|&(_, nn)| lh[nn]).collect();
                let sol = super::super::solve::solve_f64(&f[m].q[c], &f[m].p[c], kc).unwrap();
                let true_q: Vec<f64> = f[m].q[c]
                    .iter()
                    .zip(&hi(&d[m].q[c]))
                    .map(|(a, b)| (a - b).norm())
                    .collect();
                let true_p: Vec<f64> = f[m].p[c]
                    .iter()
                    .zip(&hi(&d[m].p[c]))
                    .map(|(a, b)| (a - b).norm())
                    .collect();
                let u = f64::EPSILON / 2.0;
                let bq: Vec<f64> = f[m].q_abs[c].iter().map(|v| u * v).collect();
                let bp: Vec<f64> = f[m].p_abs[c].iter().map(|v| u * v).collect();
                let big = tref_big(&f, &d, m, c, &lh);
                let est_true =
                    super::super::solve::error_estimate(&sol, &sol.t, &true_q, &true_p, &lh_c, kc)
                        / big;
                let est_bound =
                    super::super::solve::error_estimate(&sol, &sol.t, &bq, &bp, &lh_c, kc) / big;
                let ratio = true_q
                    .iter()
                    .zip(&bq)
                    .map(|(a, b)| a / b.max(1e-300))
                    .fold(0.0, f64::max);
                println!(
                    "   est(true errors) {est_true:.1e}, est(bounds) {est_bound:.1e}, max true/bound {ratio:.1e}"
                );
                // Element errors of f64 relative to their row's largest element.
                let mut row_err = 0.0f64;
                let mut where_ = (0, 0);
                for i in 0..kc {
                    let rmax = (0..kc)
                        .map(|j| d[m].q[c][i * kc + j].norm().hi())
                        .fold(0.0, f64::max);
                    for j in 0..kc {
                        let v = d[m].q[c][i * kc + j];
                        let e = (f[m].q[c][i * kc + j] - Complex::new(v.re.hi(), v.im.hi())).norm()
                            / rmax;
                        if e > row_err {
                            row_err = e;
                            where_ = (f[m].index[c][i].1, f[m].index[c][j].1);
                        }
                    }
                }
                println!(
                    "m {m} c {c}: rounded dd {e_r:.1e}; f64 {e_f:.1e}; worst row-rel element err {row_err:.1e} at {where_:?}"
                );
            }
        }
    }

    /// Per mode: Skeel estimate and errors of f64 and refined T against double-double.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn refined_per_mode() {
        let s = Complex::new(1.5, 0.01);
        let (x, eps, n, nth) = (15.0, 0.333_333_333_333, 42, 90);
        let plain = tmatrix(x, eps, s, n, nth, true).unwrap();
        let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 0.0, n).unwrap();
        let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
        for m in 0..=n {
            for c in 0..2 {
                let (mut big, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
                for ((u, r), v) in plain.modes[m].0[c]
                    .iter()
                    .zip(&refined.modes[m].0[c])
                    .zip(&b.modes[m].0[c])
                {
                    let v = Complex::new(v.re.hi(), v.im.hi());
                    big = big.max(v.norm());
                    ea = ea.max((u - v).norm());
                    eb = eb.max((r - v).norm());
                }
                println!(
                    "m {m} c {c}: max|T| {big:.1e} est {:.1e} f64 {:.1e} refined {:.1e} sweeps {}",
                    report.error[m][c], ea, eb, report.sweeps[m][c]
                );
            }
        }
    }

    /// The refined path against full double-double: run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn refined_against_double_double() {
        let s = Complex::new(1.5, 0.01);
        for &(x, eps, n, nth) in &[
            (10.0, 2.0, 24, 50),
            (15.0, 4.0, 40, 100),
            (15.0, 0.333_333_333_333, 42, 90),
            (25.0, 0.333_333_333_333, 60, 120),
            (30.0, 2.0, 50, 90),
        ] {
            let t0 = std::time::Instant::now();
            let plain = tmatrix(x, eps, s, n, nth, true).unwrap();
            let t_plain = t0.elapsed();
            let t0 = std::time::Instant::now();
            let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 1e-14, n).unwrap();
            let t_ref = t0.elapsed();
            let t0 = std::time::Instant::now();
            let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
            let t_dd = t0.elapsed();
            let err = |u: &TMatrixModes<f64>| {
                let (mut big, mut diff) = (0.0f64, 0.0f64);
                for (ma, mb) in u.modes.iter().zip(&b.modes) {
                    for c in 0..2 {
                        for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
                            let v = Complex::new(v.re.hi(), v.im.hi());
                            big = big.max(v.norm());
                            diff = diff.max((u - v).norm());
                        }
                    }
                }
                diff / big
            };
            let skeel = report.error.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
            let sweeps = report.sweeps.iter().flatten().max().unwrap();
            let n_ref = report.sweeps.iter().filter(|s| s[0] + s[1] > 0).count();
            let big = refined
                .modes
                .iter()
                .flat_map(|m| m.0.iter().flatten())
                .fold(0.0f64, |a, v| a.max(v.norm()));
            let skeel = skeel / big;
            println!(
                "x {x} eps {eps:.3} N {n}: f64 {:.1e} ({t_plain:.2?}), refined {:.1e} ({t_ref:.2?}, \
                 ≤{sweeps} sweeps, {n_ref} modes), dd ({t_dd:.2?}); est {skeel:.1e}",
                err(&plain),
                err(&refined),
            );
        }
    }

    /// f64 against double-double, plain integrals: run with
    /// `cargo test --release -p scatrs --lib tmatrix::kernel::diagnostics -- --ignored --nocapture`.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn f64_against_double_double() {
        let s = Complex::new(1.5, 0.01);
        for &(x, eps, n, nth) in &[
            (5.0, 2.0, 14, 30),
            (10.0, 2.0, 24, 50),
            (20.0, 2.0, 36, 70),
            (15.0, 4.0, 40, 100),
            (15.0, 0.333_333_333_333, 42, 90),
            (30.0, 2.0, 50, 90),
        ] {
            let t = std::time::Instant::now();
            let a = tmatrix(x, eps, s, n, nth, true).unwrap();
            let t64 = t.elapsed();
            let plain = tmatrix(x, eps, s, n, nth, false).unwrap();
            let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, false).unwrap();
            let b_reg = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
            let err = |u: &TMatrixModes<f64>| {
                let (mut big, mut diff) = (0.0f64, 0.0f64);
                for (ma, mb) in u.modes.iter().zip(&b.modes) {
                    for c in 0..2 {
                        for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
                            let v = Complex::new(v.re.hi(), v.im.hi());
                            big = big.max(v.norm());
                            diff = diff.max((u - v).norm());
                        }
                    }
                }
                diff / big
            };
            let (mut big, mut diff) = (0.0f64, 0.0f64);
            for (ma, mb) in b_reg.modes.iter().zip(&b.modes) {
                for c in 0..2 {
                    for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
                        big = big.max(v.re.hi().abs() + v.im.hi().abs());
                        diff = diff.max((u.re - v.re).hi().abs() + (u.im - v.im).hi().abs());
                    }
                }
            }
            let (ea, sa) = a.cross_sections();
            let (eb, sb) = b.cross_sections();
            println!(
                "x {x} eps {eps} N {n}: T plain {:.1e}, regularized {:.1e} (dd forms agree {:.1e}); \
                 C_ext {:.1e}, C_sca {:.1e} (f64 {:?})",
                err(&plain),
                err(&a),
                diff / big,
                ((ea - eb.hi()) / eb.hi()).abs(),
                ((sa - sb.hi()) / sb.hi()).abs(),
                t64
            );
        }
    }
}

#[cfg(test)]
mod element_diagnostics {
    use super::*;
    use crate::scalar::DoubleDouble as Dd;

    /// Q element errors, f64 (plain and regularized) against double-double (plain and
    /// regularized), worst per mode-0/1: run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn q_elements() {
        let s = Complex::new(1.5, 0.01);
        let sd = Complex::new(Dd::from(1.5), Dd::from(0.01));
        for &(x, eps, n, nth) in &[
            (10.0, 2.0, 24, 50),
            (15.0, 0.333_333_333_333, 42, 90),
            (15.0, 4.0, 40, 100),
        ] {
            let sh = Spheroid::new(x, eps);
            let shd = Spheroid::new(Dd::from(x), Dd::from(eps));
            let qp = all_modes(&sh, s, n, nth, false, &vec![Some(0); n + 1]).unwrap();
            let qr = all_modes(&sh, s, n, nth, true, &vec![Some(0); n + 1]).unwrap();
            let dp = all_modes(&shd, sd, n, nth, false, &vec![Some(0); n + 1]).unwrap();
            let dr = all_modes(&shd, sd, n, nth, true, &vec![Some(0); n + 1]).unwrap();
            for m in [0usize, 1] {
                for c in 0..2 {
                    let idx = &qp[m].index[c];
                    let kc = idx.len();
                    let mut worst: Vec<(f64, f64, f64, usize, usize, usize, usize)> = Vec::new();
                    for (i, &(rt, nn)) in idx.iter().enumerate() {
                        let row_max = (0..kc)
                            .map(|j| dp[m].q[c][i * kc + j].norm().hi())
                            .fold(0.0, f64::max);
                        for (j, &(ctp, kk)) in idx.iter().enumerate() {
                            let slot = i * kc + j;
                            let refv =
                                Complex::new(dr[m].q[c][slot].re.hi(), dr[m].q[c][slot].im.hi());
                            let refp =
                                Complex::new(dp[m].q[c][slot].re.hi(), dp[m].q[c][slot].im.hi());
                            let scale = refv.norm().max(1e-30 * row_max);
                            let ep = (qp[m].q[c][slot] - refv).norm() / scale;
                            let er = (qr[m].q[c][slot] - refv).norm() / scale;
                            let ed = (refp - refv).norm() / scale;
                            worst.push((er, ep, ed, rt, nn, ctp, kk));
                        }
                    }
                    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                    for w in worst.iter().take(3) {
                        println!(
                            "x {x} eps {eps:.3} m {m} c {c}: ({},{})-({},{}): reg {:.1e} plain {:.1e} dd-forms {:.1e}",
                            w.3, w.4, w.5, w.6, w.0, w.1, w.2
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod profile {
    use super::*;

    /// One double-double `all_modes`, to profile: run with `--ignored`.
    #[test]
    #[ignore = "a profile target"]
    fn dd_all_modes() {
        let (x, eps, n, nth) = (15.0, 2.0, 40, 80);
        let sh = Spheroid::new(Dd::from(x), Dd::from(eps));
        let s = Complex::new(Dd::from(1.53), Dd::from(0.008));
        let t0 = std::time::Instant::now();
        let rows = vec![Some(0); n + 1];
        all_modes(&sh, s, n, nth, true, &rows).unwrap();
        println!("dd regularized all rows: {:.2?}", t0.elapsed());
        let t0 = std::time::Instant::now();
        all_modes(&sh, s, n, nth, false, &rows).unwrap();
        println!("dd plain all rows: {:.2?}", t0.elapsed());
        let mut one = vec![None; n + 1];
        one[0] = Some(0);
        let t0 = std::time::Instant::now();
        all_modes(&sh, s, n, nth, true, &one).unwrap();
        println!("dd regularized mode 0 only: {:.2?}", t0.elapsed());
        let shf = Spheroid::new(x, eps);
        let sf = Complex::new(1.53, 0.008);
        let t0 = std::time::Instant::now();
        all_modes(&shf, sf, n, nth, true, &rows).unwrap();
        println!("f64 regularized: {:.2?}", t0.elapsed());
        let t0 = std::time::Instant::now();
        all_modes(&shf, sf, n, nth, false, &rows).unwrap();
        println!("f64 plain: {:.2?}", t0.elapsed());
        let t0 = std::time::Instant::now();
        all_modes(&shf, sf, n, nth, true, &one).unwrap();
        println!("f64 regularized mode 0 only: {:.2?}", t0.elapsed());
    }
}

#[cfg(test)]
mod derivative_cost {
    use super::*;

    /// The cost of the derivatives against the values: run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a diagnostic, not a check"]
    fn ratio() {
        let s = Complex::new(1.53, 0.008);
        for &(x, eps, n, nth, tol) in &[
            (5.0, 2.0, 18, 48, 1e-9),
            (10.0, 2.0, 30, 70, 1e-9),
            (10.0, 2.0, 30, 70, f64::INFINITY),
            (15.0, 2.0, 45, 106, 1e-9),
        ] {
            let t0 = std::time::Instant::now();
            let (_, r) = tmatrix_refined(x, eps, s, n, nth, tol, n).unwrap();
            let tv = t0.elapsed();
            let t0 = std::time::Instant::now();
            tmatrix_derivatives(x, eps, s, n, nth, tol).unwrap();
            let td = t0.elapsed();
            let refined = r.sweeps.iter().filter(|s| s[0] + s[1] > 0).count();
            println!(
                "x {x} eps {eps} N {n} tol {tol:e}: values {tv:.2?}, with derivatives {td:.2?} ({:.1}x), {refined} refined",
                td.as_secs_f64() / tv.as_secs_f64()
            );
        }
    }
}
