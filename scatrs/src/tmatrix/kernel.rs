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
use super::regular::{Series, angular_factors};
use crate::complex::inv;
use crate::mie::kernel::{int, real};
use crate::scalar::Real;
use crate::special::gauss_legendre;
use crate::{Error, Result};

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

/// Q and P of mode m with orders max(m, 1), …, n on n_theta nodes in 0 < θ < π/2.
pub(crate) fn mode_matrices<T: Real>(
    shape: &Spheroid<T>,
    s: Complex<T>,
    m: usize,
    n: usize,
    n_theta: usize,
) -> Result<ModeMatrices<T>> {
    let n0 = m.max(1);
    if n < n0 {
        return Err(Error::InvalidInput(format!(
            "mode {m} needs at least {n0} orders, got {n}"
        )));
    }
    // Index sets of the two classes: (type, order) with (order + type) % 2 == class.
    let mut index: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
    for t in 0..2 {
        for k in n0..=n {
            index[(k + t) % 2].push((t, k));
        }
    }
    let size = [index[0].len(), index[1].len()];
    let mut q: [Vec<Complex<T>>; 2] =
        core::array::from_fn(|c| vec![real(T::zero()); size[c] * size[c]]);
    let mut p: [Vec<Complex<T>>; 2] =
        core::array::from_fn(|c| vec![real(T::zero()); size[c] * size[c]]);
    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
    let rule = gauss_legendre::<T>(2 * n_theta)?;
    let mut radial = Radial::new();
    let mut ang = Angular::new();
    let gam: Vec<T> = (0..=n)
        .map(|k| if k == 0 { T::zero() } else { gamma(k) })
        .collect();
    let i1 = Complex::new(T::zero(), T::one());
    let zero3 = [real(T::zero()); 3];
    // Field components per order: outgoing (h) and regular (j) M, N of the medium, and the
    // particle's M, N; all without γ (applied per element).
    let mut out_m = vec![zero3; n + 1];
    let mut out_n = vec![zero3; n + 1];
    let mut reg_m = vec![zero3; n + 1];
    let mut reg_n = vec![zero3; n + 1];
    let mut in_m = vec![zero3; n + 1];
    let mut in_n = vec![zero3; n + 1];
    for (&t, &w) in rule.nodes.iter().zip(&rule.weights) {
        if t <= T::zero() {
            continue;
        }
        let (ct, st) = (t, (T::one() - t * t).sqrt());
        let (r, nu) = shape.at(ct, st);
        radial.compute(r, s, n, &scales);
        ang.compute(m, n, ct, st);
        for k in n0..=n {
            let (d, pi, tau) = (ang.d[k], ang.pi[k], ang.tau[k]);
            let kk = int::<T>(k * (k + 1));
            let po = real(-pi); // the outgoing waves' −π
            let field = |h: Complex<T>, dh: Complex<T>, h_x: Complex<T>, p: Complex<T>| {
                (
                    [real(T::zero()), i1 * h * p, -(h.scale(tau))],
                    [h_x.scale(kk * d), dh.scale(tau), i1 * dh * p],
                )
            };
            (out_m[k], out_n[k]) = field(radial.h[k], radial.dh[k], radial.h_x[k], po);
            (reg_m[k], reg_n[k]) = field(radial.j[k], radial.dj[k], radial.j_x[k], po);
            (in_m[k], in_n[k]) = field(radial.jz[k], radial.djz[k], radial.jz_z[k], real(pi));
        }
        // 2 for the two halves of the range, 2π for φ, x² from dS (k = 1).
        let fac = w * r * r * T::PI() * T::from_f64(4.0);
        for c in 0..2 {
            let idx = &index[c];
            let kc = idx.len();
            for (i, &(rt, nn)) in idx.iter().enumerate() {
                let (o1, o2) = if rt == 0 {
                    (&out_m[nn], &out_n[nn])
                } else {
                    (&out_n[nn], &out_m[nn])
                };
                let (g1, g2) = if rt == 0 {
                    (&reg_m[nn], &reg_n[nn])
                } else {
                    (&reg_n[nn], &reg_m[nn])
                };
                for (j, &(ctp, kk_)) in idx.iter().enumerate() {
                    let (b1, b2) = if ctp == 0 {
                        (in_n[kk_].map(|v| v * s), &in_m[kk_])
                    } else {
                        (in_m[kk_].map(|v| v * s), &in_n[kk_])
                    };
                    let g = fac * gam[nn] * gam[kk_];
                    let qv = triple(nu, o1, &b1) + triple(nu, o2, b2);
                    let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
                    // −i·(…)
                    q[c][i * kc + j] = q[c][i * kc + j] + Complex::new(qv.im, -qv.re).scale(g);
                    p[c][i * kc + j] = p[c][i * kc + j] + Complex::new(pv.im, -pv.re).scale(g);
                }
            }
        }
    }
    Ok(ModeMatrices { q, p, index })
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
) -> Result<Vec<ModeMatrices<T>>> {
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
                        if nn == 0 || kk == 0 || nn == kk {
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
            }
            for c in 0..2 {
                let idx = &mm.index[c];
                let kc = idx.len();
                for (i, &(rt, nn)) in idx.iter().enumerate() {
                    let (g1, g2) = if rt == 0 {
                        (&reg_m[nn], &reg_n[nn])
                    } else {
                        (&reg_n[nn], &reg_m[nn])
                    };
                    let an = (ang.d[nn], ang.pi[nn], ang.tau[nn]);
                    for (j, &(ctp, kk_)) in idx.iter().enumerate() {
                        let (b1, b2) = if ctp == 0 {
                            (in_n[kk_].map(|v| v * s), &in_m[kk_])
                        } else {
                            (in_m[kk_].map(|v| v * s), &in_n[kk_])
                        };
                        let gg = gam[nn] * gam[kk_];
                        let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
                        let slot = i * kc + j;
                        mm.p[c][slot] = mm.p[c][slot] + Complex::new(pv.im, -pv.re).scale(fac * gg);
                        // Y: Σ_t a_t R_t in its three forms.
                        let ak = (ang.d[kk_], ang.pi[kk_], ang.tau[kk_]);
                        let a = angular_factors(rt, ctp, s, nu_prime, nn, kk_, an, ak);
                        let ge = g0 * gg;
                        let (mut va, mut aa) = (zero, T::zero());
                        let (mut vb, mut ab) = (zero, T::zero());
                        let (mut vc, mut ac) = (zero, T::zero());
                        for (ti, &at) in a.iter().enumerate() {
                            if at == zero {
                                continue;
                            }
                            let fi = (ti * n1 + nn) * n1 + kk_;
                            let rf = full[fi];
                            let mag = at.norm();
                            va = va + at * rf;
                            aa += mag * rf.norm();
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
                }
            }
        }
    }
    Ok(modes)
}

/// LU factorization with partial pivoting of a row-major n × n matrix, in place; returns the
/// pivots.
pub(crate) fn lu<T: Real>(a: &mut [Complex<T>], n: usize) -> Result<Vec<usize>> {
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut best = k;
        let mut big = T::zero();
        for i in k..n {
            let v = a[i * n + k].norm();
            if v > big {
                big = v;
                best = i;
            }
        }
        if big == T::zero() || !big.is_finite() {
            return Err(Error::InvalidInput(format!(
                "the EBCM matrix is singular or not finite at column {k}"
            )));
        }
        if best != k {
            for j in 0..n {
                a.swap(k * n + j, best * n + j);
            }
            piv.swap(k, best);
        }
        let inv_p = inv(a[k * n + k]);
        for i in k + 1..n {
            let f = a[i * n + k] * inv_p;
            a[i * n + k] = f;
            if f.re == T::zero() && f.im == T::zero() {
                continue;
            }
            for j in k + 1..n {
                let v = a[k * n + j];
                a[i * n + j] = a[i * n + j] - f * v;
            }
        }
    }
    Ok(piv)
}

/// Solves A x = b in place with the factors of `lu`.
pub(crate) fn lu_solve<T: Real>(a: &[Complex<T>], piv: &[usize], n: usize, b: &mut [Complex<T>]) {
    let mut x: Vec<Complex<T>> = piv.iter().map(|&i| b[i]).collect();
    for i in 0..n {
        let mut v = x[i];
        for j in 0..i {
            v = v - a[i * n + j] * x[j];
        }
        x[i] = v;
    }
    for i in (0..n).rev() {
        let mut v = x[i];
        for j in i + 1..n {
            v = v - a[i * n + j] * x[j];
        }
        x[i] = v * inv(a[i * n + i]);
    }
    b.copy_from_slice(&x);
}

/// T = −P Q⁻¹ of one class (scaled as P and Q are): solved as Qᵀ Tᵀ = −Pᵀ.
pub(crate) fn solve_t<T: Real>(
    q: &[Complex<T>],
    p: &[Complex<T>],
    n: usize,
) -> Result<Vec<Complex<T>>> {
    let mut qt: Vec<Complex<T>> = (0..n * n).map(|ij| q[(ij % n) * n + ij / n]).collect();
    let piv = lu(&mut qt, n)?;
    let mut t = vec![real(T::zero()); n * n];
    let mut col = vec![real(T::zero()); n];
    for i in 0..n {
        // Row i of T is column i of Tᵀ: Qᵀ (row i of T)ᵀ = −(row i of P)ᵀ.
        for j in 0..n {
            col[j] = -p[i * n + j];
        }
        lu_solve(&qt, &piv, n, &mut col);
        t[i * n..(i + 1) * n].copy_from_slice(&col);
    }
    Ok(t)
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
    let all = all_modes(&shape, s, n, n_theta, regularize)?;
    let mut modes = Vec::with_capacity(n + 1);
    for mm in all {
        let mut ts: [Vec<Complex<T>>; 2] = [Vec::new(), Vec::new()];
        for ((slot, index), (q, p)) in ts.iter_mut().zip(&mm.index).zip(mm.q.iter().zip(&mm.p)) {
            let kc = index.len();
            if kc == 0 {
                continue;
            }
            let mut t = solve_t(q, p, kc)?;
            // Undo the scaling: T = Σ⁻¹ T' Σ, T_{nk} = T'_{nk} σ_k/σₙ = T' |hₙ|/|h_k| (x_min).
            for (i, &(_, nn)) in index.iter().enumerate() {
                for (j, &(_, kk_)) in index.iter().enumerate() {
                    let f = T::from_f64((lh[nn] - lh[kk_]).exp());
                    t[i * kc + j] = t[i * kc + j].scale(f);
                }
            }
            *slot = t;
        }
        modes.push((ts, mm.index));
    }
    Ok(TMatrixModes { n, modes })
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
            let qp = all_modes(&sh, s, n, nth, false).unwrap();
            let qr = all_modes(&sh, s, n, nth, true).unwrap();
            let dp = all_modes(&shd, sd, n, nth, false).unwrap();
            let dr = all_modes(&shd, sd, n, nth, true).unwrap();
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
