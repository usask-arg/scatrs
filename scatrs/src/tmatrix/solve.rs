//! The dense solves of the EBCM: LU factorization, T = −P Q⁻¹, Skeel's condition estimate, and
//! mixed-precision iterative refinement of T.
//!
//! In f64 the error of T is about Skeel(Q)·2⁻⁵³ even with exact elements, carried by the rows of
//! high outgoing order (on a non-spherical surface the evanescent multipoles r^{−n−1}Pₙ, nearly
//! dependent near r_min). Refinement keeps the O(K³) factorization in f64 and corrects
//! X Q = −P with double-double residuals, from elements exact to double-double in the rows of
//! high order: it converges to double-double accuracy while Skeel·2⁻⁵³ < 1.

use num_complex::Complex;

use crate::complex::inv;
use crate::mie::kernel::real;
use crate::scalar::{DoubleDouble as Dd, Real};
use crate::{Error, Result};

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

/// Solves Aᵀ x = b in place with the factors of `lu` (P A = L U, so Aᵀ = Uᵀ Lᵀ P).
fn lu_solve_transpose(a: &[Complex<f64>], piv: &[usize], n: usize, b: &mut [Complex<f64>]) {
    let mut y = b.to_vec();
    for i in 0..n {
        let mut v = y[i];
        for j in 0..i {
            v -= a[j * n + i] * y[j];
        }
        y[i] = v / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut v = y[i];
        for j in i + 1..n {
            v -= a[j * n + i] * y[j];
        }
        y[i] = v;
    }
    for (i, &p) in piv.iter().enumerate() {
        b[p] = y[i];
    }
}

/// One class of one mode solved in f64: T = −P Q⁻¹ (scaled), with the factors of Qᵀ.
pub(crate) struct F64Solve {
    pub(crate) t: Vec<Complex<f64>>,
    qt: Vec<Complex<f64>>,
    piv: Vec<usize>,
}

pub(crate) fn solve_f64(q: &[Complex<f64>], p: &[Complex<f64>], n: usize) -> Result<F64Solve> {
    let mut qt: Vec<Complex<f64>> = (0..n * n).map(|ij| q[(ij % n) * n + ij / n]).collect();
    let piv = lu(&mut qt, n)?;
    let mut t = vec![Complex::new(0.0, 0.0); n * n];
    let mut col = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        for j in 0..n {
            col[j] = -p[i * n + j];
        }
        lu_solve(&qt, &piv, n, &mut col);
        t[i * n..(i + 1) * n].copy_from_slice(&col);
    }
    Ok(F64Solve { t, qt, piv })
}

/// A small deterministic generator of signs for the error probes.
struct Signs(u64);

impl Signs {
    fn next(&mut self) -> f64 {
        // xorshift64*.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        if self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 63 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// An estimate of the largest error of the unscaled T (`t`, scaled, solved with the factors of
/// `solved`) from the elements' error bounds `eq`, `ep`
/// (absolute, scaled units; they include the factorization's rounding, about u|Q|): to first
/// order ΔT' = −(ΔP + T' ΔQ) Q⁻¹, probed with random-signed perturbations at their bounds and
/// random-signed vectors, three times, with a factor 4 for the largest element. `lh` are the logarithms of the unscaling weights of the
/// class's orders (T_ik = T'_ik e^{lh_i − lh_k}).
pub(crate) fn error_estimate(
    solved: &F64Solve,
    t: &[Complex<f64>],
    eq: &[f64],
    ep: &[f64],
    lh: &[f64],
    n: usize,
) -> f64 {
    error_estimate_rows(solved, t, eq, ep, lh, n).0
}

/// As [`error_estimate`], with each row's share: the largest error (unscaled) that the
/// rounding of row l of Q (through column l of T') and of P (in row l of T) would leave, times
/// 4. The rows whose share is small can stay in f64 when the solve is refined.
pub(crate) fn error_estimate_rows(
    solved: &F64Solve,
    t: &[Complex<f64>],
    eq: &[f64],
    ep: &[f64],
    lh: &[f64],
    n: usize,
) -> (f64, Vec<f64>) {
    let mut signs = Signs(0x9e37_79b9_7f4a_7c15);
    let mut worst = 0.0f64;
    let mut rows = vec![0.0f64; n];
    let mut y = vec![Complex::new(0.0, 0.0); n];
    // max_i |T'_il| e^{lh_i}: how row l of Q reaches the unscaled T.
    let reach: Vec<f64> = (0..n)
        .map(|l| {
            (0..n)
                .map(|i| t[i * n + l].norm() * lh[i].exp())
                .fold(0.0, f64::max)
        })
        .collect();
    for _ in 0..3 {
        // y = Q⁻¹ r', r'_k = ±e^{−lh_k}.
        for (k, v) in y.iter_mut().enumerate() {
            *v = Complex::new(signs.next() * (-lh[k]).exp(), 0.0);
        }
        lu_solve_transpose(&solved.qt, &solved.piv, n, &mut y);
        // w = ΔQ y, v = ΔP y + T' w.
        let w: Vec<Complex<f64>> = (0..n)
            .map(|l| (0..n).map(|k| y[k] * (signs.next() * eq[l * n + k])).sum())
            .collect();
        for i in 0..n {
            let mut dp = Complex::new(0.0, 0.0);
            let mut v = Complex::new(0.0, 0.0);
            for k in 0..n {
                dp += y[k] * (signs.next() * ep[i * n + k]);
                v += t[i * n + k] * w[k];
            }
            v += dp;
            worst = worst.max(v.norm() * lh[i].exp());
            rows[i] = rows[i].max(dp.norm() * lh[i].exp());
        }
        for (l, r) in rows.iter_mut().enumerate() {
            *r = r.max(w[l].norm() * reach[l]);
        }
    }
    // The probes see a random combination of the errors: 4× covers the largest element.
    (4.0 * worst, rows.into_iter().map(|r| 4.0 * r).collect())
}

/// The result of [`refine`].
pub(crate) struct Refined {
    /// T' (scaled), rounded to f64 and in double-double.
    pub(crate) t: Vec<Complex<f64>>,
    pub(crate) t_dd: Vec<Complex<Dd>>,
    /// The mixed Q (double-double rows and f64 rows).
    pub(crate) q: Vec<Complex<Dd>>,
    /// The factors of the mixed Q rounded to f64.
    pub(crate) solved: F64Solve,
    /// The sweeps made.
    pub(crate) sweeps: usize,
    /// The last correction's largest element, unscaled (`lh` as in [`error_estimate`]).
    pub(crate) last: f64,
}

/// Mixed-precision refinement of one class: Q and P are `q`, `p` (f64) except in the rows flagged
/// by `exact`, where they are `qd`, `pd` (double-double). Factorizes the mixed matrix rounded to
/// f64, then corrects X Q = −P with double-double residuals R = −P − X Q until the correction
/// falls below 2⁻⁶⁰ of T or after `sweeps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine(
    q: &[Complex<f64>],
    p: &[Complex<f64>],
    qd: &[Complex<Dd>],
    pd: &[Complex<Dd>],
    exact: &[bool],
    lh: &[f64],
    n: usize,
    sweeps: usize,
) -> Result<Refined> {
    let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
    let hi = |c: Complex<Dd>| Complex::new(c.re.hi(), c.im.hi());
    let pick = |ij: usize, v: &[Complex<f64>], vd: &[Complex<Dd>]| {
        if exact[ij / n] { vd[ij] } else { dd(v[ij]) }
    };
    let qm: Vec<Complex<Dd>> = (0..n * n).map(|ij| pick(ij, q, qd)).collect();
    let pm: Vec<Complex<Dd>> = (0..n * n).map(|ij| pick(ij, p, pd)).collect();
    let q53: Vec<Complex<f64>> = qm.iter().map(|&c| hi(c)).collect();
    let p53: Vec<Complex<f64>> = pm.iter().map(|&c| hi(c)).collect();
    let solved = solve_f64(&q53, &p53, n)?;
    let mut x: Vec<Complex<Dd>> = solved.t.iter().map(|&c| dd(c)).collect();
    let big = solved.t.iter().map(|c| c.norm()).fold(0.0, f64::max);
    let (mut done, mut last) = (0, 0.0f64);
    let mut row = vec![Complex::new(0.0, 0.0); n];
    for _ in 0..sweeps {
        done += 1;
        let (mut worst, mut worst_unscaled) = (0.0f64, 0.0f64);
        for i in 0..n {
            for j in 0..n {
                let mut r = -pm[i * n + j];
                for l in 0..n {
                    r -= x[i * n + l] * qm[l * n + j];
                }
                row[j] = hi(r);
            }
            lu_solve(&solved.qt, &solved.piv, n, &mut row);
            for j in 0..n {
                worst = worst.max(row[j].norm());
                worst_unscaled = worst_unscaled.max(row[j].norm() * (lh[i] - lh[j]).exp());
                x[i * n + j] += dd(row[j]);
            }
        }
        last = worst_unscaled;
        if worst <= big * 2f64.powi(-60) {
            break;
        }
    }
    Ok(Refined {
        t: x.iter().map(|&c| hi(c)).collect(),
        t_dd: x,
        q: qm,
        solved,
        sweeps: done,
        last,
    })
}

/// The derivative of T' = −P Q⁻¹ along one direction: dT' = −(dP + T' dQ) Q⁻¹, with the
/// factors of Q.
pub(crate) fn derivative_f64(
    solved: &F64Solve,
    t: &[Complex<f64>],
    dq: &[Complex<f64>],
    dp: &[Complex<f64>],
    n: usize,
) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0.0, 0.0); n * n];
    let mut row = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        for j in 0..n {
            let mut v = dp[i * n + j];
            for l in 0..n {
                v += t[i * n + l] * dq[l * n + j];
            }
            row[j] = -v;
        }
        lu_solve(&solved.qt, &solved.piv, n, &mut row);
        out[i * n..(i + 1) * n].copy_from_slice(&row);
    }
    out
}

/// Refines a derivative dT' of a refined class (`r`), with dQ and dP mixed as Q and P were
/// (`dq`, `dp` in double-double, exact in the refined rows): residuals
/// R = −dP − T' dQ − dT' Q in double-double, corrected by the f64 factors, for `sweeps` sweeps or
/// until the correction falls below 2⁻⁶⁰ of dT'.
pub(crate) fn refine_derivative(
    r: &Refined,
    dq: &[Complex<Dd>],
    dp: &[Complex<Dd>],
    n: usize,
    sweeps: usize,
) -> Vec<Complex<f64>> {
    let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
    let hi = |c: Complex<Dd>| Complex::new(c.re.hi(), c.im.hi());
    // The right-hand side −dP − T' dQ, fixed over the sweeps.
    let mut rhs = vec![Complex::new(Dd::from(0.0), Dd::from(0.0)); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut v = -dp[i * n + j];
            for l in 0..n {
                v -= r.t_dd[i * n + l] * dq[l * n + j];
            }
            rhs[i * n + j] = v;
        }
    }
    let first: Vec<Complex<f64>> = rhs.iter().map(|&c| hi(c)).collect();
    let mut x: Vec<Complex<Dd>> = Vec::with_capacity(n * n);
    let mut row = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        row.copy_from_slice(&first[i * n..(i + 1) * n]);
        lu_solve(&r.solved.qt, &r.solved.piv, n, &mut row);
        x.extend(row.iter().map(|&c| dd(c)));
    }
    let big = x.iter().map(|c| hi(*c).norm()).fold(0.0, f64::max);
    for _ in 0..sweeps {
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut v = rhs[i * n + j];
                for l in 0..n {
                    v -= x[i * n + l] * r.q[l * n + j];
                }
                row[j] = hi(v);
            }
            lu_solve(&r.solved.qt, &r.solved.piv, n, &mut row);
            for j in 0..n {
                worst = worst.max(row[j].norm());
                x[i * n + j] += dd(row[j]);
            }
        }
        if worst <= big * 2f64.powi(-60) {
            break;
        }
    }
    x.iter().map(|&c| hi(c)).collect()
}
