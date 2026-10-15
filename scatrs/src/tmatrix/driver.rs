//! The convergence loop and the precision paths: T of a spheroid to a relative tolerance.
//!
//! 1. N grows from an estimate until the relative changes of mode 0's contributions to ⟨C_ext⟩
//!    and ⟨C_sca⟩ fall below `rtol`, with n_θ tied to N (as in Mishchenko's `tmatd`: mode 0
//!    involves every order, and the higher modes converge faster).
//! 2. n_θ then grows until they settle again.
//! 3. All modes are solved at the final N and n_θ.
//! 4. If Greek coefficients are asked for, one more step in N and n_θ, with all modes, must
//!    leave them and the cross sections settled.
//!
//! Each step solves every mode in f64 with an error estimate, and refines the modes whose
//! estimate exceeds a tenth of `rtol` (relative to the largest element of T). If refinement
//! leaves an estimate above that, the whole T is computed in double-double.

use num_complex::Complex;

use super::kernel::{D4, SolveReport, TMatrixModes, tmatrix, tmatrix_derivatives, tmatrix_refined};
use crate::scalar::{DoubleDouble as Dd, Dual};
use crate::{Error, Result};

/// How the modes of a T-matrix were solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrecisionPath {
    /// f64 elements and an f64 solve.
    F64,
    /// The f64 solve refined with double-double residuals, the rows of high outgoing order
    /// integrated in double-double.
    Refined,
    /// Elements and solve in double-double.
    DoubleDouble,
}

/// Which precision paths may be used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Precision {
    /// f64, refined or double-double per mode, as the error estimates require.
    #[default]
    Auto,
    /// f64 only, whatever the estimates (fastest; the estimates still report the error).
    F64,
    /// Double-double throughout (slowest; for checks).
    DoubleDouble,
}

/// One T-matrix with how it was solved.
#[derive(Clone, Debug)]
pub(crate) struct Solved {
    pub(crate) t: TMatrixModes<f64>,
    pub(crate) paths: Vec<PrecisionPath>,
    /// The estimated largest error of an element of T (absolute).
    pub(crate) error: f64,
    pub(crate) n_theta: usize,
}

impl Solved {
    fn largest(&self) -> f64 {
        self.t.largest()
    }
}

/// T of the modes m ≤ `m_max` at (N, n_θ) on the paths `precision` allows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    x: f64,
    eps: f64,
    s: Complex<f64>,
    n: usize,
    n_theta: usize,
    precision: Precision,
    rtol: f64,
    m_max: usize,
) -> Result<Solved> {
    let full_dd = || -> Result<Solved> {
        let sd = Complex::new(Dd::from(s.re), Dd::from(s.im));
        let t = tmatrix(Dd::from(x), Dd::from(eps), sd, n, n_theta, true)?;
        let modes = t
            .modes
            .into_iter()
            .map(|(ts, index)| {
                let ts = ts.map(|v| {
                    v.iter()
                        .map(|c| Complex::new(c.re.hi(), c.im.hi()))
                        .collect()
                });
                (ts, index)
            })
            .collect();
        let t = TMatrixModes { n, modes };
        let mut out = Solved {
            t,
            paths: vec![PrecisionPath::DoubleDouble; n + 1],
            error: 0.0,
            n_theta,
        };
        // The rounding to f64.
        out.error = out.largest() * f64::EPSILON;
        Ok(out)
    };
    if precision == Precision::DoubleDouble {
        return full_dd();
    }
    let tol = if precision == Precision::F64 {
        f64::INFINITY
    } else {
        rtol / 10.0
    };
    let (t, report) = tmatrix_refined(x, eps, s, n, n_theta, tol, m_max)?;
    let out = from_report(t, &report, n_theta);
    if precision == Precision::Auto && out.error > tol * out.largest() {
        return full_dd();
    }
    Ok(out)
}

fn from_report(t: TMatrixModes<f64>, report: &SolveReport, n_theta: usize) -> Solved {
    let paths = report
        .sweeps
        .iter()
        .map(|s| {
            if s[0] + s[1] > 0 {
                PrecisionPath::Refined
            } else {
                PrecisionPath::F64
            }
        })
        .collect();
    let error = report.error.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
    Solved {
        t,
        paths,
        error,
        n_theta,
    }
}

/// The settings of the convergence loop.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Loop {
    pub(crate) rtol: f64,
    pub(crate) precision: Precision,
    /// Fixed N, or the cap on N.
    pub(crate) n_fixed: Option<usize>,
    pub(crate) n_cap: usize,
    /// Fixed n_θ, or the cap on n_θ.
    pub(crate) n_theta_fixed: Option<usize>,
    pub(crate) n_theta_cap: usize,
    /// The Greek order that must settle, if any.
    pub(crate) greek_order: Option<usize>,
    /// Return the last step, with its estimate, instead of `NotConverged` at a cap.
    pub(crate) allow_unconverged: bool,
}

/// The result of the loop: the accepted T, and the last convergence change (relative).
#[derive(Clone, Debug)]
pub(crate) struct Converged {
    pub(crate) solved: Solved,
    pub(crate) change: f64,
    pub(crate) converged: bool,
}

/// x_max = k·max(a, c) for the equal-volume size parameter x: a = x ε^{1/3}, c = a/ε.
pub(crate) fn x_max(x: f64, eps: f64) -> f64 {
    let a = x * eps.cbrt();
    a.max(a / eps)
}

/// The starting N, Wiscombe's estimate at x_max, which the loop then grows:
/// N₀ = ⌈x_max + 4 x_max^{1/3}⌉ + 2, at least 4.
pub(crate) fn start_n(x: f64, eps: f64) -> usize {
    let xm = x_max(x, eps);
    ((xm + 4.0 * xm.cbrt()).ceil() as usize + 2).max(4)
}

/// n_θ tied to N: N(1 + |ln ε|) + 8 nodes in 0 < θ < π/2.
pub(crate) fn theta_of(n: usize, eps: f64) -> usize {
    (n as f64 * (1.0 + eps.ln().abs())).ceil() as usize + 8
}

fn relative_change(a: &Solved, b: &Solved) -> f64 {
    let (ea, sa) = a.t.cross_sections();
    let (eb, sb) = b.t.cross_sections();
    let rel = |u: f64, v: f64| {
        if u == v {
            0.0
        } else {
            (u - v).abs() / u.abs().max(v.abs())
        }
    };
    rel(ea, eb).max(rel(sa, sb))
}

fn greek_change(a: &Solved, b: &Solved, order: usize) -> Result<f64> {
    let ga = a.t.greek_raw(order)?;
    let gb = b.t.greek_raw(order)?;
    // Normalized by each one's α₁ at l = 0; the largest change relative to the largest α₁.
    let (na, nb) = (ga.alpha1[0], gb.alpha1[0]);
    let big = ga.alpha1.iter().fold(0.0f64, |m, v| m.max((v / na).abs()));
    let len = ga.alpha1.len().max(gb.alpha1.len());
    let at = |v: &[f64], l: usize, norm: f64| v.get(l).map_or(0.0, |x| x / norm);
    let mut worst = 0.0f64;
    for (va, vb) in [
        (&ga.alpha1, &gb.alpha1),
        (&ga.alpha2, &gb.alpha2),
        (&ga.alpha3, &gb.alpha3),
        (&ga.alpha4, &gb.alpha4),
        (&ga.beta1, &gb.beta1),
        (&ga.beta2, &gb.beta2),
    ] {
        for l in 0..len {
            worst = worst.max((at(va, l, na) - at(vb, l, nb)).abs());
        }
    }
    Ok(worst / big)
}

/// Runs the loop for the spheroid (x, ε) of index s.
pub(crate) fn converge(x: f64, eps: f64, s: Complex<f64>, opts: &Loop) -> Result<Converged> {
    let rtol = opts.rtol;
    let theta = |n: usize| {
        opts.n_theta_fixed
            .unwrap_or_else(|| theta_of(n, eps).min(opts.n_theta_cap))
    };
    let solve_at =
        |n: usize, nth: usize, m_max: usize| solve(x, eps, s, n, nth, opts.precision, rtol, m_max);
    let not_converged = |change: f64| Error::NotConverged {
        achieved: change,
        requested: rtol,
    };
    let mut change = 0.0f64;
    let mut converged = true;
    let mut stop = |change: f64| -> Result<()> {
        if opts.allow_unconverged {
            converged = false;
            Ok(())
        } else {
            Err(not_converged(change))
        }
    };
    // 1. N, on mode 0.
    let mut n = opts
        .n_fixed
        .unwrap_or_else(|| start_n(x, eps).min(opts.n_cap));
    let mut nth = theta(n);
    if opts.n_fixed.is_none() {
        let mut cur = solve_at(n, nth, 0)?;
        loop {
            if n >= opts.n_cap {
                stop(change)?;
                break;
            }
            let next = (n + (n / 8).max(2)).min(opts.n_cap);
            let trial = solve_at(next, theta(next), 0)?;
            change = relative_change(&cur, &trial);
            (n, nth, cur) = (next, theta(next), trial);
            if change < rtol {
                break;
            }
        }
        // 2. n_θ, on mode 0.
        if opts.n_theta_fixed.is_none() {
            loop {
                if nth >= opts.n_theta_cap {
                    stop(change)?;
                    break;
                }
                let next = (nth + (nth / 4).max(4)).min(opts.n_theta_cap);
                let trial = solve_at(n, next, 0)?;
                let c = relative_change(&cur, &trial);
                change = change.max(c);
                (nth, cur) = (next, trial);
                if c < rtol {
                    break;
                }
            }
        }
    } else if opts.n_theta_fixed.is_none() {
        let mut cur = solve_at(n, nth, 0)?;
        loop {
            if nth >= opts.n_theta_cap {
                stop(change)?;
                break;
            }
            let next = (nth + (nth / 4).max(4)).min(opts.n_theta_cap);
            let trial = solve_at(n, next, 0)?;
            let c = relative_change(&cur, &trial);
            change = change.max(c);
            (nth, cur) = (next, trial);
            if c < rtol {
                break;
            }
        }
    }
    // 3. All modes.
    let mut full = solve_at(n, nth, n)?;
    // 4. The Greek coefficients.
    if let Some(order) = opts.greek_order {
        let next = n + (n / 8).max(2);
        if opts.n_fixed.is_some() || next > opts.n_cap {
            // No step is possible: the order is checked against n_θ alone if it may grow.
            if opts.n_theta_fixed.is_none() {
                let nth2 = (nth + (nth / 4).max(4)).min(opts.n_theta_cap);
                let trial = solve_at(n, nth2, n)?;
                let c = greek_change(&full, &trial, order)?.max(relative_change(&full, &trial));
                change = change.max(c);
                full = trial;
                if c >= rtol {
                    stop(c)?;
                }
            }
        } else {
            let nth2 = (nth * next).div_ceil(n).min(opts.n_theta_cap);
            let nth2 = opts.n_theta_fixed.unwrap_or(nth2);
            let trial = solve_at(next, nth2, next)?;
            let c = greek_change(&full, &trial, order)?.max(relative_change(&full, &trial));
            change = change.max(c);
            full = trial;
            if c >= rtol {
                stop(c)?;
            }
        }
    }
    Ok(Converged {
        solved: full,
        change,
        converged,
    })
}

/// T with its derivatives at (N, n_θ) on the paths `precision` allows: the paths and the
/// estimated largest error of an element of T (absolute).
pub(crate) fn solve_derivatives(
    x: f64,
    eps: f64,
    s: Complex<f64>,
    n: usize,
    n_theta: usize,
    precision: Precision,
    rtol: f64,
) -> Result<(TMatrixModes<D4>, Vec<PrecisionPath>, f64)> {
    let full_dd = || -> Result<(TMatrixModes<D4>, Vec<PrecisionPath>, f64)> {
        type Dd4 = Dual<Dd, 4>;
        let [xd, ed, sr, si] =
            Dd4::variables([Dd::from(x), Dd::from(eps), Dd::from(s.re), Dd::from(s.im)]);
        let t = tmatrix(xd, ed, Complex::new(sr, si), n, n_theta, true)?;
        let round = |v: Dd4| D4::new(v.re().hi(), v.eps().map(|e| e.hi()));
        let modes = t
            .modes
            .into_iter()
            .map(|(ts, index)| {
                let ts = ts.map(|v| {
                    v.into_iter()
                        .map(|c| Complex::new(round(c.re), round(c.im)))
                        .collect()
                });
                (ts, index)
            })
            .collect();
        let t = TMatrixModes { n, modes };
        let big = t.values().largest();
        Ok((
            t,
            vec![PrecisionPath::DoubleDouble; n + 1],
            big * f64::EPSILON,
        ))
    };
    if precision == Precision::DoubleDouble {
        return full_dd();
    }
    let tol = if precision == Precision::F64 {
        f64::INFINITY
    } else {
        rtol / 10.0
    };
    let (t, report) = tmatrix_derivatives(x, eps, s, n, n_theta, tol)?;
    let solved = from_report(t.values(), &report, n_theta);
    if precision == Precision::Auto && solved.error > tol * solved.largest() {
        return full_dd();
    }
    Ok((t, solved.paths, solved.error))
}

impl TMatrixModes<D4> {
    /// The values of a T with derivatives.
    pub(crate) fn values(&self) -> TMatrixModes<f64> {
        let modes = self
            .modes
            .iter()
            .map(|(ts, index)| {
                let ts = ts.clone().map(|v| {
                    v.into_iter()
                        .map(|c| Complex::new(c.re.re(), c.im.re()))
                        .collect()
                });
                (ts, index.clone())
            })
            .collect();
        TMatrixModes { n: self.n, modes }
    }
}

impl TMatrixModes<f64> {
    fn largest(&self) -> f64 {
        self.modes
            .iter()
            .flat_map(|(ts, _)| ts.iter().flatten())
            .fold(0.0, |a, v| a.max(v.norm()))
    }
}
