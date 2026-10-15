//! Randomly oriented spheroids against the Arb reference values in `spheroid.jsonl`, from an
//! EBCM computed from its definition with no simplification (no code or algorithm shared with
//! the kernel).
//!
//! * `fixed` records, at a given N and n_θ, isolate the solve: on every precision path the
//!   efficiencies and the sampled T-matrix elements must be within C times the solution's own
//!   error estimate (relative to the largest element of T), C = 4, and at least within a few
//!   rounding units.
//! * `converged` records check the convergence loop: Q_ext, Q_sca and Q_abs within the declared
//!   tolerance C·max(`rtol`, estimate) of `tmatrix::VALIDATED_DOMAIN` (Q_abs relative to Q_ext).
//!
//! Each test prints the worst ratio of error to tolerance it measures.

mod common;

use common::Record;
use common::check::{error, records};
use num_complex::Complex;
use scatrs::mie::MieOutput;
use scatrs::tmatrix::{Precision, Shape, TMatrix, TMatrixOptions, VALIDATED_DOMAIN};

const FILE: &str = "spheroid.jsonl";

fn inputs(r: &Record) -> (f64, f64, Complex<f64>) {
    (
        r.input_f64("aspect_ratio"),
        r.input_f64("x"),
        Complex::new(r.input_f64("m_re"), r.input_f64("m_im")),
    )
}

struct Worst {
    what: String,
    ratio: f64,
    case: String,
}

impl Worst {
    fn new(what: &str) -> Self {
        Self {
            what: what.into(),
            ratio: 0.0,
            case: String::new(),
        }
    }

    fn check(&mut self, err: f64, tol: f64, case: &str) {
        let ratio = err / tol;
        if ratio > self.ratio {
            self.ratio = ratio;
            self.case = case.into();
        }
        assert!(
            err <= tol,
            "{} {case}: error {err:.3e} exceeds {tol:.3e}",
            self.what
        );
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            eprintln!(
                "{:<40} worst error/tolerance = {:.3} at {}",
                self.what, self.ratio, self.case
            );
        }
    }
}

#[test]
fn fixed_truncations_on_every_path() {
    for (path, name) in [
        (Precision::F64, "f64"),
        (Precision::Auto, "auto"),
        (Precision::DoubleDouble, "double-double"),
    ] {
        let mut q_worst = Worst::new(&format!("fixed Q, {name}"));
        let mut t_worst = Worst::new(&format!("fixed T, {name}"));
        for r in common::load(FILE)
            .into_iter()
            .filter(|r| r.case.contains(",N="))
        {
            let (eps, x, m) = inputs(&r);
            assert!(VALIDATED_DOMAIN.contains(eps, x, m), "{} outside", r.case);
            let n = r.input_i64("n_terms") as usize;
            let nth = r.input_i64("n_theta") as usize;
            let opts = TMatrixOptions::new()
                .with_precision(path)
                .with_n_terms(n)
                .with_n_theta(nth);
            let mut tm = TMatrix::new();
            let s = tm
                .compute(Shape::Spheroid { aspect_ratio: eps }, x, m, &opts)
                .unwrap();
            // The estimate, with a floor of a few rounding units.
            let est = s.error_estimate().max(16.0 * f64::EPSILON);
            let q = s.efficiencies();
            for (v, out) in [(q.q_ext, "q_ext"), (q.q_sca, "q_sca")] {
                let e = error(v, &r, out, 0);
                let scale = r.output(out).hi[0].abs();
                q_worst.check(e, 4.0 * est * scale, &format!("{} {out}", r.case));
            }
            // T: relative to the largest element of T.
            let big = (0..=n)
                .filter_map(|mm| s.block(mm))
                .flat_map(|b| {
                    [b.t11, b.t12, b.t21, b.t22]
                        .into_iter()
                        .flatten()
                        .map(|v| v.norm())
                        .collect::<Vec<_>>()
                })
                .fold(0.0f64, f64::max);
            for mm in [0, 1, n / 2] {
                let b = s.block(mm).unwrap();
                let k = b.orders.len();
                let at = |i: usize, j: usize| -> Complex<f64> {
                    let (blk, ii, jj) = match (i < k, j < k) {
                        (true, true) => (&b.t11, i, j),
                        (true, false) => (&b.t12, i, j - k),
                        (false, true) => (&b.t21, i - k, j),
                        (false, false) => (&b.t22, i - k, j - k),
                    };
                    blk[ii * k + jj]
                };
                let mut check = |stem: &str, got: &[Complex<f64>]| {
                    for (i, v) in got.iter().enumerate() {
                        let e = error(v.re, &r, &format!("t{mm}_{stem}_re"), i).hypot(error(
                            v.im,
                            &r,
                            &format!("t{mm}_{stem}_im"),
                            i,
                        ));
                        t_worst.check(
                            e,
                            4.0 * est * big,
                            &format!("{} mode {mm} {stem} {i}", r.case),
                        );
                    }
                };
                check("diag11", &(0..k).map(|i| at(i, i)).collect::<Vec<_>>());
                check(
                    "diag22",
                    &(0..k).map(|i| at(k + i, k + i)).collect::<Vec<_>>(),
                );
                let col: Vec<Complex<f64>> = (0..2 * k)
                    .map(|i| at(i, 0))
                    .chain((0..2 * k).map(|i| at(i, k)))
                    .collect();
                check("col", &col);
            }
        }
    }
}

#[test]
fn converged_efficiencies() {
    let mut worst = Worst::new("converged Q_ext, Q_sca, Q_abs");
    for r in records(FILE, ",converged") {
        let (eps, x, m) = inputs(&r);
        let opts = TMatrixOptions::new();
        let mut tm = TMatrix::new();
        let s = tm
            .compute(Shape::Spheroid { aspect_ratio: eps }, x, m, &opts)
            .unwrap();
        let est = s.error_estimate();
        let q = s.efficiencies();
        let q_ext = r.output("q_ext").hi[0];
        for (v, out, kind, scale) in [
            (q.q_ext, "q_ext", MieOutput::Extinction, q_ext),
            (
                q.q_sca,
                "q_sca",
                MieOutput::Scattering,
                r.output("q_sca").hi[0],
            ),
            (q.q_abs, "q_abs", MieOutput::Absorption, q_ext),
        ] {
            let e = error(v, &r, out, 0);
            let tol = VALIDATED_DOMAIN.tolerance(kind, opts.rtol, est) * scale.abs();
            worst.check(e, tol, &format!("{} {out}", r.case));
        }
    }
}
