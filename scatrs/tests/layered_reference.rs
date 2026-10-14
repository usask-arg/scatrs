//! Layered-sphere outputs against the Arb reference values in `layered.jsonl` and
//! `layered_derivatives.jsonl`.
//!
//! Every reference point is inside the validated domain, so every error must be within its
//! declared tolerance, `layered::VALIDATED_DOMAIN.tolerance(output, x_L, m, condition)`, with the
//! scale and condition number of the error measure from `common::layered`. Each test prints the
//! worst C = error/((κ′ + cond)·2⁻⁵²) it measures, against the declared constant. `DoubleDouble`
//! must be within the same tolerances times 16·2⁻⁵² (`DOUBLE_DOUBLE_SLACK`).

#![allow(clippy::needless_range_loop)] // parallel per-layer arrays are indexed together

mod common;

use common::Record;
use common::check::{DOUBLE_DOUBLE_SLACK, Parts, error, records};
use common::layered::{
    EFFICIENCY_NAMES, Measure, derivative_measures, q_of, reference, value_measures,
};
use num_complex::Complex;
use scatrs::layered::{Layered, VALIDATED_DOMAIN, ValidatedDomain};
use scatrs::mie::{AmplitudeBuffers, MieOptions, MieOutput, default_n_max};

const EFFICIENCY_OUTPUTS: [MieOutput; 5] = [
    MieOutput::Extinction,
    MieOutput::Scattering,
    MieOutput::Absorption,
    MieOutput::Backscattering,
    MieOutput::Asymmetry,
];

fn inputs(r: &Record) -> (Vec<f64>, Vec<Complex<f64>>) {
    let (re, im) = (r.input_f64s("m_re"), r.input_f64s("m_im"));
    (
        r.input_f64s("x"),
        re.iter()
            .zip(&im)
            .map(|(&a, &b)| Complex::new(a, b))
            .collect(),
    )
}

fn orders(r: &Record) -> Vec<usize> {
    r.input_i64s("orders").iter().map(|&n| n as usize).collect()
}

/// Enough terms for the largest stored order of `r`.
fn n_max(r: &Record, x: &[f64]) -> usize {
    let top = orders(r).last().copied().unwrap_or(0);
    default_n_max(x[x.len() - 1]).max(top)
}

fn complex_error<T: Parts>(got: Complex<T>, r: &Record, stem: &str, i: usize) -> f64 {
    error(got.re, r, &format!("{stem}_re"), i).hypot(error(got.im, r, &format!("{stem}_im"), i))
}

fn array_error<T: Parts>(r: &Record, stem: &str, got: &[Complex<T>]) -> f64 {
    orders(r)
        .iter()
        .enumerate()
        .map(|(i, &n)| complex_error(got[n - 1], r, stem, i))
        .fold(0.0, f64::max)
}

/// The largest measured C of one output, and where (as `common::check::Worst`, for layered
/// spheres).
struct Worst {
    what: String,
    unit: f64,
    slack: f64,
    c: f64,
    case: String,
}

impl Worst {
    fn new(what: &str) -> Self {
        Self {
            what: what.into(),
            unit: 1.0,
            slack: 1.0,
            c: 0.0,
            case: String::new(),
        }
    }

    fn double_double(what: &str) -> Self {
        Self {
            what: format!("DoubleDouble {what}"),
            unit: f64::EPSILON,
            slack: DOUBLE_DOUBLE_SLACK,
            c: 0.0,
            case: String::new(),
        }
    }

    fn check(&mut self, output: MieOutput, err: f64, measure: Measure, r: &Record, detail: &str) {
        let (x, m) = inputs(r);
        assert!(
            VALIDATED_DOMAIN.contains(&x, &m),
            "{} is outside the domain",
            r.case
        );
        let x_l = x[x.len() - 1];
        let tol =
            VALIDATED_DOMAIN.tolerance(output, x_l, &m, measure.condition) * self.unit * self.slack;
        assert!(
            measure.scale > 0.0 && tol.is_finite(),
            "{} {} {detail}: degenerate measure {measure:?}, tolerance {tol:e}",
            self.what,
            r.case
        );
        let rel = err / measure.scale;
        let c = rel / tol * self.slack * ValidatedDomain::constant(output);
        if c > self.c {
            self.c = c;
            self.case = format!("{} {detail}", r.case);
        }
        assert!(
            rel <= tol,
            "{} {} {detail}: error {rel:.3e} exceeds {tol:.3e} (C = {c:.2}, condition {:.3e})",
            self.what,
            r.case,
            measure.condition
        );
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            eprintln!(
                "{:<40} worst C = {:8.3} at {}",
                self.what, self.c, self.case
            );
        }
    }
}

#[test]
fn values_match_arb() {
    let mut lay = Layered::<f64>::new();
    let mut worst_c = Worst::new("a_n, b_n");
    let mut worst_q = EFFICIENCY_NAMES.map(Worst::new);
    let mut worst_s = Worst::new("S_1, S_2");
    let mut dd_c = Worst::double_double("a_n, b_n");
    let mut dd_q = EFFICIENCY_NAMES.map(Worst::double_double);
    let mut dd_s = Worst::double_double("S_1, S_2");
    let series = records("layered.jsonl", ",series");
    for c in records("layered.jsonl", ",coefficients") {
        let (x, m) = inputs(&c);
        let case = c.case.trim_end_matches(",coefficients");
        let r = series
            .iter()
            .find(|s| s.case.trim_end_matches(",series") == case)
            .expect("a series record per case");
        let mu = r.input_f64s("mu");
        let n = n_max(&c, &x);
        // DoubleDouble with its own truncation (the references sum the series far further).
        let d = reference(&x, &m, &mu, None);
        let measures = value_measures(&x, &m, &d, &orders(&c));
        let s = lay
            .compute(&x, &m, &MieOptions::new().with_n_max(n))
            .expect("inside the domain");
        for (k, (stem, got, want)) in [("a", s.a(), &d.a), ("b", s.b(), &d.b)]
            .into_iter()
            .enumerate()
        {
            let measure = measures.coefficients[k];
            worst_c.check(
                MieOutput::Coefficients,
                array_error(&c, stem, got),
                measure,
                &c,
                stem,
            );
            dd_c.check(
                MieOutput::Coefficients,
                array_error(&c, stem, want),
                measure,
                &c,
                stem,
            );
        }
        // The series with the default truncation.
        let s = lay
            .compute(&x, &m, &MieOptions::new())
            .expect("inside the domain");
        let q = s.efficiencies();
        let got = [q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g];
        let real = m.iter().all(|v| v.im == 0.0);
        for k in 0..5 {
            let name = EFFICIENCY_NAMES[k];
            if k == 2 && real {
                assert_eq!(
                    got[k], 0.0,
                    "{}: Q_abs must be exactly 0 for real m",
                    r.case
                );
                continue;
            }
            let measure = measures.efficiencies[k];
            worst_q[k].check(
                EFFICIENCY_OUTPUTS[k],
                error(got[k], r, name, 0),
                measure,
                r,
                name,
            );
            let want = q_of(&d.q, k);
            dd_q[k].check(
                EFFICIENCY_OUTPUTS[k],
                error(want, r, name, 0),
                measure,
                r,
                name,
            );
        }
        let zero = Complex::new(0.0, 0.0);
        let (mut s1, mut s2) = (vec![zero; mu.len()], vec![zero; mu.len()]);
        s.amplitudes(
            &mu,
            AmplitudeBuffers {
                s1: &mut s1,
                s2: &mut s2,
            },
        )
        .expect("valid cosines");
        for j in 0..mu.len() {
            for (i, (stem, v)) in [("s1", s1[j]), ("s2", s2[j])].into_iter().enumerate() {
                let detail = format!("{stem} at mu = {}", mu[j]);
                let measure = measures.amplitudes[j][i];
                worst_s.check(
                    MieOutput::Amplitudes,
                    complex_error(v, r, stem, j),
                    measure,
                    r,
                    &detail,
                );
                let want = complex_error(d.s[i][j], r, stem, j);
                dd_s.check(MieOutput::Amplitudes, want, measure, r, &detail);
            }
        }
    }
}

#[test]
fn derivatives_match_arb() {
    let mut lay = Layered::<f64>::new();
    let mut worst_c = Worst::new("coefficient derivatives");
    let mut worst_j = Worst::new("efficiency Jacobian");
    let mut dd_c = Worst::double_double("coefficient derivatives");
    let mut dd_j = Worst::double_double("efficiency Jacobian");
    for r in records("layered_derivatives.jsonl", "") {
        let (x, m) = inputs(&r);
        let n = n_max(&r, &x);
        let d = reference(&x, &m, &[], None);
        let measures = derivative_measures(&x, &m, &d, &orders(&r));
        let s = lay
            .compute_with_derivatives(&x, &m, &MieOptions::new().with_n_max(n))
            .expect("inside the domain");
        let jac = s.efficiencies_jacobian().expect("computed");
        for l in 0..x.len() {
            let dc = s.derivatives(l).expect("computed");
            let got = [dc.da_dx, dc.da_dm, dc.db_dx, dc.db_dm];
            for (k, stem) in ["da_dx", "da_dm", "db_dx", "db_dm"].into_iter().enumerate() {
                let stem = format!("{stem}{l}");
                let measure = measures.coefficients[l][k];
                let out = MieOutput::CoefficientDerivatives;
                worst_c.check(out, array_error(&r, &stem, got[k]), measure, &r, &stem);
                dd_c.check(out, array_error(&r, &stem, &d.d[l][k]), measure, &r, &stem);
            }
            let real = m.iter().all(|v| v.im == 0.0);
            for (p, key) in ["dx", "dm_re", "dm_im"].into_iter().enumerate() {
                let dq = [jac[l].x, jac[l].m_re, jac[l].m_im][p];
                let got = [dq.q_ext, dq.q_sca, dq.q_abs, dq.q_back, dq.g];
                for k in 0..5 {
                    let name = format!("{}_{key}{l}", EFFICIENCY_NAMES[k]);
                    if k == 2 && real && p < 2 {
                        assert_eq!(got[k], 0.0, "{}: {name} must be exactly 0", r.case);
                        continue;
                    }
                    let measure = measures.jacobian[l][p][k];
                    let out = MieOutput::EfficiencyJacobian;
                    worst_j.check(out, error(got[k], &r, &name, 0), measure, &r, &name);
                    let want = q_of(&d.jac[l][p], k);
                    dd_j.check(out, error(want, &r, &name, 0), measure, &r, &name);
                }
            }
        }
    }
}
