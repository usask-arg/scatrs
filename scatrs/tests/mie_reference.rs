//! Mie outputs against the Arb reference values in `mie.jsonl` and `mie_derivatives.jsonl`.
//!
//! Every reference point is inside the validated domain, so every error must be within its
//! declared tolerance, `VALIDATED_DOMAIN.tolerance(output, x, m, condition)`, with the scale and
//! condition number of the error measure from `common::mie`. Each test prints the worst
//! C = error/((κ′ + cond)·2⁻⁵²) it measures, against the declared constant.

mod common;

use common::mie::{EFFICIENCY_NAMES, Measure, derivative_measures, reference, value_measures};
use common::{Record, load};
use num_complex::Complex;
use scatrs::mie::{
    AmplitudeBuffers, Mie, MieOptions, MieOutput, VALIDATED_DOMAIN, ValidatedDomain, default_n_max,
};

/// The measures of the efficiencies, in the order of `EFFICIENCY_NAMES`.
const EFFICIENCY_OUTPUTS: [MieOutput; 5] = [
    MieOutput::Extinction,
    MieOutput::Scattering,
    MieOutput::Absorption,
    MieOutput::Backscattering,
    MieOutput::Asymmetry,
];

fn size_parameter(r: &Record) -> f64 {
    r.input_f64("x")
}

fn refractive_index(r: &Record) -> Complex<f64> {
    Complex::new(r.input_f64("m_re"), r.input_f64("m_im"))
}

/// |got − v| for the reference value i of `name`.
fn error(got: f64, r: &Record, name: &str, i: usize) -> f64 {
    r.output(name).dd_error(i, got, 0.0)
}

fn complex_error(got: Complex<f64>, r: &Record, stem: &str, i: usize) -> f64 {
    error(got.re, r, &format!("{stem}_re"), i).hypot(error(got.im, r, &format!("{stem}_im"), i))
}

/// The largest measured C of one output, and where.
struct Worst {
    what: String,
    c: f64,
    case: String,
}

impl Worst {
    fn new(what: &str) -> Self {
        Self {
            what: what.to_owned(),
            c: 0.0,
            case: String::new(),
        }
    }

    /// Checks the error `err` of `output` at the case of `r` against the declared tolerance in
    /// the output's measure, and records its C.
    fn check(&mut self, output: MieOutput, err: f64, measure: Measure, r: &Record, detail: &str) {
        let (x, m) = (size_parameter(r), refractive_index(r));
        assert!(
            VALIDATED_DOMAIN.contains(x, m),
            "{} is outside the domain",
            r.case
        );
        let tol = VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition);
        let rel = err / measure.scale;
        let c = rel / tol * ValidatedDomain::constant(output);
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
                "{:<26} worst C = {:8.3} at {}",
                self.what, self.c, self.case
            );
        }
    }
}

fn records(file: &str, kind: &str) -> Vec<Record> {
    load(file)
        .into_iter()
        .filter(|r| r.case.ends_with(kind))
        .collect()
}

/// The stored orders of a coefficient record (1-based).
fn orders(r: &Record) -> Vec<usize> {
    r.input_i64s("orders").iter().map(|&n| n as usize).collect()
}

/// Options with enough terms for the largest stored order of coefficient record `r`.
fn options(r: &Record) -> MieOptions {
    let top = *orders(r).last().expect("orders");
    MieOptions::new().with_n_max(default_n_max(size_parameter(r)).max(top))
}

/// max over the stored orders of |error|, for one array.
fn array_error(r: &Record, stem: &str, got: &[Complex<f64>]) -> f64 {
    orders(r)
        .iter()
        .enumerate()
        .map(|(i, &n)| complex_error(got[n - 1], r, stem, i))
        .fold(0.0, f64::max)
}

#[test]
fn coefficients_match_arb() {
    let mut mie = Mie::<f64>::new();
    let mut worst = Worst::new("a_n, b_n");
    for r in records("mie.jsonl", ",coefficients") {
        let (x, m) = (size_parameter(&r), refractive_index(&r));
        let s = mie.compute(x, m, &options(&r)).expect("inside the domain");
        let measures = value_measures(x, m, &reference(x, m, &[]), &orders(&r));
        for (k, (stem, got)) in [("a", s.a()), ("b", s.b())].into_iter().enumerate() {
            let err = array_error(&r, stem, got);
            worst.check(
                MieOutput::Coefficients,
                err,
                measures.coefficients[k],
                &r,
                stem,
            );
        }
    }
}

#[test]
fn efficiencies_and_amplitudes_match_arb() {
    let mut mie = Mie::<f64>::new();
    let mut worst = EFFICIENCY_NAMES.map(Worst::new);
    let mut worst_s = Worst::new("S_1, S_2");
    for r in records("mie.jsonl", ",series") {
        let (x, m) = (size_parameter(&r), refractive_index(&r));
        let s = mie
            .compute(x, m, &MieOptions::new())
            .expect("inside the domain");
        let q = s.efficiencies();
        let mu = r.input_f64s("mu");
        let measures = value_measures(x, m, &reference(x, m, &mu), &[1]);
        let got = [q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g];
        for k in 0..5 {
            let name = EFFICIENCY_NAMES[k];
            if k == 2 && m.im == 0.0 {
                assert_eq!(
                    got[k], 0.0,
                    "{}: Q_abs must be exactly 0 for real m",
                    r.case
                );
                continue;
            }
            let err = error(got[k], &r, name, 0);
            worst[k].check(
                EFFICIENCY_OUTPUTS[k],
                err,
                measures.efficiencies[k],
                &r,
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
                let err = complex_error(v, &r, stem, j);
                let measure = measures.amplitudes[j][i];
                worst_s.check(MieOutput::Amplitudes, err, measure, &r, &detail);
            }
        }
    }
}

#[test]
fn derivatives_match_arb() {
    let mut mie = Mie::<f64>::new();
    let mut worst_c = Worst::new("coefficient derivatives");
    let mut worst_j = Worst::new("efficiency Jacobian");
    let mut worst_s = Worst::new("amplitude derivatives");
    let coefficient_records = records("mie_derivatives.jsonl", ",coefficients");
    for r in records("mie_derivatives.jsonl", ",series") {
        let (x, m) = (size_parameter(&r), refractive_index(&r));
        let case = r.case.trim_end_matches(",series");
        let c = coefficient_records
            .iter()
            .find(|c| c.case.trim_end_matches(",coefficients") == case)
            .expect("a coefficient record per case");
        let mu = r.input_f64s("mu");
        let d = reference(x, m, &mu);
        let measures = derivative_measures(x, m, &mu, &d, &orders(c));
        let s = mie
            .compute_with_derivatives(x, m, &options(c))
            .expect("inside the domain");
        let dc = s.derivatives().expect("computed");
        let arrays = [
            ("da_dx", dc.da_dx),
            ("da_dm", dc.da_dm),
            ("db_dx", dc.db_dx),
            ("db_dm", dc.db_dm),
        ];
        for (k, (stem, got)) in arrays.into_iter().enumerate() {
            let err = array_error(c, stem, got);
            let measure = measures.coefficients[k];
            worst_c.check(MieOutput::CoefficientDerivatives, err, measure, c, stem);
        }
        let jac = s.efficiencies_jacobian().expect("computed");
        for (p, (key, dq)) in [("dx", jac.x), ("dm_re", jac.m_re), ("dm_im", jac.m_im)]
            .into_iter()
            .enumerate()
        {
            let got = [dq.q_ext, dq.q_sca, dq.q_abs, dq.q_back, dq.g];
            for k in 0..5 {
                let name = format!("{}_{key}", EFFICIENCY_NAMES[k]);
                if k == 2 && m.im == 0.0 && p < 2 {
                    assert_eq!(
                        got[k], 0.0,
                        "{}: {name} must be exactly 0 for real m",
                        r.case
                    );
                    continue;
                }
                let err = error(got[k], &r, &name, 0);
                let measure = measures.jacobian[p][k];
                worst_j.check(MieOutput::EfficiencyJacobian, err, measure, &r, &name);
            }
        }
        let zero = Complex::new(0.0, 0.0);
        let mut v: [Vec<Complex<f64>>; 6] = std::array::from_fn(|_| vec![zero; mu.len()]);
        {
            let [s1, s2, dx1, dx2, dm1, dm2] = &mut v;
            s.amplitudes_with_derivatives(
                &mu,
                AmplitudeBuffers { s1, s2 },
                AmplitudeBuffers { s1: dx1, s2: dx2 },
                AmplitudeBuffers { s1: dm1, s2: dm2 },
            )
            .expect("valid cosines");
        }
        for (p, key) in ["dx", "dm"].into_iter().enumerate() {
            for (i, stem) in ["s1", "s2"].into_iter().enumerate() {
                let name = format!("{stem}_{key}");
                let err = (0..mu.len())
                    .map(|j| complex_error(v[2 + 2 * p + i][j], &r, &name, j))
                    .fold(0.0, f64::max);
                let measure = measures.amplitudes[p][i];
                worst_s.check(MieOutput::AmplitudeDerivatives, err, measure, &r, &name);
            }
        }
    }
}

/// `DoubleDouble` stands in for the exact values in the accuracy map, so it must be accurate
/// far beyond `f64`: the efficiencies within 16·(κ + cond)·2⁻¹⁰⁴ in their measures.
#[test]
fn double_double_matches_arb() {
    let u2 = 2f64.powi(-104);
    let mut worst = 0.0f64;
    let mut at = String::new();
    for r in records("mie.jsonl", ",series") {
        let (x, m) = (size_parameter(&r), refractive_index(&r));
        let d = reference(x, m, &[]);
        let measures = value_measures(x, m, &d, &[1]);
        let kappa = ValidatedDomain::condition_factor(x, m);
        let got = [d.q.q_ext, d.q.q_sca, d.q.q_abs, d.q.q_back, d.q.g];
        for k in [0, 1, 3, 4] {
            let measure = measures.efficiencies[k];
            let name = EFFICIENCY_NAMES[k];
            let err = r.output(name).dd_error(0, got[k].hi(), got[k].lo());
            let c = err / measure.scale / ((kappa + measure.condition) * u2);
            if c > worst {
                worst = c;
                at = format!("{} {name}", r.case);
            }
            assert!(c <= 16.0, "{} {name}: C = {c:.2}", r.case);
        }
    }
    eprintln!("DoubleDouble efficiencies  worst C = {worst:8.3} at {at}");
}
