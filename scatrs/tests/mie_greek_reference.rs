//! The phase matrix and the Greek coefficients against the Arb reference values in
//! `mie_greek.jsonl`.
//!
//! Every reference point is inside the validated domain, so every error must be within its
//! declared tolerance, with the scale and condition number of the error measure from
//! `common::greek`. The records are normalized, of the N-term series with the default `f64`
//! truncation N. Each test prints the worst C = error/((κ′ + cond)·2⁻⁵²) it measures.
//! `DoubleDouble` must be within the same tolerances times 16·2⁻⁵², as in
//! `tests/mie_reference.rs`.

mod common;

use common::Record;
use common::check::{Parts, Worst, error, max_nan, records, refractive_index, size_parameter};
use common::greek::{
    GREEK_NAMES, GreekOutputs, PHASE_NAMES, greek_derivative_measures, greek_reference,
    greek_value_measures, solve_greek,
};
use scatrs::mie::{MieOutput, Normalization, default_n_max};
use scatrs::scalar::DoubleDouble as Dd;

/// The order L of a record.
fn order(r: &Record) -> usize {
    r.input_i64("order") as usize
}

/// The f64 solution of the case of `r`: the default truncation, which the record states.
fn solve_f64(r: &Record, mu: &[f64]) -> GreekOutputs<f64> {
    let (x, m) = (size_parameter(r), refractive_index(r));
    let n = r.input_i64("n_terms") as usize;
    assert_eq!(
        n,
        default_n_max(x),
        "{}: not the default truncation",
        r.case
    );
    solve_greek(x, m, mu, None, Some(order(r)), Normalization::Normalized)
}

/// The double-double solution of the case of `r`, with the same number of terms.
fn solve_dd(r: &Record, mu: &[f64]) -> GreekOutputs<Dd> {
    let (x, m) = (size_parameter(r), refractive_index(r));
    let n = r.input_i64("n_terms") as usize;
    greek_reference(x, m, mu, n, Some(order(r)), Normalization::Normalized)
}

/// The Greek records and the phase-matrix records, paired by case.
fn cases() -> Vec<(Record, Record)> {
    let mut phases = records("mie_greek.jsonl", ",phase");
    records("mie_greek.jsonl", ",greek")
        .into_iter()
        .map(|g| {
            let case = g.case.trim_end_matches(",greek").to_owned();
            let i = phases
                .iter()
                .position(|p| p.case.trim_end_matches(",phase") == case)
                .expect("a phase record per case");
            (g, phases.remove(i))
        })
        .collect()
}

/// Checks `got` against the records with `worst`, in the measures of the double-double
/// solution `d`.
fn check<T: Parts>(
    got: &GreekOutputs<T>,
    d: &GreekOutputs<Dd>,
    g: &Record,
    p: &Record,
    worst_g: &mut Worst,
    worst_f: &mut Worst,
) {
    let (x, m) = (size_parameter(g), refractive_index(g));
    let measures = greek_value_measures(x, m, d);
    let output = MieOutput::GreekCoefficients { order: order(g) };
    for (k, name) in GREEK_NAMES.into_iter().enumerate() {
        assert_eq!(got.greek[k].len(), order(g) + 1);
        let err = (0..got.greek[k].len())
            .map(|l| error(got.greek[k][l], g, name, l))
            .fold(0.0, max_nan);
        worst_g.check(output, err, measures.greek[k], g, name);
    }
    let mu = p.input_f64s("mu");
    for (j, cosine) in mu.iter().enumerate() {
        for (k, name) in PHASE_NAMES.into_iter().enumerate() {
            let err = error(got.phase[k][j], p, name, j);
            let detail = format!("{name} at mu = {cosine}");
            worst_f.check(
                MieOutput::PhaseMatrix,
                err,
                measures.phase[j][k],
                p,
                &detail,
            );
        }
    }
}

#[test]
fn phase_matrix_and_greek_coefficients_match_arb() {
    let mut worst_g = Worst::new("Greek coefficients");
    let mut worst_f = Worst::new("phase matrix");
    for (g, p) in cases() {
        let mu = p.input_f64s("mu");
        let d = solve_dd(&g, &mu);
        check(&solve_f64(&g, &mu), &d, &g, &p, &mut worst_g, &mut worst_f);
    }
}

/// Every `DoubleDouble` value within 16·C·(κ′ + cond)·2⁻¹⁰⁴.
#[test]
fn double_double_phase_matrix_and_greek_coefficients_match_arb() {
    let mut worst_g = Worst::double_double("Greek coefficients");
    let mut worst_f = Worst::double_double("phase matrix");
    for (g, p) in cases() {
        let mu = p.input_f64s("mu");
        let d = solve_dd(&g, &mu);
        check(&d, &d, &g, &p, &mut worst_g, &mut worst_f);
    }
}

/// The unnormalized forms (per unit geometric cross section) and their derivatives within the
/// same tolerances, against double-double: the Arb records hold the normalized values only, and
/// the accuracy map is normalized.
#[test]
fn unnormalized_values_and_derivatives_are_within_the_tolerances() {
    let mut worst_g = Worst::new("unnormalized Greek coefficients");
    let mut worst_f = Worst::new("unnormalized phase matrix");
    let mut worst_dg = Worst::new("unnormalized Greek derivatives");
    let mut worst_df = Worst::new("unnormalized phase-matrix derivatives");
    let diff = |a: f64, b: Dd| (Dd::from_f64(a) - b).hi().abs();
    for (g, p) in cases() {
        let (x, m) = (size_parameter(&g), refractive_index(&g));
        let n = g.input_i64("n_terms") as usize;
        let mu = p.input_f64s("mu");
        let norm = Normalization::Unnormalized;
        let f = solve_greek(x, m, &mu, None, Some(order(&g)), norm);
        let d = greek_reference(x, m, &mu, n, Some(order(&g)), norm);
        let measures = greek_value_measures(x, m, &d);
        let output = MieOutput::GreekCoefficients { order: order(&g) };
        for (k, name) in GREEK_NAMES.into_iter().enumerate() {
            let err = (0..f.greek[k].len())
                .map(|l| diff(f.greek[k][l], d.greek[k][l]))
                .fold(0.0, max_nan);
            worst_g.check(output, err, measures.greek[k], &g, name);
        }
        for (j, cosine) in mu.iter().enumerate() {
            for (k, name) in PHASE_NAMES.into_iter().enumerate() {
                let err = diff(f.phase[k][j], d.phase[k][j]);
                let detail = format!("{name} at mu = {cosine}");
                worst_f.check(
                    MieOutput::PhaseMatrix,
                    err,
                    measures.phase[j][k],
                    &p,
                    &detail,
                );
            }
        }
        let dm = greek_derivative_measures(x, m, &mu, n, order(&g), norm, &d);
        let output = MieOutput::GreekDerivatives { order: order(&g) };
        for (q, key) in ["dx", "dm_re", "dm_im"].into_iter().enumerate() {
            for (k, name) in GREEK_NAMES.into_iter().enumerate() {
                let err = (0..f.greek_d[q][k].len())
                    .map(|l| diff(f.greek_d[q][k][l], d.greek_d[q][k][l]))
                    .fold(0.0, max_nan);
                let detail = format!("{name}_{key}");
                worst_dg.check(output, err, dm.greek[q][k], &g, &detail);
            }
            for (k, name) in PHASE_NAMES.into_iter().enumerate() {
                let err = (0..mu.len())
                    .map(|j| diff(f.phase_d[q][k][j], d.phase_d[q][k][j]))
                    .fold(0.0, max_nan);
                let detail = format!("{name}_{key}");
                let measure = dm.phase[q][k];
                worst_df.check(MieOutput::PhaseMatrixDerivatives, err, measure, &p, &detail);
            }
        }
    }
}
