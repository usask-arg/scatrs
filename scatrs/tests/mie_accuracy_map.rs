//! The accuracy map of the Mie outputs: `f64` against `DoubleDouble` on a dense grid over the
//! validated domain, every output within its declared tolerance.
//!
//! The grid: x = d·10^e for the mantissas `MANTISSAS` and e = −6, …, 4, up to 2·10⁴, 104
//! values (decimal values, so the same doubles on every platform: a libm-dependent grid could
//! land next to a sharp resonance on one platform only), against the refractive indices
//! `INDICES`, with the amplitudes at `cosines()`. `DoubleDouble` is within 16 times the
//! declared constants, in units of (κ′ + cond)·2⁻¹⁰⁴, of Arb on every output
//! (tests/mie_reference.rs), so it stands in for the exact values.
//!
//! Each error is reported as C = error/((κ′ + cond)·2⁻⁵²), the constant its tolerance would
//! need; the test prints the worst C of each output. With `SCATRS_ACCURACY_MAP=path` it also
//! writes every C as JSON, which the documentation plots (docs/validation/accuracy.md):
//!
//! ```text
//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/mie-accuracy-map.json \
//!     cargo test -p scatrs --release --test mie_accuracy_map -- --nocapture
//! ```

mod common;

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::mie::{Measure, Outputs, derivative_measures, q_of, reference, solve, value_measures};
use num_complex::Complex;
use scatrs::mie::{MieOutput, VALIDATED_DOMAIN, ValidatedDomain};
use scatrs::scalar::DoubleDouble as Dd;

/// About ten size parameters per decade.
const MANTISSAS: [&str; 10] = ["1", "1.25", "1.6", "2", "2.5", "3.2", "4", "5", "6.4", "8"];

/// (Re m, Im m): near 1, aerosol and water, high index, strong absorption and metal-like, below
/// 1 (an air bubble in water, down to the domain's |m| = 0.1), at the edges of the domain.
const INDICES: [(f64, f64); 28] = [
    (1.0001, 0.0),
    (1.001, 1e-6),
    (1.01, 0.0),
    (1.1, 1e-3),
    (1.33, 0.0),
    (1.33, 1e-8),
    (1.33, 1e-4),
    (1.5, 0.0),
    (1.5, 1e-2),
    (1.5, 0.1),
    (1.5, 1.0),
    (1.5, 10.0),
    (1.75, 0.44),
    (2.0, 0.0),
    (2.5, 1.0),
    (3.0, 1e-3),
    (4.0, 0.01),
    (5.0, 5.0),
    (8.0, 0.1),
    (10.0, 0.0),
    (10.0, 10.0),
    (0.5, 2.0),
    (0.2, 3.0),
    (0.05, 4.0),
    (0.75, 0.0),
    (0.5, 1e-4),
    (0.3, 0.6),
    (0.1, 0.0),
];

/// The columns of the map.
const OUTPUTS: [(&str, MieOutput); 10] = [
    ("coefficients", MieOutput::Coefficients),
    ("q_ext", MieOutput::Extinction),
    ("q_sca", MieOutput::Scattering),
    ("q_abs", MieOutput::Absorption),
    ("q_back", MieOutput::Backscattering),
    ("g", MieOutput::Asymmetry),
    ("amplitudes", MieOutput::Amplitudes),
    ("coefficient_derivatives", MieOutput::CoefficientDerivatives),
    ("efficiency_jacobian", MieOutput::EfficiencyJacobian),
    ("amplitude_derivatives", MieOutput::AmplitudeDerivatives),
];

fn size_parameters() -> Vec<f64> {
    (-6..=4)
        .flat_map(|e| MANTISSAS.iter().map(move |d| format!("{d}e{e}")))
        .map(|s| s.parse::<f64>().expect("a decimal literal"))
        .filter(|&x| x <= VALIDATED_DOMAIN.x_max)
        .collect()
}

/// −1 + k/16 for k = 0, …, 32, and 1 − 2^-j for j = 6, 8, …, 20 in the forward peak: all
/// exact doubles.
fn cosines() -> Vec<f64> {
    let uniform = (0..=32).map(|k| -1.0 + f64::from(k) / 16.0);
    let forward = (3..=10).map(|j| 1.0 - 2f64.powi(-2 * j));
    uniform.chain(forward).collect()
}

fn diff(a: f64, b: Dd) -> f64 {
    (Dd::from_f64(a) - b).hi().abs()
}

fn cdiff(a: Complex<f64>, b: Complex<Dd>) -> f64 {
    diff(a.re, b.re).hypot(diff(a.im, b.im))
}

/// max over n < `f.len()` of |f − d|.
fn array_error(f: &[Complex<f64>], d: &[Complex<Dd>]) -> f64 {
    f.iter()
        .zip(d)
        .map(|(&a, &b)| cdiff(a, b))
        .fold(0.0, f64::max)
}

/// C of every output at (x, m), ordered as `OUTPUTS`.
fn point(x: f64, m: Complex<f64>, mu: &[f64]) -> [f64; 10] {
    let f: Outputs<f64> = solve(x, m, mu, None);
    let d = reference(x, m, mu);
    let orders: Vec<usize> = (1..=f.a.len()).collect();
    let v = value_measures(x, m, &d, &orders);
    let dm = derivative_measures(x, m, mu, &d, &orders);
    let c = |output: MieOutput, err: f64, measure: Measure| {
        let tol = VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition);
        assert!(
            measure.scale > 0.0 && tol.is_finite(),
            "{output:?} at x = {x:e}, m = {m}: degenerate measure {measure:?}, tolerance {tol:e}"
        );
        err / measure.scale / tol * ValidatedDomain::constant(output)
    };
    let mut out = [0.0f64; 10];
    // f64::max would drop a NaN: count it as a failure instead.
    let mut put = |k: usize, value: f64| {
        out[k] = out[k].max(if value.is_nan() { f64::INFINITY } else { value });
    };
    put(
        0,
        c(
            MieOutput::Coefficients,
            array_error(&f.a, &d.a),
            v.coefficients[0],
        ),
    );
    put(
        0,
        c(
            MieOutput::Coefficients,
            array_error(&f.b, &d.b),
            v.coefficients[1],
        ),
    );
    for k in 0..5 {
        let output = OUTPUTS[1 + k].1;
        if k == 2 && m.im == 0.0 {
            // Q_abs is exactly 0 for real m.
            put(3, if f.q.q_abs == 0.0 { 0.0 } else { f64::INFINITY });
            continue;
        }
        put(
            1 + k,
            c(
                output,
                diff(q_of(&f.q, k), q_of(&d.q, k)),
                v.efficiencies[k],
            ),
        );
    }
    for j in 0..mu.len() {
        for i in 0..2 {
            let err = cdiff(f.s[i][j], d.s[i][j]);
            put(6, c(MieOutput::Amplitudes, err, v.amplitudes[j][i]));
        }
    }
    for k in 0..4 {
        let err = array_error(&f.d[k], &d.d[k]);
        put(
            7,
            c(MieOutput::CoefficientDerivatives, err, dm.coefficients[k]),
        );
    }
    for p in 0..3 {
        for k in 0..5 {
            if k == 2 && m.im == 0.0 && p < 2 {
                let exact = q_of(&f.jac[p], k) == 0.0;
                put(8, if exact { 0.0 } else { f64::INFINITY });
                continue;
            }
            let err = diff(q_of(&f.jac[p], k), q_of(&d.jac[p], k));
            put(8, c(MieOutput::EfficiencyJacobian, err, dm.jacobian[p][k]));
        }
    }
    for p in 0..2 {
        for i in 0..2 {
            let k = 2 + 2 * p + i;
            let err = (0..mu.len())
                .map(|j| cdiff(f.s[k][j], d.s[k][j]))
                .fold(0.0, f64::max);
            put(
                9,
                c(MieOutput::AmplitudeDerivatives, err, dm.amplitudes[p][i]),
            );
        }
    }
    out
}

fn to_json(xs: &[f64], results: &[[f64; 10]]) -> String {
    let list = |v: Vec<String>| v.join(", ");
    let mut json = String::from("{\n");
    let _ = writeln!(
        json,
        "  \"x\": [{}],",
        list(xs.iter().map(|x| format!("{x:e}")).collect())
    );
    let indices = INDICES
        .iter()
        .map(|(r, i)| format!("[{r:e}, {i:e}]"))
        .collect();
    let _ = writeln!(json, "  \"m\": [{}],", list(indices));
    let declared = OUTPUTS
        .iter()
        .map(|(n, o)| format!("\"{n}\": {}", ValidatedDomain::constant(*o)))
        .collect();
    let _ = writeln!(json, "  \"declared\": {{{}}},", list(declared));
    json += "  \"c\": {\n";
    for (o, (name, _)) in OUTPUTS.iter().enumerate() {
        let rows = (0..xs.len())
            .map(|i| {
                let row = (0..INDICES.len())
                    .map(|j| match results[i * INDICES.len() + j][o] {
                        c if c.is_finite() => format!("{c:.3e}"),
                        _ => "null".to_owned(),
                    })
                    .collect();
                format!("[{}]", list(row))
            })
            .collect();
        let comma = if o + 1 < OUTPUTS.len() { "," } else { "" };
        let _ = writeln!(json, "    \"{name}\": [{}]{comma}", list(rows));
    }
    json + "  }\n}\n"
}

#[test]
fn accuracy_map_is_within_the_declared_tolerances() {
    let xs = size_parameters();
    assert_eq!(xs.len(), 104, "size parameters");
    let mu = cosines();
    let points: Vec<(f64, Complex<f64>)> = xs
        .iter()
        .flat_map(|&x| INDICES.iter().map(move |&(r, i)| (x, Complex::new(r, i))))
        .collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut results = vec![[0.0f64; 10]; points.len()];
    let done: Vec<Vec<(usize, [f64; 10])>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(x, m)) = points.get(k) else {
                            break mine;
                        };
                        mine.push((k, point(x, m, &mu)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("worker"))
            .collect()
    });
    let done: Vec<(usize, [f64; 10])> = done.into_iter().flatten().collect();
    assert_eq!(done.len(), points.len(), "every point computed");
    for (k, c) in done {
        results[k] = c;
    }
    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
        std::fs::write(&path, to_json(&xs, &results)).expect("write the map");
        eprintln!("wrote {path}");
    }
    let mut failures = Vec::new();
    for (o, (name, output)) in OUTPUTS.iter().enumerate() {
        let (k, worst) = results
            .iter()
            .enumerate()
            .map(|(k, c)| (k, c[o]))
            .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let declared = ValidatedDomain::constant(*output);
        let (x, m) = points[k];
        eprintln!("{name:<24} worst C = {worst:8.3} (declared {declared}) at x = {x:e}, m = {m}");
        if worst.is_nan() || worst > declared {
            failures.push(*name);
        }
    }
    assert!(
        failures.is_empty(),
        "outside the declared tolerance: {failures:?}"
    );
}
