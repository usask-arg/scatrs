//! The accuracy map of the layered-sphere outputs: `f64` against `DoubleDouble` on a dense grid
//! over the validated domain, every output within its declared tolerance.
//!
//! The grid: the outer size parameters of the sphere map (x_L = d·10^e, up to the domain's 10⁴,
//! 101 values) against `PARTICLES`: two-layer particles of the pairs of indices `PAIRS` with the
//! core fractions `FRACTIONS`, and three- and ten-layer particles. The amplitudes are at the
//! sphere map's cosines. `DoubleDouble` is within 16 times the declared constants, in units of
//! (κ′ + cond)·2⁻¹⁰⁴, of Arb on every output (tests/layered_reference.rs), so it stands in for
//! the exact values. The derivative columns cover the derivatives' domain (x_L ≤ 1000) for the
//! particles of at most four layers (the ten-layer particle takes other passes of the same code,
//! which tests/layered.rs checks against differences); above it they are null.
//!
//! Each error is reported as C = error/((κ′ + cond)·2⁻⁵²); the test prints the worst C of each
//! output. With `SCATRS_ACCURACY_MAP=path` it also writes every C as JSON, which the
//! documentation plots (docs/validation/accuracy.md):
//!
//! ```text
//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/layered-accuracy-map.json \
//!     cargo test -p scatrs --release --test layered_accuracy_map -- --nocapture
//! ```
#![allow(clippy::needless_range_loop)] // parallel per-layer arrays are indexed together

mod common;

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::layered::{
    Measure, Outputs, derivative_measures, q_of, reference, solve, value_measures,
};
use num_complex::Complex;
use scatrs::layered::{VALIDATED_DOMAIN, ValidatedDomain};
use scatrs::mie::MieOutput;
use scatrs::scalar::DoubleDouble as Dd;

const MANTISSAS: [&str; 10] = ["1", "1.25", "1.6", "2", "2.5", "3.2", "4", "5", "6.4", "8"];

/// (core, shell) as (Re m, Im m): black carbon in sulfate, water on ice, a silica core in a
/// gold-like shell, absorbing cores in a clear shell, high contrast both ways, near unity, an air
/// shell, a clear core in an absorbing shell, a metallic core, a high-index core.
const PAIRS: [((f64, f64), (f64, f64)); 12] = [
    ((1.95, 0.79), (1.43, 1e-8)),
    ((1.31, 1e-8), (1.33, 1e-9)),
    ((1.45, 0.0), (0.2, 3.0)),
    ((1.5, 1e-8), (1.33, 0.0)),
    ((1.5, 1e-3), (1.33, 0.0)),
    ((1.5, 1.0), (1.33, 0.0)),
    ((10.0, 10.0), (1.33, 0.0)),
    ((1.33, 0.0), (10.0, 0.0)),
    ((1.01, 0.0), (1.0001, 0.0)),
    ((1.5, 0.01), (1.0, 0.0)),
    ((1.33, 0.0), (1.5, 0.1)),
    ((0.2, 3.0), (1.5, 0.0)),
];

/// Core radius / outer radius.
const FRACTIONS: [f64; 4] = [1e-3, 0.3, 0.9, 0.999];

/// A particle of the grid: the radius fractions of its interfaces (the last 1) and its indices.
struct Particle {
    fractions: Vec<f64>,
    m: Vec<Complex<f64>>,
}

fn particles() -> Vec<Particle> {
    let c = |(r, i): (f64, f64)| Complex::new(r, i);
    let mut out: Vec<Particle> = PAIRS
        .iter()
        .flat_map(|&(core, shell)| {
            FRACTIONS.iter().map(move |&f| Particle {
                fractions: vec![f, 1.0],
                m: vec![c(core), c(shell)],
            })
        })
        .collect();
    out.push(Particle {
        fractions: vec![0.3, 0.7, 1.0],
        m: vec![c((1.5, 0.1)), c((2.5, 1.0)), c((1.33, 0.0))],
    });
    out.push(Particle {
        fractions: vec![0.5, 0.8, 1.0],
        m: vec![c((1.45, 0.0)), c((0.2, 3.0)), c((1.33, 0.0))],
    });
    out.push(Particle {
        fractions: (1..=10).map(|l| f64::from(l) / 10.0).collect(),
        m: (1..=10)
            .map(|l| c((1.6 - 0.02 * f64::from(l), 1e-4)))
            .collect(),
    });
    out
}

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

/// As the sphere map's: −1 + k/16 and 1 − 2^-2j.
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

fn array_error(f: &[Complex<f64>], d: &[Complex<Dd>]) -> f64 {
    f.iter()
        .zip(d)
        .map(|(&a, &b)| cdiff(a, b))
        .fold(0.0, f64::max)
}

/// C of every output of the particle `p` at outer size parameter `x_l`, ordered as `OUTPUTS`.
fn point(x_l: f64, p: &Particle, mu: &[f64]) -> [f64; 10] {
    let x: Vec<f64> = p.fractions.iter().map(|f| f * x_l).collect();
    let m = &p.m;
    let layers = x.len();
    let derivatives = layers <= 4 && VALIDATED_DOMAIN.contains_derivatives(&x, m);
    let f: Outputs<f64> = solve(&x, m, mu, None);
    let n = f.a.len();
    let d = reference(&x, m, mu, Some(n));
    let orders: Vec<usize> = (1..=n).collect();
    let v = value_measures(&x, m, &d, &orders);
    let c = |output: MieOutput, err: f64, measure: Measure| {
        let tol = VALIDATED_DOMAIN.tolerance(output, x_l, m, measure.condition);
        assert!(
            measure.scale > 0.0 && tol.is_finite(),
            "{output:?} at x = {x:?}, m = {m:?}: degenerate measure {measure:?}, tolerance {tol:e}"
        );
        err / measure.scale / tol * ValidatedDomain::constant(output)
    };
    let mut out = [0.0f64; 10];
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
    let real = m.iter().all(|v| v.im == 0.0);
    for k in 0..5 {
        if k == 2 && real {
            put(3, if f.q.q_abs == 0.0 { 0.0 } else { f64::INFINITY });
            continue;
        }
        let err = diff(q_of(&f.q, k), q_of(&d.q, k));
        put(1 + k, c(OUTPUTS[1 + k].1, err, v.efficiencies[k]));
    }
    for j in 0..mu.len() {
        for i in 0..2 {
            let err = cdiff(f.s[i][j], d.s[i][j]);
            put(6, c(MieOutput::Amplitudes, err, v.amplitudes[j][i]));
        }
    }
    if derivatives {
        let dm = derivative_measures(&x, m, mu, &d, &orders);
        for l in 0..layers {
            for k in 0..4 {
                let err = array_error(&f.d[l][k], &d.d[l][k]);
                put(
                    7,
                    c(
                        MieOutput::CoefficientDerivatives,
                        err,
                        dm.coefficients[l][k],
                    ),
                );
            }
            for p in 0..3 {
                for k in 0..5 {
                    if k == 2 && real && p < 2 {
                        let exact = q_of(&f.jac[l][p], k) == 0.0;
                        put(8, if exact { 0.0 } else { f64::INFINITY });
                        continue;
                    }
                    let err = diff(q_of(&f.jac[l][p], k), q_of(&d.jac[l][p], k));
                    put(
                        8,
                        c(MieOutput::EfficiencyJacobian, err, dm.jacobian[l][p][k]),
                    );
                }
            }
            for p in 0..2 {
                for i in 0..2 {
                    let k = 2 * p + i;
                    let err = (0..mu.len())
                        .map(|j| cdiff(f.ds[l][k][j], d.ds[l][k][j]))
                        .fold(0.0, f64::max);
                    put(
                        9,
                        c(MieOutput::AmplitudeDerivatives, err, dm.amplitudes[l][p][i]),
                    );
                }
            }
        }
    }
    out
}

fn to_json(xs: &[f64], ps: &[Particle], results: &[[f64; 10]]) -> String {
    let list = |v: Vec<String>| v.join(", ");
    let mut json = String::from("{\n");
    let _ = writeln!(
        json,
        "  \"x\": [{}],",
        list(xs.iter().map(|x| format!("{x:e}")).collect())
    );
    let particles = ps
        .iter()
        .map(|p| {
            let f = list(p.fractions.iter().map(|v| format!("{v:e}")).collect());
            let m = list(
                p.m.iter()
                    .map(|v| format!("[{:e}, {:e}]", v.re, v.im))
                    .collect(),
            );
            format!("{{\"fractions\": [{f}], \"m\": [{m}]}}")
        })
        .collect();
    let _ = writeln!(json, "  \"particles\": [{}],", list(particles));
    let declared = OUTPUTS
        .iter()
        .map(|(n, o)| format!("\"{n}\": {}", ValidatedDomain::constant(*o)))
        .collect();
    let _ = writeln!(json, "  \"declared\": {{{}}},", list(declared));
    json += "  \"c\": {\n";
    for (o, (name, _)) in OUTPUTS.iter().enumerate() {
        let rows = (0..xs.len())
            .map(|i| {
                let row = (0..ps.len())
                    .map(|j| match results[i * ps.len() + j][o] {
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
    assert_eq!(xs.len(), 101, "size parameters");
    let ps = particles();
    let mu = cosines();
    let points: Vec<(f64, usize)> = xs
        .iter()
        .flat_map(|&x| (0..ps.len()).map(move |j| (x, j)))
        .collect();
    let inside = |&(x, j): &(f64, usize)| {
        let p: &Particle = &ps[j];
        let xl: Vec<f64> = p.fractions.iter().map(|f| f * x).collect();
        VALIDATED_DOMAIN.contains(&xl, &p.m)
    };
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let done: Vec<Vec<(usize, [f64; 10])>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(x, j)) = points.get(k) else {
                            break mine;
                        };
                        // Outside the domain: no columns (null in the map).
                        let c = if inside(&(x, j)) {
                            point(x, &ps[j], &mu)
                        } else {
                            [f64::NAN; 10]
                        };
                        mine.push((k, c));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("worker"))
            .collect()
    });
    let mut results = vec![[0.0f64; 10]; points.len()];
    let done: Vec<(usize, [f64; 10])> = done.into_iter().flatten().collect();
    assert_eq!(done.len(), points.len(), "every point computed");
    for (k, c) in done {
        results[k] = c;
    }
    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
        std::fs::write(&path, to_json(&xs, &ps, &results)).expect("write the map");
        eprintln!("wrote {path}");
    }
    // With SCATRS_MAP_WORST, the worst points of the derivative columns.
    if std::env::var("SCATRS_MAP_WORST").is_ok() {
        for o in 7..10 {
            let mut v: Vec<(f64, usize)> = results
                .iter()
                .enumerate()
                .filter(|&(k, _)| inside(&points[k]))
                .map(|(k, c)| (c[o], k))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(c, k) in v.iter().take(8) {
                let (x, j) = points[k];
                eprintln!(
                    "{} C {c:.0} x_L {x:e} fractions {:?} m {:?}",
                    OUTPUTS[o].0,
                    ps[j].fractions,
                    ps[j].m.iter().map(|m| (m.re, m.im)).collect::<Vec<_>>()
                );
            }
        }
    }
    let mut failures = Vec::new();
    for (o, (name, output)) in OUTPUTS.iter().enumerate() {
        let (k, worst) = results
            .iter()
            .enumerate()
            .filter(|&(k, _)| inside(&points[k]))
            .map(|(k, c)| (k, c[o]))
            .fold(
                (0, 0.0),
                |a, b| if b.1.is_nan() || b.1 > a.1 { b } else { a },
            );
        // The worst C per decade of x_L, to see where an output degrades.
        let mut decades = [0.0f64; 11];
        for (k, c) in results.iter().enumerate() {
            if inside(&points[k]) {
                let d = (points[k].0.log10().floor() as i32 + 6).clamp(0, 10) as usize;
                decades[d] = decades[d].max(c[o]);
            }
        }
        eprintln!(
            "{name:<24} by decade from 1e-6: {:?}",
            decades.map(|c| (c * 10.0).round() / 10.0)
        );
        let declared = ValidatedDomain::constant(*output);
        let (x, j) = points[k];
        eprintln!(
            "{name:<24} worst C = {worst:8.3} (declared {declared}) at x_L = {x:e}, fractions {:?}, m = {:?}",
            ps[j].fractions, ps[j].m
        );
        if worst.is_nan() || worst > declared {
            failures.push(*name);
        }
    }
    assert!(
        failures.is_empty(),
        "outside the declared tolerance: {failures:?}"
    );
}
