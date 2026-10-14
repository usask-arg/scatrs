//! The accuracy map of the phase matrix and the Greek coefficients: `f64` against
//! `DoubleDouble` on a grid over the validated domain, every output within its declared
//! tolerance.
//!
//! The grid: x = d·10^e for the mantissas `MANTISSAS` and e = −6, …, 3, up to 10³, 46 values,
//! against the refractive indices `INDICES` of the Mie accuracy map
//! (`tests/mie_accuracy_map.rs`), with the phase matrix at `COSINES`, normalized. The Greek
//! coefficients go up to all orders (2N) for x ≤ 160 and to order 100 above. The double-double
//! Greek coefficients cost O(x²) (amplitudes at about x nodes, and all orders at as many), and
//! the derivative measures take seven of them per point, so the grid stops at 10³, where the
//! map takes about 2 CPU-minutes. The ignored test `large_spheres_are_within_the_tolerances`
//! (about a minute; CI runs it weekly) checks the values at x = 2·10³ (all orders), 10⁴ and
//! 2·10⁴ (order 64). `DoubleDouble` stands in for the exact values: its phase matrix and
//! Greek coefficients are within 16 times the declared constants, in units of
//! (κ′ + cond)·2⁻¹⁰⁴, of Arb for the 18 spheres up to x = 200 of tests/mie_greek_reference.rs.
//! Its derivatives are not compared with Arb; they come from the same double-double amplitude
//! derivatives as those of tests/mie_reference.rs, which are.
//!
//! Each error is reported as C = error/((κ′ + cond)·2⁻⁵²); the test prints the worst C of each
//! output. With `SCATRS_ACCURACY_MAP=path` it also writes every C as JSON, which the
//! documentation plots (docs/validation/accuracy.md):
//!
//! ```text
//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/greek-accuracy-map.json \
//!     cargo test -p scatrs --release --test mie_greek_accuracy_map -- --nocapture
//! ```

mod common;

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::check::max_nan;
use common::greek::{
    GreekOutputs, greek_derivative_measures, greek_reference, greek_value_measures, solve_greek,
};
use common::mie::Measure;
use num_complex::Complex;
use scatrs::mie::{MieOutput, Normalization, VALIDATED_DOMAIN, ValidatedDomain, default_n_max};
use scatrs::scalar::DoubleDouble as Dd;

/// Five size parameters per decade.
const MANTISSAS: [&str; 5] = ["1", "1.6", "2.5", "4", "6.4"];

/// The largest x of the grid.
const X_MAX: f64 = 1e3;

/// Above this x the Greek coefficients go up to `ORDER_ABOVE` only.
const ALL_ORDERS_UP_TO: f64 = 160.0;
const ORDER_ABOVE: usize = 100;

/// (Re m, Im m), as in the Mie accuracy map.
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

/// Cosines through the forward peak, the sides and the glory: exact doubles.
const COSINES: [f64; 13] = [
    1.0,
    0.999_999_046_325_683_6, // 1 − 2⁻²⁰
    0.999_755_859_375,       // 1 − 2⁻¹²
    0.996_093_75,            // 1 − 2⁻⁸
    0.9375,
    0.75,
    0.5,
    0.0,
    -0.5,
    -0.75,
    -0.9375,
    -0.996_093_75,
    -1.0,
];

const NORMALIZATION: Normalization = Normalization::Normalized;

/// The columns of the map.
const OUTPUTS: [&str; 4] = [
    "phase_matrix",
    "phase_matrix_derivatives",
    "greek",
    "greek_derivatives",
];

fn size_parameters() -> Vec<f64> {
    (-6..=3)
        .flat_map(|e| MANTISSAS.iter().map(move |d| format!("{d}e{e}")))
        .map(|s| s.parse::<f64>().expect("a decimal literal"))
        .filter(|&x| x <= X_MAX)
        .collect()
}

fn order_at(x: f64) -> Option<usize> {
    (x > ALL_ORDERS_UP_TO).then_some(ORDER_ABOVE)
}

fn diff(a: f64, b: Dd) -> f64 {
    (Dd::from_f64(a) - b).hi().abs()
}

/// max over the array of |f − d|.
fn array_error(f: &[f64], d: &[Dd]) -> f64 {
    f.iter()
        .zip(d)
        .map(|(&a, &b)| diff(a, b))
        .fold(0.0, max_nan)
}

/// C = error/((κ′ + cond)·2⁻⁵²) of `output` at (x, m); NaN counts as infinite.
fn constant_of(output: MieOutput, x: f64, m: Complex<f64>, err: f64, measure: Measure) -> f64 {
    let tol = VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition);
    assert!(
        measure.scale > 0.0 && tol.is_finite(),
        "{output:?} at x = {x:e}, m = {m}: degenerate measure {measure:?}, tolerance {tol:e}"
    );
    let c = err / measure.scale / tol * ValidatedDomain::constant(output);
    if c.is_nan() { f64::INFINITY } else { c }
}

/// C of the phase matrix and of the Greek coefficients (up to order `l`) of `f` at (x, m),
/// against the double-double solution `d`.
fn value_constants(
    x: f64,
    m: Complex<f64>,
    l: usize,
    f: &GreekOutputs<f64>,
    d: &GreekOutputs<Dd>,
) -> [f64; 2] {
    let v = greek_value_measures(x, m, d);
    let mut out = [0.0f64; 2];
    for j in 0..COSINES.len() {
        for k in 0..4 {
            let err = diff(f.phase[k][j], d.phase[k][j]);
            let c = constant_of(MieOutput::PhaseMatrix, x, m, err, v.phase[j][k]);
            out[0] = out[0].max(c);
        }
    }
    for k in 0..6 {
        let err = array_error(&f.greek[k], &d.greek[k]);
        let output = MieOutput::GreekCoefficients { order: l };
        out[1] = out[1].max(constant_of(output, x, m, err, v.greek[k]));
    }
    out
}

/// C of every output at (x, m), ordered as `OUTPUTS`.
fn point(x: f64, m: Complex<f64>) -> [f64; 4] {
    let order = order_at(x);
    let n = default_n_max(x);
    let f = solve_greek(x, m, &COSINES, None, order, NORMALIZATION);
    let d = greek_reference(x, m, &COSINES, n, order, NORMALIZATION);
    let l = f.greek[0].len() - 1;
    let dm = greek_derivative_measures(x, m, &COSINES, n, l, NORMALIZATION, &d);
    let [phase, greek] = value_constants(x, m, l, &f, &d);
    let mut out = [phase, 0.0, greek, 0.0];
    for p in 0..3 {
        for k in 0..4 {
            let err = array_error(&f.phase_d[p][k], &d.phase_d[p][k]);
            let c = constant_of(MieOutput::PhaseMatrixDerivatives, x, m, err, dm.phase[p][k]);
            out[1] = out[1].max(c);
        }
        for k in 0..6 {
            let err = array_error(&f.greek_d[p][k], &d.greek_d[p][k]);
            let output = MieOutput::GreekDerivatives { order: l };
            out[3] = out[3].max(constant_of(output, x, m, err, dm.greek[p][k]));
        }
    }
    out
}

/// The constant declared for column `o` of `OUTPUTS` (the order does not change it).
fn declared(o: usize) -> f64 {
    ValidatedDomain::constant(
        [
            MieOutput::PhaseMatrix,
            MieOutput::PhaseMatrixDerivatives,
            MieOutput::GreekCoefficients { order: 0 },
            MieOutput::GreekDerivatives { order: 0 },
        ][o],
    )
}

fn to_json(xs: &[f64], results: &[[f64; 4]]) -> String {
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
    let constants = (0..OUTPUTS.len())
        .map(|o| format!("\"{}\": {}", OUTPUTS[o], declared(o)))
        .collect();
    let _ = writeln!(json, "  \"declared\": {{{}}},", list(constants));
    json += "  \"c\": {\n";
    for (o, name) in OUTPUTS.iter().enumerate() {
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

/// Runs `f` on every item on all threads; results in the order of `items`.
fn parallel<I: Sync, R: Send + Copy + Default>(items: &[I], f: impl Fn(&I) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let done: Vec<Vec<(usize, R)>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(k) else {
                            break mine;
                        };
                        mine.push((k, f(item)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("worker"))
            .collect()
    });
    let mut results = vec![R::default(); items.len()];
    let mut count = 0;
    for (k, r) in done.into_iter().flatten() {
        results[k] = r;
        count += 1;
    }
    assert_eq!(count, items.len(), "every item computed");
    results
}

#[test]
fn accuracy_map_is_within_the_declared_tolerances() {
    let xs = size_parameters();
    assert_eq!(xs.len(), 46, "size parameters");
    let points: Vec<(f64, Complex<f64>)> = xs
        .iter()
        .flat_map(|&x| INDICES.iter().map(move |&(r, i)| (x, Complex::new(r, i))))
        .collect();
    let results = parallel(&points, |&(x, m)| point(x, m));
    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
        std::fs::write(&path, to_json(&xs, &results)).expect("write the map");
        eprintln!("wrote {path}");
    }
    let mut failures = Vec::new();
    for (o, name) in OUTPUTS.iter().enumerate() {
        let (k, worst) = results
            .iter()
            .enumerate()
            .map(|(k, c)| (k, c[o]))
            .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let (x, m) = points[k];
        eprintln!(
            "{name:<26} worst C = {worst:8.3} (declared {}) at x = {x:e}, m = {m}",
            declared(o)
        );
        if worst > declared(o) {
            failures.push(*name);
        }
    }
    assert!(
        failures.is_empty(),
        "outside the declared tolerance: {failures:?}"
    );
}

/// The values beyond the grid: the phase matrix and the Greek coefficients within their
/// tolerances (with condition numbers from the double-double derivatives) at x = 2·10³ up to
/// all orders and at x = 10⁴ and 2·10⁴ up to order 64.
#[test]
#[ignore = "double-double Greek coefficients of large spheres: about a minute with all cores"]
fn large_spheres_are_within_the_tolerances() {
    let points = [
        (2e3, Complex::new(1.33, 0.0), None),
        (2e3, Complex::new(0.3, 0.6), None),
        (1e4, Complex::new(1.5, 0.01), Some(64)),
        (2e4, Complex::new(1.33, 0.0), Some(64)),
    ];
    let results = parallel(&points, |&(x, m, order)| {
        let n = default_n_max(x);
        let f = solve_greek(x, m, &COSINES, None, order, NORMALIZATION);
        let d = greek_reference(x, m, &COSINES, n, order, NORMALIZATION);
        value_constants(x, m, f.greek[0].len() - 1, &f, &d)
    });
    for ((x, m, _), c) in points.iter().zip(&results) {
        eprintln!(
            "x = {x:e}, m = {m}: phase matrix C = {:.3}, Greek C = {:.3}",
            c[0], c[1]
        );
        assert!(
            c[0] <= declared(0),
            "phase matrix at x = {x:e}: C = {}",
            c[0]
        );
        assert!(
            c[1] <= declared(2),
            "Greek coefficients at x = {x:e}: C = {}",
            c[1]
        );
    }
}
