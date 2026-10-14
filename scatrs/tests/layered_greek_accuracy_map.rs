//! The accuracy map of the phase matrix and the Greek coefficients of layered spheres: `f64`
//! against `DoubleDouble` on a grid over the validated domain, every output within its declared
//! tolerance.
//!
//! The grid: x_L = d·10^e for the mantissas `MANTISSAS` and e = −6, …, 3, up to 10³ (46 values),
//! against `PARTICLES`, with the phase matrix at the cosines of the sphere's Greek map,
//! normalized, and the Greek coefficients up to all orders (2N) for x_L ≤ 160 and order 100
//! above, as the sphere's Greek map (`tests/mie_greek_accuracy_map.rs`) has them. The derivative
//! columns cover the derivatives' domain (x_L ≤ 100). The measures are the sphere's, with the
//! condition numbers summed over every layer's x_l, Re m_l and Im m_l (`tests/common/layered.rs`).
//!
//! With `SCATRS_ACCURACY_MAP=path` the test writes every C as JSON:
//!
//! ```text
//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/layered-greek-accuracy-map.json \
//!     cargo test -p scatrs --release --test layered_greek_accuracy_map -- --nocapture
//! ```
#![allow(clippy::needless_range_loop)] // parallel per-layer arrays are indexed together

mod common;

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::check::max_nan;
use common::greek::{buffers, greek_arrays};
use common::mie::Measure;
use num_complex::Complex;
use scatrs::Real;
use scatrs::layered::{Layered, VALIDATED_DOMAIN, ValidatedDomain};
use scatrs::mie::{MieOptions, MieOutput, Normalization};
use scatrs::scalar::DoubleDouble as Dd;

const MANTISSAS: [&str; 5] = ["1", "1.6", "2.5", "4", "6.4"];
const X_MAX: f64 = 1e3;
const ALL_ORDERS_UP_TO: f64 = 160.0;
const ORDER_ABOVE: usize = 100;

/// Two-layer particles (core fractions 0.3 and 0.9) of the pairs of the layered accuracy map,
/// and a three-layer particle.
fn particles() -> Vec<(Vec<f64>, Vec<Complex<f64>>)> {
    let c = |r: f64, i: f64| Complex::new(r, i);
    let pairs = [
        (c(1.95, 0.79), c(1.43, 1e-8)),
        (c(1.31, 1e-8), c(1.33, 1e-9)),
        (c(1.45, 0.0), c(0.2, 3.0)),
        (c(1.5, 1e-8), c(1.33, 0.0)),
        (c(1.5, 1.0), c(1.33, 0.0)),
        (c(10.0, 10.0), c(1.33, 0.0)),
        (c(1.33, 0.0), c(10.0, 0.0)),
        (c(1.01, 0.0), c(1.0001, 0.0)),
        (c(1.5, 0.01), c(1.0, 0.0)),
        (c(1.33, 0.0), c(1.5, 0.1)),
    ];
    let mut out: Vec<_> = pairs
        .iter()
        .flat_map(|&(a, b)| [0.3, 0.9].map(|f| (vec![f, 1.0], vec![a, b])))
        .collect();
    out.push((
        vec![0.3, 0.7, 1.0],
        vec![c(1.5, 0.1), c(2.5, 1.0), c(1.33, 0.0)],
    ));
    out
}

const OUTPUTS: [&str; 4] = [
    "phase_matrix",
    "greek",
    "phase_matrix_derivatives",
    "greek_derivatives",
];

fn size_parameters() -> Vec<f64> {
    (-6..=3)
        .flat_map(|e| MANTISSAS.iter().map(move |d| format!("{d}e{e}")))
        .map(|s| s.parse::<f64>().expect("a decimal literal"))
        .filter(|&x| x <= X_MAX)
        .collect()
}

/// As the sphere's Greek map: −1 + k/8 and 1 − 2^-2j.
fn cosines() -> Vec<f64> {
    let uniform = (0..=16).map(|k| -1.0 + f64::from(k) / 8.0);
    let forward = (3..=8).map(|j| 1.0 - 2f64.powi(-2 * j));
    uniform.chain(forward).collect()
}

/// The phase matrix and Greek coefficients, with their derivatives per layer.
#[derive(Clone)]
struct Out<T> {
    greek: [Vec<T>; 6],
    /// Per layer: ∂/∂x_l, ∂/∂Re m_l, ∂/∂Im m_l.
    greek_d: Vec<[[Vec<T>; 6]; 3]>,
    phase: [Vec<T>; 4],
    phase_d: Vec<[[Vec<T>; 4]; 3]>,
}

fn solve<T: Real>(
    x: &[T],
    m: &[Complex<T>],
    mu: &[T],
    n: Option<usize>,
    order: usize,
    derivatives: bool,
) -> Out<T> {
    let mut opts = MieOptions::new().with_allow_unvalidated(true);
    if let Some(n) = n {
        opts = opts.with_n_max(n);
    }
    let mut lay = Layered::<T>::new();
    let norm = Normalization::Normalized;
    let zeros = || vec![T::zero(); mu.len()];
    let sol = if derivatives {
        lay.compute_with_derivatives(x, m, &opts)
    } else {
        lay.compute(x, m, &opts)
    }
    .expect("valid inputs");
    let greek = greek_arrays(&sol.greek(Some(order), norm).expect("valid order"));
    let mut phase: [Vec<T>; 4] = std::array::from_fn(|_| zeros());
    sol.phase_matrix(mu, norm, buffers(&mut phase))
        .expect("valid cosines");
    let (mut greek_d, mut phase_d) = (Vec::new(), Vec::new());
    if derivatives {
        for l in 0..x.len() {
            let (_, dg) = sol
                .greek_with_derivatives(Some(order), l, norm)
                .expect("valid order");
            greek_d.push(dg.each_ref().map(greek_arrays));
            let mut v: [Vec<T>; 4] = std::array::from_fn(|_| zeros());
            let mut d: [[Vec<T>; 4]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| zeros()));
            {
                let [dx, dr, di] = &mut d;
                sol.phase_matrix_with_derivatives(
                    mu,
                    l,
                    norm,
                    buffers(&mut v),
                    [buffers(dx), buffers(dr), buffers(di)],
                )
                .expect("valid cosines");
            }
            phase_d.push(d);
        }
    }
    Out {
        greek,
        greek_d,
        phase,
        phase_d,
    }
}

fn dd_inputs(x: &[f64], m: &[Complex<f64>]) -> (Vec<Dd>, Vec<Complex<Dd>>) {
    (
        x.iter().map(|&v| Dd::from_f64(v)).collect(),
        m.iter()
            .map(|v| Complex::new(Dd::from_f64(v.re), Dd::from_f64(v.im)))
            .collect(),
    )
}

fn diff(a: f64, b: Dd) -> f64 {
    (Dd::from_f64(a) - b).hi().abs()
}

/// Σ_l x_l|∂y/∂x_l| + |Re m_l||∂y/∂(Re m_l)| + |Im m_l||∂y/∂(Im m_l)|.
fn sensitivity(x: &[f64], m: &[Complex<f64>], d: &dyn Fn(usize, usize) -> f64) -> f64 {
    (0..x.len())
        .map(|l| {
            x[l] * d(l, 0).abs() + m[l].re.abs() * d(l, 1).abs() + m[l].im.abs() * d(l, 2).abs()
        })
        .sum()
}

/// C of every output of the particle (`fractions`, `m`) at x_L, ordered as `OUTPUTS` (NaN for
/// the derivative columns outside the derivatives' domain).
fn point(x_l: f64, fractions: &[f64], m: &[Complex<f64>], mu: &[f64]) -> [f64; 4] {
    let x: Vec<f64> = fractions.iter().map(|f| f * x_l).collect();
    let layers = x.len();
    let order = if x_l <= ALL_ORDERS_UP_TO {
        2 * scatrs::mie::default_n_max(x_l)
    } else {
        ORDER_ABOVE
    };
    let derivatives = VALIDATED_DOMAIN.contains_derivatives(&x, m);
    let f = solve(&x, m, mu, None, order, derivatives);
    let n = scatrs::mie::default_n_max(x_l);
    let (xd, md) = dd_inputs(&x, m);
    let mu_dd: Vec<Dd> = mu.iter().map(|&v| Dd::from_f64(v)).collect();
    let d = solve(&xd, &md, &mu_dd, Some(n), order, true);
    let tol = |output: MieOutput, measure: Measure| {
        let t = VALIDATED_DOMAIN.tolerance(output, x_l, m, measure.condition);
        assert!(
            measure.scale > 0.0 && t.is_finite(),
            "{output:?} at x = {x:?}, m = {m:?}: degenerate measure {measure:?}"
        );
        t
    };
    let c = |output: MieOutput, err: f64, measure: Measure| {
        err / measure.scale / tol(output, measure) * ValidatedDomain::constant(output)
    };
    let mut out = [0.0f64; 4];
    let alpha = d.greek[0].iter().map(|v| v.hi().abs()).fold(0.0, max_nan);
    let mean = d.greek[0][0].hi();
    let greek_out = MieOutput::GreekCoefficients { order };
    for k in 0..6 {
        let num = (0..d.greek[k].len())
            .map(|l| sensitivity(&x, m, &|q, p| d.greek_d[q][p][k][l].hi()))
            .fold(0.0, max_nan);
        let measure = Measure {
            scale: alpha,
            condition: num / alpha,
        };
        let err = f.greek[k]
            .iter()
            .zip(&d.greek[k])
            .map(|(&a, &b)| diff(a, b))
            .fold(0.0, max_nan);
        out[1] = max_nan(out[1], c(greek_out, err, measure));
    }
    for j in 0..mu.len() {
        let scale = d.phase[0][j].hi().max(mean);
        for k in 0..4 {
            let condition = sensitivity(&x, m, &|q, p| d.phase_d[q][p][k][j].hi()) / scale;
            let err = diff(f.phase[k][j], d.phase[k][j]);
            out[0] = max_nan(
                out[0],
                c(MieOutput::PhaseMatrix, err, Measure { scale, condition }),
            );
        }
    }
    if !derivatives {
        out[2] = f64::NAN;
        out[3] = f64::NAN;
        return out;
    }
    // Condition numbers of the derivatives from central differences of double-double
    // derivatives, steps 2⁻³⁰ x_l and 2⁻³⁰ |m_l| (one-sided where needed, as
    // `common::layered::derivative_measures`).
    let h = 2f64.powi(-30);
    let big = m.iter().map(|v| v.norm()).fold(1.0, f64::max);
    let phase_of_x = 1.0 + big * x_l;
    let mut steps = Vec::new();
    for l in 0..layers {
        let mut per = Vec::new();
        for p in 0..3 {
            let hp = if p == 0 { x[l] * h } else { m[l].norm() * h };
            let up = p != 0 || l + 1 == layers || x[l] + hp <= x[l + 1];
            let down = match p {
                0 => l == 0 || x[l] - hp >= x[l - 1],
                1 => true,
                _ => m[l].im >= hp,
            };
            let at = |sign: f64| {
                let (mut xs, mut ms) = (xd.clone(), md.clone());
                let s = Dd::from_f64(sign * hp);
                match p {
                    0 => xs[l] += s,
                    1 => ms[l].re += s,
                    _ => ms[l].im += s,
                }
                solve(&xs, &ms, &mu_dd, Some(n), order, true)
            };
            per.push(match (up, down) {
                (true, true) => (at(1.0), at(-1.0), 2.0 * hp),
                (true, false) => (at(1.0), d.clone(), hp),
                _ => (d.clone(), at(-1.0), hp),
            });
        }
        steps.push(per);
    }
    let fd = |a: Dd, b: Dd, w: f64| ((a - b) / Dd::from_f64(w)).hi();
    let greek_d_out = MieOutput::GreekDerivatives { order };
    for l in 0..layers {
        for p in 0..3 {
            let p_size = if p == 0 { x[l] } else { m[l].norm() };
            for k in 0..6 {
                let natural = alpha * phase_of_x / p_size;
                let values = &d.greek_d[l][p][k];
                let scale = values
                    .iter()
                    .map(|v| v.hi().abs().max(natural))
                    .fold(0.0, max_nan);
                let num = (0..values.len())
                    .map(|o| {
                        sensitivity(&x, m, &|q, r| {
                            let (pl, mi, w) = &steps[q][r];
                            fd(pl.greek_d[l][p][k][o], mi.greek_d[l][p][k][o], *w)
                        })
                    })
                    .fold(0.0, max_nan);
                let measure = Measure {
                    scale,
                    condition: num / scale,
                };
                let err = f.greek_d[l][p][k]
                    .iter()
                    .zip(values)
                    .map(|(&a, &b)| diff(a, b))
                    .fold(0.0, max_nan);
                out[3] = max_nan(out[3], c(greek_d_out, err, measure));
            }
            for k in 0..4 {
                let scale = (0..mu.len())
                    .map(|j| {
                        let natural = d.phase[0][j].hi().max(mean) * phase_of_x / p_size;
                        d.phase_d[l][p][k][j].hi().abs().max(natural)
                    })
                    .fold(0.0, max_nan);
                let num = (0..mu.len())
                    .map(|j| {
                        sensitivity(&x, m, &|q, r| {
                            let (pl, mi, w) = &steps[q][r];
                            fd(pl.phase_d[l][p][k][j], mi.phase_d[l][p][k][j], *w)
                        })
                    })
                    .fold(0.0, max_nan);
                let measure = Measure {
                    scale,
                    condition: num / scale,
                };
                let err = (0..mu.len())
                    .map(|j| diff(f.phase_d[l][p][k][j], d.phase_d[l][p][k][j]))
                    .fold(0.0, max_nan);
                out[2] = max_nan(out[2], c(MieOutput::PhaseMatrixDerivatives, err, measure));
            }
        }
    }
    out
}

/// The declared constants of the columns.
fn declared() -> [f64; 4] {
    [
        ValidatedDomain::constant(MieOutput::PhaseMatrix),
        ValidatedDomain::constant(MieOutput::GreekCoefficients { order: 0 }),
        ValidatedDomain::constant(MieOutput::PhaseMatrixDerivatives),
        ValidatedDomain::constant(MieOutput::GreekDerivatives { order: 0 }),
    ]
}

fn to_json(xs: &[f64], ps: &[(Vec<f64>, Vec<Complex<f64>>)], results: &[[f64; 4]]) -> String {
    let list = |v: Vec<String>| v.join(", ");
    let mut json = String::from("{\n");
    let _ = writeln!(
        json,
        "  \"x\": [{}],",
        list(xs.iter().map(|x| format!("{x:e}")).collect())
    );
    let particles = ps
        .iter()
        .map(|(f, m)| {
            let f = list(f.iter().map(|v| format!("{v:e}")).collect());
            let m = list(
                m.iter()
                    .map(|v| format!("[{:e}, {:e}]", v.re, v.im))
                    .collect(),
            );
            format!("{{\"fractions\": [{f}], \"m\": [{m}]}}")
        })
        .collect();
    let _ = writeln!(json, "  \"particles\": [{}],", list(particles));
    let declared = OUTPUTS
        .iter()
        .zip(declared())
        .map(|(n, c)| format!("\"{n}\": {c}"))
        .collect();
    let _ = writeln!(json, "  \"declared\": {{{}}},", list(declared));
    json += "  \"c\": {\n";
    for (o, name) in OUTPUTS.iter().enumerate() {
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
fn greek_accuracy_map_is_within_the_declared_tolerances() {
    let xs = size_parameters();
    assert_eq!(xs.len(), 46, "size parameters");
    let ps = particles();
    let mu = cosines();
    let points: Vec<(f64, usize)> = xs
        .iter()
        .flat_map(|&x| (0..ps.len()).map(move |j| (x, j)))
        .collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let done: Vec<Vec<(usize, [f64; 4])>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(x, j)) = points.get(k) else {
                            break mine;
                        };
                        mine.push((k, point(x, &ps[j].0, &ps[j].1, &mu)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("worker"))
            .collect()
    });
    let mut results = vec![[0.0f64; 4]; points.len()];
    for (k, c) in done.into_iter().flatten() {
        results[k] = c;
    }
    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
        std::fs::write(&path, to_json(&xs, &ps, &results)).expect("write the map");
        eprintln!("wrote {path}");
    }
    let declared = declared();
    let mut failures = Vec::new();
    for (o, name) in OUTPUTS.iter().enumerate() {
        let (k, worst) = results
            .iter()
            .enumerate()
            .filter(|(_, c)| !(o >= 2 && c[o].is_nan()))
            .map(|(k, c)| (k, c[o]))
            .fold(
                (0, 0.0),
                |a, b| if b.1.is_nan() || b.1 > a.1 { b } else { a },
            );
        let (x, j) = points[k];
        eprintln!(
            "{name:<26} worst C = {worst:8.3} (declared {}) at x_L = {x:e}, fractions {:?}, m = {:?}",
            declared[o], ps[j].0, ps[j].1
        );
        if worst.is_nan() || worst > declared[o] {
            failures.push(*name);
        }
    }
    assert!(
        failures.is_empty(),
        "outside the declared tolerance: {failures:?}"
    );
}
