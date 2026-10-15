//! The accuracy map of randomly oriented spheroids: the default results (`rtol` = 10⁻⁸) against
//! a reference with N + max(4, N/5) orders and 1.5 times the nodes the default converged at,
//! every mode refined to an estimate below 10⁻¹³ (the refined path, whose own accuracy the Arb
//! references of `spheroid_reference.rs` check), over a grid of aspect ratios, sizes and
//! indices, inside the validated domain and past its edge in |m|·x_max. The reference thus
//! measures the truncation, the quadrature and the solve of the default.
//!
//! For each output it prints the worst ratio error/max(`rtol`, estimate), in the measures of
//! `MieOutput` (Q_abs relative to Q_ext; g absolute; Q_back relative to max(Q_back, Q_sca); the
//! Greek coefficients relative to the largest |α₁|; the phase matrix relative to max(F₁₁, 1),
//! normalized), separately inside and outside the domain. The declared constants of
//! `tmatrix::VALIDATED_DOMAIN` are at least 4 times the worst ratio inside.
//!
//! Run with `cargo test --release -p scatrs --test tmatrix_accuracy_map -- --ignored --nocapture`
//! (about an hour on an Apple M4 Pro). With `SCATRS_ACCURACY_MAP` set to a path it also writes
//! the points as JSON there, for the documentation's plots:
//!
//! ```text
//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/spheroid-accuracy-map.json \
//!     cargo test --release -p scatrs --test tmatrix_accuracy_map -- --ignored
//! ```

use num_complex::Complex;
use scatrs::mie::{MieOutput, Normalization};
use scatrs::tmatrix::{PhaseBuffers, Shape, TMatrix, TMatrixOptions, VALIDATED_DOMAIN};

const OUTPUTS: [&str; 7] = ["q_ext", "q_sca", "q_abs", "g", "q_back", "greek", "phase"];

/// The `MieOutput` whose constant applies to each output.
fn output_of(name: &str) -> MieOutput {
    match name {
        "q_ext" => MieOutput::Extinction,
        "q_sca" => MieOutput::Scattering,
        "q_abs" => MieOutput::Absorption,
        "g" => MieOutput::Asymmetry,
        "q_back" => MieOutput::Backscattering,
        "greek" => MieOutput::GreekCoefficients { order: 0 },
        _ => MieOutput::PhaseMatrix,
    }
}

struct Point {
    values: [f64; 5],
    greek: [Vec<f64>; 6],
    phase: [Vec<f64>; 6],
    est: f64,
    n: usize,
    n_theta: usize,
}

fn solve(eps: f64, x: f64, m: Complex<f64>, opts: &TMatrixOptions, mu: &[f64]) -> Point {
    let mut tm = TMatrix::new();
    let s = tm
        .compute(Shape::Spheroid { aspect_ratio: eps }, x, m, opts)
        .unwrap();
    let q = s.efficiencies();
    let g = s.greek(None, Normalization::Normalized).unwrap();
    let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; mu.len()]);
    {
        let [f11, f12, f22, f33, f34, f44] = &mut f;
        s.phase_matrix(
            mu,
            Normalization::Normalized,
            PhaseBuffers {
                f11,
                f12,
                f22,
                f33,
                f34,
                f44,
            },
        )
        .unwrap();
    }
    Point {
        values: [q.q_ext, q.q_sca, q.q_abs, q.g, q.q_back],
        greek: [g.alpha1, g.alpha2, g.alpha3, g.alpha4, g.beta1, g.beta2],
        phase: f,
        est: s.error_estimate(),
        n: s.n_terms(),
        n_theta: s.n_theta(),
    }
}

/// The errors of `a` against `b` in the measures of the module doc.
fn errors(a: &Point, b: &Point) -> [f64; 7] {
    let [e, s, ab, g, back] = b.values;
    let rel = |u: f64, v: f64, scale: f64| (u - v).abs() / scale;
    let big = b.greek[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut greek = 0.0f64;
    for (u, v) in a.greek.iter().zip(&b.greek) {
        let len = u.len().max(v.len());
        for l in 0..len {
            let (x, y) = (
                u.get(l).copied().unwrap_or(0.0),
                v.get(l).copied().unwrap_or(0.0),
            );
            greek = greek.max((x - y).abs() / big);
        }
    }
    let fbig = b.phase[0].iter().fold(1.0f64, |m, v| m.max(*v));
    let mut phase = 0.0f64;
    for (u, v) in a.phase.iter().zip(&b.phase) {
        for (x, y) in u.iter().zip(v) {
            phase = phase.max((x - y).abs() / fbig);
        }
    }
    [
        rel(a.values[0], e, e),
        rel(a.values[1], s, s),
        rel(a.values[2], ab, e),
        rel(a.values[3], g, 1.0),
        rel(a.values[4], back, back.max(s)),
        greek,
        phase,
    ]
}

#[test]
#[ignore = "the accuracy map: about an hour"]
fn accuracy_map() {
    let xs: [f64; 7] = [0.3, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0];
    let eps: [f64; 8] = [0.2, 1.0 / 3.0, 0.5, 0.8, 1.25, 2.0, 3.0, 5.0];
    let ms = [
        Complex::new(1.33, 0.0),
        Complex::new(1.53, 0.008),
        Complex::new(1.5, 0.5),
        Complex::new(2.5, 0.1),
    ];
    let mu: Vec<f64> = (0..=20).map(|i| -1.0 + 0.1 * f64::from(i)).collect();
    let f64_opts = TMatrixOptions::new().with_allow_unvalidated(true);
    // The reference: more orders and nodes than the default converged at, every mode refined.
    let ref_opts = |n: usize, nth: usize| {
        TMatrixOptions::new()
            .with_rtol(1e-12)
            .with_n_terms(n + (n / 5).max(4))
            .with_n_theta(nth * 3 / 2)
            .with_allow_unvalidated(true)
    };
    // Worst ratio per output, inside and outside the domain, with where.
    let mut rows: Vec<String> = Vec::new();
    let mut worst: [[(f64, String); 7]; 2] =
        core::array::from_fn(|_| core::array::from_fn(|_| (0.0, String::new())));
    for &e in &eps {
        for &x in &xs {
            for &m in &ms {
                let a = x * e.cbrt();
                let phase = m.norm() * a.max(a / e);
                if phase > 70.0 {
                    continue;
                }
                let t0 = std::time::Instant::now();
                let p = solve(e, x, m, &f64_opts, &mu);
                let t64 = t0.elapsed();
                let r = solve(e, x, m, &ref_opts(p.n, p.n_theta), &mu);
                let errs = errors(&p, &r);
                let inside = VALIDATED_DOMAIN.contains(e, x, m);
                let unit = 1e-8f64.max(p.est);
                let ratios = errs.map(|v| v / unit);
                println!(
                    "eps {e:.3} x {x:5.1} m {m:.3} |m|x_max {phase:5.1} {}: est {:.1e} ({t64:.2?}) ratios {}",
                    if inside { "in " } else { "out" },
                    p.est,
                    ratios
                        .iter()
                        .map(|v| format!("{v:.1e}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                rows.push(format!(
                    "    {{\"aspect_ratio\": {e}, \"x\": {x}, \"m_re\": {}, \"m_im\": {}, \"phase\": {phase}, \"inside\": {inside}, \"estimate\": {:e}, \"ratios\": [{}]}}",
                    m.re,
                    m.im,
                    p.est,
                    ratios.iter().map(|v| format!("{v:e}")).collect::<Vec<_>>().join(", ")
                ));
                let w = &mut worst[usize::from(!inside)];
                for (k, &v) in ratios.iter().enumerate() {
                    if v > w[k].0 {
                        w[k] = (v, format!("eps {e:.3} x {x} m {m}"));
                    }
                }
            }
        }
    }
    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
        let outputs = OUTPUTS
            .iter()
            .map(|o| format!("\"{o}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let declared = OUTPUTS
            .iter()
            .map(|o| {
                format!(
                    "\"{o}\": {}",
                    VALIDATED_DOMAIN.tolerance(output_of(o), 1.0, 0.0)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            "{{\n  \"outputs\": [{outputs}],\n  \"declared\": {{{declared}}},\n  \"points\": [\n{}\n  ]\n}}\n",
            rows.join(",\n")
        );
        std::fs::write(&path, json).expect("write the map");
    }
    for (i, label) in ["inside", "outside"].iter().enumerate() {
        println!("worst {label}:");
        for (k, name) in OUTPUTS.iter().enumerate() {
            println!("  {name:7} {:.3e} at {}", worst[i][k].0, worst[i][k].1);
        }
    }
}
