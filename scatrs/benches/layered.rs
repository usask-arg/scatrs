//! Wall-clock benchmarks of the layered-sphere kernel (criterion): the efficiencies of two-,
//! three- and ten-layer particles across outer size parameters, and with every derivative (in
//! the derivatives' domain), against a sphere of the same outer size parameter. The theory
//! chapter's "Cost" section records the results.
//!
//! ```text
//! cargo bench -p scatrs --bench layered
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use scatrs::layered::{Layered, VALIDATED_DOMAIN};
use scatrs::mie::{Mie, MieOptions};

const SIZES: [f64; 5] = [0.1, 1.0, 10.0, 100.0, 1000.0];

/// A particle: its name, radius fractions and indices.
type Particle = (&'static str, Vec<f64>, Vec<Complex<f64>>);

fn particles() -> Vec<Particle> {
    let c = Complex::new;
    vec![
        (
            "2-layer",
            vec![0.5, 1.0],
            vec![c(1.95, 0.79), c(1.43, 1e-8)],
        ),
        (
            "3-layer",
            vec![0.3, 0.7, 1.0],
            vec![c(1.5, 0.1), c(2.5, 1.0), c(1.33, 0.0)],
        ),
        (
            "10-layer",
            (1..=10).map(|l| f64::from(l) / 10.0).collect(),
            (1..=10)
                .map(|l| c(1.6 - 0.02 * f64::from(l), 1e-4))
                .collect(),
        ),
    ]
}

fn layered(c: &mut Criterion) {
    let opts = MieOptions::new();
    let mut group = c.benchmark_group("layered");
    for x_l in SIZES {
        let mut mie = Mie::<f64>::new();
        let m = Complex::new(1.43, 1e-8);
        group.bench_with_input(BenchmarkId::new("sphere", x_l), &x_l, |b, &x| {
            b.iter(|| mie.compute(black_box(x), m, &opts).unwrap().efficiencies())
        });
        for (name, fractions, m) in particles() {
            let x: Vec<f64> = fractions.iter().map(|f| f * x_l).collect();
            let mut lay = Layered::<f64>::new();
            group.bench_with_input(BenchmarkId::new(name, x_l), &x, |b, x| {
                b.iter(|| lay.compute(black_box(x), &m, &opts).unwrap().efficiencies())
            });
            if x_l <= VALIDATED_DOMAIN.derivative_x_max {
                let id = BenchmarkId::new(format!("{name}+derivatives"), x_l);
                group.bench_with_input(id, &x, |b, x| {
                    b.iter(|| {
                        lay.compute_with_derivatives(black_box(x), &m, &opts)
                            .unwrap()
                            .efficiencies_jacobian()
                    })
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, layered);
criterion_main!(benches);
