//! Wall-clock benchmarks of the Mie kernel (criterion): values, values with the full
//! (x, Re m, Im m) Jacobian, and amplitudes at 100 angles, across size parameters.
//!
//! ```text
//! cargo bench -p scatrs --bench mie
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use scatrs::mie::{AmplitudeBuffers, Mie, MieOptions};

const SIZES: [f64; 6] = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0];

fn mie(c: &mut Criterion) {
    let m = Complex::new(1.5, 0.01);
    let opts = MieOptions::new();
    let mut group = c.benchmark_group("mie");
    for x in SIZES {
        let mut ws = Mie::<f64>::new();
        group.bench_with_input(BenchmarkId::new("efficiencies", x), &x, |b, &x| {
            b.iter(|| ws.compute(black_box(x), m, &opts).unwrap().efficiencies())
        });
        group.bench_with_input(BenchmarkId::new("efficiencies+jacobian", x), &x, |b, &x| {
            b.iter(|| {
                ws.compute_with_derivatives(black_box(x), m, &opts)
                    .unwrap()
                    .efficiencies_jacobian()
            })
        });
        let mu: Vec<f64> = (0..100)
            .map(|k| (std::f64::consts::PI * f64::from(k) / 99.0).cos())
            .collect();
        let (mut s1, mut s2) = (
            vec![Complex::new(0.0, 0.0); 100],
            vec![Complex::new(0.0, 0.0); 100],
        );
        group.bench_with_input(BenchmarkId::new("amplitudes-100", x), &x, |b, &x| {
            b.iter(|| {
                let sol = ws.compute(black_box(x), m, &opts).unwrap();
                sol.amplitudes(
                    &mu,
                    AmplitudeBuffers {
                        s1: &mut s1,
                        s2: &mut s2,
                    },
                )
                .unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(benches, mie);
criterion_main!(benches);
