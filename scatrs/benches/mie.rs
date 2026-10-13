//! Wall-clock benchmarks of the Mie kernel (criterion): values, values with the full
//! (x, Re m, Im m) Jacobian, and amplitudes at 100 angles, across size parameters, at
//! m = 1.5 + 0.01i; and the values and Jacobian at the real m = 1.5 (`-real-m`), whose values
//! take the cheaper real recurrence, so the Jacobian costs relatively more there. The theory
//! chapter's "Cost" section records the results.
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
    let opts = MieOptions::new();
    let mut group = c.benchmark_group("mie");
    let mu: Vec<f64> = (0..100)
        .map(|k| (std::f64::consts::PI * f64::from(k) / 99.0).cos())
        .collect();
    for x in SIZES {
        let mut ws = Mie::<f64>::new();
        for (suffix, m) in [
            ("", Complex::new(1.5, 0.01)),
            ("-real-m", Complex::new(1.5, 0.0)),
        ] {
            let id = BenchmarkId::new(format!("efficiencies{suffix}"), x);
            group.bench_with_input(id, &x, |b, &x| {
                b.iter(|| ws.compute(black_box(x), m, &opts).unwrap().efficiencies())
            });
            let id = BenchmarkId::new(format!("efficiencies+jacobian{suffix}"), x);
            group.bench_with_input(id, &x, |b, &x| {
                b.iter(|| {
                    ws.compute_with_derivatives(black_box(x), m, &opts)
                        .unwrap()
                        .efficiencies_jacobian()
                })
            });
        }
        let m = Complex::new(1.5, 0.01);
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
                black_box((&s1, &s2));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, mie);
criterion_main!(benches);
