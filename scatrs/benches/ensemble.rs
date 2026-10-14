//! Wall-clock benchmarks of the ensemble averages (criterion), with the Greek coefficients to
//! order 63 as an optics database computes them, at rtol 1e-4:
//!
//! * `w1`: one broad lognormal (r_g = 1 µm, σ_g = 2) at 0.3, 0.5 and 1 µm, m = 1.33 + 1e-4i;
//! * `w1-derivatives`: the same with the derivatives with respect to every parameter;
//! * `w1-mu`: the same with the phase matrix at 181 angles instead;
//! * `large`: a lognormal of large spheres (r_g = 5 µm, σ_g = 1.5) at 0.5 µm;
//! * `batch`: 12 lognormals (r_g in [0.05, 1] µm × σ_g in {1.2, 1.6, 2.0}) at 5 wavelengths in
//!   [0.28, 1.02] µm, m = 1.43 + 1e-8i, by `compute_batch`.
//!
//! Set `RAYON_NUM_THREADS=1` for one thread.
//!
//! ```text
//! cargo bench -p scatrs --bench ensemble
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use scatrs::ensemble::{
    EnsembleOptions, GreekOrder, SizeDistribution, compute, compute_batch, parameters,
};

fn ensemble(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble");
    group.sample_size(10);
    let options = EnsembleOptions::new()
        .with_rtol(1e-4)
        .with_greek(GreekOrder::Order(63));
    let broad = SizeDistribution::LogNormal {
        median_radius: 1.0,
        width: 2.0,
    };
    let wavelengths = [0.3, 0.5, 1.0];
    let m = [Complex::new(1.33, 1e-4); 3];
    group.bench_function("w1", |b| {
        b.iter(|| compute(black_box(&broad), &wavelengths, &m, &options).unwrap())
    });
    let with_derivatives = options.clone().with_derivatives(parameters(&broad));
    group.bench_function("w1-derivatives", |b| {
        b.iter(|| compute(black_box(&broad), &wavelengths, &m, &with_derivatives).unwrap())
    });
    let mu: Vec<f64> = (0..181)
        .map(|k| (f64::from(k).to_radians()).cos())
        .collect();
    let with_mu = EnsembleOptions::new().with_rtol(1e-4).with_mu(mu);
    group.bench_function("w1-mu", |b| {
        b.iter(|| compute(black_box(&broad), &wavelengths, &m, &with_mu).unwrap())
    });
    let large = SizeDistribution::LogNormal {
        median_radius: 5.0,
        width: 1.5,
    };
    group.bench_function("large", |b| {
        b.iter(|| {
            compute(
                black_box(&large),
                &[0.5],
                &[Complex::new(1.5, 1e-3)],
                &options,
            )
            .unwrap()
        })
    });
    let radii = [0.05, 0.135, 0.37, 1.0];
    let batch: Vec<SizeDistribution> = radii
        .iter()
        .flat_map(|&r| {
            [1.2, 1.6, 2.0].map(|s| SizeDistribution::LogNormal {
                median_radius: r,
                width: s,
            })
        })
        .collect();
    let wavelengths: Vec<f64> = (0..5).map(|k| 0.28 + 0.185 * f64::from(k)).collect();
    let m = vec![Complex::new(1.43, 1e-8); wavelengths.len()];
    group.bench_function("batch", |b| {
        b.iter(|| compute_batch(black_box(&batch), &wavelengths, &m, &options).unwrap())
    });
    group.finish();
}

criterion_group!(benches, ensemble);
criterion_main!(benches);
