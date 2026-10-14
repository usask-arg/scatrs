//! Ensembles of spheres: identities, mixtures, Jacobians against finite differences, the
//! calibration of the error estimates against tight reference integrals, determinism, and
//! the rejection of bad inputs.
#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together

use core::f64::consts::PI;

use num_complex::Complex;
use scatrs::Error;
use scatrs::ensemble::{
    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, ParamKind,
    SizeDistribution, compute, compute_mixture, mixture_parameters, parameters,
};
use scatrs::mie::{GreekCoefficients, Mie, MieOptions, Normalization, PhaseBuffers};
use scatrs::special::gauss_legendre;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn lognormal(rg: f64, width: f64) -> SizeDistribution {
    SizeDistribution::LogNormal {
        median_radius: rg,
        width,
    }
}

fn one(
    d: &SizeDistribution,
    lambda: f64,
    m: Complex<f64>,
    opts: &EnsembleOptions,
) -> EnsembleResult {
    compute(d, &[lambda], &[m], opts).unwrap().remove(0)
}

fn greek_arrays(g: &GreekCoefficients<f64>) -> [&Vec<f64>; 6] {
    [
        &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
    ]
}

/// The scalar outputs, in a fixed order.
fn scalars(o: &Outputs) -> [f64; 11] {
    [
        o.c_ext,
        o.c_sca,
        o.c_abs,
        o.c_back,
        o.ssa,
        o.g,
        o.lidar_ratio,
        o.depolarization,
        o.effective_radius,
        o.mean_area,
        o.mean_volume,
    ]
}

#[test]
fn a_monodisperse_ensemble_is_the_single_sphere() {
    let (r, lambda, m) = (0.7, 0.55, c(1.5, 0.01));
    let mu = vec![1.0, 0.3, -1.0];
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::All)
        .with_mu(mu.clone());
    let e = one(
        &SizeDistribution::Monodisperse { radius: r },
        lambda,
        m,
        &opts,
    );
    let x = 2.0 * PI * r / lambda;
    let mut mie = Mie::<f64>::new();
    let s = mie.compute(x, m, &MieOptions::new()).unwrap();
    let q = s.efficiencies();
    let area = PI * r * r;
    assert_eq!(e.values.c_ext, area * q.q_ext);
    assert_eq!(e.values.c_sca, area * q.q_sca);
    assert_eq!(e.values.c_abs, area * q.q_abs);
    assert!((e.values.g - q.g).abs() < 1e-15);
    assert!((e.values.effective_radius - r).abs() < 1e-15);
    assert!(e.converged && e.panels == 0 && e.ranges == vec![(r, r)]);
    let g = s.greek(None, Normalization::Normalized).unwrap();
    let eg = e.values.greek.as_ref().unwrap();
    for (a, b) in greek_arrays(eg).into_iter().zip(greek_arrays(&g)) {
        assert_eq!(a.len(), b.len());
        for l in 0..a.len() {
            assert!(
                (a[l] - b[l]).abs() <= 1e-13 * g.alpha1[1].abs().max(1.0),
                "order {l}"
            );
        }
    }
    let mut f: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; mu.len()]);
    {
        let [f11, f12, f33, f34] = &mut f;
        s.phase_matrix(
            &mu,
            Normalization::Normalized,
            PhaseBuffers { f11, f12, f33, f34 },
        )
        .unwrap();
    }
    let p = e.values.phase.as_ref().unwrap();
    for (a, b) in [&p.f11, &p.f12, &p.f33, &p.f34].into_iter().zip(&f) {
        for j in 0..mu.len() {
            assert!((a[j] - b[j]).abs() <= 1e-13 * f[0][0], "{a:?} vs {b:?}");
        }
    }
}

#[test]
fn moments_and_normalizations_match_the_closed_forms() {
    let opts = EnsembleOptions::new()
        .with_rtol(1e-8)
        .with_greek(GreekOrder::Order(8));
    for d in [
        lognormal(0.08, 1.6),
        SizeDistribution::Gamma {
            effective_radius: 0.5,
            effective_variance: 0.15,
        },
        SizeDistribution::ModifiedGamma {
            alpha: 2.0,
            b: 15.0,
            gamma: 0.5,
        },
        SizeDistribution::PowerLaw {
            exponent: 3.0,
            min_radius: 0.02,
            max_radius: 1.0,
        },
    ] {
        let e = one(&d, 0.5, c(1.45, 0.001), &opts);
        let (m2, m3) = (d.moment(2).unwrap(), d.moment(3).unwrap());
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-7 * b;
        assert!(close(e.values.mean_area, PI * m2), "{d:?} area");
        assert!(
            close(e.values.mean_volume, 4.0 * PI / 3.0 * m3),
            "{d:?} volume"
        );
        assert!(close(e.values.effective_radius, m3 / m2), "{d:?} r_eff");
        let g = e.values.greek.as_ref().unwrap();
        assert_eq!(g.alpha1[0], 1.0);
        assert!(
            (g.alpha1[1] - 3.0 * e.values.g).abs() < 1e-12,
            "{d:?}: alpha1[1] = 3g"
        );
        assert!((e.values.c_ext - e.values.c_sca - e.values.c_abs).abs() < 1e-12 * e.values.c_ext);
        assert_eq!(e.values.depolarization, 0.0);
        let s = e.values;
        assert!((s.lidar_ratio - 4.0 * PI * s.c_ext / s.c_back).abs() < 1e-12 * s.lidar_ratio);
    }
}

#[test]
fn the_rayleigh_limit_is_the_sixth_moment() {
    // x ≪ 1: Q_sca = (8/3) x⁴ |K|² (1 + O(x²)), K = (m² − 1)/(m² + 2), so
    // ⟨C_sca⟩ = (8/3)(2π/λ)⁴ |K|² π ⟨r⁶⟩.
    let (lambda, m) = (1.0, c(1.5, 0.0));
    let d = lognormal(2e-4, 1.4);
    let e = one(&d, lambda, m, &EnsembleOptions::new().with_rtol(1e-9));
    let k = (m * m - 1.0) / (m * m + 2.0);
    let want = 8.0 / 3.0 * (2.0 * PI / lambda).powi(4) * k.norm_sqr() * PI * d.moment(6).unwrap();
    assert!(
        (e.values.c_sca / want - 1.0).abs() < 1e-5,
        "{} vs {want}",
        e.values.c_sca
    );
    assert_eq!(e.values.c_abs, 0.0);
    assert!(e.values.ssa == 1.0 && e.values.g.abs() < 1e-5);
}

#[test]
fn the_normalized_phase_function_averages_to_one() {
    let (nodes, weights) = {
        let r = gauss_legendre::<f64>(400).unwrap();
        (r.nodes, r.weights)
    };
    let opts = EnsembleOptions::new().with_mu(nodes.clone());
    let e = one(&lognormal(0.3, 1.5), 0.5, c(1.33, 1e-6), &opts);
    let f11 = &e.values.phase.as_ref().unwrap().f11;
    let mean: f64 = f11.iter().zip(&weights).map(|(f, w)| f * w).sum::<f64>() / 2.0;
    assert!((mean - 1.0).abs() < 1e-5, "{mean}");
    // Unnormalized: ⟨C_sca F⟩, so the mean is ⟨C_sca⟩, and α₁⁰ = ⟨C_sca⟩.
    let u = one(
        &lognormal(0.3, 1.5),
        0.5,
        c(1.33, 1e-6),
        &opts
            .clone()
            .with_normalization(Normalization::Unnormalized)
            .with_greek(GreekOrder::Order(4)),
    );
    let fu = &u.values.phase.as_ref().unwrap().f11;
    let mean: f64 = fu.iter().zip(&weights).map(|(f, w)| f * w).sum::<f64>() / 2.0;
    assert!((mean / u.values.c_sca - 1.0).abs() < 1e-5);
    let a0 = u.values.greek.as_ref().unwrap().alpha1[0];
    assert!((a0 / u.values.c_sca - 1.0).abs() < 1e-12);
}

#[test]
fn mixtures_average_with_the_number_weights() {
    let (lambda, opts) = (
        0.6,
        EnsembleOptions::new()
            .with_greek(GreekOrder::Order(10))
            .with_mu(vec![0.5]),
    );
    let (d1, m1, w1) = (lognormal(0.1, 1.5), c(1.45, 0.0), 2.0);
    let (d2, m2, w2) = (lognormal(1.0, 1.3), c(1.53, 0.01), 0.5);
    let mix = compute_mixture(
        &[
            Component::new(d1.clone(), vec![m1], w1),
            Component::new(d2.clone(), vec![m2], w2),
        ],
        &[lambda],
        &opts,
    )
    .unwrap()
    .remove(0);
    let (a, b) = (one(&d1, lambda, m1, &opts), one(&d2, lambda, m2, &opts));
    let (fa, fb) = (w1 / (w1 + w2), w2 / (w1 + w2));
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * y.abs().max(1e-30);
    assert!(close(
        mix.values.c_ext,
        fa * a.values.c_ext + fb * b.values.c_ext
    ));
    assert!(close(
        mix.values.mean_volume,
        fa * a.values.mean_volume + fb * b.values.mean_volume
    ));
    // Phase matrices average with wᵢ C_sca,ᵢ.
    let sca = fa * a.values.c_sca + fb * b.values.c_sca;
    let p = |r: &EnsembleResult| r.values.phase.as_ref().unwrap().f11[0];
    let want = (fa * a.values.c_sca * p(&a) + fb * b.values.c_sca * p(&b)) / sca;
    assert!((p(&mix) - want).abs() < 1e-10 * want);
    let g = |r: &EnsembleResult| r.values.greek.as_ref().unwrap().beta1[4];
    let want = (fa * a.values.c_sca * g(&a) + fb * b.values.c_sca * g(&b)) / sca;
    assert!((g(&mix) - want).abs() < 1e-10 * want.abs().max(1e-6));
    // A bimodal distribution is a mixture of its modes with one refractive index.
    let bimodal = SizeDistribution::MultiModal {
        modes: vec![(w1, d1.clone()), (w2, d2.clone())],
    };
    let bi = one(&bimodal, lambda, m1, &opts);
    let both = compute_mixture(
        &[
            Component::new(d1, vec![m1], w1),
            Component::new(d2, vec![m1], w2),
        ],
        &[lambda],
        &opts,
    )
    .unwrap()
    .remove(0);
    assert_eq!(scalars(&bi.values), scalars(&both.values));
}

/// The derivative of every scalar output with respect to `p`, against a central difference of
/// the whole computation at a tight tolerance.
fn check_derivative(
    run: &dyn Fn(f64, &EnsembleOptions) -> EnsembleResult,
    p: EnsembleParam,
    value: f64,
    what: &str,
) {
    let tight = EnsembleOptions::new()
        .with_rtol(1e-11)
        .with_max_panels(50_000)
        .with_greek(GreekOrder::Order(6))
        .with_mu(vec![0.8, -0.6]);
    let r = run(value, &tight.clone().with_derivatives(vec![p]));
    let h = 1e-4 * value.abs().max(1e-3);
    let (plus, minus) = (run(value + h, &tight), run(value - h, &tight));
    let d = scalars(&r.derivatives[0]);
    let (vp, vm, v) = (
        scalars(&plus.values),
        scalars(&minus.values),
        scalars(&r.values),
    );
    for k in 0..11 {
        let fd = (vp[k] - vm[k]) / (2.0 * h);
        let scale = fd.abs().max(v[k].abs() / value.abs().max(1e-3)).max(1e-300);
        assert!(
            (d[k] - fd).abs() <= 1e-5 * scale,
            "{what}, scalar {k}: {} vs {fd}",
            d[k]
        );
    }
    let gd = &r.derivatives[0].greek.as_ref().unwrap().beta1;
    let (gp, gm) = (
        &plus.values.greek.as_ref().unwrap().beta1,
        &minus.values.greek.as_ref().unwrap().beta1,
    );
    for l in 0..gd.len() {
        let fd = (gp[l] - gm[l]) / (2.0 * h);
        assert!(
            (gd[l] - fd).abs() <= 1e-5 * fd.abs().max(1e-3),
            "{what}, beta1[{l}]"
        );
    }
    let phase = |o: &Outputs| o.phase.as_ref().unwrap().f12.clone();
    let (pd, pp, pm) = (
        phase(&r.derivatives[0]),
        phase(&plus.values),
        phase(&minus.values),
    );
    let f11 = &r.values.phase.as_ref().unwrap().f11;
    for j in 0..pd.len() {
        let fd = (pp[j] - pm[j]) / (2.0 * h);
        let scale = fd.abs().max(f11[j] / value.abs().max(1e-3));
        assert!(
            (pd[j] - fd).abs() <= 1e-5 * scale,
            "{what}, F12 at cosine {j}: {} vs {fd}",
            pd[j]
        );
    }
}

#[test]
fn jacobians_match_finite_differences() {
    let lambda = 0.55;
    let m = c(1.45, 0.02);
    let base = lognormal(0.2, 1.6);
    let ps = parameters(&base);
    let median = ps[0];
    check_derivative(
        &|v, o| one(&lognormal(v, 1.6), lambda, m, o),
        median,
        0.2,
        "median_radius",
    );
    check_derivative(
        &|v, o| one(&lognormal(0.2, v), lambda, m, o),
        ps[1],
        1.6,
        "width",
    );
    let re = EnsembleParam::new(0, None, ParamKind::RefrReal);
    let im = EnsembleParam::new(0, None, ParamKind::RefrImag);
    check_derivative(
        &|v, o| one(&base, lambda, c(v, 0.02), o),
        re,
        1.45,
        "m_real",
    );
    check_derivative(
        &|v, o| one(&base, lambda, c(1.45, v), o),
        im,
        0.02,
        "m_imag",
    );
    let gamma = |a: f64, b: f64| SizeDistribution::Gamma {
        effective_radius: a,
        effective_variance: b,
    };
    let gp = parameters(&gamma(0.5, 0.1));
    check_derivative(
        &|v, o| one(&gamma(v, 0.1), lambda, m, o),
        gp[0],
        0.5,
        "effective_radius",
    );
    check_derivative(
        &|v, o| one(&gamma(0.5, v), lambda, m, o),
        gp[1],
        0.1,
        "effective_variance",
    );
    let mono = |r: f64| SizeDistribution::Monodisperse { radius: r };
    let mp = parameters(&mono(0.3))[0];
    check_derivative(&|v, o| one(&mono(v), lambda, m, o), mp, 0.3, "radius");
    let mg = |b: f64, g: f64| SizeDistribution::ModifiedGamma {
        alpha: 2.0,
        b,
        gamma: g,
    };
    let mgp = parameters(&mg(15.0, 0.5));
    check_derivative(&|v, o| one(&mg(v, 0.5), lambda, m, o), mgp[1], 15.0, "b");
    check_derivative(
        &|v, o| one(&mg(15.0, v), lambda, m, o),
        mgp[2],
        0.5,
        "gamma",
    );
    let pl = |a: f64| SizeDistribution::PowerLaw {
        exponent: a,
        min_radius: 0.02,
        max_radius: 1.0,
    };
    let plp = parameters(&pl(3.0))[0];
    check_derivative(&|v, o| one(&pl(v), lambda, m, o), plp, 3.0, "exponent");
    // Mixture weights.
    let mix = |w: f64, o: &EnsembleOptions| {
        compute_mixture(
            &[
                Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], w),
                Component::new(lognormal(0.8, 1.3), vec![c(1.53, 0.01)], 0.5),
            ],
            &[lambda],
            o,
        )
        .unwrap()
        .remove(0)
    };
    let all = mixture_parameters(&[
        Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], 1.0),
        Component::new(lognormal(0.8, 1.3), vec![c(1.53, 0.01)], 0.5),
    ]);
    let w0 = *all
        .iter()
        .find(|p| p.kind == ParamKind::Weight && p.component == 0)
        .unwrap();
    check_derivative(&mix, w0, 2.0, "c0.weight");
    // The second component's refractive index.
    let mix_m = |v: f64, o: &EnsembleOptions| {
        compute_mixture(
            &[
                Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], 2.0),
                Component::new(lognormal(0.8, 1.3), vec![c(v, 0.01)], 0.5),
            ],
            &[lambda],
            o,
        )
        .unwrap()
        .remove(0)
    };
    let re1 = EnsembleParam::new(1, None, ParamKind::RefrReal);
    check_derivative(&mix_m, re1, 1.53, "c1.m_real");
    // A mode weight.
    let bi = |w: f64, o: &EnsembleOptions| {
        one(
            &SizeDistribution::MultiModal {
                modes: vec![(w, lognormal(0.1, 1.5)), (0.5, lognormal(0.8, 1.3))],
            },
            lambda,
            m,
            o,
        )
    };
    let mode_w = EnsembleParam::new(0, Some(0), ParamKind::Weight);
    check_derivative(&bi, mode_w, 2.0, "mode0.weight");
    // A mode's distribution parameter.
    let modal = |w: f64, o: &EnsembleOptions| {
        one(
            &SizeDistribution::MultiModal {
                modes: vec![(2.0, lognormal(0.1, w)), (0.5, lognormal(0.8, 1.3))],
            },
            lambda,
            m,
            o,
        )
    };
    let mode_width = EnsembleParam::new(0, Some(0), ParamKind::Distribution("width"));
    check_derivative(&modal, mode_width, 1.5, "mode0.width");
}

/// The calibration of the error estimates on the grid below: the refined values (cross
/// sections, g, the moments) are within their estimates; every other output, derivatives
/// included, within twice its estimate (|K15 − G7| is an estimate, not a bound). Measured on
/// this grid: at most 0.002 of the estimate for the refined outputs and 1.023 for the others
/// (the test prints the worst ratios).
const ESTIMATE_FACTOR: f64 = 2.0;

/// Integration grid: every output of a result at rtol 1e-6 against the same at 1e-11.
#[test]
fn error_estimates_bound_the_true_errors() {
    let cases = [
        (lognormal(0.08, 1.6), c(1.43, 1e-8), 0.5),
        (lognormal(5.0, 1.3), c(1.5, 0.01), 0.5),
        (lognormal(0.001, 2.0), c(1.5, 0.0), 0.5),
        (lognormal(0.5, 1.05), c(1.33, 1e-3), 0.6),
        (
            SizeDistribution::Gamma {
                effective_radius: 2.0,
                effective_variance: 0.1,
            },
            c(1.33, 1e-3),
            0.65,
        ),
        (
            SizeDistribution::ModifiedGamma {
                alpha: 2.0,
                b: 15.0,
                gamma: 0.5,
            },
            c(1.5, 0.001),
            0.5,
        ),
        (
            SizeDistribution::PowerLaw {
                exponent: 3.5,
                min_radius: 0.01,
                max_radius: 5.0,
            },
            c(1.5, 0.01),
            0.5,
        ),
        (
            SizeDistribution::Tabulated {
                radius: vec![0.05, 0.1, 0.4, 1.0],
                density: vec![0.0, 3.0, 1.0, 0.2],
            },
            c(2.0, 0.5),
            0.4,
        ),
        (
            SizeDistribution::MultiModal {
                modes: vec![(1000.0, lognormal(0.05, 1.5)), (1.0, lognormal(1.0, 1.4))],
            },
            c(1.53, 0.005),
            0.55,
        ),
    ];
    let mu = vec![1.0, 0.7, 0.0, -1.0];
    // The worst ratio of true error to estimate, of refined and of other outputs.
    let worst = std::cell::Cell::new([0.0f64; 2]);
    let note = |refined: bool, err: f64, est: f64| {
        let mut w = worst.get();
        let i = usize::from(!refined);
        if est > 0.0 {
            w[i] = w[i].max(err / est);
        }
        worst.set(w);
    };
    for (d, m, lambda) in cases {
        let params = parameters(&d);
        // With backscattering refined, so that every estimate is a bound.
        let opts = EnsembleOptions::new()
            .with_greek(GreekOrder::Order(24))
            .with_mu(mu.clone())
            .with_derivatives(params.clone())
            .with_refine_backscatter(true);
        let a = one(&d, lambda, m, &opts);
        let b = one(
            &d,
            lambda,
            m,
            &opts.clone().with_rtol(1e-11).with_max_panels(50_000),
        );
        assert!(a.converged && b.converged, "{d:?}");
        let check = |x: &Outputs, y: &Outputs, e: &Outputs, what: &str| {
            let (xs, ys, es) = (scalars(x), scalars(y), scalars(e));
            for k in 0..11 {
                let err = (xs[k] - ys[k]).abs();
                let refined = matches!(k, 0 | 1 | 3 | 5 | 8 | 9 | 10) && what == "values";
                let factor = if refined { 1.0 } else { ESTIMATE_FACTOR };
                note(refined, err, es[k]);
                assert!(
                    err <= factor * es[k] + 1e-14 * ys[k].abs(),
                    "{d:?} {what} scalar {k}: {err:e} > {factor} × {:e}",
                    es[k]
                );
            }
            let (gx, gy, ge) = (
                x.greek.as_ref().unwrap(),
                y.greek.as_ref().unwrap(),
                e.greek.as_ref().unwrap(),
            );
            for ((ax, ay), ae) in greek_arrays(gx)
                .into_iter()
                .zip(greek_arrays(gy))
                .zip(greek_arrays(ge))
            {
                for l in 0..ax.len() {
                    let err = (ax[l] - ay[l]).abs();
                    note(false, err, ae[l]);
                    assert!(
                        err <= ESTIMATE_FACTOR * ae[l] + 1e-13,
                        "{d:?} {what} Greek l = {l}: {err:e} > {:e}",
                        ae[l]
                    );
                }
            }
            let (px, py, pe) = (
                x.phase.as_ref().unwrap(),
                y.phase.as_ref().unwrap(),
                e.phase.as_ref().unwrap(),
            );
            for (k, (vx, vy)) in [
                (&px.f11, &py.f11),
                (&px.f12, &py.f12),
                (&px.f33, &py.f33),
                (&px.f34, &py.f34),
            ]
            .into_iter()
            .enumerate()
            {
                let ve = [&pe.f11, &pe.f12, &pe.f33, &pe.f34][k];
                for j in 0..mu.len() {
                    let err = (vx[j] - vy[j]).abs();
                    note(false, err, ve[j]);
                    assert!(
                        err <= ESTIMATE_FACTOR * ve[j] + 1e-13 * vy[j].abs(),
                        "{d:?} {what} phase {k} at {}",
                        mu[j]
                    );
                }
            }
        };
        check(&a.values, &b.values, &a.errors, "values");
        for k in 0..params.len() {
            check(
                &a.derivatives[k],
                &b.derivatives[k],
                &a.derivative_errors[k],
                "derivative",
            );
        }
        // The refined quantities meet rtol.
        let rel = |x: f64, y: f64| (x - y).abs() / y.abs();
        assert!(rel(a.values.c_ext, b.values.c_ext) <= 1e-6, "{d:?} C_ext");
        assert!(rel(a.values.c_sca, b.values.c_sca) <= 1e-6, "{d:?} C_sca");
        assert!((a.values.g - b.values.g).abs() <= 2e-6, "{d:?} g");
    }
    let [refined, other] = worst.get();
    eprintln!(
        "worst error/estimate: {refined:.3} for the refined outputs, {other:.3} for the others"
    );
}

/// A broad, weakly absorbing distribution converges without backscattering refined, and
/// costs far more panels with it.
#[test]
fn backscattering_refinement_is_optional() {
    let d = lognormal(1.0, 2.0);
    let plain = one(&d, 1.0, c(1.33, 1e-4), &EnsembleOptions::new());
    assert!(plain.converged, "{} panels", plain.panels);
    let refined = one(
        &d,
        1.0,
        c(1.33, 1e-4),
        &EnsembleOptions::new().with_refine_backscatter(true),
    );
    assert!(refined.panels > plain.panels);
    assert!((refined.values.c_ext / plain.values.c_ext - 1.0).abs() < 2e-6);
}

/// Without absorption the derivatives with respect to Im m have a spike at every resonance,
/// which pass 1 cannot resolve: the result says so.
#[test]
fn unresolvable_derivatives_are_flagged() {
    let d = SizeDistribution::Gamma {
        effective_radius: 2.0,
        effective_variance: 0.1,
    };
    let im = EnsembleParam::new(0, None, ParamKind::RefrImag);
    let opts = EnsembleOptions::new()
        .with_max_panels(512)
        .with_derivatives(vec![im]);
    let r = one(&d, 0.65, c(1.33, 0.0), &opts);
    assert!(!r.converged && r.panels <= 512);
    // The values alone converge.
    assert!(one(&d, 0.65, c(1.33, 0.0), &EnsembleOptions::new()).converged);
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let d = lognormal(0.2, 1.4);
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::All)
        .with_mu(vec![0.2])
        .with_derivatives(parameters(&d));
    let wavelengths = [0.4, 0.55, 0.8];
    let m = vec![c(1.45, 0.001); 3];
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute(&d, &wavelengths, &m, &opts).unwrap())
    };
    assert_eq!(run(1), run(4));
    // A mixture with refractive-index derivatives.
    let components = [
        Component::new(d.clone(), m.clone(), 1.0),
        Component::new(lognormal(1.0, 1.3), vec![c(1.53, 0.01); 3], 0.01),
    ];
    let mopts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(8))
        .with_derivatives(mixture_parameters(&components));
    let run_mix = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute_mixture(&components, &wavelengths, &mopts).unwrap())
    };
    assert_eq!(run_mix(1), run_mix(3));
}

#[test]
fn a_uniform_table_is_a_flat_power_law() {
    let opts = EnsembleOptions::new().with_rtol(1e-9);
    let table = SizeDistribution::Tabulated {
        radius: vec![0.1, 0.5],
        density: vec![1.0, 1.0],
    };
    let flat = SizeDistribution::PowerLaw {
        exponent: 0.0,
        min_radius: 0.1,
        max_radius: 0.5,
    };
    let (a, b) = (
        one(&table, 0.5, c(1.5, 0.0), &opts),
        one(&flat, 0.5, c(1.5, 0.0), &opts),
    );
    assert!((a.values.c_ext / b.values.c_ext - 1.0).abs() < 1e-8);
}

#[test]
fn bad_inputs_are_rejected() {
    let d = lognormal(0.1, 1.5);
    let m = [c(1.5, 0.0)];
    let bad_opts = [
        EnsembleOptions::new().with_rtol(0.5),
        EnsembleOptions::new().with_rtol(1e-15),
        EnsembleOptions::new().with_max_panels(0),
        EnsembleOptions::new().with_mu(vec![1.5]),
        EnsembleOptions::new().with_greek(GreekOrder::Order(usize::MAX)),
        EnsembleOptions::new().with_derivatives(vec![EnsembleParam::new(
            0,
            None,
            ParamKind::Distribution("alpha"),
        )]),
        EnsembleOptions::new().with_derivatives(vec![EnsembleParam::new(
            1,
            None,
            ParamKind::RefrReal,
        )]),
    ];
    for o in bad_opts {
        let r = compute(&d, &[0.5], &m, &o);
        assert!(matches!(r, Err(Error::InvalidInput(_))), "{o:?}: {r:?}");
    }
    let o = EnsembleOptions::new();
    for (wl, mm) in [
        (vec![0.0], m.to_vec()),
        (vec![0.5, 0.6], m.to_vec()),
        (vec![f64::NAN], m.to_vec()),
    ] {
        assert!(matches!(
            compute(&d, &wl, &mm, &o),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(matches!(
        compute(&lognormal(0.1, 0.9), &[0.5], &m, &o),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        compute_mixture(&[], &[0.5], &o),
        Err(Error::InvalidInput(_))
    ));
    let zero = [Component::new(d.clone(), m.to_vec(), 0.0)];
    assert!(matches!(
        compute_mixture(&zero, &[0.5], &o),
        Err(Error::InvalidInput(_))
    ));
    let nested = SizeDistribution::MultiModal {
        modes: vec![(
            1.0,
            SizeDistribution::MultiModal {
                modes: vec![(1.0, d.clone())],
            },
        )],
    };
    assert!(matches!(
        compute(&nested, &[0.5], &m, &o),
        Err(Error::InvalidInput(_))
    ));
    // A size outside the validated domain names the wavelength and the radius.
    let big = SizeDistribution::Monodisperse { radius: 2000.0 };
    match compute(&big, &[0.5], &m, &o) {
        Err(Error::OutsideValidatedDomain(msg)) => assert!(msg.contains("wavelength"), "{msg}"),
        other => panic!("{other:?}"),
    }
    assert!(compute(&big, &[0.5], &m, &o.clone().with_allow_unvalidated(true)).is_ok());
    // An invalid refractive index.
    assert!(matches!(
        compute(&d, &[0.5], &[c(1.5, -0.1)], &o),
        Err(Error::InvalidInput(_))
    ));
    // No wavelengths: nothing to do.
    assert!(compute(&d, &[], &[], &o).unwrap().is_empty());
}

#[test]
fn derivative_parameters_are_named() {
    let bi = SizeDistribution::MultiModal {
        modes: vec![
            (1.0, lognormal(0.1, 1.5)),
            (0.5, SizeDistribution::Monodisperse { radius: 1.0 }),
        ],
    };
    let names: Vec<String> = parameters(&bi).iter().map(|p| p.name(false)).collect();
    assert_eq!(
        names,
        [
            "mode0.median_radius",
            "mode0.width",
            "mode0.weight",
            "mode1.radius",
            "mode1.weight",
            "m_real",
            "m_imag"
        ]
    );
    let mix = mixture_parameters(&[Component::new(lognormal(0.1, 1.5), vec![], 1.0)]);
    assert_eq!(mix.last().unwrap().name(true), "c0.weight");
}

/// Narrow, broad and degenerate inputs give finite results or an error, never a silent NaN.
#[test]
fn narrow_and_extreme_distributions() {
    let m = c(1.45, 0.001);
    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(4));
    let finite = |r: &EnsembleResult| {
        scalars(&r.values).iter().all(|v| v.is_finite())
            && r.values
                .greek
                .as_ref()
                .unwrap()
                .alpha1
                .iter()
                .all(|v| v.is_finite())
    };
    // Nearly monodisperse: the single sphere.
    let mono = one(
        &SizeDistribution::Monodisperse { radius: 0.3 },
        0.5,
        m,
        &opts,
    );
    for d in [
        lognormal(0.3, 1.000_001),
        SizeDistribution::Gamma {
            effective_radius: 0.3,
            effective_variance: 1e-9,
        },
    ] {
        let r = one(&d, 0.5, m, &opts);
        assert!(r.converged && finite(&r), "{d:?}");
        assert!(
            (r.values.c_ext / mono.values.c_ext - 1.0).abs() < 1e-4,
            "{d:?}"
        );
    }
    // A broad gamma (b near 1/2) and a steeply falling power law.
    for d in [
        SizeDistribution::Gamma {
            effective_radius: 0.3,
            effective_variance: 0.45,
        },
        SizeDistribution::PowerLaw {
            exponent: -200.0,
            min_radius: 0.01,
            max_radius: 1.0,
        },
    ] {
        // (The broad gamma's number tail reaches x < 10⁻⁶.)
        let unvalidated = opts.clone().with_allow_unvalidated(true);
        assert!(finite(&one(&d, 0.5, m, &unvalidated)), "{d:?}");
    }
    // A spike table with fewer panels allowed than it has intervals.
    let spike = SizeDistribution::Tabulated {
        radius: vec![0.1, 0.2, 0.2001, 0.2002, 0.5],
        density: vec![0.0, 0.0, 1.0, 0.0, 0.0],
    };
    let r = one(&spike, 0.5, m, &opts.clone().with_max_panels(1));
    assert!(finite(&r) && r.values.c_ext > 0.0);
    let invalid = |r: Result<Vec<EnsembleResult>, Error>| matches!(r, Err(Error::InvalidInput(_)));
    // m = 1: nothing scatters, so the ratios are undefined.
    assert!(invalid(compute(
        &lognormal(0.3, 1.5),
        &[0.5],
        &[c(1.0, 0.0)],
        &opts
    )));
    // Size parameters beyond what Mie accepts.
    let unvalidated = opts.clone().with_allow_unvalidated(true);
    assert!(invalid(compute(
        &lognormal(0.3, 1.5),
        &[1e-7],
        &[m],
        &unvalidated
    )));
    // Duplicate parameters and too high a refinement order.
    let p = parameters(&lognormal(0.3, 1.5));
    for o in [
        EnsembleOptions::new().with_derivatives(vec![p[1], p[1]]),
        EnsembleOptions::new().with_refine_order(100_000),
    ] {
        assert!(invalid(compute(&lognormal(0.3, 1.5), &[0.5], &[m], &o)));
    }
    // Weights whose sum overflows.
    let huge = [
        Component::new(lognormal(0.3, 1.5), vec![m], f64::MAX),
        Component::new(lognormal(0.3, 1.5), vec![m], f64::MAX),
    ];
    assert!(invalid(compute_mixture(
        &huge,
        &[0.5],
        &EnsembleOptions::new()
    )));
}
