//! Ensembles of randomly oriented spheroids (the engine's direct path): an aspect ratio of 1 is
//! the sphere ensemble within the error estimates, a monodisperse ensemble is the single
//! spheroid, Jacobians against finite differences (the aspect ratio, Re m, Im m, the
//! distribution, the radius of a monodisperse ensemble), mixtures with spheres, determinism,
//! and the error estimates against a tight reference.
#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together

use core::f64::consts::PI;

use num_complex::Complex;
use scatrs::ensemble::{
    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, ParamKind,
    Particle, SizeDistribution, compute, compute_batch_with_particle, compute_mixture,
    compute_with_particle, mixture_parameters, parameters_with_particle,
};
use scatrs::mie::Normalization;
use scatrs::tmatrix::{PhaseBuffers, Shape, TMatrix, TMatrixOptions};

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn lognormal(rg: f64, width: f64) -> SizeDistribution {
    SizeDistribution::LogNormal {
        median_radius: rg,
        width,
    }
}

const DUST: Complex<f64> = Complex::new(1.53, 0.008);
const LAMBDA: f64 = 1.0;

fn one(d: &SizeDistribution, p: &Particle, opts: &EnsembleOptions) -> EnsembleResult {
    compute_with_particle(d, &[LAMBDA], p, opts)
        .unwrap()
        .remove(0)
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

/// The six phase-matrix elements.
fn phase(o: &Outputs) -> [Vec<f64>; 6] {
    let p = o.phase.as_ref().unwrap();
    [
        p.f11.clone(),
        p.f12.clone(),
        p.f22.clone(),
        p.f33.clone(),
        p.f34.clone(),
        p.f44.clone(),
    ]
}

/// The six Greek arrays.
fn greek(o: &Outputs) -> [Vec<f64>; 6] {
    let g = o.greek.as_ref().unwrap();
    [
        g.alpha1.clone(),
        g.alpha2.clone(),
        g.alpha3.clone(),
        g.alpha4.clone(),
        g.beta1.clone(),
        g.beta2.clone(),
    ]
}

fn param(kind: ParamKind) -> EnsembleParam {
    EnsembleParam::new(0, None, kind)
}

#[test]
fn aspect_ratio_one_is_the_sphere_ensemble() {
    let d = lognormal(0.05, 1.5);
    let mu = vec![1.0, 0.6, 0.0, -0.7, -1.0];
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(12))
        .with_mu(mu);
    let sphere = compute(&d, &[LAMBDA], &[DUST], &opts).unwrap().remove(0);
    let spheroid = one(&d, &Particle::spheroid(1.0, vec![DUST]), &opts);
    assert!(spheroid.converged && spheroid.panels > 0);
    let (v, e, s) = (
        scalars(&spheroid.values),
        scalars(&spheroid.errors),
        scalars(&sphere.values),
    );
    for k in 0..11 {
        // The two integrations differ by at most both estimates; the depolarization is the
        // spheroid's own (exactly 0 for the sphere), within its estimate.
        let bound = e[k] + scalars(&sphere.errors)[k] + 1e-14 * s[k].abs();
        assert!(
            (v[k] - s[k]).abs() <= bound,
            "scalar {k}: {} vs {}",
            v[k],
            s[k]
        );
    }
    let (ga, gs, ge) = (
        greek(&spheroid.values),
        greek(&sphere.values),
        greek(&spheroid.errors),
    );
    for k in 0..6 {
        for l in 0..ga[k].len() {
            assert!(
                (ga[k][l] - gs[k][l]).abs() <= ge[k][l] + 1e-12,
                "Greek {k}, order {l}: {} vs {}",
                ga[k][l],
                gs[k][l]
            );
        }
    }
    // F₂₂ = F₁₁ and F₄₄ = F₃₃, as for the sphere.
    let (pa, ps, pe) = (
        phase(&spheroid.values),
        phase(&sphere.values),
        phase(&spheroid.errors),
    );
    for k in 0..6 {
        for j in 0..pa[k].len() {
            assert!(
                (pa[k][j] - ps[k][j]).abs() <= pe[k][j] + 1e-10 * ps[0][j],
                "phase {k}, cosine {j}: {} vs {}",
                pa[k][j],
                ps[k][j]
            );
        }
    }
    assert_eq!(ps[2], ps[0]);
    assert_eq!(ps[5], ps[3]);
}

#[test]
fn a_monodisperse_ensemble_is_the_single_spheroid() {
    let (r, eps) = (0.3, 2.0);
    let d = SizeDistribution::Monodisperse { radius: r };
    let mu = vec![0.9, 0.0, -1.0];
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::All)
        .with_mu(mu.clone())
        .with_normalization(Normalization::Unnormalized);
    let e = one(&d, &Particle::spheroid(eps, vec![DUST]), &opts);
    let x = 2.0 * PI * r / LAMBDA;
    let mut tm = TMatrix::new();
    let s = tm
        .compute(
            Shape::Spheroid { aspect_ratio: eps },
            x,
            DUST,
            &TMatrixOptions::new().with_rtol(1e-8),
        )
        .unwrap();
    let area = PI * r * r;
    let q = s.efficiencies();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * b.abs();
    assert!(close(e.values.c_ext, area * q.q_ext));
    assert!(close(e.values.c_sca, area * q.q_sca));
    assert!(close(e.values.c_back, area * q.q_back));
    assert!((e.values.g - q.g).abs() <= 1e-15);
    // The node's declared errors: C·max(rtol, estimate), C = 4 for C_ext.
    let tau = 1e-8f64.max(s.error_estimate());
    assert!((e.errors.c_ext - 4.0 * tau * area * q.q_ext).abs() <= 1e-12 * e.errors.c_ext);
    let g = s.greek(None, Normalization::Unnormalized).unwrap();
    let ge = greek(&e.values);
    assert_eq!(ge[0].len(), g.alpha1.len());
    for (k, a) in [
        &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
    ]
    .into_iter()
    .enumerate()
    {
        for l in 0..a.len() {
            assert!(
                (ge[k][l] - area * a[l]).abs() <= 1e-14 * area * g.alpha1[0],
                "{k} {l}"
            );
        }
    }
    let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; mu.len()]);
    {
        let [f11, f12, f22, f33, f34, f44] = &mut f;
        s.phase_matrix(
            &mu,
            Normalization::Unnormalized,
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
    let pe = phase(&e.values);
    for k in 0..6 {
        for j in 0..mu.len() {
            assert!(
                (pe[k][j] - area * f[k][j]).abs() <= 1e-13 * area * f[0][0],
                "phase {k} at {j}: {} vs {}",
                pe[k][j],
                area * f[k][j]
            );
        }
    }
    // The depolarization ratio from F₁₁ and F₂₂ at 180°.
    let (f11, f22) = (f[0][2], f[2][2]);
    assert!((e.values.depolarization - (f11 - f22) / (f11 + f22)).abs() <= 1e-12);
    assert!(e.values.depolarization > 0.01);
}

/// Checks the derivative with respect to `p` (at `value`) against central differences of
/// `run`, for every scalar output, the Greek coefficients α₁ and β₁ and the phase matrix.
fn check_derivative(
    run: &dyn Fn(f64, &EnsembleOptions) -> EnsembleResult,
    p: EnsembleParam,
    value: f64,
    what: &str,
) {
    let tight = EnsembleOptions::new()
        .with_rtol(1e-9)
        .with_max_panels(50_000)
        .with_greek(GreekOrder::Order(6))
        .with_mu(vec![0.8, -0.6, -1.0]);
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
    for k in [0, 4] {
        let (gd, gp, gm) = (
            &greek(&r.derivatives[0])[k],
            &greek(&plus.values)[k],
            &greek(&minus.values)[k],
        );
        for l in 0..gd.len() {
            let fd = (gp[l] - gm[l]) / (2.0 * h);
            assert!(
                (gd[l] - fd).abs() <= 1e-5 * fd.abs().max(1e-3),
                "{what}, Greek {k}[{l}]: {} vs {fd}",
                gd[l]
            );
        }
    }
    let f11 = &phase(&r.values)[0];
    let (pd, pp, pm) = (
        phase(&r.derivatives[0]),
        phase(&plus.values),
        phase(&minus.values),
    );
    for k in 0..6 {
        for j in 0..pd[k].len() {
            let fd = (pp[k][j] - pm[k][j]) / (2.0 * h);
            let scale = fd.abs().max(f11[j] / value.abs().max(1e-3));
            assert!(
                (pd[k][j] - fd).abs() <= 1e-5 * scale,
                "{what}, element {k} at cosine {j}: {} vs {fd}",
                pd[k][j]
            );
        }
    }
}

#[test]
fn jacobians_match_finite_differences() {
    let d = lognormal(0.03, 1.4);
    let eps = 2.0;
    let spheroid = |eps: f64, m: Complex<f64>| Particle::spheroid(eps, vec![m]);
    check_derivative(
        &|v, o| one(&d, &spheroid(v, DUST), o),
        param(ParamKind::AspectRatio),
        eps,
        "aspect_ratio",
    );
    check_derivative(
        &|v, o| one(&d, &spheroid(v, DUST), o),
        param(ParamKind::AspectRatio),
        0.5,
        "aspect_ratio (prolate)",
    );
    check_derivative(
        &|v, o| one(&d, &spheroid(eps, c(v, DUST.im)), o),
        param(ParamKind::RefrReal),
        DUST.re,
        "m_real",
    );
    check_derivative(
        &|v, o| one(&d, &spheroid(eps, c(DUST.re, v)), o),
        param(ParamKind::RefrImag),
        DUST.im,
        "m_imag",
    );
    let p = spheroid(eps, DUST);
    let ps = parameters_with_particle(&d, &p);
    check_derivative(
        &|v, o| one(&lognormal(v, 1.4), &p, o),
        ps[0],
        0.03,
        "median_radius",
    );
    check_derivative(&|v, o| one(&lognormal(0.03, v), &p, o), ps[1], 1.4, "width");
    // A monodisperse ensemble's radius, at a fixed aspect ratio: ∂/∂x.
    let mono = |r: f64| SizeDistribution::Monodisperse { radius: r };
    let mp = parameters_with_particle(&mono(0.2), &p)[0];
    check_derivative(&|v, o| one(&mono(v), &p, o), mp, 0.2, "radius");
    // A mixture's spheroid component.
    let mix = |v: f64, o: &EnsembleOptions| {
        compute_mixture(
            &[
                Component::new(lognormal(0.05, 1.5), vec![c(1.45, 0.0)], 2.0),
                Component::with_particle(lognormal(0.03, 1.4), spheroid(v, DUST), 0.5),
            ],
            &[LAMBDA],
            o,
        )
        .unwrap()
        .remove(0)
    };
    let ratio = EnsembleParam::new(1, None, ParamKind::AspectRatio);
    check_derivative(&mix, ratio, 0.5, "c1.aspect_ratio");
}

#[test]
fn mixtures_of_spheres_and_spheroids_average_with_the_weights() {
    let mu = vec![0.5, -1.0];
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(4))
        .with_mu(mu)
        .with_normalization(Normalization::Unnormalized);
    let wl = [LAMBDA];
    let (d0, d1) = (lognormal(0.05, 1.5), lognormal(0.08, 1.4));
    let p1 = Particle::spheroid(3.0, vec![DUST]);
    let components = [
        Component::new(d0.clone(), vec![c(1.45, 0.0)], 2.0),
        Component::with_particle(d1.clone(), p1.clone(), 0.5),
    ];
    let mixed = compute_mixture(&components, &wl, &opts).unwrap().remove(0);
    let a = compute(&d0, &wl, &[c(1.45, 0.0)], &opts).unwrap().remove(0);
    let b = one(&d1, &p1, &opts);
    let (fa, fb) = (2.0 / 2.5, 0.5 / 2.5);
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-14 * y.abs();
    assert!(close(
        mixed.values.c_ext,
        fa * a.values.c_ext + fb * b.values.c_ext
    ));
    // Unnormalized, the Greek coefficients and the phase matrix average with the number weights.
    let (gm, ga, gb) = (greek(&mixed.values), greek(&a.values), greek(&b.values));
    for k in 0..6 {
        for l in 0..gm[k].len() {
            let want = fa * ga[k][l] + fb * gb[k][l];
            assert!(
                (gm[k][l] - want).abs() <= 1e-14 * ga[0][0],
                "Greek {k}[{l}]"
            );
        }
    }
    let (pm, pa, pb) = (phase(&mixed.values), phase(&a.values), phase(&b.values));
    for k in 0..6 {
        for j in 0..pm[k].len() {
            let want = fa * pa[k][j] + fb * pb[k][j];
            assert!(
                (pm[k][j] - want).abs() <= 1e-13 * pa[0][j],
                "phase {k} at {j}"
            );
        }
    }
    // The sphere's F₂₂ is its F₁₁; the mixture's is not, and it depolarizes, less than the
    // spheroids alone.
    assert_eq!(pa[2], pa[0]);
    assert!(mixed.values.depolarization > 0.0);
    assert!(mixed.values.depolarization < b.values.depolarization);
    let names: Vec<String> = mixture_parameters(&components)
        .iter()
        .map(|p| p.name(true))
        .collect();
    assert_eq!(
        names,
        [
            "c0.median_radius",
            "c0.width",
            "c0.m_real",
            "c0.m_imag",
            "c0.weight",
            "c1.median_radius",
            "c1.width",
            "c1.m_real",
            "c1.m_imag",
            "c1.aspect_ratio",
            "c1.weight",
        ]
    );
}

#[test]
fn a_batch_matches_each_distribution() {
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(6))
        .with_mu(vec![-1.0]);
    let p = Particle::spheroid(2.0, vec![DUST]);
    let ds = [lognormal(0.03, 1.4), lognormal(0.045, 1.3)];
    let batch = compute_batch_with_particle(&ds, &[LAMBDA], &p, &opts).unwrap();
    for (d, b) in ds.iter().zip(&batch) {
        let (b, own) = (&b[0], one(d, &p, &opts));
        assert!(b.converged);
        let (vb, eb, vo, eo) = (
            scalars(&b.values),
            scalars(&b.errors),
            scalars(&own.values),
            scalars(&own.errors),
        );
        for k in 0..11 {
            assert!((vb[k] - vo[k]).abs() <= eb[k] + eo[k], "scalar {k}");
        }
        let (pb, po) = (phase(&b.values), phase(&own.values));
        let (peb, peo) = (phase(&b.errors), phase(&own.errors));
        for k in 0..6 {
            assert!(
                (pb[k][0] - po[k][0]).abs() <= peb[k][0] + peo[k][0],
                "phase {k}"
            );
        }
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let d = lognormal(0.04, 1.4);
    let p = Particle::spheroid(0.5, vec![DUST]);
    let ps = parameters_with_particle(&d, &p);
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(6))
        .with_mu(vec![0.2, -1.0])
        .with_derivatives(vec![ps[0], ps[4]]);
    let wavelengths = [LAMBDA];
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute_with_particle(&d, &wavelengths, &p, &opts).unwrap())
    };
    let base = run(1);
    assert_eq!(base, run(2));
    assert_eq!(base, run(8));
}

#[test]
fn error_estimates_bound_the_true_errors() {
    // Small enough that even the reference's range stays inside the validated domain.
    let d = lognormal(0.03, 1.4);
    let p = Particle::spheroid(1.0 / 3.0, vec![DUST]);
    let opts = |rtol: f64| {
        EnsembleOptions::new()
            .with_rtol(rtol)
            .with_max_panels(50_000)
            .with_refine_backscatter(true)
            .with_greek(GreekOrder::Order(8))
            .with_mu(vec![1.0, 0.0, -1.0])
    };
    let r = one(&d, &p, &opts(1e-6));
    let reference = one(&d, &p, &opts(1e-10));
    assert!(r.converged && reference.converged);
    let (v, e, x) = (
        scalars(&r.values),
        scalars(&r.errors),
        scalars(&reference.values),
    );
    let names = [
        "c_ext", "c_sca", "c_abs", "c_back", "ssa", "g", "lidar", "depol", "r_eff", "area", "vol",
    ];
    for k in 0..11 {
        let err = (v[k] - x[k]).abs();
        eprintln!("{:6} error {err:.2e} estimate {:.2e}", names[k], e[k]);
        assert!(
            err <= e[k],
            "{}: error {err:e} above its estimate {:e}",
            names[k],
            e[k]
        );
    }
    let (gv, ge, gx) = (greek(&r.values), greek(&r.errors), greek(&reference.values));
    let (pv, pe, px) = (phase(&r.values), phase(&r.errors), phase(&reference.values));
    for k in 0..6 {
        for l in 0..gv[k].len() {
            assert!(
                (gv[k][l] - gx[k][l]).abs() <= 2.0 * ge[k][l] + 1e-15,
                "Greek {k}[{l}]"
            );
        }
        for j in 0..pv[k].len() {
            assert!(
                (pv[k][j] - px[k][j]).abs() <= 2.0 * pe[k][j] + 1e-15,
                "phase {k} at {j}"
            );
        }
    }
}
