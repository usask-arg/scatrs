//! Ensembles of layered spheres: one layer is the sphere ensemble bitwise, Jacobians against
//! finite differences (each layer's index, the radius fractions, the distribution, the radius
//! of a monodisperse ensemble), mixtures with spheres, batches, determinism, and the rejection
//! of bad inputs.
#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together

use num_complex::Complex;
use scatrs::Error;
use scatrs::ensemble::{
    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, ParamKind,
    Particle, SizeDistribution, compute, compute_batch_with_particle, compute_mixture,
    compute_with_particle, mixture_parameters, parameters, parameters_with_particle,
};

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn lognormal(rg: f64, width: f64) -> SizeDistribution {
    SizeDistribution::LogNormal {
        median_radius: rg,
        width,
    }
}

/// Black carbon in sulfate, with the core's radius fraction `f`, at one wavelength.
fn coated(f: f64, core: Complex<f64>, shell: Complex<f64>) -> Particle {
    Particle::layered_sphere(vec![f], vec![vec![core], vec![shell]])
}

const CORE: Complex<f64> = Complex::new(1.95, 0.79);
const SHELL: Complex<f64> = Complex::new(1.43, 0.01);

fn one(d: &SizeDistribution, lambda: f64, p: &Particle, opts: &EnsembleOptions) -> EnsembleResult {
    compute_with_particle(d, &[lambda], p, opts)
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

fn param(kind: ParamKind) -> EnsembleParam {
    EnsembleParam::new(0, None, kind)
}

#[test]
fn one_layer_is_the_sphere_ensemble_bitwise() {
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(12))
        .with_mu(vec![-0.7, 0.0, 0.9]);
    let m = vec![c(1.45, 0.02), c(1.5, 1e-4)];
    let wavelengths = [0.45, 0.8];
    let layered = Particle::layered_sphere(Vec::new(), vec![m.clone()]);
    for d in [
        lognormal(0.2, 1.6),
        SizeDistribution::Monodisperse { radius: 0.3 },
    ] {
        let sphere_params = parameters(&d);
        let layered_params = parameters_with_particle(&d, &layered);
        let n = sphere_params.len();
        assert_eq!(n, layered_params.len());
        assert_eq!(layered_params[n - 2].kind, ParamKind::LayerRefrReal(0));
        let want = compute(
            &d,
            &wavelengths,
            &m,
            &opts.clone().with_derivatives(sphere_params),
        )
        .unwrap();
        let got = compute_with_particle(
            &d,
            &wavelengths,
            &layered,
            &opts.clone().with_derivatives(layered_params.clone()),
        )
        .unwrap();
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(w.values, g.values);
            assert_eq!(w.errors, g.errors);
            assert_eq!(w.derivatives, g.derivatives);
            assert_eq!(w.derivative_errors, g.derivative_errors);
            assert_eq!((w.panels, &w.ranges), (g.panels, &g.ranges));
            assert_eq!(g.params, layered_params);
        }
    }
}

#[test]
fn layers_of_one_index_are_the_sphere_ensemble_bitwise() {
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::Order(8))
        .with_mu(vec![0.3]);
    let m = c(1.5, 0.01);
    let d = lognormal(0.3, 1.5);
    let want = compute(&d, &[0.55], &[m], &opts).unwrap();
    let three = Particle::layered_sphere(vec![0.3, 0.7], vec![vec![m]; 3]);
    let got = compute_with_particle(&d, &[0.55], &three, &opts).unwrap();
    assert_eq!(want[0].values, got[0].values);
}

/// The derivative of every scalar output, β₁ and F₁₁, F₁₂ with respect to `p`, against a
/// central difference of the whole computation at a tight tolerance.
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
    let greek = |o: &Outputs| o.greek.as_ref().unwrap().beta1.clone();
    let (gd, gp, gm) = (
        greek(&r.derivatives[0]),
        greek(&plus.values),
        greek(&minus.values),
    );
    for l in 0..gd.len() {
        let fd = (gp[l] - gm[l]) / (2.0 * h);
        assert!(
            (gd[l] - fd).abs() <= 1e-5 * fd.abs().max(1e-3),
            "{what}, beta1[{l}]: {} vs {fd}",
            gd[l]
        );
    }
    let f11 = &r.values.phase.as_ref().unwrap().f11;
    for element in 0..2 {
        let phase = |o: &Outputs| {
            let p = o.phase.as_ref().unwrap();
            if element == 0 {
                p.f11.clone()
            } else {
                p.f12.clone()
            }
        };
        let (pd, pp, pm) = (
            phase(&r.derivatives[0]),
            phase(&plus.values),
            phase(&minus.values),
        );
        for j in 0..pd.len() {
            let fd = (pp[j] - pm[j]) / (2.0 * h);
            let scale = fd.abs().max(f11[j] / value.abs().max(1e-3));
            assert!(
                (pd[j] - fd).abs() <= 1e-5 * scale,
                "{what}, element {element} at cosine {j}: {} vs {fd}",
                pd[j]
            );
        }
    }
}

#[test]
fn jacobians_match_finite_differences() {
    let lambda = 0.55;
    let d = lognormal(0.15, 1.5);
    let f = 0.6;
    let with = |f: f64, core: Complex<f64>, shell: Complex<f64>| {
        let d = d.clone();
        move |o: &EnsembleOptions| one(&d, lambda, &coated(f, core, shell), o)
    };
    check_derivative(
        &|v, o| with(v, CORE, SHELL)(o),
        param(ParamKind::RadiusFraction(0)),
        f,
        "layer0.radius_fraction",
    );
    check_derivative(
        &|v, o| with(f, c(v, CORE.im), SHELL)(o),
        param(ParamKind::LayerRefrReal(0)),
        CORE.re,
        "layer0.m_real",
    );
    check_derivative(
        &|v, o| with(f, c(CORE.re, v), SHELL)(o),
        param(ParamKind::LayerRefrImag(0)),
        CORE.im,
        "layer0.m_imag",
    );
    check_derivative(
        &|v, o| with(f, CORE, c(v, SHELL.im))(o),
        param(ParamKind::LayerRefrReal(1)),
        SHELL.re,
        "layer1.m_real",
    );
    check_derivative(
        &|v, o| with(f, CORE, c(SHELL.re, v))(o),
        param(ParamKind::LayerRefrImag(1)),
        SHELL.im,
        "layer1.m_imag",
    );
    // The distribution, of the outer radius.
    let p = coated(f, CORE, SHELL);
    let ps = parameters_with_particle(&d, &p);
    check_derivative(
        &|v, o| one(&lognormal(v, 1.5), lambda, &p, o),
        ps[0],
        0.15,
        "median_radius",
    );
    // A monodisperse ensemble's radius, at fixed fractions: Σ f_l ∂/∂x_l.
    let mono = |r: f64| SizeDistribution::Monodisperse { radius: r };
    let mp = parameters_with_particle(&mono(0.3), &p)[0];
    check_derivative(&|v, o| one(&mono(v), lambda, &p, o), mp, 0.3, "radius");
    // Three layers: the middle fraction and the middle index.
    let three = |f1: f64, m1: Complex<f64>| {
        Particle::layered_sphere(
            vec![0.3, f1],
            vec![vec![c(1.6, 0.3)], vec![m1], vec![c(1.33, 1e-6)]],
        )
    };
    check_derivative(
        &|v, o| one(&d, lambda, &three(v, c(2.0, 0.05)), o),
        param(ParamKind::RadiusFraction(1)),
        0.7,
        "three layers: layer1.radius_fraction",
    );
    check_derivative(
        &|v, o| one(&d, lambda, &three(0.7, c(2.0, v)), o),
        param(ParamKind::LayerRefrImag(1)),
        0.05,
        "three layers: layer1.m_imag",
    );
    // A mixture's layered component.
    let mix = |v: f64, o: &EnsembleOptions| {
        compute_mixture(
            &[
                Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], 2.0),
                Component::with_particle(lognormal(0.3, 1.4), coated(v, CORE, SHELL), 0.5),
            ],
            &[lambda],
            o,
        )
        .unwrap()
        .remove(0)
    };
    let fraction = EnsembleParam::new(1, None, ParamKind::RadiusFraction(0));
    check_derivative(&mix, fraction, 0.5, "c1.layer0.radius_fraction");
}

#[test]
fn mixtures_of_spheres_and_layered_spheres_average_with_the_number_weights() {
    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(4));
    let wl = [0.55];
    let (d0, d1) = (lognormal(0.1, 1.5), lognormal(0.3, 1.4));
    let p1 = coated(0.5, CORE, SHELL);
    let components = [
        Component::new(d0.clone(), vec![c(1.45, 0.0)], 2.0),
        Component::with_particle(d1.clone(), p1.clone(), 0.5),
    ];
    let mixed = compute_mixture(&components, &wl, &opts).unwrap().remove(0);
    let a = compute(&d0, &wl, &[c(1.45, 0.0)], &opts).unwrap().remove(0);
    let b = one(&d1, wl[0], &p1, &opts);
    let (fa, fb) = (2.0 / 2.5, 0.5 / 2.5);
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-14 * y.abs();
    assert!(close(
        mixed.values.c_ext,
        fa * a.values.c_ext + fb * b.values.c_ext
    ));
    assert!(close(
        mixed.values.c_abs,
        fa * a.values.c_abs + fb * b.values.c_abs
    ));
    let sca = fa * a.values.c_sca + fb * b.values.c_sca;
    let g = (fa * a.values.c_sca * a.values.g + fb * b.values.c_sca * b.values.g) / sca;
    assert!((mixed.values.g - g).abs() <= 1e-13);
    // The names, prefixed by the component.
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
            "c1.layer0.m_real",
            "c1.layer0.m_imag",
            "c1.layer1.m_real",
            "c1.layer1.m_imag",
            "c1.layer0.radius_fraction",
            "c1.weight",
        ]
    );
}

#[test]
fn a_batch_matches_each_distribution() {
    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(6));
    let p = Particle::layered_sphere(
        vec![0.4],
        vec![vec![CORE, c(1.9, 0.7)], vec![SHELL, c(1.42, 0.0)]],
    );
    let wl = [0.45, 0.9];
    let ds = [lognormal(0.1, 1.5), lognormal(0.4, 1.3)];
    let batch = compute_batch_with_particle(&ds, &wl, &p, &opts).unwrap();
    for (d, rs) in ds.iter().zip(&batch) {
        let alone = compute_with_particle(d, &wl, &p, &opts).unwrap();
        for (b, a) in rs.iter().zip(&alone) {
            for (x, y, e) in [
                (b.values.c_ext, a.values.c_ext, a.errors.c_ext),
                (b.values.c_abs, a.values.c_abs, a.errors.c_abs),
                (b.values.g, a.values.g, a.errors.g),
            ] {
                assert!(
                    (x - y).abs() <= 2.0 * e + 1e-15 * y.abs(),
                    "{x} vs {y} ± {e}"
                );
            }
        }
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let d = lognormal(0.2, 1.4);
    let p = Particle::layered_sphere(
        vec![0.3, 0.8],
        vec![vec![CORE; 3], vec![c(1.6, 0.01); 3], vec![c(1.43, 1e-8); 3]],
    );
    let opts = EnsembleOptions::new()
        .with_greek(GreekOrder::All)
        .with_mu(vec![0.2])
        .with_derivatives(parameters_with_particle(&d, &p));
    let wavelengths = [0.4, 0.55, 0.8];
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| compute_with_particle(&d, &wavelengths, &p, &opts).unwrap())
    };
    let one_thread = run(1);
    assert_eq!(one_thread, run(2));
    assert_eq!(one_thread, run(8));
}

#[test]
fn derivatives_need_the_layered_derivative_domain() {
    // x_L reaches about 2·10³ at λ = 0.5: beyond the derivatives' validated 10³.
    let d = lognormal(80.0, 1.1);
    let p = coated(0.5, CORE, SHELL);
    let fraction = vec![param(ParamKind::RadiusFraction(0))];
    let opts = EnsembleOptions::new()
        .with_rtol(1e-3)
        .with_derivatives(fraction);
    let e = compute_with_particle(&d, &[0.5], &p, &opts).unwrap_err();
    assert!(
        matches!(&e, Error::OutsideValidatedDomain(s) if s.contains("derivatives")),
        "{e}"
    );
    // The values alone are validated there.
    let values = EnsembleOptions::new().with_rtol(1e-3);
    assert!(compute_with_particle(&d, &[0.5], &p, &values).is_ok());
}

#[test]
fn bad_layered_particles_are_rejected() {
    let d = lognormal(0.1, 1.5);
    let wl = [0.5];
    let opts = EnsembleOptions::new();
    let bad = |p: Particle, needle: &str| {
        let e = compute_with_particle(&d, &wl, &p, &opts).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidInput(s) if s.contains(needle)),
            "{needle}: {e}"
        );
    };
    bad(Particle::layered_sphere(Vec::new(), Vec::new()), "layers");
    bad(
        Particle::layered_sphere(vec![0.5, 0.6], vec![vec![CORE], vec![SHELL]]),
        "radius fractions",
    );
    bad(coated(0.0, CORE, SHELL), "(0, 1]");
    bad(coated(1.5, CORE, SHELL), "(0, 1]");
    bad(coated(f64::NAN, CORE, SHELL), "(0, 1]");
    bad(
        Particle::layered_sphere(vec![0.6, 0.5], vec![vec![CORE], vec![SHELL], vec![SHELL]]),
        "nondecreasing",
    );
    bad(
        Particle::layered_sphere(vec![0.5], vec![vec![CORE], vec![SHELL, SHELL]]),
        "one per wavelength",
    );
    bad(
        Particle::layered_sphere(vec![0.5], vec![vec![CORE, CORE], vec![SHELL, SHELL]]),
        "wavelengths",
    );
    // A derivative the particle does not have.
    let p = coated(0.5, CORE, SHELL);
    for kind in [
        ParamKind::RefrReal,
        ParamKind::LayerRefrReal(2),
        ParamKind::RadiusFraction(1),
    ] {
        let o = EnsembleOptions::new().with_derivatives(vec![param(kind)]);
        let e = compute_with_particle(&d, &wl, &p, &o).unwrap_err();
        assert!(
            matches!(&e, Error::InvalidInput(s) if s.contains("no parameter")),
            "{e}"
        );
    }
    // A sphere has no layers.
    let o = EnsembleOptions::new().with_derivatives(vec![param(ParamKind::LayerRefrReal(0))]);
    assert!(compute(&d, &wl, &[SHELL], &o).is_err());
}
