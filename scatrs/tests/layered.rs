//! Properties of layered spheres that need no reference data: input checks, the identities
//! with homogeneous spheres, the Rayleigh limit, invariants, and the `Dual` derivatives against
//! central differences in double-double.

use num_complex::Complex;
use scatrs::Error;
use scatrs::layered::{Layered, MAX_LAYERS, VALIDATED_DOMAIN};
use scatrs::mie::{
    AmplitudeBuffers, AngularTable, Efficiencies, Mie, MieOptions, Normalization, PhaseBuffers,
};
use scatrs::scalar::{DoubleDouble, Real};

const EPS: f64 = f64::EPSILON;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn opts() -> MieOptions {
    MieOptions::new()
}

/// Every output of a solution as bits: coefficients, efficiencies, amplitudes, phase matrix
/// (both normalizations) and Greek coefficients.
fn bits_mie(s: &scatrs::mie::MieSolution<'_, f64>, mu: &[f64]) -> Vec<u64> {
    let mut out = Vec::new();
    for z in s.a().iter().chain(s.b()) {
        out.extend([z.re.to_bits(), z.im.to_bits()]);
    }
    push_eff(&mut out, &s.efficiencies());
    let z = Complex::default();
    let (mut s1, mut s2) = (vec![z; mu.len()], vec![z; mu.len()]);
    s.amplitudes(
        mu,
        AmplitudeBuffers {
            s1: &mut s1,
            s2: &mut s2,
        },
    )
    .unwrap();
    for z in s1.iter().chain(&s2) {
        out.extend([z.re.to_bits(), z.im.to_bits()]);
    }
    for norm in [Normalization::Normalized, Normalization::Unnormalized] {
        let mut f: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; mu.len()]);
        let [f11, f12, f33, f34] = &mut f;
        s.phase_matrix(mu, norm, PhaseBuffers { f11, f12, f33, f34 })
            .unwrap();
        out.extend(f.iter().flatten().map(|v| v.to_bits()));
        let g = s.greek(None, norm).unwrap();
        for a in [
            &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
        ] {
            out.extend(a.iter().map(|v| v.to_bits()));
        }
    }
    out
}

fn bits_layered(s: &scatrs::layered::LayeredSolution<'_, f64>, mu: &[f64]) -> Vec<u64> {
    let mut out = Vec::new();
    for z in s.a().iter().chain(s.b()) {
        out.extend([z.re.to_bits(), z.im.to_bits()]);
    }
    push_eff(&mut out, &s.efficiencies());
    let z = Complex::default();
    let (mut s1, mut s2) = (vec![z; mu.len()], vec![z; mu.len()]);
    s.amplitudes(
        mu,
        AmplitudeBuffers {
            s1: &mut s1,
            s2: &mut s2,
        },
    )
    .unwrap();
    for z in s1.iter().chain(&s2) {
        out.extend([z.re.to_bits(), z.im.to_bits()]);
    }
    for norm in [Normalization::Normalized, Normalization::Unnormalized] {
        let mut f: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; mu.len()]);
        let [f11, f12, f33, f34] = &mut f;
        s.phase_matrix(mu, norm, PhaseBuffers { f11, f12, f33, f34 })
            .unwrap();
        out.extend(f.iter().flatten().map(|v| v.to_bits()));
        let g = s.greek(None, norm).unwrap();
        for a in [
            &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
        ] {
            out.extend(a.iter().map(|v| v.to_bits()));
        }
    }
    out
}

fn push_eff(out: &mut Vec<u64>, q: &Efficiencies<f64>) {
    out.extend([q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g].map(f64::to_bits));
}

const MU: [f64; 5] = [-1.0, -0.4, 0.0, 0.7, 1.0];

const SPHERES: [(f64, Complex<f64>); 6] = [
    (1e-6, Complex::new(1.5, 0.01)),
    (0.3, Complex::new(1.33, 0.0)),
    (3.0, Complex::new(1.95, 0.79)),
    (25.0, Complex::new(1.43, 1e-8)),
    (140.0, Complex::new(2.5, 1.0)),
    (60.0, Complex::new(0.2, 3.0)),
];

#[test]
fn one_layer_is_a_sphere_bitwise() {
    let (mut mie, mut lay) = (Mie::<f64>::new(), Layered::<f64>::new());
    for (x, m) in SPHERES {
        let want = bits_mie(&mie.compute(x, m, &opts()).unwrap(), &MU);
        let got = bits_layered(&lay.compute(&[x], &[m], &opts()).unwrap(), &MU);
        assert_eq!(got, want, "x = {x}, m = {m}");
        // With derivatives: the values are the same, and so is every derivative.
        let s = mie.compute_with_derivatives(x, m, &opts()).unwrap();
        let (d, jac) = (s.derivatives().unwrap(), s.efficiencies_jacobian().unwrap());
        let (want_d, want_j) = ([d.da_dx, d.da_dm, d.db_dx, d.db_dm].map(<[_]>::to_vec), jac);
        let l = lay.compute_with_derivatives(&[x], &[m], &opts()).unwrap();
        let ld = l.derivatives(0).unwrap();
        assert_eq!(
            [ld.da_dx, ld.da_dm, ld.db_dx, ld.db_dm].map(<[_]>::to_vec),
            want_d
        );
        assert_eq!(l.efficiencies_jacobian().unwrap(), vec![want_j]);
    }
}

#[test]
fn identical_layers_are_the_homogeneous_sphere_bitwise() {
    let (mut mie, mut lay) = (Mie::<f64>::new(), Layered::<f64>::new());
    for (x, m) in SPHERES {
        let want = bits_mie(&mie.compute(x, m, &opts()).unwrap(), &MU);
        for xs in [vec![0.3 * x, x], vec![0.1 * x, 0.5 * x, 0.9 * x, x]] {
            let ms = vec![m; xs.len()];
            let got = bits_layered(&lay.compute(&xs, &ms, &opts()).unwrap(), &MU);
            assert_eq!(got, want, "x = {xs:?}, m = {m}");
        }
    }
}

/// Max over n of |got − want| / max over n of |want|, for aₙ and bₙ.
fn coefficient_error(
    got: &scatrs::layered::LayeredSolution<'_, f64>,
    a: &[Complex<f64>],
    b: &[Complex<f64>],
) -> f64 {
    let scale = a.iter().chain(b).map(|z| z.norm()).fold(0.0, f64::max);
    let err = got
        .a()
        .iter()
        .zip(a)
        .chain(got.b().iter().zip(b))
        .map(|(g, w)| (g - w).norm())
        .fold(0.0, f64::max);
    err / scale
}

#[test]
fn a_shell_of_zero_thickness_changes_nothing() {
    let mut lay = Layered::<f64>::new();
    let mut mie = Mie::<f64>::new();
    for (x, m) in SPHERES {
        let s = mie.compute(x, m, &opts()).unwrap();
        let (a, b) = (s.a().to_vec(), s.b().to_vec());
        for shell in [c(1.5, 0.0), c(2.0, 0.5), c(0.2, 3.0)] {
            let got = lay.compute(&[x, x], &[m, shell], &opts()).unwrap();
            let err = coefficient_error(&got, &a, &b);
            assert!(
                err <= 1e3 * (1.0 + 3.0 * x) * EPS,
                "x = {x}, m = {m}, shell {shell}: {err:e}"
            );
        }
    }
}

#[test]
fn an_air_shell_gives_the_bare_core() {
    let mut lay = Layered::<f64>::new();
    let mut mie = Mie::<f64>::new();
    for (x, m) in SPHERES {
        let s = mie.compute(x, m, &opts()).unwrap();
        let n = s.n_terms();
        let (a, b) = (s.a().to_vec(), s.b().to_vec());
        // The coated sphere has more terms; the core's vanish beyond its own.
        let opts = opts().with_n_max(n).with_allow_unvalidated(true);
        let got = lay
            .compute(&[x, 1.7 * x], &[m, c(1.0, 0.0)], &opts)
            .unwrap();
        let err = coefficient_error(&got, &a, &b);
        assert!(
            err <= 1e3 * (1.0 + 3.0 * x) * EPS,
            "x = {x}, m = {m}: {err:e}"
        );
    }
}

#[test]
fn the_rayleigh_limit_is_the_coated_polarizability() {
    let mut lay = Layered::<f64>::new();
    for (m1, m2) in [
        (c(1.95, 0.79), c(1.43, 0.01)),
        (c(1.33, 0.0), c(10.0, 10.0)),
    ] {
        for f in [0.1, 0.5, 0.9] {
            let x2 = 1e-4;
            let x1 = f * x2;
            let s = lay.compute(&[x1, x2], &[m1, m2], &opts()).unwrap();
            let (e1, e2, v) = (m1 * m1, m2 * m2, f * f * f);
            let ratio = ((e2 - 1.0) * (e1 + e2 * 2.0) + (e2 * 2.0 + 1.0) * (e1 - e2) * v)
                / ((e2 + 2.0) * (e1 + e2 * 2.0) + (e2 - 1.0) * (e1 - e2) * (2.0 * v));
            let want = c(0.0, -2.0 * x2 * x2 * x2 / 3.0) * ratio;
            let err = (s.a()[0] - want).norm() / want.norm();
            assert!(err <= 10.0 * x2 * x2, "m = {m1}, {m2}, f = {f}: {err:e}");
        }
    }
}

#[test]
fn lossless_particles_absorb_nothing_and_absorbing_ones_satisfy_the_optical_theorem() {
    let mut lay = Layered::<f64>::new();
    let q = lay
        .compute(
            &[2.0, 5.0, 8.0],
            &[c(1.5, 0.0), c(2.5, 0.0), c(1.33, 0.0)],
            &opts(),
        )
        .unwrap()
        .efficiencies();
    assert_eq!(q.q_abs, 0.0);
    assert_eq!(q.q_ext, q.q_sca);
    for (xs, ms) in [
        (vec![0.7, 1.3], vec![c(1.95, 0.79), c(1.43, 1e-8)]),
        (
            vec![3.0, 5.0, 8.0],
            vec![c(1.5, 0.1), c(2.5, 1.0), c(1.33, 0.0)],
        ),
        (vec![10.0, 12.0], vec![c(1.45, 0.0), c(0.2, 3.0)]),
    ] {
        let s = lay.compute(&xs, &ms, &opts()).unwrap();
        let q = s.efficiencies();
        let x = xs[xs.len() - 1];
        let theorem: f64 = s
            .a()
            .iter()
            .zip(s.b())
            .enumerate()
            .map(|(k, (a, b))| (2 * k + 3) as f64 * (a.re + b.re))
            .sum::<f64>()
            * 2.0
            / (x * x);
        assert!(q.q_abs > 0.0);
        assert!(
            (q.q_ext - theorem).abs() <= 1e3 * EPS * q.q_ext,
            "{xs:?}: {} vs {theorem}",
            q.q_ext
        );
    }
}

/// f64 against double-double on a set of hard cases: coefficients within a loose bound here;
/// the accuracy map and the Arb references measure them properly.
#[test]
fn double_precision_agrees_with_double_double() {
    let cases: Vec<(Vec<f64>, Vec<Complex<f64>>)> = vec![
        (vec![0.7, 1.3], vec![c(1.95, 0.79), c(1.43, 1e-8)]),
        (
            vec![3.0, 5.0, 8.0],
            vec![c(1.5, 0.1), c(2.5, 1.0), c(1.33, 0.0)],
        ),
        (vec![0.01, 0.02], vec![c(10.0, 10.0), c(1.33, 0.0)]),
        (vec![1e-6, 2e-6], vec![c(1.5, 0.01), c(1.33, 0.0)]),
        (vec![40.0, 40.02], vec![c(1.5, 0.001), c(0.2, 3.0)]),
        (vec![30.0, 50.0], vec![c(1.33, 0.0), c(1.5, 0.0)]),
        (vec![20.0, 60.0], vec![c(1.95, 0.79), c(1.43, 0.0)]),
        (vec![200.0, 1000.0], vec![c(1.5, 0.01), c(1.33, 0.0)]),
        (
            (1..=10).map(|l| f64::from(l) * 5.0).collect(),
            (1..=10)
                .map(|l| c(1.6 - 0.02 * f64::from(l), 1e-4))
                .collect(),
        ),
    ];
    let mut f = Layered::<f64>::new();
    let mut dd = Layered::<DoubleDouble>::new();
    for (xs, ms) in cases {
        let n = f.compute(&xs, &ms, &opts()).unwrap().n_terms();
        let xd: Vec<DoubleDouble> = xs.iter().map(|&v| DoubleDouble::from_f64(v)).collect();
        let md: Vec<Complex<DoubleDouble>> = ms
            .iter()
            .map(|v| Complex::new(DoubleDouble::from_f64(v.re), DoubleDouble::from_f64(v.im)))
            .collect();
        let r = dd
            .compute(&xd, &md, &opts().with_n_max(n).with_allow_unvalidated(true))
            .unwrap();
        let to = |v: &[Complex<DoubleDouble>]| -> Vec<Complex<f64>> {
            v.iter()
                .map(|z| c(z.re.approx_f64(), z.im.approx_f64()))
                .collect()
        };
        let (a, b) = (to(r.a()), to(r.b()));
        let s = f.compute(&xs, &ms, &opts()).unwrap();
        let err = coefficient_error(&s, &a, &b);
        let x = xs[xs.len() - 1];
        eprintln!(
            "{} layers, x_L = {x}: coefficient error {:.1} ulps",
            xs.len(),
            err / EPS
        );
        assert!(err <= 1e3 * (1.0 + 3.0 * x) * EPS, "{xs:?}: {err:e}");
        let (qf, qd) = (s.efficiencies(), r.efficiencies());
        for (g, w) in [
            (qf.q_ext, qd.q_ext),
            (qf.q_sca, qd.q_sca),
            (qf.q_abs, qd.q_abs),
        ] {
            let w = w.approx_f64();
            assert!(
                (g - w).abs() <= 1e3 * (1.0 + 3.0 * x) * EPS * w.abs(),
                "{xs:?}: {g} vs {w}"
            );
        }
    }
}

/// The dual-number derivatives against central differences of double-double solutions, for
/// every layer and parameter, in one pass (L ≤ 4) and per layer (L > 4).
#[test]
fn derivatives_match_central_differences() {
    let cases: Vec<(Vec<f64>, Vec<Complex<f64>>)> = vec![
        (vec![0.7, 1.3], vec![c(1.95, 0.79), c(1.43, 1e-3)]),
        (
            vec![3.0, 5.0, 8.0],
            vec![c(1.5, 0.1), c(2.5, 1.0), c(1.33, 0.0)],
        ),
        (
            vec![1.0, 2.0, 3.0, 4.0],
            vec![c(1.4, 0.0), c(1.6, 0.01), c(1.8, 0.0), c(1.5, 0.1)],
        ),
        (
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![
                c(1.4, 0.0),
                c(1.6, 0.01),
                c(1.8, 0.0),
                c(1.5, 0.1),
                c(1.33, 0.0),
            ],
        ),
    ];
    let mut f = Layered::<f64>::new();
    let mut dd = Layered::<DoubleDouble>::new();
    let h = 1e-12;
    for (xs, ms) in cases {
        let s = f.compute_with_derivatives(&xs, &ms, &opts()).unwrap();
        let n = s.n_terms();
        let derivs: Vec<_> = (0..xs.len()).map(|l| s.derivatives(l).unwrap()).collect();
        let jac = s.efficiencies_jacobian().unwrap();
        let o = opts().with_n_max(n).with_allow_unvalidated(true);
        let mut solve = |l: usize, p: usize, step: f64| {
            let mut xd: Vec<DoubleDouble> = xs.iter().map(|&v| DoubleDouble::from_f64(v)).collect();
            let mut md: Vec<Complex<DoubleDouble>> = ms
                .iter()
                .map(|v| Complex::new(DoubleDouble::from_f64(v.re), DoubleDouble::from_f64(v.im)))
                .collect();
            let hd = DoubleDouble::from_f64(step);
            match p {
                0 => xd[l] += hd,
                1 => md[l].re += hd,
                _ => md[l].im += hd,
            }
            let r = dd.compute(&xd, &md, &o).unwrap();
            (r.a().to_vec(), r.b().to_vec(), r.efficiencies())
        };
        for l in 0..xs.len() {
            for p in 0..3 {
                // Im m = 0 cannot step down (gain media are rejected): a one-sided
                // second-order difference there.
                let one_sided = p == 2 && ms[l].im == 0.0;
                let pts: Vec<(f64, f64)> = if one_sided {
                    vec![(0.0, -1.5), (h, 2.0), (2.0 * h, -0.5)]
                } else {
                    vec![(h, 0.5), (-h, -0.5)]
                };
                let sols: Vec<_> = pts.iter().map(|&(step, _)| solve(l, p, step)).collect();
                let combine = |f: &dyn Fn(usize) -> DoubleDouble| -> f64 {
                    pts.iter()
                        .enumerate()
                        .map(|(i, &(_, w))| f(i) * DoubleDouble::from_f64(w))
                        .fold(DoubleDouble::from_f64(0.0), |a, b| a + b)
                        .approx_f64()
                        / h
                };
                let d = &derivs[l];
                let want_a: Vec<Complex<f64>> = (0..sols[0].0.len())
                    .map(|k| c(combine(&|i| sols[i].0[k].re), combine(&|i| sols[i].0[k].im)))
                    .collect();
                let got_a: Vec<Complex<f64>> = match p {
                    0 => d.da_dx.to_vec(),
                    1 => d.da_dm.to_vec(),
                    _ => d.da_dm.iter().map(|z| z * c(0.0, 1.0)).collect(),
                };
                let scale = want_a.iter().map(|z| z.norm()).fold(1e-300, f64::max);
                let err = got_a
                    .iter()
                    .zip(&want_a)
                    .map(|(g, w)| (g - w).norm())
                    .fold(0.0, f64::max)
                    / scale;
                assert!(
                    err <= 1e-8,
                    "{xs:?} layer {l} parameter {p}: da error {err:e}"
                );
                let dq = combine(&|i| sols[i].2.q_abs);
                let got = [jac[l].x.q_abs, jac[l].m_re.q_abs, jac[l].m_im.q_abs][p];
                let qe = sols[0].2.q_ext.approx_f64();
                assert!(
                    (got - dq).abs() <= 1e-7 * (dq.abs() + qe),
                    "{xs:?} layer {l} parameter {p}: dQ_abs {got} vs {dq}"
                );
            }
        }
    }
}

#[test]
fn derivatives_of_the_outputs_follow_the_layer() {
    let mut lay = Layered::<f64>::new();
    let (xs, ms) = ([1.0, 2.5], [c(1.95, 0.79), c(1.43, 0.0)]);
    let s = lay.compute_with_derivatives(&xs, &ms, &opts()).unwrap();
    let z = Complex::default();
    let table = AngularTable::new(&MU, s.n_terms()).unwrap();
    for layer in 0..2 {
        let mut b: [Vec<Complex<f64>>; 6] = core::array::from_fn(|_| vec![z; MU.len()]);
        let [s1, s2, x1, x2, m1, m2] = &mut b;
        s.amplitudes_with_derivatives(
            &MU,
            layer,
            AmplitudeBuffers { s1, s2 },
            AmplitudeBuffers { s1: x1, s2: x2 },
            AmplitudeBuffers { s1: m1, s2: m2 },
        )
        .unwrap();
        let mut t: [Vec<Complex<f64>>; 6] = core::array::from_fn(|_| vec![z; MU.len()]);
        let [s1, s2, x1, x2, m1, m2] = &mut t;
        s.amplitudes_with_derivatives_from_table(
            &table,
            layer,
            AmplitudeBuffers { s1, s2 },
            AmplitudeBuffers { s1: x1, s2: x2 },
            AmplitudeBuffers { s1: m1, s2: m2 },
        )
        .unwrap();
        assert_eq!(b, t);
        let (_, d) = s
            .greek_with_derivatives(Some(8), layer, Normalization::Normalized)
            .unwrap();
        assert!(d[1].alpha1.iter().any(|v| *v != 0.0));
    }
    assert!(matches!(
        s.greek_with_derivatives(Some(8), 2, Normalization::Normalized),
        Err(Error::InvalidInput(_))
    ));
    let s = lay.compute(&xs, &ms, &opts()).unwrap();
    assert!(s.efficiencies_jacobian().is_none() && s.derivatives(0).is_none());
    assert!(matches!(
        s.greek_with_derivatives(Some(8), 0, Normalization::Normalized),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn invalid_inputs_are_rejected() {
    let mut lay = Layered::<f64>::new();
    let o = opts().with_allow_unvalidated(true);
    let invalid = |r: Result<(), Error>| matches!(r, Err(Error::InvalidInput(_)));
    let m2 = [c(1.5, 0.0), c(1.33, 0.0)];
    for xs in [
        vec![],
        vec![1.0],
        vec![2.0, 1.0],
        vec![f64::NAN, 1.0],
        vec![1e-31, 1.0],
        vec![1.0, 1.1e6],
    ] {
        let ms = &m2[..xs.len().min(2).max(if xs.len() == 1 { 2 } else { 0 })];
        assert!(invalid(lay.compute(&xs, ms, &o).map(|_| ())), "x = {xs:?}");
    }
    for m in [
        c(f64::NAN, 0.0),
        c(-1.5, 0.1),
        c(1.5, -1e-9),
        c(0.0, 0.0),
        c(2e3, 0.0),
    ] {
        assert!(
            invalid(lay.compute(&[1.0, 2.0], &[m, c(1.33, 0.0)], &o).map(|_| ())),
            "m = {m}"
        );
    }
    let many = MAX_LAYERS + 1;
    let xs: Vec<f64> = (1..=many).map(|l| l as f64).collect();
    let ms = vec![c(1.5, 0.0); many];
    assert!(invalid(lay.compute(&xs, &ms, &o).map(|_| ())));
    // Outside the validated domain unless allowed.
    let outside = lay.compute(&[1.0, 2e4], &m2, &opts()).map(|_| ());
    assert!(matches!(outside, Err(Error::OutsideValidatedDomain(_))));
    assert!(lay.compute(&[1.0, 2e4], &m2, &o).is_ok());
    assert!(!VALIDATED_DOMAIN.contains(&[1.0; 17], &[c(1.5, 0.0); 17]));
}
