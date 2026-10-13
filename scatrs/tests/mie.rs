//! Properties of the Mie solutions that need no reference data: input checks, published
//! tables, invariants, the `Dual` derivative oracle, determinism and robustness.
#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together

mod common;

use common::mie::{q_of, solve};
use num_complex::Complex;
use proptest::prelude::*;
use scatrs::Error;
use scatrs::mie::{
    AmplitudeBuffers, MAX_MIE_ORDER, Mie, MieOptions, VALIDATED_DOMAIN, ValidatedDomain,
    default_n_max, n_terms,
};
use scatrs::scalar::Dual;

const EPS: f64 = f64::EPSILON;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

#[test]
fn default_truncation() {
    assert_eq!(default_n_max(1e-6), 4); // 3.06
    assert_eq!(default_n_max(0.1), 6); // 5.785
    assert_eq!(default_n_max(1.0), 10); // exactly 10
    assert_eq!(default_n_max(1000.0), 1063); // exactly 1063
    assert_eq!(default_n_max(2e4), 20166); // 20000 + 162.86 + 3
}

#[test]
fn invalid_inputs_are_rejected() {
    let mut mie = Mie::<f64>::new();
    let opts = MieOptions::new().with_allow_unvalidated(true);
    let invalid = |r: Result<_, Error>| matches!(r, Err(Error::InvalidInput(_)));
    for x in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-31, 1.1e6] {
        assert!(
            invalid(mie.compute(x, c(1.5, 0.0), &opts).map(|_| ())),
            "x = {x}"
        );
    }
    for m in [
        c(f64::NAN, 0.0),
        c(1.5, f64::INFINITY),
        c(-1.5, 0.1),
        c(1.5, -1e-9),
        c(0.0, 0.0),
        c(1e-7, 0.0),
        c(2e3, 0.0),
    ] {
        assert!(invalid(mie.compute(1.0, m, &opts).map(|_| ())), "m = {m}");
    }
    for n in [0, MAX_MIE_ORDER + 1] {
        let opts = opts.with_n_max(n);
        assert!(
            invalid(mie.compute(1.0, c(1.5, 0.0), &opts).map(|_| ())),
            "n_max = {n}"
        );
    }
    // `Dual` inputs whose values are fine but whose derivative parts are not.
    let mut dual = Mie::<D3>::new();
    let nan_seed = D3::new(1.0, [f64::NAN, 0.0, 0.0]);
    let m = Complex::new(D3::constant(1.5), D3::constant(0.0));
    let r = dual.compute(nan_seed, m, &opts).map(|_| ());
    assert!(matches!(r, Err(Error::InvalidInput(_))));
    let m = Complex::new(D3::new(1.5, [0.0, f64::INFINITY, 0.0]), D3::constant(0.0));
    let r = dual.compute(D3::constant(1.0), m, &opts).map(|_| ());
    assert!(matches!(r, Err(Error::InvalidInput(_))));
    let m = Complex::new(D3::constant(1.5), D3::constant(0.0));
    let s = dual.compute(D3::constant(1.0), m, &opts).unwrap();
    let zero = Complex::new(D3::constant(0.0), D3::constant(0.0));
    let (mut s1, mut s2) = ([zero], [zero]);
    let r = s.amplitudes(
        &[D3::new(0.5, [f64::NAN, 0.0, 0.0])],
        AmplitudeBuffers {
            s1: &mut s1,
            s2: &mut s2,
        },
    );
    assert!(matches!(r, Err(Error::InvalidInput(_))));
}

#[test]
fn the_validated_domain_is_enforced_unless_allowed() {
    let mut mie = Mie::<f64>::new();
    let outside = |r: Result<_, Error>| matches!(r, Err(Error::OutsideValidatedDomain(_)));
    let strict = MieOptions::new();
    let lenient = MieOptions::new().with_allow_unvalidated(true);
    for (x, m) in [
        (2.0001e4, c(1.5, 0.0)),
        (9e-7, c(1.5, 0.0)),
        (1.0, c(10.5, 0.0)),
        (1.0, c(1.5, 10.5)),
        (1.0, c(0.0, 2.0)),
        // Below |m| = 0.1, where Q_abs and the Jacobian of nearly real m lose accuracy.
        (1.0, c(0.05, 0.0)),
        (1.0, c(0.01, 0.01)),
    ] {
        assert!(!VALIDATED_DOMAIN.contains(x, m));
        assert!(
            outside(mie.compute(x, m, &strict).map(|_| ())),
            "x = {x}, m = {m}"
        );
        assert!(mie.compute(x, m, &lenient).is_ok(), "x = {x}, m = {m}");
    }
    // Fewer terms than the default.
    let few = MieOptions::new().with_n_max(default_n_max(10.0) - 1);
    assert!(outside(mie.compute(10.0, c(1.5, 0.0), &few).map(|_| ())));
    assert!(
        mie.compute(10.0, c(1.5, 0.0), &few.with_allow_unvalidated(true))
            .is_ok()
    );
    // The corners are inside.
    for (x, m) in [
        (1e-6, c(10.0, 10.0)),
        (2e4, c(0.1, 0.0)),
        (1.0, c(1.0, 0.0)),
        (1.0, c(0.06, 0.08)),
    ] {
        assert!(mie.compute(x, m, &strict).is_ok(), "x = {x}, m = {m}");
    }
}

#[test]
fn n_terms_checks_and_counts_as_compute_does() {
    let mut mie = Mie::<f64>::new();
    let strict = MieOptions::new();
    for (x, m, opts) in [
        (1.0, c(1.5, 0.0), strict),
        (1e3, c(1.33, 1e-8), strict),
        (10.0, c(1.5, 0.0), strict.with_n_max(40)),
        (-1.0, c(1.5, 0.0), strict),
        (1.0, c(1.5, -1e-9), strict),
        (1.0, c(f64::NAN, 0.0), strict),
        (3e4, c(1.5, 0.0), strict),
        (10.0, c(1.5, 0.0), strict.with_n_max(5)),
        (1.0, c(1.5, 0.0), strict.with_n_max(MAX_MIE_ORDER + 1)),
        (3e4, c(1.5, 0.0), strict.with_allow_unvalidated(true)),
    ] {
        let solved = mie.compute(x, m, &opts).map(|s| s.n_terms());
        assert_eq!(n_terms(x, m, &opts), solved, "x = {x}, m = {m}");
    }
}

/// Wiscombe (1979), *Mie scattering calculations: advances in technique and fast,
/// vector-speed computer codes*, NCAR/TN-140+STR, test case of x = 1000, m = 0.75: Q_sca and
/// S₁, S₂ at Θ = 0°, 30°, …, 180°, to the 7 digits printed. MIEV0 uses e^(+iωt), so its S are
/// the complex conjugates of ours.
#[test]
fn wiscombe_published_values() {
    let mut mie = Mie::<f64>::new();
    let s = mie
        .compute(1000.0, c(0.75, 0.0), &MieOptions::new())
        .unwrap();
    assert!((s.efficiencies().q_sca - 1.997908).abs() <= 1e-6);
    let table = [
        (c(4.994770e5, -1.336502e4), c(4.994770e5, -1.336502e4)),
        (c(-3.999296e2, -3.316361e2), c(-3.946018e2, -1.147791e2)),
        (c(-5.209852e2, -5.776614e2), c(-1.970767e2, -6.937470e2)),
        (c(-1.600887e2, 1.348013e2), c(-4.152365e1, 1.143000e2)),
        (c(8.431720e1, -1.209493e2), c(-4.261732e1, 5.535055e1)),
        (c(-7.556092e1, -8.134810e1), c(4.218303e1, 9.100831e1)),
        (c(1.705778e1, 4.842510e2), c(-1.705778e1, -4.842510e2)),
    ];
    let mu: Vec<f64> = (0..7)
        .map(|k| (f64::from(k) * std::f64::consts::PI / 6.0).cos())
        .collect();
    let mut s1 = vec![c(0.0, 0.0); 7];
    let mut s2 = s1.clone();
    s.amplitudes(
        &mu,
        AmplitudeBuffers {
            s1: &mut s1,
            s2: &mut s2,
        },
    )
    .unwrap();
    for (k, (t1, t2)) in table.into_iter().enumerate() {
        for (got, want) in [(s1[k], t1.conj()), (s2[k], t2.conj())] {
            assert!(
                (got - want).norm() <= 1e-6 * want.norm(),
                "angle {}: {got} vs {want}",
                30 * k
            );
        }
    }
}

#[test]
fn rayleigh_limit() {
    // Q_sca = (8/3) x⁴ |(m² − 1)/(m² + 2)|² (1 + O(x²)).
    let mut mie = Mie::<f64>::new();
    for m in [c(1.33, 0.0), c(1.5, 0.1), c(10.0, 10.0)] {
        let x = 1e-4;
        let q = mie
            .compute(x, m, &MieOptions::new())
            .unwrap()
            .efficiencies();
        let k = (m * m - 1.0) / (m * m + 2.0);
        let rayleigh = 8.0 / 3.0 * x.powi(4) * k.norm_sqr();
        assert!((q.q_sca / rayleigh - 1.0).abs() <= 1e-6, "m = {m}");
    }
}

#[test]
fn derivatives_need_compute_with_derivatives() {
    let mut mie = Mie::<f64>::new();
    let s = mie.compute(5.0, c(1.5, 0.1), &MieOptions::new()).unwrap();
    assert!(s.derivatives().is_none());
    assert!(s.efficiencies_jacobian().is_none());
    let mut v = [c(0.0, 0.0); 6];
    let [a, b, d, e, f, g] = &mut v[..] else {
        unreachable!()
    };
    let r = s.amplitudes_with_derivatives(
        &[0.5],
        AmplitudeBuffers {
            s1: std::slice::from_mut(a),
            s2: std::slice::from_mut(b),
        },
        AmplitudeBuffers {
            s1: std::slice::from_mut(d),
            s2: std::slice::from_mut(e),
        },
        AmplitudeBuffers {
            s1: std::slice::from_mut(f),
            s2: std::slice::from_mut(g),
        },
    );
    assert!(matches!(r, Err(Error::InvalidInput(_))));
}

#[test]
fn bad_cosines_and_buffers_are_rejected_without_writing() {
    let mut mie = Mie::<f64>::new();
    let s = mie.compute(5.0, c(1.5, 0.1), &MieOptions::new()).unwrap();
    let marker = c(7.0, 7.0);
    for mu in [vec![0.5, 1.5], vec![f64::NAN], vec![-1.0 - 1e-15]] {
        let (mut s1, mut s2) = (vec![marker; mu.len()], vec![marker; mu.len()]);
        let r = s.amplitudes(
            &mu,
            AmplitudeBuffers {
                s1: &mut s1,
                s2: &mut s2,
            },
        );
        assert!(matches!(r, Err(Error::InvalidInput(_))), "{mu:?}");
        assert!(s1.iter().chain(&s2).all(|&v| v == marker));
    }
    let (mut s1, mut s2) = (vec![marker; 2], vec![marker; 3]);
    let r = s.amplitudes(
        &[0.1, 0.2],
        AmplitudeBuffers {
            s1: &mut s1,
            s2: &mut s2,
        },
    );
    assert!(matches!(r, Err(Error::InvalidInput(_))));
}

/// Forward and backward symmetries hold exactly: πₙ(±1) = ±τₙ(±1) in exact integers (while
/// n(n + 1)(2n + 1)/2 < 2⁵³, the largest product the π recurrence forms, so for x up to about
/// 2·10⁵).
#[test]
fn forward_and_backward_amplitudes() {
    let mut mie = Mie::<f64>::new();
    for (x, m) in [
        (0.3, c(1.33, 0.0)),
        (25.0, c(1.5, 0.1)),
        (700.0, c(0.2, 3.0)),
    ] {
        let s = mie.compute(x, m, &MieOptions::new()).unwrap();
        let (mut s1, mut s2) = (vec![c(0.0, 0.0); 2], vec![c(0.0, 0.0); 2]);
        s.amplitudes(
            &[1.0, -1.0],
            AmplitudeBuffers {
                s1: &mut s1,
                s2: &mut s2,
            },
        )
        .unwrap();
        assert_eq!(s1[0], s2[0]);
        assert_eq!(s1[1], -s2[1]);
        // Optical theorem: Q_ext = (4/x²) Re S(0).
        let q = s.efficiencies();
        let kappa = ValidatedDomain::condition_factor(x, m);
        assert!((4.0 / (x * x) * s1[0].re / q.q_ext - 1.0).abs() <= 64.0 * kappa * EPS);
        // Q_back = 4|S₁(π)|²/x².
        assert!((4.0 * s1[1].norm_sqr() / (x * x) / q.q_back - 1.0).abs() <= 64.0 * kappa * EPS);
    }
}

#[test]
fn workspace_reuse_is_bitwise_deterministic() {
    let mut mie = Mie::<f64>::new();
    let first: Vec<_> = mie
        .compute_with_derivatives(300.0, c(1.5, 0.01), &MieOptions::new())
        .unwrap()
        .a()
        .to_vec();
    mie.compute(2.0, c(4.0, 0.1), &MieOptions::new()).unwrap();
    mie.compute(5000.0, c(1.33, 0.0), &MieOptions::new())
        .unwrap();
    let again = mie
        .compute_with_derivatives(300.0, c(1.5, 0.01), &MieOptions::new())
        .unwrap()
        .a()
        .to_vec();
    assert_eq!(first, again);
    let fresh = Mie::<f64>::new()
        .compute(300.0, c(1.5, 0.01), &MieOptions::new())
        .unwrap()
        .a()
        .to_vec();
    assert_eq!(first, fresh);
}

type D3 = Dual<f64, 3>;

/// The closed-form derivatives against `Dual<f64, 3>` (seeds x, Re m, Im m) propagated
/// through the same recurrences: the values agree bitwise, and the derivatives within
/// 64·κ′·2⁻⁵² of their array's (or the efficiency's natural) scale, with κ′ the a-priori
/// factor of the derivative tolerances (`ValidatedDomain`): both carry rounding errors of
/// that size, from different formulas. Relative to the array's largest entry, errors in its
/// smallest entries (orders far above x) go unseen, as in the tolerance model.
#[test]
fn closed_form_derivatives_agree_with_dual() {
    let mu = [1.0, 0.6, 0.0, -0.3, -1.0];
    for (x, m) in [
        (1e-6, c(1.33, 1e-8)),
        (0.5, c(1.5, 0.01)),
        (3.0, c(1.0001, 0.0)),
        (10.0, c(2.5, 1.0)),
        (40.0, c(1.75, 0.44)),
        (150.0, c(1.33, 0.0)),
        (25.0, c(0.2, 3.0)),
        (300.0, c(4.0, 0.01)),
        (1000.0, c(1.5, 5.0)),
        (3000.0, c(10.0, 10.0)),
    ] {
        let f = solve(x, m, &mu, None);
        let near_one = (m.norm() / (m - 1.0).norm()).max(1.0);
        let bound = 64.0 * ValidatedDomain::condition_factor(x, m) * near_one * EPS;
        let xd = D3::new(x, [1.0, 0.0, 0.0]);
        let md = Complex::new(
            D3::new(m.re, [0.0, 1.0, 0.0]),
            D3::new(m.im, [0.0, 0.0, 1.0]),
        );
        let mu_d: Vec<D3> = mu.iter().map(|&v| D3::constant(v)).collect();
        let d = solve(xd, md, &mu_d, None);
        let value = |z: Complex<D3>| c(z.re.re(), z.im.re());
        let deriv = |z: Complex<D3>, k: usize| c(z.re.eps()[k], z.im.eps()[k]);
        let i1 = c(0.0, 1.0);
        // Coefficients: ∂/∂x, ∂/∂Re m = d/dm, ∂/∂Im m = i d/dm.
        for (vals, dx, dm) in [(&d.a, &f.d[0], &f.d[1]), (&d.b, &f.d[2], &f.d[3])] {
            for (k, analytic) in [
                (0, dx.clone()),
                (1, dm.clone()),
                (2, dm.iter().map(|&v| i1 * v).collect()),
            ] {
                let scale = analytic
                    .iter()
                    .map(|v: &Complex<f64>| v.norm())
                    .fold(0.0, f64::max);
                for (n, z) in vals.iter().enumerate() {
                    let err = (deriv(*z, k) - analytic[n]).norm();
                    assert!(
                        err <= bound * scale,
                        "x = {x}, m = {m}, n = {}, seed {k}: {err:e}",
                        n + 1
                    );
                }
            }
        }
        for (n, (za, zb)) in d.a.iter().zip(&d.b).enumerate() {
            assert_eq!(
                value(*za),
                f.a[n],
                "a differs: x = {x}, m = {m}, n = {}",
                n + 1
            );
            assert_eq!(
                value(*zb),
                f.b[n],
                "b differs: x = {x}, m = {m}, n = {}",
                n + 1
            );
        }
        for i in 0..6 {
            for j in 0..mu.len() {
                assert_eq!(value(d.s[i][j]), f.s[i][j], "S differs: x = {x}, m = {m}");
            }
        }
        for p in 0..3 {
            for k in 0..5 {
                let (vd, vf) = (q_of(&d.jac[p], k).re(), q_of(&f.jac[p], k));
                assert_eq!(vd, vf, "Jacobian differs: x = {x}, m = {m}, Q{k} p{p}");
            }
        }
        // Efficiencies: the Jacobian against the derivative parts.
        let sizes = [x, m.norm(), m.norm()];
        for k in 0..5 {
            let q = q_of(&d.q, k);
            assert_eq!(q.re(), q_of(&f.q, k));
            for p in 0..3 {
                let analytic = q_of(&f.jac[p], k);
                let natural =
                    q_of(&f.q, 0).abs().max(q.re().abs()) * (1.0 + m.norm() * x) / sizes[p];
                let err = (q.eps()[p] - analytic).abs();
                assert!(
                    err <= bound * analytic.abs().max(natural),
                    "x = {x}, m = {m}, Q{k} p{p}: {err:e}"
                );
            }
        }
        // Amplitudes: ∂S/∂x and dS/dm.
        for j in 0..mu.len() {
            for i in 0..2 {
                let z = d.s[i][j];
                for (k, analytic) in [(0, f.s[2 + i][j]), (1, f.s[4 + i][j])] {
                    let scale = analytic.norm().max(f.s[i][j].norm());
                    let err = (deriv(z, k) - analytic).norm();
                    assert!(
                        err <= bound * scale,
                        "x = {x}, m = {m}, S{} mu {}: {err:e}",
                        i + 1,
                        mu[j]
                    );
                }
            }
        }
    }
}

/// The corners of the accepted inputs that the property test below skips for their cost: every
/// output finite.
#[test]
fn accepted_corners_are_finite() {
    let lenient = MieOptions::new().with_allow_unvalidated(true);
    let mut mie = Mie::<f64>::new();
    for (x, m, opts) in [
        (1e6, c(1.5, 0.0), lenient),
        (1e6, c(1e-6, 0.0), lenient),
        (1e-30, c(999.0, 1.0), lenient.with_n_max(MAX_MIE_ORDER)),
        (1e-30, c(1e-6, 0.0), lenient.with_n_max(MAX_MIE_ORDER)),
    ] {
        let s = mie.compute_with_derivatives(x, m, &opts).unwrap();
        let j = s.efficiencies_jacobian().unwrap();
        for e in [s.efficiencies(), j.x, j.m_re, j.m_im] {
            for k in 0..5 {
                assert!(q_of(&e, k).is_finite(), "x = {x:e}, m = {m}: Q{k}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Inside the domain: Q_abs ≥ 0 (exactly 0 for real m), Q_sca ≤ Q_ext, |g| ≤ 1, and
    /// |aₙ|² ≤ Re aₙ up to rounding (passive media).
    #[test]
    fn physical_bounds(log_x in -6.0f64..4.3, re in 0.1f64..10.0, log_im in -10.0f64..1.0, real in proptest::bool::ANY) {
        let x = 10f64.powf(log_x).min(2e4);
        let m = c(re, if real { 0.0 } else { 10f64.powf(log_im) });
        let mut mie = Mie::<f64>::new();
        let s = mie.compute(x, m, &MieOptions::new()).unwrap();
        let q = s.efficiencies();
        if real {
            prop_assert_eq!(q.q_abs, 0.0);
        } else {
            prop_assert!(q.q_abs >= -1e-14 * q.q_ext);
        }
        prop_assert!(q.q_sca <= q.q_ext * (1.0 + 1e-14) && q.q_sca > 0.0);
        prop_assert!(q.g.abs() <= 1.0 + 1e-12);
        let scale = s.a().iter().map(|a| a.norm()).fold(0.0, f64::max);
        let kappa = ValidatedDomain::condition_factor(x, m);
        for a in s.a().iter().chain(s.b()) {
            prop_assert!(a.norm_sqr() <= a.re + 64.0 * kappa * EPS * scale);
        }
    }

    /// Over the accepted inputs, outside the domain too, up to x = 10⁵ and |m|x = 10⁶ to bound the
    /// cost (`accepted_corners_are_finite` covers the rest): every output finite. A third of the
    /// cases have real m (the real recurrence for Dₙ) and a third purely imaginary m.
    #[test]
    fn no_nan_or_infinity_over_the_supported_inputs(
        log_x in -30.0f64..5.0,
        log_abs_m in -6.0f64..3.0,
        angle in 0.0f64..std::f64::consts::FRAC_PI_2,
        kind in 0u8..3,
    ) {
        let x = 10f64.powf(log_x);
        let r = 10f64.powf(log_abs_m);
        let m = match kind {
            0 => c(r, 0.0),
            1 => c(0.0, r),
            _ => c(r * angle.cos(), r * angle.sin()),
        };
        prop_assume!(m.norm() * x <= 1e6); // keep the cost of Dₙ(mx) bounded
        let opts = MieOptions::new().with_allow_unvalidated(true);
        let mut mie = Mie::<f64>::new();
        let s = mie.compute_with_derivatives(x, m, &opts).unwrap();
        let q = s.efficiencies();
        let j = s.efficiencies_jacobian().unwrap();
        for e in [q, j.x, j.m_re, j.m_im] {
            for k in 0..5 {
                prop_assert!(q_of(&e, k).is_finite(), "x = {x:e}, m = {m}: Q{k} = {}", q_of(&e, k));
            }
        }
        let mu = [1.0, 0.3, -1.0];
        let zero = c(0.0, 0.0);
        let mut v: [Vec<Complex<f64>>; 6] = std::array::from_fn(|_| vec![zero; 3]);
        let [s1, s2, dx1, dx2, dm1, dm2] = &mut v;
        s.amplitudes_with_derivatives(
            &mu,
            AmplitudeBuffers { s1, s2 },
            AmplitudeBuffers { s1: dx1, s2: dx2 },
            AmplitudeBuffers { s1: dm1, s2: dm2 },
        ).unwrap();
        for z in v.iter().flatten() {
            prop_assert!(z.re.is_finite() && z.im.is_finite(), "x = {x:e}, m = {m}");
        }
    }
}
