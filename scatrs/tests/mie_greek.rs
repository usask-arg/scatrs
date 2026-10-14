//! The phase matrix and the Greek coefficients of spheres: identities, exactness of the
//! projection, the `Dual` derivative oracle, and the rejection of bad inputs.
#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together

mod common;

use common::greek::{GreekOutputs, solve_greek};
use num_complex::Complex;
use scatrs::Error;
use scatrs::mie::{
    GreekCoefficients, GreekProjector, Mie, MieOptions, Normalization, PhaseBuffers, PhaseTables,
    ValidatedDomain,
};
use scatrs::scalar::Dual;
use scatrs::special::{gauss_legendre, wigner_d_vec};

const EPS: f64 = f64::EPSILON;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Spheres from the Rayleigh regime to large, weakly and strongly absorbing, below and above
/// m = 1.
const SPHERES: [(f64, Complex<f64>); 7] = [
    (0.05, Complex::new(1.5, 0.0)),
    (1.0, Complex::new(1.33, 1e-8)),
    (3.0, Complex::new(1.0001, 0.0)),
    (5.0, Complex::new(1.5, 0.01)),
    (12.0, Complex::new(0.75, 0.0)),
    (40.0, Complex::new(1.75, 0.44)),
    (150.0, Complex::new(2.5, 1.0)),
];

fn bufs(t: &mut [Vec<f64>; 4]) -> PhaseBuffers<'_, f64> {
    let [f11, f12, f33, f34] = t;
    PhaseBuffers { f11, f12, f33, f34 }
}

fn solve(x: f64, m: Complex<f64>, mu: &[f64], normalization: Normalization) -> GreekOutputs<f64> {
    solve_greek(x, m, mu, None, None, normalization)
}

/// The sum Σ cₗ d^l_{mn}(μ) over the coefficients `coef`.
fn expand(coef: &[f64], m: i32, n: i32, mu: f64) -> f64 {
    let d = wigner_d_vec(m, n, coef.len() - 1, mu).unwrap();
    coef.iter().zip(&d).map(|(c, d)| c * d).sum()
}

#[test]
fn a_sphere_is_a_pure_scatterer() {
    // For a single sphere F₁₁² = F₁₂² + F₃₃² + F₃₄² (Bohren & Huffman Eq. 4.79), and the
    // normalized F₁₁ averages to 1 over μ.
    let mu: Vec<f64> = (0..=40).map(|k| -1.0 + f64::from(k) / 20.0).collect();
    for (x, m) in SPHERES {
        let f = solve(x, m, &mu, Normalization::Normalized);
        for j in 0..mu.len() {
            let [f11, f12, f33, f34] = [0, 1, 2, 3].map(|k| f.phase[k][j]);
            let rest = (f12 * f12 + f33 * f33 + f34 * f34).sqrt();
            assert!(
                (f11 - rest).abs() <= 1e-12 * f11,
                "x = {x}, m = {m}, mu = {}: F11 = {f11:e}, |(F12, F33, F34)| = {rest:e}",
                mu[j]
            );
        }
        let n = 2 * x as usize + 40;
        let rule = gauss_legendre::<f64>(n).unwrap();
        let g = solve(x, m, &rule.nodes, Normalization::Normalized);
        let mean: f64 = rule
            .weights
            .iter()
            .zip(&g.phase[0])
            .map(|(w, f)| w * f)
            .sum::<f64>()
            / 2.0;
        assert!(
            (mean - 1.0).abs() <= 1e-12,
            "x = {x}, m = {m}: mean F11 = {mean}"
        );
    }
}

#[test]
fn unnormalized_is_normalized_times_q_sca() {
    let mu = [1.0, 0.5, 0.0, -1.0];
    for (x, m) in SPHERES {
        let q = Mie::<f64>::new()
            .compute(x, m, &MieOptions::new())
            .unwrap()
            .efficiencies()
            .q_sca;
        let n = solve(x, m, &mu, Normalization::Normalized);
        let u = solve(x, m, &mu, Normalization::Unnormalized);
        let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 8.0 * EPS * scale;
        for k in 0..4 {
            for j in 0..mu.len() {
                let s = n.phase[0][j] * q;
                assert!(
                    close(u.phase[k][j], n.phase[k][j] * q, s),
                    "x = {x}: F{k} at {}",
                    mu[j]
                );
            }
        }
        for k in 0..6 {
            let s = n.greek[0].iter().fold(0.0f64, |a, v| a.max(v.abs())) * q;
            for l in 0..n.greek[k].len() {
                assert!(
                    close(u.greek[k][l], n.greek[k][l] * q, s),
                    "x = {x}: Greek {k} l = {l}"
                );
            }
        }
        assert_eq!(n.greek[0][0], 1.0, "x = {x}: alpha1 at l = 0");
    }
}

#[test]
fn greek_coefficients_expand_the_phase_matrix() {
    // α₁ at l = 1 is 3g; the expansions (MTL 2002 Eqs. 4.75–4.80) reproduce the phase matrix.
    let mu = [1.0, 0.95, 0.3, -0.4, -1.0];
    for (x, m) in SPHERES {
        let f = solve(x, m, &mu, Normalization::Normalized);
        let g = Mie::<f64>::new()
            .compute(x, m, &MieOptions::new())
            .unwrap()
            .efficiencies()
            .g;
        let [a1, a2, a3, a4, b1, b2] = &f.greek;
        assert!(
            (a1[1] - 3.0 * g).abs() <= 1e-13 * a1[1].abs().max(1e-3),
            "x = {x}: 3g"
        );
        let plus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a + b).collect();
        let minus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a - b).collect();
        let scale = a1.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        for (j, &u) in mu.iter().enumerate() {
            let (f11, f12, f33, f34) = (f.phase[0][j], f.phase[1][j], f.phase[2][j], f.phase[3][j]);
            let tol = 1e-12 * scale * a1.len() as f64 * f11.max(1.0);
            let checks = [
                ("F11", expand(a1, 0, 0, u), f11),
                ("F22 + F33", expand(&plus, 2, 2, u), f11 + f33),
                ("F22 - F33", expand(&minus, 2, -2, u), f11 - f33),
                ("F44", expand(a4, 0, 0, u), f33),
                ("F12", expand(b1, 0, 2, u), f12),
                ("F34", expand(b2, 0, 2, u), f34),
            ];
            for (name, sum, value) in checks {
                assert!(
                    (sum - value).abs() <= tol,
                    "x = {x}, m = {m}, mu = {u}: {name} = {value:e}, expansion {sum:e}"
                );
            }
        }
    }
}

#[test]
fn orders_above_2n_are_exactly_zero_and_lower_orders_a_prefix() {
    let (x, m) = (5.0, c(1.5, 0.01));
    let mut mie = Mie::<f64>::new();
    let s = mie.compute(x, m, &MieOptions::new()).unwrap();
    let n = s.n_terms();
    let all = s.greek(None, Normalization::Normalized).unwrap();
    assert_eq!(all.order(), 2 * n);
    let more = s.greek(Some(2 * n + 7), Normalization::Normalized).unwrap();
    assert_eq!(more.order(), 2 * n + 7);
    let fields = |g: &GreekCoefficients<f64>| {
        [
            g.alpha1.clone(),
            g.alpha2.clone(),
            g.alpha3.clone(),
            g.alpha4.clone(),
            g.beta1.clone(),
            g.beta2.clone(),
        ]
    };
    for (a, b) in fields(&all).iter().zip(fields(&more).iter()) {
        assert_eq!(&b[..=2 * n], &a[..], "orders up to 2N agree bitwise");
        assert!(
            b[2 * n + 1..].iter().all(|&v| v == 0.0),
            "orders above 2N are 0"
        );
    }
    let low = s.greek(Some(6), Normalization::Normalized).unwrap();
    assert_eq!(low.order(), 6);
    for (a, b) in fields(&all).iter().zip(fields(&low).iter()) {
        for l in 0..=6 {
            assert!(
                (a[l] - b[l]).abs() <= 1e-13 * a[l].abs().max(1.0),
                "order {l}"
            );
        }
    }
    // β₁, β₂ and the d²² parts start at l = 2.
    assert_eq!(
        [all.beta1[0], all.beta1[1], all.beta2[0], all.beta2[1]],
        [0.0; 4]
    );
    assert_eq!(
        [all.alpha2[0], all.alpha2[1], all.alpha3[0], all.alpha3[1]],
        [0.0; 4]
    );
}

#[test]
fn the_projector_is_exact_for_polynomials() {
    // Tables built from known coefficients up to order K, projected with a rule exact for
    // degree K + L: the coefficients come back, and those above K are 0.
    let (k_max, order) = (37usize, 45usize);
    let coef = |s: f64| -> Vec<f64> {
        (0..=k_max)
            .map(|l| ((l as f64 + s) * 0.7).sin() / (1.0 + l as f64))
            .collect()
    };
    let mut known = [
        coef(0.1),
        coef(0.2),
        coef(0.3),
        coef(0.4),
        coef(0.5),
        coef(0.6),
    ];
    for k in [1, 2, 4, 5] {
        known[k][0] = 0.0;
        known[k][1] = 0.0;
    }
    let [a1, a2, a3, a4, b1, b2] = &known;
    let n_nodes = (k_max + order) / 2 + 1;
    let p = GreekProjector::<f64>::new(order, n_nodes).unwrap();
    assert_eq!(p.order(), order);
    let nodes = p.nodes().to_vec();
    let plus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a + b).collect();
    let minus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a - b).collect();
    let f11: Vec<f64> = nodes.iter().map(|&u| expand(a1, 0, 0, u)).collect();
    let f33: Vec<f64> = nodes.iter().map(|&u| expand(a4, 0, 0, u)).collect();
    // F₂₂ = F₁₁ for a sphere, so α₂ ± α₃ must expand F₁₁ ± F₃₃: build tables that satisfy it by
    // taking F₁₁ + F₃₃ and F₁₁ − F₃₃ from α₂ ± α₃ instead, with F₁₁ and F₃₃ from those.
    let sum: Vec<f64> = nodes.iter().map(|&u| expand(&plus, 2, 2, u)).collect();
    let diff: Vec<f64> = nodes.iter().map(|&u| expand(&minus, 2, -2, u)).collect();
    let (f11_s, f33_s): (Vec<f64>, Vec<f64>) = sum
        .iter()
        .zip(&diff)
        .map(|(s, d)| ((s + d) / 2.0, (s - d) / 2.0))
        .unzip();
    let f12: Vec<f64> = nodes.iter().map(|&u| expand(b1, 0, 2, u)).collect();
    let f34: Vec<f64> = nodes.iter().map(|&u| expand(b2, 0, 2, u)).collect();
    let out = p
        .project(&[
            PhaseTables {
                f11: &f11,
                f12: &f12,
                f33: &f33,
                f34: &f34,
            },
            PhaseTables {
                f11: &f11_s,
                f12: &f12,
                f33: &f33_s,
                f34: &f34,
            },
        ])
        .unwrap();
    let tol = 1e-13;
    let get = |g: &GreekCoefficients<f64>| {
        [
            g.alpha1.clone(),
            g.alpha2.clone(),
            g.alpha3.clone(),
            g.alpha4.clone(),
            g.beta1.clone(),
            g.beta2.clone(),
        ]
    };
    let [g1, g2] = [get(&out[0]), get(&out[1])];
    for l in 0..=order {
        let want = |k: usize| if l <= k_max { known[k][l] } else { 0.0 };
        // α₁, α₄, β₁, β₂ from the first table; α₂, α₃ from the second.
        for k in [0, 3, 4, 5] {
            assert!(
                (g1[k][l] - want(k)).abs() <= tol,
                "table 1, element {k}, l = {l}"
            );
        }
        for k in [1, 2] {
            assert!(
                (g2[k][l] - want(k)).abs() <= tol,
                "table 2, element {k}, l = {l}"
            );
        }
    }
}

type D3 = Dual<f64, 3>;

/// The closed-form derivatives against `Dual<f64, 3>` (seeds x, Re m, Im m) propagated through
/// the same computation: the values agree bitwise, and the derivatives within 16·κ′·2⁻⁵² of
/// the measure scales of `MieOutput::PhaseMatrixDerivatives` and `GreekDerivatives`, with κ′
/// their a-priori factors.
#[test]
fn closed_form_derivatives_agree_with_dual() {
    let mu = [1.0, 0.6, 0.0, -0.3, -1.0];
    for normalization in [Normalization::Normalized, Normalization::Unnormalized] {
        for (x, m) in SPHERES {
            let f = solve(x, m, &mu, normalization);
            let xd = D3::new(x, [1.0, 0.0, 0.0]);
            let md = Complex::new(
                D3::new(m.re, [0.0, 1.0, 0.0]),
                D3::new(m.im, [0.0, 0.0, 1.0]),
            );
            let mu_d: Vec<D3> = mu.iter().map(|&v| D3::constant(v)).collect();
            let d = solve_greek(xd, md, &mu_d, None, None, normalization);
            let near_one = (m.norm() / (m - 1.0).norm()).max(1.0);
            let kappa = ValidatedDomain::condition_factor(x, m) * near_one;
            let l = f.greek[0].len() as f64;
            let bound_f = 16.0 * kappa * EPS;
            let bound_g = 16.0 * (kappa + (l * l + x * x) / 8.0) * EPS;
            let phase = 1.0 + m.norm() * x;
            let sizes = [x, m.norm(), m.norm()];
            let mean = f.greek[0][0];
            for k in 0..4 {
                for j in 0..mu.len() {
                    assert_eq!(d.phase[k][j].re(), f.phase[k][j], "x = {x}: F{k} value");
                }
                for p in 0..3 {
                    let natural = |j: usize| f.phase[0][j].max(mean) * phase / sizes[p];
                    let scale = (0..mu.len())
                        .map(|j| f.phase_d[p][k][j].abs().max(natural(j)))
                        .fold(0.0, f64::max);
                    for j in 0..mu.len() {
                        let err = (d.phase[k][j].eps()[p] - f.phase_d[p][k][j]).abs();
                        assert!(
                            err <= bound_f * scale,
                            "{normalization:?} x = {x}, m = {m}: dF{k}/dp{p} at {}: {err:e}",
                            mu[j]
                        );
                    }
                }
            }
            let a = f.greek[0].iter().fold(0.0f64, |s, v| s.max(v.abs()));
            for k in 0..6 {
                for (lv, v) in d.greek[k].iter().enumerate() {
                    assert_eq!(
                        v.re(),
                        f.greek[k][lv],
                        "x = {x}: Greek {k} value at l = {lv}"
                    );
                }
                for p in 0..3 {
                    let natural = a * phase / sizes[p];
                    let scale = f.greek_d[p][k]
                        .iter()
                        .map(|v| v.abs().max(natural))
                        .fold(0.0, f64::max);
                    for (lv, v) in d.greek[k].iter().enumerate() {
                        let err = (v.eps()[p] - f.greek_d[p][k][lv]).abs();
                        assert!(
                            err <= bound_g * scale,
                            "{normalization:?} x = {x}, m = {m}: Greek {k} p{p} l = {lv}: {err:e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bad_buffers_tables_and_solutions_are_rejected() {
    let mut mie = Mie::<f64>::new();
    let s = mie.compute(2.0, c(1.5, 0.0), &MieOptions::new()).unwrap();
    let mu = [0.5, 0.0];
    let mut f: [Vec<f64>; 4] = std::array::from_fn(|_| vec![7.0; 2]);
    f[2] = vec![7.0; 3];
    let [f11, f12, f33, f34] = &mut f;
    let err = s.phase_matrix(
        &mu,
        Normalization::Normalized,
        PhaseBuffers { f11, f12, f33, f34 },
    );
    assert!(matches!(err, Err(Error::InvalidInput(_))), "{err:?}");
    assert!(
        f.iter().all(|v| v.iter().all(|&x| x == 7.0)),
        "nothing written"
    );
    let mut g: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; 2]);
    let [f11, f12, f33, f34] = &mut g;
    let bad = s.phase_matrix(
        &[1.5],
        Normalization::Normalized,
        PhaseBuffers {
            f11: &mut f11[..1],
            f12: &mut f12[..1],
            f33: &mut f33[..1],
            f34: &mut f34[..1],
        },
    );
    assert!(matches!(bad, Err(Error::InvalidInput(_))), "{bad:?}");
    // Derivatives need a solution with derivatives.
    let mut v: [[Vec<f64>; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; 2]));
    let [a, b, cc, dd] = &mut v;
    let r = s.phase_matrix_with_derivatives(
        &mu,
        Normalization::Normalized,
        bufs(a),
        [bufs(b), bufs(cc), bufs(dd)],
    );
    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
    let r = s.greek_with_derivatives(None, Normalization::Normalized);
    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
    // Tables of the wrong length.
    let p = GreekProjector::<f64>::new(4, 6).unwrap();
    let short = vec![0.0; 5];
    let r = p.project(&[PhaseTables {
        f11: &short,
        f12: &short,
        f33: &short,
        f34: &short,
    }]);
    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
    assert!(matches!(
        GreekProjector::<f64>::new(4, 0),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn derivatives_values_match_the_plain_values_bitwise() {
    let mu = [1.0, 0.2, -1.0];
    for (x, m) in SPHERES {
        let mut mie = Mie::<f64>::new();
        let plain = mie.compute(x, m, &MieOptions::new()).unwrap();
        let g = plain.greek(None, Normalization::Normalized).unwrap();
        let mut f: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; mu.len()]);
        {
            let [f11, f12, f33, f34] = &mut f;
            plain
                .phase_matrix(
                    &mu,
                    Normalization::Normalized,
                    PhaseBuffers { f11, f12, f33, f34 },
                )
                .unwrap();
        }
        let both = solve(x, m, &mu, Normalization::Normalized);
        assert_eq!(both.phase, f, "x = {x}: phase matrix");
        assert_eq!(both.greek[0], g.alpha1, "x = {x}: alpha1");
        assert_eq!(both.greek[5], g.beta2, "x = {x}: beta2");
    }
}

/// The generic code runs in every scalar type.
#[test]
fn double_double_greek_runs() {
    use scatrs::scalar::DoubleDouble as Dd;
    let mut mie = Mie::<Dd>::new();
    let s = mie
        .compute(
            Dd::from_f64(2.0),
            Complex::new(Dd::from_f64(1.5), Dd::from_f64(0.0)),
            &MieOptions::new(),
        )
        .unwrap();
    let g = s.greek(None, Normalization::Normalized).unwrap();
    assert_eq!(g.alpha1[0], Dd::from_f64(1.0));
}

/// The sign conventions, pinned by the Rayleigh limit (x → 0): F₁₁ = ¾(1 + μ²) and
/// F₁₂ = −¾(1 − μ²) (Bohren & Huffman's F₁₂ = (|S₂|² − |S₁|²)/2 is negative at 90°), so with
/// `wigner_d`'s d²₀₂ = (√6/4) sin²θ ≥ 0, β₁ at l = 2 is −√6/2.
#[test]
fn rayleigh_limit_pins_the_sign_conventions() {
    let f = solve(1e-4, c(1.5, 0.0), &[0.0], Normalization::Normalized);
    let expected: [[f64; 4]; 6] = [
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0; 4],
        [0.0, 1.5, 0.0, 0.0],
        [0.0, 0.0, -(6f64.sqrt()) / 2.0, 0.0],
        [0.0; 4],
    ];
    for (k, row) in expected.iter().enumerate() {
        for (l, &want) in row.iter().enumerate() {
            let got = f.greek[k][l];
            assert!(
                (got - want).abs() <= 1e-7,
                "Greek element {k} at l = {l}: {got}"
            );
        }
    }
    assert!((f.phase[0][0] - 0.75).abs() <= 1e-7 && (f.phase[1][0] + 0.75).abs() <= 1e-7);
}
