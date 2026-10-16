//! The spheroid T-matrix against the analytic long-wavelength expansion of Majic, Pratley,
//! Schebarchov, Somerville, Auguié & Le Ru, "Approximate T matrix and optical properties of
//! spheroidal particles to third order with respect to size parameter", Phys. Rev. A 99, 013853
//! (2019), arXiv:1810.06107: the elements of T for multipoles up to octupoles, Eqs. (33)–(66),
//! after their radiative correction (Eqs. 48, 59–64, 68), correct to O(X⁶).
//!
//! Their errors are O(X⁷) for every element, so the relative error of |T|² falls like X⁴ for the
//! two dipolar elements (of order X³) and like X² for the others (of order X⁵); and the
//! orientation-averaged cross sections, dominated by the dipoles, like X⁴. The tests check those
//! rates between two sizes. They compare |T|², and Re T for the diagonal elements, which do not
//! depend on the phase conventions of the vector spherical wave functions.
//!
//! Notation of the paper: h = c/a (ours is ε = a/c = 1/h), e = √(h² − 1)/h (imaginary for oblate
//! spheroids), X = k c = X̃ h^{2/3} with X̃ the equal-volume size parameter, s the relative index.

use num_complex::Complex;
use scatrs::tmatrix::{ModeBlocks, Precision, Shape, TMatrix, TMatrixOptions};

type C = Complex<f64>;

/// The paper's closed forms at (h, s, X): the elements as (m, i, j, n, k, T).
fn approximate(h: f64, s: C, x: f64) -> Vec<(usize, usize, usize, usize, usize, C)> {
    let one = C::new(1.0, 0.0);
    let i = C::new(0.0, 1.0);
    let e2 = C::new((h * h - 1.0) / (h * h), 0.0);
    let e = e2.sqrt();
    // atanh(e)/e: real for both shapes (atan(|e|)/|e| for oblate).
    let ae = if e2.re.abs() < 1e-30 {
        one
    } else {
        e.atanh() / e
    };
    let s21 = s * s - one;
    let lz = (one - e2) / e2 * (ae - one);
    let lx = -(one / (2.0 * e2)) * ((one - e2) * ae - one);
    let l20 = 1.5 * (one - e2) / e2 * ((C::new(3.0, 0.0) - e2) / e2 * ae - C::new(3.0, 0.0) / e2);
    let l21 = -(C::new(2.0, 0.0) - e2) / (2.0 * e2 * e2)
        * (3.0 * (one - e2) * ae - C::new(3.0, 0.0) + 2.0 * e2);
    let l22 = (one / (4.0 * e2 * e2))
        * (3.0 * (one - e2) * (one - e2) * ae - C::new(3.0, 0.0) + 5.0 * e2);
    let h2 = h * h;
    let h4 = h2 * h2;
    let k0 = 2.0 / (9.0 * h2) * s21 / (one + s21 * lz);
    let k1 = 2.0 / (9.0 * h2) * s21 / (one + s21 * lx);
    let om0 =
        (4.0 * e2 - 5.0 + s21 * (5.0 * (one - e2) + 9.0 * e2 * lz)) / (25.0 * (one + s21 * lz));
    let om1 =
        (3.0 * e2 - 5.0 + s21 * (C::new(5.0, 0.0) - 12.0 * e2 * lx)) / (25.0 * (one + s21 * lx));
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    // m = 0, Eqs. (39), (40), (48).
    let k11_11_0 = s21 / (45.0 * h4) * x5;
    let k22_22_0 = (C::new(3.0, 0.0) - e2) / (225.0 * h2) * s21 / (one + s21 * l20) * x5;
    let t22_11_0 = i * k0 * x3 / (one - om0 * x2 - i * k0 * x3);
    let t22_13_0 = 14f64.sqrt() / 175.0 * i * e2 * k0 * x5 / (one - i * k0 * x3);
    let t11_11_0 = i * k11_11_0 / (one - i * k11_11_0);
    let t22_22_0 = i * k22_22_0 / (one - i * k22_22_0);
    // m = 1, Eqs. (53)–(64).
    let k22_13_1 = 2.0 * e2 * 21f64.sqrt() / 525.0 * k1 * x5;
    let k21_12_1 = i * e2 * 15f64.sqrt() / 150.0 * k1 * x5;
    let k11_11_1 = s21 * (h2 * (C::new(2.0, 0.0) - e2) * (C::new(2.0, 0.0) - e2) + 4.0 * s21 * l21)
        / (90.0 * h4 * (C::new(2.0, 0.0) - e2) * (one + s21 * l21))
        * x5;
    let k22_22_1 = (C::new(2.0, 0.0) - e2) / (150.0 * h2) * s21 / (one + s21 * l21) * x5;
    let k21_21_1 = i * e2 * x5 / (30.0 * 15f64.sqrt() * h2) * s21 / (one + s21 * l21);
    let t22_11_1 = i * k1 * x3 / (one - om1 * x2 - i * k1 * x3);
    let t21_12_1 = i * k21_12_1 / (one - i * k1 * x3);
    let t22_31_1 = i * k22_13_1 / (one - i * k1 * x3);
    let t11_11_1 = i * k11_11_1 / (one - i * (k11_11_1 - k21_21_1 * k21_21_1 / k11_11_1));
    let t22_22_1 = i * k22_22_1 / (one - i * (k22_22_1 - k21_21_1 * k21_21_1 / k22_22_1));
    let t21_21_1 = i * k21_21_1 / (one - i * (k11_11_1 + k22_22_1));
    // m = 2, Eqs. (65)–(68).
    let k22_22_2 = x5 / (75.0 * h4) * s21 / (one + s21 * l22);
    let t22_22_2 = i * k22_22_2 / (one - i * k22_22_2);
    vec![
        (0, 1, 1, 1, 1, t11_11_0),
        (0, 2, 2, 1, 1, t22_11_0),
        (0, 2, 2, 1, 3, t22_13_0),
        (0, 2, 2, 3, 1, t22_13_0),
        (0, 2, 2, 2, 2, t22_22_0),
        (1, 1, 1, 1, 1, t11_11_1),
        (1, 2, 2, 1, 1, t22_11_1),
        (1, 2, 1, 1, 2, t21_12_1),
        (1, 1, 2, 2, 1, -t21_12_1),
        (1, 2, 2, 3, 1, t22_31_1),
        (1, 2, 2, 1, 3, t22_31_1),
        (1, 2, 2, 2, 2, t22_22_1),
        (1, 2, 1, 2, 1, t21_21_1),
        (1, 1, 2, 1, 2, -t21_21_1),
        (2, 2, 2, 2, 2, t22_22_2),
    ]
}

fn element(b: &ModeBlocks, i: usize, j: usize, n: usize, k: usize) -> C {
    let n0 = b.orders[0];
    let size = b.orders.len();
    let blk = match (i, j) {
        (1, 1) => &b.t11,
        (1, 2) => &b.t12,
        (2, 1) => &b.t21,
        _ => &b.t22,
    };
    blk[(n - n0) * size + (k - n0)]
}

struct Computed {
    blocks: Vec<ModeBlocks>,
    q_ext: f64,
    q_sca: f64,
}

/// Our T at equal-volume size parameter `xv`, to rounding (double-double, ample N and nodes).
fn exact(h: f64, s: C, xv: f64) -> Computed {
    let opts = TMatrixOptions::new()
        .with_precision(Precision::DoubleDouble)
        .with_n_terms(10)
        .with_n_theta(60)
        .with_allow_unvalidated(true);
    let mut tm = TMatrix::new();
    let sol = tm
        .compute(
            Shape::Spheroid {
                aspect_ratio: 1.0 / h,
            },
            xv,
            s,
            &opts,
        )
        .unwrap();
    let q = sol.efficiencies();
    Computed {
        blocks: (0..=2).map(|m| sol.block(m).unwrap()).collect(),
        q_ext: q.q_ext,
        q_sca: q.q_sca,
    }
}

/// The averaged efficiencies from the closed forms, Eqs. (76)–(77), divided by π X̃².
fn approximate_efficiencies(t: &[(usize, usize, usize, usize, usize, C)], xv: f64) -> (f64, f64) {
    let (mut ext, mut sca) = (0.0, 0.0);
    for &(m, i, j, n, k, v) in t {
        let w = if m == 0 { 1.0 } else { 2.0 };
        if i == j && n == k {
            ext -= w * v.re;
        }
        sca += w * v.norm_sqr();
    }
    let area = core::f64::consts::PI * xv * xv;
    let two_pi = 2.0 * core::f64::consts::PI;
    (two_pi * ext / area, two_pi * sca / area)
}

/// Relative errors of every element (|T|², and Re T on the diagonal) and of Q_ext, Q_sca.
fn errors(h: f64, s: C, xv: f64) -> Vec<(String, f64)> {
    let x = xv * h.powf(2.0 / 3.0);
    let t = approximate(h, s, x);
    let c = exact(h, s, xv);
    let mut out = Vec::new();
    for &(m, i, j, n, k, v) in &t {
        let w = element(&c.blocks[m], i, j, n, k);
        let name = format!("T{i}{j}_{n}{k}|{m}");
        out.push((
            format!("|{name}|²"),
            (v.norm_sqr() / w.norm_sqr() - 1.0).abs(),
        ));
        if i == j && n == k {
            out.push((format!("Re {name}"), (v.re / w.re - 1.0).abs()));
        }
    }
    let (qe, qs) = approximate_efficiencies(&t, xv);
    out.push(("Q_ext".into(), (qe / c.q_ext - 1.0).abs()));
    out.push(("Q_sca".into(), (qs / c.q_sca - 1.0).abs()));
    out
}

/// The rate of every element and of the averaged efficiencies between X̃ = 0.08 and 0.04: 4
/// for the dipoles and the efficiencies, 2 for the others (measured: within 0.03 of those for
/// all four cases), with the errors at X̃ = 0.04 where the expansion puts them.
#[test]
fn the_long_wavelength_expansion() {
    for &(h, s) in &[
        (3.0, C::new(1.3, 0.0)),
        (3.0, C::new(1.3, 0.2)),
        (0.5, C::new(1.5, 0.01)),
        (0.25, C::new(1.3, 0.2)),
    ] {
        let a = errors(h, s, 0.08);
        let b = errors(h, s, 0.04);
        for ((name, ea), (_, eb)) in a.iter().zip(&b) {
            let dipole = name.contains("22_11|") || name.starts_with('Q');
            let (rate, bound) = if dipole { (4.0, 1e-6) } else { (2.0, 1e-2) };
            let got = (ea / eb).log2();
            assert!(
                (got - rate).abs() < 0.15 && *eb < bound,
                "h {h} s {s} {name}: rate {got:.3} (expected {rate}), error {eb:.2e} at X̃ = 0.04"
            );
        }
    }
}
