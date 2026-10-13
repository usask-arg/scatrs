#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::{Record, load};
use scatrs::Error;
use scatrs::scalar::{DoubleDouble as Dd, Dual};
use scatrs::special::{gauss_legendre, wigner_d, wigner_d_vec};

const EPS: f64 = f64::EPSILON;

/// The unit of the double-double contract, 2⁻⁹⁷.
fn dd_unit() -> f64 {
    2f64.powi(-97)
}

/// The degrees a reference record holds: `ls` for partial records, else 0..=l_max.
fn degrees(r: &Record) -> Vec<usize> {
    if r.inputs.contains_key("ls") {
        r.input_i64s("ls").iter().map(|&l| l as usize).collect()
    } else {
        (0..=r.input_i64("l_max") as usize).collect()
    }
}

fn l_min(m: i32, n: i32) -> usize {
    m.unsigned_abs().max(n.unsigned_abs()) as usize
}

#[test]
fn closed_forms() {
    for x in [-1.0, -0.6, 0.0, 0.25, 0.9, 1.0_f64] {
        let s = (1.0 - x * x).sqrt();
        let close = |a: f64, b: f64| assert!((a - b).abs() <= 4.0 * EPS, "x={x}: {a} vs {b}");
        let d00 = wigner_d_vec(0, 0, 2, x).unwrap();
        close(d00[0], 1.0);
        close(d00[1], x);
        close(d00[2], (3.0 * x * x - 1.0) / 2.0);
        close(wigner_d_vec(1, 1, 1, x).unwrap()[1], (1.0 + x) / 2.0);
        close(wigner_d_vec(1, -1, 1, x).unwrap()[1], (1.0 - x) / 2.0);
        close(wigner_d_vec(0, 1, 1, x).unwrap()[1], s / 2f64.sqrt());
        close(wigner_d_vec(1, 0, 1, x).unwrap()[1], -s / 2f64.sqrt());
        close(
            wigner_d_vec(2, 2, 2, x).unwrap()[2],
            ((1.0 + x) / 2.0).powi(2),
        );
        close(
            wigner_d_vec(2, -2, 2, x).unwrap()[2],
            ((1.0 - x) / 2.0).powi(2),
        );
        close(
            wigner_d_vec(0, 2, 2, x).unwrap()[2],
            6f64.sqrt() / 4.0 * s * s,
        );
        assert_eq!(wigner_d_vec(2, 2, 2, x).unwrap()[..2], [0.0, 0.0]);
    }
}

#[test]
fn symmetry_relations() {
    for (m, n) in [(1, 3), (-2, 4), (5, -1), (0, 2)] {
        for x in [-0.7, 0.1, 0.95_f64] {
            let a = wigner_d_vec(m, n, 30, x).unwrap();
            let b = wigner_d_vec(n, m, 30, x).unwrap();
            let c = wigner_d_vec(-n, -m, 30, x).unwrap();
            let sign = if (m - n) % 2 == 0 { 1.0 } else { -1.0 };
            for l in 0..=30 {
                // Each side carries up to ~16(l+1) ulps of recurrence error.
                let tol = 32.0 * (l as f64 + 1.0) * EPS;
                assert!((a[l] - sign * b[l]).abs() <= tol);
                assert!((a[l] - c[l]).abs() <= tol);
            }
        }
    }
}

/// d^l_mn = (−1)^(m−n) d^l_nm = d^l_(−n)(−m) and d^l_mn(−x) = (−1)^(l+m) d^l_m(−n)(x), for every
/// sign combination of m and n, m = ±n, l_min = 0, 1 and 50, and x at and next to ±1 and 0.
#[test]
fn symmetry_relations_over_signs_and_poles() {
    let pairs = [
        (0, 0),
        (1, 0),
        (0, -1),
        (1, 1),
        (-1, -1),
        (2, -2),
        (-3, 3),
        (3, 5),
        (-3, 5),
        (3, -5),
        (-3, -5),
        (7, -2),
        (-6, -1),
        (50, -37),
        (-50, 37),
        (37, 50),
    ];
    let xs = [
        -1.0,
        -1.0 + 2f64.powi(-53),
        -0.7,
        -0.0,
        0.0,
        0.1,
        0.95,
        1.0 - 3.0 * 2f64.powi(-53),
        1.0 - 2f64.powi(-53),
        1.0,
    ];
    let l_max = 120;
    for (m, n) in pairs {
        for x in xs {
            let d = wigner_d_vec(m, n, l_max, x).unwrap();
            let swapped = wigner_d_vec(n, m, l_max, x).unwrap();
            let negated = wigner_d_vec(-n, -m, l_max, x).unwrap();
            let reflected = wigner_d_vec(m, -n, l_max, -x).unwrap();
            let parity = |k: i32| if k.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            for l in 0..=l_max {
                // Each side carries up to 16(l+1) ulps (the accuracy contract).
                let tol = 32.0 * (l as f64 + 1.0) * EPS;
                let case = format!("m={m} n={n} x={x} l={l}");
                assert!((d[l] - parity(m - n) * swapped[l]).abs() <= tol, "{case}");
                assert!((d[l] - negated[l]).abs() <= tol, "{case}");
                let r = parity(l as i32 + m) * reflected[l];
                assert!((d[l] - r).abs() <= tol, "{case}: {} vs {r}", d[l]);
            }
        }
    }
}

/// The rows of the unitary matrix d^l(θ) have unit norm: Σₙ (d^l_mn)² = 1 for every m, which
/// checks every (m, n) with |m|, |n| ≤ 40 at once, including those next to the poles.
#[test]
fn rows_are_unit_vectors() {
    let l_max = 40_usize;
    let big = l_max as i32;
    for x in [
        -1.0,
        -1.0 + 2f64.powi(-53),
        -0.3,
        0.0,
        0.7,
        0.9999999,
        1.0 - 2f64.powi(-53),
        1.0,
    ] {
        for m in -big..=big {
            let rows: Vec<Vec<f64>> = (-big..=big)
                .map(|n| wigner_d_vec(m, n, l_max, x).unwrap())
                .collect();
            for l in m.unsigned_abs() as usize..=l_max {
                let sum: f64 = rows.iter().map(|d| d[l] * d[l]).sum();
                // From the contract: Σ 2|d|·16(l+1)ε ≤ 32(l+1)ε·√(2l+1) by Cauchy–Schwarz, plus
                // (2l+1)ε for rounding the sum.
                let lf = l as f64;
                let tol = (32.0 * (lf + 1.0) * (2.0 * lf + 1.0).sqrt() + 2.0 * lf + 1.0) * EPS;
                assert!((sum - 1.0).abs() <= tol, "x={x} m={m} l={l}: {sum}");
            }
        }
    }
}

/// d^l_mn(0) = δ_mn and d^l_mn(π) = (−1)^(l−n) δ_(m,−n), with exact zeros where m ≠ ±n.
#[test]
fn values_at_the_poles() {
    for (m, n) in [
        (0, 0),
        (3, 3),
        (-4, -4),
        (2, -2),
        (-5, 5),
        (1, 0),
        (3, -7),
        (-2, 6),
    ] {
        for x in [1.0, -1.0_f64] {
            let d = wigner_d_vec(m, n, 300, x).unwrap();
            for l in 0..=300 {
                let pole = if x > 0.0 { m == n } else { m == -n };
                let sign = if x > 0.0 || (l as i32 - n) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                if l < l_min(m, n) || !pole {
                    assert_eq!(d[l], 0.0, "m={m} n={n} x={x} l={l}");
                } else {
                    let tol = 16.0 * (l as f64 + 1.0) * EPS;
                    assert!(
                        (d[l] - sign).abs() <= tol,
                        "m={m} n={n} x={x} l={l}: {}",
                        d[l]
                    );
                }
            }
        }
    }
}

#[test]
fn orthogonality_under_gauss_legendre() {
    let rule = gauss_legendre::<f64>(64).unwrap();
    for (m, n) in [(0, 0), (2, 2), (2, -2), (0, 2), (1, 3)] {
        let table: Vec<Vec<f64>> = rule
            .nodes
            .iter()
            .map(|&x| wigner_d_vec(m, n, 40, x).unwrap())
            .collect();
        let l_min = m.unsigned_abs().max(n.unsigned_abs()) as usize;
        for l in l_min..=40 {
            for k in l_min..=40 {
                let s: f64 = rule
                    .weights
                    .iter()
                    .zip(&table)
                    .map(|(w, d)| w * d[l] * d[k])
                    .sum();
                let expected = if l == k {
                    2.0 / (2.0 * l as f64 + 1.0)
                } else {
                    0.0
                };
                assert!(
                    (s - expected).abs() <= 2e-12,
                    "m={m} n={n} l={l} k={k}: {s}"
                );
            }
        }
    }
}

#[test]
fn matches_arb_reference() {
    let (mut worst, mut worst_dd) = (0.0_f64, 0.0_f64);
    let (mut worst_large, mut worst_large_dd) = (0.0_f64, 0.0_f64);
    for r in load("wigner_d.jsonl") {
        let (m, n) = (r.input_i64("m") as i32, r.input_i64("n") as i32);
        let (x, l_max) = (r.input_f64("x"), r.input_i64("l_max") as usize);
        let d = r.output("d");
        let got = wigner_d_vec(m, n, l_max, x).unwrap();
        let got_dd = wigner_d_vec(m, n, l_max, Dd::from_f64(x)).unwrap();
        for (i, l) in degrees(&r).into_iter().enumerate() {
            let scale = l as f64 + 1.0;
            let err = d.dd_error(i, got[l], 0.0);
            let tol = 16.0 * scale * EPS;
            worst = worst.max(err / (scale * EPS));
            assert!(err <= tol, "{}: l={l} f64 error {err:e} > {tol:e}", r.case);
            let err_dd = d.dd_error(i, got_dd[l].hi(), got_dd[l].lo());
            let tol_dd = scale * dd_unit();
            worst_dd = worst_dd.max(err_dd / (scale * dd_unit()));
            assert!(
                err_dd <= tol_dd,
                "{}: l={l} double-double error {err_dd:e}",
                r.case
            );
            if l >= 1000 {
                worst_large = worst_large.max(err / (scale * EPS));
                worst_large_dd = worst_large_dd.max(err_dd / (scale * dd_unit()));
            }
        }
    }
    eprintln!("wigner_d f64: worst error {worst:.2} (l+1) eps ({worst_large:.2} for l >= 1000)");
    eprintln!(
        "wigner_d double-double: worst error {worst_dd:.4} (l+1) 2^-97 ({worst_large_dd:.4} for \
         l >= 1000)"
    );
}

/// Where the start value of the recurrence, at l = max(|m|, |n|), lies below the normal range,
/// the values that are normal are computed to the contract relative to their size (not flushed
/// to zero), and smaller ones to that bound plus the rounding to a subnormal. The seven records
/// cover the three-term form, both difference forms and the reflection x → −x.
#[test]
fn values_beyond_an_underflowing_start_are_accurate() {
    let (mut worst, mut worst_dd, mut normals) = (0.0_f64, 0.0_f64, 0);
    let records = load("wigner_d.jsonl");
    let cases: Vec<&Record> = records
        .iter()
        .filter(|r| r.case.ends_with(",underflow"))
        .collect();
    assert_eq!(cases.len(), 7);
    for r in cases {
        let (m, n) = (r.input_i64("m") as i32, r.input_i64("n") as i32);
        let (x, l_max) = (r.input_f64("x"), r.input_i64("l_max") as usize);
        let d = r.output("d");
        let got = wigner_d_vec(m, n, l_max, x).unwrap();
        let got_dd = wigner_d_vec(m, n, l_max, Dd::from_f64(x)).unwrap();
        for (i, l) in degrees(r).into_iter().enumerate() {
            let v = d.hi[i].abs();
            let scale = l as f64 + 1.0;
            let err = d.dd_error(i, got[l], 0.0);
            let err_dd = d.dd_error(i, got_dd[l].hi(), got_dd[l].lo());
            let case = format!("{}: l={l}, reference {}", r.case, d.dec[i]);
            if v >= f64::MIN_POSITIVE {
                normals += 1;
                // The absolute contract, relative to the value: the recurrence is stable here.
                let rel = err / v;
                worst = worst.max(rel / (scale * EPS));
                assert!(
                    rel <= 16.0 * scale * EPS,
                    "{case}: f64 relative error {rel:e}"
                );
            } else {
                // The same relative bound, plus half the spacing of subnormal doubles, 2⁻¹⁰⁷⁵,
                // for rounding the scaled value once. `err` includes the reference's own bound
                // d.err[i], which is 2⁻¹⁰⁷⁴ for values below the subnormal range (stored as 0).
                let tol = 16.0 * scale * EPS * v + 2f64.powi(-1075) + d.err[i];
                assert!(err <= tol, "{case}: f64 error {err:e}");
            }
            // Double-double has full precision above 2⁻⁹⁶⁹; below, its low word is subnormal.
            if v >= 2f64.powi(-969) {
                let rel = err_dd / v;
                worst_dd = worst_dd.max(rel / (scale * dd_unit()));
                assert!(
                    rel <= scale * dd_unit(),
                    "{case}: double-double error {rel:e}"
                );
            } else {
                // The same relative bound, plus the rounding of the words of the scaled value
                // to multiples of 2⁻¹⁰⁷⁴, plus the reference's own bound (as above).
                let tol = scale * dd_unit() * v + 2f64.powi(-1074) + d.err[i];
                assert!(err_dd <= tol, "{case}: double-double error {err_dd:e}");
            }
        }
    }
    assert!(normals >= 30, "only {normals} normal reference values");
    eprintln!(
        "wigner_d past an underflowing start: worst relative error {worst:.2} (l+1) eps (f64), \
         {worst_dd:.4} (l+1) 2^-97 (double-double)"
    );
}

#[test]
fn invalid_arguments() {
    let mut out = [0.0; 4];
    for bad in [
        1.5,
        -1.0000001,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0 + EPS,
    ] {
        match wigner_d(0, 0, bad, &mut out) {
            Err(Error::InvalidInput(msg)) => assert!(msg.contains("-1 <= x <= 1"), "{msg}"),
            other => panic!("expected InvalidInput for x={bad}, got {other:?}"),
        }
        match wigner_d_vec(2, -1, 10, bad) {
            Err(Error::InvalidInput(msg)) => assert!(msg.contains("-1 <= x <= 1"), "{msg}"),
            other => panic!("expected InvalidInput for x={bad}, got {other:?}"),
        }
    }
    assert!(wigner_d::<f64>(0, 0, 0.5, &mut []).is_ok());
    // An invalid x with an empty output is still an error.
    assert!(wigner_d::<f64>(0, 0, 2.0, &mut []).is_err());
    // Double-double and Dual arguments are checked on their full value.
    let above_one = Dd::new(1.0, 1e-20);
    assert!(matches!(
        wigner_d_vec(0, 0, 3, above_one),
        Err(Error::InvalidInput(_))
    ));
    assert!(wigner_d_vec(0, 0, 3, Dd::new(1.0, -1e-20)).is_ok());
    let [nan] = Dual::<f64, 1>::variables([f64::NAN]);
    assert!(matches!(
        wigner_d_vec(0, 0, 3, nan),
        Err(Error::InvalidInput(_))
    ));
}

/// Orders up to `i32::MIN` neither panic nor overflow: entries below l = max(|m|, |n|) are zero.
#[test]
fn extreme_orders() {
    let extremes = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX];
    for m in extremes {
        for n in extremes {
            let mut out = [1.0_f64; 8];
            wigner_d(m, n, 0.3, &mut out).unwrap();
            let lm = l_min(m, n);
            for (l, &v) in out.iter().enumerate() {
                if l < lm {
                    assert_eq!(v, 0.0, "m={m} n={n} l={l}");
                } else {
                    assert!(v.is_finite() && v.abs() <= 1.0, "m={m} n={n} l={l}: {v}");
                }
            }
            assert_eq!(wigner_d_vec(m, n, 3, -1.0).unwrap().len(), 4);
        }
    }
}

/// `wigner_d_vec` reports a length it cannot allocate instead of panicking or aborting.
#[test]
fn enormous_l_max_is_an_error() {
    for l_max in [usize::MAX, usize::MAX - 1, usize::MAX / 2, usize::MAX / 8] {
        match wigner_d_vec(0, 0, l_max, 0.5) {
            Err(Error::InvalidInput(msg)) => assert!(msg.contains("l_max"), "{msg}"),
            other => panic!("expected InvalidInput for l_max={l_max}, got {other:?}"),
        }
    }
    // An invalid x is reported first, before any allocation.
    match wigner_d_vec(0, 0, usize::MAX / 8, f64::NAN) {
        Err(Error::InvalidInput(msg)) => assert!(msg.contains("-1 <= x <= 1"), "{msg}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn derivative_in_x_matches_legendre_derivative() {
    let x0 = 0.3_f64;
    let [x] = Dual::<f64, 1>::variables([x0]);
    let d = wigner_d_vec(0, 0, 20, x).unwrap();
    let p = wigner_d_vec(0, 0, 20, x0).unwrap();
    for l in 1..=20 {
        // P_l'(x) = l (x P_l − P_{l−1}) / (x² − 1)
        let expected = l as f64 * (x0 * p[l] - p[l - 1]) / (x0 * x0 - 1.0);
        assert!(
            (d[l].eps()[0] - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "l={l}"
        );
    }
}

/// The ladder relation d/dθ d^l_mn = ½[√((l+n)(l−n+1)) d^l_m(n−1) − √((l−n)(l+n+1)) d^l_m(n+1)],
/// evaluated in double-double, with dx = −sin θ dθ.
fn ladder_derivative(m: i32, n: i32, l: usize, x: f64) -> Dd {
    let x_dd = Dd::from_f64(x);
    let (lf, nf) = (l as f64, n as f64);
    let lower = wigner_d_vec(m, n - 1, l, x_dd).unwrap()[l];
    let upper = wigner_d_vec(m, n + 1, l, x_dd).unwrap()[l];
    let c_lower = Dd::from_f64((lf + nf) * (lf - nf + 1.0)).sqrt();
    let c_upper = Dd::from_f64((lf - nf) * (lf + nf + 1.0)).sqrt();
    let d_theta = (c_lower * lower - c_upper * upper) * 0.5;
    let sin = ((Dd::ONE - x_dd) * (Dd::ONE + x_dd)).sqrt();
    -d_theta / sin
}

/// `Dual` derivatives in x, for general (m, n), against the ladder relation, away from ±1.
#[test]
fn derivative_in_x_matches_ladder_relation() {
    let (mut worst, mut worst_dd) = (0.0_f64, 0.0_f64);
    for (m, n) in [
        (0, 2),
        (2, 2),
        (2, -2),
        (1, 0),
        (-3, 5),
        (5, -3),
        (-4, -4),
        (10, -7),
    ] {
        for x0 in [-0.999, -0.6, 0.0, 0.3, 0.9, 0.9999999] {
            let [x] = Dual::<f64, 1>::variables([x0]);
            let d = wigner_d_vec(m, n, 60, x).unwrap();
            let [x_dd] = Dual::<Dd, 1>::variables([Dd::from_f64(x0)]);
            let d_dd = wigner_d_vec(m, n, 60, x_dd).unwrap();
            let values = wigner_d_vec(m, n, 60, x0).unwrap();
            for l in l_min(m, n).max(1)..=60 {
                let exact = ladder_derivative(m, n, l, x0);
                let scale = exact.abs().hi().max(1.0);
                // The value is bitwise the f64 value.
                assert_eq!(d[l].re(), values[l]);
                // f64: the values err by up to 16(l+1) ulps, and the derivative is a degree-l
                // polynomial in x (times half-angle factors), so its error is up to about
                // l·16(l+1) ulps relative to its size, amplified near ±1 by 1/(1 − x²).
                let err = (d[l].eps()[0] - exact.hi()).abs() / scale;
                let lf = l as f64;
                let tol = 16.0 * (lf + 1.0) * lf * EPS / (1.0 - x0 * x0);
                worst = worst.max(err / tol);
                assert!(
                    err <= tol,
                    "m={m} n={n} x={x0} l={l}: {} vs {exact:?}",
                    d[l].eps()[0]
                );
                let err_dd = (d_dd[l].eps()[0] - exact).abs().hi() / scale;
                let tol_dd = (lf + 1.0) * lf * dd_unit() / (1.0 - x0 * x0);
                worst_dd = worst_dd.max(err_dd / tol_dd);
                assert!(err_dd <= tol_dd, "m={m} n={n} x={x0} l={l}: double-double");
            }
        }
    }
    eprintln!(
        "wigner_d x-derivatives: worst {worst:.4} of the f64 bound, {worst_dd:.4} of the \
         double-double bound"
    );
}

/// At x = ±1 a constant `Dual` (zero derivative) gives the `f64` values bitwise and exactly zero
/// derivatives, whatever the half-angle powers.
#[test]
fn constant_dual_at_the_poles() {
    for (m, n) in [(0, 0), (1, 0), (0, 3), (2, 2), (2, -2), (-3, 5), (50, -37)] {
        for x0 in [-1.0, 1.0_f64] {
            let x = Dual::<f64, 2>::constant(x0);
            let d = wigner_d_vec(m, n, 80, x).unwrap();
            let v = wigner_d_vec(m, n, 80, x0).unwrap();
            for l in 0..=80 {
                assert_eq!(
                    d[l].re().to_bits(),
                    v[l].to_bits(),
                    "m={m} n={n} x={x0} l={l}"
                );
                assert_eq!(d[l].eps(), &[0.0, 0.0], "m={m} n={n} x={x0} l={l}");
            }
        }
    }
}

/// At x = ±1 with a nonzero derivative in x: d^l_mn contains sin^a(θ/2) = ((1 − x)/2)^(a/2),
/// a = |m − n|, and cos^b(θ/2) = ((1 + x)/2)^(b/2), b = |m + n|. The derivative at x = 1 is
/// infinite for a = 1, so it must not be finite there; for a = 0 it is finite and computed.
/// (For a ≥ 2 the documented result is NaN.) The same holds at x = −1 with b.
#[test]
fn derivative_at_the_poles() {
    let l_max = 30;
    for x0 in [1.0, -1.0_f64] {
        let [x] = Dual::<f64, 1>::variables([x0]);
        for m in -4..=4_i32 {
            for n in -4..=4_i32 {
                let d = wigner_d_vec(m, n, l_max, x).unwrap();
                let power = if x0 > 0.0 { m - n } else { m + n }.unsigned_abs();
                for l in l_min(m, n)..=l_max {
                    let dx = d[l].eps()[0];
                    let case = format!("m={m} n={n} x={x0} l={l}: derivative {dx}");
                    match power {
                        0 => {
                            // d^l_mm(θ) = 1 − (l(l+1) − m²)θ²/4 + O(θ⁴) and θ² ≈ 2(1 − x) at
                            // x = 1; at x = −1, d^l_m(−m) = (−1)^(l+m) d^l_mm(π − θ).
                            let lf = l as f64;
                            let slope = (lf * (lf + 1.0) - f64::from(m * m)) / 2.0;
                            let expected = if x0 > 0.0 {
                                slope
                            } else if (l as i32 + m) % 2 == 0 {
                                -slope
                            } else {
                                slope
                            };
                            let tol = 16.0 * (lf + 1.0) * lf * EPS * slope.abs().max(1.0);
                            assert!((dx - expected).abs() <= tol, "{case}, expected {expected}");
                        }
                        1 => assert!(!dx.is_finite(), "{case}"),
                        _ => assert!(dx.is_nan(), "{case}"),
                    }
                }
            }
        }
    }
}
