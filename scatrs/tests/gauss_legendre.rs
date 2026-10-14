#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{Record, load};
use scatrs::Error;
use scatrs::scalar::{DoubleDouble as Dd, Dual};
use scatrs::special::{
    GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre, gauss_legendre_into,
};

const EPS: f64 = f64::EPSILON; // 2^-52
/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

/// From this many points on, the `f64` rule comes from the asymptotic expansions, within
/// 32·2⁻⁵²; below, it is the double-double rule rounded, within 2⁻⁵².
const ASYMPTOTIC_FROM: usize = 101;

/// The `f64` contract at n points: node absolute error and weight relative error at most this.
fn f64_tolerance(n: usize) -> f64 {
    if n >= ASYMPTOTIC_FROM {
        32.0 * EPS
    } else {
        EPS
    }
}

/// The indices into the rule of the values that reference record `r` holds, in the order
/// stored: every node of the rule, or only those listed in its `indices` input.
fn indices(r: &Record) -> Vec<usize> {
    if r.inputs.contains_key("indices") {
        r.input_i64s("indices")
            .into_iter()
            .map(|i| i as usize)
            .collect()
    } else {
        (0..r.input_i64("n") as usize).collect()
    }
}

/// Double-double rules above this size take seconds each (the cost is O(n²)), so their records
/// are checked in double-double only by the ignored test `largest_double_double_rules_match_arb`.
const SLOW_N: usize = 10_000;

/// The reference records whose rules have at most `SLOW_N` points, or more if `slow`, or all
/// (`None`).
///
/// Records whose case ends in `,roots` are skipped: they hold single roots of the largest
/// accepted rule, whose double-double form takes minutes to compute whole, and a unit test in
/// `src/special/gauss_legendre.rs` checks them root by root.
fn records(slow: Option<bool>) -> Vec<Record> {
    load("gauss_legendre.jsonl")
        .into_iter()
        .filter(|r| !r.case.ends_with(",roots"))
        .filter(|r| slow.is_none_or(|s| (r.input_i64("n") as usize > SLOW_N) == s))
        .collect()
}

/// Checks the `f64` rule against every record; returns the worst node and weight errors in
/// units of eps.
fn check_f64(records: &[Record]) -> (f64, f64) {
    let (mut worst_x, mut worst_w) = (0.0_f64, 0.0_f64);
    for r in records {
        let n = r.input_i64("n") as usize;
        let rule = gauss_legendre::<f64>(n).unwrap();
        let (xs, ws) = (r.output("nodes"), r.output("weights"));
        for (k, i) in indices(r).into_iter().enumerate() {
            let ex = xs.dd_error(k, rule.nodes[i], 0.0);
            let ew = ws.dd_error(k, rule.weights[i], 0.0) / ws.hi[k];
            worst_x = worst_x.max(ex / EPS);
            worst_w = worst_w.max(ew / EPS);
            let tol = f64_tolerance(n);
            assert!(ex <= tol, "n={n} node {i}: error {ex:e}");
            assert!(ew <= tol, "n={n} weight {i}: relative error {ew:e}");
        }
    }
    (worst_x, worst_w)
}

/// Checks the double-double rule against every record; returns the worst node and weight
/// errors in units of u², and the worst weight error as a fraction of its bound.
fn check_dd(records: &[Record]) -> (f64, f64, f64) {
    let tol = 2f64.powi(-100);
    let (mut worst_x, mut worst_w, mut worst_w_tol) = (0.0_f64, 0.0_f64, 0.0_f64);
    for r in records {
        let n = r.input_i64("n") as usize;
        let rule = gauss_legendre::<Dd>(n).unwrap();
        let (xs, ws) = (r.output("nodes"), r.output("weights"));
        for (k, i) in indices(r).into_iter().enumerate() {
            let (x, w) = (rule.nodes[i], rule.weights[i]);
            let ex = xs.dd_error(k, x.hi(), x.lo());
            let ew = ws.dd_error(k, w.hi(), w.lo()) / ws.hi[k];
            assert!(ex <= 4.0 * tol, "n={n} node {i}: error {ex:e}");
            // Weights next to ±1 inherit the n² conditioning of 1 − x².
            let wtol = tol * (4.0 + (n * n) as f64);
            assert!(
                ew <= wtol,
                "n={n} weight {i}: relative error {ew:e} > {wtol:e}"
            );
            worst_x = worst_x.max(ex / U2);
            worst_w = worst_w.max(ew / U2);
            worst_w_tol = worst_w_tol.max(ew / wtol);
        }
    }
    (worst_x, worst_w, worst_w_tol)
}

#[test]
fn f64_rule_matches_arb() {
    let small: Vec<Record> = records(None)
        .into_iter()
        .filter(|r| (r.input_i64("n") as usize) < ASYMPTOTIC_FROM)
        .collect();
    let (x, w) = check_f64(&small);
    eprintln!(
        "f64, n < {ASYMPTOTIC_FROM}: worst node error {x:.3} eps, worst weight error {w:.3} eps"
    );
    let large: Vec<Record> = records(None)
        .into_iter()
        .filter(|r| (r.input_i64("n") as usize) >= ASYMPTOTIC_FROM)
        .collect();
    let (x, w) = check_f64(&large);
    eprintln!(
        "f64, n >= {ASYMPTOTIC_FROM}: worst node error {x:.3} eps, worst weight error {w:.3} eps"
    );
}

#[test]
fn double_double_rule_matches_arb() {
    let (x, w, w_tol) = check_dd(&records(Some(false)));
    eprintln!(
        "DD, n <= {SLOW_N}: worst node error {x:.3} u², worst weight error {w:.3e} u² \
         ({w_tol:.3e} of its bound); u² = 2^-106"
    );
}

#[test]
#[ignore = "computes two 20000-point double-double rules, about 11 s each in release builds"]
fn largest_double_double_rules_match_arb() {
    let recs = records(Some(true));
    assert!(!recs.is_empty());
    let (x, w, w_tol) = check_dd(&recs);
    eprintln!(
        "DD, n > {SLOW_N}: worst node error {x:.3} u², worst weight error {w:.3e} u² \
         ({w_tol:.3e} of its bound); u² = 2^-106"
    );
}

#[test]
fn integrates_even_monomials_exactly() {
    for n in [1usize, 2, 5, 16, 64, 257] {
        let rule = gauss_legendre::<f64>(n).unwrap();
        for k in (0..=(2 * n - 1).min(40)).step_by(2) {
            let s: f64 = rule
                .nodes
                .iter()
                .zip(&rule.weights)
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = 2.0 / (k as f64 + 1.0);
            // Sequential summation of n terms: allow 64 eps of accumulated rounding.
            assert!(
                (s - exact).abs() <= 64.0 * EPS,
                "n={n} k={k}: {s} vs {exact}"
            );
        }
    }
}

/// Checks what every `n`-point rule must satisfy, whatever its accuracy: exact symmetry, a
/// `+0` middle node for odd `n`, each node strictly inside its own interval of Bruns's
/// inequality, and positive weights summing to 2 within the accuracy contract.
///
/// Bruns: with the roots of Pₙ written cos θⱼ, θ₁ < … < θₙ (so the largest root first),
/// (j − ½)π/(n + ½) < θⱼ < jπ/(n + ½) (Szegő, *Orthogonal Polynomials*, §6.21). The
/// intervals are disjoint, so this also proves that the nodes ascend strictly and that the method
/// found every root once. Each root lies at least about 0.7/(n + ½) inside its interval in θ
/// (it is close to Tricomi's (j − ¼)π/(n + ½), the middle of the interval, and θ₁ is close to
/// 2.405/(n + ½)), so at least about 1/n² inside in x: far above the rounding of the `f64`
/// bounds computed here.
fn check_structure(n: usize, rule: &GaussLegendre<f64>) {
    assert_eq!(rule.nodes.len(), n);
    assert_eq!(rule.weights.len(), n);
    for i in 0..n {
        assert_eq!(rule.nodes[i], -rule.nodes[n - 1 - i], "n={n} node {i}");
        assert_eq!(rule.weights[i].to_bits(), rule.weights[n - 1 - i].to_bits());
    }
    if n % 2 == 1 {
        let mid = rule.nodes[n / 2];
        assert_eq!(
            mid.to_bits(),
            0.0_f64.to_bits(),
            "n={n}: middle node {mid:e}"
        );
    }
    let h = n as f64 + 0.5;
    for i in 0..n {
        let j = (n - i) as f64; // ascending index i holds the (n − i)-th largest root
        let (below, above) = ((j * PI / h).cos(), ((j - 0.5) * PI / h).cos());
        let x = rule.nodes[i];
        assert!(
            below < x && x < above,
            "n={n} node {i}: {x} outside ({below}, {above})"
        );
    }
    assert!(rule.weights.iter().all(|&w| w > 0.0));
    // Summed in double-double, so the sum's own error is negligible; each weight is within
    // `f64_tolerance` relative of the exact one, and the exact weights sum to 2.
    let sum = rule.weights.iter().fold(Dd::ZERO, |s, &w| s + w);
    assert!(
        (sum - 2.0).abs().hi() <= 2.0 * f64_tolerance(n),
        "n={n}: weights sum to {sum:?}"
    );
}

#[test]
fn rule_is_symmetric_and_sorted() {
    for n in [1usize, 2, 7, 64] {
        let rule = gauss_legendre::<f64>(n).unwrap();
        for i in 0..n {
            assert_eq!(rule.nodes[i], -rule.nodes[n - 1 - i]);
            assert_eq!(rule.weights[i], rule.weights[n - 1 - i]);
        }
        assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]));
        if n % 2 == 1 {
            assert!(rule.nodes[n / 2] == 0.0 && rule.nodes[n / 2].is_sign_positive());
        }
    }
}

#[test]
fn every_small_rule_converges_to_the_right_roots() {
    for n in 1..=300 {
        check_structure(n, &gauss_legendre::<f64>(n).unwrap());
    }
}

/// Every `f64` rule up to 3000 points: the Newton method up to 100 points and the asymptotic
/// rule above, which takes about 0.2 s for all of them.
#[test]
fn every_rule_up_to_3000_points_converges_to_the_right_roots() {
    let next = AtomicUsize::new(1);
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let n = next.fetch_add(1, Ordering::Relaxed);
                    if n > 3000 {
                        break;
                    }
                    check_structure(n, &gauss_legendre::<f64>(n).unwrap());
                }
            });
        }
    });
}

#[test]
fn large_rules_converge_to_the_right_roots() {
    for n in [
        511usize,
        512,
        1023,
        1024,
        2047,
        2048,
        4999,
        5000,
        20_000,
        MAX_GAUSS_LEGENDRE_POINTS - 1,
        MAX_GAUSS_LEGENDRE_POINTS,
    ] {
        check_structure(n, &gauss_legendre::<f64>(n).unwrap());
    }
}

#[test]
fn double_double_middle_node_is_positive_zero() {
    for n in [1usize, 3, 33, 257] {
        let x = gauss_legendre::<Dd>(n).unwrap().nodes[n / 2];
        assert_eq!((x.hi().to_bits(), x.lo().to_bits()), (0, 0), "n={n}");
    }
}

#[test]
fn zero_points_is_invalid() {
    assert!(matches!(
        gauss_legendre::<f64>(0),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn too_many_points_is_invalid() {
    for n in [
        MAX_GAUSS_LEGENDRE_POINTS + 1,
        usize::MAX / 2 + 1,
        usize::MAX,
    ] {
        assert!(
            matches!(gauss_legendre::<f64>(n), Err(Error::InvalidInput(_))),
            "n={n}"
        );
        assert!(matches!(
            gauss_legendre::<Dual<Dd, 2>>(n),
            Err(Error::InvalidInput(_))
        ));
    }
}

#[test]
fn dual_instantiation_has_zero_derivatives() {
    let rule = gauss_legendre::<Dual<f64, 1>>(4).unwrap();
    assert!(
        rule.nodes
            .iter()
            .chain(&rule.weights)
            .all(|v| v.eps()[0] == 0.0)
    );
}

/// The nodes followed by the weights.
fn values<T: Copy>(rule: &GaussLegendre<T>) -> Vec<T> {
    rule.nodes.iter().chain(&rule.weights).copied().collect()
}

fn dd_bits(x: Dd) -> (u64, u64) {
    (x.hi().to_bits(), x.lo().to_bits())
}

#[test]
fn every_type_gets_the_same_rule_bit_for_bit() {
    for n in [1usize, 2, 3, 4, 7, 8, 64, 257, 1000] {
        let f = gauss_legendre::<f64>(n).unwrap();
        let d = gauss_legendre::<Dd>(n).unwrap();
        let df = gauss_legendre::<Dual<f64, 2>>(n).unwrap();
        let dd = gauss_legendre::<Dual<Dd, 3>>(n).unwrap();
        let (f, d, df, dd) = (values(&f), values(&d), values(&df), values(&dd));
        for i in 0..2 * n {
            // Below the asymptotic range, the f64 rule is the double-double rule rounded to
            // nearest.
            if n < ASYMPTOTIC_FROM {
                assert_eq!(d[i].hi().to_bits(), f[i].to_bits(), "n={n} value {i}");
            }
            assert_eq!(df[i].re().to_bits(), f[i].to_bits(), "n={n} value {i}");
            assert_eq!(dd_bits(dd[i].re()), dd_bits(d[i]), "n={n} value {i}");
            assert!(
                df[i].eps().iter().all(|e| e.to_bits() == 0),
                "n={n} value {i}"
            );
            assert!(
                dd[i].eps().iter().all(|&e| dd_bits(e) == (0, 0)),
                "n={n} value {i}"
            );
        }
    }
}

#[test]
fn caller_buffers_get_the_same_rule() {
    for n in [1usize, 7, 100, 101, 1000] {
        let rule = gauss_legendre::<f64>(n).unwrap();
        let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
        gauss_legendre_into(&mut x, &mut w).unwrap();
        assert_eq!((x, w), (rule.nodes, rule.weights), "n={n}");
    }
    let (mut x, mut w) = (vec![5.0; 4], vec![5.0; 3]);
    assert!(matches!(
        gauss_legendre_into(&mut x, &mut w),
        Err(Error::InvalidInput(_))
    ));
    assert!(x.iter().chain(&w).all(|&v| v == 5.0), "nothing written");
    let mut empty: [f64; 0] = [];
    assert!(matches!(
        gauss_legendre_into(&mut empty, &mut []),
        Err(Error::InvalidInput(_))
    ));
}

/// The bits of the `f64` rules, hashed (FNV-1a over the nodes and weights). The rules are
/// documented to be the same on every platform: CI runs this on Linux, macOS and Windows, so a
/// platform maths library slipping into either method changes the hash there.
#[test]
fn rules_are_the_same_bits_on_every_platform() {
    let hash = |n: usize| {
        let rule = gauss_legendre::<f64>(n).unwrap();
        rule.nodes
            .iter()
            .chain(&rule.weights)
            .fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
                (h ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
            })
    };
    let got = [64, 101, 1000, 4097].map(hash);
    assert_eq!(got, EXPECTED_HASHES);
}

/// Recorded on aarch64 macOS.
const EXPECTED_HASHES: [u64; 4] = [
    0xc771cc0bd3f163c7,
    0xea9a76b75974d1d5,
    0x17987a1f4b364a3b,
    0x89c2776c18210222,
];
