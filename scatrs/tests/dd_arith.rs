#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use std::cmp::Ordering;

use common::{find, load};
use proptest::prelude::*;
use scatrs::scalar::DoubleDouble as Dd;

/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    let next = f64::from_bits(a.to_bits() + 1);
    if next.is_infinite() {
        a - f64::from_bits(a.to_bits() - 1) // ulp(MAX) = 2^971
    } else {
        next - a
    }
}

fn p(e: i32) -> f64 {
    2f64.powi(e)
}

/// Checks |op(a, b) - exact| <= bound * u² * scale(b, exact) over every case of the reference
/// record `case`, measuring the error against the three-word reference (`RefValues::dd_error`),
/// so the measured error is the true error to within about 2^-150 |exact|. A zero scale (an
/// exact zero result) requires an exact answer.
fn check_scaled(
    case: &str,
    bound: f64,
    f: impl Fn(Dd, Dd, f64) -> Dd,
    scale: impl Fn(Dd, Dd) -> f64,
) {
    let recs = load("dd_arith.jsonl");
    let r = find(&recs, case);
    let (a_hi, a_lo) = (r.input_f64s("a_hi"), r.input_f64s("a_lo"));
    let (b_hi, b_lo) = (r.input_f64s("b_hi"), r.input_f64s("b_lo"));
    let z = r.output("z");
    let mut worst = 0.0_f64;
    for i in 0..a_hi.len() {
        let a = Dd::new(a_hi[i], a_lo[i]);
        let b = Dd::new(b_hi[i], b_lo[i]);
        let got = f(a, b, b_hi[i]);
        let exact = Dd::new(z.hi[i], z.lo[i]);
        let err = z.dd_error(i, got.hi(), got.lo());
        let s = scale(b, exact);
        let rel = match (s == 0.0, err == 0.0) {
            (false, _) => err / s,
            (true, true) => 0.0,
            (true, false) => f64::INFINITY,
        };
        worst = worst.max(rel / U2);
        assert!(
            rel <= bound * U2,
            "{case} case {i}: {:.3} u^2 > {bound} u^2 ({got:?} vs {exact:?})",
            rel / U2
        );
    }
    eprintln!("{case}: worst {worst:.3} u^2 (bound {bound} u^2)");
}

/// Checks the relative error |op(a, b) - exact| <= bound * u² * |exact|.
fn check(case: &str, bound: f64, f: impl Fn(Dd, Dd, f64) -> Dd) {
    check_scaled(case, bound, f, |_, exact| exact.abs().hi())
}

#[test]
fn add_within_3u2() {
    check("add", 3.0, |a, b, _| a + b)
}

#[test]
fn sub_within_3u2() {
    check("sub", 3.0, |a, b, _| a - b)
}

#[test]
fn mul_within_4u2() {
    check("mul", 4.0, |a, b, _| a * b)
}

#[test]
fn div_within_10u2() {
    check("div", 10.0, |a, b, _| a / b)
}

#[test]
fn div_by_huge_within_10u2() {
    // |b| in [2^950, MAX], where 1/b.hi is subnormal unless div_dd rescales.
    check("div_big_y", 10.0, |a, b, _| a / b)
}

#[test]
fn add_f64_within_2u2() {
    check("add_f64", 2.0, |a, _, b| a + b)
}

#[test]
fn mul_f64_within_2u2() {
    check("mul_f64", 2.0, |a, _, b| a * b)
}

#[test]
fn div_f64_within_3u2() {
    check("div_f64", 3.0, |a, _, b| a / b)
}

#[test]
fn rem_within_2u2_of_divisor() {
    // `%` is within 2u^2 |y| of the true remainder (see `Rem for DoubleDouble`).
    check_scaled("rem", 2.0, |a, b, _| a % b, |b, _| b.abs().hi());
    let recs = load("dd_arith.jsonl");
    let r = find(&recs, "rem");
    let (a_hi, a_lo) = (r.input_f64s("a_hi"), r.input_f64s("a_lo"));
    let (b_hi, b_lo) = (r.input_f64s("b_hi"), r.input_f64s("b_lo"));
    for i in 0..a_hi.len() {
        let (a, b) = (Dd::new(a_hi[i], a_lo[i]), Dd::new(b_hi[i], b_lo[i]));
        let got = a % b;
        assert_eq!(
            got.hi().is_sign_negative(),
            a.hi().is_sign_negative(),
            "case {i}"
        );
        assert!(got.abs() < b.abs(), "case {i}: {got:?} vs {b:?}");
    }
}

#[test]
fn special_values() {
    let inf = Dd::INFINITY;
    assert_eq!((inf + 1.0).hi(), f64::INFINITY);
    assert_eq!((inf * Dd::from_f64(2.0)).hi(), f64::INFINITY);
    assert_eq!((Dd::ONE / Dd::ZERO).hi(), f64::INFINITY);
    assert!((Dd::ZERO / Dd::ZERO).hi().is_nan());
    assert!((inf / inf).hi().is_nan());
    assert_eq!((Dd::ONE / inf).hi(), 0.0);
    assert_eq!((Dd::from_f64(1e308) * 10.0).hi(), f64::INFINITY);
    assert_eq!(Dd::NAN.partial_cmp(&Dd::ONE), None);
    assert_eq!((-inf).hi(), f64::NEG_INFINITY);
    // A finite quotient by a subnormal divisor (where 1/b.hi overflows) must not be NaN.
    // `exact` is Arb's split of 1e-300 / 1e-310 (|error| <= 0.2 u^2 relative).
    let q = Dd::from_f64(1e-300) / Dd::from_f64(1e-310);
    let exact = Dd::new(10000000000.00003, 2.8368529045448975e-7);
    assert!(
        ((q - exact) / exact).abs().hi() + 0.2 * U2 <= 10.0 * U2,
        "{q:?}"
    );
    // Like f64: x % ±inf == x for finite x.
    assert_eq!(Dd::from_f64(7.0) % inf, Dd::from_f64(7.0));
    assert_eq!(Dd::from_f64(-7.0) % -inf, Dd::from_f64(-7.0));
    assert!((inf % Dd::ONE).hi().is_nan());
    assert!((Dd::ONE % Dd::ZERO).hi().is_nan());
    assert!((Dd::NAN % Dd::ONE).hi().is_nan());
}

#[test]
fn sums_near_max_do_not_overflow_in_intermediate_steps() {
    // fl(a - MAX) is finite, but 2Sum's intermediate fl(s - a) overflowed and gave NaN.
    let a = 8.228043741193672e307;
    let want = a - f64::MAX; // -9.748887607429486e307
    let exact = Dd::from_f64(-f64::MAX) + Dd::from_f64(a); // operand order that never failed
    assert_eq!(exact.hi(), want);
    assert_eq!(Dd::from_f64(a) - Dd::from_f64(f64::MAX), exact);
    assert_eq!(Dd::from_f64(a) - f64::MAX, exact);
    assert_eq!(a - Dd::from_f64(f64::MAX), exact);
    assert_eq!(Dd::new(a, -f64::MAX), exact);
    assert_eq!(Dd::new(-f64::MAX, a), exact);
    assert_eq!(Dd::from_f64(-a) + Dd::from_f64(f64::MAX), -exact);
    // Halving makes every step overflow-free, and the double-double sum scales exactly.
    let halved = (Dd::from_f64(a * 0.5) - Dd::from_f64(f64::MAX * 0.5)).mul_pow2(2.0);
    assert_eq!(exact, halved);
}

#[test]
fn remainder_near_max_terminates_and_is_exact() {
    // These looped forever: adding the small terms first overflowed the expansion to NaN.
    // Expected values are exact remainders from rational arithmetic (all double-doubles).
    let tenth = Dd::ONE / Dd::from_f64(10.0); // (0.1, -5.551115123125783e-18)
    let cases = [
        (
            Dd::from_f64(f64::MAX),
            tenth,
            Dd::from_f64(1.3877787807814455e-17),
        ),
        (
            Dd::from_f64(f64::MAX),
            Dd::new(1.0 + p(-52), -0.99 * p(-53)),
            Dd::new(0.1236719002634557, 1.1200685685205952e-18),
        ),
        (
            Dd::new(f64::MAX, 1.9 * p(969)),
            Dd::new(f64::MAX, -1.9 * p(969)),
            Dd::from_f64(1.8960482940579837e292),
        ),
    ];
    for (x, y, want) in cases {
        for (x, y, want) in [(x, y, want), (-x, y, -want), (x, -y, want), (-x, -y, -want)] {
            let got = x % y;
            let err = (got - want).abs().hi();
            assert!(
                err <= 2.0 * U2 * y.abs().hi(),
                "{x:?} % {y:?} = {got:?}, want {want:?}"
            );
            assert!(got.abs() < y.abs());
        }
    }
}

#[test]
fn zero_results_have_a_positive_zero_low_word() {
    let zeros = [
        Dd::from_f64(-0.3).ceil(),
        Dd::from_f64(-0.3).round(),
        Dd::from_f64(-0.3).trunc(),
        Dd::from_f64(-4.0) % Dd::from_f64(2.0),
        Dd::from_f64(4.0) % Dd::from_f64(-2.0),
        Dd::from_f64(-0.0).abs(),
        -Dd::ZERO,
        Dd::from_f64(-0.0) * Dd::ONE,
        Dd::ZERO / Dd::from_f64(-3.0),
    ];
    for z in zeros {
        assert_eq!(z.hi(), 0.0);
        assert_eq!(z.lo().to_bits(), 0, "{z:?}");
    }
}

#[test]
fn overflow_gives_exact_infinity() {
    // Each true result exceeds the overflow threshold MAX + 2^970 only through the low words,
    // so the final renormalisation overflows; it must give (±inf, 0), not NaN or (inf, -inf).
    let near_max = Dd::new(f64::MAX, 1.9 * p(969));
    assert_eq!(near_max + Dd::from_f64(0.2 * p(969)), Dd::INFINITY);
    assert_eq!(-near_max - Dd::from_f64(0.2 * p(969)), Dd::NEG_INFINITY);
    assert_eq!(near_max + 0.2 * p(969), Dd::INFINITY);
    assert_eq!(-near_max - 0.2 * p(969), Dd::NEG_INFINITY);
    let one_plus = Dd::new(1.0, 0.9 * p(-53));
    assert_eq!(Dd::from_f64(f64::MAX) * one_plus, Dd::INFINITY);
    assert_eq!(Dd::from_f64(-f64::MAX) * one_plus, Dd::NEG_INFINITY);
    assert_eq!(Dd::new(1.0, p(-60)) * f64::MAX * 2.0, Dd::INFINITY);
    let one_minus = Dd::new(1.0, -p(-54)); // 1 - 2^-54, normalised by the tie to even
    assert_eq!(Dd::new(f64::MAX, 0.95 * p(970)) / one_minus, Dd::INFINITY);
    assert_eq!(
        Dd::new(-f64::MAX, -0.95 * p(970)) / one_minus,
        Dd::NEG_INFINITY
    );
}

#[test]
fn quotients_near_max_are_finite() {
    // fl(fl(MAX / y) * y) overflows for these y; the quotient itself is finite.
    // Expected values are Arb's splits of MAX / y (|error| <= 0.3 u^2 relative).
    let cases = [
        (3.0, 5.992310449541053e307, -3.3264005158911995e291),
        (7.0, 2.5681330498033083e307, -1.4256002210962284e291),
        (1.3, 1.3828408729710121e308, -7.794405942561626e291),
    ];
    for (y, hi, lo) in cases {
        let exact = Dd::new(hi, lo);
        for got in [
            Dd::from_f64(f64::MAX) / y,
            Dd::from_f64(f64::MAX) / Dd::from_f64(y),
        ] {
            let rel = ((got - exact) / exact).abs().hi() + 0.3 * U2;
            assert!(rel <= 10.0 * U2, "MAX / {y}: {got:?}");
        }
        let got = Dd::from_f64(f64::MAX) / y;
        assert!(
            ((got - exact) / exact).abs().hi() + 0.3 * U2 <= 3.0 * U2,
            "MAX / {y}: {got:?}"
        );
    }
}

#[test]
fn zero_signs_and_exact_results_match_f64() {
    // Every result here is exact in f64, so the double-double must equal it bit for bit,
    // including the sign of zero, with a zero low word.
    let values = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];
    type Op = (&'static str, fn(f64, f64) -> f64, fn(Dd, Dd) -> Dd);
    let ops: [Op; 13] = [
        ("dd + dd", |a, b| a + b, |a, b| a + b),
        ("dd - dd", |a, b| a - b, |a, b| a - b),
        ("dd * dd", |a, b| a * b, |a, b| a * b),
        ("dd / dd", |a, b| a / b, |a, b| a / b),
        ("dd % dd", |a, b| a % b, |a, b| a % b),
        ("dd + f64", |a, b| a + b, |a, b| a + b.hi()),
        ("dd - f64", |a, b| a - b, |a, b| a - b.hi()),
        ("dd * f64", |a, b| a * b, |a, b| a * b.hi()),
        ("dd / f64", |a, b| a / b, |a, b| a / b.hi()),
        ("f64 + dd", |a, b| a + b, |a, b| a.hi() + b),
        ("f64 - dd", |a, b| a - b, |a, b| a.hi() - b),
        ("f64 * dd", |a, b| a * b, |a, b| a.hi() * b),
        ("f64 / dd", |a, b| a / b, |a, b| a.hi() / b),
    ];
    for (name, f64_op, dd_op) in ops {
        for a in values {
            for b in values {
                let want = f64_op(a, b);
                let got = dd_op(Dd::from_f64(a), Dd::from_f64(b));
                if want.is_nan() {
                    assert!(got.hi().is_nan(), "{a:?} {name} {b:?}");
                } else {
                    assert_eq!(got.hi().to_bits(), want.to_bits(), "{a:?} {name} {b:?}");
                    assert_eq!(got.lo(), 0.0, "{a:?} {name} {b:?}");
                    if want == 0.0 {
                        // A zero result is (±0, +0).
                        assert_eq!(got.lo().to_bits(), 0, "{a:?} {name} {b:?}");
                    }
                }
            }
        }
    }
    let x = Dd::new(1.0, p(-60));
    assert!(!(x - x).hi().is_sign_negative()); // exact cancellation gives +0, as in f64
    assert!((-x + x).hi() == 0.0 && !(-x + x).hi().is_sign_negative());
    assert!((Dd::from_f64(-0.0) * x).hi().is_sign_negative());
    assert!((Dd::from_f64(-0.0) / x).hi().is_sign_negative());
    assert!((Dd::from_f64(-0.0) % x).hi().is_sign_negative());
    assert!((Dd::from_f64(-0.0).ldexp(10)).hi().is_sign_negative());
}

#[test]
fn rounding_functions() {
    let big = Dd::new(9007199254740992.0, 0.5); // 2^53 + 1/2
    assert_eq!(big.floor(), Dd::from_f64(9007199254740992.0));
    assert_eq!(big.ceil(), Dd::new(9007199254740992.0, 1.0));
    let below_one = Dd::new(1.0, -(2f64.powi(-80)));
    assert_eq!(below_one.floor(), Dd::ZERO);
    assert_eq!(below_one.ceil(), Dd::ONE);
    assert_eq!(below_one.trunc(), Dd::ZERO);
    assert_eq!(Dd::new(-1.0, 2f64.powi(-80)).floor(), Dd::from_f64(-1.0));
    assert_eq!(Dd::from_f64(2.5).round(), Dd::from_f64(3.0));
    assert_eq!(Dd::from_f64(-2.5).round(), Dd::from_f64(-3.0));
    assert_eq!(Dd::new(0.5, -(2f64.powi(-60))).round(), Dd::ZERO);
    assert_eq!(Dd::INFINITY.floor(), Dd::INFINITY);
    assert_eq!(Dd::from_f64(7.5) % Dd::from_f64(2.0), Dd::from_f64(1.5));
    assert_eq!(Dd::from_f64(-7.5) % Dd::from_f64(2.0), Dd::from_f64(-1.5));
    // Signed zeros, as for f64.
    let neg_below_one = -below_one; // -(1 - 2^-80)
    assert!(neg_below_one.ceil().hi().is_sign_negative());
    assert!(neg_below_one.trunc().hi().is_sign_negative());
    assert_eq!(neg_below_one.round(), Dd::from_f64(-1.0));
    assert!(
        Dd::new(-0.5, 2f64.powi(-60))
            .round()
            .hi()
            .is_sign_negative()
    );
    assert!(
        Dd::new(-0.25, 2f64.powi(-60))
            .round()
            .hi()
            .is_sign_negative()
    );
}

#[test]
fn remainder_of_exact_multiples_is_zero() {
    let x = Dd::from_f64(6.625);
    assert_eq!(x % x, Dd::ZERO);
    assert_eq!(Dd::from_f64(13.25) % x, Dd::ZERO);
    assert!((-x % x).hi().is_sign_negative());
    // k * y is exact in double-double for f64 y and integer k < 2^53.
    let ys = [0.1, 1.0 / 3.0, 6.625, 1.1, 3.0e-200, 7.0e200];
    let ks = [
        1.0,
        2.0,
        3.0,
        7.0,
        10.0,
        12345.0,
        2f64.powi(40) + 1.0,
        2f64.powi(53) - 1.0,
    ];
    for y in ys {
        for k in ks {
            let m = Dd::from_f64(y) * k;
            assert_eq!(m % Dd::from_f64(y), Dd::ZERO, "{k} * {y}");
            let r = (-m) % Dd::from_f64(-y);
            assert!(
                r == Dd::ZERO && r.hi().is_sign_negative(),
                "-{k} * {y}: {r:?}"
            );
            // y + ulp(y)/4 spans 55 bits, so k * yd is exact for k < 2^40.
            let yd = Dd::new(y, 0.25 * ulp(y));
            if k < 2f64.powi(40) {
                let m = yd * Dd::from_f64(k);
                assert_eq!(m % yd, Dd::ZERO, "{k} * {yd:?}");
            }
        }
    }
}

#[test]
fn remainder_just_below_divisor_stays_below_it() {
    // The exact remainder is y - 6.162975822039155e-33 (found with exact rational
    // arithmetic); it is not a double-double, and its nearest double-double is y itself.
    let delta = 6.162975822039155e-33;
    let cases = [
        (
            3.1024344525450998,
            4.2425352317579365e-17,
            1.0341448175150332,
            8.815665241420356e-17,
        ),
        (
            4.917979373039168,
            3.63277576954868e-17,
            1.6393264576797228,
            -6.19056157431815e-17,
        ),
        (
            5.027839745694515,
            -3.3840608386463627e-17,
            1.6759465818981718,
            -8.529507110383164e-17,
        ),
    ];
    for (xh, xl, yh, yl) in cases {
        let (x, y) = (Dd::new(xh, xl), Dd::new(yh, yl));
        for (x, y) in [(x, y), (x, -y), (-x, y), (-x, -y)] {
            let r = x % y;
            assert!(r.abs() < y.abs(), "{x:?} % {y:?} = {r:?}");
            assert_eq!(r.hi().is_sign_negative(), x.hi().is_sign_negative());
            let err = ((y.abs() - r.abs()).hi() - delta).abs();
            assert!(
                err <= 2.0 * U2 * y.abs().hi(),
                "{x:?} % {y:?}: error {err:e}"
            );
        }
    }
}

#[test]
fn remainder_with_huge_quotients_matches_f64() {
    assert_eq!(
        Dd::from_f64(1e40) % Dd::from_f64(3.0),
        Dd::from_f64(1e40 % 3.0)
    );
    assert_eq!((Dd::from_f64(1e40) % Dd::from_f64(3.0)).hi(), 1.0);
    assert_eq!(
        Dd::from_f64(f64::MAX) % Dd::from_f64(5e-324),
        Dd::from_f64(f64::MAX % 5e-324)
    );
    assert_eq!(
        Dd::from_f64(-f64::MAX) % Dd::from_f64(0.1),
        Dd::from_f64(-f64::MAX % 0.1)
    );
    assert_eq!(
        Dd::from_f64(1e308) % Dd::from_f64(1e-10),
        Dd::from_f64(1e308 % 1e-10)
    );
}

#[test]
fn ldexp_and_recip() {
    assert_eq!(Dd::new(1.0, 2f64.powi(-60)).ldexp(1500).hi(), f64::INFINITY);
    assert_eq!(Dd::new(1.0, 2f64.powi(-60)).ldexp(1500), Dd::INFINITY); // lo is cleared too
    assert_eq!(
        Dd::new(-1.0, 2f64.powi(-60)).mul_pow2(f64::INFINITY),
        Dd::NEG_INFINITY
    );
    assert_eq!(
        Dd::new(0.75, 2f64.powi(-60)).ldexp(1024).hi(),
        0.75 * 2f64.powi(1023) * 2.0
    );
    assert_eq!(
        Dd::new(1.0, 2f64.powi(-60)).ldexp(-10),
        Dd::new(2f64.powi(-10), 2f64.powi(-70))
    );
    assert_eq!(Dd::from_f64(4.0).recip(), Dd::from_f64(0.25));
    // Extreme exponents are clamped, not looped over.
    assert_eq!(Dd::ONE.ldexp(i32::MAX), Dd::INFINITY);
    assert_eq!(Dd::ONE.ldexp(i32::MIN), Dd::ZERO);
    assert!(Dd::from_f64(-1.0).ldexp(i32::MIN).hi().is_sign_negative());
    assert_eq!(Dd::from_f64(5e-324).ldexp(2097), Dd::from_f64(p(1023)));
    // MAX * 2^-2097 = (2^53 - 1) * 2^-1126 rounds once, to the subnormal 2^-1073.
    assert_eq!(
        Dd::from_f64(f64::MAX).ldexp(-2097),
        Dd::from_f64(f64::from_bits(2))
    );
}

#[test]
fn scaling_into_the_subnormal_range_stays_normalised() {
    // The low word rounded to 2^-1074 = ulp(hi)/2 with the tie away from the odd hi, giving
    // (4.450147717014404e-308, 5e-324) with hi + lo != hi.
    let x = Dd::new(1.0 + f64::EPSILON, 0.4 * f64::EPSILON);
    for y in [x.ldexp(-1021), x.mul_pow2(p(-1021))] {
        assert_eq!(y.hi() + y.lo(), y.hi(), "{y:?} is not normalised");
        assert!((y.ldexp(1021) - x).abs().hi() <= p(-53), "{y:?}"); // one rounding, 2^-1075
    }
    // A zero result is (±0, +0) even when the low word underflows to -0.
    let negative = Dd::new(-1.0, -p(-60));
    for z in [
        negative.ldexp(-1100),
        negative.mul_pow2(p(-1022)).mul_pow2(p(-100)),
    ] {
        assert!(z.hi() == 0.0 && z.hi().is_sign_negative(), "{z:?}");
        assert_eq!(z.lo().to_bits(), 0, "{z:?}");
    }
    let z = Dd::new(1.0, -p(-60)).mul_pow2(0.0);
    assert!(z.hi().to_bits() == 0 && z.lo().to_bits() == 0, "{z:?}");
}

#[test]
fn mul_pow2_falls_back_for_other_factors() {
    let x = Dd::new(1.0, p(-53));
    let y = x.mul_pow2(3.0);
    assert_eq!(y.hi() + y.lo(), y.hi(), "{y:?} is not normalised");
    assert_eq!(y, x * 3.0);
    assert_eq!(x.mul_pow2(0.25), Dd::new(0.25, p(-55)));
    assert_eq!(x.mul_pow2(-2.0), Dd::new(-2.0, -p(-52)));
    assert_eq!(x.mul_pow2(0.0), Dd::ZERO);
    assert!(x.mul_pow2(f64::NAN).hi().is_nan());
    assert_eq!(x.mul_pow2(5e-324), x * 5e-324); // subnormal powers of two take the general path
}

#[test]
fn ordering_uses_the_low_word_on_ties() {
    let above = Dd::new(1.0, p(-60));
    let below = Dd::new(1.0, -p(-60));
    assert!(below < Dd::ONE && Dd::ONE < above);
    assert_eq!(above.partial_cmp(&below), Some(Ordering::Greater));
    assert_eq!(below.partial_cmp(&above), Some(Ordering::Less));
    assert_eq!(above.partial_cmp(&above), Some(Ordering::Equal));
    assert!(-above < -below);
    assert_eq!(Dd::new(1.0, f64::NAN).partial_cmp(&Dd::ONE), None);
}

#[test]
fn mixed_and_assign_operators() {
    let x = Dd::new(3.0, p(-55));
    let y = Dd::new(1.5, -p(-60));
    assert_eq!(2.0 - x, Dd::new(-1.0, -p(-55)));
    assert_eq!(2.0 - x, Dd::from_f64(2.0) - x);
    assert_eq!(2.0 / x, Dd::from_f64(2.0) / x);
    assert_eq!(2.0 + x, x + 2.0);
    assert_eq!(2.0 * x, x * 2.0);
    assert_eq!(x - 2.0, Dd::new(1.0, p(-55)));
    type AssignOp<T> = (fn(&mut Dd, T), fn(Dd, T) -> Dd);
    let ops: [AssignOp<Dd>; 5] = [
        (|z, y| *z += y, |a, b| a + b),
        (|z, y| *z -= y, |a, b| a - b),
        (|z, y| *z *= y, |a, b| a * b),
        (|z, y| *z /= y, |a, b| a / b),
        (|z, y| *z %= y, |a, b| a % b),
    ];
    for (assign, op) in ops {
        let mut z = x;
        assign(&mut z, y);
        assert_eq!(z, op(x, y));
    }
    let ops: [AssignOp<f64>; 4] = [
        (|z, y| *z += y, |a, b| a + b),
        (|z, y| *z -= y, |a, b| a - b),
        (|z, y| *z *= y, |a, b| a * b),
        (|z, y| *z /= y, |a, b| a / b),
    ];
    for (assign, op) in ops {
        let mut z = x;
        assign(&mut z, 0.75);
        assert_eq!(z, op(x, 0.75));
    }
}

fn dd(hi: f64, frac: f64) -> Dd {
    Dd::new(hi, frac * ulp(hi))
}

/// Finite doubles of any magnitude, including subnormals and both zeros.
fn any_finite() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("finite", |x| x.is_finite())
}

/// ±MAX, or within 2^16 ulps of it.
fn near_max() -> impl Strategy<Value = f64> {
    (prop_oneof![Just(0u64), 0u64..1 << 16], any::<bool>()).prop_map(|(k, neg)| {
        let x = f64::from_bits(f64::MAX.to_bits() - k);
        if neg { -x } else { x }
    })
}

/// |x| in [2^1021, MAX] with a random significand.
fn big_hi() -> impl Strategy<Value = f64> {
    (1021u64..=1023, any::<u64>(), any::<bool>()).prop_map(|(e, m, neg)| {
        let x = f64::from_bits(((e + 1023) << 52) | (m & ((1 << 52) - 1)));
        if neg { -x } else { x }
    })
}

/// A low word as a fraction of ulp(hi), often near ±1/2.
fn lo_frac() -> impl Strategy<Value = f64> {
    prop_oneof![
        -0.5..0.5f64,
        0.49..=0.5f64,
        -0.5..=-0.49f64,
        Just(0.5),
        Just(-0.5)
    ]
}

fn finite_dd(hi: impl Strategy<Value = f64>) -> impl Strategy<Value = Dd> {
    (hi, lo_frac()).prop_map(|(hi, f)| {
        let d = dd(hi, f);
        // MAX + ulp/2 overflows; keep the leading word alone rather than rejecting the case.
        if d.hi().is_finite() {
            d
        } else {
            Dd::from_f64(hi)
        }
    })
}

/// Finite double-doubles from the whole range, weighted towards ±MAX and the top binades,
/// with low words often near ±ulp(hi)/2.
fn wide_dd() -> impl Strategy<Value = Dd> {
    finite_dd(prop_oneof![any_finite(), near_max(), big_hi()])
}

/// Finite double-doubles with |hi| in [2^1021, MAX].
fn big_dd() -> impl Strategy<Value = Dd> {
    finite_dd(prop_oneof![near_max(), big_hi()])
}

proptest! {
    #[test]
    fn results_are_normalised(ah in -1e30..1e30f64, af in -0.5..0.5f64, bh in -1e30..1e30f64, bf in -0.5..0.5f64) {
        prop_assume!(bh != 0.0);
        let (a, b) = (dd(ah, af), dd(bh, bf));
        for z in [a + b, a - b, a * b, a / b, a + bh, a * bh, a / bh, a % b] {
            prop_assert_eq!(z.hi() + z.lo(), z.hi());
        }
    }

    #[test]
    fn rounding_matches_f64_for_f64_inputs(
        x in prop_oneof![-1e15..1e15f64, -4.0..4.0f64, Just(0.0), Just(-0.0), any_finite()]
    ) {
        // Bitwise, so the sign of zero results must match too.
        let d = Dd::from_f64(x);
        for (got, want) in [
            (d.floor(), x.floor()),
            (d.ceil(), x.ceil()),
            (d.trunc(), x.trunc()),
            (d.round(), x.round()),
        ] {
            prop_assert_eq!(got.hi().to_bits(), want.to_bits(), "{:?}", x);
            prop_assert_eq!(got.lo().to_bits(), 0); // +0, not -0
        }
    }

    #[test]
    fn remainder_matches_f64_for_f64_inputs(x in any_finite(), y in any_finite()) {
        // fmod is exact, so the double-double remainder must equal it bit for bit.
        prop_assume!(y != 0.0);
        let got = Dd::from_f64(x) % Dd::from_f64(y);
        prop_assert_eq!(got.hi().to_bits(), (x % y).to_bits(), "{:?} % {:?}", x, y);
        prop_assert_eq!(got.lo().to_bits(), 0); // +0, not -0
    }

    #[test]
    fn remainder_is_smaller_than_divisor_with_sign_of_dividend(
        ah in -1e30..1e30f64, af in -0.5..0.5f64, bh in -1e3..1e3f64, bf in -0.5..0.5f64
    ) {
        prop_assume!(bh != 0.0);
        let (a, b) = (dd(ah, af), dd(bh, bf));
        let r = a % b;
        prop_assert!(r.abs() < b.abs(), "{:?} % {:?} = {:?}", a, b, r);
        prop_assert_eq!(r.hi().is_sign_negative(), a.hi().is_sign_negative());
    }
}

proptest! {
    // Whole-range inputs near ±MAX hit rare overflow paths; use more cases than the default.
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn results_are_normalised_over_the_whole_range(a in wide_dd(), b in wide_dd()) {
        prop_assume!(b.hi() != 0.0);
        let bh = b.hi();
        for z in [a + b, a - b, a * b, a / b, a + bh, a - bh, a * bh, a / bh, a % b] {
            prop_assert!(!z.hi().is_nan() && !z.lo().is_nan(), "{:?} {:?} -> {:?}", a, b, z);
            prop_assert_eq!(z.hi() + z.lo(), z.hi());
        }
    }

    #[test]
    fn addition_commutes(a in wide_dd(), b in wide_dd()) {
        let s = a + b;
        prop_assert!(!s.hi().is_nan(), "{:?} + {:?} = {:?}", a, b, s);
        prop_assert_eq!(s, b + a);
        prop_assert_eq!(a + b.hi(), b.hi() + a);
    }

    #[test]
    fn sums_near_max_match_halved_sums(a in big_dd(), b in big_dd()) {
        // Halving is exact here and removes every intermediate overflow; double-double sums
        // commute with exact scaling. (When fl(a.hi + b.hi) itself overflows, the sum may
        // overflow early by design, so those cases are skipped.)
        let h = |x: Dd| x.mul_pow2(0.5);
        let twice = |x: Dd| x.mul_pow2(2.0);
        if (a.hi() + b.hi()).is_finite() {
            prop_assert_eq!(a + b, twice(h(a) + h(b)));
            prop_assert_eq!(a + b.hi(), twice(h(a) + 0.5 * b.hi()));
            prop_assert_eq!(Dd::new(a.hi(), b.hi()), twice(Dd::new(0.5 * a.hi(), 0.5 * b.hi())));
        }
        if (a.hi() - b.hi()).is_finite() {
            prop_assert_eq!(a - b, twice(h(a) - h(b)));
            prop_assert_eq!(a - b.hi(), twice(h(a) - 0.5 * b.hi()));
        }
    }

    #[test]
    fn scaling_down_to_tiny_magnitudes_stays_normalised(
        h in 1.0..2.0f64, neg in any::<bool>(), f in lo_frac(), e in -1100..=-900i32,
    ) {
        let x = dd(if neg { -h } else { h }, f);
        let mut ys = vec![x.ldexp(e)];
        if e >= -1022 {
            ys.push(x.mul_pow2(p(e)));
        }
        for y in ys {
            prop_assert_eq!(y.hi() + y.lo(), y.hi(), "{:?} * 2^{} = {:?}", x, e, y);
            if y.hi() == 0.0 {
                prop_assert_eq!(y.lo().to_bits(), 0);
                prop_assert_eq!(y.hi().is_sign_negative(), neg);
            }
            // Scaling back up is exact; ldexp rounds at most three times by <= 2^-1075 each.
            let err = (y.ldexp(-e) - x).abs().hi();
            prop_assert!(err <= p(-1073 - e), "{:?} * 2^{} = {:?}", x, e, y);
        }
    }

    #[test]
    fn remainder_terminates_and_is_reduced_over_the_whole_range(x in wide_dd(), y in wide_dd()) {
        prop_assume!(y.hi() != 0.0);
        let r = x % y;
        prop_assert!(!r.hi().is_nan(), "{:?} % {:?} = {:?}", x, y, r);
        prop_assert!(r.abs() < y.abs(), "{:?} % {:?} = {:?}", x, y, r);
        prop_assert_eq!(r.hi().is_sign_negative(), x.hi().is_sign_negative());
        prop_assert_eq!(r.hi() + r.lo(), r.hi());
    }
}
