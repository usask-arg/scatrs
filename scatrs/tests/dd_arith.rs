#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::{find, load};
use proptest::prelude::*;
use scatrs::scalar::DoubleDouble as Dd;

/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    f64::from_bits(a.to_bits() + 1) - a
}

/// Checks |op(a, b) - exact| + ref_err <= bound * u² * |exact| over every reference case.
fn check(op: &str, bound: f64, f: impl Fn(Dd, Dd, f64) -> Dd) {
    let recs = load("dd_arith.jsonl");
    let r = find(&recs, op);
    let (a_hi, a_lo) = (r.input_f64s("a_hi"), r.input_f64s("a_lo"));
    let (b_hi, b_lo) = (r.input_f64s("b_hi"), r.input_f64s("b_lo"));
    let z = r.output("z");
    let mut worst = 0.0_f64;
    for i in 0..a_hi.len() {
        let a = Dd::new(a_hi[i], a_lo[i]);
        let b = Dd::new(b_hi[i], b_lo[i]);
        let got = f(a, b, b_hi[i]);
        let exact = Dd::new(z.hi[i], z.lo[i]);
        let rel = ((got - exact).abs().hi() + z.err[i]) / exact.abs().hi();
        worst = worst.max(rel / U2);
        assert!(
            rel <= bound * U2,
            "{op} case {i}: {:.3} u^2 > {bound} u^2",
            rel / U2
        );
    }
    eprintln!("{op}: worst {worst:.3} u^2 (bound {bound} u^2)");
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
}

fn dd(hi: f64, frac: f64) -> Dd {
    Dd::new(hi, frac * ulp(hi))
}

proptest! {
    #[test]
    fn results_are_normalised(ah in -1e30..1e30f64, af in -0.5..0.5f64, bh in -1e30..1e30f64, bf in -0.5..0.5f64) {
        prop_assume!(bh != 0.0);
        let (a, b) = (dd(ah, af), dd(bh, bf));
        for z in [a + b, a - b, a * b, a / b, a + bh, a * bh, a / bh] {
            prop_assert_eq!(z.hi() + z.lo(), z.hi());
        }
    }

    #[test]
    fn addition_commutes(ah in -1e30..1e30f64, af in -0.5..0.5f64, bh in -1e30..1e30f64, bf in -0.5..0.5f64) {
        let (a, b) = (dd(ah, af), dd(bh, bf));
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn rounding_matches_f64_for_f64_inputs(x in -1e15..1e15f64) {
        let d = Dd::from_f64(x);
        prop_assert_eq!(d.floor().hi(), x.floor());
        prop_assert_eq!(d.ceil().hi(), x.ceil());
        prop_assert_eq!(d.trunc().hi(), x.trunc());
        prop_assert_eq!(d.round().hi(), x.round());
    }
}
