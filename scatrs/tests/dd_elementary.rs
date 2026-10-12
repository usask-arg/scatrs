#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use std::f64::consts::LOG2_E;

use common::{Record, find, load};
use proptest::prelude::*;
use scatrs::scalar::DoubleDouble as Dd;

/// 2^-100.
const TOL: f64 = 7.888_609_052_210_118e-31;
/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;
/// 1024·fl(ln 2), the largest double whose exponential is finite (just below ln MAX).
const LN_MAX: f64 = 709.782712893384;
/// The smallest positive subnormal, 2^-1074.
const TINY: f64 = 5e-324;

fn dd_input(r: &Record, prefix: &str) -> Vec<Dd> {
    let hi = r.input_f64s(&format!("{prefix}_hi"));
    let lo = r.input_f64s(&format!("{prefix}_lo"));
    hi.iter().zip(&lo).map(|(&h, &l)| Dd::new(h, l)).collect()
}

/// Asserts |got − ref| / |ref| <= tol(i) for every case of `case` and of its edge-case
/// record `{case}_edges`, measuring the error against the three-word reference.
fn check(case: &str, got: impl Fn(&Record, usize) -> Dd, tol: impl Fn(&Record, usize) -> f64) {
    let recs = load("dd_elementary.jsonl");
    for name in [case.to_owned(), format!("{case}_edges")] {
        let r = find(&recs, &name);
        let v = r.output("value");
        let mut worst = 0.0_f64;
        for i in 0..v.hi.len() {
            let exact = Dd::new(v.hi[i], v.lo[i]);
            let g = got(r, i);
            let rel = v.dd_error(i, g.hi(), g.lo()) / exact.abs().hi();
            let t = tol(r, i);
            worst = worst.max(rel / t);
            assert!(
                rel <= t,
                "{name} case {i}: got {g:?}, reference {exact:?}, rel {rel:e} > {t:e}"
            );
        }
        eprintln!("{name}: worst error = {worst:.3} x tolerance");
    }
}

fn unary(case: &str, f: fn(Dd) -> Dd, tol: f64) {
    check(case, |r, i| f(dd_input(r, "x")[i]), |_, _| tol);
}

#[test]
fn sqrt() {
    unary("sqrt", Dd::sqrt, TOL)
}
#[test]
fn exp() {
    unary("exp", Dd::exp, TOL)
}
#[test]
fn exp_m1() {
    unary("exp_m1", Dd::exp_m1, TOL)
}
#[test]
fn ln() {
    unary("ln", Dd::ln, TOL)
}
#[test]
fn ln_1p() {
    unary("ln_1p", Dd::ln_1p, TOL)
}
#[test]
fn log2() {
    unary("log2", Dd::log2, TOL)
}
#[test]
fn log10() {
    unary("log10", Dd::log10, TOL)
}
#[test]
fn cbrt() {
    unary("cbrt", Dd::cbrt, TOL)
}
#[test]
fn sinh() {
    unary("sinh", Dd::sinh, 2.0 * TOL)
}
#[test]
fn cosh() {
    unary("cosh", Dd::cosh, 2.0 * TOL)
}
#[test]
fn tanh() {
    unary("tanh", Dd::tanh, TOL)
}
#[test]
fn asinh() {
    unary("asinh", Dd::asinh, TOL)
}
#[test]
fn acosh() {
    unary("acosh", Dd::acosh, TOL)
}
#[test]
fn atanh() {
    unary("atanh", Dd::atanh, TOL)
}

#[test]
fn exp2() {
    // exp2(x) = exp(x ln 2): rounding x·ln 2 costs a factor |x| (conditioning).
    check(
        "exp2",
        |r, i| dd_input(r, "x")[i].exp2(),
        |r, i| TOL * dd_input(r, "x")[i].hi().abs().max(1.0),
    );
}

#[test]
fn hypot() {
    check(
        "hypot",
        |r, i| dd_input(r, "x")[i].hypot(dd_input(r, "y")[i]),
        |_, _| TOL,
    );
}

#[test]
fn powf() {
    // exp(y ln x): the error in y·ln x is amplified by |y ln x| (conditioning). Negative
    // bases (edge cases with integer y) use ln |x|.
    check(
        "powf",
        |r, i| dd_input(r, "x")[i].powf(dd_input(r, "y")[i]),
        |r, i| {
            let (x, y) = (dd_input(r, "x")[i].hi(), dd_input(r, "y")[i].hi());
            2.0 * TOL * (y * x.abs().ln()).abs().max(1.0)
        },
    );
}

#[test]
fn powi() {
    // Binary powering: each squaring doubles the accumulated relative error.
    check(
        "powi",
        |r, i| dd_input(r, "x")[i].powi(r.input_i64s("n")[i] as i32),
        |r, i| 16.0 * (r.input_i64s("n")[i].unsigned_abs() as f64 + 1.0) * U2,
    );
}

#[test]
fn special_values() {
    let d = Dd::from_f64;
    assert!(d(-1.0).sqrt().hi().is_nan());
    assert_eq!(d(0.0).sqrt(), d(0.0));
    assert_eq!(d(f64::INFINITY).sqrt().hi(), f64::INFINITY);
    assert_eq!(d(1000.0).exp().hi(), f64::INFINITY);
    assert_eq!(d(-1000.0).exp(), Dd::ZERO);
    assert!(Dd::NAN.exp().hi().is_nan());
    assert_eq!(d(0.0).exp(), Dd::ONE);
    assert_eq!(d(0.0).ln().hi(), f64::NEG_INFINITY);
    assert!(d(-1.0).ln().hi().is_nan());
    assert_eq!(d(1.0).ln(), Dd::ZERO);
    assert_eq!(d(0.0).exp_m1(), d(0.0));
    assert_eq!(d(f64::NEG_INFINITY).exp_m1(), d(-1.0));
    assert_eq!(d(-1.0).ln_1p().hi(), f64::NEG_INFINITY);
    assert!(d(-2.0).ln_1p().hi().is_nan());
    assert_eq!(d(1.0).atanh().hi(), f64::INFINITY);
    assert!(d(0.5).acosh().hi().is_nan());
    assert_eq!(d(-8.0).cbrt(), d(-2.0));
    assert_eq!(d(3.0).hypot(d(4.0)), d(5.0));
    assert_eq!(d(0.0).powf(d(2.0)), Dd::ZERO);
    assert_eq!(d(-2.0).powf(d(3.0)), d(-8.0));
    assert!(d(-2.0).powf(d(0.5)).hi().is_nan());
    assert_eq!(d(7.0).powi(0), Dd::ONE);
    assert_eq!(d(2.0).powi(-2), d(0.25));
    assert_eq!(d(100.0).log(d(10.0)).hi(), 2.0);
}

// ---- Agreement with `f64` at special values and range ends ------------------------------

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == f64::MAX {
        return a - f64::from_bits(a.to_bits() - 1);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// 2^n for -1074 <= n <= 1023, including subnormal powers.
fn two_pow(n: i32) -> f64 {
    if n >= -1022 {
        f64::from_bits(((n + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (n + 1074))
    }
}

fn next_up(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    f64::from_bits(x.to_bits() + 1)
}

fn next_down(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    f64::from_bits(x.to_bits() - 1)
}

/// Checks a double-double result against the `f64` result for the same input: NaN exactly
/// when `want` is NaN; bitwise equal (with a `+0` low word) when `want` is zero or infinite;
/// otherwise within 2^-46 relative, or 2^-1072 absolute for subnormal results.
///
/// This catches wrong special values, signs, overflow and garbage; accuracy is tested against
/// Arb above. The 2^-46 (tens of ulps) allows for platform libm errors of a few ulps (all of
/// these agree to 2^-50 on macOS).
fn agrees(what: &str, got: Dd, want: f64) -> Result<(), String> {
    let ok = if want.is_nan() {
        got.hi().is_nan()
    } else if want == 0.0 || want.is_infinite() {
        got.hi().to_bits() == want.to_bits() && got.lo().to_bits() == 0
    } else {
        got.hi().is_finite()
            && (got.hi() - want).abs() <= 2f64.powi(-46) * want.abs() + 2f64.powi(-1072)
    };
    if ok {
        Ok(())
    } else {
        Err(format!("{what}: got {got:?}, f64 gives {want:?}"))
    }
}

type Unary = (&'static str, fn(Dd) -> Dd, fn(f64) -> f64);

/// Functions whose `f64` counterparts are accurate over the whole range.
const UNARY: [Unary; 12] = [
    ("sqrt", Dd::sqrt, f64::sqrt),
    ("exp", Dd::exp, f64::exp),
    ("exp_m1", Dd::exp_m1, f64::exp_m1),
    ("ln", Dd::ln, f64::ln),
    ("ln_1p", Dd::ln_1p, f64::ln_1p),
    ("exp2", Dd::exp2, f64::exp2),
    ("log2", Dd::log2, f64::log2),
    ("log10", Dd::log10, f64::log10),
    ("cbrt", Dd::cbrt, f64::cbrt),
    ("sinh", Dd::sinh, f64::sinh),
    ("cosh", Dd::cosh, f64::cosh),
    ("tanh", Dd::tanh, f64::tanh),
];

/// The inverse hyperbolic functions. `f64::asinh` and `f64::acosh` overflow to infinity
/// for huge inputs, so these are compared with `f64` only at special values.
const INVERSE_HYPERBOLIC: [Unary; 3] = [
    ("asinh", Dd::asinh, f64::asinh),
    ("acosh", Dd::acosh, f64::acosh),
    ("atanh", Dd::atanh, f64::atanh),
];

const SPECIAL: [f64; 13] = [
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    -0.5,
    TINY,
    -TINY,
];

#[test]
fn special_values_and_signed_zeros_match_f64() {
    // For example sinh(-0) = -0, exp_m1(-inf) = -1, ln_1p(-1) = -inf, atanh(-1) = -inf.
    for (name, dd_fn, f64_fn) in UNARY.iter().chain(&INVERSE_HYPERBOLIC) {
        for x in SPECIAL {
            agrees(&format!("{name}({x:?})"), dd_fn(Dd::from_f64(x)), f64_fn(x)).unwrap();
        }
    }
}

#[test]
fn range_ends_match_f64() {
    let ends = [
        f64::MAX,
        -f64::MAX,
        next_down(f64::MAX),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1e-310,
        -1e-310,
        3.0 * TINY,
        1e300,
        -1e300,
        1e-300,
        -1e-300,
        700.0,
        -700.0,
        711.0,
        -711.0,
        750.0,
        -750.0,
        1100.0,
        -1100.0,
    ];
    for (name, dd_fn, f64_fn) in UNARY {
        for x in ends {
            agrees(&format!("{name}({x:?})"), dd_fn(Dd::from_f64(x)), f64_fn(x)).unwrap();
        }
    }
}

#[test]
fn binary_special_values_match_f64() {
    let xs = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,
        TINY,
    ];
    let ys = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e300,
        -1e300,
        1025.0,
        -1075.0,
    ];
    let d = Dd::from_f64;
    for x in xs {
        for y in ys {
            let ctx = format!("({x:?}, {y:?})");
            agrees(&format!("powf{ctx}"), d(x).powf(d(y)), x.powf(y)).unwrap();
            agrees(&format!("hypot{ctx}"), d(x).hypot(d(y)), x.hypot(y)).unwrap();
            agrees(&format!("log{ctx}"), d(x).log(d(y)), x.log(y)).unwrap();
        }
        for n in [
            0,
            1,
            -1,
            2,
            -2,
            3,
            -3,
            1023,
            -1023,
            1024,
            -1074,
            -1075,
            i32::MAX,
            i32::MIN,
        ] {
            // f64::powi multiplies repeatedly (and inverts x^|n|), so it underflows or
            // overflows early, e.g. 2.0.powi(-1074) == 0; f64::powf is the reference.
            let want = x.powf(f64::from(n));
            agrees(&format!("powi({x:?}, {n})"), d(x).powi(n), want).unwrap();
        }
    }
}

#[test]
fn powf_matches_f64_signs_for_negative_bases() {
    // Integer exponents (beyond i32 too) give the sign of the parity; others give NaN.
    let d = Dd::from_f64;
    for (x, y) in [
        (-1.0000001, 1e10 + 1.0),
        (-1.0000001, 1e10),
        (-1.0000001, -(1e10 + 1.0)),
        (-0.9999999, 3e9 + 1.0),
        (-1.5, 1001.0),
        (-1.5, -1001.0),
        (-1.5, 1000.0),
        (-2.0, 2f64.powi(60)),
        (-1.0, 9007199254740991.0),
    ] {
        agrees(&format!("powf({x:?}, {y:?})"), d(x).powf(d(y)), x.powf(y)).unwrap();
    }
    // A double-double integer exponent whose oddness is in its low word.
    let odd = Dd::new(2f64.powi(60), 1.0);
    assert!(Dd::from_f64(-1.0).powf(odd) == Dd::from_f64(-1.0));
    assert!(Dd::from_f64(-1.0).powf(Dd::new(2f64.powi(60), 2.0)) == Dd::ONE);
    assert!(
        Dd::from_f64(-1.0)
            .powf(Dd::new(2f64.powi(60), 0.5))
            .hi()
            .is_nan()
    );
}

#[test]
fn overflow_and_underflow_thresholds() {
    let d = Dd::from_f64;
    // exp overflows exactly when x >= ln(MAX + ulp/2) = 1024 ln 2 - 5.55e-17 = LN_MAX + 2.3692e-14.
    assert!(d(LN_MAX).exp().hi().is_finite());
    assert!(Dd::new(LN_MAX, 2.36e-14).exp().hi().is_finite());
    assert_eq!(Dd::new(LN_MAX, 2.38e-14).exp(), Dd::INFINITY);
    assert_eq!(d(next_up(LN_MAX)).exp(), Dd::INFINITY);
    assert_eq!(Dd::new(LN_MAX, 2.38e-14).exp_m1(), Dd::INFINITY);
    // Subnormal and vanishing results agree with f64 (no NaN, no garbage).
    for x in [
        -708.4,
        -709.0,
        -720.0,
        -740.0,
        -745.0,
        -745.1332191019411,
        -745.1332191019412,
        -746.0,
        -1e300,
    ] {
        agrees(&format!("exp({x})"), d(x).exp(), x.exp()).unwrap();
        agrees(
            &format!("exp2({x})"),
            d(x * LOG2_E).exp2(),
            (x * LOG2_E).exp2(),
        )
        .unwrap();
    }
    // exp2 of an integer is exact (f64::powi(2.0, n) is not: it gives 0 for n < -1023).
    for n in -1074..=1023 {
        assert_eq!(d(f64::from(n)).exp2(), d(two_pow(n)), "exp2({n})");
    }
    assert_eq!(d(1024.0).exp2(), Dd::INFINITY);
    assert!(d(next_down(1024.0)).exp2().hi().is_finite());
    assert_eq!(Dd::new(1024.0, -(2f64.powi(-60))).exp2(), Dd::INFINITY); // above MAX + ulp/2
    assert!(Dd::new(1024.0, -(2f64.powi(-50))).exp2().hi().is_finite());
    assert_eq!(d(-1075.0).exp2(), Dd::ZERO);
    // sinh and cosh stay finite up to ln(2 MAX) = 710.4758600739439...
    for x in [709.9, 710.0, 710.4, 710.475] {
        for s in [1.0, -1.0] {
            agrees(&format!("sinh({})", s * x), d(s * x).sinh(), (s * x).sinh()).unwrap();
            agrees(&format!("cosh({})", s * x), d(s * x).cosh(), (s * x).cosh()).unwrap();
            assert!(d(s * x).sinh().hi().is_finite() && d(s * x).cosh().hi().is_finite());
        }
    }
    assert_eq!(d(710.48).sinh(), Dd::INFINITY);
    assert_eq!(d(-710.48).sinh(), Dd::NEG_INFINITY);
    assert_eq!(d(-710.48).cosh(), Dd::INFINITY);
    // powf and powi overflow to infinity and underflow to zero.
    assert_eq!(d(2.0).powf(d(1024.0)), Dd::INFINITY);
    assert!(d(2.0).powf(d(1023.99)).hi().is_finite());
    assert_eq!(d(0.5).powf(d(1100.0)), Dd::ZERO);
    assert_eq!(d(2.0).powi(1024), Dd::INFINITY);
    assert_eq!(d(-2.0).powi(1025), Dd::NEG_INFINITY);
    assert_eq!(d(2.0).powi(-1074), d(TINY));
    assert_eq!(d(2.0).powi(-1075), Dd::ZERO);
    assert_eq!(d(10.0).powi(309), Dd::INFINITY);
    assert_eq!(d(0.1).powi(-309), Dd::INFINITY);
    assert_eq!(d(10.0).powi(-400), Dd::ZERO);
    // Intermediate powers far outside the range must not spoil an in-range result.
    assert_eq!(d(0.5).powi(-1023), d(two_pow(1023)));
    assert_eq!(d(two_pow(-1000)).powi(-1), d(two_pow(1000)));
    assert_eq!(d(TINY).powi(-1), Dd::INFINITY);
    assert_eq!(d(4.0).powi(i32::MIN), Dd::ZERO);
    assert_eq!(d(0.25).powi(i32::MAX), Dd::ZERO);
    assert_eq!(d(-1.0).powi(i32::MIN), Dd::ONE);
    assert_eq!(d(-1.0).powi(i32::MAX), d(-1.0));
    // tanh saturates; the inverse functions are finite at the range ends.
    assert_eq!(d(41.0).tanh(), Dd::ONE);
    assert_eq!(d(-f64::MAX).tanh(), d(-1.0));
    for x in [f64::MAX, 1e300, 1e-300, TINY] {
        assert!(d(x).asinh().hi().is_finite() && d(-x).asinh().hi() < 0.0);
        assert!(d(x).ln().hi().is_finite() && d(x).ln_1p().hi().is_finite());
    }
    assert!(d(f64::MAX).acosh().hi().is_finite());
    assert!(Dd::new(-1.0, 2f64.powi(-100)).atanh().hi().is_finite());
    assert!(Dd::new(1.0, 2f64.powi(-100)).atanh().hi().is_nan()); // |x| > 1
    assert!(Dd::new(1.0, -(2f64.powi(-100))).acosh().hi().is_nan()); // x < 1
    assert!(Dd::new(-1.0, -(2f64.powi(-100))).ln_1p().hi().is_nan()); // x < -1
}

#[test]
fn zero_results_keep_their_sign_with_a_positive_zero_low_word() {
    // Functions that are odd near zero map ±0 to (±0, +0); exact zeros at 1 are +0.
    let neg_zero = Dd::from_f64(-0.0);
    for (name, f) in [
        ("sqrt", Dd::sqrt as fn(Dd) -> Dd),
        ("exp_m1", Dd::exp_m1),
        ("ln_1p", Dd::ln_1p),
        ("cbrt", Dd::cbrt),
        ("sinh", Dd::sinh),
        ("tanh", Dd::tanh),
        ("asinh", Dd::asinh),
        ("atanh", Dd::atanh),
    ] {
        let z = f(neg_zero);
        assert!(
            z.hi() == 0.0 && z.hi().is_sign_negative() && z.lo().to_bits() == 0,
            "{name}(-0) = {z:?}"
        );
        let z = f(Dd::ZERO);
        assert!(
            z.hi().to_bits() == 0 && z.lo().to_bits() == 0,
            "{name}(+0) = {z:?}"
        );
    }
    for z in [
        Dd::ONE.acosh(),
        Dd::ONE.ln(),
        Dd::ONE.log2(),
        Dd::ONE.log10(),
    ] {
        assert!(z.hi().to_bits() == 0 && z.lo().to_bits() == 0, "{z:?}");
    }
    // Underflow to zero also gives (±0, +0).
    for z in [
        Dd::new(-745.2, 1e-14).exp(),
        Dd::new(-1080.0, 1e-13).exp2(),
        Dd::new(3.0, 1e-16).powi(-700),
        Dd::new(-3.0, 1e-16).powi(-701),
        Dd::new(0.5, 1e-17).powf(Dd::from_f64(1080.0)),
    ] {
        assert!(z.hi() == 0.0 && z.lo().to_bits() == 0, "{z:?}");
    }
    assert!(Dd::new(-3.0, 1e-16).powi(-701).hi().is_sign_negative());
}

// ---- Whole-range properties ---------------------------------------------------------------

/// Finite doubles of any magnitude, including subnormals and both zeros.
fn any_finite() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("finite", |x| x.is_finite())
}

/// A normalised double-double with leading word near `hi` and a low word of `frac`·ulp(hi).
fn with_lo(hi: f64, frac: f64) -> Dd {
    let d = Dd::new(hi, frac * ulp(hi));
    if d.hi().is_finite() {
        d
    } else {
        Dd::from_f64(hi)
    }
}

/// Finite double-doubles from the whole range, with low words often near ±ulp(hi)/2.
fn wide_dd() -> impl Strategy<Value = Dd> {
    let frac = prop_oneof![-0.5..0.5f64, Just(0.0), Just(0.49), Just(-0.49)];
    (any_finite(), frac).prop_map(|(h, f)| with_lo(h, f))
}

/// Double-doubles in (-1, 1), including values within 2^-60 .. 2^-1000 of ±1.
fn unit_interval_dd() -> impl Strategy<Value = Dd> {
    prop_oneof![
        (-1.0..1.0f64, -0.5..0.5f64).prop_map(|(h, f)| with_lo(h, f)),
        (60..1000i32, any::<bool>()).prop_map(|(k, neg)| {
            let x = Dd::new(1.0, -(2f64.powi(-k)));
            if neg { -x } else { x }
        }),
    ]
}

/// Rescales `x` by an exact power of two so that its magnitude lies in [2^-500, 2^500] and
/// products of a few such numbers stay well inside the full-precision range.
fn scale_of(x: Dd) -> i32 {
    let e = x.hi().abs().log2().floor() as i32;
    if e < -500 {
        600
    } else if e > 500 {
        -600
    } else {
        0
    }
}

fn rel_err(got: Dd, want: Dd) -> f64 {
    ((got - want) / want).abs().hi()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn f64_inputs_agree_with_f64(x in prop_oneof![any_finite(), -800.0..800.0f64, 0.0..2.0f64]) {
        for (name, dd_fn, f64_fn) in UNARY {
            let got = dd_fn(Dd::from_f64(x));
            agrees(&format!("{name}({x:?})"), got, f64_fn(x)).map_err(TestCaseError::fail)?;
            prop_assert!(
                got.hi().is_nan() || got.hi() + got.lo() == got.hi(),
                "{}({:?}) = {:?} is not normalised", name, x, got
            );
        }
    }

    #[test]
    fn binary_f64_inputs_agree_with_f64(
        x in prop_oneof![any_finite(), -10.0..10.0f64],
        y in prop_oneof![any_finite(), (-2000..2000i32).prop_map(f64::from), -50.0..50.0f64],
        n in prop_oneof![-2000..2000i32, any::<i32>()],
    ) {
        let d = Dd::from_f64;
        agrees(&format!("powf({x:?}, {y:?})"), d(x).powf(d(y)), x.powf(y))
            .map_err(TestCaseError::fail)?;
        agrees(&format!("hypot({x:?}, {y:?})"), d(x).hypot(d(y)), x.hypot(y))
            .map_err(TestCaseError::fail)?;
        // f64::powi multiplies repeatedly and loses accuracy for large n; compare with powf.
        let got = d(x).powi(n);
        agrees(&format!("powi({x:?}, {n})"), got, x.powf(f64::from(n)))
            .map_err(TestCaseError::fail)?;
    }

    #[test]
    fn sqrt_and_cbrt_invert_squares_and_cubes(x in wide_dd()) {
        let k = scale_of(x);
        let s = x.abs().sqrt();
        let c = x.cbrt();
        prop_assert!(s.hi() >= 0.0 && c.hi().is_finite(), "{:?}: {:?} {:?}", x, s, c);
        prop_assert_eq!(c.hi().is_sign_negative(), x.hi().is_sign_negative());
        if x.hi() != 0.0 {
            // (s·2^(k/2))² = |x|·2^k and (c·2^(k/3))³ = x·2^k, all exactly scaled.
            let (s2, c3) = (s.ldexp(k / 2), c.ldexp(k / 3));
            let xs = x.ldexp(k);
            prop_assert!(rel_err(s2 * s2, xs.abs()) <= 3.0 * TOL, "sqrt {:?} = {:?}", x, s);
            prop_assert!(rel_err(c3 * c3 * c3, xs) <= 4.0 * TOL, "cbrt {:?} = {:?}", x, c);
        }
    }

    #[test]
    fn exp_inverts_ln(x in wide_dd()) {
        let x = x.abs();
        let y = x.ln();
        prop_assert!(!y.hi().is_nan(), "ln {:?} = {:?}", x, y);
        if x.hi() >= 2f64.powi(-966) {
            // exp(y) has relative error <= TOL + |y|·(ln's relative error).
            let tol = (2.0 + y.hi().abs()) * TOL;
            prop_assert!(rel_err(y.exp(), x) <= tol, "ln {:?} = {:?}", x, y);
            let t = x.log2();
            prop_assert!(rel_err(t.exp2(), x) <= tol, "log2 {:?} = {:?}", x, t);
        }
    }

    #[test]
    fn exp_m1_inverts_ln_1p(x in wide_dd()) {
        // x in (-1, MAX], including values within 2^-1074 of -1.
        let x = if x.hi() <= -1.0 { x.abs().recip() - 1.0 } else { x };
        prop_assume!(x.hi() > -1.0 || x.lo() > 0.0);
        let y = x.ln_1p();
        prop_assert!(!y.hi().is_nan(), "ln_1p {:?} = {:?}", x, y);
        if x.hi().abs() >= 2f64.powi(-968) {
            let tol = 2.0 * (2.0 + y.hi().abs()) * TOL;
            prop_assert!(rel_err(y.exp_m1(), x) <= tol, "ln_1p {:?} = {:?}", x, y);
        }
    }

    #[test]
    fn hyperbolic_functions_invert(x in wide_dd(), u in unit_interval_dd()) {
        let y = x.asinh();
        prop_assert!(y.hi().is_finite(), "asinh {:?} = {:?}", x, y);
        prop_assert_eq!(y.hi().is_sign_negative(), x.hi().is_sign_negative());
        if x.hi().abs() >= 2f64.powi(-968) {
            let tol = 2.0 * (4.0 + y.hi().abs()) * TOL;
            prop_assert!(rel_err(y.sinh(), x) <= tol, "asinh {:?} = {:?}", x, y);
        }
        let a = x.abs() + 1.0; // [1, MAX]
        let y = a.acosh();
        prop_assert!(y.hi().is_finite() && y.hi() >= 0.0, "acosh {:?} = {:?}", a, y);
        let tol = 2.0 * (4.0 + y.hi()) * TOL;
        prop_assert!(rel_err(y.cosh(), a) <= tol, "acosh {:?} = {:?}", a, y);
        let y = u.atanh();
        prop_assert!(y.hi().is_finite(), "atanh {:?} = {:?}", u, y);
        if u.hi().abs() >= 2f64.powi(-968) {
            prop_assert!(rel_err(y.tanh(), u) <= 4.0 * TOL, "atanh {:?} = {:?}", u, y);
        }
    }

    #[test]
    fn powi_agrees_with_powf(x in wide_dd(), n in -64..=64i32) {
        prop_assume!(x.hi() != 0.0);
        let (p, q) = (x.powi(n), x.powf(Dd::from_f64(f64::from(n))));
        let t = p.hi().abs();
        if t >= 2f64.powi(-960) && t <= 2f64.powi(1020) {
            let ln = x.hi().abs().ln();
            let tol = 16.0 * f64::from(n.unsigned_abs() + 1) * U2
                + 2.0 * TOL * (f64::from(n) * ln).abs().max(1.0);
            prop_assert!(rel_err(p, q) <= tol, "{:?}^{}: powi {:?} powf {:?}", x, n, p, q);
        }
    }

    #[test]
    fn results_are_normalised_and_nan_only_outside_the_domain(
        x in wide_dd(),
        y in prop_oneof![wide_dd(), (-3000..3000i32).prop_map(|n| Dd::from_f64(f64::from(n)))],
        n in any::<i32>(),
    ) {
        let one = Dd::ONE;
        let x_minus_1 = x.hi() < -1.0 || (x.hi() == -1.0 && x.lo() < 0.0);
        let cases = [
            ("sqrt", x.sqrt(), x.hi() < 0.0),
            ("exp", x.exp(), false),
            ("exp_m1", x.exp_m1(), false),
            ("ln", x.ln(), x.hi() < 0.0),
            ("ln_1p", x.ln_1p(), x_minus_1),
            ("exp2", x.exp2(), false),
            ("log2", x.log2(), x.hi() < 0.0),
            ("log10", x.log10(), x.hi() < 0.0),
            ("cbrt", x.cbrt(), false),
            ("sinh", x.sinh(), false),
            ("cosh", x.cosh(), false),
            ("tanh", x.tanh(), false),
            ("asinh", x.asinh(), false),
            ("acosh", x.acosh(), x < one),
            ("atanh", x.atanh(), x.abs() > one),
            ("powf", x.powf(y), x.hi() < 0.0 && y != y.trunc()),
            ("powi", x.powi(n), false),
            ("hypot", x.hypot(y), false),
        ];
        for (name, z, nan_expected) in cases {
            prop_assert_eq!(z.hi().is_nan(), nan_expected, "{}({:?}, {:?}, {}) = {:?}", name, x, y, n, z);
            prop_assert!(
                z.hi().is_nan() || z.hi() + z.lo() == z.hi(),
                "{}({:?}, {:?}, {}) = {:?} is not normalised", name, x, y, n, z
            );
            if z.hi() == 0.0 || z.hi().is_infinite() {
                prop_assert_eq!(z.lo().to_bits(), 0, "{}({:?}, {:?}, {}) = {:?}", name, x, y, n, z);
            }
        }
    }

    #[test]
    fn hypot_is_symmetric_and_bounded(x in wide_dd(), y in wide_dd()) {
        let h = x.hypot(y);
        prop_assert_eq!(h, y.hypot(x));
        prop_assert_eq!(h, (-x).hypot(y));
        prop_assert!(h >= x.abs() && h >= y.abs(), "hypot({:?}, {:?}) = {:?}", x, y, h);
        prop_assert!(h.hi().is_infinite() || h <= x.abs() + y.abs());
    }
}
