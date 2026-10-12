#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use std::num::FpCategory;

use common::{Record, find, load};
use num_complex::Complex;
use num_traits::{Float, FloatConst, Num, NumCast, One, ToPrimitive, Zero};
use proptest::prelude::*;
use scatrs::Real;
use scatrs::scalar::{DoubleDouble as Dd, ParseDoubleDoubleError};

/// 2^-100.
const TOL: f64 = 7.888_609_052_210_118e-31;
/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

fn horner<T: Real>(coeffs: &[f64], x: T) -> T {
    coeffs.iter().rev().fold(T::zero(), |acc, &c| acc * x + c)
}

/// 2^n for -1074 <= n <= 1023, including subnormal powers.
fn two_pow(n: i32) -> f64 {
    if n >= -1022 {
        f64::from_bits(((n + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (n + 1074))
    }
}

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == f64::MAX {
        return a - f64::from_bits(a.to_bits() - 1);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// Both words equal bit for bit.
fn same(a: Dd, b: Dd) -> bool {
    a.hi().to_bits() == b.hi().to_bits() && a.lo().to_bits() == b.lo().to_bits()
}

/// `got` is exactly the double `want`: NaN for NaN; otherwise the same leading word bit for
/// bit (so the same sign of zero) and a `+0` trailing word.
fn is_f64(got: Dd, want: f64) -> bool {
    if want.is_nan() {
        got.hi().is_nan() && got.lo().to_bits() == 0
    } else {
        got.hi().to_bits() == want.to_bits() && got.lo().to_bits() == 0
    }
}

fn parse(s: &str) -> Dd {
    Dd::from_decimal_str(s).unwrap_or_else(|e| panic!("{s:?}: {e}"))
}

#[test]
fn float_const_matches_reference() {
    let recs = load("constants.jsonl");
    let table: [(&str, Dd); 19] = [
        ("PI", Dd::PI()),
        ("TAU", Dd::TAU()),
        ("FRAC_PI_2", Dd::FRAC_PI_2()),
        ("FRAC_PI_3", Dd::FRAC_PI_3()),
        ("FRAC_PI_4", Dd::FRAC_PI_4()),
        ("FRAC_PI_6", Dd::FRAC_PI_6()),
        ("FRAC_PI_8", Dd::FRAC_PI_8()),
        ("FRAC_1_PI", Dd::FRAC_1_PI()),
        ("FRAC_2_PI", Dd::FRAC_2_PI()),
        ("FRAC_2_SQRT_PI", Dd::FRAC_2_SQRT_PI()),
        ("SQRT_2", Dd::SQRT_2()),
        ("FRAC_1_SQRT_2", Dd::FRAC_1_SQRT_2()),
        ("E", Dd::E()),
        ("LN_2", Dd::LN_2()),
        ("LN_10", Dd::LN_10()),
        ("LOG2_E", Dd::LOG2_E()),
        ("LOG10_E", Dd::LOG10_E()),
        ("LOG10_2", Dd::LOG10_2()),
        ("LOG2_10", Dd::LOG2_10()),
    ];
    for (name, value) in table {
        let v = find(&recs, name).output("value");
        assert_eq!((value.hi(), value.lo()), (v.hi[0], v.lo[0]), "{name}");
    }
}

#[test]
fn complex_double_double_arithmetic() {
    let d = Dd::from_f64;
    let a = Complex::new(d(1.0), d(2.0));
    let b = Complex::new(d(3.0), d(-1.0));
    assert_eq!(a * b, Complex::new(d(5.0), d(5.0)));
    let back = (a * b) / b;
    assert!((back - a).norm_sqr().sqrt().hi() <= TOL);
    let minus_one = Complex::new(Dd::ZERO, Dd::PI()).exp();
    assert!(
        (minus_one + Complex::new(Dd::ONE, Dd::ZERO))
            .norm_sqr()
            .sqrt()
            .hi()
            <= TOL
    );
}

#[test]
fn generic_kernel_runs_in_double_double() {
    let x = Dd::from_f64(1.0) / Dd::from_f64(3.0);
    let p = horner(&[1.0, 2.0, 3.0], x);
    let expected = Dd::from_f64(2.0); // 1 + 2/3 + 3/9 = 2
    assert!((p - expected).abs().hi() <= 4.0 * TOL);
    assert_eq!(
        <Dd as Real>::from_f64_pair(1.0, 2f64.powi(-60)).lo(),
        2f64.powi(-60)
    );
    assert_eq!(Real::approx_f64(Dd::new(1.0, 2f64.powi(-60))), 1.0);
}

fn affine_in_place<T: Real>(mut x: T) -> T {
    x += 1.0;
    x *= 2.0;
    x -= 0.5;
    x /= 4.0;
    x
}

fn abs2_of_inverse<T: Real>(z: Complex<T>) -> T {
    (Complex::new(T::one(), T::zero()) / z).norm_sqr()
}

fn pi_over<T: Real>(n: f64) -> T {
    T::PI() / n
}

#[test]
fn generic_kernels_with_assignment_complex_and_constants() {
    assert!(same(
        affine_in_place(Dd::from_f64(3.0)),
        Dd::from_f64(1.875)
    ));
    let v = abs2_of_inverse(Complex::new(Dd::from_f64(3.0), Dd::from_f64(4.0)));
    assert!((v - Dd::from_f64(1.0) / Dd::from_f64(25.0)).abs().hi() <= TOL / 25.0);
    assert!(same(pi_over::<Dd>(2.0), Dd::FRAC_PI_2()));
    assert_eq!(<Dd as Zero>::zero(), Dd::ZERO);
    assert!(<Dd as Zero>::is_zero(&Dd::from_f64(-0.0)));
    assert!(!<Dd as Zero>::is_zero(&Dd::new(0.0, 0.0).add_f64(1e-300)));
    assert_eq!(<Dd as One>::one(), Dd::ONE);
    assert!(<Dd as One>::is_one(&Dd::ONE));
    assert!(!<Dd as One>::is_one(&Dd::new(1.0, 1e-20)));
}

// ---- Complex<DoubleDouble> against Arb ----------------------------------------------------

fn dd_pair(r: &Record, re: &str, im: &str) -> Vec<Complex<Dd>> {
    let f = |p: &str| {
        let hi = r.input_f64s(&format!("{p}_hi"));
        let lo = r.input_f64s(&format!("{p}_lo"));
        hi.iter()
            .zip(&lo)
            .map(|(&h, &l)| Dd::new(h, l))
            .collect::<Vec<_>>()
    };
    f(re)
        .into_iter()
        .zip(f(im))
        .map(|(a, b)| Complex::new(a, b))
        .collect()
}

/// Checks |got − exact| <= tol · scale(exact) over the record `case` of dd_complex.jsonl,
/// with the error of each part measured against the three-word reference.
fn check_complex(
    case: &str,
    tol: f64,
    f: impl Fn(Complex<Dd>, usize) -> Complex<Dd>,
    scale: impl Fn(Complex<f64>) -> f64,
) {
    let recs = load("dd_complex.jsonl");
    let r = find(&recs, case);
    let zs = dd_pair(r, "re", "im");
    let (re, im) = (r.output("re"), r.output("im"));
    let mut worst = 0.0_f64;
    for (i, &z) in zs.iter().enumerate() {
        let got = f(z, i);
        let err = re.dd_error(i, got.re.hi(), got.re.lo()).hypot(im.dd_error(
            i,
            got.im.hi(),
            got.im.lo(),
        ));
        let rel = err / scale(Complex::new(re.hi[i], im.hi[i]));
        worst = worst.max(rel);
        assert!(
            rel <= tol,
            "{case} case {i}: {z:?} gives {got:?}, reference ({:e}, {:e}), {rel:e} > {tol:e}",
            re.hi[i],
            im.hi[i]
        );
    }
    eprintln!("complex {case}: worst {worst:.3e} = {:.2} u²", worst / U2);
}

#[test]
fn complex_exp_matches_reference() {
    // e^a (cos b, sin b): exp 2^-100 relative, cos and sin 2^-100 absolute, products 4u²; so
    // the norm-wise error is at most about (2√2 + 1/16)·2^-100 < 2^-98.
    check_complex("exp", 4.0 * TOL, |z, _| z.exp(), |e| e.norm());
}

#[test]
fn complex_ln_matches_reference() {
    // (ln |z|, arg z): ln of the 2^-100-accurate |z| errs by 2^-100 (1 + |ln |z||), and arg z by
    // 2^-99 |arg z|; so the error is at most 4·2^-100 max(1, |ln z|). Relative to |ln z| alone
    // it is unbounded next to z = 1, where ln z vanishes but |z| is still rounded.
    check_complex("ln", 4.0 * TOL, |z, _| z.ln(), |l| l.norm().max(1.0));
}

#[test]
fn complex_sqrt_matches_reference() {
    // √|z| (cos, sin)(arg z / 2): √ of the rounded |z| errs by 1.5·2^-100, arg z / 2 by
    // 2^-99·π/2 absolute, cos and sin by 2^-100 absolute each; so the norm-wise error is
    // below 6.2·2^-100 < 8·2^-100.
    check_complex("sqrt", 8.0 * TOL, |z, _| z.sqrt(), |s| s.norm());
}

#[test]
fn complex_division_matches_reference() {
    // ((ac + bd), (bc − ad))/(c² + d²): about 27u² norm-wise.
    let recs = load("dd_complex.jsonl");
    let ys = dd_pair(find(&recs, "div"), "y_re", "y_im");
    check_complex("div", TOL, |x, i| x / ys[i], |q| q.norm());
}

// ---- Decimal parsing ----------------------------------------------------------------------

#[test]
fn decimal_parsing_matches_reference() {
    let recs = load("dd_parse.jsonl");
    for case in ["literals", "literals_edges"] {
        let r = find(&recs, case);
        let v = r.output("value");
        let mut worst = 0.0_f64;
        for (i, text) in r.input_strs("text").iter().enumerate() {
            let got = Dd::from_decimal_str(text).expect("valid literal");
            let exact = Dd::new(v.hi[i], v.lo[i]);
            let rel = v.dd_error(i, got.hi(), got.lo()) / exact.abs().hi();
            worst = worst.max(rel);
            // Tightened from the plan's 4·TOL to the documented bound 2^-100 (the error
            // analysis in parse.rs gives at most 34u²).
            assert!(
                rel <= TOL,
                "{text}: got {got:?}, reference {exact:?}, rel {rel:e}"
            );
            // No reference value lies within 2^-100 of a rounding boundary, so the leading word
            // is the correctly rounded double.
            assert_eq!(got.hi(), v.hi[i], "{text}");
        }
        eprintln!(
            "parse {case}: worst relative error {worst:e} = {:.2} u²",
            worst / U2
        );
    }
}

#[test]
fn decimal_parsing_edge_cases() {
    for bad in ["", "abc", "1e", "1.2.3", "--1", "e5", "."] {
        assert!(
            Dd::from_decimal_str(bad).is_err(),
            "{bad:?} should not parse"
        );
    }
    assert_eq!(Dd::from_decimal_str("inf").unwrap().hi(), f64::INFINITY);
    assert_eq!(
        Dd::from_decimal_str("-Infinity").unwrap().hi(),
        f64::NEG_INFINITY
    );
    assert!(Dd::from_decimal_str("NaN").unwrap().hi().is_nan());
    assert_eq!(Dd::from_decimal_str("+0.0").unwrap(), Dd::ZERO);
    assert_eq!(Dd::from_decimal_str("1e-400").unwrap(), Dd::ZERO);
    assert_eq!(Dd::from_decimal_str("1e400").unwrap().hi(), f64::INFINITY);
    // Decimal fractions go through a double-double division, so compare with a tolerance.
    let half = Dd::from_decimal_str(".5").unwrap();
    assert!((half - Dd::from_f64(0.5)).abs().hi() <= TOL * 0.5);
    assert!(<Dd as Num>::from_str_radix("10", 2).is_err());
    let two_and_a_half = <Dd as Num>::from_str_radix("2.5", 10).unwrap();
    assert!((two_and_a_half - Dd::from_f64(2.5)).abs().hi() <= TOL * 2.5);
}

#[test]
fn malformed_literals_are_rejected_like_f64() {
    let bad = [
        "+",
        "-",
        "+-1",
        "-+1",
        "1e+",
        "1e-",
        "1e5.5",
        "1e1e1",
        "1.5e",
        ".e1",
        "-.e1",
        "1_000",
        "0x10",
        "1,5",
        "1 5",
        "- 1",
        "+ inf",
        "١",
        "１",
        "∞",
        "infinit",
        "infinityx",
        "nana",
        "nan(1)",
        "in",
        "e",
        "E5",
        "1f",
        "0.0.0",
        "\u{0}1",
        "1e٣",
        "inf inity",
        "1..2",
    ];
    for s in bad {
        assert!(s.parse::<f64>().is_err(), "{s:?} is a valid f64");
        assert!(Dd::from_decimal_str(s).is_err(), "{s:?}");
        assert!(s.parse::<Dd>().is_err(), "{s:?}");
    }
    for radix in [2, 8, 16, 36] {
        assert!(<Dd as Num>::from_str_radix("1", radix).is_err());
    }
    let e: ParseDoubleDoubleError = "x".parse::<Dd>().unwrap_err();
    assert_eq!(e.to_string(), "invalid double-double literal");
    let boxed: Box<dyn std::error::Error> = Box::new(e.clone());
    assert_eq!(boxed.to_string(), e.to_string());
    assert_eq!(Dd::from_decimal_str("1e"), Err(e));
}

#[test]
fn from_str_is_strict_like_f64() {
    // from_decimal_str and Num::from_str_radix ignore surrounding whitespace; str::parse does
    // not, as for f64.
    for s in [" 1.5", "1.5 ", "\t-inf", "1.5\n", " nan "] {
        assert!(s.parse::<f64>().is_err(), "{s:?}");
        assert!(s.parse::<Dd>().is_err(), "{s:?}");
        assert!(Dd::from_decimal_str(s).is_ok(), "{s:?}");
        assert!(<Dd as Num>::from_str_radix(s, 10).is_ok(), "{s:?}");
    }
    assert!(same("1.5".parse::<Dd>().unwrap(), Dd::from_f64(1.5)));
    assert!(same("-0".parse::<Dd>().unwrap(), Dd::from_f64(-0.0)));
    // So Complex<DoubleDouble> parses exactly the strings that Complex<f64> parses.
    for s in [
        "1.5+2.5i",
        "1.5 + 2.5i",
        " 1.5+2.5i",
        "1.5+2.5i ",
        "2i",
        "-1e3-4.5e-2i",
        "1.5+ 2.5 i",
        "i",
        "-i",
        "1.5",
        "inf-nani",
        "1.5+2.5j",
        "1.5++2.5i",
        "",
    ] {
        let (got, want) = (s.parse::<Complex<Dd>>(), s.parse::<Complex<f64>>());
        assert_eq!(got.is_ok(), want.is_ok(), "{s:?}");
        if let (Ok(g), Ok(w)) = (got, want) {
            let ok = |a: Dd, b: f64| a.hi().to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
            assert!(ok(g.re, w.re) && ok(g.im, w.im), "{s:?}: {g:?} vs {w:?}");
        }
    }
}

#[test]
fn signed_zeros_underflow_and_overflow() {
    // Zeros, and values that round to zero, keep the sign of the literal (as for f64), with a
    // +0 trailing word.
    let zeros = [
        ("0", 0.0),
        ("-0", -0.0),
        ("+0", 0.0),
        ("-0.000e-5", -0.0),
        ("-.0", -0.0),
        ("0.", 0.0),
        ("00.00", 0.0),
        ("0e99999999999999999999999", 0.0),
        ("-0e-99999999999999999999999", -0.0),
        ("1e-400", 0.0),
        ("-1e-400", -0.0),
        ("-2.4703282292062327e-324", -0.0), // just below half the smallest subnormal
        ("2e-324", 0.0),
        ("1e-9223372036854775808", 0.0),
        ("-9e-99999999999999999999", -0.0),
    ];
    for (s, want) in zeros {
        assert!(is_f64(parse(s), want), "{s}: {:?}", parse(s));
        assert_eq!(s.parse::<f64>().unwrap().to_bits(), want.to_bits(), "{s}");
    }
    let infinities = [
        ("1e400", f64::INFINITY),
        ("-1e400", f64::NEG_INFINITY),
        ("1e99999999999999999999999", f64::INFINITY),
        ("1e+9223372036854775807", f64::INFINITY),
        ("1e9223372036854775808", f64::INFINITY),
        ("-1.7976931348623159e308", f64::NEG_INFINITY),
        ("1.797693134862315808e308", f64::INFINITY), // just above MAX + ulp/2
        ("inf", f64::INFINITY),
        ("+Infinity", f64::INFINITY),
        ("-INF", f64::NEG_INFINITY),
    ];
    for (s, want) in infinities {
        assert!(is_f64(parse(s), want), "{s}: {:?}", parse(s));
        assert_eq!(s.parse::<f64>().unwrap(), want, "{s}");
    }
    // NaN keeps the sign of the literal, like f64.
    for (s, negative) in [("nan", false), ("-NaN", true), ("+nAn", false)] {
        let x = parse(s);
        assert!(x.hi().is_nan() && x.lo().to_bits() == 0, "{s}");
        assert_eq!(x.hi().is_sign_negative(), negative, "{s}");
        assert_eq!(
            s.parse::<f64>().unwrap().is_sign_negative(),
            negative,
            "{s}"
        );
    }
}

#[test]
fn range_ends_match_f64() {
    // The leading word is the correctly rounded double at the ends of the range: no early
    // overflow next to MAX, and subnormal results rounded once.
    for s in [
        "1.7976931348623157e308",
        "1.7976931348623158e308",
        "-1.797693134862315807e308",
        "8.98846567431158e307",
        "2.2250738585072014e-308",
        "2.2250738585072011e-308",
        "4.9406564584124654e-324",
        "2.4703282292062328e-324",
        "-7.4109846876186982e-324",
        "1e-310",
        "1.23456789012345678901234567890123456789e-315",
        "3.5e-323",
        "2.004168360008973e-292",
    ] {
        let want: f64 = s.parse().unwrap();
        let got = parse(s);
        assert_eq!(got.hi().to_bits(), want.to_bits(), "{s}: {got:?}");
        if want.abs() < f64::MIN_POSITIVE {
            assert_eq!(got.lo().to_bits(), 0, "{s}: {got:?}");
        }
    }
    assert_eq!(parse("1.7976931348623157e308").hi(), f64::MAX);
    assert_eq!(parse(&f64::MAX.to_string()).hi(), f64::MAX);
    // All 309 digits of MAX, which go through the 38-digit rounding and the scaling.
    let max = parse(&format!("{:.0}", f64::MAX));
    assert!(
        max.hi() == f64::MAX && max.lo().abs() <= TOL * f64::MAX,
        "{max:?}"
    );
    assert!(same(
        parse(&format!("{:.1074}", two_pow(-1074))),
        Dd::from_f64(two_pow(-1074))
    ));
}

#[test]
fn whitespace_long_inputs_and_rounding_of_the_mantissa() {
    assert!(same(parse(" 2.5\n"), Dd::from_f64(2.5)));
    assert!(is_f64(parse("\t-inf "), f64::NEG_INFINITY));
    assert!(same("2.5".parse::<Dd>().unwrap(), Dd::from_f64(2.5)));
    let zeros = "0".repeat(10_000);
    assert!(is_f64(parse(&format!("1{zeros}")), f64::INFINITY));
    assert!(is_f64(parse(&format!("0.{zeros}1")), 0.0));
    assert!(is_f64(parse(&format!("-0.{zeros}1")), -0.0));
    // Long runs of digits move the exponent without overflowing or underflowing early.
    assert!(is_f64(parse(&format!("0.{zeros}1e10010")), 1e9));
    assert!(is_f64(parse(&format!("1{zeros}e-10000")), 1.0));
    assert!(is_f64(parse(&format!("0.{zeros}2e+10001")), 2.0));
    // Digits beyond the 38th significant one round the 38th, half up.
    let nines = "9".repeat(100);
    assert!(same(parse(&nines), parse("1e100")));
    let base = "12345678901234567890123456789012345678"; // 38 digits
    let base_up = "12345678901234567890123456789012345679";
    assert!(same(
        parse(&format!("{base}499")),
        parse(&format!("{base}000"))
    ));
    assert!(same(
        parse(&format!("{base}500")),
        parse(&format!("{base_up}000"))
    ));
    assert!(same(
        parse(&format!("{base}49999999999")),
        parse(&format!("{base}e11"))
    ));
    assert!(same(
        parse("1.2345678901234567890123456789012345678500000000000000000000001"),
        parse("1.23456789012345678901234567890123456785")
    ));
    assert!(same(
        parse(&format!("0.{}{}", "0".repeat(50), "1".repeat(60))),
        parse(&format!("{}e-51", "1".repeat(38)).replacen('1', "1.", 1))
    ));
}

#[test]
fn decimal_parsing_is_exact_for_representable_values() {
    // Integers below 2^106, in any form.
    let n = (1u128 << 106) - 1;
    let exact = Dd::new(two_pow(106), -1.0);
    for s in [
        n.to_string(),
        format!("{n}.000"),
        format!("{n}e0"),
        format!("{n}0000e-4"),
        "8112963841460668169578900514406.3e1".to_owned(),
        "0.81129638414606681695789005144063e32".to_owned(),
    ] {
        assert!(same(parse(&s), exact), "{s}: {:?}", parse(&s));
    }
    assert!(same(parse("9007199254740993"), Dd::new(two_pow(53), 1.0)));
    let ten31 = 10u128.pow(31);
    let want = Dd::new(1e31, (ten31 as i128 - 1e31 as i128) as f64);
    assert!(same(parse("1e31"), want));
    assert!(same(parse("10000000000000000000000000000000"), want));
    // Doubles m·10^-k with m < 2^53 and k <= 22.
    for (s, x) in [
        ("0.5", 0.5),
        ("2.5", 2.5),
        ("-0.125", -0.125),
        ("0.375", 0.375),
        ("4.76837158203125e-7", two_pow(-21)),
        ("2.384185791015625e-7", two_pow(-22)),
        ("1e22", 1e22),
        ("123.456e3", 123456.0),
        ("1000e-3", 1.0),
    ] {
        assert!(is_f64(parse(s), x), "{s}: {:?}", parse(s));
    }
}

/// Finite doubles of any magnitude, including subnormals and both zeros.
fn any_finite() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("finite", |x| x.is_finite())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn parsing_never_panics_and_accepts_what_f64_accepts(s in any::<String>()) {
        let got = Dd::from_decimal_str(&s);
        prop_assert_eq!(got.is_ok(), s.trim().parse::<f64>().is_ok(), "{:?}", s);
        prop_assert_eq!(s.parse::<Dd>().is_ok(), s.parse::<f64>().is_ok(), "{:?}", s);
    }

    #[test]
    fn number_like_strings_parse_like_f64(
        s in "[ +-]{0,2}[0-9]{0,30}[.]?[0-9]{0,30}([eE][+-]?[0-9]{0,4})?[ ]?",
        t in "[-+.eE0-9iInNaAfFtTyY]{0,12}",
    ) {
        for s in [s, t] {
            // str::parse is strict: it accepts what f64 accepts, and agrees with
            // from_decimal_str there.
            let strict = s.parse::<Dd>();
            prop_assert_eq!(strict.is_ok(), s.parse::<f64>().is_ok(), "{:?}", s);
            if let Ok(x) = strict {
                prop_assert!(same(x, Dd::from_decimal_str(&s).unwrap()), "{:?}", s);
            }
            let want = s.trim().parse::<f64>();
            let got = Dd::from_decimal_str(&s);
            prop_assert_eq!(got.is_ok(), want.is_ok(), "{:?}", s);
            if let (Ok(got), Ok(want)) = (got, want) {
                // The leading word is the correctly rounded double, including the sign of
                // zero, infinities and subnormals.
                prop_assert!(
                    got.hi().to_bits() == want.to_bits() || (got.hi().is_nan() && want.is_nan()),
                    "{:?}: {:?} vs {:?}", s, got, want
                );
            }
        }
    }

    #[test]
    fn printed_doubles_parse_back(x in prop_oneof![any_finite(), -1e3..1e3f64]) {
        for s in [format!("{x:e}"), format!("{x:?}"), format!("{x}"), format!("{x:.40e}")] {
            let got = parse(&s);
            prop_assert_eq!(got.hi().to_bits(), x.to_bits(), "{}: {:?}", s, got);
            prop_assert!(got.hi() + got.lo() == got.hi(), "{}: {:?} is not normalised", s, got);
        }
        // 41 significant digits: the value is x to 10^-40 relative.
        if x.abs() >= two_pow(-969) {
            let got = parse(&format!("{x:.40e}"));
            prop_assert!(got.lo().abs() <= TOL * x.abs(), "{:?}", got);
        }
    }

    #[test]
    fn integers_below_2_pow_106_parse_exactly(n in 0u128..(1u128 << 106), k in 0usize..4) {
        let exact = <Dd as NumCast>::from(n).unwrap();
        prop_assert_eq!(exact.to_u128(), Some(n));
        let zeros = "0".repeat(k);
        for s in [n.to_string(), format!("{n}.{zeros}"), format!("+{n}{zeros}e-{k}")] {
            prop_assert!(same(parse(&s), exact), "{}: {:?} vs {:?}", s, parse(&s), exact);
        }
        let negative = format!("-{n}");
        prop_assert!(same(parse(&negative), -exact));
    }

    #[test]
    fn negation_is_exact(s in "[0-9]{1,45}([.][0-9]{0,45})?(e[+-]?[0-9]{1,3})?") {
        let (x, y) = (parse(&s), parse(&format!("-{s}")));
        prop_assert!(same(y, -x), "{}: {:?} vs {:?}", s, x, y);
    }
}

// ---- ToPrimitive and NumCast --------------------------------------------------------------

#[test]
fn num_cast_and_to_primitive() {
    assert_eq!(<Dd as NumCast>::from(2.5_f64), Some(Dd::from_f64(2.5)));
    let big = <Dd as NumCast>::from(i64::MAX).unwrap();
    assert_eq!(big.to_i64(), Some(i64::MAX));
    assert_eq!(
        <Dd as NumCast>::from(u64::MAX).unwrap().to_u64(),
        Some(u64::MAX)
    );
    assert_eq!(Dd::from_f64(-3.7).to_i64(), Some(-3));
    assert_eq!(Dd::from_f64(-1.0).to_u64(), None);
    assert_eq!(Dd::from_f64(1e30).to_i64(), None);
    assert_eq!(Dd::new(1.0, 2f64.powi(-60)).to_f64(), Some(1.0));
}

#[test]
fn integer_conversions_are_exact_at_the_extremes() {
    let cast = |x: i128| <Dd as NumCast>::from(x).unwrap();
    assert!(same(cast(i64::MAX.into()), Dd::new(two_pow(63), -1.0)));
    assert!(same(cast(i64::MIN.into()), Dd::from_f64(-two_pow(63))));
    assert!(same(cast(i128::MAX), Dd::new(two_pow(127), -1.0)));
    assert!(same(cast(i128::MIN), Dd::from_f64(-two_pow(127))));
    assert!(same(
        <Dd as NumCast>::from(u64::MAX).unwrap(),
        Dd::new(two_pow(64), -1.0)
    ));
    assert!(same(
        <Dd as NumCast>::from(u128::MAX).unwrap(),
        Dd::new(f64::from_bits((1023 + 128) << 52), -1.0)
    ));
    assert_eq!(cast(i128::MAX).to_i128(), Some(i128::MAX));
    assert_eq!(cast(i128::MIN).to_i128(), Some(i128::MIN));
    assert_eq!(
        <Dd as NumCast>::from(u128::MAX).unwrap().to_u128(),
        Some(u128::MAX)
    );
    assert_eq!(cast(i64::MIN.into()).to_i64(), Some(i64::MIN));
    // Next to a tie of the leading word: 2^126 + 2^73 + 1 gives (2^126, 2^73), off by 1, although
    // the nearest double is 2^126 + 2^74.
    let n: i128 = (1 << 126) + (1 << 73) + 1;
    assert!(same(cast(n), Dd::new(two_pow(126), two_pow(73))));
    assert!(same(cast(-n), -Dd::new(two_pow(126), two_pow(73))));
    // Beyond 2^106 the nearest double-double: 2^120 + 2^66 + 1 needs 121 bits.
    let n: i128 = (1 << 120) + (1 << 66) + 1;
    assert!(same(cast(n), Dd::new(two_pow(120), two_pow(66))));
    // Every primitive type converts exactly.
    assert!(same(
        <Dd as NumCast>::from(-7i8).unwrap(),
        Dd::from_f64(-7.0)
    ));
    assert!(same(
        <Dd as NumCast>::from(65535u16).unwrap(),
        Dd::from_f64(65535.0)
    ));
    assert!(same(
        <Dd as NumCast>::from(usize::MAX).unwrap(),
        Dd::new(two_pow(64), -1.0)
    ));
    assert!(is_f64(
        <Dd as NumCast>::from(0.1f32).unwrap(),
        <f64 as From<f32>>::from(0.1f32)
    ));
    assert!(is_f64(<Dd as NumCast>::from(-0.0f64).unwrap(), -0.0));
    assert!(is_f64(<Dd as NumCast>::from(f32::NAN).unwrap(), f64::NAN));
    assert!(is_f64(
        <Dd as NumCast>::from(f64::NEG_INFINITY).unwrap(),
        f64::NEG_INFINITY
    ));
    // Another double-double keeps its integer part beyond 2^53 (but only its leading word below).
    assert!(same(
        <Dd as NumCast>::from(Dd::new(two_pow(60), 3.0)).unwrap(),
        Dd::new(two_pow(60), 3.0)
    ));
}

#[test]
fn to_primitive_truncates_double_doubles_towards_zero() {
    let p = two_pow;
    let cases: [(Dd, Option<i64>, Option<u64>); 14] = [
        (
            Dd::new(p(60), -0.5),
            Some((1 << 60) - 1),
            Some((1 << 60) - 1),
        ),
        (Dd::new(p(60), 0.5), Some(1 << 60), Some(1 << 60)),
        (Dd::new(-p(60), 0.5), Some(-(1 << 60) + 1), None),
        (Dd::new(p(63), -1.0), Some(i64::MAX), Some((1 << 63) - 1)),
        (Dd::new(p(63), -0.5), Some(i64::MAX), Some((1 << 63) - 1)),
        (Dd::from_f64(p(63)), None, Some(1 << 63)),
        (Dd::new(p(63), 0.5), None, Some(1 << 63)),
        (Dd::new(-p(63), -0.5), Some(i64::MIN), None),
        (Dd::new(-p(63), -1.0), None, None),
        (Dd::new(p(64), -1.0), None, Some(u64::MAX)),
        (Dd::new(p(64), -0.5), None, Some(u64::MAX)),
        (Dd::from_f64(p(64)), None, None),
        (Dd::from_f64(-0.5), Some(0), Some(0)),
        (Dd::new(-1.0, p(-60)), Some(0), Some(0)),
    ];
    for (x, i, u) in cases {
        assert_eq!(x.to_i64(), i, "{x:?}");
        assert_eq!(x.to_u64(), u, "{x:?}");
    }
    assert_eq!(Dd::new(p(127), -1.0).to_i128(), Some(i128::MAX));
    assert_eq!(Dd::new(p(127), -0.5).to_i128(), Some(i128::MAX));
    assert_eq!(Dd::from_f64(p(127)).to_i128(), None);
    assert_eq!(Dd::new(-p(127), -0.5).to_i128(), Some(i128::MIN));
    assert_eq!(Dd::new(-p(127), -1.0).to_i128(), None);
    assert_eq!(Dd::from_f64(p(127)).to_u128(), Some(1 << 127));
    assert_eq!(Dd::new(p(128), -0.5).to_u128(), Some(u128::MAX));
    assert_eq!(Dd::from_f64(p(128)).to_u128(), None);
    assert_eq!(Dd::new(p(31), -0.5).to_i32(), Some(i32::MAX));
    assert_eq!(Dd::from_f64(p(31)).to_i32(), None);
    assert_eq!(Dd::from_f64(255.9).to_u8(), Some(255));
    assert_eq!(<u8 as NumCast>::from(Dd::from_f64(256.0)), None);
    assert_eq!(<i32 as NumCast>::from(Dd::from_f64(-2.7)), Some(-2));
    for x in [Dd::NAN, Dd::INFINITY, Dd::NEG_INFINITY] {
        assert_eq!(x.to_i64(), None);
        assert_eq!(x.to_u64(), None);
        assert_eq!(x.to_i128(), None);
        assert_eq!(x.to_u128(), None);
    }
    assert!(Dd::NAN.to_f64().unwrap().is_nan());
    assert_eq!(Dd::NEG_INFINITY.to_f32(), Some(f32::NEG_INFINITY));
}

#[test]
fn to_f32_rounds_once() {
    let p = two_pow;
    // 1 + 2^-24 is halfway between two f32s; the low word decides.
    assert_eq!(
        Dd::new(1.0 + p(-24), p(-80)).to_f32(),
        Some(1.0 + 2f32.powi(-23))
    );
    assert_eq!(Dd::new(1.0 + p(-24), -p(-80)).to_f32(), Some(1.0));
    assert_eq!(Dd::from_f64(1.0 + p(-24)).to_f32(), Some(1.0));
    assert_eq!(
        Dd::new(-(1.0 + p(-24)), -p(-80)).to_f32(),
        Some(-(1.0 + 2f32.powi(-23)))
    );
    // f32 subnormals: 1.5·2^-149 is halfway between 2^-149 and 2^-148.
    assert_eq!(
        Dd::new(1.5 * p(-149), p(-200)).to_f32(),
        Some(2f32.powi(-148))
    );
    assert_eq!(
        Dd::new(1.5 * p(-149), -p(-200)).to_f32(),
        Some(2f32.powi(-149))
    );
    // Overflow to infinity, as f64 to f32 does.
    assert_eq!(Dd::from_f64(1e300).to_f32(), Some(f32::INFINITY));
    let f32_max_half_ulp = <f64 as From<f32>>::from(f32::MAX) + p(103);
    assert_eq!(Dd::new(f32_max_half_ulp, -p(20)).to_f32(), Some(f32::MAX));
    assert_eq!(
        Dd::new(f32_max_half_ulp, p(20)).to_f32(),
        Some(f32::INFINITY)
    );
    assert_eq!(
        <f32 as NumCast>::from(Dd::new(1.0 + p(-24), p(-80))),
        Some(1.0 + 2f32.powi(-23))
    );
}

/// Double-doubles next to the limits of the integer types, with fractional low words.
fn integer_edge_dd() -> impl Strategy<Value = Dd> {
    let exponents = prop::sample::select(vec![
        None,
        Some(0),
        Some(1),
        Some(31),
        Some(32),
        Some(53),
        Some(60),
        Some(63),
        Some(64),
        Some(106),
        Some(127),
        Some(128),
        Some(129),
    ]);
    let lo = prop::sample::select(vec![
        -2.0, -1.5, -1.0, -0.75, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5,
    ]);
    (exponents, lo, any::<bool>()).prop_map(|(e, l, neg)| {
        let x = Dd::new(e.map_or(0.0, two_pow), l);
        if neg { -x } else { x }
    })
}

/// Finite double-doubles from the whole range, with low words up to ±ulp(hi)/2.
fn wide_dd() -> impl Strategy<Value = Dd> {
    let frac = prop_oneof![-0.5..0.5f64, Just(0.0), Just(0.49), Just(-0.49)];
    (any_finite(), frac).prop_map(|(h, f)| {
        let d = Dd::new(h, f * ulp(h));
        if d.hi().is_finite() {
            d
        } else {
            Dd::from_f64(h)
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn to_integer_is_the_truncation_when_in_range(
        x in prop_oneof![integer_edge_dd(), wide_dd(), (-1e40..1e40f64, -0.5..0.5f64)
            .prop_map(|(h, f)| Dd::new(h, f * ulp(h)))]
    ) {
        let t = Float::trunc(x);
        let back = |n: Option<i128>| n.map(|n| <Dd as NumCast>::from(n).unwrap());
        let in_range = |lo: Dd, hi: Dd| lo <= t && t <= hi;
        let (p63, p64, p127, p128) = (two_pow(63), two_pow(64), two_pow(127), two_pow(127) * 2.0);
        let checks = [
            (x.to_i64().map(<i128 as From<_>>::from), in_range(Dd::from_f64(-p63), Dd::new(p63, -1.0))),
            (x.to_u64().map(<i128 as From<_>>::from), in_range(Dd::ZERO, Dd::new(p64, -1.0))),
            (x.to_i32().map(<i128 as From<_>>::from), in_range(Dd::from_f64(-2147483648.0), Dd::from_f64(2147483647.0))),
            (x.to_i128(), in_range(Dd::from_f64(-p127), Dd::new(p127, -1.0))),
        ];
        for (n, ok) in checks {
            prop_assert_eq!(n.is_some(), ok, "{:?}", x);
            if let Some(v) = back(n) {
                prop_assert!(v == t, "{:?}: {:?} vs trunc {:?}", x, v, t);
            }
        }
        let u = x.to_u128();
        prop_assert_eq!(u.is_some(), in_range(Dd::ZERO, Dd::new(p128, -1.0)), "{:?}", x);
        if let Some(u) = u {
            prop_assert!(<Dd as NumCast>::from(u).unwrap() == t, "{:?}", x);
        }
    }

    #[test]
    fn integers_round_trip(i in any::<i64>(), u in any::<u64>(), j in any::<i128>(), v in any::<u128>()) {
        prop_assert_eq!(<Dd as NumCast>::from(i).unwrap().to_i64(), Some(i));
        prop_assert_eq!(<Dd as NumCast>::from(u).unwrap().to_u64(), Some(u));
        let j = j >> 21; // |j| < 2^106: exact
        prop_assert_eq!(<Dd as NumCast>::from(j).unwrap().to_i128(), Some(j));
        let v = v >> 22;
        prop_assert_eq!(<Dd as NumCast>::from(v).unwrap().to_u128(), Some(v));
    }

    #[test]
    fn wide_integers_are_within_u2(
        j in prop_oneof![any::<i128>(), near_tie().prop_map(|n| (n >> 1) as i128)],
        v in prop_oneof![any::<u128>(), near_tie()],
        neg in any::<bool>(),
    ) {
        // Exact below 2^106 in magnitude, within u²|n| above. (Next to a tie the leading word
        // need not be the nearest double: see from_u128.)
        let j = if neg { j.wrapping_neg() } else { j };
        let x = <Dd as NumCast>::from(j).unwrap();
        prop_assert!(x.hi() + x.lo() == x.hi(), "{:?}", x);
        // Both words are integers: the error x − j, exactly.
        let d = if x.hi() == two_pow(127) { (i128::MAX - j) + 1 } else { x.hi() as i128 - j };
        let err = d + x.lo() as i128;
        prop_assert!(within_u2(err, j.unsigned_abs()), "{}: {:?} is off by {}", j, x, err);
        let y = <Dd as NumCast>::from(v).unwrap();
        prop_assert!(y.hi() + y.lo() == y.hi(), "{:?}", y);
        let d = if y.hi() == two_pow(127) * 2.0 {
            ((u128::MAX - v) + 1) as i128
        } else if y.hi() as u128 >= v {
            (y.hi() as u128 - v) as i128
        } else {
            -((v - y.hi() as u128) as i128)
        };
        let err = d + y.lo() as i128;
        prop_assert!(within_u2(err, v), "{}: {:?} is off by {}", v, y, err);
    }
}

/// Integers within 2 of a tie between two doubles (a double plus half its ulp), up to 2^128.
fn near_tie() -> impl Strategy<Value = u128> {
    (54..=127u32, (1u128 << 52)..(1u128 << 53), -2i128..=2).prop_map(|(e, m, delta)| {
        let tie = (m << (e - 52)) + (1u128 << (e - 53));
        tie.checked_add_signed(delta).unwrap_or(tie)
    })
}

/// |err| is 0 for n below 2^106, and at most u²·n above.
fn within_u2(err: i128, n: u128) -> bool {
    if n < 1 << 106 {
        err == 0
    } else {
        err.unsigned_abs() as f64 <= U2 * n as f64
    }
}

// ---- Float -------------------------------------------------------------------------------

#[test]
fn float_constants_and_limits() {
    assert_eq!(<Dd as Float>::epsilon().hi(), 2f64.powi(-104));
    assert_eq!(<Dd as Float>::min_positive_value().hi(), 2f64.powi(-969));
    assert_eq!(<Dd as Float>::max_value().hi(), f64::MAX);
    assert!(<Dd as Float>::nan().is_nan());
    assert!(<Dd as Float>::neg_zero().is_sign_negative());
    assert_eq!(Float::max(Dd::NAN, Dd::ONE), Dd::ONE);
    assert_eq!(Float::min(Dd::ONE, Dd::NAN), Dd::ONE);
    let deg = Float::to_degrees(Dd::PI());
    assert!((deg - Dd::from_f64(180.0)).abs().hi() <= 180.0 * TOL);
}

#[test]
fn float_limits_and_classification() {
    assert!(is_f64(<Dd as Float>::min_value(), f64::MIN));
    assert!(is_f64(<Dd as Float>::max_value(), f64::MAX));
    assert!(is_f64(<Dd as Float>::neg_zero(), -0.0));
    assert!(is_f64(<Dd as Float>::infinity(), f64::INFINITY));
    assert!(is_f64(<Dd as Float>::neg_infinity(), f64::NEG_INFINITY));
    assert!(is_f64(<Dd as Float>::epsilon(), two_pow(-104)));
    assert!(is_f64(<Dd as Float>::min_positive_value(), two_pow(-969)));
    // Classification describes the leading word: values in [2^-1022, 2^-969) are normal,
    // though below full precision.
    assert_eq!(
        Float::classify(Dd::new(two_pow(-1000), 0.0)),
        FpCategory::Normal
    );
    assert_eq!(Float::classify(Dd::from_f64(1e-310)), FpCategory::Subnormal);
    assert!(Float::is_subnormal(Dd::from_f64(1e-310)));
    assert!(Float::is_normal(Dd::new(1.0, 1e-20)));
    assert_eq!(Float::classify(Dd::new(1.0, 1e-300)), FpCategory::Normal);
    assert_eq!(
        Float::integer_decode(Dd::new(1.5, 1e-17)),
        Float::integer_decode(1.5f64)
    );
}

#[test]
fn rounding_double_doubles() {
    let p = two_pow;
    let cases = [
        // x, floor, ceil, trunc, round, fract
        (
            Dd::new(p(60), -0.5),
            Dd::new(p(60), -1.0),
            Dd::from_f64(p(60)),
            Dd::new(p(60), -1.0),
            Dd::from_f64(p(60)),
            Dd::from_f64(0.5),
        ),
        (
            Dd::new(p(60), 0.5),
            Dd::from_f64(p(60)),
            Dd::new(p(60), 1.0),
            Dd::from_f64(p(60)),
            Dd::new(p(60), 1.0),
            Dd::from_f64(0.5),
        ),
        (
            Dd::new(p(60), -0.25),
            Dd::new(p(60), -1.0),
            Dd::from_f64(p(60)),
            Dd::new(p(60), -1.0),
            Dd::from_f64(p(60)),
            Dd::from_f64(0.75),
        ),
        (
            Dd::new(-p(60), 0.25),
            Dd::from_f64(-p(60)),
            Dd::new(-p(60), 1.0),
            Dd::new(-p(60), 1.0),
            Dd::from_f64(-p(60)),
            Dd::from_f64(-0.75),
        ),
        (
            Dd::new(-p(60), -0.5),
            Dd::new(-p(60), -1.0),
            Dd::from_f64(-p(60)),
            Dd::from_f64(-p(60)),
            Dd::new(-p(60), -1.0),
            Dd::from_f64(-0.5),
        ),
        (
            Dd::new(0.5, -p(-60)),
            Dd::ZERO,
            Dd::ONE,
            Dd::ZERO,
            Dd::ZERO,
            Dd::new(0.5, -p(-60)),
        ),
        (
            Dd::new(-0.5, p(-60)),
            -Dd::ONE,
            Dd::from_f64(-0.0),
            Dd::from_f64(-0.0),
            Dd::from_f64(-0.0),
            Dd::new(-0.5, p(-60)),
        ),
        (
            Dd::new(p(106), 0.5),
            Dd::from_f64(p(106)),
            Dd::new(p(106), 1.0),
            Dd::from_f64(p(106)),
            Dd::new(p(106), 1.0),
            Dd::from_f64(0.5),
        ),
    ];
    for (x, fl, ce, tr, ro, fr) in cases {
        assert!(
            same(Float::floor(x), fl),
            "floor {x:?}: {:?}",
            Float::floor(x)
        );
        assert!(same(Float::ceil(x), ce), "ceil {x:?}: {:?}", Float::ceil(x));
        assert!(
            same(Float::trunc(x), tr),
            "trunc {x:?}: {:?}",
            Float::trunc(x)
        );
        assert!(
            same(Float::round(x), ro),
            "round {x:?}: {:?}",
            Float::round(x)
        );
        assert!(
            same(Float::fract(x), fr),
            "fract {x:?}: {:?}",
            Float::fract(x)
        );
    }
    // fract of huge values and of negative integers is +0, as for f64.
    for x in [1e300, -1e300, -3.0, -0.0, f64::MAX] {
        assert!(is_f64(Float::fract(Dd::from_f64(x)), x.fract()), "{x}");
    }
    assert!(Float::fract(Dd::INFINITY).is_nan());
}

#[test]
fn signs_max_min_and_clamp() {
    let d = Dd::from_f64;
    assert!(is_f64(Float::signum(d(0.0)), 1.0));
    assert!(is_f64(Float::signum(d(-0.0)), -1.0));
    assert!(is_f64(Float::signum(Dd::new(1.0, -1e-30)), 1.0));
    assert!(Float::signum(Dd::NAN).is_nan());
    assert!(same(
        Float::copysign(Dd::new(1.0, 1e-20), d(-0.0)),
        Dd::new(-1.0, -1e-20)
    ));
    assert!(Float::copysign(Dd::NAN, d(-1.0)).hi().is_sign_negative());
    assert!(same(Float::abs(Dd::new(-1.0, 1e-20)), Dd::new(1.0, -1e-20)));
    assert!(is_f64(Float::abs(d(-0.0)), 0.0));
    // max and min compare whole double-doubles, ignore a NaN argument, and of two zeros give
    // +0 and −0 respectively unless both are equal (IEEE 754-2019), in either order.
    let (a, b) = (Dd::new(1.0, 1e-20), Dd::new(1.0, -1e-20));
    assert!(same(Float::max(a, b), a) && same(Float::max(b, a), a));
    assert!(same(Float::min(a, b), b) && same(Float::min(b, a), b));
    for (x, y, max, min) in [
        (0.0, -0.0, 0.0, -0.0),
        (-0.0, 0.0, 0.0, -0.0),
        (-0.0, -0.0, -0.0, -0.0),
        (0.0, 0.0, 0.0, 0.0),
        (f64::NAN, 1.0, 1.0, 1.0),
        (1.0, f64::NAN, 1.0, 1.0),
        (f64::NAN, -0.0, -0.0, -0.0),
    ] {
        assert!(is_f64(Float::max(d(x), d(y)), max), "max({x}, {y})");
        assert!(is_f64(Float::min(d(x), d(y)), min), "min({x}, {y})");
    }
    assert!(Float::max(Dd::NAN, Dd::NAN).is_nan());
    // clamp follows f64 for valid bounds, and gives NaN (instead of panicking) for invalid ones.
    assert!(same(
        Float::clamp(Dd::new(2.0, 1e-16), Dd::ONE, Dd::new(2.0, 1e-17)),
        Dd::new(2.0, 1e-17)
    ));
    assert!(is_f64(Float::clamp(d(-0.0), d(0.0), d(1.0)), -0.0));
    assert!(Float::clamp(Dd::NAN, d(0.0), d(1.0)).is_nan());
    assert!(Float::clamp(d(0.5), d(1.0), d(0.0)).is_nan());
    assert!(Float::clamp(d(0.5), Dd::NAN, d(1.0)).is_nan());
    assert!(Float::clamp(d(0.5), d(0.0), Dd::NAN).is_nan());
    // abs_sub is fdim: +0 unless self > other, NaN propagates.
    assert!(same(
        Float::abs_sub(d(3.0), Dd::new(1.0, 1e-20)),
        Dd::new(2.0, -1e-20)
    ));
    assert!(is_f64(Float::abs_sub(d(1.0), d(3.0)), 0.0));
    assert!(is_f64(Float::abs_sub(d(-0.0), d(0.0)), 0.0));
    assert!(is_f64(Float::abs_sub(Dd::INFINITY, Dd::INFINITY), 0.0));
    assert!(Float::abs_sub(Dd::NAN, d(1.0)).is_nan());
    assert!(Float::abs_sub(d(1.0), Dd::NAN).is_nan());
    assert!(is_f64(
        Float::abs_sub(d(f64::MAX), d(-f64::MAX)),
        f64::INFINITY
    ));
}

#[test]
fn mul_add_follows_f64_at_the_edges() {
    let d = Dd::from_f64;
    let cases = [
        (f64::MAX, 2.0, -f64::MAX), // the product overflows, the sum does not
        (-f64::MAX, 2.0, f64::MAX),
        (f64::MAX, 4.0, -f64::MAX), // 3·MAX: overflows
        (1e300, 1e300, 1.0),
        (f64::MAX, f64::MAX, f64::NEG_INFINITY),
        (f64::MAX, -f64::MAX, f64::INFINITY),
        (1e-200, -1e-200, 0.0), // the product underflows to -0
        (1e-200, 1e-200, -0.0),
        (-1e-200, 1e-200, -0.0),
        (0.0, -1.0, 0.0),
        (-0.0, 1.0, -0.0),
        (0.0, -1.0, -0.0),
        (2.0, -3.0, 6.0), // exact zero: +0
        (f64::INFINITY, 0.0, 1.0),
        (f64::INFINITY, 1.0, f64::NEG_INFINITY),
        (2.0, 3.0, f64::INFINITY),
        (f64::NAN, 0.0, 1.0),
        (1.0, 1.0, f64::NAN),
        (two_pow(-600), two_pow(-600), two_pow(-1074)),
        (two_pow(-600), -two_pow(-475), two_pow(-1074)),
        (1e-160, 1e-160, -1e-320),
        (3e-170, -5e-160, 1e-310),
        (1e-300, 1e-300, 1e-300),
        (f64::MIN_POSITIVE, -5e-324, 0.0), // underflows even at the internal scale
        (-1e-300, 1e-300, 0.0),
        (1e-300, -1e-300, -5e-324),
    ];
    for (x, y, z) in cases {
        let want = x.mul_add(y, z);
        let got = Float::mul_add(d(x), d(y), d(z));
        assert!(
            is_f64(got, want),
            "mul_add({x:e}, {y:e}, {z:e}) = {got:?}, f64 gives {want:e}"
        );
    }
    // Double-double arguments: (1 + 2^-60)·3 − 3 = 3·2^-60 has an exact product; the product
    // (1 + 2^-60)(1 − 2^-60) = 1 − 2^-120 is rounded (to within 4u²), so the result −2^-120 is
    // only accurate to 4u² + 3u²·2^-120.
    let p = two_pow;
    let got = Float::mul_add(Dd::new(1.0, p(-60)), d(3.0), d(-3.0));
    assert!(same(got, d(3.0 * p(-60))), "{got:?}");
    let got = Float::mul_add(Dd::new(1.0, p(-60)), Dd::new(1.0, -p(-60)), -Dd::ONE);
    assert!((got + d(p(-120))).abs().hi() <= 4.0 * U2, "{got:?}");
}

#[test]
fn degrees_and_radians() {
    let d = Dd::from_f64;
    let rad = Float::to_radians(d(180.0));
    assert!((rad - Dd::PI()).abs().hi() <= Dd::PI().hi() * TOL);
    let deg = Float::to_degrees(Dd::FRAC_PI_6());
    assert!((deg - d(30.0)).abs().hi() <= 30.0 * TOL);
    assert!(is_f64(Float::to_degrees(d(-0.0)), -0.0));
    assert!(is_f64(Float::to_radians(d(-0.0)), -0.0));
    assert!(is_f64(
        Float::to_degrees(Dd::NEG_INFINITY),
        f64::NEG_INFINITY
    ));
    assert!(Float::to_radians(Dd::NAN).is_nan());
    // No intermediate overflow: MAX in radians is finite, in degrees it overflows (as f64).
    assert!(is_f64(
        Float::to_degrees(d(f64::MAX)),
        f64::MAX.to_degrees()
    ));
    let r = Float::to_radians(d(f64::MAX));
    assert!(r.hi().is_finite() && (r.hi() - f64::MAX.to_radians()).abs() <= 2.0 * ulp(r.hi()));
    // f64's to_degrees multiplies by the rounded 180/π, so allow it two ulps.
    let big = Float::to_degrees(d(1e306));
    assert!((big.hi() - 1e306f64.to_degrees()).abs() <= 2.0 * ulp(big.hi()));
}

/// A `Float` method and its `f64` counterpart.
type Unary = (&'static str, fn(Dd) -> Dd, fn(f64) -> f64);

/// Doubles from the whole range, with special values and half-integers over-represented.
fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        any::<u64>().prop_map(f64::from_bits),
        prop::sample::select(vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.5,
            -2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            4503599627370495.5,
            -4503599627370495.5,
            9007199254740993.0,
            1e300,
            -1e-300,
            0.49999999999999994,
            -0.49999999999999994,
            4503599627370497.0,
            2147483647.5,
            -2147483648.5,
            9.223372036854775e18,
            -9.223372036854776e18,
            1.8446744073709552e19,
            1.7014118346046923e38,
            -1.7014118346046923e38,
            3.402823669209385e38,
        ]),
        (-1e6..1e6f64).prop_map(|x| (x * 2.0).round() / 2.0),
        -1e3..1e3f64,
        // Next to the limits of the integer types.
        (
            prop::sample::select(vec![31, 32, 63, 64, 127, 128]),
            -2i64..=2,
            any::<bool>()
        )
            .prop_map(|(e, k, neg)| {
                let x = two_pow(e) + k as f64 * ulp(two_pow(e));
                if neg { -x } else { x }
            }),
    ]
}

/// Doubles halfway between two adjacent `f32`s (normal and subnormal), or just next to them.
fn f32_ties() -> impl Strategy<Value = (f64, f64)> {
    let f = prop_oneof![
        any::<u32>().prop_map(|b| f32::from_bits(b & 0x7fff_ffff)),
        (1u32..1 << 23).prop_map(f32::from_bits), // subnormal
    ];
    (
        f.prop_filter("finite, below MAX", |x| x.is_finite() && *x < f32::MAX),
        -2i32..=2,
    )
        .prop_map(|(f, k)| {
            let next = f32::from_bits(f.to_bits() + 1);
            let mid = (f as f64 + next as f64) / 2.0; // exact
            (mid, k as f64 * ulp(mid) / 4.0)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn float_semantics_match_f64(x in prop::num::f64::NORMAL | prop::num::f64::ZERO) {
        let d = Dd::from_f64(x);
        prop_assert_eq!(Float::floor(d).hi(), x.floor());
        prop_assert_eq!(Float::ceil(d).hi(), x.ceil());
        prop_assert_eq!(Float::round(d).hi(), x.round());
        prop_assert_eq!(Float::trunc(d).hi(), x.trunc());
        prop_assert_eq!(Float::fract(d).hi(), x.fract());
        prop_assert_eq!(Float::abs(d).hi(), x.abs());
        prop_assert_eq!(Float::signum(d).hi(), x.signum());
        prop_assert_eq!(Float::classify(d), x.classify());
        prop_assert_eq!(Float::is_sign_negative(d), x.is_sign_negative());
        prop_assert_eq!(Float::is_finite(d), x.is_finite());
    }

    #[test]
    fn unary_float_methods_match_f64_bitwise(x in any_f64()) {
        let d = Dd::from_f64(x);
        let unary: [Unary; 7] = [
            ("floor", Float::floor, f64::floor),
            ("ceil", Float::ceil, f64::ceil),
            ("round", Float::round, f64::round),
            ("trunc", Float::trunc, f64::trunc),
            ("fract", Float::fract, f64::fract),
            ("abs", Float::abs, f64::abs),
            ("signum", Float::signum, f64::signum),
        ];
        for (name, f, g) in unary {
            prop_assert!(is_f64(f(d), g(x)), "{}({:?}) = {:?}, f64 gives {:?}", name, x, f(d), g(x));
        }
        prop_assert_eq!(Float::classify(d), x.classify());
        prop_assert_eq!(Float::is_nan(d), x.is_nan());
        prop_assert_eq!(Float::is_infinite(d), x.is_infinite());
        prop_assert_eq!(Float::is_finite(d), x.is_finite());
        prop_assert_eq!(Float::is_normal(d), x.is_normal());
        prop_assert_eq!(Float::is_subnormal(d), x.is_subnormal());
        prop_assert_eq!(Float::is_sign_positive(d), x.is_sign_positive());
        prop_assert_eq!(Float::is_sign_negative(d), x.is_sign_negative());
        prop_assert_eq!(Float::integer_decode(d), Float::integer_decode(x));
        prop_assert_eq!(<Dd as Zero>::is_zero(&d), x == 0.0);
        // Conversions to every primitive type agree with f64's.
        prop_assert_eq!(d.to_i8(), x.to_i8());
        prop_assert_eq!(d.to_i16(), x.to_i16());
        prop_assert_eq!(d.to_i32(), x.to_i32());
        prop_assert_eq!(d.to_i64(), x.to_i64());
        prop_assert_eq!(d.to_i128(), x.to_i128());
        prop_assert_eq!(d.to_isize(), x.to_isize());
        prop_assert_eq!(d.to_u8(), x.to_u8());
        prop_assert_eq!(d.to_u16(), x.to_u16());
        prop_assert_eq!(d.to_u32(), x.to_u32());
        prop_assert_eq!(d.to_u64(), x.to_u64());
        prop_assert_eq!(d.to_u128(), x.to_u128());
        prop_assert_eq!(d.to_usize(), x.to_usize());
        prop_assert_eq!(d.to_f64().map(f64::to_bits), Some(x.to_bits()));
        let f = d.to_f32().unwrap();
        prop_assert!(f.to_bits() == (x as f32).to_bits() || (f.is_nan() && x.is_nan()));
        prop_assert!(is_f64(<Dd as NumCast>::from(x).unwrap(), x));
    }

    #[test]
    fn binary_float_methods_match_f64_bitwise(x in any_f64(), y in any_f64(), z in any_f64()) {
        let (dx, dy, dz) = (Dd::from_f64(x), Dd::from_f64(y), Dd::from_f64(z));
        // max and min ignore a NaN argument, as documented for f64 (aarch64's f64::max and
        // f64::min instead return NaN for a signaling NaN, which no double-double operation
        // produces), and of two zeros follow IEEE 754-2019 (f64 leaves that unspecified).
        let (max, min) = match (x.is_nan(), y.is_nan()) {
            (true, _) => (y, y),
            (_, true) => (x, x),
            _ if x == 0.0 && y == 0.0 => (
                if x.is_sign_negative() && y.is_sign_negative() { -0.0 } else { 0.0 },
                if x.is_sign_positive() && y.is_sign_positive() { 0.0 } else { -0.0 },
            ),
            _ => (x.max(y), x.min(y)),
        };
        prop_assert!(is_f64(Float::max(dx, dy), max), "max({:?}, {:?})", x, y);
        prop_assert!(is_f64(Float::min(dx, dy), min), "min({:?}, {:?})", x, y);
        // abs_sub: the exact difference (which f64 rounds), or +0, or NaN.
        let (got, want) = (Float::abs_sub(dx, dy), Float::abs_sub(x, y));
        if x > y {
            prop_assert!(same(got, dx - dy) && got.hi().to_bits() == want.to_bits(), "abs_sub({:?}, {:?})", x, y);
        } else {
            prop_assert!(is_f64(got, want), "abs_sub({:?}, {:?})", x, y);
        }
        prop_assert!(is_f64(Float::copysign(dx, dy), x.copysign(y)), "copysign({:?}, {:?})", x, y);
        let (lo, hi) = (y.min(z), y.max(z));
        if lo <= hi {
            prop_assert!(is_f64(Float::clamp(dx, Dd::from_f64(lo), Dd::from_f64(hi)), x.clamp(lo, hi)));
        }
        // mul_add: special values, signed zeros and overflow as f64; finite nonzero results
        // within an ulp of the fused f64 result (f64 products are exact in double-double).
        let want = x.mul_add(y, z);
        let got = Float::mul_add(dx, dy, dz);
        if want.is_nan() || want == 0.0 || want.is_infinite() {
            prop_assert!(is_f64(got, want), "mul_add({:?}, {:?}, {:?}) = {:?}, f64 gives {:?}", x, y, z, got, want);
        } else {
            prop_assert!(
                got.hi().is_finite() && (got.hi() - want).abs() <= ulp(want)
                    && got.hi().is_sign_negative() == want.is_sign_negative(),
                "mul_add({:?}, {:?}, {:?}) = {:?}, f64 gives {:?}", x, y, z, got, want
            );
        }
    }

    #[test]
    fn to_f32_rounds_ties_by_the_trailing_word((mid, lo) in f32_ties(), neg in any::<bool>()) {
        let x = if neg { -Dd::new(mid, lo) } else { Dd::new(mid, lo) };
        let (down, up) = ((mid - ulp(mid)) as f32, (mid + ulp(mid)) as f32);
        let want = match lo.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => up,
            Some(std::cmp::Ordering::Less) => down,
            _ => mid as f32, // ties to even
        };
        let want = if neg { -want } else { want };
        prop_assert_eq!(x.to_f32().map(f32::to_bits), Some(want.to_bits()), "{:?}", x);
    }

    #[test]
    fn rounding_double_doubles_is_consistent(x in wide_dd()) {
        let (fl, ce, tr, ro, fr) = (Float::floor(x), Float::ceil(x), Float::trunc(x), Float::round(x), Float::fract(x));
        prop_assert!(fl <= x && x <= ce, "{:?}", x);
        prop_assert!(Float::floor(fl) == fl && Float::ceil(ce) == ce, "{:?}", x);
        if fl == x {
            prop_assert!(ce == x && tr == x && ro == x && fr.hi() == 0.0, "{:?}", x);
        } else {
            prop_assert!(ce == fl + 1.0, "{:?}: {:?} {:?}", x, fl, ce);
            let towards_zero = if x.hi() < 0.0 { ce } else { fl };
            prop_assert!(tr == towards_zero);
            let up = x - fl; // exact: x and its floor share their leading binades
            let want = if up > Dd::from_f64(0.5) || (up == Dd::from_f64(0.5) && x.hi() > 0.0) { ce } else { fl };
            prop_assert!(ro == want, "{:?}: round {:?}", x, ro);
            prop_assert!(fr.abs() < Dd::ONE && fr == x - tr, "{:?}: fract {:?}", x, fr);
            prop_assert_eq!(fr.hi().is_sign_negative(), x.hi() < 0.0);
        }
        for r in [fl, ce, tr, ro, fr] {
            prop_assert!(r.hi() + r.lo() == r.hi(), "{:?} is not normalised", r);
        }
        prop_assert_eq!(Float::signum(x).hi(), x.hi().signum());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn complex_identities(
        (ar, ai, br, bi) in (-1e100..1e100f64, -1e100..1e100f64, -1e50..1e50f64, -1e50..1e50f64),
        (ea, eb) in (-30..30i32, -30..30i32),
    ) {
        let z = Complex::new(Dd::from_f64(ar * two_pow(ea)), Dd::from_f64(ai));
        let w = Complex::new(Dd::from_f64(br), Dd::from_f64(bi * two_pow(eb)));
        prop_assume!(z.norm_sqr().hi() > 0.0 && w.norm_sqr().hi() > 0.0);
        let back = (z * w) / w;
        prop_assert!((back - z).norm().hi() <= 2.0 * TOL * z.norm().hi(), "{:?}", back);
        // sqrt errs by at most 6.2·2^-100 norm-wise (see complex_sqrt_matches_reference), and
        // squaring doubles that.
        let s = z.sqrt();
        prop_assert!((s * s - z).norm().hi() <= 16.0 * TOL * z.norm().hi(), "{:?}", s);
    }
}
