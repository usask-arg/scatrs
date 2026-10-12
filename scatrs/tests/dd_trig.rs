#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::{Record, find, load};
use proptest::prelude::*;
use scatrs::scalar::DoubleDouble as Dd;

/// 2^-100.
const TOL: f64 = 7.888_609_052_210_118e-31;
/// The smallest positive subnormal, 2^-1074.
const TINY: f64 = 5e-324;
/// 2^50, the largest |x| accepted by sin, cos and tan.
const MAX_REDUCIBLE: f64 = 1_125_899_906_842_624.0;

/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

// Double-double constants rounded to nearest; all four lie above the true values.
fn pi() -> Dd {
    Dd::new(std::f64::consts::PI, 1.2246467991473532e-16)
}
fn frac_pi_2() -> Dd {
    Dd::new(std::f64::consts::FRAC_PI_2, 6.123233995736766e-17)
}
fn frac_pi_4() -> Dd {
    Dd::new(std::f64::consts::FRAC_PI_4, 3.061616997868383e-17)
}
fn frac_3pi_4() -> Dd {
    Dd::new(2.356194490192345, 9.184850993605148e-17)
}

fn dd_input(r: &Record, prefix: &str) -> Vec<Dd> {
    let hi = r.input_f64s(&format!("{prefix}_hi"));
    let lo = r.input_f64s(&format!("{prefix}_lo"));
    hi.iter().zip(&lo).map(|(&h, &l)| Dd::new(h, l)).collect()
}

/// The arguments of one reference record, parsed once: `x`, and `y` for atan2.
struct Inputs {
    x: Vec<Dd>,
    y: Vec<Dd>,
}

impl Inputs {
    fn parse(r: &Record) -> Self {
        Self {
            x: dd_input(r, "x"),
            y: if r.inputs.contains_key("y_hi") {
                dd_input(r, "y")
            } else {
                Vec::new()
            },
        }
    }
}

/// Asserts |got − ref| <= abs_tol(inputs, i, ref) for every case of `case` and of its
/// edge-case record `{case}_edges`, measuring the error against the three-word reference.
fn check(
    case: &str,
    got: impl Fn(&Inputs, usize) -> Dd,
    abs_tol: impl Fn(&Inputs, usize, Dd) -> f64,
) {
    let recs = load("dd_trig.jsonl");
    for name in [case.to_owned(), format!("{case}_edges")] {
        let r = find(&recs, &name);
        let inputs = Inputs::parse(r);
        let v = r.output("value");
        let mut worst = 0.0_f64;
        for i in 0..v.hi.len() {
            let exact = Dd::new(v.hi[i], v.lo[i]);
            let g = got(&inputs, i);
            let err = v.dd_error(i, g.hi(), g.lo());
            let t = abs_tol(&inputs, i, exact);
            worst = worst.max(err / t);
            assert!(
                err <= t,
                "{name} case {i}: got {g:?}, reference {exact:?}, error {err:e} > {t:e}"
            );
        }
        eprintln!("{name}: worst error = {worst:.3} x tolerance");
    }
}

/// Relative for |x| <= π/4 (no reduction), absolute otherwise.
fn sin_cos_tol(inp: &Inputs, i: usize, exact: Dd) -> f64 {
    if inp.x[i].hi().abs() <= std::f64::consts::FRAC_PI_4 {
        TOL * exact.abs().hi()
    } else {
        TOL
    }
}

#[test]
fn sin() {
    check("sin", |inp, i| inp.x[i].sin(), sin_cos_tol);
}

#[test]
fn cos() {
    check("cos", |inp, i| inp.x[i].cos(), sin_cos_tol);
}

#[test]
fn sin_and_cos_are_relatively_accurate_next_to_multiples_of_half_pi() {
    // The documented reduction floor: next to kπ/2 the error is at most 2^-100 |f| plus
    // |k| 2^-161 for f64 arguments (|k| 2^-158 for double-double ones), so the error is
    // within 2^-100 relative wherever |x - kπ/2| >= |k| 2^-60 (|k| 2^-57). Checked on every
    // reduced argument, including the closest doubles to multiples of π/2 below 2^50.
    let recs = load("dd_trig.jsonl");
    for case in ["sin", "sin_edges", "cos", "cos_edges"] {
        let r = find(&recs, case);
        let xs = dd_input(r, "x");
        let v = r.output("value");
        let (mut worst_floor, mut worst_rel) = (0.0_f64, 0.0_f64);
        for (i, &x) in xs.iter().enumerate() {
            if x.hi().abs() <= std::f64::consts::FRAC_PI_4 {
                continue; // not reduced: relative 2^-100 (tested above)
            }
            let g = if case.starts_with("sin") {
                x.sin()
            } else {
                x.cos()
            };
            let err = v.dd_error(i, g.hi(), g.lo());
            let f = v.hi[i].abs();
            let k = (x.hi() * std::f64::consts::FRAC_2_PI).round();
            let (floor, near) = if x.lo() == 0.0 {
                (2f64.powi(-161), 2f64.powi(-60))
            } else {
                (2f64.powi(-158), 2f64.powi(-57))
            };
            let t = TOL * f + k.abs() * floor;
            worst_floor = worst_floor.max(err / t);
            assert!(
                err <= t,
                "{case} case {i}: {x:?} gives {g:?}, error {err:e} > {t:e}"
            );
            if (x - frac_pi_2() * k).abs().hi() >= k.abs() * near {
                worst_rel = worst_rel.max(err / (TOL * f));
                assert!(
                    err <= TOL * f,
                    "{case} case {i}: {x:?} gives {g:?}, error {err:e}"
                );
            }
        }
        eprintln!(
            "{case}: worst error = {worst_floor:.3} x (2^-100 |f| + |k| floor); \
             {worst_rel:.3} x 2^-100 |f| away from the multiples"
        );
    }
}

#[test]
fn sin_cos_agrees_with_sin_and_cos() {
    let recs = load("dd_trig.jsonl");
    for case in ["sin", "sin_edges"] {
        for x in dd_input(find(&recs, case), "x") {
            let (s, c) = x.sin_cos();
            assert!(same(s, x.sin()) && same(c, x.cos()), "{x:?}: {s:?} {c:?}");
        }
    }
}

#[test]
fn tan() {
    // 2^-99 relative for |x| <= π/4 (sin and cos are both relative there), and otherwise
    // 2^-99 (1 + |tan x|) / |cos x|. The planned |cos(x.hi)| differs from |cos x| where the
    // low word matters (by 10% near 2^50 with |x.lo| = 0.06, and from 6e-17 to 1.5e-33 at the
    // double-double nearest π/2), so the tighter of it and the exact |cos x| = 1/√(1 + tan² x)
    // is used.
    check(
        "tan",
        |inp, i| inp.x[i].tan(),
        |inp, i, exact| {
            let t = exact.abs().hi();
            if inp.x[i].hi().abs() <= std::f64::consts::FRAC_PI_4 {
                return 2.0 * TOL * t;
            }
            let inv_cos = (1.0 / inp.x[i].hi().cos().abs()).min(t.hypot(1.0));
            2.0 * TOL * (1.0 + t) * inv_cos
        },
    );
}

#[test]
fn inverse_functions() {
    for (case, f) in [
        ("atan", Dd::atan as fn(Dd) -> Dd),
        ("asin", Dd::asin),
        ("acos", Dd::acos),
    ] {
        check(
            case,
            |inp, i| f(inp.x[i]),
            |_, _, exact| 2.0 * TOL * exact.abs().hi(),
        );
    }
}

#[test]
fn atan2() {
    check(
        "atan2",
        |inp, i| inp.y[i].atan2(inp.x[i]),
        |_, _, exact| 2.0 * TOL * exact.abs().hi(),
    );
}

#[test]
fn special_values() {
    let d = Dd::from_f64;
    let pi = Dd::new(std::f64::consts::PI, 1.2246467991473532e-16);
    let half_pi = Dd::new(std::f64::consts::FRAC_PI_2, 6.123233995736766e-17);
    assert_eq!(d(0.0).sin(), d(0.0));
    assert_eq!(d(0.0).cos(), Dd::ONE);
    assert!(Dd::NAN.sin().hi().is_nan());
    assert!(Dd::INFINITY.cos().hi().is_nan());
    assert!(d(2f64.powi(51)).sin().hi().is_nan());
    assert_eq!(d(0.0).atan2(d(-1.0)), pi);
    assert_eq!(d(1.0).atan2(d(0.0)), half_pi);
    let neg_zero = d(-0.0).atan2(d(1.0));
    assert!(neg_zero.hi() == 0.0 && neg_zero.hi().is_sign_negative());
    assert_eq!(d(1.0).asin(), half_pi);
    assert_eq!(d(-1.0).acos(), pi);
    assert!(d(2.0).acos().hi().is_nan());
    assert_eq!(Dd::INFINITY.atan(), half_pi);
}

// ---- Agreement with `f64` at special values; signed zeros ----------------------------------

/// Bitwise equality of both words (distinguishes -0 from +0).
fn same(a: Dd, b: Dd) -> bool {
    (a.hi().is_nan() && b.hi().is_nan())
        || (a.hi().to_bits() == b.hi().to_bits() && a.lo().to_bits() == b.lo().to_bits())
}

/// Identical words, except that the low word of a nonzero value may be a zero of either sign
/// (it carries no information then; zero results are checked separately to be `(±0, +0)`).
fn same_value(a: Dd, b: Dd) -> bool {
    same(a, b)
        || (a.hi() != 0.0 && a.hi().to_bits() == b.hi().to_bits() && a.lo() == 0.0 && b.lo() == 0.0)
}

fn next_up(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    f64::from_bits(x.to_bits() + 1)
}

fn next_down(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    f64::from_bits(x.to_bits() - 1)
}

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == f64::MAX {
        return a - f64::from_bits(a.to_bits() - 1);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// Checks a double-double result against the `f64` result for the same input: NaN exactly
/// when `want` is NaN; bitwise equal (with a `+0` low word) when `want` is zero or infinite;
/// otherwise within 2^-46 relative, or 2^-1072 absolute for subnormal results.
///
/// This catches wrong special values, signs, quadrants and garbage; accuracy is tested against
/// Arb above. The 2^-46 (tens of ulps) allows for platform libm errors of a few ulps.
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

const UNARY: [Unary; 6] = [
    ("sin", Dd::sin, f64::sin),
    ("cos", Dd::cos, f64::cos),
    ("tan", Dd::tan, f64::tan),
    ("atan", Dd::atan, f64::atan),
    ("asin", Dd::asin, f64::asin),
    ("acos", Dd::acos, f64::acos),
];

const SPECIAL: [f64; 21] = [
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    TINY,
    -TINY,
    f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE,
    1e-300,
    -1e-300,
    0.785,
    -0.786,
    std::f64::consts::FRAC_PI_2,
    -std::f64::consts::PI,
];

#[test]
fn special_values_and_signed_zeros_match_f64() {
    // For example sin(-0) = -0, tan(-0) = -0, atan(-0) = -0, asin(-0) = -0, acos(1) = +0,
    // atan(±inf) = ±π/2 and asin(2) = NaN.
    for (name, dd_fn, f64_fn) in UNARY {
        for x in SPECIAL {
            agrees(&format!("{name}({x:?})"), dd_fn(Dd::from_f64(x)), f64_fn(x)).unwrap();
        }
    }
    for x in SPECIAL {
        let (s, c) = Dd::from_f64(x).sin_cos();
        let (fs, fc) = x.sin_cos();
        agrees(&format!("sin_cos({x:?}).0"), s, fs).unwrap();
        agrees(&format!("sin_cos({x:?}).1"), c, fc).unwrap();
    }
}

#[test]
fn atan2_special_values_match_f64() {
    // Every combination of signed zeros, infinities, NaN and finite values, including the
    // extremes; exact results (0, ±π/4, ±π/2, ±3π/4, ±π) are the double-double constants.
    let vals = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -1.0,
        TINY,
        -TINY,
        f64::MAX,
        -f64::MAX,
        1e-300,
        -1e300,
        3.0,
    ];
    let d = Dd::from_f64;
    for y in vals {
        for x in vals {
            let got = d(y).atan2(d(x));
            let want = y.atan2(x);
            agrees(&format!("atan2({y:?}, {x:?})"), got, want).unwrap();
            let exact = [frac_pi_4(), frac_pi_2(), frac_3pi_4(), pi()]
                .into_iter()
                .find(|c| c.hi() == want.abs());
            if let Some(c) = exact {
                if y.is_infinite() || x.is_infinite() || y == 0.0 || x == 0.0 {
                    let c = if want < 0.0 { -c } else { c };
                    assert!(same(got, c), "atan2({y:?}, {x:?}) = {got:?}, want {c:?}");
                }
            }
        }
    }
}

#[test]
fn zero_results_keep_their_sign_with_a_positive_zero_low_word() {
    let d = Dd::from_f64;
    for (name, f) in [
        ("sin", Dd::sin as fn(Dd) -> Dd),
        ("tan", Dd::tan),
        ("atan", Dd::atan),
        ("asin", Dd::asin),
        ("sin_cos.0", |x: Dd| x.sin_cos().0),
    ] {
        for z in [0.0, -0.0] {
            let got = f(d(z));
            assert!(
                got.hi().to_bits() == z.to_bits() && got.lo().to_bits() == 0,
                "{name}({z:?}) = {got:?}"
            );
        }
    }
    for (y, x) in [
        (-0.0, 1.0),
        (-0.0, f64::INFINITY),
        (-0.0, 0.0),
        (-1e-300, 1e300),
        (-TINY, f64::MAX),
        (-TINY, 3.0),
        (0.0, 1.0),
        (1e-300, 1e300),
    ] {
        let got = d(y).atan2(d(x));
        let want = y.atan2(x);
        assert!(
            got.hi().to_bits() == want.to_bits() && got.lo().to_bits() == 0,
            "atan2({y:?}, {x:?}) = {got:?}, f64 gives {want:?}"
        );
    }
    assert!(same(Dd::ONE.acos(), Dd::ZERO));
    assert!(same(d(-1.0).asin(), -frac_pi_2()));
}

#[test]
fn reduction_limit_is_two_to_the_fifty() {
    // NaN exactly for |x| > 2^50, as a double-double value: (2^50, -0.05) is inside and
    // (2^50, 0.1) outside, although both have the leading word 2^50.
    let (below, above) = (Dd::new(MAX_REDUCIBLE, -0.05), Dd::new(MAX_REDUCIBLE, 0.1));
    assert!(below.hi() == MAX_REDUCIBLE && below.lo() < 0.0, "{below:?}");
    assert!(above.hi() == MAX_REDUCIBLE && above.lo() > 0.0, "{above:?}");
    let inside = [
        Dd::from_f64(MAX_REDUCIBLE),
        below,
        Dd::from_f64(next_down(MAX_REDUCIBLE)),
    ];
    let outside = [
        above,
        Dd::from_f64(next_up(MAX_REDUCIBLE)),
        Dd::from_f64(1e300),
        Dd::from_f64(f64::MAX),
        Dd::INFINITY,
        Dd::NAN,
    ];
    for s in [1.0, -1.0] {
        for x in inside {
            let x = x * s;
            for (name, v) in [("sin", x.sin()), ("cos", x.cos()), ("tan", x.tan())] {
                assert!(v.hi().is_finite(), "{name}({x:?}) = {v:?}");
                assert!(v.abs() <= Dd::ONE || name == "tan", "{name}({x:?}) = {v:?}");
            }
            let (si, co) = x.sin_cos();
            assert!(same(si, x.sin()) && same(co, x.cos()));
        }
        for x in outside {
            let x = x * s;
            for (name, v) in [("sin", x.sin()), ("cos", x.cos()), ("tan", x.tan())] {
                assert!(v.hi().is_nan(), "{name}({x:?}) = {v:?}");
            }
            let (si, co) = x.sin_cos();
            assert!(si.hi().is_nan() && co.hi().is_nan());
        }
    }
}

// ---- Ranges, continuity and monotonicity at branch points ---------------------------------

/// A deterministic xorshift generator of uniform doubles in [0, 1).
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn inverse_functions_stay_within_their_ranges_next_to_the_limits() {
    // Results within an ulp of the low word of ±π/2 or ±π: the double-double constants lie
    // above the true values, so rounding errors could otherwise step past them.
    let mut next = uniform(0x2545_f491_4f6c_dd1d);
    for _ in 0..100_000 {
        let e = -(60.0 + 1000.0 * next()); // |ratio| = 2^e, from 2^-60 to 2^-1060
        let t = Dd::new(1.0, (next() - 0.5) * 2f64.powi(-53)) * 2f64.powf(e.round());
        let one = Dd::new(1.0, (next() - 0.5) * 2f64.powi(-53));
        for s in [1.0, -1.0] {
            let r = (t * s).atan2(-one); // next to ±π
            assert!(r.abs() <= pi(), "atan2({:?}, {:?}) = {r:?}", t * s, -one);
            let r = (one * s).atan2(t); // next to ±π/2 with x > 0
            assert!(
                r.abs() <= frac_pi_2(),
                "atan2({:?}, {t:?}) = {r:?}",
                one * s
            );
            let r = (one * s).atan2(-t); // next to ±π/2 with x < 0
            assert!(r.abs() <= pi() && r.abs() >= Dd::ONE, "atan2 = {r:?}");
            let big = (one * s) / t;
            let r = big.atan();
            assert!(r.abs() <= frac_pi_2(), "atan({big:?}) = {r:?}");
        }
        let u = Dd::ONE - t * t; // 1 - tiny (|t| <= 2^-60)
        for s in [1.0, -1.0] {
            let x = u * s;
            let a = x.asin();
            assert!(a.abs() <= frac_pi_2(), "asin({x:?}) = {a:?}");
            let c = x.acos();
            assert!(c >= Dd::ZERO && c <= pi(), "acos({x:?}) = {c:?}");
        }
    }
}

#[test]
fn sin_and_cos_stay_within_one_next_to_multiples_of_half_pi() {
    // Results next to ±1 (and to 0) at multiples of π/2 of both signs, up to 2^50.
    let mut next = uniform(0x9e37_79b9_7f4a_7c15);
    for i in 0..100_000 {
        let k = if i % 2 == 0 {
            (next() * 64.0).floor() - 32.0
        } else {
            ((next() - 0.5) * 2.0 * 7.1e14).round()
        };
        let x = frac_pi_2() * k + Dd::new((next() - 0.5) * 2f64.powi(-40), 0.0) * next().powi(8);
        if x.abs() > Dd::from_f64(MAX_REDUCIBLE) {
            continue;
        }
        let (s, c) = x.sin_cos();
        assert!(
            s.abs() <= Dd::ONE && c.abs() <= Dd::ONE,
            "{x:?}: {s:?} {c:?}"
        );
        // s² + c² = 1 to within 2(|s| + |c|) TOL + rounding <= 3 TOL (from the 2^-100 bounds).
        let one = s * s + c * c;
        assert!(
            (one - 1.0).abs().hi() <= 3.0 * TOL,
            "{x:?}: s² + c² - 1 = {:?}",
            one - 1.0
        );
    }
}

#[test]
fn atan2_is_monotonic_and_continuous_across_its_branch_points() {
    // Through |y| = |x| (the octant boundaries), the fast-path limits 2^±450 and, after
    // scaling, the tiny-ratio limit 2^-968: atan2(y, x) must be non-decreasing in y for x > 0
    // in steps far above the rounding error (2^-90 relative here; the error is below 2^-99).
    for &x in &[
        1.0,
        3.0,
        1e-200,
        2f64.powi(450),
        2f64.powi(-450),
        1e300,
        5e-310,
    ] {
        let x = Dd::from_f64(x);
        for centre in [x, -x] {
            let mut prev: Option<Dd> = None;
            for j in -2000..=2000 {
                let y = centre + x * (f64::from(j) * 2f64.powi(-90)); // increasing in j
                let r = y.atan2(x);
                if let Some(p) = prev {
                    assert!(r >= p, "atan2({y:?}, {x:?}) = {r:?} < {p:?}");
                }
                prev = Some(r);
            }
        }
    }
    // Arguments straddling the fast-path limits and the tiny-ratio limit, in ulp steps.
    for (y0, x) in [
        (2f64.powi(450), 1.0),
        (2f64.powi(-450), 1.0),
        (2f64.powi(-968), 1.0),
        (2f64.powi(-968), -1.0),
        (1.0, 2f64.powi(-968)),
        (1.0, 2f64.powi(-969)), // x scales to 2^-968 below y = 1 and to 2^-969 above
        (1.0, 2f64.powi(-450)),
    ] {
        let mut prev: Option<Dd> = None;
        let mut y = y0;
        for _ in 0..64 {
            y = next_down(y);
        }
        for _ in 0..128 {
            let r = Dd::from_f64(y).atan2(Dd::from_f64(x));
            if let Some(p) = prev {
                // Non-decreasing in y for x > 0, non-increasing for x < 0.
                assert!(
                    if x > 0.0 { r >= p } else { r <= p },
                    "atan2({y:?}, {x:?}) = {r:?}, previous {p:?}"
                );
            }
            prev = Some(r);
            y = next_up(y);
        }
    }
}

#[test]
fn asin_and_acos_are_monotonic_next_to_one() {
    for s in [1.0, -1.0] {
        let mut prev: Option<(Dd, Dd)> = None;
        for k in 1..=110 {
            for frac in [2.0, 1.5, 1.25, 1.0] {
                let x = Dd::new(1.0, -frac * 2f64.powi(-k)) * s; // |x| increases towards 1
                let (a, c) = (x.asin(), x.acos());
                if let Some((pa, pc)) = prev {
                    assert!(
                        if s > 0.0 { a >= pa } else { a <= pa },
                        "asin({x:?}) = {a:?}"
                    );
                    assert!(
                        if s > 0.0 { c <= pc } else { c >= pc },
                        "acos({x:?}) = {c:?}"
                    );
                }
                prev = Some((a, c));
            }
        }
        let x = Dd::from_f64(s);
        let (a, c) = (x.asin(), x.acos());
        let (pa, pc) = prev.unwrap();
        assert!(if s > 0.0 {
            a >= pa && c <= pc
        } else {
            a <= pa && c >= pc
        });
    }
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

/// Angles within the reduction limit: anywhere up to 2^50, small, and next to k·π/2.
fn angle_dd() -> impl Strategy<Value = Dd> {
    let frac = prop_oneof![-0.5..0.5f64, Just(0.0)];
    prop_oneof![
        (-MAX_REDUCIBLE..MAX_REDUCIBLE, frac.clone()).prop_map(|(h, f)| with_lo(h, f)),
        (-10.0..10.0f64, frac).prop_map(|(h, f)| with_lo(h, f)),
        (-7.1e14..7.1e14f64, -1.0..1.0f64, 20..110i32)
            .prop_map(|(k, t, e)| { frac_pi_2() * k.round() + Dd::from_f64(t * 2f64.powi(-e)) }),
    ]
}

/// Double-doubles in [-1, 1], including values within 2^-60 .. 2^-1000 of ±1.
fn unit_interval_dd() -> impl Strategy<Value = Dd> {
    prop_oneof![
        (-1.0..1.0f64, -0.5..0.5f64).prop_map(|(h, f)| with_lo(h, f)),
        (53..1000i32, any::<bool>()).prop_map(|(k, neg)| {
            let x = Dd::new(1.0, -(2f64.powi(-k)));
            if neg { -x } else { x }
        }),
        Just(Dd::ONE),
        Just(-Dd::ONE),
    ]
}

/// Finite nonzero double-doubles with leading words of magnitude in [2^-960, 2^1020).
fn ranged_dd() -> impl Strategy<Value = Dd> {
    (1.0..2.0f64, -960..1020i32, any::<bool>(), -0.5..0.5f64).prop_map(|(m, e, neg, f)| {
        let h = m * 2f64.powi(e);
        with_lo(if neg { -h } else { h }, f)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn f64_inputs_agree_with_f64(x in prop_oneof![any_finite(), -10.0..10.0f64, -1.0..1.0f64]) {
        for (name, dd_fn, f64_fn) in UNARY {
            if name != "atan" && name != "asin" && name != "acos" && x.abs() > MAX_REDUCIBLE {
                continue;
            }
            // Where |cos x| < 2^-50 the bound on tan's error, 2^-99 (1 + |tan x|)/|cos x|, does
            // not imply 2^-46 relative.
            if name == "tan" && x.cos().abs() < 2f64.powi(-50) {
                continue;
            }
            let got = dd_fn(Dd::from_f64(x));
            let want = f64_fn(x);
            // sin and cos are absolutely accurate (2^-100) for |x| > π/4, and f64 to an ulp.
            if (name == "sin" || name == "cos") && want.abs() < 2f64.powi(-40) {
                prop_assert!((got.hi() - want).abs() <= 2f64.powi(-80), "{}({:?}) = {:?}", name, x, got);
                continue;
            }
            agrees(&format!("{name}({x:?})"), got, want).map_err(TestCaseError::fail)?;
        }
    }

    #[test]
    fn sin_cos_tan_are_bounded_odd_or_even_and_normalised(x in angle_dd()) {
        let (s, c) = x.sin_cos();
        prop_assert!(same(s, x.sin()) && same(c, x.cos()), "{:?}", x);
        prop_assert!(s.abs() <= Dd::ONE && c.abs() <= Dd::ONE, "{:?}: {:?} {:?}", x, s, c);
        prop_assert!(same_value((-x).sin(), -s), "sin is not odd at {:?}", x);
        prop_assert!(same_value((-x).cos(), c), "cos is not even at {:?}", x);
        let t = x.tan();
        prop_assert!(same_value((-x).tan(), -t), "tan is not odd at {:?}", x);
        for v in [s, c, t] {
            prop_assert!(v.hi() + v.lo() == v.hi(), "{:?}: {:?} is not normalised", x, v);
        }
        let one = s * s + c * c;
        prop_assert!((one - 1.0).abs().hi() <= 3.0 * TOL, "{:?}: s² + c² = {:?}", x, one);
    }

    #[test]
    fn inverse_functions_are_bounded_odd_and_normalised(
        x in wide_dd(),
        y in wide_dd(),
        u in unit_interval_dd(),
    ) {
        // atan, asin and atan2 are odd only as far as the platform's f64::atan2 is odd in y.
        let a = x.atan();
        prop_assert!(a.abs() <= frac_pi_2(), "atan {:?} = {:?}", x, a);
        prop_assert!(same_value((-x).atan(), -a), "atan is not odd at {:?}", x);
        let z = y.atan2(x);
        prop_assert!(z.abs() <= pi(), "atan2({:?}, {:?}) = {:?}", y, x, z);
        prop_assert!(same_value((-y).atan2(x), -z), "atan2 is not odd in y at ({:?}, {:?})", y, x);
        prop_assert_eq!(z.hi().is_sign_negative(), y.hi().is_sign_negative());
        if x.hi() > 0.0 {
            prop_assert!(z.abs() <= frac_pi_2(), "atan2({:?}, {:?}) = {:?}", y, x, z);
        }
        let (s, c) = (u.asin(), u.acos());
        prop_assert!(s.abs() <= frac_pi_2(), "asin {:?} = {:?}", u, s);
        prop_assert!(same_value((-u).asin(), -s), "asin is not odd at {:?}", u);
        prop_assert!(c >= Dd::ZERO && c <= pi(), "acos {:?} = {:?}", u, c);
        for v in [a, z, s, c] {
            prop_assert!(!v.hi().is_nan(), "NaN for ({:?}, {:?}, {:?})", x, y, u);
            prop_assert!(v.hi() + v.lo() == v.hi(), "{:?} is not normalised", v);
            if v.hi() == 0.0 {
                prop_assert_eq!(v.lo().to_bits(), 0);
            }
        }
        // acos(u) + acos(-u) = π: each within 2^-99 relative, so the sum within 2^-99 π, plus
        // the rounding of the sum (3u²) and of the constant π (below u²).
        let sum = c + (-u).acos();
        prop_assert!(
            ((sum - pi()) / pi()).abs().hi() <= 2.0 * TOL + 4.0 * U2,
            "acos({:?}) + acos(-u) = {:?}", u, sum
        );
    }

    #[test]
    fn results_are_nan_only_outside_the_domain(x in wide_dd(), y in wide_dd()) {
        let reducible = x.abs() <= Dd::from_f64(MAX_REDUCIBLE);
        let unit = x.abs() <= Dd::ONE;
        let (s, c) = x.sin_cos();
        let cases = [
            ("sin", x.sin(), !reducible),
            ("cos", x.cos(), !reducible),
            ("sin_cos.0", s, !reducible),
            ("sin_cos.1", c, !reducible),
            ("tan", x.tan(), !reducible),
            ("atan", x.atan(), false),
            ("asin", x.asin(), !unit),
            ("acos", x.acos(), !unit),
            ("atan2", y.atan2(x), false),
        ];
        for (name, z, nan_expected) in cases {
            prop_assert_eq!(z.hi().is_nan(), nan_expected, "{}({:?}, {:?}) = {:?}", name, x, y, z);
            prop_assert!(
                z.hi().is_nan() || z.hi() + z.lo() == z.hi(),
                "{}({:?}, {:?}) = {:?} is not normalised", name, x, y, z
            );
        }
    }

    #[test]
    fn atan2_is_scale_invariant(a in ranged_dd(), b in ranged_dd(), t in 0.0..1.0f64) {
        // atan2(y 2^k, x 2^k) = atan2(y, x): both within 2^-99 relative of the same value
        // while the arguments and the result stay in the full-precision range [2^-960, 2^1020).
        let e = |v: Dd| v.hi().abs().log2().floor() as i32;
        let (y, x) = if e(a) - e(b) >= -959 { (a, b) } else { (b, a) }; // |y/x| >= 2^-960
        let (e_min, e_max) = (e(y).min(e(x)), e(y).max(e(x)));
        let (k_lo, k_hi) = (-960 - e_min, 1019 - e_max); // k_lo <= k_hi
        let k = k_lo + (t * f64::from(k_hi - k_lo)) as i32;
        let (ys, xs) = (y.ldexp(k), x.ldexp(k));
        let (a, b) = (y.atan2(x), ys.atan2(xs));
        prop_assert!(
            (a - b).abs().hi() <= 4.0 * TOL * a.abs().hi(),
            "atan2({:?}, {:?}) = {:?} but scaled by 2^{} gives {:?}", y, x, a, k, b
        );
    }

    #[test]
    fn inverse_functions_invert(x in -1.5..1.5f64, f in -0.5..0.5f64) {
        let x = with_lo(x, f);
        // atan(tan x) = x: tan's error is amplified by 1/(1 + tan²) = cos², giving at most
        // 2^-99 (1 + |tan|) |cos| <= 2^-99 √2, plus atan's own 2^-99 |x|.
        let t = x.tan();
        let back = t.atan();
        prop_assert!((back - x).abs().hi() <= 2.0 * TOL * (std::f64::consts::SQRT_2 + x.abs().hi()),
            "atan(tan {:?}) = {:?}", x, back);
        // atan2(sin x, cos x) = x: errors e in sin and cos move the angle by at most
        // |c| e_s + |s| e_c <= √2 · 2^-100, plus atan2's own 2^-99 |x|.
        let (s, c) = x.sin_cos();
        let back = s.atan2(c);
        prop_assert!((back - x).abs().hi() <= 2.0 * TOL * (1.0 + x.abs().hi()),
            "atan2(sin, cos) of {:?} = {:?}", x, back);
        // asin(sin x) = x for |x| <= π/2: sin's error is amplified by 1/|cos x|.
        if x.abs().hi() <= std::f64::consts::FRAC_PI_2 {
            let back = s.asin();
            let tol = 2.0 * TOL * (x.abs().hi() + 1.0 / c.abs().hi());
            prop_assert!((back - x).abs().hi() <= tol, "asin(sin {:?}) = {:?}", x, back);
        }
    }
}
