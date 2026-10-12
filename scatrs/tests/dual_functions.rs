use std::num::FpCategory;

use num_complex::Complex;
use num_traits::{Float, FloatConst, One, ToPrimitive, Zero};
use proptest::prelude::*;
use scatrs::Real;
use scatrs::scalar::{DoubleDouble as Dd, Dual};

type D1 = Dual<Dd, 1>;
type D2 = Dual<Dd, 2>;

/// Central difference in double-double: truncation ~2^-60 relative, rounding ~2^-74.
fn central(f: &dyn Fn(Dd) -> Dd, x: Dd) -> Dd {
    let h = Dd::from_f64(x.hi().abs().max(1.0) * 2f64.powi(-30));
    (f(x + h) - f(x - h)) / (h * 2.0)
}

fn assert_close(got: Dd, expected: Dd, rel: f64, what: &str) {
    let err = (got - expected).abs().hi();
    let scale = expected.abs().hi();
    assert!(
        err <= rel * scale || (scale == 0.0 && err == 0.0),
        "{what}: got {got:?}, expected {expected:?}"
    );
}

macro_rules! unary {
    ($name:ident, $method:ident, [$($x:expr),* $(,)?]) => {
        #[test]
        fn $name() {
            for x in [$($x),*] {
                let xd = Dd::from_f64(x);
                let [v] = D1::variables([xd]);
                let got = Float::$method(v);
                assert_eq!(got.re(), Float::$method(xd), "value of {}({x})", stringify!($method));
                let fd = central(&|t| Float::$method(t), xd);
                assert_close(got.eps()[0], fd, 1e-15, &format!("d/dx {}({x})", stringify!($method)));
                // The f64 instantiation agrees with the double-double one.
                let [v64] = Dual::<f64, 1>::variables([x]);
                let g64 = Float::$method(v64).eps()[0];
                let gdd = got.eps()[0].hi();
                assert!((g64 - gdd).abs() <= 1e-13 * gdd.abs(), "f64 path {}({x})", stringify!($method));
            }
        }
    };
}

unary!(sqrt, sqrt, [0.7, 2.3, 1e5]);
unary!(exp, exp, [-2.0, 0.5, 10.0]);
unary!(exp2, exp2, [-3.7, 3.7]);
unary!(exp_m1, exp_m1, [1e-3, 2.0]);
unary!(ln, ln, [0.3, 3.0]);
unary!(ln_1p, ln_1p, [1e-3, 2.0]);
unary!(log2, log2, [3.7]);
unary!(log10, log10, [3.7]);
unary!(cbrt, cbrt, [-8.5, 3.0]);
unary!(sin, sin, [0.4, 2.0, -5.0]);
unary!(cos, cos, [0.4, 2.0, -5.0]);
unary!(tan, tan, [0.4, 2.0]);
unary!(asin, asin, [-0.3, 0.9]);
unary!(acos, acos, [-0.3, 0.9]);
unary!(atan, atan, [3.0, -0.2]);
unary!(sinh, sinh, [0.2, 3.0]);
unary!(cosh, cosh, [0.2, 3.0]);
unary!(tanh, tanh, [0.2, 3.0]);
unary!(asinh, asinh, [-4.0, 0.1]);
unary!(acosh, acosh, [1.5, 10.0]);
unary!(atanh, atanh, [-0.3, 0.9]);
unary!(recip, recip, [-2.5, 0.3]);
unary!(abs, abs, [-2.5, 0.3]);
unary!(fract, fract, [2.25, -1.75]);

#[test]
fn piecewise_constant_functions_have_zero_derivative() {
    let [v] = D1::variables([Dd::from_f64(2.25)]);
    let fs: [fn(D1) -> D1; 5] = [
        Float::floor,
        Float::ceil,
        Float::round,
        Float::trunc,
        Float::signum,
    ];
    for f in fs {
        assert_eq!(f(v).eps()[0], Dd::ZERO);
    }
}

#[test]
fn integer_powers() {
    let [v] = D1::variables([Dd::from_f64(1.7)]);
    for n in [5, -3, 1, 0] {
        let fd = central(&|t| t.powi(n), Dd::from_f64(1.7));
        assert_close(Float::powi(v, n).eps()[0], fd, 1e-15, &format!("powi {n}"));
    }
}

/// A binary function on `D2` and the same function on `Dd`.
type PlanBinary = (&'static str, fn(D2, D2) -> D2, fn(Dd, Dd) -> Dd);

#[test]
fn binary_functions() {
    let (x0, y0) = (Dd::from_f64(1.7), Dd::from_f64(-0.6));
    let [x, y] = D2::variables([x0, y0]);
    let cases: [PlanBinary; 5] = [
        ("powf", |a, b| a.powf(b), |a, b| a.powf(b)),
        ("hypot", |a, b| a.hypot(b), |a, b| a.hypot(b)),
        ("atan2", |a, b| a.atan2(b), |a, b| a.atan2(b)),
        ("log", |a, b| a.log(b * b + 2.0), |a, b| a.log(b * b + 2.0)),
        ("mul_add", |a, b| a.mul_add(b, a), |a, b| a.mul_add(b, a)),
    ];
    for (name, fd2, fdd) in cases {
        let got = fd2(x, y);
        assert_close(
            got.eps()[0],
            central(&|t| fdd(t, y0), x0),
            1e-15,
            &format!("d/dx {name}"),
        );
        assert_close(
            got.eps()[1],
            central(&|t| fdd(x0, t), y0),
            1e-15,
            &format!("d/dy {name}"),
        );
    }
}

#[test]
fn max_min_follow_the_chosen_branch() {
    let [x, y] = D2::variables([Dd::from_f64(1.0), Dd::from_f64(2.0)]);
    assert_eq!(x.max(y).eps(), y.eps());
    assert_eq!(x.min(y).eps(), x.eps());
}

#[test]
fn complex_of_dual_propagates_derivatives() {
    let [x, y] = Dual::<f64, 2>::variables([0.6, -1.3]);
    let z = Complex::new(x, y);
    let one = Complex::new(Dual::<f64, 2>::one(), Dual::zero());
    let f = (one / (z * z)).norm_sqr(); // 1 / (x² + y²)²
    let r2: f64 = 0.6 * 0.6 + 1.3 * 1.3;
    assert!((f.re() - 1.0 / (r2 * r2)).abs() < 1e-15);
    assert!((f.eps()[0] - (-4.0 * 0.6 / (r2 * r2 * r2))).abs() < 1e-14);
    assert!((f.eps()[1] - (-4.0 * -1.3 / (r2 * r2 * r2))).abs() < 1e-14);
    let w = Complex::new(Dual::<f64, 2>::zero(), y).exp(); // e^{iy}
    assert!((w.re.eps()[1] - (-(-1.3f64).sin())).abs() < 1e-15);
}

#[test]
fn constants_nan_detection_and_real() {
    assert_eq!(Dual::<f64, 3>::PI().re(), std::f64::consts::PI);
    assert_eq!(Dual::<f64, 3>::PI().eps(), &[0.0; 3]);
    assert!(Float::is_nan(Dual::<f64, 1>::new(1.0, [f64::NAN])));
    assert!(!Float::is_finite(Dual::<f64, 1>::new(1.0, [f64::INFINITY])));
    let c = <Dual<f64, 2> as Real>::from_f64(1.5);
    assert_eq!((c.re(), c.eps()), (1.5, &[0.0, 0.0]));
    assert_eq!(
        Real::approx_f64(Dual::<Dd, 1>::new(Dd::new(1.0, 1e-20), [Dd::ONE])),
        1.0
    );
}

#[test]
fn generic_kernel_differentiates() {
    fn horner<T: Real>(coeffs: &[f64], x: T) -> T {
        coeffs.iter().rev().fold(T::zero(), |acc, &c| acc * x + c)
    }
    let [x] = Dual::<f64, 1>::variables([2.0]);
    let p = horner(&[1.0, 2.0, 3.0], x);
    assert_eq!((p.re(), p.eps()[0]), (17.0, 14.0));
}

// ---- Shared helpers and method tables ----------------------------------------------------------

type F1 = Dual<f64, 1>;
type F3 = Dual<f64, 3>;

/// u = 2^-53.
const U: f64 = 1.110_223_024_625_156_5e-16;
/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

/// 2^n for -1074 <= n <= 1023, including subnormal powers.
fn two_pow(n: i32) -> f64 {
    if n >= -1022 {
        f64::from_bits(((n + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (n + 1074))
    }
}

/// Equal bit for bit (so with the same sign of zero), or both NaN.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Both words equal bit for bit, or both NaN.
fn same_dd(a: Dd, b: Dd) -> bool {
    (a.hi().is_nan() && b.hi().is_nan())
        || (a.hi().to_bits() == b.hi().to_bits() && a.lo().to_bits() == b.lo().to_bits())
}

/// Equal bit for bit, both NaN, or both zero (the sign of a zero derivative is unspecified).
fn same_derivative(a: f64, b: f64) -> bool {
    same(a, b) || (a == 0.0 && b == 0.0)
}

/// Zeros, normals, subnormals, extremes, infinities, NaN, and the edges of the domains.
const SPECIAL: [f64; 30] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    0.1,
    -3.5,
    7.5,
    2.0,
    -2.0,
    1.5,
    1.0 - f64::EPSILON / 2.0,
    -1.0 + f64::EPSILON / 2.0,
    1.0 + f64::EPSILON,
    std::f64::consts::FRAC_PI_2,
    100.0,
    800.0,
    -800.0,
    1e300,
    -1e-300,
    1e-310,
    -5e-324,
    f64::MIN_POSITIVE,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    -f64::NAN,
];

/// The special values as double-doubles, and double-doubles with a trailing word.
fn dd_special() -> Vec<Dd> {
    let mut v: Vec<Dd> = SPECIAL.iter().map(|&x| Dd::from_f64(x)).collect();
    v.extend([
        Dd::new(1.0, 1e-17),
        Dd::new(1.0, -1e-17),
        Dd::new(-1.0, 1e-17),
        Dd::new(0.5, 3e-18),
        Dd::new(-2.5, 1e-17),
        Dd::new(1e300, 1e283),
        Dd::FRAC_PI_2(),
        Dd::new(1e-300, -2e-317),
    ]);
    v
}

/// A unary method on `Dual<T, N>` and the matching method on `T`.
type Unary<T, const N: usize> = (&'static str, fn(Dual<T, N>) -> Dual<T, N>, fn(T) -> T);

/// Every unary `Float` method, with powers and logarithms at a few fixed second arguments.
/// `sin_cos` is compared with `T::sin` and `T::cos`: the `Dual` methods take their values from
/// `T::sin_cos`.
fn unary_methods<T: Real, const N: usize>() -> Vec<Unary<T, N>> {
    vec![
        ("floor", Float::floor, Float::floor),
        ("ceil", Float::ceil, Float::ceil),
        ("round", Float::round, Float::round),
        ("trunc", Float::trunc, Float::trunc),
        ("fract", Float::fract, Float::fract),
        ("abs", Float::abs, Float::abs),
        ("signum", Float::signum, Float::signum),
        ("recip", Float::recip, Float::recip),
        ("sqrt", Float::sqrt, Float::sqrt),
        ("cbrt", Float::cbrt, Float::cbrt),
        ("exp", Float::exp, Float::exp),
        ("exp2", Float::exp2, Float::exp2),
        ("exp_m1", Float::exp_m1, Float::exp_m1),
        ("ln", Float::ln, Float::ln),
        ("ln_1p", Float::ln_1p, Float::ln_1p),
        ("log2", Float::log2, Float::log2),
        ("log10", Float::log10, Float::log10),
        ("sin", Float::sin, Float::sin),
        ("cos", Float::cos, Float::cos),
        ("tan", Float::tan, Float::tan),
        ("sin_cos.0", |x| x.sin_cos().0, Float::sin),
        ("sin_cos.1", |x| x.sin_cos().1, Float::cos),
        ("asin", Float::asin, Float::asin),
        ("acos", Float::acos, Float::acos),
        ("atan", Float::atan, Float::atan),
        ("sinh", Float::sinh, Float::sinh),
        ("cosh", Float::cosh, Float::cosh),
        ("tanh", Float::tanh, Float::tanh),
        ("asinh", Float::asinh, Float::asinh),
        ("acosh", Float::acosh, Float::acosh),
        ("atanh", Float::atanh, Float::atanh),
        ("to_degrees", Float::to_degrees, Float::to_degrees),
        ("to_radians", Float::to_radians, Float::to_radians),
        ("powi(0)", |x| x.powi(0), |x| x.powi(0)),
        ("powi(1)", |x| x.powi(1), |x| x.powi(1)),
        ("powi(2)", |x| x.powi(2), |x| x.powi(2)),
        ("powi(3)", |x| x.powi(3), |x| x.powi(3)),
        ("powi(-1)", |x| x.powi(-1), |x| x.powi(-1)),
        ("powi(-2)", |x| x.powi(-2), |x| x.powi(-2)),
        ("powi(-3)", |x| x.powi(-3), |x| x.powi(-3)),
        ("powi(MAX)", |x| x.powi(i32::MAX), |x| x.powi(i32::MAX)),
        ("powi(MIN)", |x| x.powi(i32::MIN), |x| x.powi(i32::MIN)),
        (
            "powf(0)",
            |x| x.powf(Real::from_f64(0.0)),
            |x| x.powf(T::from_f64(0.0)),
        ),
        (
            "powf(1)",
            |x| x.powf(Real::from_f64(1.0)),
            |x| x.powf(T::from_f64(1.0)),
        ),
        (
            "powf(2)",
            |x| x.powf(Real::from_f64(2.0)),
            |x| x.powf(T::from_f64(2.0)),
        ),
        (
            "powf(3)",
            |x| x.powf(Real::from_f64(3.0)),
            |x| x.powf(T::from_f64(3.0)),
        ),
        (
            "powf(0.5)",
            |x| x.powf(Real::from_f64(0.5)),
            |x| x.powf(T::from_f64(0.5)),
        ),
        (
            "powf(1/3)",
            |x| x.powf(Real::from_f64(1.0 / 3.0)),
            |x| x.powf(T::from_f64(1.0 / 3.0)),
        ),
        (
            "powf(-0.5)",
            |x| x.powf(Real::from_f64(-0.5)),
            |x| x.powf(T::from_f64(-0.5)),
        ),
        (
            "powf(-1)",
            |x| x.powf(Real::from_f64(-1.0)),
            |x| x.powf(T::from_f64(-1.0)),
        ),
        (
            "powf(-2.5)",
            |x| x.powf(Real::from_f64(-2.5)),
            |x| x.powf(T::from_f64(-2.5)),
        ),
        (
            "powf(inf)",
            |x| x.powf(Float::infinity()),
            |x| x.powf(T::infinity()),
        ),
        (
            "2^x",
            |x| Dual::from_f64(2.0).powf(x),
            |x| T::from_f64(2.0).powf(x),
        ),
        (
            "0.5^x",
            |x| Dual::from_f64(0.5).powf(x),
            |x| T::from_f64(0.5).powf(x),
        ),
        (
            "(-2)^x",
            |x| Dual::from_f64(-2.0).powf(x),
            |x| T::from_f64(-2.0).powf(x),
        ),
        (
            "log(x, 3)",
            |x| x.log(Real::from_f64(3.0)),
            |x| x.log(T::from_f64(3.0)),
        ),
        (
            "log(3, x)",
            |x| Dual::from_f64(3.0).log(x),
            |x| T::from_f64(3.0).log(x),
        ),
    ]
}

/// A binary method on `Dual<T, N>` and the matching method on `T`.
type Binary<T, const N: usize> = (
    &'static str,
    fn(Dual<T, N>, Dual<T, N>) -> Dual<T, N>,
    fn(T, T) -> T,
);

/// Every binary `Float` method (`clamp`, which panics for `f64` on bad bounds, is tested apart).
fn binary_methods<T: Real, const N: usize>() -> Vec<Binary<T, N>> {
    vec![
        ("powf", Float::powf, Float::powf),
        ("hypot", Float::hypot, Float::hypot),
        ("atan2", Float::atan2, Float::atan2),
        ("log", Float::log, Float::log),
        ("max", Float::max, Float::max),
        ("min", Float::min, Float::min),
        ("abs_sub", Float::abs_sub, Float::abs_sub),
        ("copysign", Float::copysign, Float::copysign),
        (
            "mul_add(x, y, x)",
            |a, b| a.mul_add(b, a),
            |a, b| a.mul_add(b, a),
        ),
        (
            "mul_add(x, x, y)",
            |a, b| a.mul_add(a, b),
            |a, b| a.mul_add(a, b),
        ),
        (
            "mul_add(y, 3, x)",
            |a, b| b.mul_add(Real::from_f64(3.0), a),
            |a, b| b.mul_add(T::from_f64(3.0), a),
        ),
    ]
}

/// Piecewise-constant methods, whose derivatives are zero whatever the argument.
const PIECEWISE_CONSTANT: [&str; 6] = ["floor", "ceil", "round", "trunc", "signum", "powi(0)"];

// ---- Values follow T bit for bit; strong zeros ------------------------------------------------

#[test]
fn values_follow_f64_bit_for_bit() {
    for (name, d, f) in unary_methods::<f64, 3>() {
        for a in SPECIAL {
            for seeds in [[1.0, 0.0, -0.0], [f64::NAN, f64::INFINITY, -2.5]] {
                let got = d(F3::new(a, seeds)).re();
                assert!(
                    same(got, f(a)),
                    "{name}({a:?}) = {got:?}, f64 gives {:?}",
                    f(a)
                );
            }
        }
    }
    for (name, d, f) in binary_methods::<f64, 3>() {
        for a in SPECIAL {
            for b in SPECIAL {
                let (x, y) = (F3::new(a, [1.0, 0.0, 0.0]), F3::new(b, [0.0, 1.0, 0.0]));
                let got = d(x, y).re();
                assert!(
                    same(got, f(a, b)),
                    "{name}({a:?}, {b:?}) = {got:?}, f64 gives {:?}",
                    f(a, b)
                );
            }
        }
    }
    for a in SPECIAL {
        let (s, c) = F3::new(a, [1.0, 0.0, 0.0]).sin_cos();
        assert!(
            same(s.re(), a.sin_cos().0) && same(c.re(), a.sin_cos().1),
            "sin_cos({a:?})"
        );
        assert_eq!(
            Float::integer_decode(F3::new(a, [1.0; 3])),
            Float::integer_decode(a)
        );
        for lo in SPECIAL {
            for hi in SPECIAL {
                if lo <= hi {
                    let got = F3::new(a, [1.0, 0.0, 0.0]).clamp(F3::constant(lo), F3::constant(hi));
                    assert!(
                        same(got.re(), a.clamp(lo, hi)),
                        "clamp({a:?}, {lo:?}, {hi:?})"
                    );
                }
            }
        }
    }
}

#[test]
fn values_follow_double_double_bit_for_bit() {
    type X = Dual<Dd, 2>;
    let specials = dd_special();
    for (name, d, f) in unary_methods::<Dd, 2>() {
        for &a in &specials {
            let got = d(X::new(a, [Dd::ONE, Dd::NAN])).re();
            assert!(
                same_dd(got, f(a)),
                "{name}({a:?}) = {got:?}, Dd gives {:?}",
                f(a)
            );
        }
    }
    for (name, d, f) in binary_methods::<Dd, 2>() {
        for &a in &specials {
            for &b in &specials {
                let got = d(
                    X::new(a, [Dd::ONE, Dd::ZERO]),
                    X::new(b, [Dd::ZERO, Dd::ONE]),
                )
                .re();
                assert!(same_dd(got, f(a, b)), "{name}({a:?}, {b:?}) = {got:?}");
            }
        }
    }
    for &a in &specials {
        for &lo in &specials {
            for &hi in &specials {
                let got = X::new(a, [Dd::ONE, Dd::ZERO]).clamp(X::constant(lo), X::constant(hi));
                assert!(
                    same_dd(got.re(), Float::clamp(a, lo, hi)),
                    "clamp({a:?}, {lo:?}, {hi:?})"
                );
            }
        }
    }
}

#[test]
fn unseeded_directions_and_constants_have_exactly_zero_derivatives() {
    for (name, d, _) in unary_methods::<f64, 3>() {
        for a in SPECIAL {
            for seed in [1.0, -0.5, f64::INFINITY, f64::NAN] {
                let r = d(F3::new(a, [seed, 0.0, -0.0]));
                assert_eq!(
                    &r.eps()[1..],
                    &[0.0, 0.0],
                    "{name}({a:?}) seeded {seed}: {r:?}"
                );
            }
            assert_eq!(d(F3::constant(a)).eps(), &[0.0; 3], "{name}(const {a:?})");
        }
    }
    for (name, d, _) in binary_methods::<f64, 3>() {
        for a in SPECIAL {
            for b in SPECIAL {
                for (sa, sb) in [(1.0, 1.0), (f64::NAN, f64::INFINITY)] {
                    let r = d(F3::new(a, [sa, 0.0, 0.0]), F3::new(b, [0.0, sb, 0.0]));
                    assert_eq!(r.eps()[2], 0.0, "{name}({a:?}, {b:?}): {r:?}");
                }
                let c = d(F3::constant(a), F3::constant(b));
                assert_eq!(c.eps(), &[0.0; 3], "{name}(const {a:?}, const {b:?})");
            }
        }
    }
    type X = Dual<Dd, 2>;
    for (name, d, _) in unary_methods::<Dd, 2>() {
        for a in dd_special() {
            let r = d(X::new(a, [Dd::ONE, Dd::ZERO]));
            assert_eq!(r.eps()[1], Dd::ZERO, "{name}({a:?})");
            assert_eq!(
                d(X::constant(a)).eps(),
                &[Dd::ZERO; 2],
                "{name}(const {a:?})"
            );
        }
    }
    for (name, d, _) in binary_methods::<Dd, 2>() {
        for a in dd_special() {
            for b in dd_special() {
                let c = d(X::constant(a), X::constant(b));
                assert_eq!(c.eps(), &[Dd::ZERO; 2], "{name}(const {a:?}, const {b:?})");
            }
        }
    }
}

#[test]
fn piecewise_constant_methods_have_zero_derivatives_everywhere() {
    for (name, d, _) in unary_methods::<f64, 1>() {
        if PIECEWISE_CONSTANT.contains(&name) {
            for a in SPECIAL {
                for seed in [1.0, f64::NAN, f64::INFINITY] {
                    assert_eq!(d(F1::new(a, [seed])).eps(), &[0.0], "{name}({a:?})");
                }
            }
        }
    }
}

// ---- Derivatives at special values and the edges of the domains -------------------------------

/// The derivative of the method `name` at `x` in direction `x′ = 1` with `Dual<f64, 1>`, checked
/// against `Dual<Dd, 1>` (leading word): the same special value (up to the sign of zero), or
/// within 64u of it.
fn derivative_at(name: &str, x: f64) -> f64 {
    let (_, d64, _) = unary_methods::<f64, 1>()
        .into_iter()
        .find(|m| m.0 == name)
        .unwrap();
    let (_, ddd, _) = unary_methods::<Dd, 1>()
        .into_iter()
        .find(|m| m.0 == name)
        .unwrap();
    let a = d64(F1::new(x, [1.0])).eps()[0];
    let b = ddd(Dual::new(Dd::from_f64(x), [Dd::ONE])).eps()[0].hi();
    let close = (a - b).abs() <= 64.0 * U * b.abs();
    assert!(
        same_derivative(a, b) || close,
        "{name}({x:?}): f64 {a:?}, Dd {b:?}"
    );
    a
}

#[test]
fn derivatives_at_singular_points_are_the_one_sided_limits() {
    let inf = f64::INFINITY;
    let cases: &[(&str, f64, f64)] = &[
        // f′ infinite at the edge of the domain: the limit from inside, at either zero.
        ("sqrt", 0.0, inf),
        ("sqrt", -0.0, inf),
        ("cbrt", 0.0, inf),
        ("cbrt", -0.0, inf),
        ("ln", 0.0, inf),
        ("ln", -0.0, inf),
        ("log2", 0.0, inf),
        ("log2", -0.0, inf),
        ("log10", -0.0, inf),
        ("ln_1p", -1.0, inf),
        ("asin", 1.0, inf),
        ("asin", -1.0, inf),
        ("acos", 1.0, -inf),
        ("acos", -1.0, -inf),
        ("acosh", 1.0, inf),
        ("atanh", 1.0, inf),
        ("atanh", -1.0, inf),
        ("powf(0.5)", 0.0, inf),
        ("powf(0.5)", -0.0, inf),
        ("powf(1/3)", 0.0, inf),
        ("recip", 0.0, -inf),
        ("recip", -0.0, -inf),
        ("powi(-1)", 0.0, -inf),
        ("powi(-2)", 0.0, -inf),
        ("powi(-2)", -0.0, inf),
        ("powf(-1)", 0.0, -inf),
        ("powf(-0.5)", 0.0, -inf),
        // Finite derivatives at zero, including where x^(−1) is infinite.
        ("powi(1)", 0.0, 1.0),
        ("powi(2)", 0.0, 0.0),
        ("powi(3)", -0.0, 0.0),
        ("powf(0)", 0.0, 0.0),
        ("powf(1)", 0.0, 1.0),
        ("powf(2)", 0.0, 0.0),
        ("powf(2)", -0.0, 0.0),
        ("powf(3)", -0.0, 0.0),
        ("sin", -0.0, 1.0),
        ("cos", 0.0, 0.0),
        ("tan", -0.0, 1.0),
        ("atan", 0.0, 1.0),
        ("sinh", -0.0, 1.0),
        ("cosh", 0.0, 0.0),
        ("tanh", -0.0, 1.0),
        ("asinh", 0.0, 1.0),
        ("atanh", -0.0, 1.0),
        ("exp_m1", -0.0, 1.0),
        ("ln_1p", -0.0, 1.0),
        ("exp", 0.0, 1.0),
        // At ±∞: the limits of f′.
        ("sqrt", inf, 0.0),
        ("cbrt", inf, 0.0),
        ("cbrt", -inf, 0.0),
        ("ln", inf, 0.0),
        ("log2", inf, 0.0),
        ("ln_1p", inf, 0.0),
        ("exp", -inf, 0.0),
        ("exp", inf, inf),
        ("exp2", -inf, 0.0),
        ("exp2", inf, inf),
        ("exp_m1", -inf, 0.0),
        ("exp_m1", inf, inf),
        ("atan", inf, 0.0),
        ("atan", -inf, 0.0),
        ("sinh", inf, inf),
        ("sinh", -inf, inf),
        ("cosh", inf, inf),
        ("cosh", -inf, -inf),
        ("tanh", inf, 0.0),
        ("tanh", -inf, 0.0),
        ("asinh", inf, 0.0),
        ("asinh", -inf, 0.0),
        ("acosh", inf, 0.0),
        ("recip", inf, 0.0),
        ("recip", -inf, 0.0),
        ("powi(2)", inf, inf),
        ("powi(2)", -inf, -inf),
        ("powi(-1)", inf, 0.0),
        ("powf(0.5)", inf, 0.0),
        ("powf(2)", inf, inf),
        ("powf(-2.5)", inf, 0.0),
        ("powf(0)", inf, 0.0),
        ("powf(1)", inf, 1.0),
        ("to_degrees", inf, 180.0 / std::f64::consts::PI),
        // Overflowing values with finite derivatives.
        ("powi(2)", 1e200, 2e200),
        ("exp", -800.0, 0.0),
    ];
    for &(name, x, want) in cases {
        let got = derivative_at(name, x);
        assert!(got == want, "{name}'({x:?}) = {got:?}, expected {want:?}");
    }
}

#[test]
fn derivatives_are_nan_outside_the_domain_and_for_nan() {
    let nan = f64::NAN;
    for (name, x) in [
        ("sqrt", -1.0),
        ("sqrt", -5e-324),
        ("ln", -1.0),
        ("log2", -2.0),
        ("log10", -2.0),
        ("ln_1p", -2.0),
        ("asin", 1.5),
        ("asin", -1.0 - f64::EPSILON),
        ("acos", -1.5),
        ("acosh", 0.5),
        ("acosh", -2.0),
        ("atanh", 1.5),
        ("atanh", -2.0),
        ("sin", f64::INFINITY),
        ("cos", f64::NEG_INFINITY),
        ("tan", f64::INFINITY),
        ("fract", f64::INFINITY),
        ("powf(0.5)", -1.0),
        ("powf(1/3)", -8.0),
        ("(-2)^x", 0.5),
        ("log(x, 3)", -1.0),
        ("log(3, x)", -1.0),
    ] {
        let got = derivative_at(name, x);
        assert!(got.is_nan(), "{name}'({x:?}) = {got:?}");
    }
    // A NaN argument gives NaN derivatives, except for the piecewise-constant methods and
    // those that only choose a sign (abs) or whose derivative is constant there (x⁰, x¹).
    for (name, d, _) in unary_methods::<f64, 1>() {
        let r = d(F1::new(nan, [1.0]));
        let exempt = PIECEWISE_CONSTANT.contains(&name)
            || ["abs", "powf(0)", "powi(1)", "powf(1)"].contains(&name);
        if !exempt {
            assert!(r.eps()[0].is_nan(), "{name}'(NaN) = {:?}", r.eps()[0]);
        }
    }
}

#[test]
fn derivatives_that_naive_formulas_get_wrong() {
    let d = |name: &str, x: f64| derivative_at(name, x);
    let rel = |got: f64, want: f64| ((got - want) / want).abs();
    // e^x as (e^x − 1) + 1 would be 0 here; 1 − tanh² would be 0.
    assert!(rel(d("exp_m1", -40.0), (-40f64).exp()) <= 4.0 * U);
    assert!(
        rel(
            d("tanh", 20.0),
            4.0 * (-40f64).exp() / (1.0 + (-40f64).exp()).powi(2)
        ) <= 16.0 * U
    );
    assert!(rel(d("tanh", -300.0), 4.0 * (-600f64).exp()) <= 16.0 * U);
    // √(x² + 1) and √(x² − 1) would overflow: the derivatives are 1/|x|.
    assert_eq!(d("asinh", 1e200), 1e-200);
    assert_eq!(d("asinh", -1e300), 1e-300);
    assert_eq!(d("acosh", 1e200), 1e-200);
    // 1 + sinh² would overflow: cosh x = |sinh x|.
    let s = 700f64.sinh();
    assert_eq!(d("sinh", -700.0), s);
    // asin and acos next to ±1: 1/√((1 − x)(1 + x)) with 1 − x exact; 1 − x² would cancel.
    let x = 1.0 - f64::EPSILON / 2.0; // 1 − 2^-53
    let want = 1.0 / (two_pow(-53) * (2.0 - two_pow(-53))).sqrt();
    assert!(rel(d("asin", x), want) <= 4.0 * U && rel(d("acos", -x), -want) <= 4.0 * U);
    assert!(rel(d("atanh", -x), 1.0 / (two_pow(-53) * (2.0 - two_pow(-53)))) <= 4.0 * U);
    let x = 1.0 + f64::EPSILON;
    let want = 1.0 / (f64::EPSILON * (2.0 + f64::EPSILON)).sqrt();
    assert!(rel(d("acosh", x), want) <= 4.0 * U);
    // 2^x overflows for x >= 1024, but 2^x·ln 2 does not below 1024.53.
    let got = F1::new(1024.25, [1.0]).exp2();
    let want = (Dd::from_f64(1023.25).exp2() * Dd::LN_2() * 2.0).hi();
    assert_eq!(got.re(), f64::INFINITY);
    assert!(rel(got.eps()[0], want) <= 8.0 * U, "{got:?} vs {want:e}");
    assert_eq!(F1::new(1025.0, [1.0]).exp2().eps()[0], f64::INFINITY);
}

#[test]
fn reciprocal_keeps_derivatives_whose_intermediates_leave_the_range() {
    // 1/x overflows for this subnormal x, yet −x′/x² is finite for a tiny x′.
    let (x, dx) = (1e-310, 1e-320);
    let r = F1::new(x, [dx]).recip();
    let want = -(Dd::from_f64(dx) / Dd::from_f64(x) / Dd::from_f64(x)).hi();
    assert_eq!(r.re(), f64::INFINITY);
    assert!(
        ((r.eps()[0] - want) / want).abs() <= 1e-12,
        "{r:?} vs {want:e}"
    ); // dx is subnormal
    // 1/x is subnormal: −x′/x² is divided directly.
    let (x, dx) = (1.01 * two_pow(1022), 1.5 * two_pow(1023));
    let r = F1::new(x, [dx]).recip();
    let want = -(Dd::from_f64(dx) / Dd::from_f64(x) / Dd::from_f64(x)).hi();
    assert!(
        r.re().is_subnormal() && ((r.eps()[0] - want) / want).abs() <= 2.0 * U,
        "{r:?} vs {want:e}"
    );
    // In double-double, a reciprocal below 2^-969 loses trailing bits (here about 2^-72
    // relative); dividing directly keeps the full precision.
    let x = Dd::from_f64(3.0 * two_pow(1000));
    let dx = Dd::from_f64(two_pow(1000)) / Dd::from_f64(7.0);
    let r = Dual::<Dd, 1>::new(x, [dx]).recip();
    let want = -(dx / x / x);
    let err = ((r.eps()[0] - want) / want).abs().hi();
    assert!(err <= 32.0 * U2, "{r:?} vs {want:?}: {err:e}");
    // The fast path: −(x′·r)·r, not −x′·r², which would overflow here.
    let r = F1::new(1e-200, [1e-300]).recip();
    assert!(((r.eps()[0] + 1e100) / 1e100).abs() <= 4.0 * U, "{r:?}");
}

#[test]
fn power_derivatives_at_the_special_cases() {
    let inf = f64::INFINITY;
    // (x, y, ∂/∂x, ∂/∂y), with x and y both seeded.
    let ln2 = std::f64::consts::LN_2;
    let cases: &[(f64, f64, f64, f64)] = &[
        (2.0, 3.0, 12.0, 8.0 * ln2),
        (0.0, 2.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, 0.0),
        (0.0, 0.5, inf, 0.0),
        (-0.0, 3.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, -inf), // 0^y jumps from 1 to 0 at y = 0
        (0.0, -1.0, -inf, -inf),
        (inf, 2.0, inf, inf),
        (inf, -1.0, -0.0, 0.0), // ∞^y = 0 for all y < 0
        (inf, 0.0, 0.0, inf),
        (1.0, 5.0, 5.0, 0.0),
        (1.0, inf, inf, 0.0),
        (-2.0, 3.0, 12.0, f64::NAN), // (−2)^y has no derivative in y
        (-2.0, 0.0, 0.0, f64::NAN),
    ];
    for &(x, y, dx, dy) in cases {
        let [a, b] = Dual::<f64, 2>::variables([x, y]);
        let r = a.powf(b);
        assert!(same(r.re(), x.powf(y)));
        let close = |got: f64, want: f64| {
            same_derivative(got, want) || ((got - want) / want).abs() <= 16.0 * U
        };
        assert!(
            close(r.eps()[0], dx),
            "∂/∂x {x}^{y}: {:?}, expected {dx:?}",
            r.eps()[0]
        );
        assert!(
            close(r.eps()[1], dy),
            "∂/∂y {x}^{y}: {:?}, expected {dy:?}",
            r.eps()[1]
        );
    }
    // f underflows or overflows while ∂/∂x = y·x^(y−1) is finite: computed directly, not as
    // y·f/x. ∂/∂y = f·ln x is then subnormal, 0 or infinite, as f is.
    for (x, y) in [
        (1e-300, 1.5),
        (1e300, 1.1),
        (1e-200, 1.6),
        (1e-100, 3.1),
        (0.5, 1100.0),
    ] {
        let [a, b] = Dual::<f64, 2>::variables([x, y]);
        let r = a.powf(b);
        let (xd, yd) = (Dd::from_f64(x), Dd::from_f64(y));
        let want = yd * xd.powf(yd - 1.0);
        assert!(
            rel_err(r.eps()[0], want) <= 8.0 * U,
            "∂/∂x {x}^{y}: {r:?} vs {want:?}"
        );
        assert!(
            same_derivative(r.eps()[1], r.re() * x.ln()),
            "∂/∂y {x}^{y}: {r:?}"
        );
    }
    // A constant exponent never computes ln x: (−2)^3 has a finite derivative in x and 0 in
    // the unseeded direction.
    let r = F3::new(-2.0, [1.0, 0.0, 0.0]).powf(F3::constant(3.0));
    assert_eq!(r.eps(), &[12.0, 0.0, 0.0]);
    // The exponent's seed matters only in its own direction.
    let r = F3::new(-2.0, [1.0, 0.0, 0.0]).powf(F3::new(3.0, [0.0, 1.0, 0.0]));
    assert!(
        r.eps()[0] == 12.0 && r.eps()[1].is_nan() && r.eps()[2] == 0.0,
        "{r:?}"
    );
}

#[test]
fn hypot_and_atan2_at_the_origin_infinities_and_range_ends() {
    let inf = f64::INFINITY;
    let [x, y] = Dual::<f64, 2>::variables([0.0, -0.0]);
    // |z| has no derivative at the origin; hypot gives zero derivatives (a subgradient).
    assert_eq!(x.hypot(y).eps(), &[0.0, 0.0]);
    // atan2 jumps at the origin: NaN in each direction an argument depends on.
    let a = y.atan2(x);
    assert!(
        a.re() == 0.0 && a.eps()[0].is_nan() && a.eps()[1].is_nan(),
        "{a:?}"
    );
    let a = Dual::<f64, 2>::new(0.0, [0.0, 0.0]).atan2(Dual::new(-0.0, [0.0, 0.0]));
    assert_eq!(a.eps(), &[0.0, 0.0]);
    // Infinite arguments: the limits.
    for (a, b, dh, da) in [
        (inf, 1.0, [1.0, 0.0], [0.0, 0.0]),
        (-inf, 1.0, [-1.0, 0.0], [0.0, 0.0]),
        (2.0, -inf, [0.0, -1.0], [0.0, 0.0]),
        (inf, inf, [f64::NAN, f64::NAN], [0.0, 0.0]),
    ] {
        let [p, q] = Dual::<f64, 2>::variables([a, b]);
        let h = p.hypot(q);
        assert!(same(h.re(), a.hypot(b)));
        assert!(
            same_derivative(h.eps()[0], dh[0]) && same_derivative(h.eps()[1], dh[1]),
            "hypot({a}, {b}): {h:?}"
        );
        let t = p.atan2(q);
        assert_eq!(t.eps(), &da, "atan2({a}, {b})");
    }
    // Overflowing and underflowing squares: rescaled, so the derivatives keep full precision.
    // (The atan2 derivatives, of order 1/|z|, overflow for |z| < 1/MAX: only hypot there.)
    for (a, b, atan2_too) in [
        (3e300, 4e300, true),
        (1e308, 1e308, true),
        (1e200, 1e-200, true),
        (3e-300, -4e-300, true),
        (1e-160, 1e-160, true),
        (1e-200, 3e-200, true),
        (3e-320, -4e-320, false),
        (5e-324, 5e-324, false),
    ] {
        let [p, q] = Dual::<f64, 2>::variables([a, b]);
        let (ad, bd) = (Dd::from_f64(a), Dd::from_f64(b));
        let h = p.hypot(q);
        let dh = scaled_hypot_partials(ad, bd);
        for (&got, &want) in h.eps().iter().zip(&dh) {
            assert!(rel_err(got, want) <= 8.0 * U, "hypot({a:e}, {b:e}): {h:?}");
        }
        if atan2_too {
            let t = p.atan2(q);
            let dt = scaled_atan2_partials(ad, bd);
            for (&got, &want) in t.eps().iter().zip(&dt) {
                assert!(rel_err(got, want) <= 8.0 * U, "atan2({a:e}, {b:e}): {t:?}");
            }
        }
    }
}

/// |got − want|/|want| with `want` in double-double, or 0 if both are zero.
fn rel_err(got: f64, want: Dd) -> f64 {
    if got == 0.0 && want.hi() == 0.0 {
        return 0.0;
    }
    (Dd::from_f64(got) - want).abs().hi() / want.abs().hi()
}

/// v·2^e in double-double, scaling in steps so that no factor overflows.
fn ldexp(mut v: Dd, mut e: i32) -> Dd {
    while e > 500 {
        v *= two_pow(500);
        e -= 500;
    }
    while e < -500 {
        v *= two_pow(-500);
        e += 500;
    }
    v * two_pow(e)
}

/// x = m·2^e with |m| in [1, 2) (exact; m = 0 for x = 0).
fn split(x: Dd) -> (Dd, i32) {
    if x.hi() == 0.0 {
        return (Dd::ZERO, 0);
    }
    let e = x.hi().abs().log2().floor() as i32;
    (ldexp(x, -e), e)
}

/// (x/h, y/h) for h = hypot(x, y), in double-double with the binary exponents kept apart, so no
/// intermediate over- or underflows: x/h = (m_x/h_s)·2^(e_x − E) with h = h_s·2^E.
fn scaled_hypot_partials(x: Dd, y: Dd) -> [Dd; 2] {
    let ((mx, ex), (my, ey)) = (split(x), split(y));
    let e = ex.max(ey);
    let (xs, ys) = (ldexp(mx, ex - e), ldexp(my, ey - e));
    let h = (xs * xs + ys * ys).sqrt();
    [ldexp(mx / h, ex - e), ldexp(my / h, ey - e)]
}

/// (∂/∂y, ∂/∂x) of atan2(y, x) = (x, −y)/(x² + y²) in double-double, with the binary exponents
/// kept apart: x/(x² + y²) = (m_x/r_s)·2^(e_x − 2E) with x² + y² = r_s·2^(2E).
fn scaled_atan2_partials(y: Dd, x: Dd) -> [Dd; 2] {
    let ((my, ey), (mx, ex)) = (split(y), split(x));
    let e = ex.max(ey);
    let (xs, ys) = (ldexp(mx, ex - e), ldexp(my, ey - e));
    let r = xs * xs + ys * ys;
    [ldexp(mx / r, ex - 2 * e), ldexp(-(my / r), ey - 2 * e)]
}

#[test]
fn max_min_clamp_abs_and_copysign_choose_the_branch_derivative() {
    let [x, y] = Dual::<f64, 2>::variables([1.0, 2.0]);
    let nan = Dual::<f64, 2>::new(f64::NAN, [5.0, 5.0]);
    assert_eq!((x.max(y).eps(), x.min(y).eps()), (y.eps(), x.eps()));
    assert_eq!((y.max(x).eps(), y.min(x).eps()), (y.eps(), x.eps()));
    // A NaN operand is ignored, with its derivatives.
    assert_eq!((nan.max(x).eps(), x.max(nan).eps()), (x.eps(), x.eps()));
    assert_eq!((nan.min(x).eps(), x.min(nan).eps()), (x.eps(), x.eps()));
    // At a tie of zeros, the derivatives of the operand whose sign the result has.
    let (p, n) = (
        Dual::<f64, 2>::new(0.0, [1.0, 0.0]),
        Dual::<f64, 2>::new(-0.0, [0.0, 1.0]),
    );
    for (a, b) in [(p, n), (n, p)] {
        let (mx, mn) = (a.max(b), a.min(b));
        assert!(same(mx.re(), a.re().max(b.re())) && same(mn.re(), a.re().min(b.re())));
        let from = |r: Dual<f64, 2>| if r.re().is_sign_negative() { n } else { p };
        assert_eq!(mx.eps(), from(mx).eps());
        assert_eq!(mn.eps(), from(mn).eps());
    }
    // DoubleDouble follows IEEE 754-2019 on zeros: max(−0, +0) = +0 with +0's derivatives.
    type X = Dual<Dd, 2>;
    let (pd, nd) = (
        X::new(Dd::ZERO, [Dd::ONE, Dd::ZERO]),
        X::new(-Dd::ZERO, [Dd::ZERO, Dd::ONE]),
    );
    assert_eq!((nd.max(pd).eps(), pd.max(nd).eps()), (pd.eps(), pd.eps()));
    assert_eq!((nd.min(pd).eps(), pd.min(nd).eps()), (nd.eps(), nd.eps()));
    // A tie of equal nonzero values: self's derivatives.
    let z = Dual::<f64, 2>::new(1.0, [0.0, 7.0]);
    assert_eq!((x.max(z).eps(), z.max(x).eps()), (x.eps(), z.eps()));
    // clamp: the bound's derivatives outside, self's inside and at the bounds.
    let (lo, hi) = (
        Dual::<f64, 2>::new(0.0, [3.0, 0.0]),
        Dual::<f64, 2>::new(1.5, [0.0, 4.0]),
    );
    assert_eq!(
        Dual::<f64, 2>::new(-1.0, [1.0, 1.0]).clamp(lo, hi).eps(),
        lo.eps()
    );
    assert_eq!(
        Dual::<f64, 2>::new(9.0, [1.0, 1.0]).clamp(lo, hi).eps(),
        hi.eps()
    );
    assert_eq!(x.clamp(lo, hi).eps(), x.eps());
    assert_eq!(
        Dual::<f64, 2>::new(1.5, [1.0, 1.0]).clamp(lo, hi).eps(),
        &[1.0, 1.0]
    );
    let c = Dual::<f64, 2>::new(-0.0, [1.0, 1.0]).clamp(lo, hi);
    assert!(same(c.re(), -0.0) && c.eps() == &[1.0, 1.0]);
    // Bad bounds give NaN instead of the panic of f64::clamp.
    assert!(x.clamp(hi, lo).re().is_nan());
    assert!(x.clamp(nan, hi).re().is_nan() && x.clamp(lo, nan).re().is_nan());
    // abs: −x′ for a negative sign bit (−0 included), x′ otherwise.
    for (v, s) in [(-2.0, -1.0), (-0.0, -1.0), (0.0, 1.0), (3.0, 1.0)] {
        let a = Dual::<f64, 2>::new(v, [1.0, -2.0]).abs();
        assert!(
            same(a.re(), v.abs()) && a.eps() == &[s, -2.0 * s],
            "abs({v:?})"
        );
    }
    // copysign: ±x′, nothing from the sign argument.
    let m = Dual::<f64, 2>::new(-0.0, [0.0, 9.0]);
    assert_eq!(x.copysign(m).eps(), &[-1.0, 0.0]);
    assert_eq!(x.copysign(y).eps(), &[1.0, 0.0]);
    assert!(same(x.copysign(m).re(), -1.0));
    // abs_sub: x′ − y′ where x > y, zero otherwise.
    assert_eq!(y.abs_sub(x).eps(), &[-1.0, 1.0]);
    assert_eq!(x.abs_sub(y).eps(), &[0.0, 0.0]);
}

#[test]
fn classification_inspects_derivatives_only_for_nan_and_finite() {
    let f = |re: f64, e: f64| F1::new(re, [e]);
    // NaN and finiteness see the derivatives, so a NaN derivative is never silent.
    assert!(Float::is_nan(f(1.0, f64::NAN)) && !Float::is_finite(f(1.0, f64::NAN)));
    assert!(!Float::is_nan(f(1.0, f64::INFINITY)) && !Float::is_finite(f(1.0, f64::INFINITY)));
    // Infinity, normality, the category and the sign describe the value: a finite value with
    // an infinite derivative is neither NaN, infinite nor finite.
    assert!(!Float::is_infinite(f(1.0, f64::INFINITY)));
    assert!(Float::is_infinite(f(f64::INFINITY, 0.0)) && !Float::is_finite(f(f64::INFINITY, 0.0)));
    assert!(Float::is_normal(f(1.0, f64::NAN)) && Float::is_subnormal(f(1e-310, 1.0)));
    assert_eq!(Float::classify(f(0.0, f64::NAN)), FpCategory::Zero);
    assert_eq!(Float::classify(f(1e-310, 1.0)), FpCategory::Subnormal);
    assert!(Float::is_sign_negative(f(-0.0, 1.0)) && Float::is_sign_positive(f(0.0, -1.0)));
    // Limits and constants have zero derivatives.
    type X = Dual<f64, 2>;
    for c in [
        X::nan(),
        X::infinity(),
        X::neg_infinity(),
        X::neg_zero(),
        X::min_value(),
        X::min_positive_value(),
        X::epsilon(),
        X::max_value(),
        X::E(),
        X::LN_2(),
        X::FRAC_PI_2(),
        X::TAU(),
        X::LOG2_10(),
    ] {
        assert_eq!(c.eps(), &[0.0, 0.0]);
    }
    assert!(same(X::neg_zero().re(), -0.0) && X::epsilon().re() == f64::EPSILON);
    let d = Dual::<Dd, 1>::PI();
    assert!(same_dd(d.re(), Dd::PI()) && d.eps() == &[Dd::ZERO]);
    assert!(same_dd(Dual::<Dd, 1>::epsilon().re(), Dd::epsilon()));
    assert!(same_dd(
        Dual::<Dd, 1>::min_positive_value().re(),
        Dd::min_positive_value()
    ));
    let p = <Dual<Dd, 2> as Real>::from_f64_pair(1.0, 2f64.powi(-70));
    assert_eq!((p.re().lo(), p.eps()), (2f64.powi(-70), &[Dd::ZERO; 2]));
}

// ---- Accuracy over each domain ----------------------------------------------------------------

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    /// Seeded from a name, so each function gets its own reproducible stream.
    fn new(name: &str) -> Self {
        let h = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        Rng(h | 1)
    }

    /// Uniform in [0, 1).
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// m·2^e with m uniform in [1, 2) and e uniform in [lo, hi), rounded to the subnormal grid
    /// below 2^-1022.
    fn pow2(&mut self, lo: i32, hi: i32) -> f64 {
        let e = lo + ((hi - lo) as f64 * self.f()).floor() as i32;
        let m = 1.0 + self.f();
        if e >= -1022 {
            m * two_pow(e)
        } else {
            m * two_pow(e + 64) * two_pow(-64)
        }
    }

    /// `x` or `−x`.
    fn sign(&mut self, x: f64) -> f64 {
        if self.f() < 0.5 { -x } else { x }
    }

    /// Uniform in [a, b).
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }

    /// True with probability p.
    fn chance(&mut self, p: f64) -> bool {
        self.f() < p
    }
}

/// How one unary method is checked: an analytic derivative in double-double, the domain to
/// sample, and where central differences are accurate.
struct Spec {
    /// The method's name in [`unary_methods`].
    name: &'static str,
    /// f′(x) in double-double, from a formula independent of the implementation where possible.
    deriv: fn(Dd) -> Dd,
    /// Arguments over the whole domain, with extra weight next to singular points.
    sample: fn(&mut Rng) -> f64,
    /// Arguments where central differences with step 2^-20·scale(x) are accurate.
    moderate: fn(&mut Rng) -> f64,
    /// The length over which f′ changes by a factor of order one.
    scale: fn(f64) -> f64,
    /// Bound for `Dual<f64, 1>` against the analytic derivative, in u (see `specs`).
    bound: f64,
    /// Condition factor of `T`'s function for the double-double bound (1 unless noted).
    cond: fn(f64) -> f64,
}

/// ±(1 − δ) with δ down to 2^-53, ±tiny, or uniform in (−1, 1): arguments of asin, acos, atanh.
fn sample_unit(r: &mut Rng) -> f64 {
    let x = match (r.f() * 3.0) as u32 {
        0 => 1.0 - r.pow2(-54, -1),
        1 => r.pow2(-60, 0).min(1.0),
        _ => r.f(),
    };
    r.sign(x)
}

/// Inside (−1 + 2^-20, 1 − 2^-20).
fn moderate_unit(r: &mut Rng) -> f64 {
    r.range(-1.0 + two_pow(-20), 1.0 - two_pow(-20))
}

/// The bounds in u (2^-53) for `Dual<f64, 1>` assume each platform `f64` function (libm) is
/// within 2 ulps = 4u of its exact value (glibc and Apple libm document at most 1–2 ulps for
/// these), and each correctly rounded operation within 1u: they sum those errors through the
/// derivative formula, with any amplification noted. The double-double bound is 2^-96: each
/// `DoubleDouble` function is within 2^-99 (and the reference within as much again), plus a
/// few operations of a few u² each.
fn specs() -> Vec<Spec> {
    let abs = |x: f64| x.abs();
    let unit = |_: f64| 1.0;
    let at_least_one = |x: f64| x.abs().max(1.0);
    vec![
        // 1/(2s): s correctly rounded (1u), division (1u).
        Spec {
            name: "sqrt",
            deriv: |x| x.sqrt() / (x * 2.0),
            sample: |r| r.pow2(-1074, 1024),
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 2.0,
            cond: unit,
        },
        // 1/(3c²): c (4u) squared (8u + 1u), times 3 (1u), reciprocal (1u).
        Spec {
            name: "cbrt",
            deriv: |x| x.cbrt() / (x * 3.0),
            sample: |r| {
                let x = r.pow2(-1074, 1024);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-100, 100);
                r.sign(x)
            },
            scale: abs,
            bound: 11.0,
            cond: unit,
        },
        // eˣ (4u).
        Spec {
            name: "exp",
            deriv: |x| x.exp(),
            sample: |r| {
                if r.chance(0.2) {
                    let x = r.pow2(-60, 0);
                    r.sign(x)
                } else {
                    r.range(-745.0, 709.78)
                }
            },
            moderate: |r| r.range(-40.0, 40.0),
            scale: unit,
            bound: 4.0,
            cond: unit,
        },
        // 2ˣ (4u) times ln 2 (1u + 1u).
        Spec {
            name: "exp2",
            deriv: |x| x.exp2() * Dd::LN_2(),
            sample: |r| {
                if r.chance(0.2) {
                    let x = r.pow2(-60, 0);
                    r.sign(x)
                } else {
                    r.range(-1074.0, 1023.99)
                }
            },
            moderate: |r| r.range(-60.0, 60.0),
            scale: unit,
            bound: 6.0,
            cond: unit,
        },
        // (eˣ − 1) + 1 for eˣ − 1 >= −½: 4u·|eˣ − 1|/eˣ <= 4u, plus 1u; else eˣ (4u).
        Spec {
            name: "exp_m1",
            deriv: |x| x.exp(),
            sample: |r| {
                if r.chance(0.3) {
                    let x = r.pow2(-60, 0);
                    r.sign(x)
                } else {
                    r.range(-745.0, 709.78)
                }
            },
            moderate: |r| r.range(-5.0, 40.0),
            scale: unit,
            bound: 5.0,
            cond: unit,
        },
        // 1/x (1u).
        Spec {
            name: "ln",
            deriv: |x| Dd::ONE / x,
            sample: |r| {
                if r.chance(0.3) {
                    let d = r.pow2(-53, -1);
                    1.0 + r.sign(d)
                } else {
                    r.pow2(-1074, 1024)
                }
            },
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 1.0,
            cond: unit,
        },
        // log2(e)/x: the constant (1u) and the division (1u).
        Spec {
            name: "log2",
            deriv: |x| Dd::ONE / x / Dd::LN_2(),
            sample: |r| r.pow2(-1074, 1024),
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 2.0,
            cond: unit,
        },
        Spec {
            name: "log10",
            deriv: |x| Dd::ONE / x / Dd::LN_10(),
            sample: |r| r.pow2(-1074, 1024),
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 2.0,
            cond: unit,
        },
        // 1/(1 + x): the sum (1u, exact next to −1) and the reciprocal (1u).
        Spec {
            name: "ln_1p",
            deriv: |x| Dd::ONE / (x + 1.0),
            sample: |r| match (r.f() * 3.0) as u32 {
                0 => -1.0 + r.pow2(-53, 0),
                1 => {
                    let x = r.pow2(-1074, 0);
                    r.sign(x)
                }
                _ => r.pow2(0, 1024),
            },
            moderate: |r| -1.0 + r.pow2(-20, 6),
            scale: |x| 1.0 + x,
            bound: 2.0,
            cond: unit,
        },
        // cos x and −sin x (4u).
        Spec {
            name: "sin",
            deriv: |x| x.cos(),
            sample: |r| {
                let x = r.pow2(-60, 50);
                r.sign(x)
            },
            moderate: |r| r.range(-100.0, 100.0),
            scale: unit,
            bound: 4.0,
            cond: unit,
        },
        Spec {
            name: "cos",
            deriv: |x| -x.sin(),
            sample: |r| {
                let x = r.pow2(-60, 50);
                r.sign(x)
            },
            moderate: |r| r.range(-100.0, 100.0),
            scale: unit,
            bound: 4.0,
            cond: unit,
        },
        // 1 + t²: t (4u) squared (8u + 1u), plus 1 (1u); the reference is 1/cos²x.
        Spec {
            name: "tan",
            deriv: |x| {
                let c = x.cos();
                Dd::ONE / (c * c)
            },
            sample: sample_tan,
            moderate: |r| loop {
                let x = r.range(-100.0, 100.0);
                if x.cos().abs() > 0.01 {
                    return x;
                }
            },
            scale: |x| x.cos().abs().min(1.0),
            bound: 10.0,
            cond: unit,
        },
        // 1/√((1 − x)(1 + x)): the product (3u), its root (1.5u + 1u), the reciprocal (1u); the
        // reference is 1/√(1 − x²) with x² exact.
        Spec {
            name: "asin",
            deriv: |x| Dd::ONE / (Dd::ONE - x * x).sqrt(),
            sample: sample_unit,
            moderate: moderate_unit,
            scale: |x| (1.0 - x.abs()).min(1.0),
            bound: 3.5,
            cond: unit,
        },
        Spec {
            name: "acos",
            deriv: |x| -(Dd::ONE / (Dd::ONE - x * x).sqrt()),
            sample: sample_unit,
            moderate: moderate_unit,
            scale: |x| (1.0 - x.abs()).min(1.0),
            bound: 3.5,
            cond: unit,
        },
        // 1/(x² + 1): the square and sum (2u), the reciprocal (1u).
        Spec {
            name: "atan",
            deriv: |x| Dd::ONE / (x * x + 1.0),
            sample: |r| {
                let x = r.pow2(-60, 1024);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 20);
                r.sign(x)
            },
            scale: at_least_one,
            bound: 3.0,
            cond: unit,
        },
        // √(1 + sinh²x): sinh (4u) squared and plus 1 (8u + 2u), halved by the root, plus 1u.
        Spec {
            name: "sinh",
            deriv: |x| x.cosh(),
            sample: |r| {
                if r.chance(0.2) {
                    let x = r.pow2(-60, 0);
                    r.sign(x)
                } else {
                    r.range(-710.0, 710.0)
                }
            },
            moderate: |r| r.range(-30.0, 30.0),
            scale: unit,
            bound: 6.0,
            cond: unit,
        },
        // sinh x (4u).
        Spec {
            name: "cosh",
            deriv: |x| x.sinh(),
            sample: |r| {
                if r.chance(0.2) {
                    let x = r.pow2(-60, 0);
                    r.sign(x)
                } else {
                    r.range(-710.0, 710.0)
                }
            },
            moderate: |r| r.range(-30.0, 30.0),
            scale: unit,
            bound: 4.0,
            cond: unit,
        },
        // (1 − t)(1 + t) for |t| <= ½: 4u·2t²/(1 − t²) <= 2.7u, plus 3u; 1/cosh²x beyond: cosh
        // (4u) and its reciprocal (1u), squared (10u + 1u). The reference is 4e/(1 + e)²,
        // e = e^(−2|x|).
        Spec {
            name: "tanh",
            deriv: |x| {
                let e = (x.abs() * -2.0).exp();
                e * 4.0 / ((e + 1.0) * (e + 1.0))
            },
            sample: |r| {
                if r.chance(0.3) {
                    let x = r.pow2(-60, 0);
                    r.sign(x)
                } else {
                    r.range(-360.0, 360.0)
                }
            },
            moderate: |r| r.range(-4.0, 4.0),
            scale: unit,
            bound: 11.0,
            cond: unit,
        },
        // 1/√(x² + 1) (as atan, then the root: 1u + 1u + 1u), or 1/|x| beyond 2^54.
        Spec {
            name: "asinh",
            deriv: |x| {
                if x.abs().hi() > two_pow(60) {
                    Dd::ONE / x.abs()
                } else {
                    Dd::ONE / (x * x + 1.0).sqrt()
                }
            },
            sample: |r| {
                let x = r.pow2(-60, 1024);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 40);
                r.sign(x)
            },
            scale: at_least_one,
            bound: 3.0,
            cond: unit,
        },
        // 1/√((x − 1)(x + 1)) (as asin), or 1/x beyond 2^54; the reference uses x² − 1.
        Spec {
            name: "acosh",
            deriv: |x| {
                if x.hi() > two_pow(60) {
                    Dd::ONE / x
                } else {
                    Dd::ONE / (x * x - 1.0).sqrt()
                }
            },
            sample: |r| {
                if r.chance(0.5) {
                    1.0 + r.pow2(-52, 0)
                } else {
                    r.pow2(0, 1024)
                }
            },
            moderate: |r| 1.0 + r.pow2(-20, 40),
            scale: |x| x - 1.0,
            bound: 3.5,
            cond: unit,
        },
        // 1/((1 − x)(1 + x)): 3u and the reciprocal (1u).
        Spec {
            name: "atanh",
            deriv: |x| Dd::ONE / (Dd::ONE - x * x),
            sample: sample_unit,
            moderate: moderate_unit,
            scale: |x| (1.0 - x.abs()).min(1.0),
            bound: 4.0,
            cond: unit,
        },
        // −(x′·r)·r: r (1u), two products (2u). The reference divides twice, as x² can fall
        // below the double-double range of full precision.
        Spec {
            name: "recip",
            deriv: |x| -(Dd::ONE / x / x),
            sample: |r| {
                let x = r.pow2(-511, 511);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-100, 100);
                r.sign(x)
            },
            scale: abs,
            bound: 3.0,
            cond: unit,
        },
        // n·x^(n−1): x^(n−1) by repeated squaring and the product with n.
        Spec {
            name: "powi(2)",
            deriv: |x| x * 2.0,
            sample: |r| {
                let x = r.pow2(-1000, 1000);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 20);
                r.sign(x)
            },
            scale: abs,
            bound: 0.0,
            cond: unit,
        },
        Spec {
            name: "powi(3)",
            deriv: |x| x * x * 3.0,
            sample: |r| {
                let x = r.pow2(-500, 500);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 20);
                r.sign(x)
            },
            scale: abs,
            bound: 2.0,
            cond: unit,
        },
        Spec {
            name: "powi(-2)",
            deriv: |x| x.powi(-3) * -2.0,
            sample: |r| {
                let x = r.pow2(-300, 300);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 20);
                r.sign(x)
            },
            scale: abs,
            bound: 3.0,
            cond: unit,
        },
        Spec {
            name: "powi(-3)",
            deriv: |x| x.powi(-4) * -3.0,
            sample: |r| {
                let x = r.pow2(-250, 250);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 20);
                r.sign(x)
            },
            scale: abs,
            bound: 5.0,
            cond: unit,
        },
        // y·(xʸ/x): the power (4u), the quotient and product (2u). DoubleDouble's powf errs by
        // up to 2^-99·|y ln x|.
        Spec {
            name: "powf(0.5)",
            deriv: |x| x.sqrt() / (x * 2.0),
            sample: |r| r.pow2(-1074, 1024),
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 6.0,
            cond: |x| (0.5 * x.ln()).abs().max(1.0),
        },
        Spec {
            name: "powf(1/3)",
            deriv: |x| {
                let y = Dd::from_f64(1.0 / 3.0);
                x.powf(y - 1.0) * y
            },
            sample: |r| r.pow2(-1074, 1024),
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 6.0,
            cond: |x| (x.ln() / 3.0).abs().max(1.0),
        },
        Spec {
            name: "powf(-2.5)",
            deriv: |x| x.powf(Dd::from_f64(-3.5)) * -2.5,
            sample: |r| r.pow2(-280, 280),
            moderate: |r| r.pow2(-20, 20),
            scale: abs,
            bound: 6.0,
            cond: |x| (2.5 * x.ln()).abs().max(1.0),
        },
        // bˣ·ln b: the power (4u), ln b (4u), the product (1u).
        Spec {
            name: "2^x",
            deriv: |x| x.exp2() * Dd::LN_2(),
            sample: |r| r.range(-1074.0, 1023.99),
            moderate: |r| r.range(-60.0, 60.0),
            scale: unit,
            bound: 9.0,
            cond: |x| (x * std::f64::consts::LN_2).abs().max(1.0),
        },
        Spec {
            name: "0.5^x",
            deriv: |x| -((-x).exp2() * Dd::LN_2()),
            sample: |r| r.range(-1023.99, 1074.0),
            moderate: |r| r.range(-60.0, 60.0),
            scale: unit,
            bound: 9.0,
            cond: |x| (x * std::f64::consts::LN_2).abs().max(1.0),
        },
        // The quotient rule for ln x/ln 3 with constant ln 3: 1/x (1u), 1/ln 3 (4u + 1u), the
        // product (1u).
        Spec {
            name: "log(x, 3)",
            deriv: |x| Dd::ONE / x / Dd::from_f64(3.0).ln(),
            sample: |r| r.pow2(-1074, 1024),
            moderate: |r| r.pow2(-100, 100),
            scale: abs,
            bound: 7.0,
            cond: unit,
        },
        // 180/π and π/180: the rounded constant (1u) and its quotient (1u).
        Spec {
            name: "to_degrees",
            deriv: |_| Dd::from_f64(180.0) / Dd::PI(),
            sample: |r| {
                let x = r.pow2(-1000, 1000);
                r.sign(x)
            },
            moderate: |r| r.range(-10.0, 10.0),
            scale: unit,
            bound: 2.0,
            cond: unit,
        },
        Spec {
            name: "to_radians",
            deriv: |_| Dd::PI() / 180.0,
            sample: |r| {
                let x = r.pow2(-1000, 1000);
                r.sign(x)
            },
            moderate: |r| r.range(-10.0, 10.0),
            scale: unit,
            bound: 2.0,
            cond: unit,
        },
        // Exactly 1 and ±1.
        Spec {
            name: "fract",
            deriv: |_| Dd::ONE,
            sample: |r| {
                let x = r.pow2(-60, 52);
                r.sign(x)
            },
            moderate: |r| r.range(-50.0, 50.0).floor() + r.range(0.1, 0.9),
            scale: |_| 0.05,
            bound: 0.0,
            cond: unit,
        },
        Spec {
            name: "abs",
            deriv: |x| Dd::from_f64(x.hi().signum()),
            sample: |r| {
                let x = r.pow2(-1074, 1024);
                r.sign(x)
            },
            moderate: |r| {
                let x = r.pow2(-20, 20);
                r.sign(x)
            },
            scale: abs,
            bound: 0.0,
            cond: unit,
        },
    ]
}

/// Arguments of tan: anywhere up to 2^50, or within a few ulps of a pole (k + ½)π.
fn sample_tan(r: &mut Rng) -> f64 {
    if r.chance(0.3) {
        let k = r.range(-1000.0, 1000.0).floor();
        let pole = (Dd::from_f64(k) + 0.5) * Dd::PI();
        let ulps = r.range(-4.0, 4.0).round() as i64;
        f64::from_bits((pole.hi().to_bits() as i64 + ulps) as u64)
    } else {
        let x = r.pow2(-60, 50);
        r.sign(x)
    }
}

/// The central difference at `x`, Richardson-extrapolated from steps h and h/2, with h the power
/// of two at or below 2^-20·scale (so x ± h is exact in double-double). The truncation error is
/// of order (h/scale)⁴ ≈ 2^-80 and the rounding error about 2^-78·|f(x)|/(scale·|f′(x)|).
fn richardson(f: fn(Dd) -> Dd, x: f64, scale: f64) -> Dd {
    let h = two_pow((scale.log2() - 20.0).floor() as i32);
    let xd = Dd::from_f64(x);
    let d = |h: f64| {
        let hd = Dd::from_f64(h);
        (f(xd + hd) - f(xd - hd)) / (hd * 2.0)
    };
    (d(h / 2.0) * 4.0 - d(h)) / 3.0
}

/// Whether a reference derivative is in the normal range of `f64` (outside it, the derivative
/// over- or underflows and is not compared).
fn normal(d: Dd) -> bool {
    (f64::MIN_POSITIVE..=f64::MAX).contains(&d.hi().abs())
}

#[test]
fn derivative_accuracy_over_each_domain() {
    let f64_methods = unary_methods::<f64, 1>();
    let dd_methods = unary_methods::<Dd, 1>();
    let find64 = |name: &str| f64_methods.iter().find(|m| m.0 == name).unwrap().1;
    let find_dd = |name: &str| dd_methods.iter().find(|m| m.0 == name).unwrap();
    eprintln!(
        "worst derivative error: f64 vs analytic (u) | DD vs analytic (u²) | f64 vs central (u) | DD vs central (relative, where |f| <= 63·scale·|f′|)"
    );
    for spec in specs() {
        let (d64, (_, ddd, value)) = (find64(spec.name), find_dd(spec.name));
        let mut r = Rng::new(spec.name);
        let (mut w64, mut wdd, mut c64, mut cdd) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut compared = 0;
        // Against the analytic derivative, over the whole domain.
        for i in 0..20_000 {
            let x = (spec.sample)(&mut r);
            let want = (spec.deriv)(Dd::from_f64(x));
            if !normal(want) {
                continue;
            }
            compared += 1;
            let got = d64(F1::new(x, [1.0])).eps()[0];
            let err = (Dd::from_f64(got) - want).abs().hi() / (want.abs().hi() * U);
            assert!(
                err <= spec.bound + 1e-6,
                "{}'({x:e}) = {got:e}, expected {want:?}: {err:.2} u",
                spec.name
            );
            w64 = w64.max(err);
            if i % 4 == 0 && want.abs().hi() >= two_pow(-960) && want.abs().hi() <= two_pow(1000) {
                let got = ddd(Dual::new(Dd::from_f64(x), [Dd::ONE])).eps()[0];
                let err = ((got - want) / want).abs().hi();
                let bound = two_pow(-96) * (spec.cond)(x);
                assert!(
                    err <= bound,
                    "{}'({x:e}) in DD = {got:?}, expected {want:?}: {err:e}",
                    spec.name
                );
                wdd = wdd.max(err / U2);
            }
        }
        assert!(
            compared >= 10_000,
            "{}: only {compared} points compared",
            spec.name
        );
        // Against central differences, where they are accurate.
        for _ in 0..300 {
            let x = (spec.moderate)(&mut r);
            let s = (spec.scale)(x);
            let cd = richardson(*value, x, s);
            if cd.hi() == 0.0 {
                continue;
            }
            // Central differences lose accuracy where |f| ≫ scale·|f′|.
            let kappa = 1.0 + value(Dd::from_f64(x)).abs().hi() / (s * cd.abs().hi());
            let got = d64(F1::new(x, [1.0])).eps()[0];
            let e64 = (Dd::from_f64(got) - cd).abs().hi() / cd.abs().hi();
            assert!(
                e64 <= spec.bound * U + two_pow(-70) * kappa,
                "{}'({x:e}) = {got:e}, central {cd:?}",
                spec.name
            );
            let got = ddd(Dual::new(Dd::from_f64(x), [Dd::ONE])).eps()[0];
            let edd = ((got - cd) / cd).abs().hi();
            assert!(
                edd <= two_pow(-70) * kappa,
                "{}'({x:e}) in DD = {got:?}, central {cd:?}",
                spec.name
            );
            // Reported where the central difference itself is accurate to about 2^-70.
            if kappa <= 64.0 {
                c64 = c64.max(e64 / U);
                cdd = cdd.max(edd);
            }
        }
        eprintln!(
            "  {:<11} {w64:5.2} u (bound {:4.1}) | {wdd:7.2} u² | {c64:5.2} u | {cdd:.1e}",
            spec.name, spec.bound
        );
    }
}

/// Exponents and bases for `powf` in six regimes: general; x next to 0 with y in (0, 1),
/// (1, 3) or (−1, 0); y next to 0; x next to 1 with |y| up to 1000; x < 0 with integer y; and
/// |y| up to 2^40 with x within 2^-10 of 1.
fn sample_powf(r: &mut Rng) -> (f64, f64) {
    match (r.f() * 6.0) as u32 {
        0 => (r.pow2(-30, 30), r.range(-20.0, 20.0)),
        1 => {
            let x = r.pow2(-1074, -30);
            let y = match (r.f() * 3.0) as u32 {
                0 => r.range(0.0, 1.0),
                1 => r.range(1.0, 3.0),
                _ => r.range(-1.0, 0.0),
            };
            (x, y)
        }
        2 => {
            let y = r.pow2(-60, -1);
            (r.pow2(-30, 30), r.sign(y))
        }
        3 => {
            let d = r.pow2(-52, -1);
            (1.0 + r.sign(d), r.range(-1e3, 1e3))
        }
        4 => (-r.pow2(-20, 20), r.range(-10.0, 10.0).round()),
        _ => {
            let (d, y) = (r.pow2(-40, -10), r.pow2(5, 40));
            (1.0 + r.sign(d), r.sign(y))
        }
    }
}

/// A pair over the whole range, with independent or nearly equal magnitudes.
fn sample_pair(r: &mut Rng) -> (f64, f64) {
    let x = r.pow2(-1074, 1024);
    let y = if r.chance(0.3) {
        (x * r.range(0.5, 2.0)).min(f64::MAX)
    } else {
        r.pow2(-1074, 1024)
    };
    let x = r.sign(x);
    (x, r.sign(y))
}

#[test]
fn binary_derivative_accuracy_over_each_domain() {
    type X = Dual<Dd, 2>;
    let dd = Dd::from_f64;
    let mut r = Rng::new("binary");
    // Bounds in u, as for `specs`: x^y (4u) over x and times y (2u); x^y times ln x (4u + 4u +
    // 1u); hypot (4u) and the scaled quotient (2u); x² + y² (2u), its reciprocal and the
    // product (2u), or the rescaled form (7u); the quotient rule for ln x/ln b (7u and 17u).
    let (mut wx, mut wy, mut dx_dd, mut dy_dd) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for _ in 0..30_000 {
        let (x, y) = sample_powf(&mut r);
        let [a, b] = Dual::<f64, 2>::variables([x, y]);
        let p = a.powf(b);
        let (xd, yd) = (dd(x), dd(y));
        let want_x = yd * xd.powf(yd - 1.0);
        let cond = (y * x.abs().ln()).abs().max(1.0);
        if normal(want_x) {
            let err = rel_err(p.eps()[0], want_x) / U;
            assert!(
                err <= 6.0,
                "∂/∂x {x:e}^{y:e} = {p:?}, expected {want_x:?}: {err} u"
            );
            wx = wx.max(err);
            if want_x.abs().hi() >= two_pow(-960) && want_x.abs().hi() <= two_pow(1000) {
                let q = X::variables([xd, yd]);
                let got = q[0].powf(q[1]).eps()[0];
                let err = ((got - want_x) / want_x).abs().hi();
                assert!(
                    err <= two_pow(-96) * cond,
                    "DD ∂/∂x {x:e}^{y:e}: {got:?} vs {want_x:?}"
                );
                dx_dd = dx_dd.max(err / U2 / cond);
            }
        }
        if x < 0.0 {
            assert!(
                p.eps()[1].is_nan(),
                "∂/∂y {x:e}^{y:e} = {p:?}: (negative x)^y has none"
            );
            continue;
        }
        let want_y = xd.powf(yd) * xd.ln();
        if normal(want_y) {
            let err = rel_err(p.eps()[1], want_y) / U;
            assert!(
                err <= 9.0,
                "∂/∂y {x:e}^{y:e} = {p:?}, expected {want_y:?}: {err} u"
            );
            wy = wy.max(err);
            if want_y.abs().hi() >= two_pow(-960) && want_y.abs().hi() <= two_pow(1000) {
                let q = X::variables([xd, yd]);
                let got = q[0].powf(q[1]).eps()[1];
                let err = ((got - want_y) / want_y).abs().hi();
                assert!(
                    err <= two_pow(-96) * cond,
                    "DD ∂/∂y {x:e}^{y:e}: {got:?} vs {want_y:?}"
                );
                dy_dd = dy_dd.max(err / U2 / cond);
            }
        }
    }
    eprintln!(
        "powf: ∂/∂x worst {wx:.2} u (bound 6), ∂/∂y {wy:.2} u (bound 9); DD {dx_dd:.1} and {dy_dd:.1} u² per unit of max(1, |y ln x|)"
    );

    let (mut wh, mut wa, mut hdd, mut add) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for _ in 0..30_000 {
        let (x, y) = sample_pair(&mut r);
        let [a, b] = Dual::<f64, 2>::variables([x, y]);
        let (h, t) = (a.hypot(b), a.atan2(b));
        let (xd, yd) = (dd(x), dd(y));
        let (want_h, want_t) = (scaled_hypot_partials(xd, yd), scaled_atan2_partials(xd, yd));
        let moderate = |v: f64| v.abs() >= two_pow(-900) && v.abs() <= two_pow(900);
        let q = X::variables([xd, yd]);
        let (hq, tq) = (q[0].hypot(q[1]), q[0].atan2(q[1]));
        for i in 0..2 {
            if normal(want_h[i]) {
                let err = rel_err(h.eps()[i], want_h[i]) / U;
                assert!(
                    err <= 6.0,
                    "hypot({x:e}, {y:e}) = {h:?}, expected {want_h:?}: {err} u"
                );
                wh = wh.max(err);
                if moderate(x) && moderate(y) && want_h[i].abs().hi() >= two_pow(-960) {
                    let err = ((hq.eps()[i] - want_h[i]) / want_h[i]).abs().hi();
                    assert!(
                        err <= two_pow(-96),
                        "DD hypot({x:e}, {y:e}): {hq:?} vs {want_h:?}"
                    );
                    hdd = hdd.max(err / U2);
                }
            }
            if normal(want_t[i]) {
                let err = rel_err(t.eps()[i], want_t[i]) / U;
                assert!(
                    err <= 7.0,
                    "atan2({x:e}, {y:e}) = {t:?}, expected {want_t:?}: {err} u"
                );
                wa = wa.max(err);
                let tiny =
                    want_t[i].abs().hi() < two_pow(-960) || want_t[i].abs().hi() > two_pow(1000);
                if moderate(x) && moderate(y) && !tiny {
                    let err = ((tq.eps()[i] - want_t[i]) / want_t[i]).abs().hi();
                    assert!(
                        err <= two_pow(-96),
                        "DD atan2({x:e}, {y:e}): {tq:?} vs {want_t:?}"
                    );
                    add = add.max(err / U2);
                }
            }
        }
    }
    eprintln!(
        "hypot: worst {wh:.2} u (bound 6), DD {hdd:.1} u²; atan2: worst {wa:.2} u (bound 7), DD {add:.1} u²"
    );

    let (mut lx, mut lb) = (0.0_f64, 0.0_f64);
    for _ in 0..20_000 {
        let x = r.pow2(-100, 100);
        let b = loop {
            let b = r.pow2(-10, 10);
            if (b - 1.0).abs() > two_pow(-10) {
                break b;
            }
        };
        let [p, q] = Dual::<f64, 2>::variables([x, b]);
        let l = p.log(q);
        let (xd, bd) = (dd(x), dd(b));
        let (lnx, lnb) = (xd.ln(), bd.ln());
        let want = [Dd::ONE / xd / lnb, -(lnx / lnb / (bd * lnb))];
        let ex = rel_err(l.eps()[0], want[0]) / U;
        assert!(
            ex <= 7.0,
            "∂/∂x log({x:e}, {b:e}) = {l:?}, expected {want:?}: {ex} u"
        );
        lx = lx.max(ex);
        if normal(want[1]) {
            let eb = rel_err(l.eps()[1], want[1]) / U;
            assert!(
                eb <= 17.0,
                "∂/∂b log({x:e}, {b:e}) = {l:?}, expected {want:?}: {eb} u"
            );
            lb = lb.max(eb);
        }
    }
    eprintln!("log(x, b): ∂/∂x worst {lx:.2} u (bound 7), ∂/∂b {lb:.2} u (bound 17)");
}

// ---- Complex<Dual> ------------------------------------------------------------------------------

type C2 = Complex<Dual<f64, 2>>;
type Cdd = Complex<Dd>;

/// z with ∂/∂(Re z) and ∂/∂(Im z) seeded.
fn seeded(re: f64, im: f64) -> C2 {
    let [x, y] = Dual::<f64, 2>::variables([re, im]);
    Complex::new(x, y)
}

fn cdd(re: f64, im: f64) -> Cdd {
    Complex::new(Dd::from_f64(re), Dd::from_f64(im))
}

/// |a − b| in double-double.
fn cdist(a: Cdd, b: Cdd) -> f64 {
    (a - b).norm_sqr().sqrt().hi()
}

/// For a holomorphic f: ∂f/∂x = f′(z) and ∂f/∂y = i·f′(z). The larger error, relative to
/// |f′(z)|, in u.
fn holomorphic_error(w: C2, fp: Cdd) -> f64 {
    let part = |k: usize| Complex::new(Dd::from_f64(w.re.eps()[k]), Dd::from_f64(w.im.eps()[k]));
    let i_fp = Complex::new(-fp.im, fp.re);
    let scale = fp.norm_sqr().sqrt().hi();
    cdist(part(0), fp).max(cdist(part(1), i_fp)) / (scale * U)
}

/// The value of `w` against `want`, relative to |want|, in u.
fn value_error(w: C2, want: Cdd) -> f64 {
    cdist(cdd(w.re.re(), w.im.re()), want) / (want.norm_sqr().sqrt().hi() * U)
}

/// Points on both axes, on the real axis with either sign of zero, and off the axes.
const POINTS: [(f64, f64); 14] = [
    (2.0, 0.0),
    (2.0, -0.0),
    (-3.0, 0.0),
    (-3.0, -0.0),
    (0.0, 1.5),
    (-0.0, 1.5),
    (0.0, -0.7),
    (0.6, -1.3),
    (-2.5, 0.4),
    (-2.5, -0.4),
    (1e-3, 2e-3),
    (30.0, -20.0),
    (0.25, 0.0),
    (-1e-4, 0.0),
];

#[test]
fn complex_of_dual_derivatives_match_the_analytic_ones() {
    // num-complex evaluates these through f64 functions of a few ulps each (polar form for
    // sqrt, cbrt and powf; exp(w·ln z) for powc; (ac + bd)/(c² + d²) for division), and each
    // term of a derivative is bounded by |f′(z)|, so the error is a few u per function; powc
    // errs as its value does, by about 4u·|w·ln z| (≤ 8.5 at these points). Bound: 64u.
    type C64 = Complex<f64>;
    type Case = (
        &'static str,
        fn(C2) -> C2,
        fn(C64) -> C64,
        fn(Cdd) -> (Cdd, Cdd),
    );
    let cases: [Case; 9] = [
        ("exp", |z| z.exp(), |z| z.exp(), |z| (z.exp(), z.exp())),
        ("ln", |z| z.ln(), |z| z.ln(), |z| (z.ln(), z.inv())),
        (
            "sqrt",
            |z| z.sqrt(),
            |z| z.sqrt(),
            |z| {
                let s = z.sqrt();
                (s, (s + s).inv())
            },
        ),
        (
            "cbrt",
            |z| z.cbrt(),
            |z| z.cbrt(),
            |z| {
                let c = z.cbrt();
                (c, (c * c * Dd::from_f64(3.0)).inv())
            },
        ),
        (
            "powf(2.5)",
            |z| z.powf(Real::from_f64(2.5)),
            |z| z.powf(2.5),
            |z| {
                let f = z.powf(Dd::from_f64(2.5));
                (f, f / z * Dd::from_f64(2.5))
            },
        ),
        (
            "powf(-1/3)",
            |z| z.powf(Real::from_f64(-1.0 / 3.0)),
            |z| z.powf(-1.0 / 3.0),
            |z| {
                let y = Dd::from_f64(-1.0 / 3.0);
                let f = z.powf(y);
                (f, f / z * y)
            },
        ),
        (
            "powc(0.5 − 2i)",
            |z| z.powc(Complex::new(Real::from_f64(0.5), Real::from_f64(-2.0))),
            |z| z.powc(Complex::new(0.5, -2.0)),
            |z| {
                let w = cdd(0.5, -2.0);
                let f = z.powc(w);
                (f, f / z * w)
            },
        ),
        (
            "1/z",
            |z| Complex::new(Dual::one(), Dual::zero()) / z,
            |z| 1.0 / z,
            |z| (z.inv(), -(z * z).inv()),
        ),
        ("sin", |z| z.sin(), |z| z.sin(), |z| (z.sin(), z.cos())),
    ];
    let mut report = String::new();
    for (name, f, f64_version, reference) in cases {
        let (mut wd, mut wv, mut w64) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (re, im) in POINTS {
            let w = f(seeded(re, im));
            let (value, fp) = reference(cdd(re, im));
            let err = holomorphic_error(w, fp);
            assert!(
                err <= 64.0,
                "{name}({re:?}, {im:?}): {w:?}, f′ = {fp:?}: {err:.1} u"
            );
            wd = wd.max(err);
            let ev = value_error(w, value);
            assert!(
                ev <= 64.0,
                "value of {name}({re:?}, {im:?}): {w:?} vs {value:?}: {ev:.1} u"
            );
            wv = wv.max(ev);
            // Against Complex<f64> with the same num-complex function: equal, or (where a part
            // is zero in value but has a derivative, so another path is taken) within a few
            // ulps of the result, 8u.
            let v64 = f64_version(Complex::new(re, im));
            let e64 = value_error(w, cdd(v64.re, v64.im));
            assert!(
                e64 <= 8.0,
                "value of {name}({re:?}, {im:?}): {w:?} vs Complex<f64> {v64:?}: {e64:.1} u"
            );
            w64 = w64.max(e64);
        }
        report += &format!(" {name} {wd:.1} u ({wv:.1} u in value, {w64:.1} u from Complex<f64>);");
    }
    eprintln!("Complex<Dual<f64, 2>> worst derivative errors:{report}");
}

#[test]
fn complex_of_dual_norm_and_quotient() {
    let mut worst = 0.0_f64;
    // |z|: ∂/∂x = x/|z|, ∂/∂y = y/|z| (not holomorphic).
    for (re, im) in POINTS {
        let n = seeded(re, im).norm();
        let h = cdd(re, im).norm_sqr().sqrt();
        for (k, c) in [re, im].into_iter().enumerate() {
            let want = Dd::from_f64(c) / h;
            let err = (Dd::from_f64(n.eps()[k]) - want).abs().hi() / U;
            assert!(err <= 8.0, "|({re:?}, {im:?})|: {n:?}");
            worst = worst.max(err);
        }
    }
    // At the origin |z| has no derivative: hypot's zero derivatives.
    assert_eq!(seeded(0.0, 0.0).norm().eps(), &[0.0, 0.0]);
    // z/w with all four parts seeded: ∂/∂z = 1/w and ∂/∂w = −z/w².
    for ((a, b), (c, d)) in [
        ((1.0, 2.0), (3.0, -1.0)),
        ((-2.0, 0.0), (0.0, 0.5)),
        ((0.0, 1.0), (-4.0, 0.0)),
    ] {
        let v = Dual::<f64, 4>::variables([a, b, c, d]);
        let q = Complex::new(v[0], v[1]) / Complex::new(v[2], v[3]);
        let (z, w) = (cdd(a, b), cdd(c, d));
        let (dz, dw) = (w.inv(), -(z / (w * w)));
        for (k, fp) in [(0, dz), (2, dw)] {
            let scale = fp.norm_sqr().sqrt().hi();
            let part =
                |k: usize| Complex::new(Dd::from_f64(q.re.eps()[k]), Dd::from_f64(q.im.eps()[k]));
            let err = cdist(part(k), fp).max(cdist(part(k + 1), Complex::new(-fp.im, fp.re)))
                / (scale * U);
            assert!(err <= 64.0, "({a}, {b})/({c}, {d}) direction {k}: {q:?}");
            worst = worst.max(err);
        }
    }
    eprintln!("Complex<Dual> norm and quotient: worst {worst:.1} u");
}

#[test]
fn complex_of_dual_exponent_derivatives_at_zero_exponents() {
    // Strict is_zero: an exponent zero in value but with a derivative does not take num-complex's
    // `exp.is_zero()` shortcut, so ∂/∂y z^y = z^y·ln z = ln z at y = 0.
    type D3 = Dual<f64, 3>;
    for (re, im) in POINTS {
        let [x, y, e] = D3::variables([re, im, 0.0]);
        let z = Complex::new(x, y);
        let lnz = cdd(re, im).ln();
        let scale = lnz.norm_sqr().sqrt().hi();
        // powf: ∂/∂e is ln z; ∂/∂z is e·z^(e−1) = 0.
        let p = z.powf(e);
        assert!(
            p.re.re() == 1.0 && p.im.re() == 0.0,
            "{re:?}, {im:?}: {p:?}"
        );
        let got = Complex::new(Dd::from_f64(p.re.eps()[2]), Dd::from_f64(p.im.eps()[2]));
        assert!(
            cdist(got, lnz) <= 16.0 * U * scale,
            "∂/∂y ({re:?}, {im:?})^y: {p:?} vs {lnz:?}"
        );
        assert_eq!(
            (&p.re.eps()[..2], &p.im.eps()[..2]),
            (&[0.0, 0.0][..], &[0.0, 0.0][..])
        );
        // powc with w = e + 0i: ∂/∂(Re w) = ln z.
        let c = z.powc(Complex::new(e, D3::zero()));
        let got = Complex::new(Dd::from_f64(c.re.eps()[2]), Dd::from_f64(c.im.eps()[2]));
        assert!(
            cdist(got, lnz) <= 16.0 * U * scale,
            "∂/∂w ({re:?}, {im:?})^w: {c:?} vs {lnz:?}"
        );
    }
    // Away from zero: ∂/∂y z^y = z^y·ln z, ∂/∂(Im w) z^w = i·z^w·ln z.
    let [x, y, e] = D3::variables([-1.5, 0.5, 0.75]);
    let (z, zd) = (Complex::new(x, y), cdd(-1.5, 0.5));
    let want = zd.powf(Dd::from_f64(0.75)) * zd.ln();
    let p = z.powf(e);
    let got = Complex::new(Dd::from_f64(p.re.eps()[2]), Dd::from_f64(p.im.eps()[2]));
    assert!(
        cdist(got, want) <= 64.0 * U * want.norm_sqr().sqrt().hi(),
        "{p:?} vs {want:?}"
    );
    let c = z.powc(Complex::new(D3::constant(0.3), e));
    let want = zd.powc(cdd(0.3, 0.75)) * zd.ln();
    let got = Complex::new(Dd::from_f64(c.re.eps()[2]), Dd::from_f64(c.im.eps()[2]));
    let i_want = Complex::new(-want.im, want.re);
    assert!(
        cdist(got, i_want) <= 64.0 * U * want.norm_sqr().sqrt().hi(),
        "{c:?} vs {i_want:?}"
    );
}

#[test]
fn complex_of_double_double_dual_keeps_full_precision_at_im_zero() {
    // sqrt and cbrt at im = 0 with ∂/∂(Im z) seeded, for re > 0 and re < 0, in double-double.
    // The polar path chains hypot, atan2, sqrt or cbrt, cos and sin, each within 2^-99, so
    // 2^-94 bounds the error relative to |f′|.
    for re in [2.0, -3.0, 0.3, -1e-5] {
        let [x, y] = Dual::<Dd, 2>::variables([Dd::from_f64(re), Dd::ZERO]);
        let z = Complex::new(x, y);
        for (name, w, fp) in [
            ("sqrt", z.sqrt(), {
                let s = cdd(re, 0.0).sqrt();
                (s + s).inv()
            }),
            ("cbrt", z.cbrt(), {
                let c = cdd(re, 0.0).cbrt();
                (c * c * Dd::from_f64(3.0)).inv()
            }),
        ] {
            let scale = fp.norm_sqr().sqrt().hi();
            let dx = Complex::new(w.re.eps()[0], w.im.eps()[0]);
            let dy = Complex::new(w.re.eps()[1], w.im.eps()[1]);
            let err = cdist(dx, fp).max(cdist(dy, Complex::new(-fp.im, fp.re))) / scale;
            assert!(
                err <= two_pow(-94),
                "{name}({re}): {w:?} vs {fp:?}: {err:e}"
            );
        }
    }
}

#[test]
fn documented_value_consequences_of_strict_zero_tests() {
    // With im zero in value but carrying a derivative, num-complex takes the general polar path
    // instead of its shortcuts; values then differ from Complex<f64> by a few ulps (documented on
    // `Dual`, "Comparisons and zero tests").
    type D = Dual<f64, 2>;
    let k = D::new(0.0, [0.0, 1.0]); // zero value, nonzero derivative
    let ulps = |a: f64, b: f64| (a - b).abs() / (f64::EPSILON * b.abs().max(f64::MIN_POSITIVE));
    // sqrt(−4 + 0i): Re = 2·cos(π/2) ≈ 1.2e-16 instead of 0.
    let s = Complex::new(D::constant(-4.0), k).sqrt();
    assert_eq!(Complex::new(-4.0_f64, 0.0).sqrt(), Complex::new(0.0, 2.0));
    assert!(
        s.re.re() > 0.0 && s.re.re() < 2.0 * f64::EPSILON && s.im.re() == 2.0,
        "{s:?}"
    );
    // On the imaginary axis with re′ ≠ 0: sqrt and cbrt within 2 ulps of Complex<f64> (each
    // path is within about an ulp of the exact value; cbrt always differs).
    let mut r = Rng::new("imaginary axis");
    let (mut ws, mut wc, mut cbrt_equal) = (0.0_f64, 0.0_f64, 0);
    for i in 0..20_000 {
        let b = if i < 6 {
            [1.5, -0.7, 4.0, 1e-3, 1e10, -2.5e-7][i]
        } else {
            let b = r.pow2(-300, 300);
            r.sign(b)
        };
        let z = Complex::new(D::new(0.0, [1.0, 0.0]), D::constant(b));
        let (s, c) = (z.sqrt(), z.cbrt());
        let (s64, c64) = (Complex::new(0.0, b).sqrt(), Complex::new(0.0, b).cbrt());
        let es = ulps(s.re.re(), s64.re).max(ulps(s.im.re(), s64.im));
        let ec = ulps(c.re.re(), c64.re).max(ulps(c.im.re(), c64.im));
        assert!(es <= 2.0, "sqrt(i{b}): {s:?} vs {s64:?}");
        assert!(ec <= 2.0, "cbrt(i{b}): {c:?} vs {c64:?}");
        (ws, wc) = (ws.max(es), wc.max(ec));
        cbrt_equal += usize::from(c.re.re() == c64.re && c.im.re() == c64.im);
    }
    eprintln!(
        "imaginary axis, against Complex<f64>: sqrt within {ws:.2} ulps, cbrt within {wc:.2} ulps \
         ({cbrt_equal} of 20000 cbrt values equal)"
    );
    // powf with a value-zero exponent: 1 with Im = −0 where arg z < 0, NaN for a NaN base.
    let p = Complex::new(D::constant(-1.0), D::constant(-1.0)).powf(k);
    assert!(p.re.re() == 1.0 && same(p.im.re(), -0.0), "{p:?}");
    assert_eq!(
        Complex::new(-1.0_f64, -1.0).powf(0.0),
        Complex::new(1.0, 0.0)
    );
    assert!(
        Complex::new(D::constant(f64::NAN), D::zero())
            .powf(k)
            .re
            .re()
            .is_nan()
    );
    // powc returns NaN at z = 0 and z = ∞ instead of 1.
    let w = Complex::new(k, D::zero());
    assert!(Complex::new(D::zero(), D::zero()).powc(w).re.re().is_nan());
    assert!(
        Complex::new(D::constant(f64::INFINITY), D::zero())
            .powc(w)
            .re
            .re()
            .is_nan()
    );
    assert_eq!(
        Complex::new(0.0_f64, 0.0).powc(Complex::new(0.0, 0.0)),
        Complex::new(1.0, 0.0)
    );
    // ToPrimitive: None when im is zero in value but carries a derivative.
    assert_eq!(Complex::new(D::constant(2.0), k).to_f64(), None);
    assert_eq!(
        Complex::new(D::constant(2.0), D::zero()).to_f64(),
        Some(2.0)
    );
    // `is_finite` sees derivatives, so num-complex's exp takes its corner case for Re z = +∞ when
    // Im z has an infinite derivative, and returns Im z unchanged: (∞, 1), not (∞·cos 1, ∞·sin 1).
    let e = Complex::new(
        D::constant(f64::INFINITY),
        D::new(1.0, [0.0, f64::INFINITY]),
    )
    .exp();
    assert_eq!((e.re.re(), e.im.re()), (f64::INFINITY, 1.0));
    assert_eq!(
        Complex::new(f64::INFINITY, 1.0).exp(),
        Complex::new(f64::INFINITY, f64::INFINITY)
    );
}

// ---- Generic kernels --------------------------------------------------------------------------

/// P_n(x) and P_n′(x) by the three-term recurrence.
fn legendre<T: Real>(n: usize, x: T) -> (T, T) {
    let (mut p0, mut p1) = (T::one(), x);
    for k in 2..=n {
        let k = k as f64;
        let p2 = (x * p1 * (2.0 * k - 1.0) - p0 * (k - 1.0)) / k;
        p0 = p1;
        p1 = p2;
    }
    (p1, (x * p1 - p0) * (n as f64) / (x * x - 1.0))
}

/// Gauss–Legendre nodes and weights on [−1, 1]: Newton's method from cos(π(i − ¼)/(n + ½))
/// until a step is below √ε, then one more step.
fn gauss_legendre<T: Real>(n: usize) -> Vec<(T, T)> {
    let tol = T::epsilon().sqrt();
    (1..=n)
        .map(|i| {
            let mut x = (T::PI() * ((i as f64 - 0.25) / (n as f64 + 0.5))).cos();
            for step in 0.. {
                assert!(step < 50, "Newton's method did not converge");
                let (p, dp) = legendre(n, x);
                let dx = p / dp;
                x -= dx;
                if dx.abs() <= tol {
                    break;
                }
            }
            let (p, dp) = legendre(n, x);
            x -= p / dp;
            let (_, dp) = legendre(n, x);
            (x, (dp * dp * (T::one() - x * x)).recip() * 2.0)
        })
        .collect()
}

#[test]
fn gauss_legendre_newton_loops_run_on_duals() {
    // As constants the nodes and weights are those of T, bit for bit, with zero derivatives.
    let plain = gauss_legendre::<f64>(20);
    for (i, &(x, w)) in gauss_legendre::<Dual<f64, 4>>(20).iter().enumerate() {
        assert!(
            same(x.re(), plain[i].0) && same(w.re(), plain[i].1),
            "node {i}"
        );
        assert_eq!((x.eps(), w.eps()), (&[0.0; 4], &[0.0; 4]));
    }
    let plain = gauss_legendre::<Dd>(24);
    let dual = gauss_legendre::<Dual<Dd, 2>>(24);
    for (i, &(x, w)) in dual.iter().enumerate() {
        assert!(
            same_dd(x.re(), plain[i].0) && same_dd(w.re(), plain[i].1),
            "node {i}"
        );
        assert_eq!((x.eps(), w.eps()), (&[Dd::ZERO; 2], &[Dd::ZERO; 2]));
    }
    let total = plain.iter().fold(Dd::ZERO, |s, &(_, w)| s + w);
    assert!((total - 2.0).abs().hi() <= 64.0 * U2, "{total:?}");

    // Differentiating through the rule: ∫ e^(px)(1 + qx) dx over [−1, 1] is
    // 2 sinh(p)/p + q·(2 cosh(p)/p − 2 sinh(p)/p²); its derivatives in p and q are smooth, so a
    // 24-point rule is exact to double-double precision.
    let (p0, q0) = (0.7, -1.3);
    let [p, q] = Dual::<Dd, 2>::variables([Dd::from_f64(p0), Dd::from_f64(q0)]);
    let integral = dual.iter().fold(Dual::zero(), |s, &(x, w)| {
        s + w * (p * x).exp() * (q * x + 1.0)
    });
    let (pd, qd) = (Dd::from_f64(p0), Dd::from_f64(q0));
    let (sh, ch) = (pd.sinh(), pd.cosh());
    let value = sh * 2.0 / pd + qd * (ch * 2.0 / pd - sh * 2.0 / (pd * pd));
    // d/dp: d/dp [2 sinh p/p] = 2(p cosh p − sinh p)/p²; d/dp [2 cosh p/p − 2 sinh p/p²] =
    // 2(p sinh p − cosh p)/p² − 2(p cosh p − 2 sinh p)/p³.
    let p2 = pd * pd;
    let d_p = (pd * ch - sh) * 2.0 / p2
        + qd * ((pd * sh - ch) * 2.0 / p2 - (pd * ch - sh * 2.0) * 2.0 / (p2 * pd));
    let d_q = ch * 2.0 / pd - sh * 2.0 / p2;
    for (got, want, what) in [
        (integral.re(), value, "value"),
        (integral.eps()[0], d_p, "d/dp"),
        (integral.eps()[1], d_q, "d/dq"),
    ] {
        let err = ((got - want) / want).abs().hi();
        assert!(err <= 1024.0 * U2, "{what}: {got:?} vs {want:?}: {err:e}");
    }
    // The same in f64.
    let rule = gauss_legendre::<Dual<f64, 1>>(20);
    let [p] = Dual::<f64, 1>::variables([p0]);
    let i = rule
        .iter()
        .fold(Dual::zero(), |s, &(x, w)| s + w * (p * x).exp());
    assert!((i.re() - 2.0 * p0.sinh() / p0).abs() <= 1e-15);
    assert!((i.eps()[0] - 2.0 * (p0 * p0.cosh() - p0.sinh()) / (p0 * p0)).abs() <= 1e-15);
}

/// The root of x³ + p·x − 1 by Newton's method, stopping on the value.
fn cubic_root<T: Real>(p: T) -> T {
    let mut x = T::one();
    for step in 0.. {
        assert!(step < 50, "Newton's method did not converge");
        let dx = (x * x * x + p * x - 1.0) / (x * x * 3.0 + p);
        x -= dx;
        if dx.abs() <= T::epsilon() * 4.0 {
            break;
        }
    }
    x
}

#[test]
fn newton_iteration_differentiates_through_the_root() {
    // dx/dp = −x/(3x² + p) by the implicit function theorem; the last Newton step, taken where
    // the residual is at rounding level, also converges the derivative.
    let x = cubic_root(Dual::<f64, 1>::variables([2.0])[0]);
    let want = -x.re() / (3.0 * x.re() * x.re() + 2.0);
    assert!((x.re().powi(3) + 2.0 * x.re() - 1.0).abs() <= 4.0 * f64::EPSILON);
    assert!(
        ((x.eps()[0] - want) / want).abs() <= 8.0 * U,
        "{x:?} vs {want}"
    );
    let pd = Dd::from_f64(2.0);
    let x = cubic_root(Dual::<Dd, 1>::variables([pd])[0]);
    let want = -(x.re() / (x.re() * x.re() * 3.0 + pd));
    assert!(
        (((x.eps()[0] - want) / want).abs().hi()) <= 64.0 * U2,
        "{x:?} vs {want:?}"
    );
    // Horner over double-double duals.
    let [t] = Dual::<Dd, 1>::variables([Dd::from_f64(1.0) / Dd::from_f64(3.0)]);
    let h = [1.0, -2.0, 0.5, 4.0]
        .iter()
        .rev()
        .fold(Dual::zero(), |acc, &c| acc * t + c);
    let td = t.re();
    let want = Dd::from_f64(-2.0) + td * 1.0 + td * td * 12.0; // d/dt (1 − 2t + t²/2 + 4t³)
    assert!(
        ((h.eps()[0] - want) / want).abs().hi() <= 16.0 * U2,
        "{h:?}"
    );
}

// ---- Properties over arbitrary arguments ------------------------------------------------------

/// Doubles from the whole range, with the special values over-represented.
fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        any::<u64>().prop_map(f64::from_bits),
        prop::sample::select(SPECIAL.to_vec()),
    ]
}

/// Finite doubles, often of moderate size.
fn finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        any::<u64>()
            .prop_map(f64::from_bits)
            .prop_filter("finite", |x| x.is_finite()),
        -10.0..10.0f64,
        prop::sample::select(
            SPECIAL
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .collect::<Vec<_>>()
        ),
    ]
}

/// An expression using most methods, with special paths reachable from its arguments.
fn composite<T: Real>(x: T, y: T) -> T {
    let a = x.abs().sqrt() + y.exp_m1() * x.sin() - x.cos().powi(3);
    let b = x.hypot(y).ln_1p() + y.atan2(x) - (x * y).tanh() + x.cbrt();
    let c = x.abs().powf(y.abs().min(T::from_f64(3.0))) + y.cosh().recip() + x.asinh();
    let d = (x * 0.25).atan() + x.max(y) - x.min(y).abs_sub(y) + y.exp2().log2() + x.tan();
    let e = (x * 0.5).sinh() + y.abs().ln() + x.abs().log10() + y.mul_add(x, x).exp();
    let f = (y * 0.1).acos() + (x * 0.1).asin() + (x.abs() + 1.0).acosh() + (y * 0.5).atanh();
    let g = x.recip().powf(y) + y.copysign(x).fract() + x.clamp(-y.abs(), y.abs()) + y.log(x.abs());
    a * b + c / d + e - f * g
}

/// `composite` with seeds `s` in direction `k` of an N-direction dual and `other` elsewhere: the
/// value and direction `k`'s derivative.
fn composite_seeded<const N: usize>(
    v: [f64; 2],
    s: [f64; 2],
    other: [[f64; 8]; 2],
    k: usize,
) -> (f64, f64) {
    let d = |i: usize| {
        let mut eps: [f64; N] = core::array::from_fn(|j| other[i][j]);
        eps[k] = s[i];
        Dual::<f64, N>::new(v[i], eps)
    };
    let r = composite(d(0), d(1));
    (r.re(), r.eps()[k])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn each_direction_is_independent_of_n_and_of_the_other_seeds(
        v in [any_f64(), any_f64()],
        s in [any_f64(), any_f64()],
        other in [prop::array::uniform8(any_f64()), prop::array::uniform8(any_f64())],
        k in 0usize..8,
    ) {
        // Every rule is componentwise; a path choice depends on the values (and for powf on
        // whether the exponent varies, which changes only the sign of a zero derivative).
        let (re1, d1) = composite_seeded::<1>(v, s, other, 0);
        for (re, d) in [
            composite_seeded::<2>(v, s, other, k % 2),
            composite_seeded::<4>(v, s, other, k % 4),
            composite_seeded::<8>(v, s, other, k),
        ] {
            prop_assert!(same(re, re1), "value {re:?} vs {re1:?}");
            prop_assert!(same_derivative(d, d1), "derivative {d:?} vs {d1:?}");
        }
    }

    #[test]
    fn unary_derivatives_are_nan_only_with_nan_values(x in any_f64()) {
        // Inside the closed domain every derivative is finite or a signed infinity. Exceptions
        // by design: (−2)^y is real only at integers y and has no derivative in y; and a base of
        // ±0 is no base: log(3, ±0) = −0 by IEEE rules, and the quotient rule for
        // ln 3/ln b meets ∞·0 there.
        let excepted = |name: &str| name == "(-2)^x" || (name == "log(3, x)" && x == 0.0);
        for (name, d, _) in unary_methods::<f64, 1>().into_iter().filter(|m| !excepted(m.0)) {
            let r = d(F1::new(x, [1.0]));
            prop_assert!(r.re().is_nan() || !r.eps()[0].is_nan(), "{name}({x:?}) = {r:?}");
        }
        for (name, d, _) in unary_methods::<Dd, 1>().into_iter().filter(|m| !excepted(m.0)) {
            let r = d(Dual::new(Dd::from_f64(x), [Dd::ONE]));
            prop_assert!(r.re().hi().is_nan() || !r.eps()[0].hi().is_nan(), "{name}({x:?}) = {r:?}");
        }
    }

    #[test]
    fn binary_derivatives_of_finite_arguments_are_never_nan_without_reason(
        x in finite_f64(), y in finite_f64(), dx in -1.0..1.0f64, dy in -1.0..1.0f64,
    ) {
        let (a, b) = (F3::new(x, [dx, 0.0, 0.0]), F3::new(y, [0.0, dy, 0.0]));
        let not_nan = |r: F3, i: usize| r.re().is_nan() || !r.eps()[i].is_nan();
        let p = a.powf(b);
        // (negative x)^y has no derivative in y.
        prop_assert!(not_nan(p, 0) && (x < 0.0 || not_nan(p, 1)), "{x:?}^{y:?} = {p:?}");
        let h = a.hypot(b);
        prop_assert!(!h.eps()[0].is_nan() && !h.eps()[1].is_nan(), "hypot({x:?}, {y:?}) = {h:?}");
        // atan2 jumps at the origin.
        let t = a.atan2(b);
        prop_assert!((x == 0.0 && y == 0.0) || (not_nan(t, 0) && not_nan(t, 1)), "atan2({x:?}, {y:?}) = {t:?}");
        let m = a.mul_add(b, a);
        prop_assert!(not_nan(m, 0) && not_nan(m, 1), "mul_add({x:?}, {y:?}) = {m:?}");
        if x > 0.0 && y > 0.0 && y != 1.0 {
            let l = a.log(b);
            prop_assert!(not_nan(l, 0) && not_nan(l, 1), "log({x:?}, {y:?}) = {l:?}");
        }
        for r in [p, h, t, m] {
            prop_assert_eq!(r.eps()[2], 0.0);
        }
    }
}

#[test]
fn nested_duals_give_second_derivatives() {
    // Dual<Dual<T, 1>, 1> is a Real too: seeding both levels in the same direction gives f″.
    type In = Dual<f64, 1>;
    type Out = Dual<In, 1>;
    let at = |x: f64| Out::new(In::new(x, [1.0]), [In::new(1.0, [0.0])]);
    let second = |r: Out| r.eps()[0].eps()[0];
    let x = 0.7_f64;
    type Case = (&'static str, fn(Out) -> Out, f64);
    let cases: [Case; 7] = [
        ("sin", |v| v.sin(), -x.sin()),
        ("exp", |v| v.exp(), x.exp()),
        ("sqrt", |v| v.sqrt(), -0.25 * x.powf(-1.5)),
        ("ln", |v| v.ln(), -1.0 / (x * x)),
        ("atan", |v| v.atan(), -2.0 * x / (1.0 + x * x).powi(2)),
        ("tanh", |v| v.tanh(), -2.0 * x.tanh() / x.cosh().powi(2)),
        (
            "powf(2.5)",
            |v| v.powf(Real::from_f64(2.5)),
            2.5 * 1.5 * x.powf(0.5),
        ),
    ];
    for (name, f, want) in cases {
        let got = second(f(at(x)));
        assert!(
            ((got - want) / want).abs() <= 16.0 * U,
            "{name}″({x}) = {got}, expected {want}"
        );
    }
    // Strong zeros nest: an unseeded outer direction stays zero at a singular point.
    let r = Out::new(In::new(0.0, [1.0]), [In::new(0.0, [0.0])]).sqrt();
    assert_eq!((r.eps()[0].re(), r.eps()[0].eps()[0]), (0.0, 0.0));
}
