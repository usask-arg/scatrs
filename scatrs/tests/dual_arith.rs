use num_complex::Complex;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use proptest::prelude::*;
use scatrs::scalar::{DoubleDouble as Dd, Dual};

type D2 = Dual<f64, 2>;

#[test]
fn product_quotient_and_sum_rules() {
    let [x, y] = D2::variables([3.0, 2.0]);
    let f = x * y + x / y - y;
    assert_eq!(f.re(), 3.0 * 2.0 + 1.5 - 2.0);
    assert_eq!(f.eps(), &[2.0 + 0.5, 3.0 - 0.75 - 1.0]);
}

#[test]
fn mixed_f64_operations() {
    let [x, _] = D2::variables([3.0, 2.0]);
    let f = (x * 2.0 + 1.0 - 0.5) / 4.0;
    assert_eq!(f.re(), 1.625);
    assert_eq!(f.eps(), &[0.5, 0.0]);
    let mut g = x;
    g += x;
    g *= 2.0;
    g -= 1.0;
    g /= x;
    assert_eq!(g.re(), 11.0 / 3.0);
    assert!((g.eps()[0] - 1.0 / 9.0).abs() < 1e-16);
}

#[test]
fn negation_and_remainder() {
    let [x] = Dual::<f64, 1>::variables([7.5]);
    assert_eq!((-x).eps(), &[-1.0]);
    let r = x % Dual::constant(2.0);
    assert_eq!(r.re(), 1.5);
    assert_eq!(r.eps(), &[1.0]);
}

#[test]
fn comparisons_use_value_only() {
    assert_eq!(D2::new(1.0, [5.0, 0.0]), D2::new(1.0, [0.0, 0.0]));
    assert!(D2::new(1.0, [9.0, 9.0]) < D2::new(2.0, [0.0, 0.0]));
    // Controller decision after the Task 7 review (overrides the plan): `is_zero` and `is_one`
    // are strict, also requiring zero derivatives, because num-complex shortcuts on them
    // (`sqrt`, `cbrt`, `powf`, `powc`) and a value-only test silently drops derivatives there.
    assert!(!D2::new(0.0, [1.0, 0.0]).is_zero());
    assert!(D2::new(0.0, [0.0, 0.0]).is_zero());
}

#[test]
fn num_traits() {
    assert_eq!(D2::one().re(), 1.0);
    assert_eq!(<D2 as Num>::from_str_radix("2.5", 10).unwrap().re(), 2.5);
    assert_eq!(<D2 as NumCast>::from(3_i32).unwrap().eps(), &[0.0, 0.0]);
    assert_eq!(D2::new(-3.7, [1.0, 1.0]).to_i64(), Some(-3));
}

#[test]
fn works_over_double_double() {
    let [x, y] = Dual::<Dd, 2>::variables([Dd::from_f64(3.0), Dd::from_f64(2.0)]);
    let f = x * y;
    assert_eq!(f.eps(), &[Dd::from_f64(2.0), Dd::from_f64(3.0)]);
}

proptest! {
    #[test]
    fn cube_derivative(x in -1e3..1e3f64) {
        let [v] = Dual::<f64, 1>::variables([x]);
        let c = v * v * v;
        let expected = 3.0 * x * x;
        prop_assert!((c.eps()[0] - expected).abs() <= 4.0 * f64::EPSILON * expected.abs());
    }
}

// ---- Edge cases: values follow `T`, strong zeros, `%` quotients, accuracy, N = 0 and 1 -----

type D1 = Dual<f64, 1>;
type D3 = Dual<f64, 3>;
type Dd2 = Dual<Dd, 2>;

/// u = 2^-53.
const U: f64 = 1.110_223_024_625_156_5e-16;
/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

/// Equal bit for bit (so with the same sign of zero), or both NaN.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Both words equal bit for bit, or both NaN.
fn same_dd(a: Dd, b: Dd) -> bool {
    (a.hi().is_nan() && b.hi().is_nan())
        || (a.hi().to_bits() == b.hi().to_bits() && a.lo().to_bits() == b.lo().to_bits())
}

/// Zeros, normals, subnormals, extremes, infinities and NaN.
const SPECIAL: [f64; 18] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    -3.5,
    7.5,
    2.0,
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
];

type BinaryOp<D, T> = (&'static str, fn(D, D) -> D, fn(T, T) -> T);
type F64Op<D, T> = (&'static str, fn(D, f64) -> D, fn(T, f64) -> T);

fn binary_ops<T: scatrs::Real, const N: usize>() -> [BinaryOp<Dual<T, N>, T>; 5] {
    [
        ("+", |a, b| a + b, |a, b| a + b),
        ("-", |a, b| a - b, |a, b| a - b),
        ("*", |a, b| a * b, |a, b| a * b),
        ("/", |a, b| a / b, |a, b| a / b),
        ("%", |a, b| a % b, |a, b| a % b),
    ]
}

fn f64_ops<T: scatrs::Real, const N: usize>() -> [F64Op<Dual<T, N>, T>; 4] {
    [
        ("+ f64", |a, y| a + y, |a, y| a + y),
        ("- f64", |a, y| a - y, |a, y| a - y),
        ("* f64", |a, y| a * y, |a, y| a * y),
        ("/ f64", |a, y| a / y, |a, y| a / y),
    ]
}

/// A compound assignment and the index of its operation in `binary_ops` or `f64_ops`.
type AssignOp<D, Y> = (fn(&mut D, Y), usize);

/// The compound assignments, each paired with its binary operation.
fn assign_ops<T: scatrs::Real, const N: usize>() -> [AssignOp<Dual<T, N>, Dual<T, N>>; 5] {
    [
        (|a, b| *a += b, 0),
        (|a, b| *a -= b, 1),
        (|a, b| *a *= b, 2),
        (|a, b| *a /= b, 3),
        (|a, b| *a %= b, 4),
    ]
}

fn f64_assign_ops<T: scatrs::Real, const N: usize>() -> [AssignOp<Dual<T, N>, f64>; 4] {
    [
        (|a, y| *a += y, 0),
        (|a, y| *a -= y, 1),
        (|a, y| *a *= y, 2),
        (|a, y| *a /= y, 3),
    ]
}

#[test]
fn values_and_signed_zeros_follow_f64_bit_for_bit() {
    for a in SPECIAL {
        let x = D3::new(a, [1.0, 0.0, 0.0]);
        assert!(same((-x).re(), -a), "-{a:?}");
        for b in SPECIAL {
            let y = D3::new(b, [0.0, 1.0, 0.0]);
            for (name, d, f) in binary_ops::<f64, 3>() {
                let got = d(x, y).re();
                assert!(same(got, f(a, b)), "{a:?} {name} {b:?}: {got:?}");
            }
            for (name, d, f) in f64_ops::<f64, 3>() {
                let got = d(x, b).re();
                assert!(same(got, f(a, b)), "{a:?} {name} {b:?}: {got:?}");
            }
        }
    }
}

#[test]
fn compound_assignments_match_the_binary_operations() {
    let binary = binary_ops::<f64, 3>();
    let with_f64 = f64_ops::<f64, 3>();
    for a in SPECIAL {
        for b in SPECIAL {
            let x = D3::new(a, [1.0, 0.5, 0.0]);
            let y = D3::new(b, [0.0, -2.0, 0.0]);
            for (op, i) in assign_ops::<f64, 3>() {
                let mut z = x;
                op(&mut z, y);
                let want = binary[i].1(x, y);
                assert!(same(z.re(), want.re()), "{a:?} {}= {b:?}", binary[i].0);
                for k in 0..3 {
                    assert!(
                        same(z.eps()[k], want.eps()[k]),
                        "{a:?} {}= {b:?}",
                        binary[i].0
                    );
                }
            }
            for (op, i) in f64_assign_ops::<f64, 3>() {
                let mut z = x;
                op(&mut z, b);
                let want = with_f64[i].1(x, b);
                assert!(same(z.re(), want.re()), "{a:?} {}= {b:?}", with_f64[i].0);
                for k in 0..3 {
                    assert!(
                        same(z.eps()[k], want.eps()[k]),
                        "{a:?} {}= {b:?}",
                        with_f64[i].0
                    );
                }
            }
        }
    }
}

#[test]
fn directions_no_operand_depends_on_stay_exactly_zero() {
    for a in SPECIAL {
        for b in SPECIAL {
            let x = D3::new(a, [1.0, 0.0, 0.0]);
            let y = D3::new(b, [0.0, 1.0, 0.0]);
            for (name, d, _) in binary_ops::<f64, 3>() {
                let r = d(x, y);
                assert_eq!(r.eps()[2], 0.0, "{a:?} {name} {b:?}: {r:?}");
                let c = d(D3::constant(a), D3::constant(b));
                assert_eq!(c.eps(), &[0.0; 3], "const {a:?} {name} const {b:?}");
            }
            for (name, d, _) in f64_ops::<f64, 3>() {
                let r = d(x, b);
                assert_eq!(&r.eps()[1..], &[0.0, 0.0], "{a:?} {name} {b:?}: {r:?}");
            }
        }
    }
}

#[test]
fn derivatives_at_infinite_values_and_zero_divisors() {
    let inf = f64::INFINITY;
    let [x, y0] = D2::variables([2.0, 0.0]);
    let c = D2::constant;
    // x·∞ = ∞ with d/dx = ∞; the other direction is untouched (the 0·∞ there is a strong zero).
    assert_eq!((x * c(inf)).eps(), &[inf, 0.0]);
    assert_eq!((x * inf).eps(), &[inf, 0.0]);
    assert_eq!((x * -inf).eps(), &[-inf, 0.0]);
    assert_eq!((c(2.0) * c(inf)).eps(), &[0.0, 0.0]);
    // x/±0 = ±∞ with d/dx = 1/±0 = ±∞, through both `/ Self` and `/ f64`.
    for (z, s) in [(0.0, inf), (-0.0, -inf)] {
        let q = x / c(z);
        assert_eq!((q.re(), q.eps()), (s, &[s, 0.0]));
        let q = x / z;
        assert_eq!((q.re(), q.eps()), (s, &[s, 0.0]));
    }
    // x/y at y = 0 with y′ ≠ 0: the −x·y′/y² term dominates.
    let q = x / y0;
    assert_eq!((q.re(), q.eps()), (inf, &[inf, -inf]));
    // 1/y at y = −0: −∞, with d/dy = −1/y² = −∞.
    let q = c(1.0) / D2::new(-0.0, [0.0, 1.0]);
    assert_eq!((q.re(), q.eps()), (-inf, &[0.0, -inf]));
    // ∞/x: d/dx = −∞/x² = −∞; x/∞ = 0 with zero derivatives.
    let q = c(inf) / x;
    assert_eq!((q.re(), q.eps()), (inf, &[-inf, 0.0]));
    let q = x / c(inf);
    assert_eq!((q.re(), q.eps()), (0.0, &[0.0, 0.0]));
    // 0/0 is NaN, and so is the derivative in the direction of the divisor: never hidden.
    let [p, z] = D2::variables([0.0, 0.0]);
    let q = p / z;
    assert!(q.re().is_nan() && q.eps()[1].is_nan());
    // A NaN operand: NaN where an operand depends on the direction, zero elsewhere.
    let q = x * c(f64::NAN);
    assert!(q.re().is_nan() && q.eps()[0].is_nan() && q.eps()[1] == 0.0);
    let q = x / f64::NAN;
    assert!(q.re().is_nan() && q.eps()[0].is_nan() && q.eps()[1] == 0.0);
    // A product or quotient that overflows from finite operands keeps finite derivatives.
    let [m] = D1::variables([f64::MAX]);
    assert_eq!(((m * 2.0).re(), (m * 2.0).eps()), (inf, &[2.0]));
    assert_eq!((m * D1::constant(2.0)).eps(), &[2.0]);
    assert_eq!((m / D1::constant(0.5)).eps(), &[2.0]);
    assert_eq!((m * m).eps(), &[inf]);
    // An infinite or NaN derivative times a zero value is NaN, as in f64.
    assert!((D1::new(1.0, [inf]) * D1::constant(0.0)).eps()[0].is_nan());
}

#[test]
fn quotients_with_subnormal_and_huge_divisors() {
    // 1/b overflows for this subnormal b: the value must still be a/b (not a·(1/b) = ∞), and the
    // derivatives are divided directly.
    let (a, b) = (1e-320, 1e-309);
    let q = D2::new(a, [1.0, 0.0]) / D2::new(b, [0.0, 1.0]);
    assert_eq!(q.re(), a / b);
    assert_eq!(q.eps()[0], f64::INFINITY); // 1/b ≈ 1e309 overflows
    let want = -(Dd::from_f64(a) / Dd::from_f64(b) / Dd::from_f64(b)).hi(); // −a/b² ≈ −1e298
    // Two roundings (the quotient q and then q/b): within 2u relative.
    assert!(
        (q.eps()[1] - want).abs() <= 2.0 * U * want.abs(),
        "{q:?} vs {want:e}"
    );
    // 1/b is subnormal for |b| > 2^1022: the derivatives are again divided directly.
    let b = 1.5 * 2f64.powi(1022);
    let q = D1::new(3.0, [2f64.powi(100)]) / D1::constant(b);
    assert_eq!(q.eps()[0], 2f64.powi(100) / b);
}

#[test]
fn remainder_derivative_uses_the_exact_integer_quotient() {
    let inf = f64::INFINITY;
    // 1.0 / 0.1 rounds to 10, but 1.0 % 0.1 == 1 − 9·0.1 (0.09999999999999995).
    let [a, b] = D2::variables([1.0, 0.1]);
    let r = a % b;
    assert_eq!(r.re(), 0.09999999999999995);
    assert_eq!(r.eps(), &[1.0, -9.0]);
    // At an exact multiple the remainder is ±0 and t = a/b: the derivative of that branch.
    for (a, b, r, t) in [
        (7.5, 2.5, 0.0, 3.0),
        (-7.5, 2.5, -0.0, -3.0),
        (7.5, 2.0, 1.5, 3.0),
        (-7.5, 2.0, -1.5, -3.0),
        (7.5, -2.0, 1.5, -3.0),
        (-7.5, -2.0, -1.5, 3.0),
        (0.5, 2.0, 0.5, 0.0),
        (-0.0, 2.0, -0.0, 0.0),
        (2.0, inf, 2.0, 0.0),
        (2.0, -inf, 2.0, 0.0),
    ] {
        let [x, y] = D2::variables([a, b]);
        let rem = x % y;
        assert!(same(rem.re(), r), "{a} % {b}: {rem:?}");
        assert_eq!(rem.eps(), &[1.0, -t], "{a} % {b}");
    }
    // The quotient overflows (t ≈ 1e600) but the remainder is finite: d/da = 1, d/db = −∞, and
    // a direction no operand depends on stays 0.
    let r = D3::new(1e300, [1.0, 0.0, 0.0]) % D3::new(1e-300, [0.0, 1.0, 0.0]);
    assert_eq!(r.re(), 1e300 % 1e-300);
    assert_eq!(r.eps(), &[1.0, -inf, 0.0]);
    // Double-double: Dd(1) / Dd(0.01) rounds to 100, but 1 % 0.01 = 1 − 99·0.01; and 17·b
    // divided by b rounds below 17 although the remainder is 0.
    let p = |s: &str| Dd::from_decimal_str(s).unwrap();
    let b = Dd::from_f64(1.0) / Dd::from_f64(5.5);
    for (a, b, t) in [
        (p("1"), p("0.01"), 99.0),
        (p("3"), p("0.3"), 9.0),
        (p("1.1"), p("0.1"), 10.0),
        (b * 17.0, b, 17.0),
    ] {
        let [x, y] = Dd2::variables([a, b]);
        let r = x % y;
        assert!(same_dd(r.re(), a % b));
        assert_eq!(r.eps(), &[Dd::ONE, Dd::from_f64(-t)], "{a:?} % {b:?}");
    }
}

#[test]
fn variables_seed_unit_directions_for_any_n() {
    let none: [Dual<f64, 0>; 0] = Dual::variables([]);
    assert!(none.is_empty());
    // N = 0 is plain value arithmetic.
    let z = Dual::<f64, 0>::new(2.0, []);
    let w = (z * z + z / Dual::constant(4.0) - 1.0) % Dual::constant(2.0);
    assert_eq!((w.re(), w.eps()), ((4.0 + 0.5 - 1.0) % 2.0, &[]));
    let [x] = D1::variables([2.5]);
    assert_eq!((x.re(), x.eps()), (2.5, &[1.0]));
    let vars = Dual::<f64, 4>::variables([1.0, 2.0, 3.0, 4.0]);
    for (i, v) in vars.iter().enumerate() {
        assert_eq!(v.re(), (i + 1) as f64);
        for (j, &e) in v.eps().iter().enumerate() {
            let want: f64 = if i == j { 1.0 } else { 0.0 };
            assert_eq!(e.to_bits(), want.to_bits(), "variable {i}, direction {j}");
        }
    }
    let c = Dual::<f64, 3>::constant(-0.0);
    assert!(same(c.re(), -0.0));
    assert!(c.eps().iter().all(|e| e.to_bits() == 0));
}

#[test]
fn comparisons_and_zero_with_special_values() {
    let nan = D2::constant(f64::NAN);
    assert!(nan != nan);
    assert_eq!(nan.partial_cmp(&D2::zero()), None);
    assert_eq!(D2::new(-0.0, [1.0, 2.0]), D2::zero());
    assert!(D2::new(f64::NEG_INFINITY, [0.0; 2]) < D2::new(f64::MIN, [0.0; 2]));
    // Strict zero and one tests (controller decision after the Task 7 review): the value and
    // every derivative, each of either sign of zero.
    assert!(!D2::new(-0.0, [1.0, 2.0]).is_zero());
    assert!(!D2::new(0.0, [0.0, -5e-324]).is_zero());
    assert!(D2::new(-0.0, [0.0, -0.0]).is_zero());
    assert!(!D2::new(5e-324, [0.0, 0.0]).is_zero());
    assert!(!D2::new(1.0, [3.0, 0.0]).is_one());
    assert!(!D2::new(1.0, [0.0, f64::NAN]).is_one());
    assert!(D2::new(1.0, [-0.0, 0.0]).is_one());
    assert!(!D2::new(1.0 + f64::EPSILON, [0.0; 2]).is_one());
    assert!(D2::zero().is_zero() && D2::one().is_one());
    assert!(!D2::constant(f64::NAN).is_zero() && !D2::constant(f64::NAN).is_one());
    // Over double-double, and for a dual whose derivative parts are themselves zero tests.
    assert!(!Dual::<Dd, 1>::new(Dd::ZERO, [Dd::from_f64(1e-300)]).is_zero());
    assert!(Dual::<Dd, 1>::new(-Dd::ZERO, [Dd::ZERO]).is_zero());
}

#[test]
fn complex_zero_and_one_tests_see_derivatives() {
    // num-complex shortcuts `sqrt`, `cbrt`, `powf` and `powc` on these tests. With m = n + 0i
    // and ∂/∂(Im m) seeded, the imaginary part is zero in value but not as a dual, and must not
    // be treated as zero.
    let [n, k] = D2::variables([1.33, 0.0]);
    assert!(!Complex::new(D2::zero(), k).is_zero());
    assert!(!Complex::new(n, k).im.is_zero());
    assert!(Complex::new(D2::zero(), D2::zero()).is_zero());
    assert!(!Complex::new(D2::one(), k).is_one());
    assert!(Complex::new(D2::one(), D2::zero()).is_one());
    assert!(!Complex::new(D2::new(1.0, [1.0, 0.0]), D2::zero()).is_one());
    // Equality stays value-only, also inside a complex number.
    assert_eq!(
        Complex::new(D2::zero(), k),
        Complex::new(D2::zero(), D2::zero())
    );
}

#[test]
fn conversions_forward_to_the_value_type() {
    // Each conversion is `T`'s on the value, including those `ToPrimitive` would otherwise
    // derive from `to_i64` or `to_f64`.
    let x = Dual::<Dd, 1>::new(Dd::new(2f64.powi(100), 1.0), [Dd::ONE]);
    assert_eq!(x.to_i128(), Some((1 << 100) + 1));
    assert_eq!(x.to_u128(), Some((1 << 100) + 1));
    assert_eq!(x.to_i64(), None);
    // 1 + 2^-24 + 2^-80 rounds once to 1 + 2^-23; through to_f64 it would tie down to 1.
    let t = Dual::<Dd, 1>::new(Dd::new(1.0 + 2f64.powi(-24), 2f64.powi(-80)), [Dd::ONE]);
    assert_eq!(t.to_f32(), Some(1.0 + 2f32.powi(-23)));
    assert_eq!(D2::new(-3.7, [0.0; 2]).to_u8(), None);
    assert_eq!(D2::new(300.5, [0.0; 2]).to_u8(), None);
    assert_eq!(D2::new(255.9, [0.0; 2]).to_u8(), Some(255));
    assert_eq!(D2::new(f64::NAN, [0.0; 2]).to_i32(), None);
    let n = <Dual<Dd, 1> as NumCast>::from((1u128 << 100) + 1).unwrap();
    assert_eq!(
        (n.re(), n.eps()),
        (Dd::new(2f64.powi(100), 1.0), &[Dd::ZERO])
    );
    let from_dual = <D2 as NumCast>::from(D1::new(2.5, [7.0])).unwrap();
    assert_eq!((from_dual.re(), from_dual.eps()), (2.5, &[0.0, 0.0]));
    let s = <Dual<Dd, 1> as Num>::from_str_radix("0.1", 10).unwrap();
    assert_eq!(
        (s.re(), s.eps()),
        (Dd::from_decimal_str("0.1").unwrap(), &[Dd::ZERO])
    );
    assert!(<Dual<Dd, 1> as Num>::from_str_radix("0.1", 16).is_err());
    assert!(<D2 as Num>::from_str_radix("x", 10).is_err());
}

#[test]
fn double_double_values_are_bitwise_and_derivatives_keep_full_precision() {
    let d = Dd::from_f64;
    let (one, three) = (d(1.0), d(3.0));
    let [x, y] = Dd2::variables([one, three]);
    // Quotient: value exactly Dd's, d/dx = 1/3 and d/dy = −1/9 to double-double accuracy.
    let q = x / y;
    assert!(same_dd(q.re(), one / three));
    let (third, ninth) = (one / three, one / d(9.0));
    assert!((q.eps()[0] - third).abs().hi() <= 4.0 * U2 * third.hi());
    assert!((q.eps()[1] + ninth).abs().hi() <= 32.0 * U2 * ninth.hi());
    // Product, sum and mixed operations with f64.
    let f = (x * y + x - y * 0.5) / 4.0;
    let fv = (one * three + one - three * 0.5) / 4.0;
    assert!(same_dd(f.re(), fv));
    assert_eq!(f.eps(), &[(three + 1.0) / 4.0, (one - 0.5) / 4.0]);
    let mut g = x;
    g += y;
    g -= 0.25;
    g *= 3.0;
    g /= y;
    g %= Dd2::constant(d(0.7));
    let gv = (((one + three) - 0.25) * 3.0 / three) % d(0.7);
    assert!(same_dd(g.re(), gv));
    for (op, i) in f64_assign_ops::<Dd, 2>() {
        let mut z = x;
        op(&mut z, 2.5);
        let want = f64_ops::<Dd, 2>()[i].1(x, 2.5);
        assert!(same_dd(z.re(), want.re()) && z.eps() == want.eps());
    }
    // Values follow Dd bit for bit on special values too.
    for a in SPECIAL {
        for b in SPECIAL {
            let (u, v) = (
                Dd2::new(d(a), [Dd::ONE, Dd::ZERO]),
                Dd2::new(d(b), [Dd::ZERO, Dd::ONE]),
            );
            for (name, op, f) in binary_ops::<Dd, 2>() {
                assert!(same_dd(op(u, v).re(), f(d(a), d(b))), "{a:?} {name} {b:?}");
            }
            for (name, op, f) in f64_ops::<Dd, 2>() {
                assert!(same_dd(op(u, b).re(), f(d(a), b)), "{a:?} {name} {b:?}");
                assert_eq!(op(u, b).eps()[1], Dd::ZERO, "{a:?} {name} {b:?}");
            }
        }
    }
    // Strong zeros over Dd.
    assert_eq!(
        (x * Dd2::constant(Dd::INFINITY)).eps(),
        &[Dd::INFINITY, Dd::ZERO]
    );
    assert_eq!(
        (x / Dd2::constant(Dd::ZERO)).eps(),
        &[Dd::INFINITY, Dd::ZERO]
    );
}

#[test]
fn double_double_quotient_keeps_full_precision_for_huge_divisors() {
    // 1/b = (3/7)·2^-1000 has a subnormal trailing word, so (a′ − q·b′)·(1/b) would lose about
    // 30 bits; the derivative is divided directly instead.
    let b = Dd::from_f64(2f64.powi(1000)) * (Dd::from_f64(7.0) / Dd::from_f64(3.0));
    let da = Dd::from_f64(2f64.powi(100)) / Dd::from_f64(7.0);
    let q = Dual::<Dd, 1>::new(Dd::ONE, [da]) / Dual::constant(b);
    let want = da / b; // ≈ 2^-900: full precision
    assert!(
        (q.eps()[0] - want).abs().hi() <= 4.0 * U2 * want.abs().hi(),
        "{:?} vs {want:?}",
        q.eps()[0]
    );
}

// ---- Exact expansion arithmetic: a reference for double-double derivatives -----------------

/// Error-free sum: `s + e == a + b` exactly (Knuth's TwoSum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// An exact sum of doubles, kept as a nonoverlapping expansion (Shewchuk's Grow-Expansion).
#[derive(Default)]
struct Exact(Vec<f64>);

impl Exact {
    fn add(&mut self, b: f64) {
        let mut q = b;
        for e in self.0.iter_mut() {
            let (s, h) = two_sum(q, *e);
            *e = h;
            q = s;
        }
        self.0.push(q);
    }

    /// Adds the exact product of the factors (each split by TwoProd, so no rounding is lost;
    /// valid while no partial product underflows).
    fn add_product(&mut self, factors: &[f64]) {
        let mut terms = vec![factors[0]];
        for &f in &factors[1..] {
            terms = terms
                .iter()
                .flat_map(|&t| {
                    let p = t * f;
                    [p, t.mul_add(f, -p)]
                })
                .collect();
        }
        for t in terms {
            self.add(t);
        }
    }

    /// Adds `sign` times the exact product of double-doubles (every combination of words).
    fn add_dd_product(&mut self, sign: f64, factors: &[Dd]) {
        let words: Vec<[f64; 2]> = factors.iter().map(|x| [x.hi(), x.lo()]).collect();
        for mask in 0..1usize << words.len() {
            let mut f: Vec<f64> = (0..words.len())
                .map(|j| words[j][(mask >> j) & 1])
                .collect();
            f[0] *= sign;
            self.add_product(&f);
        }
    }

    /// The value, rounded (to within a few ulps).
    fn approx(&self) -> f64 {
        self.0.iter().sum()
    }
}

#[test]
fn double_double_product_and_quotient_rule_accuracy() {
    // Random double-doubles over 2^±200, derivatives over 2^±20, so no intermediate overflows
    // or underflows. The error is measured exactly: for the quotient, got·b² − (a′b − ab′) is
    // summed exactly and divided by b².
    let mut next = uniform(0x2545_f491_4f6c_dd1d);
    let mut dd = |lo: f64, hi: f64| {
        let s = if next() < 0.5 { -1.0 } else { 1.0 };
        let h = s * (1.0 + next()) * 2f64.powi((lo + (hi - lo) * next()).floor() as i32);
        Dd::new(h, h * (next() - 0.5) * 2f64.powi(-53))
    };
    let (mut worst_mul, mut worst_div) = (0.0_f64, 0.0_f64);
    for _ in 0..50_000 {
        let (a, b) = (dd(-200.0, 200.0), dd(-200.0, 200.0));
        let (da, db) = (dd(-20.0, 20.0), dd(-20.0, 20.0));
        let (x, y) = (Dual::<Dd, 1>::new(a, [da]), Dual::<Dd, 1>::new(b, [db]));

        // a′·b and a·b′ each err by 4u², their sum by 3u²: 7u²·(|a′b| + |ab′|) to first order
        // (the second-order terms are below 2^-100 of that).
        let got = (x * y).eps()[0];
        let mut e = Exact::default();
        e.add_dd_product(1.0, &[da, b]);
        e.add_dd_product(1.0, &[a, db]);
        e.add(-got.hi());
        e.add(-got.lo());
        let scale = (da.hi() * b.hi()).abs() + (a.hi() * db.hi()).abs();
        let err = e.approx().abs() / (U2 * scale);
        worst_mul = worst_mul.max(err);
        assert!(err <= 7.0 + 1e-9, "{x:?} * {y:?}: {err} u²");

        // q = a/b and 1/b err by 10u² each, q·b′ and the final product by 4u², a′ − q·b′ by
        // 3u²: 31u²·(|a′| + |q·b′|)/|b| to first order.
        let got = (x / y).eps()[0];
        let mut e = Exact::default();
        e.add_dd_product(1.0, &[got, b, b]);
        e.add_dd_product(-1.0, &[da, b]);
        e.add_dd_product(1.0, &[a, db]);
        let b2 = b.hi() * b.hi();
        let scale = (da.hi().abs() + (a.hi() / b.hi() * db.hi()).abs()) / b.hi().abs();
        let err = e.approx().abs() / b2 / (U2 * scale);
        worst_div = worst_div.max(err);
        assert!(err <= 31.0 + 1e-9, "{x:?} / {y:?}: {err} u²");
    }
    eprintln!("double-double product rule: worst {worst_mul:.3} u² (bound 7 u²)");
    eprintln!("double-double quotient rule: worst {worst_div:.3} u² (bound 31 u²)");
}

/// One expression that uses every operation, with special paths reachable from its inputs.
fn every_operation<T: scatrs::Real, const N: usize>(
    x: Dual<T, N>,
    y: Dual<T, N>,
    z: Dual<T, N>,
    c: f64,
) -> Dual<T, N> {
    let p = x * y - z / x;
    let q = (p % y + c) * (z - c) / (y * c);
    let mut r = q * p - x / (z + 1.0);
    r *= y;
    r /= x;
    r %= z;
    r += -p / c;
    r -= q * c;
    r * (x - y) + z
}

/// `every_operation` with the seeds `s` in direction `k` of an `N`-direction dual and the
/// `other` seeds elsewhere: the value and direction `k`'s derivative.
fn seeded<const N: usize>(
    v: [f64; 3],
    s: [f64; 3],
    other: [[f64; 8]; 3],
    k: usize,
    c: f64,
) -> (f64, f64) {
    let d = |i: usize| {
        let mut eps: [f64; N] = core::array::from_fn(|j| other[i][j]);
        eps[k] = s[i];
        Dual::<f64, N>::new(v[i], eps)
    };
    let r = every_operation(d(0), d(1), d(2), c);
    (r.re(), r.eps()[k])
}

/// A double from the whole range or a small set of specials, as an array of three.
fn three() -> impl Strategy<Value = [f64; 3]> {
    [any_f64(), any_f64(), any_f64()]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn each_direction_is_independent_of_n_and_of_the_other_seeds(
        v in three(),
        s in three(),
        other in [prop::array::uniform8(any_f64()), prop::array::uniform8(any_f64()), prop::array::uniform8(any_f64())],
        k in 0usize..8,
        c in any_f64(),
    ) {
        // Every rule is componentwise and its special-case choice depends only on the values,
        // so direction k's derivative is the same bit for bit (or NaN in both) for any N and any
        // seeds in the other directions.
        let (re1, d1) = seeded::<1>(v, s, other, 0, c);
        for (re, d) in [
            seeded::<2>(v, s, other, k % 2, c),
            seeded::<4>(v, s, other, k % 4, c),
            seeded::<8>(v, s, other, k, c),
        ] {
            prop_assert!(same(re, re1), "value {re:?} vs {re1:?}");
            prop_assert!(same(d, d1), "derivative {d:?} vs {d1:?}");
        }
    }

    #[test]
    fn remainder_quotient_beyond_2_51_is_within_its_bound(
        b in wide_f64(),
        k in (1u64 << 50)..(1u64 << 53),
        nudge in -4i64..=4,
        neg in any::<bool>(),
    ) {
        let a0 = (k as f64 * b).abs();
        let a = f64::from_bits((a0.to_bits() as i64 + nudge) as u64) * if neg { -1.0 } else { 1.0 };
        let r = D1::constant(a) % D1::new(b, [1.0]);
        let t = -r.eps()[0];
        prop_assert_eq!(t, t.trunc());
        // The exact quotient: a − r is exact in double-double and the quotient errs by about
        // 13u²·2^53 ≪ ½ there; a − t·b == r confirms it (t·b is exact for |t| < 2^54).
        let dd = Dd::from_f64;
        let exact = ((dd(a) - dd(r.re())) / dd(b)).round();
        prop_assert_eq!(exact.lo(), 0.0);
        let resid = (dd(a) - exact * dd(b) - dd(r.re())).abs().hi();
        prop_assert!(resid <= b.abs() * 2f64.powi(-60), "{a:?} % {b:?}: {exact:?}");
        // The computed (a − r)/b errs by at most (2u + u²)·|t|, and rounding adds at most ½.
        let tx = exact.hi();
        prop_assert!((t - tx).abs() <= (2.0 * U + U * U) * tx.abs() + 0.5, "{a:?} % {b:?}: {t} vs {tx}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn double_double_remainder_quotient_is_exact_far_beyond_f64(
        bh in wide_f64(),
        bl in -0.5..0.5f64,
        m in 1u64..(1 << 53),
        e in 0..=40i32,
        nudge in -4i64..=4,
    ) {
        // k up to 2^93, and a within a few u²·|a| of k·b.
        let b = Dd::new(bh, bl * bh.abs() * 2f64.powi(-53));
        let a0 = b * (m as f64 * 2f64.powi(e));
        let a = Dd::new(a0.hi(), a0.lo() + nudge as f64 * a0.hi().abs() * 2f64.powi(-106));
        let r = Dual::<Dd, 1>::constant(a) % Dual::new(b, [Dd::ONE]);
        let t = -r.eps()[0];
        prop_assert_eq!(t, t.trunc());
        // |a − t·b − r| <= 4u²|t·b| + 3u²|a| + 2u²|b| <= 2^-9·|b| for |t| < 2^94, while a wrong
        // t leaves |b|.
        let resid = (a - b * t - r.re()).abs().hi();
        prop_assert!(resid <= b.abs().hi() * 2f64.powi(-8), "{a:?} % {b:?}: t = {t:?}");
    }
}

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
fn product_and_quotient_rule_accuracy() {
    // Random values over 2^±300 and derivatives over 2^±20 (one in eight exactly zero), so no
    // intermediate overflows or underflows. References are evaluated in double-double, with
    // errors near u² against bounds near u.
    let mut next = uniform(0x9e37_79b9_7f4a_7c15);
    // `±m·2^e` with m in [1, 2) and e in [lo, hi), or zero once in `zero_one_in` draws (never if 0).
    let mut signed_pow2 = |lo: f64, hi: f64, zero_one_in: f64| {
        if zero_one_in > 0.0 && next() * zero_one_in < 1.0 {
            return 0.0;
        }
        let s = if next() < 0.5 { -1.0 } else { 1.0 };
        s * (1.0 + next()) * 2f64.powi((lo + (hi - lo) * next()).floor() as i32)
    };
    let (mut worst_mul, mut worst_div) = (0.0_f64, 0.0_f64);
    for _ in 0..200_000 {
        let (a, b) = (
            signed_pow2(-300.0, 300.0, 0.0),
            signed_pow2(-300.0, 300.0, 0.0),
        );
        let (da, db) = (signed_pow2(-20.0, 20.0, 8.0), signed_pow2(-20.0, 20.0, 8.0));
        let (x, y) = (D1::new(a, [da]), D1::new(b, [db]));
        let dd = Dd::from_f64;

        // fl(fl(a′b) + fl(ab′)) errs by at most (2u + u²)·(|a′b| + |ab′|).
        let p = x * y;
        assert_eq!(p.re(), a * b);
        let exact = dd(da) * dd(b) + dd(a) * dd(db);
        let scale = (da * b).abs() + (a * db).abs();
        if scale > 0.0 {
            let err = (dd(p.eps()[0]) - exact).abs().hi() / (U * scale);
            worst_mul = worst_mul.max(err);
            assert!(err <= 2.0 + 1e-10, "{x:?} * {y:?}: {err} u");
        }

        // (a′ − q̂·b′)·(1/b), q̂ = fl(a/b): five roundings, so the error is at most
        // (5u + O(u²))·(|a′| + |q·b′|)/|b|.
        let q = x / y;
        assert_eq!(q.re(), a / b);
        let qd = dd(a) / dd(b);
        let exact = (dd(da) - qd * dd(db)) / dd(b);
        let scale = (da.abs() + (qd.hi() * db).abs()) / b.abs();
        if scale > 0.0 {
            let err = (dd(q.eps()[0]) - exact).abs().hi() / (U * scale);
            worst_div = worst_div.max(err);
            assert!(err <= 5.0 + 1e-10, "{x:?} / {y:?}: {err} u");
        }
    }
    eprintln!("product rule: worst {worst_mul:.3} u (bound 2 u)");
    eprintln!("quotient rule: worst {worst_div:.3} u (bound 5 u)");
}

/// Finite doubles of any magnitude, including subnormals and both zeros.
fn any_finite() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("finite", |x| x.is_finite())
}

/// Doubles from the whole range, with the special values over-represented.
fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        any::<u64>().prop_map(f64::from_bits),
        prop::sample::select(SPECIAL.to_vec()),
    ]
}

/// A derivative component of magnitude at most 1, often exactly ±0: its products with finite
/// values cannot overflow, so a NaN could only come from 0·∞ or ∞ − ∞ that the rules invent.
fn small_derivative() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(-0.0),
        -1.0..=1.0f64,
        any_finite().prop_map(|x| x / f64::MAX),
    ]
}

/// `±m·2^e` with `m` in [1, 2) and |e| <= 500.
fn wide_f64() -> impl Strategy<Value = f64> {
    (1.0..2.0f64, -500..=500i32, any::<bool>())
        .prop_map(|(m, e, neg)| if neg { -m } else { m } * 2f64.powi(e))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn finite_operands_give_nan_derivatives_only_with_nan_values(
        a in prop_oneof![any_finite(), wide_f64()],
        b in prop_oneof![any_finite(), wide_f64(), Just(0.0), Just(-0.0)],
        da in small_derivative(),
        db in small_derivative(),
    ) {
        let (x, y) = (D2::new(a, [da, 0.0]), D2::new(b, [db, 0.0]));
        for (name, d, _) in binary_ops::<f64, 2>() {
            let r = d(x, y);
            prop_assert!(r.re().is_nan() || !r.eps()[0].is_nan(), "{x:?} {name} {y:?}: {r:?}");
            prop_assert_eq!(r.eps()[1], 0.0);
        }
        for (name, d, _) in f64_ops::<f64, 2>() {
            let r = d(x, b);
            prop_assert!(r.re().is_nan() || !r.eps()[0].is_nan(), "{x:?} {name} {b:?}: {r:?}");
            prop_assert_eq!(r.eps()[1], 0.0);
        }
    }

    #[test]
    fn values_follow_f64_and_unrelated_directions_stay_zero(
        a in any_f64(), b in any_f64(), da in any_f64(), db in any_f64(),
    ) {
        let (x, y) = (D3::new(a, [da, 0.0, 0.0]), D3::new(b, [0.0, db, 0.0]));
        for (name, d, f) in binary_ops::<f64, 3>() {
            let r = d(x, y);
            prop_assert!(same(r.re(), f(a, b)), "{a:?} {name} {b:?}: {r:?}");
            prop_assert_eq!(r.eps()[2], 0.0);
        }
        for (name, d, f) in f64_ops::<f64, 3>() {
            let r = d(x, b);
            prop_assert!(same(r.re(), f(a, b)), "{a:?} {name} {b:?}: {r:?}");
            prop_assert_eq!(&r.eps()[1..], &[0.0, 0.0]);
        }
    }

    #[test]
    fn remainder_quotient_is_the_exact_integer_quotient(
        b in wide_f64(),
        k in prop_oneof![1u64..64, 1u64..((1 << 51) - 4)],
        nudge in -4i64..=4,
        neg in any::<bool>(),
    ) {
        // a lies within a few ulps of k·b, where fl(a/b) often rounds onto the integer k; the
        // exact quotient is k or k ± 1, below 2^51, where the recovered t is exact.
        let a0 = (k as f64 * b).abs();
        let a = f64::from_bits((a0.to_bits() as i64 + nudge) as u64) * if neg { -1.0 } else { 1.0 };
        let r = D1::constant(a) % D1::new(b, [1.0]);
        let t = -r.eps()[0];
        prop_assert_eq!(t, t.trunc());
        // a − t·b, evaluated exactly (t·b is exact in double-double), must be the remainder; a
        // quotient off by one would leave |b| over.
        let dd = Dd::from_f64;
        let resid = (dd(a) - dd(t) * dd(b) - dd(r.re())).abs().hi();
        prop_assert!(resid <= b.abs() * 2f64.powi(-60), "{a:?} % {b:?}: t = {t}, r = {:?}", r.re());
    }

    #[test]
    fn double_double_remainder_quotient_is_the_exact_integer_quotient(
        bh in wide_f64(),
        bl in -0.5..0.5f64,
        k in prop_oneof![1u64..64, 1u64..(1 << 30)],
        nudge in -4i64..=4,
    ) {
        // a lies within a few u²·|a| of k·b, where the rounded quotient often lands on k.
        let b = Dd::new(bh, bl * bh.abs() * 2f64.powi(-53));
        let a0 = b * k as f64;
        let a = Dd::new(a0.hi(), a0.lo() + nudge as f64 * a0.hi().abs() * 2f64.powi(-106));
        let r = Dual::<Dd, 1>::constant(a) % Dual::new(b, [Dd::ONE]);
        let t = -r.eps()[0];
        prop_assert_eq!(t, t.trunc());
        prop_assert_eq!(t.lo(), 0.0);
        // |a − t·b − r| <= ~5u²·|a| <= 2^-72·|b| for |t| < 2^31; a wrong t leaves |b|.
        let resid = (a - b * t.hi() - r.re()).abs().hi();
        prop_assert!(resid <= b.abs().hi() * 2f64.powi(-60), "{a:?} % {b:?}: t = {t:?}");
    }
}
