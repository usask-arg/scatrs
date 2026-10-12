use num_complex::Complex;
use scatrs::Real;

fn horner<T: Real>(coeffs: &[f64], x: T) -> T {
    coeffs.iter().rev().fold(T::zero(), |acc, &c| acc * x + c)
}

fn affine_in_place<T: Real>(mut x: T) -> T {
    x += 1.0;
    x *= 2.0;
    x
}

fn abs2_of_inverse<T: Real>(z: Complex<T>) -> T {
    (Complex::new(T::one(), T::zero()) / z).norm_sqr()
}

#[test]
fn f64_supports_generic_kernels() {
    assert_eq!(horner(&[1.0, 2.0, 3.0], 2.0_f64), 17.0);
    assert_eq!(affine_in_place(3.0_f64), 8.0);
}

#[test]
fn complex_over_real_works() {
    let v = abs2_of_inverse(Complex::new(3.0_f64, 4.0));
    assert!((v - 1.0 / 25.0).abs() < 1e-17);
}

#[test]
fn from_f64_pair_rounds_once_for_f64() {
    assert_eq!(<f64 as Real>::from_f64_pair(1.0, 2f64.powi(-60)), 1.0);
    assert_eq!(<f64 as Real>::from_f64_pair(1.0, 0.5), 1.5);
}

#[test]
fn approx_f64_is_identity_for_f64() {
    assert_eq!(Real::approx_f64(0.1_f64), 0.1);
    assert_eq!(<f64 as Real>::from_f64(0.1), 0.1);
}
