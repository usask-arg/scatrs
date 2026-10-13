//! One generic kernel evaluated with each scalar type: the example of the documentation page
//! "Scalar types (Rust)". Run it with `cargo run --example scalar_types`.

use scatrs::Real;
use scatrs::scalar::{DoubleDouble, Dual};
use scatrs::special::{gauss_legendre, wigner_d_vec};

/// ∫₋₁¹ exp(a·x) dx by an n-point Gauss–Legendre rule, for any scalar type.
fn integral_of_exp<T: Real>(a: T, n: usize) -> scatrs::Result<T> {
    let rule = gauss_legendre::<T>(n)?;
    let mut sum = T::zero();
    for (&x, &w) in rule.nodes.iter().zip(&rule.weights) {
        sum += w * (a * x).exp();
    }
    Ok(sum)
}

fn main() -> scatrs::Result<()> {
    // Values, in f64 and in double-double. The exact integral is 2 sinh(a)/a.
    let value = integral_of_exp(0.5_f64, 20)?;
    let value_dd = integral_of_exp(DoubleDouble::from_f64(0.5), 20)?;
    println!("f64: {value}, double-double: {value_dd:?}");

    // A first derivative: seed a as the only variable (value 0.5, derivative 1).
    let [a] = Dual::<f64, 1>::variables([0.5]);
    let first = integral_of_exp(a, 20)?;
    println!("value {}, d/da {}", first.re(), first.eps()[0]);

    // A second derivative: nest two duals and seed both levels in the same direction.
    let a2 = Dual::<Dual<f64, 1>, 1>::new(
        Dual::new(0.5, [1.0]),   // value 0.5, inner derivative 1
        [Dual::new(1.0, [0.0])], // outer derivative 1, whose own derivative is 0
    );
    let second = integral_of_exp(a2, 20)?;
    println!(
        "value {}, d/da {} (= {}), d²/da² {}",
        second.re().re(),
        second.re().eps()[0],
        second.eps()[0].re(),
        second.eps()[0].eps()[0],
    );

    // Wigner d-functions with their derivative in x = cos θ.
    let [x] = Dual::<f64, 1>::variables([0.3]);
    let d = wigner_d_vec(2, -2, 100, x)?;
    println!(
        "d^50_(2,-2)(0.3) = {}, derivative {}",
        d[50].re(),
        d[50].eps()[0]
    );
    Ok(())
}
