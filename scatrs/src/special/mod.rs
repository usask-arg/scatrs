//! Special functions and quadrature rules shared by all scattering methods.
//!
//! * [`gauss_legendre`](fn@gauss_legendre) (and [`gauss_legendre_into`], into caller-owned buffers): the
//!   n-point Gauss–Legendre rule on [−1, 1], exact for polynomials of degree up to 2n − 1.
//!   The Mie module uses it to compute the Greek coefficients exactly.
//! * [`wigner_d`] (and [`wigner_d_vec`]): the Wigner d-functions d^l_mn(θ) at x = cos θ for
//!   every degree l up to a maximum, in the convention of Mishchenko, Travis & Lacis (2002),
//!   the basis of the Greek coefficients. d^l_00 is the Legendre polynomial P_l.
//!
//! Both are generic over [`Real`](crate::Real), and both have a declared accuracy, tested
//! against arbitrary-precision reference values; see each function.
//!
//! ```
//! use scatrs::special::{gauss_legendre, wigner_d_vec};
//!
//! // ∫ P₂(x)² dx over [−1, 1] = 2/5, exactly by a 3-point rule.
//! let rule = gauss_legendre::<f64>(3)?;
//! let mut integral = 0.0;
//! for (&x, &w) in rule.nodes.iter().zip(&rule.weights) {
//!     let p = wigner_d_vec(0, 0, 2, x)?; // P₀, P₁, P₂ at x
//!     integral += w * p[2] * p[2];
//! }
//! assert!((integral - 0.4).abs() < 1e-15);
//! # Ok::<(), scatrs::Error>(())
//! ```

pub(crate) mod clebsch;
mod gauss_legendre;
mod gauss_legendre_asymptotic;
mod wigner;

pub use gauss_legendre::{
    GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre, gauss_legendre_into,
};
pub use wigner::{wigner_d, wigner_d_vec};
