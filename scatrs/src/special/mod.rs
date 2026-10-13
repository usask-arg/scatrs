//! Special functions and quadrature rules shared by all scattering methods.

mod gauss_legendre;
mod wigner;

pub use gauss_legendre::{GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre};
pub use wigner::{wigner_d, wigner_d_vec};
