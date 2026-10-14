//! Fast, linearized, rigorously validated single-scattering calculations for spherical
//! particles.
//!
//! scatrs computes the optical properties of homogeneous spheres by Lorenz–Mie theory, and
//! their averages over size distributions and external mixtures, with closed-form derivatives
//! and a declared, tested accuracy. The same crate is the core of the Python package `scatrs`.
//! This reference covers the Rust API; the scatrs documentation also has a user guide with
//! examples, the theory behind every method, and the validation of the results.
//!
//! | Module | Contents |
//! |---|---|
//! | [`mie`] | One sphere: Mie coefficients, efficiencies, amplitudes, phase matrix and Greek coefficients, with their derivatives. |
//! | [`ensemble`] | Averages over size distributions, external mixtures and batches of distributions, with error estimates and derivatives. |
//! | [`special`] | Gauss–Legendre quadrature and Wigner d-functions. |
//! | [`scalar`] | The [`Real`] trait the numerical kernels are generic over, and its double-double and dual-number types. |
//! | [`error`] | The [`Error`] that every fallible function returns. |
//!
//! # One sphere
//!
//! A [`Mie`](mie::Mie) workspace solves one sphere at a time; everything else is read from
//! the solution.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::mie::{Mie, MieOptions, Normalization, PhaseBuffers};
//!
//! let mut mie = Mie::<f64>::new();
//! let sphere = mie.compute(10.0, Complex::new(1.33, 1e-8), &MieOptions::new())?;
//!
//! // Efficiencies (cross sections divided by πr²) and the asymmetry parameter.
//! let q = sphere.efficiencies();
//! println!("Q_ext = {}, Q_sca = {}, g = {}", q.q_ext, q.q_sca, q.g);
//!
//! // The normalized phase matrix at cosines of the scattering angle.
//! let mu = [1.0, 0.5, 0.0, -0.5, -1.0];
//! let mut f = [[0.0; 5]; 4];
//! let [f11, f12, f33, f34] = &mut f;
//! sphere.phase_matrix(&mu, Normalization::Normalized, PhaseBuffers { f11, f12, f33, f34 })?;
//!
//! // Its Greek coefficients, every nonzero order: α₁⁰ = 1 and α₁¹ = 3g.
//! let greek = sphere.greek(None, Normalization::Normalized)?;
//! assert_eq!(greek.alpha1[0], 1.0);
//! assert!((greek.alpha1[1] - 3.0 * q.g).abs() < 1e-12);
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! [`compute_with_derivatives`](mie::Mie::compute_with_derivatives) also gives the derivatives
//! of every output with respect to the size parameter and the real and imaginary parts of the
//! refractive index, in closed form, for 1.2 to 2.2 times the cost of the values:
//!
//! ```
//! # use num_complex::Complex;
//! # use scatrs::mie::{Mie, MieOptions};
//! let mut mie = Mie::<f64>::new();
//! let sphere = mie.compute_with_derivatives(10.0, Complex::new(1.5, 0.01), &MieOptions::new())?;
//! let jacobian = sphere.efficiencies_jacobian().expect("computed with derivatives");
//! println!("dQ_ext/dx = {}, dQ_ext/d(Im m) = {}", jacobian.x.q_ext, jacobian.m_im.q_ext);
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! # A size distribution
//!
//! [`ensemble::compute`] averages over a size distribution at each wavelength, in parallel.
//! Radii and wavelengths share any length unit, and cross sections are per particle, in that
//! unit squared. Every output has an error estimate, and `converged` says whether the size
//! integration met its tolerance.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::ensemble::{
//!     EnsembleOptions, EnsembleParam, ParamKind, SizeDistribution, compute,
//! };
//!
//! let sulfate = SizeDistribution::LogNormal { median_radius: 0.08, width: 1.6 }; // µm
//! let wavelengths = [0.35, 0.55, 1.0]; // µm
//! let m = [Complex::new(1.43, 1e-8); 3]; // one refractive index per wavelength
//! let median_radius = EnsembleParam::new(0, None, ParamKind::Distribution("median_radius"));
//! let options = EnsembleOptions::new().with_derivatives(vec![median_radius]);
//! for r in compute(&sulfate, &wavelengths, &m, &options)? {
//!     assert!(r.converged);
//!     let dc_ext = r.derivatives[0].c_ext; // the derivatives follow the order of r.params
//!     println!(
//!         "C_ext = {} ± {} µm², ω = {}, dC_ext/dr_g = {dc_ext} µm",
//!         r.values.c_ext, r.errors.c_ext, r.values.ssa,
//!     );
//! }
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! # Conventions
//!
//! * **Refractive index.** m = n + ik, with k ≥ 0 for absorbing particles (Bohren & Huffman,
//!   time dependence e^(−iωt)), relative to the surrounding medium. A negative imaginary part
//!   is an error, never silently flipped. Codes that write m = n − ik, such as MIEV0 and
//!   sasktran2, need the complex conjugate, and report the complex conjugates of the
//!   coefficients and amplitudes.
//! * **Size.** [`mie`] takes the size parameter x = 2πr/λ and returns efficiencies;
//!   [`ensemble`] takes radii and wavelengths and returns cross sections.
//! * **Angles.** Functions of angle take μ = cos Θ, with Θ the scattering angle.
//! * **Phase matrix.** Normalized, ½∫F₁₁ dμ = 1; the degree of linear polarization is
//!   −F₁₂/F₁₁.
//! * **Greek coefficients.** The expansion of Mishchenko, Travis & Lacis (2002) in Wigner
//!   d-functions; normalized, α₁⁰ = 1 and α₁¹ = 3g.
//!
//! # Accuracy
//!
//! Every Mie output has a declared accuracy inside the validated domain,
//! 10⁻⁶ ≤ x ≤ 2·10⁴, 0 < Re m ≤ 10, 0 ≤ Im m ≤ 10, |m| ≥ 0.1 (see
//! [`VALIDATED_DOMAIN`](mie::VALIDATED_DOMAIN)), checked in the tests against independent
//! arbitrary-precision reference values. Outside it, calls return
//! [`Error::OutsideValidatedDomain`] unless
//! [`allow_unvalidated`](mie::MieOptions::with_allow_unvalidated) is set. The ensembles
//! estimate the error of their size integration instead.
//!
//! # Precision and derivatives through generic code
//!
//! The numerical kernels are generic over [`Real`]: besides `f64`, they run in double-double
//! precision ([`DoubleDouble`](scalar::DoubleDouble), about 32 significant digits) and with
//! forward-mode derivatives ([`Dual`](scalar::Dual)). The ensembles are `f64` only.
//!
//! ```
//! use num_complex::Complex;
//! use scatrs::mie::{Mie, MieOptions};
//! use scatrs::scalar::DoubleDouble;
//!
//! let dd = DoubleDouble::from_f64;
//! let mut mie = Mie::<DoubleDouble>::new();
//! let sphere = mie.compute(dd(10.0), Complex::new(dd(1.33), dd(1e-8)), &MieOptions::new())?;
//! println!("Q_ext = {:?}", sphere.efficiencies().q_ext);
//! # Ok::<(), scatrs::Error>(())
//! ```
//!
//! # Threads and determinism
//!
//! A [`Mie`](mie::Mie) workspace serves one thread; make one per thread. The ensembles run on
//! rayon's global thread pool, or on a pool the caller [`install`](rayon::ThreadPool::install)s,
//! and give bitwise identical results for any number of threads.
// The only unsafe code is the call of `matrixmultiply::dgemm` in `ensemble::matrix`.
#![deny(unsafe_code)]
#![warn(missing_docs)]

mod complex;
pub mod ensemble;
pub mod error;
pub mod mie;
pub mod scalar;
pub mod special;
#[cfg(test)]
mod test_data;

pub use error::{Error, Result};
pub use scalar::Real;
