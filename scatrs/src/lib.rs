//! Fast, linearized, rigorously validated single-scattering calculations.
//!
//! The design is described in `docs/specs/2026-10-12-scatrs-design.md` in the repository.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod scalar;
pub mod special;
#[cfg(test)]
mod test_data;

pub use error::{Error, Result};
pub use scalar::Real;
