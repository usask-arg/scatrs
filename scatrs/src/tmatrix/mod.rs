//! The T-matrix of spheroids by the extended boundary condition method (EBCM), and the optical
//! properties of randomly oriented spheroids. See the theory chapter of the documentation.

// The kernel has no public entry point until the API lands later in phase 4a.
#![allow(dead_code)]

pub(crate) mod functions;
pub(crate) mod kernel;
pub(crate) mod random;
pub(crate) mod regular;
pub(crate) mod solve;
