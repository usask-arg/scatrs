//! Python extension module `scatrs._scatrs`.
//!
//! The functions here take and return plain numpy arrays and Python integers; argument
//! conversion and the user-facing documentation live in the `scatrs` Python package. Every
//! function releases the GIL while it computes.

use pyo3::prelude::*;

mod errors;
mod special;
mod testing;

#[pymodule]
fn _scatrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "MAX_GAUSS_LEGENDRE_POINTS",
        scatrs::special::MAX_GAUSS_LEGENDRE_POINTS,
    )?;
    m.add("MAX_WIGNER_D_VALUES", special::MAX_WIGNER_D_VALUES)?;
    m.add_function(wrap_pyfunction!(special::gauss_legendre, m)?)?;
    m.add_function(wrap_pyfunction!(special::wigner_d, m)?)?;
    m.add_function(wrap_pyfunction!(testing::raise_core_error, m)?)?;
    m.add_function(wrap_pyfunction!(testing::with_write_borrow, m)?)?;
    Ok(())
}
