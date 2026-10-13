//! Hooks for the Python tests (`tests/*.py`); not part of the scatrs API.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::to_pyerr;

/// For tests only: raises the exception that the core error `variant` maps to, to test
/// [`to_pyerr`]. Not part of the scatrs API.
///
/// `variant` is `"InvalidInput"`, `"NotConverged"` or `"OutsideValidatedDomain"`; `message` is
/// the payload of the string variants, and `achieved`/`requested` that of `NotConverged`.
#[pyfunction]
#[pyo3(
    name = "_raise_core_error",
    signature = (variant, message = "", achieved = 0.0, requested = 0.0)
)]
pub fn raise_core_error(
    py: Python<'_>,
    variant: &str,
    message: &str,
    achieved: f64,
    requested: f64,
) -> PyResult<()> {
    let err = match variant {
        "InvalidInput" => scatrs::Error::InvalidInput(message.into()),
        "NotConverged" => scatrs::Error::NotConverged {
            achieved,
            requested,
        },
        "OutsideValidatedDomain" => scatrs::Error::OutsideValidatedDomain(message.into()),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown scatrs::Error variant {variant:?}"
            )));
        }
    };
    Err(to_pyerr(py, err))
}

/// For tests only: calls `f()` while holding rust-numpy's writeable borrow of `x`, as another
/// rust-numpy extension could. Not part of the scatrs API.
#[pyfunction]
#[pyo3(name = "_with_write_borrow")]
pub fn with_write_borrow<'py>(
    x: Bound<'py, PyArray1<f64>>,
    f: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _borrow = x.try_readwrite()?;
    f.call0()
}
