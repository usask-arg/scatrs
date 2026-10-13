//! Conversion of `scatrs::Error` into the exception classes defined in `scatrs.errors`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// The matching `scatrs.errors` exception for a core error.
///
/// The message is the error's `Display` text. `NotConverged` passes `achieved` and `requested`
/// (either may be +∞) on to `NotConvergedError`. Variants added to the non-exhaustive
/// `scatrs::Error` later map to `ScatrsError` until they get a class of their own. If
/// `scatrs.errors` cannot be imported, the result is a `RuntimeError` caused by the import error.
pub fn to_pyerr(py: Python<'_>, err: scatrs::Error) -> PyErr {
    let message = err.to_string();
    let build = || -> PyResult<PyErr> {
        let module = py.import("scatrs.errors")?;
        let instance = match &err {
            scatrs::Error::InvalidInput(_) => module
                .getattr("InvalidInputError")?
                .call1((message.as_str(),))?,
            scatrs::Error::NotConverged {
                achieved,
                requested,
            } => module.getattr("NotConvergedError")?.call1((
                message.as_str(),
                *achieved,
                *requested,
            ))?,
            scatrs::Error::OutsideValidatedDomain(_) => module
                .getattr("OutsideValidatedDomainError")?
                .call1((message.as_str(),))?,
            _ => module.getattr("ScatrsError")?.call1((message.as_str(),))?,
        };
        Ok(PyErr::from_value(instance))
    };
    build().unwrap_or_else(|import_error| {
        let e = PyRuntimeError::new_err(message);
        e.set_cause(py, Some(import_error));
        e
    })
}

/// Raises the exception that the core error `variant` maps to (testing hook for `to_pyerr`).
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
