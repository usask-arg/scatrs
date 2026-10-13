//! Conversion of `scatrs::Error` into the exception classes defined in `scatrs.errors`.

use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;

/// The matching `scatrs.errors` exception for a core error.
///
/// The message is the error's `Display` text. `NotConverged` passes `achieved` and `requested`
/// (either may be +∞) on to `NotConvergedError`. Variants added to the non-exhaustive
/// `scatrs::Error` later map to `ScatrsError` until they get a class of their own. If building
/// the exception fails with an `Exception` (say `scatrs.errors` cannot be imported), the result
/// is a `RuntimeError` with the core message, caused by that exception; any other failure
/// (`KeyboardInterrupt`, `SystemExit`, …) is returned unchanged.
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
    build().unwrap_or_else(|failure| {
        if !failure.is_instance_of::<PyException>(py) {
            return failure;
        }
        let e = PyRuntimeError::new_err(message);
        e.set_cause(py, Some(failure));
        e
    })
}
