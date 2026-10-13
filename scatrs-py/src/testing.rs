//! Hooks for the Python tests (`tests/*.py`), in the submodule `scatrs._scatrs._testing`; not
//! part of the scatrs API.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::to_pyerr;

/// Name under which [`register`] puts the submodule in `sys.modules`.
const NAME: &str = "scatrs._scatrs._testing";

/// Adds the `_testing` submodule to `parent` (the extension module).
///
/// It is set as an attribute rather than with `add_submodule`, which would also list it in the
/// parent's `__all__`; and it is put in `sys.modules`, so that `import scatrs._scatrs._testing`
/// works too (an extension's submodules are otherwise reachable only as attributes).
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, NAME)?;
    module.add_function(wrap_pyfunction!(raise_core_error, &module)?)?;
    module.add_function(wrap_pyfunction!(with_write_borrow, &module)?)?;
    parent.setattr("_testing", &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(NAME, &module)?;
    Ok(())
}

/// For tests only: raises the exception that the core error `variant` maps to, to test
/// [`to_pyerr`]. Not part of the scatrs API.
///
/// `variant` is `"InvalidInput"`, `"NotConverged"` or `"OutsideValidatedDomain"`; `message` is
/// the payload of the string variants, and `achieved`/`requested` that of `NotConverged`.
#[pyfunction]
#[pyo3(signature = (variant, message = "", achieved = 0.0, requested = 0.0))]
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
///
/// Changing the dtype, shape or strides of `x`, or resizing it, inside `f` can abort the
/// interpreter: rust-numpy then fails to find the borrow it releases (an unwrap in
/// `release_mut`). It is a test-only hook.
#[pyfunction]
pub fn with_write_borrow<'py>(
    x: Bound<'py, PyArray1<f64>>,
    f: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _borrow = x.try_readwrite()?;
    f.call0()
}
