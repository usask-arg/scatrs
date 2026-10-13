//! Bindings for `scatrs::special`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError};
use pyo3::prelude::*;

use crate::errors::to_pyerr;

/// The most values one `wigner_d` call returns: 2³¹, 16 GiB of `f64`.
///
/// The bound is on max(len(x), 1)·(l_max + 1), so `l_max` is bounded even for an empty `x`.
/// Larger requests raise `InvalidInputError` rather than attempt the allocation, since on
/// systems that overcommit memory an allocation can succeed and the process then be killed
/// when the values are written.
pub const MAX_WIGNER_D_VALUES: usize = 1 << 31;

/// Nodes and weights of a Gauss–Legendre rule.
type Rule<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Gauss–Legendre nodes and weights on [-1, 1]; see `scatrs.special.gauss_legendre`.
///
/// Raises `InvalidInputError` for n == 0 or n > MAX_GAUSS_LEGENDRE_POINTS (checked by the core
/// before it allocates), and `OverflowError` for n < 0 or n >= 2**64.
#[pyfunction]
pub fn gauss_legendre(py: Python<'_>, n: usize) -> PyResult<Rule<'_>> {
    let rule = py
        .detach(|| scatrs::special::gauss_legendre::<f64>(n))
        .map_err(|e| to_pyerr(py, e))?;
    Ok((
        PyArray1::from_vec(py, rule.nodes),
        PyArray1::from_vec(py, rule.weights),
    ))
}

/// Wigner d^l_{mn} for l = 0..=l_max at each x = cos θ, shape (len(x), l_max + 1); see
/// `scatrs.special.wigner_d`.
///
/// `x` must be an aligned, contiguous, native-endian 1-D float64 array (otherwise
/// `TypeError`); it is read in place. Raises `InvalidInputError` for an x outside [-1, 1] or
/// NaN, and for max(len(x), 1)·(l_max + 1) > MAX_WIGNER_D_VALUES; `MemoryError` if the output
/// cannot be allocated; `OverflowError` for m or n outside the 32-bit range, or l_max < 0 or
/// l_max >= 2**64.
#[pyfunction]
pub fn wigner_d<'py>(
    py: Python<'py>,
    m: i32,
    n: i32,
    l_max: usize,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xs = x.as_slice()?;
    let too_many = || {
        to_pyerr(
            py,
            scatrs::Error::InvalidInput(format!(
                "wigner_d needs max(x.size, 1) * (l_max + 1) <= {MAX_WIGNER_D_VALUES}, \
                 got x.size = {}, l_max = {l_max}",
                xs.len()
            )),
        )
    };
    let width = l_max
        .checked_add(1)
        .filter(|&w| w <= MAX_WIGNER_D_VALUES)
        .ok_or_else(too_many)?;
    let len = xs
        .len()
        .checked_mul(width)
        .filter(|&len| len <= MAX_WIGNER_D_VALUES)
        .ok_or_else(too_many)?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|e| {
        PyMemoryError::new_err(format!(
            "wigner_d cannot allocate {len} float64 values for its output: {e}"
        ))
    })?;
    let values = py
        .detach(move || -> scatrs::Result<Vec<f64>> {
            // Within the reserved capacity, so this does not reallocate.
            values.resize(len, 0.0);
            for (&xi, row) in xs.iter().zip(values.chunks_exact_mut(width)) {
                scatrs::special::wigner_d(m, n, xi, row)?;
            }
            Ok(values)
        })
        .map_err(|e| to_pyerr(py, e))?;
    let values = Array2::from_shape_vec((xs.len(), width), values)
        .map_err(|e| PyRuntimeError::new_err(format!("wigner_d output shape: {e}")))?;
    Ok(values.into_pyarray(py))
}
