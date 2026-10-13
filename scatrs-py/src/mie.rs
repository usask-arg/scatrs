//! Bindings for `scatrs::mie`.
//!
//! Every function takes `x` (float64) and `m` (complex128) arrays of one shape, usually views
//! that `np.broadcast_arrays` made (zero strides where broadcast), and reads them in place, in
//! C order, with the GIL released; so they must not be modified from another thread during the
//! call. The outputs have one row per element in that order; `scatrs.mie` reshapes them.
//! Arrays are borrowed with `try_readonly` (see `special.rs` for why).

use numpy::ndarray::{Array2, Array3, ArrayViewD};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyBufferError, PyMemoryError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use scatrs::mie::{AmplitudeBuffers, Mie, MieOptions, MieSolution};

use crate::errors::to_pyerr;

/// The most complex values one call returns, 2²⁷ (2 GiB). Larger requests raise
/// `InvalidInputError`, as in `special.rs`.
pub const MAX_MIE_VALUES: usize = 1 << 27;

fn borrow<'py, T: numpy::Element>(
    a: &'py Bound<'py, PyArrayDyn<T>>,
    name: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    if !a.is_aligned() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an aligned array"
        )));
    }
    a.try_readonly()
        .map_err(|e| PyBufferError::new_err(format!("{name} cannot be read: {e}")))
}

/// The options, and a check that x and m have the same shape.
fn prepare(
    x: &ArrayViewD<'_, f64>,
    m: &ArrayViewD<'_, Complex64>,
    n_max: Option<usize>,
    allow_unvalidated: bool,
) -> PyResult<MieOptions> {
    if x.shape() != m.shape() {
        return Err(PyTypeError::new_err(format!(
            "x and m must have one shape (broadcast them first), got {:?} and {:?}",
            x.shape(),
            m.shape()
        )));
    }
    let mut opts = MieOptions::new().with_allow_unvalidated(allow_unvalidated);
    if let Some(n) = n_max {
        opts = opts.with_n_max(n);
    }
    Ok(opts)
}

/// Solves element `i`, prefixing any error with its index.
fn solve<'m>(
    mie: &'m mut Mie<f64>,
    i: usize,
    x: f64,
    m: Complex64,
    opts: &MieOptions,
    derivatives: bool,
) -> scatrs::Result<MieSolution<'m, f64>> {
    let r = if derivatives {
        mie.compute_with_derivatives(x, m, opts)
    } else {
        mie.compute(x, m, opts)
    };
    r.map_err(|e| match e {
        scatrs::Error::InvalidInput(s) => scatrs::Error::InvalidInput(format!("element {i}: {s}")),
        scatrs::Error::OutsideValidatedDomain(s) => {
            scatrs::Error::OutsideValidatedDomain(format!("element {i}: {s}"))
        }
        other => other,
    })
}

fn too_many(py: Python<'_>, what: &str, count: Option<usize>) -> PyErr {
    let count = count.map_or_else(|| "more than usize::MAX".to_owned(), |c| c.to_string());
    to_pyerr(
        py,
        scatrs::Error::InvalidInput(format!(
            "{what} would hold {count} values, more than MAX_MIE_VALUES = {MAX_MIE_VALUES}"
        )),
    )
}

/// A zeroed vector of `len` values, or `MemoryError`.
fn zeros<T: Clone + Default>(len: usize, what: &str) -> PyResult<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|e| {
        PyMemoryError::new_err(format!("cannot allocate {len} values for {what}: {e}"))
    })?;
    v.resize(len, T::default());
    Ok(v)
}

fn shape_error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("output shape: {e}"))
}

/// The default number of terms for each x (C order); see `scatrs.mie.default_n_max`.
#[pyfunction]
pub fn mie_default_n_max<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let x = borrow(&x, "x")?;
    let n: Vec<u64> = x
        .as_array()
        .iter()
        .map(|&v| scatrs::mie::default_n_max(v) as u64)
        .collect();
    Ok(PyArray1::from_vec(py, n))
}

type EfficiencyArrays<'py> = (Bound<'py, PyArray2<f64>>, Option<Bound<'py, PyArray3<f64>>>);

/// Efficiencies (5, len) in the order Q_ext, Q_sca, Q_abs, Q_back, g, and with
/// `derivatives` their Jacobian (3, 5, len) with respect to x, Re m and Im m.
#[pyfunction]
pub fn mie_efficiencies<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
) -> PyResult<EfficiencyArrays<'py>> {
    let (x, m) = (borrow(&x, "x")?, borrow(&m, "m")?);
    let (xv, mv) = (x.as_array(), m.as_array());
    let opts = prepare(&xv, &mv, n_max, allow_unvalidated)?;
    let len = xv.len();
    let mut values = zeros::<f64>(5 * len, "the efficiencies")?;
    let mut jacobian = zeros::<f64>(if derivatives { 15 * len } else { 0 }, "the Jacobian")?;
    py.detach(|| -> scatrs::Result<()> {
        let mut mie = Mie::<f64>::new();
        for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
            let s = solve(&mut mie, i, xi, mi, &opts, derivatives)?;
            let q = s.efficiencies();
            for (k, v) in [q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g]
                .into_iter()
                .enumerate()
            {
                values[k * len + i] = v;
            }
            if let Some(j) = s.efficiencies_jacobian() {
                for (p, d) in [j.x, j.m_re, j.m_im].into_iter().enumerate() {
                    let row = [d.q_ext, d.q_sca, d.q_abs, d.q_back, d.g];
                    for (k, v) in row.into_iter().enumerate() {
                        jacobian[(p * 5 + k) * len + i] = v;
                    }
                }
            }
        }
        Ok(())
    })
    .map_err(|e| to_pyerr(py, e))?;
    let values = Array2::from_shape_vec((5, len), values).map_err(shape_error)?;
    let jacobian = if derivatives {
        let a = Array3::from_shape_vec((3, 5, len), jacobian).map_err(shape_error)?;
        Some(a.into_pyarray(py))
    } else {
        None
    };
    Ok((values.into_pyarray(py), jacobian))
}

type CoefficientArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
    Option<Bound<'py, PyArray3<Complex64>>>,
);

/// The number of terms of each element (len), aₙ and bₙ (len, N) zero-padded to the largest
/// N, and with `derivatives` ∂a/∂x, da/dm, ∂b/∂x, db/dm (4, len, N).
#[pyfunction]
pub fn mie_coefficients<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
) -> PyResult<CoefficientArrays<'py>> {
    let (x, m) = (borrow(&x, "x")?, borrow(&m, "m")?);
    let (xv, mv) = (x.as_array(), m.as_array());
    let opts = prepare(&xv, &mv, n_max, allow_unvalidated)?;
    let len = xv.len();
    // The core checks each n_max; this only sizes the output (wrong sizes fail there first).
    let terms: Vec<u64> = xv
        .iter()
        .map(|&v| n_max.unwrap_or_else(|| scatrs::mie::default_n_max(v)) as u64)
        .collect();
    let width = terms.iter().copied().max().unwrap_or(0) as usize;
    let arrays = if derivatives { 6 } else { 2 };
    let total = len
        .checked_mul(width)
        .and_then(|v| v.checked_mul(arrays))
        .filter(|&v| v <= MAX_MIE_VALUES);
    let Some(per_array) = total.map(|t| t / arrays) else {
        let count = len.checked_mul(width).and_then(|v| v.checked_mul(arrays));
        return Err(too_many(py, "the coefficients", count));
    };
    let mut a = zeros::<Complex64>(per_array, "the coefficients")?;
    let mut b = zeros::<Complex64>(per_array, "the coefficients")?;
    let mut d = zeros::<Complex64>(if derivatives { 4 * per_array } else { 0 }, "derivatives")?;
    py.detach(|| -> scatrs::Result<()> {
        let mut mie = Mie::<f64>::new();
        for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
            let s = solve(&mut mie, i, xi, mi, &opts, derivatives)?;
            let row = i * width..i * width + s.n_terms();
            a[row.clone()].copy_from_slice(s.a());
            b[row.clone()].copy_from_slice(s.b());
            if let Some(dc) = s.derivatives() {
                for (k, src) in [dc.da_dx, dc.da_dm, dc.db_dx, dc.db_dm]
                    .into_iter()
                    .enumerate()
                {
                    let off = k * per_array;
                    d[off + row.start..off + row.end].copy_from_slice(src);
                }
            }
        }
        Ok(())
    })
    .map_err(|e| to_pyerr(py, e))?;
    let a = Array2::from_shape_vec((len, width), a).map_err(shape_error)?;
    let b = Array2::from_shape_vec((len, width), b).map_err(shape_error)?;
    let d = if derivatives {
        let d = Array3::from_shape_vec((4, len, width), d).map_err(shape_error)?;
        Some(d.into_pyarray(py))
    } else {
        None
    };
    Ok((
        PyArray1::from_vec(py, terms),
        a.into_pyarray(py),
        b.into_pyarray(py),
        d,
    ))
}

type AmplitudeArrays<'py> = (
    Bound<'py, PyArray3<Complex64>>,
    Option<Bound<'py, PyArray3<Complex64>>>,
);

/// S₁ and S₂ (2, len, n_mu) at the cosines `mu`, and with `derivatives` ∂S₁/∂x, ∂S₂/∂x,
/// dS₁/dm, dS₂/dm (4, len, n_mu).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mie_amplitudes<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    mu: Bound<'py, PyArrayDyn<f64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
) -> PyResult<AmplitudeArrays<'py>> {
    let (x, m, mu) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&mu, "mu")?);
    let (xv, mv) = (x.as_array(), m.as_array());
    let opts = prepare(&xv, &mv, n_max, allow_unvalidated)?;
    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
    let (len, n_mu) = (xv.len(), mu.len());
    let arrays = if derivatives { 6 } else { 2 };
    let per_array = len
        .checked_mul(n_mu)
        .filter(|v| v.checked_mul(arrays).is_some_and(|t| t <= MAX_MIE_VALUES));
    let Some(per_array) = per_array else {
        let count = len.checked_mul(n_mu).and_then(|v| v.checked_mul(arrays));
        return Err(too_many(py, "the amplitudes", count));
    };
    let mut s = zeros::<Complex64>(2 * per_array, "the amplitudes")?;
    let mut d = zeros::<Complex64>(if derivatives { 4 * per_array } else { 0 }, "derivatives")?;
    py.detach(|| -> scatrs::Result<()> {
        let mut mie = Mie::<f64>::new();
        let (s1, s2) = s.split_at_mut(per_array);
        // Empty unless `derivatives`.
        let half = d.len() / 2;
        let (dx, dm) = d.split_at_mut(half);
        let (dx1, dx2) = dx.split_at_mut(half / 2);
        let (dm1, dm2) = dm.split_at_mut(half / 2);
        for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
            let sol = solve(&mut mie, i, xi, mi, &opts, derivatives)?;
            let row = i * n_mu..(i + 1) * n_mu;
            let values = AmplitudeBuffers {
                s1: &mut s1[row.clone()],
                s2: &mut s2[row.clone()],
            };
            if derivatives {
                sol.amplitudes_with_derivatives(
                    &mu,
                    values,
                    AmplitudeBuffers {
                        s1: &mut dx1[row.clone()],
                        s2: &mut dx2[row.clone()],
                    },
                    AmplitudeBuffers {
                        s1: &mut dm1[row.clone()],
                        s2: &mut dm2[row.clone()],
                    },
                )?;
            } else {
                sol.amplitudes(&mu, values)?;
            }
        }
        Ok(())
    })
    .map_err(|e| to_pyerr(py, e))?;
    let s = Array3::from_shape_vec((2, len, n_mu), s).map_err(shape_error)?;
    let d = if derivatives {
        let d = Array3::from_shape_vec((4, len, n_mu), d).map_err(shape_error)?;
        Some(d.into_pyarray(py))
    } else {
        None
    };
    Ok((s.into_pyarray(py), d))
}
