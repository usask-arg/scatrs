//! Bindings for `scatrs::mie`.
//!
//! Every function takes `x` (float64) and `m` (complex128) arrays of one shape: fresh views,
//! usually from `np.broadcast_arrays` (zero strides where broadcast), which it reads in place, in
//! C order, with the GIL released; so they must not be modified from another thread during the
//! call. The outputs have one row per element in that order; `scatrs.mie` reshapes them.
//! Arrays are borrowed with `try_readonly` (see `special.rs` for why).
//!
//! Broadcasting makes outputs far larger than inputs, so every output is bounded by
//! `MAX_MIE_VALUES` before anything is allocated, and then allocated and filled without the GIL.

use numpy::ndarray::{Array2, Array3, ArrayViewD};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyBufferError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use scatrs::mie::{
    AmplitudeBuffers, MAX_MIE_ORDER, MAX_SIZE_PARAMETER, MIN_SIZE_PARAMETER, Mie, MieOptions,
    MieSolution, n_terms,
};

use crate::errors::to_pyerr;

/// The most complex values one call returns, counting derivatives, 2²⁷ (2 GiB); the
/// efficiencies, which are real, may have twice as many. Larger requests raise
/// `InvalidInputError`, as in `special.rs`.
pub const MAX_MIE_VALUES: usize = 1 << 27;

/// An error raised without the GIL: a core error, converted to its `scatrs.errors` class once
/// the GIL is held again, or a Python exception that needs no GIL to create.
enum Failure {
    Core(scatrs::Error),
    Py(PyErr),
}

impl From<scatrs::Error> for Failure {
    fn from(e: scatrs::Error) -> Self {
        Self::Core(e)
    }
}

impl From<PyErr> for Failure {
    fn from(e: PyErr) -> Self {
        Self::Py(e)
    }
}

impl Failure {
    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            Self::Core(e) => to_pyerr(py, e),
            Self::Py(e) => e,
        }
    }
}

fn invalid(message: String) -> scatrs::Error {
    scatrs::Error::InvalidInput(message)
}

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

/// The options, after checking that x and m have one shape and that `n_max` is in range (here
/// too, so that an empty input does not hide a bad `n_max`).
fn prepare(
    py: Python<'_>,
    x: &ArrayViewD<'_, f64>,
    m: &ArrayViewD<'_, Complex64>,
    n_max: Option<usize>,
    allow_unvalidated: bool,
) -> PyResult<MieOptions> {
    if x.shape() != m.shape() {
        return Err(PyValueError::new_err(format!(
            "x and m must have one shape (broadcast them first), got {:?} and {:?}",
            x.shape(),
            m.shape()
        )));
    }
    let mut opts = MieOptions::new().with_allow_unvalidated(allow_unvalidated);
    if let Some(n) = n_max {
        if n == 0 || n > MAX_MIE_ORDER {
            return Err(to_pyerr(
                py,
                invalid(format!(
                    "the number of terms must be in [1, {MAX_MIE_ORDER}], got {n}"
                )),
            ));
        }
        opts = opts.with_n_max(n);
    }
    Ok(opts)
}

/// Prefixes an input error with the index of the failing element.
fn at(i: usize) -> impl FnOnce(scatrs::Error) -> scatrs::Error {
    move |e| match e {
        scatrs::Error::InvalidInput(s) => invalid(format!("element {i}: {s}")),
        scatrs::Error::OutsideValidatedDomain(s) => {
            scatrs::Error::OutsideValidatedDomain(format!("element {i}: {s}"))
        }
        other => other,
    }
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
    r.map_err(at(i))
}

/// `count` (`None` if it overflowed) values of `kind`, unless that exceeds `limit` (named
/// `limit_name` in the message).
fn bounded(
    what: &str,
    count: Option<usize>,
    kind: &str,
    derivatives: bool,
    limit: usize,
    limit_name: &str,
) -> scatrs::Result<usize> {
    match count {
        Some(c) if c <= limit => Ok(c),
        _ => {
            let count = count.map_or_else(|| "more than usize::MAX".to_owned(), |c| c.to_string());
            let counting = if derivatives {
                " (counting derivatives)"
            } else {
                ""
            };
            Err(invalid(format!(
                "{what} would hold {count} {kind}{counting}, more than {limit_name} = {limit}"
            )))
        }
    }
}

/// `count` complex values, at most `MAX_MIE_VALUES`.
fn complex_values(what: &str, count: Option<usize>, derivatives: bool) -> scatrs::Result<usize> {
    bounded(
        what,
        count,
        "complex values",
        derivatives,
        MAX_MIE_VALUES,
        "MAX_MIE_VALUES",
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

/// The default number of terms for each x (C order); see `scatrs.mie.default_n_max`. Raises
/// `InvalidInputError` for an x outside the accepted range.
#[pyfunction]
pub fn mie_default_n_max<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let x = borrow(&x, "x")?;
    let n = x
        .as_array()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if (MIN_SIZE_PARAMETER..=MAX_SIZE_PARAMETER).contains(&v) {
                Ok(scatrs::mie::default_n_max(v) as u64)
            } else {
                Err(invalid(format!(
                    "element {i}: the size parameter x must be in \
                     [{MIN_SIZE_PARAMETER:e}, {MAX_SIZE_PARAMETER:e}], got {v:e}"
                )))
            }
        })
        .collect::<scatrs::Result<Vec<u64>>>()
        .map_err(|e| to_pyerr(py, e))?;
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
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let len = xv.len();
    // Real values: twice as many fit in the bytes of MAX_MIE_VALUES complex values.
    let count = len.checked_mul(if derivatives { 20 } else { 5 });
    bounded(
        "the efficiencies",
        count,
        "real values",
        derivatives,
        2 * MAX_MIE_VALUES,
        "2 * MAX_MIE_VALUES",
    )
    .map_err(|e| to_pyerr(py, e))?;
    let (values, jacobian) = py
        .detach(|| -> Result<_, Failure> {
            let mut values = zeros::<f64>(5 * len, "the efficiencies")?;
            let mut jacobian =
                zeros::<f64>(if derivatives { 15 * len } else { 0 }, "the Jacobian")?;
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
            Ok((values, jacobian))
        })
        .map_err(|f| f.into_pyerr(py))?;
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
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let len = xv.len();
    let arrays = if derivatives { 6 } else { 2 };
    // Every sphere has at least one term, so this bounds the sizing pass below.
    complex_values("the coefficients", len.checked_mul(arrays), derivatives)
        .map_err(|e| to_pyerr(py, e))?;
    let (terms, a, b, d, width) = py
        .detach(|| -> Result<_, Failure> {
            // Check every element and find the largest number of terms, before allocating.
            let mut width = 0;
            for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
                width = width.max(n_terms(xi, mi, &opts).map_err(at(i))?);
            }
            let count = len.checked_mul(width).and_then(|v| v.checked_mul(arrays));
            let per_array = complex_values("the coefficients", count, derivatives)? / arrays;
            let mut terms = zeros::<u64>(len, "the numbers of terms")?;
            let mut a = zeros::<Complex64>(per_array, "the coefficients")?;
            let mut b = zeros::<Complex64>(per_array, "the coefficients")?;
            let mut d =
                zeros::<Complex64>(if derivatives { 4 * per_array } else { 0 }, "derivatives")?;
            let mut mie = Mie::<f64>::new();
            for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
                let s = solve(&mut mie, i, xi, mi, &opts, derivatives)?;
                // `n_terms` above made the same checks and choice, unless x or m changed since.
                if s.n_terms() > width {
                    return Err(
                        invalid(format!("element {i}: x or m changed during the call")).into(),
                    );
                }
                terms[i] = s.n_terms() as u64;
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
            Ok((terms, a, b, d, width))
        })
        .map_err(|f| f.into_pyerr(py))?;
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
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
    // The core checks the cosines too; checking here also covers an empty x.
    if let Some((j, v)) = mu
        .iter()
        .enumerate()
        .find(|&(_, v)| !(-1.0..=1.0).contains(v))
    {
        return Err(to_pyerr(
            py,
            invalid(format!(
                "element {j} of mu: a cosine of the scattering angle must be in [-1, 1], got {v}"
            )),
        ));
    }
    let (len, n_mu) = (xv.len(), mu.len());
    let arrays = if derivatives { 6 } else { 2 };
    // At least one angle, so that an empty mu does not leave the number of spheres (each still
    // checked) unbounded.
    let count = len
        .checked_mul(n_mu.max(1))
        .and_then(|v| v.checked_mul(arrays));
    complex_values("the amplitudes", count, derivatives).map_err(|e| to_pyerr(py, e))?;
    let per_array = len * n_mu;
    let (s, d) = py
        .detach(|| -> Result<_, Failure> {
            let mut s = zeros::<Complex64>(2 * per_array, "the amplitudes")?;
            let mut d =
                zeros::<Complex64>(if derivatives { 4 * per_array } else { 0 }, "derivatives")?;
            let mut mie = Mie::<f64>::new();
            let (s1, s2) = s.split_at_mut(per_array);
            // Empty unless `derivatives`.
            let half = d.len() / 2;
            let (dx, dm) = d.split_at_mut(half);
            let (dx1, dx2) = dx.split_at_mut(half / 2);
            let (dm1, dm2) = dm.split_at_mut(half / 2);
            for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
                if n_mu == 0 {
                    // Nothing to compute, but the inputs are still checked.
                    n_terms(xi, mi, &opts).map_err(at(i))?;
                    continue;
                }
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
            Ok((s, d))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let s = Array3::from_shape_vec((2, len, n_mu), s).map_err(shape_error)?;
    let d = if derivatives {
        let d = Array3::from_shape_vec((4, len, n_mu), d).map_err(shape_error)?;
        Some(d.into_pyarray(py))
    } else {
        None
    };
    Ok((s.into_pyarray(py), d))
}
