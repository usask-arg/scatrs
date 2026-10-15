//! Bindings for `scatrs::layered`.
//!
//! Every function takes `x` (float64) and `m` (complex128) arrays of one shape (P, L): one
//! layered sphere per row, its interfaces' size parameters and its layers' indices, core first,
//! C-contiguous (`scatrs.layered` makes them so). The outputs have one row per particle, as in
//! `mie.rs`, whose helpers and bounds (`MAX_MIE_VALUES`) these share; derivatives come per
//! layer, first. The GIL is released while computing.
#![allow(clippy::needless_range_loop)] // particles are rows of x and m, indexed with the outputs

use numpy::ndarray::{ArrayD, ArrayView2, Ix2, IxDyn};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scatrs::layered::{Layered, LayeredSolution, n_terms};
use scatrs::mie::{
    AmplitudeBuffers, AngularTable, GreekCoefficients, MAX_GREEK_ORDER, MAX_MIE_ORDER, MieOptions,
};

use crate::errors::to_pyerr;
use crate::mie::{
    Failure, MAX_TABLE_VALUES, at, borrow, complex_values, cosines, greek_arrays, invalid,
    normalization, phase_buffers, real_values, shape_error, zeros,
};

/// The inputs as (P, L) views and the options, after checking their shapes and `n_max`.
fn prepare<'a>(
    py: Python<'_>,
    x: &'a ArrayView2<'a, f64>,
    m: &'a ArrayView2<'a, Complex64>,
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
    if !(x.is_standard_layout() && m.is_standard_layout()) {
        return Err(PyValueError::new_err("x and m must be C-contiguous"));
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

/// Particle `i`: its size parameters and indices.
fn row<'a>(
    x: &'a ArrayView2<'_, f64>,
    m: &'a ArrayView2<'_, Complex64>,
    i: usize,
) -> (&'a [f64], &'a [Complex64]) {
    (
        x.row(i).to_slice().expect("C-contiguous"),
        m.row(i).to_slice().expect("C-contiguous"),
    )
}

/// Checks every particle and returns the largest number of terms, before anything is allocated.
fn largest_n_terms(
    x: &ArrayView2<'_, f64>,
    m: &ArrayView2<'_, Complex64>,
    opts: &MieOptions,
) -> scatrs::Result<usize> {
    let mut most = 0;
    for i in 0..x.nrows() {
        let (xi, mi) = row(x, m, i);
        most = most.max(n_terms(xi, mi, opts).map_err(at(i))?);
    }
    Ok(most)
}

fn solve<'w>(
    lay: &'w mut Layered<f64>,
    i: usize,
    x: &[f64],
    m: &[Complex64],
    opts: &MieOptions,
    derivatives: bool,
) -> scatrs::Result<LayeredSolution<'w, f64>> {
    if derivatives {
        lay.compute_with_derivatives(x, m, opts)
    } else {
        lay.compute(x, m, opts)
    }
    .map_err(at(i))
}

fn views<'a>(
    x: &'a numpy::PyReadonlyArrayDyn<'_, f64>,
    m: &'a numpy::PyReadonlyArrayDyn<'_, Complex64>,
) -> PyResult<(ArrayView2<'a, f64>, ArrayView2<'a, Complex64>)> {
    let two = |e| PyValueError::new_err(format!("x and m must be two-dimensional: {e}"));
    Ok((
        x.as_array().into_dimensionality::<Ix2>().map_err(two)?,
        m.as_array().into_dimensionality::<Ix2>().map_err(two)?,
    ))
}

fn array<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    v: Vec<T>,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), v)
        .map_err(shape_error)?
        .into_pyarray(py))
}

type Arrays<'py, T> = (Bound<'py, PyArrayDyn<T>>, Option<Bound<'py, PyArrayDyn<T>>>);

/// Efficiencies (5, P) in the order Q_ext, Q_sca, Q_abs, Q_back, g, and with `derivatives`
/// their Jacobian (L, 3, 5, P) with respect to each layer's x, Re m and Im m.
#[pyfunction]
pub fn layered_efficiencies<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
) -> PyResult<Arrays<'py, f64>> {
    let (x, m) = (borrow(&x, "x")?, borrow(&m, "m")?);
    let (xv, mv) = views(&x, &m)?;
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let (len, layers) = xv.dim();
    let per = if derivatives { 5 + 15 * layers } else { 5 };
    real_values("the efficiencies", len.checked_mul(per), derivatives)
        .map_err(|e| to_pyerr(py, e))?;
    let (values, jacobian) = py
        .detach(|| -> Result<_, Failure> {
            let mut values = zeros::<f64>(5 * len, "the efficiencies")?;
            let mut jac = zeros::<f64>(
                if derivatives { 15 * layers * len } else { 0 },
                "the Jacobian",
            )?;
            let mut lay = Layered::<f64>::new();
            for i in 0..len {
                let (xi, mi) = row(&xv, &mv, i);
                let s = solve(&mut lay, i, xi, mi, &opts, derivatives)?;
                let q = s.efficiencies();
                for (k, v) in [q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g]
                    .into_iter()
                    .enumerate()
                {
                    values[k * len + i] = v;
                }
                for (l, j) in s
                    .efficiencies_jacobian()
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                {
                    for (p, d) in [j.x, j.m_re, j.m_im].into_iter().enumerate() {
                        for (k, v) in [d.q_ext, d.q_sca, d.q_abs, d.q_back, d.g]
                            .into_iter()
                            .enumerate()
                        {
                            jac[((l * 3 + p) * 5 + k) * len + i] = v;
                        }
                    }
                }
            }
            Ok((values, jac))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let jacobian = if derivatives {
        Some(array(py, &[layers, 3, 5, len], jacobian)?)
    } else {
        None
    };
    Ok((array(py, &[5, len], values)?, jacobian))
}

type CoefficientArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArrayDyn<Complex64>>,
    Bound<'py, PyArrayDyn<Complex64>>,
    Option<Bound<'py, PyArrayDyn<Complex64>>>,
);

/// The number of terms of each particle (P), aₙ and bₙ (P, N) zero-padded to the largest N, and
/// with `derivatives` ∂a/∂x_l, da/dm_l, ∂b/∂x_l, db/dm_l per layer (L, 4, P, N).
#[pyfunction]
pub fn layered_coefficients<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
) -> PyResult<CoefficientArrays<'py>> {
    let (x, m) = (borrow(&x, "x")?, borrow(&m, "m")?);
    let (xv, mv) = views(&x, &m)?;
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let (len, layers) = xv.dim();
    let arrays = if derivatives { 2 + 4 * layers } else { 2 };
    complex_values("the coefficients", len.checked_mul(arrays), derivatives)
        .map_err(|e| to_pyerr(py, e))?;
    let (terms, a, b, d, width) = py
        .detach(|| -> Result<_, Failure> {
            let width = largest_n_terms(&xv, &mv, &opts)?;
            let count = len.checked_mul(width).and_then(|v| v.checked_mul(arrays));
            let per_array = complex_values("the coefficients", count, derivatives)? / arrays;
            let mut terms = zeros::<u64>(len, "the numbers of terms")?;
            let mut a = zeros::<Complex64>(per_array, "the coefficients")?;
            let mut b = zeros::<Complex64>(per_array, "the coefficients")?;
            let mut d = zeros::<Complex64>(
                if derivatives {
                    4 * layers * per_array
                } else {
                    0
                },
                "derivatives",
            )?;
            let mut lay = Layered::<f64>::new();
            for i in 0..len {
                let (xi, mi) = row(&xv, &mv, i);
                let s = solve(&mut lay, i, xi, mi, &opts, derivatives)?;
                if s.n_terms() > width {
                    return Err(
                        invalid(format!("element {i}: x or m changed during the call")).into(),
                    );
                }
                terms[i] = s.n_terms() as u64;
                let r = i * width..i * width + s.n_terms();
                a[r.clone()].copy_from_slice(s.a());
                b[r.clone()].copy_from_slice(s.b());
                for l in 0..if derivatives { layers } else { 0 } {
                    let dc = s.derivatives(l).expect("computed");
                    for (k, src) in [dc.da_dx, dc.da_dm, dc.db_dx, dc.db_dm]
                        .into_iter()
                        .enumerate()
                    {
                        let off = (4 * l + k) * per_array;
                        d[off + r.start..off + r.end].copy_from_slice(src);
                    }
                }
            }
            Ok((terms, a, b, d, width))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let d = if derivatives {
        Some(array(py, &[layers, 4, len, width], d)?)
    } else {
        None
    };
    Ok((
        PyArray1::from_vec(py, terms),
        array(py, &[len, width], a)?,
        array(py, &[len, width], b)?,
        d,
    ))
}

/// An angular table at the cosines `mu` for every particle, if there are several to share it
/// and it holds at most `MAX_TABLE_VALUES` (as `mie.rs` builds one).
fn angular_table(
    x: &ArrayView2<'_, f64>,
    m: &ArrayView2<'_, Complex64>,
    opts: &MieOptions,
    mu: &[f64],
) -> scatrs::Result<Option<AngularTable<f64>>> {
    if x.nrows() < 2 || mu.is_empty() {
        return Ok(None);
    }
    let most = largest_n_terms(x, m, opts)?;
    if most
        .checked_mul(2 * mu.len())
        .is_none_or(|v| v > MAX_TABLE_VALUES)
    {
        return Ok(None);
    }
    AngularTable::new(mu, most).map(Some)
}

/// S₁ and S₂ (2, P, n_mu) at the cosines `mu`, and with `derivatives` ∂S₁/∂x_l, ∂S₂/∂x_l,
/// dS₁/dm_l, dS₂/dm_l per layer (L, 4, P, n_mu).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn layered_amplitudes<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    mu: Bound<'py, PyArrayDyn<f64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
) -> PyResult<Arrays<'py, Complex64>> {
    let (x, m, mu) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&mu, "mu")?);
    let (xv, mv) = views(&x, &m)?;
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let mu = cosines(py, &mu)?;
    let ((len, layers), n_mu) = (xv.dim(), mu.len());
    let arrays = if derivatives { 2 + 4 * layers } else { 2 };
    let count = len
        .checked_mul(n_mu.max(1))
        .and_then(|v| v.checked_mul(arrays));
    complex_values("the amplitudes", count, derivatives).map_err(|e| to_pyerr(py, e))?;
    let per_array = len * n_mu;
    let (s, d) = py
        .detach(|| -> Result<_, Failure> {
            let mut s = zeros::<Complex64>(2 * per_array, "the amplitudes")?;
            let mut d = zeros::<Complex64>(
                if derivatives {
                    4 * layers * per_array
                } else {
                    0
                },
                "derivatives",
            )?;
            let table = angular_table(&xv, &mv, &opts, &mu)?;
            let mut lay = Layered::<f64>::new();
            let (s1, s2) = s.split_at_mut(per_array);
            for i in 0..len {
                let (xi, mi) = row(&xv, &mv, i);
                if n_mu == 0 {
                    n_terms(xi, mi, &opts).map_err(at(i))?;
                    continue;
                }
                let sol = solve(&mut lay, i, xi, mi, &opts, derivatives)?;
                let r = i * n_mu..(i + 1) * n_mu;
                let values = AmplitudeBuffers {
                    s1: &mut s1[r.clone()],
                    s2: &mut s2[r.clone()],
                };
                match &table {
                    Some(t) => sol.amplitudes_from_table(t, values),
                    None => sol.amplitudes(&mu, values),
                }
                .map_err(at(i))?;
                for l in 0..if derivatives { layers } else { 0 } {
                    let block = &mut d[4 * l * per_array..4 * (l + 1) * per_array];
                    let (dx, dm) = block.split_at_mut(2 * per_array);
                    let (dx1, dx2) = dx.split_at_mut(per_array);
                    let (dm1, dm2) = dm.split_at_mut(per_array);
                    // The values again, into scratch: they were written above.
                    let z = Complex64::default();
                    let (mut v1, mut v2) = (vec![z; n_mu], vec![z; n_mu]);
                    let values = AmplitudeBuffers {
                        s1: &mut v1,
                        s2: &mut v2,
                    };
                    let d_dx = AmplitudeBuffers {
                        s1: &mut dx1[r.clone()],
                        s2: &mut dx2[r.clone()],
                    };
                    let d_dm = AmplitudeBuffers {
                        s1: &mut dm1[r.clone()],
                        s2: &mut dm2[r.clone()],
                    };
                    match &table {
                        Some(t) => {
                            sol.amplitudes_with_derivatives_from_table(t, l, values, d_dx, d_dm)
                        }
                        None => sol.amplitudes_with_derivatives(&mu, l, values, d_dx, d_dm),
                    }
                    .map_err(at(i))?;
                }
            }
            Ok((s, d))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let d = if derivatives {
        Some(array(py, &[layers, 4, len, n_mu], d)?)
    } else {
        None
    };
    Ok((array(py, &[2, len, n_mu], s)?, d))
}

/// F₁₁, F₁₂, F₃₃, F₃₄ (4, P, n_mu) at the cosines `mu`, normalized or not, and with
/// `derivatives` their derivatives with respect to each layer's x, Re m and Im m
/// (L, 3, 4, P, n_mu).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn layered_phase_matrix<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    mu: Bound<'py, PyArrayDyn<f64>>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
    normalized: bool,
) -> PyResult<Arrays<'py, f64>> {
    let (x, m, mu) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&mu, "mu")?);
    let (xv, mv) = views(&x, &m)?;
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    let mu = cosines(py, &mu)?;
    let ((len, layers), n_mu) = (xv.dim(), mu.len());
    let arrays = if derivatives { 4 + 12 * layers } else { 4 };
    let count = len
        .checked_mul(n_mu.max(1))
        .and_then(|v| v.checked_mul(arrays));
    real_values("the phase matrix", count, derivatives).map_err(|e| to_pyerr(py, e))?;
    let per_array = len * n_mu;
    let norm = normalization(normalized);
    let (f, d) = py
        .detach(|| -> Result<_, Failure> {
            let mut f = zeros::<f64>(4 * per_array, "the phase matrix")?;
            let mut d = zeros::<f64>(
                if derivatives {
                    12 * layers * per_array
                } else {
                    0
                },
                "derivatives",
            )?;
            let table = angular_table(&xv, &mv, &opts, &mu)?;
            let mut lay = Layered::<f64>::new();
            for i in 0..len {
                let (xi, mi) = row(&xv, &mv, i);
                if n_mu == 0 {
                    n_terms(xi, mi, &opts).map_err(at(i))?;
                    continue;
                }
                let sol = solve(&mut lay, i, xi, mi, &opts, derivatives)?;
                let r = i * n_mu..(i + 1) * n_mu;
                match &table {
                    Some(t) => sol.phase_matrix_from_table(
                        t,
                        norm,
                        phase_buffers(&mut f, per_array, r.clone()),
                    ),
                    None => {
                        sol.phase_matrix(&mu, norm, phase_buffers(&mut f, per_array, r.clone()))
                    }
                }
                .map_err(at(i))?;
                for l in 0..if derivatives { layers } else { 0 } {
                    let block = &mut d[12 * l * per_array..12 * (l + 1) * per_array];
                    let (dx, rest) = block.split_at_mut(4 * per_array);
                    let (dr, di) = rest.split_at_mut(4 * per_array);
                    let mut v = vec![0.0; 4 * n_mu];
                    let values = phase_buffers(&mut v, n_mu, 0..n_mu);
                    let dd = [
                        phase_buffers(dx, per_array, r.clone()),
                        phase_buffers(dr, per_array, r.clone()),
                        phase_buffers(di, per_array, r.clone()),
                    ];
                    match &table {
                        Some(t) => {
                            sol.phase_matrix_with_derivatives_from_table(t, l, norm, values, dd)
                        }
                        None => sol.phase_matrix_with_derivatives(&mu, l, norm, values, dd),
                    }
                    .map_err(at(i))?;
                }
            }
            Ok((f, d))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let d = if derivatives {
        Some(array(py, &[layers, 3, 4, len, n_mu], d)?)
    } else {
        None
    };
    Ok((array(py, &[4, len, n_mu], f)?, d))
}

type GreekArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Option<Bound<'py, PyArrayDyn<f64>>>,
);

/// The number of terms of each particle (P), the Greek coefficients (6, P, L + 1) up to `order`
/// (by default the largest 2N), normalized or not, and with `derivatives` their derivatives
/// with respect to each layer's x, Re m and Im m (layers, 3, 6, P, L + 1).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn layered_greek<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    order: Option<usize>,
    n_max: Option<usize>,
    derivatives: bool,
    allow_unvalidated: bool,
    normalized: bool,
) -> PyResult<GreekArrays<'py>> {
    let (x, m) = (borrow(&x, "x")?, borrow(&m, "m")?);
    let (xv, mv) = views(&x, &m)?;
    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
    if let Some(l) = order.filter(|&l| l > MAX_GREEK_ORDER) {
        return Err(to_pyerr(
            py,
            invalid(format!(
                "the Greek order must be at most {MAX_GREEK_ORDER}, got {l}"
            )),
        ));
    }
    let (len, layers) = xv.dim();
    let arrays = if derivatives { 6 + 18 * layers } else { 6 };
    real_values(
        "the Greek coefficients",
        len.checked_mul(arrays),
        derivatives,
    )
    .map_err(|e| to_pyerr(py, e))?;
    let norm = normalization(normalized);
    let (terms, g, d, width) = py
        .detach(|| -> Result<_, Failure> {
            let most = largest_n_terms(&xv, &mv, &opts)?;
            let width = order.unwrap_or(2 * most) + 1;
            let count = len.checked_mul(width).and_then(|v| v.checked_mul(arrays));
            let per_array = real_values("the Greek coefficients", count, derivatives)? / arrays;
            let mut terms = zeros::<u64>(len, "the numbers of terms")?;
            let mut g = zeros::<f64>(6 * per_array, "the Greek coefficients")?;
            let mut d = zeros::<f64>(
                if derivatives {
                    18 * layers * per_array
                } else {
                    0
                },
                "derivatives",
            )?;
            let mut lay = Layered::<f64>::new();
            for i in 0..len {
                let (xi, mi) = row(&xv, &mv, i);
                let s = solve(&mut lay, i, xi, mi, &opts, derivatives)?;
                if order.is_none() && s.n_terms() > most {
                    return Err(
                        invalid(format!("element {i}: x or m changed during the call")).into(),
                    );
                }
                terms[i] = s.n_terms() as u64;
                let r = i * width..(i + 1) * width;
                let put = |out: &mut [f64], first: usize, c: &GreekCoefficients<f64>| {
                    for (k, src) in greek_arrays(c).into_iter().enumerate() {
                        let off = (first + k) * per_array;
                        out[off + r.start..off + r.end].copy_from_slice(src);
                    }
                };
                put(&mut g, 0, &s.greek(Some(width - 1), norm).map_err(at(i))?);
                for l in 0..if derivatives { layers } else { 0 } {
                    let (_, jac) = s
                        .greek_with_derivatives(Some(width - 1), l, norm)
                        .map_err(at(i))?;
                    for (p, c) in jac.iter().enumerate() {
                        put(&mut d, 6 * (3 * l + p), c);
                    }
                }
            }
            Ok((terms, g, d, width))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let d = if derivatives {
        Some(array(py, &[layers, 3, 6, len, width], d)?)
    } else {
        None
    };
    Ok((
        PyArray1::from_vec(py, terms),
        array(py, &[6, len, width], g)?,
        d,
    ))
}
