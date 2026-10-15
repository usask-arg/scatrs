//! Bindings for `scatrs::tmatrix`.
//!
//! `tmatrix_random` takes `x` (float64), `m` (complex128) and `eps` (float64) arrays of one shape
//! (`scatrs.tmatrix` broadcasts them) and solves each spheroid once, returning every output asked
//! for in a dict of arrays with one column (or row) per particle. `TMatrixObject` holds one solved
//! spheroid for `scatrs.tmatrix.compute`. The GIL is released while computing.

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{Complex64, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scatrs::mie::{Efficiencies, GreekCoefficients};
use scatrs::tmatrix::{
    Derivatives, PhaseBuffers, Precision, PrecisionPath, Shape, TMatrix, TMatrixOptions,
    TMatrixSolution,
};

use crate::errors::to_pyerr;
use crate::mie::{
    Failure, at, borrow, cosines, greek_arrays, invalid, normalization, real_values, shape_error,
    zeros,
};

fn precision(py: Python<'_>, name: &str) -> PyResult<Precision> {
    match name {
        "auto" => Ok(Precision::Auto),
        "f64" => Ok(Precision::F64),
        "double-double" => Ok(Precision::DoubleDouble),
        other => Err(to_pyerr(
            py,
            invalid(format!(
                "precision must be 'auto', 'f64' or 'double-double', got {other:?}"
            )),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn options(
    py: Python<'_>,
    rtol: f64,
    precision_name: &str,
    n_terms: Option<usize>,
    n_theta: Option<usize>,
    greek_order: Option<usize>,
    allow_unconverged: bool,
    allow_unvalidated: bool,
) -> PyResult<TMatrixOptions> {
    let mut opts = TMatrixOptions::new()
        .with_rtol(rtol)
        .with_precision(precision(py, precision_name)?)
        .with_allow_unconverged(allow_unconverged)
        .with_allow_unvalidated(allow_unvalidated);
    if let Some(n) = n_terms {
        opts = opts.with_n_terms(n);
    }
    if let Some(n) = n_theta {
        opts = opts.with_n_theta(n);
    }
    if let Some(l) = greek_order {
        opts = opts.with_greek_order(l);
    }
    Ok(opts)
}

fn efficiency_values(q: &Efficiencies<f64>) -> [f64; 5] {
    [q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g]
}

fn path_name(p: PrecisionPath) -> &'static str {
    match p {
        PrecisionPath::F64 => "f64",
        PrecisionPath::Refined => "refined",
        PrecisionPath::DoubleDouble => "double-double",
        _ => "other",
    }
}

fn solve<'w>(
    tm: &'w mut TMatrix,
    eps: f64,
    x: f64,
    m: Complex64,
    opts: &TMatrixOptions,
    derivatives: bool,
) -> scatrs::Result<TMatrixSolution<'w>> {
    let shape = Shape::Spheroid { aspect_ratio: eps };
    if derivatives {
        tm.compute_with_derivatives(shape, x, m, opts)
    } else {
        tm.compute(shape, x, m, opts)
    }
}

/// Six phase buffers over row `j` of six consecutive (P, Q) arrays in `buf`.
fn rows(buf: &mut [f64], per: usize, j: usize, q: usize) -> PhaseBuffers<'_> {
    let mut parts = buf.chunks_mut(per).map(|a| &mut a[j * q..(j + 1) * q]);
    let mut next = || parts.next().expect("six arrays");
    PhaseBuffers {
        f11: next(),
        f12: next(),
        f22: next(),
        f33: next(),
        f34: next(),
        f44: next(),
    }
}

/// Solves each spheroid (x, m, eps of one shape, flattened to P) and returns, in a dict:
/// `n_terms`, `n_theta`, `error` (P); `efficiencies` (5, P) in the order Q_ext, Q_sca, Q_abs,
/// Q_back, g; with `greek`, `greek` (6, P, L + 1) (L = `order`, or the largest 2N); with `mu`,
/// `phase` (6, P, Q) in the order F11, F12, F22, F33, F34, F44; and with `derivatives`, each one's
/// `*_d` with a leading axis of 4 (x, aspect ratio, Re m, Im m).
#[pyfunction]
#[pyo3(signature = (x, m, eps, rtol, precision, n_terms, n_theta, allow_unconverged, allow_unvalidated, derivatives, greek, order, mu, normalized))]
#[allow(clippy::too_many_arguments)]
pub fn tmatrix_random<'py>(
    py: Python<'py>,
    x: Bound<'py, PyArrayDyn<f64>>,
    m: Bound<'py, PyArrayDyn<Complex64>>,
    eps: Bound<'py, PyArrayDyn<f64>>,
    rtol: f64,
    precision: &str,
    n_terms: Option<usize>,
    n_theta: Option<usize>,
    allow_unconverged: bool,
    allow_unvalidated: bool,
    derivatives: bool,
    greek: bool,
    order: Option<usize>,
    mu: Option<Bound<'py, PyArrayDyn<f64>>>,
    normalized: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (x, m, eps) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&eps, "eps")?);
    let (xv, mv, ev) = (x.as_array(), m.as_array(), eps.as_array());
    if xv.shape() != mv.shape() || xv.shape() != ev.shape() {
        return Err(PyValueError::new_err(
            "x, m and eps must have one shape (broadcast them first)",
        ));
    }
    let opts = options(
        py,
        rtol,
        precision,
        n_terms,
        n_theta,
        if greek { order } else { None },
        allow_unconverged,
        allow_unvalidated,
    )?;
    let mu = match &mu {
        Some(a) => Some(cosines(py, &borrow(a, "mu")?)?),
        None => None,
    };
    let norm = normalization(normalized);
    let len = xv.len();
    let n_mu = mu.as_ref().map_or(0, Vec::len);
    let per_phase = len * n_mu;
    let count = len.checked_mul(n_mu.max(1)).and_then(|v| v.checked_mul(30));
    real_values("the phase matrix", count, derivatives).map_err(|e| to_pyerr(py, e))?;
    type Out = (
        Vec<f64>,
        Vec<f64>,
        Vec<usize>,
        Vec<usize>,
        Vec<f64>,
        Vec<GreekCoefficients<f64>>,
        Vec<Derivatives<GreekCoefficients<f64>>>,
        Vec<f64>,
        Vec<f64>,
    );
    let out: Out = py
        .detach(|| -> Result<Out, Failure> {
            let mut q = zeros::<f64>(5 * len, "the efficiencies")?;
            let mut qd = zeros::<f64>(if derivatives { 20 * len } else { 0 }, "derivatives")?;
            let mut terms = vec![0usize; len];
            let mut thetas = vec![0usize; len];
            let mut error = vec![0.0f64; len];
            let mut greeks = Vec::new();
            let mut greeks_d = Vec::new();
            let mut f = zeros::<f64>(6 * per_phase, "the phase matrix")?;
            let mut fd = zeros::<f64>(if derivatives { 24 * per_phase } else { 0 }, "derivatives")?;
            let mut tm = TMatrix::new();
            for (i, ((&xi, &mi), &ei)) in xv.iter().zip(mv.iter()).zip(ev.iter()).enumerate() {
                let sol = solve(&mut tm, ei, xi, mi, &opts, derivatives).map_err(at(i))?;
                terms[i] = sol.n_terms();
                thetas[i] = sol.n_theta();
                error[i] = sol.error_estimate();
                for (k, v) in efficiency_values(&sol.efficiencies())
                    .into_iter()
                    .enumerate()
                {
                    q[k * len + i] = v;
                }
                if derivatives {
                    let j = sol.efficiencies_jacobian().ok_or_else(|| {
                        invalid("the efficiency derivatives are unavailable".into())
                    })?;
                    for (p, e) in [&j.x, &j.aspect_ratio, &j.m_re, &j.m_im]
                        .into_iter()
                        .enumerate()
                    {
                        for (k, v) in efficiency_values(e).into_iter().enumerate() {
                            qd[(p * 5 + k) * len + i] = v;
                        }
                    }
                }
                if greek {
                    if derivatives {
                        let (g, d) = sol.greek_with_derivatives(None, norm).map_err(at(i))?;
                        greeks.push(g);
                        greeks_d.push(d);
                    } else {
                        greeks.push(sol.greek(None, norm).map_err(at(i))?);
                    }
                }
                if let Some(mu) = &mu {
                    if n_mu > 0 {
                        let values = rows(&mut f, per_phase, i, n_mu);
                        if derivatives {
                            let mut parts = fd.chunks_mut(6 * per_phase);
                            let mut next = || rows(parts.next().expect("four"), per_phase, i, n_mu);
                            let d = Derivatives::new(next(), next(), next(), next());
                            sol.phase_matrix_with_derivatives(mu, norm, values, d)
                                .map_err(at(i))?;
                        } else {
                            sol.phase_matrix(mu, norm, values).map_err(at(i))?;
                        }
                    }
                }
            }
            Ok((q, qd, terms, thetas, error, greeks, greeks_d, f, fd))
        })
        .map_err(|f| f.into_pyerr(py))?;
    let (q, qd, terms, thetas, error, greeks, greeks_d, f, fd) = out;
    let dict = PyDict::new(py);
    dict.set_item("n_terms", Array1::from(terms).into_pyarray(py))?;
    dict.set_item("n_theta", Array1::from(thetas).into_pyarray(py))?;
    dict.set_item("error", Array1::from(error).into_pyarray(py))?;
    dict.set_item(
        "efficiencies",
        Array2::from_shape_vec((5, len), q)
            .map_err(shape_error)?
            .into_pyarray(py),
    )?;
    if derivatives {
        dict.set_item(
            "efficiencies_d",
            Array3::from_shape_vec((4, 5, len), qd)
                .map_err(shape_error)?
                .into_pyarray(py),
        )?;
    }
    if greek {
        let width = order.map_or_else(
            || greeks.iter().map(|g| g.alpha1.len()).max().unwrap_or(1),
            |l| l + 1,
        );
        let pack = |gs: &[&GreekCoefficients<f64>]| -> Vec<f64> {
            let mut v = vec![0.0; 6 * len * width];
            for (i, g) in gs.iter().enumerate() {
                for (k, a) in greek_arrays(g).into_iter().enumerate() {
                    let row = (k * len + i) * width;
                    let n = a.len().min(width);
                    v[row..row + n].copy_from_slice(&a[..n]);
                }
            }
            v
        };
        let refs: Vec<&GreekCoefficients<f64>> = greeks.iter().collect();
        dict.set_item(
            "greek",
            Array3::from_shape_vec((6, len, width), pack(&refs))
                .map_err(shape_error)?
                .into_pyarray(py),
        )?;
        if derivatives {
            let mut all = Vec::with_capacity(4 * 6 * len * width);
            for p in 0..4 {
                let refs: Vec<&GreekCoefficients<f64>> = greeks_d
                    .iter()
                    .map(|d| [&d.x, &d.aspect_ratio, &d.m_re, &d.m_im][p])
                    .collect();
                all.extend(pack(&refs));
            }
            dict.set_item(
                "greek_d",
                Array4::from_shape_vec((4, 6, len, width), all)
                    .map_err(shape_error)?
                    .into_pyarray(py),
            )?;
        }
    }
    if mu.is_some() {
        dict.set_item(
            "phase",
            Array3::from_shape_vec((6, len, n_mu), f)
                .map_err(shape_error)?
                .into_pyarray(py),
        )?;
        if derivatives {
            dict.set_item(
                "phase_d",
                Array4::from_shape_vec((4, 6, len, n_mu), fd)
                    .map_err(shape_error)?
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(dict)
}

/// Values (6, ·) and, if computed, their derivatives (4, 6, ·).
type WithDerivatives<'py> = (
    Bound<'py, numpy::PyArray2<f64>>,
    Option<Bound<'py, numpy::PyArray3<f64>>>,
);

/// One solved spheroid, for `scatrs.tmatrix.compute`.
#[pyclass(module = "scatrs._scatrs", name = "TMatrixObject", frozen)]
pub struct TMatrixObject {
    inner: TMatrix,
}

impl TMatrixObject {
    fn sol(&self) -> TMatrixSolution<'_> {
        self.inner.solution().expect("solved at construction")
    }
}

#[pymethods]
impl TMatrixObject {
    #[new]
    #[pyo3(signature = (x, m_re, m_im, eps, rtol, precision, n_terms, n_theta, allow_unconverged, allow_unvalidated, derivatives))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        x: f64,
        m_re: f64,
        m_im: f64,
        eps: f64,
        rtol: f64,
        precision: &str,
        n_terms: Option<usize>,
        n_theta: Option<usize>,
        allow_unconverged: bool,
        allow_unvalidated: bool,
        derivatives: bool,
    ) -> PyResult<Self> {
        let opts = options(
            py,
            rtol,
            precision,
            n_terms,
            n_theta,
            None,
            allow_unconverged,
            allow_unvalidated,
        )?;
        let m = Complex64::new(m_re, m_im);
        let inner = py
            .detach(|| -> scatrs::Result<TMatrix> {
                let mut tm = TMatrix::new();
                solve(&mut tm, eps, x, m, &opts, derivatives)?;
                Ok(tm)
            })
            .map_err(|e| to_pyerr(py, e))?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_terms(&self) -> usize {
        self.sol().n_terms()
    }

    #[getter]
    fn n_theta(&self) -> usize {
        self.sol().n_theta()
    }

    #[getter]
    fn error_estimate(&self) -> f64 {
        self.sol().error_estimate()
    }

    #[getter]
    fn converged(&self) -> bool {
        self.sol().converged()
    }

    #[getter]
    fn precision_paths(&self) -> Vec<&'static str> {
        self.sol()
            .precision_paths()
            .iter()
            .map(|&p| path_name(p))
            .collect()
    }

    /// (orders, T11, T12, T21, T22) of mode `m`, or None above N.
    #[allow(clippy::type_complexity)]
    fn block<'py>(
        &self,
        py: Python<'py>,
        m: usize,
    ) -> PyResult<
        Option<(
            Bound<'py, numpy::PyArray1<i64>>,
            [Bound<'py, numpy::PyArray2<Complex64>>; 4],
        )>,
    > {
        let Some(b) = self.sol().block(m) else {
            return Ok(None);
        };
        let k = b.orders.len();
        let orders = Array1::from_iter(b.orders.iter().map(|&n| n as i64)).into_pyarray(py);
        let mat = |v: Vec<Complex64>| -> PyResult<_> {
            Ok(Array2::from_shape_vec((k, k), v)
                .map_err(shape_error)?
                .into_pyarray(py))
        };
        Ok(Some((
            orders,
            [mat(b.t11)?, mat(b.t12)?, mat(b.t21)?, mat(b.t22)?],
        )))
    }

    /// The efficiencies (5,) and, if computed, their derivatives (4, 5).
    fn efficiencies<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, numpy::PyArray1<f64>>,
        Option<Bound<'py, numpy::PyArray2<f64>>>,
    ) {
        let s = self.sol();
        let v = Array1::from(efficiency_values(&s.efficiencies()).to_vec());
        let d = s.efficiencies_jacobian().map(|j| {
            let rows: Vec<f64> = [&j.x, &j.aspect_ratio, &j.m_re, &j.m_im]
                .into_iter()
                .flat_map(efficiency_values)
                .collect();
            Array2::from_shape_vec((4, 5), rows)
                .expect("4 x 5")
                .into_pyarray(py)
        });
        (v.into_pyarray(py), d)
    }

    /// The Greek coefficients (6, L + 1) and, if computed, their derivatives (4, 6, L + 1).
    #[pyo3(signature = (order, normalized))]
    fn greek<'py>(
        &self,
        py: Python<'py>,
        order: Option<usize>,
        normalized: bool,
    ) -> PyResult<WithDerivatives<'py>> {
        let s = self.sol();
        let norm = normalization(normalized);
        let flat = |g: &GreekCoefficients<f64>| -> Vec<f64> {
            greek_arrays(g).into_iter().flatten().copied().collect()
        };
        if s.efficiencies_jacobian().is_some() {
            let (g, d) = s
                .greek_with_derivatives(order, norm)
                .map_err(|e| to_pyerr(py, e))?;
            let w = g.alpha1.len();
            let v = Array2::from_shape_vec((6, w), flat(&g)).map_err(shape_error)?;
            let all: Vec<f64> = [&d.x, &d.aspect_ratio, &d.m_re, &d.m_im]
                .into_iter()
                .flat_map(flat)
                .collect();
            let d = Array3::from_shape_vec((4, 6, w), all).map_err(shape_error)?;
            Ok((v.into_pyarray(py), Some(d.into_pyarray(py))))
        } else {
            let g = s.greek(order, norm).map_err(|e| to_pyerr(py, e))?;
            let w = g.alpha1.len();
            let v = Array2::from_shape_vec((6, w), flat(&g)).map_err(shape_error)?;
            Ok((v.into_pyarray(py), None))
        }
    }

    /// The phase matrix (6, Q) at the cosines `mu`, and, if computed, its derivatives
    /// (4, 6, Q).
    fn phase_matrix<'py>(
        &self,
        py: Python<'py>,
        mu: PyReadonlyArrayDyn<'py, f64>,
        normalized: bool,
    ) -> PyResult<WithDerivatives<'py>> {
        let mu = cosines(py, &mu)?;
        let q = mu.len();
        let s = self.sol();
        let norm = normalization(normalized);
        let mut f = vec![0.0; 6 * q];
        if s.efficiencies_jacobian().is_some() {
            let mut fd = vec![0.0; 24 * q];
            {
                let mut parts = fd.chunks_mut(6 * q);
                let mut next = || rows(parts.next().expect("four"), q, 0, q);
                let d = Derivatives::new(next(), next(), next(), next());
                s.phase_matrix_with_derivatives(&mu, norm, rows(&mut f, q, 0, q), d)
                    .map_err(|e| to_pyerr(py, e))?;
            }
            let v = Array2::from_shape_vec((6, q), f).map_err(shape_error)?;
            let d = Array3::from_shape_vec((4, 6, q), fd).map_err(shape_error)?;
            Ok((v.into_pyarray(py), Some(d.into_pyarray(py))))
        } else {
            s.phase_matrix(&mu, norm, rows(&mut f, q, 0, q))
                .map_err(|e| to_pyerr(py, e))?;
            let v = Array2::from_shape_vec((6, q), f).map_err(shape_error)?;
            Ok((v.into_pyarray(py), None))
        }
    }
}
