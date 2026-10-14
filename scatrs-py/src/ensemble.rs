//! Bindings for `scatrs::ensemble`.
//!
//! Distributions arrive as tagged tuples from `scatrs.ensemble` (`("lognormal", r_g, σ_g)`,
//! `("tabulated", radii, densities)`, `("multimodal", [(weight, spec), …])`, …), components as
//! `(spec, refractive indices, weight)`. Results return as a dict of arrays, one row per
//! wavelength, which `scatrs.ensemble` wraps; derivatives are selected by their index in
//! `ensemble_parameters`.

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{Complex64, IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use scatrs::ensemble::{
    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs,
    SizeDistribution, compute_mixture, mixture_parameters, parameters,
};
use scatrs::mie::{MAX_GREEK_ORDER, Normalization};

use crate::errors::to_pyerr;
use crate::mie::MAX_MIE_VALUES;

fn invalid(message: String) -> scatrs::Error {
    scatrs::Error::InvalidInput(message)
}

/// A distribution from its tagged tuple.
fn distribution(spec: &Bound<'_, PyAny>) -> PyResult<SizeDistribution> {
    let t = spec
        .cast::<PyTuple>()
        .map_err(|_| PyTypeError::new_err("a distribution spec must be a tuple"))?;
    let kind: String = t.get_item(0)?.extract()?;
    let f = |i: usize| -> PyResult<f64> { t.get_item(i)?.extract() };
    Ok(match kind.as_str() {
        "lognormal" => SizeDistribution::LogNormal {
            median_radius: f(1)?,
            width: f(2)?,
        },
        "gamma" => SizeDistribution::Gamma {
            effective_radius: f(1)?,
            effective_variance: f(2)?,
        },
        "modified_gamma" => SizeDistribution::ModifiedGamma {
            alpha: f(1)?,
            b: f(2)?,
            gamma: f(3)?,
        },
        "power_law" => SizeDistribution::PowerLaw {
            exponent: f(1)?,
            min_radius: f(2)?,
            max_radius: f(3)?,
        },
        "monodisperse" => SizeDistribution::Monodisperse { radius: f(1)? },
        "tabulated" => SizeDistribution::Tabulated {
            radius: t.get_item(1)?.extract()?,
            density: t.get_item(2)?.extract()?,
        },
        "multimodal" => {
            let modes: Vec<(f64, Bound<'_, PyAny>)> = t.get_item(1)?.extract()?;
            SizeDistribution::MultiModal {
                modes: modes
                    .iter()
                    .map(|(w, s)| Ok((*w, distribution(s)?)))
                    .collect::<PyResult<_>>()?,
            }
        }
        other => {
            return Err(PyTypeError::new_err(format!(
                "unknown distribution kind {other:?}"
            )));
        }
    })
}

/// Components from `(spec, refractive indices, weight)` tuples.
fn components(
    items: &[(Bound<'_, PyAny>, PyReadonlyArray1<'_, Complex64>, f64)],
) -> PyResult<Vec<Component>> {
    items
        .iter()
        .map(|(spec, m, w)| {
            Ok(Component::new(
                distribution(spec)?,
                m.as_array().to_vec(),
                *w,
            ))
        })
        .collect()
}

fn all_parameters(components: &[Component], mixture: bool) -> Vec<EnsembleParam> {
    if mixture {
        mixture_parameters(components)
    } else {
        parameters(&components[0].distribution)
    }
}

/// The names of the parameters of `components` (a single distribution unless `mixture`), in
/// the order `ensemble_compute` indexes them.
#[pyfunction]
pub fn ensemble_parameters(
    items: Vec<(Bound<'_, PyAny>, PyReadonlyArray1<'_, Complex64>, f64)>,
    mixture: bool,
) -> PyResult<Vec<String>> {
    let cs = components(&items)?;
    if cs.is_empty() {
        return Ok(Vec::new());
    }
    Ok(all_parameters(&cs, mixture)
        .iter()
        .map(|p| p.name(mixture))
        .collect())
}

/// The scalar outputs, in the order of `scatrs.ensemble.SCALARS`.
fn scalars(o: &Outputs) -> [f64; 11] {
    [
        o.c_ext,
        o.c_sca,
        o.c_abs,
        o.c_back,
        o.ssa,
        o.g,
        o.lidar_ratio,
        o.depolarization,
        o.effective_radius,
        o.mean_area,
        o.mean_volume,
    ]
}

/// The ensemble outputs at each wavelength: a dict of arrays `values`/`errors` (11, n),
/// `d`/`d_errors` (P, 11, n), with the Greek coefficients (`greek`… (6, n, L + 1), zero-padded
/// to the longest) and the phase matrix (`phase`… (4, n, n_mu)) when requested, `converged`
/// (n) and `panels` (n). `greek` is the order, −1 for all orders, `None` for none.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ensemble_compute<'py>(
    py: Python<'py>,
    items: Vec<(Bound<'py, PyAny>, PyReadonlyArray1<'py, Complex64>, f64)>,
    wavelengths: PyReadonlyArray1<'py, f64>,
    mixture: bool,
    greek: Option<i64>,
    mu: Vec<f64>,
    normalized: bool,
    derivatives: Vec<usize>,
    rtol: f64,
    max_panels: usize,
    allow_unvalidated: bool,
    refine_backscatter: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cs = components(&items)?;
    let wl: Vec<f64> = wavelengths.as_array().to_vec();
    let n = wl.len();
    let err = |e| to_pyerr(py, e);
    if cs.is_empty() {
        return Err(err(invalid("a mixture needs a component".into())));
    }
    let all = all_parameters(&cs, mixture);
    let params: Vec<EnsembleParam> = derivatives
        .iter()
        .map(|&i| {
            all.get(i)
                .copied()
                .ok_or_else(|| err(invalid(format!("no parameter {i}"))))
        })
        .collect::<PyResult<_>>()?;
    let order = match greek {
        None => None,
        Some(-1) => Some(GreekOrder::All),
        Some(l) if l >= 0 && (l as u64) <= MAX_GREEK_ORDER as u64 => {
            Some(GreekOrder::Order(l as usize))
        }
        Some(l) => {
            return Err(err(invalid(format!(
                "the Greek order must be in [0, {MAX_GREEK_ORDER}], got {l}"
            ))));
        }
    };
    // Bound the outputs before computing where their size is known.
    let p = params.len();
    let per = 11
        + 4 * mu.len()
        + match order {
            Some(GreekOrder::Order(l)) => 6 * (l + 1),
            _ => 0,
        };
    if n.saturating_mul(per).saturating_mul(2 * (1 + p)) > 2 * MAX_MIE_VALUES {
        return Err(err(invalid(format!(
            "the ensemble outputs would exceed 2 * MAX_MIE_VALUES = {} real values",
            2 * MAX_MIE_VALUES
        ))));
    }
    let mut opts = EnsembleOptions::new()
        .with_rtol(rtol)
        .with_max_panels(max_panels)
        .with_mu(mu.clone())
        .with_normalization(if normalized {
            Normalization::Normalized
        } else {
            Normalization::Unnormalized
        })
        .with_derivatives(params)
        .with_allow_unvalidated(allow_unvalidated)
        .with_refine_backscatter(refine_backscatter);
    if let Some(o) = order {
        opts = opts.with_greek(o);
    }
    let results: Vec<EnsembleResult> = py
        .detach(|| compute_mixture(&cs, &wl, &opts))
        .map_err(err)?;
    let out = PyDict::new(py);
    let sc = |f: &dyn Fn(&EnsembleResult) -> &Outputs| {
        Array2::from_shape_fn((11, n), |(k, i)| scalars(f(&results[i]))[k])
    };
    let dsc = |f: &dyn Fn(&EnsembleResult, usize) -> &Outputs| {
        Array3::from_shape_fn((p, 11, n), |(c, k, i)| scalars(f(&results[i], c))[k])
    };
    out.set_item("values", sc(&|r| &r.values).into_pyarray(py))?;
    out.set_item("errors", sc(&|r| &r.errors).into_pyarray(py))?;
    out.set_item("d", dsc(&|r, c| &r.derivatives[c]).into_pyarray(py))?;
    out.set_item(
        "d_errors",
        dsc(&|r, c| &r.derivative_errors[c]).into_pyarray(py),
    )?;
    if order.is_some() {
        let width = results
            .iter()
            .map(|r| r.values.greek.as_ref().map_or(0, |g| g.alpha1.len()))
            .max()
            .unwrap_or(1);
        if n.saturating_mul(width).saturating_mul(12 * (1 + p)) > 2 * MAX_MIE_VALUES {
            return Err(err(invalid(format!(
                "the Greek coefficients would exceed 2 * MAX_MIE_VALUES = {} real values",
                2 * MAX_MIE_VALUES
            ))));
        }
        let pick = |o: &Outputs, k: usize, l: usize| {
            let g = o.greek.as_ref().expect("requested");
            let a = [
                &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
            ][k];
            a.get(l).copied().unwrap_or(0.0)
        };
        let g = |f: &dyn Fn(&EnsembleResult) -> &Outputs| {
            Array3::from_shape_fn((6, n, width), |(k, i, l)| pick(f(&results[i]), k, l))
        };
        let dg = |f: &dyn Fn(&EnsembleResult, usize) -> &Outputs| {
            Array4::from_shape_fn((p, 6, n, width), |(c, k, i, l)| {
                pick(f(&results[i], c), k, l)
            })
        };
        out.set_item("greek", g(&|r| &r.values).into_pyarray(py))?;
        out.set_item("greek_errors", g(&|r| &r.errors).into_pyarray(py))?;
        out.set_item("greek_d", dg(&|r, c| &r.derivatives[c]).into_pyarray(py))?;
        out.set_item(
            "greek_d_errors",
            dg(&|r, c| &r.derivative_errors[c]).into_pyarray(py),
        )?;
    }
    if !mu.is_empty() {
        let n_mu = mu.len();
        let pick = |o: &Outputs, k: usize, j: usize| {
            let ph = o.phase.as_ref().expect("requested");
            [&ph.f11, &ph.f12, &ph.f33, &ph.f34][k][j]
        };
        let f = |s: &dyn Fn(&EnsembleResult) -> &Outputs| {
            Array3::from_shape_fn((4, n, n_mu), |(k, i, j)| pick(s(&results[i]), k, j))
        };
        let df = |s: &dyn Fn(&EnsembleResult, usize) -> &Outputs| {
            Array4::from_shape_fn((p, 4, n, n_mu), |(c, k, i, j)| {
                pick(s(&results[i], c), k, j)
            })
        };
        out.set_item("phase", f(&|r| &r.values).into_pyarray(py))?;
        out.set_item("phase_errors", f(&|r| &r.errors).into_pyarray(py))?;
        out.set_item("phase_d", df(&|r, c| &r.derivatives[c]).into_pyarray(py))?;
        out.set_item(
            "phase_d_errors",
            df(&|r, c| &r.derivative_errors[c]).into_pyarray(py),
        )?;
    }
    out.set_item(
        "converged",
        Array1::from_iter(results.iter().map(|r| r.converged)).into_pyarray(py),
    )?;
    out.set_item(
        "panels",
        Array1::from_iter(results.iter().map(|r| r.panels as u64)).into_pyarray(py),
    )?;
    Ok(out)
}
