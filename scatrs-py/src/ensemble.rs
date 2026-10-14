//! Bindings for `scatrs::ensemble`.
//!
//! Distributions arrive as tagged tuples from `scatrs.ensemble` (`("lognormal", r_g, σ_g)`,
//! `("tabulated", radii, densities)`, `("multimodal", [(weight, spec), …])`, …), components as
//! `(spec, refractive indices, weight)`, with one refractive index per wavelength. Results
//! return as a dict of arrays, one row per wavelength, which `scatrs.ensemble` wraps;
//! derivatives are selected by their index in `ensemble_parameters`. Arrays are borrowed with
//! `try_readonly` (see `special.rs`), and the outputs are computed and laid out without the GIL.

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{Complex64, IntoPyArray, PyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use scatrs::ensemble::{
    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs,
    SizeDistribution, compute_batch, compute_mixture, mixture_parameters, parameters,
};
use scatrs::mie::{MAX_GREEK_ORDER, Normalization};

use crate::errors::to_pyerr;
use crate::mie::{MAX_MIE_VALUES, borrow};

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

/// A component as passed from Python: (spec, refractive indices, weight).
type Item<'py> = (Bound<'py, PyAny>, Bound<'py, PyArrayDyn<Complex64>>, f64);

/// Components from their items.
fn components(items: &[Item<'_>]) -> PyResult<Vec<Component>> {
    items
        .iter()
        .map(|(spec, m, w)| {
            let m = borrow(m, "refractive_index")?;
            Ok(Component::new(
                distribution(spec)?,
                m.as_array().iter().copied().collect(),
                *w,
            ))
        })
        .collect()
}

/// The parameters: of the first component's distribution, or of the mixture.
fn all_parameters(py: Python<'_>, cs: &[Component], mixture: bool) -> PyResult<Vec<EnsembleParam>> {
    if !mixture && cs.len() != 1 {
        return Err(to_pyerr(
            py,
            invalid(format!(
                "a single distribution needs exactly one component, got {}",
                cs.len()
            )),
        ));
    }
    Ok(if mixture {
        mixture_parameters(cs)
    } else {
        parameters(&cs[0].distribution)
    })
}

/// The names of the parameters of the components (a single distribution unless `mixture`),
/// in the order `ensemble_compute` indexes them.
#[pyfunction]
pub fn ensemble_parameters(
    py: Python<'_>,
    items: Vec<Item<'_>>,
    mixture: bool,
) -> PyResult<Vec<String>> {
    let cs = components(&items)?;
    if cs.is_empty() {
        return Ok(Vec::new());
    }
    Ok(all_parameters(py, &cs, mixture)?
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

/// The output arrays: values, errors, derivatives, derivative errors.
type Quad<A, B> = (A, A, B, B);

/// Every output array, laid out without the GIL.
struct Arrays {
    scalars: Quad<Array2<f64>, Array3<f64>>,
    greek: Option<Quad<Array3<f64>, Array4<f64>>>,
    phase: Option<Quad<Array3<f64>, Array4<f64>>>,
    converged: Array1<bool>,
    panels: Array1<u64>,
}

/// Lays out the results; `width` is the number of Greek orders, if requested.
fn arrays(
    results: &[EnsembleResult],
    p: usize,
    width: Option<usize>,
    n_mu: usize,
) -> scatrs::Result<Arrays> {
    let n = results.len();
    type Pick<'a> = &'a dyn Fn(&EnsembleResult, usize) -> &Outputs;
    let kinds: [Pick<'_>; 4] = [
        &|r, _| &r.values,
        &|r, _| &r.errors,
        &|r, c| &r.derivatives[c],
        &|r, c| &r.derivative_errors[c],
    ];
    let sc = |f: Pick<'_>| Array2::from_shape_fn((11, n), |(k, i)| scalars(f(&results[i], 0))[k]);
    let dsc =
        |f: Pick<'_>| Array3::from_shape_fn((p, 11, n), |(c, k, i)| scalars(f(&results[i], c))[k]);
    let scalars = (sc(kinds[0]), sc(kinds[1]), dsc(kinds[2]), dsc(kinds[3]));
    let greek = match width {
        None => None,
        Some(width) => {
            if n.saturating_mul(width).saturating_mul(12 * (1 + p)) > 2 * MAX_MIE_VALUES {
                return Err(invalid(format!(
                    "the Greek coefficients would exceed 2 * MAX_MIE_VALUES = {} real values",
                    2 * MAX_MIE_VALUES
                )));
            }
            let pick = |o: &Outputs, k: usize, l: usize| {
                let g = o.greek.as_ref().expect("requested");
                let a = [
                    &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
                ][k];
                a.get(l).copied().unwrap_or(0.0)
            };
            let g = |f: Pick<'_>| {
                Array3::from_shape_fn((6, n, width), |(k, i, l)| pick(f(&results[i], 0), k, l))
            };
            let dg = |f: Pick<'_>| {
                Array4::from_shape_fn((p, 6, n, width), |(c, k, i, l)| {
                    pick(f(&results[i], c), k, l)
                })
            };
            Some((g(kinds[0]), g(kinds[1]), dg(kinds[2]), dg(kinds[3])))
        }
    };
    let phase = (n_mu > 0).then(|| {
        let pick = |o: &Outputs, k: usize, j: usize| {
            let ph = o.phase.as_ref().expect("requested");
            [&ph.f11, &ph.f12, &ph.f33, &ph.f34][k][j]
        };
        let f = |s: Pick<'_>| {
            Array3::from_shape_fn((4, n, n_mu), |(k, i, j)| pick(s(&results[i], 0), k, j))
        };
        let df = |s: Pick<'_>| {
            Array4::from_shape_fn((p, 4, n, n_mu), |(c, k, i, j)| {
                pick(s(&results[i], c), k, j)
            })
        };
        (f(kinds[0]), f(kinds[1]), df(kinds[2]), df(kinds[3]))
    });
    Ok(Arrays {
        scalars,
        greek,
        phase,
        converged: results.iter().map(|r| r.converged).collect(),
        panels: results.iter().map(|r| r.panels as u64).collect(),
    })
}

/// The ensemble outputs at each wavelength: a dict of arrays `values`/`errors` (11, n),
/// `d`/`d_errors` (P, 11, n), with the Greek coefficients (`greek`… (6, n, L + 1), zero-padded
/// to the longest) and the phase matrix (`phase`… (4, n, n_mu)) when requested, `converged`
/// (n) and `panels` (n). The Greek coefficients go up to `greek`, or every order with
/// `all_orders`, or are not computed.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ensemble_compute<'py>(
    py: Python<'py>,
    items: Vec<Item<'py>>,
    wavelengths: Bound<'py, PyArrayDyn<f64>>,
    mixture: bool,
    greek: Option<u64>,
    all_orders: bool,
    mu: Bound<'py, PyArrayDyn<f64>>,
    normalized: bool,
    derivatives: Vec<usize>,
    rtol: f64,
    max_panels: usize,
    allow_unvalidated: bool,
    refine_backscatter: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let err = |e| to_pyerr(py, e);
    let cs = components(&items)?;
    let wl: Vec<f64> = borrow(&wavelengths, "wavelength")?
        .as_array()
        .iter()
        .copied()
        .collect();
    let mu: Vec<f64> = borrow(&mu, "mu")?.as_array().iter().copied().collect();
    if cs.is_empty() {
        return Err(err(invalid("a mixture needs a component".into())));
    }
    let all = all_parameters(py, &cs, mixture)?;
    let mut params: Vec<EnsembleParam> = Vec::with_capacity(derivatives.len());
    for &i in &derivatives {
        let p = *all
            .get(i)
            .ok_or_else(|| err(invalid(format!("no parameter {i}"))))?;
        if params.contains(&p) {
            return Err(err(invalid(format!(
                "parameter {} requested twice",
                p.name(mixture)
            ))));
        }
        params.push(p);
    }
    let (n, p, n_mu) = (wl.len(), params.len(), mu.len());
    let settings = Settings {
        greek,
        all_orders,
        normalized,
        rtol,
        max_panels,
        allow_unvalidated,
        refine_backscatter,
    };
    let (opts, order) = settings.options(mu, params, n, p).map_err(err)?;
    let arrays = py
        .detach(|| -> scatrs::Result<Arrays> {
            let results = compute_mixture(&cs, &wl, &opts)?;
            arrays(&results, p, width(order, &results), n_mu)
        })
        .map_err(err)?;
    into_dict(py, arrays)
}

/// The options shared by `ensemble_compute` and `ensemble_compute_batch`.
struct Settings {
    greek: Option<u64>,
    all_orders: bool,
    normalized: bool,
    rtol: f64,
    max_panels: usize,
    allow_unvalidated: bool,
    refine_backscatter: bool,
}

impl Settings {
    /// The options and the Greek order, after bounding the outputs of `n` results with `p`
    /// derivatives where their size is known before computing.
    fn options(
        &self,
        mu: Vec<f64>,
        params: Vec<EnsembleParam>,
        n: usize,
        p: usize,
    ) -> scatrs::Result<(EnsembleOptions, Option<GreekOrder>)> {
        let order = match (self.greek, self.all_orders) {
            (_, true) => Some(GreekOrder::All),
            (None, false) => None,
            (Some(l), false) if l <= MAX_GREEK_ORDER as u64 => Some(GreekOrder::Order(l as usize)),
            (Some(l), false) => {
                return Err(invalid(format!(
                    "the Greek order must be at most {MAX_GREEK_ORDER}, got {l}"
                )));
            }
        };
        let per = 11
            + 4 * mu.len()
            + match order {
                Some(GreekOrder::Order(l)) => 6 * (l + 1),
                _ => 0,
            };
        if n.saturating_mul(per).saturating_mul(2 * (1 + p)) > 2 * MAX_MIE_VALUES {
            return Err(invalid(format!(
                "the ensemble outputs would exceed 2 * MAX_MIE_VALUES = {} real values",
                2 * MAX_MIE_VALUES
            )));
        }
        let mut opts = EnsembleOptions::new()
            .with_rtol(self.rtol)
            .with_max_panels(self.max_panels)
            .with_mu(mu)
            .with_normalization(if self.normalized {
                Normalization::Normalized
            } else {
                Normalization::Unnormalized
            })
            .with_derivatives(params)
            .with_allow_unvalidated(self.allow_unvalidated)
            .with_refine_backscatter(self.refine_backscatter);
        if let Some(o) = order {
            opts = opts.with_greek(o);
        }
        Ok((opts, order))
    }
}

/// The number of Greek orders of the output arrays: the requested ones, or the longest.
fn width(order: Option<GreekOrder>, results: &[EnsembleResult]) -> Option<usize> {
    order.map(|o| match o {
        GreekOrder::Order(l) => l + 1,
        _ => results
            .iter()
            .map(|r| r.values.greek.as_ref().map_or(1, |g| g.alpha1.len()))
            .max()
            .unwrap_or(1),
    })
}

/// The arrays as the dict `scatrs.ensemble` unpacks.
fn into_dict(py: Python<'_>, arrays: Arrays) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    let (v, e, d, de) = arrays.scalars;
    out.set_item("values", v.into_pyarray(py))?;
    out.set_item("errors", e.into_pyarray(py))?;
    out.set_item("d", d.into_pyarray(py))?;
    out.set_item("d_errors", de.into_pyarray(py))?;
    for (prefix, quad) in [("greek", arrays.greek), ("phase", arrays.phase)] {
        if let Some((v, e, d, de)) = quad {
            out.set_item(prefix, v.into_pyarray(py))?;
            out.set_item(format!("{prefix}_errors"), e.into_pyarray(py))?;
            out.set_item(format!("{prefix}_d"), d.into_pyarray(py))?;
            out.set_item(format!("{prefix}_d_errors"), de.into_pyarray(py))?;
        }
    }
    out.set_item("converged", arrays.converged.into_pyarray(py))?;
    out.set_item("panels", arrays.panels.into_pyarray(py))?;
    Ok(out)
}

/// The ensemble outputs of each distribution (tagged tuples `specs`) at each wavelength, sharing
/// the size nodes: as `ensemble_compute` without derivatives, one row per (distribution,
/// wavelength), distribution-major.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ensemble_compute_batch<'py>(
    py: Python<'py>,
    specs: Vec<Bound<'py, PyAny>>,
    refractive_index: Bound<'py, PyArrayDyn<Complex64>>,
    wavelengths: Bound<'py, PyArrayDyn<f64>>,
    greek: Option<u64>,
    all_orders: bool,
    mu: Bound<'py, PyArrayDyn<f64>>,
    normalized: bool,
    rtol: f64,
    max_panels: usize,
    allow_unvalidated: bool,
    refine_backscatter: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let err = |e| to_pyerr(py, e);
    let distributions: Vec<SizeDistribution> =
        specs.iter().map(distribution).collect::<PyResult<_>>()?;
    let m: Vec<Complex64> = borrow(&refractive_index, "refractive_index")?
        .as_array()
        .iter()
        .copied()
        .collect();
    let wl: Vec<f64> = borrow(&wavelengths, "wavelength")?
        .as_array()
        .iter()
        .copied()
        .collect();
    let mu: Vec<f64> = borrow(&mu, "mu")?.as_array().iter().copied().collect();
    let n_mu = mu.len();
    let settings = Settings {
        greek,
        all_orders,
        normalized,
        rtol,
        max_panels,
        allow_unvalidated,
        refine_backscatter,
    };
    let n = distributions.len().saturating_mul(wl.len());
    let (opts, order) = settings.options(mu, Vec::new(), n, 0).map_err(err)?;
    let arrays = py
        .detach(|| -> scatrs::Result<Arrays> {
            let results: Vec<EnsembleResult> = compute_batch(&distributions, &wl, &m, &opts)?
                .into_iter()
                .flatten()
                .collect();
            arrays(&results, 0, width(order, &results), n_mu)
        })
        .map_err(err)?;
    into_dict(py, arrays)
}
