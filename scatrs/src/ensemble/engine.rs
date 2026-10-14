//! The ensemble engine: per wavelength, the size integration of each component (pass 1:
//! adaptive panels on the cheap quantities; pass 2: every output at the final nodes), the
//! mixture, and the outputs with their derivatives and error estimates.

use core::f64::consts::PI;

use num_complex::Complex;
use rayon::prelude::*;

use super::distribution::{SizeDistribution, Support};
use super::lin::Lin;
use super::moments::{Moments, moments};
use super::range::{self, MIE_POWERS, MOMENT_POWERS, STEP};
use super::{
    Component, EnsembleOptions, EnsembleResult, GreekOrder, Outputs, ParamKind, PhaseValues,
    kronrod,
};
use crate::mie::{
    GreekCoefficients, GreekProjector, MAX_SIZE_PARAMETER, Mie, MieOptions, MieSolution,
    Normalization, PhaseBuffers, PhaseTables, n_terms,
};
use crate::{Error, Result};

// The per-sphere quantities, in this order, then the phase-matrix tables at the projector's
// nodes (F₁₁, F₁₂, F₃₃, F₃₄, each `n_q` values) and at the requested cosines (each `n_mu`).
// Cross sections and tables are per sphere (πr²·Q, πr²·F with F per unit geometric cross
// section, so C_sca times the normalized phase matrix); integrated with the number density
// and divided by the integral of `ONE`, they become number averages.
const ONE: usize = 0;
const EXT: usize = 1;
const SCA: usize = 2;
const SCAG: usize = 3;
const ABS: usize = 4;
const BACK: usize = 5;
const R2: usize = 6;
const R3: usize = 7;
const SCALARS: usize = 8;

/// Pass 1 integrates the per-sphere quantities 1, C_ext, C_sca, C_sca·g, C_back and C_abs (the
/// first four at the indices of the flat layout, C_back at `P1_BACK`, C_abs at `P1_ABS`), then the
/// optional Greek coefficients (each πr²·α, 6 arrays of `refine_order` + 1) and the
/// refractive-index derivatives (`M_GROUP` per part of m: C_ext, C_sca, C_sca·g, C_back,
/// C_abs), and refines on all of them but C_back and its derivatives unless
/// `refine_backscatter`. The moments r² and r³ come from their own integration (`moments`).
const N_BASE: usize = 6;
const P1_BACK: usize = 4;
const P1_ABS: usize = 5;
const M_GROUP: usize = 5;

/// Panels of pass 2 per parallel task; fixed, so the order of the sums never depends on the
/// thread count.
const CHUNK: usize = 4;

/// At most this many range extensions per side.
const MAX_EXTENSIONS: usize = 100;

/// Panels narrower than this (in ln r) are not split.
const MIN_WIDTH: f64 = 1e-10;

/// The largest increase of the size parameter across an initial panel: the efficiencies'
/// broad interference structure has a period of about π/|m − 1| in x.
const MAX_DX: f64 = 20.0;

/// A component, or one mode of a multimodal component, integrated on its own.
struct Part<'a> {
    component: usize,
    mode: Option<usize>,
    dist: &'a SizeDistribution,
    m: &'a [Complex<f64>],
    /// The component's weight, and the mode's weight and the sum of the component's mode
    /// weights (1 and 1 without modes).
    w: f64,
    v: f64,
    v_total: f64,
    support: Support,
    /// The range of the Mie integration, before any extension.
    cutoffs: (f64, f64),
    /// ⟨r²⟩ and ⟨r³⟩ (`None` for a single radius).
    moments: Option<Moments>,
    /// The requested distribution parameters: (column of the result, index into the
    /// distribution's gradient).
    theta: Vec<(usize, usize)>,
    /// The result columns of ∂/∂(Re m) and ∂/∂(Im m), if requested.
    m_cols: [Option<usize>; 2],
}

impl Part<'_> {
    fn m_derivatives(&self) -> bool {
        self.m_cols.iter().any(Option::is_some)
    }
}

fn context(e: Error, lambda: f64, r: f64) -> Error {
    let at = format!("at wavelength {lambda:e}, radius {r:e}");
    match e {
        Error::InvalidInput(s) => Error::InvalidInput(format!("{at}: {s}")),
        Error::OutsideValidatedDomain(s) => Error::OutsideValidatedDomain(format!("{at}: {s}")),
        other => other,
    }
}

fn component_context(e: Error, component: usize) -> Error {
    match e {
        Error::InvalidInput(s) => Error::InvalidInput(format!("component {component}: {s}")),
        other => other,
    }
}

/// Splits the components into parts and maps the requested parameters onto them.
fn parts<'a>(
    components: &'a [Component],
    options: &EnsembleOptions,
    rtol: f64,
) -> Result<Vec<Part<'a>>> {
    let mut out = Vec::new();
    for (ci, c) in components.iter().enumerate() {
        let modes: Vec<(Option<usize>, f64, &SizeDistribution)> = match &c.distribution {
            SizeDistribution::MultiModal { modes } => modes
                .iter()
                .enumerate()
                .map(|(j, (v, d))| (Some(j), *v, d))
                .collect(),
            d => vec![(None, 1.0, d)],
        };
        let v_total: f64 = modes.iter().map(|(_, v, _)| v).sum();
        for (mode, v, dist) in modes {
            let names = dist.parameter_names();
            let mut theta = Vec::new();
            let mut m_cols = [None, None];
            for (col, p) in options.derivatives.iter().enumerate() {
                if p.component != ci {
                    continue;
                }
                match p.kind {
                    ParamKind::Distribution(name) if p.mode == mode => {
                        let k = names.iter().position(|n| *n == name).expect("validated");
                        theta.push((col, k));
                    }
                    ParamKind::RefrReal => m_cols[0] = Some(col),
                    ParamKind::RefrImag => m_cols[1] = Some(col),
                    _ => {}
                }
            }
            let support = dist.support();
            let breaks: &[f64] = match dist {
                SizeDistribution::Tabulated { radius, .. } => radius,
                _ => &[],
            };
            let gradient: Vec<usize> = theta.iter().map(|&(_, k)| k).collect();
            let (cutoffs, moments) = match support {
                Support::Unbounded(center) => {
                    let d = dist.prepare();
                    let mie = range::cutoffs(&d, center, rtol / 10.0, MIE_POWERS)?;
                    // No Mie solutions here, so a far smaller tail: the derivatives weight
                    // the tails by (∂ ln u/∂θ)², about z², 30 at five standard deviations.
                    let eps = (1e-3 * rtol).max(1e-16);
                    let wide = range::cutoffs(&d, center, eps, MOMENT_POWERS)?;
                    let m = moments(dist, &gradient, wide, breaks, rtol, 0.2 * rtol)
                        .map_err(|e| component_context(e, ci))?;
                    (mie, Some(m))
                }
                Support::Finite(lo, hi) => {
                    let m = moments(dist, &gradient, (lo, hi), breaks, rtol, 0.0)
                        .map_err(|e| component_context(e, ci))?;
                    ((lo, hi), Some(m))
                }
                Support::Point(r) => ((r, r), None),
            };
            out.push(Part {
                component: ci,
                mode,
                dist,
                m: &c.refractive_index,
                w: c.weight,
                v,
                v_total,
                support,
                cutoffs,
                moments,
                theta,
                m_cols,
            });
        }
    }
    Ok(out)
}

/// Runs every wavelength (validated inputs).
pub(super) fn run(
    components: &[Component],
    wavelengths: &[f64],
    options: &EnsembleOptions,
) -> Result<Vec<EnsembleResult>> {
    let parts = parts(components, options, options.rtol)?;
    let w_total: f64 = components.iter().map(|c| c.weight).sum();
    let results: Vec<Result<EnsembleResult>> = (0..wavelengths.len())
        .into_par_iter()
        .map(|i| wavelength(&parts, w_total, wavelengths[i], i, options))
        .collect();
    results.into_iter().collect()
}

/// The size parameter of radius `r` at `lambda`.
fn size(r: f64, lambda: f64) -> f64 {
    2.0 * PI * r / lambda
}

/// The initial panels of [t_lo, t_hi] (t = ln r): at most `max_step` wide and at most `MAX_DX`
/// in size parameter, with the radii of `breaks` (inside the range) as edges; if that would
/// take more than `budget` panels, equal panels between the edges, at least one per interval,
/// about `budget` in all.
fn initial_panels(
    t_lo: f64,
    t_hi: f64,
    lambda: f64,
    breaks: &[f64],
    max_step: f64,
    budget: usize,
) -> Vec<(f64, f64)> {
    let mut edges = vec![t_lo];
    edges.extend(
        breaks
            .iter()
            .map(|r| r.ln())
            .filter(|&t| t > t_lo && t < t_hi),
    );
    edges.push(t_hi);
    let mut out = Vec::new();
    for w in edges.windows(2) {
        let (mut a, b) = (w[0], w[1]);
        while a < b {
            let x = size(a.exp(), lambda);
            let step = max_step.min((1.0 + MAX_DX / x).ln());
            // Equal panels up to b rather than a short last one.
            let n = ((b - a) / step).ceil().max(1.0);
            let h = (b - a) / n;
            let next = if n <= 1.0 || a + h <= a { b } else { a + h };
            out.push((a, next));
            a = next;
            if out.len() > budget {
                break;
            }
        }
        if out.len() > budget {
            break;
        }
    }
    if out.len() <= budget {
        return out;
    }
    let width = t_hi - t_lo;
    let mut equal = Vec::new();
    for w in edges.windows(2) {
        let n = ((w[1] - w[0]) / width * budget as f64).floor().max(1.0) as usize;
        let h = (w[1] - w[0]) / n as f64;
        equal.extend((0..n).map(|i| (w[0] + i as f64 * h, w[0] + (i + 1) as f64 * h)));
    }
    equal
}

/// Pass-1 sums of one panel.
#[derive(Clone, Debug)]
struct Sums1 {
    /// Kronrod sums of each refined quantity, then of each θ-weighted one (block per θ).
    k: Vec<f64>,
    /// |Kronrod − Gauss| of the same.
    err: Vec<f64>,
    /// Kronrod sums of |∂u/∂θ|·r·|q|, the scales of the θ blocks.
    abs: Vec<f64>,
    /// Kronrod sums of u·r·|q|, the scales of the refractive-index derivatives.
    absv: Vec<f64>,
    /// The largest number of terms at the nodes.
    n_terms: usize,
    /// The integrands per unit ln r of every quantity (and θ block) at the first and the last
    /// node.
    edge: [Vec<f64>; 2],
}

/// The pass-1 quantities at radius r: 1, C_ext, C_sca, C_sca·g, C_back, then the Greek
/// coefficients up to `refine_order` (πr²·α), then the derivatives of C_ext, C_sca, C_sca·g and
/// C_back with respect to Re m and Im m where `m_derivatives` asks for them; and the number of
/// terms.
fn quantities1(
    mie: &mut Mie<f64>,
    r: f64,
    lambda: f64,
    m: Complex<f64>,
    opts: &MieOptions,
    refine_order: Option<usize>,
    m_derivatives: [bool; 2],
) -> Result<(Vec<f64>, usize)> {
    let x = size(r, lambda);
    let s = if m_derivatives.iter().any(|&b| b) {
        mie.compute_with_derivatives(x, m, opts)
    } else {
        mie.compute(x, m, opts)
    }
    .map_err(|e| context(e, lambda, r))?;
    let q = s.efficiencies();
    let area = PI * r * r;
    let mut out = vec![
        1.0,
        area * q.q_ext,
        area * q.q_sca,
        area * q.q_sca * q.g,
        area * q.q_back,
        area * q.q_abs,
    ];
    if let Some(order) = refine_order {
        let g = s
            .greek(Some(order), Normalization::Unnormalized)
            .map_err(|e| context(e, lambda, r))?;
        for a in [
            &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
        ] {
            out.extend(a.iter().map(|v| area * v));
        }
    }
    if let Some(jac) = s.efficiencies_jacobian() {
        for (want, d) in m_derivatives.into_iter().zip([jac.m_re, jac.m_im]) {
            if want {
                out.extend([
                    area * d.q_ext,
                    area * d.q_sca,
                    area * (d.q_sca * q.g + q.q_sca * d.g),
                    area * d.q_back,
                    area * d.q_abs,
                ]);
            }
        }
    }
    Ok((out, s.n_terms()))
}

/// The number of pass-1 quantities: the base ones, the Greek coefficients, and `M_GROUP`
/// derivatives per requested refractive-index part.
fn n_refined(refine_order: Option<usize>, part: &Part<'_>) -> usize {
    let md = part.m_cols.iter().filter(|c| c.is_some()).count();
    N_BASE + refine_order.map_or(0, |l| 6 * (l + 1)) + M_GROUP * md
}

/// Evaluates one panel in pass 1.
#[allow(clippy::too_many_arguments)]
fn panel1(
    mie: &mut Mie<f64>,
    part: &Part<'_>,
    (a, b): (f64, f64),
    lambda: f64,
    m: Complex<f64>,
    opts: &MieOptions,
    refine_order: Option<usize>,
    n_refined: usize,
) -> Result<Sums1> {
    let d = part.dist.prepare();
    let n_grad = part.dist.parameter_names().len();
    let nt = part.theta.len();
    let mut grad = vec![0.0; n_grad];
    let mut k = vec![0.0; (1 + nt) * n_refined];
    let mut g = vec![0.0; (1 + nt) * n_refined];
    let mut abs = vec![0.0; nt * n_refined];
    let mut absv = vec![0.0; n_refined];
    let m_derivatives = [part.m_cols[0].is_some(), part.m_cols[1].is_some()];
    let mut n_max = 0;
    let mut edge = [
        vec![0.0; (1 + nt) * n_refined],
        vec![0.0; (1 + nt) * n_refined],
    ];
    let nodes = kronrod::panel(a, b);
    for (i, &(t, wk, wg)) in nodes.iter().enumerate() {
        let r = t.exp();
        let u = d.eval(r, &mut grad);
        let (q, n) = quantities1(mie, r, lambda, m, opts, refine_order, m_derivatives)?;
        n_max = n_max.max(n);
        let base = u * r;
        let end = if i == 0 {
            Some(0)
        } else if i == nodes.len() - 1 {
            Some(1)
        } else {
            None
        };
        for (j, &qj) in q.iter().enumerate() {
            k[j] += wk * base * qj;
            g[j] += wg * base * qj;
            absv[j] += wk * (base * qj).abs();
            if let Some(e) = end {
                edge[e][j] = base * qj;
            }
            for (b_idx, &(_, gi)) in part.theta.iter().enumerate() {
                let dw = grad[gi] * r;
                let o = (1 + b_idx) * n_refined + j;
                k[o] += wk * dw * qj;
                g[o] += wg * dw * qj;
                abs[b_idx * n_refined + j] += wk * (dw * qj).abs();
                if let Some(e) = end {
                    edge[e][o] = dw * qj;
                }
            }
        }
    }
    let err = k.iter().zip(&g).map(|(x, y)| (x - y).abs()).collect();
    Ok(Sums1 {
        k,
        err,
        abs,
        absv,
        n_terms: n_max,
        edge,
    })
}

/// The outcome of pass 1 for one part.
struct Pass1 {
    panels: Vec<(f64, f64)>,
    converged: bool,
    n_terms: usize,
    range: (f64, f64),
}

/// Evaluates the panels without sums, in parallel and in order.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    panels: &mut [((f64, f64), Option<Sums1>)],
    part: &Part<'_>,
    lambda: f64,
    m: Complex<f64>,
    opts: &MieOptions,
    refine_order: Option<usize>,
    n_refined: usize,
) -> Result<()> {
    // Every panel, then the first error in order: an error must not depend on the threads.
    let results: Vec<Result<()>> = panels
        .par_iter_mut()
        .filter(|(_, s)| s.is_none())
        .map_init(Mie::<f64>::new, |mie, (ab, s)| {
            *s = Some(panel1(
                mie,
                part,
                *ab,
                lambda,
                m,
                opts,
                refine_order,
                n_refined,
            )?);
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Pass 1: the adaptive panels of one part at one wavelength.
fn pass1(
    part: &Part<'_>,
    lambda: f64,
    m: Complex<f64>,
    options: &EnsembleOptions,
    mie_opts: &MieOptions,
) -> Result<Pass1> {
    let rtol = options.rtol;
    let refine_order = options.refine_order;
    let n_refined = n_refined(refine_order, part);
    let n_greek = N_BASE + refine_order.map_or(0, |l| 6 * (l + 1));
    let nt = part.theta.len();
    let breaks: &[f64] = match part.dist {
        SizeDistribution::Tabulated { radius, .. } => radius,
        _ => &[],
    };
    let (mut t_lo, mut t_hi) = (part.cutoffs.0.ln(), part.cutoffs.1.ln());
    // Before any panel: a range whose largest size parameter no Mie solution accepts.
    let x_hi = size(part.cutoffs.1, lambda);
    if x_hi > MAX_SIZE_PARAMETER {
        return Err(context(
            Error::InvalidInput(format!(
                "the integration range reaches size parameter {x_hi:e}, above the accepted \
                 {MAX_SIZE_PARAMETER:e}"
            )),
            lambda,
            part.cutoffs.1,
        ));
    }
    let extendable = matches!(part.support, Support::Unbounded(_));
    let density = part.dist.prepare();
    let max_step = density.max_step(STEP);
    // Extensions of the range: `STEP`, or the peak's width for narrower peaks, whose tails fall
    // off over that width.
    let ext = STEP.min(density.log_width);
    let mut panels: Vec<((f64, f64), Option<Sums1>)> =
        initial_panels(t_lo, t_hi, lambda, breaks, max_step, options.max_panels)
            .into_iter()
            .map(|p| (p, None))
            .collect();
    let mut extensions = [0usize; 2];
    loop {
        evaluate(
            &mut panels,
            part,
            lambda,
            m,
            mie_opts,
            refine_order,
            n_refined,
        )?;
        let n = panels.len();
        let sums: Vec<&Sums1> = panels
            .iter()
            .map(|(_, s)| s.as_ref().expect("evaluated"))
            .collect();
        let total = |j: usize| sums.iter().map(|s| s.k[j]).sum::<f64>();
        let err = |j: usize| sums.iter().map(|s| s.err[j]).sum::<f64>();
        // The targets of each refined quantity.
        let mut target = vec![0.0; (1 + nt) * n_refined];
        // C_abs and its derivatives are refined to rtol of C_ext's (the accuracy of
        // C_ext − C_sca): relative to C_abs itself, the resonances of weakly absorbing large
        // spheres would take most of the panels.
        let scale_of = |j: usize| match j {
            P1_ABS => EXT,
            j if j >= n_greek && (j - n_greek) % M_GROUP == M_GROUP - 1 => j + 1 - M_GROUP,
            j => j,
        };
        let c_sca = total(SCA).abs();
        for j in 0..n_refined {
            let scale = match j {
                SCAG => c_sca,
                // As for single spheres (`MieOutput::Backscattering`): relative to the larger
                // of ⟨C_back⟩ and ⟨C_sca⟩, since backscattering by large weakly absorbing
                // spheres is a sum of narrow glory ripples orders of magnitude below C_sca.
                P1_BACK => total(P1_BACK).abs().max(c_sca),
                // The refractive-index derivatives can cancel: their absolute integrands.
                j if j >= n_greek => sums.iter().map(|s| s.absv[scale_of(j)]).sum::<f64>(),
                j if j >= N_BASE => c_sca,
                j => total(scale_of(j)).abs(),
            };
            target[j] = rtol * scale;
            for b in 0..nt {
                let k = b * n_refined + scale_of(j);
                let abs: f64 = sums.iter().map(|s| s.abs[k]).sum();
                target[(1 + b) * n_refined + j] = rtol * abs;
            }
        }
        // The averages take Z = ∫u and its derivatives from the moment integration: no
        // targets for ∫u over the Mie range, which would extend it down the number tail.
        for b in 0..=nt {
            target[b * n_refined + ONE] = f64::INFINITY;
        }
        // ∫∂u/∂θ·∂C/∂m is no output: no targets for the refractive-index derivatives in the θ
        // blocks.
        for b in 1..=nt {
            for j in n_greek..n_refined {
                target[b * n_refined + j] = f64::INFINITY;
            }
        }
        if !options.refine_backscatter {
            // C_back and its derivatives (the fourth of each refractive-index group) are only
            // integrated.
            for b in 0..=nt {
                target[b * n_refined + P1_BACK] = f64::INFINITY;
                for j in (n_greek + 3..n_refined).step_by(M_GROUP) {
                    target[b * n_refined + j] = f64::INFINITY;
                }
            }
        }
        let errors: Vec<f64> = (0..target.len()).map(err).collect();
        let met = errors.iter().zip(&target).all(|(e, t)| e <= t);
        if met {
            // Extend the range where an integrand is not negligible at an end: above a tenth
            // of its target per unit ln r, or per `ext / STEP` units for narrow peaks (the cross
            // sections of small absorbing spheres and the derivatives can have heavier tails
            // than the area the cut-offs bound).
            // The derivative integrals cancel (∂u/∂θ changes sign) while their tails carry
            // the extra weight of ∂ ln u/∂θ: their ends are compared with the derivative's own
            // size, floored at 10⁻³ of the absolute integral.
            let end_target: Vec<f64> = (0..target.len())
                .map(|o| {
                    if o < n_refined || target[o].is_infinite() {
                        target[o]
                    } else {
                        let (b, j) = (o / n_refined - 1, o % n_refined);
                        let k = b * n_refined + scale_of(j);
                        let abs: f64 = sums.iter().map(|s| s.abs[k]).sum();
                        rtol * total(o).abs().max(1e-3 * abs)
                    }
                })
                .collect();
            let small = |density: &Vec<f64>| {
                density
                    .iter()
                    .zip(&end_target)
                    .all(|(d, t)| d.abs() * (ext / STEP) <= 0.1 * t)
            };
            let small_first = small(&sums.first().expect("a panel").edge[0]);
            let small_last = small(&sums.last().expect("a panel").edge[1]);
            let mut extended = false;
            let wants = [
                extendable && !small_first && extensions[0] < MAX_EXTENSIONS,
                extendable && !small_last && extensions[1] < MAX_EXTENSIONS,
            ];
            if wants[0] {
                let fresh =
                    initial_panels(t_lo - ext, t_lo, lambda, &[], max_step, options.max_panels);
                if panels.len() + fresh.len() > options.max_panels {
                    return Ok(finish(panels, false, (t_lo, t_hi)));
                }
                t_lo -= ext;
                extensions[0] += 1;
                let mut new: Vec<_> = fresh.into_iter().map(|p| (p, None)).collect();
                new.append(&mut panels);
                panels = new;
                extended = true;
            }
            if wants[1] {
                let fresh =
                    initial_panels(t_hi, t_hi + ext, lambda, &[], max_step, options.max_panels);
                if panels.len() + fresh.len() > options.max_panels {
                    return Ok(finish(panels, false, (t_lo, t_hi)));
                }
                t_hi += ext;
                extensions[1] += 1;
                panels.extend(fresh.into_iter().map(|p| (p, None)));
                extended = true;
            }
            if !extended {
                // Converged unless an end still wants extending after `MAX_EXTENSIONS`.
                let ends_small = !extendable || (small_first && small_last);
                return Ok(finish(panels, ends_small, (t_lo, t_hi)));
            }
            continue;
        }
        // Split every panel above its share of a target, worst first, within the budget.
        let mut candidates: Vec<(f64, usize)> = sums
            .iter()
            .enumerate()
            .filter(|&(i, _)| panels[i].0.1 - panels[i].0.0 > MIN_WIDTH)
            .filter_map(|(i, s)| {
                let badness = s
                    .err
                    .iter()
                    .zip(&target)
                    .zip(&errors)
                    // Only the quantities still above their targets.
                    .filter(|&((_, t), &total_err)| total_err > *t)
                    .map(|((e, t), _)| e * n as f64 / t.max(f64::MIN_POSITIVE))
                    .fold(0.0, f64::max);
                (badness > 1.0).then_some((badness, i))
            })
            .collect();
        let budget = options.max_panels.saturating_sub(n);
        if candidates.is_empty() || budget == 0 {
            return Ok(finish(panels, false, (t_lo, t_hi)));
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        candidates.truncate(budget);
        let mut split = vec![false; n];
        for &(_, i) in &candidates {
            split[i] = true;
        }
        let mut next = Vec::with_capacity(n + candidates.len());
        for (i, p) in panels.into_iter().enumerate() {
            if split[i] {
                let ((a, b), _) = p;
                let mid = 0.5 * (a + b);
                next.push(((a, mid), None));
                next.push(((mid, b), None));
            } else {
                next.push(p);
            }
        }
        panels = next;
    }
}

fn finish(panels: Vec<((f64, f64), Option<Sums1>)>, converged: bool, t: (f64, f64)) -> Pass1 {
    let n_terms = panels
        .iter()
        .map(|(_, s)| s.as_ref().map_or(0, |s| s.n_terms))
        .max()
        .unwrap_or(0);
    Pass1 {
        panels: panels.into_iter().map(|(p, _)| p).collect(),
        converged,
        n_terms,
        range: (t.0.exp(), t.1.exp()),
    }
}

/// Greek-coefficient projection groups the size panels into bands by their largest number of
/// terms N (`panel_terms`), each band with its own exact projector: orders up to
/// L_b = min(L, 2·N_band), where a sphere's coefficients vanish, on n_q = N_band + ⌈(L_b + 1)/2⌉
/// nodes, N_band the band's largest N. A sphere's amplitudes then cost O(N·(N_band + L_b))
/// rather than O(N·N_max), which matters for broad distributions. The band of N is the first
/// `bucket` value ≥ N, in a sequence that grows by a factor of 1.25.
fn bucket(n: usize) -> usize {
    let mut u = 1usize;
    while u < n {
        u = (u as f64 * 1.25).ceil() as usize;
    }
    u
}

/// The largest number of terms at the nodes of the panel [a, b] (t = ln r), which chooses the
/// panel's band.
fn panel_terms(a: f64, b: f64, lambda: f64, m: Complex<f64>, opts: &MieOptions) -> Result<usize> {
    let mut most = 0;
    for &(t, _, _) in &kronrod::panel(a, b) {
        let r = t.exp();
        most = most.max(n_terms(size(r, lambda), m, opts).map_err(|e| context(e, lambda, r))?);
    }
    Ok(most)
}

/// One band: its largest number of terms, the start of its tables in the flat layout, and
/// its projector.
struct Band {
    upper: usize,
    start: usize,
    projector: GreekProjector<f64>,
}

/// What pass 2 computes at every node: the per-sphere scalars, the phase matrix at the
/// requested cosines (each element `n_mu` values), and the tables of each band at its
/// projector's nodes.
struct Layout<'a> {
    mu: &'a [f64],
    bands: Vec<Band>,
    len: usize,
}

impl Layout<'_> {
    /// The scalars and the cosine tables: the part of the layout every node fills.
    fn common(&self) -> usize {
        SCALARS + 4 * self.mu.len()
    }

    fn band(&self, n: usize) -> Option<&Band> {
        let u = bucket(n);
        self.bands.iter().find(|b| b.upper == u)
    }
}

/// The per-sphere quantities of pass 2 at radius r into `g` (the common part and the node's
/// band, whose (start, length) it returns), and their derivatives with respect to Re m, Im m
/// into `dm` and with respect to r into `dr`, if given.
#[allow(clippy::too_many_arguments)]
fn quantities2(
    mie: &mut Mie<f64>,
    r: f64,
    lambda: f64,
    m: Complex<f64>,
    opts: &MieOptions,
    layout: &Layout<'_>,
    g: &mut [f64],
    mut dm: Option<[&mut [f64]; 2]>,
    dr: Option<&mut [f64]>,
    band_terms: usize,
) -> Result<Option<(usize, usize)>> {
    let x = size(r, lambda);
    let derivatives = dm.is_some() || dr.is_some();
    let s = if derivatives {
        mie.compute_with_derivatives(x, m, opts)
    } else {
        mie.compute(x, m, opts)
    }
    .map_err(|e| context(e, lambda, r))?;
    let area = PI * r * r;
    let q = s.efficiencies();
    g[..SCALARS].copy_from_slice(&[
        1.0,
        area * q.q_ext,
        area * q.q_sca,
        area * q.q_sca * q.g,
        area * q.q_abs,
        area * q.q_back,
        r * r,
        r * r * r,
    ]);
    // The band of the panel (its largest number of terms), so that every node of a panel
    // fills the same tables: per-node bands would make each band's table a step function
    // within the panel, and its Kronrod–Gauss difference a measure of the step.
    let n = band_terms.max(s.n_terms());
    let band = layout.band(n).map(|b| (b.start, b.projector.nodes()));
    if !layout.bands.is_empty() && band.is_none() {
        return Err(context(
            Error::InvalidInput(format!("no projector band for {n} terms")),
            lambda,
            r,
        ));
    }
    let mut tables = vec![(SCALARS, layout.mu)];
    tables.extend(band);
    let segment = band.map(|(start, nodes)| (start, 4 * nodes.len()));
    if !derivatives {
        for (start, cosines) in tables {
            let n = cosines.len();
            if n == 0 {
                continue;
            }
            phase_into(&s, cosines, &mut g[start..start + 4 * n])
                .map_err(|e| context(e, lambda, r))?;
            g[start..start + 4 * n].iter_mut().for_each(|v| *v *= area);
        }
        return Ok(segment);
    }
    let jac = s
        .efficiencies_jacobian()
        .expect("computed with derivatives");
    let scalar_d = |e: &crate::mie::Efficiencies<f64>| {
        [
            0.0,
            area * e.q_ext,
            area * e.q_sca,
            area * (e.q_sca * q.g + q.q_sca * e.g),
            area * e.q_abs,
            area * e.q_back,
            0.0,
            0.0,
        ]
    };
    let k = 2.0 * PI / lambda;
    if let Some([dre, dim]) = dm.as_mut() {
        dre[..SCALARS].copy_from_slice(&scalar_d(&jac.m_re));
        dim[..SCALARS].copy_from_slice(&scalar_d(&jac.m_im));
    }
    let mut dr = dr;
    if let Some(dr) = dr.as_deref_mut() {
        // ∂(πr²Q)/∂r = 2πr Q + πr² (2π/λ) ∂Q/∂x.
        let dx = scalar_d(&jac.x);
        let values = [0.0, q.q_ext, q.q_sca, q.q_sca * q.g, q.q_abs, q.q_back];
        for j in 1..6 {
            dr[j] = 2.0 * PI * r * values[j] + k * dx[j];
        }
        dr[ONE] = 0.0;
        dr[R2] = 2.0 * r;
        dr[R3] = 3.0 * r * r;
    }
    for (start, cosines) in tables {
        let n = cosines.len();
        if n == 0 {
            continue;
        }
        let span = start..start + 4 * n;
        let mut dx = vec![0.0; 4 * n];
        let (mut dre_buf, mut dim_buf) = (vec![0.0; 4 * n], vec![0.0; 4 * n]);
        {
            let [v11, v12, v33, v34] = quarter(&mut g[span.clone()]);
            let [x11, x12, x33, x34] = quarter(&mut dx);
            let [r11, r12, r33, r34] = quarter(&mut dre_buf);
            let [i11, i12, i33, i34] = quarter(&mut dim_buf);
            s.phase_matrix_with_derivatives(
                cosines,
                Normalization::Unnormalized,
                PhaseBuffers {
                    f11: v11,
                    f12: v12,
                    f33: v33,
                    f34: v34,
                },
                [
                    PhaseBuffers {
                        f11: x11,
                        f12: x12,
                        f33: x33,
                        f34: x34,
                    },
                    PhaseBuffers {
                        f11: r11,
                        f12: r12,
                        f33: r33,
                        f34: r34,
                    },
                    PhaseBuffers {
                        f11: i11,
                        f12: i12,
                        f33: i33,
                        f34: i34,
                    },
                ],
            )
            .map_err(|e| context(e, lambda, r))?;
        }
        if let Some(dr) = dr.as_deref_mut() {
            for (j, o) in span.clone().enumerate() {
                dr[o] = 2.0 * PI * r * g[o] + area * k * dx[j];
            }
        }
        if let Some([dre, dim]) = dm.as_mut() {
            for (j, o) in span.clone().enumerate() {
                dre[o] = area * dre_buf[j];
                dim[o] = area * dim_buf[j];
            }
        }
        g[span].iter_mut().for_each(|v| *v *= area);
    }
    Ok(segment)
}

/// A table of four elements as phase-matrix tables.
fn view(t: &[Vec<f64>; 4]) -> PhaseTables<'_, f64> {
    PhaseTables {
        f11: &t[0],
        f12: &t[1],
        f33: &t[2],
        f34: &t[3],
    }
}

/// Splits a table of 4·n values into its four elements.
fn quarter(t: &mut [f64]) -> [&mut [f64]; 4] {
    let n = t.len() / 4;
    let (a, rest) = t.split_at_mut(n);
    let (b, rest) = rest.split_at_mut(n);
    let (c, d) = rest.split_at_mut(n);
    [a, b, c, d]
}

/// The unnormalized phase matrix of `s` at `cosines` into the 4·n values of `out`.
fn phase_into(s: &MieSolution<'_, f64>, cosines: &[f64], out: &mut [f64]) -> Result<()> {
    let [f11, f12, f33, f34] = quarter(out);
    s.phase_matrix(
        cosines,
        Normalization::Unnormalized,
        PhaseBuffers { f11, f12, f33, f34 },
    )
}

/// Pass-2 integrals of one part: blocks of the flat layout (values; each requested θ;
/// Re m, Im m if requested), Kronrod sums and Σ over panels of |Kronrod − Gauss|.
struct Sums2 {
    s: Vec<f64>,
    err: Vec<f64>,
    blocks: usize,
}

/// Pass 2 of one part over its final panels.
fn pass2(
    part: &Part<'_>,
    panels: &[(f64, f64)],
    lambda: f64,
    m: Complex<f64>,
    opts: &MieOptions,
    layout: &Layout<'_>,
) -> Result<Sums2> {
    let len = layout.len;
    let common = layout.common();
    let nt = part.theta.len();
    let nm = if part.m_derivatives() { 2 } else { 0 };
    let blocks = 1 + nt + nm;
    let n_grad = part.dist.parameter_names().len();
    // Per-thread scratch: the dense per-panel sums are only touched on the ranges the panel's
    // nodes fill; each chunk returns its sums of those ranges alone.
    struct Scratch {
        mie: Mie<f64>,
        grad: Vec<f64>,
        g: Vec<f64>,
        dmv: Vec<f64>,
        pk: Vec<f64>,
        pg: Vec<f64>,
    }
    /// A chunk's sums and errors on one range [lo, hi) of the layout, every block.
    type Segment = ((usize, usize), Vec<f64>, Vec<f64>);
    let chunks: Vec<Result<Vec<Segment>>> = panels
        .par_chunks(CHUNK)
        .map_init(
            || Scratch {
                mie: Mie::new(),
                grad: vec![0.0; n_grad],
                g: vec![0.0; len],
                dmv: vec![0.0; 2 * len],
                pk: vec![0.0; blocks * len],
                pg: vec![0.0; blocks * len],
            },
            |sc, chunk| {
                let d = part.dist.prepare();
                let mut segments: Vec<Segment> = Vec::new();
                for &(a, b) in chunk {
                    let band_terms = if layout.bands.is_empty() {
                        0
                    } else {
                        panel_terms(a, b, lambda, m, opts)?
                    };
                    let mut ranges = vec![(0, common)];
                    for &(t, wk, wg) in &kronrod::panel(a, b) {
                        let r = t.exp();
                        let u = d.eval(r, &mut sc.grad);
                        let dm = if nm > 0 {
                            let (re, im) = sc.dmv.split_at_mut(len);
                            Some([re, im])
                        } else {
                            None
                        };
                        let segment = quantities2(
                            &mut sc.mie,
                            r,
                            lambda,
                            m,
                            opts,
                            layout,
                            &mut sc.g,
                            dm,
                            None,
                            band_terms,
                        )?;
                        let band = segment.map(|(s, n)| (s, s + n));
                        if let Some(range) = band {
                            if !ranges.contains(&range) {
                                ranges.push(range);
                            }
                        }
                        let base = u * r;
                        for (lo, hi) in [Some((0, common)), band].into_iter().flatten() {
                            for i in lo..hi {
                                let gi = sc.g[i];
                                sc.pk[i] += wk * base * gi;
                                sc.pg[i] += wg * base * gi;
                                for (bi, &(_, k)) in part.theta.iter().enumerate() {
                                    let o = (1 + bi) * len + i;
                                    let dw = sc.grad[k] * r * gi;
                                    sc.pk[o] += wk * dw;
                                    sc.pg[o] += wg * dw;
                                }
                                for c in 0..nm {
                                    let o = (1 + nt + c) * len + i;
                                    let dv = base * sc.dmv[c * len + i];
                                    sc.pk[o] += wk * dv;
                                    sc.pg[o] += wg * dv;
                                }
                            }
                        }
                    }
                    for &(lo, hi) in &ranges {
                        let w = hi - lo;
                        let k = match segments.iter().position(|(rg, _, _)| *rg == (lo, hi)) {
                            Some(k) => k,
                            None => {
                                segments.push((
                                    (lo, hi),
                                    vec![0.0; blocks * w],
                                    vec![0.0; blocks * w],
                                ));
                                segments.len() - 1
                            }
                        };
                        let (_, ssum, serr) = &mut segments[k];
                        for blk in 0..blocks {
                            for i in 0..w {
                                let o = blk * len + lo + i;
                                ssum[blk * w + i] += sc.pk[o];
                                serr[blk * w + i] += (sc.pk[o] - sc.pg[o]).abs();
                                sc.pk[o] = 0.0;
                                sc.pg[o] = 0.0;
                            }
                        }
                    }
                }
                Ok(segments)
            },
        )
        .collect();
    let mut s = vec![0.0; blocks * len];
    let mut err = vec![0.0; blocks * len];
    for c in chunks {
        for ((lo, hi), cs, ce) in c? {
            let w = hi - lo;
            for blk in 0..blocks {
                for i in 0..w {
                    s[blk * len + lo + i] += cs[blk * w + i];
                    err[blk * len + lo + i] += ce[blk * w + i];
                }
            }
        }
    }
    Ok(Sums2 { s, err, blocks })
}

/// A part's number averages with derivatives (by result column) and error estimates.
struct PartAverages {
    v: Vec<f64>,
    e: Vec<f64>,
    /// (column, derivative, its error).
    d: Vec<(usize, Vec<f64>, Vec<f64>)>,
}

/// Number averages from the pass-2 integrals: E = S/Z, with Z = ∫u dr from the moment
/// integration (whose range bounds the number tails), the quotient rule for the θ blocks (the
/// renormalization) and first-order error propagation. Each error estimate also includes
/// `truncation`·|value| for the cut-offs: the Mie range omits at most rtol/10 of the area (and
/// is extended until every refined integrand is small at both ends), which moves an average by
/// at most about twice that.
fn averages(part: &Part<'_>, sums: &Sums2, len: usize, truncation: f64) -> PartAverages {
    let block = |b: usize| &sums.s[b * len..(b + 1) * len];
    let eblock = |b: usize| &sums.err[b * len..(b + 1) * len];
    let (s0, e0) = (block(0), eblock(0));
    let mom = part.moments.as_ref().expect("an integrated part");
    let (z, ez) = (mom.z, mom.ez);
    let v: Vec<f64> = s0.iter().map(|x| x / z).collect();
    let e: Vec<f64> = (0..len)
        .map(|i| (e0[i] + v[i].abs() * ez) / z + truncation * v[i].abs())
        .collect();
    let mut d = Vec::new();
    for (bi, &(col, _)) in part.theta.iter().enumerate() {
        let (sb, eb) = (block(1 + bi), eblock(1 + bi));
        let (dz, edz) = (mom.dz[bi], mom.dez[bi]);
        let dv: Vec<f64> = (0..len).map(|i| (sb[i] - v[i] * dz) / z).collect();
        let de: Vec<f64> = (0..len)
            .map(|i| {
                (eb[i] + v[i].abs() * edz + dv[i].abs() * ez + e[i] * dz.abs()) / z
                    + truncation * dv[i].abs()
            })
            .collect();
        d.push((col, dv, de));
    }
    let nt = part.theta.len();
    let mut c = 0;
    for col in part.m_cols {
        if let Some(col) = col {
            let (sb, eb) = (block(1 + nt + c), eblock(1 + nt + c));
            let dv: Vec<f64> = (0..len).map(|i| sb[i] / z).collect();
            let de: Vec<f64> = (0..len)
                .map(|i| (eb[i] + dv[i].abs() * ez) / z + truncation * dv[i].abs())
                .collect();
            d.push((col, dv, de));
        }
        if part.m_derivatives() {
            c += 1;
        }
    }
    let mut out = PartAverages { v, e, d };
    if let Some(m) = &part.moments {
        for (k, idx) in [R2, R3].into_iter().enumerate() {
            out.v[idx] = m.v[k];
            out.e[idx] = m.e[k];
            for (col, dv, de) in &mut out.d {
                match part.theta.iter().position(|&(c, _)| c == *col) {
                    Some(b) => {
                        dv[idx] = m.d[b][k];
                        de[idx] = m.de[b][k];
                    }
                    None => {
                        dv[idx] = 0.0;
                        de[idx] = 0.0;
                    }
                }
            }
        }
    }
    out
}

/// A monodisperse part: the sphere itself, exact (no integration).
fn monodisperse(
    part: &Part<'_>,
    r: f64,
    lambda: f64,
    m: Complex<f64>,
    opts: &MieOptions,
    layout: &Layout<'_>,
) -> Result<PartAverages> {
    let len = layout.len;
    let mut mie = Mie::<f64>::new();
    let mut g = vec![0.0; len];
    let mut dmv = vec![0.0; 2 * len];
    let mut drv = vec![0.0; len];
    let radius_col = part.theta.first().map(|&(col, _)| col);
    {
        let dm = if part.m_derivatives() {
            let (re, im) = dmv.split_at_mut(len);
            Some([re, im])
        } else {
            None
        };
        let dr = radius_col.map(|_| drv.as_mut_slice());
        let n = n_terms(size(r, lambda), m, opts).map_err(|e| context(e, lambda, r))?;
        quantities2(&mut mie, r, lambda, m, opts, layout, &mut g, dm, dr, n)?;
    }
    let zeros = vec![0.0; len];
    let mut d = Vec::new();
    if let Some(col) = radius_col {
        d.push((col, drv, zeros.clone()));
    }
    for (c, col) in part.m_cols.iter().enumerate() {
        if let Some(col) = col {
            d.push((*col, dmv[c * len..(c + 1) * len].to_vec(), zeros.clone()));
        }
    }
    Ok(PartAverages { v: g, e: zeros, d })
}

/// One wavelength.
fn wavelength(
    parts: &[Part<'_>],
    w_total: f64,
    lambda: f64,
    index: usize,
    options: &EnsembleOptions,
) -> Result<EnsembleResult> {
    let mie_opts = MieOptions::new().with_allow_unvalidated(options.allow_unvalidated);
    let n_cols = options.derivatives.len();
    // Pass 1 for every part, and the largest number of terms.
    let mut firsts = Vec::with_capacity(parts.len());
    let mut n_max = 0;
    for part in parts {
        let m = part.m[index];
        let p1 = match part.support {
            Support::Point(r) => {
                let n =
                    n_terms(size(r, lambda), m, &mie_opts).map_err(|e| context(e, lambda, r))?;
                Pass1 {
                    panels: Vec::new(),
                    converged: true,
                    n_terms: n,
                    range: (r, r),
                }
            }
            _ => pass1(part, lambda, m, options, &mie_opts)?,
        };
        n_max = n_max.max(p1.n_terms);
        firsts.push(p1);
    }
    // The projector bands, from the numbers of terms at the final nodes.
    let order = match options.greek {
        None => 0,
        Some(GreekOrder::All) => 2 * n_max,
        Some(GreekOrder::Order(l)) => l,
    };
    let mut uppers = std::collections::BTreeSet::new();
    if options.greek.is_some() {
        for (part, p1) in parts.iter().zip(&firsts) {
            let m = part.m[index];
            if let Support::Point(_) = part.support {
                uppers.insert(bucket(p1.n_terms));
                continue;
            }
            for &(a, b) in &p1.panels {
                uppers.insert(bucket(panel_terms(a, b, lambda, m, &mie_opts)?));
            }
        }
    }
    let mut bands = Vec::with_capacity(uppers.len());
    let mut len = SCALARS + 4 * options.mu.len();
    for upper in uppers {
        let projected = order.min(2 * upper);
        let projector = GreekProjector::<f64>::new(projected, upper + (projected + 1).div_ceil(2))?;
        let n_q = projector.nodes().len();
        bands.push(Band {
            upper,
            start: len,
            projector,
        });
        len += 4 * n_q;
    }
    let layout = Layout {
        mu: &options.mu,
        bands,
        len,
    };
    // Pass 2 and the mixture.
    let mut v = vec![0.0; len];
    let mut e = vec![0.0; len];
    let mut d = vec![vec![0.0; len]; n_cols];
    let mut de = vec![vec![0.0; len]; n_cols];
    let mut averages_of = Vec::with_capacity(parts.len());
    for (part, p1) in parts.iter().zip(&firsts) {
        let m = part.m[index];
        let avg = match part.support {
            Support::Point(r) => monodisperse(part, r, lambda, m, &mie_opts, &layout)?,
            _ => {
                let sums = pass2(part, &p1.panels, lambda, m, &mie_opts, &layout)?;
                debug_assert_eq!(
                    sums.blocks,
                    1 + part.theta.len() + if part.m_derivatives() { 2 } else { 0 }
                );
                let truncation = if matches!(part.support, Support::Unbounded(_)) {
                    0.2 * options.rtol
                } else {
                    0.0
                };
                averages(part, &sums, len, truncation)
            }
        };
        let f = part.w / w_total * part.v / part.v_total;
        for i in 0..len {
            v[i] += f * avg.v[i];
            e[i] += f * avg.e[i];
        }
        for (col, dv, dev) in &avg.d {
            for i in 0..len {
                d[*col][i] += f * dv[i];
                de[*col][i] += f * dev[i];
            }
        }
        averages_of.push(avg);
    }
    // The weight derivatives: ∂f_p/∂w and ∂f_p/∂v of the number fractions.
    for (col, p) in options.derivatives.iter().enumerate() {
        if p.kind != ParamKind::Weight {
            continue;
        }
        for (part, avg) in parts.iter().zip(&averages_of) {
            let f = part.w / w_total * part.v / part.v_total;
            let df = match p.mode {
                None => {
                    let own = if part.component == p.component {
                        part.v / part.v_total
                    } else {
                        0.0
                    };
                    (own - f) / w_total
                }
                Some(j) => {
                    if part.component != p.component {
                        0.0
                    } else {
                        let own = if part.mode == Some(j) { 1.0 } else { 0.0 };
                        part.w / w_total * (own - part.v / part.v_total) / part.v_total
                    }
                }
            };
            for i in 0..len {
                d[col][i] += df * avg.v[i];
                de[col][i] += df.abs() * avg.e[i];
            }
        }
    }
    let lin = |i: usize| Lin {
        v: v[i],
        e: e[i],
        d: d.iter().map(|c| c[i]).collect(),
        de: de.iter().map(|c| c[i]).collect(),
    };
    let greek = options.greek.is_some();
    let outputs = assemble(&lin, greek, order, &layout, &d, &de, &v, &e, options)?;
    Ok(EnsembleResult {
        values: outputs.pick(|l| l.v),
        errors: outputs.pick(|l| l.e),
        derivatives: (0..n_cols).map(|k| outputs.pick(|l| l.d[k])).collect(),
        derivative_errors: (0..n_cols).map(|k| outputs.pick(|l| l.de[k])).collect(),
        params: options.derivatives.clone(),
        converged: firsts.iter().all(|p| p.converged),
        panels: firsts.iter().map(|p| p.panels.len()).sum(),
        ranges: firsts.iter().map(|p| p.range).collect(),
    })
}

/// Every output as a [`Lin`].
struct LinOutputs {
    scalars: [Lin; 11],
    /// The Greek coefficients up to the largest band's order (zero above it), and the
    /// requested order, to which `pick` pads them with zeros.
    greek: Option<([Vec<Lin>; 6], usize)>,
    phase: Option<[Vec<Lin>; 4]>,
}

impl LinOutputs {
    fn pick(&self, f: impl Fn(&Lin) -> f64) -> Outputs {
        let s: Vec<f64> = self.scalars.iter().map(&f).collect();
        let arr = |v: &Vec<Lin>| v.iter().map(&f).collect::<Vec<f64>>();
        let padded = |v: &Vec<Lin>, order: usize| {
            let mut out = arr(v);
            out.resize(order + 1, 0.0);
            out
        };
        Outputs {
            c_ext: s[0],
            c_sca: s[1],
            c_abs: s[2],
            c_back: s[3],
            ssa: s[4],
            g: s[5],
            lidar_ratio: s[6],
            depolarization: s[7],
            effective_radius: s[8],
            mean_area: s[9],
            mean_volume: s[10],
            greek: self.greek.as_ref().map(|(g, order)| GreekCoefficients {
                alpha1: padded(&g[0], *order),
                alpha2: padded(&g[1], *order),
                alpha3: padded(&g[2], *order),
                alpha4: padded(&g[3], *order),
                beta1: padded(&g[4], *order),
                beta2: padded(&g[5], *order),
            }),
            phase: self.phase.as_ref().map(|p| PhaseValues {
                f11: arr(&p[0]),
                f12: arr(&p[1]),
                f33: arr(&p[2]),
                f34: arr(&p[3]),
            }),
        }
    }
}

/// The outputs from the mixed averages.
#[allow(clippy::too_many_arguments)]
fn assemble(
    lin: &dyn Fn(usize) -> Lin,
    greek: bool,
    order: usize,
    layout: &Layout<'_>,
    d: &[Vec<f64>],
    de: &[Vec<f64>],
    v: &[f64],
    e: &[f64],
    options: &EnsembleOptions,
) -> Result<LinOutputs> {
    let n_cols = d.len();
    let (ext, sca, scag, abs, back, r2, r3) = (
        lin(EXT),
        lin(SCA),
        lin(SCAG),
        lin(ABS),
        lin(BACK),
        lin(R2),
        lin(R3),
    );
    // The ratios need non-zero denominators: an ensemble that does not scatter (m = 1) has
    // no albedo, asymmetry parameter or lidar ratio.
    for (what, l) in [("<C_ext>", &ext), ("<C_sca>", &sca), ("<C_back>", &back)] {
        if !(l.v.is_finite() && l.v != 0.0) {
            return Err(Error::InvalidInput(format!(
                "the ensemble's {what} is {}, so its ratios (albedo, asymmetry parameter, lidar \
                 ratio) are undefined (does m = 1?)",
                l.v
            )));
        }
    }
    let scalars = [
        ext.clone(),
        sca.clone(),
        abs,
        back.clone(),
        sca.div(&ext),
        scag.div(&sca),
        ext.div(&back).scale(4.0 * PI),
        Lin::constant(0.0, n_cols),
        r3.div(&r2),
        r2.scale(PI),
        r3.scale(4.0 * PI / 3.0),
    ];
    let normalized = options.normalization == Normalization::Normalized;
    let greek = if !greek {
        None
    } else {
        let arrays = |g: &GreekCoefficients<f64>| -> [Vec<f64>; 6] {
            [
                g.alpha1.clone(),
                g.alpha2.clone(),
                g.alpha3.clone(),
                g.alpha4.clone(),
                g.beta1.clone(),
                g.beta2.clone(),
            ]
        };
        // Sum the bands' coefficients: values and derivatives, and the error bounds.
        // Up to the largest band's order; `pick` pads to `order` with plain zeros.
        let top = layout
            .bands
            .iter()
            .map(|b| order.min(2 * b.upper))
            .max()
            .unwrap_or(0);
        let mut out: [Vec<Lin>; 6] =
            core::array::from_fn(|_| vec![Lin::constant(0.0, n_cols); top + 1]);
        for band in &layout.bands {
            let n_q = band.projector.nodes().len();
            let start = band.start;
            let table = |x: &[f64]| -> [Vec<f64>; 4] {
                core::array::from_fn(|k| x[start + k * n_q..start + (k + 1) * n_q].to_vec())
            };
            let mut values = vec![table(v)];
            values.extend(d.iter().map(|c| table(c)));
            let mut errs = vec![table(e)];
            errs.extend(de.iter().map(|c| table(c)));
            let c = band
                .projector
                .project(&values.iter().map(view).collect::<Vec<_>>())?;
            let ce = band
                .projector
                .project_bounds(&errs.iter().map(view).collect::<Vec<_>>())?;
            let ca: Vec<[Vec<f64>; 6]> = c.iter().map(arrays).collect();
            let cea: Vec<[Vec<f64>; 6]> = ce.iter().map(arrays).collect();
            for (k, arr) in out.iter_mut().enumerate() {
                for (l, lin) in arr.iter_mut().enumerate().take(ca[0][k].len()) {
                    lin.v += ca[0][k][l];
                    lin.e += cea[0][k][l];
                    for col in 0..n_cols {
                        lin.d[col] += ca[1 + col][k][l];
                        lin.de[col] += cea[1 + col][k][l];
                    }
                }
            }
        }
        if normalized {
            let a0 = out[0][0].clone();
            for arr in &mut out {
                for l in arr.iter_mut() {
                    *l = l.div(&a0);
                }
            }
            out[0][0] = Lin::constant(1.0, n_cols);
        }
        Some((out, order))
    };
    let n_mu = layout.mu.len();
    let phase = if n_mu == 0 {
        None
    } else {
        let start = SCALARS;
        Some(core::array::from_fn(|k| {
            (0..n_mu)
                .map(|j| {
                    let l = lin(start + k * n_mu + j);
                    if normalized { l.div(&sca) } else { l }
                })
                .collect()
        }))
    };
    Ok(LinOutputs {
        scalars,
        greek,
        phase,
    })
}
