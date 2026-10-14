//! The ensemble engine: per wavelength, the size integration of each component (pass 1:
//! adaptive panels on the cheap quantities; pass 2: every output at the final nodes), the
//! mixture, and the outputs with their derivatives and error estimates.

use core::f64::consts::PI;

use super::matrix::{Mat, matmul};
use rayon::prelude::*;

use super::distribution::{SizeDistribution, Support};
use super::fast::{BandProjection, BandTables, PanelProducts};
use super::lin::Lin;
use super::moments::{Moments, moments};
use super::particle::{Optics, Solver};
use super::range::{self, MIE_POWERS, MOMENT_POWERS, STEP};
use super::{
    Component, EnsembleOptions, EnsembleResult, GreekOrder, Outputs, ParamKind, Particle,
    PhaseValues, kronrod,
};
use crate::mie::{
    Coefficients, EfficiencyJacobian, GreekCoefficients, GreekProjector, MAX_SIZE_PARAMETER,
    MieOptions, Normalization, PhaseBuffers,
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
/// derivatives along the requested directions of the particle (`M_GROUP` per direction: C_ext,
/// C_sca, C_sca·g, C_back, C_abs), and refines on all of them but C_back and its derivatives unless
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
    particle: &'a Particle,
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
    /// The result column of each of the particle's derivative directions, if requested.
    dir_cols: Vec<Option<usize>>,
}

impl Part<'_> {
    /// Whether any derivative along the particle's directions is requested.
    fn dir_derivatives(&self) -> bool {
        self.dir_cols.iter().any(Option::is_some)
    }
}

/// The derivative of the efficiencies along direction `k`: Re m or Im m of group k/2.
fn direction(jacs: &[EfficiencyJacobian<f64>], k: usize) -> &crate::mie::Efficiencies<f64> {
    let j = &jacs[k / 2];
    if k % 2 == 0 { &j.m_re } else { &j.m_im }
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
            let mut dir_cols = vec![None; c.particle.directions()];
            for (col, p) in options.derivatives.iter().enumerate() {
                if p.component != ci {
                    continue;
                }
                match p.kind {
                    ParamKind::Distribution(name) if p.mode == mode => {
                        let k = names.iter().position(|n| *n == name).expect("validated");
                        theta.push((col, k));
                    }
                    ParamKind::RefrReal => dir_cols[0] = Some(col),
                    ParamKind::RefrImag => dir_cols[1] = Some(col),
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
                particle: &c.particle,
                w: c.weight,
                v,
                v_total,
                support,
                cutoffs,
                moments,
                theta,
                dir_cols,
            });
        }
    }
    Ok(out)
}

/// Parts mixed into one result, and the sum of their components' weights.
struct Unit {
    parts: Vec<usize>,
    w_total: f64,
}

/// Runs every wavelength (validated inputs): each part on its own panels, every part mixed
/// into one result.
pub(super) fn run(
    components: &[Component],
    wavelengths: &[f64],
    options: &EnsembleOptions,
) -> Result<Vec<EnsembleResult>> {
    let parts = parts(components, options, options.rtol)?;
    let w_total: f64 = components.iter().map(|c| c.weight).sum();
    let groups: Vec<Vec<usize>> = (0..parts.len()).map(|p| vec![p]).collect();
    let units = [Unit {
        parts: (0..parts.len()).collect(),
        w_total,
    }];
    let results: Vec<Result<EnsembleResult>> = (0..wavelengths.len())
        .into_par_iter()
        .map(|i| {
            wavelength(&parts, &groups, &units, wavelengths[i], i, options).map(|mut r| r.remove(0))
        })
        .collect();
    results.into_iter().collect()
}

/// Runs every wavelength of a batch (validated inputs, one component per distribution): the
/// parts of every distribution with a range on shared panels, a result per distribution.
/// Results by distribution, then wavelength.
pub(super) fn run_batch(
    components: &[Component],
    wavelengths: &[f64],
    options: &EnsembleOptions,
) -> Result<Vec<Vec<EnsembleResult>>> {
    let parts = parts(components, options, options.rtol)?;
    let (points, shared): (Vec<usize>, Vec<usize>) =
        (0..parts.len()).partition(|&p| matches!(parts[p].support, Support::Point(_)));
    let mut groups: Vec<Vec<usize>> = points.into_iter().map(|p| vec![p]).collect();
    if !shared.is_empty() {
        groups.push(shared);
    }
    let units: Vec<Unit> = components
        .iter()
        .enumerate()
        .map(|(c, comp)| Unit {
            parts: (0..parts.len())
                .filter(|&p| parts[p].component == c)
                .collect(),
            w_total: comp.weight,
        })
        .collect();
    let results: Vec<Result<Vec<EnsembleResult>>> = (0..wavelengths.len())
        .into_par_iter()
        .map(|i| wavelength(&parts, &groups, &units, wavelengths[i], i, options))
        .collect();
    let mut by_wavelength = Vec::with_capacity(wavelengths.len());
    for r in results {
        by_wavelength.push(r?);
    }
    let mut out: Vec<Vec<EnsembleResult>> = (0..components.len())
        .map(|_| Vec::with_capacity(wavelengths.len()))
        .collect();
    for per_unit in by_wavelength {
        for (u, r) in per_unit.into_iter().enumerate() {
            out[u].push(r);
        }
    }
    Ok(out)
}

/// The size parameter of radius `r` at `lambda`.
fn size(r: f64, lambda: f64) -> f64 {
    2.0 * PI * r / lambda
}

/// The initial panels of [t_lo, t_hi] (t = ln r): starting at t, at most `max_step(t)` wide
/// and at most `MAX_DX` in size parameter, with the radii of `breaks` (inside the range) as
/// edges; if that would take more than `budget` panels, equal panels between the edges, at
/// least one per interval, about `budget` in all.
fn initial_panels(
    t_lo: f64,
    t_hi: f64,
    lambda: f64,
    breaks: &[f64],
    max_step: &dyn Fn(f64) -> f64,
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
            let step = max_step(a).min((1.0 + MAX_DX / x).ln());
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
    /// Kronrod sums of u·r·|q|, the scales of the derivatives along the particle's directions.
    absv: Vec<f64>,
    /// The largest number of terms at the nodes.
    n_terms: usize,
    /// The integrands per unit ln r of every quantity (and θ block) at the first and the last
    /// node.
    edge: [Vec<f64>; 2],
}

/// The pass-1 quantities at radius r: 1, C_ext, C_sca, C_sca·g, C_back, then the Greek
/// coefficients up to `refine_order` (πr²·α), then the derivatives of C_ext, C_sca, C_sca·g,
/// C_back and C_abs along each of the particle's directions that `dirs` asks for; and the
/// number of terms.
#[allow(clippy::too_many_arguments)]
fn quantities1(
    solver: &mut Solver,
    r: f64,
    lambda: f64,
    optics: Optics,
    opts: &MieOptions,
    refine_order: Option<usize>,
    dirs: &[bool],
) -> Result<(Vec<f64>, usize)> {
    let x = size(r, lambda);
    let s = solver
        .solve(optics, x, opts, dirs.iter().any(|&b| b))
        .map_err(|e| context(e, lambda, r))?;
    let c = s.c;
    let q = c.efficiencies();
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
        let g = c
            .greek(Some(order), Normalization::Unnormalized)
            .map_err(|e| context(e, lambda, r))?;
        for a in [
            &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
        ] {
            out.extend(a.iter().map(|v| area * v));
        }
    }
    let jacs: Vec<_> = s
        .groups
        .iter()
        .map(|g| c.efficiencies_jacobian(g))
        .collect();
    for (k, _) in dirs.iter().enumerate().filter(|&(_, &want)| want) {
        let d = direction(&jacs, k);
        out.extend([
            area * d.q_ext,
            area * d.q_sca,
            area * (d.q_sca * q.g + q.q_sca * d.g),
            area * d.q_back,
            area * d.q_abs,
        ]);
    }
    Ok((out, c.n_terms()))
}

/// The number of pass-1 quantities: the base ones, the Greek coefficients, and `M_GROUP`
/// derivatives per requested direction.
fn n_refined(refine_order: Option<usize>, part: &Part<'_>) -> usize {
    let md = part.dir_cols.iter().filter(|c| c.is_some()).count();
    N_BASE + refine_order.map_or(0, |l| 6 * (l + 1)) + M_GROUP * md
}

/// Evaluates one panel in pass 1 for every part of a group: each node's Mie solution once,
/// its quantities weighted by each part's density.
#[allow(clippy::too_many_arguments)]
fn panel1(
    solver: &mut Solver,
    group: &[&Part<'_>],
    (a, b): (f64, f64),
    lambda: f64,
    optics: Optics,
    opts: &MieOptions,
    refine_order: Option<usize>,
    n_refined: usize,
) -> Result<Vec<Sums1>> {
    let nodes = kronrod::panel(a, b);
    let first = group.first().expect("a part");
    let dirs: Vec<bool> = first.dir_cols.iter().map(Option::is_some).collect();
    let mut n_max = 0;
    let mut qs = Vec::with_capacity(nodes.len());
    for &(t, _, _) in &nodes {
        let (q, n) = quantities1(solver, t.exp(), lambda, optics, opts, refine_order, &dirs)?;
        n_max = n_max.max(n);
        qs.push(q);
    }
    let mut out = Vec::with_capacity(group.len());
    for part in group {
        let d = part.dist.prepare();
        let n_grad = part.dist.parameter_names().len();
        let nt = part.theta.len();
        let mut grad = vec![0.0; n_grad];
        let mut k = vec![0.0; (1 + nt) * n_refined];
        let mut g = vec![0.0; (1 + nt) * n_refined];
        let mut abs = vec![0.0; nt * n_refined];
        let mut absv = vec![0.0; n_refined];
        let mut edge = [
            vec![0.0; (1 + nt) * n_refined],
            vec![0.0; (1 + nt) * n_refined],
        ];
        for (i, (&(t, wk, wg), q)) in nodes.iter().zip(&qs).enumerate() {
            let r = t.exp();
            let u = d.eval(r, &mut grad);
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
        out.push(Sums1 {
            k,
            err,
            abs,
            absv,
            n_terms: n_max,
            edge,
        });
    }
    Ok(out)
}

/// The outcome of pass 1 for a group of parts: their shared panels, and whether each part
/// converged.
struct Pass1 {
    panels: Vec<(f64, f64)>,
    converged: Vec<bool>,
    n_terms: usize,
    range: (f64, f64),
}

/// The panels of pass 1, with each part's sums once evaluated.
type Panels1 = Vec<((f64, f64), Option<Vec<Sums1>>)>;

/// Evaluates the panels without sums, in parallel and in order.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    panels: &mut Panels1,
    group: &[&Part<'_>],
    lambda: f64,
    optics: Optics,
    opts: &MieOptions,
    refine_order: Option<usize>,
    n_refined: usize,
) -> Result<()> {
    // Every panel, then the first error in order: an error must not depend on the threads.
    let results: Vec<Result<()>> = panels
        .par_iter_mut()
        .filter(|(_, s)| s.is_none())
        .map_init(Solver::new, |solver, (ab, s)| {
            *s = Some(panel1(
                solver,
                group,
                *ab,
                lambda,
                optics,
                opts,
                refine_order,
                n_refined,
            )?);
            Ok(())
        })
        .collect();
    results.into_iter().collect()
}

/// Where one part of a group stands after a round of pass 1.
struct Status {
    /// The targets of every refined quantity (and θ block), and the errors.
    target: Vec<f64>,
    errors: Vec<f64>,
    met: bool,
    /// Whether every integrand is negligible at the first and at the last node.
    small: [bool; 2],
}

/// The targets, errors and ends of one part from its panel sums.
fn status(
    part: &Part<'_>,
    sums: &[&Sums1],
    options: &EnsembleOptions,
    n_refined: usize,
    n_greek: usize,
    ext: f64,
) -> Status {
    let rtol = options.rtol;
    let nt = part.theta.len();
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
            // The derivatives along the particle's directions can cancel: their absolute
            // integrands.
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
    // ∫∂u/∂θ·∂C/∂m is no output: no targets for the derivatives along the particle's
    // directions in the θ blocks.
    for b in 1..=nt {
        for j in n_greek..n_refined {
            target[b * n_refined + j] = f64::INFINITY;
        }
    }
    if !options.refine_backscatter {
        // C_back and its derivatives (the fourth of each direction's group) are only
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
    // An integrand is not negligible at an end above a tenth of its target per unit ln r, or
    // per `ext / STEP` units for narrow peaks (the cross sections of small absorbing spheres
    // and the derivatives can have heavier tails than the area the cut-offs bound). The
    // derivative integrals cancel (∂u/∂θ changes sign) while their tails carry the extra
    // weight of ∂ ln u/∂θ: their ends are compared with the derivative's own size, floored at
    // 10⁻³ of the absolute integral.
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
    let small = [
        small(&sums.first().expect("a panel").edge[0]),
        small(&sums.last().expect("a panel").edge[1]),
    ];
    Status {
        target,
        errors,
        met,
        small,
    }
}

/// Pass 1: the adaptive panels of a group of parts (sharing a particle) at one wavelength,
/// refined until every part meets its targets.
fn pass1(
    group: &[&Part<'_>],
    lambda: f64,
    optics: Optics,
    options: &EnsembleOptions,
    mie_opts: &MieOptions,
) -> Result<Pass1> {
    let refine_order = options.refine_order;
    let first = group.first().expect("a part");
    let n_refined = n_refined(refine_order, first);
    let n_greek = N_BASE + refine_order.map_or(0, |l| 6 * (l + 1));
    // Panel edges where a part's density has a kink or a jump: a table's radii, and the ends
    // of a finite support (inside the union range of a group).
    let mut breaks: Vec<f64> = group
        .iter()
        .flat_map(|p| {
            let ends = match p.support {
                Support::Finite(lo, hi) => vec![lo, hi],
                _ => Vec::new(),
            };
            let table: &[f64] = match p.dist {
                SizeDistribution::Tabulated { radius, .. } => radius,
                _ => &[],
            };
            ends.into_iter().chain(table.iter().copied())
        })
        .collect();
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let lo = group
        .iter()
        .map(|p| p.cutoffs.0)
        .fold(f64::INFINITY, f64::min);
    let hi = group.iter().map(|p| p.cutoffs.1).fold(0.0, f64::max);
    let (mut t_lo, mut t_hi) = (lo.ln(), hi.ln());
    // Before any panel: a range whose largest size parameter no Mie solution accepts.
    let x_hi = size(hi, lambda);
    if x_hi > MAX_SIZE_PARAMETER {
        return Err(context(
            Error::InvalidInput(format!(
                "the integration range reaches size parameter {x_hi:e}, above the accepted \
                 {MAX_SIZE_PARAMETER:e}"
            )),
            lambda,
            hi,
        ));
    }
    let extendable: Vec<bool> = group
        .iter()
        .map(|p| matches!(p.support, Support::Unbounded(_)))
        .collect();
    let densities: Vec<_> = group.iter().map(|p| p.dist.prepare()).collect();
    // Each part's largest step applies within its own range only, so that a narrow member of
    // a batch does not set the step of the whole range.
    let steps: Vec<(f64, f64, f64)> = group
        .iter()
        .zip(&densities)
        .map(|(p, d)| (p.cutoffs.0.ln(), p.cutoffs.1.ln(), d.max_step(STEP)))
        .collect();
    let max_step = |t: f64| {
        steps
            .iter()
            .filter(|&&(lo, hi, _)| lo <= t && t < hi)
            .map(|&(_, _, h)| h)
            .fold(STEP, f64::min)
    };
    // Each part's extensions of the range: `STEP`, or its peak's width for narrower peaks,
    // whose tails fall off over that width.
    let exts: Vec<f64> = densities.iter().map(|d| STEP.min(d.log_width)).collect();
    let mut panels: Panels1 =
        initial_panels(t_lo, t_hi, lambda, &breaks, &max_step, options.max_panels)
            .into_iter()
            .map(|p| (p, None))
            .collect();
    let mut extensions = [0usize; 2];
    loop {
        evaluate(
            &mut panels,
            group,
            lambda,
            optics,
            mie_opts,
            refine_order,
            n_refined,
        )?;
        let n = panels.len();
        let statuses: Vec<Status> = group
            .iter()
            .enumerate()
            .map(|(p, part)| {
                let sums: Vec<&Sums1> = panels
                    .iter()
                    .map(|(_, s)| &s.as_ref().expect("evaluated")[p])
                    .collect();
                status(part, &sums, options, n_refined, n_greek, exts[p])
            })
            .collect();
        // Converged: the targets met, and the ends negligible unless the support ends there.
        let converged = || -> Vec<bool> {
            statuses
                .iter()
                .zip(&extendable)
                .map(|(s, &e)| s.met && (!e || (s.small[0] && s.small[1])))
                .collect()
        };
        if statuses.iter().all(|s| s.met) {
            // Extend the range at an end where an integrand of any part is not negligible.
            // At each end, the parts that want an extension set its width and its steps.
            let wanting = |end: usize| -> Option<(f64, f64)> {
                if extensions[end] >= MAX_EXTENSIONS {
                    return None;
                }
                statuses
                    .iter()
                    .zip(&extendable)
                    .enumerate()
                    .filter(|&(_, (s, &e))| e && !s.small[end])
                    .map(|(p, _)| (exts[p], steps[p].2))
                    .reduce(|a, b| (a.0.min(b.0), a.1.min(b.1)))
            };
            let wants = [wanting(0), wanting(1)];
            let mut extended = false;
            if let Some((ext, step)) = wants[0] {
                let fresh =
                    initial_panels(t_lo - ext, t_lo, lambda, &[], &|_| step, options.max_panels);
                if panels.len() + fresh.len() > options.max_panels {
                    return Ok(finish(panels, converged(), (t_lo, t_hi)));
                }
                t_lo -= ext;
                extensions[0] += 1;
                let mut new: Panels1 = fresh.into_iter().map(|p| (p, None)).collect();
                new.append(&mut panels);
                panels = new;
                extended = true;
            }
            if let Some((ext, step)) = wants[1] {
                let fresh =
                    initial_panels(t_hi, t_hi + ext, lambda, &[], &|_| step, options.max_panels);
                if panels.len() + fresh.len() > options.max_panels {
                    return Ok(finish(panels, converged(), (t_lo, t_hi)));
                }
                t_hi += ext;
                extensions[1] += 1;
                panels.extend(fresh.into_iter().map(|p| (p, None)));
                extended = true;
            }
            if !extended {
                // Converged unless an end still wants extending after `MAX_EXTENSIONS`.
                return Ok(finish(panels, converged(), (t_lo, t_hi)));
            }
            continue;
        }
        // Split every panel above its share of a target of any part, worst first, within the
        // budget.
        let mut candidates: Vec<(f64, usize)> = panels
            .iter()
            .enumerate()
            .filter(|&(i, _)| panels[i].0.1 - panels[i].0.0 > MIN_WIDTH)
            .filter_map(|(i, (_, sums))| {
                let sums = sums.as_ref().expect("evaluated");
                let badness = sums
                    .iter()
                    .zip(&statuses)
                    .flat_map(|(s, st)| {
                        s.err
                            .iter()
                            .zip(&st.target)
                            .zip(&st.errors)
                            // Only the quantities still above their targets.
                            .filter(|&((_, t), &total_err)| total_err > *t)
                            .map(|((e, t), _)| e * n as f64 / t.max(f64::MIN_POSITIVE))
                    })
                    .fold(0.0, f64::max);
                (badness > 1.0).then_some((badness, i))
            })
            .collect();
        let budget = options.max_panels.saturating_sub(n);
        if candidates.is_empty() || budget == 0 {
            return Ok(finish(panels, converged(), (t_lo, t_hi)));
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

fn finish(panels: Panels1, converged: Vec<bool>, t: (f64, f64)) -> Pass1 {
    let n_terms = panels
        .iter()
        .filter_map(|(_, s)| s.as_ref().and_then(|s| s.first()).map(|s| s.n_terms))
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
fn panel_terms(a: f64, b: f64, lambda: f64, optics: Optics, opts: &MieOptions) -> Result<usize> {
    let mut most = 0;
    for &(t, _, _) in &kronrod::panel(a, b) {
        let r = t.exp();
        most = most.max(
            optics
                .n_terms(size(r, lambda), opts)
                .map_err(|e| context(e, lambda, r))?,
        );
    }
    Ok(most)
}

/// One band: its largest number of terms, the start of its tables in the flat layout, its
/// projector, the πₙ, τₙ tables at the projector's nodes, and the projection's matrices.
struct Band {
    upper: usize,
    start: usize,
    projector: GreekProjector<f64>,
    tables: BandTables,
    projection: BandProjection,
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
/// band, whose (start, length) it returns), their derivatives along each of the particle's
/// directions into `dm` and with respect to r into `dr`, if given. With `products` (the panel's
/// products and the node's index in them), the band's tables are left to the products: the
/// node's coefficients go into them instead.
#[allow(clippy::too_many_arguments)]
fn quantities2(
    solver: &mut Solver,
    r: f64,
    lambda: f64,
    optics: Optics,
    opts: &MieOptions,
    layout: &Layout<'_>,
    g: &mut [f64],
    mut dm: Option<&mut [&mut [f64]]>,
    dr: Option<&mut [f64]>,
    band_terms: usize,
    products: Option<(&mut PanelProducts, usize)>,
) -> Result<Option<(usize, usize)>> {
    let x = size(r, lambda);
    let derivatives = dm.is_some() || dr.is_some();
    let s = solver
        .solve(optics, x, opts, derivatives)
        .map_err(|e| context(e, lambda, r))?;
    let c = s.c;
    let area = PI * r * r;
    let q = c.efficiencies();
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
    let n = band_terms.max(c.n_terms());
    let band = layout.band(n).map(|b| (b.start, b.projector.nodes()));
    if !layout.bands.is_empty() && band.is_none() {
        return Err(context(
            Error::InvalidInput(format!("no projector band for {n} terms")),
            lambda,
            r,
        ));
    }
    let mut tables = vec![(SCALARS, layout.mu)];
    match products {
        Some((p, k)) => p.set(k, &c, &s.groups),
        None => tables.extend(band),
    }
    let segment = band.map(|(start, nodes)| (start, 4 * nodes.len()));
    if !derivatives {
        for (start, cosines) in tables {
            let n = cosines.len();
            if n == 0 {
                continue;
            }
            phase_into(&c, cosines, &mut g[start..start + 4 * n])
                .map_err(|e| context(e, lambda, r))?;
            g[start..start + 4 * n].iter_mut().for_each(|v| *v *= area);
        }
        return Ok(segment);
    }
    let jacs: Vec<_> = s
        .groups
        .iter()
        .map(|gr| c.efficiencies_jacobian(gr))
        .collect();
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
    if let Some(dm) = dm.as_deref_mut() {
        for (dir, d) in dm.iter_mut().enumerate() {
            d[..SCALARS].copy_from_slice(&scalar_d(direction(&jacs, dir)));
        }
    }
    let mut dr = dr;
    if let Some(dr) = dr.as_deref_mut() {
        // ∂(πr²Q)/∂r = 2πr Q + πr² (2π/λ) ∂Q/∂x.
        let dx = scalar_d(&jacs[0].x);
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
        // Per group: ∂/∂x, ∂/∂(Re m), ∂/∂(Im m).
        let mut d: Vec<[Vec<f64>; 3]> = s
            .groups
            .iter()
            .map(|_| core::array::from_fn(|_| vec![0.0; 4 * n]))
            .collect();
        for (group, [dx, dre_buf, dim_buf]) in s.groups.iter().zip(&mut d) {
            let [v11, v12, v33, v34] = quarter(&mut g[span.clone()]);
            let [x11, x12, x33, x34] = quarter(dx);
            let [r11, r12, r33, r34] = quarter(dre_buf);
            let [i11, i12, i33, i34] = quarter(dim_buf);
            c.phase_matrix_with_derivatives(
                cosines,
                group,
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
                dr[o] = 2.0 * PI * r * g[o] + area * k * d[0][0][j];
            }
        }
        if let Some(dm) = dm.as_deref_mut() {
            for (dir, out) in dm.iter_mut().enumerate() {
                let buf = &d[dir / 2][1 + dir % 2];
                for (j, o) in span.clone().enumerate() {
                    out[o] = area * buf[j];
                }
            }
        }
        g[span].iter_mut().for_each(|v| *v *= area);
    }
    Ok(segment)
}

/// Splits a table of 4·n values into its four elements.
fn quarter(t: &mut [f64]) -> [&mut [f64]; 4] {
    let n = t.len() / 4;
    let (a, rest) = t.split_at_mut(n);
    let (b, rest) = rest.split_at_mut(n);
    let (c, d) = rest.split_at_mut(n);
    [a, b, c, d]
}

/// The unnormalized phase matrix of `c` at `cosines` into the 4·n values of `out`.
fn phase_into(c: &Coefficients<'_, f64>, cosines: &[f64], out: &mut [f64]) -> Result<()> {
    let [f11, f12, f33, f34] = quarter(out);
    c.phase_matrix(
        cosines,
        Normalization::Unnormalized,
        PhaseBuffers { f11, f12, f33, f34 },
    )
}

/// Pass-2 integrals of one part: blocks of the flat layout (values; each requested θ; every
/// direction of the particle if any is requested), Kronrod sums and Σ over panels of
/// |Kronrod − Gauss|.
struct Sums2 {
    s: Vec<f64>,
    err: Vec<f64>,
    blocks: usize,
}

/// The number of pass-2 blocks of a part.
fn blocks(part: &Part<'_>) -> usize {
    let dirs = if part.dir_derivatives() {
        part.dir_cols.len()
    } else {
        0
    };
    1 + part.theta.len() + dirs
}

/// At most this many `CHUNK`s of panels have their sums held at once before they are added up
/// in order, and fewer for large groups: about `WINDOW_BYTES` of sums. The window depends on
/// the inputs only, never on the thread count.
const WINDOW: usize = 32;
const WINDOW_BYTES: usize = 64 << 20;

/// The weights of a group's blocks at a panel's nodes: a row per (part, block) for the
/// values (the values block, weight u·r, and each θ block, ∂u/∂θ·r) and a row per part with
/// derivatives along the particle's directions (weight u·r, applied to each direction's
/// derivatives), Kronrod rows first, then Gauss rows. A panel's sums for every part and block are then
/// matrix products of these weights with the nodes' values.
struct Rows {
    /// The first values row of each part.
    first: Vec<usize>,
    /// The derivative row of each part with derivatives along the particle's directions.
    m_row: Vec<Option<usize>>,
    values: usize,
    m: usize,
}

impl Rows {
    fn new(group: &[&Part<'_>]) -> Self {
        let (mut first, mut m_row) = (Vec::new(), Vec::new());
        let (mut values, mut m) = (0, 0);
        for part in group {
            first.push(values);
            values += 1 + part.theta.len();
            m_row.push(part.dir_derivatives().then(|| {
                m += 1;
                m - 1
            }));
        }
        Self {
            first,
            m_row,
            values,
            m,
        }
    }
}

/// Pass 2 of a group of parts (sharing a particle) over their shared final panels: each node's
/// solution and tables once; each part's panel sums as matrix products of its weights with the
/// nodes' values.
fn pass2(
    group: &[&Part<'_>],
    panels: &[(f64, f64)],
    lambda: f64,
    optics: Optics,
    opts: &MieOptions,
    layout: &Layout<'_>,
) -> Result<Vec<Sums2>> {
    let len = layout.len;
    let common = layout.common();
    let blocks: Vec<usize> = group.iter().map(|p| blocks(p)).collect();
    let derivatives = group.iter().any(|p| p.dir_derivatives());
    // Every direction of the particle when any is requested.
    let n_dirs = if derivatives {
        group[0].dir_cols.len()
    } else {
        0
    };
    let rows = Rows::new(group);
    let n_nodes = kronrod::panel(0.0, 1.0).len();
    // Per-thread scratch. A panel touches the common part of the layout and its band's
    // tables: `stride` values, the common part first.
    struct Scratch {
        solver: Solver,
        grads: Vec<Vec<f64>>,
        g: Vec<f64>,
        dmv: Vec<f64>,
        products: Option<PanelProducts>,
        /// The nodes' values (nodes × stride), and their derivatives along each direction.
        v: Mat,
        dv: Vec<Mat>,
        /// The weights (2·rows × nodes) and the products (2·rows × stride).
        w: Mat,
        wm: Mat,
        pv: Mat,
        pm: Vec<Mat>,
    }
    /// A chunk's sums and errors on one range [lo, hi) of the layout, every block.
    type Segment = ((usize, usize), Vec<f64>, Vec<f64>);
    let mut out: Vec<Sums2> = blocks
        .iter()
        .map(|&b| Sums2 {
            s: vec![0.0; b * len],
            err: vec![0.0; b * len],
            blocks: b,
        })
        .collect();
    let gemm = |dst: &mut Mat, w: &Mat, v: &Mat| matmul(dst, w, v, w.ncols());
    // A chunk's sums: every part's blocks on the common part and at most a band per panel,
    // with their errors.
    let widest = layout
        .bands
        .iter()
        .map(|b| 4 * b.projector.nodes().len())
        .max()
        .unwrap_or(0);
    let per_chunk = blocks.iter().sum::<usize>() * (common + CHUNK * widest) * 2 * 8;
    let window = (WINDOW_BYTES / per_chunk.max(1)).clamp(1, WINDOW);
    for window in panels.chunks(CHUNK * window) {
        let chunks: Vec<Result<Vec<Vec<Segment>>>> = window
            .par_chunks(CHUNK)
            .map_init(
                || Scratch {
                    solver: Solver::new(),
                    grads: group
                        .iter()
                        .map(|p| vec![0.0; p.dist.parameter_names().len()])
                        .collect(),
                    g: vec![0.0; len],
                    dmv: vec![0.0; n_dirs * len],
                    products: None,
                    v: Mat::zeros(0, 0),
                    dv: (0..n_dirs).map(|_| Mat::zeros(0, 0)).collect(),
                    w: Mat::zeros(2 * rows.values, n_nodes),
                    wm: Mat::zeros(2 * rows.m, n_nodes),
                    pv: Mat::zeros(0, 0),
                    pm: (0..n_dirs).map(|_| Mat::zeros(0, 0)).collect(),
                },
                |sc, chunk| {
                    let densities: Vec<_> = group.iter().map(|p| p.dist.prepare()).collect();
                    let mut segments: Vec<Vec<Segment>> = vec![Vec::new(); group.len()];
                    for &(a, b) in chunk {
                        let band_terms = if layout.bands.is_empty() {
                            0
                        } else {
                            panel_terms(a, b, lambda, optics, opts)?
                        };
                        let band = if layout.bands.is_empty() {
                            None
                        } else {
                            Some(layout.band(band_terms).ok_or_else(|| {
                                Error::InvalidInput(format!(
                                    "no projector band for {band_terms} terms"
                                ))
                            })?)
                        };
                        let nodes = kronrod::panel(a, b);
                        if let Some(band) = band {
                            let fits = sc
                                .products
                                .as_ref()
                                .is_some_and(|p| p.fits(&band.tables, nodes.len(), n_dirs / 2));
                            if !fits {
                                sc.products =
                                    Some(PanelProducts::new(&band.tables, nodes.len(), n_dirs / 2));
                            }
                        }
                        let (lo, hi) = band.map_or((0, 0), |b| {
                            (b.start, b.start + 4 * b.projector.nodes().len())
                        });
                        let stride = common + (hi - lo);
                        if sc.v.nrows() != nodes.len() || sc.v.ncols() != stride {
                            sc.v = Mat::zeros(nodes.len(), stride);
                            for dv in &mut sc.dv {
                                *dv = Mat::zeros(nodes.len(), stride);
                            }
                        }
                        for (k, &(t, wk, wg)) in nodes.iter().enumerate() {
                            let r = t.exp();
                            let mut dm: Vec<&mut [f64]> = sc.dmv.chunks_mut(len).collect();
                            let products = match (band, sc.products.as_mut()) {
                                (Some(_), Some(p)) => Some((p, k)),
                                _ => None,
                            };
                            quantities2(
                                &mut sc.solver,
                                r,
                                lambda,
                                optics,
                                opts,
                                layout,
                                &mut sc.g,
                                derivatives.then_some(&mut dm[..]),
                                None,
                                band_terms,
                                products,
                            )?;
                            for j in 0..common {
                                sc.v[(k, j)] = sc.g[j];
                            }
                            if derivatives {
                                for (c, dv) in sc.dv.iter_mut().enumerate() {
                                    for j in 0..common {
                                        dv[(k, j)] = sc.dmv[c * len + j];
                                    }
                                }
                            }
                            // The weights of every part and block at this node.
                            for (p, part) in group.iter().enumerate() {
                                let u = densities[p].eval(r, &mut sc.grads[p]);
                                let row = rows.first[p];
                                let set = |w: &mut Mat, i: usize, n: usize, x: f64| {
                                    w[(i, k)] = wk * x;
                                    w[(n + i, k)] = wg * x;
                                };
                                set(&mut sc.w, row, rows.values, u * r);
                                for (bi, &(_, gi)) in part.theta.iter().enumerate() {
                                    set(&mut sc.w, row + 1 + bi, rows.values, sc.grads[p][gi] * r);
                                }
                                if let Some(i) = rows.m_row[p] {
                                    set(&mut sc.wm, i, rows.m, u * r);
                                }
                            }
                        }
                        // The band's tables of every node at once.
                        if let (Some(band), Some(prod)) = (band, sc.products.as_mut()) {
                            prod.multiply(&band.tables, band_terms);
                            for (k, &(t, _, _)) in nodes.iter().enumerate() {
                                let r = t.exp();
                                let x = size(r, lambda);
                                let scale = 4.0 / (x * x) * (PI * r * r);
                                let mut dm: Vec<&mut [f64]> =
                                    sc.dmv.chunks_mut(len).map(|d| &mut d[lo..hi]).collect();
                                prod.phase(&band.tables, k, scale, &mut sc.g[lo..hi], &mut dm);
                                for (j, &gv) in sc.g[lo..hi].iter().enumerate() {
                                    sc.v[(k, common + j)] = gv;
                                }
                                if derivatives {
                                    for (c, dv) in sc.dv.iter_mut().enumerate() {
                                        for j in 0..hi - lo {
                                            dv[(k, common + j)] = sc.dmv[c * len + lo + j];
                                        }
                                    }
                                }
                            }
                        }
                        gemm(&mut sc.pv, &sc.w, &sc.v);
                        if rows.m > 0 {
                            for (pm, dv) in sc.pm.iter_mut().zip(&sc.dv) {
                                gemm(pm, &sc.wm, dv);
                            }
                        }
                        // Each part's panel sums into the chunk's segments.
                        for (p, part) in group.iter().enumerate() {
                            let nb = blocks[p];
                            let nt = part.theta.len();
                            // (matrix of products, its Kronrod row, its number of rows).
                            let block_rows = |blk: usize| -> (&Mat, usize, usize) {
                                if blk <= nt {
                                    (&sc.pv, rows.first[p] + blk, rows.values)
                                } else {
                                    let i = rows.m_row[p].expect("m derivatives");
                                    (&sc.pm[blk - nt - 1], i, rows.m)
                                }
                            };
                            for (range, off) in [((0, common), 0), ((lo, hi), common)] {
                                let w = range.1 - range.0;
                                if w == 0 {
                                    continue;
                                }
                                let segs = &mut segments[p];
                                let k = match segs.iter().position(|(rg, _, _)| *rg == range) {
                                    Some(k) => k,
                                    None => {
                                        segs.push((range, vec![0.0; nb * w], vec![0.0; nb * w]));
                                        segs.len() - 1
                                    }
                                };
                                let (_, ssum, serr) = &mut segs[k];
                                for blk in 0..nb {
                                    let (mat, row, n) = block_rows(blk);
                                    for i in 0..w {
                                        let (kv, gv) =
                                            (mat[(row, off + i)], mat[(n + row, off + i)]);
                                        ssum[blk * w + i] += kv;
                                        serr[blk * w + i] += (kv - gv).abs();
                                    }
                                }
                            }
                        }
                    }
                    Ok(segments)
                },
            )
            .collect();
        for c in chunks {
            for (p, segs) in c?.into_iter().enumerate() {
                let sums = &mut out[p];
                for ((lo, hi), cs, ce) in segs {
                    let w = hi - lo;
                    for blk in 0..sums.blocks {
                        for i in 0..w {
                            sums.s[blk * len + lo + i] += cs[blk * w + i];
                            sums.err[blk * len + lo + i] += ce[blk * w + i];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
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
    for (c, col) in part.dir_cols.iter().enumerate() {
        if let &Some(col) = col {
            let (sb, eb) = (block(1 + nt + c), eblock(1 + nt + c));
            let dv: Vec<f64> = (0..len).map(|i| sb[i] / z).collect();
            let de: Vec<f64> = (0..len)
                .map(|i| (eb[i] + dv[i].abs() * ez) / z + truncation * dv[i].abs())
                .collect();
            d.push((col, dv, de));
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
    optics: Optics,
    opts: &MieOptions,
    layout: &Layout<'_>,
) -> Result<PartAverages> {
    let len = layout.len;
    let mut solver = Solver::new();
    let mut g = vec![0.0; len];
    let mut dmv = vec![0.0; part.dir_cols.len() * len];
    let mut drv = vec![0.0; len];
    let radius_col = part.theta.first().map(|&(col, _)| col);
    {
        let mut dm: Vec<&mut [f64]> = dmv.chunks_mut(len).collect();
        let dm = part.dir_derivatives().then_some(&mut dm[..]);
        let dr = radius_col.map(|_| drv.as_mut_slice());
        let n = optics
            .n_terms(size(r, lambda), opts)
            .map_err(|e| context(e, lambda, r))?;
        quantities2(
            &mut solver,
            r,
            lambda,
            optics,
            opts,
            layout,
            &mut g,
            dm,
            dr,
            n,
            None,
        )?;
    }
    let zeros = vec![0.0; len];
    let mut d = Vec::new();
    if let Some(col) = radius_col {
        d.push((col, drv, zeros.clone()));
    }
    for (c, col) in part.dir_cols.iter().enumerate() {
        if let Some(col) = col {
            d.push((*col, dmv[c * len..(c + 1) * len].to_vec(), zeros.clone()));
        }
    }
    Ok(PartAverages { v: g, e: zeros, d })
}

/// One wavelength: pass 1 and pass 2 per group of parts sharing panels, then a result per
/// unit.
fn wavelength(
    parts: &[Part<'_>],
    groups: &[Vec<usize>],
    units: &[Unit],
    lambda: f64,
    index: usize,
    options: &EnsembleOptions,
) -> Result<Vec<EnsembleResult>> {
    let mie_opts = MieOptions::new().with_allow_unvalidated(options.allow_unvalidated);
    let n_cols = options.derivatives.len();
    let members = |g: &Vec<usize>| -> Vec<&Part<'_>> { g.iter().map(|&p| &parts[p]).collect() };
    // Pass 1 for every group, and the largest number of terms.
    let mut firsts = Vec::with_capacity(groups.len());
    let mut n_max = 0;
    for group in groups {
        let g = members(group);
        let optics = g[0].particle.optics(index);
        let p1 = match g[0].support {
            Support::Point(r) => {
                let n = optics
                    .n_terms(size(r, lambda), &mie_opts)
                    .map_err(|e| context(e, lambda, r))?;
                Pass1 {
                    panels: Vec::new(),
                    converged: vec![true],
                    n_terms: n,
                    range: (r, r),
                }
            }
            _ => pass1(&g, lambda, optics, options, &mie_opts)?,
        };
        n_max = n_max.max(p1.n_terms);
        firsts.push(p1);
    }
    // Each part's group and its place in it.
    let mut place = vec![(0, 0); parts.len()];
    for (gi, group) in groups.iter().enumerate() {
        for (k, &p) in group.iter().enumerate() {
            place[p] = (gi, k);
        }
    }
    // The projector bands, from the numbers of terms at the final nodes.
    let order = match options.greek {
        None => 0,
        Some(GreekOrder::All) => 2 * n_max,
        Some(GreekOrder::Order(l)) => l,
    };
    let mut uppers = std::collections::BTreeSet::new();
    if options.greek.is_some() {
        for (group, p1) in groups.iter().zip(&firsts) {
            let first = &parts[group[0]];
            let optics = first.particle.optics(index);
            if let Support::Point(_) = first.support {
                uppers.insert(bucket(p1.n_terms));
                continue;
            }
            for &(a, b) in &p1.panels {
                uppers.insert(bucket(panel_terms(a, b, lambda, optics, &mie_opts)?));
            }
        }
    }
    let mut bands = Vec::with_capacity(uppers.len());
    let mut len = SCALARS + 4 * options.mu.len();
    for upper in uppers {
        let projected = order.min(2 * upper);
        let projector = GreekProjector::<f64>::new(projected, upper + (projected + 1).div_ceil(2))?;
        let n_q = projector.nodes().len();
        let tables = BandTables::new(projector.nodes(), upper);
        let projection = BandProjection::new(&projector)?;
        bands.push(Band {
            upper,
            start: len,
            projector,
            tables,
            projection,
        });
        len += 4 * n_q;
    }
    let layout = Layout {
        mu: &options.mu,
        bands,
        len,
    };
    // Pass 2: every part's averages.
    let mut averages_of: Vec<Option<PartAverages>> = (0..parts.len()).map(|_| None).collect();
    for (group, p1) in groups.iter().zip(&firsts) {
        let g = members(group);
        let optics = g[0].particle.optics(index);
        if let Support::Point(r) = g[0].support {
            averages_of[group[0]] =
                Some(monodisperse(g[0], r, lambda, optics, &mie_opts, &layout)?);
            continue;
        }
        let sums = pass2(&g, &p1.panels, lambda, optics, &mie_opts, &layout)?;
        for ((&p, part), sums) in group.iter().zip(&g).zip(&sums) {
            debug_assert_eq!(sums.blocks, blocks(part));
            let truncation = if matches!(part.support, Support::Unbounded(_)) {
                0.2 * options.rtol
            } else {
                0.0
            };
            averages_of[p] = Some(averages(part, sums, len, truncation));
        }
    }
    let averages_of: Vec<PartAverages> = averages_of
        .into_iter()
        .map(|a| a.expect("every part averaged"))
        .collect();
    // Each unit's mixture of its parts: values, errors, derivatives and their errors.
    let mut mixed = Vec::with_capacity(units.len());
    for unit in units {
        let w_total = unit.w_total;
        let mut v = vec![0.0; len];
        let mut e = vec![0.0; len];
        let mut d = vec![vec![0.0; len]; n_cols];
        let mut de = vec![vec![0.0; len]; n_cols];
        for &p in &unit.parts {
            let (part, avg) = (&parts[p], &averages_of[p]);
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
        }
        // The weight derivatives: ∂f_p/∂w and ∂f_p/∂v of the number fractions.
        for (col, p) in options.derivatives.iter().enumerate() {
            if p.kind != ParamKind::Weight {
                continue;
            }
            for &q in &unit.parts {
                let (part, avg) = (&parts[q], &averages_of[q]);
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
        mixed.push((v, e, d, de));
    }
    // The Greek coefficients of every unit, band by band: one projection of all the units'
    // tables shares the generalized spherical functions.
    let greek = if options.greek.is_some() {
        Some(project(&layout, order, &mixed)?)
    } else {
        None
    };
    let mut out = Vec::with_capacity(units.len());
    for (u, (unit, (v, e, d, de))) in units.iter().zip(&mixed).enumerate() {
        let lin = |i: usize| Lin {
            v: v[i],
            e: e[i],
            d: d.iter().map(|c| c[i]).collect(),
            de: de.iter().map(|c| c[i]).collect(),
        };
        let unit_greek = greek.as_ref().map(|g| g[u].clone());
        let outputs = assemble(&lin, unit_greek, order, &layout, n_cols, options)?;
        let mut unit_groups: Vec<usize> = unit.parts.iter().map(|&p| place[p].0).collect();
        unit_groups.sort_unstable();
        unit_groups.dedup();
        out.push(EnsembleResult {
            values: outputs.pick(|l| l.v),
            errors: outputs.pick(|l| l.e),
            derivatives: (0..n_cols).map(|k| outputs.pick(|l| l.d[k])).collect(),
            derivative_errors: (0..n_cols).map(|k| outputs.pick(|l| l.de[k])).collect(),
            params: options.derivatives.clone(),
            converged: unit.parts.iter().all(|&p| {
                let (gi, k) = place[p];
                firsts[gi].converged[k]
            }),
            panels: unit_groups.iter().map(|&gi| firsts[gi].panels.len()).sum(),
            ranges: unit
                .parts
                .iter()
                .map(|&p| firsts[place[p].0].range)
                .collect(),
        });
    }
    Ok(out)
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

/// The Greek coefficients of each unit's mixed tables (values, errors, derivatives and their
/// errors, as `wavelength` mixes them), summed over the bands, up to the largest band's order
/// (`LinOutputs::pick` pads them to `order`): values and derivatives projected, errors as
/// bounds. Each band projects every unit's tables at once.
fn project(layout: &Layout<'_>, order: usize, mixed: &[Mixed]) -> Result<Vec<[Vec<Lin>; 6]>> {
    let n_cols = mixed.first().map_or(0, |m| m.2.len());
    let top = layout
        .bands
        .iter()
        .map(|b| order.min(2 * b.upper))
        .max()
        .unwrap_or(0);
    let mut out: Vec<[Vec<Lin>; 6]> = mixed
        .iter()
        .map(|_| core::array::from_fn(|_| vec![Lin::constant(0.0, n_cols); top + 1]))
        .collect();
    let per = 1 + n_cols;
    let rows = mixed.len() * per;
    for band in &layout.bands {
        let n_q = band.projector.nodes().len();
        let start = band.start;
        // A row per unit and kind (the values, then each derivative), a matrix per element.
        let tables = |pick: &dyn Fn(&Mixed, usize) -> &[f64]| -> [Mat; 4] {
            core::array::from_fn(|k| {
                Mat::from_fn(rows, n_q, |row, j| {
                    pick(&mixed[row / per], row % per)[start + k * n_q + j]
                })
            })
        };
        let c = band.projection.project(&tables(&|m, i| {
            if i == 0 { &m.0 } else { &m.2[i - 1] }
        }));
        let ce = band.projection.bounds(&tables(&|m, i| {
            if i == 0 { &m.1 } else { &m.3[i - 1] }
        }));
        for (u, lins) in out.iter_mut().enumerate() {
            for (k, arr) in lins.iter_mut().enumerate() {
                for (l, lin) in arr.iter_mut().enumerate().take(c[k].ncols()) {
                    lin.v += c[k][(u * per, l)];
                    lin.e += ce[k][(u * per, l)];
                    for col in 0..n_cols {
                        lin.d[col] += c[k][(u * per + 1 + col, l)];
                        lin.de[col] += ce[k][(u * per + 1 + col, l)];
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A unit's mixed values, errors, derivatives and their errors over the flat layout.
type Mixed = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The outputs from the mixed averages and the unit's Greek coefficients (from `project`).
fn assemble(
    lin: &dyn Fn(usize) -> Lin,
    greek: Option<[Vec<Lin>; 6]>,
    order: usize,
    layout: &Layout<'_>,
    n_cols: usize,
    options: &EnsembleOptions,
) -> Result<LinOutputs> {
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
                "the ensemble's {what} is {}, so its ratios (albedo, asymmetry parameter, lidar \\
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
    let greek = greek.map(|mut out| {
        if normalized {
            let a0 = out[0][0].clone();
            for arr in &mut out {
                for l in arr.iter_mut() {
                    *l = l.div(&a0);
                }
            }
            out[0][0] = Lin::constant(1.0, n_cols);
        }
        (out, order)
    });
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
