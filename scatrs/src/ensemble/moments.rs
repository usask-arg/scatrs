//! The distribution moments ⟨r²⟩ and ⟨r³⟩: adaptive integrals of the density alone (no Mie
//! solutions), on a range whose volume tail is also bounded, independent of the wavelength.

use super::distribution::SizeDistribution;
use super::kronrod;
use super::range::STEP;
use crate::{Error, Result};

/// At most this many panels; far more than smooth densities need.
const MAX_PANELS: usize = 100_000;

/// ⟨r²⟩ and ⟨r³⟩ with their error estimates, and their derivatives (one per requested
/// distribution parameter) with error estimates; and the normalization Z = ∫u dr of the
/// unnormalized density over the whole range (number tails included), with its error estimate
/// and derivatives, which normalizes the Mie averages too.
#[derive(Clone, Debug)]
pub(crate) struct Moments {
    pub(crate) v: [f64; 2],
    pub(crate) e: [f64; 2],
    pub(crate) d: Vec<[f64; 2]>,
    pub(crate) de: Vec<[f64; 2]>,
    pub(crate) z: f64,
    pub(crate) ez: f64,
    pub(crate) dz: Vec<f64>,
    pub(crate) dez: Vec<f64>,
}

/// The Kronrod sums of [1, r², r³] and of their ∂u/∂θ-weighted forms (θ in `theta`, indices
/// into the gradient) on the panel [a, b] of t = ln r, |Kronrod − Gauss| of the same, and the
/// Kronrod sums of the absolute θ-weighted integrands.
fn panel(dist: &SizeDistribution, theta: &[usize], a: f64, b: f64) -> [Vec<f64>; 3] {
    let d = dist.prepare();
    let n = 3 * (1 + theta.len());
    let mut grad = vec![0.0; dist.parameter_names().len()];
    let (mut k, mut g, mut abs) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for &(t, wk, wg) in &kronrod::panel(a, b) {
        let r = t.exp();
        let u = d.eval(r, &mut grad);
        let q = [1.0, r * r, r * r * r];
        for (j, &qj) in q.iter().enumerate() {
            k[j] += wk * u * r * qj;
            g[j] += wg * u * r * qj;
            for (bi, &gi) in theta.iter().enumerate() {
                let o = 3 * (1 + bi) + j;
                let x = grad[gi] * r * qj;
                k[o] += wk * x;
                g[o] += wg * x;
                abs[o] += wk * x.abs();
            }
        }
    }
    let err = k.iter().zip(&g).map(|(x, y)| (x - y).abs()).collect();
    [k, err, abs]
}

/// The moments over [lo, hi] (with `breaks` as panel edges), refined until every sum meets
/// `rtol`; `truncation`·|value| is added to each estimate for the omitted tails.
///
/// # Errors
///
/// [`Error::InvalidInput`] if the density has no finite, positive mass on the range.
pub(crate) fn moments(
    dist: &SizeDistribution,
    theta: &[usize],
    (lo, hi): (f64, f64),
    breaks: &[f64],
    rtol: f64,
    truncation: f64,
) -> Result<Moments> {
    let step = dist.prepare().max_step(STEP);
    let (t_lo, t_hi) = (lo.ln(), hi.ln());
    let mut edges = vec![t_lo];
    edges.extend(
        breaks
            .iter()
            .map(|r| r.ln())
            .filter(|&t| t > t_lo && t < t_hi),
    );
    edges.push(t_hi);
    let mut panels: Vec<(f64, f64)> = Vec::new();
    for w in edges.windows(2) {
        let n = ((w[1] - w[0]) / step).ceil().max(1.0) as usize;
        let h = (w[1] - w[0]) / n as f64;
        panels.extend((0..n).map(|i| (w[0] + i as f64 * h, w[0] + (i + 1) as f64 * h)));
    }
    let n_q = 3 * (1 + theta.len());
    loop {
        let sums: Vec<[Vec<f64>; 3]> = panels
            .iter()
            .map(|&(a, b)| panel(dist, theta, a, b))
            .collect();
        let total = |which: usize, j: usize| sums.iter().map(|s| s[which][j]).sum::<f64>();
        let target: Vec<f64> = (0..n_q)
            .map(|j| {
                if j < 3 {
                    rtol * total(0, j).abs()
                } else {
                    rtol * total(2, j)
                }
            })
            .collect();
        let n = panels.len();
        let met = (0..n_q).all(|j| total(1, j) <= target[j]);
        if met || n >= MAX_PANELS {
            let s: Vec<f64> = (0..n_q).map(|j| total(0, j)).collect();
            let e: Vec<f64> = (0..n_q).map(|j| total(1, j)).collect();
            return averages(&s, &e, theta.len(), truncation);
        }
        let mut next = Vec::with_capacity(2 * n);
        for (p, s) in panels.iter().zip(&sums) {
            let bad = (0..n_q).any(|j| s[1][j] * n as f64 > target[j] && total(1, j) > target[j]);
            if bad && p.1 - p.0 > 1e-10 {
                let mid = 0.5 * (p.0 + p.1);
                next.push((p.0, mid));
                next.push((mid, p.1));
            } else {
                next.push(*p);
            }
        }
        if next.len() == n {
            let s: Vec<f64> = (0..n_q).map(|j| total(0, j)).collect();
            let e: Vec<f64> = (0..n_q).map(|j| total(1, j)).collect();
            return averages(&s, &e, theta.len(), truncation);
        }
        panels = next;
    }
}

/// ⟨r^k⟩ = S_k/S₀ with the quotient rule and first-order error propagation.
fn averages(s: &[f64], e: &[f64], nt: usize, truncation: f64) -> Result<Moments> {
    let (z, ez) = (s[0], e[0]);
    if !(z.is_finite() && z > 0.0) {
        return Err(Error::InvalidInput(format!(
            "the size distribution has no finite positive mass on its range (got {z:e})"
        )));
    }
    let v = [s[1] / z, s[2] / z];
    let ev = [
        (e[1] + v[0].abs() * ez) / z + truncation * v[0].abs(),
        (e[2] + v[1].abs() * ez) / z + truncation * v[1].abs(),
    ];
    let mut d = Vec::with_capacity(nt);
    let mut de = Vec::with_capacity(nt);
    for b in 0..nt {
        let o = 3 * (1 + b);
        let dz = s[o];
        let dv = [(s[o + 1] - v[0] * dz) / z, (s[o + 2] - v[1] * dz) / z];
        let dev = core::array::from_fn(|k| {
            (e[o + 1 + k] + v[k].abs() * e[o] + dv[k].abs() * ez + ev[k] * dz.abs()) / z
                + truncation * dv[k].abs()
        });
        d.push(dv);
        de.push(dev);
    }
    let dz = (0..nt).map(|b| s[3 * (1 + b)]).collect();
    let dez = (0..nt).map(|b| e[3 * (1 + b)]).collect();
    Ok(Moments {
        v,
        e: ev,
        d,
        de,
        z,
        ez,
        dz,
        dez,
    })
}

#[cfg(test)]
mod tests {
    use super::super::range::{MOMENT_POWERS, cutoffs};
    use super::*;

    #[test]
    fn moments_match_the_closed_forms() {
        for dist in [
            SizeDistribution::LogNormal {
                median_radius: 0.08,
                width: 1.6,
            },
            SizeDistribution::LogNormal {
                median_radius: 1.0,
                width: 1.05,
            },
            SizeDistribution::Gamma {
                effective_radius: 2.0,
                effective_variance: 0.1,
            },
            SizeDistribution::ModifiedGamma {
                alpha: 2.0,
                b: 15.0,
                gamma: 0.5,
            },
            SizeDistribution::PowerLaw {
                exponent: 3.5,
                min_radius: 0.01,
                max_radius: 10.0,
            },
        ] {
            for rtol in [1e-6, 1e-10] {
                let range = match dist {
                    SizeDistribution::PowerLaw {
                        min_radius,
                        max_radius,
                        ..
                    } => (min_radius, max_radius),
                    _ => {
                        let d = dist.prepare();
                        cutoffs(&d, d.reference, rtol / 10.0, MOMENT_POWERS).unwrap()
                    }
                };
                let m = moments(&dist, &[], range, &[], rtol, 0.2 * rtol).unwrap();
                for (k, (&got, &est)) in m.v.iter().zip(&m.e).enumerate() {
                    let want = dist.moment(2 + k as u32).unwrap();
                    let err = (got - want).abs();
                    assert!(
                        err <= rtol * want,
                        "{dist:?} rtol {rtol:e} r^{}: {err:e}",
                        2 + k
                    );
                    assert!(err <= est, "{dist:?} rtol {rtol:e} r^{}: estimate", 2 + k);
                }
            }
        }
    }
}
