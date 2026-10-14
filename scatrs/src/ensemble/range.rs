//! Cut-offs of distributions with unbounded support.

use super::distribution::Density;
use super::kronrod;
use crate::{Error, Result};

/// The step of the march in t = ln r.
pub(crate) const STEP: f64 = 0.5;

/// At most this many steps on each side: e^(400·½) is far beyond any radius of interest.
const MAX_STEPS: usize = 400;

/// The moments whose tails bound the range of the Mie integration: the area (the extinction
/// of large spheres; the cross sections of small spheres fall faster than r², so the area tail
/// bounds them below). The averages are normalized by the number integral of the moment
/// integration, whose range bounds the number tail, so the tiny spheres of a long number tail
/// need no Mie solutions.
pub(crate) const MIE_POWERS: &[i32] = &[2];

/// The moments whose tails bound the range of the moment integration (and of the
/// normalization): the number, the area and the volume.
pub(crate) const MOMENT_POWERS: &[i32] = &[0, 2, 3];

/// The masses ∫ rᵏ u dr, k in `powers`, over [t₀, t₁] in t = ln r (dr = r dt).
fn masses(d: &Density<'_>, t0: f64, t1: f64, powers: &[i32]) -> Vec<f64> {
    let mut out = vec![0.0; powers.len()];
    for &(t, w, _) in &kronrod::panel(t0, t1) {
        let r = t.exp();
        let wu = w * r * d.eval(r, &mut []);
        for (o, &k) in out.iter_mut().zip(powers) {
            *o += wu * r.powi(k);
        }
    }
    out
}

/// The range [r_lo, r_hi] outside which each tail holds at most `eps` of each moment rᵏ, k in
/// `powers`: from `center`, steps of `STEP` in ln r (or half the peak's width, if narrower)
/// outward, until for each moment the last
/// step's mass, with the geometric bound q/(1 − q) on the rest (q the ratio of the last two
/// steps), is below `eps` of the mass found so far.
///
/// # Errors
///
/// [`Error::InvalidInput`] if a side needs more than 400 steps (the distribution is too
/// broad or the density does not decay).
pub(crate) fn cutoffs(
    d: &Density<'_>,
    center: f64,
    eps: f64,
    powers: &[i32],
) -> Result<(f64, f64)> {
    let t0 = center.ln();
    let step = d.max_step(STEP);
    let np = powers.len();
    let mut total = vec![0.0; np];
    let mut ends = [t0, t0];
    let mut last = [vec![f64::INFINITY; np], vec![f64::INFINITY; np]];
    let mut done = [false; 2];
    for i in 0..MAX_STEPS {
        for (side, sign) in [(0usize, -1.0f64), (1, 1.0)] {
            if done[side] {
                continue;
            }
            let a = ends[side];
            let b = a + sign * step;
            let mass = masses(d, a.min(b), a.max(b), powers);
            if mass.iter().any(|m| !m.is_finite()) {
                return Err(Error::InvalidInput(format!(
                    "the size distribution's density is not finite near r = {:e}",
                    b.exp()
                )));
            }
            ends[side] = b;
            let mut small = i >= 1;
            for k in 0..np {
                total[k] += mass[k];
                let q = mass[k] / last[side][k];
                last[side][k] = mass[k];
                // At least two steps, so that q compares two masses of this side.
                small &= mass[k] == 0.0 || (q < 1.0 && mass[k] / (1.0 - q) <= eps * total[k]);
            }
            done[side] = small;
        }
        if done[0] && done[1] {
            return Ok((ends[0].exp(), ends[1].exp()));
        }
    }
    Err(Error::InvalidInput(format!(
        "the size distribution is too broad: its tails do not decay within e^{} of r = {center:e}",
        MAX_STEPS as f64 * step
    )))
}

#[cfg(test)]
mod tests {
    use super::super::SizeDistribution;
    use super::*;

    #[test]
    fn the_tails_beyond_the_cutoffs_are_below_eps() {
        for dist in [
            SizeDistribution::LogNormal {
                median_radius: 0.08,
                width: 1.6,
            },
            SizeDistribution::LogNormal {
                median_radius: 1.0,
                width: 3.0,
            },
            SizeDistribution::Gamma {
                effective_radius: 2.0,
                effective_variance: 0.1,
            },
            SizeDistribution::Gamma {
                effective_radius: 2.0,
                effective_variance: 0.45,
            },
            SizeDistribution::ModifiedGamma {
                alpha: 2.0,
                b: 15.0,
                gamma: 0.5,
            },
        ] {
            let d = dist.prepare();
            for eps in [1e-4, 1e-8, 1e-13] {
                let (lo, hi) = cutoffs(&d, d.reference, eps, MOMENT_POWERS).unwrap();
                // Each moment's mass inside against the whole (on a far wider range).
                let inside = integrate(&d, lo.ln(), hi.ln(), 2000);
                let wide = integrate(&d, lo.ln() - 40.0, hi.ln() + 40.0, 20_000);
                for k in 0..3 {
                    let outside = (wide[k] - inside[k]) / wide[k];
                    assert!(
                        outside <= 2.0 * eps,
                        "{dist:?}, eps {eps:e}, power {}: outside {outside:e}",
                        MOMENT_POWERS[k]
                    );
                }
                assert!(lo < d.reference && d.reference < hi);
            }
        }
    }

    fn integrate(d: &Density<'_>, a: f64, b: f64, n: usize) -> [f64; 3] {
        let h = (b - a) / n as f64;
        let mut out = [0.0; 3];
        for i in 0..n {
            let m = masses(d, a + i as f64 * h, a + (i + 1) as f64 * h, MOMENT_POWERS);
            for k in 0..3 {
                out[k] += m[k];
            }
        }
        out
    }
}
