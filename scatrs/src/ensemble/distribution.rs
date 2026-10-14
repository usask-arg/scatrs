//! Size distributions: number densities n(r), unnormalized, with their parameter gradients.
//!
//! The engine normalizes every distribution numerically over its integration range, so each
//! density here is known only up to a factor (chosen to keep it near 1 at its peak, which
//! avoids overflow and underflow of powers like r^α), and its gradient is the exact partial
//! derivative of that same expression with the factor's reference radius held fixed. The
//! renormalization cancels any parameter-dependent factor, so the derivatives of the
//! normalized outputs are exact.

use crate::{Error, Result};

/// A number size distribution n(r) of spheres (radii in the unit of the wavelengths).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SizeDistribution {
    /// n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: `median_radius` r_g > 0, `width` σ_g > 1.
    LogNormal {
        /// The median radius r_g.
        median_radius: f64,
        /// The geometric standard deviation σ_g.
        width: f64,
    },
    /// Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
    /// `effective_radius` a > 0, `effective_variance` 0 < b < ½.
    Gamma {
        /// The effective radius a.
        effective_radius: f64,
        /// The effective variance b.
        effective_variance: f64,
    },
    /// Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): `alpha` > −1,
    /// `b` > 0, `gamma` > 0.
    ModifiedGamma {
        /// α.
        alpha: f64,
        /// b.
        b: f64,
        /// γ.
        gamma: f64,
    },
    /// n(r) ∝ r^(−α) on [`min_radius`, `max_radius`]: `exponent` α (the limits are fixed).
    PowerLaw {
        /// α.
        exponent: f64,
        /// The smallest radius, > 0.
        min_radius: f64,
        /// The largest radius, > `min_radius`.
        max_radius: f64,
    },
    /// All spheres of one `radius` > 0.
    Monodisperse {
        /// The radius.
        radius: f64,
    },
    /// n(r) linear between the points (`radius[i]`, `density[i]`) and 0 outside them: radii
    /// increasing and positive, densities ≥ 0 and not all 0. No parameters.
    Tabulated {
        /// The radii, increasing.
        radius: Vec<f64>,
        /// The number densities at `radius`.
        density: Vec<f64>,
    },
    /// Σⱼ wⱼ nⱼ(r) with each mode normalized: `modes` of (weight wⱼ > 0, distribution).
    MultiModal {
        /// The modes and their number weights.
        modes: Vec<(f64, SizeDistribution)>,
    },
}

/// Where a distribution's density is non-zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Support {
    /// [lo, hi].
    Finite(f64, f64),
    /// (0, ∞), with most of its mass near `center`.
    Unbounded(f64),
    /// A single radius.
    Point(f64),
}

fn invalid(msg: String) -> Error {
    Error::InvalidInput(msg)
}

fn positive(name: &str, v: f64) -> Result<()> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(invalid(format!("{name} must be finite and > 0, got {v}")))
    }
}

impl SizeDistribution {
    /// Checks the parameters.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the first invalid parameter.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::LogNormal {
                median_radius,
                width,
            } => {
                positive("median_radius", *median_radius)?;
                if !(width.is_finite() && *width > 1.0) {
                    return Err(invalid(format!(
                        "the lognormal width must be finite and > 1, got {width}"
                    )));
                }
            }
            Self::Gamma {
                effective_radius,
                effective_variance,
            } => {
                positive("effective_radius", *effective_radius)?;
                if !(*effective_variance > 0.0 && *effective_variance < 0.5) {
                    return Err(invalid(format!(
                        "effective_variance must be in (0, 0.5), got {effective_variance}"
                    )));
                }
            }
            Self::ModifiedGamma { alpha, b, gamma } => {
                if !(alpha.is_finite() && *alpha > -1.0) {
                    return Err(invalid(format!(
                        "alpha must be finite and > -1, got {alpha}"
                    )));
                }
                positive("b", *b)?;
                positive("gamma", *gamma)?;
            }
            Self::PowerLaw {
                exponent,
                min_radius,
                max_radius,
            } => {
                if !exponent.is_finite() {
                    return Err(invalid(format!("exponent must be finite, got {exponent}")));
                }
                positive("min_radius", *min_radius)?;
                positive("max_radius", *max_radius)?;
                if min_radius >= max_radius {
                    return Err(invalid(format!(
                        "min_radius must be < max_radius, got {min_radius} and {max_radius}"
                    )));
                }
            }
            Self::Monodisperse { radius } => positive("radius", *radius)?,
            Self::Tabulated { radius, density } => {
                if radius.len() < 2 || radius.len() != density.len() {
                    return Err(invalid(format!(
                        "a tabulated distribution needs at least 2 points and as many densities \
                         as radii, got {} radii and {} densities",
                        radius.len(),
                        density.len()
                    )));
                }
                for &r in radius {
                    positive("radius", r)?;
                }
                if radius.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(invalid("tabulated radii must increase".into()));
                }
                if density.iter().any(|d| !(d.is_finite() && *d >= 0.0)) {
                    return Err(invalid(
                        "tabulated densities must be finite and >= 0".into(),
                    ));
                }
                if density.iter().all(|&d| d == 0.0) {
                    return Err(invalid("tabulated densities must not all be 0".into()));
                }
            }
            Self::MultiModal { modes } => {
                if modes.is_empty() {
                    return Err(invalid("a multimodal distribution needs a mode".into()));
                }
                for (j, (w, mode)) in modes.iter().enumerate() {
                    positive(&format!("the weight of mode {j}"), *w)?;
                    mode.validate()
                        .map_err(|e| invalid(format!("mode {j}: {e}")))?;
                }
            }
        }
        Ok(())
    }

    /// The names of the differentiable parameters, in the order of the gradients (empty for
    /// `Tabulated`; for `MultiModal`, see [`super::EnsembleParam`]).
    pub fn parameter_names(&self) -> &'static [&'static str] {
        match self {
            Self::LogNormal { .. } => &["median_radius", "width"],
            Self::Gamma { .. } => &["effective_radius", "effective_variance"],
            Self::ModifiedGamma { .. } => &["alpha", "b", "gamma"],
            Self::PowerLaw { .. } => &["exponent"],
            Self::Monodisperse { .. } => &["radius"],
            Self::Tabulated { .. } | Self::MultiModal { .. } => &[],
        }
    }

    /// The parameter values, in the order of [`parameter_names`](Self::parameter_names).
    #[cfg(test)]
    pub(crate) fn parameter_values(&self) -> Vec<f64> {
        match *self {
            Self::LogNormal {
                median_radius,
                width,
            } => vec![median_radius, width],
            Self::Gamma {
                effective_radius,
                effective_variance,
            } => vec![effective_radius, effective_variance],
            Self::ModifiedGamma { alpha, b, gamma } => vec![alpha, b, gamma],
            Self::PowerLaw { exponent, .. } => vec![exponent],
            Self::Monodisperse { radius } => vec![radius],
            Self::Tabulated { .. } | Self::MultiModal { .. } => vec![],
        }
    }

    /// The normalized moment ⟨rᵏ⟩ of the whole distribution, where it has a closed form (not
    /// for `Tabulated`).
    pub fn moment(&self, k: u32) -> Option<f64> {
        let kf = f64::from(k);
        match self {
            Self::LogNormal {
                median_radius,
                width,
            } => {
                let s = width.ln();
                Some(median_radius.powf(kf) * (0.5 * kf * kf * s * s).exp())
            }
            Self::Gamma {
                effective_radius: a,
                effective_variance: b,
            } => {
                let alpha = (1.0 - 3.0 * b) / b;
                Some((1..=k).fold(1.0, |p, j| p * a * b * (alpha + f64::from(j))))
            }
            Self::ModifiedGamma { alpha, b, gamma } => {
                let s = (alpha + 1.0) / gamma;
                let ratio = (ln_gamma(s + kf / gamma) - ln_gamma(s)).exp();
                Some(ratio * b.powf(-kf / gamma))
            }
            Self::PowerLaw {
                exponent,
                min_radius,
                max_radius,
            } => {
                let integral = |p: f64| {
                    if p.abs() < 1e-12 {
                        (max_radius / min_radius).ln()
                    } else {
                        (max_radius.powf(p) - min_radius.powf(p)) / p
                    }
                };
                Some(integral(kf + 1.0 - exponent) / integral(1.0 - exponent))
            }
            Self::Monodisperse { radius } => Some(radius.powf(kf)),
            Self::Tabulated { .. } => None,
            Self::MultiModal { modes } => {
                let total: f64 = modes.iter().map(|(w, _)| w).sum();
                let mut sum = 0.0;
                for (w, mode) in modes {
                    sum += w * mode.moment(k)?;
                }
                Some(sum / total)
            }
        }
    }

    /// Where the density is non-zero (not for `MultiModal`, which the engine splits into
    /// its modes).
    pub(crate) fn support(&self) -> Support {
        match *self {
            Self::LogNormal { median_radius, .. } => Support::Unbounded(median_radius),
            Self::Gamma {
                effective_radius, ..
            } => Support::Unbounded(effective_radius),
            Self::ModifiedGamma { .. } => Support::Unbounded(self.prepare().reference),
            Self::PowerLaw {
                min_radius,
                max_radius,
                ..
            } => Support::Finite(min_radius, max_radius),
            Self::Monodisperse { radius } => Support::Point(radius),
            Self::Tabulated { ref radius, .. } => {
                Support::Finite(radius[0], radius[radius.len() - 1])
            }
            Self::MultiModal { .. } => unreachable!("multimodal distributions are split"),
        }
    }

    /// The density, ready to evaluate.
    pub(crate) fn prepare(&self) -> Density<'_> {
        let reference = match *self {
            Self::LogNormal { median_radius, .. } => median_radius,
            Self::Gamma {
                effective_radius: a,
                effective_variance: b,
            } => {
                let alpha = (1.0 - 3.0 * b) / b;
                if alpha > 0.0 { alpha * a * b } else { a }
            }
            Self::ModifiedGamma { alpha, b, gamma } => {
                (alpha.max(1.0) / (b * gamma)).powf(1.0 / gamma)
            }
            // The density's largest value is at one of the limits.
            Self::PowerLaw {
                exponent,
                min_radius,
                max_radius,
            } => {
                if exponent >= 0.0 {
                    min_radius
                } else {
                    max_radius
                }
            }
            _ => 1.0,
        };
        // The width of the peak in ln r: steps and panels must not skip over a narrow one.
        let log_width = match *self {
            Self::LogNormal { width, .. } => width.ln(),
            Self::Gamma {
                effective_variance: b,
                ..
            } => (b / (1.0 - 2.0 * b)).sqrt(),
            Self::ModifiedGamma { alpha, gamma, .. } => {
                if alpha > 0.0 {
                    1.0 / (alpha * gamma).sqrt()
                } else {
                    1.0 / gamma
                }
            }
            _ => f64::INFINITY,
        };
        Density {
            dist: self,
            reference,
            log_width,
        }
    }
}

/// A distribution's unnormalized density with its reference radius fixed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Density<'a> {
    dist: &'a SizeDistribution,
    /// The radius where the density is scaled to about 1 (and the start of the cut-off
    /// search).
    pub(crate) reference: f64,
    /// The width of the density's peak in ln r (∞ for finite support).
    pub(crate) log_width: f64,
}

impl Density<'_> {
    /// The largest step in ln r that samples the peak: ½ (`range::STEP`), or half the peak's
    /// width.
    pub(crate) fn max_step(&self, step: f64) -> f64 {
        step.min(0.5 * self.log_width)
    }
}

impl Density<'_> {
    /// u(r), and its partial derivatives with respect to the parameters into `grad` (as many
    /// as the parameters).
    pub(crate) fn eval(&self, r: f64, grad: &mut [f64]) -> f64 {
        let r0 = self.reference;
        // ln u and ∂(ln u)/∂θ for the analytic forms.
        let (ln_u, dln): (f64, [f64; 3]) = match *self.dist {
            SizeDistribution::LogNormal {
                median_radius,
                width,
            } => {
                let (mu, s) = (median_radius.ln(), width.ln());
                let z = r.ln() - mu;
                let ln_u = -z * z / (2.0 * s * s) - s.ln() - (r / r0).ln();
                let d_rg = z / (s * s * median_radius);
                let d_width = (z * z / (s * s * s) - 1.0 / s) / width;
                (ln_u, [d_rg, d_width, 0.0])
            }
            SizeDistribution::Gamma {
                effective_radius: a,
                effective_variance: b,
            } => {
                let alpha = (1.0 - 3.0 * b) / b;
                let scale = a * b;
                let l = (r / r0).ln();
                let ln_u = alpha * l - (r - r0) / scale;
                let d_a = (r - r0) * b / (scale * scale);
                let d_b = -l / (b * b) + (r - r0) * a / (scale * scale);
                (ln_u, [d_a, d_b, 0.0])
            }
            SizeDistribution::ModifiedGamma { alpha, b, gamma } => {
                let (rg, r0g) = (r.powf(gamma), r0.powf(gamma));
                let l = (r / r0).ln();
                let ln_u = alpha * l - b * (rg - r0g);
                let d_gamma = -b * (rg * r.ln() - r0g * r0.ln());
                (ln_u, [l, -(rg - r0g), d_gamma])
            }
            SizeDistribution::PowerLaw {
                exponent,
                min_radius,
                max_radius,
            } => {
                if r < min_radius || r > max_radius {
                    grad.iter_mut().for_each(|g| *g = 0.0);
                    return 0.0;
                }
                let l = (r / r0).ln();
                (-exponent * l, [-l, 0.0, 0.0])
            }
            SizeDistribution::Monodisperse { .. } => (0.0, [0.0; 3]),
            SizeDistribution::Tabulated {
                ref radius,
                ref density,
            } => {
                grad.iter_mut().for_each(|g| *g = 0.0);
                return interpolate(radius, density, r);
            }
            SizeDistribution::MultiModal { .. } => {
                unreachable!("multimodal distributions are split")
            }
        };
        let u = ln_u.exp();
        for (g, d) in grad.iter_mut().zip(dln) {
            *g = u * d;
        }
        u
    }
}

/// Linear interpolation through (xs, ys), 0 outside.
fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x < xs[0] || x > xs[xs.len() - 1] {
        return 0.0;
    }
    let i = xs.partition_point(|&v| v <= x).clamp(1, xs.len() - 1);
    let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    ys[i - 1] + t * (ys[i] - ys[i - 1])
}

/// ln Γ(x) for x > 0 (Lanczos, g = 7, n = 9; relative error about 1e-15).
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1 − x) = π/sin(πx).
        let pi = core::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * core::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<SizeDistribution> {
        vec![
            SizeDistribution::LogNormal {
                median_radius: 0.08,
                width: 1.6,
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
        ]
    }

    #[test]
    fn gradients_match_finite_differences() {
        for dist in all() {
            let names = dist.parameter_names();
            let values = dist.parameter_values();
            let d = dist.prepare();
            for r in [0.01, 0.05, 0.3, 1.0, 3.0] {
                let mut grad = vec![0.0; names.len()];
                let u = d.eval(r, &mut grad);
                for (p, name) in names.iter().enumerate() {
                    // Perturb one parameter, keeping the reference radius of `d`.
                    let mut v = values.clone();
                    let h = 1e-6 * v[p].abs();
                    let at = |delta: f64, v: &mut Vec<f64>| {
                        v[p] = values[p] + delta;
                        let other = with_values(&dist, v);
                        let dens = Density {
                            dist: &other,
                            reference: d.reference,
                            log_width: d.log_width,
                        };
                        dens.eval(r, &mut vec![0.0; names.len()])
                    };
                    let fd = (at(h, &mut v) - at(-h, &mut v)) / (2.0 * h);
                    let scale = grad[p].abs().max(1e-6 * u.abs());
                    assert!(
                        (fd - grad[p]).abs() <= 1e-6 * scale.max(u / values[p].abs()),
                        "{dist:?} {name} at r = {r}: {fd} vs {}",
                        grad[p]
                    );
                }
            }
        }
    }

    fn with_values(dist: &SizeDistribution, v: &[f64]) -> SizeDistribution {
        match *dist {
            SizeDistribution::LogNormal { .. } => SizeDistribution::LogNormal {
                median_radius: v[0],
                width: v[1],
            },
            SizeDistribution::Gamma { .. } => SizeDistribution::Gamma {
                effective_radius: v[0],
                effective_variance: v[1],
            },
            SizeDistribution::ModifiedGamma { .. } => SizeDistribution::ModifiedGamma {
                alpha: v[0],
                b: v[1],
                gamma: v[2],
            },
            SizeDistribution::PowerLaw {
                min_radius,
                max_radius,
                ..
            } => SizeDistribution::PowerLaw {
                exponent: v[0],
                min_radius,
                max_radius,
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        for (x, want) in [
            (0.5, 0.572_364_942_924_700_1),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.5, 1.200_973_602_347_074_3),
            (10.0, 12.801_827_480_081_469),
            (0.1, 2.252_712_651_734_206),
        ] {
            assert!(
                (ln_gamma(x) - want).abs() <= 1e-13 * want.abs().max(1.0),
                "x = {x}"
            );
        }
    }

    #[test]
    fn gamma_moments_agree_with_the_modified_gamma_form() {
        // Hansen–Travis gamma is the modified gamma with γ = 1, α = (1 − 3b)/b, rate 1/(ab).
        let (a, b) = (2.0, 0.1);
        let g = SizeDistribution::Gamma {
            effective_radius: a,
            effective_variance: b,
        };
        let m = SizeDistribution::ModifiedGamma {
            alpha: (1.0 - 3.0 * b) / b,
            b: 1.0 / (a * b),
            gamma: 1.0,
        };
        for k in 0..=6 {
            let (x, y) = (g.moment(k).unwrap(), m.moment(k).unwrap());
            assert!((x - y).abs() <= 1e-12 * x, "k = {k}: {x} vs {y}");
        }
        // r_eff = ⟨r³⟩/⟨r²⟩ = a.
        assert!((g.moment(3).unwrap() / g.moment(2).unwrap() - a).abs() < 1e-12);
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let bad = [
            SizeDistribution::LogNormal {
                median_radius: 0.1,
                width: 1.0,
            },
            SizeDistribution::LogNormal {
                median_radius: -0.1,
                width: 2.0,
            },
            SizeDistribution::Gamma {
                effective_radius: 1.0,
                effective_variance: 0.5,
            },
            SizeDistribution::ModifiedGamma {
                alpha: -1.0,
                b: 1.0,
                gamma: 1.0,
            },
            SizeDistribution::PowerLaw {
                exponent: 3.0,
                min_radius: 2.0,
                max_radius: 1.0,
            },
            SizeDistribution::Monodisperse { radius: f64::NAN },
            SizeDistribution::Tabulated {
                radius: vec![1.0, 1.0],
                density: vec![1.0, 1.0],
            },
            SizeDistribution::Tabulated {
                radius: vec![1.0, 2.0],
                density: vec![0.0, 0.0],
            },
            SizeDistribution::MultiModal { modes: vec![] },
            SizeDistribution::MultiModal {
                modes: vec![(0.0, SizeDistribution::Monodisperse { radius: 1.0 })],
            },
        ];
        for d in bad {
            assert!(matches!(d.validate(), Err(Error::InvalidInput(_))), "{d:?}");
        }
    }
}
