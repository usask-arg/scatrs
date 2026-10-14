# Phase 2a: Ensembles of Spheres — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Number-averaged optical properties of sphere ensembles (seven size distributions, external mixtures) at many wavelengths, with error estimates and Jacobians, in Rust and Python.

**Architecture:** `scatrs::ensemble`: distributions as an enum with unnormalized densities and gradients (normalized numerically over the range); cut-offs from the number and area tails; two-pass adaptive Gauss–Kronrod integration in ln r per component and wavelength (pass 1 refines the cheap quantities, pass 2 evaluates everything at the final nodes); exact Greek coefficients with one `GreekProjector` per band of the number of terms; first-order propagation of derivatives and error estimates through the ratios; rayon over wavelengths and panel chunks with fixed reduction order.

**Tech Stack:** Rust 2024 (MSRV 1.85; CI lints with Rust 1.99), rayon 1.10, pyo3/numpy 0.29.

**Spec:** `docs/specs/2026-10-14-phase-2a-ensembles-design.md` (section 12 records the changes made while prototyping).

## Global Constraints

- MSRV 1.85: no let-chains or newer APIs; `cargo +1.99.0 clippy --workspace --all-targets -- -D warnings` must be clean (CI runs 1.99).
- No panics and no silent NaN for any finite input; invalid inputs are `InvalidInput`; sizes outside the Mie validated domain are `OutsideValidatedDomain` unless `allow_unvalidated`.
- Results bitwise identical for any thread count.
- Every quoted figure in `docs/` that appears in more than one place is pinned by `tests/test_docs_consistency.py`.
- Never send personal data in network requests. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## How this plan was made

As in phase 1b, tasks 1 and 2 were prototyped and verified before this plan was written (fmt, clippy 1.99, the workspace tests, MSRV, ruff, pytest, stubtest); their code is the exact diff of their commits. Task 3 (documentation) is written from the requirements below.

## Measured results the plan relies on

| What | Measured |
|---|---|
| Calibration grid (8 distributions, rtol 1e-6 vs 1e-11, backscattering refined) | worst true error/estimate 0.085 (refined outputs), 0.37 (others) |
| Lognormal r_g = 0.08, σ_g = 1.6, m = 1.43 + 1e-8i, λ = 0.5 | 25–49 panels, 3–10 ms |
| Broad lognormal r_g = 1, σ_g = 2, m = 1.33 + 1e-4i | cross sections 0.04/0.11/0.23 s at λ = 1/0.5/0.3; Greek to order 32 1.1/5.9/22.6 s (1553–4096 panels) |
| Same, m = 1.33 (no absorption) | not converged at 4096 panels (ripples), estimate 1e-5 |
| Rust ensemble tests / Python ensemble tests | 2.3 s / 1.2 s |

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `scatrs/src/ensemble/mod.rs` | 1 | public API: `SizeDistribution`, `Component`, `EnsembleOptions`, `GreekOrder`, `EnsembleParam`, `ParamKind`, `Outputs`, `PhaseValues`, `EnsembleResult`, `compute`, `compute_mixture`, `parameters`, `mixture_parameters`; input validation |
| `scatrs/src/ensemble/distribution.rs` | 1 | the distributions, densities and gradients, moments, ln Γ |
| `scatrs/src/ensemble/kronrod.rs` | 1 | QUADPACK's G7–K15 |
| `scatrs/src/ensemble/range.rs` | 1 | cut-offs from moment tails |
| `scatrs/src/ensemble/moments.rs` | 1 | ⟨r²⟩, ⟨r³⟩ without Mie |
| `scatrs/src/ensemble/lin.rs` | 1 | first-order propagation of derivatives and errors |
| `scatrs/src/ensemble/engine.rs` | 1 | passes 1 and 2, bands, mixing, outputs |
| `scatrs/src/mie/greek.rs` | 1 | `GreekProjector::project_bounds` |
| `scatrs/tests/ensemble.rs` | 1 | identities, Jacobians, calibration, determinism, errors |
| `scatrs-py/src/ensemble.rs`, `python/scatrs/ensemble.py`, stubs, `tests/test_ensemble.py` | 2 | Python |
| `docs/…` | 3 | theory, example, accuracy, limitations, API, changelog, follow-ups |

---

### Task 1: The ensemble engine

**Interfaces — Produces:** `scatrs::ensemble::{compute(&SizeDistribution, &[f64], &[Complex<f64>], &EnsembleOptions) -> Result<Vec<EnsembleResult>>, compute_mixture(&[Component], &[f64], &EnsembleOptions) -> Result<Vec<EnsembleResult>>, parameters, mixture_parameters}`; `EnsembleOptions` builder (`with_rtol`, `with_max_panels`, `with_greek`, `with_mu`, `with_normalization`, `with_derivatives`, `with_allow_unvalidated`, `with_refine_order`, `with_refine_backscatter`); `EnsembleParam::name(mixture)`; `GreekProjector::project_bounds(&[PhaseTables]) -> Result<Vec<GreekCoefficients>>`.

- [ ] **Step 1: Apply the diff:**

````diff
diff --git a/Cargo.toml b/Cargo.toml
index a605d1a..a9f5a22 100644
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -12,6 +12,7 @@ repository = "https://github.com/usask-arg/scatrs"
 [workspace.dependencies]
 num-complex = "0.4.6"
 num-traits = "0.2.19"
+rayon = "1.10"
 thiserror = "2"
 
 [profile.release]
diff --git a/scatrs/Cargo.toml b/scatrs/Cargo.toml
index 25ff3f1..7a2e382 100644
--- a/scatrs/Cargo.toml
+++ b/scatrs/Cargo.toml
@@ -16,11 +16,13 @@ exclude = ["/tests/", "*.proptest-regressions"]
 [dependencies]
 num-complex.workspace = true
 num-traits.workspace = true
+rayon.workspace = true
 thiserror.workspace = true
 
 [dev-dependencies]
 criterion = { version = "0.7", default-features = false, features = ["cargo_bench_support"] }
 proptest = "1"
+rayon.workspace = true
 serde = { version = "1", features = ["derive"] }
 # float_roundtrip makes f64 parsing correctly rounded; without it, reference values can be 1 ulp off.
 serde_json = { version = "1", features = ["float_roundtrip"] }
diff --git a/scatrs/src/ensemble/distribution.rs b/scatrs/src/ensemble/distribution.rs
new file mode 100644
index 0000000..ec5c994
--- /dev/null
+++ b/scatrs/src/ensemble/distribution.rs
@@ -0,0 +1,602 @@
+//! Size distributions: number densities n(r), unnormalized, with their parameter gradients.
+//!
+//! The engine normalizes every distribution numerically over its integration range, so each
+//! density here is known only up to a factor (chosen to keep it near 1 at its peak, which
+//! avoids overflow and underflow of powers like r^α), and its gradient is the exact partial
+//! derivative of that same expression with the factor's reference radius held fixed. The
+//! renormalization cancels any parameter-dependent factor, so the derivatives of the
+//! normalized outputs are exact.
+
+use crate::{Error, Result};
+
+/// A number size distribution n(r) of spheres (radii in the unit of the wavelengths).
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub enum SizeDistribution {
+    /// n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: `median_radius` r_g > 0, `width` σ_g > 1.
+    LogNormal {
+        /// The median radius r_g.
+        median_radius: f64,
+        /// The geometric standard deviation σ_g.
+        width: f64,
+    },
+    /// Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
+    /// `effective_radius` a > 0, `effective_variance` 0 < b < ½.
+    Gamma {
+        /// The effective radius a.
+        effective_radius: f64,
+        /// The effective variance b.
+        effective_variance: f64,
+    },
+    /// Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): `alpha` > −1,
+    /// `b` > 0, `gamma` > 0.
+    ModifiedGamma {
+        /// α.
+        alpha: f64,
+        /// b.
+        b: f64,
+        /// γ.
+        gamma: f64,
+    },
+    /// n(r) ∝ r^(−α) on [`min_radius`, `max_radius`]: `exponent` α (the limits are fixed).
+    PowerLaw {
+        /// α.
+        exponent: f64,
+        /// The smallest radius, > 0.
+        min_radius: f64,
+        /// The largest radius, > `min_radius`.
+        max_radius: f64,
+    },
+    /// All spheres of one `radius` > 0.
+    Monodisperse {
+        /// The radius.
+        radius: f64,
+    },
+    /// n(r) linear between the points (`radius[i]`, `density[i]`) and 0 outside them: radii
+    /// increasing and positive, densities ≥ 0 and not all 0. No parameters.
+    Tabulated {
+        /// The radii, increasing.
+        radius: Vec<f64>,
+        /// The number densities at `radius`.
+        density: Vec<f64>,
+    },
+    /// Σⱼ wⱼ nⱼ(r) with each mode normalized: `modes` of (weight wⱼ > 0, distribution).
+    MultiModal {
+        /// The modes and their number weights.
+        modes: Vec<(f64, SizeDistribution)>,
+    },
+}
+
+/// Where a distribution's density is non-zero.
+#[derive(Clone, Copy, Debug, PartialEq)]
+pub(crate) enum Support {
+    /// [lo, hi].
+    Finite(f64, f64),
+    /// (0, ∞), with most of its mass near `center`.
+    Unbounded(f64),
+    /// A single radius.
+    Point(f64),
+}
+
+fn invalid(msg: String) -> Error {
+    Error::InvalidInput(msg)
+}
+
+fn positive(name: &str, v: f64) -> Result<()> {
+    if v.is_finite() && v > 0.0 {
+        Ok(())
+    } else {
+        Err(invalid(format!("{name} must be finite and > 0, got {v}")))
+    }
+}
+
+impl SizeDistribution {
+    /// Checks the parameters.
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] naming the first invalid parameter.
+    pub fn validate(&self) -> Result<()> {
+        match self {
+            Self::LogNormal {
+                median_radius,
+                width,
+            } => {
+                positive("median_radius", *median_radius)?;
+                if !(width.is_finite() && *width > 1.0) {
+                    return Err(invalid(format!(
+                        "the lognormal width must be finite and > 1, got {width}"
+                    )));
+                }
+            }
+            Self::Gamma {
+                effective_radius,
+                effective_variance,
+            } => {
+                positive("effective_radius", *effective_radius)?;
+                if !(*effective_variance > 0.0 && *effective_variance < 0.5) {
+                    return Err(invalid(format!(
+                        "effective_variance must be in (0, 0.5), got {effective_variance}"
+                    )));
+                }
+            }
+            Self::ModifiedGamma { alpha, b, gamma } => {
+                if !(alpha.is_finite() && *alpha > -1.0) {
+                    return Err(invalid(format!(
+                        "alpha must be finite and > -1, got {alpha}"
+                    )));
+                }
+                positive("b", *b)?;
+                positive("gamma", *gamma)?;
+            }
+            Self::PowerLaw {
+                exponent,
+                min_radius,
+                max_radius,
+            } => {
+                if !exponent.is_finite() {
+                    return Err(invalid(format!("exponent must be finite, got {exponent}")));
+                }
+                positive("min_radius", *min_radius)?;
+                positive("max_radius", *max_radius)?;
+                if min_radius >= max_radius {
+                    return Err(invalid(format!(
+                        "min_radius must be < max_radius, got {min_radius} and {max_radius}"
+                    )));
+                }
+            }
+            Self::Monodisperse { radius } => positive("radius", *radius)?,
+            Self::Tabulated { radius, density } => {
+                if radius.len() < 2 || radius.len() != density.len() {
+                    return Err(invalid(format!(
+                        "a tabulated distribution needs at least 2 points and as many densities \
+                         as radii, got {} radii and {} densities",
+                        radius.len(),
+                        density.len()
+                    )));
+                }
+                for &r in radius {
+                    positive("radius", r)?;
+                }
+                if radius.windows(2).any(|w| w[0] >= w[1]) {
+                    return Err(invalid("tabulated radii must increase".into()));
+                }
+                if density.iter().any(|d| !(d.is_finite() && *d >= 0.0)) {
+                    return Err(invalid(
+                        "tabulated densities must be finite and >= 0".into(),
+                    ));
+                }
+                if density.iter().all(|&d| d == 0.0) {
+                    return Err(invalid("tabulated densities must not all be 0".into()));
+                }
+            }
+            Self::MultiModal { modes } => {
+                if modes.is_empty() {
+                    return Err(invalid("a multimodal distribution needs a mode".into()));
+                }
+                for (j, (w, mode)) in modes.iter().enumerate() {
+                    positive(&format!("the weight of mode {j}"), *w)?;
+                    mode.validate()
+                        .map_err(|e| invalid(format!("mode {j}: {e}")))?;
+                }
+            }
+        }
+        Ok(())
+    }
+
+    /// The names of the differentiable parameters, in the order of the gradients (empty for
+    /// `Tabulated`; for `MultiModal`, see [`super::EnsembleParam`]).
+    pub fn parameter_names(&self) -> &'static [&'static str] {
+        match self {
+            Self::LogNormal { .. } => &["median_radius", "width"],
+            Self::Gamma { .. } => &["effective_radius", "effective_variance"],
+            Self::ModifiedGamma { .. } => &["alpha", "b", "gamma"],
+            Self::PowerLaw { .. } => &["exponent"],
+            Self::Monodisperse { .. } => &["radius"],
+            Self::Tabulated { .. } | Self::MultiModal { .. } => &[],
+        }
+    }
+
+    /// The parameter values, in the order of [`parameter_names`](Self::parameter_names).
+    #[cfg(test)]
+    pub(crate) fn parameter_values(&self) -> Vec<f64> {
+        match *self {
+            Self::LogNormal {
+                median_radius,
+                width,
+            } => vec![median_radius, width],
+            Self::Gamma {
+                effective_radius,
+                effective_variance,
+            } => vec![effective_radius, effective_variance],
+            Self::ModifiedGamma { alpha, b, gamma } => vec![alpha, b, gamma],
+            Self::PowerLaw { exponent, .. } => vec![exponent],
+            Self::Monodisperse { radius } => vec![radius],
+            Self::Tabulated { .. } | Self::MultiModal { .. } => vec![],
+        }
+    }
+
+    /// The normalized moment ⟨rᵏ⟩ of the whole distribution, where it has a closed form (not
+    /// for `Tabulated`).
+    pub fn moment(&self, k: u32) -> Option<f64> {
+        let kf = f64::from(k);
+        match self {
+            Self::LogNormal {
+                median_radius,
+                width,
+            } => {
+                let s = width.ln();
+                Some(median_radius.powf(kf) * (0.5 * kf * kf * s * s).exp())
+            }
+            Self::Gamma {
+                effective_radius: a,
+                effective_variance: b,
+            } => {
+                let alpha = (1.0 - 3.0 * b) / b;
+                Some((1..=k).fold(1.0, |p, j| p * a * b * (alpha + f64::from(j))))
+            }
+            Self::ModifiedGamma { alpha, b, gamma } => {
+                let s = (alpha + 1.0) / gamma;
+                let ratio = (ln_gamma(s + kf / gamma) - ln_gamma(s)).exp();
+                Some(ratio * b.powf(-kf / gamma))
+            }
+            Self::PowerLaw {
+                exponent,
+                min_radius,
+                max_radius,
+            } => {
+                let integral = |p: f64| {
+                    if p.abs() < 1e-12 {
+                        (max_radius / min_radius).ln()
+                    } else {
+                        (max_radius.powf(p) - min_radius.powf(p)) / p
+                    }
+                };
+                Some(integral(kf + 1.0 - exponent) / integral(1.0 - exponent))
+            }
+            Self::Monodisperse { radius } => Some(radius.powf(kf)),
+            Self::Tabulated { .. } => None,
+            Self::MultiModal { modes } => {
+                let total: f64 = modes.iter().map(|(w, _)| w).sum();
+                let mut sum = 0.0;
+                for (w, mode) in modes {
+                    sum += w * mode.moment(k)?;
+                }
+                Some(sum / total)
+            }
+        }
+    }
+
+    /// Where the density is non-zero (not for `MultiModal`, which the engine splits into
+    /// its modes).
+    pub(crate) fn support(&self) -> Support {
+        match *self {
+            Self::LogNormal { median_radius, .. } => Support::Unbounded(median_radius),
+            Self::Gamma {
+                effective_radius, ..
+            } => Support::Unbounded(effective_radius),
+            Self::ModifiedGamma { .. } => Support::Unbounded(self.prepare().reference),
+            Self::PowerLaw {
+                min_radius,
+                max_radius,
+                ..
+            } => Support::Finite(min_radius, max_radius),
+            Self::Monodisperse { radius } => Support::Point(radius),
+            Self::Tabulated { ref radius, .. } => {
+                Support::Finite(radius[0], radius[radius.len() - 1])
+            }
+            Self::MultiModal { .. } => unreachable!("multimodal distributions are split"),
+        }
+    }
+
+    /// The density, ready to evaluate.
+    pub(crate) fn prepare(&self) -> Density<'_> {
+        let reference = match *self {
+            Self::LogNormal { median_radius, .. } => median_radius,
+            Self::Gamma {
+                effective_radius: a,
+                effective_variance: b,
+            } => {
+                let alpha = (1.0 - 3.0 * b) / b;
+                if alpha > 0.0 { alpha * a * b } else { a }
+            }
+            Self::ModifiedGamma { alpha, b, gamma } => {
+                (alpha.max(1.0) / (b * gamma)).powf(1.0 / gamma)
+            }
+            Self::PowerLaw { min_radius, .. } => min_radius,
+            _ => 1.0,
+        };
+        Density {
+            dist: self,
+            reference,
+        }
+    }
+}
+
+/// A distribution's unnormalized density with its reference radius fixed.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct Density<'a> {
+    dist: &'a SizeDistribution,
+    /// The radius where the density is scaled to about 1 (and the start of the cut-off
+    /// search).
+    pub(crate) reference: f64,
+}
+
+impl Density<'_> {
+    /// u(r), and its partial derivatives with respect to the parameters into `grad` (as many
+    /// as the parameters).
+    pub(crate) fn eval(&self, r: f64, grad: &mut [f64]) -> f64 {
+        let r0 = self.reference;
+        // ln u and ∂(ln u)/∂θ for the analytic forms.
+        let (ln_u, dln): (f64, [f64; 3]) = match *self.dist {
+            SizeDistribution::LogNormal {
+                median_radius,
+                width,
+            } => {
+                let (mu, s) = (median_radius.ln(), width.ln());
+                let z = r.ln() - mu;
+                let ln_u = -z * z / (2.0 * s * s) - s.ln() - (r / r0).ln();
+                let d_rg = z / (s * s * median_radius);
+                let d_width = (z * z / (s * s * s) - 1.0 / s) / width;
+                (ln_u, [d_rg, d_width, 0.0])
+            }
+            SizeDistribution::Gamma {
+                effective_radius: a,
+                effective_variance: b,
+            } => {
+                let alpha = (1.0 - 3.0 * b) / b;
+                let scale = a * b;
+                let l = (r / r0).ln();
+                let ln_u = alpha * l - (r - r0) / scale;
+                let d_a = (r - r0) * b / (scale * scale);
+                let d_b = -l / (b * b) + (r - r0) * a / (scale * scale);
+                (ln_u, [d_a, d_b, 0.0])
+            }
+            SizeDistribution::ModifiedGamma { alpha, b, gamma } => {
+                let (rg, r0g) = (r.powf(gamma), r0.powf(gamma));
+                let l = (r / r0).ln();
+                let ln_u = alpha * l - b * (rg - r0g);
+                let d_gamma = -b * (rg * r.ln() - r0g * r0.ln());
+                (ln_u, [l, -(rg - r0g), d_gamma])
+            }
+            SizeDistribution::PowerLaw {
+                exponent,
+                min_radius,
+                max_radius,
+            } => {
+                if r < min_radius || r > max_radius {
+                    grad.iter_mut().for_each(|g| *g = 0.0);
+                    return 0.0;
+                }
+                let l = (r / min_radius).ln();
+                (-exponent * l, [-l, 0.0, 0.0])
+            }
+            SizeDistribution::Monodisperse { .. } => (0.0, [0.0; 3]),
+            SizeDistribution::Tabulated {
+                ref radius,
+                ref density,
+            } => {
+                grad.iter_mut().for_each(|g| *g = 0.0);
+                return interpolate(radius, density, r);
+            }
+            SizeDistribution::MultiModal { .. } => {
+                unreachable!("multimodal distributions are split")
+            }
+        };
+        let u = ln_u.exp();
+        for (g, d) in grad.iter_mut().zip(dln) {
+            *g = u * d;
+        }
+        u
+    }
+}
+
+/// Linear interpolation through (xs, ys), 0 outside.
+fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
+    if x < xs[0] || x > xs[xs.len() - 1] {
+        return 0.0;
+    }
+    let i = xs.partition_point(|&v| v <= x).clamp(1, xs.len() - 1);
+    let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
+    ys[i - 1] + t * (ys[i] - ys[i - 1])
+}
+
+/// ln Γ(x) for x > 0 (Lanczos, g = 7, n = 9; relative error about 1e-15).
+pub(crate) fn ln_gamma(x: f64) -> f64 {
+    const G: f64 = 7.0;
+    const C: [f64; 9] = [
+        0.999_999_999_999_809_9,
+        676.520_368_121_885_1,
+        -1_259.139_216_722_402_8,
+        771.323_428_777_653_1,
+        -176.615_029_162_140_6,
+        12.507_343_278_686_905,
+        -0.138_571_095_265_720_12,
+        9.984_369_578_019_572e-6,
+        1.505_632_735_149_311_6e-7,
+    ];
+    if x < 0.5 {
+        // Reflection: Γ(x)Γ(1 − x) = π/sin(πx).
+        let pi = core::f64::consts::PI;
+        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
+    }
+    let x = x - 1.0;
+    let mut a = C[0];
+    for (i, &c) in C.iter().enumerate().skip(1) {
+        a += c / (x + i as f64);
+    }
+    let t = x + G + 0.5;
+    0.5 * (2.0 * core::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    fn all() -> Vec<SizeDistribution> {
+        vec![
+            SizeDistribution::LogNormal {
+                median_radius: 0.08,
+                width: 1.6,
+            },
+            SizeDistribution::Gamma {
+                effective_radius: 2.0,
+                effective_variance: 0.1,
+            },
+            SizeDistribution::ModifiedGamma {
+                alpha: 2.0,
+                b: 15.0,
+                gamma: 0.5,
+            },
+            SizeDistribution::PowerLaw {
+                exponent: 3.5,
+                min_radius: 0.01,
+                max_radius: 10.0,
+            },
+        ]
+    }
+
+    #[test]
+    fn gradients_match_finite_differences() {
+        for dist in all() {
+            let names = dist.parameter_names();
+            let values = dist.parameter_values();
+            let d = dist.prepare();
+            for r in [0.01, 0.05, 0.3, 1.0, 3.0] {
+                let mut grad = vec![0.0; names.len()];
+                let u = d.eval(r, &mut grad);
+                for (p, name) in names.iter().enumerate() {
+                    // Perturb one parameter, keeping the reference radius of `d`.
+                    let mut v = values.clone();
+                    let h = 1e-6 * v[p].abs();
+                    let at = |delta: f64, v: &mut Vec<f64>| {
+                        v[p] = values[p] + delta;
+                        let other = with_values(&dist, v);
+                        let dens = Density {
+                            dist: &other,
+                            reference: d.reference,
+                        };
+                        dens.eval(r, &mut vec![0.0; names.len()])
+                    };
+                    let fd = (at(h, &mut v) - at(-h, &mut v)) / (2.0 * h);
+                    let scale = grad[p].abs().max(1e-6 * u.abs());
+                    assert!(
+                        (fd - grad[p]).abs() <= 1e-6 * scale.max(u / values[p].abs()),
+                        "{dist:?} {name} at r = {r}: {fd} vs {}",
+                        grad[p]
+                    );
+                }
+            }
+        }
+    }
+
+    fn with_values(dist: &SizeDistribution, v: &[f64]) -> SizeDistribution {
+        match *dist {
+            SizeDistribution::LogNormal { .. } => SizeDistribution::LogNormal {
+                median_radius: v[0],
+                width: v[1],
+            },
+            SizeDistribution::Gamma { .. } => SizeDistribution::Gamma {
+                effective_radius: v[0],
+                effective_variance: v[1],
+            },
+            SizeDistribution::ModifiedGamma { .. } => SizeDistribution::ModifiedGamma {
+                alpha: v[0],
+                b: v[1],
+                gamma: v[2],
+            },
+            SizeDistribution::PowerLaw {
+                min_radius,
+                max_radius,
+                ..
+            } => SizeDistribution::PowerLaw {
+                exponent: v[0],
+                min_radius,
+                max_radius,
+            },
+            _ => unreachable!(),
+        }
+    }
+
+    #[test]
+    fn ln_gamma_matches_known_values() {
+        for (x, want) in [
+            (0.5, 0.572_364_942_924_700_1),
+            (1.0, 0.0),
+            (2.0, 0.0),
+            (3.5, 1.200_973_602_347_074_3),
+            (10.0, 12.801_827_480_081_469),
+            (0.1, 2.252_712_651_734_206),
+        ] {
+            assert!(
+                (ln_gamma(x) - want).abs() <= 1e-13 * want.abs().max(1.0),
+                "x = {x}"
+            );
+        }
+    }
+
+    #[test]
+    fn gamma_moments_agree_with_the_modified_gamma_form() {
+        // Hansen–Travis gamma is the modified gamma with γ = 1, α = (1 − 3b)/b, rate 1/(ab).
+        let (a, b) = (2.0, 0.1);
+        let g = SizeDistribution::Gamma {
+            effective_radius: a,
+            effective_variance: b,
+        };
+        let m = SizeDistribution::ModifiedGamma {
+            alpha: (1.0 - 3.0 * b) / b,
+            b: 1.0 / (a * b),
+            gamma: 1.0,
+        };
+        for k in 0..=6 {
+            let (x, y) = (g.moment(k).unwrap(), m.moment(k).unwrap());
+            assert!((x - y).abs() <= 1e-12 * x, "k = {k}: {x} vs {y}");
+        }
+        // r_eff = ⟨r³⟩/⟨r²⟩ = a.
+        assert!((g.moment(3).unwrap() / g.moment(2).unwrap() - a).abs() < 1e-12);
+    }
+
+    #[test]
+    fn invalid_parameters_are_rejected() {
+        let bad = [
+            SizeDistribution::LogNormal {
+                median_radius: 0.1,
+                width: 1.0,
+            },
+            SizeDistribution::LogNormal {
+                median_radius: -0.1,
+                width: 2.0,
+            },
+            SizeDistribution::Gamma {
+                effective_radius: 1.0,
+                effective_variance: 0.5,
+            },
+            SizeDistribution::ModifiedGamma {
+                alpha: -1.0,
+                b: 1.0,
+                gamma: 1.0,
+            },
+            SizeDistribution::PowerLaw {
+                exponent: 3.0,
+                min_radius: 2.0,
+                max_radius: 1.0,
+            },
+            SizeDistribution::Monodisperse { radius: f64::NAN },
+            SizeDistribution::Tabulated {
+                radius: vec![1.0, 1.0],
+                density: vec![1.0, 1.0],
+            },
+            SizeDistribution::Tabulated {
+                radius: vec![1.0, 2.0],
+                density: vec![0.0, 0.0],
+            },
+            SizeDistribution::MultiModal { modes: vec![] },
+            SizeDistribution::MultiModal {
+                modes: vec![(0.0, SizeDistribution::Monodisperse { radius: 1.0 })],
+            },
+        ];
+        for d in bad {
+            assert!(matches!(d.validate(), Err(Error::InvalidInput(_))), "{d:?}");
+        }
+    }
+}
diff --git a/scatrs/src/ensemble/engine.rs b/scatrs/src/ensemble/engine.rs
new file mode 100644
index 0000000..c8f759b
--- /dev/null
+++ b/scatrs/src/ensemble/engine.rs
@@ -0,0 +1,1313 @@
+//! The ensemble engine: per wavelength, the size integration of each component (pass 1:
+//! adaptive panels on the cheap quantities; pass 2: every output at the final nodes), the
+//! mixture, and the outputs with their derivatives and error estimates.
+
+use core::f64::consts::PI;
+
+use num_complex::Complex;
+use rayon::prelude::*;
+
+use super::distribution::{SizeDistribution, Support};
+use super::lin::Lin;
+use super::moments::{Moments, moments};
+use super::range::{self, MIE_POWERS, MOMENT_POWERS, STEP};
+use super::{
+    Component, EnsembleOptions, EnsembleResult, GreekOrder, Outputs, ParamKind, PhaseValues,
+    kronrod,
+};
+use crate::mie::{
+    GreekCoefficients, GreekProjector, Mie, MieOptions, MieSolution, Normalization, PhaseBuffers,
+    PhaseTables, n_terms,
+};
+use crate::{Error, Result};
+
+// The per-sphere quantities, in this order, then the phase-matrix tables at the projector's
+// nodes (F₁₁, F₁₂, F₃₃, F₃₄, each `n_q` values) and at the requested cosines (each `n_mu`).
+// Cross sections and tables are per sphere (πr²·Q, πr²·F with F per unit geometric cross
+// section, so C_sca times the normalized phase matrix); integrated with the number density
+// and divided by the integral of `ONE`, they become number averages.
+const ONE: usize = 0;
+const EXT: usize = 1;
+const SCA: usize = 2;
+const SCAG: usize = 3;
+const ABS: usize = 4;
+const BACK: usize = 5;
+const R2: usize = 6;
+const R3: usize = 7;
+const SCALARS: usize = 8;
+
+/// Pass 1 integrates the per-sphere quantities 1, C_ext, C_sca, C_sca·g and C_back (the first
+/// four at the indices of the flat layout, C_back at 4), then the optional Greek coefficients
+/// (each πr²·α, 6 arrays of `refine_order` + 1) and the refractive-index derivatives (four per
+/// part of m), and refines on all of them but C_back and its derivatives unless
+/// `refine_backscatter`. The moments r² and r³ come from their own integration (`moments`).
+const N_BASE: usize = 5;
+
+/// Panels of pass 2 per parallel task; fixed, so the order of the sums never depends on the
+/// thread count.
+const CHUNK: usize = 4;
+
+/// At most this many range extensions per side.
+const MAX_EXTENSIONS: usize = 100;
+
+/// Panels narrower than this (in ln r) are not split.
+const MIN_WIDTH: f64 = 1e-10;
+
+/// The largest increase of the size parameter across an initial panel: the efficiencies'
+/// broad interference structure has a period of about π/|m − 1| in x.
+const MAX_DX: f64 = 20.0;
+
+/// A component, or one mode of a multimodal component, integrated on its own.
+struct Part<'a> {
+    component: usize,
+    mode: Option<usize>,
+    dist: &'a SizeDistribution,
+    m: &'a [Complex<f64>],
+    /// The component's weight, and the mode's weight and the sum of the component's mode
+    /// weights (1 and 1 without modes).
+    w: f64,
+    v: f64,
+    v_total: f64,
+    support: Support,
+    /// The range of the Mie integration, before any extension.
+    cutoffs: (f64, f64),
+    /// ⟨r²⟩ and ⟨r³⟩ (`None` for a single radius).
+    moments: Option<Moments>,
+    /// The requested distribution parameters: (column of the result, index into the
+    /// distribution's gradient).
+    theta: Vec<(usize, usize)>,
+    /// The result columns of ∂/∂(Re m) and ∂/∂(Im m), if requested.
+    m_cols: [Option<usize>; 2],
+}
+
+impl Part<'_> {
+    fn m_derivatives(&self) -> bool {
+        self.m_cols.iter().any(Option::is_some)
+    }
+}
+
+fn context(e: Error, lambda: f64, r: f64) -> Error {
+    let at = format!("at wavelength {lambda:e}, radius {r:e}");
+    match e {
+        Error::InvalidInput(s) => Error::InvalidInput(format!("{at}: {s}")),
+        Error::OutsideValidatedDomain(s) => Error::OutsideValidatedDomain(format!("{at}: {s}")),
+        other => other,
+    }
+}
+
+/// Splits the components into parts and maps the requested parameters onto them.
+fn parts<'a>(
+    components: &'a [Component],
+    options: &EnsembleOptions,
+    rtol: f64,
+) -> Result<Vec<Part<'a>>> {
+    let mut out = Vec::new();
+    for (ci, c) in components.iter().enumerate() {
+        let modes: Vec<(Option<usize>, f64, &SizeDistribution)> = match &c.distribution {
+            SizeDistribution::MultiModal { modes } => modes
+                .iter()
+                .enumerate()
+                .map(|(j, (v, d))| (Some(j), *v, d))
+                .collect(),
+            d => vec![(None, 1.0, d)],
+        };
+        let v_total: f64 = modes.iter().map(|(_, v, _)| v).sum();
+        for (mode, v, dist) in modes {
+            let names = dist.parameter_names();
+            let mut theta = Vec::new();
+            let mut m_cols = [None, None];
+            for (col, p) in options.derivatives.iter().enumerate() {
+                if p.component != ci {
+                    continue;
+                }
+                match p.kind {
+                    ParamKind::Distribution(name) if p.mode == mode => {
+                        let k = names.iter().position(|n| *n == name).expect("validated");
+                        theta.push((col, k));
+                    }
+                    ParamKind::RefrReal => m_cols[0] = Some(col),
+                    ParamKind::RefrImag => m_cols[1] = Some(col),
+                    _ => {}
+                }
+            }
+            let support = dist.support();
+            let breaks: &[f64] = match dist {
+                SizeDistribution::Tabulated { radius, .. } => radius,
+                _ => &[],
+            };
+            let gradient: Vec<usize> = theta.iter().map(|&(_, k)| k).collect();
+            let (cutoffs, moments) = match support {
+                Support::Unbounded(center) => {
+                    let d = dist.prepare();
+                    let mie = range::cutoffs(&d, center, rtol / 10.0, MIE_POWERS)?;
+                    let wide = range::cutoffs(&d, center, rtol / 10.0, MOMENT_POWERS)?;
+                    let m = moments(dist, &gradient, wide, breaks, rtol, 0.2 * rtol);
+                    (mie, Some(m))
+                }
+                Support::Finite(lo, hi) => {
+                    let m = moments(dist, &gradient, (lo, hi), breaks, rtol, 0.0);
+                    ((lo, hi), Some(m))
+                }
+                Support::Point(r) => ((r, r), None),
+            };
+            out.push(Part {
+                component: ci,
+                mode,
+                dist,
+                m: &c.refractive_index,
+                w: c.weight,
+                v,
+                v_total,
+                support,
+                cutoffs,
+                moments,
+                theta,
+                m_cols,
+            });
+        }
+    }
+    Ok(out)
+}
+
+/// Runs every wavelength (validated inputs).
+pub(super) fn run(
+    components: &[Component],
+    wavelengths: &[f64],
+    options: &EnsembleOptions,
+) -> Result<Vec<EnsembleResult>> {
+    let parts = parts(components, options, options.rtol)?;
+    let w_total: f64 = components.iter().map(|c| c.weight).sum();
+    let results: Vec<Result<EnsembleResult>> = (0..wavelengths.len())
+        .into_par_iter()
+        .map(|i| wavelength(&parts, w_total, wavelengths[i], i, options))
+        .collect();
+    results.into_iter().collect()
+}
+
+/// The size parameter of radius `r` at `lambda`.
+fn size(r: f64, lambda: f64) -> f64 {
+    2.0 * PI * r / lambda
+}
+
+/// The initial panels of [t_lo, t_hi] (t = ln r): at most `STEP` wide and at most `MAX_DX` in
+/// size parameter, with the radii of `breaks` (inside the range) as edges.
+fn initial_panels(t_lo: f64, t_hi: f64, lambda: f64, breaks: &[f64]) -> Vec<(f64, f64)> {
+    let mut edges = vec![t_lo];
+    edges.extend(
+        breaks
+            .iter()
+            .map(|r| r.ln())
+            .filter(|&t| t > t_lo && t < t_hi),
+    );
+    edges.push(t_hi);
+    let mut out = Vec::new();
+    for w in edges.windows(2) {
+        let (mut a, b) = (w[0], w[1]);
+        while a < b {
+            let x = size(a.exp(), lambda);
+            let step = STEP.min((1.0 + MAX_DX / x).ln());
+            // Equal panels up to b rather than a short last one.
+            let n = ((b - a) / step).ceil().max(1.0);
+            let h = (b - a) / n;
+            let next = if n <= 1.0 { b } else { a + h };
+            out.push((a, next));
+            a = next;
+        }
+    }
+    out
+}
+
+/// Pass-1 sums of one panel.
+#[derive(Clone, Debug)]
+struct Sums1 {
+    /// Kronrod sums of each refined quantity, then of each θ-weighted one (block per θ).
+    k: Vec<f64>,
+    /// |Kronrod − Gauss| of the same.
+    err: Vec<f64>,
+    /// Kronrod sums of |∂u/∂θ|·r·|q|, the scales of the θ blocks.
+    abs: Vec<f64>,
+    /// Kronrod sums of u·r·|q|, the scales of the refractive-index derivatives.
+    absv: Vec<f64>,
+    /// The largest number of terms at the nodes.
+    n_terms: usize,
+    /// u·r·C_ext at the first and the last node (the integrand per unit ln r).
+    edge: [f64; 2],
+}
+
+/// The pass-1 quantities at radius r: 1, C_ext, C_sca, C_sca·g, C_back, then the Greek
+/// coefficients up to `refine_order` (πr²·α), then the derivatives of C_ext, C_sca, C_sca·g and
+/// C_back with respect to Re m and Im m where `m_derivatives` asks for them; and the number of
+/// terms.
+fn quantities1(
+    mie: &mut Mie<f64>,
+    r: f64,
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &MieOptions,
+    refine_order: Option<usize>,
+    m_derivatives: [bool; 2],
+) -> Result<(Vec<f64>, usize)> {
+    let x = size(r, lambda);
+    let s = if m_derivatives.iter().any(|&b| b) {
+        mie.compute_with_derivatives(x, m, opts)
+    } else {
+        mie.compute(x, m, opts)
+    }
+    .map_err(|e| context(e, lambda, r))?;
+    let q = s.efficiencies();
+    let area = PI * r * r;
+    let mut out = vec![
+        1.0,
+        area * q.q_ext,
+        area * q.q_sca,
+        area * q.q_sca * q.g,
+        area * q.q_back,
+    ];
+    if let Some(order) = refine_order {
+        let g = s
+            .greek(Some(order), Normalization::Unnormalized)
+            .map_err(|e| context(e, lambda, r))?;
+        for a in [
+            &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
+        ] {
+            out.extend(a.iter().map(|v| area * v));
+        }
+    }
+    if let Some(jac) = s.efficiencies_jacobian() {
+        for (want, d) in m_derivatives.into_iter().zip([jac.m_re, jac.m_im]) {
+            if want {
+                out.extend([
+                    area * d.q_ext,
+                    area * d.q_sca,
+                    area * (d.q_sca * q.g + q.q_sca * d.g),
+                    area * d.q_back,
+                ]);
+            }
+        }
+    }
+    Ok((out, s.n_terms()))
+}
+
+/// The number of pass-1 quantities: the base ones, the Greek coefficients, and four
+/// derivatives per requested refractive-index part.
+fn n_refined(refine_order: Option<usize>, part: &Part<'_>) -> usize {
+    let md = part.m_cols.iter().filter(|c| c.is_some()).count();
+    N_BASE + refine_order.map_or(0, |l| 6 * (l + 1)) + 4 * md
+}
+
+/// Evaluates one panel in pass 1.
+#[allow(clippy::too_many_arguments)]
+fn panel1(
+    mie: &mut Mie<f64>,
+    part: &Part<'_>,
+    (a, b): (f64, f64),
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &MieOptions,
+    refine_order: Option<usize>,
+    n_refined: usize,
+) -> Result<Sums1> {
+    let d = part.dist.prepare();
+    let n_grad = part.dist.parameter_names().len();
+    let nt = part.theta.len();
+    let mut grad = vec![0.0; n_grad];
+    let mut k = vec![0.0; (1 + nt) * n_refined];
+    let mut g = vec![0.0; (1 + nt) * n_refined];
+    let mut abs = vec![0.0; nt * n_refined];
+    let mut absv = vec![0.0; n_refined];
+    let m_derivatives = [part.m_cols[0].is_some(), part.m_cols[1].is_some()];
+    let mut n_max = 0;
+    let mut edge = [0.0; 2];
+    let nodes = kronrod::panel(a, b);
+    for (i, &(t, wk, wg)) in nodes.iter().enumerate() {
+        let r = t.exp();
+        let u = d.eval(r, &mut grad);
+        let (q, n) = quantities1(mie, r, lambda, m, opts, refine_order, m_derivatives)?;
+        n_max = n_max.max(n);
+        let base = u * r;
+        if i == 0 {
+            edge[0] = base * q[EXT];
+        }
+        if i == nodes.len() - 1 {
+            edge[1] = base * q[EXT];
+        }
+        for (j, &qj) in q.iter().enumerate() {
+            k[j] += wk * base * qj;
+            g[j] += wg * base * qj;
+            absv[j] += wk * (base * qj).abs();
+            for (b_idx, &(_, gi)) in part.theta.iter().enumerate() {
+                let dw = grad[gi] * r;
+                let o = (1 + b_idx) * n_refined + j;
+                k[o] += wk * dw * qj;
+                g[o] += wg * dw * qj;
+                abs[b_idx * n_refined + j] += wk * (dw * qj).abs();
+            }
+        }
+    }
+    let err = k.iter().zip(&g).map(|(x, y)| (x - y).abs()).collect();
+    Ok(Sums1 {
+        k,
+        err,
+        abs,
+        absv,
+        n_terms: n_max,
+        edge,
+    })
+}
+
+/// The outcome of pass 1 for one part.
+struct Pass1 {
+    panels: Vec<(f64, f64)>,
+    converged: bool,
+    n_terms: usize,
+    range: (f64, f64),
+}
+
+/// Evaluates the panels without sums, in parallel and in order.
+#[allow(clippy::too_many_arguments)]
+fn evaluate(
+    panels: &mut [((f64, f64), Option<Sums1>)],
+    part: &Part<'_>,
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &MieOptions,
+    refine_order: Option<usize>,
+    n_refined: usize,
+) -> Result<()> {
+    panels
+        .par_iter_mut()
+        .filter(|(_, s)| s.is_none())
+        .map_init(Mie::<f64>::new, |mie, (ab, s)| {
+            *s = Some(panel1(
+                mie,
+                part,
+                *ab,
+                lambda,
+                m,
+                opts,
+                refine_order,
+                n_refined,
+            )?);
+            Ok(())
+        })
+        .collect::<Result<Vec<()>>>()?;
+    Ok(())
+}
+
+/// Pass 1: the adaptive panels of one part at one wavelength.
+fn pass1(
+    part: &Part<'_>,
+    lambda: f64,
+    m: Complex<f64>,
+    options: &EnsembleOptions,
+    mie_opts: &MieOptions,
+) -> Result<Pass1> {
+    let rtol = options.rtol;
+    let refine_order = options.refine_order;
+    let n_refined = n_refined(refine_order, part);
+    let n_greek = N_BASE + refine_order.map_or(0, |l| 6 * (l + 1));
+    let nt = part.theta.len();
+    let breaks: &[f64] = match part.dist {
+        SizeDistribution::Tabulated { radius, .. } => radius,
+        _ => &[],
+    };
+    let (mut t_lo, mut t_hi) = (part.cutoffs.0.ln(), part.cutoffs.1.ln());
+    let extendable = matches!(part.support, Support::Unbounded(_));
+    let mut initial = initial_panels(t_lo, t_hi, lambda, breaks);
+    if initial.len() > options.max_panels {
+        // Within the budget: equal panels in ln r.
+        let n = options.max_panels;
+        let h = (t_hi - t_lo) / n as f64;
+        initial = (0..n)
+            .map(|i| (t_lo + i as f64 * h, t_lo + (i + 1) as f64 * h))
+            .collect();
+    }
+    let mut panels: Vec<((f64, f64), Option<Sums1>)> =
+        initial.into_iter().map(|p| (p, None)).collect();
+    let mut extensions = [0usize; 2];
+    loop {
+        evaluate(
+            &mut panels,
+            part,
+            lambda,
+            m,
+            mie_opts,
+            refine_order,
+            n_refined,
+        )?;
+        let n = panels.len();
+        let sums: Vec<&Sums1> = panels
+            .iter()
+            .map(|(_, s)| s.as_ref().expect("evaluated"))
+            .collect();
+        let total = |j: usize| sums.iter().map(|s| s.k[j]).sum::<f64>();
+        let err = |j: usize| sums.iter().map(|s| s.err[j]).sum::<f64>();
+        // The targets of each refined quantity.
+        let mut target = vec![0.0; (1 + nt) * n_refined];
+        let c_sca = total(SCA).abs();
+        for j in 0..n_refined {
+            let scale = match j {
+                SCAG => c_sca,
+                // As for single spheres (`MieOutput::Backscattering`): relative to the larger
+                // of ⟨C_back⟩ and ⟨C_sca⟩, since backscattering by large weakly absorbing
+                // spheres is a sum of narrow glory ripples orders of magnitude below C_sca.
+                4 => total(4).abs().max(c_sca),
+                // The refractive-index derivatives can cancel: their absolute integrands.
+                j if j >= n_greek => sums.iter().map(|s| s.absv[j]).sum::<f64>(),
+                j if j >= N_BASE => c_sca,
+                j => total(j).abs(),
+            };
+            target[j] = rtol * scale;
+            for b in 0..nt {
+                let abs: f64 = sums.iter().map(|s| s.abs[b * n_refined + j]).sum();
+                target[(1 + b) * n_refined + j] = rtol * abs;
+            }
+        }
+        if !options.refine_backscatter {
+            // C_back and its derivatives (every fourth refractive-index derivative) are only
+            // integrated.
+            for b in 0..=nt {
+                target[b * n_refined + 4] = f64::INFINITY;
+                for j in (n_greek + 3..n_refined).step_by(4) {
+                    target[b * n_refined + j] = f64::INFINITY;
+                }
+            }
+        }
+        let errors: Vec<f64> = (0..target.len()).map(err).collect();
+        let met = errors.iter().zip(&target).all(|(e, t)| e <= t);
+        if met {
+            // Extend the range where the extinction integrand is not negligible at an end.
+            let ext = total(EXT);
+            let first = sums.first().expect("a panel").edge[0];
+            let last = sums.last().expect("a panel").edge[1];
+            let small = |density: f64| density * STEP <= 0.1 * rtol * ext.abs();
+            let mut extended = false;
+            if extendable && !small(first) && extensions[0] < MAX_EXTENSIONS {
+                let fresh = initial_panels(t_lo - STEP, t_lo, lambda, &[]);
+                t_lo -= STEP;
+                extensions[0] += 1;
+                let mut new: Vec<_> = fresh.into_iter().map(|p| (p, None)).collect();
+                new.append(&mut panels);
+                panels = new;
+                extended = true;
+            }
+            if extendable && !small(last) && extensions[1] < MAX_EXTENSIONS {
+                let fresh = initial_panels(t_hi, t_hi + STEP, lambda, &[]);
+                t_hi += STEP;
+                extensions[1] += 1;
+                panels.extend(fresh.into_iter().map(|p| (p, None)));
+                extended = true;
+            }
+            if !extended {
+                return Ok(finish(panels, true, (t_lo, t_hi)));
+            }
+            continue;
+        }
+        // Split every panel above its share of a target, worst first, within the budget.
+        let mut candidates: Vec<(f64, usize)> = sums
+            .iter()
+            .enumerate()
+            .filter(|&(i, _)| panels[i].0.1 - panels[i].0.0 > MIN_WIDTH)
+            .filter_map(|(i, s)| {
+                let badness = s
+                    .err
+                    .iter()
+                    .zip(&target)
+                    .zip(&errors)
+                    // Only the quantities still above their targets.
+                    .filter(|&((_, t), &total_err)| total_err > *t)
+                    .map(|((e, t), _)| e * n as f64 / t.max(f64::MIN_POSITIVE))
+                    .fold(0.0, f64::max);
+                (badness > 1.0).then_some((badness, i))
+            })
+            .collect();
+        let budget = options.max_panels.saturating_sub(n);
+        if candidates.is_empty() || budget == 0 {
+            return Ok(finish(panels, false, (t_lo, t_hi)));
+        }
+        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
+        candidates.truncate(budget);
+        let mut split = vec![false; n];
+        for &(_, i) in &candidates {
+            split[i] = true;
+        }
+        let mut next = Vec::with_capacity(n + candidates.len());
+        for (i, p) in panels.into_iter().enumerate() {
+            if split[i] {
+                let ((a, b), _) = p;
+                let mid = 0.5 * (a + b);
+                next.push(((a, mid), None));
+                next.push(((mid, b), None));
+            } else {
+                next.push(p);
+            }
+        }
+        panels = next;
+    }
+}
+
+fn finish(panels: Vec<((f64, f64), Option<Sums1>)>, converged: bool, t: (f64, f64)) -> Pass1 {
+    let n_terms = panels
+        .iter()
+        .map(|(_, s)| s.as_ref().map_or(0, |s| s.n_terms))
+        .max()
+        .unwrap_or(0);
+    Pass1 {
+        panels: panels.into_iter().map(|(p, _)| p).collect(),
+        converged,
+        n_terms,
+        range: (t.0.exp(), t.1.exp()),
+    }
+}
+
+/// Greek-coefficient projection groups the size nodes into bands by their number of terms N,
+/// each band with its own exact projector (n_q = N_band + ⌈(L + 1)/2⌉, N_band the band's
+/// largest N): a sphere's amplitudes then cost O(N·(N_band + L)) rather than O(N·N_max), which
+/// matters for broad distributions. The band of N is the first `bucket` value ≥ N, in a
+/// sequence that grows by a factor of 1.25.
+fn bucket(n: usize) -> usize {
+    let mut u = 1usize;
+    while u < n {
+        u = (u as f64 * 1.25).ceil() as usize;
+    }
+    u
+}
+
+/// One band: its largest number of terms, the start of its tables in the flat layout, and
+/// its projector.
+struct Band {
+    upper: usize,
+    start: usize,
+    projector: GreekProjector<f64>,
+}
+
+/// What pass 2 computes at every node: the per-sphere scalars, the phase matrix at the
+/// requested cosines (each element `n_mu` values), and the tables of each band at its
+/// projector's nodes.
+struct Layout<'a> {
+    mu: &'a [f64],
+    bands: Vec<Band>,
+    len: usize,
+}
+
+impl Layout<'_> {
+    /// The scalars and the cosine tables: the part of the layout every node fills.
+    fn common(&self) -> usize {
+        SCALARS + 4 * self.mu.len()
+    }
+
+    fn band(&self, n: usize) -> Option<&Band> {
+        let u = bucket(n);
+        self.bands.iter().find(|b| b.upper == u)
+    }
+}
+
+/// The per-sphere quantities of pass 2 at radius r into `g` (the common part and the node's
+/// band, whose (start, length) it returns), and their derivatives with respect to Re m, Im m
+/// into `dm` and with respect to r into `dr`, if given.
+#[allow(clippy::too_many_arguments)]
+fn quantities2(
+    mie: &mut Mie<f64>,
+    r: f64,
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &MieOptions,
+    layout: &Layout<'_>,
+    g: &mut [f64],
+    mut dm: Option<[&mut [f64]; 2]>,
+    dr: Option<&mut [f64]>,
+) -> Result<Option<(usize, usize)>> {
+    let x = size(r, lambda);
+    let derivatives = dm.is_some() || dr.is_some();
+    let s = if derivatives {
+        mie.compute_with_derivatives(x, m, opts)
+    } else {
+        mie.compute(x, m, opts)
+    }
+    .map_err(|e| context(e, lambda, r))?;
+    let area = PI * r * r;
+    let q = s.efficiencies();
+    g[..SCALARS].copy_from_slice(&[
+        1.0,
+        area * q.q_ext,
+        area * q.q_sca,
+        area * q.q_sca * q.g,
+        area * q.q_abs,
+        area * q.q_back,
+        r * r,
+        r * r * r,
+    ]);
+    let band = layout
+        .band(s.n_terms())
+        .map(|b| (b.start, b.projector.nodes()));
+    if !layout.bands.is_empty() && band.is_none() {
+        return Err(context(
+            Error::InvalidInput(format!("no projector band for {} terms", s.n_terms())),
+            lambda,
+            r,
+        ));
+    }
+    let mut tables = vec![(SCALARS, layout.mu)];
+    tables.extend(band);
+    let segment = band.map(|(start, nodes)| (start, 4 * nodes.len()));
+    if !derivatives {
+        for (start, cosines) in tables {
+            let n = cosines.len();
+            if n == 0 {
+                continue;
+            }
+            phase_into(&s, cosines, &mut g[start..start + 4 * n])
+                .map_err(|e| context(e, lambda, r))?;
+            g[start..start + 4 * n].iter_mut().for_each(|v| *v *= area);
+        }
+        return Ok(segment);
+    }
+    let jac = s
+        .efficiencies_jacobian()
+        .expect("computed with derivatives");
+    let scalar_d = |e: &crate::mie::Efficiencies<f64>| {
+        [
+            0.0,
+            area * e.q_ext,
+            area * e.q_sca,
+            area * (e.q_sca * q.g + q.q_sca * e.g),
+            area * e.q_abs,
+            area * e.q_back,
+            0.0,
+            0.0,
+        ]
+    };
+    let k = 2.0 * PI / lambda;
+    if let Some([dre, dim]) = dm.as_mut() {
+        dre[..SCALARS].copy_from_slice(&scalar_d(&jac.m_re));
+        dim[..SCALARS].copy_from_slice(&scalar_d(&jac.m_im));
+    }
+    let mut dr = dr;
+    if let Some(dr) = dr.as_deref_mut() {
+        // ∂(πr²Q)/∂r = 2πr Q + πr² (2π/λ) ∂Q/∂x.
+        let dx = scalar_d(&jac.x);
+        let values = [0.0, q.q_ext, q.q_sca, q.q_sca * q.g, q.q_abs, q.q_back];
+        for j in 1..6 {
+            dr[j] = 2.0 * PI * r * values[j] + k * dx[j];
+        }
+        dr[ONE] = 0.0;
+        dr[R2] = 2.0 * r;
+        dr[R3] = 3.0 * r * r;
+    }
+    for (start, cosines) in tables {
+        let n = cosines.len();
+        if n == 0 {
+            continue;
+        }
+        let span = start..start + 4 * n;
+        let mut dx = vec![0.0; 4 * n];
+        let (mut dre_buf, mut dim_buf) = (vec![0.0; 4 * n], vec![0.0; 4 * n]);
+        {
+            let [v11, v12, v33, v34] = quarter(&mut g[span.clone()]);
+            let [x11, x12, x33, x34] = quarter(&mut dx);
+            let [r11, r12, r33, r34] = quarter(&mut dre_buf);
+            let [i11, i12, i33, i34] = quarter(&mut dim_buf);
+            s.phase_matrix_with_derivatives(
+                cosines,
+                Normalization::Unnormalized,
+                PhaseBuffers {
+                    f11: v11,
+                    f12: v12,
+                    f33: v33,
+                    f34: v34,
+                },
+                [
+                    PhaseBuffers {
+                        f11: x11,
+                        f12: x12,
+                        f33: x33,
+                        f34: x34,
+                    },
+                    PhaseBuffers {
+                        f11: r11,
+                        f12: r12,
+                        f33: r33,
+                        f34: r34,
+                    },
+                    PhaseBuffers {
+                        f11: i11,
+                        f12: i12,
+                        f33: i33,
+                        f34: i34,
+                    },
+                ],
+            )
+            .map_err(|e| context(e, lambda, r))?;
+        }
+        if let Some(dr) = dr.as_deref_mut() {
+            for (j, o) in span.clone().enumerate() {
+                dr[o] = 2.0 * PI * r * g[o] + area * k * dx[j];
+            }
+        }
+        if let Some([dre, dim]) = dm.as_mut() {
+            for (j, o) in span.clone().enumerate() {
+                dre[o] = area * dre_buf[j];
+                dim[o] = area * dim_buf[j];
+            }
+        }
+        g[span].iter_mut().for_each(|v| *v *= area);
+    }
+    Ok(segment)
+}
+
+/// A table of four elements as phase-matrix tables.
+fn view(t: &[Vec<f64>; 4]) -> PhaseTables<'_, f64> {
+    PhaseTables {
+        f11: &t[0],
+        f12: &t[1],
+        f33: &t[2],
+        f34: &t[3],
+    }
+}
+
+/// Splits a table of 4·n values into its four elements.
+fn quarter(t: &mut [f64]) -> [&mut [f64]; 4] {
+    let n = t.len() / 4;
+    let (a, rest) = t.split_at_mut(n);
+    let (b, rest) = rest.split_at_mut(n);
+    let (c, d) = rest.split_at_mut(n);
+    [a, b, c, d]
+}
+
+/// The unnormalized phase matrix of `s` at `cosines` into the 4·n values of `out`.
+fn phase_into(s: &MieSolution<'_, f64>, cosines: &[f64], out: &mut [f64]) -> Result<()> {
+    let [f11, f12, f33, f34] = quarter(out);
+    s.phase_matrix(
+        cosines,
+        Normalization::Unnormalized,
+        PhaseBuffers { f11, f12, f33, f34 },
+    )
+}
+
+/// Pass-2 integrals of one part: blocks of the flat layout (values; each requested θ;
+/// Re m, Im m if requested), Kronrod sums and Σ over panels of |Kronrod − Gauss|.
+struct Sums2 {
+    s: Vec<f64>,
+    err: Vec<f64>,
+    blocks: usize,
+}
+
+/// Pass 2 of one part over its final panels.
+fn pass2(
+    part: &Part<'_>,
+    panels: &[(f64, f64)],
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &MieOptions,
+    layout: &Layout<'_>,
+) -> Result<Sums2> {
+    let len = layout.len;
+    let common = layout.common();
+    let nt = part.theta.len();
+    let nm = if part.m_derivatives() { 2 } else { 0 };
+    let blocks = 1 + nt + nm;
+    let n_grad = part.dist.parameter_names().len();
+    let chunks: Vec<Result<(Vec<f64>, Vec<f64>)>> = panels
+        .par_chunks(CHUNK)
+        .map(|chunk| {
+            let mut mie = Mie::<f64>::new();
+            let d = part.dist.prepare();
+            let mut grad = vec![0.0; n_grad];
+            let mut sum = vec![0.0; blocks * len];
+            let mut err = vec![0.0; blocks * len];
+            let mut g = vec![0.0; len];
+            let mut dmv = vec![0.0; 2 * len];
+            let (mut pk, mut pg) = (vec![0.0; blocks * len], vec![0.0; blocks * len]);
+            for &(a, b) in chunk {
+                // The ranges of the flat layout this panel touches.
+                let mut ranges = vec![(0, common)];
+                for &(t, wk, wg) in &kronrod::panel(a, b) {
+                    let r = t.exp();
+                    let u = d.eval(r, &mut grad);
+                    let dm = if nm > 0 {
+                        let (re, im) = dmv.split_at_mut(len);
+                        Some([re, im])
+                    } else {
+                        None
+                    };
+                    let segment =
+                        quantities2(&mut mie, r, lambda, m, opts, layout, &mut g, dm, None)?;
+                    if let Some((start, n)) = segment {
+                        if !ranges.contains(&(start, start + n)) {
+                            ranges.push((start, start + n));
+                        }
+                    }
+                    let base = u * r;
+                    let node_ranges = [Some((0, common)), segment.map(|(s, n)| (s, s + n))];
+                    for (lo, hi) in node_ranges.into_iter().flatten() {
+                        for i in lo..hi {
+                            let gi = g[i];
+                            pk[i] += wk * base * gi;
+                            pg[i] += wg * base * gi;
+                            for (bi, &(_, k)) in part.theta.iter().enumerate() {
+                                let o = (1 + bi) * len + i;
+                                let dw = grad[k] * r * gi;
+                                pk[o] += wk * dw;
+                                pg[o] += wg * dw;
+                            }
+                            for c in 0..nm {
+                                let o = (1 + nt + c) * len + i;
+                                let dv = base * dmv[c * len + i];
+                                pk[o] += wk * dv;
+                                pg[o] += wg * dv;
+                            }
+                        }
+                    }
+                }
+                for &(lo, hi) in &ranges {
+                    for blk in 0..blocks {
+                        for i in blk * len + lo..blk * len + hi {
+                            sum[i] += pk[i];
+                            err[i] += (pk[i] - pg[i]).abs();
+                            pk[i] = 0.0;
+                            pg[i] = 0.0;
+                        }
+                    }
+                }
+            }
+            Ok((sum, err))
+        })
+        .collect();
+    let mut s = vec![0.0; blocks * len];
+    let mut err = vec![0.0; blocks * len];
+    for c in chunks {
+        let (cs, ce) = c?;
+        for i in 0..blocks * len {
+            s[i] += cs[i];
+            err[i] += ce[i];
+        }
+    }
+    Ok(Sums2 { s, err, blocks })
+}
+
+/// A part's number averages with derivatives (by result column) and error estimates.
+struct PartAverages {
+    v: Vec<f64>,
+    e: Vec<f64>,
+    /// (column, derivative, its error).
+    d: Vec<(usize, Vec<f64>, Vec<f64>)>,
+}
+
+/// Number averages from the pass-2 integrals: E = S/S[ONE], with the quotient rule for the
+/// θ blocks (the renormalization) and first-order error propagation. Each error estimate also
+/// includes `truncation`·|value| for the cut-offs: they omit at most rtol/10 of the number,
+/// area and volume moments (and the range is extended until the extinction integrand is as
+/// small at both ends), which moves an average by at most about twice that.
+fn averages(part: &Part<'_>, sums: &Sums2, len: usize, truncation: f64) -> PartAverages {
+    let block = |b: usize| &sums.s[b * len..(b + 1) * len];
+    let eblock = |b: usize| &sums.err[b * len..(b + 1) * len];
+    let (s0, e0) = (block(0), eblock(0));
+    let z = s0[ONE];
+    let ez = e0[ONE];
+    let v: Vec<f64> = s0.iter().map(|x| x / z).collect();
+    let e: Vec<f64> = (0..len)
+        .map(|i| (e0[i] + v[i].abs() * ez) / z + truncation * v[i].abs())
+        .collect();
+    let mut d = Vec::new();
+    for (bi, &(col, _)) in part.theta.iter().enumerate() {
+        let (sb, eb) = (block(1 + bi), eblock(1 + bi));
+        let dz = sb[ONE];
+        let dv: Vec<f64> = (0..len).map(|i| (sb[i] - v[i] * dz) / z).collect();
+        let de: Vec<f64> = (0..len)
+            .map(|i| {
+                (eb[i] + v[i].abs() * eb[ONE] + dv[i].abs() * ez + e[i] * dz.abs()) / z
+                    + truncation * dv[i].abs()
+            })
+            .collect();
+        d.push((col, dv, de));
+    }
+    let nt = part.theta.len();
+    let mut c = 0;
+    for col in part.m_cols {
+        if let Some(col) = col {
+            let (sb, eb) = (block(1 + nt + c), eblock(1 + nt + c));
+            let dv: Vec<f64> = (0..len).map(|i| sb[i] / z).collect();
+            let de: Vec<f64> = (0..len)
+                .map(|i| (eb[i] + dv[i].abs() * ez) / z + truncation * dv[i].abs())
+                .collect();
+            d.push((col, dv, de));
+        }
+        if part.m_derivatives() {
+            c += 1;
+        }
+    }
+    let mut out = PartAverages { v, e, d };
+    if let Some(m) = &part.moments {
+        for (k, idx) in [R2, R3].into_iter().enumerate() {
+            out.v[idx] = m.v[k];
+            out.e[idx] = m.e[k];
+            for (col, dv, de) in &mut out.d {
+                match part.theta.iter().position(|&(c, _)| c == *col) {
+                    Some(b) => {
+                        dv[idx] = m.d[b][k];
+                        de[idx] = m.de[b][k];
+                    }
+                    None => {
+                        dv[idx] = 0.0;
+                        de[idx] = 0.0;
+                    }
+                }
+            }
+        }
+    }
+    out
+}
+
+/// A monodisperse part: the sphere itself, exact (no integration).
+fn monodisperse(
+    part: &Part<'_>,
+    r: f64,
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &MieOptions,
+    layout: &Layout<'_>,
+) -> Result<PartAverages> {
+    let len = layout.len;
+    let mut mie = Mie::<f64>::new();
+    let mut g = vec![0.0; len];
+    let mut dmv = vec![0.0; 2 * len];
+    let mut drv = vec![0.0; len];
+    let radius_col = part.theta.first().map(|&(col, _)| col);
+    {
+        let dm = if part.m_derivatives() {
+            let (re, im) = dmv.split_at_mut(len);
+            Some([re, im])
+        } else {
+            None
+        };
+        let dr = radius_col.map(|_| drv.as_mut_slice());
+        quantities2(&mut mie, r, lambda, m, opts, layout, &mut g, dm, dr)?;
+    }
+    let zeros = vec![0.0; len];
+    let mut d = Vec::new();
+    if let Some(col) = radius_col {
+        d.push((col, drv, zeros.clone()));
+    }
+    for (c, col) in part.m_cols.iter().enumerate() {
+        if let Some(col) = col {
+            d.push((*col, dmv[c * len..(c + 1) * len].to_vec(), zeros.clone()));
+        }
+    }
+    Ok(PartAverages { v: g, e: zeros, d })
+}
+
+/// One wavelength.
+fn wavelength(
+    parts: &[Part<'_>],
+    w_total: f64,
+    lambda: f64,
+    index: usize,
+    options: &EnsembleOptions,
+) -> Result<EnsembleResult> {
+    let mie_opts = MieOptions::new().with_allow_unvalidated(options.allow_unvalidated);
+    let n_cols = options.derivatives.len();
+    // Pass 1 for every part, and the largest number of terms.
+    let mut firsts = Vec::with_capacity(parts.len());
+    let mut n_max = 0;
+    for part in parts {
+        let m = part.m[index];
+        let p1 = match part.support {
+            Support::Point(r) => {
+                let n =
+                    n_terms(size(r, lambda), m, &mie_opts).map_err(|e| context(e, lambda, r))?;
+                Pass1 {
+                    panels: Vec::new(),
+                    converged: true,
+                    n_terms: n,
+                    range: (r, r),
+                }
+            }
+            _ => pass1(part, lambda, m, options, &mie_opts)?,
+        };
+        n_max = n_max.max(p1.n_terms);
+        firsts.push(p1);
+    }
+    // The projector bands, from the numbers of terms at the final nodes.
+    let order = match options.greek {
+        None => 0,
+        Some(GreekOrder::All) => 2 * n_max,
+        Some(GreekOrder::Order(l)) => l,
+    };
+    let mut uppers = std::collections::BTreeSet::new();
+    if options.greek.is_some() {
+        for (part, p1) in parts.iter().zip(&firsts) {
+            let m = part.m[index];
+            if let Support::Point(_) = part.support {
+                uppers.insert(bucket(p1.n_terms));
+                continue;
+            }
+            for &(a, b) in &p1.panels {
+                for &(t, _, _) in &kronrod::panel(a, b) {
+                    let r = t.exp();
+                    let n = n_terms(size(r, lambda), m, &mie_opts)
+                        .map_err(|e| context(e, lambda, r))?;
+                    uppers.insert(bucket(n));
+                }
+            }
+        }
+    }
+    let mut bands = Vec::with_capacity(uppers.len());
+    let mut len = SCALARS + 4 * options.mu.len();
+    for upper in uppers {
+        let projected = order.min(2 * upper);
+        let projector = GreekProjector::<f64>::new(projected, upper + (projected + 1).div_ceil(2))?;
+        let n_q = projector.nodes().len();
+        bands.push(Band {
+            upper,
+            start: len,
+            projector,
+        });
+        len += 4 * n_q;
+    }
+    let layout = Layout {
+        mu: &options.mu,
+        bands,
+        len,
+    };
+    // Pass 2 and the mixture.
+    let mut v = vec![0.0; len];
+    let mut e = vec![0.0; len];
+    let mut d = vec![vec![0.0; len]; n_cols];
+    let mut de = vec![vec![0.0; len]; n_cols];
+    let mut averages_of = Vec::with_capacity(parts.len());
+    for (part, p1) in parts.iter().zip(&firsts) {
+        let m = part.m[index];
+        let avg = match part.support {
+            Support::Point(r) => monodisperse(part, r, lambda, m, &mie_opts, &layout)?,
+            _ => {
+                let sums = pass2(part, &p1.panels, lambda, m, &mie_opts, &layout)?;
+                debug_assert_eq!(
+                    sums.blocks,
+                    1 + part.theta.len() + if part.m_derivatives() { 2 } else { 0 }
+                );
+                let truncation = if matches!(part.support, Support::Unbounded(_)) {
+                    0.2 * options.rtol
+                } else {
+                    0.0
+                };
+                averages(part, &sums, len, truncation)
+            }
+        };
+        let f = part.w / w_total * part.v / part.v_total;
+        for i in 0..len {
+            v[i] += f * avg.v[i];
+            e[i] += f * avg.e[i];
+        }
+        for (col, dv, dev) in &avg.d {
+            for i in 0..len {
+                d[*col][i] += f * dv[i];
+                de[*col][i] += f * dev[i];
+            }
+        }
+        averages_of.push(avg);
+    }
+    // The weight derivatives: ∂f_p/∂w and ∂f_p/∂v of the number fractions.
+    for (col, p) in options.derivatives.iter().enumerate() {
+        if p.kind != ParamKind::Weight {
+            continue;
+        }
+        for (part, avg) in parts.iter().zip(&averages_of) {
+            let f = part.w / w_total * part.v / part.v_total;
+            let df = match p.mode {
+                None => {
+                    let own = if part.component == p.component {
+                        part.v / part.v_total
+                    } else {
+                        0.0
+                    };
+                    (own - f) / w_total
+                }
+                Some(j) => {
+                    if part.component != p.component {
+                        0.0
+                    } else {
+                        let own = if part.mode == Some(j) { 1.0 } else { 0.0 };
+                        part.w / w_total * (own - part.v / part.v_total) / part.v_total
+                    }
+                }
+            };
+            for i in 0..len {
+                d[col][i] += df * avg.v[i];
+                de[col][i] += df.abs() * avg.e[i];
+            }
+        }
+    }
+    let lin = |i: usize| Lin {
+        v: v[i],
+        e: e[i],
+        d: d.iter().map(|c| c[i]).collect(),
+        de: de.iter().map(|c| c[i]).collect(),
+    };
+    let greek = options.greek.is_some();
+    let outputs = assemble(&lin, greek, order, &layout, &d, &de, &v, &e, options)?;
+    Ok(EnsembleResult {
+        values: outputs.pick(|l| l.v),
+        errors: outputs.pick(|l| l.e),
+        derivatives: (0..n_cols).map(|k| outputs.pick(|l| l.d[k])).collect(),
+        derivative_errors: (0..n_cols).map(|k| outputs.pick(|l| l.de[k])).collect(),
+        params: options.derivatives.clone(),
+        converged: firsts.iter().all(|p| p.converged),
+        panels: firsts.iter().map(|p| p.panels.len()).sum(),
+        ranges: firsts.iter().map(|p| p.range).collect(),
+    })
+}
+
+/// Every output as a [`Lin`].
+struct LinOutputs {
+    scalars: [Lin; 11],
+    greek: Option<[Vec<Lin>; 6]>,
+    phase: Option<[Vec<Lin>; 4]>,
+}
+
+impl LinOutputs {
+    fn pick(&self, f: impl Fn(&Lin) -> f64) -> Outputs {
+        let s: Vec<f64> = self.scalars.iter().map(&f).collect();
+        let arr = |v: &Vec<Lin>| v.iter().map(&f).collect::<Vec<f64>>();
+        Outputs {
+            c_ext: s[0],
+            c_sca: s[1],
+            c_abs: s[2],
+            c_back: s[3],
+            ssa: s[4],
+            g: s[5],
+            lidar_ratio: s[6],
+            depolarization: s[7],
+            effective_radius: s[8],
+            mean_area: s[9],
+            mean_volume: s[10],
+            greek: self.greek.as_ref().map(|g| GreekCoefficients {
+                alpha1: arr(&g[0]),
+                alpha2: arr(&g[1]),
+                alpha3: arr(&g[2]),
+                alpha4: arr(&g[3]),
+                beta1: arr(&g[4]),
+                beta2: arr(&g[5]),
+            }),
+            phase: self.phase.as_ref().map(|p| PhaseValues {
+                f11: arr(&p[0]),
+                f12: arr(&p[1]),
+                f33: arr(&p[2]),
+                f34: arr(&p[3]),
+            }),
+        }
+    }
+}
+
+/// The outputs from the mixed averages.
+#[allow(clippy::too_many_arguments)]
+fn assemble(
+    lin: &dyn Fn(usize) -> Lin,
+    greek: bool,
+    order: usize,
+    layout: &Layout<'_>,
+    d: &[Vec<f64>],
+    de: &[Vec<f64>],
+    v: &[f64],
+    e: &[f64],
+    options: &EnsembleOptions,
+) -> Result<LinOutputs> {
+    let n_cols = d.len();
+    let (ext, sca, scag, abs, back, r2, r3) = (
+        lin(EXT),
+        lin(SCA),
+        lin(SCAG),
+        lin(ABS),
+        lin(BACK),
+        lin(R2),
+        lin(R3),
+    );
+    let scalars = [
+        ext.clone(),
+        sca.clone(),
+        abs,
+        back.clone(),
+        sca.div(&ext),
+        scag.div(&sca),
+        ext.div(&back).scale(4.0 * PI),
+        Lin::constant(0.0, n_cols),
+        r3.div(&r2),
+        r2.scale(PI),
+        r3.scale(4.0 * PI / 3.0),
+    ];
+    let normalized = options.normalization == Normalization::Normalized;
+    let greek = if !greek {
+        None
+    } else {
+        let arrays = |g: &GreekCoefficients<f64>| -> [Vec<f64>; 6] {
+            [
+                g.alpha1.clone(),
+                g.alpha2.clone(),
+                g.alpha3.clone(),
+                g.alpha4.clone(),
+                g.beta1.clone(),
+                g.beta2.clone(),
+            ]
+        };
+        // Sum the bands' coefficients: values and derivatives, and the error bounds.
+        let mut out: [Vec<Lin>; 6] =
+            core::array::from_fn(|_| vec![Lin::constant(0.0, n_cols); order + 1]);
+        for band in &layout.bands {
+            let n_q = band.projector.nodes().len();
+            let start = band.start;
+            let table = |x: &[f64]| -> [Vec<f64>; 4] {
+                core::array::from_fn(|k| x[start + k * n_q..start + (k + 1) * n_q].to_vec())
+            };
+            let mut values = vec![table(v)];
+            values.extend(d.iter().map(|c| table(c)));
+            let mut errs = vec![table(e)];
+            errs.extend(de.iter().map(|c| table(c)));
+            let c = band
+                .projector
+                .project(&values.iter().map(view).collect::<Vec<_>>())?;
+            let ce = band
+                .projector
+                .project_bounds(&errs.iter().map(view).collect::<Vec<_>>())?;
+            let ca: Vec<[Vec<f64>; 6]> = c.iter().map(arrays).collect();
+            let cea: Vec<[Vec<f64>; 6]> = ce.iter().map(arrays).collect();
+            for (k, arr) in out.iter_mut().enumerate() {
+                for (l, lin) in arr.iter_mut().enumerate().take(ca[0][k].len()) {
+                    lin.v += ca[0][k][l];
+                    lin.e += cea[0][k][l];
+                    for col in 0..n_cols {
+                        lin.d[col] += ca[1 + col][k][l];
+                        lin.de[col] += cea[1 + col][k][l];
+                    }
+                }
+            }
+        }
+        if normalized {
+            let a0 = out[0][0].clone();
+            for arr in &mut out {
+                for l in arr.iter_mut() {
+                    *l = l.div(&a0);
+                }
+            }
+            out[0][0] = Lin::constant(1.0, n_cols);
+        }
+        Some(out)
+    };
+    let n_mu = layout.mu.len();
+    let phase = if n_mu == 0 {
+        None
+    } else {
+        let start = SCALARS;
+        Some(core::array::from_fn(|k| {
+            (0..n_mu)
+                .map(|j| {
+                    let l = lin(start + k * n_mu + j);
+                    if normalized { l.div(&sca) } else { l }
+                })
+                .collect()
+        }))
+    };
+    Ok(LinOutputs {
+        scalars,
+        greek,
+        phase,
+    })
+}
diff --git a/scatrs/src/ensemble/kronrod.rs b/scatrs/src/ensemble/kronrod.rs
new file mode 100644
index 0000000..109b569
--- /dev/null
+++ b/scatrs/src/ensemble/kronrod.rs
@@ -0,0 +1,80 @@
+//! The 15-point Gauss–Kronrod rule and its embedded 7-point Gauss rule (QUADPACK's `qk15`).
+
+/// The abscissae of the 15-point Kronrod rule on [−1, 1] in descending order of the positive
+/// half: ±`NODES[i]`, i < 7, and 0. The Gauss nodes are those with odd i, and 0.
+const NODES: [f64; 8] = [
+    0.991_455_371_120_812_6,
+    0.949_107_912_342_758_5,
+    0.864_864_423_359_769_1,
+    0.741_531_185_599_394_4,
+    0.586_087_235_467_691_1,
+    0.405_845_151_377_397_2,
+    0.207_784_955_007_898_47,
+    0.0,
+];
+
+/// The Kronrod weights of `NODES`.
+const KRONROD: [f64; 8] = [
+    0.022_935_322_010_529_225,
+    0.063_092_092_629_978_55,
+    0.104_790_010_322_250_18,
+    0.140_653_259_715_525_92,
+    0.169_004_726_639_267_9,
+    0.190_350_578_064_785_4,
+    0.204_432_940_075_298_9,
+    0.209_482_141_084_727_83,
+];
+
+/// The Gauss weights of `NODES[1]`, `NODES[3]`, `NODES[5]` and `NODES[7]` = 0.
+const GAUSS: [f64; 4] = [
+    0.129_484_966_168_869_7,
+    0.279_705_391_489_276_7,
+    0.381_830_050_505_118_9,
+    0.417_959_183_673_469_4,
+];
+
+/// The number of nodes of a panel.
+pub(crate) const POINTS: usize = 15;
+
+/// The nodes of the panel [a, b] (ascending) with their Kronrod and Gauss weights (the Gauss
+/// weight is 0 at the nodes that only the Kronrod rule uses).
+pub(crate) fn panel(a: f64, b: f64) -> [(f64, f64, f64); POINTS] {
+    let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
+    let mut out = [(0.0, 0.0, 0.0); POINTS];
+    for i in 0..7 {
+        let gauss = if i % 2 == 1 { h * GAUSS[i / 2] } else { 0.0 };
+        let k = h * KRONROD[i];
+        out[i] = (c - h * NODES[i], k, gauss);
+        out[POINTS - 1 - i] = (c + h * NODES[i], k, gauss);
+    }
+    out[7] = (c, h * KRONROD[7], h * GAUSS[3]);
+    out
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn the_rules_integrate_polynomials_exactly() {
+        // Kronrod: degree 22; Gauss: degree 13.
+        let p = panel(-0.3, 1.7);
+        for k in 0..=22 {
+            let exact = (1.7f64.powi(k + 1) - (-0.3f64).powi(k + 1)) / f64::from(k + 1);
+            let kr: f64 = p.iter().map(|&(t, w, _)| w * t.powi(k)).sum();
+            assert!(
+                (kr - exact).abs() <= 1e-14 * exact.abs().max(1.0),
+                "Kronrod, degree {k}"
+            );
+            if k <= 13 {
+                let g: f64 = p.iter().map(|&(t, _, w)| w * t.powi(k)).sum();
+                assert!(
+                    (g - exact).abs() <= 1e-14 * exact.abs().max(1.0),
+                    "Gauss, degree {k}"
+                );
+            }
+        }
+        assert!(p.windows(2).all(|w| w[0].0 < w[1].0));
+        assert_eq!(p.iter().filter(|&&(_, _, g)| g != 0.0).count(), 7);
+    }
+}
diff --git a/scatrs/src/ensemble/lin.rs b/scatrs/src/ensemble/lin.rs
new file mode 100644
index 0000000..c94cb34
--- /dev/null
+++ b/scatrs/src/ensemble/lin.rs
@@ -0,0 +1,85 @@
+//! Values with derivatives and error estimates, propagated to first order through the
+//! ratios that turn integrals into ensemble outputs.
+
+/// A value, its error estimate, its derivatives with respect to the requested parameters,
+/// and their error estimates.
+#[derive(Clone, Debug, PartialEq)]
+pub(crate) struct Lin {
+    pub(crate) v: f64,
+    pub(crate) e: f64,
+    pub(crate) d: Vec<f64>,
+    pub(crate) de: Vec<f64>,
+}
+
+impl Lin {
+    /// An exact value without derivatives (`n` parameters).
+    pub(crate) fn constant(v: f64, n: usize) -> Self {
+        Self {
+            v,
+            e: 0.0,
+            d: vec![0.0; n],
+            de: vec![0.0; n],
+        }
+    }
+
+    /// `self`/`b`: the quotient rule, and the first-order error bounds
+    /// e(a/b) ≈ (e_a + |a/b| e_b)/|b| and
+    /// e(∂(a/b)) ≈ (e_∂a + |a/b| e_∂b + |∂(a/b)| e_b + e(a/b) |∂b|)/|b|.
+    pub(crate) fn div(&self, b: &Self) -> Self {
+        let v = self.v / b.v;
+        let e = (self.e + v.abs() * b.e) / b.v.abs();
+        let d: Vec<f64> = self
+            .d
+            .iter()
+            .zip(&b.d)
+            .map(|(da, db)| (da - v * db) / b.v)
+            .collect();
+        let de = (0..d.len())
+            .map(|k| {
+                (self.de[k] + v.abs() * b.de[k] + d[k].abs() * b.e + e * b.d[k].abs()) / b.v.abs()
+            })
+            .collect();
+        Self { v, e, d, de }
+    }
+
+    /// `c`·`self` for an exact constant `c`.
+    pub(crate) fn scale(&self, c: f64) -> Self {
+        Self {
+            v: c * self.v,
+            e: c.abs() * self.e,
+            d: self.d.iter().map(|x| c * x).collect(),
+            de: self.de.iter().map(|x| c.abs() * x).collect(),
+        }
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn the_quotient_rule_matches_finite_differences() {
+        // a(θ) = 3 + 2θ, b(θ) = 5 − θ at θ = 0.
+        let a = Lin {
+            v: 3.0,
+            e: 1e-3,
+            d: vec![2.0],
+            de: vec![1e-4],
+        };
+        let b = Lin {
+            v: 5.0,
+            e: 2e-3,
+            d: vec![-1.0],
+            de: vec![0.0],
+        };
+        let q = a.div(&b);
+        let h = 1e-6;
+        let fd = ((3.0 + 2.0 * h) / (5.0 - h) - (3.0 - 2.0 * h) / (5.0 + h)) / (2.0 * h);
+        assert!((q.d[0] - fd).abs() < 1e-9);
+        assert!((q.v - 0.6).abs() < 1e-15);
+        // Perturbing a and b by their errors moves a/b by the bound, to first order.
+        let worst = ((3.0f64 + 1e-3) / (5.0 - 2e-3) - 0.6).abs();
+        assert!(worst <= q.e * (1.0 + 1e-3) && worst >= q.e * (1.0 - 1e-3));
+        assert_eq!(a.scale(2.0).d, vec![4.0]);
+    }
+}
diff --git a/scatrs/src/ensemble/mod.rs b/scatrs/src/ensemble/mod.rs
new file mode 100644
index 0000000..e704131
--- /dev/null
+++ b/scatrs/src/ensemble/mod.rs
@@ -0,0 +1,513 @@
+//! Ensembles of spheres: size distributions, adaptive size integration and external
+//! mixtures.
+//!
+//! [`compute`] averages the optical properties of spheres over a [`SizeDistribution`] at each
+//! wavelength; [`compute_mixture`] averages over several [`Component`]s. The size integral is
+//! adaptive (Gauss–Kronrod panels in ln r, refined until the cross sections, g and the
+//! distribution moments meet `rtol`), and every output comes with an error estimate and,
+//! on request, derivatives with respect to the distribution parameters, the refractive
+//! indices and the mixture weights. See the theory chapter of the documentation.
+//!
+//! ```
+//! use num_complex::Complex;
+//! use scatrs::ensemble::{EnsembleOptions, GreekOrder, SizeDistribution, compute};
+//!
+//! let aerosol = SizeDistribution::LogNormal { median_radius: 0.08, width: 1.6 };
+//! let options = EnsembleOptions::new().with_greek(GreekOrder::Order(16));
+//! let result = compute(&aerosol, &[0.5], &[Complex::new(1.43, 1e-8)], &options)?;
+//! let r = &result[0];
+//! assert!(r.converged && r.values.ssa > 0.99);
+//! assert_eq!(r.values.greek.as_ref().unwrap().alpha1[0], 1.0);
+//! # Ok::<(), scatrs::Error>(())
+//! ```
+
+mod distribution;
+mod engine;
+mod kronrod;
+mod lin;
+mod moments;
+mod range;
+
+use num_complex::Complex;
+
+pub use distribution::SizeDistribution;
+
+use crate::mie::{GreekCoefficients, MAX_GREEK_ORDER, Normalization};
+use crate::{Error, Result};
+
+/// One component of an external mixture: spheres of one size distribution and refractive
+/// index (one per wavelength), in relative number concentration `weight`.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct Component {
+    /// The size distribution.
+    pub distribution: SizeDistribution,
+    /// The refractive index at each wavelength.
+    pub refractive_index: Vec<Complex<f64>>,
+    /// The relative number concentration, ≥ 0.
+    pub weight: f64,
+}
+
+impl Component {
+    /// A component of weight `weight`.
+    pub fn new(
+        distribution: SizeDistribution,
+        refractive_index: Vec<Complex<f64>>,
+        weight: f64,
+    ) -> Self {
+        Self {
+            distribution,
+            refractive_index,
+            weight,
+        }
+    }
+}
+
+/// The orders of the Greek coefficients to compute.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+#[non_exhaustive]
+pub enum GreekOrder {
+    /// Every order, up to 2N for the largest number of terms N at any size node.
+    All,
+    /// Orders 0 to L (exactly 0 above 2N).
+    Order(usize),
+}
+
+/// What a derivative is taken with respect to.
+#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
+#[non_exhaustive]
+pub enum ParamKind {
+    /// A parameter of the size distribution, by its name in
+    /// [`SizeDistribution::parameter_names`].
+    Distribution(&'static str),
+    /// The real part of the component's refractive index (at each wavelength).
+    RefrReal,
+    /// The imaginary part of the component's refractive index (at each wavelength).
+    RefrImag,
+    /// The weight of a mixture component (`mode` = `None`) or of a mode of a multimodal
+    /// distribution.
+    Weight,
+}
+
+/// A parameter the outputs can be differentiated with respect to.
+#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
+#[non_exhaustive]
+pub struct EnsembleParam {
+    /// The mixture component (0 for [`compute`]).
+    pub component: usize,
+    /// The mode of a multimodal distribution, if the parameter belongs to one.
+    pub mode: Option<usize>,
+    /// The parameter.
+    pub kind: ParamKind,
+}
+
+impl EnsembleParam {
+    /// A parameter of `component` (and `mode`).
+    pub fn new(component: usize, mode: Option<usize>, kind: ParamKind) -> Self {
+        Self {
+            component,
+            mode,
+            kind,
+        }
+    }
+
+    /// Its name: `median_radius`, `m_real`, `m_imag`, `weight`, …, prefixed by `mode{j}.`
+    /// for a mode and, with `mixture`, by `c{i}.` for the component.
+    pub fn name(&self, mixture: bool) -> String {
+        let base = match self.kind {
+            ParamKind::Distribution(name) => name,
+            ParamKind::RefrReal => "m_real",
+            ParamKind::RefrImag => "m_imag",
+            ParamKind::Weight => "weight",
+        };
+        let mut out = String::new();
+        if mixture {
+            out += &format!("c{}.", self.component);
+        }
+        if let Some(j) = self.mode {
+            out += &format!("mode{j}.");
+        }
+        out + base
+    }
+}
+
+/// The differentiable parameters of `distribution` as component `index`, with the component
+/// weight if `weight`.
+fn component_parameters(
+    index: usize,
+    distribution: &SizeDistribution,
+    weight: bool,
+) -> Vec<EnsembleParam> {
+    let mut out = Vec::new();
+    if let SizeDistribution::MultiModal { modes } = distribution {
+        for (j, (_, mode)) in modes.iter().enumerate() {
+            for name in mode.parameter_names() {
+                out.push(EnsembleParam::new(
+                    index,
+                    Some(j),
+                    ParamKind::Distribution(name),
+                ));
+            }
+            out.push(EnsembleParam::new(index, Some(j), ParamKind::Weight));
+        }
+    } else {
+        for name in distribution.parameter_names() {
+            out.push(EnsembleParam::new(
+                index,
+                None,
+                ParamKind::Distribution(name),
+            ));
+        }
+    }
+    out.push(EnsembleParam::new(index, None, ParamKind::RefrReal));
+    out.push(EnsembleParam::new(index, None, ParamKind::RefrImag));
+    if weight {
+        out.push(EnsembleParam::new(index, None, ParamKind::Weight));
+    }
+    out
+}
+
+/// Every parameter of a single distribution, for [`compute`].
+pub fn parameters(distribution: &SizeDistribution) -> Vec<EnsembleParam> {
+    component_parameters(0, distribution, false)
+}
+
+/// Every parameter of a mixture, for [`compute_mixture`].
+pub fn mixture_parameters(components: &[Component]) -> Vec<EnsembleParam> {
+    components
+        .iter()
+        .enumerate()
+        .flat_map(|(i, c)| component_parameters(i, &c.distribution, true))
+        .collect()
+}
+
+/// Options for [`compute`] and [`compute_mixture`].
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct EnsembleOptions {
+    /// The relative tolerance of the size integration, in [1e-13, 0.1]; default 1e-6.
+    pub rtol: f64,
+    /// The most panels per component (or mode) and wavelength; default 4096. Reaching it sets
+    /// [`EnsembleResult::converged`] to false.
+    pub max_panels: usize,
+    /// The Greek coefficients to compute, if any.
+    pub greek: Option<GreekOrder>,
+    /// Cosines of the scattering angle at which to compute the phase matrix.
+    pub mu: Vec<f64>,
+    /// The form of the phase matrix and the Greek coefficients.
+    pub normalization: Normalization,
+    /// The parameters to differentiate with respect to.
+    pub derivatives: Vec<EnsembleParam>,
+    /// Compute at sizes outside the validated domain of the Mie solution instead of
+    /// returning [`Error::OutsideValidatedDomain`].
+    pub allow_unvalidated: bool,
+    /// Also refine the size integration on the Greek coefficients up to this order.
+    pub refine_order: Option<usize>,
+    /// Also refine on ⟨C_back⟩ (to rtol of the larger of ⟨C_back⟩ and ⟨C_sca⟩), which makes
+    /// its error estimate (and the lidar ratio's) a calibrated bound. Off by default: the
+    /// backscattering of weakly absorbing spheres is a sum of narrow glory ripples that can
+    /// take far more panels than the cross sections; without it the estimate is indicative.
+    pub refine_backscatter: bool,
+}
+
+impl Default for EnsembleOptions {
+    fn default() -> Self {
+        Self {
+            rtol: 1e-6,
+            max_panels: 4096,
+            greek: None,
+            mu: Vec::new(),
+            normalization: Normalization::Normalized,
+            derivatives: Vec::new(),
+            allow_unvalidated: false,
+            refine_order: None,
+            refine_backscatter: false,
+        }
+    }
+}
+
+impl EnsembleOptions {
+    /// The defaults.
+    pub fn new() -> Self {
+        Self::default()
+    }
+
+    /// With relative tolerance `rtol`.
+    #[must_use]
+    pub fn with_rtol(mut self, rtol: f64) -> Self {
+        self.rtol = rtol;
+        self
+    }
+
+    /// With at most `max_panels` panels per component and wavelength.
+    #[must_use]
+    pub fn with_max_panels(mut self, max_panels: usize) -> Self {
+        self.max_panels = max_panels;
+        self
+    }
+
+    /// With the Greek coefficients of `order`.
+    #[must_use]
+    pub fn with_greek(mut self, order: GreekOrder) -> Self {
+        self.greek = Some(order);
+        self
+    }
+
+    /// With the phase matrix at the cosines `mu`.
+    #[must_use]
+    pub fn with_mu(mut self, mu: Vec<f64>) -> Self {
+        self.mu = mu;
+        self
+    }
+
+    /// With the phase matrix and Greek coefficients in `normalization`.
+    #[must_use]
+    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
+        self.normalization = normalization;
+        self
+    }
+
+    /// With the derivatives with respect to `params`.
+    #[must_use]
+    pub fn with_derivatives(mut self, params: Vec<EnsembleParam>) -> Self {
+        self.derivatives = params;
+        self
+    }
+
+    /// Allowing (or not) sizes outside the validated domain.
+    #[must_use]
+    pub fn with_allow_unvalidated(mut self, allow: bool) -> Self {
+        self.allow_unvalidated = allow;
+        self
+    }
+
+    /// Also refining on the Greek coefficients up to `order`.
+    #[must_use]
+    pub fn with_refine_order(mut self, order: usize) -> Self {
+        self.refine_order = Some(order);
+        self
+    }
+
+    /// Also refining (or not) on ⟨C_back⟩.
+    #[must_use]
+    pub fn with_refine_backscatter(mut self, refine: bool) -> Self {
+        self.refine_backscatter = refine;
+        self
+    }
+}
+
+/// The phase-matrix elements at each requested cosine.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct PhaseValues {
+    /// F₁₁ (= F₂₂ for spheres).
+    pub f11: Vec<f64>,
+    /// F₁₂.
+    pub f12: Vec<f64>,
+    /// F₃₃ (= F₄₄ for spheres).
+    pub f33: Vec<f64>,
+    /// F₃₄.
+    pub f34: Vec<f64>,
+}
+
+/// The outputs of an ensemble at one wavelength (or their error estimates, or their
+/// derivatives with respect to one parameter). Cross sections are per particle, averaged
+/// over the number distribution, in the square of the length unit of the radii.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct Outputs {
+    /// ⟨C_ext⟩.
+    pub c_ext: f64,
+    /// ⟨C_sca⟩.
+    pub c_sca: f64,
+    /// ⟨C_abs⟩.
+    pub c_abs: f64,
+    /// ⟨C_back⟩, Bohren & Huffman's backscattering cross section (4π dσ/dΩ at 180°).
+    pub c_back: f64,
+    /// The single-scattering albedo ⟨C_sca⟩/⟨C_ext⟩.
+    pub ssa: f64,
+    /// The asymmetry parameter ⟨C_sca g⟩/⟨C_sca⟩.
+    pub g: f64,
+    /// The lidar ratio 4π⟨C_ext⟩/⟨C_back⟩.
+    pub lidar_ratio: f64,
+    /// The linear depolarization ratio at 180°: exactly 0 for spheres.
+    pub depolarization: f64,
+    /// The effective radius ⟨r³⟩/⟨r²⟩.
+    pub effective_radius: f64,
+    /// The mean geometric cross section ⟨πr²⟩.
+    pub mean_area: f64,
+    /// The mean volume ⟨4πr³/3⟩.
+    pub mean_volume: f64,
+    /// The Greek coefficients, if requested: normalized (α₁⁰ = 1) or unnormalized
+    /// (α₁⁰ = ⟨C_sca⟩), as in [`EnsembleOptions::normalization`].
+    pub greek: Option<GreekCoefficients<f64>>,
+    /// The phase matrix at [`EnsembleOptions::mu`], if requested: normalized (½∫F₁₁ dμ = 1)
+    /// or unnormalized (⟨C_sca F⟩).
+    pub phase: Option<PhaseValues>,
+}
+
+/// The result at one wavelength.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct EnsembleResult {
+    /// The outputs.
+    pub values: Outputs,
+    /// Their error estimates (non-negative; see the theory chapter).
+    pub errors: Outputs,
+    /// The derivatives with respect to each of [`params`](Self::params).
+    pub derivatives: Vec<Outputs>,
+    /// The error estimates of the derivatives.
+    pub derivative_errors: Vec<Outputs>,
+    /// The parameters of `derivatives`, as requested.
+    pub params: Vec<EnsembleParam>,
+    /// Whether every refined quantity met `rtol` within `max_panels` panels.
+    pub converged: bool,
+    /// The number of panels, summed over the components (and modes).
+    pub panels: usize,
+    /// The integration range [r_lo, r_hi] of each component (and mode), in order (a single
+    /// radius r gives (r, r)).
+    pub ranges: Vec<(f64, f64)>,
+}
+
+fn invalid(msg: String) -> Error {
+    Error::InvalidInput(msg)
+}
+
+/// Averages over `distribution` of spheres of refractive index `refractive_index[i]` at
+/// `wavelengths[i]` (in the unit of the radii).
+///
+/// # Errors
+///
+/// [`Error::InvalidInput`] for invalid distribution parameters, wavelengths, refractive
+/// indices or options, or a derivative parameter the distribution does not have;
+/// [`Error::OutsideValidatedDomain`] if a size node of the integration is outside the Mie
+/// solution's validated domain, unless [`EnsembleOptions::allow_unvalidated`].
+pub fn compute(
+    distribution: &SizeDistribution,
+    wavelengths: &[f64],
+    refractive_index: &[Complex<f64>],
+    options: &EnsembleOptions,
+) -> Result<Vec<EnsembleResult>> {
+    let component = Component::new(distribution.clone(), refractive_index.to_vec(), 1.0);
+    let allowed = parameters(distribution);
+    validate(
+        core::slice::from_ref(&component),
+        wavelengths,
+        options,
+        &allowed,
+    )?;
+    engine::run(&[component], wavelengths, options)
+}
+
+/// Averages over the external mixture `components`: cross sections and moments weighted by
+/// wᵢ/Σw, phase matrices by wᵢ C_sca,ᵢ.
+///
+/// # Errors
+///
+/// As [`compute`], and [`Error::InvalidInput`] for an empty mixture, a negative weight or
+/// weights that are all 0.
+pub fn compute_mixture(
+    components: &[Component],
+    wavelengths: &[f64],
+    options: &EnsembleOptions,
+) -> Result<Vec<EnsembleResult>> {
+    if components.is_empty() {
+        return Err(invalid("a mixture needs a component".into()));
+    }
+    let allowed = mixture_parameters(components);
+    validate(components, wavelengths, options, &allowed)?;
+    engine::run(components, wavelengths, options)
+}
+
+/// Checks the inputs common to [`compute`] and [`compute_mixture`].
+fn validate(
+    components: &[Component],
+    wavelengths: &[f64],
+    options: &EnsembleOptions,
+    allowed: &[EnsembleParam],
+) -> Result<()> {
+    if !(1e-13..=0.1).contains(&options.rtol) {
+        return Err(invalid(format!(
+            "rtol must be in [1e-13, 0.1], got {}",
+            options.rtol
+        )));
+    }
+    if options.max_panels == 0 {
+        return Err(invalid("max_panels must be at least 1".into()));
+    }
+    let greek_order = match options.greek {
+        Some(GreekOrder::Order(l)) => Some(l),
+        _ => None,
+    };
+    for order in [options.refine_order, greek_order].into_iter().flatten() {
+        if order > MAX_GREEK_ORDER {
+            return Err(invalid(format!(
+                "the Greek order must be at most {MAX_GREEK_ORDER}, got {order}"
+            )));
+        }
+    }
+    if let Some((j, v)) = options
+        .mu
+        .iter()
+        .enumerate()
+        .find(|&(_, v)| !(-1.0..=1.0).contains(v))
+    {
+        return Err(invalid(format!("cosine {j} must be in [-1, 1], got {v}")));
+    }
+    if let Some((i, w)) = wavelengths
+        .iter()
+        .enumerate()
+        .find(|&(_, w)| !(w.is_finite() && *w > 0.0))
+    {
+        return Err(invalid(format!(
+            "wavelength {i} must be finite and > 0, got {w}"
+        )));
+    }
+    let mut total = 0.0;
+    for (i, c) in components.iter().enumerate() {
+        c.distribution
+            .validate()
+            .map_err(|e| invalid(format!("component {i}: {e}")))?;
+        if let SizeDistribution::MultiModal { modes } = &c.distribution {
+            if modes
+                .iter()
+                .any(|(_, m)| matches!(m, SizeDistribution::MultiModal { .. }))
+            {
+                return Err(invalid(format!(
+                    "component {i}: the modes of a multimodal distribution cannot be multimodal"
+                )));
+            }
+        }
+        if c.refractive_index.len() != wavelengths.len() {
+            return Err(invalid(format!(
+                "component {i} has {} refractive indices for {} wavelengths",
+                c.refractive_index.len(),
+                wavelengths.len()
+            )));
+        }
+        if !(c.weight.is_finite() && c.weight >= 0.0) {
+            return Err(invalid(format!(
+                "the weight of component {i} must be finite and >= 0, got {}",
+                c.weight
+            )));
+        }
+        total += c.weight;
+    }
+    if total <= 0.0 {
+        return Err(invalid("the mixture weights must not all be 0".into()));
+    }
+    let mixture = components.len() > 1
+        || allowed
+            .iter()
+            .any(|p| p.kind == ParamKind::Weight && p.mode.is_none());
+    for p in &options.derivatives {
+        if !allowed.contains(p) {
+            return Err(invalid(format!(
+                "no parameter {} here; the parameters are {:?}",
+                p.name(mixture),
+                allowed.iter().map(|q| q.name(mixture)).collect::<Vec<_>>()
+            )));
+        }
+    }
+    Ok(())
+}
diff --git a/scatrs/src/ensemble/moments.rs b/scatrs/src/ensemble/moments.rs
new file mode 100644
index 0000000..5c81734
--- /dev/null
+++ b/scatrs/src/ensemble/moments.rs
@@ -0,0 +1,198 @@
+//! The distribution moments ⟨r²⟩ and ⟨r³⟩: adaptive integrals of the density alone (no Mie
+//! solutions), on a range whose volume tail is also bounded, independent of the wavelength.
+
+use super::distribution::SizeDistribution;
+use super::kronrod;
+use super::range::STEP;
+
+/// At most this many panels; far more than smooth densities need.
+const MAX_PANELS: usize = 100_000;
+
+/// ⟨r²⟩ and ⟨r³⟩ with their error estimates, and their derivatives (one per requested
+/// distribution parameter) with error estimates.
+#[derive(Clone, Debug)]
+pub(crate) struct Moments {
+    pub(crate) v: [f64; 2],
+    pub(crate) e: [f64; 2],
+    pub(crate) d: Vec<[f64; 2]>,
+    pub(crate) de: Vec<[f64; 2]>,
+}
+
+/// The Kronrod sums of [1, r², r³] and of their ∂u/∂θ-weighted forms (θ in `theta`, indices
+/// into the gradient) on the panel [a, b] of t = ln r, |Kronrod − Gauss| of the same, and the
+/// Kronrod sums of the absolute θ-weighted integrands.
+fn panel(dist: &SizeDistribution, theta: &[usize], a: f64, b: f64) -> [Vec<f64>; 3] {
+    let d = dist.prepare();
+    let n = 3 * (1 + theta.len());
+    let mut grad = vec![0.0; dist.parameter_names().len()];
+    let (mut k, mut g, mut abs) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
+    for &(t, wk, wg) in &kronrod::panel(a, b) {
+        let r = t.exp();
+        let u = d.eval(r, &mut grad);
+        let q = [1.0, r * r, r * r * r];
+        for (j, &qj) in q.iter().enumerate() {
+            k[j] += wk * u * r * qj;
+            g[j] += wg * u * r * qj;
+            for (bi, &gi) in theta.iter().enumerate() {
+                let o = 3 * (1 + bi) + j;
+                let x = grad[gi] * r * qj;
+                k[o] += wk * x;
+                g[o] += wg * x;
+                abs[o] += wk * x.abs();
+            }
+        }
+    }
+    let err = k.iter().zip(&g).map(|(x, y)| (x - y).abs()).collect();
+    [k, err, abs]
+}
+
+/// The moments over [lo, hi] (with `breaks` as panel edges), refined until every sum meets
+/// `rtol`; `truncation`·|value| is added to each estimate for the omitted tails.
+pub(crate) fn moments(
+    dist: &SizeDistribution,
+    theta: &[usize],
+    (lo, hi): (f64, f64),
+    breaks: &[f64],
+    rtol: f64,
+    truncation: f64,
+) -> Moments {
+    let (t_lo, t_hi) = (lo.ln(), hi.ln());
+    let mut edges = vec![t_lo];
+    edges.extend(
+        breaks
+            .iter()
+            .map(|r| r.ln())
+            .filter(|&t| t > t_lo && t < t_hi),
+    );
+    edges.push(t_hi);
+    let mut panels: Vec<(f64, f64)> = Vec::new();
+    for w in edges.windows(2) {
+        let n = ((w[1] - w[0]) / STEP).ceil().max(1.0) as usize;
+        let h = (w[1] - w[0]) / n as f64;
+        panels.extend((0..n).map(|i| (w[0] + i as f64 * h, w[0] + (i + 1) as f64 * h)));
+    }
+    let n_q = 3 * (1 + theta.len());
+    loop {
+        let sums: Vec<[Vec<f64>; 3]> = panels
+            .iter()
+            .map(|&(a, b)| panel(dist, theta, a, b))
+            .collect();
+        let total = |which: usize, j: usize| sums.iter().map(|s| s[which][j]).sum::<f64>();
+        let target: Vec<f64> = (0..n_q)
+            .map(|j| {
+                if j < 3 {
+                    rtol * total(0, j).abs()
+                } else {
+                    rtol * total(2, j)
+                }
+            })
+            .collect();
+        let n = panels.len();
+        let met = (0..n_q).all(|j| total(1, j) <= target[j]);
+        if met || n >= MAX_PANELS {
+            let s: Vec<f64> = (0..n_q).map(|j| total(0, j)).collect();
+            let e: Vec<f64> = (0..n_q).map(|j| total(1, j)).collect();
+            return averages(&s, &e, theta.len(), truncation);
+        }
+        let mut next = Vec::with_capacity(2 * n);
+        for (p, s) in panels.iter().zip(&sums) {
+            let bad = (0..n_q).any(|j| s[1][j] * n as f64 > target[j] && total(1, j) > target[j]);
+            if bad && p.1 - p.0 > 1e-10 {
+                let mid = 0.5 * (p.0 + p.1);
+                next.push((p.0, mid));
+                next.push((mid, p.1));
+            } else {
+                next.push(*p);
+            }
+        }
+        if next.len() == n {
+            let s: Vec<f64> = (0..n_q).map(|j| total(0, j)).collect();
+            let e: Vec<f64> = (0..n_q).map(|j| total(1, j)).collect();
+            return averages(&s, &e, theta.len(), truncation);
+        }
+        panels = next;
+    }
+}
+
+/// ⟨r^k⟩ = S_k/S₀ with the quotient rule and first-order error propagation.
+fn averages(s: &[f64], e: &[f64], nt: usize, truncation: f64) -> Moments {
+    let (z, ez) = (s[0], e[0]);
+    let v = [s[1] / z, s[2] / z];
+    let ev = [
+        (e[1] + v[0].abs() * ez) / z + truncation * v[0].abs(),
+        (e[2] + v[1].abs() * ez) / z + truncation * v[1].abs(),
+    ];
+    let mut d = Vec::with_capacity(nt);
+    let mut de = Vec::with_capacity(nt);
+    for b in 0..nt {
+        let o = 3 * (1 + b);
+        let dz = s[o];
+        let dv = [(s[o + 1] - v[0] * dz) / z, (s[o + 2] - v[1] * dz) / z];
+        let dev = core::array::from_fn(|k| {
+            (e[o + 1 + k] + v[k].abs() * e[o] + dv[k].abs() * ez + ev[k] * dz.abs()) / z
+                + truncation * dv[k].abs()
+        });
+        d.push(dv);
+        de.push(dev);
+    }
+    Moments { v, e: ev, d, de }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::super::range::{MOMENT_POWERS, cutoffs};
+    use super::*;
+
+    #[test]
+    fn moments_match_the_closed_forms() {
+        for dist in [
+            SizeDistribution::LogNormal {
+                median_radius: 0.08,
+                width: 1.6,
+            },
+            SizeDistribution::LogNormal {
+                median_radius: 1.0,
+                width: 1.05,
+            },
+            SizeDistribution::Gamma {
+                effective_radius: 2.0,
+                effective_variance: 0.1,
+            },
+            SizeDistribution::ModifiedGamma {
+                alpha: 2.0,
+                b: 15.0,
+                gamma: 0.5,
+            },
+            SizeDistribution::PowerLaw {
+                exponent: 3.5,
+                min_radius: 0.01,
+                max_radius: 10.0,
+            },
+        ] {
+            for rtol in [1e-6, 1e-10] {
+                let range = match dist {
+                    SizeDistribution::PowerLaw {
+                        min_radius,
+                        max_radius,
+                        ..
+                    } => (min_radius, max_radius),
+                    _ => {
+                        let d = dist.prepare();
+                        cutoffs(&d, d.reference, rtol / 10.0, MOMENT_POWERS).unwrap()
+                    }
+                };
+                let m = moments(&dist, &[], range, &[], rtol, 0.2 * rtol);
+                for (k, (&got, &est)) in m.v.iter().zip(&m.e).enumerate() {
+                    let want = dist.moment(2 + k as u32).unwrap();
+                    let err = (got - want).abs();
+                    assert!(
+                        err <= rtol * want,
+                        "{dist:?} rtol {rtol:e} r^{}: {err:e}",
+                        2 + k
+                    );
+                    assert!(err <= est, "{dist:?} rtol {rtol:e} r^{}: estimate", 2 + k);
+                }
+            }
+        }
+    }
+}
diff --git a/scatrs/src/ensemble/range.rs b/scatrs/src/ensemble/range.rs
new file mode 100644
index 0000000..124d83d
--- /dev/null
+++ b/scatrs/src/ensemble/range.rs
@@ -0,0 +1,150 @@
+//! Cut-offs of distributions with unbounded support.
+
+use super::distribution::Density;
+use super::kronrod;
+use crate::{Error, Result};
+
+/// The step of the march in t = ln r.
+pub(crate) const STEP: f64 = 0.5;
+
+/// At most this many steps on each side: e^(400·½) is far beyond any radius of interest.
+const MAX_STEPS: usize = 400;
+
+/// The moments whose tails bound the range of the Mie integration: the number (every
+/// per-particle average is renormalized over the range, so it moves by the omitted number
+/// fraction) and the area (the extinction of large spheres).
+pub(crate) const MIE_POWERS: &[i32] = &[0, 2];
+
+/// The moments whose tails bound the range of the moment integration: also the volume.
+pub(crate) const MOMENT_POWERS: &[i32] = &[0, 2, 3];
+
+/// The masses ∫ rᵏ u dr, k in `powers`, over [t₀, t₁] in t = ln r (dr = r dt).
+fn masses(d: &Density<'_>, t0: f64, t1: f64, powers: &[i32]) -> Vec<f64> {
+    let mut out = vec![0.0; powers.len()];
+    for &(t, w, _) in &kronrod::panel(t0, t1) {
+        let r = t.exp();
+        let wu = w * r * d.eval(r, &mut []);
+        for (o, &k) in out.iter_mut().zip(powers) {
+            *o += wu * r.powi(k);
+        }
+    }
+    out
+}
+
+/// The range [r_lo, r_hi] outside which each tail holds at most `eps` of each moment rᵏ, k in
+/// `powers`: from `center`, steps of `STEP` in ln r outward, until for each moment the last
+/// step's mass, with the geometric bound q/(1 − q) on the rest (q the ratio of the last two
+/// steps), is below `eps` of the mass found so far.
+///
+/// # Errors
+///
+/// [`Error::InvalidInput`] if a side needs more than 400 steps (the distribution is too
+/// broad or the density does not decay).
+pub(crate) fn cutoffs(
+    d: &Density<'_>,
+    center: f64,
+    eps: f64,
+    powers: &[i32],
+) -> Result<(f64, f64)> {
+    let t0 = center.ln();
+    let np = powers.len();
+    let mut total = vec![0.0; np];
+    let mut ends = [t0, t0];
+    let mut last = [vec![f64::INFINITY; np], vec![f64::INFINITY; np]];
+    let mut done = [false; 2];
+    for step in 0..MAX_STEPS {
+        for (side, sign) in [(0usize, -1.0f64), (1, 1.0)] {
+            if done[side] {
+                continue;
+            }
+            let a = ends[side];
+            let b = a + sign * STEP;
+            let mass = masses(d, a.min(b), a.max(b), powers);
+            if mass.iter().any(|m| !m.is_finite()) {
+                return Err(Error::InvalidInput(format!(
+                    "the size distribution's density is not finite near r = {:e}",
+                    b.exp()
+                )));
+            }
+            ends[side] = b;
+            let mut small = step >= 1;
+            for k in 0..np {
+                total[k] += mass[k];
+                let q = mass[k] / last[side][k];
+                last[side][k] = mass[k];
+                // At least two steps, so that q compares two masses of this side.
+                small &= mass[k] == 0.0 || (q < 1.0 && mass[k] / (1.0 - q) <= eps * total[k]);
+            }
+            done[side] = small;
+        }
+        if done[0] && done[1] {
+            return Ok((ends[0].exp(), ends[1].exp()));
+        }
+    }
+    Err(Error::InvalidInput(format!(
+        "the size distribution is too broad: its tails do not decay within e^{} of r = {center:e}",
+        MAX_STEPS as f64 * STEP
+    )))
+}
+
+#[cfg(test)]
+mod tests {
+    use super::super::SizeDistribution;
+    use super::*;
+
+    #[test]
+    fn the_tails_beyond_the_cutoffs_are_below_eps() {
+        for dist in [
+            SizeDistribution::LogNormal {
+                median_radius: 0.08,
+                width: 1.6,
+            },
+            SizeDistribution::LogNormal {
+                median_radius: 1.0,
+                width: 3.0,
+            },
+            SizeDistribution::Gamma {
+                effective_radius: 2.0,
+                effective_variance: 0.1,
+            },
+            SizeDistribution::Gamma {
+                effective_radius: 2.0,
+                effective_variance: 0.45,
+            },
+            SizeDistribution::ModifiedGamma {
+                alpha: 2.0,
+                b: 15.0,
+                gamma: 0.5,
+            },
+        ] {
+            let d = dist.prepare();
+            for eps in [1e-4, 1e-8, 1e-13] {
+                let (lo, hi) = cutoffs(&d, d.reference, eps, MOMENT_POWERS).unwrap();
+                // Each moment's mass inside against the whole (on a far wider range).
+                let inside = integrate(&d, lo.ln(), hi.ln(), 2000);
+                let wide = integrate(&d, lo.ln() - 40.0, hi.ln() + 40.0, 20_000);
+                for k in 0..3 {
+                    let outside = (wide[k] - inside[k]) / wide[k];
+                    assert!(
+                        outside <= 2.0 * eps,
+                        "{dist:?}, eps {eps:e}, power {}: outside {outside:e}",
+                        MOMENT_POWERS[k]
+                    );
+                }
+                assert!(lo < d.reference && d.reference < hi);
+            }
+        }
+    }
+
+    fn integrate(d: &Density<'_>, a: f64, b: f64, n: usize) -> [f64; 3] {
+        let h = (b - a) / n as f64;
+        let mut out = [0.0; 3];
+        for i in 0..n {
+            let m = masses(d, a + i as f64 * h, a + (i + 1) as f64 * h, MOMENT_POWERS);
+            for k in 0..3 {
+                out[k] += m[k];
+            }
+        }
+        out
+    }
+}
diff --git a/scatrs/src/lib.rs b/scatrs/src/lib.rs
index b5faf76..ed11930 100644
--- a/scatrs/src/lib.rs
+++ b/scatrs/src/lib.rs
@@ -5,6 +5,7 @@
 #![warn(missing_docs)]
 
 mod complex;
+pub mod ensemble;
 pub mod error;
 pub mod mie;
 pub mod scalar;
diff --git a/scatrs/src/mie/greek.rs b/scatrs/src/mie/greek.rs
index 12dab5a..a8d5daa 100644
--- a/scatrs/src/mie/greek.rs
+++ b/scatrs/src/mie/greek.rs
@@ -28,7 +28,7 @@ pub struct GreekCoefficients<T> {
 }
 
 impl<T: Real> GreekCoefficients<T> {
-    fn zeros(len: usize) -> Self {
+    pub(crate) fn zeros(len: usize) -> Self {
         let z = || vec![T::zero(); len];
         Self {
             alpha1: z(),
@@ -214,6 +214,74 @@ impl<T: Real> GreekProjector<T> {
     }
 }
 
+impl<T: Real> GreekProjector<T> {
+    /// Bounds on the coefficients' changes when each table value changes by at most the given
+    /// amount (the tables hold non-negative bounds on |ΔF₁₁|, |ΔF₁₂|, |ΔF₃₃|, |ΔF₃₄|): by the
+    /// triangle inequality, (2l + 1)/2·Σₖ wₖ|ΔF(μₖ)||dₗ(μₖ)| for α₁, α₄, β₁, β₂, and half the
+    /// sum of the bounds of α₂ + α₃ and α₂ − α₃ (each from |ΔF₁₁| + |ΔF₃₃|) for α₂ and α₃.
+    ///
+    /// # Errors
+    ///
+    /// As [`project`](Self::project).
+    pub fn project_bounds(
+        &self,
+        tables: &[PhaseTables<'_, T>],
+    ) -> Result<Vec<GreekCoefficients<T>>> {
+        let n = self.rule.nodes.len();
+        for t in tables {
+            let lens = [t.f11.len(), t.f12.len(), t.f33.len(), t.f34.len()];
+            if lens.iter().any(|&l| l != n) {
+                return Err(Error::InvalidInput(format!(
+                    "phase-matrix tables must have one value per node ({n}), got {lens:?}"
+                )));
+            }
+        }
+        if tables.is_empty() {
+            return Ok(Vec::new());
+        }
+        let len = self.order + 1;
+        let mut out: Vec<_> = tables
+            .iter()
+            .map(|_| GreekCoefficients::zeros(len))
+            .collect();
+        let mut d: [Vec<T>; 4] = core::array::from_fn(|_| vec![T::zero(); len]);
+        for (k, (&mu, &w)) in self.rule.nodes.iter().zip(&self.rule.weights).enumerate() {
+            let [d00, d22, d2m2, d02] = &mut d;
+            wigner_d(0, 0, mu, d00)?;
+            wigner_d(2, 2, mu, d22)?;
+            wigner_d(2, -2, mu, d2m2)?;
+            wigner_d(0, 2, mu, d02)?;
+            for (t, c) in tables.iter().zip(&mut out) {
+                let (f11, f33) = (w * t.f11[k], w * t.f33[k]);
+                let sum = f11 + f33;
+                let (f12, f34) = (w * t.f12[k], w * t.f34[k]);
+                for l in 0..len {
+                    c.alpha1[l] += f11 * d00[l].abs();
+                    c.alpha4[l] += f33 * d00[l].abs();
+                    c.alpha2[l] += sum * d22[l].abs(); // bound of α₂ + α₃ for now
+                    c.alpha3[l] += sum * d2m2[l].abs(); // bound of α₂ − α₃ for now
+                    c.beta1[l] += f12 * d02[l].abs();
+                    c.beta2[l] += f34 * d02[l].abs();
+                }
+            }
+        }
+        let half = T::from_f64(0.5);
+        for c in &mut out {
+            for l in 0..len {
+                let s = T::from_f64((2 * l + 1) as f64) * half;
+                let both = s * half * (c.alpha2[l] + c.alpha3[l]);
+                c.alpha1[l] *= s;
+                c.alpha4[l] *= s;
+                c.alpha2[l] = both;
+                c.alpha3[l] = both;
+                c.beta1[l] *= s;
+                c.beta2[l] *= s;
+            }
+        }
+        Ok(out)
+    }
+}
+
 /// The highest order [`MieSolution::greek`] accepts, 2²¹: twice [`MAX_MIE_ORDER`], since the
 /// coefficients of an N-term sphere vanish above order 2N.
 ///
diff --git a/scatrs/tests/ensemble.rs b/scatrs/tests/ensemble.rs
new file mode 100644
index 0000000..b16ede8
--- /dev/null
+++ b/scatrs/tests/ensemble.rs
@@ -0,0 +1,721 @@
+//! Ensembles of spheres: identities, mixtures, Jacobians against finite differences, the
+//! calibration of the error estimates against tight reference integrals, determinism, and
+//! the rejection of bad inputs.
+#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together
+
+use core::f64::consts::PI;
+
+use num_complex::Complex;
+use scatrs::Error;
+use scatrs::ensemble::{
+    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, ParamKind,
+    SizeDistribution, compute, compute_mixture, mixture_parameters, parameters,
+};
+use scatrs::mie::{GreekCoefficients, Mie, MieOptions, Normalization, PhaseBuffers};
+use scatrs::special::gauss_legendre;
+
+fn c(re: f64, im: f64) -> Complex<f64> {
+    Complex::new(re, im)
+}
+
+fn lognormal(rg: f64, width: f64) -> SizeDistribution {
+    SizeDistribution::LogNormal {
+        median_radius: rg,
+        width,
+    }
+}
+
+fn one(
+    d: &SizeDistribution,
+    lambda: f64,
+    m: Complex<f64>,
+    opts: &EnsembleOptions,
+) -> EnsembleResult {
+    compute(d, &[lambda], &[m], opts).unwrap().remove(0)
+}
+
+fn greek_arrays(g: &GreekCoefficients<f64>) -> [&Vec<f64>; 6] {
+    [
+        &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
+    ]
+}
+
+/// The scalar outputs, in a fixed order.
+fn scalars(o: &Outputs) -> [f64; 11] {
+    [
+        o.c_ext,
+        o.c_sca,
+        o.c_abs,
+        o.c_back,
+        o.ssa,
+        o.g,
+        o.lidar_ratio,
+        o.depolarization,
+        o.effective_radius,
+        o.mean_area,
+        o.mean_volume,
+    ]
+}
+
+#[test]
+fn a_monodisperse_ensemble_is_the_single_sphere() {
+    let (r, lambda, m) = (0.7, 0.55, c(1.5, 0.01));
+    let mu = vec![1.0, 0.3, -1.0];
+    let opts = EnsembleOptions::new()
+        .with_greek(GreekOrder::All)
+        .with_mu(mu.clone());
+    let e = one(
+        &SizeDistribution::Monodisperse { radius: r },
+        lambda,
+        m,
+        &opts,
+    );
+    let x = 2.0 * PI * r / lambda;
+    let mut mie = Mie::<f64>::new();
+    let s = mie.compute(x, m, &MieOptions::new()).unwrap();
+    let q = s.efficiencies();
+    let area = PI * r * r;
+    assert_eq!(e.values.c_ext, area * q.q_ext);
+    assert_eq!(e.values.c_sca, area * q.q_sca);
+    assert_eq!(e.values.c_abs, area * q.q_abs);
+    assert!((e.values.g - q.g).abs() < 1e-15);
+    assert!((e.values.effective_radius - r).abs() < 1e-15);
+    assert!(e.converged && e.panels == 0 && e.ranges == vec![(r, r)]);
+    let g = s.greek(None, Normalization::Normalized).unwrap();
+    let eg = e.values.greek.as_ref().unwrap();
+    for (a, b) in greek_arrays(eg).into_iter().zip(greek_arrays(&g)) {
+        assert_eq!(a.len(), b.len());
+        for l in 0..a.len() {
+            assert!(
+                (a[l] - b[l]).abs() <= 1e-13 * g.alpha1[1].abs().max(1.0),
+                "order {l}"
+            );
+        }
+    }
+    let mut f: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; mu.len()]);
+    {
+        let [f11, f12, f33, f34] = &mut f;
+        s.phase_matrix(
+            &mu,
+            Normalization::Normalized,
+            PhaseBuffers { f11, f12, f33, f34 },
+        )
+        .unwrap();
+    }
+    let p = e.values.phase.as_ref().unwrap();
+    for (a, b) in [&p.f11, &p.f12, &p.f33, &p.f34].into_iter().zip(&f) {
+        for j in 0..mu.len() {
+            assert!((a[j] - b[j]).abs() <= 1e-13 * f[0][0], "{a:?} vs {b:?}");
+        }
+    }
+}
+
+#[test]
+fn moments_and_normalizations_match_the_closed_forms() {
+    let opts = EnsembleOptions::new()
+        .with_rtol(1e-8)
+        .with_greek(GreekOrder::Order(8));
+    for d in [
+        lognormal(0.08, 1.6),
+        SizeDistribution::Gamma {
+            effective_radius: 0.5,
+            effective_variance: 0.15,
+        },
+        SizeDistribution::ModifiedGamma {
+            alpha: 2.0,
+            b: 15.0,
+            gamma: 0.5,
+        },
+        SizeDistribution::PowerLaw {
+            exponent: 3.0,
+            min_radius: 0.02,
+            max_radius: 1.0,
+        },
+    ] {
+        let e = one(&d, 0.5, c(1.45, 0.001), &opts);
+        let (m2, m3) = (d.moment(2).unwrap(), d.moment(3).unwrap());
+        let close = |a: f64, b: f64| (a - b).abs() <= 1e-7 * b;
+        assert!(close(e.values.mean_area, PI * m2), "{d:?} area");
+        assert!(
+            close(e.values.mean_volume, 4.0 * PI / 3.0 * m3),
+            "{d:?} volume"
+        );
+        assert!(close(e.values.effective_radius, m3 / m2), "{d:?} r_eff");
+        let g = e.values.greek.as_ref().unwrap();
+        assert_eq!(g.alpha1[0], 1.0);
+        assert!(
+            (g.alpha1[1] - 3.0 * e.values.g).abs() < 1e-12,
+            "{d:?}: alpha1[1] = 3g"
+        );
+        assert!((e.values.c_ext - e.values.c_sca - e.values.c_abs).abs() < 1e-12 * e.values.c_ext);
+        assert_eq!(e.values.depolarization, 0.0);
+        let s = e.values;
+        assert!((s.lidar_ratio - 4.0 * PI * s.c_ext / s.c_back).abs() < 1e-12 * s.lidar_ratio);
+    }
+}
+
+#[test]
+fn the_rayleigh_limit_is_the_sixth_moment() {
+    // x ≪ 1: Q_sca = (8/3) x⁴ |K|² (1 + O(x²)), K = (m² − 1)/(m² + 2), so
+    // ⟨C_sca⟩ = (8/3)(2π/λ)⁴ |K|² π ⟨r⁶⟩.
+    let (lambda, m) = (1.0, c(1.5, 0.0));
+    let d = lognormal(2e-4, 1.4);
+    let e = one(&d, lambda, m, &EnsembleOptions::new().with_rtol(1e-9));
+    let k = (m * m - 1.0) / (m * m + 2.0);
+    let want = 8.0 / 3.0 * (2.0 * PI / lambda).powi(4) * k.norm_sqr() * PI * d.moment(6).unwrap();
+    assert!(
+        (e.values.c_sca / want - 1.0).abs() < 1e-5,
+        "{} vs {want}",
+        e.values.c_sca
+    );
+    assert_eq!(e.values.c_abs, 0.0);
+    assert!(e.values.ssa == 1.0 && e.values.g.abs() < 1e-5);
+}
+
+#[test]
+fn the_normalized_phase_function_averages_to_one() {
+    let (nodes, weights) = {
+        let r = gauss_legendre::<f64>(400).unwrap();
+        (r.nodes, r.weights)
+    };
+    let opts = EnsembleOptions::new().with_mu(nodes.clone());
+    let e = one(&lognormal(0.3, 1.5), 0.5, c(1.33, 1e-6), &opts);
+    let f11 = &e.values.phase.as_ref().unwrap().f11;
+    let mean: f64 = f11.iter().zip(&weights).map(|(f, w)| f * w).sum::<f64>() / 2.0;
+    assert!((mean - 1.0).abs() < 1e-5, "{mean}");
+    // Unnormalized: ⟨C_sca F⟩, so the mean is ⟨C_sca⟩, and α₁⁰ = ⟨C_sca⟩.
+    let u = one(
+        &lognormal(0.3, 1.5),
+        0.5,
+        c(1.33, 1e-6),
+        &opts
+            .clone()
+            .with_normalization(Normalization::Unnormalized)
+            .with_greek(GreekOrder::Order(4)),
+    );
+    let fu = &u.values.phase.as_ref().unwrap().f11;
+    let mean: f64 = fu.iter().zip(&weights).map(|(f, w)| f * w).sum::<f64>() / 2.0;
+    assert!((mean / u.values.c_sca - 1.0).abs() < 1e-5);
+    let a0 = u.values.greek.as_ref().unwrap().alpha1[0];
+    assert!((a0 / u.values.c_sca - 1.0).abs() < 1e-12);
+}
+
+#[test]
+fn mixtures_average_with_the_number_weights() {
+    let (lambda, opts) = (
+        0.6,
+        EnsembleOptions::new()
+            .with_greek(GreekOrder::Order(10))
+            .with_mu(vec![0.5]),
+    );
+    let (d1, m1, w1) = (lognormal(0.1, 1.5), c(1.45, 0.0), 2.0);
+    let (d2, m2, w2) = (lognormal(1.0, 1.3), c(1.53, 0.01), 0.5);
+    let mix = compute_mixture(
+        &[
+            Component::new(d1.clone(), vec![m1], w1),
+            Component::new(d2.clone(), vec![m2], w2),
+        ],
+        &[lambda],
+        &opts,
+    )
+    .unwrap()
+    .remove(0);
+    let (a, b) = (one(&d1, lambda, m1, &opts), one(&d2, lambda, m2, &opts));
+    let (fa, fb) = (w1 / (w1 + w2), w2 / (w1 + w2));
+    let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * y.abs().max(1e-30);
+    assert!(close(
+        mix.values.c_ext,
+        fa * a.values.c_ext + fb * b.values.c_ext
+    ));
+    assert!(close(
+        mix.values.mean_volume,
+        fa * a.values.mean_volume + fb * b.values.mean_volume
+    ));
+    // Phase matrices average with wᵢ C_sca,ᵢ.
+    let sca = fa * a.values.c_sca + fb * b.values.c_sca;
+    let p = |r: &EnsembleResult| r.values.phase.as_ref().unwrap().f11[0];
+    let want = (fa * a.values.c_sca * p(&a) + fb * b.values.c_sca * p(&b)) / sca;
+    assert!((p(&mix) - want).abs() < 1e-10 * want);
+    let g = |r: &EnsembleResult| r.values.greek.as_ref().unwrap().beta1[4];
+    let want = (fa * a.values.c_sca * g(&a) + fb * b.values.c_sca * g(&b)) / sca;
+    assert!((g(&mix) - want).abs() < 1e-10 * want.abs().max(1e-6));
+    // A bimodal distribution is a mixture of its modes with one refractive index.
+    let bimodal = SizeDistribution::MultiModal {
+        modes: vec![(w1, d1.clone()), (w2, d2.clone())],
+    };
+    let bi = one(&bimodal, lambda, m1, &opts);
+    let both = compute_mixture(
+        &[
+            Component::new(d1, vec![m1], w1),
+            Component::new(d2, vec![m1], w2),
+        ],
+        &[lambda],
+        &opts,
+    )
+    .unwrap()
+    .remove(0);
+    assert_eq!(scalars(&bi.values), scalars(&both.values));
+}
+
+/// The derivative of every scalar output with respect to `p`, against a central difference of
+/// the whole computation at a tight tolerance.
+fn check_derivative(
+    run: &dyn Fn(f64, &EnsembleOptions) -> EnsembleResult,
+    p: EnsembleParam,
+    value: f64,
+    what: &str,
+) {
+    let tight = EnsembleOptions::new()
+        .with_rtol(1e-11)
+        .with_max_panels(50_000)
+        .with_greek(GreekOrder::Order(6));
+    let r = run(value, &tight.clone().with_derivatives(vec![p]));
+    let h = 1e-4 * value.abs().max(1e-3);
+    let (plus, minus) = (run(value + h, &tight), run(value - h, &tight));
+    let d = scalars(&r.derivatives[0]);
+    let (vp, vm, v) = (
+        scalars(&plus.values),
+        scalars(&minus.values),
+        scalars(&r.values),
+    );
+    for k in 0..11 {
+        let fd = (vp[k] - vm[k]) / (2.0 * h);
+        let scale = fd.abs().max(v[k].abs() / value.abs().max(1e-3)).max(1e-300);
+        assert!(
+            (d[k] - fd).abs() <= 1e-5 * scale,
+            "{what}, scalar {k}: {} vs {fd}",
+            d[k]
+        );
+    }
+    let gd = &r.derivatives[0].greek.as_ref().unwrap().beta1;
+    let (gp, gm) = (
+        &plus.values.greek.as_ref().unwrap().beta1,
+        &minus.values.greek.as_ref().unwrap().beta1,
+    );
+    for l in 0..gd.len() {
+        let fd = (gp[l] - gm[l]) / (2.0 * h);
+        assert!(
+            (gd[l] - fd).abs() <= 1e-5 * fd.abs().max(1e-3),
+            "{what}, beta1[{l}]"
+        );
+    }
+}
+
+#[test]
+fn jacobians_match_finite_differences() {
+    let lambda = 0.55;
+    let m = c(1.45, 0.02);
+    let base = lognormal(0.2, 1.6);
+    let ps = parameters(&base);
+    let median = ps[0];
+    check_derivative(
+        &|v, o| one(&lognormal(v, 1.6), lambda, m, o),
+        median,
+        0.2,
+        "median_radius",
+    );
+    check_derivative(
+        &|v, o| one(&lognormal(0.2, v), lambda, m, o),
+        ps[1],
+        1.6,
+        "width",
+    );
+    let re = EnsembleParam::new(0, None, ParamKind::RefrReal);
+    let im = EnsembleParam::new(0, None, ParamKind::RefrImag);
+    check_derivative(
+        &|v, o| one(&base, lambda, c(v, 0.02), o),
+        re,
+        1.45,
+        "m_real",
+    );
+    check_derivative(
+        &|v, o| one(&base, lambda, c(1.45, v), o),
+        im,
+        0.02,
+        "m_imag",
+    );
+    let gamma = |a: f64, b: f64| SizeDistribution::Gamma {
+        effective_radius: a,
+        effective_variance: b,
+    };
+    let gp = parameters(&gamma(0.5, 0.1));
+    check_derivative(
+        &|v, o| one(&gamma(v, 0.1), lambda, m, o),
+        gp[0],
+        0.5,
+        "effective_radius",
+    );
+    check_derivative(
+        &|v, o| one(&gamma(0.5, v), lambda, m, o),
+        gp[1],
+        0.1,
+        "effective_variance",
+    );
+    let mono = |r: f64| SizeDistribution::Monodisperse { radius: r };
+    let mp = parameters(&mono(0.3))[0];
+    check_derivative(&|v, o| one(&mono(v), lambda, m, o), mp, 0.3, "radius");
+    // Mixture weights.
+    let mix = |w: f64, o: &EnsembleOptions| {
+        compute_mixture(
+            &[
+                Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], w),
+                Component::new(lognormal(0.8, 1.3), vec![c(1.53, 0.01)], 0.5),
+            ],
+            &[lambda],
+            o,
+        )
+        .unwrap()
+        .remove(0)
+    };
+    let all = mixture_parameters(&[
+        Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], 1.0),
+        Component::new(lognormal(0.8, 1.3), vec![c(1.53, 0.01)], 0.5),
+    ]);
+    let w0 = *all
+        .iter()
+        .find(|p| p.kind == ParamKind::Weight && p.component == 0)
+        .unwrap();
+    check_derivative(&mix, w0, 2.0, "c0.weight");
+    // A mode weight.
+    let bi = |w: f64, o: &EnsembleOptions| {
+        one(
+            &SizeDistribution::MultiModal {
+                modes: vec![(w, lognormal(0.1, 1.5)), (0.5, lognormal(0.8, 1.3))],
+            },
+            lambda,
+            m,
+            o,
+        )
+    };
+    let mode_w = EnsembleParam::new(0, Some(0), ParamKind::Weight);
+    check_derivative(&bi, mode_w, 2.0, "mode0.weight");
+}
+
+/// The calibration of the error estimates: on the grid below, every true error is within its
+/// estimate (measured: at most 0.085 of it for the refined outputs, 0.37 for the others).
+const ESTIMATE_FACTOR: f64 = 1.0;
+
+/// Integration grid: every output of a result at rtol 1e-6 against the same at 1e-11.
+#[test]
+fn error_estimates_bound_the_true_errors() {
+    let cases = [
+        (lognormal(0.08, 1.6), c(1.43, 1e-8), 0.5),
+        (lognormal(5.0, 1.3), c(1.5, 0.01), 0.5),
+        (lognormal(0.001, 2.0), c(1.5, 0.0), 0.5),
+        (lognormal(0.5, 1.05), c(1.33, 1e-3), 0.6),
+        (
+            SizeDistribution::Gamma {
+                effective_radius: 2.0,
+                effective_variance: 0.1,
+            },
+            c(1.33, 1e-3),
+            0.65,
+        ),
+        (
+            SizeDistribution::ModifiedGamma {
+                alpha: 2.0,
+                b: 15.0,
+                gamma: 0.5,
+            },
+            c(1.5, 0.001),
+            0.5,
+        ),
+        (
+            SizeDistribution::PowerLaw {
+                exponent: 3.5,
+                min_radius: 0.01,
+                max_radius: 5.0,
+            },
+            c(1.5, 0.01),
+            0.5,
+        ),
+        (
+            SizeDistribution::Tabulated {
+                radius: vec![0.05, 0.1, 0.4, 1.0],
+                density: vec![0.0, 3.0, 1.0, 0.2],
+            },
+            c(2.0, 0.5),
+            0.4,
+        ),
+    ];
+    let mu = vec![1.0, 0.7, 0.0, -1.0];
+    // The worst ratio of true error to estimate, of refined and of other outputs.
+    let worst = std::cell::Cell::new([0.0f64; 2]);
+    let note = |refined: bool, err: f64, est: f64| {
+        let mut w = worst.get();
+        let i = usize::from(!refined);
+        if est > 0.0 {
+            w[i] = w[i].max(err / est);
+        }
+        worst.set(w);
+    };
+    for (d, m, lambda) in cases {
+        let params = parameters(&d);
+        // With backscattering refined, so that every estimate is a bound.
+        let opts = EnsembleOptions::new()
+            .with_greek(GreekOrder::Order(24))
+            .with_mu(mu.clone())
+            .with_derivatives(params.clone())
+            .with_refine_backscatter(true);
+        let a = one(&d, lambda, m, &opts);
+        let b = one(
+            &d,
+            lambda,
+            m,
+            &opts.clone().with_rtol(1e-11).with_max_panels(50_000),
+        );
+        assert!(a.converged && b.converged, "{d:?}");
+        let check = |x: &Outputs, y: &Outputs, e: &Outputs, what: &str| {
+            let (xs, ys, es) = (scalars(x), scalars(y), scalars(e));
+            for k in 0..11 {
+                let err = (xs[k] - ys[k]).abs();
+                let refined = matches!(k, 0 | 1 | 3 | 5 | 8 | 9 | 10) && what == "values";
+                let factor = ESTIMATE_FACTOR;
+                note(refined, err, es[k]);
+                assert!(
+                    err <= factor * es[k] + 1e-14 * ys[k].abs(),
+                    "{d:?} {what} scalar {k}: {err:e} > {factor} × {:e}",
+                    es[k]
+                );
+            }
+            let (gx, gy, ge) = (
+                x.greek.as_ref().unwrap(),
+                y.greek.as_ref().unwrap(),
+                e.greek.as_ref().unwrap(),
+            );
+            for ((ax, ay), ae) in greek_arrays(gx)
+                .into_iter()
+                .zip(greek_arrays(gy))
+                .zip(greek_arrays(ge))
+            {
+                for l in 0..ax.len() {
+                    let err = (ax[l] - ay[l]).abs();
+                    note(false, err, ae[l]);
+                    assert!(
+                        err <= ESTIMATE_FACTOR * ae[l] + 1e-13,
+                        "{d:?} {what} Greek l = {l}: {err:e} > {:e}",
+                        ae[l]
+                    );
+                }
+            }
+            let (px, py, pe) = (
+                x.phase.as_ref().unwrap(),
+                y.phase.as_ref().unwrap(),
+                e.phase.as_ref().unwrap(),
+            );
+            for (k, (vx, vy)) in [
+                (&px.f11, &py.f11),
+                (&px.f12, &py.f12),
+                (&px.f33, &py.f33),
+                (&px.f34, &py.f34),
+            ]
+            .into_iter()
+            .enumerate()
+            {
+                let ve = [&pe.f11, &pe.f12, &pe.f33, &pe.f34][k];
+                for j in 0..mu.len() {
+                    let err = (vx[j] - vy[j]).abs();
+                    note(false, err, ve[j]);
+                    assert!(
+                        err <= ESTIMATE_FACTOR * ve[j] + 1e-13 * vy[j].abs(),
+                        "{d:?} {what} phase {k} at {}",
+                        mu[j]
+                    );
+                }
+            }
+        };
+        check(&a.values, &b.values, &a.errors, "values");
+        for k in 0..params.len() {
+            check(
+                &a.derivatives[k],
+                &b.derivatives[k],
+                &a.derivative_errors[k],
+                "derivative",
+            );
+        }
+        // The refined quantities meet rtol.
+        let rel = |x: f64, y: f64| (x - y).abs() / y.abs();
+        assert!(rel(a.values.c_ext, b.values.c_ext) <= 1e-6, "{d:?} C_ext");
+        assert!(rel(a.values.c_sca, b.values.c_sca) <= 1e-6, "{d:?} C_sca");
+        assert!((a.values.g - b.values.g).abs() <= 2e-6, "{d:?} g");
+    }
+    let [refined, other] = worst.get();
+    eprintln!(
+        "worst error/estimate: {refined:.3} for the refined outputs, {other:.3} for the others"
+    );
+}
+
+/// A broad, weakly absorbing distribution converges without backscattering refined, and
+/// costs far more panels with it.
+#[test]
+fn backscattering_refinement_is_optional() {
+    let d = lognormal(1.0, 2.0);
+    let plain = one(&d, 1.0, c(1.33, 1e-4), &EnsembleOptions::new());
+    assert!(plain.converged, "{} panels", plain.panels);
+    let refined = one(
+        &d,
+        1.0,
+        c(1.33, 1e-4),
+        &EnsembleOptions::new().with_refine_backscatter(true),
+    );
+    assert!(refined.panels > plain.panels);
+    assert!((refined.values.c_ext / plain.values.c_ext - 1.0).abs() < 2e-6);
+}
+
+/// Without absorption the derivatives with respect to Im m have a spike at every resonance,
+/// which pass 1 cannot resolve: the result says so.
+#[test]
+fn unresolvable_derivatives_are_flagged() {
+    let d = SizeDistribution::Gamma {
+        effective_radius: 2.0,
+        effective_variance: 0.1,
+    };
+    let im = EnsembleParam::new(0, None, ParamKind::RefrImag);
+    let opts = EnsembleOptions::new()
+        .with_max_panels(512)
+        .with_derivatives(vec![im]);
+    let r = one(&d, 0.65, c(1.33, 0.0), &opts);
+    assert!(!r.converged && r.panels <= 512);
+    // The values alone converge.
+    assert!(one(&d, 0.65, c(1.33, 0.0), &EnsembleOptions::new()).converged);
+}
+
+#[test]
+fn results_do_not_depend_on_the_thread_count() {
+    let d = lognormal(0.2, 1.4);
+    let opts = EnsembleOptions::new()
+        .with_greek(GreekOrder::All)
+        .with_mu(vec![0.2])
+        .with_derivatives(parameters(&d));
+    let wavelengths = [0.4, 0.55, 0.8];
+    let m = vec![c(1.45, 0.001); 3];
+    let run = |threads: usize| {
+        rayon::ThreadPoolBuilder::new()
+            .num_threads(threads)
+            .build()
+            .unwrap()
+            .install(|| compute(&d, &wavelengths, &m, &opts).unwrap())
+    };
+    assert_eq!(run(1), run(4));
+}
+
+#[test]
+fn a_uniform_table_is_a_flat_power_law() {
+    let opts = EnsembleOptions::new().with_rtol(1e-9);
+    let table = SizeDistribution::Tabulated {
+        radius: vec![0.1, 0.5],
+        density: vec![1.0, 1.0],
+    };
+    let flat = SizeDistribution::PowerLaw {
+        exponent: 0.0,
+        min_radius: 0.1,
+        max_radius: 0.5,
+    };
+    let (a, b) = (
+        one(&table, 0.5, c(1.5, 0.0), &opts),
+        one(&flat, 0.5, c(1.5, 0.0), &opts),
+    );
+    assert!((a.values.c_ext / b.values.c_ext - 1.0).abs() < 1e-8);
+}
+
+#[test]
+fn bad_inputs_are_rejected() {
+    let d = lognormal(0.1, 1.5);
+    let m = [c(1.5, 0.0)];
+    let bad_opts = [
+        EnsembleOptions::new().with_rtol(0.5),
+        EnsembleOptions::new().with_rtol(1e-15),
+        EnsembleOptions::new().with_max_panels(0),
+        EnsembleOptions::new().with_mu(vec![1.5]),
+        EnsembleOptions::new().with_greek(GreekOrder::Order(usize::MAX)),
+        EnsembleOptions::new().with_derivatives(vec![EnsembleParam::new(
+            0,
+            None,
+            ParamKind::Distribution("alpha"),
+        )]),
+        EnsembleOptions::new().with_derivatives(vec![EnsembleParam::new(
+            1,
+            None,
+            ParamKind::RefrReal,
+        )]),
+    ];
+    for o in bad_opts {
+        let r = compute(&d, &[0.5], &m, &o);
+        assert!(matches!(r, Err(Error::InvalidInput(_))), "{o:?}: {r:?}");
+    }
+    let o = EnsembleOptions::new();
+    for (wl, mm) in [
+        (vec![0.0], m.to_vec()),
+        (vec![0.5, 0.6], m.to_vec()),
+        (vec![f64::NAN], m.to_vec()),
+    ] {
+        assert!(matches!(
+            compute(&d, &wl, &mm, &o),
+            Err(Error::InvalidInput(_))
+        ));
+    }
+    assert!(matches!(
+        compute(&lognormal(0.1, 0.9), &[0.5], &m, &o),
+        Err(Error::InvalidInput(_))
+    ));
+    assert!(matches!(
+        compute_mixture(&[], &[0.5], &o),
+        Err(Error::InvalidInput(_))
+    ));
+    let zero = [Component::new(d.clone(), m.to_vec(), 0.0)];
+    assert!(matches!(
+        compute_mixture(&zero, &[0.5], &o),
+        Err(Error::InvalidInput(_))
+    ));
+    let nested = SizeDistribution::MultiModal {
+        modes: vec![(
+            1.0,
+            SizeDistribution::MultiModal {
+                modes: vec![(1.0, d.clone())],
+            },
+        )],
+    };
+    assert!(matches!(
+        compute(&nested, &[0.5], &m, &o),
+        Err(Error::InvalidInput(_))
+    ));
+    // A size outside the validated domain names the wavelength and the radius.
+    let big = SizeDistribution::Monodisperse { radius: 2000.0 };
+    match compute(&big, &[0.5], &m, &o) {
+        Err(Error::OutsideValidatedDomain(msg)) => assert!(msg.contains("wavelength"), "{msg}"),
+        other => panic!("{other:?}"),
+    }
+    assert!(compute(&big, &[0.5], &m, &o.clone().with_allow_unvalidated(true)).is_ok());
+    // An invalid refractive index.
+    assert!(matches!(
+        compute(&d, &[0.5], &[c(1.5, -0.1)], &o),
+        Err(Error::InvalidInput(_))
+    ));
+    // No wavelengths: nothing to do.
+    assert!(compute(&d, &[], &[], &o).unwrap().is_empty());
+}
+
+#[test]
+fn derivative_parameters_are_named() {
+    let bi = SizeDistribution::MultiModal {
+        modes: vec![
+            (1.0, lognormal(0.1, 1.5)),
+            (0.5, SizeDistribution::Monodisperse { radius: 1.0 }),
+        ],
+    };
+    let names: Vec<String> = parameters(&bi).iter().map(|p| p.name(false)).collect();
+    assert_eq!(
+        names,
+        [
+            "mode0.median_radius",
+            "mode0.width",
+            "mode0.weight",
+            "mode1.radius",
+            "mode1.weight",
+            "m_real",
+            "m_imag"
+        ]
+    );
+    let mix = mixture_parameters(&[Component::new(lognormal(0.1, 1.5), vec![], 1.0)]);
+    assert_eq!(mix.last().unwrap().name(true), "c0.weight");
+}
````

- [ ] **Step 2: Verify.** `cargo test -p scatrs --release --lib ensemble` (8 unit tests) and `cargo test -p scatrs --release --test ensemble -- --nocapture` (13 tests; prints the worst error/estimate ratios); `cargo +1.99.0 clippy --workspace --all-targets -- -D warnings`; `cargo +1.85.0 check -p scatrs --lib`.

- [ ] **Step 3: Commit** — `feat(ensemble): size distributions, adaptive size integration and mixtures`

---

### Task 2: Python `scatrs.ensemble`

**Interfaces — Consumes:** Task 1. **Produces:** `_scatrs.ensemble_parameters(items, mixture) -> list[str]`, `_scatrs.ensemble_compute(items, wavelengths, mixture, greek, mu, normalized, derivatives, rtol, max_panels, allow_unvalidated, refine_backscatter) -> dict`; `scatrs.ensemble.{LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal, Component, compute, compute_mixture, EnsembleResult, EnsembleOutputs, EnsembleGreek, EnsemblePhase, SCALARS}`.

- [ ] **Step 1: Apply the diff:**

````diff
diff --git a/pyproject.toml b/pyproject.toml
index 03cc15f..346d1ae 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -74,8 +74,9 @@ extend-exclude = ["docs/plans", "docs/specs"]
 [tool.ruff.lint]
 select = ["E", "F", "W", "I", "UP", "B", "NPY", "RUF"]
 # Deliberate typography in docstrings and comments: minus sign and en dash (Gauss–Legendre), and
-# the mathematical notation of the Mie docstrings (ρ, ×, the prime of D′).
-allowed-confusables = ["−", "–", "ρ", "×", "′"]
+# the mathematical notation of the Mie docstrings (ρ, ×, the prime of D′) and of the size
+# distributions (α, γ, σ).
+allowed-confusables = ["−", "–", "ρ", "×", "′", "α", "γ", "σ"]
 
 [tool.ruff.lint.per-file-ignores]
 # l is the degree of d^l_{mn}, the notation of the papers the generator follows.
diff --git a/python/scatrs/__init__.py b/python/scatrs/__init__.py
index 4d1a245..99c2dc6 100644
--- a/python/scatrs/__init__.py
+++ b/python/scatrs/__init__.py
@@ -3,7 +3,7 @@
 from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
 from importlib.metadata import version as _version
 
-from scatrs import _scatrs, mie, special
+from scatrs import _scatrs, ensemble, mie, special
 from scatrs.errors import (
     InvalidInputError,
     NotConvergedError,
@@ -17,6 +17,7 @@ __all__ = [
     "OutsideValidatedDomainError",
     "ScatrsError",
     "__version__",
+    "ensemble",
     "mie",
     "special",
 ]
diff --git a/python/scatrs/_scatrs/__init__.pyi b/python/scatrs/_scatrs/__init__.pyi
index 6ce5bde..089f0bb 100644
--- a/python/scatrs/_scatrs/__init__.pyi
+++ b/python/scatrs/_scatrs/__init__.pyi
@@ -1,5 +1,5 @@
 # Stubs for the extension module scatrs._scatrs (a stub package, for its _testing submodule).
-from typing import Final
+from typing import Any, Final
 
 import numpy as np
 from numpy.typing import NDArray
@@ -13,6 +13,8 @@ __all__ = [
     "MAX_MIE_VALUES",
     "MAX_WIGNER_D_VALUES",
     "__version__",
+    "ensemble_compute",
+    "ensemble_parameters",
     "gauss_legendre",
     "mie_amplitudes",
     "mie_coefficients",
@@ -78,3 +80,19 @@ def mie_greek(
     allow_unvalidated: bool,
     normalized: bool,
 ) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64] | None]: ...
+def ensemble_parameters(
+    items: list[tuple[Any, NDArray[np.complex128], float]], mixture: bool
+) -> list[str]: ...
+def ensemble_compute(
+    items: list[tuple[Any, NDArray[np.complex128], float]],
+    wavelengths: NDArray[np.float64],
+    mixture: bool,
+    greek: int | None,
+    mu: list[float],
+    normalized: bool,
+    derivatives: list[int],
+    rtol: float,
+    max_panels: int,
+    allow_unvalidated: bool,
+    refine_backscatter: bool,
+) -> dict[str, Any]: ...
diff --git a/python/scatrs/ensemble.py b/python/scatrs/ensemble.py
new file mode 100644
index 0000000..5f01a91
--- /dev/null
+++ b/python/scatrs/ensemble.py
@@ -0,0 +1,505 @@
+"""Ensembles of spheres: size distributions, adaptive size integration and mixtures.
+
+:func:`compute` averages the optical properties of spheres over a size distribution at each
+wavelength; :func:`compute_mixture` averages over several :class:`Component` s. Radii and
+wavelengths are in the same (any) unit; cross sections are per particle, averaged over the
+number distribution, in that unit squared. See :doc:`/theory/ensembles` for the method and
+the error estimates.
+
+The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the cross
+sections, g, the backscattering and the distribution moments meet ``rtol``). Every output
+comes with an error estimate (:attr:`EnsembleResult.error`) and, with ``derivatives``, its
+derivatives with respect to the distribution parameters, ``"m_real"``, ``"m_imag"`` and the
+mixture weights (:meth:`EnsembleResult.d`).
+
+Each call runs in Rust with the GIL released, in parallel over wavelengths and size panels;
+results are bitwise the same for any number of threads. Results are frozen dataclasses of
+read-only arrays.
+"""
+
+from __future__ import annotations
+
+from collections.abc import Iterable, Sequence
+from dataclasses import dataclass, field
+from typing import Literal, SupportsIndex, TypeAlias
+
+import numpy as np
+from numpy.typing import ArrayLike, NDArray
+
+from scatrs import _args, _scatrs
+
+__all__ = [
+    "SCALARS",
+    "Component",
+    "EnsembleGreek",
+    "EnsembleOutputs",
+    "EnsemblePhase",
+    "EnsembleResult",
+    "Gamma",
+    "LogNormal",
+    "ModifiedGamma",
+    "Monodisperse",
+    "MultiModal",
+    "PowerLaw",
+    "Tabulated",
+    "compute",
+    "compute_mixture",
+]
+
+SCALARS = (
+    "c_ext",
+    "c_sca",
+    "c_abs",
+    "c_back",
+    "ssa",
+    "g",
+    "lidar_ratio",
+    "depolarization",
+    "effective_radius",
+    "mean_area",
+    "mean_volume",
+)
+"""The scalar outputs of :class:`EnsembleOutputs`, in order."""
+
+
+@dataclass(frozen=True)
+class LogNormal:
+    """n(r) ∝ exp(−(ln r − ln r_g)²/(2 ln²σ_g))/r: ``median_radius`` r_g > 0, ``width``
+    σ_g > 1."""
+
+    median_radius: float
+    width: float
+
+    def _spec(self) -> tuple:
+        return ("lognormal", float(self.median_radius), float(self.width))
+
+
+@dataclass(frozen=True)
+class Gamma:
+    """Hansen & Travis's gamma distribution, n(r) ∝ r^((1−3b)/b) exp(−r/(ab)):
+    ``effective_radius`` a > 0, ``effective_variance`` 0 < b < 1/2."""
+
+    effective_radius: float
+    effective_variance: float
+
+    def _spec(self) -> tuple:
+        return ("gamma", float(self.effective_radius), float(self.effective_variance))
+
+
+@dataclass(frozen=True)
+class ModifiedGamma:
+    """Deirmendjian's modified gamma distribution, n(r) ∝ r^α exp(−b r^γ): ``alpha`` > −1,
+    ``b`` > 0, ``gamma`` > 0."""
+
+    alpha: float
+    b: float
+    gamma: float
+
+    def _spec(self) -> tuple:
+        return ("modified_gamma", float(self.alpha), float(self.b), float(self.gamma))
+
+
+@dataclass(frozen=True)
+class PowerLaw:
+    """n(r) ∝ r^(−α) on [``min_radius``, ``max_radius``]: ``exponent`` α (the limits are not
+    differentiable)."""
+
+    exponent: float
+    min_radius: float
+    max_radius: float
+
+    def _spec(self) -> tuple:
+        return (
+            "power_law",
+            float(self.exponent),
+            float(self.min_radius),
+            float(self.max_radius),
+        )
+
+
+@dataclass(frozen=True)
+class Monodisperse:
+    """Spheres of one ``radius``."""
+
+    radius: float
+
+    def _spec(self) -> tuple:
+        return ("monodisperse", float(self.radius))
+
+
+@dataclass(frozen=True)
+class Tabulated:
+    """n(r) linear between the points (``radius[i]``, ``density[i]``) and 0 outside: radii
+    increasing, densities ≥ 0. No parameters."""
+
+    radius: Sequence[float]
+    density: Sequence[float]
+
+    def _spec(self) -> tuple:
+        return (
+            "tabulated",
+            [float(r) for r in np.asarray(self.radius, dtype=float).ravel()],
+            [float(d) for d in np.asarray(self.density, dtype=float).ravel()],
+        )
+
+
+@dataclass(frozen=True)
+class MultiModal:
+    """Σⱼ wⱼ nⱼ(r), each mode normalized: ``modes`` is a sequence of (weight wⱼ > 0, mode).
+    The modes' parameters are named ``"mode{j}.<name>"``, their weights ``"mode{j}.weight"``.
+    """
+
+    modes: Sequence[tuple[float, Distribution]]
+
+    def _spec(self) -> tuple:
+        return ("multimodal", [(float(w), d._spec()) for w, d in self.modes])
+
+
+Distribution: TypeAlias = (
+    LogNormal | Gamma | ModifiedGamma | PowerLaw | Monodisperse | Tabulated | MultiModal
+)
+
+
+@dataclass(frozen=True)
+class Component:
+    """One component of an external mixture: a size distribution, its refractive index
+    (broadcast to the wavelengths), and its relative number concentration ``weight`` ≥ 0."""
+
+    distribution: Distribution
+    refractive_index: ArrayLike
+    weight: float = 1.0
+
+
+def _frozen(a: np.ndarray) -> np.ndarray:
+    a.flags.writeable = False
+    return a.view()
+
+
+@dataclass(frozen=True, eq=False)
+class EnsembleGreek:
+    """Greek coefficients per wavelength, arrays of shape ``wavelength.shape + (L + 1,)``,
+    zero-padded to the largest order (see :class:`scatrs.mie.MieGreek` for the expansion)."""
+
+    alpha1: NDArray[np.float64]
+    alpha2: NDArray[np.float64]
+    alpha3: NDArray[np.float64]
+    alpha4: NDArray[np.float64]
+    beta1: NDArray[np.float64]
+    beta2: NDArray[np.float64]
+
+
+@dataclass(frozen=True, eq=False)
+class EnsemblePhase:
+    """The phase matrix per wavelength at the cosines ``mu``, arrays of shape
+    ``wavelength.shape + mu.shape``."""
+
+    f11: NDArray[np.float64]
+    f12: NDArray[np.float64]
+    f33: NDArray[np.float64]
+    f34: NDArray[np.float64]
+
+
+@dataclass(frozen=True, eq=False)
+class EnsembleOutputs:
+    """The outputs of an ensemble (or their error estimates, or their derivatives with
+    respect to one parameter), each an array of the wavelengths' shape.
+
+    Attributes
+    ----------
+    c_ext, c_sca, c_abs, c_back : numpy.ndarray
+        Number-averaged cross sections per particle; ``c_back`` is Bohren & Huffman's
+        backscattering cross section (4π times dσ/dΩ at 180°).
+    ssa, g : numpy.ndarray
+        Single-scattering albedo and asymmetry parameter.
+    lidar_ratio, depolarization : numpy.ndarray
+        4π ``c_ext``/``c_back``, and the linear depolarization ratio (exactly 0 for spheres).
+    effective_radius, mean_area, mean_volume : numpy.ndarray
+        ⟨r³⟩/⟨r²⟩, ⟨πr²⟩ and ⟨4πr³/3⟩.
+    greek : EnsembleGreek or None
+        The Greek coefficients, if requested.
+    phase : EnsemblePhase or None
+        The phase matrix, if requested.
+    """
+
+    c_ext: NDArray[np.float64]
+    c_sca: NDArray[np.float64]
+    c_abs: NDArray[np.float64]
+    c_back: NDArray[np.float64]
+    ssa: NDArray[np.float64]
+    g: NDArray[np.float64]
+    lidar_ratio: NDArray[np.float64]
+    depolarization: NDArray[np.float64]
+    effective_radius: NDArray[np.float64]
+    mean_area: NDArray[np.float64]
+    mean_volume: NDArray[np.float64]
+    greek: EnsembleGreek | None = None
+    phase: EnsemblePhase | None = None
+
+
+@dataclass(frozen=True, eq=False)
+class EnsembleResult(EnsembleOutputs):
+    """:class:`EnsembleOutputs` with their error estimates and derivatives.
+
+    Attributes
+    ----------
+    error : EnsembleOutputs
+        Non-negative error estimates of every output (see :doc:`/validation/accuracy`).
+    converged : numpy.ndarray
+        bool per wavelength: whether every refined quantity met ``rtol`` within
+        ``max_panels`` panels.
+    panels : numpy.ndarray
+        int64 per wavelength: the number of size panels.
+    params : tuple of str
+        The parameters :meth:`d` accepts.
+    """
+
+    error: EnsembleOutputs | None = None
+    converged: NDArray[np.bool_] | None = None
+    panels: NDArray[np.int64] | None = None
+    params: tuple[str, ...] = ()
+    _d: tuple[EnsembleOutputs, ...] = field(default=(), repr=False)
+    _d_errors: tuple[EnsembleOutputs, ...] = field(default=(), repr=False)
+
+    def _index(self, param: str) -> int:
+        if param not in self.params:
+            raise ValueError(f"no derivative with respect to {param!r}; computed: {self.params}")
+        return self.params.index(param)
+
+    def d(self, param: str) -> EnsembleOutputs:
+        """The derivatives with respect to ``param``.
+
+        Raises
+        ------
+        ValueError
+            If the derivative was not requested.
+        """
+        return self._d[self._index(param)]
+
+    def d_error(self, param: str) -> EnsembleOutputs:
+        """The error estimates of the derivatives with respect to ``param``.
+
+        Raises
+        ------
+        ValueError
+            If the derivative was not requested.
+        """
+        return self._d_errors[self._index(param)]
+
+
+def _outputs(
+    raw: dict, which: str, k: int | None, shape: tuple[int, ...], mu_shape
+) -> EnsembleOutputs:
+    """The outputs of kind ``which`` ("" for values, "_errors", "_d" with derivative ``k``,
+    "_d_errors")."""
+    key = {"": "values", "_errors": "errors", "_d": "d", "_d_errors": "d_errors"}[which]
+    rows = raw[key] if k is None else raw[key][k]
+    scalars = [_frozen(np.ascontiguousarray(r).reshape(shape)) for r in rows]
+    greek = phase = None
+    if "greek" in raw:
+        g = raw["greek" + which]
+        g = g if k is None else g[k]
+        greek = EnsembleGreek(*(_frozen(a.reshape((*shape, a.shape[-1]))) for a in g))
+    if "phase" in raw:
+        p = raw["phase" + which]
+        p = p if k is None else p[k]
+        phase = EnsemblePhase(*(_frozen(a.reshape((*shape, *mu_shape))) for a in p))
+    return EnsembleOutputs(*scalars, greek=greek, phase=phase)
+
+
+def _specs(components: Sequence[Component], n: int) -> list:
+    out = []
+    for i, c in enumerate(components):
+        if not isinstance(c, Component):
+            raise TypeError(f"component {i} must be a Component, not {type(c).__name__}")
+        m = _args.complex_array(c.refractive_index, "refractive_index")
+        m = np.ascontiguousarray(np.broadcast_to(m.ravel() if m.ndim else m, (n,)))
+        out.append((c.distribution._spec(), m, float(c.weight)))
+    return out
+
+
+def _greek_order(greek: SupportsIndex | Literal["all"] | None) -> int | None:
+    if greek is None:
+        return None
+    if isinstance(greek, str):
+        if greek != "all":
+            raise ValueError(f"greek must be an order, 'all' or None, not {greek!r}")
+        return -1
+    return _args.size(greek, "greek")
+
+
+def _run(
+    items: list,
+    wavelength: ArrayLike,
+    mixture: bool,
+    greek: SupportsIndex | Literal["all"] | None,
+    mu: ArrayLike | None,
+    normalized: bool,
+    derivatives: bool | Sequence[str],
+    rtol: float,
+    max_panels: SupportsIndex,
+    allow_unvalidated: bool,
+    refine_backscatter: bool,
+) -> EnsembleResult:
+    wl = _args.real_array(wavelength, "wavelength")
+    shape = wl.shape
+    names = tuple(_scatrs.ensemble_parameters(items, mixture))
+    if isinstance(derivatives, (bool, np.bool_)):
+        params = names if derivatives else ()
+    elif isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
+        raise TypeError(
+            "derivatives must be a bool or a sequence of parameter names, "
+            f"not {type(derivatives).__name__}"
+        )
+    else:
+        params = tuple(derivatives)
+        for p in params:
+            if p not in names:
+                raise ValueError(f"unknown parameter {p!r}; expected one of {names}")
+    mus = None if mu is None else _args.real_array(mu, "mu")
+    raw = _scatrs.ensemble_compute(
+        items,
+        np.ascontiguousarray(wl.ravel()),
+        mixture,
+        _greek_order(greek),
+        [] if mus is None else [float(v) for v in mus.ravel()],
+        _args.flag(normalized, "normalized"),
+        [names.index(p) for p in params],
+        float(rtol),
+        _args.size(max_panels, "max_panels"),
+        _args.flag(allow_unvalidated, "allow_unvalidated"),
+        _args.flag(refine_backscatter, "refine_backscatter"),
+    )
+    mu_shape = () if mus is None else mus.shape
+    values = _outputs(raw, "", None, shape, mu_shape)
+    return EnsembleResult(
+        **{f: getattr(values, f) for f in (*SCALARS, "greek", "phase")},
+        error=_outputs(raw, "_errors", None, shape, mu_shape),
+        converged=_frozen(raw["converged"].reshape(shape)),
+        panels=_frozen(raw["panels"].astype(np.int64).reshape(shape)),
+        params=params,
+        _d=tuple(_outputs(raw, "_d", k, shape, mu_shape) for k in range(len(params))),
+        _d_errors=tuple(_outputs(raw, "_d_errors", k, shape, mu_shape) for k in range(len(params))),
+    )
+
+
+_SHARED_DOC = """
+    Parameters
+    ----------
+    {first}wavelength : array_like
+        Wavelengths, in the unit of the radii (real, scalar or any shape).
+    greek : int, "all" or None, optional
+        The Greek coefficients to compute: up to this order, every order, or none (the
+        default).
+    mu : array_like, optional
+        Cosines of the scattering angle at which to compute the phase matrix.
+    normalized : bool, optional
+        Normalize the phase matrix and the Greek coefficients (``alpha1[..., 0]`` = 1, the
+        default) or give them per particle (``alpha1[..., 0]`` = ``c_sca``).
+    derivatives : bool or sequence of str, optional
+        The parameters to differentiate with respect to (``True`` for all): the
+        distribution's (``"median_radius"``, …), ``"m_real"``, ``"m_imag"``{weights}.
+    rtol : float, optional
+        Relative tolerance of the size integration, in [1e-13, 0.1].
+    max_panels : int, optional
+        The most size panels per component and wavelength; reaching it sets ``converged``
+        to False.
+    allow_unvalidated : bool, optional
+        Compute sizes outside the validated domain of the Mie solution instead of raising.
+    refine_backscatter : bool, optional
+        Also refine on ``c_back`` (to ``rtol`` of the larger of ``c_back`` and ``c_sca``),
+        which makes its error estimate, and the lidar ratio's, a calibrated bound. Off by
+        default: the backscattering of weakly absorbing spheres is a sum of narrow glory
+        ripples that can take far more size panels than the cross sections.
+
+    Returns
+    -------
+    EnsembleResult
+
+    Raises
+    ------
+    InvalidInputError
+        For invalid distribution parameters, wavelengths, refractive indices or options.
+    OutsideValidatedDomainError
+        If a size node is outside the validated domain, unless ``allow_unvalidated``; the
+        message names the wavelength and the radius.
+    TypeError, ValueError
+        For arguments of the wrong type, or unknown parameter names.
+"""
+
+
+def compute(
+    distribution: Distribution,
+    wavelength: ArrayLike,
+    refractive_index: ArrayLike,
+    *,
+    greek: SupportsIndex | Literal["all"] | None = None,
+    mu: ArrayLike | None = None,
+    normalized: bool = True,
+    derivatives: bool | Sequence[str] = False,
+    rtol: float = 1e-6,
+    max_panels: SupportsIndex = 4096,
+    allow_unvalidated: bool = False,
+    refine_backscatter: bool = False,
+) -> EnsembleResult:
+    wl = _args.real_array(wavelength, "wavelength")
+    items = _specs([Component(distribution, refractive_index)], wl.size)
+    return _run(
+        items,
+        wl,
+        False,
+        greek,
+        mu,
+        normalized,
+        derivatives,
+        rtol,
+        max_panels,
+        allow_unvalidated,
+        refine_backscatter,
+    )
+
+
+compute.__doc__ = """Averages over spheres of one size distribution.
+""" + _SHARED_DOC.format(
+    first="distribution : LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated "
+    "or MultiModal\n        The number size distribution.\n    refractive_index : array_like\n"
+    "        The refractive index, broadcast to the wavelengths.\n    ",
+    weights=', and ``"mode{j}.weight"`` for multimodal distributions',
+)
+
+
+def compute_mixture(
+    components: Sequence[Component],
+    wavelength: ArrayLike,
+    *,
+    greek: SupportsIndex | Literal["all"] | None = None,
+    mu: ArrayLike | None = None,
+    normalized: bool = True,
+    derivatives: bool | Sequence[str] = False,
+    rtol: float = 1e-6,
+    max_panels: SupportsIndex = 4096,
+    allow_unvalidated: bool = False,
+    refine_backscatter: bool = False,
+) -> EnsembleResult:
+    wl = _args.real_array(wavelength, "wavelength")
+    items = _specs(list(components), wl.size)
+    return _run(
+        items,
+        wl,
+        True,
+        greek,
+        mu,
+        normalized,
+        derivatives,
+        rtol,
+        max_panels,
+        allow_unvalidated,
+        refine_backscatter,
+    )
+
+
+compute_mixture.__doc__ = """Averages over an external mixture: cross sections and moments
+    weighted by wᵢ/Σw, phase matrices by wᵢ C_sca,ᵢ.
+""" + _SHARED_DOC.format(
+    first="components : sequence of Component\n        The components.\n    ",
+    weights=' (each prefixed by ``"c{i}."``), and the weights ``"c{i}.weight"``',
+)
diff --git a/scatrs-py/src/ensemble.rs b/scatrs-py/src/ensemble.rs
new file mode 100644
index 0000000..6b30dcf
--- /dev/null
+++ b/scatrs-py/src/ensemble.rs
@@ -0,0 +1,293 @@
+//! Bindings for `scatrs::ensemble`.
+//!
+//! Distributions arrive as tagged tuples from `scatrs.ensemble` (`("lognormal", r_g, σ_g)`,
+//! `("tabulated", radii, densities)`, `("multimodal", [(weight, spec), …])`, …), components as
+//! `(spec, refractive indices, weight)`. Results return as a dict of arrays, one row per
+//! wavelength, which `scatrs.ensemble` wraps; derivatives are selected by their index in
+//! `ensemble_parameters`.
+
+use numpy::ndarray::{Array1, Array2, Array3, Array4};
+use numpy::{Complex64, IntoPyArray, PyReadonlyArray1};
+use pyo3::exceptions::PyTypeError;
+use pyo3::prelude::*;
+use pyo3::types::{PyDict, PyTuple};
+use scatrs::ensemble::{
+    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs,
+    SizeDistribution, compute_mixture, mixture_parameters, parameters,
+};
+use scatrs::mie::{MAX_GREEK_ORDER, Normalization};
+
+use crate::errors::to_pyerr;
+use crate::mie::MAX_MIE_VALUES;
+
+fn invalid(message: String) -> scatrs::Error {
+    scatrs::Error::InvalidInput(message)
+}
+
+/// A distribution from its tagged tuple.
+fn distribution(spec: &Bound<'_, PyAny>) -> PyResult<SizeDistribution> {
+    let t = spec
+        .cast::<PyTuple>()
+        .map_err(|_| PyTypeError::new_err("a distribution spec must be a tuple"))?;
+    let kind: String = t.get_item(0)?.extract()?;
+    let f = |i: usize| -> PyResult<f64> { t.get_item(i)?.extract() };
+    Ok(match kind.as_str() {
+        "lognormal" => SizeDistribution::LogNormal {
+            median_radius: f(1)?,
+            width: f(2)?,
+        },
+        "gamma" => SizeDistribution::Gamma {
+            effective_radius: f(1)?,
+            effective_variance: f(2)?,
+        },
+        "modified_gamma" => SizeDistribution::ModifiedGamma {
+            alpha: f(1)?,
+            b: f(2)?,
+            gamma: f(3)?,
+        },
+        "power_law" => SizeDistribution::PowerLaw {
+            exponent: f(1)?,
+            min_radius: f(2)?,
+            max_radius: f(3)?,
+        },
+        "monodisperse" => SizeDistribution::Monodisperse { radius: f(1)? },
+        "tabulated" => SizeDistribution::Tabulated {
+            radius: t.get_item(1)?.extract()?,
+            density: t.get_item(2)?.extract()?,
+        },
+        "multimodal" => {
+            let modes: Vec<(f64, Bound<'_, PyAny>)> = t.get_item(1)?.extract()?;
+            SizeDistribution::MultiModal {
+                modes: modes
+                    .iter()
+                    .map(|(w, s)| Ok((*w, distribution(s)?)))
+                    .collect::<PyResult<_>>()?,
+            }
+        }
+        other => {
+            return Err(PyTypeError::new_err(format!(
+                "unknown distribution kind {other:?}"
+            )));
+        }
+    })
+}
+
+/// Components from `(spec, refractive indices, weight)` tuples.
+fn components(
+    items: &[(Bound<'_, PyAny>, PyReadonlyArray1<'_, Complex64>, f64)],
+) -> PyResult<Vec<Component>> {
+    items
+        .iter()
+        .map(|(spec, m, w)| {
+            Ok(Component::new(
+                distribution(spec)?,
+                m.as_array().to_vec(),
+                *w,
+            ))
+        })
+        .collect()
+}
+
+fn all_parameters(components: &[Component], mixture: bool) -> Vec<EnsembleParam> {
+    if mixture {
+        mixture_parameters(components)
+    } else {
+        parameters(&components[0].distribution)
+    }
+}
+
+/// The names of the parameters of `components` (a single distribution unless `mixture`), in
+/// the order `ensemble_compute` indexes them.
+#[pyfunction]
+pub fn ensemble_parameters(
+    items: Vec<(Bound<'_, PyAny>, PyReadonlyArray1<'_, Complex64>, f64)>,
+    mixture: bool,
+) -> PyResult<Vec<String>> {
+    let cs = components(&items)?;
+    if cs.is_empty() {
+        return Ok(Vec::new());
+    }
+    Ok(all_parameters(&cs, mixture)
+        .iter()
+        .map(|p| p.name(mixture))
+        .collect())
+}
+
+/// The scalar outputs, in the order of `scatrs.ensemble.SCALARS`.
+fn scalars(o: &Outputs) -> [f64; 11] {
+    [
+        o.c_ext,
+        o.c_sca,
+        o.c_abs,
+        o.c_back,
+        o.ssa,
+        o.g,
+        o.lidar_ratio,
+        o.depolarization,
+        o.effective_radius,
+        o.mean_area,
+        o.mean_volume,
+    ]
+}
+
+/// The ensemble outputs at each wavelength: a dict of arrays `values`/`errors` (11, n),
+/// `d`/`d_errors` (P, 11, n), with the Greek coefficients (`greek`… (6, n, L + 1), zero-padded
+/// to the longest) and the phase matrix (`phase`… (4, n, n_mu)) when requested, `converged`
+/// (n) and `panels` (n). `greek` is the order, −1 for all orders, `None` for none.
+#[pyfunction]
+#[allow(clippy::too_many_arguments)]
+pub fn ensemble_compute<'py>(
+    py: Python<'py>,
+    items: Vec<(Bound<'py, PyAny>, PyReadonlyArray1<'py, Complex64>, f64)>,
+    wavelengths: PyReadonlyArray1<'py, f64>,
+    mixture: bool,
+    greek: Option<i64>,
+    mu: Vec<f64>,
+    normalized: bool,
+    derivatives: Vec<usize>,
+    rtol: f64,
+    max_panels: usize,
+    allow_unvalidated: bool,
+    refine_backscatter: bool,
+) -> PyResult<Bound<'py, PyDict>> {
+    let cs = components(&items)?;
+    let wl: Vec<f64> = wavelengths.as_array().to_vec();
+    let n = wl.len();
+    let err = |e| to_pyerr(py, e);
+    if cs.is_empty() {
+        return Err(err(invalid("a mixture needs a component".into())));
+    }
+    let all = all_parameters(&cs, mixture);
+    let params: Vec<EnsembleParam> = derivatives
+        .iter()
+        .map(|&i| {
+            all.get(i)
+                .copied()
+                .ok_or_else(|| err(invalid(format!("no parameter {i}"))))
+        })
+        .collect::<PyResult<_>>()?;
+    let order = match greek {
+        None => None,
+        Some(-1) => Some(GreekOrder::All),
+        Some(l) if l >= 0 && (l as u64) <= MAX_GREEK_ORDER as u64 => {
+            Some(GreekOrder::Order(l as usize))
+        }
+        Some(l) => {
+            return Err(err(invalid(format!(
+                "the Greek order must be in [0, {MAX_GREEK_ORDER}], got {l}"
+            ))));
+        }
+    };
+    // Bound the outputs before computing where their size is known.
+    let p = params.len();
+    let per = 11
+        + 4 * mu.len()
+        + match order {
+            Some(GreekOrder::Order(l)) => 6 * (l + 1),
+            _ => 0,
+        };
+    if n.saturating_mul(per).saturating_mul(2 * (1 + p)) > 2 * MAX_MIE_VALUES {
+        return Err(err(invalid(format!(
+            "the ensemble outputs would exceed 2 * MAX_MIE_VALUES = {} real values",
+            2 * MAX_MIE_VALUES
+        ))));
+    }
+    let mut opts = EnsembleOptions::new()
+        .with_rtol(rtol)
+        .with_max_panels(max_panels)
+        .with_mu(mu.clone())
+        .with_normalization(if normalized {
+            Normalization::Normalized
+        } else {
+            Normalization::Unnormalized
+        })
+        .with_derivatives(params)
+        .with_allow_unvalidated(allow_unvalidated)
+        .with_refine_backscatter(refine_backscatter);
+    if let Some(o) = order {
+        opts = opts.with_greek(o);
+    }
+    let results: Vec<EnsembleResult> = py
+        .detach(|| compute_mixture(&cs, &wl, &opts))
+        .map_err(err)?;
+    let out = PyDict::new(py);
+    let sc = |f: &dyn Fn(&EnsembleResult) -> &Outputs| {
+        Array2::from_shape_fn((11, n), |(k, i)| scalars(f(&results[i]))[k])
+    };
+    let dsc = |f: &dyn Fn(&EnsembleResult, usize) -> &Outputs| {
+        Array3::from_shape_fn((p, 11, n), |(c, k, i)| scalars(f(&results[i], c))[k])
+    };
+    out.set_item("values", sc(&|r| &r.values).into_pyarray(py))?;
+    out.set_item("errors", sc(&|r| &r.errors).into_pyarray(py))?;
+    out.set_item("d", dsc(&|r, c| &r.derivatives[c]).into_pyarray(py))?;
+    out.set_item(
+        "d_errors",
+        dsc(&|r, c| &r.derivative_errors[c]).into_pyarray(py),
+    )?;
+    if order.is_some() {
+        let width = results
+            .iter()
+            .map(|r| r.values.greek.as_ref().map_or(0, |g| g.alpha1.len()))
+            .max()
+            .unwrap_or(1);
+        if n.saturating_mul(width).saturating_mul(12 * (1 + p)) > 2 * MAX_MIE_VALUES {
+            return Err(err(invalid(format!(
+                "the Greek coefficients would exceed 2 * MAX_MIE_VALUES = {} real values",
+                2 * MAX_MIE_VALUES
+            ))));
+        }
+        let pick = |o: &Outputs, k: usize, l: usize| {
+            let g = o.greek.as_ref().expect("requested");
+            let a = [
+                &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
+            ][k];
+            a.get(l).copied().unwrap_or(0.0)
+        };
+        let g = |f: &dyn Fn(&EnsembleResult) -> &Outputs| {
+            Array3::from_shape_fn((6, n, width), |(k, i, l)| pick(f(&results[i]), k, l))
+        };
+        let dg = |f: &dyn Fn(&EnsembleResult, usize) -> &Outputs| {
+            Array4::from_shape_fn((p, 6, n, width), |(c, k, i, l)| {
+                pick(f(&results[i], c), k, l)
+            })
+        };
+        out.set_item("greek", g(&|r| &r.values).into_pyarray(py))?;
+        out.set_item("greek_errors", g(&|r| &r.errors).into_pyarray(py))?;
+        out.set_item("greek_d", dg(&|r, c| &r.derivatives[c]).into_pyarray(py))?;
+        out.set_item(
+            "greek_d_errors",
+            dg(&|r, c| &r.derivative_errors[c]).into_pyarray(py),
+        )?;
+    }
+    if !mu.is_empty() {
+        let n_mu = mu.len();
+        let pick = |o: &Outputs, k: usize, j: usize| {
+            let ph = o.phase.as_ref().expect("requested");
+            [&ph.f11, &ph.f12, &ph.f33, &ph.f34][k][j]
+        };
+        let f = |s: &dyn Fn(&EnsembleResult) -> &Outputs| {
+            Array3::from_shape_fn((4, n, n_mu), |(k, i, j)| pick(s(&results[i]), k, j))
+        };
+        let df = |s: &dyn Fn(&EnsembleResult, usize) -> &Outputs| {
+            Array4::from_shape_fn((p, 4, n, n_mu), |(c, k, i, j)| {
+                pick(s(&results[i], c), k, j)
+            })
+        };
+        out.set_item("phase", f(&|r| &r.values).into_pyarray(py))?;
+        out.set_item("phase_errors", f(&|r| &r.errors).into_pyarray(py))?;
+        out.set_item("phase_d", df(&|r, c| &r.derivatives[c]).into_pyarray(py))?;
+        out.set_item(
+            "phase_d_errors",
+            df(&|r, c| &r.derivative_errors[c]).into_pyarray(py),
+        )?;
+    }
+    out.set_item(
+        "converged",
+        Array1::from_iter(results.iter().map(|r| r.converged)).into_pyarray(py),
+    )?;
+    out.set_item(
+        "panels",
+        Array1::from_iter(results.iter().map(|r| r.panels as u64)).into_pyarray(py),
+    )?;
+    Ok(out)
+}
diff --git a/scatrs-py/src/lib.rs b/scatrs-py/src/lib.rs
index e36e375..dc65896 100644
--- a/scatrs-py/src/lib.rs
+++ b/scatrs-py/src/lib.rs
@@ -6,6 +6,7 @@
 
 use pyo3::prelude::*;
 
+mod ensemble;
 mod errors;
 mod mie;
 mod special;
@@ -30,6 +31,8 @@ fn _scatrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
     m.add("MAX_GREEK_ORDER", scatrs::mie::MAX_GREEK_ORDER)?;
     m.add_function(wrap_pyfunction!(mie::mie_phase_matrix, m)?)?;
     m.add_function(wrap_pyfunction!(mie::mie_greek, m)?)?;
+    m.add_function(wrap_pyfunction!(ensemble::ensemble_parameters, m)?)?;
+    m.add_function(wrap_pyfunction!(ensemble::ensemble_compute, m)?)?;
     testing::register(m)?;
     Ok(())
 }
diff --git a/tests/test_ensemble.py b/tests/test_ensemble.py
new file mode 100644
index 0000000..0fe552b
--- /dev/null
+++ b/tests/test_ensemble.py
@@ -0,0 +1,147 @@
+"""scatrs.ensemble: values, shapes, derivatives, mixtures and errors."""
+
+import threading
+import time
+
+import numpy as np
+import pytest
+
+from scatrs import InvalidInputError, OutsideValidatedDomainError, ensemble, mie
+
+LOGNORMAL = ensemble.LogNormal(0.08, 1.6)
+
+
+def test_monodisperse_is_the_single_sphere():
+    r, wl, m = 0.7, np.array([0.45, 0.55]), 1.5 + 0.01j
+    e = ensemble.compute(ensemble.Monodisperse(r), wl, m, greek="all", mu=[1.0, -1.0])
+    q = mie.efficiencies(2 * np.pi * r / wl, m)
+    np.testing.assert_allclose(e.c_ext, np.pi * r**2 * q.q_ext, rtol=1e-15)
+    np.testing.assert_allclose(e.g, q.g, rtol=1e-14)
+    g = mie.greek(2 * np.pi * r / wl, m)
+    np.testing.assert_allclose(e.greek.beta1, g.beta1, atol=1e-12)
+    assert np.all(e.converged) and np.all(e.panels == 0)
+
+
+def test_shapes_and_frozen_results():
+    wl = np.array([[0.35, 0.5], [0.75, 1.0]])
+    r = ensemble.compute(LOGNORMAL, wl, 1.43, greek=8, mu=np.linspace(-1, 1, 5), derivatives=True)
+    assert r.c_ext.shape == (2, 2) and r.converged.dtype == bool
+    assert r.greek.alpha1.shape == (2, 2, 9) and r.phase.f34.shape == (2, 2, 5)
+    assert r.error.greek.beta2.shape == (2, 2, 9)
+    assert r.params == ("median_radius", "width", "m_real", "m_imag")
+    np.testing.assert_array_equal(r.greek.alpha1[..., 0], 1.0)
+    for a in (r.c_ext, r.greek.alpha1, r.d("width").g, r.error.ssa, r.converged):
+        with pytest.raises(ValueError, match="read-only"):
+            a[...] = 0
+    s = ensemble.compute(LOGNORMAL, 0.5, 1.43)
+    assert s.c_ext.shape == () and s.greek is None and s.phase is None
+
+
+def test_refractive_index_broadcasts_to_the_wavelengths():
+    wl = [0.4, 0.6]
+    both = ensemble.compute(LOGNORMAL, wl, [1.45, 1.43 + 1e-3j])
+    one = ensemble.compute(LOGNORMAL, wl[1], 1.43 + 1e-3j)
+    assert both.c_ext[1] == one.c_ext
+
+
+def _fd(f, h):
+    return (f(+h) - f(-h)) / (2 * h)
+
+
+def test_derivatives_match_finite_differences():
+    wl, m = 0.55, 1.45 + 0.02j
+    kw = {"rtol": 1e-11, "max_panels": 50_000, "greek": 4}
+    r = ensemble.compute(ensemble.LogNormal(0.2, 1.6), wl, m, derivatives=True, **kw)
+    steps = {
+        "median_radius": lambda e: (ensemble.LogNormal(0.2 + e, 1.6), m),
+        "width": lambda e: (ensemble.LogNormal(0.2, 1.6 + e), m),
+        "m_real": lambda e: (ensemble.LogNormal(0.2, 1.6), m + e),
+        "m_imag": lambda e: (ensemble.LogNormal(0.2, 1.6), m + 1j * e),
+    }
+    for p, at in steps.items():
+        h = 1e-4
+        for name in ("c_ext", "ssa", "g", "c_back"):
+            fd = _fd(
+                lambda e, at=at, name=name: getattr(
+                    ensemble.compute(at(e)[0], wl, at(e)[1], **kw), name
+                ),
+                h,
+            )
+            assert abs(getattr(r.d(p), name) - fd) <= 1e-5 * abs(fd) + 1e-12, (p, name)
+        fd = _fd(lambda e, at=at: ensemble.compute(at(e)[0], wl, at(e)[1], **kw).greek.alpha1[2], h)
+        assert abs(r.d(p).greek.alpha1[2] - fd) <= 1e-5 * abs(fd) + 1e-10, p
+
+
+def test_mixtures_and_their_weights():
+    a = ensemble.Component(ensemble.LogNormal(0.1, 1.5), 1.45, 2.0)
+    b = ensemble.Component(ensemble.Gamma(1.0, 0.1), 1.53 + 0.01j, 0.5)
+    m = ensemble.compute_mixture([a, b], 0.6, derivatives=["c0.weight"])
+    ra = ensemble.compute(a.distribution, 0.6, a.refractive_index)
+    rb = ensemble.compute(b.distribution, 0.6, b.refractive_index)
+    np.testing.assert_allclose(m.c_ext, (2.0 * ra.c_ext + 0.5 * rb.c_ext) / 2.5, rtol=1e-13)
+    # ∂/∂w₀ of (w₀ A + w₁ B)/(w₀ + w₁) = (A − mix)/(w₀ + w₁).
+    np.testing.assert_allclose(m.d("c0.weight").c_ext, (ra.c_ext - m.c_ext) / 2.5, rtol=1e-12)
+    names = ensemble.compute_mixture([a, b], 0.6, derivatives=True).params
+    assert names[:3] == ("c0.median_radius", "c0.width", "c0.m_real") and names[-1] == "c1.weight"
+    bi = ensemble.MultiModal(
+        [(2.0, ensemble.LogNormal(0.1, 1.5)), (0.5, ensemble.Monodisperse(1.0))]
+    )
+    assert ensemble.compute(bi, 0.5, 1.5, derivatives=True).params[:3] == (
+        "mode0.median_radius",
+        "mode0.width",
+        "mode0.weight",
+    )
+
+
+def test_tabulated_and_power_law_agree():
+    t = ensemble.compute(ensemble.Tabulated([0.1, 0.5], [1.0, 1.0]), 0.5, 1.5, rtol=1e-9)
+    p = ensemble.compute(ensemble.PowerLaw(0.0, 0.1, 0.5), 0.5, 1.5, rtol=1e-9)
+    np.testing.assert_allclose(t.c_ext, p.c_ext, rtol=1e-8)
+
+
+def test_errors():
+    with pytest.raises(InvalidInputError, match="width"):
+        ensemble.compute(ensemble.LogNormal(0.1, 0.5), 0.5, 1.5)
+    with pytest.raises(InvalidInputError, match="rtol"):
+        ensemble.compute(LOGNORMAL, 0.5, 1.5, rtol=1.0)
+    with pytest.raises(InvalidInputError, match="wavelength"):
+        ensemble.compute(LOGNORMAL, -0.5, 1.5)
+    with pytest.raises(OutsideValidatedDomainError, match="radius"):
+        ensemble.compute(ensemble.Monodisperse(2000.0), 0.5, 1.5)
+    with pytest.raises(ValueError, match="unknown parameter"):
+        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives=["radius"])
+    with pytest.raises(ValueError, match="no derivative"):
+        ensemble.compute(LOGNORMAL, 0.5, 1.5).d("width")
+    with pytest.raises(ValueError, match="greek must be"):
+        ensemble.compute(LOGNORMAL, 0.5, 1.5, greek="some")
+    with pytest.raises(TypeError, match="Component"):
+        ensemble.compute_mixture([LOGNORMAL], 0.5)
+    with pytest.raises(TypeError, match="derivatives must be"):
+        ensemble.compute(LOGNORMAL, 0.5, 1.5, derivatives="width")
+    with pytest.raises(InvalidInputError, match="weights"):
+        ensemble.compute_mixture([ensemble.Component(LOGNORMAL, 1.5, 0.0)], 0.5)
+    with pytest.raises(ValueError):
+        ensemble.compute(LOGNORMAL, [0.4, 0.5], [1.5, 1.5, 1.5])
+
+
+def test_gil_is_released_while_computing():
+    span = {}
+
+    def worker():
+        span["start"] = time.perf_counter()
+        ensemble.compute(
+            ensemble.LogNormal(0.3, 1.6), np.linspace(0.3, 1.0, 8), 1.45 + 1e-3j, greek=32
+        )
+        span["end"] = time.perf_counter()
+
+    stamps = []
+    thread = threading.Thread(target=worker)
+    thread.start()
+    while thread.is_alive():
+        time.sleep(0.0005)
+        stamps.append(time.perf_counter())
+    thread.join()
+    quarter = (span["end"] - span["start"]) / 4
+    inside = sum(span["start"] + quarter < t < span["end"] - quarter for t in stamps)
+    assert span["end"] - span["start"] > 0.02
+    assert inside >= 5
diff --git a/tests/test_package.py b/tests/test_package.py
index 2cbbed2..a5ce15c 100644
--- a/tests/test_package.py
+++ b/tests/test_package.py
@@ -7,7 +7,7 @@ from pathlib import Path
 from packaging.version import Version
 
 import scatrs
-from scatrs import _scatrs, mie, special
+from scatrs import _scatrs, ensemble, mie, special
 
 ROOT = Path(__file__).parents[1]
 
@@ -48,6 +48,7 @@ def test_crate_license_is_the_repository_license():
 
 def test_public_names():
     for name in (
+        "ensemble",
         "mie",
         "special",
         "ScatrsError",
@@ -57,7 +58,7 @@ def test_public_names():
         "__version__",
     ):
         assert hasattr(scatrs, name)
-    for module in (special, mie):
+    for module in (special, mie, ensemble):
         for name in module.__all__:
             assert hasattr(module, name)
 
@@ -78,6 +79,8 @@ def test_test_hooks_live_outside_the_extension_namespace():
         "MAX_MIE_VALUES",
         "MAX_WIGNER_D_VALUES",
         "__version__",
+        "ensemble_compute",
+        "ensemble_parameters",
         "gauss_legendre",
         "mie_amplitudes",
         "mie_coefficients",
````

- [ ] **Step 2: Verify.** `uv run pytest -q`, `uv run python -m mypy.stubtest scatrs._scatrs`, `uv run ruff check . && uv run ruff format --check .`.

- [ ] **Step 3: Commit** — `feat(python): scatrs.ensemble`

---

### Task 3: Documentation

Requirements (facts from the spec, section 12, and "Measured results"):

1. `docs/theory/ensembles.md` (new, in the toctree after `theory/mie`): the distributions and their parameters; numeric normalization over the range; the cut-offs (number and area tails at rtol/10 for the Mie range, volume tail for the moments, the extension rule); the two passes and the refined quantities with their targets (including `refine_backscatter`, `refine_order`, the m-derivatives); initial panels (½ in ln r, Δx ≤ 20); projector bands; error estimates (|K15 − G7|, the projected Greek bound, the truncation term, first-order propagation); Jacobians (renormalization, monodisperse radius, mixture and mode weights); mixtures; parallelism and determinism; cost.
2. `docs/examples/ensemble.md` (executed): a lognormal sulfate-like aerosol (r_g = 0.08 µm, σ_g = 1.6, m = 1.43 + 10⁻⁸i) at several wavelengths: cross sections, ω, g, Ångström-like wavelength dependence, Greek coefficients (α₁⁰ = 1, α₁¹ = 3g asserted), error estimates and `converged`, a derivative against a finite difference, a two-component mixture; in the toctree next to the other examples.
3. `docs/validation/accuracy.md`: a section on the ensembles: the calibration grid and its result, what is a bound (refined quantities) and what is indicative (backscattering unless refined), the flagged cases.
4. `docs/limitations.md`: the ensemble cost (the measured table), the Greek pass-2 cost and the planned remedies, non-absorbing ripples and Im m derivatives (unconverged flags), backscattering opt-in, `f64` only, sizes outside the validated domain.
5. `docs/api.md`: `scatrs.ensemble`. `docs/index.md`: status. `CHANGELOG.md`: Added (ensembles, Python), Changed (rayon dependency). `docs/plans/phase-2a-followups.md`: Clebsch–Gordan Greek coefficients, shared πₙ/τₙ tables, distributions over other parameters (phase 4), 2b/2c.
6. Contracts in `tests/test_docs_consistency.py` for figures quoted in more than one place.

- [ ] **Step 1:** Write the pages. **Step 2:** `uv run --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html` and `uv run pytest -q tests/test_docs_consistency.py`. **Step 3: Commit** — `docs: ensembles of spheres (theory, example, accuracy, limitations)`.

### Task 4: Final verification and review

The CONTRIBUTING checklist (with clippy 1.99), a whole-branch review, then finishing the branch.
