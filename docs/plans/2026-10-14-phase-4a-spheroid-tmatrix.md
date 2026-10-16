# Phase 4a: The Spheroid T-matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** randomly oriented spheroids in Rust (`scatrs::tmatrix`) and Python (`scatrs.tmatrix`): orientation-averaged efficiencies, Greek coefficients and the six-element phase matrix, converged to a tolerance with an error estimate, with derivatives with respect to x, the aspect ratio, Re m and Im m, validated against an independent Arb EBCM and an accuracy map.

**Architecture:**
- **Kernel** (`tmatrix/kernel.rs`, `functions.rs`): the EBCM per azimuthal mode, node-outer over all modes, with scaled radial functions and the mirror-symmetry split into two classes per mode.
- **Cancellation-free integrals** (`tmatrix/regular.rs`): each element in three exact forms (plain; minus the Laurent powers < 0; minus the powers ≤ 0, the electrostatic limit, which integrates to zero for n ≠ k), keeping the one adding the least magnitude.
- **Solve** (`tmatrix/solve.rs`): f64 LU with a probe error estimate per mode; mixed-precision refinement (rows n ≥ N/4 in double-double, double-double residuals) where the estimate exceeds rtol/10; full double-double as the last path.
- **Driver** (`tmatrix/driver.rs`): N then n_θ converged on mode 0, all modes solved once, an optional Greek check.
- **Averaging** (`tmatrix/random.rs`): the rank decomposition of T by Clebsch–Gordan coefficients; the Greek coefficients by an exact six-element projection.
- **Derivatives**: the kernel in `Dual<f64, 4>`, dT = −(dP + T dQ) Q⁻¹ with the solve's factors.

**Tech Stack:**
- Rust 2024 (MSRV 1.85; CI lints with 1.99), pyo3/numpy 0.29.
- Python ≥ 3.11.
- python-flint 0.9.0 for the references.
- Sphinx 9 + myst-nb.

**Spec:** `docs/specs/2026-10-14-phase-4-spheroids-design.md`, sections 1.1, 2, 3, 5.1, 5.2, 6–10. Section 13 records what 4a changed, and why.

## Global Constraints

- **Spheres, layered spheres and ensembles stay bitwise unchanged.** The only change outside `tmatrix` is `GreekProjector::project_six` and three `pub(crate)` helpers on `GreekCoefficients`.
- **Declared accuracy:** the Arb references and the accuracy map meet their tolerances, and each declared constant is at least 4 times the worst case found.
- **Checks at every commit:**
  - `cargo fmt --all`;
  - `cargo +1.99.0 clippy --workspace --all-targets -- -D warnings` (stop on any output);
  - `cargo test --workspace --release`;
  - `RUSTDOCFLAGS="-D warnings" cargo doc -p scatrs --no-deps --document-private-items`;
  - `uv run pytest -q`, `ruff check`, `ruff format --check`;
  - `uv run python -m mypy.stubtest scatrs._scatrs`;
  - `sphinx-build -E -W --keep-going`.
- **Docs are user-facing:** no phases or development history.
- Downloads (the `tmatd` source, pytmatrix) need the user's approval first.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never send personal data in network requests.

## How this plan was made

Every task was prototyped and verified before this plan was written. The diffs below are the commits on `phase-4`, without the generated data (`reference/data/spheroid.jsonl` and the manifest, from `python -m scatrs_reference generate --only spheroid`, about 32 minutes; `docs/validation/data/spheroid-accuracy-map.json`, written by the map with `SCATRS_ACCURACY_MAP`, about 96 minutes) and without the precision study in `research/tmatrix/` (commit `0d734e9`).

## Measured results the plan relies on

| What | Measured |
|---|---|
| Regularized f64 T, x = 15, ε = 4 (N = 40) | error against double-double 43 (classical) → 3.2·10⁻⁴; double-double forms agree to rounding once the series are summed to x_max (a fixed 48 terms truncated them) |
| f64 elements, prolate (x = 15, ε = 1/3) | row-relative errors near 10⁻⁵ in rows n ≈ N: f64 T wrong by 0.15; the rounded double-double elements give 3·10⁻¹⁰ |
| Refinement (rows n ≥ N/4 in double-double, mixed matrix factorized) | 2 sweeps: x = 15, ε = 1/3: 0.15 → 3·10⁻¹⁴; x = 25, ε = 1/3: 11 → 4·10⁻¹⁴; x = 15, ε = 4: 3·10⁻⁴ → 4·10⁻¹³; x = 30, ε = 2: 6·10⁻⁷ → 1·10⁻¹⁴ |
| Error estimate | fed with the true element errors, 3× above the true T error; element bounds 2N·u·Σ|terms| (the true worst is about N·u·Σ|terms|); Hager's Skeel(Qᵀ) estimate mispredicted by up to 10⁸ and was dropped |
| Speed (N = 40, n_θ = 80) | f64 `all_modes` 283 → 85 ms, double-double 5.8 → 2.1 s (suffix sums, hoisting, sparse angular factors, \|re\| + \|im\| bookkeeping) |
| Convergence loop at rtol 10⁻⁸ | x = 15, ε = 2: 23 s → 4.9 s with the mode-0 loop; ε = 5, x = 12: 19 s; ε = 1/5, x = 12: 192 s |
| Derivatives | against central differences to 10⁻⁷ (kernel) and 10⁻⁶ (public outputs) of their natural size; cost 3.8× (x = 5) to 13× the values |
| Arb references (10 spheroids) | fixed truncations on the f64, refined and double-double paths within 0.8 of 4× the estimate; converged efficiencies within 0.004 of the tolerance |
| Accuracy map (221 spheroids) | worst error/max(rtol, estimate) inside the domain: Q_back 1.04, Greek 0.57, phase 0.10, cross sections and g ≤ 0.033 |
| Rayleigh limit (x = 0.01, ε = 1/4 … 4) | Q_abs, Q_sca within 10⁻³ of the depolarization-factor polarizabilities |

## File Structure

| File | Task | Role |
|---|---|---|
| `scatrs/src/special/clebsch.rs` | 1 | Clebsch–Gordan coefficients (Schulten–Gordon recursion) |
| `scatrs/src/tmatrix/{functions,kernel}.rs` | 1, 2, 3, 5 | radial and angular functions; `Spheroid`, `all_modes`, `tmatrix*`, `tmatrix_refined`, `tmatrix_derivatives` |
| `scatrs/src/tmatrix/random.rs` | 1, 4 | rank decomposition, `random_orientation`, `greek_raw` |
| `scatrs/src/tmatrix/regular.rs` | 2 | the series and the three forms; `angular_factors` |
| `scatrs/src/tmatrix/solve.rs` | 3, 5 | LU, `solve_f64`, `error_estimate`, `refine`, derivative solves |
| `scatrs/src/tmatrix/{mod,driver,domain}.rs` | 4, 5, 8 | public API, convergence loop and paths, validated domain |
| `scatrs/src/mie/greek.rs` | 4 | `GreekProjector::project_six` |
| `scatrs-py/src/tmatrix.rs`, `python/scatrs/tmatrix.py`, stubs | 6 | Python API |
| `reference/scatrs_reference/generators/spheroid.py`, `scatrs/tests/spheroid_reference.rs` | 7 | Arb references and their test |
| `scatrs/tests/tmatrix_accuracy_map.rs` | 8 | the accuracy map |
| `docs/theory/spheroids.md` and pages | 9 | documentation |

### Task 1: The EBCM kernel and the random-orientation averages

- [ ] Port the lab's EBCM (`research/tmatrix/ebcm.py`) to `tmatrix/kernel.rs`: scaled radial functions (σₙ = 1/|hₙ(x_min)|, c_k = 1/E_k), node-outer over all modes, the two mirror classes, half-range quadrature.
- [ ] Add `special/clebsch.rs` and its tests (Racah to 10⁻¹³, orthonormality at j ≈ 100 to 10⁻¹²).
- [ ] Add `tmatrix/random.rs`: helicity blocks, rank components, Φ_sq, the products and `mueller`.
- [ ] Test: a sphere equals Mie (T and phase matrix to 10⁻¹²), lossless energy, the brute-force average of `ranks.py`, 2π∫F₁₁ = C_sca.

The commit (`3ed812b`):

````diff
diff --git a/pyproject.toml b/pyproject.toml
index 5fc2d10..f293cab 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -80,6 +80,8 @@ select = ["E", "F", "W", "I", "UP", "B", "NPY", "RUF"]
 allowed-confusables = ["−", "–", "ρ", "×", "′", "α", "γ", "σ"]
 
 [tool.ruff.lint.per-file-ignores]
+# The precision study writes its mathematics in its own notation (ν, Greek letters in docstrings).
+"research/**" = ["RUF002", "RUF003"]
 # l is the degree of d^l_{mn}, the notation of the papers the generator follows.
 "reference/scatrs_reference/generators/mie_greek.py" = ["E741"]
 "reference/scatrs_reference/generators/wigner_d.py" = ["E741"]
diff --git a/scatrs/src/lib.rs b/scatrs/src/lib.rs
index eba745d..a7f1ac5 100644
--- a/scatrs/src/lib.rs
+++ b/scatrs/src/lib.rs
@@ -149,6 +149,7 @@ pub mod scalar;
 pub mod special;
 #[cfg(test)]
 mod test_data;
+pub mod tmatrix;
 
 pub use error::{Error, Result};
 pub use scalar::Real;
diff --git a/scatrs/src/special/clebsch.rs b/scatrs/src/special/clebsch.rs
new file mode 100644
index 0000000..978d9ad
--- /dev/null
+++ b/scatrs/src/special/clebsch.rs
@@ -0,0 +1,232 @@
+//! Clebsch–Gordan coefficients C^{J M}_{ja ma; jb mb} (Condon–Shortley phases) for every J at
+//! once, by the Schulten–Gordon recursion of the 3j symbol (Schulten & Gordon 1975):
+//! f(J) = (J ja jb; −M ma mb) satisfies
+//!
+//!   J A(J+1) f(J+1) + B(J) f(J) + (J+1) A(J) f(J−1) = 0,
+//!   A(J) = √((J² − (ja − jb)²)((ja + jb + 1)² − J²)(J² − M²)),
+//!   B(J) = −(2J + 1)[ja(ja+1)(−M) − jb(jb+1)(−M) − J(J+1)(mb − ma)],
+//!
+//! run backward from J_max = ja + jb and forward from J_min = max(|ja − jb|, |M|), each stable
+//! where the solution grows, joined where the backward one peaks, normalized by
+//! Σ (2J + 1) f² = 1 and signed by the closed form at J_max, sgn = (−1)^{ja − jb + M}. Then
+//! C^{J M} = (−1)^{ja − jb + M} √(2J + 1) f(J).
+#![allow(dead_code)] // used by the T-matrix kernel, whose API lands later in phase 4a
+
+use crate::scalar::Real;
+
+fn int<T: Real>(k: i64) -> T {
+    T::from_f64(k as f64)
+}
+
+/// The recursion coefficients A(J), B(J).
+fn ab<T: Real>(j: i64, ja: i64, jb: i64, ma: i64, mb: i64) -> (T, T) {
+    let m = ma + mb;
+    let a2 =
+        (j * j - (ja - jb) * (ja - jb)) * ((ja + jb + 1) * (ja + jb + 1) - j * j) * (j * j - m * m);
+    let a = if a2 > 0 {
+        int::<T>(a2).sqrt()
+    } else {
+        T::zero()
+    };
+    let b = -int::<T>(2 * j + 1)
+        * (int::<T>(ja * (ja + 1)) * int::<T>(-m)
+            - int::<T>(jb * (jb + 1)) * int::<T>(-m)
+            - int::<T>(j * (j + 1)) * int::<T>(mb - ma));
+    (a, b)
+}
+
+/// C^{J, ma+mb}_{ja ma; jb mb} for J = J_min, …, ja + jb, with J_min (the first value's J);
+/// empty if |ma| > ja or |mb| > jb.
+pub(crate) fn clebsch_gordan<T: Real>(
+    ja: usize,
+    ma: i64,
+    jb: usize,
+    mb: i64,
+    out: &mut Vec<T>,
+) -> usize {
+    out.clear();
+    let (ja, jb) = (ja as i64, jb as i64);
+    let m = ma + mb;
+    if ma.abs() > ja || mb.abs() > jb {
+        return 0;
+    }
+    let j_min = (ja - jb).abs().max(m.abs());
+    let j_max = ja + jb;
+    if j_min > j_max {
+        return j_min as usize;
+    }
+    let len = (j_max - j_min + 1) as usize;
+    let idx = |j: i64| (j - j_min) as usize;
+    // Backward, from f(J_max) = 1.
+    let mut back = vec![T::zero(); len];
+    back[idx(j_max)] = T::one();
+    for j in (j_min + 1..=j_max).rev() {
+        let (a_j, b_j) = ab::<T>(j, ja, jb, ma, mb);
+        let (a_j1, _) = ab::<T>(j + 1, ja, jb, ma, mb);
+        let next = if j < j_max {
+            back[idx(j + 1)]
+        } else {
+            T::zero()
+        };
+        let den = int::<T>(j + 1) * a_j;
+        back[idx(j - 1)] = if den == T::zero() {
+            T::zero()
+        } else {
+            -(int::<T>(j) * a_j1 * next + b_j * back[idx(j)]) / den
+        };
+    }
+    // Where the backward solution peaks.
+    let peak = (0..len)
+        .max_by(|&p, &q| {
+            back[p]
+                .abs()
+                .partial_cmp(&back[q].abs())
+                .unwrap_or(core::cmp::Ordering::Equal)
+        })
+        .unwrap_or(0);
+    let mut f = back.clone();
+    if peak > 0 && j_min > 0 {
+        // Forward, from f(J_min) = 1, up to the peak; scaled to meet the backward one there.
+        let mut fwd = vec![T::zero(); peak + 1];
+        fwd[0] = T::one();
+        for p in 0..peak {
+            let j = j_min + p as i64;
+            let (a_j, b_j) = ab::<T>(j, ja, jb, ma, mb);
+            let (a_j1, _) = ab::<T>(j + 1, ja, jb, ma, mb);
+            let prev = if p > 0 { fwd[p - 1] } else { T::zero() };
+            let den = int::<T>(j) * a_j1;
+            fwd[p + 1] = if den == T::zero() {
+                T::zero()
+            } else {
+                -(b_j * fwd[p] + int::<T>(j + 1) * a_j * prev) / den
+            };
+        }
+        if fwd[peak] != T::zero() {
+            let scale = back[peak] / fwd[peak];
+            for p in 0..peak {
+                f[p] = fwd[p] * scale;
+            }
+        }
+    }
+    let norm: T = (0..len).fold(T::zero(), |acc, p| {
+        acc + int::<T>(2 * (j_min + p as i64) + 1) * f[p] * f[p]
+    });
+    let mut scale = norm.sqrt().recip();
+    // sgn f(J_max) = (−1)^{ja − jb + M}.
+    let want_negative = (ja - jb + m).rem_euclid(2) == 1;
+    if (f[len - 1] < T::zero()) != want_negative {
+        scale = -scale;
+    }
+    let phase = if (ja - jb + m).rem_euclid(2) == 1 {
+        -T::one()
+    } else {
+        T::one()
+    };
+    for (p, &fp) in f.iter().enumerate() {
+        let j = j_min + p as i64;
+        out.push(phase * int::<T>(2 * j + 1).sqrt() * fp * scale);
+    }
+    j_min as usize
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    fn fact(n: i64) -> f64 {
+        (1..=n).map(|k| k as f64).product()
+    }
+
+    /// Racah's formula, for small arguments.
+    fn racah(j1: i64, m1: i64, j2: i64, m2: i64, j: i64, m: i64) -> f64 {
+        if m1 + m2 != m
+            || j < (j1 - j2).abs()
+            || j > j1 + j2
+            || m1.abs() > j1
+            || m2.abs() > j2
+            || m.abs() > j
+        {
+            return 0.0;
+        }
+        let pre = ((2 * j + 1) as f64 * fact(j + j1 - j2) * fact(j - j1 + j2) * fact(j1 + j2 - j)
+            / fact(j1 + j2 + j + 1))
+        .sqrt()
+            * (fact(j + m)
+                * fact(j - m)
+                * fact(j1 - m1)
+                * fact(j1 + m1)
+                * fact(j2 - m2)
+                * fact(j2 + m2))
+            .sqrt();
+        let mut sum = 0.0;
+        for k in 0..=(j1 + j2 + j) {
+            let terms = [
+                k,
+                j1 + j2 - j - k,
+                j1 - m1 - k,
+                j2 + m2 - k,
+                j - j2 + m1 + k,
+                j - j1 - m2 + k,
+            ];
+            if terms.iter().any(|&t| t < 0) {
+                continue;
+            }
+            let den: f64 = terms.iter().map(|&t| fact(t)).product();
+            sum += if k % 2 == 0 { 1.0 } else { -1.0 } / den;
+        }
+        pre * sum
+    }
+
+    #[test]
+    fn matches_racah_for_small_arguments() {
+        let mut out = Vec::new();
+        let mut worst = 0.0f64;
+        for ja in 0..7usize {
+            for jb in 0..7usize {
+                for ma in -(ja as i64)..=ja as i64 {
+                    for mb in -(jb as i64)..=jb as i64 {
+                        let j_min = clebsch_gordan::<f64>(ja, ma, jb, mb, &mut out);
+                        for (p, &v) in out.iter().enumerate() {
+                            let j = (j_min + p) as i64;
+                            let want = racah(ja as i64, ma, jb as i64, mb, j, ma + mb);
+                            worst = worst.max((v - want).abs());
+                        }
+                    }
+                }
+            }
+        }
+        assert!(worst < 1e-13, "worst {worst:e}");
+    }
+
+    /// Orthonormality at large arguments: Σ_{ma} C^{J,M}_{ja ma; jb M−ma} C^{J',M}_{…} = δ_{JJ'}.
+    #[test]
+    fn orthonormal_at_large_arguments() {
+        let (ja, jb, m) = (60usize, 45usize, 7i64);
+        let mut table: Vec<(usize, Vec<f64>)> = Vec::new();
+        let mut out = Vec::new();
+        for ma in -(ja as i64)..=ja as i64 {
+            let mb = m - ma;
+            if mb.abs() > jb as i64 {
+                continue;
+            }
+            let j_min = clebsch_gordan::<f64>(ja, ma, jb, mb, &mut out);
+            table.push((j_min, out.clone()));
+        }
+        let get = |t: &(usize, Vec<f64>), j: usize| {
+            if j >= t.0 {
+                t.1.get(j - t.0).copied().unwrap_or(0.0)
+            } else {
+                0.0
+            }
+        };
+        let mut worst = 0.0f64;
+        for j in [15usize, 30, 60, 90, 105] {
+            for j2 in [15usize, 30, 105] {
+                let s: f64 = table.iter().map(|t| get(t, j) * get(t, j2)).sum();
+                let want = if j == j2 { 1.0 } else { 0.0 };
+                worst = worst.max((s - want).abs());
+            }
+        }
+        assert!(worst < 1e-12, "worst {worst:e}");
+    }
+}
diff --git a/scatrs/src/special/mod.rs b/scatrs/src/special/mod.rs
index 7fb9d34..148de83 100644
--- a/scatrs/src/special/mod.rs
+++ b/scatrs/src/special/mod.rs
@@ -24,6 +24,7 @@
 //! # Ok::<(), scatrs::Error>(())
 //! ```
 
+pub(crate) mod clebsch;
 mod gauss_legendre;
 mod gauss_legendre_asymptotic;
 mod wigner;
diff --git a/scatrs/src/tmatrix/functions.rs b/scatrs/src/tmatrix/functions.rs
new file mode 100644
index 0000000..ece7900
--- /dev/null
+++ b/scatrs/src/tmatrix/functions.rs
@@ -0,0 +1,239 @@
+//! The functions the EBCM integrands are built from, at one quadrature node: the outgoing and
+//! regular radial functions of the medium (x = k r(θ)), the regular radial functions of the
+//! particle (z = s x), and the normalized angular functions of one azimuthal mode.
+//!
+//! The radial functions span hundreds of orders of magnitude over the orders and the surface
+//! (hₙ(x) grows like (2n − 1)!!/x^{n+1}), so they are kept scaled: the medium's of order n by
+//! σₙ = 1/|hₙ(x_min)|, x_min = k r_min, and the particle's of order k by c_k = 1/E_k, a smooth
+//! envelope of |j_k| at z_max = s k r_max. Every scaled value stays of order 1 or below over the
+//! surface. The scaling is undone on T (`super::kernel`).
+
+use num_complex::Complex;
+
+use crate::complex::inv;
+use crate::mie::kernel::{int, ratios_complex, ratios_real, real};
+use crate::scalar::Real;
+
+/// The scale factors of the radial functions: σₙ = 1/|hₙ(x_min)| through the moduli of the
+/// upward ratios hₙ/hₙ₋₁ at x_min, and c_k = 1/E_k with E_k = e^{|Im z_max|}/|z_max| ·
+/// Π_{i ≤ k} min(1, |z_max|/(2i + 1)).
+#[derive(Clone, Debug)]
+pub(crate) struct Scales<T> {
+    /// |hₙ(x_min)|/|hₙ₋₁(x_min)| for n = 1, …, N + 1 (index n), and |h₀(x_min)| at index 0.
+    pub(crate) h_ratio: Vec<T>,
+    /// E_k/E_{k−1} for k = 1, …, N + 1 (index k), and E₀ at index 0 (without the e^{|Im z_max|}).
+    pub(crate) e_ratio: Vec<T>,
+    /// |Im z_max|.
+    pub(crate) im_z_max: T,
+}
+
+impl<T: Real> Scales<T> {
+    pub(crate) fn new(x_min: T, z_max: Complex<T>, n: usize) -> Self {
+        let mut h_ratio = Vec::with_capacity(n + 2);
+        h_ratio.push(x_min.recip());
+        // h₁/h₀ = 1/x − i, h_{n+1}/hₙ = (2n + 1)/x − hₙ₋₁/hₙ.
+        let inv_x = x_min.recip();
+        let mut r = Complex::new(inv_x, -T::one());
+        h_ratio.push(r.norm());
+        for k in 1..=n {
+            r = real(inv_x * int::<T>(2 * k + 1)) - inv(r);
+            h_ratio.push(r.norm());
+        }
+        let za = z_max.norm();
+        let mut e_ratio = Vec::with_capacity(n + 2);
+        e_ratio.push(za.recip());
+        for k in 1..=n + 1 {
+            e_ratio.push((za / int::<T>(2 * k + 1)).min(T::one()));
+        }
+        Self {
+            h_ratio,
+            e_ratio,
+            im_z_max: z_max.im.abs(),
+        }
+    }
+}
+
+/// The scaled radial functions of one node.
+#[derive(Clone, Debug)]
+pub(crate) struct Radial<T> {
+    /// σₙ hₙ(x), σₙ [x hₙ(x)]'/x and σₙ hₙ(x)/x (index n = 0, …, N).
+    pub(crate) h: Vec<Complex<T>>,
+    pub(crate) dh: Vec<Complex<T>>,
+    pub(crate) h_x: Vec<Complex<T>>,
+    /// σₙ jₙ(x), σₙ [x jₙ]'/x, σₙ jₙ/x.
+    pub(crate) j: Vec<Complex<T>>,
+    pub(crate) dj: Vec<Complex<T>>,
+    pub(crate) j_x: Vec<Complex<T>>,
+    /// c_k j_k(z), c_k [z j_k]'/z, c_k j_k/z.
+    pub(crate) jz: Vec<Complex<T>>,
+    pub(crate) djz: Vec<Complex<T>>,
+    pub(crate) jz_z: Vec<Complex<T>>,
+    u_x: Vec<T>,
+    u_z: Vec<Complex<T>>,
+}
+
+impl<T: Real> Radial<T> {
+    pub(crate) fn new() -> Self {
+        Self {
+            h: Vec::new(),
+            dh: Vec::new(),
+            h_x: Vec::new(),
+            j: Vec::new(),
+            dj: Vec::new(),
+            j_x: Vec::new(),
+            jz: Vec::new(),
+            djz: Vec::new(),
+            jz_z: Vec::new(),
+            u_x: Vec::new(),
+            u_z: Vec::new(),
+        }
+    }
+
+    /// The functions at x (real) and z = s x, orders 0, …, n.
+    pub(crate) fn compute(&mut self, x: T, s: Complex<T>, n: usize, scales: &Scales<T>) {
+        let z = s.scale(x);
+        let inv_x = x.recip();
+        let inv_z = inv(z);
+        for v in [
+            &mut self.h,
+            &mut self.dh,
+            &mut self.h_x,
+            &mut self.j,
+            &mut self.dj,
+            &mut self.j_x,
+            &mut self.jz,
+            &mut self.djz,
+            &mut self.jz_z,
+        ] {
+            v.clear();
+        }
+        // Outgoing: h₀ = −i e^{ix}/x, hₙ = hₙ₋₁ rₙ with the upward ratios; scaled by σₙ, so
+        // σₙ hₙ = (σₙ₋₁ hₙ₋₁)·rₙ/|rₙ(x_min)|.
+        let (sx, cx) = x.sin_cos();
+        let h0 = Complex::new(sx, -cx).scale(inv_x);
+        let mut hs = h0.scale(scales.h_ratio[0].recip());
+        let mut prev = Complex::new(cx, sx).scale(inv_x * scales.h_ratio[0].recip());
+        // h₋₁ = e^{ix}/x (h_{−1} = j_{−1} + i y_{−1} = cos x/x + i sin x/x); σ₋₁ := σ₀.
+        let mut r = Complex::new(inv_x, -T::one());
+        for k in 0..=n {
+            if k > 0 {
+                r = if k == 1 {
+                    Complex::new(inv_x, -T::one())
+                } else {
+                    real(inv_x * int::<T>(2 * k - 1)) - inv(r)
+                };
+                prev = hs.scale(scales.h_ratio[k].recip());
+                hs = (hs * r).scale(scales.h_ratio[k].recip());
+            }
+            self.h.push(hs);
+            self.h_x.push(hs.scale(inv_x));
+            // [x hₖ]'/x = hₖ₋₁ − k hₖ/x, both scaled by σₖ.
+            self.dh.push(prev - hs.scale(inv_x * int::<T>(k)));
+        }
+        // Regular jₙ(x) = j₀ Π u_i with the downward ratios u_i = jᵢ/jᵢ₋₁, scaled by σₙ.
+        ratios_real(&mut self.u_x, x, 1, n);
+        let j0 = sx * inv_x;
+        let mut js = j0 / scales.h_ratio[0];
+        let mut jprev = cx * inv_x / scales.h_ratio[0]; // j₋₁ = cos x/x
+        for k in 0..=n {
+            if k > 0 {
+                jprev = js / scales.h_ratio[k];
+                js = js * self.u_x[k] / scales.h_ratio[k];
+            }
+            self.j.push(real(js));
+            self.j_x.push(real(js * inv_x));
+            self.dj.push(real(jprev - js * inv_x * int::<T>(k)));
+        }
+        // Particle: j_k(z) = j₀(z) Π u_i(z), scaled by c_k; j₀(z) e^{−|Im z_max|} without
+        // overflow: sin z = (e^{iz} − e^{−iz})/(2i).
+        ratios_complex(&mut self.u_z, z, n);
+        let sin_scaled = sin_scaled(z, scales.im_z_max);
+        let cos_scaled = cos_scaled(z, scales.im_z_max);
+        let mut jz = sin_scaled * inv_z.scale(scales.e_ratio[0].recip());
+        let mut jzprev = cos_scaled * inv_z.scale(scales.e_ratio[0].recip());
+        for k in 0..=n {
+            if k > 0 {
+                jzprev = jz.scale(scales.e_ratio[k].recip());
+                jz = (jz * self.u_z[k]).scale(scales.e_ratio[k].recip());
+            }
+            self.jz.push(jz);
+            self.jz_z.push(jz * inv_z);
+            self.djz.push(jzprev - jz * inv_z.scale(int::<T>(k)));
+        }
+    }
+}
+
+/// e^{iz}, e^{−iz} times e^{−a}, a ≥ |Im z|: with exponents ≤ 0.
+fn exps_scaled<T: Real>(z: Complex<T>, a: T) -> (Complex<T>, Complex<T>) {
+    let (s, c) = z.re.sin_cos();
+    let plus = Complex::new(c, s).scale((-z.im - a).exp());
+    let minus = Complex::new(c, -s).scale((z.im - a).exp());
+    (plus, minus)
+}
+
+/// sin z · e^{−a}.
+fn sin_scaled<T: Real>(z: Complex<T>, a: T) -> Complex<T> {
+    let (p, m) = exps_scaled(z, a);
+    // (e^{iz} − e^{−iz})/(2i) = −i (p − m)/2.
+    let d = p - m;
+    Complex::new(d.im, -d.re).scale(T::from_f64(0.5))
+}
+
+/// cos z · e^{−a}.
+fn cos_scaled<T: Real>(z: Complex<T>, a: T) -> Complex<T> {
+    let (p, m) = exps_scaled(z, a);
+    (p + m).scale(T::from_f64(0.5))
+}
+
+/// The normalized angular functions of mode m at cos θ = `ct`, sin θ = `st` > 0, for orders
+/// 0, …, n: d_{mn} = √((n−m)!/(n+m)!) Pₙᵐ(cos θ) (Mishchenko's dₙ, with his sign), πₙ = m dₙ/sin θ
+/// and τₙ = d dₙ/dθ.
+#[derive(Clone, Debug)]
+pub(crate) struct Angular<T> {
+    pub(crate) d: Vec<T>,
+    pub(crate) pi: Vec<T>,
+    pub(crate) tau: Vec<T>,
+}
+
+impl<T: Real> Angular<T> {
+    pub(crate) fn new() -> Self {
+        Self {
+            d: Vec::new(),
+            pi: Vec::new(),
+            tau: Vec::new(),
+        }
+    }
+
+    pub(crate) fn compute(&mut self, m: usize, n: usize, ct: T, st: T) {
+        let zero = T::zero();
+        self.d.clear();
+        self.d.resize(n + 1, zero);
+        self.pi.clear();
+        self.pi.resize(n + 1, zero);
+        self.tau.clear();
+        self.tau.resize(n + 1, zero);
+        if m > n {
+            return;
+        }
+        let mut am = T::one();
+        for k in 1..=m {
+            am *= (int::<T>(2 * k - 1) / int::<T>(2 * k)).sqrt();
+        }
+        self.d[m] = am * st.powi(m as i32);
+        if m < n {
+            self.d[m + 1] = int::<T>(2 * m + 1).sqrt() * ct * self.d[m];
+        }
+        for k in m + 1..n {
+            let a = int::<T>(k * k - m * m).sqrt();
+            let b = int::<T>((k + 1) * (k + 1) - m * m).sqrt();
+            self.d[k + 1] = (int::<T>(2 * k + 1) * ct * self.d[k] - a * self.d[k - 1]) / b;
+        }
+        let inv_st = st.recip();
+        for k in m.max(1)..=n {
+            let prev = if k > m { self.d[k - 1] } else { zero };
+            let a = int::<T>(k * k - m * m).sqrt();
+            self.tau[k] = (int::<T>(k) * ct * self.d[k] - a * prev) * inv_st;
+            self.pi[k] = int::<T>(m) * self.d[k] * inv_st;
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/kernel.rs b/scatrs/src/tmatrix/kernel.rs
new file mode 100644
index 0000000..806c1d5
--- /dev/null
+++ b/scatrs/src/tmatrix/kernel.rs
@@ -0,0 +1,461 @@
+//! The EBCM for a spheroid: per azimuthal mode m, the matrices Q and P = RgQ from surface
+//! integrals of the vector spherical wave functions, and Tᵐ = −P Q⁻¹ (Mishchenko, Travis & Lacis
+//! 2002, §5.8; Waterman 1971).
+//!
+//! Every element is a value of the reciprocity form B(X, Y) = ∫_S n̂·(X × ∇×Y − Y × ∇×X):
+//! Q_{nk} = −i B(Xₙ, Y_k) with X an outgoing wave of the medium and Y a regular wave of the
+//! particle (P: X regular). With k = 1, x = r(θ), z = s x, ν = −r'/r and the components
+//!
+//! Mₙ = γₙ (0, i hₙ pₙ, −hₙ τₙ),  Nₙ = γₙ (n(n+1) hₙ/x dₙ, Dhₙ τₙ, i Dhₙ pₙ),
+//!
+//! (pₙ = −πₙ for the outgoing wave, the (−1)^m X_{−mn} of the integrals; Dh = [x h]'/x), the
+//! element of row type a and column type b is −i 2π γₙ γ_k Σ w x² [T(A₁, B₁) + T(A₂, B₂)], with
+//! T(A, B) = A_θ B_φ − A_φ B_θ + ν (A_φ B_r − A_r B_φ), rows (A₁, A₂) = (Mₙ, Nₙ) or (Nₙ, Mₙ) and
+//! columns (B₁, B₂) = (s N_k, M_k) or (s M_k, N_k). A sphere gives T¹¹ = −bₙ, T²² = −aₙ.
+//!
+//! A spheroid is mirror-symmetric: the diagonal blocks vanish unless n + k is even and the
+//! off-diagonal ones unless it is odd, so each mode splits into two systems by the class
+//! (n + type) mod 2, and the integrands of the non-zero elements are even in cos θ: the
+//! quadrature runs over 0 < θ < π/2 and doubles.
+
+use num_complex::Complex;
+
+use super::functions::{Angular, Radial, Scales};
+use crate::complex::inv;
+use crate::mie::kernel::{int, real};
+use crate::scalar::Real;
+use crate::special::gauss_legendre;
+use crate::{Error, Result};
+
+/// A spheroid: semi-axes a (equatorial) and c (polar) in units of 1/k, from the equal-volume
+/// size parameter x and the aspect ratio ε = a/c.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct Spheroid<T> {
+    pub(crate) a: T,
+    pub(crate) c: T,
+}
+
+impl<T: Real> Spheroid<T> {
+    pub(crate) fn new(x: T, eps: T) -> Self {
+        let a = x * eps.cbrt();
+        Self { a, c: a / eps }
+    }
+
+    /// r(θ) and ν = −r'/r at cos θ = `ct`, sin θ = `st`.
+    fn at(&self, ct: T, st: T) -> (T, T) {
+        let (a2, c2) = (self.a * self.a, self.c * self.c);
+        let den = c2 * st * st + a2 * ct * ct;
+        let r = self.a * self.c / den.sqrt();
+        // r'/r = −(c² − a²) sin θ cos θ / den.
+        let nu = (c2 - a2) * st * ct / den;
+        (r, nu)
+    }
+
+    fn r_min(&self) -> T {
+        self.a.min(self.c)
+    }
+
+    fn r_max(&self) -> T {
+        self.a.max(self.c)
+    }
+}
+
+/// One mode's matrices, scaled (row n by σₙ, column k by c_k; see `super::functions`), in the
+/// two parity classes: `q[c]`, `p[c]` are K_c × K_c, row-major, indexed by `index[c]`.
+pub(crate) struct ModeMatrices<T> {
+    pub(crate) q: [Vec<Complex<T>>; 2],
+    pub(crate) p: [Vec<Complex<T>>; 2],
+    /// (type, order) of each index of a class.
+    pub(crate) index: [Vec<(usize, usize)>; 2],
+}
+
+/// γₙ = √((2n + 1)/(4π n(n + 1))).
+fn gamma<T: Real>(n: usize) -> T {
+    (int::<T>(2 * n + 1) / (T::PI() * int::<T>(4 * n * (n + 1)))).sqrt()
+}
+
+type Vec3<T> = [Complex<T>; 3];
+
+/// n̂·(A × B) for n̂ ∝ (1, ν, 0).
+fn triple<T: Real>(nu: T, a: &Vec3<T>, b: &Vec3<T>) -> Complex<T> {
+    a[1] * b[2] - a[2] * b[1] + (a[2] * b[0] - a[0] * b[2]).scale(nu)
+}
+
+/// Q and P of mode m with orders max(m, 1), …, n on n_theta nodes in 0 < θ < π/2.
+pub(crate) fn mode_matrices<T: Real>(
+    shape: &Spheroid<T>,
+    s: Complex<T>,
+    m: usize,
+    n: usize,
+    n_theta: usize,
+) -> Result<ModeMatrices<T>> {
+    let n0 = m.max(1);
+    if n < n0 {
+        return Err(Error::InvalidInput(format!(
+            "mode {m} needs at least {n0} orders, got {n}"
+        )));
+    }
+    // Index sets of the two classes: (type, order) with (order + type) % 2 == class.
+    let mut index: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
+    for t in 0..2 {
+        for k in n0..=n {
+            index[(k + t) % 2].push((t, k));
+        }
+    }
+    let size = [index[0].len(), index[1].len()];
+    let mut q: [Vec<Complex<T>>; 2] =
+        core::array::from_fn(|c| vec![real(T::zero()); size[c] * size[c]]);
+    let mut p: [Vec<Complex<T>>; 2] =
+        core::array::from_fn(|c| vec![real(T::zero()); size[c] * size[c]]);
+    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
+    let rule = gauss_legendre::<T>(2 * n_theta)?;
+    let mut radial = Radial::new();
+    let mut ang = Angular::new();
+    let gam: Vec<T> = (0..=n)
+        .map(|k| if k == 0 { T::zero() } else { gamma(k) })
+        .collect();
+    let i1 = Complex::new(T::zero(), T::one());
+    let zero3 = [real(T::zero()); 3];
+    // Field components per order: outgoing (h) and regular (j) M, N of the medium, and the
+    // particle's M, N; all without γ (applied per element).
+    let mut out_m = vec![zero3; n + 1];
+    let mut out_n = vec![zero3; n + 1];
+    let mut reg_m = vec![zero3; n + 1];
+    let mut reg_n = vec![zero3; n + 1];
+    let mut in_m = vec![zero3; n + 1];
+    let mut in_n = vec![zero3; n + 1];
+    for (&t, &w) in rule.nodes.iter().zip(&rule.weights) {
+        if t <= T::zero() {
+            continue;
+        }
+        let (ct, st) = (t, (T::one() - t * t).sqrt());
+        let (r, nu) = shape.at(ct, st);
+        radial.compute(r, s, n, &scales);
+        ang.compute(m, n, ct, st);
+        for k in n0..=n {
+            let (d, pi, tau) = (ang.d[k], ang.pi[k], ang.tau[k]);
+            let kk = int::<T>(k * (k + 1));
+            let po = real(-pi); // the outgoing waves' −π
+            let field = |h: Complex<T>, dh: Complex<T>, h_x: Complex<T>, p: Complex<T>| {
+                (
+                    [real(T::zero()), i1 * h * p, -(h.scale(tau))],
+                    [h_x.scale(kk * d), dh.scale(tau), i1 * dh * p],
+                )
+            };
+            (out_m[k], out_n[k]) = field(radial.h[k], radial.dh[k], radial.h_x[k], po);
+            (reg_m[k], reg_n[k]) = field(radial.j[k], radial.dj[k], radial.j_x[k], po);
+            (in_m[k], in_n[k]) = field(radial.jz[k], radial.djz[k], radial.jz_z[k], real(pi));
+        }
+        // 2 for the two halves of the range, 2π for φ, x² from dS (k = 1).
+        let fac = w * r * r * T::PI() * T::from_f64(4.0);
+        for c in 0..2 {
+            let idx = &index[c];
+            let kc = idx.len();
+            for (i, &(rt, nn)) in idx.iter().enumerate() {
+                let (o1, o2) = if rt == 0 {
+                    (&out_m[nn], &out_n[nn])
+                } else {
+                    (&out_n[nn], &out_m[nn])
+                };
+                let (g1, g2) = if rt == 0 {
+                    (&reg_m[nn], &reg_n[nn])
+                } else {
+                    (&reg_n[nn], &reg_m[nn])
+                };
+                for (j, &(ctp, kk_)) in idx.iter().enumerate() {
+                    let (b1, b2) = if ctp == 0 {
+                        (in_n[kk_].map(|v| v * s), &in_m[kk_])
+                    } else {
+                        (in_m[kk_].map(|v| v * s), &in_n[kk_])
+                    };
+                    let g = fac * gam[nn] * gam[kk_];
+                    let qv = triple(nu, o1, &b1) + triple(nu, o2, b2);
+                    let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
+                    // −i·(…)
+                    q[c][i * kc + j] = q[c][i * kc + j] + Complex::new(qv.im, -qv.re).scale(g);
+                    p[c][i * kc + j] = p[c][i * kc + j] + Complex::new(pv.im, -pv.re).scale(g);
+                }
+            }
+        }
+    }
+    Ok(ModeMatrices { q, p, index })
+}
+
+/// LU factorization with partial pivoting of a row-major n × n matrix, in place; returns the
+/// pivots.
+pub(crate) fn lu<T: Real>(a: &mut [Complex<T>], n: usize) -> Result<Vec<usize>> {
+    let mut piv: Vec<usize> = (0..n).collect();
+    for k in 0..n {
+        let mut best = k;
+        let mut big = T::zero();
+        for i in k..n {
+            let v = a[i * n + k].norm();
+            if v > big {
+                big = v;
+                best = i;
+            }
+        }
+        if big == T::zero() || !big.is_finite() {
+            return Err(Error::InvalidInput(format!(
+                "the EBCM matrix is singular or not finite at column {k}"
+            )));
+        }
+        if best != k {
+            for j in 0..n {
+                a.swap(k * n + j, best * n + j);
+            }
+            piv.swap(k, best);
+        }
+        let inv_p = inv(a[k * n + k]);
+        for i in k + 1..n {
+            let f = a[i * n + k] * inv_p;
+            a[i * n + k] = f;
+            if f.re == T::zero() && f.im == T::zero() {
+                continue;
+            }
+            for j in k + 1..n {
+                let v = a[k * n + j];
+                a[i * n + j] = a[i * n + j] - f * v;
+            }
+        }
+    }
+    Ok(piv)
+}
+
+/// Solves A x = b in place with the factors of `lu`.
+pub(crate) fn lu_solve<T: Real>(a: &[Complex<T>], piv: &[usize], n: usize, b: &mut [Complex<T>]) {
+    let mut x: Vec<Complex<T>> = piv.iter().map(|&i| b[i]).collect();
+    for i in 0..n {
+        let mut v = x[i];
+        for j in 0..i {
+            v = v - a[i * n + j] * x[j];
+        }
+        x[i] = v;
+    }
+    for i in (0..n).rev() {
+        let mut v = x[i];
+        for j in i + 1..n {
+            v = v - a[i * n + j] * x[j];
+        }
+        x[i] = v * inv(a[i * n + i]);
+    }
+    b.copy_from_slice(&x);
+}
+
+/// T = −P Q⁻¹ of one class (scaled as P and Q are): solved as Qᵀ Tᵀ = −Pᵀ.
+pub(crate) fn solve_t<T: Real>(
+    q: &[Complex<T>],
+    p: &[Complex<T>],
+    n: usize,
+) -> Result<Vec<Complex<T>>> {
+    let mut qt: Vec<Complex<T>> = (0..n * n).map(|ij| q[(ij % n) * n + ij / n]).collect();
+    let piv = lu(&mut qt, n)?;
+    let mut t = vec![real(T::zero()); n * n];
+    let mut col = vec![real(T::zero()); n];
+    for i in 0..n {
+        // Row i of T is column i of Tᵀ: Qᵀ (row i of T)ᵀ = −(row i of P)ᵀ.
+        for j in 0..n {
+            col[j] = -p[i * n + j];
+        }
+        lu_solve(&qt, &piv, n, &mut col);
+        t[i * n..(i + 1) * n].copy_from_slice(&col);
+    }
+    Ok(t)
+}
+
+/// The unscaling factor of T's element (row order n, column order k): σ_k/σₙ =
+/// |hₙ(x_min)|/|h_k(x_min)|, from the cumulative logarithms of the scale ratios.
+pub(crate) fn log_h(scales: &Scales<impl Real>) -> Vec<f64> {
+    let mut out = Vec::with_capacity(scales.h_ratio.len());
+    let mut acc = 0.0;
+    for v in &scales.h_ratio {
+        acc += v.approx_f64().ln();
+        out.push(acc);
+    }
+    out
+}
+
+/// One mode's T: the two parity classes' matrices (unscaled, row-major) and index sets.
+pub(crate) type ModeT<T> = ([Vec<Complex<T>>; 2], [Vec<(usize, usize)>; 2]);
+
+/// The T-matrix of a spheroid, mode by mode: for each m = 0, …, N the two classes' T (unscaled)
+/// with their index sets.
+pub(crate) struct TMatrixModes<T> {
+    pub(crate) n: usize,
+    pub(crate) modes: Vec<ModeT<T>>,
+}
+
+/// T of every mode for orders up to n and n_theta nodes in 0 < θ < π/2.
+pub(crate) fn tmatrix<T: Real>(
+    x: T,
+    eps: T,
+    s: Complex<T>,
+    n: usize,
+    n_theta: usize,
+) -> Result<TMatrixModes<T>> {
+    let shape = Spheroid::new(x, eps);
+    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
+    let lh = log_h(&scales);
+    let mut modes = Vec::with_capacity(n + 1);
+    for m in 0..=n {
+        let mm = mode_matrices(&shape, s, m, n, n_theta)?;
+        let mut ts: [Vec<Complex<T>>; 2] = [Vec::new(), Vec::new()];
+        for ((slot, index), (q, p)) in ts.iter_mut().zip(&mm.index).zip(mm.q.iter().zip(&mm.p)) {
+            let kc = index.len();
+            if kc == 0 {
+                continue;
+            }
+            let mut t = solve_t(q, p, kc)?;
+            // Undo the scaling: T = Σ⁻¹ T' Σ, T_{nk} = T'_{nk} σ_k/σₙ = T' |hₙ|/|h_k| (x_min).
+            for (i, &(_, nn)) in index.iter().enumerate() {
+                for (j, &(_, kk_)) in index.iter().enumerate() {
+                    let f = T::from_f64((lh[nn] - lh[kk_]).exp());
+                    t[i * kc + j] = t[i * kc + j].scale(f);
+                }
+            }
+            *slot = t;
+        }
+        modes.push((ts, mm.index));
+    }
+    Ok(TMatrixModes { n, modes })
+}
+
+impl<T: Real> TMatrixModes<T> {
+    /// The orientation-averaged ⟨C_ext⟩ and ⟨C_sca⟩ in units of 1/k² (Mishchenko 1991):
+    /// −2π Σ (2 − δ_{m0}) Re(T¹¹ₙₙ + T²²ₙₙ) and 2π Σ (2 − δ_{m0}) |T|².
+    pub(crate) fn cross_sections(&self) -> (T, T) {
+        let (mut ext, mut sca) = (T::zero(), T::zero());
+        for (m, (ts, index)) in self.modes.iter().enumerate() {
+            let w = if m == 0 { T::one() } else { T::from_f64(2.0) };
+            for (t, idx) in ts.iter().zip(index) {
+                let kc = idx.len();
+                for i in 0..kc {
+                    ext -= w * t[i * kc + i].re;
+                    for v in &t[i * kc..(i + 1) * kc] {
+                        sca += w * v.norm_sqr();
+                    }
+                }
+            }
+        }
+        let two_pi = T::PI() * T::from_f64(2.0);
+        (two_pi * ext, two_pi * sca)
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::mie::{Mie, MieOptions};
+    use crate::scalar::DoubleDouble as Dd;
+
+    /// Aspect ratio 1: T is diagonal with T¹¹ = −bₙ, T²² = −aₙ, in every mode.
+    #[test]
+    fn a_sphere_gives_the_mie_coefficients() {
+        let (x, m) = (3.0, Complex::new(1.5, 0.01));
+        let n = 12;
+        let t = tmatrix(x, 1.0, m, n, 30).unwrap();
+        let mut mie = Mie::<f64>::new();
+        let opts = MieOptions::new().with_n_max(n).with_allow_unvalidated(true);
+        let s = mie.compute(x, m, &opts).unwrap();
+        let (a, b) = (s.a(), s.b());
+        let mut worst = 0.0f64;
+        for (mm, (ts, index)) in t.modes.iter().enumerate() {
+            for c in 0..2 {
+                let kc = index[c].len();
+                for (i, &(ty, k)) in index[c].iter().enumerate() {
+                    let want = if ty == 0 { -b[k - 1] } else { -a[k - 1] };
+                    worst = worst.max((ts[c][i * kc + i] - want).norm() / want.norm().max(1e-300));
+                    for j in 0..kc {
+                        if j != i {
+                            assert!(ts[c][i * kc + j].norm() < 1e-13, "mode {mm} off-diagonal");
+                        }
+                    }
+                }
+            }
+        }
+        assert!(worst < 1e-12, "worst relative error {worst:e}");
+    }
+
+    /// A lossless spheroid conserves energy: ⟨C_ext⟩ = ⟨C_sca⟩.
+    #[test]
+    fn lossless_spheroids_conserve_energy() {
+        for (x, eps) in [(3.0, 2.0), (3.0, 0.5), (5.0, 1.5)] {
+            let t = tmatrix(x, eps, Complex::new(1.5, 0.0), 16, 40).unwrap();
+            let (ext, sca) = t.cross_sections();
+            assert!(
+                (ext - sca).abs() < 1e-11 * ext,
+                "x {x} eps {eps}: {ext} vs {sca}"
+            );
+        }
+    }
+
+    /// The study's lab (research/tmatrix, mpmath at 30 digits): x = 3, ε = 2, s = 1.5, N = 12,
+    /// 80 nodes over [−1, 1] give ⟨C_ext⟩ = 91.9311274132168 (in units of 1/k²).
+    #[test]
+    fn matches_the_reference_ebcm() {
+        let t = tmatrix(
+            Dd::from(3.0),
+            Dd::from(2.0),
+            Complex::new(Dd::from(1.5), Dd::from(0.0)),
+            12,
+            40,
+        )
+        .unwrap();
+        let (ext, _) = t.cross_sections();
+        assert!(
+            (ext.hi() - 91.931_127_413_216_8).abs() < 1e-12,
+            "{}",
+            ext.hi()
+        );
+    }
+}
+
+#[cfg(test)]
+mod diagnostics {
+    use super::*;
+    use crate::scalar::DoubleDouble as Dd;
+
+    fn dd(c: Complex<f64>) -> Complex<Dd> {
+        Complex::new(Dd::from(c.re), Dd::from(c.im))
+    }
+
+    /// f64 against double-double, plain integrals: run with
+    /// `cargo test --release -p scatrs --lib tmatrix::kernel::diagnostics -- --ignored --nocapture`.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn f64_against_double_double() {
+        let s = Complex::new(1.5, 0.01);
+        for &(x, eps, n, nth) in &[
+            (5.0, 2.0, 14, 30),
+            (10.0, 2.0, 24, 50),
+            (20.0, 2.0, 36, 70),
+            (15.0, 4.0, 40, 100),
+            (15.0, 0.333_333_333_333, 42, 90),
+            (30.0, 2.0, 50, 90),
+        ] {
+            let t = std::time::Instant::now();
+            let a = tmatrix(x, eps, s, n, nth).unwrap();
+            let t64 = t.elapsed();
+            let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth).unwrap();
+            let (mut big, mut diff) = (0.0f64, 0.0f64);
+            for (ma, mb) in a.modes.iter().zip(&b.modes) {
+                for c in 0..2 {
+                    for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
+                        let v = Complex::new(v.re.hi(), v.im.hi());
+                        big = big.max(v.norm());
+                        diff = diff.max((u - v).norm());
+                    }
+                }
+            }
+            let (ea, sa) = a.cross_sections();
+            let (eb, sb) = b.cross_sections();
+            println!(
+                "x {x} eps {eps} N {n}: T {:.1e}, C_ext {:.1e}, C_sca {:.1e} (f64 {:?})",
+                diff / big,
+                ((ea - eb.hi()) / eb.hi()).abs(),
+                ((sa - sb.hi()) / sb.hi()).abs(),
+                t64
+            );
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
new file mode 100644
index 0000000..3095b46
--- /dev/null
+++ b/scatrs/src/tmatrix/mod.rs
@@ -0,0 +1,9 @@
+//! The T-matrix of spheroids by the extended boundary condition method (EBCM), and the optical
+//! properties of randomly oriented spheroids. See the theory chapter of the documentation.
+
+// The kernel has no public entry point until the API lands later in phase 4a.
+#![allow(dead_code)]
+
+pub(crate) mod functions;
+pub(crate) mod kernel;
+pub(crate) mod random;
diff --git a/scatrs/src/tmatrix/random.rs b/scatrs/src/tmatrix/random.rs
new file mode 100644
index 0000000..2411847
--- /dev/null
+++ b/scatrs/src/tmatrix/random.rs
@@ -0,0 +1,393 @@
+//! The phase matrix of a randomly oriented axisymmetric particle from its T-matrix, by the rank
+//! decomposition of T (verified against a brute-force average over orientations to 10⁻¹⁵ in
+//! research/tmatrix/ranks.py).
+//!
+//! In the helicity basis W^p = (M + pN)/√2, T^{pσ} = (T¹¹ + σT¹² + pT²¹ + pσT²²)/2, and
+//! T^{pσ}_{−m} = T^{−p,−σ}_m. The rank-s components of T^{pσ} between orders n and n' are
+//! t⁽ˢ⁾(n, n') = Σ_μ (−1)^{n'−μ} C^{s0}_{nμ;n'−μ} T^{pσ}_{μ}(n, n') (T is diagonal in the mode for
+//! an axisymmetric particle); rotating the particle by R multiplies the q-th component by
+//! D^s_{q0}(R). For incident helicity σ along ẑ and scattered helicity p at (Θ, φ = 0),
+//!
+//!   A_{pσ}(Θ; R) = Σ_{s,q} D^s_{q0}(R) Φ_{sq},
+//!   Φ_{sq} = Σ_n farₙ(q + σ, p; Θ) w_n,  w_n = Σ_{n'} inc_{n'} (−1)^{n'−σ} C^{sq}_{n,q+σ;n',−σ} t⁽ˢ⁾(n, n'),
+//!
+//! with inc_{n'} = 2 i^{n'−1} √(π(2n'+1)) the plane wave's helicity coefficients and
+//! farₙ(m, p; Θ) = γₙ (−i)ⁿ (π_{mn} + p τ_{mn})(Θ) the far field of W^p_{nm}. Over orientations
+//! ⟨D^s_{q0} D^{s'*}_{q'0}⟩ = δ_{ss'} δ_{qq'}/(2s + 1), so ⟨A_{pσ} A*_{p'σ'}⟩ = Σ_{s,q} Φ Φ'*/(2s + 1).
+//! The linear amplitude matrix is S = U A U⁻¹, U = [ê₊ ê₋] = [[1, 1], [i, −i]]/√2 in (θ̂, φ̂),
+//! with S = [[S₂, S₃], [S₄, S₁]], and the phase matrix follows from ⟨S_{ij} S*_{kl}⟩.
+//!
+//! Cost: O(N⁴) for N orders and N + L/2 cosines.
+
+use num_complex::Complex;
+
+use super::functions::Angular;
+use super::kernel::TMatrixModes;
+use crate::mie::kernel::real;
+use crate::scalar::Real;
+use crate::special::clebsch::clebsch_gordan;
+
+/// The orientation-averaged phase matrix ⟨Z⟩ at cosines μ: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ in units
+/// of 1/k² per steradian (∫ F₁₁ dΩ = ⟨C_sca⟩).
+#[derive(Clone, Debug)]
+pub(crate) struct AveragedPhase<T> {
+    pub(crate) f11: Vec<T>,
+    pub(crate) f12: Vec<T>,
+    pub(crate) f22: Vec<T>,
+    pub(crate) f33: Vec<T>,
+    pub(crate) f34: Vec<T>,
+    pub(crate) f44: Vec<T>,
+}
+
+fn sgn<T: Real>(k: i64) -> T {
+    if k.rem_euclid(2) == 0 {
+        T::one()
+    } else {
+        -T::one()
+    }
+}
+
+/// i^k.
+fn i_pow<T: Real>(k: i64) -> Complex<T> {
+    match k.rem_euclid(4) {
+        0 => real(T::one()),
+        1 => Complex::new(T::zero(), T::one()),
+        2 => real(-T::one()),
+        _ => Complex::new(T::zero(), -T::one()),
+    }
+}
+
+impl<T: Real> TMatrixModes<T> {
+    /// T^{ij}_m(n, n') of mode m ≥ 0 as four dense (N+1)×(N+1) blocks (index by order).
+    fn dense(&self, m: usize) -> [Vec<Complex<T>>; 4] {
+        let n = self.n;
+        let size = (n + 1) * (n + 1);
+        let mut out: [Vec<Complex<T>>; 4] = core::array::from_fn(|_| vec![real(T::zero()); size]);
+        let (ts, index) = &self.modes[m];
+        for c in 0..2 {
+            let kc = index[c].len();
+            for (i, &(ti, ni)) in index[c].iter().enumerate() {
+                for (j, &(tj, nj)) in index[c].iter().enumerate() {
+                    out[ti * 2 + tj][ni * (n + 1) + nj] = ts[c][i * kc + j];
+                }
+            }
+        }
+        out
+    }
+
+    /// ⟨Z⟩ at the cosines `mu`.
+    pub(crate) fn random_orientation(&self, mu: &[T]) -> AveragedPhase<T> {
+        let n = self.n;
+        let n1 = n + 1;
+        let zero = real(T::zero());
+        let half = T::from_f64(0.5);
+        // Helicity blocks for μ = −N..N: hel[(p, σ)][μ + N][n * n1 + n'].
+        let mut hel: Vec<Vec<Vec<Complex<T>>>> = vec![vec![Vec::new(); 2 * n + 1]; 4];
+        for m in 0..=n {
+            let [t11, t12, t21, t22] = self.dense(m);
+            for (pi, p) in [1i64, -1].into_iter().enumerate() {
+                for (si, s) in [1i64, -1].into_iter().enumerate() {
+                    let block = |p: i64, s: i64| -> Vec<Complex<T>> {
+                        (0..n1 * n1)
+                            .map(|k| {
+                                (t11[k]
+                                    + t12[k].scale(T::from_f64(s as f64))
+                                    + t21[k].scale(T::from_f64(p as f64))
+                                    + t22[k].scale(T::from_f64((p * s) as f64)))
+                                .scale(half)
+                            })
+                            .collect()
+                    };
+                    hel[pi * 2 + si][n + m] = block(p, s);
+                    if m > 0 {
+                        // T^{pσ}_{−m} = T^{−p,−σ}_m.
+                        hel[pi * 2 + si][n - m] = block(-p, -s);
+                    }
+                }
+            }
+        }
+        // Rank components t[(p,σ)][n][n'] for s = |n − n'|, …, n + n'.
+        let mut cg = Vec::new();
+        let mut t_rank: Vec<Vec<Vec<Complex<T>>>> = vec![vec![Vec::new(); n1 * n1]; 4];
+        for nn in 1..=n {
+            for np in 1..=n {
+                let s_min = nn.abs_diff(np);
+                let len = nn + np - s_min + 1;
+                let mut acc: [Vec<Complex<T>>; 4] = core::array::from_fn(|_| vec![zero; len]);
+                let mu_max = nn.min(np) as i64;
+                for mu in -mu_max..=mu_max {
+                    let j0 = clebsch_gordan::<T>(nn, mu, np, -mu, &mut cg);
+                    let ph: T = sgn((np as i64) - mu);
+                    for (k, a) in acc.iter_mut().enumerate() {
+                        let tv = hel[k][(n as i64 + mu) as usize][nn * n1 + np];
+                        if tv == zero {
+                            continue;
+                        }
+                        for (q, &c) in cg.iter().enumerate() {
+                            let s = j0 + q;
+                            a[s - s_min] = a[s - s_min] + tv.scale(ph * c);
+                        }
+                    }
+                }
+                for k in 0..4 {
+                    t_rank[k][nn * n1 + np] = core::mem::take(&mut acc[k]);
+                }
+            }
+        }
+        let inc: Vec<Complex<T>> = (0..=n)
+            .map(|k| {
+                if k == 0 {
+                    zero
+                } else {
+                    i_pow::<T>(k as i64 - 1).scale(
+                        T::from_f64(2.0) * (T::PI() * T::from_f64((2 * k + 1) as f64)).sqrt(),
+                    )
+                }
+            })
+            .collect();
+        let gam: Vec<T> = (0..=n)
+            .map(|k| {
+                if k == 0 {
+                    T::zero()
+                } else {
+                    (T::from_f64((2 * k + 1) as f64)
+                        / (T::PI() * T::from_f64((4 * k * (k + 1)) as f64)))
+                    .sqrt()
+                }
+            })
+            .collect();
+        let nq = mu.len();
+        let s_max = 2 * n;
+        // prod[a][b][c][d][j] = ⟨A_{p_a σ_b} A*_{p_c σ_d}⟩ at node j.
+        let mut prod = vec![zero; 16 * nq];
+        let mut ang = Angular::new();
+        // far[p][n][j] for the current m, and Φ[(p,σ)][s][j] for the current q.
+        let mut phi: Vec<Vec<Complex<T>>> = vec![vec![zero; (s_max + 1) * nq]; 4];
+        let mut active = [false; 4];
+        let mut w = vec![zero; (s_max + 1) * n1];
+        let qmax = n as i64 + 1;
+        for q in -qmax..=qmax {
+            for v in &mut phi {
+                v.iter_mut().for_each(|x| *x = zero);
+            }
+            active = [false; 4];
+            for (si, sigma) in [1i64, -1].into_iter().enumerate() {
+                let m = q + sigma;
+                if m.unsigned_abs() as usize > n {
+                    continue;
+                }
+                let am = m.unsigned_abs() as usize;
+                // far_n(m, p; Θ_j) = γₙ (−i)ⁿ (π + p τ), the signed-m functions from |m|:
+                // d_{−m} = (−1)^m d_m, τ likewise, π_{−m} = (−m) d_{−m}/sinθ.
+                let mut far = vec![zero; 2 * n1 * nq];
+                for (j, &c) in mu.iter().enumerate() {
+                    let st = (T::one() - c * c).sqrt();
+                    ang.compute(am, n, c, st);
+                    let flip: T = if m < 0 { sgn(am as i64) } else { T::one() };
+                    for k in am.max(1)..=n {
+                        let tau = ang.tau[k] * flip;
+                        let pi_ = if m < 0 {
+                            -(ang.pi[k] * flip)
+                        } else {
+                            ang.pi[k]
+                        };
+                        let f = i_pow::<T>(-(k as i64)).scale(gam[k]);
+                        far[k * nq + j] = f.scale(pi_ + tau);
+                        far[(n1 + k) * nq + j] = f.scale(pi_ - tau);
+                    }
+                }
+                for (pi, _p) in [1i64, -1].into_iter().enumerate() {
+                    let key = pi * 2 + si;
+                    w.iter_mut().for_each(|x| *x = zero);
+                    for nn in am.max(1)..=n {
+                        for np in 1..=n {
+                            let tr = &t_rank[key][nn * n1 + np];
+                            if tr.is_empty() {
+                                continue;
+                            }
+                            let s_min = nn.abs_diff(np);
+                            let j0 = clebsch_gordan::<T>(nn, m, np, -sigma, &mut cg);
+                            let ph: T = sgn(np as i64 - sigma);
+                            let pre = inc[np].scale(ph);
+                            for (k, &c) in cg.iter().enumerate() {
+                                let s = j0 + k;
+                                if s < s_min || s - s_min >= tr.len() {
+                                    continue;
+                                }
+                                w[s * n1 + nn] = w[s * n1 + nn] + pre * tr[s - s_min].scale(c);
+                            }
+                        }
+                    }
+                    let fr = &far[pi * n1 * nq..(pi + 1) * n1 * nq];
+                    for s in 0..=s_max {
+                        for nn in am.max(1)..=n {
+                            let wv = w[s * n1 + nn];
+                            if wv == zero {
+                                continue;
+                            }
+                            for j in 0..nq {
+                                phi[key][s * nq + j] = phi[key][s * nq + j] + wv * fr[nn * nq + j];
+                            }
+                        }
+                    }
+                    active[key] = true;
+                }
+            }
+            for a in 0..4 {
+                if !active[a] {
+                    continue;
+                }
+                for b in 0..4 {
+                    if !active[b] {
+                        continue;
+                    }
+                    let (pa, sa) = (a / 2, a % 2);
+                    let (pb, sb) = (b / 2, b % 2);
+                    let slot = ((pa * 2 + sa) * 2 + pb) * 2 + sb;
+                    for s in 0..=s_max {
+                        let wt = T::from_f64((2 * s + 1) as f64).recip();
+                        for j in 0..nq {
+                            let v = phi[a][s * nq + j] * phi[b][s * nq + j].conj();
+                            prod[slot * nq + j] = prod[slot * nq + j] + v.scale(wt);
+                        }
+                    }
+                }
+            }
+        }
+        let _ = active;
+        mueller(&prod, nq)
+    }
+}
+
+/// The phase matrix from the helicity products: ⟨S_{ij} S*_{kl}⟩ = Σ U_{ia} U⁻¹_{bj} conj(U_{kc}
+/// U⁻¹_{dl}) ⟨A_{ab} A*_{cd}⟩.
+fn mueller<T: Real>(prod: &[Complex<T>], nq: usize) -> AveragedPhase<T> {
+    let r = T::from_f64(0.5).sqrt();
+    let one = real(r);
+    let i1 = Complex::new(T::zero(), r);
+    let u = [[one, one], [i1, -i1]];
+    let ui = [[one, -i1], [one, i1]];
+    let mut out = AveragedPhase {
+        f11: Vec::with_capacity(nq),
+        f12: Vec::with_capacity(nq),
+        f22: Vec::with_capacity(nq),
+        f33: Vec::with_capacity(nq),
+        f34: Vec::with_capacity(nq),
+        f44: Vec::with_capacity(nq),
+    };
+    let half = T::from_f64(0.5);
+    for j in 0..nq {
+        let ss = |i: usize, jj: usize, k: usize, l: usize| -> Complex<T> {
+            let mut acc = real(T::zero());
+            for a in 0..2 {
+                for b in 0..2 {
+                    for c in 0..2 {
+                        for d in 0..2 {
+                            let slot = ((a * 2 + b) * 2 + c) * 2 + d;
+                            let f = u[i][a] * ui[b][jj] * (u[k][c] * ui[d][l]).conj();
+                            acc = acc + f * prod[slot * nq + j];
+                        }
+                    }
+                }
+            }
+            acc
+        };
+        let s2s2 = ss(0, 0, 0, 0).re;
+        let s1s1 = ss(1, 1, 1, 1).re;
+        let s3s3 = ss(0, 1, 0, 1).re;
+        let s4s4 = ss(1, 0, 1, 0).re;
+        let s1s2 = ss(1, 1, 0, 0);
+        let s3s4 = ss(0, 1, 1, 0);
+        out.f11.push(half * (s1s1 + s2s2 + s3s3 + s4s4));
+        out.f12.push(half * (s2s2 - s1s1 + s4s4 - s3s3));
+        out.f22.push(half * (s2s2 + s1s1 - s4s4 - s3s3));
+        out.f33.push(s1s2.re + s3s4.re);
+        out.f34.push(-s1s2.im - s3s4.im);
+        out.f44.push(s1s2.re - s3s4.re);
+    }
+    out
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::tmatrix::kernel::tmatrix;
+
+    /// research/tmatrix/ranks.py: x = 2, ε = 1.8, s = 1.5 + 0.02i, N = 6, 30 nodes over [−1, 1]
+    /// (15 here), at Θ = 0.4: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ (brute-force average, 10⁻¹⁵).
+    #[test]
+    fn matches_the_brute_force_average() {
+        let t = tmatrix(2.0, 1.8, Complex::new(1.5, 0.02), 6, 15).unwrap();
+        let mu = [0.4f64.cos()];
+        let f = t.random_orientation(&mu);
+        let got = [f.f11[0], f.f12[0], f.f22[0], f.f33[0], f.f34[0], f.f44[0]];
+        let want = [6.980506, -0.33857, 6.912394, 6.902738, 0.118319, 6.84077];
+        for (g, w) in got.iter().zip(want) {
+            assert!((g - w).abs() < 2e-6, "{got:?} vs {want:?}");
+        }
+    }
+
+    /// Aspect ratio 1: the average is the sphere's phase matrix, (4/x²)⟨Z⟩ = Mie's unnormalized
+    /// one, with F₂₂ = F₁₁ and F₄₄ = F₃₃.
+    #[test]
+    fn a_sphere_gives_the_mie_phase_matrix() {
+        use crate::mie::{Mie, MieOptions, Normalization, PhaseBuffers};
+        let (x, m, n) = (3.0, Complex::new(1.5, 0.01), 14);
+        let t = tmatrix(x, 1.0, m, n, 30).unwrap();
+        let mu: Vec<f64> = (0..7).map(|k| -0.95 + 0.3 * f64::from(k)).collect();
+        let f = t.random_orientation(&mu);
+        let mut mie = Mie::<f64>::new();
+        let opts = MieOptions::new().with_n_max(n).with_allow_unvalidated(true);
+        let s = mie.compute(x, m, &opts).unwrap();
+        let (mut f11, mut f12, mut f33, mut f34) =
+            (vec![0.0; 7], vec![0.0; 7], vec![0.0; 7], vec![0.0; 7]);
+        s.phase_matrix(
+            &mu,
+            Normalization::Unnormalized,
+            PhaseBuffers {
+                f11: &mut f11,
+                f12: &mut f12,
+                f33: &mut f33,
+                f34: &mut f34,
+            },
+        )
+        .unwrap();
+        let k = 4.0 / (x * x);
+        for j in 0..7 {
+            let scale = f11[j].abs();
+            for (got, want) in [
+                (k * f.f11[j], f11[j]),
+                (k * f.f22[j], f11[j]),
+                (k * f.f12[j], f12[j]),
+                (k * f.f33[j], f33[j]),
+                (k * f.f44[j], f33[j]),
+                (k * f.f34[j], f34[j]),
+            ] {
+                assert!(
+                    (got - want).abs() < 1e-12 * scale,
+                    "μ {}: {got} vs {want}",
+                    mu[j]
+                );
+            }
+        }
+    }
+
+    /// 2π ∫ F₁₁ dμ = ⟨C_sca⟩ from T, for a spheroid (Gauss–Legendre, exact for the polynomial).
+    #[test]
+    fn the_phase_function_integrates_to_the_scattering_cross_section() {
+        let n = 10;
+        let t = tmatrix(3.0, 0.5, Complex::new(1.5, 0.02), n, 30).unwrap();
+        let rule = crate::special::gauss_legendre::<f64>(2 * n + 2).unwrap();
+        let f = t.random_orientation(&rule.nodes);
+        let integral: f64 = f
+            .f11
+            .iter()
+            .zip(&rule.weights)
+            .map(|(v, w)| v * w)
+            .sum::<f64>()
+            * 2.0
+            * core::f64::consts::PI;
+        let (_, sca) = t.cross_sections();
+        assert!((integral - sca).abs() < 1e-12 * sca, "{integral} vs {sca}");
+    }
+}
````

### Task 2: Cancellation-free outgoing-wave integrals

- [ ] Add `tmatrix/regular.rs`: the eight radial product types, `Series` (lengths adaptive to x_max), `NodeSeries::forms` (R⁺ by its series or as R − R_{≤cut}).
- [ ] In `all_modes`, accumulate the three forms per element with their magnitude sums and keep the smallest (n ≠ k; powers < 0 only for |n − k| ≥ 3).
- [ ] Test: the series reproduce the Bessel factors on [x_min, x_max] to rounding in double-double.
- [ ] Diagnose with the ignored `diagnostics` tests: f64 against double-double, x = 15, ε = 4 → 3·10⁻⁴.

The commit (`8b2aa3d`):

````diff
diff --git a/scatrs/src/tmatrix/kernel.rs b/scatrs/src/tmatrix/kernel.rs
index 806c1d5..3a214f2 100644
--- a/scatrs/src/tmatrix/kernel.rs
+++ b/scatrs/src/tmatrix/kernel.rs
@@ -21,6 +21,7 @@
 use num_complex::Complex;
 
 use super::functions::{Angular, Radial, Scales};
+use super::regular::{Series, angular_factors};
 use crate::complex::inv;
 use crate::mie::kernel::{int, real};
 use crate::scalar::Real;
@@ -181,6 +182,216 @@ pub(crate) fn mode_matrices<T: Real>(
     Ok(ModeMatrices { q, p, index })
 }
 
+/// Q and P of every mode m = 0, …, n, on n_theta nodes in 0 < θ < π/2, nodes outside and modes
+/// inside so that each node's radial functions and regularized products serve every mode. With
+/// `regularize`, each element of Q keeps the cleanest of its three exact forms (`super::regular`).
+pub(crate) fn all_modes<T: Real>(
+    shape: &Spheroid<T>,
+    s: Complex<T>,
+    n: usize,
+    n_theta: usize,
+    regularize: bool,
+) -> Result<Vec<ModeMatrices<T>>> {
+    let zero = real(T::zero());
+    let mut modes: Vec<ModeMatrices<T>> = Vec::with_capacity(n + 1);
+    // Per mode and class: Y's three forms and their magnitude sums.
+    let mut yv: Vec<[[Vec<Complex<T>>; 3]; 2]> = Vec::with_capacity(n + 1);
+    let mut ya: Vec<[[Vec<T>; 3]; 2]> = Vec::with_capacity(n + 1);
+    for m in 0..=n {
+        let n0 = m.max(1);
+        let mut index: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
+        for t in 0..2 {
+            for k in n0..=n {
+                index[(k + t) % 2].push((t, k));
+            }
+        }
+        let size = [index[0].len(), index[1].len()];
+        let mk = |c: usize| vec![zero; size[c] * size[c]];
+        yv.push(core::array::from_fn(|c| core::array::from_fn(|_| mk(c))));
+        ya.push(core::array::from_fn(|c| {
+            core::array::from_fn(|_| vec![T::zero(); size[c] * size[c]])
+        }));
+        modes.push(ModeMatrices {
+            q: core::array::from_fn(mk),
+            p: core::array::from_fn(mk),
+            index,
+        });
+    }
+    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
+    let series = Series::new(shape.r_min(), shape.r_max(), s, n, &scales);
+    let rule = gauss_legendre::<T>(2 * n_theta)?;
+    let mut radial = Radial::new();
+    let mut ang = Angular::new();
+    let gam: Vec<T> = (0..=n)
+        .map(|k| if k == 0 { T::zero() } else { gamma(k) })
+        .collect();
+    let i1 = Complex::new(T::zero(), T::one());
+    let zero3 = [zero; 3];
+    let mut reg_m = vec![zero3; n + 1];
+    let mut reg_n = vec![zero3; n + 1];
+    let mut in_m = vec![zero3; n + 1];
+    let mut in_n = vec![zero3; n + 1];
+    let n1 = n + 1;
+    // forms[(t * n1 + n) * n1 + k] at the node.
+    let mut forms = Vec::new();
+    let mut full = vec![zero; 8 * n1 * n1];
+    let (a2, c2) = (shape.a * shape.a, shape.c * shape.c);
+    let beta = c2.recip() - a2.recip();
+    for (&t, &w) in rule.nodes.iter().zip(&rule.weights) {
+        if t <= T::zero() {
+            continue;
+        }
+        let (ct, st) = (t, (T::one() - t * t).sqrt());
+        let (r, nu) = shape.at(ct, st);
+        radial.compute(r, s, n, &scales);
+        let r2 = r * r;
+        let x_e = [r2, r2 * r2];
+        // The y-part radial factors (scaled yₙ = Im of the scaled hₙ for real x) and the
+        // particle's, per kind: plain, [x f]'/x, f/x.
+        let yk = |kind: usize, nn: usize| match kind {
+            0 => radial.h[nn].im,
+            1 => radial.dh[nn].im,
+            _ => radial.h_x[nn].im,
+        };
+        let jk = |kind: usize, kk: usize| match kind {
+            0 => radial.jz[kk],
+            1 => radial.djz[kk],
+            _ => radial.jz_z[kk],
+        };
+        const KINDS: [(usize, usize, usize); 8] = [
+            (0, 0, 0),
+            (0, 1, 0),
+            (1, 0, 0),
+            (1, 1, 0),
+            (0, 2, 1),
+            (1, 2, 1),
+            (2, 0, 1),
+            (2, 1, 1),
+        ];
+        for (ti, &(yki, jki, ei)) in KINDS.iter().enumerate() {
+            for nn in 1..=n {
+                for kk in 1..=n {
+                    full[(ti * n1 + nn) * n1 + kk] = jk(jki, kk).scale(yk(yki, nn) * x_e[ei]);
+                }
+            }
+        }
+        if regularize {
+            let ns = series.at(r / shape.r_min());
+            forms.clear();
+            for ti in 0..8 {
+                for nn in 0..=n {
+                    for kk in 0..=n {
+                        if nn == 0 || kk == 0 || nn == kk {
+                            forms.push(None);
+                            continue;
+                        }
+                        let f = full[(ti * n1 + nn) * n1 + kk];
+                        forms.push(Some(ns.forms(ti, nn, kk, f, x_e)));
+                    }
+                }
+            }
+        }
+        // 2 for the two halves of the range, 2π for φ.
+        let g0 = w * T::PI() * T::from_f64(4.0);
+        let fac = g0 * r2;
+        let nu_prime = -(beta * st * ct);
+        for (m, mm) in modes.iter_mut().enumerate() {
+            let n0 = m.max(1);
+            ang.compute(m, n, ct, st);
+            for k in n0..=n {
+                let (d, pi, tau) = (ang.d[k], ang.pi[k], ang.tau[k]);
+                let kk = int::<T>(k * (k + 1));
+                let field = |h: Complex<T>, dh: Complex<T>, h_x: Complex<T>, p: Complex<T>| {
+                    (
+                        [real(T::zero()), i1 * h * p, -(h.scale(tau))],
+                        [h_x.scale(kk * d), dh.scale(tau), i1 * dh * p],
+                    )
+                };
+                (reg_m[k], reg_n[k]) = field(radial.j[k], radial.dj[k], radial.j_x[k], real(-pi));
+                (in_m[k], in_n[k]) = field(radial.jz[k], radial.djz[k], radial.jz_z[k], real(pi));
+            }
+            for c in 0..2 {
+                let idx = &mm.index[c];
+                let kc = idx.len();
+                for (i, &(rt, nn)) in idx.iter().enumerate() {
+                    let (g1, g2) = if rt == 0 {
+                        (&reg_m[nn], &reg_n[nn])
+                    } else {
+                        (&reg_n[nn], &reg_m[nn])
+                    };
+                    let an = (ang.d[nn], ang.pi[nn], ang.tau[nn]);
+                    for (j, &(ctp, kk_)) in idx.iter().enumerate() {
+                        let (b1, b2) = if ctp == 0 {
+                            (in_n[kk_].map(|v| v * s), &in_m[kk_])
+                        } else {
+                            (in_m[kk_].map(|v| v * s), &in_n[kk_])
+                        };
+                        let gg = gam[nn] * gam[kk_];
+                        let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
+                        let slot = i * kc + j;
+                        mm.p[c][slot] = mm.p[c][slot] + Complex::new(pv.im, -pv.re).scale(fac * gg);
+                        // Y: Σ_t a_t R_t in its three forms.
+                        let ak = (ang.d[kk_], ang.pi[kk_], ang.tau[kk_]);
+                        let a = angular_factors(rt, ctp, s, nu_prime, nn, kk_, an, ak);
+                        let ge = g0 * gg;
+                        let (mut va, mut aa) = (zero, T::zero());
+                        let (mut vb, mut ab) = (zero, T::zero());
+                        let (mut vc, mut ac) = (zero, T::zero());
+                        for (ti, &at) in a.iter().enumerate() {
+                            if at == zero {
+                                continue;
+                            }
+                            let fi = (ti * n1 + nn) * n1 + kk_;
+                            let rf = full[fi];
+                            let mag = at.norm();
+                            va = va + at * rf;
+                            aa += mag * rf.norm();
+                            if let Some(Some(f)) = forms.get(fi) {
+                                vc = vc + at * f.le0;
+                                ac += mag * f.le0_abs;
+                                vb = vb + at * f.lt0;
+                                ab += mag * f.lt0_abs;
+                            }
+                        }
+                        let ys = &mut yv[m][c];
+                        let ys_abs = &mut ya[m][c];
+                        ys[0][slot] = ys[0][slot] + va.scale(ge);
+                        ys_abs[0][slot] += aa * ge.abs();
+                        ys[1][slot] = ys[1][slot] + vb.scale(ge);
+                        ys_abs[1][slot] += ab * ge.abs();
+                        ys[2][slot] = ys[2][slot] + vc.scale(ge);
+                        ys_abs[2][slot] += ac * ge.abs();
+                    }
+                }
+            }
+        }
+    }
+    // Q = P + Y in the cleanest exact form of each element: the plain one, minus the powers < 0
+    // (|n − k| ≥ 3), or minus the powers ≤ 0 (n ≠ k).
+    for (m, mm) in modes.iter_mut().enumerate() {
+        for c in 0..2 {
+            let idx = mm.index[c].clone();
+            let kc = idx.len();
+            for (i, &(_, nn)) in idx.iter().enumerate() {
+                for (j, &(_, kk_)) in idx.iter().enumerate() {
+                    let slot = i * kc + j;
+                    let mut best = 0;
+                    if regularize && nn != kk_ {
+                        if ya[m][c][2][slot] < ya[m][c][best][slot] {
+                            best = 2;
+                        }
+                        if nn.abs_diff(kk_) >= 3 && ya[m][c][1][slot] < ya[m][c][best][slot] {
+                            best = 1;
+                        }
+                    }
+                    mm.q[c][slot] = mm.p[c][slot] + yv[m][c][best][slot];
+                }
+            }
+        }
+    }
+    Ok(modes)
+}
+
 /// LU factorization with partial pivoting of a row-major n × n matrix, in place; returns the
 /// pivots.
 pub(crate) fn lu<T: Real>(a: &mut [Complex<T>], n: usize) -> Result<Vec<usize>> {
@@ -285,20 +496,22 @@ pub(crate) struct TMatrixModes<T> {
     pub(crate) modes: Vec<ModeT<T>>,
 }
 
-/// T of every mode for orders up to n and n_theta nodes in 0 < θ < π/2.
+/// T of every mode for orders up to n and n_theta nodes in 0 < θ < π/2, with or without the
+/// regularized integrals.
 pub(crate) fn tmatrix<T: Real>(
     x: T,
     eps: T,
     s: Complex<T>,
     n: usize,
     n_theta: usize,
+    regularize: bool,
 ) -> Result<TMatrixModes<T>> {
     let shape = Spheroid::new(x, eps);
     let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
     let lh = log_h(&scales);
+    let all = all_modes(&shape, s, n, n_theta, regularize)?;
     let mut modes = Vec::with_capacity(n + 1);
-    for m in 0..=n {
-        let mm = mode_matrices(&shape, s, m, n, n_theta)?;
+    for mm in all {
         let mut ts: [Vec<Complex<T>>; 2] = [Vec::new(), Vec::new()];
         for ((slot, index), (q, p)) in ts.iter_mut().zip(&mm.index).zip(mm.q.iter().zip(&mm.p)) {
             let kc = index.len();
@@ -353,7 +566,7 @@ mod tests {
     fn a_sphere_gives_the_mie_coefficients() {
         let (x, m) = (3.0, Complex::new(1.5, 0.01));
         let n = 12;
-        let t = tmatrix(x, 1.0, m, n, 30).unwrap();
+        let t = tmatrix(x, 1.0, m, n, 30, true).unwrap();
         let mut mie = Mie::<f64>::new();
         let opts = MieOptions::new().with_n_max(n).with_allow_unvalidated(true);
         let s = mie.compute(x, m, &opts).unwrap();
@@ -380,7 +593,7 @@ mod tests {
     #[test]
     fn lossless_spheroids_conserve_energy() {
         for (x, eps) in [(3.0, 2.0), (3.0, 0.5), (5.0, 1.5)] {
-            let t = tmatrix(x, eps, Complex::new(1.5, 0.0), 16, 40).unwrap();
+            let t = tmatrix(x, eps, Complex::new(1.5, 0.0), 16, 40, true).unwrap();
             let (ext, sca) = t.cross_sections();
             assert!(
                 (ext - sca).abs() < 1e-11 * ext,
@@ -399,6 +612,7 @@ mod tests {
             Complex::new(Dd::from(1.5), Dd::from(0.0)),
             12,
             40,
+            true,
         )
         .unwrap();
         let (ext, _) = t.cross_sections();
@@ -434,23 +648,40 @@ mod diagnostics {
             (30.0, 2.0, 50, 90),
         ] {
             let t = std::time::Instant::now();
-            let a = tmatrix(x, eps, s, n, nth).unwrap();
+            let a = tmatrix(x, eps, s, n, nth, true).unwrap();
             let t64 = t.elapsed();
-            let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth).unwrap();
+            let plain = tmatrix(x, eps, s, n, nth, false).unwrap();
+            let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, false).unwrap();
+            let b_reg = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
+            let err = |u: &TMatrixModes<f64>| {
+                let (mut big, mut diff) = (0.0f64, 0.0f64);
+                for (ma, mb) in u.modes.iter().zip(&b.modes) {
+                    for c in 0..2 {
+                        for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
+                            let v = Complex::new(v.re.hi(), v.im.hi());
+                            big = big.max(v.norm());
+                            diff = diff.max((u - v).norm());
+                        }
+                    }
+                }
+                diff / big
+            };
             let (mut big, mut diff) = (0.0f64, 0.0f64);
-            for (ma, mb) in a.modes.iter().zip(&b.modes) {
+            for (ma, mb) in b_reg.modes.iter().zip(&b.modes) {
                 for c in 0..2 {
                     for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
-                        let v = Complex::new(v.re.hi(), v.im.hi());
-                        big = big.max(v.norm());
-                        diff = diff.max((u - v).norm());
+                        big = big.max(v.re.hi().abs() + v.im.hi().abs());
+                        diff = diff.max((u.re - v.re).hi().abs() + (u.im - v.im).hi().abs());
                     }
                 }
             }
             let (ea, sa) = a.cross_sections();
             let (eb, sb) = b.cross_sections();
             println!(
-                "x {x} eps {eps} N {n}: T {:.1e}, C_ext {:.1e}, C_sca {:.1e} (f64 {:?})",
+                "x {x} eps {eps} N {n}: T plain {:.1e}, regularized {:.1e} (dd forms agree {:.1e}); \
+                 C_ext {:.1e}, C_sca {:.1e} (f64 {:?})",
+                err(&plain),
+                err(&a),
                 diff / big,
                 ((ea - eb.hi()) / eb.hi()).abs(),
                 ((sa - sb.hi()) / sb.hi()).abs(),
@@ -459,3 +690,61 @@ mod diagnostics {
         }
     }
 }
+
+#[cfg(test)]
+mod element_diagnostics {
+    use super::*;
+    use crate::scalar::DoubleDouble as Dd;
+
+    /// Q element errors, f64 (plain and regularized) against double-double (plain and
+    /// regularized), worst per mode-0/1: run with `--ignored --nocapture`.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn q_elements() {
+        let s = Complex::new(1.5, 0.01);
+        let sd = Complex::new(Dd::from(1.5), Dd::from(0.01));
+        for &(x, eps, n, nth) in &[
+            (10.0, 2.0, 24, 50),
+            (15.0, 0.333_333_333_333, 42, 90),
+            (15.0, 4.0, 40, 100),
+        ] {
+            let sh = Spheroid::new(x, eps);
+            let shd = Spheroid::new(Dd::from(x), Dd::from(eps));
+            let qp = all_modes(&sh, s, n, nth, false).unwrap();
+            let qr = all_modes(&sh, s, n, nth, true).unwrap();
+            let dp = all_modes(&shd, sd, n, nth, false).unwrap();
+            let dr = all_modes(&shd, sd, n, nth, true).unwrap();
+            for m in [0usize, 1] {
+                for c in 0..2 {
+                    let idx = &qp[m].index[c];
+                    let kc = idx.len();
+                    let mut worst: Vec<(f64, f64, f64, usize, usize, usize, usize)> = Vec::new();
+                    for (i, &(rt, nn)) in idx.iter().enumerate() {
+                        let row_max = (0..kc)
+                            .map(|j| dp[m].q[c][i * kc + j].norm().hi())
+                            .fold(0.0, f64::max);
+                        for (j, &(ctp, kk)) in idx.iter().enumerate() {
+                            let slot = i * kc + j;
+                            let refv =
+                                Complex::new(dr[m].q[c][slot].re.hi(), dr[m].q[c][slot].im.hi());
+                            let refp =
+                                Complex::new(dp[m].q[c][slot].re.hi(), dp[m].q[c][slot].im.hi());
+                            let scale = refv.norm().max(1e-30 * row_max);
+                            let ep = (qp[m].q[c][slot] - refv).norm() / scale;
+                            let er = (qr[m].q[c][slot] - refv).norm() / scale;
+                            let ed = (refp - refv).norm() / scale;
+                            worst.push((er, ep, ed, rt, nn, ctp, kk));
+                        }
+                    }
+                    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
+                    for w in worst.iter().take(3) {
+                        println!(
+                            "x {x} eps {eps:.3} m {m} c {c}: ({},{})-({},{}): reg {:.1e} plain {:.1e} dd-forms {:.1e}",
+                            w.3, w.4, w.5, w.6, w.0, w.1, w.2
+                        );
+                    }
+                }
+            }
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
index 3095b46..3a79e3e 100644
--- a/scatrs/src/tmatrix/mod.rs
+++ b/scatrs/src/tmatrix/mod.rs
@@ -7,3 +7,4 @@
 pub(crate) mod functions;
 pub(crate) mod kernel;
 pub(crate) mod random;
+pub(crate) mod regular;
diff --git a/scatrs/src/tmatrix/random.rs b/scatrs/src/tmatrix/random.rs
index 2411847..1a21bf6 100644
--- a/scatrs/src/tmatrix/random.rs
+++ b/scatrs/src/tmatrix/random.rs
@@ -317,7 +317,7 @@ mod tests {
     /// (15 here), at Θ = 0.4: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ (brute-force average, 10⁻¹⁵).
     #[test]
     fn matches_the_brute_force_average() {
-        let t = tmatrix(2.0, 1.8, Complex::new(1.5, 0.02), 6, 15).unwrap();
+        let t = tmatrix(2.0, 1.8, Complex::new(1.5, 0.02), 6, 15, true).unwrap();
         let mu = [0.4f64.cos()];
         let f = t.random_orientation(&mu);
         let got = [f.f11[0], f.f12[0], f.f22[0], f.f33[0], f.f34[0], f.f44[0]];
@@ -333,7 +333,7 @@ mod tests {
     fn a_sphere_gives_the_mie_phase_matrix() {
         use crate::mie::{Mie, MieOptions, Normalization, PhaseBuffers};
         let (x, m, n) = (3.0, Complex::new(1.5, 0.01), 14);
-        let t = tmatrix(x, 1.0, m, n, 30).unwrap();
+        let t = tmatrix(x, 1.0, m, n, 30, true).unwrap();
         let mu: Vec<f64> = (0..7).map(|k| -0.95 + 0.3 * f64::from(k)).collect();
         let f = t.random_orientation(&mu);
         let mut mie = Mie::<f64>::new();
@@ -376,7 +376,7 @@ mod tests {
     #[test]
     fn the_phase_function_integrates_to_the_scattering_cross_section() {
         let n = 10;
-        let t = tmatrix(3.0, 0.5, Complex::new(1.5, 0.02), n, 30).unwrap();
+        let t = tmatrix(3.0, 0.5, Complex::new(1.5, 0.02), n, 30, true).unwrap();
         let rule = crate::special::gauss_legendre::<f64>(2 * n + 2).unwrap();
         let f = t.random_orientation(&rule.nodes);
         let integral: f64 = f
diff --git a/scatrs/src/tmatrix/regular.rs b/scatrs/src/tmatrix/regular.rs
new file mode 100644
index 0000000..e0e6135
--- /dev/null
+++ b/scatrs/src/tmatrix/regular.rs
@@ -0,0 +1,406 @@
+//! The regularized yₙ part of the EBCM integrals.
+//!
+//! Q = P + Y, with P the integrals of the regular jₙ part of hₙ = jₙ + i yₙ and Y those of i yₙ:
+//! per element Y = 4π γₙ γ_k Σ w Σ_t a_t(θ) R_t(x), a sum of eight radial products
+//! R_t = x^e Y_t(x) J_t(s x) (Y_t ∈ {yₙ, [x yₙ]'/x, yₙ/x}, J_t ∈ {j_k, [z j_k]'/z, j_k/z}, e = 2 or 4)
+//! with angular factors a_t (see `angular_factors`).
+//!
+//! For a spheroid the terms of power ≤ 0 of every R_t's Laurent series in x integrate, together, to
+//! exactly zero in every element with n ≠ k, and those of power < 0 alone when |n − k| ≥ 3
+//! (research/tmatrix, findings 11–12): they are the electrostatic limit, whose outgoing and regular
+//! multipoles pair to zero off the diagonal on any surface. They carry the catastrophic
+//! cancellation of the plain integrals, up to 10¹⁶ at x = 15, ε = 4. Each element is therefore
+//! accumulated in three exact forms, plain, minus the powers < 0, minus the powers ≤ 0, each with
+//! the sum of the magnitudes of what it adds, and the form with the smallest sum is kept.
+//!
+//! At each node the kept part R⁺ of a product is evaluated either by its own series (the positive
+//! powers) or as R − R_{≤0} from the Bessel values, whichever adds less in magnitude.
+//!
+//! The series are normalized at x_min = k r_min (ξ = x/x_min ≥ 1): yₙ(x) = Σ_l a_l x^{2l−n−1}
+//! (a₀ = −(2n − 1)!!, a_l = −a_{l−1}/(2l(2l − 2n − 1))), scaled by σₙ: Ã_l = σₙ a_l x_min^{2l−n−1},
+//! bounded; j_k(z) = Σ_i b_i z^{k+2i}, scaled by c_k: B̃_i = c_k b_i (s x_min)^{k+2i}, bounded by
+//! (x_min/x_max)^k.
+
+use num_complex::Complex;
+
+use super::functions::Scales;
+use crate::mie::kernel::{int, real};
+use crate::scalar::Real;
+
+/// The y-kinds and j-kinds of the radial products, and e.
+const TYPES: [(Kind, Kind, i64); 8] = [
+    (Kind::Plain, Kind::Plain, 2),
+    (Kind::Plain, Kind::Deriv, 2),
+    (Kind::Deriv, Kind::Plain, 2),
+    (Kind::Deriv, Kind::Deriv, 2),
+    (Kind::Plain, Kind::Over, 4),
+    (Kind::Deriv, Kind::Over, 4),
+    (Kind::Over, Kind::Plain, 4),
+    (Kind::Over, Kind::Deriv, 4),
+];
+
+/// f, [z f]'/z or f/z.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+enum Kind {
+    Plain,
+    Deriv,
+    Over,
+}
+
+/// The series terms at one node, for every order: the y-part Y_l = Ã_l ξ^{a_l} (with the kind's
+/// coefficient) and the j-part Taylor terms t_i = B̃_i ξ^{b_i}, as (exponent of the first term,
+/// values), with the prefix and suffix sums of the j-part (and of their magnitudes).
+pub(crate) struct NodeSeries<T> {
+    /// y[n][kind] = (a₀ exponent, terms Y_l).
+    y: Vec<[(i64, Vec<T>); 3]>,
+    /// j[k][kind] = (b₀ exponent, terms, prefix, prefix |·|, suffix, suffix |·|).
+    j: Vec<[JTerms<T>; 3]>,
+}
+
+#[derive(Clone, Debug)]
+struct JTerms<T> {
+    b0: i64,
+    head: Vec<Complex<T>>,
+    head_abs: Vec<T>,
+    tail: Vec<Complex<T>>,
+    tail_abs: Vec<T>,
+}
+
+/// The regularized products of one pair at one node: R_full is supplied; the forms and their
+/// magnitude sums.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct Forms<T> {
+    /// R⁺ minus the powers ≤ 0, and minus the powers < 0; their magnitude sums (the plain form's
+    /// is |R_full|).
+    pub(crate) le0: Complex<T>,
+    pub(crate) le0_abs: T,
+    pub(crate) lt0: Complex<T>,
+    pub(crate) lt0_abs: T,
+}
+
+/// The coefficients Ã_l (per order n) and B̃_i (per order k), node-independent.
+pub(crate) struct Series<T> {
+    x_min: T,
+    s: Complex<T>,
+    a: Vec<Vec<T>>,
+    b: Vec<Vec<Complex<T>>>,
+}
+
+impl<T: Real> Series<T> {
+    /// The series up to the terms negligible (below ε/2⁸ of the largest) on x ≤ x_max.
+    pub(crate) fn new(x_min: T, x_max: T, s: Complex<T>, n: usize, scales: &Scales<T>) -> Self {
+        let mut a = vec![Vec::new(); n + 1];
+        let x2 = x_min * x_min;
+        let xi_max = x_max / x_min;
+        let xi2_max = xi_max * xi_max;
+        let tol = T::epsilon() * T::from_f64(1.0 / 256.0);
+        let half = T::from_f64(0.5);
+        for (nn, slot) in a.iter_mut().enumerate().skip(1) {
+            // Ã₀ = −Π_{i=1}^{n} (2i − 1)/(x_min |r_i(x_min)|) (|h₀(x_min)| = 1/x_min).
+            let mut a0 = -T::one();
+            for i in 1..=nn {
+                a0 *= int::<T>(2 * i - 1) / (x_min * scales.h_ratio[i]);
+            }
+            // Terms until they are past their peak and negligible at ξ_max (the term of l at
+            // ξ_max is Ã_l ξ_max^{2l−n−1}; track it relative to Ã₀ ξ_max^{−n−1}).
+            let mut v = vec![a0];
+            let (mut at_max, mut peak) = (a0.abs(), a0.abs());
+            for l in 1.. {
+                let prev = v[l - 1];
+                let den = int::<T>(2 * l) * (int::<T>(2 * l) - int::<T>(2 * nn + 1));
+                let next = -(prev * x2) / den;
+                v.push(next);
+                let ratio = (x2 * xi2_max / den).abs();
+                at_max *= ratio;
+                peak = peak.max(at_max);
+                if 2 * l > 2 * nn + 1 && ratio < half && at_max < tol * peak {
+                    break;
+                }
+            }
+            *slot = v;
+        }
+        let sx = s.scale(x_min);
+        let sx2 = sx * sx;
+        let mut b = vec![Vec::new(); n + 1];
+        for (kk, slot) in b.iter_mut().enumerate() {
+            // B̃₀ = Π_{i=1}^{k} s x_min/max(2i + 1, |z_max|) · |z_max| e^{−|Im z_max|} / 1 (j₀'s
+            // b₀ = 1), the c_k of `Scales`.
+            let mut b0 = real(scales.e_ratio[0].recip() * (-scales.im_z_max).exp());
+            for i in 1..=kk {
+                b0 = b0 * sx.scale((int::<T>(2 * i + 1) * scales.e_ratio[i]).recip());
+            }
+            let mut v = vec![b0];
+            let (mut at_max, mut peak) = (b0.norm(), b0.norm());
+            for i in 1.. {
+                let den = int::<T>(2 * i) * int::<T>(2 * i + 2 * kk + 1);
+                let prev = v[i - 1];
+                v.push(-(prev * sx2).scale(den.recip()));
+                let ratio = sx2.norm() * xi2_max / den;
+                at_max *= ratio;
+                peak = peak.max(at_max);
+                if ratio < half && at_max < tol * peak {
+                    break;
+                }
+            }
+            *slot = v;
+        }
+        Self { x_min, s, a, b }
+    }
+
+    /// The per-node terms at ξ = x/x_min.
+    pub(crate) fn at(&self, xi: T) -> NodeSeries<T> {
+        let n = self.a.len() - 1;
+        let inv_xmin = self.x_min.recip();
+        let inv_sx = (self.s.scale(self.x_min)).inv();
+        let mut y = Vec::with_capacity(n + 1);
+        for (nn, a) in self.a.iter().enumerate() {
+            if nn == 0 {
+                y.push(core::array::from_fn(|_| (0, Vec::new())));
+                continue;
+            }
+            let e0 = -(nn as i64) - 1;
+            let start = xi.powi(e0 as i32);
+            let xi2 = xi * xi;
+            let mut plain = Vec::with_capacity(a.len());
+            let mut deriv = Vec::with_capacity(a.len());
+            let mut over = Vec::with_capacity(a.len());
+            let mut p = start;
+            for (l, &al) in a.iter().enumerate() {
+                let v = al * p;
+                plain.push(v);
+                // [x y]'/x: exponent one less, coefficient (2l − n); y/x: exponent one less.
+                let v1 = v / xi * inv_xmin;
+                deriv.push(v1 * int::<T>(2 * l) - v1 * int::<T>(nn));
+                over.push(v1);
+                p *= xi2;
+            }
+            y.push([(e0, plain), (e0 - 1, deriv), (e0 - 1, over)]);
+        }
+        let mut j = Vec::with_capacity(n + 1);
+        for (kk, b) in self.b.iter().enumerate() {
+            let e0 = kk as i64;
+            let xi2 = xi * xi;
+            let mut p = xi.powi(e0 as i32);
+            let mut plain = Vec::with_capacity(b.len());
+            let mut deriv = Vec::with_capacity(b.len());
+            let mut over = Vec::with_capacity(b.len());
+            for (i, &bi) in b.iter().enumerate() {
+                let v = bi.scale(p);
+                plain.push(v);
+                let v1 = (v * inv_sx).scale(xi.recip());
+                deriv.push(v1.scale(int::<T>(kk + 1 + 2 * i)));
+                over.push(v1);
+                p *= xi2;
+            }
+            j.push([sums(e0, &plain), sums(e0 - 1, &deriv), sums(e0 - 1, &over)]);
+        }
+        NodeSeries { y, j }
+    }
+}
+
+fn sums<T: Real>(b0: i64, terms: &[Complex<T>]) -> JTerms<T> {
+    let len = terms.len();
+    let zero = real(T::zero());
+    let mut head = Vec::with_capacity(len);
+    let mut head_abs = Vec::with_capacity(len);
+    let (mut acc, mut acc_abs) = (zero, T::zero());
+    for &t in terms {
+        acc = acc + t;
+        acc_abs += t.norm();
+        head.push(acc);
+        head_abs.push(acc_abs);
+    }
+    let mut tail = vec![zero; len + 1];
+    let mut tail_abs = vec![T::zero(); len + 1];
+    for i in (0..len).rev() {
+        tail[i] = tail[i + 1] + terms[i];
+        tail_abs[i] = tail_abs[i + 1] + terms[i].norm();
+    }
+    JTerms {
+        b0,
+        head,
+        head_abs,
+        tail,
+        tail_abs,
+    }
+}
+
+impl<T: Real> NodeSeries<T> {
+    /// The forms of the product of type t for orders (n, k) at this node; `x_e` = (x², x⁴) at the
+    /// node.
+    pub(crate) fn forms(
+        &self,
+        t: usize,
+        n: usize,
+        k: usize,
+        full: Complex<T>,
+        x_e: [T; 2],
+    ) -> Forms<T> {
+        let (yk, jk, e) = TYPES[t];
+        let (a0, ys) = &self.y[n][kind_index(yk)];
+        let js = &self.j[k][kind_index(jk)];
+        let xe = if e == 2 { x_e[0] } else { x_e[1] };
+        // For each l: the largest i with a₀ + 2l + b₀ + 2i + e ≤ cut.
+        let calc = |cut: i64| -> (Complex<T>, T, Complex<T>, T) {
+            let (mut neg, mut neg_abs) = (real(T::zero()), T::zero());
+            let (mut pos, mut pos_abs) = (real(T::zero()), T::zero());
+            let len = js.head.len() as i64;
+            for (l, &yl) in ys.iter().enumerate() {
+                let base = a0 + 2 * l as i64 + js.b0 + e;
+                // p(i) = base + 2i ≤ cut ⇔ i ≤ (cut − base)/2.
+                let i_max = if cut >= base { (cut - base) / 2 } else { -1 };
+                let yv = yl * xe;
+                if i_max >= 0 {
+                    let i = i_max.min(len - 1) as usize;
+                    neg = neg + js.head[i].scale(yv);
+                    neg_abs += js.head_abs[i] * yv.abs();
+                }
+                let from = (i_max + 1).clamp(0, len) as usize;
+                pos = pos + js.tail[from].scale(yv);
+                pos_abs += js.tail_abs[from] * yv.abs();
+            }
+            (neg, neg_abs, pos, pos_abs)
+        };
+        let pick = |cut: i64| -> (Complex<T>, T) {
+            let (neg, neg_abs, pos, pos_abs) = calc(cut);
+            // R⁺ by its series, or as R − R_{≤cut}: whichever adds less.
+            let sub_abs = full.norm() + neg_abs;
+            if pos_abs <= sub_abs {
+                (pos, pos_abs)
+            } else {
+                (full - neg, sub_abs)
+            }
+        };
+        let (le0, le0_abs) = pick(0);
+        let (lt0, lt0_abs) = pick(-1);
+        Forms {
+            le0,
+            le0_abs,
+            lt0,
+            lt0_abs,
+        }
+    }
+}
+
+fn kind_index(k: Kind) -> usize {
+    match k {
+        Kind::Plain => 0,
+        Kind::Deriv => 1,
+        Kind::Over => 2,
+    }
+}
+
+/// The eight angular factors of element (row type rt, order n; column type ct, order k) at a
+/// node, for the products of `TYPES` (with s on the first column field, and ν' = −β sin θ cos θ on
+/// the ν terms, whose x² is in e): rows (A₁, A₂) = (Mₙ, Nₙ) | (Nₙ, Mₙ), columns (B₁, B₂) =
+/// (s N_k, M_k) | (s M_k, N_k), and n̂·(A × B) with the y-part components
+/// Mₙ = (0, i yₙ pₙ, −yₙ τₙ), Nₙ = (n(n+1) (yₙ/x) dₙ, Dyₙ τₙ, i Dyₙ pₙ), pₙ = −πₙ.
+#[allow(clippy::too_many_arguments)]
+pub(crate) fn angular_factors<T: Real>(
+    rt: usize,
+    ct: usize,
+    s: Complex<T>,
+    nu: T,
+    n: usize,
+    k: usize,
+    (dn, pin, tn): (T, T, T),
+    (dk, pik, tk): (T, T, T),
+) -> [Complex<T>; 8] {
+    let i1 = Complex::new(T::zero(), T::one());
+    let pn = -pin;
+    let nn1 = int::<T>(n * (n + 1));
+    let kk1 = int::<T>(k * (k + 1));
+    let zero = real(T::zero());
+    // n̂·(A × B) of each (A, B) as coefficients of the TYPES.
+    let mm = [
+        i1.scale(tn * pik - pn * tk),
+        zero,
+        zero,
+        zero,
+        zero,
+        zero,
+        zero,
+        zero,
+    ];
+    let mn = [
+        zero,
+        real(tn * tk - pn * pik),
+        zero,
+        zero,
+        real(-(nu * tn * kk1 * dk)),
+        zero,
+        zero,
+        zero,
+    ];
+    let nm = [
+        zero,
+        zero,
+        real(pn * pik - tn * tk),
+        zero,
+        zero,
+        zero,
+        real(nu * nn1 * dn * tk),
+        zero,
+    ];
+    let nnv = [
+        zero,
+        zero,
+        zero,
+        i1.scale(tn * pik - pn * tk),
+        zero,
+        i1.scale(nu * pn * kk1 * dk),
+        zero,
+        i1.scale(-(nu * nn1 * dn * pik)),
+    ];
+    let combine = |a: [Complex<T>; 8], b: [Complex<T>; 8]| -> [Complex<T>; 8] {
+        core::array::from_fn(|t| s * a[t] + b[t])
+    };
+    match (rt, ct) {
+        (0, 0) => combine(mn, nm),
+        (0, _) => combine(mm, nnv),
+        (_, 0) => combine(nnv, mm),
+        _ => combine(nm, mn),
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::scalar::DoubleDouble as Dd;
+    use crate::tmatrix::functions::Radial;
+
+    /// The whole series reproduce the scaled Bessel factors on [x_min, x_max] to the rounding of
+    /// their largest term, in double-double (so none is truncated).
+    #[test]
+    fn series_reproduce_the_functions() {
+        let x_min = Dd::from(9.0);
+        let x_max = Dd::from(36.0);
+        let s = Complex::new(Dd::from(1.5), Dd::from(0.01));
+        let n = 40;
+        let scales = Scales::new(x_min, s.scale(x_max), n);
+        let series = Series::new(x_min, x_max, s, n, &scales);
+        let mut rad = Radial::new();
+        let tol = <Dd as num_traits::Float>::epsilon() * Dd::from(64.0);
+        for xi in [1.0, 1.7, 2.5, 4.0] {
+            rad.compute(x_min * Dd::from(xi), s, n, &scales);
+            let ns = series.at(Dd::from(xi));
+            for nn in [1usize, 10, 40] {
+                for (kind, v) in [(0, rad.h[nn].im), (1, rad.dh[nn].im), (2, rad.h_x[nn].im)] {
+                    let terms = &ns.y[nn][kind].1;
+                    let sum = terms.iter().fold(Dd::from(0.0), |a, &b| a + b);
+                    let mag = terms.iter().fold(Dd::from(0.0), |a, &b| a + b.abs());
+                    assert!((sum - v).abs() <= tol * mag, "y n {nn} kind {kind} ξ {xi}");
+                }
+            }
+            for kk in [1usize, 2, 40] {
+                for (kind, v) in [(0, rad.jz[kk]), (1, rad.djz[kk]), (2, rad.jz_z[kk])] {
+                    let js = &ns.j[kk][kind];
+                    assert!(
+                        (js.tail[0] - v).norm() <= tol * js.tail_abs[0],
+                        "j k {kk} kind {kind} ξ {xi}"
+                    );
+                }
+            }
+        }
+    }
+}
````

### Task 3: Mixed-precision refinement and the error estimate

- [ ] Move the LU to `tmatrix/solve.rs`; add `solve_f64`, the probe `error_estimate` (×4) and `refine` (factorizes the mixed matrix rounded to f64).
- [ ] Add the per-element magnitude bounds `q_abs`, `p_abs` to `ModeMatrices`, and `tmatrix_refined` with a per-mode decision.
- [ ] Test: refined against full double-double (< 10⁻¹³), estimates bound the errors on both paths.

The commit (`3eeac33`):

````diff
diff --git a/scatrs/src/tmatrix/kernel.rs b/scatrs/src/tmatrix/kernel.rs
index 3a214f2..2b45e16 100644
--- a/scatrs/src/tmatrix/kernel.rs
+++ b/scatrs/src/tmatrix/kernel.rs
@@ -22,11 +22,12 @@ use num_complex::Complex;
 
 use super::functions::{Angular, Radial, Scales};
 use super::regular::{Series, angular_factors};
-use crate::complex::inv;
+use super::solve::{error_estimate, refine, solve_f64, solve_t};
+use crate::Result;
 use crate::mie::kernel::{int, real};
+use crate::scalar::DoubleDouble as Dd;
 use crate::scalar::Real;
 use crate::special::gauss_legendre;
-use crate::{Error, Result};
 
 /// A spheroid: semi-axes a (equatorial) and c (polar) in units of 1/k, from the equal-volume
 /// size parameter x and the aspect ratio ε = a/c.
@@ -66,6 +67,11 @@ impl<T: Real> Spheroid<T> {
 pub(crate) struct ModeMatrices<T> {
     pub(crate) q: [Vec<Complex<T>>; 2],
     pub(crate) p: [Vec<Complex<T>>; 2],
+    /// A bound on the rounding of each element of Q: the magnitudes it was summed from (Y's in the
+    /// kept form, and |P|), to be multiplied by the unit roundoff.
+    pub(crate) q_abs: [Vec<T>; 2],
+    /// The same for P.
+    pub(crate) p_abs: [Vec<T>; 2],
     /// (type, order) of each index of a class.
     pub(crate) index: [Vec<(usize, usize)>; 2],
 }
@@ -82,106 +88,6 @@ fn triple<T: Real>(nu: T, a: &Vec3<T>, b: &Vec3<T>) -> Complex<T> {
     a[1] * b[2] - a[2] * b[1] + (a[2] * b[0] - a[0] * b[2]).scale(nu)
 }
 
-/// Q and P of mode m with orders max(m, 1), …, n on n_theta nodes in 0 < θ < π/2.
-pub(crate) fn mode_matrices<T: Real>(
-    shape: &Spheroid<T>,
-    s: Complex<T>,
-    m: usize,
-    n: usize,
-    n_theta: usize,
-) -> Result<ModeMatrices<T>> {
-    let n0 = m.max(1);
-    if n < n0 {
-        return Err(Error::InvalidInput(format!(
-            "mode {m} needs at least {n0} orders, got {n}"
-        )));
-    }
-    // Index sets of the two classes: (type, order) with (order + type) % 2 == class.
-    let mut index: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
-    for t in 0..2 {
-        for k in n0..=n {
-            index[(k + t) % 2].push((t, k));
-        }
-    }
-    let size = [index[0].len(), index[1].len()];
-    let mut q: [Vec<Complex<T>>; 2] =
-        core::array::from_fn(|c| vec![real(T::zero()); size[c] * size[c]]);
-    let mut p: [Vec<Complex<T>>; 2] =
-        core::array::from_fn(|c| vec![real(T::zero()); size[c] * size[c]]);
-    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
-    let rule = gauss_legendre::<T>(2 * n_theta)?;
-    let mut radial = Radial::new();
-    let mut ang = Angular::new();
-    let gam: Vec<T> = (0..=n)
-        .map(|k| if k == 0 { T::zero() } else { gamma(k) })
-        .collect();
-    let i1 = Complex::new(T::zero(), T::one());
-    let zero3 = [real(T::zero()); 3];
-    // Field components per order: outgoing (h) and regular (j) M, N of the medium, and the
-    // particle's M, N; all without γ (applied per element).
-    let mut out_m = vec![zero3; n + 1];
-    let mut out_n = vec![zero3; n + 1];
-    let mut reg_m = vec![zero3; n + 1];
-    let mut reg_n = vec![zero3; n + 1];
-    let mut in_m = vec![zero3; n + 1];
-    let mut in_n = vec![zero3; n + 1];
-    for (&t, &w) in rule.nodes.iter().zip(&rule.weights) {
-        if t <= T::zero() {
-            continue;
-        }
-        let (ct, st) = (t, (T::one() - t * t).sqrt());
-        let (r, nu) = shape.at(ct, st);
-        radial.compute(r, s, n, &scales);
-        ang.compute(m, n, ct, st);
-        for k in n0..=n {
-            let (d, pi, tau) = (ang.d[k], ang.pi[k], ang.tau[k]);
-            let kk = int::<T>(k * (k + 1));
-            let po = real(-pi); // the outgoing waves' −π
-            let field = |h: Complex<T>, dh: Complex<T>, h_x: Complex<T>, p: Complex<T>| {
-                (
-                    [real(T::zero()), i1 * h * p, -(h.scale(tau))],
-                    [h_x.scale(kk * d), dh.scale(tau), i1 * dh * p],
-                )
-            };
-            (out_m[k], out_n[k]) = field(radial.h[k], radial.dh[k], radial.h_x[k], po);
-            (reg_m[k], reg_n[k]) = field(radial.j[k], radial.dj[k], radial.j_x[k], po);
-            (in_m[k], in_n[k]) = field(radial.jz[k], radial.djz[k], radial.jz_z[k], real(pi));
-        }
-        // 2 for the two halves of the range, 2π for φ, x² from dS (k = 1).
-        let fac = w * r * r * T::PI() * T::from_f64(4.0);
-        for c in 0..2 {
-            let idx = &index[c];
-            let kc = idx.len();
-            for (i, &(rt, nn)) in idx.iter().enumerate() {
-                let (o1, o2) = if rt == 0 {
-                    (&out_m[nn], &out_n[nn])
-                } else {
-                    (&out_n[nn], &out_m[nn])
-                };
-                let (g1, g2) = if rt == 0 {
-                    (&reg_m[nn], &reg_n[nn])
-                } else {
-                    (&reg_n[nn], &reg_m[nn])
-                };
-                for (j, &(ctp, kk_)) in idx.iter().enumerate() {
-                    let (b1, b2) = if ctp == 0 {
-                        (in_n[kk_].map(|v| v * s), &in_m[kk_])
-                    } else {
-                        (in_m[kk_].map(|v| v * s), &in_n[kk_])
-                    };
-                    let g = fac * gam[nn] * gam[kk_];
-                    let qv = triple(nu, o1, &b1) + triple(nu, o2, b2);
-                    let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
-                    // −i·(…)
-                    q[c][i * kc + j] = q[c][i * kc + j] + Complex::new(qv.im, -qv.re).scale(g);
-                    p[c][i * kc + j] = p[c][i * kc + j] + Complex::new(pv.im, -pv.re).scale(g);
-                }
-            }
-        }
-    }
-    Ok(ModeMatrices { q, p, index })
-}
-
 /// Q and P of every mode m = 0, …, n, on n_theta nodes in 0 < θ < π/2, nodes outside and modes
 /// inside so that each node's radial functions and regularized products serve every mode. With
 /// `regularize`, each element of Q keeps the cleanest of its three exact forms (`super::regular`).
@@ -191,7 +97,10 @@ pub(crate) fn all_modes<T: Real>(
     n: usize,
     n_theta: usize,
     regularize: bool,
+    rows: &[Option<usize>],
 ) -> Result<Vec<ModeMatrices<T>>> {
+    // The lowest row order integrated in any mode (rows below are left zero).
+    let low = rows.iter().flatten().copied().min().unwrap_or(n + 1);
     let zero = real(T::zero());
     let mut modes: Vec<ModeMatrices<T>> = Vec::with_capacity(n + 1);
     // Per mode and class: Y's three forms and their magnitude sums.
@@ -214,6 +123,8 @@ pub(crate) fn all_modes<T: Real>(
         modes.push(ModeMatrices {
             q: core::array::from_fn(mk),
             p: core::array::from_fn(mk),
+            q_abs: core::array::from_fn(|c| vec![T::zero(); size[c] * size[c]]),
+            p_abs: core::array::from_fn(|c| vec![T::zero(); size[c] * size[c]]),
             index,
         });
     }
@@ -281,7 +192,7 @@ pub(crate) fn all_modes<T: Real>(
             for ti in 0..8 {
                 for nn in 0..=n {
                     for kk in 0..=n {
-                        if nn == 0 || kk == 0 || nn == kk {
+                        if nn == 0 || kk == 0 || nn == kk || nn < low {
                             forms.push(None);
                             continue;
                         }
@@ -296,6 +207,7 @@ pub(crate) fn all_modes<T: Real>(
         let fac = g0 * r2;
         let nu_prime = -(beta * st * ct);
         for (m, mm) in modes.iter_mut().enumerate() {
+            let Some(from) = rows[m] else { continue };
             let n0 = m.max(1);
             ang.compute(m, n, ct, st);
             for k in n0..=n {
@@ -314,6 +226,9 @@ pub(crate) fn all_modes<T: Real>(
                 let idx = &mm.index[c];
                 let kc = idx.len();
                 for (i, &(rt, nn)) in idx.iter().enumerate() {
+                    if nn < from {
+                        continue;
+                    }
                     let (g1, g2) = if rt == 0 {
                         (&reg_m[nn], &reg_n[nn])
                     } else {
@@ -330,6 +245,7 @@ pub(crate) fn all_modes<T: Real>(
                         let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
                         let slot = i * kc + j;
                         mm.p[c][slot] = mm.p[c][slot] + Complex::new(pv.im, -pv.re).scale(fac * gg);
+                        mm.p_abs[c][slot] += pv.norm() * (fac * gg).abs();
                         // Y: Σ_t a_t R_t in its three forms.
                         let ak = (ang.d[kk_], ang.pi[kk_], ang.tau[kk_]);
                         let a = angular_factors(rt, ctp, s, nu_prime, nn, kk_, an, ak);
@@ -385,6 +301,7 @@ pub(crate) fn all_modes<T: Real>(
                         }
                     }
                     mm.q[c][slot] = mm.p[c][slot] + yv[m][c][best][slot];
+                    mm.q_abs[c][slot] = ya[m][c][best][slot] + mm.p_abs[c][slot];
                 }
             }
         }
@@ -392,88 +309,6 @@ pub(crate) fn all_modes<T: Real>(
     Ok(modes)
 }
 
-/// LU factorization with partial pivoting of a row-major n × n matrix, in place; returns the
-/// pivots.
-pub(crate) fn lu<T: Real>(a: &mut [Complex<T>], n: usize) -> Result<Vec<usize>> {
-    let mut piv: Vec<usize> = (0..n).collect();
-    for k in 0..n {
-        let mut best = k;
-        let mut big = T::zero();
-        for i in k..n {
-            let v = a[i * n + k].norm();
-            if v > big {
-                big = v;
-                best = i;
-            }
-        }
-        if big == T::zero() || !big.is_finite() {
-            return Err(Error::InvalidInput(format!(
-                "the EBCM matrix is singular or not finite at column {k}"
-            )));
-        }
-        if best != k {
-            for j in 0..n {
-                a.swap(k * n + j, best * n + j);
-            }
-            piv.swap(k, best);
-        }
-        let inv_p = inv(a[k * n + k]);
-        for i in k + 1..n {
-            let f = a[i * n + k] * inv_p;
-            a[i * n + k] = f;
-            if f.re == T::zero() && f.im == T::zero() {
-                continue;
-            }
-            for j in k + 1..n {
-                let v = a[k * n + j];
-                a[i * n + j] = a[i * n + j] - f * v;
-            }
-        }
-    }
-    Ok(piv)
-}
-
-/// Solves A x = b in place with the factors of `lu`.
-pub(crate) fn lu_solve<T: Real>(a: &[Complex<T>], piv: &[usize], n: usize, b: &mut [Complex<T>]) {
-    let mut x: Vec<Complex<T>> = piv.iter().map(|&i| b[i]).collect();
-    for i in 0..n {
-        let mut v = x[i];
-        for j in 0..i {
-            v = v - a[i * n + j] * x[j];
-        }
-        x[i] = v;
-    }
-    for i in (0..n).rev() {
-        let mut v = x[i];
-        for j in i + 1..n {
-            v = v - a[i * n + j] * x[j];
-        }
-        x[i] = v * inv(a[i * n + i]);
-    }
-    b.copy_from_slice(&x);
-}
-
-/// T = −P Q⁻¹ of one class (scaled as P and Q are): solved as Qᵀ Tᵀ = −Pᵀ.
-pub(crate) fn solve_t<T: Real>(
-    q: &[Complex<T>],
-    p: &[Complex<T>],
-    n: usize,
-) -> Result<Vec<Complex<T>>> {
-    let mut qt: Vec<Complex<T>> = (0..n * n).map(|ij| q[(ij % n) * n + ij / n]).collect();
-    let piv = lu(&mut qt, n)?;
-    let mut t = vec![real(T::zero()); n * n];
-    let mut col = vec![real(T::zero()); n];
-    for i in 0..n {
-        // Row i of T is column i of Tᵀ: Qᵀ (row i of T)ᵀ = −(row i of P)ᵀ.
-        for j in 0..n {
-            col[j] = -p[i * n + j];
-        }
-        lu_solve(&qt, &piv, n, &mut col);
-        t[i * n..(i + 1) * n].copy_from_slice(&col);
-    }
-    Ok(t)
-}
-
 /// The unscaling factor of T's element (row order n, column order k): σ_k/σₙ =
 /// |hₙ(x_min)|/|h_k(x_min)|, from the cumulative logarithms of the scale ratios.
 pub(crate) fn log_h(scales: &Scales<impl Real>) -> Vec<f64> {
@@ -509,7 +344,7 @@ pub(crate) fn tmatrix<T: Real>(
     let shape = Spheroid::new(x, eps);
     let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
     let lh = log_h(&scales);
-    let all = all_modes(&shape, s, n, n_theta, regularize)?;
+    let all = all_modes(&shape, s, n, n_theta, regularize, &vec![Some(0); n + 1])?;
     let mut modes = Vec::with_capacity(n + 1);
     for mm in all {
         let mut ts: [Vec<Complex<T>>; 2] = [Vec::new(), Vec::new()];
@@ -519,13 +354,7 @@ pub(crate) fn tmatrix<T: Real>(
                 continue;
             }
             let mut t = solve_t(q, p, kc)?;
-            // Undo the scaling: T = Σ⁻¹ T' Σ, T_{nk} = T'_{nk} σ_k/σₙ = T' |hₙ|/|h_k| (x_min).
-            for (i, &(_, nn)) in index.iter().enumerate() {
-                for (j, &(_, kk_)) in index.iter().enumerate() {
-                    let f = T::from_f64((lh[nn] - lh[kk_]).exp());
-                    t[i * kc + j] = t[i * kc + j].scale(f);
-                }
-            }
+            unscale(&mut t, index, &lh);
             *slot = t;
         }
         modes.push((ts, mm.index));
@@ -533,6 +362,142 @@ pub(crate) fn tmatrix<T: Real>(
     Ok(TMatrixModes { n, modes })
 }
 
+/// Undoes the scaling of one class's T: T = Σ⁻¹ T' Σ, T_{nk} = T'_{nk} σ_k/σₙ =
+/// T' |hₙ|/|h_k| (x_min).
+fn unscale<T: Real>(t: &mut [Complex<T>], index: &[(usize, usize)], lh: &[f64]) {
+    let kc = index.len();
+    for (i, &(_, nn)) in index.iter().enumerate() {
+        for (j, &(_, kk_)) in index.iter().enumerate() {
+            let f = T::from_f64((lh[nn] - lh[kk_]).exp());
+            t[i * kc + j] = t[i * kc + j].scale(f);
+        }
+    }
+}
+
+/// How each mode's classes were solved: the estimated largest error of T (unscaled, absolute)
+/// and the refinement sweeps (0: plain f64).
+#[derive(Clone, Debug)]
+pub(crate) struct SolveReport {
+    pub(crate) error: Vec<[f64; 2]>,
+    pub(crate) sweeps: Vec<[usize; 2]>,
+}
+
+/// The lowest row order integrated in double-double for refinement in mode m: the upper three
+/// quarters of the orders (findings 5 and 7 of the study: the f64 floor lives there).
+fn refine_from(m: usize, n: usize) -> usize {
+    let n0 = m.max(1);
+    n0 + (n + 1 - n0) / 4
+}
+
+/// T of every mode in f64, refined (mixed precision) in the modes whose error estimate exceeds
+/// `tol` times the largest |T|: their rows of order ≥ [`refine_from`] are integrated in
+/// double-double, and T is corrected with double-double residuals.
+pub(crate) fn tmatrix_refined(
+    x: f64,
+    eps: f64,
+    s: Complex<f64>,
+    n: usize,
+    n_theta: usize,
+    tol: f64,
+) -> Result<(TMatrixModes<f64>, SolveReport)> {
+    // The elements' rounding: u Σ|terms| times 2N for the accuracy of the function values (the
+    // largest true error measured is about N u Σ|terms|).
+    let u = f64::EPSILON / 2.0 * (2 * n) as f64;
+    let u_dd = u * f64::EPSILON / 2.0;
+    let shape = Spheroid::new(x, eps);
+    let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
+    let lh = log_h(&scales);
+    let all = all_modes(&shape, s, n, n_theta, true, &vec![Some(0); n + 1])?;
+    let mut report = SolveReport {
+        error: vec![[0.0; 2]; n + 1],
+        sweeps: vec![[0; 2]; n + 1],
+    };
+    let mut ts: Vec<[Vec<Complex<f64>>; 2]> = Vec::with_capacity(n + 1);
+    let mut big = 0.0f64;
+    for (m, mm) in all.iter().enumerate() {
+        let mut pair: [Vec<Complex<f64>>; 2] = [Vec::new(), Vec::new()];
+        for (c, slot) in pair.iter_mut().enumerate() {
+            let index = &mm.index[c];
+            let kc = index.len();
+            if kc == 0 {
+                continue;
+            }
+            let f = solve_f64(&mm.q[c], &mm.p[c], kc)?;
+            let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
+            let eq: Vec<f64> = mm.q_abs[c].iter().map(|&v| u * v).collect();
+            let ep: Vec<f64> = mm.p_abs[c].iter().map(|&v| u * v).collect();
+            report.error[m][c] = error_estimate(&f, &f.t, &eq, &ep, &lh_c, kc);
+            let mut t = f.t;
+            unscale(&mut t, index, &lh);
+            big = t.iter().fold(big, |a, v| a.max(v.norm()));
+            *slot = t;
+        }
+        ts.push(pair);
+    }
+    let rows: Vec<Option<usize>> = (0..=n)
+        .map(|m| {
+            let worst = report.error[m][0].max(report.error[m][1]);
+            (worst > tol * big).then(|| refine_from(m, n))
+        })
+        .collect();
+    if rows.iter().any(Option::is_some) {
+        let shape_dd = Spheroid::new(Dd::from(x), Dd::from(eps));
+        let s_dd = Complex::new(Dd::from(s.re), Dd::from(s.im));
+        let exact = all_modes(&shape_dd, s_dd, n, n_theta, true, &rows)?;
+        for (m, from) in rows.iter().enumerate() {
+            let Some(from) = *from else { continue };
+            let mm = &all[m];
+            for (c, slot) in ts[m].iter_mut().enumerate() {
+                let index = &mm.index[c];
+                let kc = index.len();
+                if kc == 0 {
+                    continue;
+                }
+                let flags: Vec<bool> = index.iter().map(|&(_, nn)| nn >= from).collect();
+                let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
+                let (mut t, solved, sweeps, last) = refine(
+                    &mm.q[c],
+                    &mm.p[c],
+                    &exact[m].q[c],
+                    &exact[m].p[c],
+                    &flags,
+                    &lh_c,
+                    kc,
+                    4,
+                )?;
+                // The f64 rows' rounding remains, beside the double-double rows' and the last
+                // correction.
+                let bound = |v: &[f64]| -> Vec<f64> {
+                    v.iter()
+                        .enumerate()
+                        .map(|(ij, &a)| a * if flags[ij / kc] { u_dd } else { u })
+                        .collect()
+                };
+                let est = error_estimate(
+                    &solved,
+                    &t,
+                    &bound(&mm.q_abs[c]),
+                    &bound(&mm.p_abs[c]),
+                    &lh_c,
+                    kc,
+                );
+                unscale(&mut t, index, &lh);
+                // T is returned in f64: its own rounding.
+                let rounding = t.iter().fold(0.0f64, |a, v| a.max(v.norm())) * f64::EPSILON;
+                report.sweeps[m][c] = sweeps;
+                report.error[m][c] = est + last + rounding;
+                *slot = t;
+            }
+        }
+    }
+    let modes = ts
+        .into_iter()
+        .zip(all)
+        .map(|(t, mm)| (t, mm.index))
+        .collect();
+    Ok((TMatrixModes { n, modes }, report))
+}
+
 impl<T: Real> TMatrixModes<T> {
     /// The orientation-averaged ⟨C_ext⟩ and ⟨C_sca⟩ in units of 1/k² (Mishchenko 1991):
     /// −2π Σ (2 − δ_{m0}) Re(T¹¹ₙₙ + T²²ₙₙ) and 2π Σ (2 − δ_{m0}) |T|².
@@ -558,6 +523,48 @@ impl<T: Real> TMatrixModes<T> {
 #[cfg(test)]
 mod tests {
     use super::*;
+
+    /// The refined path matches full double-double, its error estimate bounds its error, and the
+    /// f64 path's estimate bounds the f64 error.
+    #[test]
+    fn refinement_reaches_double_double() {
+        use crate::scalar::DoubleDouble as Dd;
+        let (x, eps, n, nth) = (7.0, 3.0, 20, 40);
+        let s = Complex::new(1.5, 0.01);
+        let (f, f_rep) = tmatrix_refined(x, eps, s, n, nth, 1.0).unwrap();
+        let (r, r_rep) = tmatrix_refined(x, eps, s, n, nth, 0.0).unwrap();
+        let sd = Complex::new(Dd::from(s.re), Dd::from(s.im));
+        let d = tmatrix(Dd::from(x), Dd::from(eps), sd, n, nth, true).unwrap();
+        for (u, rep) in [(&f, &f_rep), (&r, &r_rep)] {
+            for (m, (mu, md)) in u.modes.iter().zip(&d.modes).enumerate() {
+                for c in 0..2 {
+                    let err = mu.0[c]
+                        .iter()
+                        .zip(&md.0[c])
+                        .map(|(a, b)| (a - Complex::new(b.re.hi(), b.im.hi())).norm())
+                        .fold(0.0, f64::max);
+                    assert!(
+                        err <= rep.error[m][c].max(1e-300),
+                        "m {m} c {c}: {err:e} est {:e} sweeps {}",
+                        rep.error[m][c],
+                        rep.sweeps[m][c]
+                    );
+                }
+            }
+        }
+        assert!(f_rep.sweeps.iter().flatten().all(|&s| s == 0));
+        assert!(r_rep.sweeps.iter().flatten().all(|&s| s > 0));
+        let worst = r_rep.error.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
+        assert!(worst < 1e-12, "{worst:e}");
+        let err = r
+            .modes
+            .iter()
+            .zip(&d.modes)
+            .flat_map(|(a, b)| a.0.iter().flatten().zip(b.0.iter().flatten()))
+            .map(|(a, b)| (a - Complex::new(b.re.hi(), b.im.hi())).norm())
+            .fold(0.0, f64::max);
+        assert!(err < 1e-13, "{err:e}");
+    }
     use crate::mie::{Mie, MieOptions};
     use crate::scalar::DoubleDouble as Dd;
 
@@ -633,6 +640,205 @@ mod diagnostics {
         Complex::new(Dd::from(c.re), Dd::from(c.im))
     }
 
+    fn tref_big(
+        f: &[ModeMatrices<f64>],
+        d: &[ModeMatrices<Dd>],
+        m: usize,
+        c: usize,
+        lh: &[f64],
+    ) -> f64 {
+        let kc = f[m].index[c].len();
+        let mut t = solve_t(&d[m].q[c], &d[m].p[c], kc).unwrap();
+        unscale(&mut t, &f[m].index[c], lh);
+        t.iter().map(|v| v.norm().hi()).fold(0.0, f64::max)
+    }
+
+    /// Mode 1, prolate: f64 solve of the double-double elements rounded, against the f64
+    /// elements.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn rounded_elements() {
+        let s = Complex::new(1.5, 0.01);
+        let (x, eps, n, nth) = (15.0, 0.333_333_333_333, 42, 90);
+        let rows = vec![Some(0); n + 1];
+        let f = all_modes(&Spheroid::new(x, eps), s, n, nth, true, &rows).unwrap();
+        let d = all_modes(
+            &Spheroid::new(Dd::from(x), Dd::from(eps)),
+            dd(s),
+            n,
+            nth,
+            true,
+            &rows,
+        )
+        .unwrap();
+        let lh = log_h(&Scales::new(
+            Spheroid::new(x, eps).r_min(),
+            s.scale(Spheroid::new(x, eps).r_max()),
+            n,
+        ));
+        let hi = |v: &[Complex<Dd>]| {
+            v.iter()
+                .map(|c| Complex::new(c.re.hi(), c.im.hi()))
+                .collect::<Vec<_>>()
+        };
+        for m in [0usize, 1, 2, 8] {
+            for c in 0..2 {
+                let kc = f[m].index[c].len();
+                let tref = {
+                    let mut t = solve_t(&d[m].q[c], &d[m].p[c], kc).unwrap();
+                    unscale(&mut t, &f[m].index[c], &lh);
+                    hi(&t)
+                };
+                let run = |q: &[Complex<f64>], p: &[Complex<f64>]| {
+                    let sol = super::super::solve::solve_f64(q, p, kc).unwrap();
+                    let mut t = sol.t.clone();
+                    unscale(&mut t, &f[m].index[c], &lh);
+                    let big = tref.iter().map(|v| v.norm()).fold(0.0, f64::max);
+                    let e = t
+                        .iter()
+                        .zip(&tref)
+                        .map(|(a, b)| (a - b).norm())
+                        .fold(0.0, f64::max);
+                    e / big
+                };
+                let e_r = run(&hi(&d[m].q[c]), &hi(&d[m].p[c]));
+                let e_f = run(&f[m].q[c], &f[m].p[c]);
+                // The estimator fed with the true element errors, and with the bounds.
+                let lh_c: Vec<f64> = f[m].index[c].iter().map(|&(_, nn)| lh[nn]).collect();
+                let sol = super::super::solve::solve_f64(&f[m].q[c], &f[m].p[c], kc).unwrap();
+                let true_q: Vec<f64> = f[m].q[c]
+                    .iter()
+                    .zip(&hi(&d[m].q[c]))
+                    .map(|(a, b)| (a - b).norm())
+                    .collect();
+                let true_p: Vec<f64> = f[m].p[c]
+                    .iter()
+                    .zip(&hi(&d[m].p[c]))
+                    .map(|(a, b)| (a - b).norm())
+                    .collect();
+                let u = f64::EPSILON / 2.0;
+                let bq: Vec<f64> = f[m].q_abs[c].iter().map(|v| u * v).collect();
+                let bp: Vec<f64> = f[m].p_abs[c].iter().map(|v| u * v).collect();
+                let big = tref_big(&f, &d, m, c, &lh);
+                let est_true =
+                    super::super::solve::error_estimate(&sol, &sol.t, &true_q, &true_p, &lh_c, kc)
+                        / big;
+                let est_bound =
+                    super::super::solve::error_estimate(&sol, &sol.t, &bq, &bp, &lh_c, kc) / big;
+                let ratio = true_q
+                    .iter()
+                    .zip(&bq)
+                    .map(|(a, b)| a / b.max(1e-300))
+                    .fold(0.0, f64::max);
+                println!(
+                    "   est(true errors) {est_true:.1e}, est(bounds) {est_bound:.1e}, max true/bound {ratio:.1e}"
+                );
+                // Element errors of f64 relative to their row's largest element.
+                let mut row_err = 0.0f64;
+                let mut where_ = (0, 0);
+                for i in 0..kc {
+                    let rmax = (0..kc)
+                        .map(|j| d[m].q[c][i * kc + j].norm().hi())
+                        .fold(0.0, f64::max);
+                    for j in 0..kc {
+                        let v = d[m].q[c][i * kc + j];
+                        let e = (f[m].q[c][i * kc + j] - Complex::new(v.re.hi(), v.im.hi())).norm()
+                            / rmax;
+                        if e > row_err {
+                            row_err = e;
+                            where_ = (f[m].index[c][i].1, f[m].index[c][j].1);
+                        }
+                    }
+                }
+                println!(
+                    "m {m} c {c}: rounded dd {e_r:.1e}; f64 {e_f:.1e}; worst row-rel element err {row_err:.1e} at {where_:?}"
+                );
+            }
+        }
+    }
+
+    /// Per mode: Skeel estimate and errors of f64 and refined T against double-double.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn refined_per_mode() {
+        let s = Complex::new(1.5, 0.01);
+        let (x, eps, n, nth) = (15.0, 0.333_333_333_333, 42, 90);
+        let plain = tmatrix(x, eps, s, n, nth, true).unwrap();
+        let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 0.0).unwrap();
+        let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
+        for m in 0..=n {
+            for c in 0..2 {
+                let (mut big, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
+                for ((u, r), v) in plain.modes[m].0[c]
+                    .iter()
+                    .zip(&refined.modes[m].0[c])
+                    .zip(&b.modes[m].0[c])
+                {
+                    let v = Complex::new(v.re.hi(), v.im.hi());
+                    big = big.max(v.norm());
+                    ea = ea.max((u - v).norm());
+                    eb = eb.max((r - v).norm());
+                }
+                println!(
+                    "m {m} c {c}: max|T| {big:.1e} est {:.1e} f64 {:.1e} refined {:.1e} sweeps {}",
+                    report.error[m][c], ea, eb, report.sweeps[m][c]
+                );
+            }
+        }
+    }
+
+    /// The refined path against full double-double: run with `--ignored --nocapture`.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn refined_against_double_double() {
+        let s = Complex::new(1.5, 0.01);
+        for &(x, eps, n, nth) in &[
+            (10.0, 2.0, 24, 50),
+            (15.0, 4.0, 40, 100),
+            (15.0, 0.333_333_333_333, 42, 90),
+            (25.0, 0.333_333_333_333, 60, 120),
+            (30.0, 2.0, 50, 90),
+        ] {
+            let t0 = std::time::Instant::now();
+            let plain = tmatrix(x, eps, s, n, nth, true).unwrap();
+            let t_plain = t0.elapsed();
+            let t0 = std::time::Instant::now();
+            let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 1e-14).unwrap();
+            let t_ref = t0.elapsed();
+            let t0 = std::time::Instant::now();
+            let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
+            let t_dd = t0.elapsed();
+            let err = |u: &TMatrixModes<f64>| {
+                let (mut big, mut diff) = (0.0f64, 0.0f64);
+                for (ma, mb) in u.modes.iter().zip(&b.modes) {
+                    for c in 0..2 {
+                        for (u, v) in ma.0[c].iter().zip(&mb.0[c]) {
+                            let v = Complex::new(v.re.hi(), v.im.hi());
+                            big = big.max(v.norm());
+                            diff = diff.max((u - v).norm());
+                        }
+                    }
+                }
+                diff / big
+            };
+            let skeel = report.error.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
+            let sweeps = report.sweeps.iter().flatten().max().unwrap();
+            let n_ref = report.sweeps.iter().filter(|s| s[0] + s[1] > 0).count();
+            let big = refined
+                .modes
+                .iter()
+                .flat_map(|m| m.0.iter().flatten())
+                .fold(0.0f64, |a, v| a.max(v.norm()));
+            let skeel = skeel / big;
+            println!(
+                "x {x} eps {eps:.3} N {n}: f64 {:.1e} ({t_plain:.2?}), refined {:.1e} ({t_ref:.2?}, \
+                 ≤{sweeps} sweeps, {n_ref} modes), dd ({t_dd:.2?}); est {skeel:.1e}",
+                err(&plain),
+                err(&refined),
+            );
+        }
+    }
+
     /// f64 against double-double, plain integrals: run with
     /// `cargo test --release -p scatrs --lib tmatrix::kernel::diagnostics -- --ignored --nocapture`.
     #[test]
@@ -710,10 +916,10 @@ mod element_diagnostics {
         ] {
             let sh = Spheroid::new(x, eps);
             let shd = Spheroid::new(Dd::from(x), Dd::from(eps));
-            let qp = all_modes(&sh, s, n, nth, false).unwrap();
-            let qr = all_modes(&sh, s, n, nth, true).unwrap();
-            let dp = all_modes(&shd, sd, n, nth, false).unwrap();
-            let dr = all_modes(&shd, sd, n, nth, true).unwrap();
+            let qp = all_modes(&sh, s, n, nth, false, &vec![Some(0); n + 1]).unwrap();
+            let qr = all_modes(&sh, s, n, nth, true, &vec![Some(0); n + 1]).unwrap();
+            let dp = all_modes(&shd, sd, n, nth, false, &vec![Some(0); n + 1]).unwrap();
+            let dr = all_modes(&shd, sd, n, nth, true, &vec![Some(0); n + 1]).unwrap();
             for m in [0usize, 1] {
                 for c in 0..2 {
                     let idx = &qp[m].index[c];
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
index 3a79e3e..b1e3992 100644
--- a/scatrs/src/tmatrix/mod.rs
+++ b/scatrs/src/tmatrix/mod.rs
@@ -8,3 +8,4 @@ pub(crate) mod functions;
 pub(crate) mod kernel;
 pub(crate) mod random;
 pub(crate) mod regular;
+pub(crate) mod solve;
diff --git a/scatrs/src/tmatrix/solve.rs b/scatrs/src/tmatrix/solve.rs
new file mode 100644
index 0000000..2e7239e
--- /dev/null
+++ b/scatrs/src/tmatrix/solve.rs
@@ -0,0 +1,253 @@
+//! The dense solves of the EBCM: LU factorization, T = −P Q⁻¹, Skeel's condition estimate, and
+//! mixed-precision iterative refinement of T.
+//!
+//! In f64 the error of T is about Skeel(Q)·2⁻⁵³ even with exact elements, carried by the rows of
+//! high outgoing order (on a non-spherical surface the evanescent multipoles r^{−n−1}Pₙ, nearly
+//! dependent near r_min). Refinement keeps the O(K³) factorization in f64 and corrects
+//! X Q = −P with double-double residuals, from elements exact to double-double in the rows of
+//! high order: it converges to double-double accuracy while Skeel·2⁻⁵³ < 1.
+
+use num_complex::Complex;
+
+use crate::complex::inv;
+use crate::mie::kernel::real;
+use crate::scalar::{DoubleDouble as Dd, Real};
+use crate::{Error, Result};
+
+/// LU factorization with partial pivoting of a row-major n × n matrix, in place; returns the
+/// pivots.
+pub(crate) fn lu<T: Real>(a: &mut [Complex<T>], n: usize) -> Result<Vec<usize>> {
+    let mut piv: Vec<usize> = (0..n).collect();
+    for k in 0..n {
+        let mut best = k;
+        let mut big = T::zero();
+        for i in k..n {
+            let v = a[i * n + k].norm();
+            if v > big {
+                big = v;
+                best = i;
+            }
+        }
+        if big == T::zero() || !big.is_finite() {
+            return Err(Error::InvalidInput(format!(
+                "the EBCM matrix is singular or not finite at column {k}"
+            )));
+        }
+        if best != k {
+            for j in 0..n {
+                a.swap(k * n + j, best * n + j);
+            }
+            piv.swap(k, best);
+        }
+        let inv_p = inv(a[k * n + k]);
+        for i in k + 1..n {
+            let f = a[i * n + k] * inv_p;
+            a[i * n + k] = f;
+            if f.re == T::zero() && f.im == T::zero() {
+                continue;
+            }
+            for j in k + 1..n {
+                let v = a[k * n + j];
+                a[i * n + j] = a[i * n + j] - f * v;
+            }
+        }
+    }
+    Ok(piv)
+}
+
+/// Solves A x = b in place with the factors of `lu`.
+pub(crate) fn lu_solve<T: Real>(a: &[Complex<T>], piv: &[usize], n: usize, b: &mut [Complex<T>]) {
+    let mut x: Vec<Complex<T>> = piv.iter().map(|&i| b[i]).collect();
+    for i in 0..n {
+        let mut v = x[i];
+        for j in 0..i {
+            v = v - a[i * n + j] * x[j];
+        }
+        x[i] = v;
+    }
+    for i in (0..n).rev() {
+        let mut v = x[i];
+        for j in i + 1..n {
+            v = v - a[i * n + j] * x[j];
+        }
+        x[i] = v * inv(a[i * n + i]);
+    }
+    b.copy_from_slice(&x);
+}
+
+/// T = −P Q⁻¹ of one class (scaled as P and Q are): solved as Qᵀ Tᵀ = −Pᵀ.
+pub(crate) fn solve_t<T: Real>(
+    q: &[Complex<T>],
+    p: &[Complex<T>],
+    n: usize,
+) -> Result<Vec<Complex<T>>> {
+    let mut qt: Vec<Complex<T>> = (0..n * n).map(|ij| q[(ij % n) * n + ij / n]).collect();
+    let piv = lu(&mut qt, n)?;
+    let mut t = vec![real(T::zero()); n * n];
+    let mut col = vec![real(T::zero()); n];
+    for i in 0..n {
+        // Row i of T is column i of Tᵀ: Qᵀ (row i of T)ᵀ = −(row i of P)ᵀ.
+        for j in 0..n {
+            col[j] = -p[i * n + j];
+        }
+        lu_solve(&qt, &piv, n, &mut col);
+        t[i * n..(i + 1) * n].copy_from_slice(&col);
+    }
+    Ok(t)
+}
+
+/// Solves Aᵀ x = b in place with the factors of `lu` (P A = L U, so Aᵀ = Uᵀ Lᵀ P).
+fn lu_solve_transpose(a: &[Complex<f64>], piv: &[usize], n: usize, b: &mut [Complex<f64>]) {
+    let mut y = b.to_vec();
+    for i in 0..n {
+        let mut v = y[i];
+        for j in 0..i {
+            v -= a[j * n + i] * y[j];
+        }
+        y[i] = v / a[i * n + i];
+    }
+    for i in (0..n).rev() {
+        let mut v = y[i];
+        for j in i + 1..n {
+            v -= a[j * n + i] * y[j];
+        }
+        y[i] = v;
+    }
+    for (i, &p) in piv.iter().enumerate() {
+        b[p] = y[i];
+    }
+}
+
+/// One class of one mode solved in f64: T = −P Q⁻¹ (scaled), with the factors of Qᵀ.
+pub(crate) struct F64Solve {
+    pub(crate) t: Vec<Complex<f64>>,
+    qt: Vec<Complex<f64>>,
+    piv: Vec<usize>,
+}
+
+pub(crate) fn solve_f64(q: &[Complex<f64>], p: &[Complex<f64>], n: usize) -> Result<F64Solve> {
+    let mut qt: Vec<Complex<f64>> = (0..n * n).map(|ij| q[(ij % n) * n + ij / n]).collect();
+    let piv = lu(&mut qt, n)?;
+    let mut t = vec![Complex::new(0.0, 0.0); n * n];
+    let mut col = vec![Complex::new(0.0, 0.0); n];
+    for i in 0..n {
+        for j in 0..n {
+            col[j] = -p[i * n + j];
+        }
+        lu_solve(&qt, &piv, n, &mut col);
+        t[i * n..(i + 1) * n].copy_from_slice(&col);
+    }
+    Ok(F64Solve { t, qt, piv })
+}
+
+/// A small deterministic generator of signs for the error probes.
+struct Signs(u64);
+
+impl Signs {
+    fn next(&mut self) -> f64 {
+        // xorshift64*.
+        self.0 ^= self.0 >> 12;
+        self.0 ^= self.0 << 25;
+        self.0 ^= self.0 >> 27;
+        if self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 63 == 0 {
+            1.0
+        } else {
+            -1.0
+        }
+    }
+}
+
+/// An estimate of the largest error of the unscaled T (`t`, scaled, solved with the factors of
+/// `solved`) from the elements' error bounds `eq`, `ep`
+/// (absolute, scaled units; they include the factorization's rounding, about u|Q|): to first
+/// order ΔT' = −(ΔP + T' ΔQ) Q⁻¹, probed with random-signed perturbations at their bounds and
+/// random-signed vectors, three times, with a factor 4 for the largest element. `lh` are the logarithms of the unscaling weights of the
+/// class's orders (T_ik = T'_ik e^{lh_i − lh_k}).
+pub(crate) fn error_estimate(
+    solved: &F64Solve,
+    t: &[Complex<f64>],
+    eq: &[f64],
+    ep: &[f64],
+    lh: &[f64],
+    n: usize,
+) -> f64 {
+    let mut signs = Signs(0x9e37_79b9_7f4a_7c15);
+    let mut worst = 0.0f64;
+    let mut y = vec![Complex::new(0.0, 0.0); n];
+    for _ in 0..3 {
+        // y = Q⁻¹ r', r'_k = ±e^{−lh_k}.
+        for (k, v) in y.iter_mut().enumerate() {
+            *v = Complex::new(signs.next() * (-lh[k]).exp(), 0.0);
+        }
+        lu_solve_transpose(&solved.qt, &solved.piv, n, &mut y);
+        // w = ΔQ y, v = ΔP y + T' w.
+        let w: Vec<Complex<f64>> = (0..n)
+            .map(|l| (0..n).map(|k| y[k] * (signs.next() * eq[l * n + k])).sum())
+            .collect();
+        for i in 0..n {
+            let mut v = Complex::new(0.0, 0.0);
+            for k in 0..n {
+                v += y[k] * (signs.next() * ep[i * n + k]) + t[i * n + k] * w[k];
+            }
+            worst = worst.max(v.norm() * lh[i].exp());
+        }
+    }
+    // The probes see a random combination of the errors: 4× covers the largest element.
+    4.0 * worst
+}
+
+/// Mixed-precision refinement of one class: Q and P are `q`, `p` (f64) except in the rows flagged
+/// by `exact`, where they are `qd`, `pd` (double-double). Factorizes the mixed matrix rounded to
+/// f64, then corrects X Q = −P with double-double residuals R = −P − X Q until the correction
+/// falls below 2⁻⁶⁰ of T or after `sweeps`. Returns T' (scaled), the factors used, the sweeps
+/// made, and the last correction's largest element unscaled (`lh` as in [`error_estimate`]).
+#[allow(clippy::too_many_arguments)]
+pub(crate) fn refine(
+    q: &[Complex<f64>],
+    p: &[Complex<f64>],
+    qd: &[Complex<Dd>],
+    pd: &[Complex<Dd>],
+    exact: &[bool],
+    lh: &[f64],
+    n: usize,
+    sweeps: usize,
+) -> Result<(Vec<Complex<f64>>, F64Solve, usize, f64)> {
+    let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
+    let hi = |c: Complex<Dd>| Complex::new(c.re.hi(), c.im.hi());
+    let pick = |ij: usize, v: &[Complex<f64>], vd: &[Complex<Dd>]| {
+        if exact[ij / n] { vd[ij] } else { dd(v[ij]) }
+    };
+    let qm: Vec<Complex<Dd>> = (0..n * n).map(|ij| pick(ij, q, qd)).collect();
+    let pm: Vec<Complex<Dd>> = (0..n * n).map(|ij| pick(ij, p, pd)).collect();
+    let q53: Vec<Complex<f64>> = qm.iter().map(|&c| hi(c)).collect();
+    let p53: Vec<Complex<f64>> = pm.iter().map(|&c| hi(c)).collect();
+    let solved = solve_f64(&q53, &p53, n)?;
+    let mut x: Vec<Complex<Dd>> = solved.t.iter().map(|&c| dd(c)).collect();
+    let big = solved.t.iter().map(|c| c.norm()).fold(0.0, f64::max);
+    let (mut done, mut last) = (0, 0.0f64);
+    let mut row = vec![Complex::new(0.0, 0.0); n];
+    for _ in 0..sweeps {
+        done += 1;
+        let (mut worst, mut worst_unscaled) = (0.0f64, 0.0f64);
+        for i in 0..n {
+            for j in 0..n {
+                let mut r = -pm[i * n + j];
+                for l in 0..n {
+                    r -= x[i * n + l] * qm[l * n + j];
+                }
+                row[j] = hi(r);
+            }
+            lu_solve(&solved.qt, &solved.piv, n, &mut row);
+            for j in 0..n {
+                worst = worst.max(row[j].norm());
+                worst_unscaled = worst_unscaled.max(row[j].norm() * (lh[i] - lh[j]).exp());
+                x[i * n + j] += dd(row[j]);
+            }
+        }
+        last = worst_unscaled;
+        if worst <= big * 2f64.powi(-60) {
+            break;
+        }
+    }
+    Ok((x.iter().map(|&c| hi(c)).collect(), solved, done, last))
+}
````

### Task 4: Public API, convergence loop and Greek coefficients

- [ ] `GreekProjector::project_six`; `TMatrixModes::greek_raw`.
- [ ] `tmatrix/driver.rs`: `Precision`, `PrecisionPath`, `solve` (paths), `converge` (mode-0 loop, all modes, Greek check).
- [ ] `tmatrix/mod.rs`: `Shape`, `TMatrixOptions`, `TMatrix`, `TMatrixSolution`, `ModeBlocks`, six-element `PhaseBuffers`; outputs from the Greek expansion.
- [ ] Speed: suffix sums for the series tail, hoisted s-scaled fields, sparse `angular_factors`, |re| + |im| bookkeeping; the double-double path solves only the modes the loop needs (`tmatrix_upto`).
- [ ] Test: ε = 1 equals Mie (efficiencies, Greek, phase), lossless and normalizations, input errors, caps.

The commits (`86394e0`, `b7c20cf`):

````diff
diff --git a/scatrs/src/mie/greek.rs b/scatrs/src/mie/greek.rs
index 1e62457..d40ab13 100644
--- a/scatrs/src/mie/greek.rs
+++ b/scatrs/src/mie/greek.rs
@@ -29,7 +29,7 @@ pub struct GreekCoefficients<T> {
 }
 
 impl<T: Real> GreekCoefficients<T> {
-    fn zeros(len: usize) -> Self {
+    pub(crate) fn zeros(len: usize) -> Self {
         let z = || vec![T::zero(); len];
         Self {
             alpha1: z(),
@@ -70,14 +70,14 @@ impl<T: Real> GreekCoefficients<T> {
     }
 
     /// Extends every array with zeros to `len` values.
-    fn pad(&mut self, len: usize) {
+    pub(crate) fn pad(&mut self, len: usize) {
         for v in self.arrays_mut() {
             v.resize(len, T::zero());
         }
     }
 
     /// Applies `f` to every coefficient.
-    fn map(&mut self, f: impl Fn(T) -> T) {
+    pub(crate) fn map(&mut self, f: impl Fn(T) -> T) {
         for v in self.arrays_mut() {
             for c in v.iter_mut() {
                 *c = f(*c);
@@ -220,6 +220,52 @@ impl<T: Real> GreekProjector<T> {
     }
 }
 
+impl<T: Real> GreekProjector<T> {
+    /// The Greek coefficients of a phase matrix with six independent elements at the nodes,
+    /// [F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄] (a sphere's projection, [`project`](Self::project), takes
+    /// F₂₂ = F₁₁ and F₄₄ = F₃₃).
+    pub(crate) fn project_six(&self, f: [&[T]; 6]) -> Result<GreekCoefficients<T>> {
+        let n = self.rule.nodes.len();
+        if f.iter().any(|v| v.len() != n) {
+            return Err(Error::InvalidInput(format!(
+                "phase-matrix tables must have one value per node ({n})"
+            )));
+        }
+        let len = self.order + 1;
+        let mut c = GreekCoefficients::zeros(len);
+        let mut d: [Vec<T>; 4] = core::array::from_fn(|_| vec![T::zero(); len]);
+        for (k, (&mu, &w)) in self.rule.nodes.iter().zip(&self.rule.weights).enumerate() {
+            let [d00, d22, d2m2, d02] = &mut d;
+            wigner_d(0, 0, mu, d00)?;
+            wigner_d(2, 2, mu, d22)?;
+            wigner_d(2, -2, mu, d2m2)?;
+            wigner_d(0, 2, mu, d02)?;
+            let [f11, f12, f22, f33, f34, f44] = f.map(|v| w * v[k]);
+            let (sum, diff) = (f22 + f33, f22 - f33);
+            for l in 0..len {
+                c.alpha1[l] += f11 * d00[l];
+                c.alpha4[l] += f44 * d00[l];
+                c.alpha2[l] += sum * d22[l]; // α₂ + α₃ for now
+                c.alpha3[l] += diff * d2m2[l]; // α₂ − α₃ for now
+                c.beta1[l] += f12 * d02[l];
+                c.beta2[l] += f34 * d02[l];
+            }
+        }
+        let half = T::from_f64(0.5);
+        for l in 0..len {
+            let s = T::from_f64((2 * l + 1) as f64) * half;
+            let (plus, minus) = (c.alpha2[l], c.alpha3[l]);
+            c.alpha1[l] *= s;
+            c.alpha4[l] *= s;
+            c.alpha2[l] = s * half * (plus + minus);
+            c.alpha3[l] = s * half * (plus - minus);
+            c.beta1[l] *= s;
+            c.beta2[l] *= s;
+        }
+        Ok(c)
+    }
+}
+
 impl<T: Real> GreekProjector<T> {
     /// Bounds on the coefficients' changes when each table value changes by at most the given
     /// amount (the tables hold non-negative bounds on |ΔF₁₁|, |ΔF₁₂|, |ΔF₃₃|, |ΔF₃₄|): by the
diff --git a/scatrs/src/tmatrix/domain.rs b/scatrs/src/tmatrix/domain.rs
new file mode 100644
index 0000000..18cb42b
--- /dev/null
+++ b/scatrs/src/tmatrix/domain.rs
@@ -0,0 +1,84 @@
+//! The validated domain of randomly oriented spheroids and their declared accuracy.
+
+use num_complex::Complex;
+
+use super::driver::x_max;
+use crate::mie::MieOutput;
+
+/// The region of inputs whose accuracy has been validated, and the declared tolerances there.
+///
+/// Inside it, every output of [`TMatrix::compute`](super::TMatrix::compute), and of its solution's
+/// methods, is within
+///
+/// C·max(`rtol`, ε̂)
+///
+/// of the exact value, in the measure of its [`MieOutput`], with ε̂ the solution's
+/// [`error_estimate`](super::TMatrixSolution::error_estimate). The constants are measured, at
+/// least 4 times the worst case found at every high-precision reference value in the domain and
+/// on the accuracy maps.
+#[derive(Clone, Copy, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct ValidatedDomain {
+    /// The smallest validated aspect ratio ε (prolate below 1).
+    pub aspect_ratio_min: f64,
+    /// The largest validated aspect ratio ε (oblate above 1).
+    pub aspect_ratio_max: f64,
+    /// The smallest validated equal-volume size parameter.
+    pub x_min: f64,
+    /// The largest validated |m|·x_max, with x_max = k·max(a, c) the size parameter of the
+    /// longest semi-axis.
+    pub max_phase: f64,
+    /// The largest validated Re m (the smallest is `abs_m_min`).
+    pub m_re_max: f64,
+    /// The largest validated Im m (the smallest is 0).
+    pub m_im_max: f64,
+    /// The smallest validated |m|.
+    pub abs_m_min: f64,
+    /// The smallest validated `rtol`.
+    pub rtol_min: f64,
+}
+
+/// The validated domain: aspect ratios 1/5 to 5, equal-volume x ≥ 10⁻³, |m|·x_max ≤ 60,
+/// 1 ≤ Re m ≤ 3, Im m ≤ 1, |m| ≥ 1, `rtol` ≥ 10⁻¹².
+pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
+    aspect_ratio_min: 0.2,
+    aspect_ratio_max: 5.0,
+    x_min: 1e-3,
+    max_phase: 60.0,
+    m_re_max: 3.0,
+    m_im_max: 1.0,
+    abs_m_min: 1.0,
+    rtol_min: 1e-12,
+};
+
+impl ValidatedDomain {
+    /// Whether the spheroid of aspect ratio `eps`, equal-volume size parameter `x` and index `m`
+    /// is inside the domain (for any `rtol` ≥ `rtol_min`).
+    pub fn contains(&self, eps: f64, x: f64, m: Complex<f64>) -> bool {
+        (self.aspect_ratio_min..=self.aspect_ratio_max).contains(&eps)
+            && x >= self.x_min
+            && m.norm() * x_max(x, eps) <= self.max_phase
+            && m.re <= self.m_re_max
+            && (0.0..=self.m_im_max).contains(&m.im)
+            && m.norm() >= self.abs_m_min
+    }
+
+    /// The declared bound on the error of `output`: C·max(`rtol`, `estimate`).
+    pub fn tolerance(&self, output: MieOutput, rtol: f64, estimate: f64) -> f64 {
+        Self::constant(output) * rtol.max(estimate)
+    }
+
+    /// The constant C of [`tolerance`](Self::tolerance), measured on the accuracy maps (at least
+    /// 4 times the worst case found there and at the references).
+    ///
+    /// | output | C |
+    /// |---|---|
+    /// | Q_ext, Q_sca, g | 4 |
+    /// | Q_abs, Q_back, phase matrix, Greek coefficients | 16 |
+    pub fn constant(output: MieOutput) -> f64 {
+        match output {
+            MieOutput::Extinction | MieOutput::Scattering | MieOutput::Asymmetry => 4.0,
+            _ => 16.0,
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/driver.rs b/scatrs/src/tmatrix/driver.rs
new file mode 100644
index 0000000..cc757c6
--- /dev/null
+++ b/scatrs/src/tmatrix/driver.rs
@@ -0,0 +1,333 @@
+//! The convergence loop and the precision paths: T of a spheroid to a relative tolerance.
+//!
+//! 1. N grows from an estimate until the relative changes of mode 0's contributions to ⟨C_ext⟩
+//!    and ⟨C_sca⟩ fall below `rtol`, with n_θ tied to N (as in Mishchenko's `tmatd`: mode 0
+//!    involves every order, and the higher modes converge faster).
+//! 2. n_θ then grows until they settle again.
+//! 3. All modes are solved at the final N and n_θ.
+//! 4. If Greek coefficients are asked for, one more step in N and n_θ, with all modes, must
+//!    leave them and the cross sections settled.
+//!
+//! Each step solves every mode in f64 with an error estimate, and refines the modes whose
+//! estimate exceeds a tenth of `rtol` (relative to the largest element of T). If refinement
+//! leaves an estimate above that, the whole T is computed in double-double.
+
+use num_complex::Complex;
+
+use super::kernel::{SolveReport, TMatrixModes, tmatrix, tmatrix_refined};
+use crate::scalar::DoubleDouble as Dd;
+use crate::{Error, Result};
+
+/// How the modes of a T-matrix were solved.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+#[non_exhaustive]
+pub enum PrecisionPath {
+    /// f64 elements and an f64 solve.
+    F64,
+    /// The f64 solve refined with double-double residuals, the rows of high outgoing order
+    /// integrated in double-double.
+    Refined,
+    /// Elements and solve in double-double.
+    DoubleDouble,
+}
+
+/// Which precision paths may be used.
+#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
+#[non_exhaustive]
+pub enum Precision {
+    /// f64, refined or double-double per mode, as the error estimates require.
+    #[default]
+    Auto,
+    /// f64 only, whatever the estimates (fastest; the estimates still report the error).
+    F64,
+    /// Double-double throughout (slowest; for checks).
+    DoubleDouble,
+}
+
+/// One T-matrix with how it was solved.
+#[derive(Clone, Debug)]
+pub(crate) struct Solved {
+    pub(crate) t: TMatrixModes<f64>,
+    pub(crate) paths: Vec<PrecisionPath>,
+    /// The estimated largest error of an element of T (absolute).
+    pub(crate) error: f64,
+    pub(crate) n_theta: usize,
+}
+
+impl Solved {
+    fn largest(&self) -> f64 {
+        self.t
+            .modes
+            .iter()
+            .flat_map(|(ts, _)| ts.iter().flatten())
+            .fold(0.0, |a, v| a.max(v.norm()))
+    }
+}
+
+/// T of the modes m ≤ `m_max` at (N, n_θ) on the paths `precision` allows.
+#[allow(clippy::too_many_arguments)]
+pub(crate) fn solve(
+    x: f64,
+    eps: f64,
+    s: Complex<f64>,
+    n: usize,
+    n_theta: usize,
+    precision: Precision,
+    rtol: f64,
+    m_max: usize,
+) -> Result<Solved> {
+    let full_dd = || -> Result<Solved> {
+        let sd = Complex::new(Dd::from(s.re), Dd::from(s.im));
+        let t = tmatrix(Dd::from(x), Dd::from(eps), sd, n, n_theta, true)?;
+        let modes = t
+            .modes
+            .into_iter()
+            .map(|(ts, index)| {
+                let ts = ts.map(|v| {
+                    v.iter()
+                        .map(|c| Complex::new(c.re.hi(), c.im.hi()))
+                        .collect()
+                });
+                (ts, index)
+            })
+            .collect();
+        let t = TMatrixModes { n, modes };
+        let mut out = Solved {
+            t,
+            paths: vec![PrecisionPath::DoubleDouble; n + 1],
+            error: 0.0,
+            n_theta,
+        };
+        // The rounding to f64.
+        out.error = out.largest() * f64::EPSILON;
+        Ok(out)
+    };
+    if precision == Precision::DoubleDouble {
+        return full_dd();
+    }
+    let tol = if precision == Precision::F64 {
+        f64::INFINITY
+    } else {
+        rtol / 10.0
+    };
+    let (t, report) = tmatrix_refined(x, eps, s, n, n_theta, tol, m_max)?;
+    let out = from_report(t, &report, n_theta);
+    if precision == Precision::Auto && out.error > tol * out.largest() {
+        return full_dd();
+    }
+    Ok(out)
+}
+
+fn from_report(t: TMatrixModes<f64>, report: &SolveReport, n_theta: usize) -> Solved {
+    let paths = report
+        .sweeps
+        .iter()
+        .map(|s| {
+            if s[0] + s[1] > 0 {
+                PrecisionPath::Refined
+            } else {
+                PrecisionPath::F64
+            }
+        })
+        .collect();
+    let error = report.error.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
+    Solved {
+        t,
+        paths,
+        error,
+        n_theta,
+    }
+}
+
+/// The settings of the convergence loop.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct Loop {
+    pub(crate) rtol: f64,
+    pub(crate) precision: Precision,
+    /// Fixed N, or the cap on N.
+    pub(crate) n_fixed: Option<usize>,
+    pub(crate) n_cap: usize,
+    /// Fixed n_θ, or the cap on n_θ.
+    pub(crate) n_theta_fixed: Option<usize>,
+    pub(crate) n_theta_cap: usize,
+    /// The Greek order that must settle, if any.
+    pub(crate) greek_order: Option<usize>,
+    /// Return the last step, with its estimate, instead of `NotConverged` at a cap.
+    pub(crate) allow_unconverged: bool,
+}
+
+/// The result of the loop: the accepted T, and the last convergence change (relative).
+#[derive(Clone, Debug)]
+pub(crate) struct Converged {
+    pub(crate) solved: Solved,
+    pub(crate) change: f64,
+    pub(crate) converged: bool,
+}
+
+/// x_max = k·max(a, c) for the equal-volume size parameter x: a = x ε^{1/3}, c = a/ε.
+pub(crate) fn x_max(x: f64, eps: f64) -> f64 {
+    let a = x * eps.cbrt();
+    a.max(a / eps)
+}
+
+/// The starting N, Wiscombe's estimate at x_max, which the loop then grows:
+/// N₀ = ⌈x_max + 4 x_max^{1/3}⌉ + 2, at least 4.
+pub(crate) fn start_n(x: f64, eps: f64) -> usize {
+    let xm = x_max(x, eps);
+    ((xm + 4.0 * xm.cbrt()).ceil() as usize + 2).max(4)
+}
+
+/// n_θ tied to N: N(1 + |ln ε|) + 8 nodes in 0 < θ < π/2.
+pub(crate) fn theta_of(n: usize, eps: f64) -> usize {
+    (n as f64 * (1.0 + eps.ln().abs())).ceil() as usize + 8
+}
+
+fn relative_change(a: &Solved, b: &Solved) -> f64 {
+    let (ea, sa) = a.t.cross_sections();
+    let (eb, sb) = b.t.cross_sections();
+    let rel = |u: f64, v: f64| {
+        if u == v {
+            0.0
+        } else {
+            (u - v).abs() / u.abs().max(v.abs())
+        }
+    };
+    rel(ea, eb).max(rel(sa, sb))
+}
+
+fn greek_change(a: &Solved, b: &Solved, order: usize) -> Result<f64> {
+    let ga = a.t.greek_raw(order)?;
+    let gb = b.t.greek_raw(order)?;
+    // Normalized by each one's α₁ at l = 0; the largest change relative to the largest α₁.
+    let (na, nb) = (ga.alpha1[0], gb.alpha1[0]);
+    let big = ga.alpha1.iter().fold(0.0f64, |m, v| m.max((v / na).abs()));
+    let len = ga.alpha1.len().max(gb.alpha1.len());
+    let at = |v: &[f64], l: usize, norm: f64| v.get(l).map_or(0.0, |x| x / norm);
+    let mut worst = 0.0f64;
+    for (va, vb) in [
+        (&ga.alpha1, &gb.alpha1),
+        (&ga.alpha2, &gb.alpha2),
+        (&ga.alpha3, &gb.alpha3),
+        (&ga.alpha4, &gb.alpha4),
+        (&ga.beta1, &gb.beta1),
+        (&ga.beta2, &gb.beta2),
+    ] {
+        for l in 0..len {
+            worst = worst.max((at(va, l, na) - at(vb, l, nb)).abs());
+        }
+    }
+    Ok(worst / big)
+}
+
+/// Runs the loop for the spheroid (x, ε) of index s.
+pub(crate) fn converge(x: f64, eps: f64, s: Complex<f64>, opts: &Loop) -> Result<Converged> {
+    let rtol = opts.rtol;
+    let theta = |n: usize| {
+        opts.n_theta_fixed
+            .unwrap_or_else(|| theta_of(n, eps).min(opts.n_theta_cap))
+    };
+    let solve_at =
+        |n: usize, nth: usize, m_max: usize| solve(x, eps, s, n, nth, opts.precision, rtol, m_max);
+    let not_converged = |change: f64| Error::NotConverged {
+        achieved: change,
+        requested: rtol,
+    };
+    let mut change = 0.0f64;
+    let mut converged = true;
+    let mut stop = |change: f64| -> Result<()> {
+        if opts.allow_unconverged {
+            converged = false;
+            Ok(())
+        } else {
+            Err(not_converged(change))
+        }
+    };
+    // 1. N, on mode 0.
+    let mut n = opts
+        .n_fixed
+        .unwrap_or_else(|| start_n(x, eps).min(opts.n_cap));
+    let mut nth = theta(n);
+    if opts.n_fixed.is_none() {
+        let mut cur = solve_at(n, nth, 0)?;
+        loop {
+            if n >= opts.n_cap {
+                stop(change)?;
+                break;
+            }
+            let next = (n + (n / 8).max(2)).min(opts.n_cap);
+            let trial = solve_at(next, theta(next), 0)?;
+            change = relative_change(&cur, &trial);
+            (n, nth, cur) = (next, theta(next), trial);
+            if change < rtol {
+                break;
+            }
+        }
+        // 2. n_θ, on mode 0.
+        if opts.n_theta_fixed.is_none() {
+            loop {
+                if nth >= opts.n_theta_cap {
+                    stop(change)?;
+                    break;
+                }
+                let next = (nth + (nth / 4).max(4)).min(opts.n_theta_cap);
+                let trial = solve_at(n, next, 0)?;
+                let c = relative_change(&cur, &trial);
+                change = change.max(c);
+                (nth, cur) = (next, trial);
+                if c < rtol {
+                    break;
+                }
+            }
+        }
+    } else if opts.n_theta_fixed.is_none() {
+        let mut cur = solve_at(n, nth, 0)?;
+        loop {
+            if nth >= opts.n_theta_cap {
+                stop(change)?;
+                break;
+            }
+            let next = (nth + (nth / 4).max(4)).min(opts.n_theta_cap);
+            let trial = solve_at(n, next, 0)?;
+            let c = relative_change(&cur, &trial);
+            change = change.max(c);
+            (nth, cur) = (next, trial);
+            if c < rtol {
+                break;
+            }
+        }
+    }
+    // 3. All modes.
+    let mut full = solve_at(n, nth, n)?;
+    // 4. The Greek coefficients.
+    if let Some(order) = opts.greek_order {
+        let next = n + (n / 8).max(2);
+        if opts.n_fixed.is_some() || next > opts.n_cap {
+            // No step is possible: the order is checked against n_θ alone if it may grow.
+            if opts.n_theta_fixed.is_none() {
+                let nth2 = (nth + (nth / 4).max(4)).min(opts.n_theta_cap);
+                let trial = solve_at(n, nth2, n)?;
+                let c = greek_change(&full, &trial, order)?.max(relative_change(&full, &trial));
+                change = change.max(c);
+                full = trial;
+                if c >= rtol {
+                    stop(c)?;
+                }
+            }
+        } else {
+            let nth2 = (nth * next).div_ceil(n).min(opts.n_theta_cap);
+            let nth2 = opts.n_theta_fixed.unwrap_or(nth2);
+            let trial = solve_at(next, nth2, next)?;
+            let c = greek_change(&full, &trial, order)?.max(relative_change(&full, &trial));
+            change = change.max(c);
+            full = trial;
+            if c >= rtol {
+                stop(c)?;
+            }
+        }
+    }
+    Ok(Converged {
+        solved: full,
+        change,
+        converged,
+    })
+}
diff --git a/scatrs/src/tmatrix/kernel.rs b/scatrs/src/tmatrix/kernel.rs
index 2b45e16..8b82cb1 100644
--- a/scatrs/src/tmatrix/kernel.rs
+++ b/scatrs/src/tmatrix/kernel.rs
@@ -21,7 +21,7 @@
 use num_complex::Complex;
 
 use super::functions::{Angular, Radial, Scales};
-use super::regular::{Series, angular_factors};
+use super::regular::{Series, angular_factors, l1};
 use super::solve::{error_estimate, refine, solve_f64, solve_t};
 use crate::Result;
 use crate::mie::kernel::{int, real};
@@ -142,6 +142,8 @@ pub(crate) fn all_modes<T: Real>(
     let mut reg_n = vec![zero3; n + 1];
     let mut in_m = vec![zero3; n + 1];
     let mut in_n = vec![zero3; n + 1];
+    let mut in_ms = vec![zero3; n + 1];
+    let mut in_ns = vec![zero3; n + 1];
     let n1 = n + 1;
     // forms[(t * n1 + n) * n1 + k] at the node.
     let mut forms = Vec::new();
@@ -221,6 +223,8 @@ pub(crate) fn all_modes<T: Real>(
                 };
                 (reg_m[k], reg_n[k]) = field(radial.j[k], radial.dj[k], radial.j_x[k], real(-pi));
                 (in_m[k], in_n[k]) = field(radial.jz[k], radial.djz[k], radial.jz_z[k], real(pi));
+                in_ms[k] = in_m[k].map(|v| v * s);
+                in_ns[k] = in_n[k].map(|v| v * s);
             }
             for c in 0..2 {
                 let idx = &mm.index[c];
@@ -237,15 +241,15 @@ pub(crate) fn all_modes<T: Real>(
                     let an = (ang.d[nn], ang.pi[nn], ang.tau[nn]);
                     for (j, &(ctp, kk_)) in idx.iter().enumerate() {
                         let (b1, b2) = if ctp == 0 {
-                            (in_n[kk_].map(|v| v * s), &in_m[kk_])
+                            (&in_ns[kk_], &in_m[kk_])
                         } else {
-                            (in_m[kk_].map(|v| v * s), &in_n[kk_])
+                            (&in_ms[kk_], &in_n[kk_])
                         };
                         let gg = gam[nn] * gam[kk_];
-                        let pv = triple(nu, g1, &b1) + triple(nu, g2, b2);
+                        let pv = triple(nu, g1, b1) + triple(nu, g2, b2);
                         let slot = i * kc + j;
                         mm.p[c][slot] = mm.p[c][slot] + Complex::new(pv.im, -pv.re).scale(fac * gg);
-                        mm.p_abs[c][slot] += pv.norm() * (fac * gg).abs();
+                        mm.p_abs[c][slot] += l1(pv) * (fac * gg).abs();
                         // Y: Σ_t a_t R_t in its three forms.
                         let ak = (ang.d[kk_], ang.pi[kk_], ang.tau[kk_]);
                         let a = angular_factors(rt, ctp, s, nu_prime, nn, kk_, an, ak);
@@ -253,15 +257,12 @@ pub(crate) fn all_modes<T: Real>(
                         let (mut va, mut aa) = (zero, T::zero());
                         let (mut vb, mut ab) = (zero, T::zero());
                         let (mut vc, mut ac) = (zero, T::zero());
-                        for (ti, &at) in a.iter().enumerate() {
-                            if at == zero {
-                                continue;
-                            }
+                        for &(ti, at) in &a {
                             let fi = (ti * n1 + nn) * n1 + kk_;
                             let rf = full[fi];
-                            let mag = at.norm();
+                            let mag = l1(at);
                             va = va + at * rf;
-                            aa += mag * rf.norm();
+                            aa += mag * l1(rf);
                             if let Some(Some(f)) = forms.get(fi) {
                                 vc = vc + at * f.le0;
                                 ac += mag * f.le0_abs;
@@ -326,6 +327,7 @@ pub(crate) type ModeT<T> = ([Vec<Complex<T>>; 2], [Vec<(usize, usize)>; 2]);
 
 /// The T-matrix of a spheroid, mode by mode: for each m = 0, …, N the two classes' T (unscaled)
 /// with their index sets.
+#[derive(Clone, Debug)]
 pub(crate) struct TMatrixModes<T> {
     pub(crate) n: usize,
     pub(crate) modes: Vec<ModeT<T>>,
@@ -389,7 +391,8 @@ fn refine_from(m: usize, n: usize) -> usize {
     n0 + (n + 1 - n0) / 4
 }
 
-/// T of every mode in f64, refined (mixed precision) in the modes whose error estimate exceeds
+/// T of the modes m ≤ `m_max` (the others are left empty) in f64, refined (mixed precision) in
+/// the modes whose error estimate exceeds
 /// `tol` times the largest |T|: their rows of order ≥ [`refine_from`] are integrated in
 /// double-double, and T is corrected with double-double residuals.
 pub(crate) fn tmatrix_refined(
@@ -399,6 +402,7 @@ pub(crate) fn tmatrix_refined(
     n: usize,
     n_theta: usize,
     tol: f64,
+    m_max: usize,
 ) -> Result<(TMatrixModes<f64>, SolveReport)> {
     // The elements' rounding: u Σ|terms| times 2N for the accuracy of the function values (the
     // largest true error measured is about N u Σ|terms|).
@@ -407,7 +411,8 @@ pub(crate) fn tmatrix_refined(
     let shape = Spheroid::new(x, eps);
     let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
     let lh = log_h(&scales);
-    let all = all_modes(&shape, s, n, n_theta, true, &vec![Some(0); n + 1])?;
+    let modes_in: Vec<Option<usize>> = (0..=n).map(|m| (m <= m_max).then_some(0)).collect();
+    let all = all_modes(&shape, s, n, n_theta, true, &modes_in)?;
     let mut report = SolveReport {
         error: vec![[0.0; 2]; n + 1],
         sweeps: vec![[0; 2]; n + 1],
@@ -419,7 +424,7 @@ pub(crate) fn tmatrix_refined(
         for (c, slot) in pair.iter_mut().enumerate() {
             let index = &mm.index[c];
             let kc = index.len();
-            if kc == 0 {
+            if kc == 0 || m > m_max {
                 continue;
             }
             let f = solve_f64(&mm.q[c], &mm.p[c], kc)?;
@@ -437,7 +442,7 @@ pub(crate) fn tmatrix_refined(
     let rows: Vec<Option<usize>> = (0..=n)
         .map(|m| {
             let worst = report.error[m][0].max(report.error[m][1]);
-            (worst > tol * big).then(|| refine_from(m, n))
+            (m <= m_max && worst > tol * big).then(|| refine_from(m, n))
         })
         .collect();
     if rows.iter().any(Option::is_some) {
@@ -507,6 +512,9 @@ impl<T: Real> TMatrixModes<T> {
             let w = if m == 0 { T::one() } else { T::from_f64(2.0) };
             for (t, idx) in ts.iter().zip(index) {
                 let kc = idx.len();
+                if t.is_empty() {
+                    continue;
+                }
                 for i in 0..kc {
                     ext -= w * t[i * kc + i].re;
                     for v in &t[i * kc..(i + 1) * kc] {
@@ -531,8 +539,8 @@ mod tests {
         use crate::scalar::DoubleDouble as Dd;
         let (x, eps, n, nth) = (7.0, 3.0, 20, 40);
         let s = Complex::new(1.5, 0.01);
-        let (f, f_rep) = tmatrix_refined(x, eps, s, n, nth, 1.0).unwrap();
-        let (r, r_rep) = tmatrix_refined(x, eps, s, n, nth, 0.0).unwrap();
+        let (f, f_rep) = tmatrix_refined(x, eps, s, n, nth, 1.0, n).unwrap();
+        let (r, r_rep) = tmatrix_refined(x, eps, s, n, nth, 0.0, n).unwrap();
         let sd = Complex::new(Dd::from(s.re), Dd::from(s.im));
         let d = tmatrix(Dd::from(x), Dd::from(eps), sd, n, nth, true).unwrap();
         for (u, rep) in [(&f, &f_rep), (&r, &r_rep)] {
@@ -764,7 +772,7 @@ mod diagnostics {
         let s = Complex::new(1.5, 0.01);
         let (x, eps, n, nth) = (15.0, 0.333_333_333_333, 42, 90);
         let plain = tmatrix(x, eps, s, n, nth, true).unwrap();
-        let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 0.0).unwrap();
+        let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 0.0, n).unwrap();
         let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
         for m in 0..=n {
             for c in 0..2 {
@@ -803,7 +811,7 @@ mod diagnostics {
             let plain = tmatrix(x, eps, s, n, nth, true).unwrap();
             let t_plain = t0.elapsed();
             let t0 = std::time::Instant::now();
-            let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 1e-14).unwrap();
+            let (refined, report) = tmatrix_refined(x, eps, s, n, nth, 1e-14, n).unwrap();
             let t_ref = t0.elapsed();
             let t0 = std::time::Instant::now();
             let b = tmatrix(Dd::from(x), Dd::from(eps), dd(s), n, nth, true).unwrap();
@@ -954,3 +962,40 @@ mod element_diagnostics {
         }
     }
 }
+
+#[cfg(test)]
+mod profile {
+    use super::*;
+
+    /// One double-double `all_modes`, to profile: run with `--ignored`.
+    #[test]
+    #[ignore = "a profile target"]
+    fn dd_all_modes() {
+        let (x, eps, n, nth) = (15.0, 2.0, 40, 80);
+        let sh = Spheroid::new(Dd::from(x), Dd::from(eps));
+        let s = Complex::new(Dd::from(1.53), Dd::from(0.008));
+        let t0 = std::time::Instant::now();
+        let rows = vec![Some(0); n + 1];
+        all_modes(&sh, s, n, nth, true, &rows).unwrap();
+        println!("dd regularized all rows: {:.2?}", t0.elapsed());
+        let t0 = std::time::Instant::now();
+        all_modes(&sh, s, n, nth, false, &rows).unwrap();
+        println!("dd plain all rows: {:.2?}", t0.elapsed());
+        let mut one = vec![None; n + 1];
+        one[0] = Some(0);
+        let t0 = std::time::Instant::now();
+        all_modes(&sh, s, n, nth, true, &one).unwrap();
+        println!("dd regularized mode 0 only: {:.2?}", t0.elapsed());
+        let shf = Spheroid::new(x, eps);
+        let sf = Complex::new(1.53, 0.008);
+        let t0 = std::time::Instant::now();
+        all_modes(&shf, sf, n, nth, true, &rows).unwrap();
+        println!("f64 regularized: {:.2?}", t0.elapsed());
+        let t0 = std::time::Instant::now();
+        all_modes(&shf, sf, n, nth, false, &rows).unwrap();
+        println!("f64 plain: {:.2?}", t0.elapsed());
+        let t0 = std::time::Instant::now();
+        all_modes(&shf, sf, n, nth, true, &one).unwrap();
+        println!("f64 regularized mode 0 only: {:.2?}", t0.elapsed());
+    }
+}
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
index b1e3992..5cf2b1f 100644
--- a/scatrs/src/tmatrix/mod.rs
+++ b/scatrs/src/tmatrix/mod.rs
@@ -1,11 +1,772 @@
-//! The T-matrix of spheroids by the extended boundary condition method (EBCM), and the optical
-//! properties of randomly oriented spheroids. See the theory chapter of the documentation.
-
-// The kernel has no public entry point until the API lands later in phase 4a.
-#![allow(dead_code)]
+//! Scattering by randomly oriented spheroids, by the T-matrix method.
+//!
+//! [`TMatrix`] is a reusable workspace: [`TMatrix::compute`] solves one spheroid, converging the
+//! T-matrix to a relative tolerance, and returns a [`TMatrixSolution`] with the
+//! orientation-averaged efficiencies, Greek coefficients and phase matrix (six independent
+//! elements: F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄), the T-matrix itself, and how it was obtained.
+//!
+//! ```
+//! use num_complex::Complex;
+//! use scatrs::tmatrix::{Shape, TMatrix, TMatrixOptions};
+//!
+//! // An oblate dust spheroid, twice as wide as tall, of equal-volume size parameter 5.
+//! let shape = Shape::Spheroid { aspect_ratio: 2.0 };
+//! let mut tm = TMatrix::new();
+//! let s = tm.compute(shape, 5.0, Complex::new(1.53, 0.008), &TMatrixOptions::new())?;
+//! let q = s.efficiencies();
+//! assert!(q.q_abs > 0.0 && q.q_ext > q.q_sca);
+//! assert!(s.error_estimate() < 1e-8);
+//! # Ok::<(), scatrs::Error>(())
+//! ```
+//!
+//! # Conventions
+//!
+//! A spheroid has semi-axes a, a, c, with c along its symmetry axis, and aspect ratio
+//! ε = a/c: oblate for ε > 1, prolate for ε < 1, a sphere for ε = 1. Its size is the size
+//! parameter x = 2πr_v/λ of the sphere of equal volume (r_v³ = a²c), which also normalizes the
+//! efficiencies (Q = C/(πr_v²)) and the unnormalized phase matrix and Greek coefficients. The
+//! refractive index m is relative to the medium. The phase matrix is that of
+//! [`crate::mie`] (Bohren & Huffman's), averaged uniformly over orientations, and its Greek
+//! coefficients are those of [`crate::mie::GreekCoefficients`] with F₂₂ and F₄₄ independent.
+//!
+//! # Method
+//!
+//! See the theory chapter of the documentation, "Spheroids". The extended boundary condition
+//! method gives the T-matrix of each azimuthal mode from surface integrals. Their outgoing-wave
+//! parts are summed in a form free of the electrostatic cancellation that limits the classical
+//! method. Each mode is solved in `f64` with an estimate of its error, and the modes that need
+//! it are refined in mixed precision (double-double residuals, the high outgoing orders
+//! integrated in double-double). The orientation average follows from the rank decomposition of
+//! T (Mishchenko 1991), and the Greek coefficients from its exact projection. The number of
+//! orders N and of quadrature nodes n_θ grow until the cross sections (and the Greek
+//! coefficients, if asked for) settle to `rtol`.
+//!
+//! # Accuracy
+//!
+//! See [`VALIDATED_DOMAIN`].
 
+mod domain;
+mod driver;
 pub(crate) mod functions;
 pub(crate) mod kernel;
 pub(crate) mod random;
 pub(crate) mod regular;
 pub(crate) mod solve;
+
+use std::sync::OnceLock;
+
+use num_complex::Complex;
+
+pub use domain::{VALIDATED_DOMAIN, ValidatedDomain};
+pub use driver::{Precision, PrecisionPath};
+
+use crate::mie::{Efficiencies, GreekCoefficients, MAX_GREEK_ORDER, Normalization};
+use crate::special::wigner_d;
+use crate::{Error, Result};
+
+/// The shape of a particle.
+#[derive(Clone, Copy, Debug, PartialEq)]
+#[non_exhaustive]
+pub enum Shape {
+    /// A spheroid of aspect ratio ε = a/c (horizontal over polar semi-axis): oblate for ε > 1,
+    /// prolate for ε < 1.
+    Spheroid {
+        /// ε = a/c.
+        aspect_ratio: f64,
+    },
+}
+
+/// Options for [`TMatrix::compute`].
+#[derive(Clone, Copy, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct TMatrixOptions {
+    /// The relative tolerance of the convergence in N and n_θ (default 10⁻⁸).
+    pub rtol: f64,
+    /// Which precision paths may be used (default [`Precision::Auto`]).
+    pub precision: Precision,
+    /// A fixed number of orders N instead of the convergence loop's.
+    pub n_terms: Option<usize>,
+    /// A fixed number of quadrature nodes in 0 < θ < π/2 instead of the loop's.
+    pub n_theta: Option<usize>,
+    /// The cap on N (default: twice the starting estimate, plus 20).
+    pub n_terms_max: Option<usize>,
+    /// The cap on n_θ (default: 4 times N's starting estimate (1 + |ln ε|), plus 40).
+    pub n_theta_max: Option<usize>,
+    /// The Greek order that must settle too (default: none, only the cross sections).
+    pub greek_order: Option<usize>,
+    /// At a cap, return the last step with its estimate instead of [`Error::NotConverged`].
+    pub allow_unconverged: bool,
+    /// Compute even outside [`VALIDATED_DOMAIN`] instead of returning
+    /// [`Error::OutsideValidatedDomain`].
+    pub allow_unvalidated: bool,
+}
+
+impl Default for TMatrixOptions {
+    fn default() -> Self {
+        Self {
+            rtol: 1e-8,
+            precision: Precision::Auto,
+            n_terms: None,
+            n_theta: None,
+            n_terms_max: None,
+            n_theta_max: None,
+            greek_order: None,
+            allow_unconverged: false,
+            allow_unvalidated: false,
+        }
+    }
+}
+
+impl TMatrixOptions {
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
+    /// With the precision paths `precision`.
+    #[must_use]
+    pub fn with_precision(mut self, precision: Precision) -> Self {
+        self.precision = precision;
+        self
+    }
+
+    /// With N fixed at `n`.
+    #[must_use]
+    pub fn with_n_terms(mut self, n: usize) -> Self {
+        self.n_terms = Some(n);
+        self
+    }
+
+    /// With n_θ fixed at `n_theta`.
+    #[must_use]
+    pub fn with_n_theta(mut self, n_theta: usize) -> Self {
+        self.n_theta = Some(n_theta);
+        self
+    }
+
+    /// Requiring the Greek coefficients up to `order` to settle.
+    #[must_use]
+    pub fn with_greek_order(mut self, order: usize) -> Self {
+        self.greek_order = Some(order);
+        self
+    }
+
+    /// Allowing (or not) an unconverged result at a cap.
+    #[must_use]
+    pub fn with_allow_unconverged(mut self, allow: bool) -> Self {
+        self.allow_unconverged = allow;
+        self
+    }
+
+    /// Allowing (or not) inputs outside the validated domain.
+    #[must_use]
+    pub fn with_allow_unvalidated(mut self, allow: bool) -> Self {
+        self.allow_unvalidated = allow;
+        self
+    }
+}
+
+/// The largest N [`TMatrix::compute`] accepts, 500.
+pub const MAX_TMATRIX_ORDER: usize = 500;
+
+/// A workspace for T-matrix calculations, one per thread.
+#[derive(Debug, Default)]
+pub struct TMatrix {
+    state: Option<State>,
+}
+
+#[derive(Debug)]
+struct State {
+    x: f64,
+    eps: f64,
+    m: Complex<f64>,
+    rtol: f64,
+    converged: driver::Converged,
+    /// The Greek coefficients up to 2N (unnormalized, 1/k²), computed on first use.
+    greek: OnceLock<Result<GreekCoefficients<f64>>>,
+}
+
+impl TMatrix {
+    /// An empty workspace.
+    pub fn new() -> Self {
+        Self::default()
+    }
+
+    /// Solves the particle of `shape`, equal-volume size parameter `x` and refractive index `m`.
+    ///
+    /// # Errors
+    ///
+    /// * [`Error::InvalidInput`] if ε or x is not finite and positive, if m is not finite or
+    ///   has Re m ≤ 0 or Im m < 0, if `rtol` is not in (0, 1), or if N would exceed
+    ///   [`MAX_TMATRIX_ORDER`].
+    /// * [`Error::OutsideValidatedDomain`] outside [`VALIDATED_DOMAIN`], unless
+    ///   `allow_unvalidated` is set.
+    /// * [`Error::NotConverged`] if N or n_θ reaches its cap first, unless `allow_unconverged`
+    ///   is set.
+    pub fn compute(
+        &mut self,
+        shape: Shape,
+        x: f64,
+        m: Complex<f64>,
+        opts: &TMatrixOptions,
+    ) -> Result<TMatrixSolution<'_>> {
+        let Shape::Spheroid { aspect_ratio: eps } = shape;
+        let invalid = |s: String| Err(Error::InvalidInput(s));
+        if !(eps.is_finite() && eps > 0.0) {
+            return invalid(format!(
+                "the aspect ratio must be finite and positive, got {eps}"
+            ));
+        }
+        if !(x.is_finite() && x > 0.0) {
+            return invalid(format!("x must be finite and positive, got {x}"));
+        }
+        if !(m.re.is_finite() && m.im.is_finite() && m.re > 0.0 && m.im >= 0.0) {
+            return invalid(format!(
+                "m must be finite with Re m > 0 and Im m ≥ 0, got {m}"
+            ));
+        }
+        if !(opts.rtol > 0.0 && opts.rtol < 1.0) {
+            return invalid(format!("rtol must be in (0, 1), got {}", opts.rtol));
+        }
+        if !opts.allow_unvalidated {
+            if !VALIDATED_DOMAIN.contains(eps, x, m) {
+                return Err(Error::OutsideValidatedDomain(format!(
+                    "aspect ratio {eps}, x = {x}, m = {m}"
+                )));
+            }
+            if opts.rtol < VALIDATED_DOMAIN.rtol_min {
+                return Err(Error::OutsideValidatedDomain(format!(
+                    "rtol {:e} is below {:e}",
+                    opts.rtol, VALIDATED_DOMAIN.rtol_min
+                )));
+            }
+        }
+        let start = driver::start_n(x, eps);
+        let n_cap = opts
+            .n_terms_max
+            .unwrap_or(2 * start + 20)
+            .min(MAX_TMATRIX_ORDER);
+        let theta_cap = opts
+            .n_theta_max
+            .unwrap_or(4 * driver::theta_of(start, eps) + 40);
+        if let Some(n) = opts.n_terms {
+            if n == 0 || n > MAX_TMATRIX_ORDER {
+                return invalid(format!("N must be in 1..={MAX_TMATRIX_ORDER}, got {n}"));
+            }
+        }
+        if opts.n_theta == Some(0) {
+            return invalid("n_theta must be at least 1".into());
+        }
+        if let Some(order) = opts.greek_order {
+            if order > MAX_GREEK_ORDER {
+                return invalid(format!("the Greek order must be at most {MAX_GREEK_ORDER}"));
+            }
+        }
+        let converged = driver::converge(
+            x,
+            eps,
+            m,
+            &driver::Loop {
+                rtol: opts.rtol,
+                precision: opts.precision,
+                n_fixed: opts.n_terms,
+                n_cap,
+                n_theta_fixed: opts.n_theta,
+                n_theta_cap: theta_cap,
+                greek_order: opts.greek_order,
+                allow_unconverged: opts.allow_unconverged,
+            },
+        )?;
+        self.state = Some(State {
+            x,
+            eps,
+            m,
+            rtol: opts.rtol,
+            converged,
+            greek: OnceLock::new(),
+        });
+        Ok(TMatrixSolution {
+            s: self.state.as_ref().expect("just set"),
+        })
+    }
+}
+
+/// The T-matrix blocks of one azimuthal mode m ≥ 0, for orders n, k = max(m, 1), …, N: row-major
+/// K×K matrices, T¹¹ (M from M), T¹² (M from N), T²¹ (N from M), T²² (N from N), in the
+/// convention of [Mishchenko, Travis & Lacis 2002, §5.2] without the (−1)^m of their vector
+/// spherical wave functions. Mode −m has T¹¹ and T²² equal, and T¹² and T²¹ negated.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct ModeBlocks {
+    /// The orders n of the rows and columns.
+    pub orders: Vec<usize>,
+    /// T¹¹.
+    pub t11: Vec<Complex<f64>>,
+    /// T¹².
+    pub t12: Vec<Complex<f64>>,
+    /// T²¹.
+    pub t21: Vec<Complex<f64>>,
+    /// T²².
+    pub t22: Vec<Complex<f64>>,
+}
+
+/// Output buffers for the six independent elements of the orientation-averaged phase matrix,
+/// one value per cosine.
+#[derive(Debug)]
+pub struct PhaseBuffers<'b> {
+    /// F₁₁.
+    pub f11: &'b mut [f64],
+    /// F₁₂.
+    pub f12: &'b mut [f64],
+    /// F₂₂.
+    pub f22: &'b mut [f64],
+    /// F₃₃.
+    pub f33: &'b mut [f64],
+    /// F₃₄.
+    pub f34: &'b mut [f64],
+    /// F₄₄.
+    pub f44: &'b mut [f64],
+}
+
+/// A solved spheroid: borrowed from its [`TMatrix`] until the next computation.
+#[derive(Clone, Copy, Debug)]
+pub struct TMatrixSolution<'a> {
+    s: &'a State,
+}
+
+impl TMatrixSolution<'_> {
+    fn solved(&self) -> &driver::Solved {
+        &self.s.converged.solved
+    }
+
+    /// The number of orders N.
+    pub fn n_terms(&self) -> usize {
+        self.solved().t.n
+    }
+
+    /// The number of quadrature nodes in 0 < θ < π/2.
+    pub fn n_theta(&self) -> usize {
+        self.solved().n_theta
+    }
+
+    /// How each mode m = 0, …, N was solved.
+    pub fn precision_paths(&self) -> &[PrecisionPath] {
+        &self.solved().paths
+    }
+
+    /// Whether the loop converged to `rtol` (it always has, unless `allow_unconverged` let it
+    /// stop at a cap).
+    pub fn converged(&self) -> bool {
+        self.s.converged.converged
+    }
+
+    /// The estimated relative error of the outputs: the last change of the convergence loop
+    /// and the solve's estimate for the largest element of T, relative to it.
+    pub fn error_estimate(&self) -> f64 {
+        let solved = self.solved();
+        let big = solved
+            .t
+            .modes
+            .iter()
+            .flat_map(|(ts, _)| ts.iter().flatten())
+            .fold(0.0f64, |a, v| a.max(v.norm()));
+        let solve = if big > 0.0 { solved.error / big } else { 0.0 };
+        self.s.converged.change.max(solve)
+    }
+
+    /// The T-matrix blocks of mode `m` (`None` above N).
+    pub fn block(&self, m: usize) -> Option<ModeBlocks> {
+        let t = &self.solved().t;
+        if m > t.n {
+            return None;
+        }
+        let (ts, index) = &t.modes[m];
+        let orders: Vec<usize> = (m.max(1)..=t.n).collect();
+        let k = orders.len();
+        let mut blocks: [Vec<Complex<f64>>; 4] =
+            core::array::from_fn(|_| vec![Complex::new(0.0, 0.0); k * k]);
+        let n0 = m.max(1);
+        for c in 0..2 {
+            let kc = index[c].len();
+            for (i, &(ti, ni)) in index[c].iter().enumerate() {
+                for (j, &(tj, nj)) in index[c].iter().enumerate() {
+                    blocks[ti * 2 + tj][(ni - n0) * k + (nj - n0)] = ts[c][i * kc + j];
+                }
+            }
+        }
+        let [t11, t12, t21, t22] = blocks;
+        Some(ModeBlocks {
+            orders,
+            t11,
+            t12,
+            t21,
+            t22,
+        })
+    }
+
+    /// The raw Greek coefficients up to 2N (1/k² units, α₁ at l = 0 is ⟨C_sca⟩/(4π)).
+    fn greek_raw(&self) -> Result<&GreekCoefficients<f64>> {
+        let t = &self.solved().t;
+        self.s
+            .greek
+            .get_or_init(|| t.greek_raw(2 * t.n))
+            .as_ref()
+            .map_err(Clone::clone)
+    }
+
+    /// πx², the geometric cross section of the equal-volume sphere in 1/k².
+    fn area(&self) -> f64 {
+        core::f64::consts::PI * self.s.x * self.s.x
+    }
+
+    /// The orientation-averaged efficiencies, per unit geometric cross section of the
+    /// equal-volume sphere: Q_ext and Q_sca from T (Mishchenko 1991), Q_abs = Q_ext − Q_sca,
+    /// g = α₁(1)/3, and Q_back = 4π F₁₁(180°)/(πx²) from the Greek expansion.
+    pub fn efficiencies(&self) -> Efficiencies<f64> {
+        let (ext, sca) = self.solved().t.cross_sections();
+        let area = self.area();
+        let (g, back) = match self.greek_raw() {
+            Ok(gr) => {
+                let a0 = gr.alpha1[0];
+                let g = if a0 > 0.0 {
+                    gr.alpha1.get(1).copied().unwrap_or(0.0) / (3.0 * a0)
+                } else {
+                    0.0
+                };
+                // F₁₁(−1) = Σ α₁ₗ (−1)ˡ.
+                let f11: f64 = gr
+                    .alpha1
+                    .iter()
+                    .enumerate()
+                    .map(|(l, &v)| if l % 2 == 0 { v } else { -v })
+                    .sum();
+                (g, 4.0 * core::f64::consts::PI * f11 / area)
+            }
+            Err(_) => (f64::NAN, f64::NAN),
+        };
+        Efficiencies {
+            q_ext: ext / area,
+            q_sca: sca / area,
+            q_abs: (ext - sca) / area,
+            q_back: back,
+            g,
+        }
+    }
+
+    /// The Greek coefficients up to `order` (default 2N; zero above 2N), normalized (α₁ at
+    /// l = 0 is 1) or unnormalized (times Q_sca).
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`].
+    pub fn greek(
+        &self,
+        order: Option<usize>,
+        normalization: Normalization,
+    ) -> Result<GreekCoefficients<f64>> {
+        let n = self.n_terms();
+        let order = order.unwrap_or(2 * n);
+        if order > MAX_GREEK_ORDER {
+            return Err(Error::InvalidInput(format!(
+                "the Greek order must be at most {MAX_GREEK_ORDER}"
+            )));
+        }
+        let mut g = self.greek_raw()?.clone();
+        let len = (order + 1).min(g.alpha1.len());
+        for v in [
+            &mut g.alpha1,
+            &mut g.alpha2,
+            &mut g.alpha3,
+            &mut g.alpha4,
+            &mut g.beta1,
+            &mut g.beta2,
+        ] {
+            v.truncate(len);
+        }
+        let a0 = g.alpha1[0];
+        let scale = match normalization {
+            Normalization::Normalized => 1.0 / a0,
+            // α₁(0) = C_sca/(4π) in 1/k²: Q_sca/α₁(0) = 4π/(πx²).
+            Normalization::Unnormalized => 4.0 * core::f64::consts::PI / self.area(),
+        };
+        g.map(|v| v * scale);
+        if normalization == Normalization::Normalized {
+            g.alpha1[0] = 1.0;
+        }
+        g.pad(order + 1);
+        Ok(g)
+    }
+
+    /// The orientation-averaged phase matrix at the cosines `mu` of the scattering angle, from
+    /// the Greek expansion (exact up to 2N), normalized (½∫F₁₁ dμ = 1) or unnormalized (times
+    /// Q_sca).
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if a buffer's length differs from `mu`'s or a cosine is outside
+    /// [−1, 1].
+    pub fn phase_matrix(
+        &self,
+        mu: &[f64],
+        normalization: Normalization,
+        out: PhaseBuffers<'_>,
+    ) -> Result<()> {
+        let lens = [
+            out.f11.len(),
+            out.f12.len(),
+            out.f22.len(),
+            out.f33.len(),
+            out.f34.len(),
+            out.f44.len(),
+        ];
+        if lens.iter().any(|&l| l != mu.len()) {
+            return Err(Error::InvalidInput(format!(
+                "phase-matrix buffers must have one value per cosine ({}), got {lens:?}",
+                mu.len()
+            )));
+        }
+        let g = self.greek(None, normalization)?;
+        let len = g.alpha1.len();
+        let mut d: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; len]);
+        for (j, &c) in mu.iter().enumerate() {
+            let [d00, d22, d2m2, d02] = &mut d;
+            wigner_d(0, 0, c, d00)?;
+            wigner_d(2, 2, c, d22)?;
+            wigner_d(2, -2, c, d2m2)?;
+            wigner_d(0, 2, c, d02)?;
+            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
+            let plus: Vec<f64> = g.alpha2.iter().zip(&g.alpha3).map(|(a, b)| a + b).collect();
+            let minus: Vec<f64> = g.alpha2.iter().zip(&g.alpha3).map(|(a, b)| a - b).collect();
+            let (sum, diff) = (dot(&plus, d22), dot(&minus, d2m2));
+            out.f11[j] = dot(&g.alpha1, d00);
+            out.f44[j] = dot(&g.alpha4, d00);
+            out.f22[j] = 0.5 * (sum + diff);
+            out.f33[j] = 0.5 * (sum - diff);
+            out.f12[j] = dot(&g.beta1, d02);
+            out.f34[j] = dot(&g.beta2, d02);
+        }
+        Ok(())
+    }
+
+    /// The aspect ratio, size parameter, index and tolerance this solution was computed for.
+    pub fn inputs(&self) -> (f64, f64, Complex<f64>, f64) {
+        (self.s.eps, self.s.x, self.s.m, self.s.rtol)
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::mie::{Mie, MieOptions, PhaseBuffers as MiePhase};
+
+    fn spheroid(eps: f64) -> Shape {
+        Shape::Spheroid { aspect_ratio: eps }
+    }
+
+    /// Aspect ratio 1 gives the sphere's efficiencies, Greek coefficients and phase matrix.
+    #[test]
+    fn a_sphere_matches_mie() {
+        let (x, m) = (4.0, Complex::new(1.5, 0.01));
+        let mut tm = TMatrix::new();
+        let s = tm
+            .compute(spheroid(1.0), x, m, &TMatrixOptions::new().with_rtol(1e-10))
+            .unwrap();
+        let q = s.efficiencies();
+        let mut mie = Mie::<f64>::new();
+        let ms = mie.compute(x, m, &MieOptions::new()).unwrap();
+        let qm = ms.efficiencies();
+        for (a, b, what) in [
+            (q.q_ext, qm.q_ext, "q_ext"),
+            (q.q_sca, qm.q_sca, "q_sca"),
+            (q.q_abs, qm.q_abs, "q_abs"),
+            (q.q_back, qm.q_back, "q_back"),
+            (q.g, qm.g, "g"),
+        ] {
+            assert!((a - b).abs() <= 1e-9 * b.abs(), "{what}: {a} vs {b}");
+        }
+        let order = 2 * ms.n_terms();
+        let g = s.greek(Some(order), Normalization::Normalized).unwrap();
+        let gm = ms.greek(Some(order), Normalization::Normalized).unwrap();
+        let big = gm.alpha1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
+        for (u, v) in [
+            (&g.alpha1, &gm.alpha1),
+            (&g.alpha2, &gm.alpha2),
+            (&g.alpha3, &gm.alpha3),
+            (&g.alpha4, &gm.alpha4),
+            (&g.beta1, &gm.beta1),
+            (&g.beta2, &gm.beta2),
+        ] {
+            for (a, b) in u.iter().zip(v) {
+                assert!((a - b).abs() <= 1e-9 * big, "{a} vs {b}");
+            }
+        }
+        let mu = [-1.0, -0.3, 0.0, 0.5, 1.0];
+        let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; mu.len()]);
+        let [f11, f12, f22, f33, f34, f44] = &mut f;
+        s.phase_matrix(
+            &mu,
+            Normalization::Unnormalized,
+            PhaseBuffers {
+                f11,
+                f12,
+                f22,
+                f33,
+                f34,
+                f44,
+            },
+        )
+        .unwrap();
+        let mut fm: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; mu.len()]);
+        let [m11, m12, m33, m34] = &mut fm;
+        ms.phase_matrix(
+            &mu,
+            Normalization::Unnormalized,
+            MiePhase {
+                f11: m11,
+                f12: m12,
+                f33: m33,
+                f34: m34,
+            },
+        )
+        .unwrap();
+        let big = fm[0].iter().fold(0.0f64, |a, v| a.max(v.abs()));
+        for (a, b) in [
+            (&f[0], &fm[0]),
+            (&f[1], &fm[1]),
+            (&f[2], &fm[0]),
+            (&f[3], &fm[2]),
+            (&f[4], &fm[3]),
+            (&f[5], &fm[2]),
+        ] {
+            for (u, v) in a.iter().zip(b) {
+                assert!((u - v).abs() <= 1e-9 * big, "{u} vs {v}");
+            }
+        }
+    }
+
+    /// A lossless spheroid: Q_ext = Q_sca; unnormalized α₁(0) = Q_sca; ½∫F₁₁ dμ = 1 normalized.
+    #[test]
+    fn lossless_and_normalizations() {
+        let mut tm = TMatrix::new();
+        let s = tm
+            .compute(
+                spheroid(0.5),
+                3.0,
+                Complex::new(1.33, 0.0),
+                &TMatrixOptions::new(),
+            )
+            .unwrap();
+        let q = s.efficiencies();
+        assert!((q.q_ext - q.q_sca).abs() <= 1e-9 * q.q_ext, "{q:?}");
+        let gu = s.greek(None, Normalization::Unnormalized).unwrap();
+        assert!((gu.alpha1[0] - q.q_sca).abs() <= 1e-12 * q.q_sca);
+        let rule = crate::special::gauss_legendre::<f64>(80).unwrap();
+        let k = rule.nodes.len();
+        let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; k]);
+        let [f11, f12, f22, f33, f34, f44] = &mut f;
+        s.phase_matrix(
+            &rule.nodes,
+            Normalization::Normalized,
+            PhaseBuffers {
+                f11,
+                f12,
+                f22,
+                f33,
+                f34,
+                f44,
+            },
+        )
+        .unwrap();
+        let half: f64 = 0.5
+            * rule
+                .weights
+                .iter()
+                .zip(&f[0])
+                .map(|(w, v)| w * v)
+                .sum::<f64>();
+        assert!((half - 1.0).abs() < 1e-12, "{half}");
+        assert!(s.converged() && s.error_estimate() < 1e-8);
+    }
+
+    #[test]
+    fn invalid_inputs_and_domain() {
+        let mut tm = TMatrix::new();
+        let m = Complex::new(1.5, 0.0);
+        let o = TMatrixOptions::new();
+        for (eps, x, mm) in [
+            (0.0, 1.0, m),
+            (f64::NAN, 1.0, m),
+            (2.0, -1.0, m),
+            (2.0, 1.0, Complex::new(1.5, -0.1)),
+        ] {
+            assert!(matches!(
+                tm.compute(spheroid(eps), x, mm, &o),
+                Err(Error::InvalidInput(_))
+            ));
+        }
+        assert!(matches!(
+            tm.compute(spheroid(8.0), 1.0, m, &o),
+            Err(Error::OutsideValidatedDomain(_))
+        ));
+        assert!(
+            tm.compute(spheroid(2.0), 3.0, m, &o.with_n_theta(3).with_n_terms(4))
+                .is_ok()
+        );
+    }
+}
+
+#[cfg(test)]
+mod timing {
+    use super::*;
+
+    /// The convergence loop on harder cases: run with `--ignored --nocapture`.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn loop_costs() {
+        for &(eps, x, m) in &[
+            (2.0, 5.0, Complex::new(1.53, 0.008)),
+            (2.0, 15.0, Complex::new(1.53, 0.008)),
+            (0.5, 15.0, Complex::new(1.53, 0.008)),
+            (4.0, 10.0, Complex::new(1.33, 0.0)),
+            (0.25, 10.0, Complex::new(1.33, 0.0)),
+            (0.2, 12.0, Complex::new(1.53, 0.008)),
+            (5.0, 12.0, Complex::new(1.53, 0.008)),
+        ] {
+            let mut tm = TMatrix::new();
+            let t0 = std::time::Instant::now();
+            let r = tm.compute(
+                Shape::Spheroid { aspect_ratio: eps },
+                x,
+                m,
+                &TMatrixOptions::new().with_allow_unvalidated(true),
+            );
+            match r {
+                Ok(s) => {
+                    let refined = s
+                        .precision_paths()
+                        .iter()
+                        .filter(|p| **p != PrecisionPath::F64)
+                        .count();
+                    let q = s.efficiencies();
+                    println!(
+                        "eps {eps} x {x}: {:.2?} N {} nθ {} refined {refined} est {:.1e} Qext {:.10} Qsca {:.10}",
+                        t0.elapsed(),
+                        s.n_terms(),
+                        s.n_theta(),
+                        s.error_estimate(),
+                        q.q_ext,
+                        q.q_sca
+                    );
+                }
+                Err(e) => println!("eps {eps} x {x}: {:.2?} {e}", t0.elapsed()),
+            }
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/random.rs b/scatrs/src/tmatrix/random.rs
index 1a21bf6..f500bbe 100644
--- a/scatrs/src/tmatrix/random.rs
+++ b/scatrs/src/tmatrix/random.rs
@@ -23,7 +23,9 @@ use num_complex::Complex;
 
 use super::functions::Angular;
 use super::kernel::TMatrixModes;
+use crate::Result;
 use crate::mie::kernel::real;
+use crate::mie::{GreekCoefficients, GreekProjector};
 use crate::scalar::Real;
 use crate::special::clebsch::clebsch_gordan;
 
@@ -259,6 +261,18 @@ impl<T: Real> TMatrixModes<T> {
     }
 }
 
+impl<T: Real> TMatrixModes<T> {
+    /// The Greek coefficients of ⟨Z⟩ up to order `order` (at most 2N, above which they vanish),
+    /// unnormalized: α₁ at l = 0 is ⟨C_sca⟩/(4π) (in 1/k²). The elements are polynomials of
+    /// degree 2N in μ, so N + ⌈(L + 1)/2⌉ Gauss–Legendre nodes project them exactly.
+    pub(crate) fn greek_raw(&self, order: usize) -> Result<GreekCoefficients<T>> {
+        let order = order.min(2 * self.n);
+        let projector = GreekProjector::<T>::new(order, self.n + (order + 1).div_ceil(2))?;
+        let f = self.random_orientation(projector.nodes());
+        projector.project_six([&f.f11, &f.f12, &f.f22, &f.f33, &f.f34, &f.f44])
+    }
+}
+
 /// The phase matrix from the helicity products: ⟨S_{ij} S*_{kl}⟩ = Σ U_{ia} U⁻¹_{bj} conj(U_{kc}
 /// U⁻¹_{dl}) ⟨A_{ab} A*_{cd}⟩.
 fn mueller<T: Real>(prod: &[Complex<T>], nq: usize) -> AveragedPhase<T> {
diff --git a/scatrs/src/tmatrix/regular.rs b/scatrs/src/tmatrix/regular.rs
index e0e6135..db7b839 100644
--- a/scatrs/src/tmatrix/regular.rs
+++ b/scatrs/src/tmatrix/regular.rs
@@ -51,9 +51,11 @@ enum Kind {
 /// coefficient) and the j-part Taylor terms t_i = B̃_i ξ^{b_i}, as (exponent of the first term,
 /// values), with the prefix and suffix sums of the j-part (and of their magnitudes).
 pub(crate) struct NodeSeries<T> {
-    /// y[n][kind] = (a₀ exponent, terms Y_l).
+    /// `y[n][kind]` = (a₀ exponent, terms Y_l).
     y: Vec<[(i64, Vec<T>); 3]>,
-    /// j[k][kind] = (b₀ exponent, terms, prefix, prefix |·|, suffix, suffix |·|).
+    /// `ysuf[n][kind]` = (Σ_{l' ≥ l} Y_l', Σ_{l' ≥ l} |Y_l'|) for l = 0, …, len (0 at len).
+    ysuf: Vec<[(Vec<T>, Vec<T>); 3]>,
+    /// `j[k][kind]` = (b₀ exponent, terms, prefix, prefix |·|, suffix, suffix |·|).
     j: Vec<[JTerms<T>; 3]>,
 }
 
@@ -153,9 +155,11 @@ impl<T: Real> Series<T> {
         let inv_xmin = self.x_min.recip();
         let inv_sx = (self.s.scale(self.x_min)).inv();
         let mut y = Vec::with_capacity(n + 1);
+        let mut ysuf = Vec::with_capacity(n + 1);
         for (nn, a) in self.a.iter().enumerate() {
             if nn == 0 {
                 y.push(core::array::from_fn(|_| (0, Vec::new())));
+                ysuf.push(core::array::from_fn(|_| (Vec::new(), Vec::new())));
                 continue;
             }
             let e0 = -(nn as i64) - 1;
@@ -174,6 +178,7 @@ impl<T: Real> Series<T> {
                 over.push(v1);
                 p *= xi2;
             }
+            ysuf.push([suffix(&plain), suffix(&deriv), suffix(&over)]);
             y.push([(e0, plain), (e0 - 1, deriv), (e0 - 1, over)]);
         }
         let mut j = Vec::with_capacity(n + 1);
@@ -194,10 +199,26 @@ impl<T: Real> Series<T> {
             }
             j.push([sums(e0, &plain), sums(e0 - 1, &deriv), sums(e0 - 1, &over)]);
         }
-        NodeSeries { y, j }
+        NodeSeries { y, ysuf, j }
     }
 }
 
+/// |Re z| + |Im z|: the magnitudes of the bookkeeping (within √2 of |z|, without a square root).
+pub(crate) fn l1<T: Real>(z: Complex<T>) -> T {
+    z.re.abs() + z.im.abs()
+}
+
+fn suffix<T: Real>(terms: &[T]) -> (Vec<T>, Vec<T>) {
+    let len = terms.len();
+    let mut sum = vec![T::zero(); len + 1];
+    let mut abs = vec![T::zero(); len + 1];
+    for i in (0..len).rev() {
+        sum[i] = sum[i + 1] + terms[i];
+        abs[i] = abs[i + 1] + terms[i].abs();
+    }
+    (sum, abs)
+}
+
 fn sums<T: Real>(b0: i64, terms: &[Complex<T>]) -> JTerms<T> {
     let len = terms.len();
     let zero = real(T::zero());
@@ -206,7 +227,7 @@ fn sums<T: Real>(b0: i64, terms: &[Complex<T>]) -> JTerms<T> {
     let (mut acc, mut acc_abs) = (zero, T::zero());
     for &t in terms {
         acc = acc + t;
-        acc_abs += t.norm();
+        acc_abs += l1(t);
         head.push(acc);
         head_abs.push(acc_abs);
     }
@@ -214,7 +235,7 @@ fn sums<T: Real>(b0: i64, terms: &[Complex<T>]) -> JTerms<T> {
     let mut tail_abs = vec![T::zero(); len + 1];
     for i in (0..len).rev() {
         tail[i] = tail[i + 1] + terms[i];
-        tail_abs[i] = tail_abs[i + 1] + terms[i].norm();
+        tail_abs[i] = tail_abs[i + 1] + l1(terms[i]);
     }
     JTerms {
         b0,
@@ -240,31 +261,39 @@ impl<T: Real> NodeSeries<T> {
         let (a0, ys) = &self.y[n][kind_index(yk)];
         let js = &self.j[k][kind_index(jk)];
         let xe = if e == 2 { x_e[0] } else { x_e[1] };
-        // For each l: the largest i with a₀ + 2l + b₀ + 2i + e ≤ cut.
+        let (ysum, yabs) = &self.ysuf[n][kind_index(yk)];
+        // For each l: the largest i with a₀ + 2l + b₀ + 2i + e ≤ cut. Past the last l with any
+        // (l > l*), every j term is kept: those l add (Σ_{l>l*} Y_l)·J, from the suffix sums.
         let calc = |cut: i64| -> (Complex<T>, T, Complex<T>, T) {
             let (mut neg, mut neg_abs) = (real(T::zero()), T::zero());
             let (mut pos, mut pos_abs) = (real(T::zero()), T::zero());
             let len = js.head.len() as i64;
-            for (l, &yl) in ys.iter().enumerate() {
-                let base = a0 + 2 * l as i64 + js.b0 + e;
+            let base0 = a0 + js.b0 + e;
+            let l_end = if cut >= base0 {
+                (((cut - base0) / 2 + 1) as usize).min(ys.len())
+            } else {
+                0
+            };
+            for (l, &yl) in ys.iter().enumerate().take(l_end) {
+                let base = base0 + 2 * l as i64;
                 // p(i) = base + 2i ≤ cut ⇔ i ≤ (cut − base)/2.
-                let i_max = if cut >= base { (cut - base) / 2 } else { -1 };
+                let i_max = (cut - base) / 2;
                 let yv = yl * xe;
-                if i_max >= 0 {
-                    let i = i_max.min(len - 1) as usize;
-                    neg = neg + js.head[i].scale(yv);
-                    neg_abs += js.head_abs[i] * yv.abs();
-                }
+                let i = i_max.min(len - 1) as usize;
+                neg = neg + js.head[i].scale(yv);
+                neg_abs += js.head_abs[i] * yv.abs();
                 let from = (i_max + 1).clamp(0, len) as usize;
                 pos = pos + js.tail[from].scale(yv);
                 pos_abs += js.tail_abs[from] * yv.abs();
             }
+            pos = pos + js.tail[0].scale(ysum[l_end] * xe);
+            pos_abs += js.tail_abs[0] * yabs[l_end] * xe.abs();
             (neg, neg_abs, pos, pos_abs)
         };
         let pick = |cut: i64| -> (Complex<T>, T) {
             let (neg, neg_abs, pos, pos_abs) = calc(cut);
             // R⁺ by its series, or as R − R_{≤cut}: whichever adds less.
-            let sub_abs = full.norm() + neg_abs;
+            let sub_abs = l1(full) + neg_abs;
             if pos_abs <= sub_abs {
                 (pos, pos_abs)
             } else {
@@ -290,8 +319,8 @@ fn kind_index(k: Kind) -> usize {
     }
 }
 
-/// The eight angular factors of element (row type rt, order n; column type ct, order k) at a
-/// node, for the products of `TYPES` (with s on the first column field, and ν' = −β sin θ cos θ on
+/// The four non-zero angular factors of element (row type rt, order n; column type ct, order k)
+/// at a node, as (index into `TYPES`, factor) (with s on the first column field, and ν' = −β sin θ cos θ on
 /// the ν terms, whose x² is in e): rows (A₁, A₂) = (Mₙ, Nₙ) | (Nₙ, Mₙ), columns (B₁, B₂) =
 /// (s N_k, M_k) | (s M_k, N_k), and n̂·(A × B) with the y-part components
 /// Mₙ = (0, i yₙ pₙ, −yₙ τₙ), Nₙ = (n(n+1) (yₙ/x) dₙ, Dyₙ τₙ, i Dyₙ pₙ), pₙ = −πₙ.
@@ -305,61 +334,31 @@ pub(crate) fn angular_factors<T: Real>(
     k: usize,
     (dn, pin, tn): (T, T, T),
     (dk, pik, tk): (T, T, T),
-) -> [Complex<T>; 8] {
+) -> [(usize, Complex<T>); 4] {
     let i1 = Complex::new(T::zero(), T::one());
     let pn = -pin;
     let nn1 = int::<T>(n * (n + 1));
     let kk1 = int::<T>(k * (k + 1));
-    let zero = real(T::zero());
-    // n̂·(A × B) of each (A, B) as coefficients of the TYPES.
-    let mm = [
-        i1.scale(tn * pik - pn * tk),
-        zero,
-        zero,
-        zero,
-        zero,
-        zero,
-        zero,
-        zero,
-    ];
-    let mn = [
-        zero,
-        real(tn * tk - pn * pik),
-        zero,
-        zero,
-        real(-(nu * tn * kk1 * dk)),
-        zero,
-        zero,
-        zero,
-    ];
-    let nm = [
-        zero,
-        zero,
-        real(pn * pik - tn * tk),
-        zero,
-        zero,
-        zero,
-        real(nu * nn1 * dn * tk),
-        zero,
-    ];
-    let nnv = [
-        zero,
-        zero,
-        zero,
+    // n̂·(A × B) of each (A, B) pair as coefficients of the TYPES: (M, M) has type 0, (M, N)
+    // types 1 and 4, (N, M) types 2 and 6, (N, N) types 3, 5 and 7.
+    let mm = i1.scale(tn * pik - pn * tk);
+    let mn = (real(tn * tk - pn * pik), real(-(nu * tn * kk1 * dk)));
+    let nm = (real(pn * pik - tn * tk), real(nu * nn1 * dn * tk));
+    let nnv = (
         i1.scale(tn * pik - pn * tk),
-        zero,
         i1.scale(nu * pn * kk1 * dk),
-        zero,
         i1.scale(-(nu * nn1 * dn * pik)),
-    ];
-    let combine = |a: [Complex<T>; 8], b: [Complex<T>; 8]| -> [Complex<T>; 8] {
-        core::array::from_fn(|t| s * a[t] + b[t])
-    };
+    );
+    // Rows (A₁, A₂), columns (B₁, B₂), s on B₁: s (A₁ × B₁) + (A₂ × B₂).
     match (rt, ct) {
-        (0, 0) => combine(mn, nm),
-        (0, _) => combine(mm, nnv),
-        (_, 0) => combine(nnv, mm),
-        _ => combine(nm, mn),
+        // (M, N) with (s N, M): s (M × N) + (N × M).
+        (0, 0) => [(1, s * mn.0), (4, s * mn.1), (2, nm.0), (6, nm.1)],
+        // (M, N) with (s M, N): s (M × M) + (N × N).
+        (0, _) => [(0, s * mm), (3, nnv.0), (5, nnv.1), (7, nnv.2)],
+        // (N, M) with (s N, M): s (N × N) + (M × M).
+        (_, 0) => [(3, s * nnv.0), (5, s * nnv.1), (7, s * nnv.2), (0, mm)],
+        // (N, M) with (s M, N): s (N × M) + (M × N).
+        _ => [(2, s * nm.0), (6, s * nm.1), (1, mn.0), (4, mn.1)],
     }
 }
 
diff --git a/docs/limitations.md b/docs/limitations.md
index 94d63c5..f9d683e 100644
--- a/docs/limitations.md
+++ b/docs/limitations.md
@@ -11,7 +11,8 @@ describes what the accuracy figures rest on.
 ## Scope and availability
 
 - scatrs computes homogeneous spheres and layered spheres (a core and concentric shells),
-  single or averaged over size distributions and external mixtures.
+  single or averaged over size distributions and external mixtures, and randomly oriented
+  spheroids, single.
 - Python exposes only `f64` results. Double-double precision and the `Dual` derivatives of
   the Rust core are available from Rust only.
 - The tests run on Linux (x86_64), macOS (arm64) and Windows (x86_64) for every change. The
@@ -47,6 +48,21 @@ describes what the accuracy figures rest on.
   `allow_unvalidated` is set. Its values, and its derivatives with respect to the
   distribution's own parameters, are validated up to $10^4$.
 
+## Spheroids
+
+- **Random orientation only.** Fixed orientations, other axisymmetric shapes (cylinders,
+  Chebyshev particles) and spheroids in size distributions and mixtures are not yet
+  available.
+- **Validated domain.** Aspect ratios from 1/5 to 5, $|m|\,x_{\max} \le 60$ with $x_{\max}$ the
+  size parameter of the longest semi-axis, $1 \le \mathrm{Re}\,m \le 3$,
+  $\mathrm{Im}\,m \le 1$, and `rtol` down to $10^{-12}$ ({doc}`validation/accuracy`).
+- **Cost.** It grows like $x_{\max}^4$, and much faster where the modes need mixed-precision
+  refinement, which is most often for prolate spheroids and at the largest $|m|\,x_{\max}$.
+  At `rtol = 1e-8` an oblate spheroid with $\varepsilon = 2$ takes about 25 ms at $x = 5$ and
+  5 s at $x = 15$; $\varepsilon = 1/5$ at $x = 12$ takes minutes. `precision="f64"` skips the
+  refinement: it is fastest, and the error estimate still reports what it costs in accuracy.
+- **Derivatives** cost 4 to 13 times the values.
+
 ## Mie scattering
 
 - **Validated domain.** $10^{-6} \le x \le 2\cdot10^4$, $0 < \operatorname{Re} m \le 10$,
diff --git a/scatrs/src/tmatrix/driver.rs b/scatrs/src/tmatrix/driver.rs
index 3c0910b..7153003 100644
--- a/scatrs/src/tmatrix/driver.rs
+++ b/scatrs/src/tmatrix/driver.rs
@@ -14,7 +14,9 @@
 
 use num_complex::Complex;
 
-use super::kernel::{D4, SolveReport, TMatrixModes, tmatrix, tmatrix_derivatives, tmatrix_refined};
+use super::kernel::{
+    D4, SolveReport, TMatrixModes, tmatrix, tmatrix_derivatives, tmatrix_refined, tmatrix_upto,
+};
 use crate::scalar::{DoubleDouble as Dd, Dual};
 use crate::{Error, Result};
 
@@ -74,7 +76,7 @@ pub(crate) fn solve(
 ) -> Result<Solved> {
     let full_dd = || -> Result<Solved> {
         let sd = Complex::new(Dd::from(s.re), Dd::from(s.im));
-        let t = tmatrix(Dd::from(x), Dd::from(eps), sd, n, n_theta, true)?;
+        let t = tmatrix_upto(Dd::from(x), Dd::from(eps), sd, n, n_theta, true, m_max)?;
         let modes = t
             .modes
             .into_iter()
diff --git a/scatrs/src/tmatrix/kernel.rs b/scatrs/src/tmatrix/kernel.rs
index 29a1b10..965b012 100644
--- a/scatrs/src/tmatrix/kernel.rs
+++ b/scatrs/src/tmatrix/kernel.rs
@@ -345,17 +345,31 @@ pub(crate) fn tmatrix<T: Real>(
     n: usize,
     n_theta: usize,
     regularize: bool,
+) -> Result<TMatrixModes<T>> {
+    tmatrix_upto(x, eps, s, n, n_theta, regularize, n)
+}
+
+/// As [`tmatrix`] for the modes m ≤ `m_max` (the others are left empty).
+pub(crate) fn tmatrix_upto<T: Real>(
+    x: T,
+    eps: T,
+    s: Complex<T>,
+    n: usize,
+    n_theta: usize,
+    regularize: bool,
+    m_max: usize,
 ) -> Result<TMatrixModes<T>> {
     let shape = Spheroid::new(x, eps);
     let scales = Scales::new(shape.r_min(), s.scale(shape.r_max()), n);
     let lh = log_h(&scales);
-    let all = all_modes(&shape, s, n, n_theta, regularize, &vec![Some(0); n + 1])?;
+    let rows: Vec<Option<usize>> = (0..=n).map(|m| (m <= m_max).then_some(0)).collect();
+    let all = all_modes(&shape, s, n, n_theta, regularize, &rows)?;
     let mut modes = Vec::with_capacity(n + 1);
-    for mm in all {
+    for (m, mm) in all.into_iter().enumerate() {
         let mut ts: [Vec<Complex<T>>; 2] = [Vec::new(), Vec::new()];
         for ((slot, index), (q, p)) in ts.iter_mut().zip(&mm.index).zip(mm.q.iter().zip(&mm.p)) {
             let kc = index.len();
-            if kc == 0 {
+            if kc == 0 || m > m_max {
                 continue;
             }
             let mut t = solve_t(q, p, kc)?;
````

### Task 5: Derivatives

- [ ] `tmatrix_derivatives` in `Dual<f64, 4>` (refined modes with `Dual<DoubleDouble, 4>` rows), `derivative_f64`, `refine_derivative`, dual unscaling.
- [ ] Generic output helpers; `compute_with_derivatives`, `Derivatives`, `efficiencies_jacobian`, `greek_with_derivatives`, `phase_matrix_with_derivatives`.
- [ ] Test against central differences (kernel and public outputs).

The commit (`31f3720`):

````diff
diff --git a/scatrs/src/tmatrix/driver.rs b/scatrs/src/tmatrix/driver.rs
index cc757c6..3c0910b 100644
--- a/scatrs/src/tmatrix/driver.rs
+++ b/scatrs/src/tmatrix/driver.rs
@@ -14,8 +14,8 @@
 
 use num_complex::Complex;
 
-use super::kernel::{SolveReport, TMatrixModes, tmatrix, tmatrix_refined};
-use crate::scalar::DoubleDouble as Dd;
+use super::kernel::{D4, SolveReport, TMatrixModes, tmatrix, tmatrix_derivatives, tmatrix_refined};
+use crate::scalar::{DoubleDouble as Dd, Dual};
 use crate::{Error, Result};
 
 /// How the modes of a T-matrix were solved.
@@ -56,11 +56,7 @@ pub(crate) struct Solved {
 
 impl Solved {
     fn largest(&self) -> f64 {
-        self.t
-            .modes
-            .iter()
-            .flat_map(|(ts, _)| ts.iter().flatten())
-            .fold(0.0, |a, v| a.max(v.norm()))
+        self.t.largest()
     }
 }
 
@@ -331,3 +327,84 @@ pub(crate) fn converge(x: f64, eps: f64, s: Complex<f64>, opts: &Loop) -> Result
         converged,
     })
 }
+
+/// T with its derivatives at (N, n_θ) on the paths `precision` allows: the paths and the
+/// estimated largest error of an element of T (absolute).
+pub(crate) fn solve_derivatives(
+    x: f64,
+    eps: f64,
+    s: Complex<f64>,
+    n: usize,
+    n_theta: usize,
+    precision: Precision,
+    rtol: f64,
+) -> Result<(TMatrixModes<D4>, Vec<PrecisionPath>, f64)> {
+    let full_dd = || -> Result<(TMatrixModes<D4>, Vec<PrecisionPath>, f64)> {
+        type Dd4 = Dual<Dd, 4>;
+        let [xd, ed, sr, si] =
+            Dd4::variables([Dd::from(x), Dd::from(eps), Dd::from(s.re), Dd::from(s.im)]);
+        let t = tmatrix(xd, ed, Complex::new(sr, si), n, n_theta, true)?;
+        let round = |v: Dd4| D4::new(v.re().hi(), v.eps().map(|e| e.hi()));
+        let modes = t
+            .modes
+            .into_iter()
+            .map(|(ts, index)| {
+                let ts = ts.map(|v| {
+                    v.into_iter()
+                        .map(|c| Complex::new(round(c.re), round(c.im)))
+                        .collect()
+                });
+                (ts, index)
+            })
+            .collect();
+        let t = TMatrixModes { n, modes };
+        let big = t.values().largest();
+        Ok((
+            t,
+            vec![PrecisionPath::DoubleDouble; n + 1],
+            big * f64::EPSILON,
+        ))
+    };
+    if precision == Precision::DoubleDouble {
+        return full_dd();
+    }
+    let tol = if precision == Precision::F64 {
+        f64::INFINITY
+    } else {
+        rtol / 10.0
+    };
+    let (t, report) = tmatrix_derivatives(x, eps, s, n, n_theta, tol)?;
+    let solved = from_report(t.values(), &report, n_theta);
+    if precision == Precision::Auto && solved.error > tol * solved.largest() {
+        return full_dd();
+    }
+    Ok((t, solved.paths, solved.error))
+}
+
+impl TMatrixModes<D4> {
+    /// The values of a T with derivatives.
+    pub(crate) fn values(&self) -> TMatrixModes<f64> {
+        let modes = self
+            .modes
+            .iter()
+            .map(|(ts, index)| {
+                let ts = ts.clone().map(|v| {
+                    v.into_iter()
+                        .map(|c| Complex::new(c.re.re(), c.im.re()))
+                        .collect()
+                });
+                (ts, index.clone())
+            })
+            .collect();
+        TMatrixModes { n: self.n, modes }
+    }
+}
+
+impl TMatrixModes<f64> {
+    fn largest(&self) -> f64 {
+        self.modes
+            .iter()
+            .flat_map(|(ts, _)| ts.iter().flatten())
+            .fold(0.0, |a, v| a.max(v.norm()))
+    }
+}
diff --git a/scatrs/src/tmatrix/kernel.rs b/scatrs/src/tmatrix/kernel.rs
index 8b82cb1..29a1b10 100644
--- a/scatrs/src/tmatrix/kernel.rs
+++ b/scatrs/src/tmatrix/kernel.rs
@@ -22,10 +22,13 @@ use num_complex::Complex;
 
 use super::functions::{Angular, Radial, Scales};
 use super::regular::{Series, angular_factors, l1};
-use super::solve::{error_estimate, refine, solve_f64, solve_t};
+use super::solve::{
+    F64Solve, derivative_f64, error_estimate, refine, refine_derivative, solve_f64, solve_t,
+};
 use crate::Result;
 use crate::mie::kernel::{int, real};
 use crate::scalar::DoubleDouble as Dd;
+use crate::scalar::Dual;
 use crate::scalar::Real;
 use crate::special::gauss_legendre;
 
@@ -460,7 +463,7 @@ pub(crate) fn tmatrix_refined(
                 }
                 let flags: Vec<bool> = index.iter().map(|&(_, nn)| nn >= from).collect();
                 let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
-                let (mut t, solved, sweeps, last) = refine(
+                let refined = refine(
                     &mm.q[c],
                     &mm.p[c],
                     &exact[m].q[c],
@@ -470,6 +473,8 @@ pub(crate) fn tmatrix_refined(
                     kc,
                     4,
                 )?;
+                let (mut t, solved, sweeps, last) =
+                    (refined.t, &refined.solved, refined.sweeps, refined.last);
                 // The f64 rows' rounding remains, beside the double-double rows' and the last
                 // correction.
                 let bound = |v: &[f64]| -> Vec<f64> {
@@ -479,7 +484,7 @@ pub(crate) fn tmatrix_refined(
                         .collect()
                 };
                 let est = error_estimate(
-                    &solved,
+                    solved,
                     &t,
                     &bound(&mm.q_abs[c]),
                     &bound(&mm.p_abs[c]),
@@ -503,6 +508,220 @@ pub(crate) fn tmatrix_refined(
     Ok((TMatrixModes { n, modes }, report))
 }
 
+/// One class's T' and its four derivatives (scaled), with the factors of its solve.
+type ClassParts = ([Vec<Complex<f64>>; 5], F64Solve);
+
+/// One direction's dual numbers: x, ε, Re m, Im m.
+pub(crate) type D4 = Dual<f64, 4>;
+
+/// The cumulative logarithms of the scale ratios, as [`log_h`], in `T` (with derivatives).
+fn log_h_t<T: Real>(scales: &Scales<T>) -> Vec<T> {
+    let mut out = Vec::with_capacity(scales.h_ratio.len());
+    let mut acc = T::zero();
+    for &v in &scales.h_ratio {
+        acc += v.ln();
+        out.push(acc);
+    }
+    out
+}
+
+/// As [`tmatrix_refined`] for all modes, with the derivatives of T with respect to x, ε, Re m
+/// and Im m (the dual parts, in that order). The elements are computed in `Dual<f64, 4>`, whose
+/// values are bitwise those of `f64`; the derivatives follow from dT' = −(dP + T' dQ) Q⁻¹ with
+/// the same factors, and in the refined modes from double-double residuals as T' is refined.
+pub(crate) fn tmatrix_derivatives(
+    x: f64,
+    eps: f64,
+    s: Complex<f64>,
+    n: usize,
+    n_theta: usize,
+    tol: f64,
+) -> Result<(TMatrixModes<D4>, SolveReport)> {
+    let u = f64::EPSILON / 2.0 * (2 * n) as f64;
+    let u_dd = u * f64::EPSILON / 2.0;
+    let [xd, ed, sr, si] = D4::variables([x, eps, s.re, s.im]);
+    let sd = Complex::new(sr, si);
+    let shape = Spheroid::new(xd, ed);
+    let scales = Scales::new(shape.r_min(), sd.scale(shape.r_max()), n);
+    let lh_d = log_h_t(&scales);
+    let lh: Vec<f64> = lh_d.iter().map(|v| v.re()).collect();
+    let all = all_modes(&shape, sd, n, n_theta, true, &vec![Some(0); n + 1])?;
+    let value = |v: &[Complex<D4>]| -> Vec<Complex<f64>> {
+        v.iter()
+            .map(|c| Complex::new(c.re.re(), c.im.re()))
+            .collect()
+    };
+    let part = |v: &[Complex<D4>], p: usize| -> Vec<Complex<f64>> {
+        v.iter()
+            .map(|c| Complex::new(c.re.eps()[p], c.im.eps()[p]))
+            .collect()
+    };
+    let mut report = SolveReport {
+        error: vec![[0.0; 2]; n + 1],
+        sweeps: vec![[0; 2]; n + 1],
+    };
+    // Per mode and class: T' and its four derivatives (scaled), and the factors.
+    let mut ts: Vec<[Option<ClassParts>; 2]> = Vec::with_capacity(n + 1);
+    let mut big = 0.0f64;
+    for (m, mm) in all.iter().enumerate() {
+        let mut pair: [Option<ClassParts>; 2] = [None, None];
+        for (c, slot) in pair.iter_mut().enumerate() {
+            let index = &mm.index[c];
+            let kc = index.len();
+            if kc == 0 {
+                continue;
+            }
+            let (qv, pv) = (value(&mm.q[c]), value(&mm.p[c]));
+            let f = solve_f64(&qv, &pv, kc)?;
+            let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
+            let eq: Vec<f64> = mm.q_abs[c].iter().map(|v| u * v.re()).collect();
+            let ep: Vec<f64> = mm.p_abs[c].iter().map(|v| u * v.re()).collect();
+            report.error[m][c] = error_estimate(&f, &f.t, &eq, &ep, &lh_c, kc);
+            for (i, &(_, a)) in index.iter().enumerate() {
+                for (j, &(_, b)) in index.iter().enumerate() {
+                    big = big.max(f.t[i * kc + j].norm() * (lh[a] - lh[b]).exp());
+                }
+            }
+            let d: [Vec<Complex<f64>>; 4] = core::array::from_fn(|p| {
+                derivative_f64(&f, &f.t, &part(&mm.q[c], p), &part(&mm.p[c], p), kc)
+            });
+            let [d0, d1, d2, d3] = d;
+            *slot = Some(([f.t.clone(), d0, d1, d2, d3], f));
+        }
+        ts.push(pair);
+    }
+    let rows: Vec<Option<usize>> = (0..=n)
+        .map(|m| {
+            let worst = report.error[m][0].max(report.error[m][1]);
+            (worst > tol * big).then(|| refine_from(m, n))
+        })
+        .collect();
+    if rows.iter().any(Option::is_some) {
+        type Dd4 = Dual<Dd, 4>;
+        let [xd, ed, sr, si] =
+            Dd4::variables([Dd::from(x), Dd::from(eps), Dd::from(s.re), Dd::from(s.im)]);
+        let exact = all_modes(
+            &Spheroid::new(xd, ed),
+            Complex::new(sr, si),
+            n,
+            n_theta,
+            true,
+            &rows,
+        )?;
+        let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
+        for (m, from) in rows.iter().enumerate() {
+            let Some(from) = *from else { continue };
+            let mm = &all[m];
+            for (c, slot) in ts[m].iter_mut().enumerate() {
+                let index = &mm.index[c];
+                let kc = index.len();
+                let Some((parts, _)) = slot else {
+                    continue;
+                };
+                let flags: Vec<bool> = index.iter().map(|&(_, nn)| nn >= from).collect();
+                let lh_c: Vec<f64> = index.iter().map(|&(_, nn)| lh[nn]).collect();
+                let ex_val = |v: &[Complex<Dd4>]| -> Vec<Complex<Dd>> {
+                    v.iter()
+                        .map(|c| Complex::new(c.re.re(), c.im.re()))
+                        .collect()
+                };
+                let ex_part = |v: &[Complex<Dd4>], p: usize| -> Vec<Complex<Dd>> {
+                    v.iter()
+                        .map(|c| Complex::new(c.re.eps()[p], c.im.eps()[p]))
+                        .collect()
+                };
+                let refined = refine(
+                    &value(&mm.q[c]),
+                    &value(&mm.p[c]),
+                    &ex_val(&exact[m].q[c]),
+                    &ex_val(&exact[m].p[c]),
+                    &flags,
+                    &lh_c,
+                    kc,
+                    4,
+                )?;
+                let mixed =
+                    |exact_v: Vec<Complex<Dd>>, f64_v: Vec<Complex<f64>>| -> Vec<Complex<Dd>> {
+                        (0..kc * kc)
+                            .map(|ij| {
+                                if flags[ij / kc] {
+                                    exact_v[ij]
+                                } else {
+                                    dd(f64_v[ij])
+                                }
+                            })
+                            .collect()
+                    };
+                for p in 0..4 {
+                    let dq = mixed(ex_part(&exact[m].q[c], p), part(&mm.q[c], p));
+                    let dp = mixed(ex_part(&exact[m].p[c], p), part(&mm.p[c], p));
+                    parts[p + 1] = refine_derivative(&refined, &dq, &dp, kc, 3);
+                }
+                let bound = |v: &[D4]| -> Vec<f64> {
+                    v.iter()
+                        .enumerate()
+                        .map(|(ij, a)| a.re() * if flags[ij / kc] { u_dd } else { u })
+                        .collect()
+                };
+                let est = error_estimate(
+                    &refined.solved,
+                    &refined.t,
+                    &bound(&mm.q_abs[c]),
+                    &bound(&mm.p_abs[c]),
+                    &lh_c,
+                    kc,
+                );
+                let largest = refined
+                    .t
+                    .iter()
+                    .zip(
+                        index
+                            .iter()
+                            .flat_map(|&(_, a)| index.iter().map(move |&(_, b)| (a, b))),
+                    )
+                    .fold(0.0f64, |acc, (v, (a, b))| {
+                        acc.max(v.norm() * (lh[a] - lh[b]).exp())
+                    });
+                report.sweeps[m][c] = refined.sweeps;
+                report.error[m][c] = est + refined.last + largest * f64::EPSILON;
+                parts[0] = refined.t;
+            }
+        }
+    }
+    // Assemble the dual T' and unscale it with the dual factors σ_k/σₙ.
+    let mut modes = Vec::with_capacity(n + 1);
+    for (pair, mm) in ts.into_iter().zip(all) {
+        let mut out: [Vec<Complex<D4>>; 2] = [Vec::new(), Vec::new()];
+        for (c, slot) in pair.into_iter().enumerate() {
+            let Some((parts, _)) = slot else { continue };
+            let index = &mm.index[c];
+            let kc = index.len();
+            let mut t: Vec<Complex<D4>> = (0..kc * kc)
+                .map(|ij| {
+                    let re = D4::new(
+                        parts[0][ij].re,
+                        core::array::from_fn(|p| parts[p + 1][ij].re),
+                    );
+                    let im = D4::new(
+                        parts[0][ij].im,
+                        core::array::from_fn(|p| parts[p + 1][ij].im),
+                    );
+                    Complex::new(re, im)
+                })
+                .collect();
+            for (i, &(_, a)) in index.iter().enumerate() {
+                for (j, &(_, b)) in index.iter().enumerate() {
+                    let f = num_traits::Float::exp(lh_d[a] - lh_d[b]);
+                    t[i * kc + j] = t[i * kc + j].scale(f);
+                }
+            }
+            out[c] = t;
+        }
+        modes.push((out, mm.index));
+    }
+    Ok((TMatrixModes { n, modes }, report))
+}
+
 impl<T: Real> TMatrixModes<T> {
     /// The orientation-averaged ⟨C_ext⟩ and ⟨C_sca⟩ in units of 1/k² (Mishchenko 1991):
     /// −2π Σ (2 − δ_{m0}) Re(T¹¹ₙₙ + T²²ₙₙ) and 2π Σ (2 − δ_{m0}) |T|².
@@ -532,6 +751,64 @@ impl<T: Real> TMatrixModes<T> {
 mod tests {
     use super::*;
 
+    /// The dual derivatives of ⟨C_ext⟩, ⟨C_sca⟩ and the Greek coefficients match central
+    /// differences of the refined values, with and without refinement; the values are those of
+    /// [`tmatrix_refined`].
+    #[test]
+    fn derivatives_match_differences() {
+        let (x, eps, n, nth) = (6.0, 3.0, 20, 40);
+        let s = Complex::new(1.5, 0.02);
+        for tol in [1.0, 0.0] {
+            let (d, rep) = tmatrix_derivatives(x, eps, s, n, nth, tol).unwrap();
+            assert_eq!(rep.sweeps.iter().flatten().any(|&k| k > 0), tol == 0.0);
+            let (ext, sca) = d.cross_sections();
+            let g = d.greek_raw(12).unwrap();
+            let at = |x: f64, eps: f64, s: Complex<f64>| {
+                let (t, _) = tmatrix_refined(x, eps, s, n, nth, 0.0, n).unwrap();
+                let (e, c) = t.cross_sections();
+                (e, c, t.greek_raw(12).unwrap())
+            };
+            let (e0, c0, g0) = at(x, eps, s);
+            assert!((ext.re() - e0).abs() < 1e-13 * e0 && (sca.re() - c0).abs() < 1e-13 * c0);
+            let h = 1e-5;
+            let steps = [
+                (h * x, 0.0, Complex::new(0.0, 0.0)),
+                (0.0, h * eps, Complex::new(0.0, 0.0)),
+                (0.0, 0.0, Complex::new(h, 0.0)),
+                (0.0, 0.0, Complex::new(0.0, h)),
+            ];
+            for (p, &(dx, de, ds)) in steps.iter().enumerate() {
+                let (ep, cp, gp) = at(x + dx, eps + de, s + ds);
+                let (em, cm, gm) = at(x - dx, eps - de, s - ds);
+                let step = dx + de + ds.re + ds.im;
+                let fd = |a: f64, b: f64| (a - b) / (2.0 * step);
+                // Measured against the natural size of a derivative, value/|parameter|.
+                let param = [x, eps, s.norm(), s.norm()][p];
+                let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-7 * scale / param;
+                assert!(
+                    close(ext.eps()[p], fd(ep, em), e0),
+                    "ext p {p}: {} vs {}",
+                    ext.eps()[p],
+                    fd(ep, em)
+                );
+                assert!(close(sca.eps()[p], fd(cp, cm), c0), "sca p {p}");
+                let big = g0.alpha1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
+                for l in 0..g.alpha1.len() {
+                    for (u, a, b) in [
+                        (&g.alpha1, &gp.alpha1, &gm.alpha1),
+                        (&g.alpha2, &gp.alpha2, &gm.alpha2),
+                        (&g.beta2, &gp.beta2, &gm.beta2),
+                    ] {
+                        assert!(
+                            close(u[l].eps()[p], fd(a[l], b[l]), big),
+                            "greek l {l} p {p}"
+                        );
+                    }
+                }
+            }
+        }
+    }
+
     /// The refined path matches full double-double, its error estimate bounds its error, and the
     /// f64 path's estimate bounds the f64 error.
     #[test]
@@ -999,3 +1276,33 @@ mod profile {
         println!("f64 regularized mode 0 only: {:.2?}", t0.elapsed());
     }
 }
+
+#[cfg(test)]
+mod derivative_cost {
+    use super::*;
+
+    /// The cost of the derivatives against the values: run with `--ignored --nocapture`.
+    #[test]
+    #[ignore = "a diagnostic, not a check"]
+    fn ratio() {
+        let s = Complex::new(1.53, 0.008);
+        for &(x, eps, n, nth, tol) in &[
+            (5.0, 2.0, 18, 48, 1e-9),
+            (10.0, 2.0, 30, 70, 1e-9),
+            (10.0, 2.0, 30, 70, f64::INFINITY),
+            (15.0, 2.0, 45, 106, 1e-9),
+        ] {
+            let t0 = std::time::Instant::now();
+            let (_, r) = tmatrix_refined(x, eps, s, n, nth, tol, n).unwrap();
+            let tv = t0.elapsed();
+            let t0 = std::time::Instant::now();
+            tmatrix_derivatives(x, eps, s, n, nth, tol).unwrap();
+            let td = t0.elapsed();
+            let refined = r.sweeps.iter().filter(|s| s[0] + s[1] > 0).count();
+            println!(
+                "x {x} eps {eps} N {n} tol {tol:e}: values {tv:.2?}, with derivatives {td:.2?} ({:.1}x), {refined} refined",
+                td.as_secs_f64() / tv.as_secs_f64()
+            );
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
index 5cf2b1f..6417388 100644
--- a/scatrs/src/tmatrix/mod.rs
+++ b/scatrs/src/tmatrix/mod.rs
@@ -60,6 +60,9 @@ use num_complex::Complex;
 pub use domain::{VALIDATED_DOMAIN, ValidatedDomain};
 pub use driver::{Precision, PrecisionPath};
 
+use crate::scalar::Real;
+use kernel::{D4, TMatrixModes};
+
 use crate::mie::{Efficiencies, GreekCoefficients, MAX_GREEK_ORDER, Normalization};
 use crate::special::wigner_d;
 use crate::{Error, Result};
@@ -182,6 +185,9 @@ pub struct TMatrix {
     state: Option<State>,
 }
 
+/// T with its derivatives, and their Greek coefficients on first use.
+type WithDerivatives = (TMatrixModes<D4>, OnceLock<Result<GreekCoefficients<D4>>>);
+
 #[derive(Debug)]
 struct State {
     x: f64,
@@ -191,6 +197,8 @@ struct State {
     converged: driver::Converged,
     /// The Greek coefficients up to 2N (unnormalized, 1/k²), computed on first use.
     greek: OnceLock<Result<GreekCoefficients<f64>>>,
+    /// T with its derivatives, if asked for, and their Greek coefficients on first use.
+    derivatives: Option<WithDerivatives>,
 }
 
 impl TMatrix {
@@ -216,6 +224,35 @@ impl TMatrix {
         x: f64,
         m: Complex<f64>,
         opts: &TMatrixOptions,
+    ) -> Result<TMatrixSolution<'_>> {
+        self.run(shape, x, m, opts, false)
+    }
+
+    /// As [`compute`](Self::compute), also computing the derivatives of every output with
+    /// respect to x, the aspect ratio, Re m and Im m, at the N and n_θ the values converged at:
+    /// the elements in dual numbers (their values are those of `f64`), and
+    /// dT = −(dP + T dQ) Q⁻¹ with the factors of the solve, refined as T is.
+    ///
+    /// # Errors
+    ///
+    /// As [`compute`](Self::compute).
+    pub fn compute_with_derivatives(
+        &mut self,
+        shape: Shape,
+        x: f64,
+        m: Complex<f64>,
+        opts: &TMatrixOptions,
+    ) -> Result<TMatrixSolution<'_>> {
+        self.run(shape, x, m, opts, true)
+    }
+
+    fn run(
+        &mut self,
+        shape: Shape,
+        x: f64,
+        m: Complex<f64>,
+        opts: &TMatrixOptions,
+        with_derivatives: bool,
     ) -> Result<TMatrixSolution<'_>> {
         let Shape::Spheroid { aspect_ratio: eps } = shape;
         let invalid = |s: String| Err(Error::InvalidInput(s));
@@ -284,6 +321,19 @@ impl TMatrix {
                 allow_unconverged: opts.allow_unconverged,
             },
         )?;
+        let mut converged = converged;
+        let derivatives = if with_derivatives {
+            let solved = &mut converged.solved;
+            let (n, nth) = (solved.t.n, solved.n_theta);
+            let d = driver::solve_derivatives(x, eps, m, n, nth, opts.precision, opts.rtol)?;
+            // The values are the dual T's, so that they match the derivatives.
+            solved.t = d.0.values();
+            solved.paths = d.1;
+            solved.error = d.2;
+            Some((d.0, OnceLock::new()))
+        } else {
+            None
+        };
         self.state = Some(State {
             x,
             eps,
@@ -291,6 +341,7 @@ impl TMatrix {
             rtol: opts.rtol,
             converged,
             greek: OnceLock::new(),
+            derivatives,
         });
         Ok(TMatrixSolution {
             s: self.state.as_ref().expect("just set"),
@@ -341,6 +392,167 @@ pub struct TMatrixSolution<'a> {
     s: &'a State,
 }
 
+/// The derivatives of an output with respect to the spheroid's parameters.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct Derivatives<V> {
+    /// ∂/∂x (equal-volume size parameter).
+    pub x: V,
+    /// ∂/∂ε (aspect ratio).
+    pub aspect_ratio: V,
+    /// ∂/∂(Re m).
+    pub m_re: V,
+    /// ∂/∂(Im m).
+    pub m_im: V,
+}
+
+impl<V> Derivatives<V> {
+    fn from_fn(mut f: impl FnMut(usize) -> V) -> Self {
+        Self {
+            x: f(0),
+            aspect_ratio: f(1),
+            m_re: f(2),
+            m_im: f(3),
+        }
+    }
+}
+
+/// The efficiencies from T, its raw Greek coefficients and x (Q = C/(πx²) in 1/k² units).
+fn efficiencies_of<T: Real>(
+    t: &kernel::TMatrixModes<T>,
+    raw: &GreekCoefficients<T>,
+    x: T,
+) -> Efficiencies<T> {
+    let (ext, sca) = t.cross_sections();
+    let area = T::PI() * x * x;
+    let a0 = raw.alpha1[0];
+    let g = if a0 > T::zero() {
+        raw.alpha1.get(1).copied().unwrap_or(T::zero()) / (T::from_f64(3.0) * a0)
+    } else {
+        T::zero()
+    };
+    // F₁₁(−1) = Σ α₁ₗ (−1)ˡ.
+    let mut f11 = T::zero();
+    for (l, &v) in raw.alpha1.iter().enumerate() {
+        f11 = if l % 2 == 0 { f11 + v } else { f11 - v };
+    }
+    Efficiencies {
+        q_ext: ext / area,
+        q_sca: sca / area,
+        q_abs: (ext - sca) / area,
+        q_back: T::from_f64(4.0) * T::PI() * f11 / area,
+        g,
+    }
+}
+
+/// The Greek coefficients up to `order` from the raw ones, normalized or unnormalized.
+fn greek_of<T: Real>(
+    raw: &GreekCoefficients<T>,
+    x: T,
+    order: usize,
+    normalization: Normalization,
+) -> GreekCoefficients<T> {
+    let mut g = raw.clone();
+    let len = (order + 1).min(g.alpha1.len());
+    for v in [
+        &mut g.alpha1,
+        &mut g.alpha2,
+        &mut g.alpha3,
+        &mut g.alpha4,
+        &mut g.beta1,
+        &mut g.beta2,
+    ] {
+        v.truncate(len);
+    }
+    let scale = match normalization {
+        Normalization::Normalized => g.alpha1[0].recip(),
+        // α₁(0) = C_sca/(4π) in 1/k²: Q_sca/α₁(0) = 4π/(πx²).
+        Normalization::Unnormalized => T::from_f64(4.0) / (x * x),
+    };
+    g.map(|v| v * scale);
+    g.pad(order + 1);
+    g
+}
+
+/// F₁₁, F₁₂, F₂₂, F₃₃, F₃₄, F₄₄ at cosine `mu` from Greek coefficients.
+fn phase_of<T: Real>(g: &GreekCoefficients<T>, d: &[Vec<f64>; 4]) -> [T; 6] {
+    let [d00, d22, d2m2, d02] = d;
+    let dot = |a: &[T], b: &[f64]| {
+        a.iter()
+            .zip(b)
+            .fold(T::zero(), |acc, (&x, &y)| acc + x * T::from_f64(y))
+    };
+    let mut sum = T::zero();
+    let mut diff = T::zero();
+    for l in 0..g.alpha2.len() {
+        sum += (g.alpha2[l] + g.alpha3[l]) * T::from_f64(d22[l]);
+        diff += (g.alpha2[l] - g.alpha3[l]) * T::from_f64(d2m2[l]);
+    }
+    let half = T::from_f64(0.5);
+    [
+        dot(&g.alpha1, d00),
+        dot(&g.beta1, d02),
+        half * (sum + diff),
+        half * (sum - diff),
+        dot(&g.beta2, d02),
+        dot(&g.alpha4, d00),
+    ]
+}
+
+fn wigner_tables(mu: f64, len: usize, d: &mut [Vec<f64>; 4]) -> Result<()> {
+    for v in d.iter_mut() {
+        v.resize(len, 0.0);
+    }
+    let [d00, d22, d2m2, d02] = d;
+    wigner_d(0, 0, mu, d00)?;
+    wigner_d(2, 2, mu, d22)?;
+    wigner_d(2, -2, mu, d2m2)?;
+    wigner_d(0, 2, mu, d02)?;
+    Ok(())
+}
+
+fn check_buffers(mu: &[f64], out: &PhaseBuffers<'_>) -> Result<()> {
+    let lens = [
+        out.f11.len(),
+        out.f12.len(),
+        out.f22.len(),
+        out.f33.len(),
+        out.f34.len(),
+        out.f44.len(),
+    ];
+    if lens.iter().any(|&l| l != mu.len()) {
+        return Err(Error::InvalidInput(format!(
+            "phase-matrix buffers must have one value per cosine ({}), got {lens:?}",
+            mu.len()
+        )));
+    }
+    Ok(())
+}
+
+fn store(out: &mut PhaseBuffers<'_>, j: usize, v: [f64; 6]) {
+    out.f11[j] = v[0];
+    out.f12[j] = v[1];
+    out.f22[j] = v[2];
+    out.f33[j] = v[3];
+    out.f34[j] = v[4];
+    out.f44[j] = v[5];
+}
+
+fn no_derivatives() -> Error {
+    Error::InvalidInput(
+        "the solution was computed without derivatives (use compute_with_derivatives)".into(),
+    )
+}
+
+fn check_order(order: usize) -> Result<()> {
+    if order > MAX_GREEK_ORDER {
+        return Err(Error::InvalidInput(format!(
+            "the Greek order must be at most {MAX_GREEK_ORDER}"
+        )));
+    }
+    Ok(())
+}
+
 impl TMatrixSolution<'_> {
     fn solved(&self) -> &driver::Solved {
         &self.s.converged.solved
@@ -381,6 +593,11 @@ impl TMatrixSolution<'_> {
         self.s.converged.change.max(solve)
     }
 
+    /// The aspect ratio, size parameter, index and tolerance this solution was computed for.
+    pub fn inputs(&self) -> (f64, f64, Complex<f64>, f64) {
+        (self.s.eps, self.s.x, self.s.m, self.s.rtol)
+    }
+
     /// The T-matrix blocks of mode `m` (`None` above N).
     pub fn block(&self, m: usize) -> Option<ModeBlocks> {
         let t = &self.solved().t;
@@ -421,45 +638,52 @@ impl TMatrixSolution<'_> {
             .map_err(Clone::clone)
     }
 
-    /// πx², the geometric cross section of the equal-volume sphere in 1/k².
-    fn area(&self) -> f64 {
-        core::f64::consts::PI * self.s.x * self.s.x
+    /// The same with derivatives.
+    fn greek_raw_dual(&self) -> Result<&GreekCoefficients<D4>> {
+        let (t, cell) = self.s.derivatives.as_ref().ok_or_else(no_derivatives)?;
+        cell.get_or_init(|| t.greek_raw(2 * t.n))
+            .as_ref()
+            .map_err(Clone::clone)
+    }
+
+    fn x_dual(&self) -> D4 {
+        D4::new(self.s.x, [1.0, 0.0, 0.0, 0.0])
     }
 
     /// The orientation-averaged efficiencies, per unit geometric cross section of the
     /// equal-volume sphere: Q_ext and Q_sca from T (Mishchenko 1991), Q_abs = Q_ext − Q_sca,
     /// g = α₁(1)/3, and Q_back = 4π F₁₁(180°)/(πx²) from the Greek expansion.
     pub fn efficiencies(&self) -> Efficiencies<f64> {
-        let (ext, sca) = self.solved().t.cross_sections();
-        let area = self.area();
-        let (g, back) = match self.greek_raw() {
-            Ok(gr) => {
-                let a0 = gr.alpha1[0];
-                let g = if a0 > 0.0 {
-                    gr.alpha1.get(1).copied().unwrap_or(0.0) / (3.0 * a0)
-                } else {
-                    0.0
-                };
-                // F₁₁(−1) = Σ α₁ₗ (−1)ˡ.
-                let f11: f64 = gr
-                    .alpha1
-                    .iter()
-                    .enumerate()
-                    .map(|(l, &v)| if l % 2 == 0 { v } else { -v })
-                    .sum();
-                (g, 4.0 * core::f64::consts::PI * f11 / area)
+        match self.greek_raw() {
+            Ok(raw) => efficiencies_of(&self.solved().t, raw, self.s.x),
+            Err(_) => {
+                let (ext, sca) = self.solved().t.cross_sections();
+                let area = core::f64::consts::PI * self.s.x * self.s.x;
+                Efficiencies {
+                    q_ext: ext / area,
+                    q_sca: sca / area,
+                    q_abs: (ext - sca) / area,
+                    q_back: f64::NAN,
+                    g: f64::NAN,
+                }
             }
-            Err(_) => (f64::NAN, f64::NAN),
-        };
-        Efficiencies {
-            q_ext: ext / area,
-            q_sca: sca / area,
-            q_abs: (ext - sca) / area,
-            q_back: back,
-            g,
         }
     }
 
+    /// The derivatives of the efficiencies (`None` without derivatives).
+    pub fn efficiencies_jacobian(&self) -> Option<Derivatives<Efficiencies<f64>>> {
+        let (t, _) = self.s.derivatives.as_ref()?;
+        let raw = self.greek_raw_dual().ok()?;
+        let q = efficiencies_of(t, raw, self.x_dual());
+        Some(Derivatives::from_fn(|p| Efficiencies {
+            q_ext: q.q_ext.eps()[p],
+            q_sca: q.q_sca.eps()[p],
+            q_abs: q.q_abs.eps()[p],
+            q_back: q.q_back.eps()[p],
+            g: q.g.eps()[p],
+        }))
+    }
+
     /// The Greek coefficients up to `order` (default 2N; zero above 2N), normalized (α₁ at
     /// l = 0 is 1) or unnormalized (times Q_sca).
     ///
@@ -471,39 +695,55 @@ impl TMatrixSolution<'_> {
         order: Option<usize>,
         normalization: Normalization,
     ) -> Result<GreekCoefficients<f64>> {
-        let n = self.n_terms();
-        let order = order.unwrap_or(2 * n);
-        if order > MAX_GREEK_ORDER {
-            return Err(Error::InvalidInput(format!(
-                "the Greek order must be at most {MAX_GREEK_ORDER}"
-            )));
-        }
-        let mut g = self.greek_raw()?.clone();
-        let len = (order + 1).min(g.alpha1.len());
-        for v in [
-            &mut g.alpha1,
-            &mut g.alpha2,
-            &mut g.alpha3,
-            &mut g.alpha4,
-            &mut g.beta1,
-            &mut g.beta2,
-        ] {
-            v.truncate(len);
-        }
-        let a0 = g.alpha1[0];
-        let scale = match normalization {
-            Normalization::Normalized => 1.0 / a0,
-            // α₁(0) = C_sca/(4π) in 1/k²: Q_sca/α₁(0) = 4π/(πx²).
-            Normalization::Unnormalized => 4.0 * core::f64::consts::PI / self.area(),
-        };
-        g.map(|v| v * scale);
+        let order = order.unwrap_or(2 * self.n_terms());
+        check_order(order)?;
+        let mut g = greek_of(self.greek_raw()?, self.s.x, order, normalization);
         if normalization == Normalization::Normalized {
             g.alpha1[0] = 1.0;
         }
-        g.pad(order + 1);
         Ok(g)
     }
 
+    /// The Greek coefficients and their derivatives.
+    ///
+    /// # Errors
+    ///
+    /// As [`greek`](Self::greek), and [`Error::InvalidInput`] without derivatives.
+    pub fn greek_with_derivatives(
+        &self,
+        order: Option<usize>,
+        normalization: Normalization,
+    ) -> Result<(GreekCoefficients<f64>, Derivatives<GreekCoefficients<f64>>)> {
+        let order = order.unwrap_or(2 * self.n_terms());
+        check_order(order)?;
+        let g = greek_of(self.greek_raw_dual()?, self.x_dual(), order, normalization);
+        let split = |f: &dyn Fn(D4) -> f64| {
+            let mut out = GreekCoefficients::zeros(order + 1);
+            for (o, i) in [
+                (&mut out.alpha1, &g.alpha1),
+                (&mut out.alpha2, &g.alpha2),
+                (&mut out.alpha3, &g.alpha3),
+                (&mut out.alpha4, &g.alpha4),
+                (&mut out.beta1, &g.beta1),
+                (&mut out.beta2, &g.beta2),
+            ] {
+                for (a, &b) in o.iter_mut().zip(i) {
+                    *a = f(b);
+                }
+            }
+            out
+        };
+        let mut values = split(&|v| v.re());
+        let mut d = Derivatives::from_fn(|p| split(&|v| v.eps()[p]));
+        if normalization == Normalization::Normalized {
+            values.alpha1[0] = 1.0;
+            for v in [&mut d.x, &mut d.aspect_ratio, &mut d.m_re, &mut d.m_im] {
+                v.alpha1[0] = 0.0;
+            }
+        }
+        Ok((values, d))
+    }
+
     /// The orientation-averaged phase matrix at the cosines `mu` of the scattering angle, from
     /// the Greek expansion (exact up to 2N), normalized (½∫F₁₁ dμ = 1) or unnormalized (times
     /// Q_sca).
@@ -516,48 +756,53 @@ impl TMatrixSolution<'_> {
         &self,
         mu: &[f64],
         normalization: Normalization,
-        out: PhaseBuffers<'_>,
+        mut out: PhaseBuffers<'_>,
     ) -> Result<()> {
-        let lens = [
-            out.f11.len(),
-            out.f12.len(),
-            out.f22.len(),
-            out.f33.len(),
-            out.f34.len(),
-            out.f44.len(),
-        ];
-        if lens.iter().any(|&l| l != mu.len()) {
-            return Err(Error::InvalidInput(format!(
-                "phase-matrix buffers must have one value per cosine ({}), got {lens:?}",
-                mu.len()
-            )));
-        }
+        check_buffers(mu, &out)?;
         let g = self.greek(None, normalization)?;
-        let len = g.alpha1.len();
-        let mut d: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; len]);
+        let mut d: [Vec<f64>; 4] = Default::default();
         for (j, &c) in mu.iter().enumerate() {
-            let [d00, d22, d2m2, d02] = &mut d;
-            wigner_d(0, 0, c, d00)?;
-            wigner_d(2, 2, c, d22)?;
-            wigner_d(2, -2, c, d2m2)?;
-            wigner_d(0, 2, c, d02)?;
-            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
-            let plus: Vec<f64> = g.alpha2.iter().zip(&g.alpha3).map(|(a, b)| a + b).collect();
-            let minus: Vec<f64> = g.alpha2.iter().zip(&g.alpha3).map(|(a, b)| a - b).collect();
-            let (sum, diff) = (dot(&plus, d22), dot(&minus, d2m2));
-            out.f11[j] = dot(&g.alpha1, d00);
-            out.f44[j] = dot(&g.alpha4, d00);
-            out.f22[j] = 0.5 * (sum + diff);
-            out.f33[j] = 0.5 * (sum - diff);
-            out.f12[j] = dot(&g.beta1, d02);
-            out.f34[j] = dot(&g.beta2, d02);
+            wigner_tables(c, g.alpha1.len(), &mut d)?;
+            store(&mut out, j, phase_of(&g, &d));
         }
         Ok(())
     }
 
-    /// The aspect ratio, size parameter, index and tolerance this solution was computed for.
-    pub fn inputs(&self) -> (f64, f64, Complex<f64>, f64) {
-        (self.s.eps, self.s.x, self.s.m, self.s.rtol)
+    /// The phase matrix and its derivatives, into `out` and `d`.
+    ///
+    /// # Errors
+    ///
+    /// As [`phase_matrix`](Self::phase_matrix), and [`Error::InvalidInput`] without
+    /// derivatives.
+    pub fn phase_matrix_with_derivatives(
+        &self,
+        mu: &[f64],
+        normalization: Normalization,
+        mut out: PhaseBuffers<'_>,
+        d: Derivatives<PhaseBuffers<'_>>,
+    ) -> Result<()> {
+        check_buffers(mu, &out)?;
+        let Derivatives {
+            x: mut dx,
+            aspect_ratio: mut de,
+            m_re: mut dr,
+            m_im: mut di,
+        } = d;
+        for b in [&dx, &de, &dr, &di] {
+            check_buffers(mu, b)?;
+        }
+        let n = self.n_terms();
+        let g = greek_of(self.greek_raw_dual()?, self.x_dual(), 2 * n, normalization);
+        let mut w: [Vec<f64>; 4] = Default::default();
+        for (j, &c) in mu.iter().enumerate() {
+            wigner_tables(c, g.alpha1.len(), &mut w)?;
+            let v = phase_of(&g, &w);
+            store(&mut out, j, v.map(|x| x.re()));
+            for (p, b) in [&mut dx, &mut de, &mut dr, &mut di].into_iter().enumerate() {
+                store(b, j, v.map(|x| x.eps()[p]));
+            }
+        }
+        Ok(())
     }
 }
 
@@ -695,6 +940,126 @@ mod tests {
         assert!(s.converged() && s.error_estimate() < 1e-8);
     }
 
+    /// The derivatives of the efficiencies, Greek coefficients and phase matrix against central
+    /// differences at fixed N and n_θ; the values equal `compute`'s.
+    #[test]
+    fn derivatives_against_differences() {
+        let (eps, x, m) = (0.6, 3.0, Complex::new(1.45, 0.03));
+        let base = TMatrixOptions::new().with_rtol(1e-10);
+        let mut tm = TMatrix::new();
+        let s = tm.compute(spheroid(eps), x, m, &base).unwrap();
+        let (n, nth) = (s.n_terms(), s.n_theta());
+        let q0 = s.efficiencies();
+        let fixed = base.with_n_terms(n).with_n_theta(nth);
+        let mut td = TMatrix::new();
+        let d = td
+            .compute_with_derivatives(spheroid(eps), x, m, &fixed)
+            .unwrap();
+        let qd = d.efficiencies();
+        assert!((qd.q_ext - q0.q_ext).abs() <= 1e-14 * q0.q_ext);
+        let jac = d.efficiencies_jacobian().unwrap();
+        let (gv, gd) = d
+            .greek_with_derivatives(Some(8), Normalization::Unnormalized)
+            .unwrap();
+        let mu = [-1.0, -0.2, 0.7];
+        let mk = || -> [Vec<f64>; 6] { core::array::from_fn(|_| vec![0.0; mu.len()]) };
+        let (mut f, mut fx, mut fe, mut fr, mut fi) = (mk(), mk(), mk(), mk(), mk());
+        fn buf(b: &mut [Vec<f64>; 6]) -> PhaseBuffers<'_> {
+            let [f11, f12, f22, f33, f34, f44] = b;
+            PhaseBuffers {
+                f11,
+                f12,
+                f22,
+                f33,
+                f34,
+                f44,
+            }
+        }
+        d.phase_matrix_with_derivatives(
+            &mu,
+            Normalization::Normalized,
+            buf(&mut f),
+            Derivatives {
+                x: buf(&mut fx),
+                aspect_ratio: buf(&mut fe),
+                m_re: buf(&mut fr),
+                m_im: buf(&mut fi),
+            },
+        )
+        .unwrap();
+        let h = 1e-5;
+        let eval = |eps: f64, x: f64, m: Complex<f64>| {
+            let mut t = TMatrix::new();
+            let s = t.compute(spheroid(eps), x, m, &fixed).unwrap();
+            let mut p = mk();
+            s.phase_matrix(&mu, Normalization::Normalized, buf(&mut p))
+                .unwrap();
+            (
+                s.efficiencies(),
+                s.greek(Some(8), Normalization::Unnormalized).unwrap(),
+                p,
+            )
+        };
+        let dirs = [
+            (0.0, h * x, Complex::new(0.0, 0.0), &jac.x, &gd.x, &fx, x),
+            (
+                h * eps,
+                0.0,
+                Complex::new(0.0, 0.0),
+                &jac.aspect_ratio,
+                &gd.aspect_ratio,
+                &fe,
+                eps,
+            ),
+            (
+                0.0,
+                0.0,
+                Complex::new(h, 0.0),
+                &jac.m_re,
+                &gd.m_re,
+                &fr,
+                m.norm(),
+            ),
+            (
+                0.0,
+                0.0,
+                Complex::new(0.0, h),
+                &jac.m_im,
+                &gd.m_im,
+                &fi,
+                m.norm(),
+            ),
+        ];
+        for (de, dx, dm, jq, jg, jf, size) in dirs {
+            let step = de + dx + dm.re + dm.im;
+            let (qp, gp, pp) = eval(eps + de, x + dx, m + dm);
+            let (qm, gm, pm) = eval(eps - de, x - dx, m - dm);
+            let fd = |a: f64, b: f64| (a - b) / (2.0 * step);
+            let ok = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-6 * scale / size;
+            for (a, p, mm, v) in [
+                (jq.q_ext, qp.q_ext, qm.q_ext, q0.q_ext),
+                (jq.q_sca, qp.q_sca, qm.q_sca, q0.q_sca),
+                (jq.q_abs, qp.q_abs, qm.q_abs, q0.q_abs),
+                (jq.q_back, qp.q_back, qm.q_back, q0.q_back),
+                (jq.g, qp.g, qm.g, q0.g),
+            ] {
+                assert!(ok(a, fd(p, mm), v.abs()), "{a} vs {}", fd(p, mm));
+            }
+            let big = gv.alpha1.iter().fold(0.0f64, |a, v| a.max(v.abs()));
+            for l in 0..=8 {
+                assert!(ok(jg.alpha1[l], fd(gp.alpha1[l], gm.alpha1[l]), big));
+                assert!(ok(jg.beta1[l], fd(gp.beta1[l], gm.beta1[l]), big));
+                assert!(ok(jg.alpha3[l], fd(gp.alpha3[l], gm.alpha3[l]), big));
+            }
+            let fbig = f[0].iter().fold(0.0f64, |a, v| a.max(v.abs()));
+            for k in 0..6 {
+                for j in 0..mu.len() {
+                    assert!(ok(jf[k][j], fd(pp[k][j], pm[k][j]), fbig), "phase {k} {j}");
+                }
+            }
+        }
+    }
+
     #[test]
     fn invalid_inputs_and_domain() {
         let mut tm = TMatrix::new();
diff --git a/scatrs/src/tmatrix/solve.rs b/scatrs/src/tmatrix/solve.rs
index 2e7239e..cf1d25b 100644
--- a/scatrs/src/tmatrix/solve.rs
+++ b/scatrs/src/tmatrix/solve.rs
@@ -196,11 +196,25 @@ pub(crate) fn error_estimate(
     4.0 * worst
 }
 
+/// The result of [`refine`].
+pub(crate) struct Refined {
+    /// T' (scaled), rounded to f64 and in double-double.
+    pub(crate) t: Vec<Complex<f64>>,
+    pub(crate) t_dd: Vec<Complex<Dd>>,
+    /// The mixed Q (double-double rows and f64 rows).
+    pub(crate) q: Vec<Complex<Dd>>,
+    /// The factors of the mixed Q rounded to f64.
+    pub(crate) solved: F64Solve,
+    /// The sweeps made.
+    pub(crate) sweeps: usize,
+    /// The last correction's largest element, unscaled (`lh` as in [`error_estimate`]).
+    pub(crate) last: f64,
+}
+
 /// Mixed-precision refinement of one class: Q and P are `q`, `p` (f64) except in the rows flagged
 /// by `exact`, where they are `qd`, `pd` (double-double). Factorizes the mixed matrix rounded to
 /// f64, then corrects X Q = −P with double-double residuals R = −P − X Q until the correction
-/// falls below 2⁻⁶⁰ of T or after `sweeps`. Returns T' (scaled), the factors used, the sweeps
-/// made, and the last correction's largest element unscaled (`lh` as in [`error_estimate`]).
+/// falls below 2⁻⁶⁰ of T or after `sweeps`.
 #[allow(clippy::too_many_arguments)]
 pub(crate) fn refine(
     q: &[Complex<f64>],
@@ -211,7 +225,7 @@ pub(crate) fn refine(
     lh: &[f64],
     n: usize,
     sweeps: usize,
-) -> Result<(Vec<Complex<f64>>, F64Solve, usize, f64)> {
+) -> Result<Refined> {
     let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
     let hi = |c: Complex<Dd>| Complex::new(c.re.hi(), c.im.hi());
     let pick = |ij: usize, v: &[Complex<f64>], vd: &[Complex<Dd>]| {
@@ -249,5 +263,93 @@ pub(crate) fn refine(
             break;
         }
     }
-    Ok((x.iter().map(|&c| hi(c)).collect(), solved, done, last))
+    Ok(Refined {
+        t: x.iter().map(|&c| hi(c)).collect(),
+        t_dd: x,
+        q: qm,
+        solved,
+        sweeps: done,
+        last,
+    })
+}
+
+/// The derivative of T' = −P Q⁻¹ along one direction: dT' = −(dP + T' dQ) Q⁻¹, with the
+/// factors of Q.
+pub(crate) fn derivative_f64(
+    solved: &F64Solve,
+    t: &[Complex<f64>],
+    dq: &[Complex<f64>],
+    dp: &[Complex<f64>],
+    n: usize,
+) -> Vec<Complex<f64>> {
+    let mut out = vec![Complex::new(0.0, 0.0); n * n];
+    let mut row = vec![Complex::new(0.0, 0.0); n];
+    for i in 0..n {
+        for j in 0..n {
+            let mut v = dp[i * n + j];
+            for l in 0..n {
+                v += t[i * n + l] * dq[l * n + j];
+            }
+            row[j] = -v;
+        }
+        lu_solve(&solved.qt, &solved.piv, n, &mut row);
+        out[i * n..(i + 1) * n].copy_from_slice(&row);
+    }
+    out
+}
+
+/// Refines a derivative dT' of a refined class (`r`), with dQ and dP mixed as Q and P were
+/// (`dq`, `dp` in double-double, exact in the refined rows): residuals
+/// R = −dP − T' dQ − dT' Q in double-double, corrected by the f64 factors, for `sweeps` sweeps or
+/// until the correction falls below 2⁻⁶⁰ of dT'.
+pub(crate) fn refine_derivative(
+    r: &Refined,
+    dq: &[Complex<Dd>],
+    dp: &[Complex<Dd>],
+    n: usize,
+    sweeps: usize,
+) -> Vec<Complex<f64>> {
+    let dd = |c: Complex<f64>| Complex::new(Dd::from(c.re), Dd::from(c.im));
+    let hi = |c: Complex<Dd>| Complex::new(c.re.hi(), c.im.hi());
+    // The right-hand side −dP − T' dQ, fixed over the sweeps.
+    let mut rhs = vec![Complex::new(Dd::from(0.0), Dd::from(0.0)); n * n];
+    for i in 0..n {
+        for j in 0..n {
+            let mut v = -dp[i * n + j];
+            for l in 0..n {
+                v -= r.t_dd[i * n + l] * dq[l * n + j];
+            }
+            rhs[i * n + j] = v;
+        }
+    }
+    let first: Vec<Complex<f64>> = rhs.iter().map(|&c| hi(c)).collect();
+    let mut x: Vec<Complex<Dd>> = Vec::with_capacity(n * n);
+    let mut row = vec![Complex::new(0.0, 0.0); n];
+    for i in 0..n {
+        row.copy_from_slice(&first[i * n..(i + 1) * n]);
+        lu_solve(&r.solved.qt, &r.solved.piv, n, &mut row);
+        x.extend(row.iter().map(|&c| dd(c)));
+    }
+    let big = x.iter().map(|c| hi(*c).norm()).fold(0.0, f64::max);
+    for _ in 0..sweeps {
+        let mut worst = 0.0f64;
+        for i in 0..n {
+            for j in 0..n {
+                let mut v = rhs[i * n + j];
+                for l in 0..n {
+                    v -= x[i * n + l] * r.q[l * n + j];
+                }
+                row[j] = hi(v);
+            }
+            lu_solve(&r.solved.qt, &r.solved.piv, n, &mut row);
+            for j in 0..n {
+                worst = worst.max(row[j].norm());
+                x[i * n + j] += dd(row[j]);
+            }
+        }
+        if worst <= big * 2f64.powi(-60) {
+            break;
+        }
+    }
+    x.iter().map(|&c| hi(c)).collect()
 }
````

### Task 6: Python `scatrs.tmatrix`

- [ ] Bindings: `tmatrix_random` (one solve per spheroid for every output asked for) and `TMatrixObject`; `TMatrix::solution`, `Derivatives::new` in the core.
- [ ] `python/scatrs/tmatrix.py`: `efficiencies`, `greek`, `phase_matrix`, `compute`, result classes; stubs; package namespace test.
- [ ] Tests: shapes, sphere limit, derivative names and a difference, the `TMatrix` object, errors.

The commit (`c9b95de`):

````diff
diff --git a/python/scatrs/__init__.py b/python/scatrs/__init__.py
index eb7c720..bf8443a 100644
--- a/python/scatrs/__init__.py
+++ b/python/scatrs/__init__.py
@@ -3,7 +3,17 @@
 from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
 from importlib.metadata import version as _version
 
-from scatrs import _scatrs, ensemble, layered, mie, particles, refractive, size, special
+from scatrs import (
+    _scatrs,
+    ensemble,
+    layered,
+    mie,
+    particles,
+    refractive,
+    size,
+    special,
+    tmatrix,
+)
 from scatrs.ensemble import (
     Ensemble,
     EnsembleGreek,
@@ -57,6 +67,7 @@ __all__ = [
     "refractive",
     "size",
     "special",
+    "tmatrix",
 ]
 
 # The distribution's version, in PEP 440 form: maturin derives it from the Cargo workspace
diff --git a/python/scatrs/_scatrs/__init__.pyi b/python/scatrs/_scatrs/__init__.pyi
index a0f0353..b5501d1 100644
--- a/python/scatrs/_scatrs/__init__.pyi
+++ b/python/scatrs/_scatrs/__init__.pyi
@@ -1,5 +1,5 @@
 # Stubs for the extension module scatrs._scatrs (a stub package, for its _testing submodule).
-from typing import Any, Final
+from typing import Any, Final, final
 
 import numpy as np
 from numpy.typing import NDArray
@@ -12,7 +12,9 @@ __all__ = [
     "MAX_LAYERS",
     "MAX_MIE_ORDER",
     "MAX_MIE_VALUES",
+    "MAX_TMATRIX_ORDER",
     "MAX_WIGNER_D_VALUES",
+    "TMatrixObject",
     "__version__",
     "ensemble_compute",
     "ensemble_compute_batch",
@@ -29,6 +31,7 @@ __all__ = [
     "mie_efficiencies",
     "mie_greek",
     "mie_phase_matrix",
+    "tmatrix_random",
     "wigner_d",
 ]
 
@@ -163,3 +166,65 @@ def ensemble_compute_batch(
     allow_unvalidated: bool,
     refine_backscatter: bool,
 ) -> dict[str, Any]: ...
+
+MAX_TMATRIX_ORDER: Final[int]
+
+def tmatrix_random(
+    x: NDArray[np.float64],
+    m: NDArray[np.complex128],
+    eps: NDArray[np.float64],
+    rtol: float,
+    precision: str,
+    n_terms: int | None,
+    n_theta: int | None,
+    allow_unconverged: bool,
+    allow_unvalidated: bool,
+    derivatives: bool,
+    greek: bool,
+    order: int | None,
+    mu: NDArray[np.float64] | None,
+    normalized: bool,
+) -> dict[str, NDArray[Any]]: ...
+
+@final
+class TMatrixObject:
+    def __new__(
+        cls,
+        x: float,
+        m_re: float,
+        m_im: float,
+        eps: float,
+        rtol: float,
+        precision: str,
+        n_terms: int | None,
+        n_theta: int | None,
+        allow_unconverged: bool,
+        allow_unvalidated: bool,
+        derivatives: bool,
+    ) -> TMatrixObject: ...
+    @property
+    def n_terms(self) -> int: ...
+    @property
+    def n_theta(self) -> int: ...
+    @property
+    def error_estimate(self) -> float: ...
+    @property
+    def converged(self) -> bool: ...
+    @property
+    def precision_paths(self) -> list[str]: ...
+    def block(
+        self, m: int
+    ) -> (
+        tuple[
+            NDArray[np.int64],
+            list[NDArray[np.complex128]],
+        ]
+        | None
+    ): ...
+    def efficiencies(self) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
+    def greek(
+        self, order: int | None, normalized: bool
+    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
+    def phase_matrix(
+        self, mu: NDArray[np.float64], normalized: bool
+    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
diff --git a/python/scatrs/tmatrix.py b/python/scatrs/tmatrix.py
new file mode 100644
index 0000000..6f8db1f
--- /dev/null
+++ b/python/scatrs/tmatrix.py
@@ -0,0 +1,583 @@
+"""Scattering by randomly oriented spheroids, by the T-matrix method (vectorized wrappers over
+the Rust core).
+
+A spheroid has semi-axes ``a, a, c``, ``c`` along its symmetry axis, and aspect ratio
+``aspect_ratio = a / c``: oblate above 1, prolate below 1, a sphere at 1. Its size is the size
+parameter ``x = 2 pi r_v / wavelength`` of the sphere of equal volume (``r_v**3 = a**2 c``),
+which also normalizes the efficiencies (cross sections divided by ``pi r_v**2``) and the
+unnormalized phase matrix and Greek coefficients. ``x``, ``m`` and ``aspect_ratio`` broadcast
+against each other to a shape ``S``; every result has that shape, extended by the orders or the
+angles. Conventions are those of :mod:`scatrs.mie`, averaged uniformly over orientations. See
+:doc:`/theory/spheroids` for the method and :doc:`/validation/accuracy` for the validated domain.
+
+Each spheroid's T-matrix is converged to the relative tolerance ``rtol`` in its number of
+orders and quadrature nodes; the results carry ``n_terms``, ``n_theta`` and an ``error``
+estimate per spheroid. ``derivatives`` is ``True`` (all of ``"x"``, ``"aspect_ratio"``,
+``"m_real"``, ``"m_imag"``) or a sequence of those names; the result's ``d`` method returns the
+derivative with respect to one of them.
+"""
+
+from __future__ import annotations
+
+from collections.abc import Iterable, Sequence
+from dataclasses import dataclass, field
+from typing import Final, Literal, SupportsIndex
+
+import numpy as np
+from numpy.typing import ArrayLike, NDArray
+
+from scatrs import _args, _scatrs
+from scatrs.mie import MieEfficiencies, MieGreek, _frozen
+
+__all__ = [
+    "MAX_TMATRIX_ORDER",
+    "PARAMETERS",
+    "SpheroidEfficiencies",
+    "SpheroidGreek",
+    "SpheroidPhaseMatrix",
+    "TMatrix",
+    "compute",
+    "efficiencies",
+    "greek",
+    "phase_matrix",
+]
+
+MAX_TMATRIX_ORDER: Final[int] = _scatrs.MAX_TMATRIX_ORDER
+"""The largest number of orders N accepted: 500."""
+
+PARAMETERS: Final = ("x", "aspect_ratio", "m_real", "m_imag")
+"""The parameters ``derivatives`` may name."""
+
+Precision = Literal["auto", "f64", "double-double"]
+
+
+def _parameters(derivatives: bool | Sequence[str]) -> tuple[str, ...]:
+    if isinstance(derivatives, (bool, np.bool_)):
+        return PARAMETERS if derivatives else ()
+    if isinstance(derivatives, (str, bytes)) or not isinstance(derivatives, Iterable):
+        raise TypeError(
+            "derivatives must be a bool or a sequence of parameter names, "
+            f"not {type(derivatives).__name__}"
+        )
+    names = tuple(derivatives)
+    for name in names:
+        if name not in PARAMETERS:
+            raise ValueError(f"unknown parameter {name!r}; expected one of {PARAMETERS}")
+    return names
+
+
+def _index(params: tuple[str, ...], param: str) -> int:
+    if param not in params:
+        if param == "m":
+            raise ValueError("these outputs are real: use 'm_real' or 'm_imag'")
+        raise ValueError(f"no derivative with respect to {param!r}; computed: {params}")
+    return PARAMETERS.index(param)
+
+
+def _inputs(
+    x: ArrayLike, m: ArrayLike, aspect_ratio: ArrayLike
+) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64], tuple[int, ...]]:
+    xs = _args.real_array(x, "x")
+    ms = _args.complex_array(m, "m")
+    es = _args.real_array(aspect_ratio, "aspect_ratio")
+    xb, mb, eb = np.broadcast_arrays(xs, ms, es)
+    shape = xb.shape
+    return (
+        np.ascontiguousarray(xb).reshape(-1),
+        np.ascontiguousarray(mb).reshape(-1),
+        np.ascontiguousarray(eb).reshape(-1),
+        shape,
+    )
+
+
+def _precision(precision: str) -> str:
+    if precision not in ("auto", "f64", "double-double"):
+        raise ValueError(f"precision must be 'auto', 'f64' or 'double-double', got {precision!r}")
+    return precision
+
+
+def _optional_size(value: SupportsIndex | None, name: str) -> int | None:
+    return None if value is None else _args.size(value, name)
+
+
+@dataclass(frozen=True, eq=False)
+class SpheroidEfficiencies(MieEfficiencies):
+    """Orientation-averaged efficiencies of spheroids (:class:`~scatrs.mie.MieEfficiencies`,
+    per unit geometric cross section of the equal-volume sphere; ``q_abs = q_ext - q_sca``),
+    with the convergence information of each spheroid.
+
+    Attributes
+    ----------
+    n_terms, n_theta : numpy.ndarray
+        int64, shape ``S``: the number of orders and of quadrature nodes (in a half range of
+        the polar angle) each T-matrix converged at.
+    error : numpy.ndarray
+        float64, shape ``S``: each spheroid's estimated relative error.
+    """
+
+    n_terms: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
+    n_theta: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
+    error: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
+
+    def d(self, param: str) -> MieEfficiencies:
+        """The derivatives with respect to ``param`` (one of :data:`PARAMETERS`)."""
+        p = _index(self.params, param)
+        assert self._jacobian is not None
+        return MieEfficiencies(*self._jacobian[p])
+
+
+@dataclass(frozen=True, eq=False)
+class SpheroidGreek(MieGreek):
+    """Greek expansion coefficients of the orientation-averaged phase matrix of spheroids
+    (:class:`~scatrs.mie.MieGreek`, with ``F22`` and ``F44`` independent); orders above a
+    spheroid's ``2 n_terms`` are exactly 0."""
+
+    def d(self, param: str) -> MieGreek:
+        """The derivatives with respect to ``param`` (one of :data:`PARAMETERS`)."""
+        p = _index(self.params, param)
+        assert self._derivatives is not None
+        return MieGreek(*self._derivatives[p], self.n_terms)
+
+
+@dataclass(frozen=True, eq=False)
+class SpheroidPhaseMatrix:
+    """The orientation-averaged phase matrix of spheroids, arrays of shape ``S + mu.shape``:
+    its six independent elements (the others are 0), normalized (``F11`` averages to 1 over the
+    sphere of directions) or unnormalized (times ``q_sca``). For a sphere ``F22 = F11`` and
+    ``F44 = F33``.
+
+    Attributes
+    ----------
+    f11, f12, f22, f33, f34, f44 : numpy.ndarray
+        The elements.
+    params : tuple of str
+        The parameters :meth:`d` accepts.
+    """
+
+    f11: NDArray[np.float64]
+    f12: NDArray[np.float64]
+    f22: NDArray[np.float64]
+    f33: NDArray[np.float64]
+    f34: NDArray[np.float64]
+    f44: NDArray[np.float64]
+    params: tuple[str, ...] = ()
+    _derivatives: NDArray[np.float64] | None = field(default=None, repr=False)
+
+    def d(self, param: str) -> SpheroidPhaseMatrix:
+        """The derivatives with respect to ``param`` (one of :data:`PARAMETERS`); normalized
+        derivatives include that of ``q_sca``."""
+        p = _index(self.params, param)
+        assert self._derivatives is not None
+        return SpheroidPhaseMatrix(*self._derivatives[p])
+
+
+def _run(
+    x: ArrayLike,
+    m: ArrayLike,
+    aspect_ratio: ArrayLike,
+    *,
+    rtol: float,
+    precision: str,
+    n_terms: SupportsIndex | None,
+    n_theta: SupportsIndex | None,
+    allow_unconverged: bool,
+    allow_unvalidated: bool,
+    derivatives: bool | Sequence[str],
+    greek: bool = False,
+    order: SupportsIndex | None = None,
+    mu: ArrayLike | None = None,
+    normalized: bool = True,
+) -> tuple[dict[str, NDArray], tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
+    params = _parameters(derivatives)
+    xb, mb, eb, shape = _inputs(x, m, aspect_ratio)
+    mus = None if mu is None else _args.real_array(mu, "mu")
+    out = _scatrs.tmatrix_random(
+        xb,
+        mb,
+        eb,
+        float(rtol),
+        _precision(precision),
+        _optional_size(n_terms, "n_terms"),
+        _optional_size(n_theta, "n_theta"),
+        _args.flag(allow_unconverged, "allow_unconverged"),
+        _args.flag(allow_unvalidated, "allow_unvalidated"),
+        bool(params),
+        greek,
+        _optional_size(order, "order"),
+        None if mus is None else mus.reshape(-1),
+        _args.flag(normalized, "normalized"),
+    )
+    return out, shape, params, () if mus is None else mus.shape
+
+
+def _info(out: dict[str, NDArray], shape: tuple[int, ...]) -> dict[str, NDArray]:
+    return {
+        "n_terms": _frozen(out["n_terms"].astype(np.int64)).reshape(shape),
+        "n_theta": _frozen(out["n_theta"].astype(np.int64)).reshape(shape),
+        "error": _frozen(out["error"]).reshape(shape),
+    }
+
+
+def efficiencies(
+    x: ArrayLike,
+    m: ArrayLike,
+    aspect_ratio: ArrayLike,
+    *,
+    rtol: float = 1e-8,
+    precision: Precision = "auto",
+    n_terms: SupportsIndex | None = None,
+    n_theta: SupportsIndex | None = None,
+    derivatives: bool | Sequence[str] = False,
+    allow_unconverged: bool = False,
+    allow_unvalidated: bool = False,
+) -> SpheroidEfficiencies:
+    out, shape, params, _ = _run(
+        x,
+        m,
+        aspect_ratio,
+        rtol=rtol,
+        precision=precision,
+        n_terms=n_terms,
+        n_theta=n_theta,
+        allow_unconverged=allow_unconverged,
+        allow_unvalidated=allow_unvalidated,
+        derivatives=derivatives,
+    )
+    q = _frozen(out["efficiencies"])
+    jac = out.get("efficiencies_d")
+    jac = None if jac is None else _frozen(jac).reshape((4, 5, *shape))
+    return SpheroidEfficiencies(*(v.reshape(shape) for v in q), params, jac, **_info(out, shape))
+
+
+def greek(
+    x: ArrayLike,
+    m: ArrayLike,
+    aspect_ratio: ArrayLike,
+    order: SupportsIndex | None = None,
+    *,
+    normalized: bool = True,
+    rtol: float = 1e-8,
+    precision: Precision = "auto",
+    n_terms: SupportsIndex | None = None,
+    n_theta: SupportsIndex | None = None,
+    derivatives: bool | Sequence[str] = False,
+    allow_unconverged: bool = False,
+    allow_unvalidated: bool = False,
+) -> SpheroidGreek:
+    out, shape, params, _ = _run(
+        x,
+        m,
+        aspect_ratio,
+        rtol=rtol,
+        precision=precision,
+        n_terms=n_terms,
+        n_theta=n_theta,
+        allow_unconverged=allow_unconverged,
+        allow_unvalidated=allow_unvalidated,
+        derivatives=derivatives,
+        greek=True,
+        order=order,
+        normalized=normalized,
+    )
+    g = _frozen(out["greek"])
+    width = g.shape[-1]
+    shaped = (*shape, width)
+    d = out.get("greek_d")
+    d = None if d is None else _frozen(d).reshape((4, 6, *shaped))
+    info = _info(out, shape)
+    return SpheroidGreek(*(v.reshape(shaped) for v in g), info["n_terms"], params, d)
+
+
+def phase_matrix(
+    x: ArrayLike,
+    m: ArrayLike,
+    aspect_ratio: ArrayLike,
+    mu: ArrayLike,
+    *,
+    normalized: bool = True,
+    rtol: float = 1e-8,
+    precision: Precision = "auto",
+    n_terms: SupportsIndex | None = None,
+    n_theta: SupportsIndex | None = None,
+    derivatives: bool | Sequence[str] = False,
+    allow_unconverged: bool = False,
+    allow_unvalidated: bool = False,
+) -> SpheroidPhaseMatrix:
+    out, shape, params, mu_shape = _run(
+        x,
+        m,
+        aspect_ratio,
+        rtol=rtol,
+        precision=precision,
+        n_terms=n_terms,
+        n_theta=n_theta,
+        allow_unconverged=allow_unconverged,
+        allow_unvalidated=allow_unvalidated,
+        derivatives=derivatives,
+        mu=mu,
+        normalized=normalized,
+    )
+    f = _frozen(out["phase"])
+    shaped = (*shape, *mu_shape)
+    d = out.get("phase_d")
+    d = None if d is None else _frozen(d).reshape((4, 6, *shaped))
+    return SpheroidPhaseMatrix(*(v.reshape(shaped) for v in f), params=params, _derivatives=d)
+
+
+_SHARED_DOC = """
+    Parameters
+    ----------
+    x : array_like
+        Equal-volume size parameters, 2 pi r_v / wavelength (real).
+    m : array_like
+        Refractive indices relative to the medium (real or complex).
+    aspect_ratio : array_like
+        ``a / c``: oblate above 1, prolate below 1 (real, positive).
+    {extra}rtol : float, optional
+        The relative tolerance of each T-matrix's convergence (default ``1e-8``).
+    precision : {{"auto", "f64", "double-double"}}, optional
+        The precision paths allowed: ``"auto"`` refines in mixed precision the modes whose
+        error estimate requires it; ``"f64"`` never does (fastest); ``"double-double"`` computes
+        everything in double-double (slowest, for checks).
+    n_terms, n_theta : int, optional
+        Fix the number of orders or of quadrature nodes instead of converging them.
+    derivatives : bool or sequence of str, optional
+        Also compute derivatives with respect to these of :data:`PARAMETERS` (``True`` for all
+        four); read them with the result's ``d`` method.
+    allow_unconverged : bool, optional
+        Return the last step of a convergence that reached its cap, with its error estimate,
+        instead of raising :class:`~scatrs.errors.NotConvergedError`.
+    allow_unvalidated : bool, optional
+        Compute outside the validated domain instead of raising
+        :class:`~scatrs.errors.OutsideValidatedDomainError`.
+
+    Raises
+    ------
+    InvalidInputError
+        For non-positive or non-finite ``x`` or ``aspect_ratio``, non-finite ``m``,
+        ``Re m <= 0`` or ``Im m < 0``, ``rtol`` outside (0, 1){extra_raise}. The message names
+        the first failing element (in C order).
+    OutsideValidatedDomainError
+        For valid inputs outside the validated domain, unless ``allow_unvalidated``.
+    NotConvergedError
+        If a T-matrix reaches the cap of its orders or nodes, unless ``allow_unconverged``.
+    ValueError
+        For an unknown parameter name or precision, or shapes that do not broadcast.
+
+    Notes
+    -----
+    The cost per spheroid grows like the fourth power of ``|m| x_max``, with ``x_max`` the size
+    parameter of the longest semi-axis, and steeply for extreme aspect ratios, where the
+    refinement engages; the call releases the GIL but cannot be interrupted.
+"""
+
+efficiencies.__doc__ = """Orientation-averaged efficiencies of spheroids.
+
+    Returns
+    -------
+    SpheroidEfficiencies
+        Shape ``S``.
+""" + _SHARED_DOC.format(extra="", extra_raise="")
+
+greek.__doc__ = """Greek expansion coefficients of the orientation-averaged phase matrix of
+    spheroids.
+
+    Returns
+    -------
+    SpheroidGreek
+        Shape ``S + (L + 1,)``, with ``L`` = ``order``, by default the largest ``2 n_terms``.
+""" + _SHARED_DOC.format(
+    extra="order : int, optional\n        The highest order L.\n"
+    "    normalized : bool, optional\n        Normalize so that ``alpha1[..., 0]`` is 1"
+    " (the default); otherwise times ``q_sca``.\n    ",
+    extra_raise=", ``order`` above ``MAX_GREEK_ORDER``",
+)
+
+phase_matrix.__doc__ = """The six-element orientation-averaged phase matrix of spheroids.
+
+    Returns
+    -------
+    SpheroidPhaseMatrix
+        Shape ``S + mu.shape``.
+""" + _SHARED_DOC.format(
+    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n"
+    "    normalized : bool, optional\n        Normalize so that F11 averages to 1 (the"
+    " default); otherwise times ``q_sca``.\n    ",
+    extra_raise=", ``mu`` outside [-1, 1]",
+)
+
+
+@dataclass(frozen=True, eq=False)
+class TMatrixBlock:
+    """The T-matrix of one azimuthal mode ``m >= 0`` for orders ``n, k = max(m, 1), ..., N``.
+
+    Attributes
+    ----------
+    orders : numpy.ndarray
+        int64, the orders of the rows and columns.
+    t11, t12, t21, t22 : numpy.ndarray
+        complex128, square: T11 (M from M), T12 (M from N), T21 (N from M), T22 (N from N).
+        Mode ``-m`` has ``t11`` and ``t22`` equal, ``t12`` and ``t21`` negated.
+    """
+
+    orders: NDArray[np.int64]
+    t11: NDArray[np.complex128]
+    t12: NDArray[np.complex128]
+    t21: NDArray[np.complex128]
+    t22: NDArray[np.complex128]
+
+
+class TMatrix:
+    """The converged T-matrix of one spheroid, from :func:`compute`.
+
+    Attributes
+    ----------
+    n_terms, n_theta : int
+        The number of orders and of quadrature nodes.
+    error_estimate : float
+        The estimated relative error of the outputs.
+    converged : bool
+        False only for a result returned at a cap with ``allow_unconverged``.
+    precision_paths : tuple of str
+        How each mode ``m = 0, ..., N`` was solved: ``"f64"``, ``"refined"`` or
+        ``"double-double"``.
+    params : tuple of str
+        The parameters of the derivatives, if computed.
+    """
+
+    __slots__ = ("_obj", "params")
+
+    def __init__(self, obj: _scatrs.TMatrixObject, params: tuple[str, ...]) -> None:
+        self._obj = obj
+        self.params = params
+
+    @property
+    def n_terms(self) -> int:
+        return int(self._obj.n_terms)
+
+    @property
+    def n_theta(self) -> int:
+        return int(self._obj.n_theta)
+
+    @property
+    def error_estimate(self) -> float:
+        return float(self._obj.error_estimate)
+
+    @property
+    def converged(self) -> bool:
+        return bool(self._obj.converged)
+
+    @property
+    def precision_paths(self) -> tuple[str, ...]:
+        return tuple(self._obj.precision_paths)
+
+    def block(self, m: SupportsIndex) -> TMatrixBlock:
+        """The blocks of mode ``m`` (0 to ``n_terms``).
+
+        Raises
+        ------
+        IndexError
+            For ``m`` outside 0 to ``n_terms``.
+        """
+        mm = _args.integer(m, "m")
+        b = None if mm < 0 else self._obj.block(mm)
+        if b is None:
+            raise IndexError(f"mode {mm} is outside 0..={self.n_terms}")
+        orders, (t11, t12, t21, t22) = b
+        return TMatrixBlock(*(_frozen(v) for v in (orders, t11, t12, t21, t22)))
+
+    def efficiencies(self) -> SpheroidEfficiencies:
+        """The orientation-averaged efficiencies (scalars), with their derivatives if
+        computed."""
+        q, d = self._obj.efficiencies()
+        jac = None if d is None else _frozen(d)
+        info = {
+            "n_terms": np.int64(self.n_terms),
+            "n_theta": np.int64(self.n_theta),
+            "error": np.float64(self.error_estimate),
+        }
+        return SpheroidEfficiencies(*q, self.params, jac, **info)
+
+    def greek(
+        self, order: SupportsIndex | None = None, *, normalized: bool = True
+    ) -> SpheroidGreek:
+        """The Greek coefficients up to ``order`` (default ``2 n_terms``)."""
+        g, d = self._obj.greek(_optional_size(order, "order"), _args.flag(normalized, "normalized"))
+        return SpheroidGreek(
+            *_frozen(g),
+            np.int64(self.n_terms),
+            self.params,
+            None if d is None else _frozen(d),
+        )
+
+    def phase_matrix(self, mu: ArrayLike, *, normalized: bool = True) -> SpheroidPhaseMatrix:
+        """The six-element phase matrix at the cosines ``mu``."""
+        mus = _args.real_array(mu, "mu")
+        f, d = self._obj.phase_matrix(mus.reshape(-1), _args.flag(normalized, "normalized"))
+        f = _frozen(f).reshape((6, *mus.shape))
+        d = None if d is None else _frozen(d).reshape((4, 6, *mus.shape))
+        return SpheroidPhaseMatrix(*f, params=self.params, _derivatives=d)
+
+    def random_orientation(
+        self,
+        mu: ArrayLike | None = None,
+        order: SupportsIndex | None = None,
+        *,
+        normalized: bool = True,
+    ) -> tuple[SpheroidEfficiencies, SpheroidGreek, SpheroidPhaseMatrix | None]:
+        """The orientation-averaged outputs: efficiencies, Greek coefficients up to ``order``
+        and, with cosines ``mu``, the phase matrix."""
+        return (
+            self.efficiencies(),
+            self.greek(order, normalized=normalized),
+            None if mu is None else self.phase_matrix(mu, normalized=normalized),
+        )
+
+    def __repr__(self) -> str:
+        return (
+            f"TMatrix(n_terms={self.n_terms}, n_theta={self.n_theta}, "
+            f"error_estimate={self.error_estimate:.1e})"
+        )
+
+
+def compute(
+    x: float,
+    m: complex,
+    aspect_ratio: float,
+    *,
+    rtol: float = 1e-8,
+    precision: Precision = "auto",
+    n_terms: SupportsIndex | None = None,
+    n_theta: SupportsIndex | None = None,
+    derivatives: bool | Sequence[str] = False,
+    allow_unconverged: bool = False,
+    allow_unvalidated: bool = False,
+) -> TMatrix:
+    params = _parameters(derivatives)
+    mc = complex(m)
+    obj = _scatrs.TMatrixObject(
+        float(x),
+        mc.real,
+        mc.imag,
+        float(aspect_ratio),
+        float(rtol),
+        _precision(precision),
+        _optional_size(n_terms, "n_terms"),
+        _optional_size(n_theta, "n_theta"),
+        _args.flag(allow_unconverged, "allow_unconverged"),
+        _args.flag(allow_unvalidated, "allow_unvalidated"),
+        bool(params),
+    )
+    return TMatrix(obj, params)
+
+
+compute.__doc__ = """The converged T-matrix of one spheroid, with its convergence information and
+    orientation-averaged outputs.
+
+    Returns
+    -------
+    TMatrix
+""" + _SHARED_DOC.format(extra="", extra_raise="").replace(
+    "x : array_like\n        Equal-volume size parameters",
+    "x : float\n        The equal-volume size parameter",
+)
diff --git a/scatrs-py/src/lib.rs b/scatrs-py/src/lib.rs
index 6c74f24..9132518 100644
--- a/scatrs-py/src/lib.rs
+++ b/scatrs-py/src/lib.rs
@@ -12,6 +12,7 @@ mod layered;
 mod mie;
 mod special;
 mod testing;
+mod tmatrix;
 
 #[pymodule]
 fn _scatrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
@@ -38,6 +39,9 @@ fn _scatrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
     m.add_function(wrap_pyfunction!(layered::layered_amplitudes, m)?)?;
     m.add_function(wrap_pyfunction!(layered::layered_phase_matrix, m)?)?;
     m.add_function(wrap_pyfunction!(layered::layered_greek, m)?)?;
+    m.add("MAX_TMATRIX_ORDER", scatrs::tmatrix::MAX_TMATRIX_ORDER)?;
+    m.add_function(wrap_pyfunction!(tmatrix::tmatrix_random, m)?)?;
+    m.add_class::<tmatrix::TMatrixObject>()?;
     m.add_function(wrap_pyfunction!(ensemble::ensemble_parameters, m)?)?;
     m.add_function(wrap_pyfunction!(ensemble::ensemble_compute, m)?)?;
     m.add_function(wrap_pyfunction!(ensemble::ensemble_compute_batch, m)?)?;
diff --git a/scatrs-py/src/tmatrix.rs b/scatrs-py/src/tmatrix.rs
new file mode 100644
index 0000000..24d2f17
--- /dev/null
+++ b/scatrs-py/src/tmatrix.rs
@@ -0,0 +1,516 @@
+//! Bindings for `scatrs::tmatrix`.
+//!
+//! `tmatrix_random` takes `x` (float64), `m` (complex128) and `eps` (float64) arrays of one shape
+//! (`scatrs.tmatrix` broadcasts them) and solves each spheroid once, returning every output asked
+//! for in a dict of arrays with one column (or row) per particle. `TMatrixObject` holds one solved
+//! spheroid for `scatrs.tmatrix.compute`. The GIL is released while computing.
+
+use numpy::ndarray::{Array1, Array2, Array3, Array4};
+use numpy::{Complex64, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
+use pyo3::exceptions::PyValueError;
+use pyo3::prelude::*;
+use pyo3::types::PyDict;
+use scatrs::mie::{Efficiencies, GreekCoefficients};
+use scatrs::tmatrix::{
+    Derivatives, PhaseBuffers, Precision, PrecisionPath, Shape, TMatrix, TMatrixOptions,
+    TMatrixSolution,
+};
+
+use crate::errors::to_pyerr;
+use crate::mie::{
+    Failure, at, borrow, cosines, greek_arrays, invalid, normalization, real_values, shape_error,
+    zeros,
+};
+
+fn precision(py: Python<'_>, name: &str) -> PyResult<Precision> {
+    match name {
+        "auto" => Ok(Precision::Auto),
+        "f64" => Ok(Precision::F64),
+        "double-double" => Ok(Precision::DoubleDouble),
+        other => Err(to_pyerr(
+            py,
+            invalid(format!(
+                "precision must be 'auto', 'f64' or 'double-double', got {other:?}"
+            )),
+        )),
+    }
+}
+
+#[allow(clippy::too_many_arguments)]
+fn options(
+    py: Python<'_>,
+    rtol: f64,
+    precision_name: &str,
+    n_terms: Option<usize>,
+    n_theta: Option<usize>,
+    greek_order: Option<usize>,
+    allow_unconverged: bool,
+    allow_unvalidated: bool,
+) -> PyResult<TMatrixOptions> {
+    let mut opts = TMatrixOptions::new()
+        .with_rtol(rtol)
+        .with_precision(precision(py, precision_name)?)
+        .with_allow_unconverged(allow_unconverged)
+        .with_allow_unvalidated(allow_unvalidated);
+    if let Some(n) = n_terms {
+        opts = opts.with_n_terms(n);
+    }
+    if let Some(n) = n_theta {
+        opts = opts.with_n_theta(n);
+    }
+    if let Some(l) = greek_order {
+        opts = opts.with_greek_order(l);
+    }
+    Ok(opts)
+}
+
+fn efficiency_values(q: &Efficiencies<f64>) -> [f64; 5] {
+    [q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g]
+}
+
+fn path_name(p: PrecisionPath) -> &'static str {
+    match p {
+        PrecisionPath::F64 => "f64",
+        PrecisionPath::Refined => "refined",
+        PrecisionPath::DoubleDouble => "double-double",
+        _ => "other",
+    }
+}
+
+fn solve<'w>(
+    tm: &'w mut TMatrix,
+    eps: f64,
+    x: f64,
+    m: Complex64,
+    opts: &TMatrixOptions,
+    derivatives: bool,
+) -> scatrs::Result<TMatrixSolution<'w>> {
+    let shape = Shape::Spheroid { aspect_ratio: eps };
+    if derivatives {
+        tm.compute_with_derivatives(shape, x, m, opts)
+    } else {
+        tm.compute(shape, x, m, opts)
+    }
+}
+
+/// Six phase buffers over row `j` of six consecutive (P, Q) arrays in `buf`.
+fn rows(buf: &mut [f64], per: usize, j: usize, q: usize) -> PhaseBuffers<'_> {
+    let mut parts = buf.chunks_mut(per).map(|a| &mut a[j * q..(j + 1) * q]);
+    let mut next = || parts.next().expect("six arrays");
+    PhaseBuffers {
+        f11: next(),
+        f12: next(),
+        f22: next(),
+        f33: next(),
+        f34: next(),
+        f44: next(),
+    }
+}
+
+/// Solves each spheroid (x, m, eps of one shape, flattened to P) and returns, in a dict:
+/// `n_terms`, `n_theta`, `error` (P); `efficiencies` (5, P) in the order Q_ext, Q_sca, Q_abs,
+/// Q_back, g; with `greek`, `greek` (6, P, L + 1) (L = `order`, or the largest 2N); with `mu`,
+/// `phase` (6, P, Q) in the order F11, F12, F22, F33, F34, F44; and with `derivatives`, each one's
+/// `*_d` with a leading axis of 4 (x, aspect ratio, Re m, Im m).
+#[pyfunction]
+#[pyo3(signature = (x, m, eps, rtol, precision, n_terms, n_theta, allow_unconverged, allow_unvalidated, derivatives, greek, order, mu, normalized))]
+#[allow(clippy::too_many_arguments)]
+pub fn tmatrix_random<'py>(
+    py: Python<'py>,
+    x: Bound<'py, PyArrayDyn<f64>>,
+    m: Bound<'py, PyArrayDyn<Complex64>>,
+    eps: Bound<'py, PyArrayDyn<f64>>,
+    rtol: f64,
+    precision: &str,
+    n_terms: Option<usize>,
+    n_theta: Option<usize>,
+    allow_unconverged: bool,
+    allow_unvalidated: bool,
+    derivatives: bool,
+    greek: bool,
+    order: Option<usize>,
+    mu: Option<Bound<'py, PyArrayDyn<f64>>>,
+    normalized: bool,
+) -> PyResult<Bound<'py, PyDict>> {
+    let (x, m, eps) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&eps, "eps")?);
+    let (xv, mv, ev) = (x.as_array(), m.as_array(), eps.as_array());
+    if xv.shape() != mv.shape() || xv.shape() != ev.shape() {
+        return Err(PyValueError::new_err(
+            "x, m and eps must have one shape (broadcast them first)",
+        ));
+    }
+    let opts = options(
+        py,
+        rtol,
+        precision,
+        n_terms,
+        n_theta,
+        if greek { order } else { None },
+        allow_unconverged,
+        allow_unvalidated,
+    )?;
+    let mu = match &mu {
+        Some(a) => Some(cosines(py, &borrow(a, "mu")?)?),
+        None => None,
+    };
+    let norm = normalization(normalized);
+    let len = xv.len();
+    let n_mu = mu.as_ref().map_or(0, Vec::len);
+    let per_phase = len * n_mu;
+    let count = len.checked_mul(n_mu.max(1)).and_then(|v| v.checked_mul(30));
+    real_values("the phase matrix", count, derivatives).map_err(|e| to_pyerr(py, e))?;
+    type Out = (
+        Vec<f64>,
+        Vec<f64>,
+        Vec<usize>,
+        Vec<usize>,
+        Vec<f64>,
+        Vec<GreekCoefficients<f64>>,
+        Vec<Derivatives<GreekCoefficients<f64>>>,
+        Vec<f64>,
+        Vec<f64>,
+    );
+    let out: Out = py
+        .detach(|| -> Result<Out, Failure> {
+            let mut q = zeros::<f64>(5 * len, "the efficiencies")?;
+            let mut qd = zeros::<f64>(if derivatives { 20 * len } else { 0 }, "derivatives")?;
+            let mut terms = vec![0usize; len];
+            let mut thetas = vec![0usize; len];
+            let mut error = vec![0.0f64; len];
+            let mut greeks = Vec::new();
+            let mut greeks_d = Vec::new();
+            let mut f = zeros::<f64>(6 * per_phase, "the phase matrix")?;
+            let mut fd = zeros::<f64>(if derivatives { 24 * per_phase } else { 0 }, "derivatives")?;
+            let mut tm = TMatrix::new();
+            for (i, ((&xi, &mi), &ei)) in xv.iter().zip(mv.iter()).zip(ev.iter()).enumerate() {
+                let sol = solve(&mut tm, ei, xi, mi, &opts, derivatives).map_err(at(i))?;
+                terms[i] = sol.n_terms();
+                thetas[i] = sol.n_theta();
+                error[i] = sol.error_estimate();
+                for (k, v) in efficiency_values(&sol.efficiencies())
+                    .into_iter()
+                    .enumerate()
+                {
+                    q[k * len + i] = v;
+                }
+                if derivatives {
+                    let j = sol.efficiencies_jacobian().ok_or_else(|| {
+                        invalid("the efficiency derivatives are unavailable".into())
+                    })?;
+                    for (p, e) in [&j.x, &j.aspect_ratio, &j.m_re, &j.m_im]
+                        .into_iter()
+                        .enumerate()
+                    {
+                        for (k, v) in efficiency_values(e).into_iter().enumerate() {
+                            qd[(p * 5 + k) * len + i] = v;
+                        }
+                    }
+                }
+                if greek {
+                    if derivatives {
+                        let (g, d) = sol.greek_with_derivatives(None, norm).map_err(at(i))?;
+                        greeks.push(g);
+                        greeks_d.push(d);
+                    } else {
+                        greeks.push(sol.greek(None, norm).map_err(at(i))?);
+                    }
+                }
+                if let Some(mu) = &mu {
+                    if n_mu > 0 {
+                        let values = rows(&mut f, per_phase, i, n_mu);
+                        if derivatives {
+                            let mut parts = fd.chunks_mut(6 * per_phase);
+                            let mut next = || rows(parts.next().expect("four"), per_phase, i, n_mu);
+                            let d = Derivatives::new(next(), next(), next(), next());
+                            sol.phase_matrix_with_derivatives(mu, norm, values, d)
+                                .map_err(at(i))?;
+                        } else {
+                            sol.phase_matrix(mu, norm, values).map_err(at(i))?;
+                        }
+                    }
+                }
+            }
+            Ok((q, qd, terms, thetas, error, greeks, greeks_d, f, fd))
+        })
+        .map_err(|f| f.into_pyerr(py))?;
+    let (q, qd, terms, thetas, error, greeks, greeks_d, f, fd) = out;
+    let dict = PyDict::new(py);
+    dict.set_item("n_terms", Array1::from(terms).into_pyarray(py))?;
+    dict.set_item("n_theta", Array1::from(thetas).into_pyarray(py))?;
+    dict.set_item("error", Array1::from(error).into_pyarray(py))?;
+    dict.set_item(
+        "efficiencies",
+        Array2::from_shape_vec((5, len), q)
+            .map_err(shape_error)?
+            .into_pyarray(py),
+    )?;
+    if derivatives {
+        dict.set_item(
+            "efficiencies_d",
+            Array3::from_shape_vec((4, 5, len), qd)
+                .map_err(shape_error)?
+                .into_pyarray(py),
+        )?;
+    }
+    if greek {
+        let width = order.map_or_else(
+            || greeks.iter().map(|g| g.alpha1.len()).max().unwrap_or(1),
+            |l| l + 1,
+        );
+        let pack = |gs: &[&GreekCoefficients<f64>]| -> Vec<f64> {
+            let mut v = vec![0.0; 6 * len * width];
+            for (i, g) in gs.iter().enumerate() {
+                for (k, a) in greek_arrays(g).into_iter().enumerate() {
+                    let row = (k * len + i) * width;
+                    let n = a.len().min(width);
+                    v[row..row + n].copy_from_slice(&a[..n]);
+                }
+            }
+            v
+        };
+        let refs: Vec<&GreekCoefficients<f64>> = greeks.iter().collect();
+        dict.set_item(
+            "greek",
+            Array3::from_shape_vec((6, len, width), pack(&refs))
+                .map_err(shape_error)?
+                .into_pyarray(py),
+        )?;
+        if derivatives {
+            let mut all = Vec::with_capacity(4 * 6 * len * width);
+            for p in 0..4 {
+                let refs: Vec<&GreekCoefficients<f64>> = greeks_d
+                    .iter()
+                    .map(|d| [&d.x, &d.aspect_ratio, &d.m_re, &d.m_im][p])
+                    .collect();
+                all.extend(pack(&refs));
+            }
+            dict.set_item(
+                "greek_d",
+                Array4::from_shape_vec((4, 6, len, width), all)
+                    .map_err(shape_error)?
+                    .into_pyarray(py),
+            )?;
+        }
+    }
+    if mu.is_some() {
+        dict.set_item(
+            "phase",
+            Array3::from_shape_vec((6, len, n_mu), f)
+                .map_err(shape_error)?
+                .into_pyarray(py),
+        )?;
+        if derivatives {
+            dict.set_item(
+                "phase_d",
+                Array4::from_shape_vec((4, 6, len, n_mu), fd)
+                    .map_err(shape_error)?
+                    .into_pyarray(py),
+            )?;
+        }
+    }
+    Ok(dict)
+}
+
+/// Values (6, ·) and, if computed, their derivatives (4, 6, ·).
+type WithDerivatives<'py> = (
+    Bound<'py, numpy::PyArray2<f64>>,
+    Option<Bound<'py, numpy::PyArray3<f64>>>,
+);
+
+/// One solved spheroid, for `scatrs.tmatrix.compute`.
+#[pyclass(module = "scatrs._scatrs", name = "TMatrixObject", frozen)]
+pub struct TMatrixObject {
+    inner: TMatrix,
+}
+
+impl TMatrixObject {
+    fn sol(&self) -> TMatrixSolution<'_> {
+        self.inner.solution().expect("solved at construction")
+    }
+}
+
+#[pymethods]
+impl TMatrixObject {
+    #[new]
+    #[pyo3(signature = (x, m_re, m_im, eps, rtol, precision, n_terms, n_theta, allow_unconverged, allow_unvalidated, derivatives))]
+    #[allow(clippy::too_many_arguments)]
+    fn new(
+        py: Python<'_>,
+        x: f64,
+        m_re: f64,
+        m_im: f64,
+        eps: f64,
+        rtol: f64,
+        precision: &str,
+        n_terms: Option<usize>,
+        n_theta: Option<usize>,
+        allow_unconverged: bool,
+        allow_unvalidated: bool,
+        derivatives: bool,
+    ) -> PyResult<Self> {
+        let opts = options(
+            py,
+            rtol,
+            precision,
+            n_terms,
+            n_theta,
+            None,
+            allow_unconverged,
+            allow_unvalidated,
+        )?;
+        let m = Complex64::new(m_re, m_im);
+        let inner = py
+            .detach(|| -> scatrs::Result<TMatrix> {
+                let mut tm = TMatrix::new();
+                solve(&mut tm, eps, x, m, &opts, derivatives)?;
+                Ok(tm)
+            })
+            .map_err(|e| to_pyerr(py, e))?;
+        Ok(Self { inner })
+    }
+
+    #[getter]
+    fn n_terms(&self) -> usize {
+        self.sol().n_terms()
+    }
+
+    #[getter]
+    fn n_theta(&self) -> usize {
+        self.sol().n_theta()
+    }
+
+    #[getter]
+    fn error_estimate(&self) -> f64 {
+        self.sol().error_estimate()
+    }
+
+    #[getter]
+    fn converged(&self) -> bool {
+        self.sol().converged()
+    }
+
+    #[getter]
+    fn precision_paths(&self) -> Vec<&'static str> {
+        self.sol()
+            .precision_paths()
+            .iter()
+            .map(|&p| path_name(p))
+            .collect()
+    }
+
+    /// (orders, T11, T12, T21, T22) of mode `m`, or None above N.
+    #[allow(clippy::type_complexity)]
+    fn block<'py>(
+        &self,
+        py: Python<'py>,
+        m: usize,
+    ) -> PyResult<
+        Option<(
+            Bound<'py, numpy::PyArray1<i64>>,
+            [Bound<'py, numpy::PyArray2<Complex64>>; 4],
+        )>,
+    > {
+        let Some(b) = self.sol().block(m) else {
+            return Ok(None);
+        };
+        let k = b.orders.len();
+        let orders = Array1::from_iter(b.orders.iter().map(|&n| n as i64)).into_pyarray(py);
+        let mat = |v: Vec<Complex64>| -> PyResult<_> {
+            Ok(Array2::from_shape_vec((k, k), v)
+                .map_err(shape_error)?
+                .into_pyarray(py))
+        };
+        Ok(Some((
+            orders,
+            [mat(b.t11)?, mat(b.t12)?, mat(b.t21)?, mat(b.t22)?],
+        )))
+    }
+
+    /// The efficiencies (5,) and, if computed, their derivatives (4, 5).
+    fn efficiencies<'py>(
+        &self,
+        py: Python<'py>,
+    ) -> (
+        Bound<'py, numpy::PyArray1<f64>>,
+        Option<Bound<'py, numpy::PyArray2<f64>>>,
+    ) {
+        let s = self.sol();
+        let v = Array1::from(efficiency_values(&s.efficiencies()).to_vec());
+        let d = s.efficiencies_jacobian().map(|j| {
+            let rows: Vec<f64> = [&j.x, &j.aspect_ratio, &j.m_re, &j.m_im]
+                .into_iter()
+                .flat_map(efficiency_values)
+                .collect();
+            Array2::from_shape_vec((4, 5), rows)
+                .expect("4 x 5")
+                .into_pyarray(py)
+        });
+        (v.into_pyarray(py), d)
+    }
+
+    /// The Greek coefficients (6, L + 1) and, if computed, their derivatives (4, 6, L + 1).
+    #[pyo3(signature = (order, normalized))]
+    fn greek<'py>(
+        &self,
+        py: Python<'py>,
+        order: Option<usize>,
+        normalized: bool,
+    ) -> PyResult<WithDerivatives<'py>> {
+        let s = self.sol();
+        let norm = normalization(normalized);
+        let flat = |g: &GreekCoefficients<f64>| -> Vec<f64> {
+            greek_arrays(g).into_iter().flatten().copied().collect()
+        };
+        if s.efficiencies_jacobian().is_some() {
+            let (g, d) = s
+                .greek_with_derivatives(order, norm)
+                .map_err(|e| to_pyerr(py, e))?;
+            let w = g.alpha1.len();
+            let v = Array2::from_shape_vec((6, w), flat(&g)).map_err(shape_error)?;
+            let all: Vec<f64> = [&d.x, &d.aspect_ratio, &d.m_re, &d.m_im]
+                .into_iter()
+                .flat_map(flat)
+                .collect();
+            let d = Array3::from_shape_vec((4, 6, w), all).map_err(shape_error)?;
+            Ok((v.into_pyarray(py), Some(d.into_pyarray(py))))
+        } else {
+            let g = s.greek(order, norm).map_err(|e| to_pyerr(py, e))?;
+            let w = g.alpha1.len();
+            let v = Array2::from_shape_vec((6, w), flat(&g)).map_err(shape_error)?;
+            Ok((v.into_pyarray(py), None))
+        }
+    }
+
+    /// The phase matrix (6, Q) at the cosines `mu`, and, if computed, its derivatives
+    /// (4, 6, Q).
+    fn phase_matrix<'py>(
+        &self,
+        py: Python<'py>,
+        mu: PyReadonlyArrayDyn<'py, f64>,
+        normalized: bool,
+    ) -> PyResult<WithDerivatives<'py>> {
+        let mu = cosines(py, &mu)?;
+        let q = mu.len();
+        let s = self.sol();
+        let norm = normalization(normalized);
+        let mut f = vec![0.0; 6 * q];
+        if s.efficiencies_jacobian().is_some() {
+            let mut fd = vec![0.0; 24 * q];
+            {
+                let mut parts = fd.chunks_mut(6 * q);
+                let mut next = || rows(parts.next().expect("four"), q, 0, q);
+                let d = Derivatives::new(next(), next(), next(), next());
+                s.phase_matrix_with_derivatives(&mu, norm, rows(&mut f, q, 0, q), d)
+                    .map_err(|e| to_pyerr(py, e))?;
+            }
+            let v = Array2::from_shape_vec((6, q), f).map_err(shape_error)?;
+            let d = Array3::from_shape_vec((4, 6, q), fd).map_err(shape_error)?;
+            Ok((v.into_pyarray(py), Some(d.into_pyarray(py))))
+        } else {
+            s.phase_matrix(&mu, norm, rows(&mut f, q, 0, q))
+                .map_err(|e| to_pyerr(py, e))?;
+            let v = Array2::from_shape_vec((6, q), f).map_err(shape_error)?;
+            Ok((v.into_pyarray(py), None))
+        }
+    }
+}
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
index 6417388..627080e 100644
--- a/scatrs/src/tmatrix/mod.rs
+++ b/scatrs/src/tmatrix/mod.rs
@@ -246,6 +246,11 @@ impl TMatrix {
         self.run(shape, x, m, opts, true)
     }
 
+    /// The last solution, if any.
+    pub fn solution(&self) -> Option<TMatrixSolution<'_>> {
+        self.state.as_ref().map(|s| TMatrixSolution { s })
+    }
+
     fn run(
         &mut self,
         shape: Shape,
@@ -407,6 +412,16 @@ pub struct Derivatives<V> {
 }
 
 impl<V> Derivatives<V> {
+    /// The derivatives with respect to x, ε, Re m and Im m (output buffers, for example).
+    pub fn new(x: V, aspect_ratio: V, m_re: V, m_im: V) -> Self {
+        Self {
+            x,
+            aspect_ratio,
+            m_re,
+            m_im,
+        }
+    }
+
     fn from_fn(mut f: impl FnMut(usize) -> V) -> Self {
         Self {
             x: f(0),
@@ -1060,6 +1075,24 @@ mod tests {
         }
     }
 
+    /// A cap reached first raises `NotConverged`, unless an unconverged result is allowed.
+    #[test]
+    fn caps() {
+        let mut tm = TMatrix::new();
+        let start = driver::start_n(6.0, 2.0);
+        let mut o = TMatrixOptions::new().with_rtol(1e-12);
+        o.n_terms_max = Some(start + 1);
+        let m = Complex::new(1.5, 0.01);
+        assert!(matches!(
+            tm.compute(spheroid(2.0), 6.0, m, &o),
+            Err(Error::NotConverged { .. })
+        ));
+        let s = tm
+            .compute(spheroid(2.0), 6.0, m, &o.with_allow_unconverged(true))
+            .unwrap();
+        assert!(!s.converged() && s.error_estimate() > 1e-12);
+    }
+
     #[test]
     fn invalid_inputs_and_domain() {
         let mut tm = TMatrix::new();
diff --git a/tests/test_package.py b/tests/test_package.py
index d9697ba..d87a6c5 100644
--- a/tests/test_package.py
+++ b/tests/test_package.py
@@ -89,7 +89,9 @@ def test_test_hooks_live_outside_the_extension_namespace():
         "MAX_LAYERS",
         "MAX_MIE_ORDER",
         "MAX_MIE_VALUES",
+        "MAX_TMATRIX_ORDER",
         "MAX_WIGNER_D_VALUES",
+        "TMatrixObject",
         "__version__",
         "ensemble_compute",
         "ensemble_compute_batch",
@@ -106,6 +108,7 @@ def test_test_hooks_live_outside_the_extension_namespace():
         "mie_efficiencies",
         "mie_greek",
         "mie_phase_matrix",
+        "tmatrix_random",
         "wigner_d",
     }
     assert not hasattr(_scatrs, "_raise_core_error")
diff --git a/tests/test_tmatrix.py b/tests/test_tmatrix.py
new file mode 100644
index 0000000..a85a59f
--- /dev/null
+++ b/tests/test_tmatrix.py
@@ -0,0 +1,91 @@
+"""scatrs.tmatrix: shapes, the sphere limit, derivative names, the TMatrix object and errors."""
+
+import numpy as np
+import pytest
+
+from scatrs import (
+    InvalidInputError,
+    OutsideValidatedDomainError,
+    mie,
+    tmatrix,
+)
+
+MU = np.array([-1.0, -0.3, 0.4, 1.0])
+
+
+def test_shapes_broadcast():
+    x = np.array([1.0, 2.5]).reshape(2, 1)
+    eps = np.array([0.5, 1.0, 2.0])
+    q = tmatrix.efficiencies(x, 1.5 + 0.01j, eps)
+    assert q.q_ext.shape == (2, 3) and q.n_terms.shape == (2, 3) and q.error.shape == (2, 3)
+    g = tmatrix.greek(x, 1.5 + 0.01j, eps, 6)
+    assert g.alpha1.shape == (2, 3, 7) and g.n_terms.shape == (2, 3)
+    f = tmatrix.phase_matrix(x, 1.5 + 0.01j, eps, MU.reshape(2, 2))
+    assert f.f22.shape == (2, 3, 2, 2) and f.f44.shape == (2, 3, 2, 2)
+
+
+def test_an_aspect_ratio_of_one_is_the_sphere():
+    x, m = 3.0, 1.5 + 0.01j
+    q, q0 = tmatrix.efficiencies(x, m, 1.0, rtol=1e-10), mie.efficiencies(x, m)
+    for name in ("q_ext", "q_sca", "q_abs", "q_back", "g"):
+        assert getattr(q, name) == pytest.approx(getattr(q0, name), rel=1e-9), name
+    f, f0 = tmatrix.phase_matrix(x, m, 1.0, MU, rtol=1e-10), mie.phase_matrix(x, m, MU)
+    scale = f0.f11.max()
+    for a, b in [(f.f11, f0.f11), (f.f22, f0.f11), (f.f33, f0.f33), (f.f44, f0.f33)]:
+        np.testing.assert_allclose(a, b, atol=1e-9 * scale)
+
+
+def test_oblate_and_prolate_differ():
+    q = tmatrix.efficiencies(3.0, 1.5, [0.5, 2.0])
+    assert abs(q.q_ext[0] - q.q_ext[1]) > 1e-3
+
+
+def test_derivative_names():
+    q = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0, derivatives=["aspect_ratio", "m_imag"])
+    assert q.params == ("aspect_ratio", "m_imag")
+    assert np.isfinite(q.d("aspect_ratio").q_ext)
+    with pytest.raises(ValueError, match="no derivative"):
+        q.d("x")
+    with pytest.raises(ValueError, match="real"):
+        q.d("m")
+    with pytest.raises(ValueError, match="unknown parameter"):
+        tmatrix.efficiencies(2.0, 1.5, 2.0, derivatives=["radius"])
+    g = tmatrix.greek(2.0, 1.5, 2.0, 4, derivatives=True)
+    assert g.d("x").alpha1.shape == (5,)
+    f = tmatrix.phase_matrix(2.0, 1.5, 2.0, MU, derivatives=True)
+    assert f.d("m_real").f22.shape == (4,)
+
+
+def test_derivative_matches_a_difference():
+    h = 1e-5
+    kw = {"n_terms": 12, "n_theta": 30}
+    q = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0, derivatives=True, **kw)
+    qp = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0 + h, **kw)
+    qm = tmatrix.efficiencies(2.0, 1.5 + 0.01j, 2.0 - h, **kw)
+    fd = (qp.q_sca - qm.q_sca) / (2 * h)
+    assert q.d("aspect_ratio").q_sca == pytest.approx(fd, rel=1e-6)
+
+
+def test_the_tmatrix_object():
+    t = tmatrix.compute(3.0, 1.33, 2.0, derivatives=True)
+    assert t.converged and t.error_estimate < 1e-8 and len(t.precision_paths) == t.n_terms + 1
+    b = t.block(2)
+    assert b.t11.shape == (t.n_terms - 1, t.n_terms - 1) and b.orders[0] == 2
+    with pytest.raises(IndexError):
+        t.block(t.n_terms + 1)
+    q, g, f = t.random_orientation(MU, 6)
+    q1 = tmatrix.efficiencies(3.0, 1.33, 2.0)
+    assert q.q_ext == pytest.approx(q1.q_ext, rel=1e-12)
+    assert g.alpha1.shape == (7,) and f is not None and f.f11.shape == (4,)
+    assert np.isfinite(q.d("x").q_ext)
+
+
+def test_errors():
+    with pytest.raises(InvalidInputError):
+        tmatrix.efficiencies(1.0, 1.5, 0.0)
+    with pytest.raises(InvalidInputError, match="element 1"):
+        tmatrix.efficiencies([1.0, -1.0], 1.5, 2.0)
+    with pytest.raises(OutsideValidatedDomainError):
+        tmatrix.efficiencies(1.0, 1.5, 9.0)
+    with pytest.raises(ValueError, match="precision"):
+        tmatrix.efficiencies(1.0, 1.5, 2.0, precision="quad")
````

### Task 7: Arb references

- [ ] `generators/spheroid.py`: the EBCM from its definition in python-flint, preconditioned solve, fixed and converged records (generate: about 32 minutes).
- [ ] `scatrs/tests/spheroid_reference.rs`: every path at the fixed truncations within 4× the estimate; converged efficiencies within the declared tolerance.
- [ ] The reference-suite page.

The commit (`af724eb`):

````diff
diff --git a/docs/theory/reference-suite.md b/docs/theory/reference-suite.md
index 38ffeb9..420a912 100644
--- a/docs/theory/reference-suite.md
+++ b/docs/theory/reference-suite.md
@@ -83,6 +83,7 @@ coefficient records are all certified. {doc}`/theory/mie` derives the bound.
 | `mie.jsonl` | 612 | see below |
 | `mie_derivatives.jsonl` | 160 | see below |
 | `mie_greek.jsonl` | 36 | see below |
+| `spheroid.jsonl` | 20 | see below |
 | `wigner_d.jsonl` | 212 | see below |
 
 The edge cases sit at the edges of each function's domain and of its algorithm's branches:
@@ -129,6 +130,18 @@ coefficients come from Gauss–Legendre quadrature on Arb's own Legendre roots,
 polynomial integrands, and the Wigner d-functions from their three-term recurrence in ball
 arithmetic, so every value is certified.
 
+**`spheroid.jsonl`** holds two records per spheroid, for 10 spheroids with aspect ratios from
+1/5 to 5 (a sphere among them), equal-volume size parameters from 0.5 to 6, and indices from
+water to 3 + 0.1i, computed without scatrs's algorithm: the generator evaluates the EBCM from
+its definition, the surface integrals of the cross products of the vector spherical wave
+functions with no simplification of the integrands, on Arb's own Gauss–Legendre rule over the
+whole range of $\cos\theta$, and solves for $T^m$ in ball arithmetic. A `,N=…,n_theta=…` record
+holds $Q_{\mathrm{ext}}$ and $Q_{\mathrm{sca}}$ at a given truncation and a sample of $T^m$ for
+three modes (the diagonals of $T^{11}$ and $T^{22}$ and the first column of each block),
+certified for that truncation; a `,converged` record holds $Q_{\mathrm{ext}}$,
+$Q_{\mathrm{sca}}$ and $Q_{\mathrm{abs}}$ with $N$ and the quadrature grown until they settle
+to $10^{-20}$, high-precision.
+
 **`layered.jsonl`** holds two records per layered sphere, for 370 particles, computed without
 scatrs's algorithm: the generator evaluates $\psi_n$ and $\xi_n$ of every layer's arguments
 and matches the fields at each interface directly, all in ball arithmetic. The particles are
@@ -144,7 +157,7 @@ of the coefficients and the efficiencies with respect to every layer's $x_l$, $\
 and $\mathrm{Im}\,m_l$, from five-point central differences in ball arithmetic
 (high-precision).
 
-The committed data take about 17.5 MB, within the 20 MB budget, which a test enforces.
+The committed data take about 18 MB, within the 20 MB budget, which a test enforces.
 
 ## How the tests use it
 
diff --git a/reference/scatrs_reference/generators/__init__.py b/reference/scatrs_reference/generators/__init__.py
index f4e00a2..fc3a118 100644
--- a/reference/scatrs_reference/generators/__init__.py
+++ b/reference/scatrs_reference/generators/__init__.py
@@ -13,6 +13,7 @@ from scatrs_reference.generators import (
     layered,
     mie,
     mie_greek,
+    spheroid,
     wigner_d,
 )
 
@@ -31,5 +32,6 @@ GENERATORS: dict[str, Generator] = {
     "mie": mie.generate,
     "mie_derivatives": mie.generate_derivatives,
     "mie_greek": mie_greek.generate,
+    "spheroid": spheroid.generate,
     "wigner_d": wigner_d.generate,
 }
diff --git a/reference/scatrs_reference/generators/spheroid.py b/reference/scatrs_reference/generators/spheroid.py
new file mode 100644
index 0000000..b9b951a
--- /dev/null
+++ b/reference/scatrs_reference/generators/spheroid.py
@@ -0,0 +1,300 @@
+"""Randomly oriented spheroids: the T-matrix by the EBCM from its definition, in Arb.
+
+For each azimuthal mode m the matrices Q and RgQ are the surface integrals of the cross
+products of the vector spherical wave functions (Mishchenko, Travis & Lacis 2002, §5.8; the
+conventions of ``research/tmatrix/ebcm.py``), evaluated component by component with no
+simplification of the integrands, on Arb's own Gauss–Legendre rule over the whole range
+-1 <= cos θ <= 1 (``arb.legendre_p_root``); the spherical Bessel functions come from
+``acb.bessel_j`` and ``acb.bessel_y``. Tᵐ = −RgQ Q⁻¹ is solved by ``acb_mat.solve``. The
+orientation averages are Mishchenko's (1991):
+⟨C_ext⟩ = −2π Σ_m (2 − δ_m0) Σ_n Re(T¹¹_nn + T²²_nn), ⟨C_sca⟩ = 2π Σ_m (2 − δ_m0) Σ |Tᵐ|²
+(k = 1), divided by πx² for the efficiencies.
+
+Two kinds of records per case (x, ε, m):
+
+* ``converged``: Q_ext, Q_sca and Q_abs = Q_ext − Q_sca with N and the number of nodes grown
+  until both change by less than 10⁻²⁰ relative. Labelled "high-precision": the truncation in
+  N is not rigorously bounded (the ball radii bound the rounding only).
+* ``fixed``: at a given N and 2n_θ nodes (the kernel's n_θ in a half range), Q_ext and Q_sca and
+  a sample of Tᵐ for modes 0, 1 and N/2 (the diagonals of T¹¹ and T²², and the first column of
+  each block), which isolate the solve from the convergence. Labelled "certified": for that
+  truncation, the ball radii bound everything.
+
+The working precision is ``PRECISION_BITS`` (Q's elements cancel by up to 10¹⁶, and its
+conditioning reaches 10¹⁵ in the cases here), doubled for a case until every stored radius is
+below 2^-``REL_BITS`` of its scale.
+"""
+
+from __future__ import annotations
+
+from flint import acb, acb_mat, arb, ctx
+
+from scatrs_reference.records import PORTABLE_BITS, nearest_double, record
+
+GENERATOR = "spheroid"
+PRECISION_BITS = 480
+REL_BITS = 120
+MAX_PRECISION_BITS = 1920
+SETTLE = arb("1e-20")
+
+#: (x, ε, Re m, Im m, fixed N, fixed n_θ in a half range).
+CASES = (
+    (0.5, 2.0, 1.5, 0.01, 6, 12),
+    (1.0, 0.5, 1.33, 0.0, 8, 16),
+    (2.0, 1.8, 1.5, 0.02, 10, 24),
+    (2.0, 0.2, 1.53, 0.008, 14, 40),
+    (3.0, 5.0, 1.31, 0.0, 14, 48),
+    (3.0, 1.0, 1.5, 0.01, 12, 20),
+    (4.0, 3.0, 1.53, 0.008, 16, 40),
+    (4.0, 1 / 3, 1.5, 0.5, 18, 44),
+    (5.0, 2.0, 3.0, 0.1, 26, 60),
+    (6.0, 0.5, 1.33, 0.0, 20, 44),
+)
+
+
+def gauss_legendre(n: int) -> tuple[list[arb], list[arb]]:
+    """Nodes and weights of the n-point rule on [-1, 1]."""
+    nodes, weights = [], []
+    for k in range(n):
+        x, w = arb.legendre_p_root(n, k, weight=True)
+        nodes.append(x)
+        weights.append(w)
+    return nodes, weights
+
+
+def shape(x: arb, eps: arb):
+    """r(θ) and dr/dθ of the spheroid of equal-volume size parameter x and aspect ratio ε."""
+    a = x * eps ** (arb(1) / 3)
+    c = a / eps
+
+    def at(ct: arb, st: arb) -> tuple[arb, arb]:
+        den = c * c * st * st + a * a * ct * ct
+        r = a * c / den.sqrt()
+        dr = -a * c * (c * c - a * a) * st * ct / (den * den.sqrt())
+        return r, dr
+
+    return at
+
+
+def angular(m: int, nmax: int, ct: arb, st: arb) -> tuple[list[arb], list[arb], list[arb]]:
+    """Normalized d_mn, π_mn = m d/sin θ and τ_mn = dd/dθ for n = 0..nmax (m >= 0)."""
+    zero = arb(0)
+    d = [zero] * (nmax + 1)
+    if m <= nmax:
+        am = arb(1)
+        for k in range(1, m + 1):
+            am *= (arb(2 * k - 1) / (2 * k)).sqrt()
+        d[m] = am * st**m
+        if m + 1 <= nmax:
+            d[m + 1] = arb(2 * m + 1).sqrt() * ct * d[m]
+        for n in range(m + 1, nmax):
+            d[n + 1] = ((2 * n + 1) * ct * d[n] - arb(n * n - m * m).sqrt() * d[n - 1]) / arb(
+                (n + 1) ** 2 - m * m
+            ).sqrt()
+    tau = [zero] * (nmax + 1)
+    pi_ = [zero] * (nmax + 1)
+    for n in range(max(m, 1), nmax + 1):
+        prev = d[n - 1] if n - 1 >= m else zero
+        tau[n] = (n * ct * d[n] - arb(n * n - m * m).sqrt() * prev) / st
+        pi_[n] = m * d[n] / st
+    return d, pi_, tau
+
+
+def bessels(z: acb, nmax: int, outgoing: bool) -> tuple[list[acb], list[acb]]:
+    """z_n(z) and [z z_n(z)]'/z for n = 0..nmax: j_n, or h_n = j_n + i y_n if ``outgoing``."""
+    pre = (acb.pi() / (2 * z)).sqrt()
+    half = arb(1) / 2
+    zn = []
+    for n in range(nmax + 1):
+        v = pre * z.bessel_j(n + half)
+        if outgoing:
+            v += 1j * pre * z.bessel_y(n + half)
+        zn.append(v)
+    # z_{-1}: j_{-1} = cos z/z, y_{-1} = sin z/z.
+    zm1 = (z.cos() + 1j * z.sin()) / z if outgoing else z.cos() / z
+    der = [(zm1 if n == 0 else zn[n - 1]) - n * zn[n] / z for n in range(nmax + 1)]
+    return zn, der
+
+
+def vswf(m, nmax, z, outgoing, d, pi_, tau, flip):
+    """(r, θ, φ) components of M_mn and N_mn (φ factor dropped); with ``flip``, π → −π (the
+    (−1)^m X_{−mn} of the integrals)."""
+    zn, der = bessels(z, nmax, outgoing)
+    out_m, out_n = [None] * (nmax + 1), [None] * (nmax + 1)
+    sgn = -1 if flip else 1
+    for n in range(max(m, 1), nmax + 1):
+        g = (arb(2 * n + 1) / (4 * arb.pi() * n * (n + 1))).sqrt()
+        p = sgn * pi_[n]
+        out_m[n] = (acb(0), g * zn[n] * 1j * p, -g * zn[n] * tau[n])
+        out_n[n] = (
+            g * n * (n + 1) * zn[n] / z * d[n],
+            g * der[n] * tau[n],
+            g * der[n] * 1j * p,
+        )
+    return out_m, out_n
+
+
+def tmatrix_mode(x, eps, s, m, nmax, nodes, weights) -> list[list[acb]]:
+    """Tᵐ (2K × 2K, blocks [[T¹¹, T¹²], [T²¹, T²²]], K = nmax − max(m, 1) + 1)."""
+    at = shape(x, eps)
+    n0 = max(m, 1)
+    kk = nmax - n0 + 1
+    acc = {
+        (a, b, h): [[acb(0)] * kk for _ in range(kk)]
+        for a in (1, 2)
+        for b in (1, 2)
+        for h in (True, False)
+    }
+    for ct, w in zip(nodes, weights, strict=True):
+        st = (1 - ct * ct).sqrt()
+        r, dr = at(ct, st)
+        nv = -dr / r
+        fac = w * r * r * 2 * arb.pi()
+        d, pi_, tau = angular(m, nmax, ct, st)
+        inner = vswf(m, nmax, s * r, False, d, pi_, tau, flip=False)
+        for out in (True, False):
+            outer = vswf(m, nmax, acb(r), out, d, pi_, tau, flip=True)
+            for a, xs in ((1, outer[0]), (2, outer[1])):
+                for b, ys in ((1, inner[0]), (2, inner[1])):
+                    mat = acc[(a, b, out)]
+                    for i in range(n0, nmax + 1):
+                        A = xs[i]
+                        row = mat[i - n0]
+                        for j in range(n0, nmax + 1):
+                            B = ys[j]
+                            trip = (A[1] * B[2] - A[2] * B[1]) + nv * (A[2] * B[0] - A[0] * B[2])
+                            row[j - n0] += fac * trip
+
+    def assemble(out: bool) -> acb_mat:
+        J = {(a, b): acc[(a, b, out)] for a in (1, 2) for b in (1, 2)}
+        q = acb_mat(2 * kk, 2 * kk)
+        mi = acb(0, -1)
+        for i in range(kk):
+            for j in range(kk):
+                q[i, j] = mi * (s * J[1, 2][i][j] + J[2, 1][i][j])
+                q[i, kk + j] = mi * (s * J[1, 1][i][j] + J[2, 2][i][j])
+                q[kk + i, j] = mi * (s * J[2, 2][i][j] + J[1, 1][i][j])
+                q[kk + i, kk + j] = mi * (s * J[2, 1][i][j] + J[1, 2][i][j])
+        return q
+
+    q, rgq = assemble(True), assemble(False)
+    # T = −RgQ Q⁻¹: Qᵀ Tᵀ = −RgQᵀ.
+    t = q.transpose().solve(-rgq.transpose(), algorithm="precond").transpose()
+    return [[t[i, j] for j in range(2 * kk)] for i in range(2 * kk)]
+
+
+def efficiencies(x, eps, s, nmax, n_nodes) -> tuple[arb, arb, list[list[list[acb]]]]:
+    """Q_ext, Q_sca and every Tᵐ (m = 0..nmax) at N = nmax and n_nodes nodes."""
+    nodes, weights = gauss_legendre(n_nodes)
+    ext, sca = arb(0), arb(0)
+    ts = []
+    for m in range(nmax + 1):
+        t = tmatrix_mode(x, eps, s, m, nmax, nodes, weights)
+        ts.append(t)
+        w = 1 if m == 0 else 2
+        size = len(t)
+        for i in range(size):
+            ext -= w * t[i][i].real
+            for v in t[i]:
+                sca += w * (v.real * v.real + v.imag * v.imag)
+    area = arb.pi() * x * x
+    two_pi = 2 * arb.pi()
+    return two_pi * ext / area, two_pi * sca / area, ts
+
+
+def start_n(x: float, eps: float) -> int:
+    """The starting N: x_max + 4 x_max^(1/3) + 3 (through Arb, for portability)."""
+    with ctx.workprec(PORTABLE_BITS):
+        a = arb(x) * arb(eps) ** (arb(1) / 3)
+        xm = max(a, a / arb(eps), key=lambda v: v.mid())
+        return max(4, int(nearest_double(xm + 4 * xm ** (arb(1) / 3))) + 3)
+
+
+def converged(x, eps, s, xf: float, ef: float):
+    """Q_ext and Q_sca with N, then the nodes, grown until they settle to SETTLE."""
+    n = start_n(xf, ef)
+    nodes = 2 * (int(n * (1 + abs(_ln(ef)))) + 8)
+    e0, c0, _ = efficiencies(x, eps, s, n, nodes)
+    while True:
+        n2 = n + max(2, n // 6)
+        nodes2 = 2 * (int(n2 * (1 + abs(_ln(ef)))) + 8)
+        e1, c1, _ = efficiencies(x, eps, s, n2, nodes2)
+        n, nodes = n2, nodes2
+        done = abs(e1 - e0) < SETTLE * abs(e1) and abs(c1 - c0) < SETTLE * abs(c1)
+        e0, c0 = e1, c1
+        if done:
+            break
+    while True:
+        nodes2 = nodes + max(8, nodes // 4)
+        e1, c1, _ = efficiencies(x, eps, s, n, nodes2)
+        nodes = nodes2
+        done = abs(e1 - e0) < SETTLE * abs(e1) and abs(c1 - c0) < SETTLE * abs(c1)
+        e0, c0 = e1, c1
+        if done:
+            break
+    return e0, c0, n, nodes
+
+
+def _ln(v: float) -> float:
+    """|ln v| as a double, through Arb (portable: see ``records.nearest_pow10``)."""
+    with ctx.workprec(PORTABLE_BITS):
+        return nearest_double(arb(v).log())
+
+
+def _relative_ok(vals: list[arb], scale: arb) -> bool:
+    bound = scale * arb(2) ** (-REL_BITS)
+    return all(v.rad() < bound for v in vals)
+
+
+def case_records(x: float, eps: float, m_re: float, m_im: float, nf: int, thf: int) -> list[dict]:
+    name = f"x={x!r},eps={eps!r},m=({m_re!r},{m_im!r})"
+    base = {"x": x, "aspect_ratio": eps, "m_re": m_re, "m_im": m_im}
+    prec = PRECISION_BITS
+    while True:
+        with ctx.workprec(prec):
+            xa, ea, s = arb(x), arb(eps), acb(m_re, m_im)
+            ef, sf, ts = efficiencies(xa, ea, s, nf, 2 * thf)
+            sample: dict[str, list[arb]] = {}
+            for m in sorted({0, 1, nf // 2}):
+                t = ts[m]
+                kk = len(t) // 2
+                diag11 = [t[i][i] for i in range(kk)]
+                diag22 = [t[kk + i][kk + i] for i in range(kk)]
+                col = [t[i][0] for i in range(2 * kk)] + [t[i][kk] for i in range(2 * kk)]
+                for label, vs in (("diag11", diag11), ("diag22", diag22), ("col", col)):
+                    sample[f"t{m}_{label}_re"] = [v.real for v in vs]
+                    sample[f"t{m}_{label}_im"] = [v.imag for v in vs]
+            qe, qs, n_conv, nodes_conv = converged(xa, ea, s, x, eps)
+            scale = max(
+                (abs(v.real) + abs(v.imag) for t in ts for row in t for v in row),
+                key=lambda a: a.mid(),
+            )
+            ok = _relative_ok([ef, sf, qe, qs], qe) and _relative_ok(
+                [v for vs in sample.values() for v in vs], scale
+            )
+            if ok or prec >= MAX_PRECISION_BITS:
+                if not ok:
+                    raise RuntimeError(f"{name}: radii too wide at {prec} bits")
+                fixed = record(
+                    GENERATOR,
+                    name + f",N={nf},n_theta={thf}",
+                    {**base, "n_terms": nf, "n_theta": thf},
+                    {"q_ext": [ef], "q_sca": [sf], **sample},
+                    label="certified",
+                )
+                conv = record(
+                    GENERATOR,
+                    name + ",converged",
+                    {**base, "n_terms": n_conv, "n_theta": nodes_conv // 2},
+                    {"q_ext": [qe], "q_sca": [qs], "q_abs": [qe - qs]},
+                    label="high-precision",
+                )
+                return [fixed, conv]
+        prec *= 2
+
+
+def generate() -> tuple[int, list[dict]]:
+    records: list[dict] = []
+    for case in CASES:
+        records.extend(case_records(*case))
+    return PRECISION_BITS, records
diff --git a/scatrs/tests/spheroid_reference.rs b/scatrs/tests/spheroid_reference.rs
new file mode 100644
index 0000000..541b1ba
--- /dev/null
+++ b/scatrs/tests/spheroid_reference.rs
@@ -0,0 +1,186 @@
+//! Randomly oriented spheroids against the Arb reference values in `spheroid.jsonl`, from an
+//! EBCM computed from its definition with no simplification (no code or algorithm shared with
+//! the kernel).
+//!
+//! * `fixed` records, at a given N and n_θ, isolate the solve: on every precision path the
+//!   efficiencies and the sampled T-matrix elements must be within C times the solution's own
+//!   error estimate (relative to the largest element of T), C = 4, and at least within a few
+//!   rounding units.
+//! * `converged` records check the convergence loop: Q_ext, Q_sca and Q_abs within the declared
+//!   tolerance C·max(`rtol`, estimate) of `tmatrix::VALIDATED_DOMAIN` (Q_abs relative to Q_ext).
+//!
+//! Each test prints the worst ratio of error to tolerance it measures.
+
+mod common;
+
+use common::Record;
+use common::check::{error, records};
+use num_complex::Complex;
+use scatrs::mie::MieOutput;
+use scatrs::tmatrix::{Precision, Shape, TMatrix, TMatrixOptions, VALIDATED_DOMAIN};
+
+const FILE: &str = "spheroid.jsonl";
+
+fn inputs(r: &Record) -> (f64, f64, Complex<f64>) {
+    (
+        r.input_f64("aspect_ratio"),
+        r.input_f64("x"),
+        Complex::new(r.input_f64("m_re"), r.input_f64("m_im")),
+    )
+}
+
+struct Worst {
+    what: String,
+    ratio: f64,
+    case: String,
+}
+
+impl Worst {
+    fn new(what: &str) -> Self {
+        Self {
+            what: what.into(),
+            ratio: 0.0,
+            case: String::new(),
+        }
+    }
+
+    fn check(&mut self, err: f64, tol: f64, case: &str) {
+        let ratio = err / tol;
+        if ratio > self.ratio {
+            self.ratio = ratio;
+            self.case = case.into();
+        }
+        assert!(
+            err <= tol,
+            "{} {case}: error {err:.3e} exceeds {tol:.3e}",
+            self.what
+        );
+    }
+}
+
+impl Drop for Worst {
+    fn drop(&mut self) {
+        if !std::thread::panicking() {
+            eprintln!(
+                "{:<40} worst error/tolerance = {:.3} at {}",
+                self.what, self.ratio, self.case
+            );
+        }
+    }
+}
+
+#[test]
+fn fixed_truncations_on_every_path() {
+    for (path, name) in [
+        (Precision::F64, "f64"),
+        (Precision::Auto, "auto"),
+        (Precision::DoubleDouble, "double-double"),
+    ] {
+        let mut q_worst = Worst::new(&format!("fixed Q, {name}"));
+        let mut t_worst = Worst::new(&format!("fixed T, {name}"));
+        for r in common::load(FILE)
+            .into_iter()
+            .filter(|r| r.case.contains(",N="))
+        {
+            let (eps, x, m) = inputs(&r);
+            assert!(VALIDATED_DOMAIN.contains(eps, x, m), "{} outside", r.case);
+            let n = r.input_i64("n_terms") as usize;
+            let nth = r.input_i64("n_theta") as usize;
+            let opts = TMatrixOptions::new()
+                .with_precision(path)
+                .with_n_terms(n)
+                .with_n_theta(nth);
+            let mut tm = TMatrix::new();
+            let s = tm
+                .compute(Shape::Spheroid { aspect_ratio: eps }, x, m, &opts)
+                .unwrap();
+            // The estimate, with a floor of a few rounding units.
+            let est = s.error_estimate().max(16.0 * f64::EPSILON);
+            let q = s.efficiencies();
+            for (v, out) in [(q.q_ext, "q_ext"), (q.q_sca, "q_sca")] {
+                let e = error(v, &r, out, 0);
+                let scale = r.output(out).hi[0].abs();
+                q_worst.check(e, 4.0 * est * scale, &format!("{} {out}", r.case));
+            }
+            // T: relative to the largest element of T.
+            let big = (0..=n)
+                .filter_map(|mm| s.block(mm))
+                .flat_map(|b| {
+                    [b.t11, b.t12, b.t21, b.t22]
+                        .into_iter()
+                        .flatten()
+                        .map(|v| v.norm())
+                        .collect::<Vec<_>>()
+                })
+                .fold(0.0f64, f64::max);
+            for mm in [0, 1, n / 2] {
+                let b = s.block(mm).unwrap();
+                let k = b.orders.len();
+                let at = |i: usize, j: usize| -> Complex<f64> {
+                    let (blk, ii, jj) = match (i < k, j < k) {
+                        (true, true) => (&b.t11, i, j),
+                        (true, false) => (&b.t12, i, j - k),
+                        (false, true) => (&b.t21, i - k, j),
+                        (false, false) => (&b.t22, i - k, j - k),
+                    };
+                    blk[ii * k + jj]
+                };
+                let mut check = |stem: &str, got: &[Complex<f64>]| {
+                    for (i, v) in got.iter().enumerate() {
+                        let e = error(v.re, &r, &format!("t{mm}_{stem}_re"), i).hypot(error(
+                            v.im,
+                            &r,
+                            &format!("t{mm}_{stem}_im"),
+                            i,
+                        ));
+                        t_worst.check(
+                            e,
+                            4.0 * est * big,
+                            &format!("{} mode {mm} {stem} {i}", r.case),
+                        );
+                    }
+                };
+                check("diag11", &(0..k).map(|i| at(i, i)).collect::<Vec<_>>());
+                check(
+                    "diag22",
+                    &(0..k).map(|i| at(k + i, k + i)).collect::<Vec<_>>(),
+                );
+                let col: Vec<Complex<f64>> = (0..2 * k)
+                    .map(|i| at(i, 0))
+                    .chain((0..2 * k).map(|i| at(i, k)))
+                    .collect();
+                check("col", &col);
+            }
+        }
+    }
+}
+
+#[test]
+fn converged_efficiencies() {
+    let mut worst = Worst::new("converged Q_ext, Q_sca, Q_abs");
+    for r in records(FILE, ",converged") {
+        let (eps, x, m) = inputs(&r);
+        let opts = TMatrixOptions::new();
+        let mut tm = TMatrix::new();
+        let s = tm
+            .compute(Shape::Spheroid { aspect_ratio: eps }, x, m, &opts)
+            .unwrap();
+        let est = s.error_estimate();
+        let q = s.efficiencies();
+        let q_ext = r.output("q_ext").hi[0];
+        for (v, out, kind, scale) in [
+            (q.q_ext, "q_ext", MieOutput::Extinction, q_ext),
+            (
+                q.q_sca,
+                "q_sca",
+                MieOutput::Scattering,
+                r.output("q_sca").hi[0],
+            ),
+            (q.q_abs, "q_abs", MieOutput::Absorption, q_ext),
+        ] {
+            let e = error(v, &r, out, 0);
+            let tol = VALIDATED_DOMAIN.tolerance(kind, opts.rtol, est) * scale.abs();
+            worst.check(e, tol, &format!("{} {out}", r.case));
+        }
+    }
+}
````

### Task 8: The accuracy map and the validated domain

- [ ] `scatrs/tests/tmatrix_accuracy_map.rs` (ignored; about 96 minutes) with the JSON for the docs.
- [ ] Set `VALIDATED_DOMAIN` and the constants from it; the accuracy page section and its plot.

The commit (`384f336`):

````diff
diff --git a/docs/limitations.md b/docs/limitations.md
index f9d683e..aa2a3ad 100644
--- a/docs/limitations.md
+++ b/docs/limitations.md
@@ -53,9 +53,10 @@ describes what the accuracy figures rest on.
 - **Random orientation only.** Fixed orientations, other axisymmetric shapes (cylinders,
   Chebyshev particles) and spheroids in size distributions and mixtures are not yet
   available.
-- **Validated domain.** Aspect ratios from 1/5 to 5, $|m|\,x_{\max} \le 60$ with $x_{\max}$ the
-  size parameter of the longest semi-axis, $1 \le \mathrm{Re}\,m \le 3$,
-  $\mathrm{Im}\,m \le 1$, and `rtol` down to $10^{-12}$ ({doc}`validation/accuracy`).
+- **Validated domain.** Aspect ratios from 1/5 to 5, equal-volume $x \ge 0.3$,
+  $|m|\,x_{\max} \le 60$ with $x_{\max}$ the size parameter of the longest semi-axis,
+  $1.3 \le \mathrm{Re}\,m \le 3$, $\mathrm{Im}\,m \le 0.5$, and `rtol` down to $10^{-12}$
+  ({doc}`validation/accuracy`). The edge in $|m|\,x_{\max}$ is set by cost, not accuracy.
 - **Cost.** It grows like $x_{\max}^4$, and much faster where the modes need mixed-precision
   refinement, which is most often for prolate spheroids and at the largest $|m|\,x_{\max}$.
   At `rtol = 1e-8` an oblate spheroid with $\varepsilon = 2$ takes about 25 ms at $x = 5$ and
diff --git a/docs/validation/accuracy.md b/docs/validation/accuracy.md
index f19d4a0..1bd26b1 100644
--- a/docs/validation/accuracy.md
+++ b/docs/validation/accuracy.md
@@ -256,6 +256,65 @@ with respect to the shell's index would exceed its constant by 1.7 times at $x_L
 rounding errors of the recurrences at the shell's two surfaces are alike from one order to the
 next, so they add up in the forward sum, and the derivatives' domain ends at 1000.
 
+## Spheroids
+
+<!-- synced with scatrs/src/tmatrix/domain.rs rustdoc -->
+
+The outputs of {mod}`scatrs.tmatrix` are converged to a tolerance `rtol` and come with an
+error estimate $\hat\varepsilon$ ({doc}`/theory/spheroids`). Inside the validated domain
+(aspect ratios from 1/5 to 5, equal-volume $x \ge 0.3$, $|m|\,x_{\max} \le 60$ with $x_{\max}$
+the size parameter of the longest semi-axis, $1.3 \le \mathrm{Re}\,m \le 3$,
+$\mathrm{Im}\,m \le 0.5$), every output is within $C\cdot\max(\mathtt{rtol}, \hat\varepsilon)$
+of the exact value, in the measures of the sphere's outputs ($Q_{\mathrm{abs}}$ relative to
+$Q_{\mathrm{ext}}$, the Greek coefficients relative to the largest $|\alpha_1|$, the phase matrix
+relative to $\max(F_{11}, 1)$, normalized). Two tests enforce it:
+
+- `scatrs/tests/spheroid_reference.rs` compares with Arb reference values of an EBCM computed
+  from its definition, with no simplification of the integrands ({doc}`/theory/reference-suite`),
+  for 10 spheroids. At a fixed truncation, the efficiencies and the T-matrix elements must be
+  within 4 times the solution's own estimate on each of the three precision paths (`f64`,
+  refined and double-double); the converged efficiencies must be within the declared
+  tolerance.
+- `scatrs/tests/tmatrix_accuracy_map.rs` compares the default results (`rtol = 1e-8`) with a
+  reference using more orders and nodes and every mode refined, for 221 spheroids: 8 aspect
+  ratios, $x$ from 0.3 to 14 and four indices, up to $|m|\,x_{\max} = 70$.
+
+The worst ratio of each error to $\max(\mathtt{rtol}, \hat\varepsilon)$ inside the domain, against
+the declared constant:
+
+```{code-cell} python
+data = json.loads(Path("data", "spheroid-accuracy-map.json").read_text())
+points = data["points"]
+ratios = np.array([p["ratios"] for p in points])
+inside = np.array([p["inside"] for p in points])
+print(f"{len(points)} spheroids, {inside.sum()} inside the domain")
+print(f"{'output':<10}{'worst inside':>14}{'worst outside':>15}{'declared C':>12}")
+for k, output in enumerate(data["outputs"]):
+    worst_in, worst_out = ratios[inside, k].max(), ratios[~inside, k].max()
+    print(f"{output:<10}{worst_in:14.3f}{worst_out:15.3f}{data['declared'][output]:12.0f}")
+    assert 4 * worst_in <= data["declared"][output], f"{output}: margin below 4"
+```
+
+```{code-cell} python
+fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
+phase = np.array([p["phase"] for p in points])
+for k, output in enumerate(data["outputs"]):
+    ax.semilogy(phase, np.maximum(ratios[:, k], 1e-12), ".", ms=4, label=output)
+ax.axvline(60, color="k", lw=0.8, ls="--")
+ax.set(
+    xlabel=r"$|m|\,x_{\max}$",
+    ylabel=r"error / max(rtol, estimate)",
+    ylim=(1e-8, 10),
+)
+ax.legend(fontsize=7, ncols=2)
+plt.show()
+```
+
+The estimates are conservative: the cross sections and $g$ stay below a twentieth of
+$\max(\mathtt{rtol}, \hat\varepsilon)$, and only $Q_{\mathrm{back}}$ reaches it (1.04 times, at
+$\varepsilon = 1/3$, $x = 4$), well within its constant. Past $|m|\,x_{\max} = 60$ the results are as accurate, but
+the modes that need refinement make one spheroid cost minutes, so the domain ends there.
+
 (ensemble-accuracy)=
 ## Ensembles
 
diff --git a/scatrs/src/tmatrix/domain.rs b/scatrs/src/tmatrix/domain.rs
index 18cb42b..0d051f6 100644
--- a/scatrs/src/tmatrix/domain.rs
+++ b/scatrs/src/tmatrix/domain.rs
@@ -28,26 +28,33 @@ pub struct ValidatedDomain {
     /// The largest validated |m|·x_max, with x_max = k·max(a, c) the size parameter of the
     /// longest semi-axis.
     pub max_phase: f64,
-    /// The largest validated Re m (the smallest is `abs_m_min`).
+    /// The smallest validated Re m.
+    pub m_re_min: f64,
+    /// The largest validated Re m.
     pub m_re_max: f64,
     /// The largest validated Im m (the smallest is 0).
     pub m_im_max: f64,
-    /// The smallest validated |m|.
-    pub abs_m_min: f64,
     /// The smallest validated `rtol`.
     pub rtol_min: f64,
 }
 
-/// The validated domain: aspect ratios 1/5 to 5, equal-volume x ≥ 10⁻³, |m|·x_max ≤ 60,
-/// 1 ≤ Re m ≤ 3, Im m ≤ 1, |m| ≥ 1, `rtol` ≥ 10⁻¹².
+/// The validated domain: aspect ratios 1/5 to 5, equal-volume x ≥ 0.3, |m|·x_max ≤ 60,
+/// 1.3 ≤ Re m ≤ 3, Im m ≤ 0.5, `rtol` ≥ 10⁻¹².
+///
+/// The accuracy map covers 221 spheroids: 8 aspect ratios from 1/5 to 5, x from 0.3 to 14 and
+/// four indices from 1.33 to 2.5 + 0.1i and 1.5 + 0.5i, up to |m|·x_max = 70. Its worst ratio of
+/// error to max(`rtol`, estimate) inside the domain is 1.04 (Q_back, ε = 1/3, x = 4), and 0.57
+/// for the Greek coefficients; the cross sections and g stay below 0.04. Past |m|·x_max = 60 the
+/// results are as accurate, but the refined modes make a spheroid cost minutes: the edge is set
+/// by cost.
 pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
     aspect_ratio_min: 0.2,
     aspect_ratio_max: 5.0,
-    x_min: 1e-3,
+    x_min: 0.3,
     max_phase: 60.0,
+    m_re_min: 1.3,
     m_re_max: 3.0,
-    m_im_max: 1.0,
-    abs_m_min: 1.0,
+    m_im_max: 0.5,
     rtol_min: 1e-12,
 };
 
@@ -58,9 +65,8 @@ impl ValidatedDomain {
         (self.aspect_ratio_min..=self.aspect_ratio_max).contains(&eps)
             && x >= self.x_min
             && m.norm() * x_max(x, eps) <= self.max_phase
-            && m.re <= self.m_re_max
+            && (self.m_re_min..=self.m_re_max).contains(&m.re)
             && (0.0..=self.m_im_max).contains(&m.im)
-            && m.norm() >= self.abs_m_min
     }
 
     /// The declared bound on the error of `output`: C·max(`rtol`, `estimate`).
@@ -73,8 +79,8 @@ impl ValidatedDomain {
     ///
     /// | output | C |
     /// |---|---|
-    /// | Q_ext, Q_sca, g | 4 |
-    /// | Q_abs, Q_back, phase matrix, Greek coefficients | 16 |
+    /// | Q_ext, Q_sca, g | 4 (worst measured: 0.03) |
+    /// | Q_abs, Q_back, phase matrix, Greek coefficients | 16 (worst measured: 1.04) |
     pub fn constant(output: MieOutput) -> f64 {
         match output {
             MieOutput::Extinction | MieOutput::Scattering | MieOutput::Asymmetry => 4.0,
diff --git a/scatrs/tests/tmatrix_accuracy_map.rs b/scatrs/tests/tmatrix_accuracy_map.rs
new file mode 100644
index 0000000..9ee1443
--- /dev/null
+++ b/scatrs/tests/tmatrix_accuracy_map.rs
@@ -0,0 +1,214 @@
+//! The accuracy map of randomly oriented spheroids: the default results (`rtol` = 10⁻⁸) against
+//! a reference with N + max(4, N/5) orders and 1.5 times the nodes the default converged at,
+//! every mode refined to an estimate below 10⁻¹³ (the refined path, whose own accuracy the Arb
+//! references of `spheroid_reference.rs` check), over a grid of aspect ratios, sizes and
+//! indices, inside the validated domain and past its edge in |m|·x_max. The reference thus
+//! measures the truncation, the quadrature and the solve of the default.
+//!
+//! For each output it prints the worst ratio error/max(`rtol`, estimate), in the measures of
+//! `MieOutput` (Q_abs relative to Q_ext; g absolute; Q_back relative to max(Q_back, Q_sca); the
+//! Greek coefficients relative to the largest |α₁|; the phase matrix relative to max(F₁₁, 1),
+//! normalized), separately inside and outside the domain. The declared constants of
+//! `tmatrix::VALIDATED_DOMAIN` are at least 4 times the worst ratio inside.
+//!
+//! Run with `cargo test --release -p scatrs --test tmatrix_accuracy_map -- --ignored --nocapture`
+//! (about an hour on an Apple M4 Pro). With `SCATRS_ACCURACY_MAP` set to a path it also writes
+//! the points as JSON there, for the documentation's plots:
+//!
+//! ```text
+//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/spheroid-accuracy-map.json \
+//!     cargo test --release -p scatrs --test tmatrix_accuracy_map -- --ignored
+//! ```
+
+use num_complex::Complex;
+use scatrs::mie::{MieOutput, Normalization};
+use scatrs::tmatrix::{PhaseBuffers, Shape, TMatrix, TMatrixOptions, VALIDATED_DOMAIN};
+
+const OUTPUTS: [&str; 7] = ["q_ext", "q_sca", "q_abs", "g", "q_back", "greek", "phase"];
+
+/// The `MieOutput` whose constant applies to each output.
+fn output_of(name: &str) -> MieOutput {
+    match name {
+        "q_ext" => MieOutput::Extinction,
+        "q_sca" => MieOutput::Scattering,
+        "q_abs" => MieOutput::Absorption,
+        "g" => MieOutput::Asymmetry,
+        "q_back" => MieOutput::Backscattering,
+        "greek" => MieOutput::GreekCoefficients { order: 0 },
+        _ => MieOutput::PhaseMatrix,
+    }
+}
+
+struct Point {
+    values: [f64; 5],
+    greek: [Vec<f64>; 6],
+    phase: [Vec<f64>; 6],
+    est: f64,
+    n: usize,
+    n_theta: usize,
+}
+
+fn solve(eps: f64, x: f64, m: Complex<f64>, opts: &TMatrixOptions, mu: &[f64]) -> Point {
+    let mut tm = TMatrix::new();
+    let s = tm
+        .compute(Shape::Spheroid { aspect_ratio: eps }, x, m, opts)
+        .unwrap();
+    let q = s.efficiencies();
+    let g = s.greek(None, Normalization::Normalized).unwrap();
+    let mut f: [Vec<f64>; 6] = core::array::from_fn(|_| vec![0.0; mu.len()]);
+    {
+        let [f11, f12, f22, f33, f34, f44] = &mut f;
+        s.phase_matrix(
+            mu,
+            Normalization::Normalized,
+            PhaseBuffers {
+                f11,
+                f12,
+                f22,
+                f33,
+                f34,
+                f44,
+            },
+        )
+        .unwrap();
+    }
+    Point {
+        values: [q.q_ext, q.q_sca, q.q_abs, q.g, q.q_back],
+        greek: [g.alpha1, g.alpha2, g.alpha3, g.alpha4, g.beta1, g.beta2],
+        phase: f,
+        est: s.error_estimate(),
+        n: s.n_terms(),
+        n_theta: s.n_theta(),
+    }
+}
+
+/// The errors of `a` against `b` in the measures of the module doc.
+fn errors(a: &Point, b: &Point) -> [f64; 7] {
+    let [e, s, ab, g, back] = b.values;
+    let rel = |u: f64, v: f64, scale: f64| (u - v).abs() / scale;
+    let big = b.greek[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
+    let mut greek = 0.0f64;
+    for (u, v) in a.greek.iter().zip(&b.greek) {
+        let len = u.len().max(v.len());
+        for l in 0..len {
+            let (x, y) = (
+                u.get(l).copied().unwrap_or(0.0),
+                v.get(l).copied().unwrap_or(0.0),
+            );
+            greek = greek.max((x - y).abs() / big);
+        }
+    }
+    let fbig = b.phase[0].iter().fold(1.0f64, |m, v| m.max(*v));
+    let mut phase = 0.0f64;
+    for (u, v) in a.phase.iter().zip(&b.phase) {
+        for (x, y) in u.iter().zip(v) {
+            phase = phase.max((x - y).abs() / fbig);
+        }
+    }
+    [
+        rel(a.values[0], e, e),
+        rel(a.values[1], s, s),
+        rel(a.values[2], ab, e),
+        rel(a.values[3], g, 1.0),
+        rel(a.values[4], back, back.max(s)),
+        greek,
+        phase,
+    ]
+}
+
+#[test]
+#[ignore = "the accuracy map: about an hour"]
+fn accuracy_map() {
+    let xs: [f64; 7] = [0.3, 1.0, 2.0, 4.0, 7.0, 10.0, 14.0];
+    let eps: [f64; 8] = [0.2, 1.0 / 3.0, 0.5, 0.8, 1.25, 2.0, 3.0, 5.0];
+    let ms = [
+        Complex::new(1.33, 0.0),
+        Complex::new(1.53, 0.008),
+        Complex::new(1.5, 0.5),
+        Complex::new(2.5, 0.1),
+    ];
+    let mu: Vec<f64> = (0..=20).map(|i| -1.0 + 0.1 * f64::from(i)).collect();
+    let f64_opts = TMatrixOptions::new().with_allow_unvalidated(true);
+    // The reference: more orders and nodes than the default converged at, every mode refined.
+    let ref_opts = |n: usize, nth: usize| {
+        TMatrixOptions::new()
+            .with_rtol(1e-12)
+            .with_n_terms(n + (n / 5).max(4))
+            .with_n_theta(nth * 3 / 2)
+            .with_allow_unvalidated(true)
+    };
+    // Worst ratio per output, inside and outside the domain, with where.
+    let mut rows: Vec<String> = Vec::new();
+    let mut worst: [[(f64, String); 7]; 2] =
+        core::array::from_fn(|_| core::array::from_fn(|_| (0.0, String::new())));
+    for &e in &eps {
+        for &x in &xs {
+            for &m in &ms {
+                let a = x * e.cbrt();
+                let phase = m.norm() * a.max(a / e);
+                if phase > 70.0 {
+                    continue;
+                }
+                let t0 = std::time::Instant::now();
+                let p = solve(e, x, m, &f64_opts, &mu);
+                let t64 = t0.elapsed();
+                let r = solve(e, x, m, &ref_opts(p.n, p.n_theta), &mu);
+                let errs = errors(&p, &r);
+                let inside = VALIDATED_DOMAIN.contains(e, x, m);
+                let unit = 1e-8f64.max(p.est);
+                let ratios = errs.map(|v| v / unit);
+                println!(
+                    "eps {e:.3} x {x:5.1} m {m:.3} |m|x_max {phase:5.1} {}: est {:.1e} ({t64:.2?}) ratios {}",
+                    if inside { "in " } else { "out" },
+                    p.est,
+                    ratios
+                        .iter()
+                        .map(|v| format!("{v:.1e}"))
+                        .collect::<Vec<_>>()
+                        .join(" ")
+                );
+                rows.push(format!(
+                    "    {{\"aspect_ratio\": {e}, \"x\": {x}, \"m_re\": {}, \"m_im\": {}, \"phase\": {phase}, \"inside\": {inside}, \"estimate\": {:e}, \"ratios\": [{}]}}",
+                    m.re,
+                    m.im,
+                    p.est,
+                    ratios.iter().map(|v| format!("{v:e}")).collect::<Vec<_>>().join(", ")
+                ));
+                let w = &mut worst[usize::from(!inside)];
+                for (k, &v) in ratios.iter().enumerate() {
+                    if v > w[k].0 {
+                        w[k] = (v, format!("eps {e:.3} x {x} m {m}"));
+                    }
+                }
+            }
+        }
+    }
+    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
+        let outputs = OUTPUTS
+            .iter()
+            .map(|o| format!("\"{o}\""))
+            .collect::<Vec<_>>()
+            .join(", ");
+        let declared = OUTPUTS
+            .iter()
+            .map(|o| {
+                format!(
+                    "\"{o}\": {}",
+                    VALIDATED_DOMAIN.tolerance(output_of(o), 1.0, 0.0)
+                )
+            })
+            .collect::<Vec<_>>()
+            .join(", ");
+        let json = format!(
+            "{{\n  \"outputs\": [{outputs}],\n  \"declared\": {{{declared}}},\n  \"points\": [\n{}\n  ]\n}}\n",
+            rows.join(",\n")
+        );
+        std::fs::write(&path, json).expect("write the map");
+    }
+    for (i, label) in ["inside", "outside"].iter().enumerate() {
+        println!("worst {label}:");
+        for (k, name) in OUTPUTS.iter().enumerate() {
+            println!("  {name:7} {:.3e} at {}", worst[i][k].0, worst[i][k].1);
+        }
+    }
+}
````

### Task 9: Documentation and the Rayleigh limit

- [ ] The theory chapter, the API pages, the index and the references; spec section 13.
- [ ] Test the Rayleigh limit.

The commits (`ef48058`, `2767521`):

````diff
diff --git a/docs/api.md b/docs/api.md
index 5218ec1..739ae4b 100644
--- a/docs/api.md
+++ b/docs/api.md
@@ -17,7 +17,7 @@ available from Rust ({doc}`rust-api`). Both APIs follow the {doc}`/theory/conven
 ```
 
 The package also re-exports the exceptions of {mod}`scatrs.errors` and the modules
-{mod}`scatrs.mie`, {mod}`scatrs.layered`, {mod}`scatrs.ensemble`, {mod}`scatrs.particles`, {mod}`scatrs.refractive`,
+{mod}`scatrs.mie`, {mod}`scatrs.layered`, {mod}`scatrs.tmatrix`, {mod}`scatrs.ensemble`, {mod}`scatrs.particles`, {mod}`scatrs.refractive`,
 {mod}`scatrs.size` and {mod}`scatrs.special`. The classes below are defined in
 {mod}`scatrs.particles`, {mod}`scatrs.size` and {mod}`scatrs.ensemble`, and used from the top
 level.
@@ -98,6 +98,13 @@ level.
    :members:
 ```
 
+## `scatrs.tmatrix`
+
+```{eval-rst}
+.. automodule:: scatrs.tmatrix
+   :members:
+```
+
 ## `scatrs.special`
 
 ```{eval-rst}
diff --git a/docs/index.md b/docs/index.md
index d6c2661..8c5f945 100644
--- a/docs/index.md
+++ b/docs/index.md
@@ -1,7 +1,7 @@
 # scatrs
 
-Fast, linearized, rigorously validated single-scattering calculations for spherical
-particles, written in Rust with a Python interface.
+Fast, linearized, rigorously validated single-scattering calculations for spherical and
+spheroidal particles, written in Rust with a Python interface.
 
 scatrs is built out of the scattering core of the
 [sasktran2](https://github.com/usask-arg/sasktran2) radiative transfer model, as a standalone
@@ -14,6 +14,9 @@ library for the optical properties of particles. Install it with `pip install sc
 - **Layered spheres** ({mod}`scatrs.layered`): a core and any number of concentric shells,
   coated spheres among them, with the same outputs and derivatives with respect to every
   layer.
+- **Randomly oriented spheroids** ({mod}`scatrs.tmatrix`): efficiencies, Greek coefficients
+  and the six-element phase matrix of oblate and prolate spheroids, by the T-matrix method,
+  converged to a tolerance with an error estimate, and their derivatives.
 - **Ensembles** of spheres and layered spheres ({class}`scatrs.Ensemble`,
   {class}`scatrs.Mixture`): averages over seven kinds of size distribution and over external
   mixtures, at many wavelengths in parallel, with an error estimate for every output, and refractive indices that vary with
@@ -70,6 +73,7 @@ theory/conventions
 theory/refractive-index
 theory/mie
 theory/layered
+theory/spheroids
 theory/ensembles
 theory/special-functions
 theory/scalar-types
diff --git a/docs/references.md b/docs/references.md
index 2005046..69b3251 100644
--- a/docs/references.md
+++ b/docs/references.md
@@ -59,6 +59,22 @@
   19.3. The error bound for Gauss quadrature of analytic functions, used in
   {doc}`utilities/legendre-moments`. The theorem is stated for rules with $n + 1$ points.
 
+**Spheroids and the T-matrix method**
+
+- P. C. Waterman, "Symmetry, unitarity, and geometry in electromagnetic scattering",
+  *Physical Review D* 3 (1971), 825–839. The extended boundary condition method.
+- M. I. Mishchenko, "Light scattering by randomly oriented axially symmetric particles",
+  *Journal of the Optical Society of America A* 8 (1991), 871–882. The orientation averages
+  from the T-matrix, by its rank decomposition.
+- M. I. Mishchenko and L. D. Travis, "T-matrix computations of light scattering by large
+  spheroidal particles", *Optics Communications* 109 (1994), 16–21. The loss of precision of
+  the EBCM for spheroids, and the extended-precision code `tmatd` of the comparison.
+- M. I. Mishchenko, L. D. Travis and A. A. Lacis, *Scattering, Absorption, and Emission of
+  Light by Small Particles*, Cambridge University Press (2002), Chapter 5. The T-matrix of an
+  axisymmetric particle (§5.8), its conventions and the convergence procedure.
+- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, SIAM (2002),
+  Chapters 12 and 15. Iterative refinement in mixed precision, and condition estimation.
+
 **Wigner d-functions and polarized radiative transfer**
 
 - M. I. Mishchenko, L. D. Travis and A. A. Lacis, *Scattering, Absorption, and Emission of
diff --git a/docs/rust-api.md b/docs/rust-api.md
index 05c1c49..bcd5126 100644
--- a/docs/rust-api.md
+++ b/docs/rust-api.md
@@ -14,6 +14,7 @@ Add it with `cargo add scatrs`. The same reference is on [docs.rs](https://docs.
 |---|---|---|
 | {rust}`scatrs::mie` | {mod}`scatrs.mie` | One sphere: coefficients, efficiencies, amplitudes, phase matrix, Greek coefficients and their derivatives. |
 | {rust}`scatrs::layered` | {mod}`scatrs.layered` | One layered sphere (a core and concentric shells): the same outputs, with derivatives with respect to every layer. |
+| {rust}`scatrs::tmatrix` | {mod}`scatrs.tmatrix` | One spheroid by the T-matrix method: orientation-averaged efficiencies, Greek coefficients and phase matrix, the T-matrix itself and its convergence, with derivatives. |
 | {rust}`scatrs::ensemble` | {class}`scatrs.Ensemble`, {class}`scatrs.Mixture` | Averages of spheres and layered spheres over size distributions, mixtures and batches (Python: grids), with error estimates and derivatives. |
 | {rust}`scatrs::special` | {mod}`scatrs.special` | Gauss–Legendre quadrature and Wigner d-functions. |
 | {rust}`scatrs::scalar` | | The {rust}`Real <scatrs::scalar::Real>` trait and the double-double and dual-number types. Rust only. |
diff --git a/docs/theory/spheroids.md b/docs/theory/spheroids.md
new file mode 100644
index 0000000..527b660
--- /dev/null
+++ b/docs/theory/spheroids.md
@@ -0,0 +1,189 @@
+# Spheroids
+
+<!-- synced with scatrs/src/tmatrix/ -->
+
+{mod}`scatrs.tmatrix` computes the optical properties of spheroids averaged uniformly over
+their orientations, the case of randomly oriented particles in a cloud or an aerosol layer.
+It uses the T-matrix method, with the T-matrix from the extended boundary condition method
+(EBCM; Waterman 1971). This chapter describes the method, the three changes that make it
+accurate in double precision where the classical method is not, and how accurate and costly
+the results are.
+
+## Geometry and conventions
+
+A spheroid has semi-axes $a$, $a$ and $c$, with $c$ along its symmetry axis, and aspect ratio
+$\varepsilon = a/c$: oblate for $\varepsilon > 1$, prolate for $\varepsilon < 1$, a sphere at
+$\varepsilon = 1$. Its surface is
+
+$$
+r(\theta) = \frac{ac}{\sqrt{c^2\sin^2\theta + a^2\cos^2\theta}} .
+$$
+
+Its size is the size parameter $x = k r_v$ of the sphere of equal volume, $r_v^3 = a^2 c$,
+with $k = 2\pi/\lambda$. That sphere's geometric cross section $\pi r_v^2$ normalizes the
+efficiencies and the unnormalized phase matrix and Greek coefficients, so that a spheroid and
+a sphere of the same volume compare directly. The longest semi-axis has the size parameter
+$x_{\max} = k\max(a, c)$, which sets the number of orders. The other conventions are those of
+{doc}`conventions`.
+
+## The T-matrix
+
+The T-matrix relates the coefficients of the scattered field, in outgoing vector spherical
+wave functions $\mathbf{M}_{mn}$ and $\mathbf{N}_{mn}$, to those of the incident field, in
+regular ones. For an axisymmetric particle in its own frame it is diagonal in the azimuthal
+order $m$: each mode $m = 0, \dots, N$ has its own matrix $T^m$ over the orders
+$n, k = \max(m, 1), \dots, N$, with four blocks ($\mathbf{M}$ and $\mathbf{N}$ in and out).
+
+The EBCM gives $T^m = -P^m (Q^m)^{-1}$, with the elements of $Q^m$ and $P^m$ surface integrals
+of cross products of the wave functions,
+
+$$
+Q_{nk} = -i \oint_S \hat{\mathbf n}\cdot\bigl[\mathbf X_n \times \nabla\times\mathbf Y_k
+  - \mathbf Y_k \times \nabla\times\mathbf X_n\bigr]\, dS ,
+$$
+
+where $\mathbf X_n$ is an outgoing wave of the medium for $Q$ and a regular one for $P$, and
+$\mathbf Y_k$ a regular wave inside the particle (Mishchenko, Travis & Lacis 2002, §5.8). The
+integral over $\phi$ is exact, and the one over $\theta$ uses Gauss–Legendre quadrature with
+$2n_\theta$ nodes in $\cos\theta$. A spheroid is symmetric under reflection in its equatorial
+plane: half of the elements vanish, each mode splits into two independent systems of half the
+size, and the integrands of the others are even in $\cos\theta$, so only the $n_\theta$ nodes
+with $\cos\theta > 0$ are evaluated.
+
+For a sphere $T^m$ is diagonal, $T^{11} = -b_n$ and $T^{22} = -a_n$, the Mie coefficients;
+scatrs checks this identity, and that every averaged output of an aspect ratio of 1 equals
+{mod}`scatrs.mie`'s.
+
+## Why the classical method loses precision
+
+The outgoing waves contain the spherical Bessel functions of the second kind,
+$y_n(kr) \sim -(2n-1)!!/(kr)^{n+1}$ for $kr < n$. On a non-spherical surface the integrand of
+an element $Q_{nk}$ with $n$ much larger than $k$ contains terms as large as
+$(n/kr_{\min})^n$, which the integration over $\theta$ must cancel down to a result many
+orders of magnitude smaller. In double precision this cancellation destroys the element: at
+$x = 15$ and $\varepsilon = 4$ the classical integrals lose up to sixteen digits, and the
+T-matrix is wrong in its leading digit. This is the well-known loss of precision of the EBCM
+for elongated or flattened particles, usually addressed by computing in extended precision
+(Mishchenko & Travis 1994).
+
+scatrs removes the cancellation instead. Expand the radial factors of each integrand, the
+products of $y_n$ (or its derivative) with $j_k(m k r)$, in powers of $kr$. For a spheroid
+$1/r^2 = \alpha + \beta\cos^2\theta$, so every term of non-positive power is a polynomial in
+$\cos\theta$ times an angular function. The terms of power at most 0 are the electrostatic
+limit of the integral, and for $n \ne k$ they integrate to exactly zero as a group: the
+outgoing and regular multipoles of different orders are orthogonal on any surface in the
+static limit. (Those of negative power alone vanish when $|n - k| \ge 3$.) They are exactly
+the terms that cancel. scatrs therefore accumulates each element in three exact forms (the
+plain integrand, without the terms of negative power, and without those of power at most 0)
+together with the sum of the magnitudes of what each adds, and keeps the form with the
+smallest sum. At each node the kept part is evaluated either from its own series or as the
+function minus the removed terms, whichever adds less in magnitude. The series are summed
+until negligible at the longest axis, so both routes agree to rounding.
+
+With this change the elements carry little cancellation, and at $x = 15$, $\varepsilon = 4$
+the T-matrix error in double precision falls from order 1 to $3\cdot10^{-4}$.
+
+## The conditioning floor and mixed-precision refinement
+
+Even with accurate elements, a double-precision solve leaves an error of about
+$\mathrm{cond}(Q)\cdot 2^{-53}$, carried by the rows of high outgoing order. On a
+non-spherical surface these are the evanescent multipoles $r^{-n-1}P_n^m(\cos\theta)$, which
+concentrate near the shortest axis and are nearly dependent. Rounding their elements to double
+precision perturbs $T$ by up to $10^{-8}$ near $|m|\,x_{\max} \approx 60$, and much more for
+prolate spheroids, whose elements in these rows keep a cancellation that the electrostatic
+removal does not reach.
+
+scatrs estimates the error of every mode and refines the modes that need it:
+
+1. **Estimate.** Each element's rounding is bounded by the unit roundoff times the magnitudes
+   it was summed from. To first order an error $\delta Q$, $\delta P$ moves $T$ by
+   $-(\delta P + T\,\delta Q)\,Q^{-1}$. scatrs applies this to random-signed perturbations at
+   the bounds and random-signed vectors, with the factors already computed, which costs
+   $O(K^2)$ for a mode of size $K$.
+2. **Refine.** If the estimate exceeds a tenth of the tolerance, the rows of order $n \ge N/4$
+   are integrated again in double-double arithmetic (about 32 digits;
+   {doc}`scalar-types`). The mixed matrix is factorized in double precision, and the solution
+   of $X Q = -P$ is corrected with residuals $-P - XQ$ computed in double-double (iterative
+   refinement in mixed precision; Higham 2002, §12). Each sweep gains the factor
+   $\mathrm{cond}(Q)\cdot 2^{-53}$, and two sweeps usually reach double-double accuracy.
+3. **Double-double.** If refinement still leaves an estimate above the tolerance, the whole
+   T-matrix is computed in double-double.
+
+The paths taken for each mode are reported (`precision_paths`). For example, at $x = 15$ and
+$\varepsilon = 1/3$ the double-precision T-matrix is wrong by $0.15$, and the refined one is
+within $3\cdot10^{-14}$ of the double-double one after two sweeps.
+
+## Convergence
+
+The number of orders $N$ and of nodes $n_\theta$ are chosen as in Mishchenko's `tmatd`:
+
+1. $N$ starts at $\lceil x_{\max} + 4x_{\max}^{1/3}\rceil + 2$ and grows by an eighth
+   (at least 2), with $n_\theta = N(1 + |\ln\varepsilon|) + 8$, until the contributions of
+   mode 0 to $\langle C_{\mathrm{ext}}\rangle$ and $\langle C_{\mathrm{sca}}\rangle$ change by
+   less than `rtol` (mode 0 involves every order, and the other modes converge faster).
+2. $n_\theta$ then grows by a quarter until they settle again.
+3. Every mode is solved at the final $N$ and $n_\theta$.
+4. If Greek coefficients up to an order must settle, one more step in $N$ and $n_\theta$ with
+   every mode must leave them, and the cross sections, within `rtol`.
+
+$N$ and $n_\theta$ have caps; reaching one raises `NotConverged` unless the unconverged result
+is asked for. The result's error estimate is the last change of the loop, or the solve's
+estimate if larger.
+
+## Random orientation
+
+The orientation averages follow from the rank decomposition of $T$ (Mishchenko 1991). In the
+helicity basis $(\mathbf M \pm \mathbf N)/\sqrt2$, the T-matrix between orders $n$ and $n'$
+splits into components of rank $s = |n - n'|, \dots, n + n'$ by the Clebsch–Gordan
+coefficients, and rotating the particle multiplies each by a Wigner function. The average over
+orientations of a product of two amplitudes is then a sum over $s$ of products of
+rank-$s$ amplitudes divided by $2s + 1$. scatrs evaluates the amplitudes for each helicity at
+the nodes of a Gauss–Legendre rule in the scattering angle, forms the orientation-averaged
+phase matrix, which has six independent elements ($F_{11}$, $F_{12}$, $F_{22}$, $F_{33}$,
+$F_{34}$, $F_{44}$), and projects it onto the generalized spherical functions. The elements
+are polynomials of degree $2N$ in $\cos\Theta$, so the projection with
+$N + \lceil (L+1)/2 \rceil$ nodes is exact, and the coefficients vanish above $L = 2N$. The
+cost is $O(N^4)$. The cross sections come from $T$ directly,
+
+$$
+\langle C_{\mathrm{ext}}\rangle = -\frac{2\pi}{k^2}\sum_{m=0}^{N} (2 - \delta_{m0})
+  \sum_n \mathrm{Re}\bigl(T^{m,11}_{nn} + T^{m,22}_{nn}\bigr),
+\qquad
+\langle C_{\mathrm{sca}}\rangle = \frac{2\pi}{k^2}\sum_{m=0}^{N} (2 - \delta_{m0})
+  \sum_{n,k}\sum_{i,j=1}^{2}\bigl|T^{m,ij}_{nk}\bigr|^2,
+$$
+
+$\langle C_{\mathrm{abs}}\rangle$ is their difference, the asymmetry parameter is
+$\alpha_1(1)/3$, and the backscattering cross section is $4\pi F_{11}(180^\circ)$ from the
+Greek expansion. The phase matrix at any angle also comes from the Greek expansion.
+
+These formulas were checked against a direct numerical average over orientations of the
+amplitude matrix, to $10^{-15}$.
+
+## Derivatives
+
+The derivatives with respect to $x$, $\varepsilon$, $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$ come
+from one more solve at the converged $N$ and $n_\theta$: the elements are computed in dual
+numbers carrying four derivatives (their values are bitwise the double-precision ones), and
+
+$$
+\frac{\partial T}{\partial p} = -\Bigl(\frac{\partial P}{\partial p}
+  + T\,\frac{\partial Q}{\partial p}\Bigr) Q^{-1}
+$$
+
+reuses the factors of the solve, refined in double-double in the modes where $T$ is. The
+averages are computed in the same dual numbers, so every output has its derivatives. They cost
+4 to 13 times the values.
+
+## Accuracy and cost
+
+The results are compared with an EBCM computed from its definition, with no simplification of
+the integrands, in Arb ball arithmetic ({doc}`reference-suite`), and with the external codes
+of {doc}`/validation/external`. Inside the validated domain every output is within a small
+multiple of the larger of `rtol` and the error estimate ({doc}`/validation/accuracy`).
+
+The cost grows like $N^4$, with $N \approx x_{\max}$, and steeply for extreme aspect ratios,
+where refinement engages. At `rtol = 1e-8` on one core of an Apple M4 Pro, an oblate dust
+spheroid with $\varepsilon = 2$ takes 25 ms at $x = 5$ and 5 s at $x = 15$. Refined modes cost
+far more than double-precision ones, and extreme cases, such as $\varepsilon = 1/5$ at
+$x = 12$, take minutes.
diff --git a/scatrs/src/tmatrix/mod.rs b/scatrs/src/tmatrix/mod.rs
index 627080e..d800856 100644
--- a/scatrs/src/tmatrix/mod.rs
+++ b/scatrs/src/tmatrix/mod.rs
@@ -1093,6 +1093,63 @@ mod tests {
         assert!(!s.converged() && s.error_estimate() > 1e-12);
     }
 
+    /// The Rayleigh limit: at x ≪ 1, ⟨C_abs⟩ = k Im ᾱ and ⟨C_sca⟩ = k⁴/(6π) mean |αⱼ|², with
+    /// αⱼ = V(m² − 1)/(1 + Lⱼ(m² − 1)) from the depolarization factors Lⱼ (Bohren & Huffman
+    /// 1983, §5.3), for oblate and prolate spheroids, to O(x²).
+    #[test]
+    fn the_rayleigh_limit() {
+        let (x, m) = (0.01, Complex::new(1.5, 0.1));
+        for eps in [0.25, 0.5, 2.0, 4.0] {
+            let a = x * f64::cbrt(eps);
+            let c = a / eps;
+            // Prolate: L along the (long) symmetry axis c; oblate: L along the two equal axes a.
+            let (l_a, l_c) = if c > a {
+                let e = (1.0 - a * a / (c * c)).sqrt();
+                let l_c =
+                    (1.0 - e * e) / (e * e) * (-1.0 + ((1.0 + e) / (1.0 - e)).ln() / (2.0 * e));
+                ((1.0 - l_c) / 2.0, l_c)
+            } else {
+                let e2 = 1.0 - c * c / (a * a);
+                let g = ((1.0 - e2) / e2).sqrt();
+                let l_a = g / (2.0 * e2) * (core::f64::consts::FRAC_PI_2 - g.atan()) - g * g / 2.0;
+                (l_a, 1.0 - 2.0 * l_a)
+            };
+            let v = 4.0 / 3.0 * core::f64::consts::PI * x * x * x;
+            let em1 = m * m - 1.0;
+            let alpha = |l: f64| em1 * v / (1.0 + em1 * l);
+            let al = [alpha(l_a), alpha(l_a), alpha(l_c)];
+            let mean: Complex<f64> = al.iter().sum::<Complex<f64>>() / 3.0;
+            let area = core::f64::consts::PI * x * x;
+            let q_abs = mean.im / area;
+            let q_sca = al.iter().map(|a| a.norm_sqr()).sum::<f64>()
+                / 3.0
+                / (6.0 * core::f64::consts::PI)
+                / area;
+            let mut tm = TMatrix::new();
+            let s = tm
+                .compute(
+                    spheroid(eps),
+                    x,
+                    m,
+                    &TMatrixOptions::new()
+                        .with_rtol(1e-12)
+                        .with_allow_unvalidated(true),
+                )
+                .unwrap();
+            let q = s.efficiencies();
+            assert!(
+                (q.q_abs - q_abs).abs() < 1e-3 * q_abs,
+                "eps {eps}: {} vs {q_abs}",
+                q.q_abs
+            );
+            assert!(
+                (q.q_sca - q_sca).abs() < 1e-3 * q_sca,
+                "eps {eps}: {} vs {q_sca}",
+                q.q_sca
+            );
+        }
+    }
+
     #[test]
     fn invalid_inputs_and_domain() {
         let mut tm = TMatrix::new();
diff --git a/docs/specs/2026-10-14-phase-4-spheroids-design.md b/docs/specs/2026-10-14-phase-4-spheroids-design.md
index c982d1a..e429222 100644
--- a/docs/specs/2026-10-14-phase-4-spheroids-design.md
+++ b/docs/specs/2026-10-14-phase-4-spheroids-design.md
@@ -311,3 +311,39 @@ impl<T: Real> TMatrix<T> {
 
   Separation of variables in spheroidal coordinates is recorded as the research track that could extend f64 further.
 - The study's lab becomes the reference generator's core.
+
+## 13. Changes made during the implementation of 4a
+
+- **The Rust API is `f64`.** `TMatrix`, `TMatrixOptions` and `TMatrixSolution` are not generic over `Real`. The precision paths and the refinement are specific to `f64`, and `TMatrixOptions::precision` (`Auto`, `F64`, `DoubleDouble`) gives double-double results. The kernel itself stays generic, and runs in `f64`, `DoubleDouble`, `Dual<f64, 4>` and `Dual<DoubleDouble, 4>`.
+- **The paths are chosen by an error estimate, not a Skeel estimate.**
+  - Hager's estimate of Skeel(Qᵀ) mispredicted the f64 error by up to 10⁸, since the error depends on which rows carry it.
+  - Each mode now gets ΔT ≈ −(ΔP + T ΔQ) Q⁻¹, probed with random-signed element perturbations at their bounds and random-signed vectors (O(K²)), times 4.
+  - The element bounds are 2N·u times the magnitudes the element was summed from (the regularized forms' bookkeeping). The largest true error measured was about N·u times that sum.
+  - The estimate covers the f64 error on every map point, conservatively by 10–100×.
+- **Refinement factorizes the mixed matrix rounded to f64**, not the f64 integrals. For prolate spheroids the f64 elements of the high outgoing rows keep row-relative errors near 10⁻⁵, a cancellation the electrostatic removal does not reach. A factorization of those elements does not contract the residuals. The mixed matrix rounded contracts them by about 10⁻¹⁰ per sweep, so two sweeps suffice.
+- **Convergence as in `tmatd`.**
+  - N and then n_θ converge on mode 0 alone, refined if its estimate requires.
+  - All modes are then solved once.
+  - The Greek coefficients, if asked for, are checked by one more full step.
+  - This made the loop about 3× faster than converging every mode.
+- **Outputs from the Greek expansion.** The phase matrix at any angle, Q_back = 4πF₁₁(180°)/(πx²) and g = α₁(1)/3 come from the Greek coefficients up to 2N, which are projected exactly. This avoids the poles of πₙ at sin Θ = 0. Q_abs = Q_ext − Q_sca.
+- **Derivatives in dual numbers.**
+  - The elements run in `Dual<f64, 4>`, and dT = −(dP + T dQ) Q⁻¹ uses the solve's factors.
+  - Refined modes get double-double rows in `Dual<DoubleDouble, 4>` and refined derivative residuals.
+  - Cost: 4–13 times the values, against the target of 3. Closed-form differentiated integrands remain the way to the target (follow-up).
+- **Cost of the refined path.**
+  - Double-double integrals cost about 25 times f64 per element. A mode refined with rows n ≥ N/4 in double-double costs about 15–20 times its f64 solve, against the target of 4 times.
+  - Mitigations made: the series tail collapses into suffix sums; norms are replaced by |re| + |im| in the bookkeeping; the s-scaled column fields are hoisted; the angular factors are sparse; refinement covers only the modes whose estimate needs it.
+  - Measured at `rtol` = 10⁻⁸: x = 15, ε = 2 takes 5 s. Extreme aspect ratios at the edge of the domain (ε = 1/5, x = 12) take minutes.
+- **Python.**
+  - `scatrs.mie.PhaseBuffers` is not `non_exhaustive`, so it was not extended. `scatrs::tmatrix::PhaseBuffers` and `SpheroidPhaseMatrix` have six elements instead.
+  - `scatrs.tmatrix.compute` returns a `TMatrix` with the mode blocks, the convergence information and `random_orientation`.
+- **References.**
+  - The generator ports the study's from-definition EBCM to python-flint (Arb), not mpmath: it is 30–100× faster, and the package's single pinned dependency stays.
+  - It stores `fixed` records (Q_ext, Q_sca and a sample of Tᵐ at a given N and n_θ, certified for that truncation), which test every precision path, and `converged` records (Q_ext, Q_sca, Q_abs settled to 10⁻²⁰, high-precision).
+  - Greek coefficients are not in the references. They are covered by:
+    - the sphere limit;
+    - the brute-force orientation average (10⁻¹⁵);
+    - 2π∫F₁₁ = C_sca;
+    - the external comparisons.
+- **External comparisons.** `tmatd` and pytmatrix need downloads, which wait for the user's approval.
````

### Task 10: External comparisons

- [ ] Ask before downloading (approved 2026-10-15): Mishchenko's `tmq.lp.f`, `lpq.f`, `tmq.par.f` from NASA GISS (about 300 KB); pytmatrix from PyPI.
- [ ] `benchmarks/compare_tmatd.py`: fetch into the ignored `benchmarks/tmatd/`, check SHA-256, patch (stdin input, single-particle radii, 16 digits, gfortran quad intrinsics, drop the leading sample output), build, compare 72 spheroids (about 25 minutes).
- [ ] Publish on the external-validation page, with the β sign convention and the attribution.
- [ ] pytmatrix 0.3.3 does not build (`numpy.distutils`, missing f2py source) and wraps the same code: record and skip.

The commit (`b8c2798`, without its results JSON):

````diff
diff --git a/.gitignore b/.gitignore
index 87e2763..bcb0d80 100644
--- a/.gitignore
+++ b/.gitignore
@@ -12,3 +12,5 @@
 *.egg-info/
 .DS_Store
 /docs/build/
+# Mishchenko's T-matrix code, fetched by benchmarks/compare_tmatd.py (not redistributed).
+benchmarks/tmatd/
diff --git a/benchmarks/README.md b/benchmarks/README.md
index eed2d50..439d464 100644
--- a/benchmarks/README.md
+++ b/benchmarks/README.md
@@ -11,3 +11,15 @@ uv run --locked --directory benchmarks python compare.py
 It writes `results/<date>-<machine>.json` and prints the tables of the documentation. The
 reference integrals (scatrs at rtol 10⁻⁹) take a few minutes on all cores. `--quick` runs a
 smaller grid to check the script itself.
+
+## Spheroids against Mishchenko's T-matrix code
+
+`compare_tmatd.py` fetches Mishchenko's extended-precision code for randomly oriented particles
+(`tmq.lp.f`, `lpq.f`, `tmq.par.f`) from its page at NASA GISS into `tmatd/` (ignored by git: the
+code is publicly available but not redistributed here), checks the SHA-256 of each file,
+patches it (input from standard input, 16 printed digits, gfortran's names for its
+quad-precision intrinsics, the single-particle radii of its notes), builds it with `gfortran`,
+and compares it with `scatrs.tmatrix` for 72 spheroids, accuracy and time. Needs `gfortran`;
+about 25 minutes:
+
+    uv run --directory benchmarks python compare_tmatd.py
diff --git a/benchmarks/compare_tmatd.py b/benchmarks/compare_tmatd.py
new file mode 100644
index 0000000..d509a22
--- /dev/null
+++ b/benchmarks/compare_tmatd.py
@@ -0,0 +1,263 @@
+"""scatrs.tmatrix against Mishchenko's T-matrix code for randomly oriented spheroids.
+
+Mishchenko's extended-precision code (``tmq.lp.f``, ``lpq.f``, ``tmq.par.f``, REAL*16) is not
+redistributed here: it is fetched from its page at NASA GISS into ``tmatd/src`` (not committed),
+and built with gfortran after two patches, which change no computation:
+
+* the input block reads the particle from standard input: a monodisperse spheroid as the code's
+  own notes describe (NDISTR = 4, NKMAX = -1, R1 and R2 within 10⁻⁷ of the radius), with
+  RAT = 1 (equal-volume radius), LAM = 2π so that the radius is the size parameter, and the
+  convergence tolerance DDELT;
+* the output formats print 16 digits instead of 6 or 7, and the sample output that opens the
+  file is dropped;
+* the vendor-specific quad-precision intrinsics (``QCONJG``, ``QSQRT``, ``QFLOAT``, …), which
+  gfortran does not have, become their generic equivalents at kind 16.
+
+For each case it compares Q_ext, Q_sca, g and the normalized Greek coefficients with scatrs at
+``rtol`` = 10⁻¹⁰, and times both (tmatd at DDELT = 10⁻⁶ and 10⁻⁹, scatrs at 10⁻⁶ and 10⁻⁸).
+tmatd's β₁ and β₂ are compared with their signs reversed: it expands in the generalized
+spherical functions P⁰²ₗ = −d⁰²ₗ (Mishchenko, Travis & Lacis 2002, Appendix B), scatrs in the
+Wigner d-functions.
+
+Usage: uv run --directory benchmarks python compare_tmatd.py [--quick]
+"""
+
+from __future__ import annotations
+
+import json
+import math
+import platform
+import re
+import shutil
+import subprocess
+import sys
+import tempfile
+import time
+import urllib.request
+from pathlib import Path
+
+import numpy as np
+
+import scatrs
+from scatrs import tmatrix
+
+HERE = Path(__file__).parent
+SRC = HERE / "tmatd" / "src"
+BUILD = HERE / "tmatd" / "build"
+URL = "https://www.giss.nasa.gov/staff/mmishchenko/ftpcode/"
+FILES = {
+    "tmq.lp.f": "612718d332fc3405a0fc50f717f543c8948637162bd994cdd633a5ee6955a66c",
+    "lpq.f": "5de1eb42504ea3058b3afe10e43d64ef75ea84d999379c213854cb29cfcd6d65",
+    "tmq.par.f": "9ecb43b79b985d182b0c794194d092669be522f88970ca7553f60a980a4138fb",
+}
+
+# The input block, from its first assignment to its last (its lines end in spaces).
+INPUT_BLOCK = re.compile(r"^      RAT=0\.5D0\n.*?^      NDGS=2 *\n", re.MULTILINE | re.DOTALL)
+NEW_INPUT = """      RAT=1D0
+      NDISTR=4
+      NPNAX=1
+      B=1D-1
+      GAM=0.5D0
+      NKMAX=-1
+      NP=-1
+      NPNA=19
+      LAM=2D0*DACOS(-1D0)
+      READ (5,*) AXMAX,EPS,MRR,MRI,DDELT,NDGS
+"""
+# The radii of the size distribution, which the notes say to replace for one particle.
+RADII = re.compile(
+    r"^         R1=0\.89031D0\*AXI *\n^         R2=1\.56538D0\*AXI *\n", re.MULTILINE
+)
+NEW_RADII = """         R1=0.9999999D0*AXI
+         R2=1.0000001D0*AXI
+"""
+
+
+#: Vendor quad-precision intrinsics and their generic gfortran forms.
+QUAD_INTRINSICS = [
+    (r"\bQCONJG\b", "CONJG"),
+    (r"\bQFLOAT\(", "REAL(KIND=16,A="),
+    (r"\bQSQRT\b", "SQRT"),
+    (r"\bQCOS\b", "COS"),
+    (r"\bQSIN\b", "SIN"),
+    (r"\bQABS\b", "ABS"),
+    (r"\bQMAX1\b", "MAX"),
+    (r"\bQARCOS\b", "ACOS"),
+    (r"\bQSINH\b", "SINH"),
+    (r"\bQCOSH\b", "COSH"),
+    (r"\bQIMAG\b", "AIMAG"),
+    (r"\bQATAN\b", "ATAN"),
+    (r"QCMPLX\(QR\(I,J\),QI\(I,J\)\)", "CMPLX(QR(I,J),QI(I,J),KIND=16)"),
+    # The Q edit descriptor of formats.
+    (r",Q(\d+\.\d+)\)", r",E\1)"),
+]
+
+
+def generic(text: str) -> str:
+    for pattern, new in QUAD_INTRINSICS:
+        text = re.sub(pattern, new, text, flags=re.IGNORECASE)
+    return text
+
+
+def fetch() -> None:
+    import hashlib
+
+    SRC.mkdir(parents=True, exist_ok=True)
+    for name, digest in FILES.items():
+        path = SRC / name
+        if not path.exists():
+            with urllib.request.urlopen(URL + name, timeout=60) as r:
+                path.write_bytes(r.read())
+        got = hashlib.sha256(path.read_bytes()).hexdigest()
+        if got != digest:
+            raise RuntimeError(f"{name}: SHA-256 {got} differs from the one compared ({digest})")
+
+
+def build() -> Path:
+    exe = BUILD / "tmq"
+    if exe.exists():
+        return exe
+    BUILD.mkdir(parents=True, exist_ok=True)
+    text = (SRC / "tmq.lp.f").read_text()
+    # The file opens with a sample of the code's output: the Fortran starts at its notes.
+    text = text[text.index("C   New release including the LAPACK") :]
+    text, count = INPUT_BLOCK.subn(NEW_INPUT, text)
+    text, count2 = RADII.subn(NEW_RADII, text)
+    if count != 1 or count2 != 1:
+        raise RuntimeError("tmq.lp.f differs from the version the patches were written for")
+    text = text.replace(
+        "9100    FORMAT('CEXT=',D12.6,2X,'CSCA=',D12.6,2X,",
+        "9100    FORMAT('CEXT=',D24.16,2X,'CSCA=',D24.16,2X,",
+    )
+    text = text.replace(
+        "     &          2X,'W=',D12.6,2X,'<COS>=',D12.6)",
+        "     &          2X,'W=',D24.16,2X,'<COS>=',D24.16)",
+    )
+    text = text.replace("  575    FORMAT(6D14.7)", "  575    FORMAT(6D25.16)")
+    (BUILD / "tmq.lp.f").write_text(generic(text))
+    (BUILD / "lpq.f").write_text(generic((SRC / "lpq.f").read_text()))
+    shutil.copy(SRC / "tmq.par.f", BUILD / "tmq.par.f")
+    subprocess.run(
+        ["gfortran", "-O2", "-std=legacy", "-o", "tmq", "tmq.lp.f", "lpq.f"],
+        cwd=BUILD,
+        check=True,
+    )
+    return exe
+
+
+def run_tmatd(exe: Path, x: float, eps: float, m: complex, ddelt: float) -> dict:
+    with tempfile.TemporaryDirectory() as tmp:
+        t0 = time.perf_counter()
+        subprocess.run(
+            [str(exe)],
+            input=f"{x!r} {eps!r} {m.real!r} {m.imag!r} {ddelt!r} 2\n",
+            text=True,
+            cwd=tmp,
+            check=True,
+            capture_output=True,
+            timeout=3600,
+        )
+        elapsed = time.perf_counter() - t0
+        out = (Path(tmp) / "test").read_text()
+        coeffs = (Path(tmp) / "tmatr.write").read_text().split("\n")
+    m_ = re.search(r"CEXT=\s*(\S+)\s+CSCA=\s*(\S+)\s+W=\s*(\S+)\s+<COS>=\s*(\S+)", out)
+    if m_ is None:
+        raise RuntimeError(f"tmatd failed:\n{out[-2000:]}")
+    num = [float(v.replace("D", "E")) for v in m_.groups()]
+    area = math.pi * x * x
+    rows = [
+        [float(v.replace("D", "E")) for v in line.split()] for line in coeffs[1:] if line.strip()
+    ]
+    greek = np.array(rows).T  # alpha1..4, beta1, beta2 by order
+    # tmatd expands F12 and F34 in the generalized spherical functions P⁰²ₗ = −d⁰²ₗ, scatrs in
+    # the Wigner d-functions: the same phase matrix has β₁ and β₂ of opposite signs.
+    greek[4:] *= -1
+    return {
+        "q_ext": num[0] / area,
+        "q_sca": num[1] / area,
+        "g": num[3],
+        "greek": greek,
+        "seconds": elapsed,
+        "converged": "WARNING" not in out,
+    }
+
+
+def run_scatrs(x: float, eps: float, m: complex, rtol: float) -> dict:
+    t0 = time.perf_counter()
+    t = tmatrix.compute(x, m, eps, rtol=rtol, allow_unvalidated=True)
+    q, g, _ = t.random_orientation()
+    elapsed = time.perf_counter() - t0
+    return {
+        "q_ext": float(q.q_ext),
+        "q_sca": float(q.q_sca),
+        "g": float(q.g),
+        "greek": np.array([g.alpha1, g.alpha2, g.alpha3, g.alpha4, g.beta1, g.beta2]),
+        "seconds": elapsed,
+        "estimate": t.error_estimate,
+    }
+
+
+CASES = [
+    (x, eps, m)
+    for eps in (0.2, 1 / 3, 0.5, 2.0, 3.0, 5.0)
+    for x in (1.0, 3.0, 6.0, 10.0)
+    for m in (1.33 + 0j, 1.53 + 0.008j, 1.5 + 0.5j)
+]
+
+
+def main() -> None:
+    quick = "--quick" in sys.argv
+    fetch()
+    exe = build()
+    cases = CASES[::7] if quick else CASES
+    results = []
+    for x, eps, m in cases:
+        ref = run_scatrs(x, eps, m, 1e-10)
+        fast = run_scatrs(x, eps, m, 1e-6)
+        mid = run_scatrs(x, eps, m, 1e-8)
+        row = {
+            "x": x,
+            "aspect_ratio": eps,
+            "m": [m.real, m.imag],
+            "scatrs_estimate": ref["estimate"],
+        }
+        for ddelt in (1e-6, 1e-9):
+            try:
+                tm = run_tmatd(exe, x, eps, m, ddelt)
+            except (RuntimeError, subprocess.SubprocessError) as e:
+                row[f"tmatd_{ddelt:g}"] = {"error": str(e)[:300]}
+                continue
+            n = min(tm["greek"].shape[1], ref["greek"].shape[1])
+            big = np.abs(ref["greek"][0]).max()
+            greek_diff = float(np.abs(tm["greek"][:, :n] - ref["greek"][:, :n]).max() / big)
+            tail = (
+                float(np.abs(tm["greek"][:, n:]).max() / big) if tm["greek"].shape[1] > n else 0.0
+            )
+            row[f"tmatd_{ddelt:g}"] = {
+                "q_ext": abs(tm["q_ext"] / ref["q_ext"] - 1),
+                "q_sca": abs(tm["q_sca"] / ref["q_sca"] - 1),
+                "g": abs(tm["g"] - ref["g"]),
+                "greek": max(greek_diff, tail),
+                "seconds": tm["seconds"],
+                "converged": tm["converged"],
+            }
+        row["scatrs_seconds"] = {"1e-6": fast["seconds"], "1e-8": mid["seconds"]}
+        row["scatrs_1e-6_q_ext"] = abs(fast["q_ext"] / ref["q_ext"] - 1)
+        results.append(row)
+        print(json.dumps(row), flush=True)
+    out = HERE / "results" / f"{time.strftime('%Y-%m-%d')}-{platform.machine()}-tmatd.json"
+    meta = {
+        "scatrs": scatrs.__version__,
+        "tmatd": {k: v for k, v in FILES.items()},
+        "machine": platform.machine(),
+        "processor": platform.processor(),
+        "gfortran": subprocess.run(
+            ["gfortran", "--version"], capture_output=True, text=True
+        ).stdout.splitlines()[0],
+    }
+    out.write_text(json.dumps({"meta": meta, "cases": results}, indent=1) + "\n")
+    print(f"wrote {out}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/docs/specs/2026-10-14-phase-4-spheroids-design.md b/docs/specs/2026-10-14-phase-4-spheroids-design.md
index e429222..28fd5d2 100644
--- a/docs/specs/2026-10-14-phase-4-spheroids-design.md
+++ b/docs/specs/2026-10-14-phase-4-spheroids-design.md
@@ -346,4 +346,10 @@ impl<T: Real> TMatrix<T> {
     - the brute-force orientation average (10⁻¹⁵);
     - 2π∫F₁₁ = C_sca;
     - the external comparisons.
-- **External comparisons.** `tmatd` and pytmatrix need downloads, which wait for the user's approval.
+- **External comparisons** (approved by the user, 2026-10-15).
+  - `tmatd`: `benchmarks/compare_tmatd.py` fetches Mishchenko's `tmq.lp.f`, `lpq.f` and `tmq.par.f`, checks them by SHA-256 and patches them (input, output digits, gfortran's quad intrinsics). The sources are not committed: no license is stated, only free public access.
+  - Agreement over 72 spheroids: Q_ext and Q_sca to 7.5·10⁻¹¹ at tmatd's tolerance 10⁻⁹; g and the Greek coefficients to the floor of its 10⁻⁷-wide size distribution (about 10⁻⁸, worst 2.9·10⁻⁷).
+  - β₁ and β₂ have opposite signs: tmatd expands in P⁰²ₗ = −d⁰²ₗ.
+  - tmatd stops at its NPN1 = 140 limit for 5 extreme spheroids that scatrs computes.
+  - Speed: scatrs is 4–7× faster up to x = 6, but tmatd is faster at x = 10, by up to 3× (the refined modes). The target of section 10 is met up to x = 6 only.
+  - pytmatrix 0.3.3 does not build: it needs `numpy.distutils`, and its sdist lacks the generated f2py source. It also wraps Mishchenko's own code, so it is left out.
diff --git a/docs/validation/external.md b/docs/validation/external.md
index d54542a..f8861f5 100644
--- a/docs/validation/external.md
+++ b/docs/validation/external.md
@@ -292,6 +292,69 @@ core radius half the outer) 4.3 to 8.4 times faster:
 | 100 | 10.9 | 83.7 | 58.3 |
 | 1000 | 104 | | 451 |
 
+## Spheroids
+
+### Limits
+
+- **A sphere.** An aspect ratio of 1 gives every output of {mod}`scatrs.mie`, to $10^{-9}$ or
+  better (`tests/test_tmatrix.py` and `scatrs/src/tmatrix/mod.rs`).
+- **The Rayleigh limit.** At $x = 0.01$ the orientation-averaged absorption and scattering of
+  oblate and prolate spheroids ($\varepsilon$ from 1/4 to 4, $m = 1.5 + 0.1i$) follow the
+  electrostatic polarizabilities $\alpha_j = V(m^2 - 1)/(1 + L_j(m^2 - 1))$ of their
+  depolarization factors $L_j$ (Bohren & Huffman 1983, §5.3) to $10^{-3}$, the size of the
+  first correction.
+
+### Comparison with Mishchenko's T-matrix code
+
+Mishchenko's extended-precision code for randomly oriented particles (`tmq.lp.f`, Mishchenko &
+Travis 1994; the "tmatd" of the literature) is the standard reference for spheroids. The script
+`benchmarks/compare_tmatd.py` fetches it from its page at NASA GISS (it is not redistributed
+here), builds it with gfortran after patches that change no computation (the input read from
+standard input, 16 printed digits, gfortran's names for its quad-precision intrinsics), and
+compares it with scatrs at `rtol = 1e-10` for 72 spheroids: aspect ratios 1/5, 1/3, 1/2, 2, 3
+and 5, $x$ = 1, 3, 6 and 10, and indices 1.33, $1.53 + 0.008i$ and $1.5 + 0.5i$. One particle is
+computed as the code's notes describe, as a size distribution $10^{-7}$ wide. The results are in
+`benchmarks/results/2026-10-15-arm64-tmatd.json`. Apple M4 Pro, macOS; scatrs 0.3.0,
+GNU Fortran 14.2, 2026-10-15.
+
+| output | at tmatd's tolerance $10^{-6}$: median, largest difference | at $10^{-9}$: median, largest |
+|---|---|---|
+| $Q_\text{ext}$ | $1.6\cdot10^{-9}$, $6.4\cdot10^{-8}$ | $7.0\cdot10^{-13}$, $7.5\cdot10^{-11}$ |
+| $Q_\text{sca}$ | $2.1\cdot10^{-9}$, $8.6\cdot10^{-8}$ | $9.3\cdot10^{-13}$, $6.8\cdot10^{-11}$ |
+| $g$ | $1.2\cdot10^{-8}$, $4.1\cdot10^{-8}$ | $1.1\cdot10^{-8}$, $3.7\cdot10^{-8}$ |
+| Greek coefficients | $3.6\cdot10^{-7}$, $4.1\cdot10^{-6}$ | $4.0\cdot10^{-8}$, $2.9\cdot10^{-7}$ |
+
+The differences are relative ($g$ absolute, the Greek coefficients relative to the largest
+$\alpha_1$, normalized). They shrink with tmatd's tolerance to the floor of its narrow size
+distribution, about $10^{-8}$ in $g$ and the Greek coefficients; scatrs's own estimates are
+$10^{-11}$ or below. tmatd expands $F_{12}$ and $F_{34}$ in the generalized spherical functions
+$P^l_{02} = -d^l_{02}$, so its $\beta_1$ and $\beta_2$ have the opposite signs of scatrs's, which
+uses the Wigner d-functions; the phase matrices agree.
+
+tmatd stops ("convergence is not obtained") for five spheroids: $\varepsilon = 1/5$ at $x = 10$
+for every index, $\varepsilon = 1/5$ at $x = 6$ and $\varepsilon = 5$ at $x = 10$ for the absorbing
+index (at the tighter tolerance). There its number of orders grows to its limit of 140 without
+the cross sections settling: the loss of precision of the classical integrals, which scatrs
+removes ({doc}`/theory/spheroids`). scatrs computes them with about 45 orders.
+
+On one thread, the median time per spheroid, scatrs at `rtol` $10^{-6}$ and $10^{-8}$, tmatd at
+its tolerances $10^{-6}$ and $10^{-9}$ (the five failures excluded):
+
+| $x$ | scatrs, $10^{-6}$ (s) | scatrs, $10^{-8}$ (s) | tmatd, $10^{-6}$ (s) | tmatd, $10^{-9}$ (s) |
+|---|---|---|---|---|
+| 1 | 0.006 | 0.008 | 0.026 | 0.041 |
+| 3 | 0.020 | 0.024 | 0.080 | 0.18 |
+| 6 | 0.061 | 0.095 | 0.35 | 0.47 |
+| 10 | 2.1 | 3.0 | 1.2 | 1.7 |
+
+scatrs is 4 to 7 times faster up to $x = 6$. At $x = 10$ the modes that need refinement in
+double-double dominate its time, and tmatd is faster for most spheroids: up to 3 times for the
+extreme aspect ratios, where scatrs refines most (the slowest, $\varepsilon = 1/5$ with the
+absorbing index, took 154 s, a spheroid tmatd cannot compute).
+
+We thank M. I. Mishchenko, L. D. Travis and the NASA Goddard Institute for Space Studies for
+making the code publicly available; its references are listed in {doc}`/references`.
+
 ## Size distributions
 
 For averages over size distributions, {doc}`sasktran2` compares scatrs with the Mie
````

## Hand-off

- **Performance.** After the speed work of 2043466, e220d7e and b34f757 (row-selective refinement, parallel node blocks, n_θ = 2N, an amplitude-based mode cutoff, only the kept forms in double-double), scatrs is faster than tmatd on one thread for all 67 spheroids of the comparison. Targets still not met: the derivatives cost 4–8 times the values (target 3); a refined mode still costs several times its f64 solve (target 4×, not re-measured per mode). Closed-form differentiated integrands remain the follow-up for the derivatives.
- **Phase 4b** (ensembles of spheroids) builds on `TMatrixSolution::greek_with_derivatives` and the aspect ratio as a real direction (spec section 4).
