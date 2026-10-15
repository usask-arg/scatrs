# Phase 3a: Coefficient Layer and Particle Abstraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the Mie outputs and the ensemble engine for layered spheres without changing any result: the outputs that need only aₙ and bₙ are computed on a shared coefficient view, and the engine solves a *particle* per component instead of a sphere of refractive index m.

**Architecture:**
- **`scatrs::mie` (crate-internal).** A `Coefficients<'a, T>` view (x, aₙ, bₙ, the absorption sum) carries the efficiencies, amplitudes (cosine and `AngularTable` forms), phase matrix and Greek coefficients. A `Group<'a, T>` carries the derivatives with respect to one size parameter and one refractive index (`CoefficientDerivatives` and the absorption sum's three derivatives). `MieSolution` keeps its public API and delegates. A layered sphere will have one group per layer.
- **`scatrs::ensemble`.** A public `Particle` enum (`Particle::Sphere { refractive_index }`, non-exhaustive) replaces `Component`'s `refractive_index` field. `Component::new(distribution, m, weight)` still makes spheres, and `Component::with_particle` is added.
  - Inside the engine, a per-thread `Solver` turns a particle at one wavelength (`Optics`) and a size parameter into coefficients and derivative groups.
  - The derivatives with respect to the particle are *directions*: Re m and Im m of each group, in order.
  - The panel products of kernel A hold 4 + 4·groups columns per sphere.
- **Binding and Python.** Components cross the binding as `(distribution spec, particle spec, weight)`, with the particle spec `("sphere", m)`. The batch binding takes a particle spec too. `Sphere._engine(wavelength)` builds the spec, which is the 2c-1 follow-up's particle-side hook.

**Tech Stack:** Rust 2024 (MSRV 1.85; CI lints with 1.99), pyo3/numpy 0.29, Python ≥ 3.11, pytest, ruff, criterion.

**Spec:** `docs/specs/2026-10-14-phase-3-layered-spheres-design.md`, sections 1.1 and 3. The refinements made while planning are recorded in its section 14.

## Global Constraints

- **Bitwise.** Every output of `scatrs::mie` and of the ensemble engine is bitwise unchanged, whether called from Rust or from Python. The checks are in the appendix: the Rust snapshot, the Python snapshot, and `benchmarks/compare.py`'s scatrs outputs.
- **No expected values change.** No existing test changes its expected values. Tests change only where they call the private bindings, whose item format changes.
- **No speed regression** beyond noise in the ensemble criterion benchmarks. Compare on one thread (`RAYON_NUM_THREADS=1`), since multi-threaded timings on a laptop vary by ±10%.
- **Public Rust API:**
  - `MieSolution` is unchanged.
  - `ensemble::Component`'s public field `refractive_index` becomes `particle`. This is a breaking change, recorded for the next release's notes in phase 3c.
  - Everything else is additive: `Particle`, `Particle::sphere` and `Component::with_particle`.
- **The Python public API is unchanged.** The private bindings change.
- **Checks at every commit:**
  - `cargo fmt --all`;
  - `cargo +1.99.0 clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace`;
  - `RUSTDOCFLAGS="-D warnings" cargo doc -p scatrs --no-deps`;
  - `uv run pytest -q`, `uv run ruff check .` and `uv run ruff format --check .`;
  - `uv run python -m mypy.stubtest scatrs._scatrs`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never send personal data in network requests.

## How this plan was made

As in phases 1b, 2a and 2c-1, every task was prototyped and verified before this plan was written. The diffs below are the prototype's.

- Task 1 is commit `e7c5483`.
- Task 2 is commit `6760658`.
- Task 3 is commit `e5a5d27`.

Each task's checks in the table below were run on the prototype.

## Measured results the plan relies on

| What | Measured |
|---|---|
| Rust snapshot (appendix A): every Mie output, cosine and table forms, with derivatives and Greek coefficients, 7 x × 5 m; ensembles of six distribution kinds × four option sets, a mixture with all derivatives, a batch | 457,459 values; bitwise equal after task 1 and after task 2 |
| Python snapshot (appendix B): `Ensemble`/`Mixture`/grid/list with derivatives, Greek coefficients and the phase matrix | 1293 arrays; bitwise equal after task 3 |
| `benchmarks/compare.py --child` scatrs outputs, W1 and W2 (full size), at every rtol | 40 arrays each; bitwise equal (old build from `5a785c3` in its own environment) |
| Ensemble criterion benchmarks, one thread, two alternating runs each | w1 1.100/1.108 s → 1.084/1.073 s; batch 2.398/2.408 s → 2.379/2.354 s (no regression; the 1–2% gain is within noise) |
| Tests | Rust 360 → 361 passed (a new doctest), 3 ignored; Python 381 → 384 passed (three new binding tests), 1 skipped |

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `scatrs/src/mie/coefficients.rs` | 1 | `Coefficients` and `Group`; `MieSolution::coefficients()` and `group()` |
| `scatrs/src/mie/{efficiencies,amplitudes,phase,greek}.rs` | 1 | implementations on `Coefficients`; `MieSolution` wrappers with the public docs |
| `scatrs/src/mie/mod.rs` | 1 | module and re-exports |
| `scatrs/src/ensemble/particle.rs` | 2 | `Particle::optics`, `wavelengths`, `directions`; `Optics`, `Solver`, `Solved` |
| `scatrs/src/ensemble/mod.rs` | 2 | public `Particle`; `Component { particle }`, `with_particle`; validation and parameters through the particle |
| `scatrs/src/ensemble/engine.rs` | 2 | passes 1 and 2, monodisperse parts and the wavelength loop on `Optics`/`Solver` and directions |
| `scatrs/src/ensemble/fast.rs` | 2 | `PanelProducts` with any number of derivative groups |
| `scatrs-py/src/ensemble.rs`, `python/scatrs/_scatrs/__init__.pyi` | 3 | particle specs in the bindings |
| `python/scatrs/particles.py`, `python/scatrs/ensemble.py` | 3 | `Sphere._engine`; `Ensemble`/`Mixture` build items through it |
| `tests/test_ensemble_engine_calls.py`, `tests/test_ensemble.py` | 3 | the new item format; binding errors |

---

### Task 1: Coefficient layer in `scatrs::mie`

**Files:**
- Create: `scatrs/src/mie/coefficients.rs`
- Modify: `scatrs/src/mie/mod.rs`, `efficiencies.rs`, `amplitudes.rs`, `phase.rs`, `greek.rs`

**Interfaces:**
- **Produces (crate-internal):**
  - `mie::Coefficients<'a, T> { x, a, b, abs_sum }` with `n_terms`, `efficiencies`, `efficiencies_jacobian(&Group)`, `amplitudes`, `amplitudes_from_table`, `amplitudes_with_derivatives(mu, &Group, …)`, `amplitudes_with_derivatives_from_table(table, &Group, …)`, `phase_matrix`, `phase_matrix_from_table`, `phase_matrix_with_derivatives(mu, &Group, norm, …)`, `phase_matrix_with_derivatives_from_table(table, &Group, norm, …)`, `greek(order, norm)` and `greek_with_derivatives(order, &Group, norm)`;
  - `mie::Group<'a, T> { d: CoefficientDerivatives, abs_sum: [T; 3] }`;
  - `MieSolution::coefficients()` and `MieSolution::group() -> Option<Group>`.
- **Unchanged:** every public method of `MieSolution`, its docs and its error messages. The derivative variants check for derivatives first and return "… derivatives need a solution from Mie::compute_with_derivatives", in the same order of checks as before.

- [ ] **Step 1: Record the baselines.** On the commit before the task, build and run the Rust snapshot (appendix A): `cargo run -q --release -- snap-before.txt`. Run the Python snapshot (appendix B): `uv run python pysnap.py py-before.npz`. Save the `ensemble` criterion baseline: `RAYON_NUM_THREADS=1 cargo bench -p scatrs --bench ensemble -- --save-baseline before`.
- [ ] **Step 2: Implement.** Apply the diff below. It moves code verbatim from `impl MieSolution` to `impl Coefficients`. The only rewrites are `self.state.x`/`a`/`b`/`abs_sum` → `self.x`/`a`/`b`/`abs_sum`, and derivatives read from the `Group` argument.
- [ ] **Step 3: Check bitwise equality.**

Run: the Rust snapshot to `snap-t1.txt`, then `cmp snap-before.txt snap-t1.txt`
Expected: no output (equal).

- [ ] **Step 4: Run the checks** (Global Constraints).

Expected: Rust `360 passed`, 3 ignored; clippy and rustdoc clean.

- [ ] **Step 5: Commit:** `refactor(mie): outputs on a shared coefficient view` (as `e7c5483`).

The diff (commit `e7c5483`):

````diff
diff --git a/scatrs/src/mie/amplitudes.rs b/scatrs/src/mie/amplitudes.rs
index 199d5b7..fb1e2ac 100644
--- a/scatrs/src/mie/amplitudes.rs
+++ b/scatrs/src/mie/amplitudes.rs
@@ -2,7 +2,7 @@
 
 use num_complex::Complex;
 
-use super::{MAX_MIE_ORDER, MieSolution};
+use super::{Coefficients, Group, MAX_MIE_ORDER, MieSolution};
 use crate::scalar::Real;
 use crate::{Error, Result};
 
@@ -174,7 +174,82 @@ impl<T: Real> AngularTable<T> {
     }
 }
 
+/// The error for derivatives of a solution computed without them.
+pub(super) fn no_derivatives(what: &str) -> Error {
+    Error::InvalidInput(format!(
+        "{what} derivatives need a solution from Mie::compute_with_derivatives"
+    ))
+}
+
 impl<T: Real> MieSolution<'_, T> {
+    /// S₁ and S₂ at the cosines of `table`, as [`amplitudes`](Self::amplitudes) gives them
+    /// (bitwise).
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if a buffer's length differs from the number of cosines, or the
+    /// solution has more terms than the table has orders. Nothing is written then.
+    pub fn amplitudes_from_table(
+        &self,
+        table: &AngularTable<T>,
+        out: AmplitudeBuffers<'_, T>,
+    ) -> Result<()> {
+        self.coefficients().amplitudes_from_table(table, out)
+    }
+
+    /// S₁, S₂ and their derivatives at the cosines of `table`, as
+    /// [`amplitudes_with_derivatives`](Self::amplitudes_with_derivatives) gives them
+    /// (bitwise).
+    ///
+    /// # Errors
+    ///
+    /// As [`amplitudes_from_table`](Self::amplitudes_from_table), and
+    /// [`Error::InvalidInput`] if the solution was computed without derivatives.
+    pub fn amplitudes_with_derivatives_from_table(
+        &self,
+        table: &AngularTable<T>,
+        values: AmplitudeBuffers<'_, T>,
+        d_dx: AmplitudeBuffers<'_, T>,
+        d_dm: AmplitudeBuffers<'_, T>,
+    ) -> Result<()> {
+        let g = self.group().ok_or_else(|| no_derivatives("amplitude"))?;
+        self.coefficients()
+            .amplitudes_with_derivatives_from_table(table, &g, values, d_dx, d_dm)
+    }
+
+    /// S₁(μ) and S₂(μ) at the cosines `mu` of the scattering angle (Bohren & Huffman 1983,
+    /// Eq. 4.74): S₁ = Σ (2n + 1)/(n(n + 1)) (aₙπₙ + bₙτₙ) and
+    /// S₂ = Σ (2n + 1)/(n(n + 1)) (aₙτₙ + bₙπₙ).
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if a cosine is NaN or outside [−1, 1] (for `Dual`, its value),
+    /// or a buffer's length differs from `mu.len()`. Nothing is written then.
+    pub fn amplitudes(&self, mu: &[T], out: AmplitudeBuffers<'_, T>) -> Result<()> {
+        self.coefficients().amplitudes(mu, out)
+    }
+
+    /// S₁, S₂ and their derivatives: `d_dx` receives ∂S/∂x and `d_dm` the complex
+    /// derivative dS/dm (∂/∂(Re m) = d/dm, ∂/∂(Im m) = i·d/dm).
+    ///
+    /// # Errors
+    ///
+    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] if the solution was
+    /// computed without derivatives.
+    pub fn amplitudes_with_derivatives(
+        &self,
+        mu: &[T],
+        values: AmplitudeBuffers<'_, T>,
+        d_dx: AmplitudeBuffers<'_, T>,
+        d_dm: AmplitudeBuffers<'_, T>,
+    ) -> Result<()> {
+        let g = self.group().ok_or_else(|| no_derivatives("amplitude"))?;
+        self.coefficients()
+            .amplitudes_with_derivatives(mu, &g, values, d_dx, d_dm)
+    }
+}
+
+impl<T: Real> Coefficients<'_, T> {
     /// The number of terms, if `table` has as many orders.
     fn table_terms(&self, table: &AngularTable<T>) -> Result<usize> {
         let n = self.n_terms();
@@ -187,14 +262,8 @@ impl<T: Real> MieSolution<'_, T> {
         Ok(n)
     }
 
-    /// S₁ and S₂ at the cosines of `table`, as [`amplitudes`](Self::amplitudes) gives them
-    /// (bitwise).
-    ///
-    /// # Errors
-    ///
-    /// [`Error::InvalidInput`] if a buffer's length differs from the number of cosines, or the
-    /// solution has more terms than the table has orders. Nothing is written then.
-    pub fn amplitudes_from_table(
+    /// As [`MieSolution::amplitudes_from_table`].
+    pub(crate) fn amplitudes_from_table(
         &self,
         table: &AngularTable<T>,
         out: AmplitudeBuffers<'_, T>,
@@ -204,7 +273,7 @@ impl<T: Real> MieSolution<'_, T> {
         let zero = Complex::new(T::zero(), T::zero());
         out.s1.fill(zero);
         out.s2.fill(zero);
-        let (a, b) = (&self.state.a, &self.state.b);
+        let (a, b) = (self.a, self.b);
         for (k, (pi, tau)) in table.orders(n).enumerate() {
             let (an, bn) = (a[k], b[k]);
             let s = out.s1.iter_mut().zip(out.s2.iter_mut());
@@ -216,26 +285,17 @@ impl<T: Real> MieSolution<'_, T> {
         Ok(())
     }
 
-    /// S₁, S₂ and their derivatives at the cosines of `table`, as
-    /// [`amplitudes_with_derivatives`](Self::amplitudes_with_derivatives) gives them
-    /// (bitwise).
-    ///
-    /// # Errors
-    ///
-    /// As [`amplitudes_from_table`](Self::amplitudes_from_table), and
-    /// [`Error::InvalidInput`] if the solution was computed without derivatives.
-    pub fn amplitudes_with_derivatives_from_table(
+    /// As [`MieSolution::amplitudes_with_derivatives_from_table`], with the derivatives of
+    /// `group`.
+    pub(crate) fn amplitudes_with_derivatives_from_table(
         &self,
         table: &AngularTable<T>,
+        group: &Group<'_, T>,
         values: AmplitudeBuffers<'_, T>,
         d_dx: AmplitudeBuffers<'_, T>,
         d_dm: AmplitudeBuffers<'_, T>,
     ) -> Result<()> {
-        let Some(d) = self.derivatives() else {
-            return Err(Error::InvalidInput(
-                "amplitude derivatives need a solution from Mie::compute_with_derivatives".into(),
-            ));
-        };
+        let d = &group.d;
         let len = table.mu.len();
         values.check(len, "amplitude")?;
         d_dx.check(len, "derivative")?;
@@ -247,7 +307,7 @@ impl<T: Real> MieSolution<'_, T> {
         for buffer in &mut s {
             buffer.fill(zero);
         }
-        let (a, b) = (&self.state.a, &self.state.b);
+        let (a, b) = (self.a, self.b);
         for (k, (pi, tau)) in table.orders(n).enumerate() {
             let pairs = [
                 (a[k], b[k]),
@@ -263,21 +323,12 @@ impl<T: Real> MieSolution<'_, T> {
         }
         Ok(())
     }
-}
 
-impl<T: Real> MieSolution<'_, T> {
-    /// S₁(μ) and S₂(μ) at the cosines `mu` of the scattering angle (Bohren & Huffman 1983,
-    /// Eq. 4.74): S₁ = Σ (2n + 1)/(n(n + 1)) (aₙπₙ + bₙτₙ) and
-    /// S₂ = Σ (2n + 1)/(n(n + 1)) (aₙτₙ + bₙπₙ).
-    ///
-    /// # Errors
-    ///
-    /// [`Error::InvalidInput`] if a cosine is NaN or outside [−1, 1] (for `Dual`, its value),
-    /// or a buffer's length differs from `mu.len()`. Nothing is written then.
-    pub fn amplitudes(&self, mu: &[T], out: AmplitudeBuffers<'_, T>) -> Result<()> {
+    /// As [`MieSolution::amplitudes`].
+    pub(crate) fn amplitudes(&self, mu: &[T], out: AmplitudeBuffers<'_, T>) -> Result<()> {
         out.check(mu.len(), "amplitude")?;
         check_cosines(mu)?;
-        let (a, b) = (&self.state.a, &self.state.b);
+        let (a, b) = (self.a, self.b);
         for (j, &muj) in mu.iter().enumerate() {
             let zero = Complex::new(T::zero(), T::zero());
             let (mut s1, mut s2) = (zero, zero);
@@ -292,30 +343,21 @@ impl<T: Real> MieSolution<'_, T> {
         Ok(())
     }
 
-    /// S₁, S₂ and their derivatives: `d_dx` receives ∂S/∂x and `d_dm` the complex
-    /// derivative dS/dm (∂/∂(Re m) = d/dm, ∂/∂(Im m) = i·d/dm).
-    ///
-    /// # Errors
-    ///
-    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] if the solution was
-    /// computed without derivatives.
-    pub fn amplitudes_with_derivatives(
+    /// As [`MieSolution::amplitudes_with_derivatives`], with the derivatives of `group`.
+    pub(crate) fn amplitudes_with_derivatives(
         &self,
         mu: &[T],
+        group: &Group<'_, T>,
         values: AmplitudeBuffers<'_, T>,
         d_dx: AmplitudeBuffers<'_, T>,
         d_dm: AmplitudeBuffers<'_, T>,
     ) -> Result<()> {
-        let Some(d) = self.derivatives() else {
-            return Err(Error::InvalidInput(
-                "amplitude derivatives need a solution from Mie::compute_with_derivatives".into(),
-            ));
-        };
+        let d = &group.d;
         values.check(mu.len(), "amplitude")?;
         d_dx.check(mu.len(), "derivative")?;
         d_dm.check(mu.len(), "derivative")?;
         check_cosines(mu)?;
-        let (a, b) = (&self.state.a, &self.state.b);
+        let (a, b) = (self.a, self.b);
         for (j, &muj) in mu.iter().enumerate() {
             let zero = Complex::new(T::zero(), T::zero());
             let mut s = [zero; 6]; // S₁, S₂, ∂S₁/∂x, ∂S₂/∂x, dS₁/dm, dS₂/dm
diff --git a/scatrs/src/mie/coefficients.rs b/scatrs/src/mie/coefficients.rs
new file mode 100644
index 0000000..25a1d00
--- /dev/null
+++ b/scatrs/src/mie/coefficients.rs
@@ -0,0 +1,57 @@
+//! The coefficients aₙ, bₙ of a particle, from which every output of [`MieSolution`] follows:
+//! the efficiencies, amplitudes, phase matrix and Greek coefficients are implemented once on a
+//! [`Coefficients`] view, and their derivatives on a [`Group`] of coefficient derivatives.
+//!
+//! A sphere has one group, with respect to x and m.
+
+use num_complex::Complex;
+
+use super::{CoefficientDerivatives, MieSolution};
+use crate::scalar::Real;
+
+/// The coefficients aₙ, bₙ (n = 1, …, N at index n − 1) of a particle of size parameter `x`,
+/// which scales every output, and the absorption sum (x²/2)·Q_abs.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct Coefficients<'a, T> {
+    pub(crate) x: T,
+    pub(crate) a: &'a [Complex<T>],
+    pub(crate) b: &'a [Complex<T>],
+    pub(crate) abs_sum: T,
+}
+
+/// The derivatives of the coefficients with respect to a real size parameter and a refractive
+/// index, in which they are holomorphic, and those of the absorption sum with respect to the
+/// size parameter, Re m and Im m.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct Group<'a, T> {
+    pub(crate) d: CoefficientDerivatives<'a, T>,
+    pub(crate) abs_sum: [T; 3],
+}
+
+impl<T: Real> Coefficients<'_, T> {
+    /// The number of terms N.
+    pub(crate) fn n_terms(&self) -> usize {
+        self.a.len()
+    }
+}
+
+impl<'a, T: Real> MieSolution<'a, T> {
+    /// The sphere's coefficients.
+    pub(crate) fn coefficients(&self) -> Coefficients<'a, T> {
+        let s = self.state;
+        Coefficients {
+            x: s.x,
+            a: &s.a,
+            b: &s.b,
+            abs_sum: s.abs_sum,
+        }
+    }
+
+    /// The derivatives with respect to x and m, if computed.
+    pub(crate) fn group(&self) -> Option<Group<'a, T>> {
+        self.derivatives().map(|d| Group {
+            d,
+            abs_sum: self.state.abs_sum_jacobian,
+        })
+    }
+}
diff --git a/scatrs/src/mie/efficiencies.rs b/scatrs/src/mie/efficiencies.rs
index 7ce3ffd..9024b80 100644
--- a/scatrs/src/mie/efficiencies.rs
+++ b/scatrs/src/mie/efficiencies.rs
@@ -2,7 +2,7 @@
 
 use num_complex::Complex;
 
-use super::MieSolution;
+use super::{Coefficients, Group, MieSolution};
 use crate::scalar::Real;
 
 /// Efficiencies (cross sections divided by the geometric cross section πr²) and the
@@ -117,17 +117,30 @@ fn derivative_sums<T: Real>(
 impl<T: Real> MieSolution<'_, T> {
     /// The efficiencies and the asymmetry parameter.
     pub fn efficiencies(&self) -> Efficiencies<T> {
-        self.efficiencies_from(&value_sums(&self.state.a, &self.state.b))
+        self.coefficients().efficiencies()
+    }
+
+    /// The derivatives of the efficiencies with respect to x, Re m and Im m, if the solution
+    /// was computed by [`Mie::compute_with_derivatives`](super::Mie::compute_with_derivatives).
+    pub fn efficiencies_jacobian(&self) -> Option<EfficiencyJacobian<T>> {
+        let g = self.group()?;
+        Some(self.coefficients().efficiencies_jacobian(&g))
+    }
+}
+
+impl<T: Real> Coefficients<'_, T> {
+    /// The efficiencies and the asymmetry parameter.
+    pub(crate) fn efficiencies(&self) -> Efficiencies<T> {
+        self.efficiencies_from(&value_sums(self.a, self.b))
     }
 
     /// The efficiencies from the value sums of [`value_sums`].
     fn efficiencies_from(&self, sums: &Sums<T>) -> Efficiencies<T> {
-        let s = self.state;
-        let inv_x = s.x.recip();
+        let inv_x = self.x.recip();
         let inv_x2 = inv_x * inv_x;
         let two_x2 = T::from_f64(2.0) * inv_x2;
         let q_sca = two_x2 * sums.sca;
-        let q_abs = two_x2 * s.abs_sum;
+        let q_abs = two_x2 * self.abs_sum;
         let g = if sums.sca == T::zero() {
             T::zero()
         } else {
@@ -142,31 +155,28 @@ impl<T: Real> MieSolution<'_, T> {
         }
     }
 
-    /// The derivatives of the efficiencies with respect to x, Re m and Im m, if the solution
-    /// was computed by [`Mie::compute_with_derivatives`](super::Mie::compute_with_derivatives).
-    pub fn efficiencies_jacobian(&self) -> Option<EfficiencyJacobian<T>> {
-        let s = self.state;
-        if !s.derivatives {
-            return None;
-        }
+    /// The derivatives of the efficiencies with respect to the size parameter and the real and
+    /// imaginary parts of the refractive index of `group`.
+    pub(crate) fn efficiencies_jacobian(&self, group: &Group<'_, T>) -> EfficiencyJacobian<T> {
+        let (a, b, d) = (self.a, self.b, &group.d);
         let i1 = Complex::new(T::zero(), T::one());
         let two = T::from_f64(2.0);
-        let inv_x = s.x.recip();
+        let inv_x = self.x.recip();
         let inv_x2 = inv_x * inv_x;
         let two_x2 = two * inv_x2;
-        let values = value_sums(&s.a, &s.b);
+        let values = value_sums(a, b);
         let q = self.efficiencies_from(&values);
         let mut out = [q; 3];
         for (p, o) in out.iter_mut().enumerate() {
             let sums = match p {
-                0 => derivative_sums(&s.a, &s.b, |k| s.da_dx[k], |k| s.db_dx[k]),
-                1 => derivative_sums(&s.a, &s.b, |k| s.da_dm[k], |k| s.db_dm[k]),
-                _ => derivative_sums(&s.a, &s.b, |k| i1 * s.da_dm[k], |k| i1 * s.db_dm[k]),
+                0 => derivative_sums(a, b, |k| d.da_dx[k], |k| d.db_dx[k]),
+                1 => derivative_sums(a, b, |k| d.da_dm[k], |k| d.db_dm[k]),
+                _ => derivative_sums(a, b, |k| i1 * d.da_dm[k], |k| i1 * d.db_dm[k]),
             };
             // Q = (2/x²)·sum: ∂Q/∂x = (2/x²)·∂sum/∂x − (2/x)·Q.
             let x_term = |v: T| if p == 0 { two * inv_x * v } else { T::zero() };
             let q_sca = two_x2 * sums.sca - x_term(q.q_sca);
-            let q_abs = two_x2 * s.abs_sum_jacobian[p] - x_term(q.q_abs);
+            let q_abs = two_x2 * group.abs_sum[p] - x_term(q.q_abs);
             let q_back = two * re_mul_conj(sums.back, values.back) * inv_x2 - x_term(q.q_back);
             // g = 2G/S with S the scattering sum: ∂g = (2∂G − g ∂S)/S (x cancels).
             let g = if values.sca == T::zero() {
@@ -183,6 +193,6 @@ impl<T: Real> MieSolution<'_, T> {
             };
         }
         let [x, m_re, m_im] = out;
-        Some(EfficiencyJacobian { x, m_re, m_im })
+        EfficiencyJacobian { x, m_re, m_im }
     }
 }
diff --git a/scatrs/src/mie/greek.rs b/scatrs/src/mie/greek.rs
index 75691b6..1e62457 100644
--- a/scatrs/src/mie/greek.rs
+++ b/scatrs/src/mie/greek.rs
@@ -1,6 +1,7 @@
 //! Greek (generalized-spherical-function) expansion coefficients of the phase matrix.
 
-use super::{MAX_MIE_ORDER, MieSolution, Normalization, PhaseBuffers};
+use super::amplitudes::no_derivatives;
+use super::{Coefficients, Group, MAX_MIE_ORDER, MieSolution, Normalization, PhaseBuffers};
 use crate::scalar::Real;
 use crate::special::{GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre, wigner_d};
 use crate::{Error, Result};
@@ -322,6 +323,46 @@ fn buffers<T>(b: &mut [Vec<T>; 4]) -> PhaseBuffers<'_, T> {
 }
 
 impl<T: Real> MieSolution<'_, T> {
+    /// The Greek coefficients up to `order` (default: all, 2N), exact up to rounding: the phase
+    /// matrix at the nodes of an (N + ⌈(L + 1)/2⌉)-point Gauss–Legendre rule, projected. Orders
+    /// above 2N are exactly 0. Normalized coefficients are divided by the projected α₁ at l = 0,
+    /// so that it is exactly 1; unnormalized ones are those times Q_sca.
+    ///
+    /// The cost is the amplitudes at the N + ⌈(L + 1)/2⌉ nodes, O(N²) for L ≈ 2N, and four
+    /// Wigner recurrences up to L at each, O(N·L): for all orders, about 0.09 s at x = 10³ and
+    /// 7.6 s at x = 10⁴ in `f64` on an Apple M4 Pro.
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`], or if the rule would need more
+    /// than [`MAX_GAUSS_LEGENDRE_POINTS`] nodes (N + ⌈(min(L, 2N) + 1)/2⌉ > 2¹⁷, so N ≥ 2¹⁶,
+    /// x above about 6.5·10⁴, for all orders).
+    pub fn greek(
+        &self,
+        order: Option<usize>,
+        normalization: Normalization,
+    ) -> Result<GreekCoefficients<T>> {
+        self.coefficients().greek(order, normalization)
+    }
+
+    /// The Greek coefficients and their derivatives with respect to x, Re m and Im m.
+    ///
+    /// # Errors
+    ///
+    /// As [`greek`](Self::greek), and [`Error::InvalidInput`] if the solution was computed
+    /// without derivatives.
+    pub fn greek_with_derivatives(
+        &self,
+        order: Option<usize>,
+        normalization: Normalization,
+    ) -> Result<(GreekCoefficients<T>, [GreekCoefficients<T>; 3])> {
+        let g = self.group().ok_or_else(|| no_derivatives("Greek"))?;
+        self.coefficients()
+            .greek_with_derivatives(order, &g, normalization)
+    }
+}
+
+impl<T: Real> Coefficients<'_, T> {
     /// The order L asked for (default 2N) and the projector for orders up to min(L, 2N): the
     /// coefficients of an N-term sphere vanish above 2N.
     fn greek_projector(&self, order: Option<usize>) -> Result<(usize, GreekProjector<T>)> {
@@ -339,21 +380,8 @@ impl<T: Real> MieSolution<'_, T> {
         Ok((order, GreekProjector::new(projected, nodes)?))
     }
 
-    /// The Greek coefficients up to `order` (default: all, 2N), exact up to rounding: the phase
-    /// matrix at the nodes of an (N + ⌈(L + 1)/2⌉)-point Gauss–Legendre rule, projected. Orders
-    /// above 2N are exactly 0. Normalized coefficients are divided by the projected α₁ at l = 0,
-    /// so that it is exactly 1; unnormalized ones are those times Q_sca.
-    ///
-    /// The cost is the amplitudes at the N + ⌈(L + 1)/2⌉ nodes, O(N²) for L ≈ 2N, and four
-    /// Wigner recurrences up to L at each, O(N·L): for all orders, about 0.09 s at x = 10³ and
-    /// 7.6 s at x = 10⁴ in `f64` on an Apple M4 Pro.
-    ///
-    /// # Errors
-    ///
-    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`], or if the rule would need more
-    /// than [`MAX_GAUSS_LEGENDRE_POINTS`] nodes (N + ⌈(min(L, 2N) + 1)/2⌉ > 2¹⁷, so N ≥ 2¹⁶,
-    /// x above about 6.5·10⁴, for all orders).
-    pub fn greek(
+    /// As [`MieSolution::greek`].
+    pub(crate) fn greek(
         &self,
         order: Option<usize>,
         normalization: Normalization,
@@ -372,22 +400,14 @@ impl<T: Real> MieSolution<'_, T> {
         Ok(g)
     }
 
-    /// The Greek coefficients and their derivatives with respect to x, Re m and Im m.
-    ///
-    /// # Errors
-    ///
-    /// As [`greek`](Self::greek), and [`Error::InvalidInput`] if the solution was computed
-    /// without derivatives.
-    pub fn greek_with_derivatives(
+    /// As [`MieSolution::greek_with_derivatives`], with the derivatives of `group`.
+    pub(crate) fn greek_with_derivatives(
         &self,
         order: Option<usize>,
+        group: &Group<'_, T>,
         normalization: Normalization,
     ) -> Result<(GreekCoefficients<T>, [GreekCoefficients<T>; 3])> {
-        let Some(jac) = self.efficiencies_jacobian() else {
-            return Err(Error::InvalidInput(
-                "Greek derivatives need a solution from Mie::compute_with_derivatives".into(),
-            ));
-        };
+        let jac = self.efficiencies_jacobian(group);
         let (order, projector) = self.greek_projector(order)?;
         let n = projector.nodes().len();
         let mut f: [[Vec<T>; 4]; 4] =
@@ -396,6 +416,7 @@ impl<T: Real> MieSolution<'_, T> {
             let [v, dx, dr, di] = &mut f;
             self.phase_matrix_with_derivatives(
                 projector.nodes(),
+                group,
                 Normalization::Unnormalized,
                 buffers(v),
                 [buffers(dx), buffers(dr), buffers(di)],
diff --git a/scatrs/src/mie/mod.rs b/scatrs/src/mie/mod.rs
index e007290..d4747ac 100644
--- a/scatrs/src/mie/mod.rs
+++ b/scatrs/src/mie/mod.rs
@@ -70,6 +70,7 @@
 //! use them to check the closed-form derivatives.
 
 mod amplitudes;
+mod coefficients;
 mod domain;
 mod efficiencies;
 mod greek;
@@ -83,6 +84,7 @@ use crate::{Error, Result};
 
 pub(crate) use amplitudes::for_each_order;
 pub use amplitudes::{AmplitudeBuffers, AngularTable};
+pub(crate) use coefficients::{Coefficients, Group};
 pub use domain::{MieOutput, VALIDATED_DOMAIN, ValidatedDomain};
 pub use efficiencies::{Efficiencies, EfficiencyJacobian};
 pub use greek::{GreekCoefficients, GreekProjector, MAX_GREEK_ORDER, PhaseTables};
diff --git a/scatrs/src/mie/phase.rs b/scatrs/src/mie/phase.rs
index beb77be..f146f72 100644
--- a/scatrs/src/mie/phase.rs
+++ b/scatrs/src/mie/phase.rs
@@ -2,7 +2,8 @@
 
 use num_complex::Complex;
 
-use super::{AmplitudeBuffers, AngularTable, MieSolution};
+use super::amplitudes::no_derivatives;
+use super::{AmplitudeBuffers, AngularTable, Coefficients, Group, MieSolution};
 use crate::scalar::Real;
 use crate::{Error, Result};
 
@@ -71,10 +72,83 @@ pub(crate) fn element_derivatives<T: Real>(
 }
 
 impl<T: Real> MieSolution<'_, T> {
+    /// The phase matrix F₁₁, F₁₂, F₃₃, F₃₄ at the cosines `mu` (Bohren & Huffman 1983, §4.4),
+    /// scaled by `normalization`.
+    ///
+    /// # Errors
+    ///
+    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] if a buffer's length
+    /// differs from `mu.len()`. Nothing is written then.
+    pub fn phase_matrix(
+        &self,
+        mu: &[T],
+        normalization: Normalization,
+        out: PhaseBuffers<'_, T>,
+    ) -> Result<()> {
+        self.coefficients().phase_matrix(mu, normalization, out)
+    }
+
+    /// The phase matrix at the cosines of `table`, as [`phase_matrix`](Self::phase_matrix)
+    /// gives it (bitwise).
+    ///
+    /// # Errors
+    ///
+    /// As [`amplitudes_from_table`](Self::amplitudes_from_table).
+    pub fn phase_matrix_from_table(
+        &self,
+        table: &AngularTable<T>,
+        normalization: Normalization,
+        out: PhaseBuffers<'_, T>,
+    ) -> Result<()> {
+        self.coefficients()
+            .phase_matrix_from_table(table, normalization, out)
+    }
+
+    /// The phase matrix and its derivatives with respect to x, Re m and Im m, in `d[0]`, `d[1]`
+    /// and `d[2]`. For [`Normalization::Normalized`] the derivatives include those of Q_sca.
+    ///
+    /// # Errors
+    ///
+    /// As [`phase_matrix`](Self::phase_matrix), and [`Error::InvalidInput`] if the solution was
+    /// computed without derivatives.
+    pub fn phase_matrix_with_derivatives(
+        &self,
+        mu: &[T],
+        normalization: Normalization,
+        values: PhaseBuffers<'_, T>,
+        d: [PhaseBuffers<'_, T>; 3],
+    ) -> Result<()> {
+        let g = self.group().ok_or_else(|| no_derivatives("phase-matrix"))?;
+        self.coefficients()
+            .phase_matrix_with_derivatives(mu, &g, normalization, values, d)
+    }
+
+    /// The phase matrix and its derivatives at the cosines of `table`, as
+    /// [`phase_matrix_with_derivatives`](Self::phase_matrix_with_derivatives) gives them
+    /// (bitwise).
+    ///
+    /// # Errors
+    ///
+    /// As [`phase_matrix_from_table`](Self::phase_matrix_from_table), and
+    /// [`Error::InvalidInput`] if the solution was computed without derivatives.
+    pub fn phase_matrix_with_derivatives_from_table(
+        &self,
+        table: &AngularTable<T>,
+        normalization: Normalization,
+        values: PhaseBuffers<'_, T>,
+        d: [PhaseBuffers<'_, T>; 3],
+    ) -> Result<()> {
+        let g = self.group().ok_or_else(|| no_derivatives("phase-matrix"))?;
+        self.coefficients()
+            .phase_matrix_with_derivatives_from_table(table, &g, normalization, values, d)
+    }
+}
+
+impl<T: Real> Coefficients<'_, T> {
     /// The scale of the phase matrix: 4/(x²Q_sca) for [`Normalization::Normalized`], 4/x² for
     /// [`Normalization::Unnormalized`].
     fn phase_scale(&self, normalization: Normalization) -> T {
-        let x = self.x();
+        let x = self.x;
         let four_x2 = T::from_f64(4.0) / (x * x);
         match normalization {
             Normalization::Normalized => four_x2 / self.efficiencies().q_sca,
@@ -82,14 +156,8 @@ impl<T: Real> MieSolution<'_, T> {
         }
     }
 
-    /// The phase matrix F₁₁, F₁₂, F₃₃, F₃₄ at the cosines `mu` (Bohren & Huffman 1983, §4.4),
-    /// scaled by `normalization`.
-    ///
-    /// # Errors
-    ///
-    /// As [`amplitudes`](Self::amplitudes), and [`Error::InvalidInput`] if a buffer's length
-    /// differs from `mu.len()`. Nothing is written then.
-    pub fn phase_matrix(
+    /// As [`MieSolution::phase_matrix`].
+    pub(crate) fn phase_matrix(
         &self,
         mu: &[T],
         normalization: Normalization,
@@ -109,13 +177,8 @@ impl<T: Real> MieSolution<'_, T> {
         Ok(())
     }
 
-    /// The phase matrix at the cosines of `table`, as [`phase_matrix`](Self::phase_matrix)
-    /// gives it (bitwise).
-    ///
-    /// # Errors
-    ///
-    /// As [`amplitudes_from_table`](Self::amplitudes_from_table).
-    pub fn phase_matrix_from_table(
+    /// As [`MieSolution::phase_matrix_from_table`].
+    pub(crate) fn phase_matrix_from_table(
         &self,
         table: &AngularTable<T>,
         normalization: Normalization,
@@ -154,16 +217,11 @@ impl<T: Real> MieSolution<'_, T> {
         }
     }
 
-    /// The phase matrix and its derivatives with respect to x, Re m and Im m, in `d[0]`, `d[1]`
-    /// and `d[2]`. For [`Normalization::Normalized`] the derivatives include those of Q_sca.
-    ///
-    /// # Errors
-    ///
-    /// As [`phase_matrix`](Self::phase_matrix), and [`Error::InvalidInput`] if the solution was
-    /// computed without derivatives.
-    pub fn phase_matrix_with_derivatives(
+    /// As [`MieSolution::phase_matrix_with_derivatives`], with the derivatives of `group`.
+    pub(crate) fn phase_matrix_with_derivatives(
         &self,
         mu: &[T],
+        group: &Group<'_, T>,
         normalization: Normalization,
         values: PhaseBuffers<'_, T>,
         d: [PhaseBuffers<'_, T>; 3],
@@ -172,26 +230,22 @@ impl<T: Real> MieSolution<'_, T> {
             let [s1, s2, dx1, dx2, dm1, dm2] = s;
             self.amplitudes_with_derivatives(
                 mu,
+                group,
                 AmplitudeBuffers { s1, s2 },
                 AmplitudeBuffers { s1: dx1, s2: dx2 },
                 AmplitudeBuffers { s1: dm1, s2: dm2 },
             )
         })?;
-        self.phase_derivatives_from_amplitudes(&s, normalization, values, d);
+        self.phase_derivatives_from_amplitudes(&s, group, normalization, values, d);
         Ok(())
     }
 
-    /// The phase matrix and its derivatives at the cosines of `table`, as
-    /// [`phase_matrix_with_derivatives`](Self::phase_matrix_with_derivatives) gives them
-    /// (bitwise).
-    ///
-    /// # Errors
-    ///
-    /// As [`phase_matrix_from_table`](Self::phase_matrix_from_table), and
-    /// [`Error::InvalidInput`] if the solution was computed without derivatives.
-    pub fn phase_matrix_with_derivatives_from_table(
+    /// As [`MieSolution::phase_matrix_with_derivatives_from_table`], with the derivatives of
+    /// `group`.
+    pub(crate) fn phase_matrix_with_derivatives_from_table(
         &self,
         table: &AngularTable<T>,
+        group: &Group<'_, T>,
         normalization: Normalization,
         values: PhaseBuffers<'_, T>,
         d: [PhaseBuffers<'_, T>; 3],
@@ -200,16 +254,17 @@ impl<T: Real> MieSolution<'_, T> {
             let [s1, s2, dx1, dx2, dm1, dm2] = s;
             self.amplitudes_with_derivatives_from_table(
                 table,
+                group,
                 AmplitudeBuffers { s1, s2 },
                 AmplitudeBuffers { s1: dx1, s2: dx2 },
                 AmplitudeBuffers { s1: dm1, s2: dm2 },
             )
         })?;
-        self.phase_derivatives_from_amplitudes(&s, normalization, values, d);
+        self.phase_derivatives_from_amplitudes(&s, group, normalization, values, d);
         Ok(())
     }
 
-    /// After checking that the solution has derivatives and the buffers' lengths are `len`,
+    /// After checking that the buffers' lengths are `len`,
     /// S₁, S₂, ∂S₁/∂x, ∂S₂/∂x, dS₁/dm, dS₂/dm from `amplitudes`.
     fn phase_derivative_amplitudes(
         &self,
@@ -218,12 +273,6 @@ impl<T: Real> MieSolution<'_, T> {
         d: &[PhaseBuffers<'_, T>; 3],
         amplitudes: impl FnOnce([&mut [Complex<T>]; 6]) -> Result<()>,
     ) -> Result<[Vec<Complex<T>>; 6]> {
-        if self.derivatives().is_none() {
-            return Err(Error::InvalidInput(
-                "phase-matrix derivatives need a solution from Mie::compute_with_derivatives"
-                    .into(),
-            ));
-        }
         values.check(len, "phase-matrix")?;
         for buffers in d {
             buffers.check(len, "derivative")?;
@@ -239,15 +288,14 @@ impl<T: Real> MieSolution<'_, T> {
     fn phase_derivatives_from_amplitudes(
         &self,
         s: &[Vec<Complex<T>>; 6],
+        group: &Group<'_, T>,
         normalization: Normalization,
         values: PhaseBuffers<'_, T>,
         d: [PhaseBuffers<'_, T>; 3],
     ) {
-        let jac = self
-            .efficiencies_jacobian()
-            .expect("a solution with derivatives");
+        let jac = self.efficiencies_jacobian(group);
         // The scale c and its derivatives: c = 4/(x²Q) or 4/x².
-        let x = self.x();
+        let x = self.x;
         let c = self.phase_scale(normalization);
         let two = T::from_f64(2.0);
         let dc = match normalization {
````

### Task 2: Particles in the ensemble engine

**Files:**
- Create: `scatrs/src/ensemble/particle.rs`
- Modify: `scatrs/src/ensemble/mod.rs`, `engine.rs`, `fast.rs`

**Interfaces:**
- **Consumes:** task 1's `Coefficients`, `Group`, `MieSolution::coefficients()` and `group()`.
- **Produces (public):**
  - `ensemble::Particle::Sphere { refractive_index }` (non-exhaustive enum) and `Particle::sphere(m)`;
  - `Component { distribution, particle, weight }`, `Component::with_particle(distribution, particle, weight)`;
  - `Component::new` unchanged.
- **Produces (crate-internal):**
  - `Particle::optics(i) -> Optics`, `Particle::wavelengths()`, `Particle::directions()` (2 per group);
  - `Optics::Sphere(m)` and `Optics::n_terms(x, opts)`;
  - `Solver::new()` and `Solver::solve(optics, x, opts, derivatives) -> Solved { c, groups }`;
  - `PanelProducts::new(tables, spheres, groups)`, `fits(…, groups)`, `set(k, &Coefficients, &[Group])`, and `phase(…, d: &mut [&mut [f64]])`, where `d` holds Re m and Im m of each group in order and is empty for none.
- **Rules that keep results bitwise:**
  - Pass 1 integrates only the requested directions.
  - Pass 2 computes every direction of the particle when any is requested, as before for Re m and Im m. A part's block count, and so the window of panels summed together, is unchanged.
  - Per group, `quantities2` calls `phase_matrix_with_derivatives` once, and r-derivatives use group 0's ∂/∂x.

- [ ] **Step 1: Implement** — the diff below. Before it, the engine read m in the solves and in the number of terms only; `MAX_DX` is a constant and does not read m.
- [ ] **Step 2: Check bitwise equality and speed.**

Run: the Rust snapshot to `snap-t2.txt` and `cmp snap-before.txt snap-t2.txt`. Then, alternating with the baseline commit, run `RAYON_NUM_THREADS=1 cargo bench -p scatrs --bench ensemble -- "batch|w1$"`.
Expected: equal; times within noise of the baseline.

- [ ] **Step 3: Run the checks.**

Expected: Rust `361 passed` (the `with_particle` doctest is new), 3 ignored; Python unchanged (`381 passed`, 1 skipped); clippy and rustdoc clean.

- [ ] **Step 4: Commit:** `refactor(ensemble): components of particles; the engine solves particles`.

The diff:

````diff
diff --git a/scatrs/src/ensemble/engine.rs b/scatrs/src/ensemble/engine.rs
index bd0e38a..cecd81c 100644
--- a/scatrs/src/ensemble/engine.rs
+++ b/scatrs/src/ensemble/engine.rs
@@ -5,21 +5,21 @@
 use core::f64::consts::PI;
 
 use super::matrix::{Mat, matmul};
-use num_complex::Complex;
 use rayon::prelude::*;
 
 use super::distribution::{SizeDistribution, Support};
 use super::fast::{BandProjection, BandTables, PanelProducts};
 use super::lin::Lin;
 use super::moments::{Moments, moments};
+use super::particle::{Optics, Solver};
 use super::range::{self, MIE_POWERS, MOMENT_POWERS, STEP};
 use super::{
-    Component, EnsembleOptions, EnsembleResult, GreekOrder, Outputs, ParamKind, PhaseValues,
-    kronrod,
+    Component, EnsembleOptions, EnsembleResult, GreekOrder, Outputs, ParamKind, Particle,
+    PhaseValues, kronrod,
 };
 use crate::mie::{
-    GreekCoefficients, GreekProjector, MAX_SIZE_PARAMETER, Mie, MieOptions, MieSolution,
-    Normalization, PhaseBuffers, n_terms,
+    Coefficients, EfficiencyJacobian, GreekCoefficients, GreekProjector, MAX_SIZE_PARAMETER,
+    MieOptions, Normalization, PhaseBuffers,
 };
 use crate::{Error, Result};
 
@@ -41,8 +41,8 @@ const SCALARS: usize = 8;
 /// Pass 1 integrates the per-sphere quantities 1, C_ext, C_sca, C_sca·g, C_back and C_abs (the
 /// first four at the indices of the flat layout, C_back at `P1_BACK`, C_abs at `P1_ABS`), then the
 /// optional Greek coefficients (each πr²·α, 6 arrays of `refine_order` + 1) and the
-/// refractive-index derivatives (`M_GROUP` per part of m: C_ext, C_sca, C_sca·g, C_back,
-/// C_abs), and refines on all of them but C_back and its derivatives unless
+/// derivatives along the requested directions of the particle (`M_GROUP` per direction: C_ext,
+/// C_sca, C_sca·g, C_back, C_abs), and refines on all of them but C_back and its derivatives unless
 /// `refine_backscatter`. The moments r² and r³ come from their own integration (`moments`).
 const N_BASE: usize = 6;
 const P1_BACK: usize = 4;
@@ -68,7 +68,7 @@ struct Part<'a> {
     component: usize,
     mode: Option<usize>,
     dist: &'a SizeDistribution,
-    m: &'a [Complex<f64>],
+    particle: &'a Particle,
     /// The component's weight, and the mode's weight and the sum of the component's mode
     /// weights (1 and 1 without modes).
     w: f64,
@@ -82,16 +82,23 @@ struct Part<'a> {
     /// The requested distribution parameters: (column of the result, index into the
     /// distribution's gradient).
     theta: Vec<(usize, usize)>,
-    /// The result columns of ∂/∂(Re m) and ∂/∂(Im m), if requested.
-    m_cols: [Option<usize>; 2],
+    /// The result column of each of the particle's derivative directions, if requested.
+    dir_cols: Vec<Option<usize>>,
 }
 
 impl Part<'_> {
-    fn m_derivatives(&self) -> bool {
-        self.m_cols.iter().any(Option::is_some)
+    /// Whether any derivative along the particle's directions is requested.
+    fn dir_derivatives(&self) -> bool {
+        self.dir_cols.iter().any(Option::is_some)
     }
 }
 
+/// The derivative of the efficiencies along direction `k`: Re m or Im m of group k/2.
+fn direction(jacs: &[EfficiencyJacobian<f64>], k: usize) -> &crate::mie::Efficiencies<f64> {
+    let j = &jacs[k / 2];
+    if k % 2 == 0 { &j.m_re } else { &j.m_im }
+}
+
 fn context(e: Error, lambda: f64, r: f64) -> Error {
     let at = format!("at wavelength {lambda:e}, radius {r:e}");
     match e {
@@ -128,7 +135,7 @@ fn parts<'a>(
         for (mode, v, dist) in modes {
             let names = dist.parameter_names();
             let mut theta = Vec::new();
-            let mut m_cols = [None, None];
+            let mut dir_cols = vec![None; c.particle.directions()];
             for (col, p) in options.derivatives.iter().enumerate() {
                 if p.component != ci {
                     continue;
@@ -138,8 +145,8 @@ fn parts<'a>(
                         let k = names.iter().position(|n| *n == name).expect("validated");
                         theta.push((col, k));
                     }
-                    ParamKind::RefrReal => m_cols[0] = Some(col),
-                    ParamKind::RefrImag => m_cols[1] = Some(col),
+                    ParamKind::RefrReal => dir_cols[0] = Some(col),
+                    ParamKind::RefrImag => dir_cols[1] = Some(col),
                     _ => {}
                 }
             }
@@ -172,7 +179,7 @@ fn parts<'a>(
                 component: ci,
                 mode,
                 dist,
-                m: &c.refractive_index,
+                particle: &c.particle,
                 w: c.weight,
                 v,
                 v_total,
@@ -180,7 +187,7 @@ fn parts<'a>(
                 cutoffs,
                 moments,
                 theta,
-                m_cols,
+                dir_cols,
             });
         }
     }
@@ -327,7 +334,7 @@ struct Sums1 {
     err: Vec<f64>,
     /// Kronrod sums of |∂u/∂θ|·r·|q|, the scales of the θ blocks.
     abs: Vec<f64>,
-    /// Kronrod sums of u·r·|q|, the scales of the refractive-index derivatives.
+    /// Kronrod sums of u·r·|q|, the scales of the derivatives along the particle's directions.
     absv: Vec<f64>,
     /// The largest number of terms at the nodes.
     n_terms: usize,
@@ -337,26 +344,25 @@ struct Sums1 {
 }
 
 /// The pass-1 quantities at radius r: 1, C_ext, C_sca, C_sca·g, C_back, then the Greek
-/// coefficients up to `refine_order` (πr²·α), then the derivatives of C_ext, C_sca, C_sca·g and
-/// C_back with respect to Re m and Im m where `m_derivatives` asks for them; and the number of
-/// terms.
+/// coefficients up to `refine_order` (πr²·α), then the derivatives of C_ext, C_sca, C_sca·g,
+/// C_back and C_abs along each of the particle's directions that `dirs` asks for; and the
+/// number of terms.
+#[allow(clippy::too_many_arguments)]
 fn quantities1(
-    mie: &mut Mie<f64>,
+    solver: &mut Solver,
     r: f64,
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     opts: &MieOptions,
     refine_order: Option<usize>,
-    m_derivatives: [bool; 2],
+    dirs: &[bool],
 ) -> Result<(Vec<f64>, usize)> {
     let x = size(r, lambda);
-    let s = if m_derivatives.iter().any(|&b| b) {
-        mie.compute_with_derivatives(x, m, opts)
-    } else {
-        mie.compute(x, m, opts)
-    }
-    .map_err(|e| context(e, lambda, r))?;
-    let q = s.efficiencies();
+    let s = solver
+        .solve(optics, x, opts, dirs.iter().any(|&b| b))
+        .map_err(|e| context(e, lambda, r))?;
+    let c = s.c;
+    let q = c.efficiencies();
     let area = PI * r * r;
     let mut out = vec![
         1.0,
@@ -367,7 +373,7 @@ fn quantities1(
         area * q.q_abs,
     ];
     if let Some(order) = refine_order {
-        let g = s
+        let g = c
             .greek(Some(order), Normalization::Unnormalized)
             .map_err(|e| context(e, lambda, r))?;
         for a in [
@@ -376,26 +382,28 @@ fn quantities1(
             out.extend(a.iter().map(|v| area * v));
         }
     }
-    if let Some(jac) = s.efficiencies_jacobian() {
-        for (want, d) in m_derivatives.into_iter().zip([jac.m_re, jac.m_im]) {
-            if want {
-                out.extend([
-                    area * d.q_ext,
-                    area * d.q_sca,
-                    area * (d.q_sca * q.g + q.q_sca * d.g),
-                    area * d.q_back,
-                    area * d.q_abs,
-                ]);
-            }
-        }
+    let jacs: Vec<_> = s
+        .groups
+        .iter()
+        .map(|g| c.efficiencies_jacobian(g))
+        .collect();
+    for (k, _) in dirs.iter().enumerate().filter(|&(_, &want)| want) {
+        let d = direction(&jacs, k);
+        out.extend([
+            area * d.q_ext,
+            area * d.q_sca,
+            area * (d.q_sca * q.g + q.q_sca * d.g),
+            area * d.q_back,
+            area * d.q_abs,
+        ]);
     }
-    Ok((out, s.n_terms()))
+    Ok((out, c.n_terms()))
 }
 
 /// The number of pass-1 quantities: the base ones, the Greek coefficients, and `M_GROUP`
-/// derivatives per requested refractive-index part.
+/// derivatives per requested direction.
 fn n_refined(refine_order: Option<usize>, part: &Part<'_>) -> usize {
-    let md = part.m_cols.iter().filter(|c| c.is_some()).count();
+    let md = part.dir_cols.iter().filter(|c| c.is_some()).count();
     N_BASE + refine_order.map_or(0, |l| 6 * (l + 1)) + M_GROUP * md
 }
 
@@ -403,22 +411,22 @@ fn n_refined(refine_order: Option<usize>, part: &Part<'_>) -> usize {
 /// its quantities weighted by each part's density.
 #[allow(clippy::too_many_arguments)]
 fn panel1(
-    mie: &mut Mie<f64>,
+    solver: &mut Solver,
     group: &[&Part<'_>],
     (a, b): (f64, f64),
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     opts: &MieOptions,
     refine_order: Option<usize>,
     n_refined: usize,
 ) -> Result<Vec<Sums1>> {
     let nodes = kronrod::panel(a, b);
     let first = group.first().expect("a part");
-    let m_derivatives = [first.m_cols[0].is_some(), first.m_cols[1].is_some()];
+    let dirs: Vec<bool> = first.dir_cols.iter().map(Option::is_some).collect();
     let mut n_max = 0;
     let mut qs = Vec::with_capacity(nodes.len());
     for &(t, _, _) in &nodes {
-        let (q, n) = quantities1(mie, t.exp(), lambda, m, opts, refine_order, m_derivatives)?;
+        let (q, n) = quantities1(solver, t.exp(), lambda, optics, opts, refine_order, &dirs)?;
         n_max = n_max.max(n);
         qs.push(q);
     }
@@ -497,7 +505,7 @@ fn evaluate(
     panels: &mut Panels1,
     group: &[&Part<'_>],
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     opts: &MieOptions,
     refine_order: Option<usize>,
     n_refined: usize,
@@ -506,13 +514,13 @@ fn evaluate(
     let results: Vec<Result<()>> = panels
         .par_iter_mut()
         .filter(|(_, s)| s.is_none())
-        .map_init(Mie::<f64>::new, |mie, (ab, s)| {
+        .map_init(Solver::new, |solver, (ab, s)| {
             *s = Some(panel1(
-                mie,
+                solver,
                 group,
                 *ab,
                 lambda,
-                m,
+                optics,
                 opts,
                 refine_order,
                 n_refined,
@@ -564,7 +572,8 @@ fn status(
             // of ⟨C_back⟩ and ⟨C_sca⟩, since backscattering by large weakly absorbing
             // spheres is a sum of narrow glory ripples orders of magnitude below C_sca.
             P1_BACK => total(P1_BACK).abs().max(c_sca),
-            // The refractive-index derivatives can cancel: their absolute integrands.
+            // The derivatives along the particle's directions can cancel: their absolute
+            // integrands.
             j if j >= n_greek => sums.iter().map(|s| s.absv[scale_of(j)]).sum::<f64>(),
             j if j >= N_BASE => c_sca,
             j => total(scale_of(j)).abs(),
@@ -581,15 +590,15 @@ fn status(
     for b in 0..=nt {
         target[b * n_refined + ONE] = f64::INFINITY;
     }
-    // ∫∂u/∂θ·∂C/∂m is no output: no targets for the refractive-index derivatives in the θ
-    // blocks.
+    // ∫∂u/∂θ·∂C/∂m is no output: no targets for the derivatives along the particle's
+    // directions in the θ blocks.
     for b in 1..=nt {
         for j in n_greek..n_refined {
             target[b * n_refined + j] = f64::INFINITY;
         }
     }
     if !options.refine_backscatter {
-        // C_back and its derivatives (the fourth of each refractive-index group) are only
+        // C_back and its derivatives (the fourth of each direction's group) are only
         // integrated.
         for b in 0..=nt {
             target[b * n_refined + P1_BACK] = f64::INFINITY;
@@ -636,12 +645,12 @@ fn status(
     }
 }
 
-/// Pass 1: the adaptive panels of a group of parts (sharing m) at one wavelength, refined
-/// until every part meets its targets.
+/// Pass 1: the adaptive panels of a group of parts (sharing a particle) at one wavelength,
+/// refined until every part meets its targets.
 fn pass1(
     group: &[&Part<'_>],
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     options: &EnsembleOptions,
     mie_opts: &MieOptions,
 ) -> Result<Pass1> {
@@ -718,7 +727,7 @@ fn pass1(
             &mut panels,
             group,
             lambda,
-            m,
+            optics,
             mie_opts,
             refine_order,
             n_refined,
@@ -869,11 +878,15 @@ fn bucket(n: usize) -> usize {
 
 /// The largest number of terms at the nodes of the panel [a, b] (t = ln r), which chooses the
 /// panel's band.
-fn panel_terms(a: f64, b: f64, lambda: f64, m: Complex<f64>, opts: &MieOptions) -> Result<usize> {
+fn panel_terms(a: f64, b: f64, lambda: f64, optics: Optics, opts: &MieOptions) -> Result<usize> {
     let mut most = 0;
     for &(t, _, _) in &kronrod::panel(a, b) {
         let r = t.exp();
-        most = most.max(n_terms(size(r, lambda), m, opts).map_err(|e| context(e, lambda, r))?);
+        most = most.max(
+            optics
+                .n_terms(size(r, lambda), opts)
+                .map_err(|e| context(e, lambda, r))?,
+        );
     }
     Ok(most)
 }
@@ -910,34 +923,32 @@ impl Layout<'_> {
 }
 
 /// The per-sphere quantities of pass 2 at radius r into `g` (the common part and the node's
-/// band, whose (start, length) it returns), and their derivatives with respect to Re m, Im m
-/// into `dm` and with respect to r into `dr`, if given. With `products` (the panel's products
-/// and the node's index in them), the band's tables are left to the products: the node's Mie
-/// coefficients go into them instead.
+/// band, whose (start, length) it returns), their derivatives along each of the particle's
+/// directions into `dm` and with respect to r into `dr`, if given. With `products` (the panel's
+/// products and the node's index in them), the band's tables are left to the products: the
+/// node's coefficients go into them instead.
 #[allow(clippy::too_many_arguments)]
 fn quantities2(
-    mie: &mut Mie<f64>,
+    solver: &mut Solver,
     r: f64,
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     opts: &MieOptions,
     layout: &Layout<'_>,
     g: &mut [f64],
-    mut dm: Option<[&mut [f64]; 2]>,
+    mut dm: Option<&mut [&mut [f64]]>,
     dr: Option<&mut [f64]>,
     band_terms: usize,
     products: Option<(&mut PanelProducts, usize)>,
 ) -> Result<Option<(usize, usize)>> {
     let x = size(r, lambda);
     let derivatives = dm.is_some() || dr.is_some();
-    let s = if derivatives {
-        mie.compute_with_derivatives(x, m, opts)
-    } else {
-        mie.compute(x, m, opts)
-    }
-    .map_err(|e| context(e, lambda, r))?;
+    let s = solver
+        .solve(optics, x, opts, derivatives)
+        .map_err(|e| context(e, lambda, r))?;
+    let c = s.c;
     let area = PI * r * r;
-    let q = s.efficiencies();
+    let q = c.efficiencies();
     g[..SCALARS].copy_from_slice(&[
         1.0,
         area * q.q_ext,
@@ -951,7 +962,7 @@ fn quantities2(
     // The band of the panel (its largest number of terms), so that every node of a panel
     // fills the same tables: per-node bands would make each band's table a step function
     // within the panel, and its Kronrod–Gauss difference a measure of the step.
-    let n = band_terms.max(s.n_terms());
+    let n = band_terms.max(c.n_terms());
     let band = layout.band(n).map(|b| (b.start, b.projector.nodes()));
     if !layout.bands.is_empty() && band.is_none() {
         return Err(context(
@@ -962,7 +973,7 @@ fn quantities2(
     }
     let mut tables = vec![(SCALARS, layout.mu)];
     match products {
-        Some((p, k)) => p.set(k, &s),
+        Some((p, k)) => p.set(k, &c, &s.groups),
         None => tables.extend(band),
     }
     let segment = band.map(|(start, nodes)| (start, 4 * nodes.len()));
@@ -972,15 +983,17 @@ fn quantities2(
             if n == 0 {
                 continue;
             }
-            phase_into(&s, cosines, &mut g[start..start + 4 * n])
+            phase_into(&c, cosines, &mut g[start..start + 4 * n])
                 .map_err(|e| context(e, lambda, r))?;
             g[start..start + 4 * n].iter_mut().for_each(|v| *v *= area);
         }
         return Ok(segment);
     }
-    let jac = s
-        .efficiencies_jacobian()
-        .expect("computed with derivatives");
+    let jacs: Vec<_> = s
+        .groups
+        .iter()
+        .map(|gr| c.efficiencies_jacobian(gr))
+        .collect();
     let scalar_d = |e: &crate::mie::Efficiencies<f64>| {
         [
             0.0,
@@ -994,14 +1007,15 @@ fn quantities2(
         ]
     };
     let k = 2.0 * PI / lambda;
-    if let Some([dre, dim]) = dm.as_mut() {
-        dre[..SCALARS].copy_from_slice(&scalar_d(&jac.m_re));
-        dim[..SCALARS].copy_from_slice(&scalar_d(&jac.m_im));
+    if let Some(dm) = dm.as_deref_mut() {
+        for (dir, d) in dm.iter_mut().enumerate() {
+            d[..SCALARS].copy_from_slice(&scalar_d(direction(&jacs, dir)));
+        }
     }
     let mut dr = dr;
     if let Some(dr) = dr.as_deref_mut() {
         // ∂(πr²Q)/∂r = 2πr Q + πr² (2π/λ) ∂Q/∂x.
-        let dx = scalar_d(&jac.x);
+        let dx = scalar_d(&jacs[0].x);
         let values = [0.0, q.q_ext, q.q_sca, q.q_sca * q.g, q.q_abs, q.q_back];
         for j in 1..6 {
             dr[j] = 2.0 * PI * r * values[j] + k * dx[j];
@@ -1016,15 +1030,20 @@ fn quantities2(
             continue;
         }
         let span = start..start + 4 * n;
-        let mut dx = vec![0.0; 4 * n];
-        let (mut dre_buf, mut dim_buf) = (vec![0.0; 4 * n], vec![0.0; 4 * n]);
-        {
+        // Per group: ∂/∂x, ∂/∂(Re m), ∂/∂(Im m).
+        let mut d: Vec<[Vec<f64>; 3]> = s
+            .groups
+            .iter()
+            .map(|_| core::array::from_fn(|_| vec![0.0; 4 * n]))
+            .collect();
+        for (group, [dx, dre_buf, dim_buf]) in s.groups.iter().zip(&mut d) {
             let [v11, v12, v33, v34] = quarter(&mut g[span.clone()]);
-            let [x11, x12, x33, x34] = quarter(&mut dx);
-            let [r11, r12, r33, r34] = quarter(&mut dre_buf);
-            let [i11, i12, i33, i34] = quarter(&mut dim_buf);
-            s.phase_matrix_with_derivatives(
+            let [x11, x12, x33, x34] = quarter(dx);
+            let [r11, r12, r33, r34] = quarter(dre_buf);
+            let [i11, i12, i33, i34] = quarter(dim_buf);
+            c.phase_matrix_with_derivatives(
                 cosines,
+                group,
                 Normalization::Unnormalized,
                 PhaseBuffers {
                     f11: v11,
@@ -1057,13 +1076,15 @@ fn quantities2(
         }
         if let Some(dr) = dr.as_deref_mut() {
             for (j, o) in span.clone().enumerate() {
-                dr[o] = 2.0 * PI * r * g[o] + area * k * dx[j];
+                dr[o] = 2.0 * PI * r * g[o] + area * k * d[0][0][j];
             }
         }
-        if let Some([dre, dim]) = dm.as_mut() {
-            for (j, o) in span.clone().enumerate() {
-                dre[o] = area * dre_buf[j];
-                dim[o] = area * dim_buf[j];
+        if let Some(dm) = dm.as_deref_mut() {
+            for (dir, out) in dm.iter_mut().enumerate() {
+                let buf = &d[dir / 2][1 + dir % 2];
+                for (j, o) in span.clone().enumerate() {
+                    out[o] = area * buf[j];
+                }
             }
         }
         g[span].iter_mut().for_each(|v| *v *= area);
@@ -1080,24 +1101,35 @@ fn quarter(t: &mut [f64]) -> [&mut [f64]; 4] {
     [a, b, c, d]
 }
 
-/// The unnormalized phase matrix of `s` at `cosines` into the 4·n values of `out`.
-fn phase_into(s: &MieSolution<'_, f64>, cosines: &[f64], out: &mut [f64]) -> Result<()> {
+/// The unnormalized phase matrix of `c` at `cosines` into the 4·n values of `out`.
+fn phase_into(c: &Coefficients<'_, f64>, cosines: &[f64], out: &mut [f64]) -> Result<()> {
     let [f11, f12, f33, f34] = quarter(out);
-    s.phase_matrix(
+    c.phase_matrix(
         cosines,
         Normalization::Unnormalized,
         PhaseBuffers { f11, f12, f33, f34 },
     )
 }
 
-/// Pass-2 integrals of one part: blocks of the flat layout (values; each requested θ;
-/// Re m, Im m if requested), Kronrod sums and Σ over panels of |Kronrod − Gauss|.
+/// Pass-2 integrals of one part: blocks of the flat layout (values; each requested θ; every
+/// direction of the particle if any is requested), Kronrod sums and Σ over panels of
+/// |Kronrod − Gauss|.
 struct Sums2 {
     s: Vec<f64>,
     err: Vec<f64>,
     blocks: usize,
 }
 
+/// The number of pass-2 blocks of a part.
+fn blocks(part: &Part<'_>) -> usize {
+    let dirs = if part.dir_derivatives() {
+        part.dir_cols.len()
+    } else {
+        0
+    };
+    1 + part.theta.len() + dirs
+}
+
 /// At most this many `CHUNK`s of panels have their sums held at once before they are added up
 /// in order, and fewer for large groups: about `WINDOW_BYTES` of sums. The window depends on
 /// the inputs only, never on the thread count.
@@ -1106,13 +1138,13 @@ const WINDOW_BYTES: usize = 64 << 20;
 
 /// The weights of a group's blocks at a panel's nodes: a row per (part, block) for the
 /// values (the values block, weight u·r, and each θ block, ∂u/∂θ·r) and a row per part with
-/// refractive-index derivatives (weight u·r, applied to the Re m and Im m derivatives),
-/// Kronrod rows first, then Gauss rows. A panel's sums for every part and block are then
+/// derivatives along the particle's directions (weight u·r, applied to each direction's
+/// derivatives), Kronrod rows first, then Gauss rows. A panel's sums for every part and block are then
 /// matrix products of these weights with the nodes' values.
 struct Rows {
     /// The first values row of each part.
     first: Vec<usize>,
-    /// The derivative row of each part with refractive-index derivatives.
+    /// The derivative row of each part with derivatives along the particle's directions.
     m_row: Vec<Option<usize>>,
     values: usize,
     m: usize,
@@ -1125,7 +1157,7 @@ impl Rows {
         for part in group {
             first.push(values);
             values += 1 + part.theta.len();
-            m_row.push(part.m_derivatives().then(|| {
+            m_row.push(part.dir_derivatives().then(|| {
                 m += 1;
                 m - 1
             }));
@@ -1139,42 +1171,45 @@ impl Rows {
     }
 }
 
-/// Pass 2 of a group of parts (sharing m) over their shared final panels: each node's Mie
-/// solution and tables once; each part's panel sums as matrix products of its weights with
-/// the nodes' values.
+/// Pass 2 of a group of parts (sharing a particle) over their shared final panels: each node's
+/// solution and tables once; each part's panel sums as matrix products of its weights with the
+/// nodes' values.
 fn pass2(
     group: &[&Part<'_>],
     panels: &[(f64, f64)],
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     opts: &MieOptions,
     layout: &Layout<'_>,
 ) -> Result<Vec<Sums2>> {
     let len = layout.len;
     let common = layout.common();
-    let blocks: Vec<usize> = group
-        .iter()
-        .map(|p| 1 + p.theta.len() + if p.m_derivatives() { 2 } else { 0 })
-        .collect();
-    let derivatives = group.iter().any(|p| p.m_derivatives());
+    let blocks: Vec<usize> = group.iter().map(|p| blocks(p)).collect();
+    let derivatives = group.iter().any(|p| p.dir_derivatives());
+    // Every direction of the particle when any is requested.
+    let n_dirs = if derivatives {
+        group[0].dir_cols.len()
+    } else {
+        0
+    };
     let rows = Rows::new(group);
     let n_nodes = kronrod::panel(0.0, 1.0).len();
     // Per-thread scratch. A panel touches the common part of the layout and its band's
     // tables: `stride` values, the common part first.
     struct Scratch {
-        mie: Mie<f64>,
+        solver: Solver,
         grads: Vec<Vec<f64>>,
         g: Vec<f64>,
         dmv: Vec<f64>,
         products: Option<PanelProducts>,
-        /// The nodes' values (nodes × stride), and their Re m and Im m derivatives.
+        /// The nodes' values (nodes × stride), and their derivatives along each direction.
         v: Mat,
-        dv: [Mat; 2],
+        dv: Vec<Mat>,
         /// The weights (2·rows × nodes) and the products (2·rows × stride).
         w: Mat,
         wm: Mat,
         pv: Mat,
-        pm: [Mat; 2],
+        pm: Vec<Mat>,
     }
     /// A chunk's sums and errors on one range [lo, hi) of the layout, every block.
     type Segment = ((usize, usize), Vec<f64>, Vec<f64>);
@@ -1202,20 +1237,20 @@ fn pass2(
             .par_chunks(CHUNK)
             .map_init(
                 || Scratch {
-                    mie: Mie::new(),
+                    solver: Solver::new(),
                     grads: group
                         .iter()
                         .map(|p| vec![0.0; p.dist.parameter_names().len()])
                         .collect(),
                     g: vec![0.0; len],
-                    dmv: vec![0.0; 2 * len],
+                    dmv: vec![0.0; n_dirs * len],
                     products: None,
                     v: Mat::zeros(0, 0),
-                    dv: [Mat::zeros(0, 0), Mat::zeros(0, 0)],
+                    dv: (0..n_dirs).map(|_| Mat::zeros(0, 0)).collect(),
                     w: Mat::zeros(2 * rows.values, n_nodes),
                     wm: Mat::zeros(2 * rows.m, n_nodes),
                     pv: Mat::zeros(0, 0),
-                    pm: [Mat::zeros(0, 0), Mat::zeros(0, 0)],
+                    pm: (0..n_dirs).map(|_| Mat::zeros(0, 0)).collect(),
                 },
                 |sc, chunk| {
                     let densities: Vec<_> = group.iter().map(|p| p.dist.prepare()).collect();
@@ -1224,7 +1259,7 @@ fn pass2(
                         let band_terms = if layout.bands.is_empty() {
                             0
                         } else {
-                            panel_terms(a, b, lambda, m, opts)?
+                            panel_terms(a, b, lambda, optics, opts)?
                         };
                         let band = if layout.bands.is_empty() {
                             None
@@ -1240,13 +1275,10 @@ fn pass2(
                             let fits = sc
                                 .products
                                 .as_ref()
-                                .is_some_and(|p| p.fits(&band.tables, nodes.len(), derivatives));
+                                .is_some_and(|p| p.fits(&band.tables, nodes.len(), n_dirs / 2));
                             if !fits {
-                                sc.products = Some(PanelProducts::new(
-                                    &band.tables,
-                                    nodes.len(),
-                                    derivatives,
-                                ));
+                                sc.products =
+                                    Some(PanelProducts::new(&band.tables, nodes.len(), n_dirs / 2));
                             }
                         }
                         let (lo, hi) = band.map_or((0, 0), |b| {
@@ -1255,34 +1287,26 @@ fn pass2(
                         let stride = common + (hi - lo);
                         if sc.v.nrows() != nodes.len() || sc.v.ncols() != stride {
                             sc.v = Mat::zeros(nodes.len(), stride);
-                            if derivatives {
-                                sc.dv = [
-                                    Mat::zeros(nodes.len(), stride),
-                                    Mat::zeros(nodes.len(), stride),
-                                ];
+                            for dv in &mut sc.dv {
+                                *dv = Mat::zeros(nodes.len(), stride);
                             }
                         }
                         for (k, &(t, wk, wg)) in nodes.iter().enumerate() {
                             let r = t.exp();
-                            let dm = if derivatives {
-                                let (re, im) = sc.dmv.split_at_mut(len);
-                                Some([re, im])
-                            } else {
-                                None
-                            };
+                            let mut dm: Vec<&mut [f64]> = sc.dmv.chunks_mut(len).collect();
                             let products = match (band, sc.products.as_mut()) {
                                 (Some(_), Some(p)) => Some((p, k)),
                                 _ => None,
                             };
                             quantities2(
-                                &mut sc.mie,
+                                &mut sc.solver,
                                 r,
                                 lambda,
-                                m,
+                                optics,
                                 opts,
                                 layout,
                                 &mut sc.g,
-                                dm,
+                                derivatives.then_some(&mut dm[..]),
                                 None,
                                 band_terms,
                                 products,
@@ -1321,13 +1345,9 @@ fn pass2(
                                 let r = t.exp();
                                 let x = size(r, lambda);
                                 let scale = 4.0 / (x * x) * (PI * r * r);
-                                let dm = if derivatives {
-                                    let (re, im) = sc.dmv.split_at_mut(len);
-                                    Some([&mut re[lo..hi], &mut im[lo..hi]])
-                                } else {
-                                    None
-                                };
-                                prod.phase(&band.tables, k, scale, &mut sc.g[lo..hi], dm);
+                                let mut dm: Vec<&mut [f64]> =
+                                    sc.dmv.chunks_mut(len).map(|d| &mut d[lo..hi]).collect();
+                                prod.phase(&band.tables, k, scale, &mut sc.g[lo..hi], &mut dm);
                                 for (j, &gv) in sc.g[lo..hi].iter().enumerate() {
                                     sc.v[(k, common + j)] = gv;
                                 }
@@ -1342,9 +1362,9 @@ fn pass2(
                         }
                         gemm(&mut sc.pv, &sc.w, &sc.v);
                         if rows.m > 0 {
-                            let [re, im] = &mut sc.pm;
-                            gemm(re, &sc.wm, &sc.dv[0]);
-                            gemm(im, &sc.wm, &sc.dv[1]);
+                            for (pm, dv) in sc.pm.iter_mut().zip(&sc.dv) {
+                                gemm(pm, &sc.wm, dv);
+                            }
                         }
                         // Each part's panel sums into the chunk's segments.
                         for (p, part) in group.iter().enumerate() {
@@ -1445,9 +1465,8 @@ fn averages(part: &Part<'_>, sums: &Sums2, len: usize, truncation: f64) -> PartA
         d.push((col, dv, de));
     }
     let nt = part.theta.len();
-    let mut c = 0;
-    for col in part.m_cols {
-        if let Some(col) = col {
+    for (c, col) in part.dir_cols.iter().enumerate() {
+        if let &Some(col) = col {
             let (sb, eb) = (block(1 + nt + c), eblock(1 + nt + c));
             let dv: Vec<f64> = (0..len).map(|i| sb[i] / z).collect();
             let de: Vec<f64> = (0..len)
@@ -1455,9 +1474,6 @@ fn averages(part: &Part<'_>, sums: &Sums2, len: usize, truncation: f64) -> PartA
                 .collect();
             d.push((col, dv, de));
         }
-        if part.m_derivatives() {
-            c += 1;
-        }
     }
     let mut out = PartAverages { v, e, d };
     if let Some(m) = &part.moments {
@@ -1486,27 +1502,35 @@ fn monodisperse(
     part: &Part<'_>,
     r: f64,
     lambda: f64,
-    m: Complex<f64>,
+    optics: Optics,
     opts: &MieOptions,
     layout: &Layout<'_>,
 ) -> Result<PartAverages> {
     let len = layout.len;
-    let mut mie = Mie::<f64>::new();
+    let mut solver = Solver::new();
     let mut g = vec![0.0; len];
-    let mut dmv = vec![0.0; 2 * len];
+    let mut dmv = vec![0.0; part.dir_cols.len() * len];
     let mut drv = vec![0.0; len];
     let radius_col = part.theta.first().map(|&(col, _)| col);
     {
-        let dm = if part.m_derivatives() {
-            let (re, im) = dmv.split_at_mut(len);
-            Some([re, im])
-        } else {
-            None
-        };
+        let mut dm: Vec<&mut [f64]> = dmv.chunks_mut(len).collect();
+        let dm = part.dir_derivatives().then_some(&mut dm[..]);
         let dr = radius_col.map(|_| drv.as_mut_slice());
-        let n = n_terms(size(r, lambda), m, opts).map_err(|e| context(e, lambda, r))?;
+        let n = optics
+            .n_terms(size(r, lambda), opts)
+            .map_err(|e| context(e, lambda, r))?;
         quantities2(
-            &mut mie, r, lambda, m, opts, layout, &mut g, dm, dr, n, None,
+            &mut solver,
+            r,
+            lambda,
+            optics,
+            opts,
+            layout,
+            &mut g,
+            dm,
+            dr,
+            n,
+            None,
         )?;
     }
     let zeros = vec![0.0; len];
@@ -1514,7 +1538,7 @@ fn monodisperse(
     if let Some(col) = radius_col {
         d.push((col, drv, zeros.clone()));
     }
-    for (c, col) in part.m_cols.iter().enumerate() {
+    for (c, col) in part.dir_cols.iter().enumerate() {
         if let Some(col) = col {
             d.push((*col, dmv[c * len..(c + 1) * len].to_vec(), zeros.clone()));
         }
@@ -1540,11 +1564,12 @@ fn wavelength(
     let mut n_max = 0;
     for group in groups {
         let g = members(group);
-        let m = g[0].m[index];
+        let optics = g[0].particle.optics(index);
         let p1 = match g[0].support {
             Support::Point(r) => {
-                let n =
-                    n_terms(size(r, lambda), m, &mie_opts).map_err(|e| context(e, lambda, r))?;
+                let n = optics
+                    .n_terms(size(r, lambda), &mie_opts)
+                    .map_err(|e| context(e, lambda, r))?;
                 Pass1 {
                     panels: Vec::new(),
                     converged: vec![true],
@@ -1552,7 +1577,7 @@ fn wavelength(
                     range: (r, r),
                 }
             }
-            _ => pass1(&g, lambda, m, options, &mie_opts)?,
+            _ => pass1(&g, lambda, optics, options, &mie_opts)?,
         };
         n_max = n_max.max(p1.n_terms);
         firsts.push(p1);
@@ -1574,13 +1599,13 @@ fn wavelength(
     if options.greek.is_some() {
         for (group, p1) in groups.iter().zip(&firsts) {
             let first = &parts[group[0]];
-            let m = first.m[index];
+            let optics = first.particle.optics(index);
             if let Support::Point(_) = first.support {
                 uppers.insert(bucket(p1.n_terms));
                 continue;
             }
             for &(a, b) in &p1.panels {
-                uppers.insert(bucket(panel_terms(a, b, lambda, m, &mie_opts)?));
+                uppers.insert(bucket(panel_terms(a, b, lambda, optics, &mie_opts)?));
             }
         }
     }
@@ -1610,17 +1635,15 @@ fn wavelength(
     let mut averages_of: Vec<Option<PartAverages>> = (0..parts.len()).map(|_| None).collect();
     for (group, p1) in groups.iter().zip(&firsts) {
         let g = members(group);
-        let m = g[0].m[index];
+        let optics = g[0].particle.optics(index);
         if let Support::Point(r) = g[0].support {
-            averages_of[group[0]] = Some(monodisperse(g[0], r, lambda, m, &mie_opts, &layout)?);
+            averages_of[group[0]] =
+                Some(monodisperse(g[0], r, lambda, optics, &mie_opts, &layout)?);
             continue;
         }
-        let sums = pass2(&g, &p1.panels, lambda, m, &mie_opts, &layout)?;
+        let sums = pass2(&g, &p1.panels, lambda, optics, &mie_opts, &layout)?;
         for ((&p, part), sums) in group.iter().zip(&g).zip(&sums) {
-            debug_assert_eq!(
-                sums.blocks,
-                1 + part.theta.len() + if part.m_derivatives() { 2 } else { 0 }
-            );
+            debug_assert_eq!(sums.blocks, blocks(part));
             let truncation = if matches!(part.support, Support::Unbounded(_)) {
                 0.2 * options.rtol
             } else {
diff --git a/scatrs/src/ensemble/fast.rs b/scatrs/src/ensemble/fast.rs
index 9b393a3..073828f 100644
--- a/scatrs/src/ensemble/fast.rs
+++ b/scatrs/src/ensemble/fast.rs
@@ -11,7 +11,10 @@ use super::matrix::{Mat, matmul, product};
 use num_complex::Complex;
 
 use crate::Result;
-use crate::mie::{GreekProjector, MieSolution, element_derivatives, elements, for_each_order};
+use crate::mie::{
+    Coefficients as MieCoefficients, GreekProjector, Group, element_derivatives, elements,
+    for_each_order,
+};
 use crate::special::wigner_d;
 
 /// G_E and G_O of one band (h × N, one column per order).
@@ -46,7 +49,7 @@ impl BandTables {
 }
 
 /// The right-hand sides of a panel's spheres (4 columns each: u and v, real and imaginary
-/// parts; 4 more with derivatives: du/dm, dv/dm) and the products.
+/// parts; 4 more per derivative group: du/dm, dv/dm) and the products.
 pub(crate) struct PanelProducts {
     per: usize,
     x: Mat,
@@ -55,9 +58,10 @@ pub(crate) struct PanelProducts {
 }
 
 impl PanelProducts {
-    /// Room for `spheres` spheres of at most `n` orders, at `h` half nodes.
-    pub(crate) fn new(tables: &BandTables, spheres: usize, derivatives: bool) -> Self {
-        let per = if derivatives { 8 } else { 4 };
+    /// Room for `spheres` spheres with `groups` derivative groups each, at the band's orders
+    /// and half nodes.
+    pub(crate) fn new(tables: &BandTables, spheres: usize, groups: usize) -> Self {
+        let per = 4 + 4 * groups;
         let (h, n) = (tables.ge.nrows(), tables.orders());
         Self {
             per,
@@ -67,46 +71,53 @@ impl PanelProducts {
         }
     }
 
-    /// Whether this fits `tables` and `spheres` spheres.
-    pub(crate) fn fits(&self, tables: &BandTables, spheres: usize, derivatives: bool) -> bool {
+    /// Whether this fits `tables` and `spheres` spheres with `groups` derivative groups.
+    pub(crate) fn fits(&self, tables: &BandTables, spheres: usize, groups: usize) -> bool {
         self.x.nrows() == tables.orders()
             && self.ce.nrows() == tables.ge.nrows()
-            && self.per == if derivatives { 8 } else { 4 }
+            && self.per == 4 + 4 * groups
             && self.x.ncols() == self.per * spheres
     }
 
-    /// Sphere `k`'s coefficients (zero beyond its number of terms); with derivatives, also
-    /// daₙ/dm and dbₙ/dm.
-    pub(crate) fn set(&mut self, k: usize, s: &MieSolution<'_, f64>) {
-        let (a, b) = (s.a(), s.b());
-        let d = if self.per == 8 { s.derivatives() } else { None };
+    /// The number of derivative groups.
+    fn groups(&self) -> usize {
+        (self.per - 4) / 4
+    }
+
+    /// Sphere `k`'s coefficients (zero beyond its number of terms) and daₙ/dm, dbₙ/dm of each
+    /// of the first `self.groups()` of `groups`.
+    pub(crate) fn set(
+        &mut self,
+        k: usize,
+        c: &MieCoefficients<'_, f64>,
+        groups: &[Group<'_, f64>],
+    ) {
+        let (a, b) = (c.a, c.b);
+        let groups = &groups[..self.groups()];
         let base = k * self.per;
         let n = self.x.nrows();
+        let mut put = |row: usize, col: usize, u: Complex<f64>, v: Complex<f64>| {
+            self.x[(row, col)] = u.re;
+            self.x[(row, col + 1)] = u.im;
+            self.x[(row, col + 2)] = v.re;
+            self.x[(row, col + 3)] = v.im;
+        };
+        // Order row + 1: u = a for odd orders (even rows), b for even ones.
+        let uv = |row: usize, a: Complex<f64>, b: Complex<f64>| {
+            if row % 2 == 0 { (a, b) } else { (b, a) }
+        };
         for row in 0..n {
-            let (u, v, du, dv) = if row < a.len() {
-                let (dan, dbn) = d
-                    .as_ref()
-                    .map_or((Complex::default(), Complex::default()), |d| {
-                        (d.da_dm[row], d.db_dm[row])
-                    });
-                // Order row + 1: u = a for odd orders (even rows), b for even ones.
-                if row % 2 == 0 {
-                    (a[row], b[row], dan, dbn)
-                } else {
-                    (b[row], a[row], dbn, dan)
+            if row < a.len() {
+                let (u, v) = uv(row, a[row], b[row]);
+                put(row, base, u, v);
+                for (i, g) in groups.iter().enumerate() {
+                    let (du, dv) = uv(row, g.d.da_dm[row], g.d.db_dm[row]);
+                    put(row, base + 4 + 4 * i, du, dv);
                 }
             } else {
-                Default::default()
-            };
-            self.x[(row, base)] = u.re;
-            self.x[(row, base + 1)] = u.im;
-            self.x[(row, base + 2)] = v.re;
-            self.x[(row, base + 3)] = v.im;
-            if self.per == 8 {
-                self.x[(row, base + 4)] = du.re;
-                self.x[(row, base + 5)] = du.im;
-                self.x[(row, base + 6)] = dv.re;
-                self.x[(row, base + 7)] = dv.im;
+                for i in 0..=groups.len() {
+                    put(row, base + 4 * i, Complex::default(), Complex::default());
+                }
             }
         }
     }
@@ -120,15 +131,15 @@ impl PanelProducts {
     }
 
     /// Sphere `k`'s phase matrix times `scale` at every node of the band (F₁₁, F₁₂, F₃₃, F₃₄,
-    /// each n_q values, into `out`), and with derivatives its derivatives with respect to Re m
-    /// and Im m into `d`.
+    /// each n_q values, into `out`), and its derivatives with respect to Re m and Im m of each
+    /// group into `d` (Re m, Im m of the first group, then of the next; empty for none).
     pub(crate) fn phase(
         &self,
         tables: &BandTables,
         k: usize,
         scale: f64,
         out: &mut [f64],
-        mut d: Option<[&mut [f64]; 2]>,
+        d: &mut [&mut [f64]],
     ) {
         let q = tables.q;
         let lo = q / 2;
@@ -139,24 +150,20 @@ impl PanelProducts {
             let j = lo + i;
             let (eu, ev) = (c(&self.ce, i, base), c(&self.ce, i, base + 2));
             let (ou, ov) = (c(&self.co, i, base), c(&self.co, i, base + 2));
-            let derivs = (self.per == 8).then(|| {
-                (
-                    c(&self.ce, i, base + 4),
-                    c(&self.ce, i, base + 6),
-                    c(&self.co, i, base + 4),
-                    c(&self.co, i, base + 6),
-                )
-            });
             let mut write = |node: usize, sign: f64| {
                 let (s1, s2) = (eu + ov * sign, ev + ou * sign);
                 let f = elements(s1, s2);
                 for (e, v) in f.into_iter().enumerate() {
                     out[e * q + node] = scale * v;
                 }
-                if let (Some([dre, dim]), Some((deu, dev, dou, dov))) = (d.as_mut(), derivs) {
+                for (g, pair) in d.chunks_exact_mut(2).enumerate() {
+                    let col = base + 4 + 4 * g;
+                    let (deu, dev) = (c(&self.ce, i, col), c(&self.ce, i, col + 2));
+                    let (dou, dov) = (c(&self.co, i, col), c(&self.co, i, col + 2));
                     let (d1, d2) = (deu + dov * sign, dev + dou * sign);
                     let fr = element_derivatives(s1, s2, d1, d2);
                     let fi = element_derivatives(s1, s2, i1 * d1, i1 * d2);
+                    let [dre, dim] = pair else { unreachable!() };
                     for e in 0..4 {
                         dre[e * q + node] = scale * fr[e];
                         dim[e * q + node] = scale * fi[e];
@@ -272,14 +279,15 @@ mod tests {
             let nodes = gauss_legendre::<f64>(q).unwrap().nodes;
             let tables = BandTables::new(&nodes, n_max);
             for derivatives in [false, true] {
-                let mut p = PanelProducts::new(&tables, cases.len(), derivatives);
+                let mut p = PanelProducts::new(&tables, cases.len(), usize::from(derivatives));
                 for (k, &(x, m)) in cases.iter().enumerate() {
                     let s = if derivatives {
                         mie.compute_with_derivatives(x, m, &opts).unwrap()
                     } else {
                         mie.compute(x, m, &opts).unwrap()
                     };
-                    p.set(k, &s);
+                    let groups: Vec<_> = s.group().into_iter().collect();
+                    p.set(k, &s.coefficients(), &groups);
                 }
                 p.multiply(&tables, n_max);
                 for (k, &(x, m)) in cases.iter().enumerate() {
@@ -300,8 +308,12 @@ mod tests {
                     let scale = 4.0 / (x * x);
                     let mut got = vec![0.0; 4 * q];
                     let (mut gre, mut gim) = (vec![0.0; 4 * q], vec![0.0; 4 * q]);
-                    let d = derivatives.then_some([&mut gre[..], &mut gim[..]]);
-                    p.phase(&tables, k, scale, &mut got, d);
+                    let mut d: Vec<&mut [f64]> = if derivatives {
+                        vec![&mut gre[..], &mut gim[..]]
+                    } else {
+                        Vec::new()
+                    };
+                    p.phase(&tables, k, scale, &mut got, &mut d);
                     let check = |got: &[f64], want: &[f64], what: &str| {
                         let size = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                         for (j, (g, w)) in got.iter().zip(want).enumerate() {
diff --git a/scatrs/src/ensemble/mod.rs b/scatrs/src/ensemble/mod.rs
index d94a965..2ac2a79 100644
--- a/scatrs/src/ensemble/mod.rs
+++ b/scatrs/src/ensemble/mod.rs
@@ -30,6 +30,7 @@ mod kronrod;
 mod lin;
 mod matrix;
 mod moments;
+mod particle;
 mod range;
 
 use num_complex::Complex;
@@ -39,29 +40,63 @@ pub use distribution::SizeDistribution;
 use crate::mie::{GreekCoefficients, MAX_GREEK_ORDER, Normalization};
 use crate::{Error, Result};
 
-/// One component of an external mixture: spheres of one size distribution and refractive
-/// index (one per wavelength), in relative number concentration `weight`.
+/// The particles of a component, given at each wavelength.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub enum Particle {
+    /// Homogeneous spheres.
+    Sphere {
+        /// The refractive index at each wavelength.
+        refractive_index: Vec<Complex<f64>>,
+    },
+}
+
+impl Particle {
+    /// Homogeneous spheres of refractive index `refractive_index[i]` at wavelength i.
+    pub fn sphere(refractive_index: Vec<Complex<f64>>) -> Self {
+        Particle::Sphere { refractive_index }
+    }
+}
+
+/// One component of an external mixture: particles of one size distribution, in relative
+/// number concentration `weight`.
 #[derive(Clone, Debug, PartialEq)]
 #[non_exhaustive]
 pub struct Component {
     /// The size distribution.
     pub distribution: SizeDistribution,
-    /// The refractive index at each wavelength.
-    pub refractive_index: Vec<Complex<f64>>,
+    /// The particles.
+    pub particle: Particle,
     /// The relative number concentration, ≥ 0.
     pub weight: f64,
 }
 
 impl Component {
-    /// A component of weight `weight`.
+    /// A component of spheres of refractive index `refractive_index[i]` at wavelength i, of
+    /// weight `weight`.
     pub fn new(
         distribution: SizeDistribution,
         refractive_index: Vec<Complex<f64>>,
         weight: f64,
     ) -> Self {
+        Self::with_particle(distribution, Particle::sphere(refractive_index), weight)
+    }
+
+    /// A component of `particle`s, of weight `weight`.
+    ///
+    /// ```
+    /// use num_complex::Complex;
+    /// use scatrs::ensemble::{Component, Particle, SizeDistribution};
+    ///
+    /// let size = SizeDistribution::LogNormal { median_radius: 0.1, width: 1.5 };
+    /// let m = vec![Complex::new(1.45, 1e-3)];
+    /// let c = Component::with_particle(size.clone(), Particle::sphere(m.clone()), 2.0);
+    /// assert_eq!(c, Component::new(size, m, 2.0));
+    /// ```
+    pub fn with_particle(distribution: SizeDistribution, particle: Particle, weight: f64) -> Self {
         Self {
             distribution,
-            refractive_index,
+            particle,
             weight,
         }
     }
@@ -140,6 +175,7 @@ impl EnsembleParam {
 fn component_parameters(
     index: usize,
     distribution: &SizeDistribution,
+    particle: &Particle,
     weight: bool,
 ) -> Vec<EnsembleParam> {
     let mut out = Vec::new();
@@ -163,8 +199,12 @@ fn component_parameters(
             ));
         }
     }
-    out.push(EnsembleParam::new(index, None, ParamKind::RefrReal));
-    out.push(EnsembleParam::new(index, None, ParamKind::RefrImag));
+    match particle {
+        Particle::Sphere { .. } => {
+            out.push(EnsembleParam::new(index, None, ParamKind::RefrReal));
+            out.push(EnsembleParam::new(index, None, ParamKind::RefrImag));
+        }
+    }
     if weight {
         out.push(EnsembleParam::new(index, None, ParamKind::Weight));
     }
@@ -173,7 +213,7 @@ fn component_parameters(
 
 /// Every parameter of a single distribution, for [`compute`].
 pub fn parameters(distribution: &SizeDistribution) -> Vec<EnsembleParam> {
-    component_parameters(0, distribution, false)
+    component_parameters(0, distribution, &Particle::sphere(Vec::new()), false)
 }
 
 /// Every parameter of a mixture, for [`compute_mixture`].
@@ -181,7 +221,7 @@ pub fn mixture_parameters(components: &[Component]) -> Vec<EnsembleParam> {
     components
         .iter()
         .enumerate()
-        .flat_map(|(i, c)| component_parameters(i, &c.distribution, true))
+        .flat_map(|(i, c)| component_parameters(i, &c.distribution, &c.particle, true))
         .collect()
 }
 
@@ -535,10 +575,10 @@ fn validate(
                 )));
             }
         }
-        if c.refractive_index.len() != wavelengths.len() {
+        if c.particle.wavelengths() != wavelengths.len() {
             return Err(invalid(format!(
                 "component {i} has {} refractive indices for {} wavelengths",
-                c.refractive_index.len(),
+                c.particle.wavelengths(),
                 wavelengths.len()
             )));
         }
diff --git a/scatrs/src/ensemble/particle.rs b/scatrs/src/ensemble/particle.rs
new file mode 100644
index 0000000..61d3dfd
--- /dev/null
+++ b/scatrs/src/ensemble/particle.rs
@@ -0,0 +1,93 @@
+//! The particles of a component as the engine solves them: at each wavelength, a particle of
+//! size parameter x gives its coefficients and, on request, their derivative groups.
+//!
+//! The per-sphere quantities are differentiated along the particle's directions: the real and
+//! imaginary parts of the refractive index of each group, in order (Re m and Im m for a
+//! sphere).
+
+use num_complex::Complex;
+
+use super::Particle;
+use crate::Result;
+use crate::mie::{Coefficients, Group, Mie, MieOptions, n_terms};
+
+impl Particle {
+    /// The particle at wavelength `i`.
+    pub(super) fn optics(&self, i: usize) -> Optics {
+        match self {
+            Particle::Sphere { refractive_index } => Optics::Sphere(refractive_index[i]),
+        }
+    }
+
+    /// The number of wavelengths it is given at.
+    pub(super) fn wavelengths(&self) -> usize {
+        match self {
+            Particle::Sphere { refractive_index } => refractive_index.len(),
+        }
+    }
+
+    /// The number of derivative directions: two per group.
+    pub(super) fn directions(&self) -> usize {
+        match self {
+            Particle::Sphere { .. } => 2,
+        }
+    }
+}
+
+/// A component's particle at one wavelength.
+#[derive(Clone, Copy, Debug)]
+pub(super) enum Optics {
+    /// A homogeneous sphere of refractive index m.
+    Sphere(Complex<f64>),
+}
+
+impl Optics {
+    /// The number of terms of the particle of size parameter `x`, after the input checks of
+    /// [`Solver::solve`].
+    pub(super) fn n_terms(self, x: f64, opts: &MieOptions) -> Result<usize> {
+        match self {
+            Optics::Sphere(m) => n_terms(x, m, opts),
+        }
+    }
+}
+
+/// One particle's solution: its coefficients and, if computed, their derivative groups.
+pub(super) struct Solved<'a> {
+    pub(super) c: Coefficients<'a, f64>,
+    pub(super) groups: Vec<Group<'a, f64>>,
+}
+
+/// A per-thread workspace for solving particles.
+pub(super) struct Solver {
+    mie: Mie<f64>,
+}
+
+impl Solver {
+    pub(super) fn new() -> Self {
+        Self { mie: Mie::new() }
+    }
+
+    /// The particle `optics` of size parameter `x`, with its derivative groups if
+    /// `derivatives`.
+    pub(super) fn solve(
+        &mut self,
+        optics: Optics,
+        x: f64,
+        opts: &MieOptions,
+        derivatives: bool,
+    ) -> Result<Solved<'_>> {
+        match optics {
+            Optics::Sphere(m) => {
+                let s = if derivatives {
+                    self.mie.compute_with_derivatives(x, m, opts)
+                } else {
+                    self.mie.compute(x, m, opts)
+                }?;
+                Ok(Solved {
+                    c: s.coefficients(),
+                    groups: s.group().into_iter().collect(),
+                })
+            }
+        }
+    }
+}
````

### Task 3: Particle specs in the bindings and the Python layer

**Files:**
- Modify: `scatrs-py/src/ensemble.rs`, `python/scatrs/_scatrs/__init__.pyi`, `python/scatrs/particles.py`, `python/scatrs/ensemble.py`, `tests/test_ensemble_engine_calls.py`, `tests/test_ensemble.py`

**Interfaces:**
- **Consumes:** task 2's `Particle`, `Particle::sphere` and `Component::with_particle`.
- **Produces:**
  - `_scatrs.ensemble_parameters(items, mixture)` and `_scatrs.ensemble_compute(items, …)` with items `(distribution spec, particle spec, weight)`;
  - `_scatrs.ensemble_compute_batch(specs, particle_spec, …)`;
  - the particle spec `("sphere", m)`, with m a complex array, one value per wavelength. A particle spec that is not a tuple, of unknown kind, or with a sphere's m not an array raises `TypeError`. The batch accepts spheres only until 3c.
  - `Sphere._engine(wavelength) -> (spec, m)`, where m has the wavelengths' shape, for `EnsembleResult.refractive_index`.

- [ ] **Step 1: Write the failing tests:** the parametrized `test_the_engine_rejects_unknown_particles` and the new item format in `tests/test_ensemble_engine_calls.py` (diff below).
- [ ] **Step 2: Run them to see them fail.**

Run: `uv run pytest -q tests/test_ensemble.py tests/test_ensemble_engine_calls.py`
Expected: FAIL. The old binding reads the particle spec as an array of refractive indices.

- [ ] **Step 3: Implement** the binding, the stub, `Sphere._engine`, and its use in `Ensemble.compute` and `Mixture.compute` (diff below).
- [ ] **Step 4: Check bitwise equality.**

Run: the Python snapshot to `py-after.npz`, then `uv run python pycmp.py py-before.npz py-after.npz`. Then `compare.py --child` for W1 and W2 in both builds (appendix C).
Expected: `bitwise equal` for each.

- [ ] **Step 5: Run the checks and the docs build** (`uv run --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html`, which executes the ensemble examples).

Expected: Python `384 passed`, 1 skipped; stubtest, ruff and the docs build clean.

- [ ] **Step 6: Commit:** `refactor(python): particle specs for the ensemble engine`.

The diff:

````diff
diff --git a/python/scatrs/_scatrs/__init__.pyi b/python/scatrs/_scatrs/__init__.pyi
index 4f1e882..63abb2e 100644
--- a/python/scatrs/_scatrs/__init__.pyi
+++ b/python/scatrs/_scatrs/__init__.pyi
@@ -81,11 +81,9 @@ def mie_greek(
     allow_unvalidated: bool,
     normalized: bool,
 ) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64] | None]: ...
-def ensemble_parameters(
-    items: list[tuple[Any, NDArray[np.complex128], float]], mixture: bool
-) -> list[str]: ...
+def ensemble_parameters(items: list[tuple[Any, Any, float]], mixture: bool) -> list[str]: ...
 def ensemble_compute(
-    items: list[tuple[Any, NDArray[np.complex128], float]],
+    items: list[tuple[Any, Any, float]],
     wavelengths: NDArray[np.float64],
     mixture: bool,
     greek: int | None,
@@ -100,7 +98,7 @@ def ensemble_compute(
 ) -> dict[str, Any]: ...
 def ensemble_compute_batch(
     specs: list[Any],
-    refractive_index: NDArray[np.complex128],
+    particle_spec: Any,
     wavelengths: NDArray[np.float64],
     greek: int | None,
     all_orders: bool,
diff --git a/python/scatrs/ensemble.py b/python/scatrs/ensemble.py
index f723972..432c091 100644
--- a/python/scatrs/ensemble.py
+++ b/python/scatrs/ensemble.py
@@ -542,14 +542,13 @@ class Ensemble:
         refine_backscatter: bool = False,
     ) -> EnsembleResult:
         wl = _args.real_array(wavelength, "wavelength")
-        m = self.particle.refractive_index.evaluate(wl)
+        particle, m = self.particle._engine(wl)
         o = _Options.of(
             greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
         )
         shape, grid, specs = self._layout()
-        flat_m = np.ascontiguousarray(m.reshape(-1))
         if shape == ():
-            items = [(specs[0], flat_m, 1.0)]
+            items = [(specs[0], particle, 1.0)]
             raw, params = _single(items, False, wl, derivatives, o)
             return _result(raw, wl.shape, o, wl, m, grid, params)
         if not (isinstance(derivatives, (bool, np.bool_)) and not derivatives):
@@ -559,7 +558,7 @@ class Ensemble:
             )
         raw = _scatrs.ensemble_compute_batch(
             specs,
-            flat_m,
+            particle,
             np.ascontiguousarray(wl.reshape(-1)),
             o.order,
             o.all_orders,
@@ -646,13 +645,14 @@ class Mixture:
         refine_backscatter: bool = False,
     ) -> EnsembleResult:
         wl = _args.real_array(wavelength, "wavelength")
-        ms = [e.particle.refractive_index.evaluate(wl) for _, e in self.components]
+        engine = [e.particle._engine(wl) for _, e in self.components]
+        ms = [m for _, m in engine]
         o = _Options.of(
             greek, mu, normalized, rtol, max_panels, allow_unvalidated, refine_backscatter
         )
         items = [
-            (_scalar_spec(e.size), np.ascontiguousarray(m.reshape(-1)), w)
-            for (w, e), m in zip(self.components, ms, strict=True)
+            (_scalar_spec(e.size), particle, w)
+            for (w, e), (particle, _) in zip(self.components, engine, strict=True)
         ]
         raw, params = _single(items, True, wl, derivatives, o)
         return _result(raw, wl.shape, o, wl, np.stack(ms), {}, params)
diff --git a/python/scatrs/particles.py b/python/scatrs/particles.py
index cf77058..66eba42 100644
--- a/python/scatrs/particles.py
+++ b/python/scatrs/particles.py
@@ -6,6 +6,7 @@ import numbers
 from dataclasses import dataclass
 
 import numpy as np
+from numpy.typing import NDArray
 
 from scatrs.refractive import Constant, Function, RefractiveIndex
 
@@ -61,3 +62,9 @@ class Sphere:
 
     def __post_init__(self) -> None:
         object.__setattr__(self, "refractive_index", _model(self.refractive_index))
+
+    def _engine(self, wavelength: NDArray[np.float64]) -> tuple[tuple, NDArray[np.complex128]]:
+        """The engine's tagged tuple for the particle at ``wavelength``, and its refractive
+        index there (the wavelengths' shape), which the result reports."""
+        m = self.refractive_index.evaluate(wavelength)
+        return ("sphere", np.ascontiguousarray(m.reshape(-1))), m
diff --git a/scatrs-py/src/ensemble.rs b/scatrs-py/src/ensemble.rs
index 62ce495..6aad463 100644
--- a/scatrs-py/src/ensemble.rs
+++ b/scatrs-py/src/ensemble.rs
@@ -1,8 +1,9 @@
 //! Bindings for `scatrs::ensemble`.
 //!
 //! Distributions arrive as tagged tuples from `scatrs.ensemble` (`("lognormal", r_g, σ_g)`,
-//! `("tabulated", radii, densities)`, `("multimodal", [(weight, spec), …])`, …), components as
-//! `(spec, refractive indices, weight)`, with one refractive index per wavelength. Results
+//! `("tabulated", radii, densities)`, `("multimodal", [(weight, spec), …])`, …), particles as
+//! tagged tuples too (`("sphere", refractive indices)`, one refractive index per wavelength),
+//! and components as `(spec, particle, weight)`. Results
 //! return as a dict of arrays, one row per wavelength, which `scatrs.ensemble` wraps;
 //! derivatives are selected by their index in `ensemble_parameters`. Arrays are borrowed with
 //! `try_readonly` (see `special.rs`), and the outputs are computed and laid out without the GIL.
@@ -13,7 +14,7 @@ use pyo3::exceptions::PyTypeError;
 use pyo3::prelude::*;
 use pyo3::types::{PyDict, PyTuple};
 use scatrs::ensemble::{
-    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs,
+    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, Particle,
     SizeDistribution, compute_batch, compute_mixture, mixture_parameters, parameters,
 };
 use scatrs::mie::{MAX_GREEK_ORDER, Normalization};
@@ -73,18 +74,38 @@ fn distribution(spec: &Bound<'_, PyAny>) -> PyResult<SizeDistribution> {
     })
 }
 
-/// A component as passed from Python: (spec, refractive indices, weight).
-type Item<'py> = (Bound<'py, PyAny>, Bound<'py, PyArrayDyn<Complex64>>, f64);
+/// A particle from its tagged tuple.
+fn particle(spec: &Bound<'_, PyAny>) -> PyResult<Particle> {
+    let t = spec
+        .cast::<PyTuple>()
+        .map_err(|_| PyTypeError::new_err("a particle spec must be a tuple"))?;
+    let kind: String = t.get_item(0)?.extract()?;
+    match kind.as_str() {
+        "sphere" => {
+            let m = t.get_item(1)?;
+            let m = m.cast::<PyArrayDyn<Complex64>>().map_err(|_| {
+                PyTypeError::new_err("a sphere's refractive indices must be an array")
+            })?;
+            let m = borrow(m, "refractive_index")?;
+            Ok(Particle::sphere(m.as_array().iter().copied().collect()))
+        }
+        other => Err(PyTypeError::new_err(format!(
+            "unknown particle kind {other:?}"
+        ))),
+    }
+}
+
+/// A component as passed from Python: (distribution spec, particle spec, weight).
+type Item<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, f64);
 
 /// Components from their items.
 fn components(items: &[Item<'_>]) -> PyResult<Vec<Component>> {
     items
         .iter()
-        .map(|(spec, m, w)| {
-            let m = borrow(m, "refractive_index")?;
-            Ok(Component::new(
+        .map(|(spec, p, w)| {
+            Ok(Component::with_particle(
                 distribution(spec)?,
-                m.as_array().iter().copied().collect(),
+                particle(p)?,
                 *w,
             ))
         })
@@ -388,15 +409,15 @@ fn into_dict(py: Python<'_>, arrays: Arrays) -> PyResult<Bound<'_, PyDict>> {
     Ok(out)
 }
 
-/// The ensemble outputs of each distribution (tagged tuples `specs`) at each wavelength, sharing
-/// the size nodes: as `ensemble_compute` without derivatives, one row per (distribution,
-/// wavelength), distribution-major.
+/// The ensemble outputs of each distribution (tagged tuples `specs`) of one particle (a tagged
+/// tuple) at each wavelength, sharing the size nodes: as `ensemble_compute` without derivatives,
+/// one row per (distribution, wavelength), distribution-major.
 #[pyfunction]
 #[allow(clippy::too_many_arguments)]
 pub fn ensemble_compute_batch<'py>(
     py: Python<'py>,
     specs: Vec<Bound<'py, PyAny>>,
-    refractive_index: Bound<'py, PyArrayDyn<Complex64>>,
+    particle_spec: Bound<'py, PyAny>,
     wavelengths: Bound<'py, PyArrayDyn<f64>>,
     greek: Option<u64>,
     all_orders: bool,
@@ -410,11 +431,10 @@ pub fn ensemble_compute_batch<'py>(
     let err = |e| to_pyerr(py, e);
     let distributions: Vec<SizeDistribution> =
         specs.iter().map(distribution).collect::<PyResult<_>>()?;
-    let m: Vec<Complex64> = borrow(&refractive_index, "refractive_index")?
-        .as_array()
-        .iter()
-        .copied()
-        .collect();
+    let m = match particle(&particle_spec)? {
+        Particle::Sphere { refractive_index } => refractive_index,
+        _ => return Err(PyTypeError::new_err("a batch needs spheres")),
+    };
     let wl: Vec<f64> = borrow(&wavelengths, "wavelength")?
         .as_array()
         .iter()
diff --git a/tests/test_ensemble.py b/tests/test_ensemble.py
index 4ae4c24..9af1605 100644
--- a/tests/test_ensemble.py
+++ b/tests/test_ensemble.py
@@ -275,11 +275,26 @@ def test_error_estimates_of_derivatives_and_convergence_flags():
 def test_a_single_distribution_has_one_component():
     from scatrs import _scatrs
 
-    spec = (("lognormal", 0.08, 1.6), np.array([1.5 + 0j]), 1.0)
+    spec = (("lognormal", 0.08, 1.6), ("sphere", np.array([1.5 + 0j])), 1.0)
     with pytest.raises(InvalidInputError, match="exactly one component"):
         _scatrs.ensemble_parameters([spec, spec], False)
 
 
+@pytest.mark.parametrize(
+    ("particle", "message"),
+    [
+        (("cube", np.array([1.5 + 0j])), "unknown particle kind"),
+        (("sphere", [1.5 + 0j]), "must be an array"),
+        ([1.5 + 0j], "must be a tuple"),
+    ],
+)
+def test_the_engine_rejects_unknown_particles(particle, message):
+    from scatrs import _scatrs
+
+    with pytest.raises(TypeError, match=message):
+        _scatrs.ensemble_parameters([(("lognormal", 0.08, 1.6), particle, 1.0)], False)
+
+
 def test_a_grid_shares_nodes_and_matches_each_distribution():
     size = LogNormal(np.array([0.1, 0.2, 0.3])[:, None], [1.5, 1.7])
     wl = np.array([[0.4, 0.6, 0.8]])
diff --git a/tests/test_ensemble_engine_calls.py b/tests/test_ensemble_engine_calls.py
index 79f6225..98074b8 100644
--- a/tests/test_ensemble_engine_calls.py
+++ b/tests/test_ensemble_engine_calls.py
@@ -1,7 +1,6 @@
-"""Ensemble and Mixture call the engine exactly as the functions they replaced did.
-
-The private bindings are unchanged, so these results are bitwise those of 0.1.0's
-``scatrs.ensemble.compute``, ``compute_batch`` and ``compute_mixture`` for the same inputs.
+"""Ensemble and Mixture call the engine exactly as the functions they replaced did, so
+these results are bitwise those of 0.1.0's ``scatrs.ensemble.compute``, ``compute_batch`` and
+``compute_mixture`` for the same inputs.
 """
 
 import numpy as np
@@ -63,7 +62,7 @@ def test_a_scalar_ensemble_is_the_single_distribution_call():
         WL, greek=8, mu=MU, derivatives=["mode0.width", "m_imag"], **OPTIONS
     )
     spec = ("multimodal", [(2.0, ("lognormal", 0.1, 1.5)), (1.0, ("gamma", 0.5, 0.1))])
-    items = [(spec, np.full(4, m), 1.0)]
+    items = [(spec, ("sphere", np.full(4, m)), 1.0)]
     names = _scatrs.ensemble_parameters(items, False)
     raw = _scatrs.ensemble_compute(
         items,
@@ -84,7 +83,7 @@ def test_a_grid_is_the_batch_call():
     )
     specs = [("lognormal", float(a), float(b)) for a in radii for b in widths]
     raw = _scatrs.ensemble_compute_batch(
-        specs, np.full(4, m), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
+        specs, ("sphere", np.full(4, m)), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
     )
     _equal(raw, r, (3, 2, *WL.shape))
 
@@ -96,8 +95,8 @@ def test_a_mixture_is_the_mixture_call():
         WL, greek=8, mu=MU, derivatives=["c1.weight"], **OPTIONS
     )
     items = [
-        (("lognormal", 0.1, 1.5), np.full(4, 1.45 + 0j), 2.0),
-        (("gamma", 1.0, 0.1), np.full(4, 1.53 + 0.01j), 0.5),
+        (("lognormal", 0.1, 1.5), ("sphere", np.full(4, 1.45 + 0j)), 2.0),
+        (("gamma", 1.0, 0.1), ("sphere", np.full(4, 1.53 + 0.01j)), 0.5),
     ]
     names = _scatrs.ensemble_parameters(items, True)
     raw = _scatrs.ensemble_compute(
@@ -117,6 +116,6 @@ def test_a_list_is_the_batch_call():
     r = Ensemble(Sphere(m), sizes).compute(WL, greek=8, mu=MU, **OPTIONS)
     specs = [("lognormal", 0.1, 1.5), ("gamma", 0.5, 0.1), ("tabulated", [0.1, 0.3], [1.0, 1.0])]
     raw = _scatrs.ensemble_compute_batch(
-        specs, np.full(4, m), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
+        specs, ("sphere", np.full(4, m)), WL.reshape(-1), *_engine_options(), *OPTIONS.values()
     )
     _equal(raw, r, (3, *WL.shape))
````

## Hand-off

- **For 3b.**
  - `Coefficients` and `Group` take the layered kernel's coefficients as they are.
  - `Group` will need to know whether its size parameter is the outer one (x_L), which scales the outputs. Only that group's ∂/∂x includes the −(2/x)·Q term of the efficiencies and the phase-matrix scale. The flag is added with its first use (spec section 4).
- **For 3c.**
  - Add `Particle::LayeredSphere` (spec section 15), its `Optics` variant and its `Solver` arm.
  - Add its directions: Re m and Im m per layer, then one *real* direction per radius fraction. This needs a real-column kind in `PanelProducts`, and in `quantities1`/`quantities2` the fraction derivative x_L·∂/∂x_l.
  - Make the r-derivative Σ_l f_l·∂/∂x_l.
  - Add the `ParamKind`s, a particle-aware `parameters` for single components, a particle-aware `compute_batch`, the `("layered", fractions, m)` spec, and `LayeredSphere._engine`.
  - Draft the release notes, including this plan's `Component` field change.
- **For phase 4** (spheroids, T-matrix). The `Solver` gains a direct path for particles that give their outputs and Greek coefficients without aₙ and bₙ (spec section 15).
- **Branch.** Task 1 and the spec commit were first made on local `main`, after an outside checkout. They were moved to `phase-3`, and tasks 2 and 3 were committed there as `6760658` and `e5a5d27`.

## Appendix A: the Rust snapshot

A throwaway crate outside the repository (`snap/`), depending on `scatrs = { path = "<repo>/scatrs" }` and `num-complex = "0.4.6"`. Run with `cargo run -q --release -- <out.txt>`. It writes one line per value, as the value's bits in hex, so `cmp` or `diff` locates any change.

```rust
//! Writes the bits of every Mie and ensemble output on a fixed grid, one line per value, to
//! compare a refactor bitwise.
use std::fmt::Write as _;

use num_complex::Complex;
use scatrs::ensemble::{
    Component, EnsembleOptions, EnsembleResult, GreekOrder, Outputs, SizeDistribution, compute,
    compute_batch, compute_mixture, mixture_parameters, parameters,
};
use scatrs::mie::{
    AmplitudeBuffers, AngularTable, Mie, MieOptions, Normalization, PhaseBuffers,
};

struct Out(String);
impl Out {
    fn f(&mut self, tag: &str, v: f64) {
        writeln!(self.0, "{tag} {:016x}", v.to_bits()).unwrap();
    }
    fn c(&mut self, tag: &str, v: Complex<f64>) {
        self.f(tag, v.re);
        self.f(tag, v.im);
    }
    fn cs(&mut self, tag: &str, v: &[Complex<f64>]) {
        for z in v {
            self.c(tag, *z);
        }
    }
    fn fs(&mut self, tag: &str, v: &[f64]) {
        for z in v {
            self.f(tag, *z);
        }
    }
    fn e(&mut self, tag: &str, q: &scatrs::mie::Efficiencies<f64>) {
        self.fs(tag, &[q.q_ext, q.q_sca, q.q_abs, q.q_back, q.g]);
    }
    fn g(&mut self, tag: &str, g: &scatrs::mie::GreekCoefficients<f64>) {
        for a in [&g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2] {
            self.fs(tag, a);
        }
    }
    fn outputs(&mut self, tag: &str, o: &Outputs) {
        self.fs(tag, &[o.c_ext, o.c_sca, o.c_abs, o.c_back, o.ssa, o.g, o.lidar_ratio,
            o.depolarization, o.effective_radius, o.mean_area, o.mean_volume]);
        if let Some(g) = &o.greek { self.g(tag, g); }
        if let Some(p) = &o.phase { for a in [&p.f11, &p.f12, &p.f33, &p.f34] { self.fs(tag, a); } }
    }
    fn results(&mut self, tag: &str, rs: &[EnsembleResult]) {
        for r in rs {
            self.outputs(tag, &r.values);
            self.outputs(tag, &r.errors);
            for d in &r.derivatives { self.outputs(tag, d); }
            for d in &r.derivative_errors { self.outputs(tag, d); }
            writeln!(self.0, "{tag} conv {} panels {} ranges {:?}", r.converged, r.panels, r.ranges).unwrap();
        }
    }
}

fn phase_bufs(v: &mut [Vec<f64>; 4]) -> PhaseBuffers<'_, f64> {
    let [f11, f12, f33, f34] = v;
    PhaseBuffers { f11, f12, f33, f34 }
}

fn mie(out: &mut Out) {
    let mu: Vec<f64> = [-1.0, -0.7, -0.2, 0.0, 0.3, 0.8, 0.999, 1.0].to_vec();
    let n = mu.len();
    let ms = [Complex::new(1.33, 0.0), Complex::new(1.5, 1e-3), Complex::new(1.0001, 0.0),
              Complex::new(2.5, 1.0), Complex::new(0.2, 3.0)];
    let xs = [1e-6, 0.01, 0.5, 3.0, 20.0, 150.0, 1200.0];
    let opts = MieOptions::new();
    let table = AngularTable::new(&mu, 1400).unwrap();
    let mut w = Mie::<f64>::new();
    for &x in &xs {
        for &m in &ms {
            let tag = format!("x={x} m={m}");
            let s = w.compute_with_derivatives(x, m, &opts).unwrap();
            out.cs(&tag, s.a());
            out.cs(&tag, s.b());
            let d = s.derivatives().unwrap();
            for v in [d.da_dx, d.da_dm, d.db_dx, d.db_dm] { out.cs(&tag, v); }
            out.e(&tag, &s.efficiencies());
            let j = s.efficiencies_jacobian().unwrap();
            for q in [&j.x, &j.m_re, &j.m_im] { out.e(&tag, q); }
            let z = Complex::default();
            let mut b: [Vec<Complex<f64>>; 6] = core::array::from_fn(|_| vec![z; n]);
            {
                let [s1, s2, a1, a2, c1, c2] = &mut b;
                s.amplitudes(&mu, AmplitudeBuffers { s1, s2 }).unwrap();
                out.cs(&tag, s1); out.cs(&tag, s2);
                s.amplitudes_from_table(&table, AmplitudeBuffers { s1, s2 }).unwrap();
                out.cs(&tag, s1); out.cs(&tag, s2);
                s.amplitudes_with_derivatives(&mu, AmplitudeBuffers { s1, s2 },
                    AmplitudeBuffers { s1: a1, s2: a2 }, AmplitudeBuffers { s1: c1, s2: c2 }).unwrap();
            }
            for v in &b { out.cs(&tag, v); }
            {
                let [s1, s2, a1, a2, c1, c2] = &mut b;
                s.amplitudes_with_derivatives_from_table(&table, AmplitudeBuffers { s1, s2 },
                    AmplitudeBuffers { s1: a1, s2: a2 }, AmplitudeBuffers { s1: c1, s2: c2 }).unwrap();
            }
            for v in &b { out.cs(&tag, v); }
            for norm in [Normalization::Normalized, Normalization::Unnormalized] {
                let mut p: [[Vec<f64>; 4]; 4] = core::array::from_fn(|_| core::array::from_fn(|_| vec![0.0; n]));
                s.phase_matrix(&mu, norm, phase_bufs(&mut p[0])).unwrap();
                out.fs(&tag, &p[0].concat());
                s.phase_matrix_from_table(&table, norm, phase_bufs(&mut p[0])).unwrap();
                out.fs(&tag, &p[0].concat());
                {
                    let [v, a, b2, c] = &mut p;
                    s.phase_matrix_with_derivatives(&mu, norm, phase_bufs(v), [phase_bufs(a), phase_bufs(b2), phase_bufs(c)]).unwrap();
                }
                for q in &p { out.fs(&tag, &q.concat()); }
                {
                    let [v, a, b2, c] = &mut p;
                    s.phase_matrix_with_derivatives_from_table(&table, norm, phase_bufs(v), [phase_bufs(a), phase_bufs(b2), phase_bufs(c)]).unwrap();
                }
                for q in &p { out.fs(&tag, &q.concat()); }
                if x <= 150.0 {
                    for order in [Some(6), None] {
                        out.g(&tag, &s.greek(order, norm).unwrap());
                        let (g, d) = s.greek_with_derivatives(order, norm).unwrap();
                        out.g(&tag, &g);
                        for q in &d { out.g(&tag, q); }
                    }
                }
            }
            let s = w.compute(x, m, &opts).unwrap();
            out.e(&tag, &s.efficiencies());
            out.cs(&tag, s.a());
        }
    }
}

fn ensembles(out: &mut Out) {
    let wl = [0.35, 0.55, 1.02];
    let m = vec![Complex::new(1.43, 1e-8), Complex::new(1.5, 0.01), Complex::new(1.33, 0.002)];
    let dists = [
        SizeDistribution::LogNormal { median_radius: 0.12, width: 1.6 },
        SizeDistribution::Gamma { effective_radius: 0.3, effective_variance: 0.1 },
        SizeDistribution::PowerLaw { exponent: 3.0, min_radius: 0.05, max_radius: 0.6 },
        SizeDistribution::Monodisperse { radius: 0.4 },
        SizeDistribution::Tabulated { radius: vec![0.05, 0.1, 0.3, 0.6], density: vec![1.0, 3.0, 2.0, 0.5] },
        SizeDistribution::MultiModal { modes: vec![
            (0.7, SizeDistribution::LogNormal { median_radius: 0.05, width: 1.5 }),
            (0.3, SizeDistribution::LogNormal { median_radius: 0.5, width: 1.8 })] },
    ];
    let mu = vec![-1.0, -0.3, 0.2, 0.9, 1.0];
    for (i, d) in dists.iter().enumerate() {
        for (k, opts) in [
            EnsembleOptions::new(),
            EnsembleOptions::new().with_greek(GreekOrder::Order(12)).with_mu(mu.clone())
                .with_derivatives(parameters(d)),
            EnsembleOptions::new().with_greek(GreekOrder::All).with_rtol(1e-4)
                .with_normalization(Normalization::Unnormalized).with_refine_order(4)
                .with_derivatives(parameters(d).into_iter().rev().take(2).collect()),
            EnsembleOptions::new().with_refine_backscatter(true).with_mu(mu.clone())
                .with_derivatives(parameters(d).into_iter().take(1).collect()),
        ].iter().enumerate() {
            let r = compute(d, &wl, &m, opts).unwrap();
            out.results(&format!("ens {i} opts {k}"), &r);
        }
    }
    let comps = vec![
        Component::new(dists[0].clone(), m.clone(), 2.0),
        Component::new(dists[5].clone(), vec![Complex::new(1.95, 0.79); 3], 0.5),
        Component::new(dists[3].clone(), vec![Complex::new(1.33, 0.0); 3], 1.0),
    ];
    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(16)).with_mu(mu.clone())
        .with_derivatives(mixture_parameters(&comps));
    out.results("mixture", &compute_mixture(&comps, &wl, &opts).unwrap());
    let batch: Vec<SizeDistribution> = [0.05, 0.1, 0.2, 0.4].iter().flat_map(|&r| [1.4, 1.8].map(|w|
        SizeDistribution::LogNormal { median_radius: r, width: w })).chain([dists[3].clone()]).collect();
    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(24)).with_mu(mu).with_rtol(1e-5);
    for rs in compute_batch(&batch, &wl, &m, &opts).unwrap() {
        out.results("batch", &rs);
    }
}

fn main() {
    let mut out = Out(String::new());
    mie(&mut out);
    ensembles(&mut out);
    let path = std::env::args().nth(1).expect("output path");
    std::fs::write(&path, out.0).unwrap();
}
```

## Appendix B: the Python snapshot

`pysnap.py <out.npz>` saves every output of the Python ensemble API on a fixed set of calls. `pycmp.py <a.npz> <b.npz>` prints `bitwise equal` or the arrays that differ.

```python
"""Every output of the Python ensemble API on a fixed set of calls, to compare a refactor bitwise."""
import sys
import numpy as np
import scatrs
from scatrs import refractive

out = {}
def put(tag, res):
    for f in scatrs.SCALARS if hasattr(scatrs, "SCALARS") else scatrs.ensemble.SCALARS:
        out[f"{tag}.{f}"] = getattr(res, f)
        out[f"{tag}.err.{f}"] = getattr(res.error, f)
        for p in res.params:
            out[f"{tag}.d.{p}.{f}"] = getattr(res.d(p), f)
            out[f"{tag}.de.{p}.{f}"] = getattr(res.d_error(p), f)
    for part in ("greek", "phase"):
        v = getattr(res, part)
        if v is not None:
            for k, a in vars(v).items():
                out[f"{tag}.{part}.{k}"] = a
            for p in res.params:
                for k, a in vars(getattr(res.d(p), part)).items():
                    out[f"{tag}.d.{p}.{part}.{k}"] = a
    out[f"{tag}.converged"] = res.converged
    out[f"{tag}.panels"] = res.panels
    out[f"{tag}.m"] = res.refractive_index

wl = np.array([0.35, 0.55, 1.02])
mu = np.array([-1.0, -0.3, 0.2, 0.9, 1.0])
sph = scatrs.Sphere(refractive.Cauchy(1.42, 0.003, k=1e-4))
for i, size in enumerate([
    scatrs.LogNormal(0.12, 1.6),
    scatrs.Gamma(0.3, 0.1),
    scatrs.Monodisperse(0.4),
    scatrs.MultiModal([(0.7, scatrs.LogNormal(0.05, 1.5)), (0.3, scatrs.LogNormal(0.5, 1.8))]),
]):
    e = scatrs.Ensemble(sph, size)
    put(f"e{i}", e.compute(wl, greek=12, mu=mu, derivatives=True))
    put(f"e{i}u", e.compute(wl, greek="all", normalized=False, rtol=1e-4))
grid = scatrs.Ensemble(sph, scatrs.LogNormal(np.array([[0.05], [0.1], [0.2]]), np.array([1.4, 1.8])))
put("grid", grid.compute(wl, greek=24, mu=mu, rtol=1e-5))
lst = scatrs.Ensemble(sph, [scatrs.LogNormal(0.1, 1.5), scatrs.Monodisperse(0.3)])
put("list", lst.compute(wl, greek=8))
mix = scatrs.Mixture([(2.0, scatrs.Ensemble(sph, scatrs.LogNormal(0.12, 1.6))),
                      (0.5, scatrs.Ensemble(scatrs.Sphere(1.95 + 0.79j), scatrs.Gamma(0.2, 0.2)))])
put("mix", mix.compute(wl, greek=16, mu=mu, derivatives=True))
np.savez(sys.argv[1], **out)
print(len(out), "arrays")
```

```python
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert set(a.files) == set(b.files), set(a.files) ^ set(b.files)
bad = [k for k in a.files if a[k].tobytes() != b[k].tobytes() or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape]
print("bitwise equal" if not bad else f"DIFFER: {bad[:20]} ({len(bad)})")
```

## Appendix C: `benchmarks/compare.py`'s scatrs outputs

In each build (the old one from `git archive 5a785c3`, extracted elsewhere, and the new one), for W1 and W2:

```bash
uv run --frozen --directory benchmarks python compare.py --child out-W1.npy --workload W1
```

`--frozen`, because `benchmarks/uv.lock` is out of date with the root `pyproject.toml` on `main` (`uv lock --check` fails there) and `--locked` refuses it. Then compare the `out` arrays of every rtol with `tobytes()` (the `time` entries differ).
