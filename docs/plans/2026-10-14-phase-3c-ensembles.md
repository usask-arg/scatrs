# Phase 3c: Layered Spheres in Ensembles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Closed-form derivatives for layered spheres, replacing the dual numbers, so that their derivatives are validated beyond x_L = 100 (asked for before 3c).
- Layered spheres in `Ensemble`, `Mixture` and grids, with Jacobians with respect to each layer's refractive index and each radius fraction.
- The coated black-carbon example.

**Architecture:**
- **Derivatives.** Within each layer, H solves the Riccati equation y′ = V − y². Its sensitivities to the two surfaces and to its starting value are closed forms. An adjoint sweep per order chains them from the outer layer inward, and the outer step uses the sphere's closed forms with H in place of D.
- **Ensembles.**
  - `Particle::LayeredSphere` gets its own `Optics` and `Solver` arms.
  - The engine's directions are Re m and Im m per layer, then one real direction per radius fraction (x_L·∂/∂x_l), with real columns in `PanelProducts`.
  - The radius derivative at fixed fractions is Σ f_l·∂/∂x_l.
- **Python.** `LayeredSphere` has an `_engine` hook giving the spec `("layered", fractions, m[L, n])`.

**Tech Stack:**
- Rust 2024 (MSRV 1.85; CI lints with 1.99), pyo3/numpy 0.29.
- Python ≥ 3.11.
- criterion, Sphinx 9 + myst-nb.

**Spec:** `docs/specs/2026-10-14-phase-3-layered-spheres-design.md`.
- Sections 1.3, 3.2, 5.2, 10 and 11 describe the design.
- Section 17 covers the closed-form derivatives.
- Section 18 records what 3c changed, and why.

## Global Constraints

- **Spheres stay bitwise.** Every output of `scatrs::mie` and of sphere ensembles is bitwise unchanged. Check with the 3a Rust snapshot (`docs/plans/2026-10-14-phase-3a-refactor.md`, appendix A) after every task.
- **One layer is the sphere.** A one-layer layered sphere gives the sphere's results bitwise, alone and in ensembles.
- **Declared accuracy.** Both layered accuracy maps and the Arb references meet their tolerances. Each declared constant is at least 4 times the worst case found.
- **Checks at every commit:**
  - `cargo fmt --all`;
  - `cargo +1.99.0 clippy --workspace --all-targets -- -D warnings` (stop on any output, not just on the exit status);
  - `cargo test --workspace --release`;
  - `RUSTDOCFLAGS="-D warnings" cargo doc -p scatrs --no-deps`;
  - `uv run pytest -q`, `ruff check` and `ruff format --check`;
  - `uv run python -m mypy.stubtest scatrs._scatrs`;
  - the docs build, `sphinx-build -E -W --keep-going`.
- **Docs are user-facing:** no phases or development history.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never send personal data in network requests.

## How this plan was made

Every task was prototyped and verified before this plan was written. The diffs below are the commits on `phase-3`, without the generated data: `docs/validation/data/layered-*.json`, written by the maps with `SCATRS_ACCURACY_MAP`.

## Measured results the plan relies on

| What | Measured |
|---|---|
| Closed-form derivatives against `Layered<Dual<DoubleDouble, 4>>` | per order to 10⁻¹⁵ after the interface closed forms. A metallic core in a thin shell (x_L = 640) lost 4.6·10⁻⁴ in the D form of the index term near poles of D_n(z₁). The per-order choice fixes it (10⁻¹⁰ of the largest). |
| Double-double closed forms against the dual oracle | equal to f64 rounding everywhere tried |
| Layered map with derivatives to x_L = 10⁴ | coefficient derivatives C ≤ 22, Jacobian ≤ 37 for every x_L. Amplitude derivatives: 163 at 10³; 609, 666 and 1722 at 2.5·10³, 5·10³ and 10⁴ (forward amplitude, air shell of 10⁻³ of the radius around 1.5 + 0.01i). So the derivative domain is x_L ≤ 1000. |
| Greek map, derivatives to 10³ | phase-matrix derivatives C 53 / 256, Greek derivatives 48 / 256 |
| Derivative cost (`cargo bench --bench layered`) | every derivative costs 1.6–2.8× the values for 2, 3 and 10 layers, x_L 0.1–1000 (duals: 4.6–7.4× and 25–32×) |
| Ensembles: one layer against spheres | bitwise: values, estimates, derivatives, panels |
| Ensembles: Jacobians against central differences (rtol 10⁻¹¹) | within 10⁻⁵ of scale for every scalar, β₁ and F₁₁, F₁₂ |
| Error-estimate calibration with two layered distributions | worst ratios unchanged: 0.003 refined, 1.210 others |
| w1 of coated spheres (`cargo bench --bench ensemble`) | 1.8× the spheres' time; with every derivative 3.5× (the shell's Im m refines to `max_panels`) |

## File Structure

| File | Task | Role |
|---|---|---|
| `scatrs/src/layered/kernel.rs` | 1 | sensitivities per layer and order (`Sens`), S recurrence, adjoint sweep, inner layers' gradients |
| `scatrs/src/mie/kernel.rs` | 1 | the outer step's derivatives with H (`OuterDerivs`), stored per-order quantities |
| `scatrs/src/layered/{mod,domain}.rs` | 1 | duals removed; `derivative_x_max` = 1000 |
| `scatrs/src/ensemble/{mod,particle,engine,fast}.rs` | 2 | `Particle::LayeredSphere`, directions, real columns, `*_with_particle` |
| `scatrs/tests/ensemble_layered.rs` | 2 | the ensemble tests of layered spheres |
| `scatrs-py/src/ensemble.rs`, `python/scatrs/{particles,ensemble,__init__}.py` | 3 | the `("layered", …)` spec, `LayeredSphere`, mixtures' `refractive_index` tuple |
| `tests/test_ensemble_layered.py` | 3 | Python tests |
| `docs/examples/coated.md` and pages | 4 | the example and the docs updates |
| `scatrs/benches/ensemble.rs` | 5 | coated-sphere benchmarks |


### Task 1: Closed-form derivatives of layered spheres

Replaces the dual numbers for L ≥ 2. Spec section 17 has the formulas. The diagnosis tools were a debug crate comparing f64 and `DoubleDouble` closed forms with a `Layered<Dual<DoubleDouble, 4>>` oracle, per order and component. The oracle is unusable for an outer index of exactly 1, because the code branches on m = 1; use the double-double closed form there.

- [ ] Add `Sens` (H, R², the index term, H′, H + zH′) per layer and order to `side`. Add the S recurrence and the adjoint sweep, with the interface closed forms J and K.
- [ ] Choose the index term's form per order: the D form unless |t|·4 < max(|D|, |e|), and the t form then.
- [ ] Extend `State::finish` with `OuterDerivs`, and store per-order quantities for the inner layers' gradients.
- [ ] Remove the dual machinery from `layered/mod.rs`.
- [ ] Run the map with `derivative_x_max` = 10⁴ and `SCATRS_MAP_WORST=1`, and set the domain to 1000.
- [ ] Regenerate both map JSONs. Re-run the Arb references, the snapshot (bitwise) and the benchmark.
- [ ] Update the theory chapter (Derivatives, Cost), validation, limitations, index, Python docstrings, spec §17 and follow-ups.

The commit (`1991f84`):

````diff
diff --git a/docs/index.md b/docs/index.md
index 7874632..2b594a9 100644
--- a/docs/index.md
+++ b/docs/index.md
@@ -19,9 +19,9 @@ library for the optical properties of particles. Install it with `pip install sc
   parallel, with an error estimate for every output, and refractive indices that vary with
   wavelength ({mod}`scatrs.refractive`). Grids of distributions share their Mie solutions, for
   building optics tables.
-- **Derivatives** of every output: in closed form with respect to a sphere's size parameter and
-  refractive index, by dual numbers with respect to every layer of a layered sphere, and for
-  ensembles, the distribution parameters and the mixture weights.
+- **Derivatives** of every output, in closed form: with respect to a sphere's size parameter
+  and refractive index, to every layer's of a layered sphere, and for ensembles, the
+  distribution parameters and the mixture weights.
 - **Validated accuracy.** Every output has a declared accuracy, checked against independent
   reference values with rigorous error bounds across the whole validated domain, and against
   published results. See {doc}`validation/overview`.
diff --git a/docs/limitations.md b/docs/limitations.md
index 00df026..185e2bd 100644
--- a/docs/limitations.md
+++ b/docs/limitations.md
@@ -30,13 +30,11 @@ describes what the accuracy figures rest on.
   layer, every index in the sphere's domain, and at most 16 layers. Beyond
   $|m|\,x_L = 3\cdot10^4$ the narrow resonances of thin lossless shells of high index cost the
   coefficients up to $10^9$ ulps. A single layer is a sphere, with the sphere's domain.
-- **Derivatives up to $x_L = 100$.** The derivatives come from dual numbers through the
-  recursion, and lose accuracy faster than the values for larger particles, worst for thin
-  lossless shells (the amplitude derivatives of a thin air shell around a large core reached
-  errors of $10^{-8}$ relative at $x_L = 5\cdot10^3$). Above $x_L = 100$,
-  `compute_with_derivatives` and `derivatives=` raise `OutsideValidatedDomain` unless
-  `allow_unvalidated` is set. They also cost more than a sphere's closed forms: 4.6 to 7.4
-  times the values for two layers, up to 32 times for ten.
+- **Derivatives up to $x_L = 1000$.** Above it the derivatives lose accuracy faster than the
+  values for thin lossless shells (the forward amplitude's derivative with respect to the index
+  of a thin air shell around a large core reached errors of $2\cdot10^{-8}$ relative at
+  $x_L = 10^4$). Above $x_L = 1000$, `compute_with_derivatives` and `derivatives=` raise
+  `OutsideValidatedDomain` unless `allow_unvalidated` is set.
 - **Cost.** About 2.5 times a sphere of the outer size for a coated sphere, and roughly
   1.5 times more per further layer ({doc}`theory/layered`).
 
diff --git a/docs/plans/phase-3b-followups.md b/docs/plans/phase-3b-followups.md
index cfa0002..791162d 100644
--- a/docs/plans/phase-3b-followups.md
+++ b/docs/plans/phase-3b-followups.md
@@ -2,15 +2,6 @@
 
 These items came out of phase 3b (the layered-sphere kernel) and were deliberately deferred. Each belongs to the phase or task named in its heading.
 
-## Derivatives (a later phase, before large coated particles need Jacobians)
-
-**Closed-form derivatives through the recursion.**
-- **What:** derivatives by the chain rule through the H recursion, with an adjoint sweep per order, O(L) per order. They would replace the dual numbers for L ≥ 2.
-- **Why:** two reasons.
-  - Dual numbers differentiate every intermediate quantity, and for thin lossless shells some of those are far more sensitive than the result. The derivatives' validated domain therefore ends at x_L = 100, where the values reach 10⁴.
-  - The cost misses (1 + 3L)× slightly at x_L = 100 (7.4× for two layers, 32× for ten).
-- **Then:** extend `derivative_x_max`, and re-measure both maps and the cost.
-
 ## Domain
 
 **Thin lossless shells of high index.**
@@ -18,6 +9,11 @@ These items came out of phase 3b (the layered-sphere kernel) and were deliberate
 - **Why deferred:** only very large, lossless, high-index shells are affected, and they are outside the declared domain.
 - **Then:** investigate the narrow whispering-gallery resonances, and whether the condition estimate captures them.
 
+**Derivatives above x_L = 1000.**
+- **What:** the forward amplitude's derivative with respect to a thin air shell's index reaches C ≈ 1700 at x_L = 10⁴. The recurrence errors at the shell's two surfaces are coherent across orders and add up in the forward sum.
+- **Why deferred:** 1000 is ten times the dual numbers' domain, and every other output stays within its constant up to 10⁴.
+- **Then:** find which recurrence carries the coherent error (P_n's error is 10⁻¹² to 10⁻⁹ relative), and whether pairing the two surfaces' terms removes it.
+
 ## Python
 
 **Double-double and dual layered results.** As for spheres, Python exposes `f64` results only.
diff --git a/docs/specs/2026-10-14-phase-3-layered-spheres-design.md b/docs/specs/2026-10-14-phase-3-layered-spheres-design.md
index 47ae9bc..56c7520 100644
--- a/docs/specs/2026-10-14-phase-3-layered-spheres-design.md
+++ b/docs/specs/2026-10-14-phase-3-layered-spheres-design.md
@@ -608,3 +608,23 @@ Each change was forced or decided by a measurement. The plan, `docs/plans/2026-1
   - Two layers cost 2.5–2.65× a sphere, against a target of about 2.5×.
   - scatrs is 4.3–8.4× faster than scattnlay.
   - Derivatives meet (1 + 3L)× up to x_L ≈ 10. At x_L = 100 they cost 7.4× for two layers (target 7) and 32× for ten (target 31). The fallback is section 2.7's closed forms (follow-up).
+
+## 17. Closed-form derivatives
+
+The fallback of section 2.7 replaced the dual numbers for L ≥ 2, before 3c. The theory chapter ("Derivatives") gives the formulas.
+
+- **Method.** Within layer l, H solves the Riccati equation y′ = V − y², V = n(n + 1)/z² − 1, from t = m_l H^(l−1)/m_{l−1} at z₁ to H^(l) at z₂. Its sensitivities are:
+  - ∂H/∂t = R², with R = −m_{l−1}W S_n/(Π_n(z₂)G₂ − P_nG₁) and S_n = ζ_n(z₂)/ζ_n(z₁), from S_n = S_{n−1}w_n(z₂)/w_n(z₁);
+  - ∂H/∂z₂ = V − H²;
+  - ∂H/∂z₁ = −(V(z₁) − t²)R².
+
+  An adjoint sweep per order chains them from the outer layer inward. The outer step uses the sphere's closed forms with H in place of D. The `Dual` machinery of `layered` is gone.
+- **Three cancellations are removed.**
+  - The two terms of an interface x_{l−1} are combined into closed forms, which are exactly 0 for equal indices. For H_b at x = 10⁻⁶ the two terms cancelled to a few digits.
+  - A layer's own index, R²[±t − z₁(V(z₁) − t²)], is written with t = D_n(z₁) + e for small z₁. Near a pole of D_n(z₁) it keeps the t form, chosen by the same factor-4 test as H. In the D form alone, a metallic core in a thin shell lost 4·10⁻⁴ relative at x_L = 640.
+  - H′ is D′ − δ(2D + δ) in the δ form.
+- **The values are bitwise those of `compute`**, as before. Spheres and ensembles are bitwise unchanged.
+- **The derivatives' domain is x_L ≤ 1000** (it was 100).
+  - Up to there, the worst C on the maps is 22 for the coefficient derivatives, 37 for the Jacobian, 163 for the amplitude derivatives, and 53 for the phase-matrix and Greek derivatives. The constants are unchanged.
+  - Above 1000, the forward amplitude's derivative with respect to a thin air shell's index reaches C ≈ 1700 at x_L = 10⁴ (0.999 fraction, core 1.5 + 0.01i). The recurrence errors at the shell's two surfaces are independent of each other and coherent across orders, so they add up in the forward sum. Each coefficient's derivative stays within its tolerance.
+- **Cost.** Every derivative (3L of them) costs 1.6 to 2.8 times the values, for 2, 3 and 10 layers, from x_L = 0.1 to 1000. The target is (1 + 3L)×. With duals it was 4.6 to 7.4 times for two layers and 25 to 32 for ten.
diff --git a/docs/theory/layered.md b/docs/theory/layered.md
index d0a7ea1..5ba81da 100644
--- a/docs/theory/layered.md
+++ b/docs/theory/layered.md
@@ -144,28 +144,70 @@ absorbs nothing; scatrs sets its $Q_\text{abs}$ to exactly 0, as for a real sphe
 ## Derivatives
 
 The derivatives with respect to each layer's $x_l$, $\mathrm{Re}\,m_l$ and $\mathrm{Im}\,m_l$
-come from dual numbers ({doc}`scalar-types`) carried through the same code. The coefficients
-are holomorphic in each $m_l$, so seeding $x_l$ and $\mathrm{Re}\,m_l$ is enough: the
-derivative with respect to $\mathrm{Re}\,m_l$ is the complex derivative $d/dm_l$, and the
-one with respect to $\mathrm{Im}\,m_l$ is $i\,d/dm_l$. The absorption sum is not holomorphic;
-its derivative with respect to $\mathrm{Im}\,m_l$ follows from the coefficients',
+are in closed form. The coefficients are holomorphic in each $m_l$: the derivative with respect
+to $\mathrm{Re}\,m_l$ is the complex derivative $d/dm_l$, and the one with respect to
+$\mathrm{Im}\,m_l$ is $i\,d/dm_l$. The absorption sum is not holomorphic; its derivative with
+respect to $\mathrm{Im}\,m_l$ follows from the coefficients',
 
 $$
 \frac{\partial}{\partial\,\mathrm{Im}\,m_l}\bigl(\mathrm{Re}\,a_n - |a_n|^2\bigr)
   = 2\,\mathrm{Im}\bigl(\bar a_n a_n'\bigr) - \mathrm{Im}\,a_n',
 $$
 
-with $a_n' = da_n/dm_l$, likewise for $b_n$. Because $\mathrm{Im}\,m$ is never seeded, the values
-take the same paths as without derivatives and are bitwise the same: a real index keeps the
-cheaper real recurrence, and a lossless particle keeps $Q_\text{abs} = 0$. For up to four
-layers one pass carries all $2L$ derivatives; above, one pass per layer carries two. One layer
-uses the sphere's closed forms.
+with $a_n' = da_n/dm_l$, likewise for $b_n$.
 
-The derivatives lose accuracy faster than the values as the particle grows. The dual numbers
-differentiate every intermediate quantity, and for thin lossless shells some of those are far
-more sensitive than the result. Their validated domain therefore ends at $x_L = 100$
-({doc}`/validation/accuracy`). Derivatives in closed form, by the chain rule through the
-recursion, would avoid this; they are planned.
+Inside layer $l$ the logarithmic derivative $y = f'/f$ of the layer's field $f$ (of order $n$)
+solves the Riccati equation
+
+$$
+y' = V - y^2, \qquad V(z) = \frac{n(n + 1)}{z^2} - 1,
+$$
+
+from $y(z_1) = t = m_l H^{(l-1)}/m_{l-1}$ at the inner surface (for $H_a$; the indices are
+exchanged for $H_b$) to $y(z_2) = H^{(l)}$. Its sensitivities follow without differentiating
+the recursion:
+
+$$
+\frac{\partial H^{(l)}}{\partial t} = R^2, \qquad
+\frac{\partial H^{(l)}}{\partial z_2} = V(z_2) - \bigl(H^{(l)}\bigr)^2, \qquad
+\frac{\partial H^{(l)}}{\partial z_1} = -\bigl(V(z_1) - t^2\bigr)R^2,
+$$
+
+with $R = f(z_1)/f(z_2)$ the ratio of the field at the two surfaces,
+
+$$
+R = -\frac{m_{l-1} W S_n}{\Pi_n(z_2) G_2 - P_n G_1}, \qquad
+S_n = \frac{\zeta_n(z_2)}{\zeta_n(z_1)} = S_{n-1}\,\frac{w_n(z_2)}{w_n(z_1)},
+$$
+
+which reuses the recursion's quantities. At each order a sweep from the outer layer inward
+chains these sensitivities to give the outer $H_a$ and $H_b$'s derivatives with respect to
+every layer's $x_l$ and $m_l$; the outer step then differentiates as the sphere's closed forms
+do, with $H$ in place of $D_n(mx)$. All $3L$ derivatives cost one pass, and the values are
+bitwise those computed without derivatives.
+
+The sweep's terms need the same care as the values, for the same reasons:
+
+- **At an interface.** $x_{l-1}$ moves both the outer surface of layer $l - 1$ and the inner
+  surface of layer $l$. For $H_b$ the two terms cancel at leading order at small $x_{l-1}$.
+  Together they are, per unit of the outer $H$'s sensitivity to $t$,
+  $m_l(m_l^2 - m_{l-1}^2)\bigl[n(n + 1)/(x_{l-1}^2 m_{l-1}^2 m_l^2) + H^2/m_{l-1}^2\bigr]$ for
+  $H_a$ and $(m_l^2 - m_{l-1}^2)/m_l$ for $H_b$ ($H$ the inner layer's), which vanish exactly
+  for equal indices.
+- **A layer's own index** enters through $z_1$ and $t$ as
+  $R^2\bigl[\pm t - z_1\bigl(V(z_1) - t^2\bigr)\bigr]$, whose two terms are both about
+  $(n + 1)/z_1$ at small $z_1$. Written with $t = D_n(z_1) + e$ it is
+  $\pm D_n - z_1 D_n' + e\bigl(\pm 1 + z_1(2D_n + e)\bigr)$, free of that cancellation; near a
+  pole of $D_n(z_1)$, where $D_n$ and $e$ cancel instead, scatrs keeps the first form.
+- **The derivative of $H$** at $z_2$, $V - H^2$, is
+  $D_n'(z_2) - \delta\bigl(2D_n(z_2) + \delta\bigr)$ in the form $H = D_n(z_2) + \delta$.
+
+The derivatives' validated domain ends at $x_L = 1000$ ({doc}`/validation/accuracy`). For thin
+lossless shells around large particles, the rounding errors of the recurrences at a shell's
+two surfaces are independent of each other and alike from one order to the next. In the
+forward-scattering amplitude's derivative with respect to the shell's index they add up instead
+of averaging out. That derivative reaches about $1.7$ times its declared constant at
+$x_L = 10^4$, for an air shell of a thousandth of the radius.
 
 ## Truncation
 
@@ -186,8 +228,8 @@ $$
 $$
 
 The derivatives of the coefficients and the amplitudes take $\kappa$ itself, near-unity factor
-included, since they share the algorithm's losses near $m = 1$; a sphere's closed forms do
-not. The constants $C$ were measured on two accuracy maps and at the Arb references, each at
+included, since they are built from the recursion's $H$ and share its losses near $m = 1$; a
+sphere's closed forms do not. The constants $C$ were measured on two accuracy maps and at the Arb references, each at
 least four times the worst case found, and differ from the sphere's where the layered kernel
 needs more ({rust}`scatrs::layered::ValidatedDomain::constant`). The validated domain:
 
@@ -196,7 +238,7 @@ needs more ({rust}`scatrs::layered::ValidatedDomain::constant`). The validated d
   $10^9$ ulps;
 - every layer's index in the sphere's domain ($0 < \mathrm{Re}\,m \le 10$,
   $0 \le \mathrm{Im}\,m \le 10$, $|m| \ge 0.1$), and at most 16 layers;
-- the derivatives up to $x_L = 100$.
+- the derivatives up to $x_L = 1000$.
 
 ## Reference values
 
@@ -213,6 +255,6 @@ recurrences of $\Pi_n$, $w_n$ and $P_n$. Measured on one thread of an Apple M4 P
 (`cargo bench -p scatrs --bench layered`), the efficiencies of a coated sphere cost about 2.5
 to 2.7 times those of a homogeneous sphere of the same outer size: 1.7 µs at $x_L = 1$, 11 µs at
 100 and 94 µs at 1000. A three-layer particle costs 3.3 to 4 times a sphere, a ten-layer one 13
-to 16 times. Every derivative of a two-layer particle costs 4.6 to 7.4 times the values,
-of a three-layer one 5.6 to 8.9 times, and of a ten-layer one (one pass per layer) 25 to 32
-times.
+to 16 times. Every derivative ($3L$ of them) costs 1.6 to 2.8 times the values, for two,
+three or ten layers alike: 3.3 µs for a coated sphere at $x_L = 1$, 33 µs at 100 and 270 µs at
+1000.
diff --git a/docs/validation/accuracy.md b/docs/validation/accuracy.md
index 9af935d..ad827f7 100644
--- a/docs/validation/accuracy.md
+++ b/docs/validation/accuracy.md
@@ -187,7 +187,7 @@ plot_map(greek, gx, gm, gnames, rows=2, height=7)
 The outputs of {mod}`scatrs.layered` follow the same model ({doc}`/theory/layered`), with the
 condition numbers summed over every layer's parameters and their own constants. Inside their
 validated domain ($10^{-6} \le x_L \le 10^4$ with $|m_l|\,x_L \le 3\cdot10^4$, every index in
-the sphere's domain, at most 16 layers; the derivatives up to $x_L = 100$), four tests enforce
+the sphere's domain, at most 16 layers; the derivatives up to $x_L = 1000$), four tests enforce
 them:
 
 - `scatrs/tests/layered_reference.rs` compares every output with the Arb reference values,
@@ -208,7 +208,7 @@ them:
   central differences.
 
 The worst $C$ of each map against its declared constant, and the worst $C$ over all particles
-at each size (empty above $x_L = 100$ for the derivatives):
+at each size (empty above $x_L = 1000$ for the derivatives):
 
 ```{code-cell} python
 def worst_layered(name):
@@ -249,10 +249,12 @@ for ax in axes:
 plt.show()
 ```
 
-The values use at most a quarter of their constants anywhere. The derivatives grow with the
-particle: for thin lossless shells (a real index of 10, or an air shell around a large core)
-the amplitude derivatives use about a fifth of their constant at $x_L = 50$, and would exceed it
-by up to 6 times at $x_L \sim 10^3$, which is why their domain ends at 100.
+The values use at most a quarter of their constants anywhere, and so do the derivatives up to
+$x_L = 1000$. Above it, the amplitude derivatives of thin lossless shells grow with the
+particle: for an air shell of a thousandth of the radius, the forward amplitude's derivative
+with respect to the shell's index would exceed its constant by 1.7 times at $x_L = 10^4$. The
+rounding errors of the recurrences at the shell's two surfaces are alike from one order to the
+next, so they add up in the forward sum, and the derivatives' domain ends at 1000.
 
 (ensemble-accuracy)=
 ## Ensembles of spheres
diff --git a/python/scatrs/layered.py b/python/scatrs/layered.py
index 8905377..adc0dc0 100644
--- a/python/scatrs/layered.py
+++ b/python/scatrs/layered.py
@@ -13,7 +13,7 @@ method and :doc:`/validation/accuracy` for the validated domain.
 ``derivatives`` is ``True`` (every layer's ``"layer{l}.x"``, ``"layer{l}.m_real"`` and
 ``"layer{l}.m_imag"``) or a sequence of those names; the result's ``d`` method returns the
 derivative with respect to one of them, or ``"layer{l}.m"`` (the complex derivative) for the
-complex outputs. The derivatives are validated up to an outer size parameter of 100.
+complex outputs. The derivatives are validated up to an outer size parameter of 1000.
 
 The results are those of :mod:`scatrs.mie` (:class:`~scatrs.mie.MieEfficiencies`, …), with
 ``d`` accepting the layers' parameter names.
@@ -215,7 +215,7 @@ _SHARED_DOC = """
     derivatives : bool or sequence of str, optional
         Also compute derivatives with respect to these of ``"layer{{l}}.x"``,
         ``"layer{{l}}.m_real"`` and ``"layer{{l}}.m_imag"`` (``True`` for all); read them with the
-        result's ``d`` method. Validated up to an outer size parameter of 100.
+        result's ``d`` method. Validated up to an outer size parameter of 1000.
     allow_unvalidated : bool, optional
         Compute outside the validated domain instead of raising
         :class:`~scatrs.errors.OutsideValidatedDomainError` (see
diff --git a/scatrs/benches/layered.rs b/scatrs/benches/layered.rs
index 3e082ab..d8465cd 100644
--- a/scatrs/benches/layered.rs
+++ b/scatrs/benches/layered.rs
@@ -11,7 +11,7 @@ use std::hint::black_box;
 
 use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
 use num_complex::Complex;
-use scatrs::layered::Layered;
+use scatrs::layered::{Layered, VALIDATED_DOMAIN};
 use scatrs::mie::{Mie, MieOptions};
 
 const SIZES: [f64; 5] = [0.1, 1.0, 10.0, 100.0, 1000.0];
@@ -57,7 +57,7 @@ fn layered(c: &mut Criterion) {
             group.bench_with_input(BenchmarkId::new(name, x_l), &x, |b, x| {
                 b.iter(|| lay.compute(black_box(x), &m, &opts).unwrap().efficiencies())
             });
-            if x_l <= 100.0 {
+            if x_l <= VALIDATED_DOMAIN.derivative_x_max {
                 let id = BenchmarkId::new(format!("{name}+derivatives"), x_l);
                 group.bench_with_input(id, &x, |b, x| {
                     b.iter(|| {
diff --git a/scatrs/src/layered/domain.rs b/scatrs/src/layered/domain.rs
index 9009e9e..fdfe4d0 100644
--- a/scatrs/src/layered/domain.rs
+++ b/scatrs/src/layered/domain.rs
@@ -43,12 +43,13 @@ pub struct ValidatedDomain {
 
 /// The validated domain: x_L ∈ [10⁻⁶, 10⁴], every layer's m with 0 < Re m ≤ 10, 0 ≤ Im m ≤ 10,
 /// |m| ≥ 0.1 and |m|·x_L ≤ 3·10⁴, at most 16 layers, any radii; the derivatives up to
-/// x_L = 100.
+/// x_L = 1000.
 ///
 /// Beyond |m|·x_L = 3·10⁴ (x_L = 10⁴ with a real index of 10), the narrow resonances of thin
-/// lossless shells cost the coefficients up to 10⁹ ulps. Above x_L = 100 the derivatives
-/// (dual numbers through the recursion) lose accuracy faster than the values, worst for thin
-/// lossless shells: the amplitude derivatives reach C ≈ 6000 at x_L ~ 10³.
+/// lossless shells cost the coefficients up to 10⁹ ulps. Above x_L = 1000 the derivatives lose
+/// accuracy faster than the values for thin lossless shells, whose two surfaces' rounding errors
+/// add up over the orders in the forward amplitude: its derivative with respect to the index of
+/// an air shell of a thousandth of the radius reaches C ≈ 1700 at x_L = 10⁴.
 pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
     x_min: 1e-6,
     x_max: 1e4,
@@ -57,7 +58,7 @@ pub const VALIDATED_DOMAIN: ValidatedDomain = ValidatedDomain {
     abs_m_min: 0.1,
     max_layers: 16,
     max_phase: 3e4,
-    derivative_x_max: 100.0,
+    derivative_x_max: 1e3,
 };
 
 impl ValidatedDomain {
@@ -119,8 +120,8 @@ impl ValidatedDomain {
         let near_one = (kappa / (1.0 + big * x)).max(1.0);
         let quadrature = |order: usize| ((order as f64 + 1.0).powi(2) + x * x) / 8.0;
         let prior = match output {
-            // The derivatives come from dual numbers through the algorithm, so they share its
-            // losses near m = 1 (a sphere's closed forms do not).
+            // The derivatives are built from the recursion's H, so they share its losses near
+            // m = 1 (a sphere's closed forms do not).
             MieOutput::CoefficientDerivatives | MieOutput::AmplitudeDerivatives => kappa,
             MieOutput::EfficiencyJacobian | MieOutput::PhaseMatrixDerivatives => kappa * near_one,
             MieOutput::GreekCoefficients { order } => kappa + quadrature(order),
diff --git a/scatrs/src/layered/kernel.rs b/scatrs/src/layered/kernel.rs
index 61e8c9a..8696f2e 100644
--- a/scatrs/src/layered/kernel.rs
+++ b/scatrs/src/layered/kernel.rs
@@ -22,7 +22,7 @@
 use num_complex::Complex;
 
 use crate::complex::{exp_m1, inv};
-use crate::mie::kernel::{Outer, Side, State, int, ratios, real};
+use crate::mie::kernel::{Outer, OuterDerivs, Side, State, int, ratios, real};
 use crate::scalar::Real;
 
 /// Π_n(z) and w_{n+1}(z) of one argument, advanced order by order by Π_n = Π_{n−1} u_n w_n
@@ -31,6 +31,7 @@ use crate::scalar::Real;
 /// for ζ = χ (χ₀ = cos z, χ₋₁ = −sin z).
 #[derive(Clone, Copy, Debug)]
 struct Arg<T> {
+    z: Complex<T>,
     inv_z: Complex<T>,
     pi: Complex<T>,
     /// w_{n+1} after `step(n)`.
@@ -57,6 +58,7 @@ impl<T: Real> Arg<T> {
             )
         };
         Self {
+            z,
             inv_z,
             pi,
             w,
@@ -80,6 +82,18 @@ impl<T: Real> Arg<T> {
     fn d3(&self, n: usize) -> Complex<T> {
         self.inv_z.scale(int(n + 1)) - self.w
     }
+
+    /// D′_n(z) = n(n + 1)/z² − 1 − D_n², with u = u_{n+1}(z), in the sphere's form without the
+    /// terms in 1/z² that cancel: (n + 1)(2u/z − 1/z²) − (1 + u²).
+    fn d1_prime(&self, n: usize, u: Complex<T>) -> Complex<T> {
+        let iz = self.inv_z;
+        (u.scale(T::from_f64(2.0)) * iz - iz * iz).scale(int(n + 1)) - (real(T::one()) + u * u)
+    }
+
+    /// D_n + zD′_n, with u = u_{n+1}(z): (2n + 1)u − z(1 + u²), which is O(z) at small z.
+    fn d1_plus(&self, n: usize, u: Complex<T>) -> Complex<T> {
+        u.scale(int(2 * n + 1)) - self.z * (real(T::one()) + u * u)
+    }
 }
 
 /// |re| + |im|.
@@ -100,6 +114,35 @@ struct Step<T> {
     p: Complex<T>,
     /// The Wronskian W of ψ and the layer's ζ.
     w: Complex<T>,
+    /// Whether to compute the derivatives' sensitivities, and for them: the order, u_{n+1}(z₁),
+    /// u_{n+1}(z₂), S = ζ_n(z₂)/ζ_n(z₁), and the arguments.
+    derivatives: bool,
+    n: usize,
+    u2: Complex<T>,
+    s: Complex<T>,
+    a1: Arg<T>,
+    a2: Arg<T>,
+}
+
+/// The sensitivities of a layer's H at one order, for the closed-form derivatives. H is the
+/// value at z₂ of the logarithmic derivative y of the layer's field, which solves the Riccati
+/// equation y′ = V − y², V = n(n + 1)/z² − 1, from y(z₁) = t (the inner layer's H times
+/// mi/mo). Hence ∂H/∂t = R², R = f(z₁)/f(z₂) the ratio of the field at the two surfaces
+/// (= −mo W S/(Π_n(z₂)G₂ − P_n G₁)), ∂H/∂z₂ = V(z₂) − H² and ∂H/∂z₁ = −(V(z₁) − t²)R².
+#[derive(Clone, Copy, Debug)]
+pub(super) struct Sens<T> {
+    /// H itself.
+    hv: Complex<T>,
+    r2: Complex<T>,
+    /// m_l·∂H/∂m_l through the inner surface, over R²: σt − z₁(V(z₁) − t²), with σ = +1 for Hₐ
+    /// and −1 for H_b (t ∝ m_l^σ). Its two terms are both about (n + 1)/z₁ and cancel for H_b; with
+    /// t = D + e (D = D_n(z₁), e = G₁/mo) it is σD − zD′ + e(σ + z(2D + e)), whose first term is
+    /// −(D + zD′) for H_b, free of the cancellation in the sphere's form.
+    em: Complex<T>,
+    /// H′ = V(z₂) − H².
+    hp: Complex<T>,
+    /// H + z₂H′.
+    hplus: Complex<T>,
 }
 
 /// H at the outer surface of a layer from H at its inner surface (`h`, as `prev` represents it),
@@ -119,7 +162,8 @@ fn side<T: Real>(
     h: Complex<T>,
     lead: Complex<T>,
     s: &Step<T>,
-) -> (Side<T>, Complex<T>) {
+    lead_is_a: bool,
+) -> (Side<T>, Complex<T>, Option<Sens<T>>) {
     let mh = mi * h;
     let g1 = match prev {
         Side::Delta(delta) => lead - mi * s.u_prev + mo * s.u1 + mi * delta,
@@ -129,11 +173,84 @@ fn side<T: Real>(
     let inv_den = inv(s.pio * g2 - s.p * g1);
     let delta = -(s.w * s.p * g1 * inv_den * inv(s.pio));
     let hd = s.d1o + delta;
-    if norm1(hd) * 4.0 >= norm1(s.d1o).max(norm1(delta)) {
+    let (out, h_out) = if norm1(hd) * 4.0 >= norm1(s.d1o).max(norm1(delta)) {
         (Side::Delta(delta), hd)
     } else {
         let hp = s.d3o - s.w * g2 * inv_den;
         (Side::Direct(hp), hp)
+    };
+    if !s.derivatives {
+        return (out, h_out, None);
+    }
+    let two = T::from_f64(2.0);
+    let inv_mo = inv(mo);
+    let t = mh * inv_mo;
+    let r = -(mo * s.w * s.s * inv_den);
+    let e = g1 * inv_mo;
+    let sigma = if lead_is_a { T::one() } else { -T::one() };
+    let d1 = s.d1i;
+    let z1 = s.a1.z;
+    let own = if lead_is_a {
+        d1.scale(two) - s.a1.d1_plus(s.n, s.u1) // D − zD′
+    } else {
+        -s.a1.d1_plus(s.n, s.u1) // −(D + zD′)
+    };
+    let em = if norm1(t) * 4.0 >= norm1(d1).max(norm1(e)) {
+        own + e * (real(sigma) + z1 * (d1.scale(two) + e))
+    } else {
+        // Near a pole of D_n(z₁), where the terms of that form cancel instead (D ≈ −e ≫ t).
+        let iz = s.a1.inv_z;
+        let v = real(int::<T>(s.n) * int::<T>(s.n + 1)) * iz * iz - real(T::one());
+        t.scale(sigma) - z1 * (v - t * t)
+    };
+    let (hp, hplus) = match out {
+        // H = D + δ: H′ = D′ − δ(2D + δ) and H + zH′ = (D + zD′) + δ(1 − z(2D + δ)), each
+        // free of the cancellation at small z.
+        Side::Delta(delta) => {
+            let d2 = s.d1o.scale(two) + delta;
+            (
+                s.a2.d1_prime(s.n, s.u2) - delta * d2,
+                s.a2.d1_plus(s.n, s.u2) + delta * (real(T::one()) - s.a2.z * d2),
+            )
+        }
+        // Near a pole of D_n(z₂), where nothing cancels.
+        Side::Direct(hv) => {
+            let iz = s.a2.inv_z;
+            let v = real(int::<T>(s.n) * int::<T>(s.n + 1)) * iz * iz - real(T::one());
+            let hp = v - hv * hv;
+            (hp, hv + s.a2.z * hp)
+        }
+    };
+    let sens = Sens {
+        hv: h_out,
+        r2: r * r,
+        em,
+        hp,
+        hplus,
+    };
+    (out, h_out, Some(sens))
+}
+
+/// The derivatives of one inner layer's (or with `outer`, the last layer's) coefficients and
+/// absorption sum, as `Layered` stores them.
+#[derive(Clone, Debug)]
+pub(super) struct LayerGrad<T> {
+    pub(super) da_dx: Vec<Complex<T>>,
+    pub(super) da_dm: Vec<Complex<T>>,
+    pub(super) db_dx: Vec<Complex<T>>,
+    pub(super) db_dm: Vec<Complex<T>>,
+    pub(super) abs_sum: [T; 3],
+}
+
+impl<T: Real> LayerGrad<T> {
+    fn new() -> Self {
+        Self {
+            da_dx: Vec::new(),
+            da_dm: Vec::new(),
+            db_dx: Vec::new(),
+            db_dm: Vec::new(),
+            abs_sum: [T::zero(); 3],
+        }
     }
 }
 
@@ -153,6 +270,15 @@ pub(super) struct Kernel<T: Real> {
     /// Whether layer l ≥ 1 takes ζ = χ (else ξ).
     chi: Vec<bool>,
     outer: Vec<Outer<T>>,
+    /// For the derivatives: S = ζ_n(z₂)/ζ_n(z₁) of each layer l ≥ 1, each layer's sensitivities
+    /// at the current order (Hₐ, H_b), the outer layer's quantities per order, and dH/dx_l,
+    /// dH/dm_l of every layer per order (for Hₐ then H_b).
+    s: Vec<Complex<T>>,
+    sens: Vec<(Sens<T>, Sens<T>)>,
+    outer_derivs: Vec<OuterDerivs<T>>,
+    dh: Vec<[Vec<Complex<T>>; 4]>,
+    /// The derivatives of every layer but the last (whose `mie` holds).
+    pub(super) grads: Vec<LayerGrad<T>>,
 }
 
 impl<T: Real> Kernel<T> {
@@ -167,12 +293,19 @@ impl<T: Real> Kernel<T> {
             lead: Vec::new(),
             chi: Vec::new(),
             outer: Vec::new(),
+            s: Vec::new(),
+            sens: Vec::new(),
+            outer_derivs: Vec::new(),
+            dh: Vec::new(),
+            grads: Vec::new(),
         }
     }
 
     /// a_n, b_n for n = 1, …, `n` and the absorption sum of the layered sphere `x`, `m`
-    /// (validated, L ≥ 2) into `self.mie`.
-    pub(super) fn solve(&mut self, x: &[T], m: &[Complex<T>], n: usize) {
+    /// (validated, L ≥ 2) into `self.mie`; with `derivatives`, their derivatives with respect
+    /// to the last layer's x and m into `self.mie` too and to every other layer's into
+    /// `self.grads`, in closed form.
+    pub(super) fn solve(&mut self, x: &[T], m: &[Complex<T>], n: usize, derivatives: bool) {
         let layers = x.len();
         debug_assert!(layers >= 2 && m.len() == layers);
         let last = layers - 1;
@@ -213,6 +346,30 @@ impl<T: Real> Kernel<T> {
             };
             self.p.push(p0);
         }
+        self.s.clear();
+        if derivatives {
+            for l in 1..layers {
+                let (z1, z2) = (m[l].scale(x[l - 1]), m[l].scale(x[l]));
+                // S₀ = ζ₀(z₂)/ζ₀(z₁): cos z₂/cos z₁, or e^(i m_l (x_l − x_{l−1})).
+                let s0 = if self.chi[l - 1] {
+                    z2.cos() * inv(z1.cos())
+                } else {
+                    let dx = x[l] - x[l - 1];
+                    let (sin_b, cos_b) = (m[l].re * dx).sin_cos();
+                    Complex::new(cos_b, sin_b).scale((-(m[l].im * dx)).exp())
+                };
+                self.s.push(s0);
+            }
+            self.dh
+                .resize_with(layers, || core::array::from_fn(|_| Vec::new()));
+            self.dh.truncate(layers);
+            for d in &mut self.dh {
+                for v in d.iter_mut() {
+                    v.clear();
+                }
+            }
+            self.outer_derivs.clear();
+        }
         // Only its D_n is used.
         let core = Arg::new(m[0].scale(x[0]), false);
         self.outer.clear();
@@ -225,7 +382,11 @@ impl<T: Real> Kernel<T> {
                 ai.step(k, u1[k]);
                 ao.step(k, u2[k]);
                 self.p[l - 1] = self.p[l - 1] * u1[k] * inv(ai.w_n) * ao.w_n * ao.w_n;
+                if derivatives {
+                    self.s[l - 1] = self.s[l - 1] * ao.w_n * inv(ai.w_n);
+                }
             }
+            self.sens.clear();
             let mut ha = core.d1(k, self.u[0][k + 1]);
             let mut hb = ha;
             let zero = Side::Delta(real(T::zero()));
@@ -251,20 +412,178 @@ impl<T: Real> Kernel<T> {
                     } else {
                         Complex::new(T::zero(), T::one())
                     },
+                    derivatives,
+                    n: k,
+                    u2: u_of(l)[k + 1],
+                    s: if derivatives {
+                        self.s[l - 1]
+                    } else {
+                        real(T::zero())
+                    },
+                    a1: *ai,
+                    a2: *ao,
                 };
                 let lead = self.lead[l - 1].scale(int::<T>(k + 1) / x[l - 1]);
-                let (sa, a) = side(m[l], m[l - 1], sides.0, ha, lead, &s);
-                let (sb, b) = side(m[l - 1], m[l], sides.1, hb, real(T::zero()), &s);
+                let (sa, a, ea) = side(m[l], m[l - 1], sides.0, ha, lead, &s, true);
+                let (sb, b, eb) = side(m[l - 1], m[l], sides.1, hb, real(T::zero()), &s, false);
                 (ha, hb, sides) = (a, b, (sa, sb));
+                if let (Some(ea), Some(eb)) = (ea, eb) {
+                    self.sens.push((ea, eb));
+                }
             }
             self.outer.push(sides);
+            if derivatives {
+                let u0 = self.u[0][k + 1];
+                let core_h = (core.d1(k, u0), core.d1_plus(k, u0));
+                let extra = Self::adjoint(&self.sens, &mut self.dh, x, m, k, core_h);
+                let (la, lb) = self.sens[last - 1];
+                self.outer_derivs.push(OuterDerivs {
+                    hp: [la.hp, lb.hp],
+                    hplus: [la.hplus, lb.hplus],
+                    extra_m: extra,
+                });
+            }
         }
+        let od = derivatives.then_some(&self.outer_derivs[..]);
         self.mie
-            .finish(x[last], m[last], n, false, Some(&self.outer));
+            .finish(x[last], m[last], n, derivatives, Some(&self.outer), od);
+        if derivatives {
+            self.inner_grads(m[last], n);
+        }
         // A lossless particle absorbs nothing; H is real then in exact arithmetic, but its
         // complex intermediates leave a rounding-level imaginary part.
         if m.iter().all(|m| m.im.is_zero()) {
             self.mie.abs_sum = T::zero();
+            // And it stays lossless as any x or Re m changes.
+            self.mie.abs_sum_jacobian[0] = T::zero();
+            self.mie.abs_sum_jacobian[1] = T::zero();
+            for g in &mut self.grads {
+                g.abs_sum[0] = T::zero();
+                g.abs_sum[1] = T::zero();
+            }
+        }
+    }
+
+    /// The adjoint sweep at order `n`, from the sensitivities of every layer: dH/dx_l and dH/dm_l
+    /// of the outer H into `dh` for every layer but the last, and for the last its index's
+    /// derivative through its inner surface, which it returns (for Hₐ, H_b). `core` is
+    /// (D_n, D_n + zD′_n) of the core's argument.
+    ///
+    /// An interface x_{l−1} moves both the inner layer's outer surface and layer l's inner one,
+    /// whose terms cancel at leading order for H_b. Together, per unit of the outer H's
+    /// sensitivity λ_l R_l² to layer l's starting value, they are the closed forms
+    /// J = m_l(m_l² − m_{l−1}²)[n(n + 1)/(x² m_{l−1}² m_l²) + H²/m_{l−1}²] for Hₐ and
+    /// (m_l² − m_{l−1}²)/m_l for H_b (H the inner layer's, x = x_{l−1}): the terms in n(n + 1)/x² and
+    /// H² cancel exactly. Likewise the inner layer's index, through its outer surface and t_l:
+    /// −(m_l/m_{l−1}²)(H − zH′) for Hₐ and (H + zH′)/m_l for H_b, z = m_{l−1}x_{l−1}.
+    fn adjoint(
+        sens: &[(Sens<T>, Sens<T>)],
+        dh: &mut [[Vec<Complex<T>>; 4]],
+        x: &[T],
+        m: &[Complex<T>],
+        n: usize,
+        core: (Complex<T>, Complex<T>),
+    ) -> [Complex<T>; 2] {
+        let layers = x.len();
+        let last = layers - 1;
+        let two = T::from_f64(2.0);
+        let nn1 = int::<T>(n) * int::<T>(n + 1);
+        let mut extra = [real(T::zero()); 2];
+        for side in 0..2 {
+            let mut dx = vec![real(T::zero()); layers];
+            let mut dm = vec![real(T::zero()); layers];
+            let mut lambda = real(T::one());
+            for l in (1..layers).rev() {
+                let e = if side == 0 {
+                    sens[l - 1].0
+                } else {
+                    sens[l - 1].1
+                };
+                let (ml, mp) = (m[l], m[l - 1]);
+                // Layer l's index through its inner surface and t (over R²: em/m_l).
+                dm[l] = dm[l] + lambda * e.r2 * e.em * inv(ml);
+                // The inner layer's H at the interface, and H + zH′ there.
+                let (h, hplus) = if l == 1 {
+                    core
+                } else {
+                    let p = if side == 0 {
+                        sens[l - 2].0
+                    } else {
+                        sens[l - 2].1
+                    };
+                    (p.hv, p.hplus)
+                };
+                let c = (ml - mp) * (ml + mp);
+                let w = lambda * e.r2;
+                let inv_mp = inv(mp);
+                let (jump, k) = if side == 0 {
+                    let inv_x = x[l - 1].recip();
+                    let j = ml
+                        * c
+                        * (real(nn1 * inv_x * inv_x) * inv(mp * mp * ml * ml)
+                            + h * h * inv_mp * inv_mp);
+                    // H − zH′ = 2H − (H + zH′).
+                    (j, -(ml * inv_mp * inv_mp * (h.scale(two) - hplus)))
+                } else {
+                    let inv_ml = inv(ml);
+                    (c * inv_ml, hplus * inv_ml)
+                };
+                dx[l - 1] = dx[l - 1] + w * jump;
+                dm[l - 1] = dm[l - 1] + w * k;
+                let (mi, mo) = if side == 0 { (ml, mp) } else { (mp, ml) };
+                lambda = w * mi * inv(mo);
+            }
+            extra[side] = dm[last];
+            for l in 0..last {
+                dh[l][2 * side].push(dx[l]);
+                dh[l][2 * side + 1].push(dm[l]);
+            }
+        }
+        extra
+    }
+
+    /// The derivatives of the coefficients and the absorption sum with respect to every layer
+    /// but the last, from `self.dh` and what the outer step stored: with B = Hₐ/m (m H_b),
+    /// ∂a/∂p = −i(∂B/∂p)/Δₐ², and the absorption term −Im(B)/|Δ|² changes by
+    /// −[Im(∂B) − 2 Im(B) Re(∂B·ξ_n/Δ)]/|Δ|².
+    fn inner_grads(&mut self, m: Complex<T>, n: usize) {
+        let last = self.dh.len() - 1;
+        let i1 = Complex::new(T::zero(), T::one());
+        let two = T::from_f64(2.0);
+        let inv_m = inv(m);
+        self.grads.resize_with(last, LayerGrad::new);
+        self.grads.truncate(last);
+        let term = |f: Complex<T>, df: Complex<T>, l: Complex<T>, n2: T| {
+            -((df.im - two * f.im * l.re) * n2)
+        };
+        for (l, g) in self.grads.iter_mut().enumerate() {
+            for v in [&mut g.da_dx, &mut g.da_dm, &mut g.db_dx, &mut g.db_dm] {
+                v.clear();
+            }
+            let mut jac = [T::zero(); 3];
+            let dh = &self.dh[l];
+            #[allow(clippy::needless_range_loop)] // `dh`'s four arrays and `stored` are parallel
+            for k in 0..n {
+                let [inv_da, inv_db, g_a, g_b, b_a, b_b] = self.mie.stored[k];
+                let (ida2, idb2) = (inv_da * inv_da, inv_db * inv_db);
+                let (na, nb) = (inv_da.norm_sqr(), inv_db.norm_sqr());
+                let ba_x = dh[0][k] * inv_m;
+                let ba_m = dh[1][k] * inv_m;
+                let bb_x = m * dh[2][k];
+                let bb_m = m * dh[3][k];
+                g.da_dx.push(-(i1 * ba_x * ida2));
+                g.da_dm.push(-(i1 * ba_m * ida2));
+                g.db_dx.push(-(i1 * bb_x * idb2));
+                g.db_dm.push(-(i1 * bb_m * idb2));
+                let weight: T = int(2 * k + 3);
+                let pair = |da: Complex<T>, db: Complex<T>| {
+                    term(b_a, da, da * g_a, na) + term(b_b, db, db * g_b, nb)
+                };
+                jac[0] += weight * pair(ba_x, bb_x);
+                jac[1] += weight * pair(ba_m, bb_m);
+                jac[2] += weight * pair(i1 * ba_m, i1 * bb_m);
+            }
+            g.abs_sum = jac;
         }
     }
 }
diff --git a/scatrs/src/layered/mod.rs b/scatrs/src/layered/mod.rs
index 6c08695..d1fdee1 100644
--- a/scatrs/src/layered/mod.rs
+++ b/scatrs/src/layered/mod.rs
@@ -36,8 +36,9 @@
 //! (Yang 2003; Peña & Pal 2009) in a form without the poles of D_n(m_l x_l), and the outer step
 //! is the homogeneous sphere's with Hₐ and H_b in place of D_n(mx), keeping its numerators free
 //! of cancellation. A single layer is a sphere: it gives [`crate::mie`]'s results bitwise.
-//! Derivatives are forward-mode [`Dual`] numbers through the generic
-//! kernel (for one layer, the sphere's closed forms).
+//! The derivatives with respect to every layer's x and m are in closed form: within a layer, H
+//! solves a Riccati equation, whose sensitivities to its two surfaces and to its starting value
+//! are chained from the outer layer inward at each order (for one layer, the sphere's).
 //!
 //! # Accuracy
 //!
@@ -56,7 +57,7 @@ use crate::mie::{
     MAX_SIZE_PARAMETER, MIN_REFRACTIVE_INDEX, MIN_SIZE_PARAMETER, MieOptions, Normalization,
     PhaseBuffers, default_terms,
 };
-use crate::scalar::{Dual, Real};
+use crate::scalar::Real;
 use crate::{Error, Result};
 
 /// The largest number of layers [`Layered::compute`] accepts, 2¹⁰.
@@ -84,38 +85,6 @@ impl<T: Real> LayerDerivatives<T> {
     }
 }
 
-/// A kernel in dual numbers with `N` derivatives, allocated on first use.
-type DualKernel<T, const N: usize> = Option<Box<kernel::Kernel<Dual<T, N>>>>;
-
-/// The kernels in dual numbers of a workspace, one per number of derivatives.
-#[derive(Clone, Debug)]
-struct DualKernels<T: Real> {
-    k2: DualKernel<T, 2>,
-    k4: DualKernel<T, 4>,
-    k6: DualKernel<T, 6>,
-    k8: DualKernel<T, 8>,
-}
-
-/// The kernel with `N` derivatives.
-trait DualSlot<T: Real, const N: usize> {
-    fn kernel(&mut self) -> &mut kernel::Kernel<Dual<T, N>>;
-}
-
-macro_rules! dual_slot {
-    ($n:literal, $field:ident) => {
-        impl<T: Real> DualSlot<T, $n> for DualKernels<T> {
-            fn kernel(&mut self) -> &mut kernel::Kernel<Dual<T, $n>> {
-                self.$field
-                    .get_or_insert_with(|| Box::new(kernel::Kernel::new()))
-            }
-        }
-    };
-}
-dual_slot!(2, k2);
-dual_slot!(4, k4);
-dual_slot!(6, k6);
-dual_slot!(8, k8);
-
 /// A reusable workspace for layered spheres, one per thread.
 ///
 /// Its buffers grow to the largest number of terms and layers used and are then reused.
@@ -128,7 +97,6 @@ pub struct Layered<T: Real> {
     abs_sum: T,
     derivatives: bool,
     layers: Vec<LayerDerivatives<T>>,
-    duals: DualKernels<T>,
 }
 
 impl<T: Real> Default for Layered<T> {
@@ -148,12 +116,6 @@ impl<T: Real> Layered<T> {
             abs_sum: T::zero(),
             derivatives: false,
             layers: Vec::new(),
-            duals: DualKernels {
-                k2: None,
-                k4: None,
-                k6: None,
-                k8: None,
-            },
         }
     }
 
@@ -183,7 +145,7 @@ impl<T: Real> Layered<T> {
         if last == 0 {
             self.kernel.mie.solve(x[0], m[0], n, false);
         } else {
-            self.kernel.solve(x, m, n);
+            self.kernel.solve(x, m, n, false);
         }
         let s = &self.kernel.mie;
         self.x = x[last];
@@ -195,15 +157,11 @@ impl<T: Real> Layered<T> {
     }
 
     /// As [`compute`](Self::compute), also computing the derivatives of the coefficients with
-    /// respect to each layer's size parameter and refractive index: forward-mode dual numbers
-    /// through the kernel, seeded in every layer's x_l and Re m_l (the coefficients are
-    /// holomorphic in m_l, so ∂/∂(Re m_l) is the complex derivative), in one pass with 2L
-    /// derivatives for L ≤ 4 layers and one pass of 2 per layer above; for one layer, the
-    /// sphere's closed forms. The derivative of the absorption sum with respect to Im m_l follows
-    /// from the coefficients' (−Im aₙ′ + 2 Im(āₙaₙ′) and likewise for bₙ, per order). Im m is
-    /// never seeded, so the values take the same paths and are bitwise those of
-    /// [`compute`](Self::compute): a real index keeps its real recurrence, and a lossless particle
-    /// its Q_abs of exactly 0.
+    /// respect to each layer's size parameter and refractive index, in closed form: the
+    /// sensitivities of each layer's logarithmic derivative (a Riccati equation's), combined by
+    /// an adjoint sweep over the layers at every order, and the sphere's closed forms at the
+    /// outer surface (for one layer, the sphere's own). The values are bitwise those of
+    /// [`compute`](Self::compute).
     ///
     /// # Errors
     ///
@@ -225,98 +183,35 @@ impl<T: Real> Layered<T> {
             )));
         }
         let layers = x.len();
-        self.x = x[layers - 1];
+        let last = layers - 1;
+        self.x = x[last];
         self.layers.resize_with(layers, LayerDerivatives::new);
         self.layers.truncate(layers);
-        match layers {
-            1 => {
-                let s = &mut self.kernel.mie;
-                s.solve(x[0], m[0], n, true);
-                self.a.clone_from(&s.a);
-                self.b.clone_from(&s.b);
-                self.abs_sum = s.abs_sum;
-                let d = &mut self.layers[0];
-                d.da_dx.clone_from(&s.da_dx);
-                d.da_dm.clone_from(&s.da_dm);
-                d.db_dx.clone_from(&s.db_dx);
-                d.db_dm.clone_from(&s.db_dm);
-                d.abs_sum = s.abs_sum_jacobian;
-            }
-            2 => self.dual::<4>(x, m, n, 0),
-            3 => self.dual::<6>(x, m, n, 0),
-            4 => self.dual::<8>(x, m, n, 0),
-            _ => {
-                for l in 0..layers {
-                    self.dual::<2>(x, m, n, l);
-                }
+        if last == 0 {
+            self.kernel.mie.solve(x[0], m[0], n, true);
+        } else {
+            self.kernel.solve(x, m, n, true);
+            for (d, g) in self.layers.iter_mut().zip(&self.kernel.grads) {
+                d.da_dx.clone_from(&g.da_dx);
+                d.da_dm.clone_from(&g.da_dm);
+                d.db_dx.clone_from(&g.db_dx);
+                d.db_dm.clone_from(&g.db_dm);
+                d.abs_sum = g.abs_sum;
             }
         }
+        let s = &self.kernel.mie;
+        self.a.clone_from(&s.a);
+        self.b.clone_from(&s.b);
+        self.abs_sum = s.abs_sum;
+        let d = &mut self.layers[last];
+        d.da_dx.clone_from(&s.da_dx);
+        d.da_dm.clone_from(&s.da_dm);
+        d.db_dx.clone_from(&s.db_dx);
+        d.db_dm.clone_from(&s.db_dm);
+        d.abs_sum = s.abs_sum_jacobian;
         self.derivatives = true;
         Ok(LayeredSolution { s: self })
     }
-
-    /// One pass in `Dual<T, N>`, seeding x and Re m of the N/2 layers from `first`; the first
-    /// pass also stores the values.
-    fn dual<const N: usize>(&mut self, x: &[T], m: &[Complex<T>], n: usize, first: usize)
-    where
-        DualKernels<T>: DualSlot<T, N>,
-    {
-        let seeded = first..first + N / 2;
-        let seed = |l: usize, k: usize, v: T| {
-            let mut eps = [T::zero(); N];
-            if seeded.contains(&l) {
-                eps[2 * (l - first) + k] = T::one();
-            }
-            Dual::new(v, eps)
-        };
-        let xd: Vec<Dual<T, N>> = x.iter().enumerate().map(|(l, &v)| seed(l, 0, v)).collect();
-        let md: Vec<Complex<Dual<T, N>>> = m
-            .iter()
-            .enumerate()
-            .map(|(l, v)| Complex::new(seed(l, 1, v.re), Dual::constant(v.im)))
-            .collect();
-        let kernel = DualSlot::<T, N>::kernel(&mut self.duals);
-        kernel.solve(&xd, &md, n);
-        let s = &kernel.mie;
-        let value = |z: &Complex<Dual<T, N>>| Complex::new(z.re.re(), z.im.re());
-        if first == 0 {
-            self.a.clear();
-            self.a.extend(s.a.iter().map(value));
-            self.b.clear();
-            self.b.extend(s.b.iter().map(value));
-            self.abs_sum = s.abs_sum.re();
-        }
-        let part = |z: &Complex<Dual<T, N>>, k: usize| Complex::new(z.re.eps()[k], z.im.eps()[k]);
-        for l in seeded.filter(|&l| l < x.len()) {
-            let j = 2 * (l - first);
-            let d = &mut self.layers[l];
-            for (out, c, k) in [
-                (&mut d.da_dx, &s.a, j),
-                (&mut d.da_dm, &s.a, j + 1),
-                (&mut d.db_dx, &s.b, j),
-                (&mut d.db_dm, &s.b, j + 1),
-            ] {
-                out.clear();
-                out.extend(c.iter().map(|z| part(z, k)));
-            }
-            // ∂/∂(Im m) of Σ (2n + 1)(Re aₙ − |aₙ|² + Re bₙ − |bₙ|²), with ∂aₙ/∂(Im m) = i aₙ′:
-            // Re(i a′) = −Im a′ and 2 Re(ā i a′) = −2 Im(ā a′).
-            let mut im = T::zero();
-            for (k, ((a, b), (da, db))) in self
-                .a
-                .iter()
-                .zip(&self.b)
-                .zip(d.da_dm.iter().zip(&d.db_dm))
-                .enumerate()
-            {
-                let w = T::from_f64((2 * k + 3) as f64);
-                let term = |c: &Complex<T>, dc: &Complex<T>| (c.conj() * dc).im * 2.0 - dc.im;
-                im += w * (term(a, da) + term(b, db));
-            }
-            let e = s.abs_sum.eps();
-            d.abs_sum = [e[j], e[j + 1], im];
-        }
-    }
 }
 
 /// The number of terms [`Layered::compute`] would use for `f64` at (`x`, `m`) with `opts`
diff --git a/scatrs/src/mie/kernel.rs b/scatrs/src/mie/kernel.rs
index 25ffb35..3be6a7c 100644
--- a/scatrs/src/mie/kernel.rs
+++ b/scatrs/src/mie/kernel.rs
@@ -26,6 +26,17 @@ pub(crate) enum Side<T> {
 /// Hₐ and H_b at one order.
 pub(crate) type Outer<T> = (Side<T>, Side<T>);
 
+/// The outer layer's quantities for the closed-form derivatives of a layered sphere at one
+/// order, for Hₐ (index 0) and H_b (index 1): H′ = dH/dz at the outer surface, H + zH′ there
+/// (free of cancellation at small z), and the derivative of H with respect to the outer layer's
+/// index through its inner surface (beyond the outer surface's own).
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct OuterDerivs<T> {
+    pub(crate) hp: [Complex<T>; 2],
+    pub(crate) hplus: [Complex<T>; 2],
+    pub(crate) extra_m: [Complex<T>; 2],
+}
+
 #[derive(Clone, Debug)]
 pub(crate) struct State<T: Real> {
     pub(crate) x: T,
@@ -47,6 +58,9 @@ pub(crate) struct State<T: Real> {
     u_x: Vec<T>,
     /// u_k(mx) for real m, before conversion into `u_z`.
     u_real: Vec<T>,
+    /// For a layered sphere's derivatives, per order: 1/Δₐ, 1/Δ_b, ξ_n/Δₐ, ξ_n/Δ_b, Hₐ/m and m H_b
+    /// (what the inner layers' derivatives need).
+    pub(crate) stored: Vec<[Complex<T>; 6]>,
 }
 
 pub(crate) fn real<T: Real>(v: T) -> Complex<T> {
@@ -155,6 +169,7 @@ impl<T: Real> State<T> {
             u_z: Vec::new(),
             u_x: Vec::new(),
             u_real: Vec::new(),
+            stored: Vec::new(),
         }
     }
 
@@ -162,7 +177,7 @@ impl<T: Real> State<T> {
     /// absorption sum.
     pub(crate) fn solve(&mut self, x: T, m: Complex<T>, n: usize, derivatives: bool) {
         self.prepare(x, m, n);
-        self.finish(x, m, n, derivatives, None);
+        self.finish(x, m, n, derivatives, None, None);
     }
 
     /// The first step of [`solve`](Self::solve): u_k(mx) for k = 1, …, `n` + 1, which
@@ -180,7 +195,8 @@ impl<T: Real> State<T> {
 
     /// The rest of [`solve`](Self::solve), after [`prepare`](Self::prepare) with the same
     /// x, m and n. With `outer`, the effective log-derivatives of a layered sphere at each
-    /// order replace D_n(mx); derivatives are then not available.
+    /// order replace D_n(mx); its derivatives then need `outer_derivs`, and give those with
+    /// respect to the outer layer's x and m (and fill `stored` for the inner layers').
     pub(crate) fn finish(
         &mut self,
         x: T,
@@ -188,8 +204,10 @@ impl<T: Real> State<T> {
         n: usize,
         derivatives: bool,
         outer: Option<&[Outer<T>]>,
+        outer_derivs: Option<&[OuterDerivs<T>]>,
     ) {
-        debug_assert!(!(derivatives && outer.is_some()));
+        debug_assert!(!(derivatives && outer.is_some() && outer_derivs.is_none()));
+        self.stored.clear();
         let one = T::one();
         let two = T::from_f64(2.0);
         let i1 = Complex::new(T::zero(), one);
@@ -250,8 +268,9 @@ impl<T: Real> State<T> {
                 Some(Side::Delta(delta)) => d + delta,
                 Some(Side::Direct(h)) => h,
             };
-            let d_m = h(side.map(|s| s.0)) * inv_m;
-            let md = h(side.map(|s| s.1)) * m;
+            let (hh_a, hh_b) = (h(side.map(|s| s.0)), h(side.map(|s| s.1)));
+            let d_m = hh_a * inv_m;
+            let md = hh_b * m;
             let big_a = d_m + real(k_x);
             let big_b = md + real(k_x);
             // a_n, b_n, 1/Δ_a, 1/Δ_b, ξ_n/Δ and ξ_{n−1}/Δ.
@@ -346,7 +365,8 @@ impl<T: Real> State<T> {
             let weight: T = int(2 * k + 1);
             // Re a_n − |a_n|² and Re b_n − |b_n|².
             abs_sum += weight * (-(d_m.im * na) - md.im * nb);
-            if derivatives {
+            let od = outer_derivs.map(|o| o[k - 1]);
+            if derivatives && od.is_none() {
                 // D′ = n(n + 1)/z² − 1 − D², D + zD′ and D − zD′ in terms of u = u_{n+1}(z),
                 // with D = (n + 1)/z − u: the terms in 1/z² and 1/z that cancel at small z
                 // (D + zD′ = O(z)) are removed exactly.
@@ -381,6 +401,38 @@ impl<T: Real> State<T> {
                 jac[2] += weight
                     * (term(d_m, i1 * fa_m, i1 * la_m, na) + term(md, i1 * fb_m, i1 * lb_m, nb));
             }
+            if let (true, Some(o)) = (derivatives, od) {
+                // A layered sphere: the sphere's formulas with Hₐ, H_b in place of D_n(mx)
+                // (∂a/∂B = −i/Δₐ², ∂a/∂x at fixed B = i(V − B²)/Δₐ², V = n(n + 1)/x² − 1, so
+                // that the outer surface's own terms reduce to the sphere's closed forms), plus
+                // the outer layer's index through its inner surface.
+                let kk1: T = int::<T>(k) * int::<T>(k + 1);
+                let ida2 = inv_da * inv_da;
+                let idb2 = inv_db * inv_db;
+                let [dpa, dpb] = o.hp;
+                let minus_a = hh_a.scale(two) - o.hplus[0]; // Hₐ − zHₐ′
+                let fa_m = -(minus_a * inv_m2) + o.extra_m[0] * inv_m; // dBₐ/dm
+                let fb_m = o.hplus[1] + m * o.extra_m[1]; // dB_b/dm
+                self.da_dm.push(-(i1 * fa_m * ida2));
+                self.db_dm.push(-(i1 * fb_m * idb2));
+                self.da_dx.push(
+                    i1 * m2_minus_1 * (hh_a * hh_a + real(kk1 * inv_x * inv_x)) * inv_m2 * ida2,
+                );
+                self.db_dx.push(i1 * m2_minus_1 * idb2);
+                let k_x2 = k_x * inv_x;
+                let la_x = g_a * (dpa - real(k_x2) - big_a.scale(k_x) + real(one)) + h_a * d_m;
+                let lb_x = g_b * (m2 * dpb - real(k_x2) - big_b.scale(k_x) + real(one)) + h_b * md;
+                let la_m = fa_m * g_a;
+                let lb_m = fb_m * g_b;
+                let term = |f: Complex<T>, df: Complex<T>, l: Complex<T>, n2: T| {
+                    -((df.im - two * f.im * l.re) * n2)
+                };
+                jac[0] += weight * (term(d_m, dpa, la_x, na) + term(md, m2 * dpb, lb_x, nb));
+                jac[1] += weight * (term(d_m, fa_m, la_m, na) + term(md, fb_m, lb_m, nb));
+                jac[2] += weight
+                    * (term(d_m, i1 * fa_m, i1 * la_m, na) + term(md, i1 * fb_m, i1 * lb_m, nb));
+                self.stored.push([inv_da, inv_db, g_a, g_b, d_m, md]);
+            }
         }
         self.abs_sum = abs_sum;
         self.abs_sum_jacobian = jac;
diff --git a/scatrs/tests/layered_accuracy_map.rs b/scatrs/tests/layered_accuracy_map.rs
index 10a5e18..3ccb074 100644
--- a/scatrs/tests/layered_accuracy_map.rs
+++ b/scatrs/tests/layered_accuracy_map.rs
@@ -6,7 +6,7 @@
 //! core fractions `FRACTIONS`, and three- and ten-layer particles. The amplitudes are at the
 //! sphere map's cosines. `DoubleDouble` is within 16 times the declared constants, in units of
 //! (κ′ + cond)·2⁻¹⁰⁴, of Arb on every output (tests/layered_reference.rs), so it stands in for
-//! the exact values. The derivative columns cover the derivatives' domain (x_L ≤ 100) for the
+//! the exact values. The derivative columns cover the derivatives' domain (x_L ≤ 1000) for the
 //! particles of at most four layers (the ten-layer particle takes other passes of the same code,
 //! which tests/layered.rs checks against differences); above it they are null.
 //!
@@ -337,7 +337,7 @@ fn accuracy_map_is_within_the_declared_tolerances() {
             let mut v: Vec<(f64, usize)> = results
                 .iter()
                 .enumerate()
-                .filter(|&(k, _)| inside(&points[k]) && points[k].0 <= 100.0)
+                .filter(|&(k, _)| inside(&points[k]))
                 .map(|(k, c)| (c[o], k))
                 .collect();
             v.sort_by(|a, b| b.0.total_cmp(&a.0));
diff --git a/scatrs/tests/layered_greek_accuracy_map.rs b/scatrs/tests/layered_greek_accuracy_map.rs
index 12613cb..786788a 100644
--- a/scatrs/tests/layered_greek_accuracy_map.rs
+++ b/scatrs/tests/layered_greek_accuracy_map.rs
@@ -6,7 +6,7 @@
 //! against `PARTICLES`, with the phase matrix at the cosines of the sphere's Greek map,
 //! normalized, and the Greek coefficients up to all orders (2N) for x_L ≤ 160 and order 100
 //! above, as the sphere's Greek map (`tests/mie_greek_accuracy_map.rs`) has them. The derivative
-//! columns cover the derivatives' domain (x_L ≤ 100). The measures are the sphere's, with the
+//! columns cover the derivatives' domain (x_L ≤ 1000). The measures are the sphere's, with the
 //! condition numbers summed over every layer's x_l, Re m_l and Im m_l (`tests/common/layered.rs`).
 //!
 //! With `SCATRS_ACCURACY_MAP=path` the test writes every C as JSON:
diff --git a/tests/test_layered.py b/tests/test_layered.py
index 3ced228..f328ebb 100644
--- a/tests/test_layered.py
+++ b/tests/test_layered.py
@@ -119,10 +119,10 @@ def test_errors():
     with pytest.raises(OutsideValidatedDomainError):
         layered.efficiencies([1.0, 2e4], [1.5, 1.33])
     assert layered.efficiencies([1.0, 2e4], [1.5, 1.33], allow_unvalidated=True).q_ext > 0
-    # Derivatives are validated up to an outer size parameter of 100.
+    # Derivatives are validated up to an outer size parameter of 1000.
     with pytest.raises(OutsideValidatedDomainError, match="derivatives"):
-        layered.efficiencies([50.0, 200.0], [1.5, 1.33], derivatives=True)
-    assert layered.efficiencies([50.0, 200.0], [1.5, 1.33]).q_ext > 0
+        layered.efficiencies([500.0, 2000.0], [1.5, 1.33], derivatives=True)
+    assert layered.efficiencies([500.0, 2000.0], [1.5, 1.33]).q_ext > 0
 
 
 def test_default_n_max_is_the_spheres_at_the_outer_size():
````

### Task 2: Layered spheres in the ensemble engine

- [ ] Add `Particle::LayeredSphere`, its constructor and validation, and `ParamKind::{LayerRefrReal, LayerRefrImag, RadiusFraction}` with their names.
- [ ] Add `Optics::Layered` (borrowing the particle) and the `Solver` arm, with `Solved::{direction, size_derivative}`. Keep the sphere's arithmetic bitwise: the fraction loops are empty for a sphere.
- [ ] Add real columns to `PanelProducts` (`new`/`fits` take `groups, reals`).
- [ ] Add `compute_with_particle`, `compute_batch_with_particle` and `parameters_with_particle`. The sphere functions delegate to them.
- [ ] Add the tests (`scatrs/tests/ensemble_layered.rs`). Mutation-check the fraction scale in `Solved::direction` and in `PanelProducts::set`.
- [ ] Run the snapshot: bitwise.

The commit (`7739c57`):

````diff
diff --git a/scatrs/src/ensemble/engine.rs b/scatrs/src/ensemble/engine.rs
index cecd81c..2225d1c 100644
--- a/scatrs/src/ensemble/engine.rs
+++ b/scatrs/src/ensemble/engine.rs
@@ -18,8 +18,8 @@ use super::{
     PhaseValues, kronrod,
 };
 use crate::mie::{
-    Coefficients, EfficiencyJacobian, GreekCoefficients, GreekProjector, MAX_SIZE_PARAMETER,
-    MieOptions, Normalization, PhaseBuffers,
+    Coefficients, GreekCoefficients, GreekProjector, MAX_SIZE_PARAMETER, MieOptions, Normalization,
+    PhaseBuffers,
 };
 use crate::{Error, Result};
 
@@ -93,12 +93,6 @@ impl Part<'_> {
     }
 }
 
-/// The derivative of the efficiencies along direction `k`: Re m or Im m of group k/2.
-fn direction(jacs: &[EfficiencyJacobian<f64>], k: usize) -> &crate::mie::Efficiencies<f64> {
-    let j = &jacs[k / 2];
-    if k % 2 == 0 { &j.m_re } else { &j.m_im }
-}
-
 fn context(e: Error, lambda: f64, r: f64) -> Error {
     let at = format!("at wavelength {lambda:e}, radius {r:e}");
     match e {
@@ -147,6 +141,11 @@ fn parts<'a>(
                     }
                     ParamKind::RefrReal => dir_cols[0] = Some(col),
                     ParamKind::RefrImag => dir_cols[1] = Some(col),
+                    ParamKind::LayerRefrReal(l) => dir_cols[2 * l] = Some(col),
+                    ParamKind::LayerRefrImag(l) => dir_cols[2 * l + 1] = Some(col),
+                    ParamKind::RadiusFraction(l) => {
+                        dir_cols[2 * c.particle.groups() + l] = Some(col);
+                    }
                     _ => {}
                 }
             }
@@ -352,7 +351,7 @@ fn quantities1(
     solver: &mut Solver,
     r: f64,
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     opts: &MieOptions,
     refine_order: Option<usize>,
     dirs: &[bool],
@@ -361,7 +360,7 @@ fn quantities1(
     let s = solver
         .solve(optics, x, opts, dirs.iter().any(|&b| b))
         .map_err(|e| context(e, lambda, r))?;
-    let c = s.c;
+    let c = &s.c;
     let q = c.efficiencies();
     let area = PI * r * r;
     let mut out = vec![
@@ -388,7 +387,7 @@ fn quantities1(
         .map(|g| c.efficiencies_jacobian(g))
         .collect();
     for (k, _) in dirs.iter().enumerate().filter(|&(_, &want)| want) {
-        let d = direction(&jacs, k);
+        let d = s.direction(&jacs, k);
         out.extend([
             area * d.q_ext,
             area * d.q_sca,
@@ -415,7 +414,7 @@ fn panel1(
     group: &[&Part<'_>],
     (a, b): (f64, f64),
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     opts: &MieOptions,
     refine_order: Option<usize>,
     n_refined: usize,
@@ -505,7 +504,7 @@ fn evaluate(
     panels: &mut Panels1,
     group: &[&Part<'_>],
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     opts: &MieOptions,
     refine_order: Option<usize>,
     n_refined: usize,
@@ -650,7 +649,7 @@ fn status(
 fn pass1(
     group: &[&Part<'_>],
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     options: &EnsembleOptions,
     mie_opts: &MieOptions,
 ) -> Result<Pass1> {
@@ -878,7 +877,13 @@ fn bucket(n: usize) -> usize {
 
 /// The largest number of terms at the nodes of the panel [a, b] (t = ln r), which chooses the
 /// panel's band.
-fn panel_terms(a: f64, b: f64, lambda: f64, optics: Optics, opts: &MieOptions) -> Result<usize> {
+fn panel_terms(
+    a: f64,
+    b: f64,
+    lambda: f64,
+    optics: Optics<'_>,
+    opts: &MieOptions,
+) -> Result<usize> {
     let mut most = 0;
     for &(t, _, _) in &kronrod::panel(a, b) {
         let r = t.exp();
@@ -932,7 +937,7 @@ fn quantities2(
     solver: &mut Solver,
     r: f64,
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     opts: &MieOptions,
     layout: &Layout<'_>,
     g: &mut [f64],
@@ -946,7 +951,7 @@ fn quantities2(
     let s = solver
         .solve(optics, x, opts, derivatives)
         .map_err(|e| context(e, lambda, r))?;
-    let c = s.c;
+    let c = &s.c;
     let area = PI * r * r;
     let q = c.efficiencies();
     g[..SCALARS].copy_from_slice(&[
@@ -973,7 +978,7 @@ fn quantities2(
     }
     let mut tables = vec![(SCALARS, layout.mu)];
     match products {
-        Some((p, k)) => p.set(k, &c, &s.groups),
+        Some((p, k)) => p.set(k, c, &s.groups),
         None => tables.extend(band),
     }
     let segment = band.map(|(start, nodes)| (start, 4 * nodes.len()));
@@ -983,7 +988,7 @@ fn quantities2(
             if n == 0 {
                 continue;
             }
-            phase_into(&c, cosines, &mut g[start..start + 4 * n])
+            phase_into(c, cosines, &mut g[start..start + 4 * n])
                 .map_err(|e| context(e, lambda, r))?;
             g[start..start + 4 * n].iter_mut().for_each(|v| *v *= area);
         }
@@ -1009,13 +1014,13 @@ fn quantities2(
     let k = 2.0 * PI / lambda;
     if let Some(dm) = dm.as_deref_mut() {
         for (dir, d) in dm.iter_mut().enumerate() {
-            d[..SCALARS].copy_from_slice(&scalar_d(direction(&jacs, dir)));
+            d[..SCALARS].copy_from_slice(&scalar_d(&s.direction(&jacs, dir)));
         }
     }
     let mut dr = dr;
     if let Some(dr) = dr.as_deref_mut() {
-        // ∂(πr²Q)/∂r = 2πr Q + πr² (2π/λ) ∂Q/∂x.
-        let dx = scalar_d(&jacs[0].x);
+        // ∂(πr²Q)/∂r = 2πr Q + πr² (2π/λ) ∂Q/∂x, at fixed radius fractions.
+        let dx = scalar_d(&s.size_derivative(&jacs));
         let values = [0.0, q.q_ext, q.q_sca, q.q_sca * q.g, q.q_abs, q.q_back];
         for j in 1..6 {
             dr[j] = 2.0 * PI * r * values[j] + k * dx[j];
@@ -1075,15 +1080,27 @@ fn quantities2(
             .map_err(|e| context(e, lambda, r))?;
         }
         if let Some(dr) = dr.as_deref_mut() {
+            // At fixed radius fractions: Σ_l f_l·∂/∂x_l, from the outer group.
+            let (outer, inner) = d.split_last().expect("a group");
             for (j, o) in span.clone().enumerate() {
-                dr[o] = 2.0 * PI * r * g[o] + area * k * d[0][0][j];
+                let mut dx = outer[0][j];
+                for (f, di) in s.fractions.iter().zip(inner) {
+                    dx += f * di[0][j];
+                }
+                dr[o] = 2.0 * PI * r * g[o] + area * k * dx;
             }
         }
         if let Some(dm) = dm.as_deref_mut() {
+            let groups = d.len();
             for (dir, out) in dm.iter_mut().enumerate() {
-                let buf = &d[dir / 2][1 + dir % 2];
+                // Re m, Im m of each group, then each radius fraction: x_L·∂/∂x_l.
+                let (buf, scale) = if dir < 2 * groups {
+                    (&d[dir / 2][1 + dir % 2], area)
+                } else {
+                    (&d[dir - 2 * groups][0], area * c.x)
+                };
                 for (j, o) in span.clone().enumerate() {
-                    out[o] = area * buf[j];
+                    out[o] = scale * buf[j];
                 }
             }
         }
@@ -1178,7 +1195,7 @@ fn pass2(
     group: &[&Part<'_>],
     panels: &[(f64, f64)],
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     opts: &MieOptions,
     layout: &Layout<'_>,
 ) -> Result<Vec<Sums2>> {
@@ -1186,12 +1203,14 @@ fn pass2(
     let common = layout.common();
     let blocks: Vec<usize> = group.iter().map(|p| blocks(p)).collect();
     let derivatives = group.iter().any(|p| p.dir_derivatives());
-    // Every direction of the particle when any is requested.
-    let n_dirs = if derivatives {
-        group[0].dir_cols.len()
+    // Every direction of the particle when any is requested: two per group, then the real
+    // ones.
+    let (n_groups, n_reals) = if derivatives {
+        (group[0].particle.groups(), group[0].particle.reals())
     } else {
-        0
+        (0, 0)
     };
+    let n_dirs = 2 * n_groups + n_reals;
     let rows = Rows::new(group);
     let n_nodes = kronrod::panel(0.0, 1.0).len();
     // Per-thread scratch. A panel touches the common part of the layout and its band's
@@ -1272,13 +1291,16 @@ fn pass2(
                         };
                         let nodes = kronrod::panel(a, b);
                         if let Some(band) = band {
-                            let fits = sc
-                                .products
-                                .as_ref()
-                                .is_some_and(|p| p.fits(&band.tables, nodes.len(), n_dirs / 2));
+                            let fits = sc.products.as_ref().is_some_and(|p| {
+                                p.fits(&band.tables, nodes.len(), n_groups, n_reals)
+                            });
                             if !fits {
-                                sc.products =
-                                    Some(PanelProducts::new(&band.tables, nodes.len(), n_dirs / 2));
+                                sc.products = Some(PanelProducts::new(
+                                    &band.tables,
+                                    nodes.len(),
+                                    n_groups,
+                                    n_reals,
+                                ));
                             }
                         }
                         let (lo, hi) = band.map_or((0, 0), |b| {
@@ -1502,7 +1524,7 @@ fn monodisperse(
     part: &Part<'_>,
     r: f64,
     lambda: f64,
-    optics: Optics,
+    optics: Optics<'_>,
     opts: &MieOptions,
     layout: &Layout<'_>,
 ) -> Result<PartAverages> {
diff --git a/scatrs/src/ensemble/fast.rs b/scatrs/src/ensemble/fast.rs
index 073828f..2d35cb2 100644
--- a/scatrs/src/ensemble/fast.rs
+++ b/scatrs/src/ensemble/fast.rs
@@ -49,43 +49,51 @@ impl BandTables {
 }
 
 /// The right-hand sides of a panel's spheres (4 columns each: u and v, real and imaginary
-/// parts; 4 more per derivative group: du/dm, dv/dm) and the products.
+/// parts; 4 more per derivative group: du/dm, dv/dm; and 4 more per real direction: x·du/dx_j,
+/// x·dv/dx_j of group j, the derivatives with respect to a radius fraction) and the products.
 pub(crate) struct PanelProducts {
     per: usize,
+    groups: usize,
+    reals: usize,
     x: Mat,
     ce: Mat,
     co: Mat,
 }
 
 impl PanelProducts {
-    /// Room for `spheres` spheres with `groups` derivative groups each, at the band's orders
-    /// and half nodes.
-    pub(crate) fn new(tables: &BandTables, spheres: usize, groups: usize) -> Self {
-        let per = 4 + 4 * groups;
+    /// Room for `spheres` spheres with `groups` derivative groups and `reals` real directions
+    /// each, at the band's orders and half nodes.
+    pub(crate) fn new(tables: &BandTables, spheres: usize, groups: usize, reals: usize) -> Self {
+        let per = 4 + 4 * (groups + reals);
         let (h, n) = (tables.ge.nrows(), tables.orders());
         Self {
             per,
+            groups,
+            reals,
             x: Mat::zeros(n, per * spheres),
             ce: Mat::zeros(h, per * spheres),
             co: Mat::zeros(h, per * spheres),
         }
     }
 
-    /// Whether this fits `tables` and `spheres` spheres with `groups` derivative groups.
-    pub(crate) fn fits(&self, tables: &BandTables, spheres: usize, groups: usize) -> bool {
+    /// Whether this fits `tables` and `spheres` spheres with `groups` derivative groups and
+    /// `reals` real directions.
+    pub(crate) fn fits(
+        &self,
+        tables: &BandTables,
+        spheres: usize,
+        groups: usize,
+        reals: usize,
+    ) -> bool {
         self.x.nrows() == tables.orders()
             && self.ce.nrows() == tables.ge.nrows()
-            && self.per == 4 + 4 * groups
+            && (self.groups, self.reals) == (groups, reals)
             && self.x.ncols() == self.per * spheres
     }
 
-    /// The number of derivative groups.
-    fn groups(&self) -> usize {
-        (self.per - 4) / 4
-    }
-
-    /// Sphere `k`'s coefficients (zero beyond its number of terms) and daₙ/dm, dbₙ/dm of each
-    /// of the first `self.groups()` of `groups`.
+    /// Sphere `k`'s coefficients (zero beyond its number of terms), daₙ/dm, dbₙ/dm of each of
+    /// the first `self.groups` of `groups`, and x·daₙ/dx_j, x·dbₙ/dx_j of the first
+    /// `self.reals` (x = `c.x`, the outer size parameter).
     pub(crate) fn set(
         &mut self,
         k: usize,
@@ -93,7 +101,8 @@ impl PanelProducts {
         groups: &[Group<'_, f64>],
     ) {
         let (a, b) = (c.a, c.b);
-        let groups = &groups[..self.groups()];
+        let reals = &groups[..self.reals];
+        let groups = &groups[..self.groups];
         let base = k * self.per;
         let n = self.x.nrows();
         let mut put = |row: usize, col: usize, u: Complex<f64>, v: Complex<f64>| {
@@ -114,8 +123,12 @@ impl PanelProducts {
                     let (du, dv) = uv(row, g.d.da_dm[row], g.d.db_dm[row]);
                     put(row, base + 4 + 4 * i, du, dv);
                 }
+                for (j, g) in reals.iter().enumerate() {
+                    let (du, dv) = uv(row, g.d.da_dx[row] * c.x, g.d.db_dx[row] * c.x);
+                    put(row, base + 4 + 4 * (groups.len() + j), du, dv);
+                }
             } else {
-                for i in 0..=groups.len() {
+                for i in 0..=groups.len() + reals.len() {
                     put(row, base + 4 * i, Complex::default(), Complex::default());
                 }
             }
@@ -131,8 +144,9 @@ impl PanelProducts {
     }
 
     /// Sphere `k`'s phase matrix times `scale` at every node of the band (F₁₁, F₁₂, F₃₃, F₃₄,
-    /// each n_q values, into `out`), and its derivatives with respect to Re m and Im m of each
-    /// group into `d` (Re m, Im m of the first group, then of the next; empty for none).
+    /// each n_q values, into `out`), and its derivatives into `d`: with respect to Re m and
+    /// Im m of each group (Re m, Im m of the first group, then of the next), then along each
+    /// real direction (empty for none).
     pub(crate) fn phase(
         &self,
         tables: &BandTables,
@@ -156,11 +170,15 @@ impl PanelProducts {
                 for (e, v) in f.into_iter().enumerate() {
                     out[e * q + node] = scale * v;
                 }
-                for (g, pair) in d.chunks_exact_mut(2).enumerate() {
-                    let col = base + 4 + 4 * g;
+                // dS₁, dS₂ along the derivative columns `col`.
+                let ds = |col: usize| {
                     let (deu, dev) = (c(&self.ce, i, col), c(&self.ce, i, col + 2));
                     let (dou, dov) = (c(&self.co, i, col), c(&self.co, i, col + 2));
-                    let (d1, d2) = (deu + dov * sign, dev + dou * sign);
+                    (deu + dov * sign, dev + dou * sign)
+                };
+                let (pairs, reals) = d.split_at_mut(2 * self.groups);
+                for (g, pair) in pairs.chunks_exact_mut(2).enumerate() {
+                    let (d1, d2) = ds(base + 4 + 4 * g);
                     let fr = element_derivatives(s1, s2, d1, d2);
                     let fi = element_derivatives(s1, s2, i1 * d1, i1 * d2);
                     let [dre, dim] = pair else { unreachable!() };
@@ -169,6 +187,13 @@ impl PanelProducts {
                         dim[e * q + node] = scale * fi[e];
                     }
                 }
+                for (j, dx) in reals.iter_mut().enumerate() {
+                    let (d1, d2) = ds(base + 4 + 4 * (self.groups + j));
+                    let fx = element_derivatives(s1, s2, d1, d2);
+                    for e in 0..4 {
+                        dx[e * q + node] = scale * fx[e];
+                    }
+                }
             };
             write(j, 1.0);
             let mirror = q - 1 - j;
@@ -279,7 +304,7 @@ mod tests {
             let nodes = gauss_legendre::<f64>(q).unwrap().nodes;
             let tables = BandTables::new(&nodes, n_max);
             for derivatives in [false, true] {
-                let mut p = PanelProducts::new(&tables, cases.len(), usize::from(derivatives));
+                let mut p = PanelProducts::new(&tables, cases.len(), usize::from(derivatives), 0);
                 for (k, &(x, m)) in cases.iter().enumerate() {
                     let s = if derivatives {
                         mie.compute_with_derivatives(x, m, &opts).unwrap()
diff --git a/scatrs/src/ensemble/mod.rs b/scatrs/src/ensemble/mod.rs
index 2ac2a79..def38a5 100644
--- a/scatrs/src/ensemble/mod.rs
+++ b/scatrs/src/ensemble/mod.rs
@@ -1,10 +1,11 @@
-//! Ensembles of spheres: size distributions, adaptive size integration and external
+//! Ensembles of particles: size distributions, adaptive size integration and external
 //! mixtures.
 //!
 //! [`compute`] averages the optical properties of spheres over a [`SizeDistribution`] at each
-//! wavelength; [`compute_mixture`] averages over several [`Component`]s, and [`compute_batch`]
-//! over each of many distributions of one material (an optics database), sharing their Mie
-//! solutions. The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the
+//! wavelength, and [`compute_with_particle`] those of any [`Particle`] (a layered sphere's size
+//! is its outer radius, with each interface at a fixed fraction of it); [`compute_mixture`]
+//! averages over several [`Component`]s, and [`compute_batch`] over each of many distributions
+//! of one material (an optics database), sharing their solutions. The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the
 //! cross sections and g meet `rtol`; the distribution's moments and normalization are
 //! integrated separately), and every output comes with an error estimate and, on request,
 //! derivatives with respect to the distribution parameters, the refractive indices and the
@@ -40,7 +41,8 @@ pub use distribution::SizeDistribution;
 use crate::mie::{GreekCoefficients, MAX_GREEK_ORDER, Normalization};
 use crate::{Error, Result};
 
-/// The particles of a component, given at each wavelength.
+/// The particles of a component, given at each wavelength. A size distribution describes the
+/// radius the particle defines: the radius of a sphere, the outer radius of a layered sphere.
 #[derive(Clone, Debug, PartialEq)]
 #[non_exhaustive]
 pub enum Particle {
@@ -49,6 +51,16 @@ pub enum Particle {
         /// The refractive index at each wavelength.
         refractive_index: Vec<Complex<f64>>,
     },
+    /// Layered spheres: a core and concentric shells ([`crate::layered`]), with the outer radius
+    /// of each layer but the last a fixed fraction of the particle's.
+    LayeredSphere {
+        /// The outer radius of layers 0, …, L − 2 (the core first) as fractions of the
+        /// particle's radius, nondecreasing in (0, 1]: L − 1 values.
+        radius_fractions: Vec<f64>,
+        /// The refractive index of each layer (the core first) at each wavelength:
+        /// `refractive_index[l][i]` is layer l's at wavelength i.
+        refractive_index: Vec<Vec<Complex<f64>>>,
+    },
 }
 
 impl Particle {
@@ -56,6 +68,30 @@ impl Particle {
     pub fn sphere(refractive_index: Vec<Complex<f64>>) -> Self {
         Particle::Sphere { refractive_index }
     }
+
+    /// Layered spheres of L layers: layer l's outer radius is `radius_fractions[l]` times the
+    /// particle's (l < L − 1), and its refractive index `refractive_index[l][i]` at wavelength
+    /// i.
+    ///
+    /// ```
+    /// use num_complex::Complex;
+    /// use scatrs::ensemble::Particle;
+    ///
+    /// // Black carbon (core radius 0.6 of the particle's) coated in sulfate, at one wavelength.
+    /// let coated = Particle::layered_sphere(
+    ///     vec![0.6],
+    ///     vec![vec![Complex::new(1.95, 0.79)], vec![Complex::new(1.43, 1e-8)]],
+    /// );
+    /// ```
+    pub fn layered_sphere(
+        radius_fractions: Vec<f64>,
+        refractive_index: Vec<Vec<Complex<f64>>>,
+    ) -> Self {
+        Particle::LayeredSphere {
+            radius_fractions,
+            refractive_index,
+        }
+    }
 }
 
 /// One component of an external mixture: particles of one size distribution, in relative
@@ -123,6 +159,14 @@ pub enum ParamKind {
     RefrReal,
     /// The imaginary part of the component's refractive index (at each wavelength).
     RefrImag,
+    /// The real part of layer l's refractive index (at each wavelength) of a layered sphere.
+    LayerRefrReal(usize),
+    /// The imaginary part of layer l's refractive index (at each wavelength) of a layered
+    /// sphere.
+    LayerRefrImag(usize),
+    /// The radius fraction of layer l's outer surface of a layered sphere (l < L − 1), at a
+    /// fixed outer radius.
+    RadiusFraction(usize),
     /// The weight of a mixture component (`mode` = `None`) or of a mode of a multimodal
     /// distribution.
     Weight,
@@ -150,14 +194,18 @@ impl EnsembleParam {
         }
     }
 
-    /// Its name: `median_radius`, `m_real`, `m_imag`, `weight`, …, prefixed by `mode{j}.`
-    /// for a mode and, with `mixture`, by `c{i}.` for the component.
+    /// Its name: `median_radius`, `m_real`, `m_imag`, `layer{l}.m_real`, `layer{l}.m_imag`,
+    /// `layer{l}.radius_fraction`, `weight`, …, prefixed by `mode{j}.` for a mode and, with
+    /// `mixture`, by `c{i}.` for the component.
     pub fn name(&self, mixture: bool) -> String {
         let base = match self.kind {
-            ParamKind::Distribution(name) => name,
-            ParamKind::RefrReal => "m_real",
-            ParamKind::RefrImag => "m_imag",
-            ParamKind::Weight => "weight",
+            ParamKind::Distribution(name) => name.to_owned(),
+            ParamKind::RefrReal => "m_real".to_owned(),
+            ParamKind::RefrImag => "m_imag".to_owned(),
+            ParamKind::LayerRefrReal(l) => format!("layer{l}.m_real"),
+            ParamKind::LayerRefrImag(l) => format!("layer{l}.m_imag"),
+            ParamKind::RadiusFraction(l) => format!("layer{l}.radius_fraction"),
+            ParamKind::Weight => "weight".to_owned(),
         };
         let mut out = String::new();
         if mixture {
@@ -166,7 +214,7 @@ impl EnsembleParam {
         if let Some(j) = self.mode {
             out += &format!("mode{j}.");
         }
-        out + base
+        out + &base
     }
 }
 
@@ -204,6 +252,22 @@ fn component_parameters(
             out.push(EnsembleParam::new(index, None, ParamKind::RefrReal));
             out.push(EnsembleParam::new(index, None, ParamKind::RefrImag));
         }
+        Particle::LayeredSphere {
+            radius_fractions,
+            refractive_index,
+        } => {
+            for l in 0..refractive_index.len() {
+                out.push(EnsembleParam::new(index, None, ParamKind::LayerRefrReal(l)));
+                out.push(EnsembleParam::new(index, None, ParamKind::LayerRefrImag(l)));
+            }
+            for l in 0..radius_fractions.len() {
+                out.push(EnsembleParam::new(
+                    index,
+                    None,
+                    ParamKind::RadiusFraction(l),
+                ));
+            }
+        }
     }
     if weight {
         out.push(EnsembleParam::new(index, None, ParamKind::Weight));
@@ -211,9 +275,20 @@ fn component_parameters(
     out
 }
 
-/// Every parameter of a single distribution, for [`compute`].
+/// Every parameter of a single distribution of spheres, for [`compute`].
 pub fn parameters(distribution: &SizeDistribution) -> Vec<EnsembleParam> {
-    component_parameters(0, distribution, &Particle::sphere(Vec::new()), false)
+    parameters_with_particle(distribution, &Particle::sphere(Vec::new()))
+}
+
+/// Every parameter of a single distribution of `particle`s, for [`compute_with_particle`]: the
+/// distribution's, then the particle's (`m_real`, `m_imag` for a sphere; for a layered sphere
+/// `layer{l}.m_real` and `layer{l}.m_imag` of each layer, then `layer{l}.radius_fraction` of
+/// each but the last).
+pub fn parameters_with_particle(
+    distribution: &SizeDistribution,
+    particle: &Particle,
+) -> Vec<EnsembleParam> {
+    component_parameters(0, distribution, particle, false)
 }
 
 /// Every parameter of a mixture, for [`compute_mixture`].
@@ -436,8 +511,49 @@ pub fn compute(
     refractive_index: &[Complex<f64>],
     options: &EnsembleOptions,
 ) -> Result<Vec<EnsembleResult>> {
-    let component = Component::new(distribution.clone(), refractive_index.to_vec(), 1.0);
-    let allowed = parameters(distribution);
+    let particle = Particle::sphere(refractive_index.to_vec());
+    compute_with_particle(distribution, wavelengths, &particle, options)
+}
+
+/// Averages over `distribution` of `particle`s at `wavelengths` (in the unit of the radii), as
+/// [`compute`] does for spheres. For a layered sphere the distribution gives the outer radius.
+///
+/// ```
+/// use num_complex::Complex;
+/// use scatrs::ensemble::{
+///     EnsembleOptions, Particle, SizeDistribution, compute_with_particle,
+///     parameters_with_particle,
+/// };
+///
+/// let size = SizeDistribution::LogNormal { median_radius: 0.1, width: 1.5 };
+/// let coated = Particle::layered_sphere(
+///     vec![0.6],
+///     vec![vec![Complex::new(1.95, 0.79)], vec![Complex::new(1.43, 1e-8)]],
+/// );
+/// // ∂/∂(core fraction), the last parameter.
+/// let fraction = *parameters_with_particle(&size, &coated).last().unwrap();
+/// let options = EnsembleOptions::new().with_derivatives(vec![fraction]);
+/// let r = &compute_with_particle(&size, &[0.55], &coated, &options)?[0];
+/// assert!(r.derivatives[0].c_abs > 0.0); // a larger core absorbs more
+/// # Ok::<(), scatrs::Error>(())
+/// ```
+///
+/// # Errors
+///
+/// As [`compute`], and [`Error::InvalidInput`] for a layered sphere without a layer, with
+/// more than [`crate::layered::MAX_LAYERS`], with fractions that are not nondecreasing in
+/// (0, 1] or not one fewer than the layers, or with a layer's refractive indices not one per
+/// wavelength. With derivatives along the particle's parameters, a layered sphere's size nodes
+/// must be within its derivatives' validated domain
+/// ([`crate::layered::ValidatedDomain::derivative_x_max`]) unless `allow_unvalidated`.
+pub fn compute_with_particle(
+    distribution: &SizeDistribution,
+    wavelengths: &[f64],
+    particle: &Particle,
+    options: &EnsembleOptions,
+) -> Result<Vec<EnsembleResult>> {
+    let component = Component::with_particle(distribution.clone(), particle.clone(), 1.0);
+    let allowed = parameters_with_particle(distribution, particle);
     validate(
         core::slice::from_ref(&component),
         wavelengths,
@@ -485,6 +601,21 @@ pub fn compute_batch(
     wavelengths: &[f64],
     refractive_index: &[Complex<f64>],
     options: &EnsembleOptions,
+) -> Result<Vec<Vec<EnsembleResult>>> {
+    let particle = Particle::sphere(refractive_index.to_vec());
+    compute_batch_with_particle(distributions, wavelengths, &particle, options)
+}
+
+/// As [`compute_batch`], for `particle`s.
+///
+/// # Errors
+///
+/// As [`compute_batch`] and [`compute_with_particle`].
+pub fn compute_batch_with_particle(
+    distributions: &[SizeDistribution],
+    wavelengths: &[f64],
+    particle: &Particle,
+    options: &EnsembleOptions,
 ) -> Result<Vec<Vec<EnsembleResult>>> {
     if distributions.is_empty() {
         return Err(invalid("a batch needs a distribution".into()));
@@ -494,10 +625,11 @@ pub fn compute_batch(
             "a batch computes values only: request derivatives from compute".into(),
         ));
     }
-    if refractive_index.len() != wavelengths.len() {
+    particle.validate().map_err(invalid)?;
+    if particle.wavelengths() != wavelengths.len() {
         return Err(invalid(format!(
             "{} refractive indices for {} wavelengths",
-            refractive_index.len(),
+            particle.wavelengths(),
             wavelengths.len()
         )));
     }
@@ -507,7 +639,7 @@ pub fn compute_batch(
     }
     let components: Vec<Component> = distributions
         .iter()
-        .map(|d| Component::new(d.clone(), refractive_index.to_vec(), 1.0))
+        .map(|d| Component::with_particle(d.clone(), particle.clone(), 1.0))
         .collect();
     validate(&components, wavelengths, options, &[])?;
     engine::run_batch(&components, wavelengths, options)
@@ -575,6 +707,9 @@ fn validate(
                 )));
             }
         }
+        c.particle
+            .validate()
+            .map_err(|e| invalid(format!("component {i}: {e}")))?;
         if c.particle.wavelengths() != wavelengths.len() {
             return Err(invalid(format!(
                 "component {i} has {} refractive indices for {} wavelengths",
diff --git a/scatrs/src/ensemble/particle.rs b/scatrs/src/ensemble/particle.rs
index 61d3dfd..4bb7f0e 100644
--- a/scatrs/src/ensemble/particle.rs
+++ b/scatrs/src/ensemble/particle.rs
@@ -3,79 +3,240 @@
 //!
 //! The per-sphere quantities are differentiated along the particle's directions: the real and
 //! imaginary parts of the refractive index of each group, in order (Re m and Im m for a
-//! sphere).
+//! sphere; of each layer for a layered sphere), then the particle's real parameters (the radius
+//! fraction of each layer of a layered sphere but the last).
 
 use num_complex::Complex;
 
 use super::Particle;
 use crate::Result;
-use crate::mie::{Coefficients, Group, Mie, MieOptions, n_terms};
+use crate::layered::{self, Layered};
+use crate::mie::{Coefficients, Efficiencies, EfficiencyJacobian, Group, Mie, MieOptions, n_terms};
 
 impl Particle {
     /// The particle at wavelength `i`.
-    pub(super) fn optics(&self, i: usize) -> Optics {
+    pub(super) fn optics(&self, i: usize) -> Optics<'_> {
         match self {
             Particle::Sphere { refractive_index } => Optics::Sphere(refractive_index[i]),
+            Particle::LayeredSphere {
+                radius_fractions,
+                refractive_index,
+            } => Optics::Layered {
+                fractions: radius_fractions,
+                m: refractive_index,
+                i,
+            },
         }
     }
 
-    /// The number of wavelengths it is given at.
+    /// The number of wavelengths it is given at (of the core, for a layered sphere; see
+    /// [`validate`](Self::validate)).
     pub(super) fn wavelengths(&self) -> usize {
         match self {
             Particle::Sphere { refractive_index } => refractive_index.len(),
+            Particle::LayeredSphere {
+                refractive_index, ..
+            } => refractive_index.first().map_or(0, Vec::len),
         }
     }
 
-    /// The number of derivative directions: two per group.
-    pub(super) fn directions(&self) -> usize {
+    /// Checks the particle's own description: a layered sphere's layers and fractions.
+    pub(super) fn validate(&self) -> core::result::Result<(), String> {
+        let Particle::LayeredSphere {
+            radius_fractions: f,
+            refractive_index: m,
+        } = self
+        else {
+            return Ok(());
+        };
+        let layers = m.len();
+        if layers == 0 || layers > layered::MAX_LAYERS {
+            return Err(format!(
+                "a layered sphere needs 1 to {} layers, got {layers}",
+                layered::MAX_LAYERS
+            ));
+        }
+        if f.len() + 1 != layers {
+            return Err(format!(
+                "a layered sphere of {layers} layers needs {} radius fractions, got {}",
+                layers - 1,
+                f.len()
+            ));
+        }
+        if let Some((l, v)) = f
+            .iter()
+            .enumerate()
+            .find(|&(_, v)| !(v.is_finite() && *v > 0.0 && *v <= 1.0))
+        {
+            return Err(format!("radius fraction {l} must be in (0, 1], got {v}"));
+        }
+        if let Some(l) = (1..f.len()).find(|&l| f[l] < f[l - 1]) {
+            return Err(format!(
+                "the radius fractions must be nondecreasing from the core out, but fraction {l} \
+                 is {} < {}",
+                f[l],
+                f[l - 1]
+            ));
+        }
+        if let Some(l) = (1..layers).find(|&l| m[l].len() != m[0].len()) {
+            return Err(format!(
+                "layer {l} has {} refractive indices and the core {}: one per wavelength",
+                m[l].len(),
+                m[0].len()
+            ));
+        }
+        Ok(())
+    }
+
+    /// The number of derivative groups (a size parameter and a refractive index each).
+    pub(super) fn groups(&self) -> usize {
+        match self {
+            Particle::Sphere { .. } => 1,
+            Particle::LayeredSphere {
+                refractive_index, ..
+            } => refractive_index.len(),
+        }
+    }
+
+    /// The number of real directions after the groups' (the radius fractions).
+    pub(super) fn reals(&self) -> usize {
         match self {
-            Particle::Sphere { .. } => 2,
+            Particle::Sphere { .. } => 0,
+            Particle::LayeredSphere {
+                radius_fractions, ..
+            } => radius_fractions.len(),
         }
     }
+
+    /// The number of derivative directions: two per group, then the real ones.
+    pub(super) fn directions(&self) -> usize {
+        2 * self.groups() + self.reals()
+    }
 }
 
 /// A component's particle at one wavelength.
 #[derive(Clone, Copy, Debug)]
-pub(super) enum Optics {
+pub(super) enum Optics<'a> {
     /// A homogeneous sphere of refractive index m.
     Sphere(Complex<f64>),
+    /// A layered sphere: its radius fractions, and its layers' refractive indices at
+    /// wavelength `i` (`m[l][i]`).
+    Layered {
+        fractions: &'a [f64],
+        m: &'a [Vec<Complex<f64>>],
+        i: usize,
+    },
 }
 
-impl Optics {
+impl Optics<'_> {
     /// The number of terms of the particle of size parameter `x`, after the input checks of
     /// [`Solver::solve`].
     pub(super) fn n_terms(self, x: f64, opts: &MieOptions) -> Result<usize> {
         match self {
             Optics::Sphere(m) => n_terms(x, m, opts),
+            Optics::Layered { .. } => {
+                let (mut xs, mut ms) = (Vec::new(), Vec::new());
+                self.layers(x, &mut xs, &mut ms);
+                layered::n_terms(&xs, &ms, opts)
+            }
         }
     }
+
+    /// A layered sphere's interface size parameters at outer size parameter `x` (x_l = f_l·x,
+    /// and x itself last) and its layers' indices, into `xs` and `ms`.
+    fn layers(self, x: f64, xs: &mut Vec<f64>, ms: &mut Vec<Complex<f64>>) {
+        let Optics::Layered { fractions, m, i } = self else {
+            unreachable!("a layered sphere")
+        };
+        xs.clear();
+        xs.extend(fractions.iter().map(|f| f * x));
+        xs.push(x);
+        ms.clear();
+        ms.extend(m.iter().map(|layer| layer[i]));
+    }
 }
 
-/// One particle's solution: its coefficients and, if computed, their derivative groups.
+/// One particle's solution: its coefficients and, if computed, their derivative groups, with
+/// the radius fractions of a layered sphere (empty for a sphere).
 pub(super) struct Solved<'a> {
     pub(super) c: Coefficients<'a, f64>,
     pub(super) groups: Vec<Group<'a, f64>>,
+    pub(super) fractions: &'a [f64],
+}
+
+/// `e` times `s`.
+fn scaled(e: &Efficiencies<f64>, s: f64) -> Efficiencies<f64> {
+    Efficiencies {
+        q_ext: s * e.q_ext,
+        q_sca: s * e.q_sca,
+        q_abs: s * e.q_abs,
+        q_back: s * e.q_back,
+        g: s * e.g,
+    }
+}
+
+impl Solved<'_> {
+    /// The derivative of the efficiencies along direction `k`, from the groups' Jacobians
+    /// `jacs`: Re m or Im m of group k/2 for k below twice the number of groups, and above,
+    /// the radius fraction j = k − 2·groups, x_L·∂/∂x_j at a fixed outer size x_L.
+    pub(super) fn direction(
+        &self,
+        jacs: &[EfficiencyJacobian<f64>],
+        k: usize,
+    ) -> Efficiencies<f64> {
+        let groups = self.groups.len();
+        if k < 2 * groups {
+            let j = &jacs[k / 2];
+            if k % 2 == 0 { j.m_re } else { j.m_im }
+        } else {
+            scaled(&jacs[k - 2 * groups].x, self.c.x)
+        }
+    }
+
+    /// The derivative of the efficiencies with respect to the outer size parameter, at fixed
+    /// radius fractions: Σ_l f_l·∂/∂x_l (∂/∂x for a sphere).
+    pub(super) fn size_derivative(&self, jacs: &[EfficiencyJacobian<f64>]) -> Efficiencies<f64> {
+        let (outer, inner) = jacs.split_last().expect("a group");
+        let mut d = outer.x;
+        for (f, j) in self.fractions.iter().zip(inner) {
+            let e = scaled(&j.x, *f);
+            d.q_ext += e.q_ext;
+            d.q_sca += e.q_sca;
+            d.q_abs += e.q_abs;
+            d.q_back += e.q_back;
+            d.g += e.g;
+        }
+        d
+    }
 }
 
 /// A per-thread workspace for solving particles.
 pub(super) struct Solver {
     mie: Mie<f64>,
+    layered: Layered<f64>,
+    xs: Vec<f64>,
+    ms: Vec<Complex<f64>>,
 }
 
 impl Solver {
     pub(super) fn new() -> Self {
-        Self { mie: Mie::new() }
+        Self {
+            mie: Mie::new(),
+            layered: Layered::new(),
+            xs: Vec::new(),
+            ms: Vec::new(),
+        }
     }
 
-    /// The particle `optics` of size parameter `x`, with its derivative groups if
+    /// The particle `optics` of (outer) size parameter `x`, with its derivative groups if
     /// `derivatives`.
-    pub(super) fn solve(
-        &mut self,
-        optics: Optics,
+    pub(super) fn solve<'a>(
+        &'a mut self,
+        optics: Optics<'a>,
         x: f64,
         opts: &MieOptions,
         derivatives: bool,
-    ) -> Result<Solved<'_>> {
+    ) -> Result<Solved<'a>> {
         match optics {
             Optics::Sphere(m) => {
                 let s = if derivatives {
@@ -86,6 +247,21 @@ impl Solver {
                 Ok(Solved {
                     c: s.coefficients(),
                     groups: s.group().into_iter().collect(),
+                    fractions: &[],
+                })
+            }
+            Optics::Layered { fractions, .. } => {
+                optics.layers(x, &mut self.xs, &mut self.ms);
+                let s = if derivatives {
+                    self.layered
+                        .compute_with_derivatives(&self.xs, &self.ms, opts)
+                } else {
+                    self.layered.compute(&self.xs, &self.ms, opts)
+                }?;
+                Ok(Solved {
+                    c: s.coefficients(),
+                    groups: (0..self.xs.len()).filter_map(|l| s.group(l)).collect(),
+                    fractions,
                 })
             }
         }
diff --git a/scatrs/tests/ensemble_layered.rs b/scatrs/tests/ensemble_layered.rs
new file mode 100644
index 0000000..689d74f
--- /dev/null
+++ b/scatrs/tests/ensemble_layered.rs
@@ -0,0 +1,446 @@
+//! Ensembles of layered spheres: one layer is the sphere ensemble bitwise, Jacobians against
+//! finite differences (each layer's index, the radius fractions, the distribution, the radius
+//! of a monodisperse ensemble), mixtures with spheres, batches, determinism, and the rejection
+//! of bad inputs.
+#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together
+
+use num_complex::Complex;
+use scatrs::Error;
+use scatrs::ensemble::{
+    Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, ParamKind,
+    Particle, SizeDistribution, compute, compute_batch_with_particle, compute_mixture,
+    compute_with_particle, mixture_parameters, parameters, parameters_with_particle,
+};
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
+/// Black carbon in sulfate, with the core's radius fraction `f`, at one wavelength.
+fn coated(f: f64, core: Complex<f64>, shell: Complex<f64>) -> Particle {
+    Particle::layered_sphere(vec![f], vec![vec![core], vec![shell]])
+}
+
+const CORE: Complex<f64> = Complex::new(1.95, 0.79);
+const SHELL: Complex<f64> = Complex::new(1.43, 0.01);
+
+fn one(d: &SizeDistribution, lambda: f64, p: &Particle, opts: &EnsembleOptions) -> EnsembleResult {
+    compute_with_particle(d, &[lambda], p, opts)
+        .unwrap()
+        .remove(0)
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
+fn param(kind: ParamKind) -> EnsembleParam {
+    EnsembleParam::new(0, None, kind)
+}
+
+#[test]
+fn one_layer_is_the_sphere_ensemble_bitwise() {
+    let opts = EnsembleOptions::new()
+        .with_greek(GreekOrder::Order(12))
+        .with_mu(vec![-0.7, 0.0, 0.9]);
+    let m = vec![c(1.45, 0.02), c(1.5, 1e-4)];
+    let wavelengths = [0.45, 0.8];
+    let layered = Particle::layered_sphere(Vec::new(), vec![m.clone()]);
+    for d in [
+        lognormal(0.2, 1.6),
+        SizeDistribution::Monodisperse { radius: 0.3 },
+    ] {
+        let sphere_params = parameters(&d);
+        let layered_params = parameters_with_particle(&d, &layered);
+        let n = sphere_params.len();
+        assert_eq!(n, layered_params.len());
+        assert_eq!(layered_params[n - 2].kind, ParamKind::LayerRefrReal(0));
+        let want = compute(
+            &d,
+            &wavelengths,
+            &m,
+            &opts.clone().with_derivatives(sphere_params),
+        )
+        .unwrap();
+        let got = compute_with_particle(
+            &d,
+            &wavelengths,
+            &layered,
+            &opts.clone().with_derivatives(layered_params.clone()),
+        )
+        .unwrap();
+        for (w, g) in want.iter().zip(&got) {
+            assert_eq!(w.values, g.values);
+            assert_eq!(w.errors, g.errors);
+            assert_eq!(w.derivatives, g.derivatives);
+            assert_eq!(w.derivative_errors, g.derivative_errors);
+            assert_eq!((w.panels, &w.ranges), (g.panels, &g.ranges));
+            assert_eq!(g.params, layered_params);
+        }
+    }
+}
+
+#[test]
+fn layers_of_one_index_are_the_sphere_ensemble_bitwise() {
+    let opts = EnsembleOptions::new()
+        .with_greek(GreekOrder::Order(8))
+        .with_mu(vec![0.3]);
+    let m = c(1.5, 0.01);
+    let d = lognormal(0.3, 1.5);
+    let want = compute(&d, &[0.55], &[m], &opts).unwrap();
+    let three = Particle::layered_sphere(vec![0.3, 0.7], vec![vec![m]; 3]);
+    let got = compute_with_particle(&d, &[0.55], &three, &opts).unwrap();
+    assert_eq!(want[0].values, got[0].values);
+}
+
+/// The derivative of every scalar output, β₁ and F₁₁, F₁₂ with respect to `p`, against a
+/// central difference of the whole computation at a tight tolerance.
+fn check_derivative(
+    run: &dyn Fn(f64, &EnsembleOptions) -> EnsembleResult,
+    p: EnsembleParam,
+    value: f64,
+    what: &str,
+) {
+    let tight = EnsembleOptions::new()
+        .with_rtol(1e-11)
+        .with_max_panels(50_000)
+        .with_greek(GreekOrder::Order(6))
+        .with_mu(vec![0.8, -0.6]);
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
+    let greek = |o: &Outputs| o.greek.as_ref().unwrap().beta1.clone();
+    let (gd, gp, gm) = (
+        greek(&r.derivatives[0]),
+        greek(&plus.values),
+        greek(&minus.values),
+    );
+    for l in 0..gd.len() {
+        let fd = (gp[l] - gm[l]) / (2.0 * h);
+        assert!(
+            (gd[l] - fd).abs() <= 1e-5 * fd.abs().max(1e-3),
+            "{what}, beta1[{l}]: {} vs {fd}",
+            gd[l]
+        );
+    }
+    let f11 = &r.values.phase.as_ref().unwrap().f11;
+    for element in 0..2 {
+        let phase = |o: &Outputs| {
+            let p = o.phase.as_ref().unwrap();
+            if element == 0 {
+                p.f11.clone()
+            } else {
+                p.f12.clone()
+            }
+        };
+        let (pd, pp, pm) = (
+            phase(&r.derivatives[0]),
+            phase(&plus.values),
+            phase(&minus.values),
+        );
+        for j in 0..pd.len() {
+            let fd = (pp[j] - pm[j]) / (2.0 * h);
+            let scale = fd.abs().max(f11[j] / value.abs().max(1e-3));
+            assert!(
+                (pd[j] - fd).abs() <= 1e-5 * scale,
+                "{what}, element {element} at cosine {j}: {} vs {fd}",
+                pd[j]
+            );
+        }
+    }
+}
+
+#[test]
+fn jacobians_match_finite_differences() {
+    let lambda = 0.55;
+    let d = lognormal(0.15, 1.5);
+    let f = 0.6;
+    let with = |f: f64, core: Complex<f64>, shell: Complex<f64>| {
+        let d = d.clone();
+        move |o: &EnsembleOptions| one(&d, lambda, &coated(f, core, shell), o)
+    };
+    check_derivative(
+        &|v, o| with(v, CORE, SHELL)(o),
+        param(ParamKind::RadiusFraction(0)),
+        f,
+        "layer0.radius_fraction",
+    );
+    check_derivative(
+        &|v, o| with(f, c(v, CORE.im), SHELL)(o),
+        param(ParamKind::LayerRefrReal(0)),
+        CORE.re,
+        "layer0.m_real",
+    );
+    check_derivative(
+        &|v, o| with(f, c(CORE.re, v), SHELL)(o),
+        param(ParamKind::LayerRefrImag(0)),
+        CORE.im,
+        "layer0.m_imag",
+    );
+    check_derivative(
+        &|v, o| with(f, CORE, c(v, SHELL.im))(o),
+        param(ParamKind::LayerRefrReal(1)),
+        SHELL.re,
+        "layer1.m_real",
+    );
+    check_derivative(
+        &|v, o| with(f, CORE, c(SHELL.re, v))(o),
+        param(ParamKind::LayerRefrImag(1)),
+        SHELL.im,
+        "layer1.m_imag",
+    );
+    // The distribution, of the outer radius.
+    let p = coated(f, CORE, SHELL);
+    let ps = parameters_with_particle(&d, &p);
+    check_derivative(
+        &|v, o| one(&lognormal(v, 1.5), lambda, &p, o),
+        ps[0],
+        0.15,
+        "median_radius",
+    );
+    // A monodisperse ensemble's radius, at fixed fractions: Σ f_l ∂/∂x_l.
+    let mono = |r: f64| SizeDistribution::Monodisperse { radius: r };
+    let mp = parameters_with_particle(&mono(0.3), &p)[0];
+    check_derivative(&|v, o| one(&mono(v), lambda, &p, o), mp, 0.3, "radius");
+    // Three layers: the middle fraction and the middle index.
+    let three = |f1: f64, m1: Complex<f64>| {
+        Particle::layered_sphere(
+            vec![0.3, f1],
+            vec![vec![c(1.6, 0.3)], vec![m1], vec![c(1.33, 1e-6)]],
+        )
+    };
+    check_derivative(
+        &|v, o| one(&d, lambda, &three(v, c(2.0, 0.05)), o),
+        param(ParamKind::RadiusFraction(1)),
+        0.7,
+        "three layers: layer1.radius_fraction",
+    );
+    check_derivative(
+        &|v, o| one(&d, lambda, &three(0.7, c(2.0, v)), o),
+        param(ParamKind::LayerRefrImag(1)),
+        0.05,
+        "three layers: layer1.m_imag",
+    );
+    // A mixture's layered component.
+    let mix = |v: f64, o: &EnsembleOptions| {
+        compute_mixture(
+            &[
+                Component::new(lognormal(0.1, 1.5), vec![c(1.45, 0.0)], 2.0),
+                Component::with_particle(lognormal(0.3, 1.4), coated(v, CORE, SHELL), 0.5),
+            ],
+            &[lambda],
+            o,
+        )
+        .unwrap()
+        .remove(0)
+    };
+    let fraction = EnsembleParam::new(1, None, ParamKind::RadiusFraction(0));
+    check_derivative(&mix, fraction, 0.5, "c1.layer0.radius_fraction");
+}
+
+#[test]
+fn mixtures_of_spheres_and_layered_spheres_average_with_the_number_weights() {
+    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(4));
+    let wl = [0.55];
+    let (d0, d1) = (lognormal(0.1, 1.5), lognormal(0.3, 1.4));
+    let p1 = coated(0.5, CORE, SHELL);
+    let components = [
+        Component::new(d0.clone(), vec![c(1.45, 0.0)], 2.0),
+        Component::with_particle(d1.clone(), p1.clone(), 0.5),
+    ];
+    let mixed = compute_mixture(&components, &wl, &opts).unwrap().remove(0);
+    let a = compute(&d0, &wl, &[c(1.45, 0.0)], &opts).unwrap().remove(0);
+    let b = one(&d1, wl[0], &p1, &opts);
+    let (fa, fb) = (2.0 / 2.5, 0.5 / 2.5);
+    let close = |x: f64, y: f64| (x - y).abs() <= 1e-14 * y.abs();
+    assert!(close(
+        mixed.values.c_ext,
+        fa * a.values.c_ext + fb * b.values.c_ext
+    ));
+    assert!(close(
+        mixed.values.c_abs,
+        fa * a.values.c_abs + fb * b.values.c_abs
+    ));
+    let sca = fa * a.values.c_sca + fb * b.values.c_sca;
+    let g = (fa * a.values.c_sca * a.values.g + fb * b.values.c_sca * b.values.g) / sca;
+    assert!((mixed.values.g - g).abs() <= 1e-13);
+    // The names, prefixed by the component.
+    let names: Vec<String> = mixture_parameters(&components)
+        .iter()
+        .map(|p| p.name(true))
+        .collect();
+    assert_eq!(
+        names,
+        [
+            "c0.median_radius",
+            "c0.width",
+            "c0.m_real",
+            "c0.m_imag",
+            "c0.weight",
+            "c1.median_radius",
+            "c1.width",
+            "c1.layer0.m_real",
+            "c1.layer0.m_imag",
+            "c1.layer1.m_real",
+            "c1.layer1.m_imag",
+            "c1.layer0.radius_fraction",
+            "c1.weight",
+        ]
+    );
+}
+
+#[test]
+fn a_batch_matches_each_distribution() {
+    let opts = EnsembleOptions::new().with_greek(GreekOrder::Order(6));
+    let p = Particle::layered_sphere(
+        vec![0.4],
+        vec![vec![CORE, c(1.9, 0.7)], vec![SHELL, c(1.42, 0.0)]],
+    );
+    let wl = [0.45, 0.9];
+    let ds = [lognormal(0.1, 1.5), lognormal(0.4, 1.3)];
+    let batch = compute_batch_with_particle(&ds, &wl, &p, &opts).unwrap();
+    for (d, rs) in ds.iter().zip(&batch) {
+        let alone = compute_with_particle(d, &wl, &p, &opts).unwrap();
+        for (b, a) in rs.iter().zip(&alone) {
+            for (x, y, e) in [
+                (b.values.c_ext, a.values.c_ext, a.errors.c_ext),
+                (b.values.c_abs, a.values.c_abs, a.errors.c_abs),
+                (b.values.g, a.values.g, a.errors.g),
+            ] {
+                assert!(
+                    (x - y).abs() <= 2.0 * e + 1e-15 * y.abs(),
+                    "{x} vs {y} ± {e}"
+                );
+            }
+        }
+    }
+}
+
+#[test]
+fn results_do_not_depend_on_the_thread_count() {
+    let d = lognormal(0.2, 1.4);
+    let p = Particle::layered_sphere(
+        vec![0.3, 0.8],
+        vec![vec![CORE; 3], vec![c(1.6, 0.01); 3], vec![c(1.43, 1e-8); 3]],
+    );
+    let opts = EnsembleOptions::new()
+        .with_greek(GreekOrder::All)
+        .with_mu(vec![0.2])
+        .with_derivatives(parameters_with_particle(&d, &p));
+    let wavelengths = [0.4, 0.55, 0.8];
+    let run = |threads: usize| {
+        rayon::ThreadPoolBuilder::new()
+            .num_threads(threads)
+            .build()
+            .unwrap()
+            .install(|| compute_with_particle(&d, &wavelengths, &p, &opts).unwrap())
+    };
+    let one_thread = run(1);
+    assert_eq!(one_thread, run(2));
+    assert_eq!(one_thread, run(8));
+}
+
+#[test]
+fn derivatives_need_the_layered_derivative_domain() {
+    // x_L reaches about 2·10³ at λ = 0.5: beyond the derivatives' validated 10³.
+    let d = lognormal(80.0, 1.1);
+    let p = coated(0.5, CORE, SHELL);
+    let fraction = vec![param(ParamKind::RadiusFraction(0))];
+    let opts = EnsembleOptions::new()
+        .with_rtol(1e-3)
+        .with_derivatives(fraction);
+    let e = compute_with_particle(&d, &[0.5], &p, &opts).unwrap_err();
+    assert!(
+        matches!(&e, Error::OutsideValidatedDomain(s) if s.contains("derivatives")),
+        "{e}"
+    );
+    // The values alone are validated there.
+    let values = EnsembleOptions::new().with_rtol(1e-3);
+    assert!(compute_with_particle(&d, &[0.5], &p, &values).is_ok());
+}
+
+#[test]
+fn bad_layered_particles_are_rejected() {
+    let d = lognormal(0.1, 1.5);
+    let wl = [0.5];
+    let opts = EnsembleOptions::new();
+    let bad = |p: Particle, needle: &str| {
+        let e = compute_with_particle(&d, &wl, &p, &opts).unwrap_err();
+        assert!(
+            matches!(&e, Error::InvalidInput(s) if s.contains(needle)),
+            "{needle}: {e}"
+        );
+    };
+    bad(Particle::layered_sphere(Vec::new(), Vec::new()), "layers");
+    bad(
+        Particle::layered_sphere(vec![0.5, 0.6], vec![vec![CORE], vec![SHELL]]),
+        "radius fractions",
+    );
+    bad(coated(0.0, CORE, SHELL), "(0, 1]");
+    bad(coated(1.5, CORE, SHELL), "(0, 1]");
+    bad(coated(f64::NAN, CORE, SHELL), "(0, 1]");
+    bad(
+        Particle::layered_sphere(vec![0.6, 0.5], vec![vec![CORE], vec![SHELL], vec![SHELL]]),
+        "nondecreasing",
+    );
+    bad(
+        Particle::layered_sphere(vec![0.5], vec![vec![CORE], vec![SHELL, SHELL]]),
+        "one per wavelength",
+    );
+    bad(
+        Particle::layered_sphere(vec![0.5], vec![vec![CORE, CORE], vec![SHELL, SHELL]]),
+        "wavelengths",
+    );
+    // A derivative the particle does not have.
+    let p = coated(0.5, CORE, SHELL);
+    for kind in [
+        ParamKind::RefrReal,
+        ParamKind::LayerRefrReal(2),
+        ParamKind::RadiusFraction(1),
+    ] {
+        let o = EnsembleOptions::new().with_derivatives(vec![param(kind)]);
+        let e = compute_with_particle(&d, &wl, &p, &o).unwrap_err();
+        assert!(
+            matches!(&e, Error::InvalidInput(s) if s.contains("no parameter")),
+            "{e}"
+        );
+    }
+    // A sphere has no layers.
+    let o = EnsembleOptions::new().with_derivatives(vec![param(ParamKind::LayerRefrReal(0))]);
+    assert!(compute(&d, &wl, &[SHELL], &o).is_err());
+}
````

### Task 3: `LayeredSphere` in Python

- [ ] Parse the `("layered", fractions, m[L, n])` spec in the binding. Single-component parameters become particle-aware, and the batch takes any particle.
- [ ] Add `LayeredSphere` (`coated`, `layers`, `_engine`) and export it.
- [ ] Let `Ensemble` accept either particle. A `Mixture`'s `refractive_index` becomes a tuple, and pickles read-only.
- [ ] Add the tests (`tests/test_ensemble_layered.py`). Run mypy, which adds no new errors over its baseline of 15.

The commit (`2c934d3`):

````diff
diff --git a/python/scatrs/__init__.py b/python/scatrs/__init__.py
index 290853c..eb7c720 100644
--- a/python/scatrs/__init__.py
+++ b/python/scatrs/__init__.py
@@ -18,7 +18,7 @@ from scatrs.errors import (
     OutsideValidatedDomainError,
     ScatrsError,
 )
-from scatrs.particles import Sphere
+from scatrs.particles import LayeredSphere, Sphere
 from scatrs.size import (
     Gamma,
     LogNormal,
@@ -37,6 +37,7 @@ __all__ = [
     "EnsembleResult",
     "Gamma",
     "InvalidInputError",
+    "LayeredSphere",
     "LogNormal",
     "Mixture",
     "ModifiedGamma",
diff --git a/python/scatrs/ensemble.py b/python/scatrs/ensemble.py
index 432c091..e05a9f2 100644
--- a/python/scatrs/ensemble.py
+++ b/python/scatrs/ensemble.py
@@ -1,7 +1,8 @@
 """Ensembles of particles: size distributions, adaptive size integration and mixtures.
 
-An :class:`Ensemble` is a particle (:class:`~scatrs.Sphere`) and a size distribution
-(:mod:`scatrs.size`); a :class:`Mixture` is several ensembles with relative number
+An :class:`Ensemble` is a particle (:class:`~scatrs.Sphere` or
+:class:`~scatrs.LayeredSphere`) and a size distribution (:mod:`scatrs.size`) of its radius (a
+layered sphere's outer radius); a :class:`Mixture` is several ensembles with relative number
 concentrations. Their ``compute`` methods average the optical properties over the sizes at
 each wavelength. Radii and wavelengths are in the same (any) unit; cross sections are per
 particle, averaged over the number distribution, in that unit squared. See
@@ -10,8 +11,9 @@ particle, averaged over the number distribution, in that unit squared. See
 The size integral is adaptive (Gauss–Kronrod panels in ln r, refined until the cross
 sections, g and the distribution moments, and optionally the backscattering, meet ``rtol``).
 Every output comes with an error estimate (:attr:`~scatrs.EnsembleResult.error`) and, with
-``derivatives``, its derivatives with respect to the distribution parameters, ``"m_real"``,
-``"m_imag"`` and the mixture weights (:meth:`~scatrs.EnsembleResult.d`).
+``derivatives``, its derivatives with respect to the distribution parameters, the particle's
+(``"m_real"`` and ``"m_imag"``; each layer's ``"layer{l}.m_real"``, ``"layer{l}.m_imag"`` and
+``"layer{l}.radius_fraction"``) and the mixture weights (:meth:`~scatrs.EnsembleResult.d`).
 
 A distribution whose parameters are arrays is a grid of distributions, computed at once with
 shared size nodes (as an optics table needs): the results have the grid's axes first.
@@ -31,7 +33,7 @@ import numpy as np
 from numpy.typing import ArrayLike, NDArray
 
 from scatrs import _args, _scatrs
-from scatrs.particles import Sphere
+from scatrs.particles import LayeredSphere, Sphere
 from scatrs.size import Distribution, _check, _grid, _scalar_spec, _specs
 
 __all__ = [
@@ -72,8 +74,11 @@ class _ReadOnly:
 
     def __setstate__(self, state: dict[str, Any]) -> None:
         for name, value in state.items():
-            frozen = _frozen(value) if isinstance(value, np.ndarray) else value
-            object.__setattr__(self, name, frozen)
+            if isinstance(value, np.ndarray):
+                value = _frozen(value)
+            elif isinstance(value, tuple) and value and isinstance(value[0], np.ndarray):
+                value = tuple(_frozen(a) for a in value)
+            object.__setattr__(self, name, value)
 
 
 class _Grid(Mapping[str, NDArray[np.float64]]):
@@ -178,9 +183,10 @@ class EnsembleResult(EnsembleOutputs):
         The parameters :meth:`d` accepts.
     wavelength : numpy.ndarray
         The wavelengths, as given.
-    refractive_index : numpy.ndarray
-        m at the wavelengths: of the wavelengths' shape, or ``(components,) +
-        wavelength.shape`` for a :class:`Mixture`.
+    refractive_index : numpy.ndarray or tuple of numpy.ndarray
+        m at the wavelengths: of the wavelengths' shape for a :class:`~scatrs.Sphere`, and
+        ``(L,) + wavelength.shape`` for a :class:`~scatrs.LayeredSphere` of L layers; for a
+        :class:`Mixture`, a tuple of each component's.
     grid : Mapping[str, numpy.ndarray]
         Each array-valued distribution parameter, by name (as in :attr:`params`), broadcast
         to the grid's shape; empty if there is no grid.
@@ -190,7 +196,9 @@ class EnsembleResult(EnsembleOutputs):
     converged: NDArray[np.bool_] = field(kw_only=True)
     panels: NDArray[np.int64] = field(kw_only=True)
     wavelength: NDArray[np.float64] = field(kw_only=True)
-    refractive_index: NDArray[np.complex128] = field(kw_only=True)
+    refractive_index: NDArray[np.complex128] | tuple[NDArray[np.complex128], ...] = field(
+        kw_only=True
+    )
     grid: Mapping[str, NDArray[np.float64]] = field(kw_only=True)
     params: tuple[str, ...] = field(default=(), kw_only=True)
     _d: tuple[EnsembleOutputs, ...] = field(default=(), kw_only=True)
@@ -332,7 +340,7 @@ def _result(
     shape: tuple[int, ...],
     o: _Options,
     wavelength: NDArray[np.float64],
-    m: NDArray[np.complex128],
+    m: NDArray[np.complex128] | tuple[NDArray[np.complex128], ...],
     grid: dict[str, NDArray[np.float64]],
     params: tuple[str, ...] = (),
 ) -> EnsembleResult:
@@ -348,7 +356,7 @@ def _result(
         converged=raw["converged"].reshape(shape),
         panels=raw["panels"].reshape(shape),
         wavelength=_frozen(wavelength.copy()),
-        refractive_index=_frozen(m),
+        refractive_index=tuple(_frozen(a) for a in m) if isinstance(m, tuple) else _frozen(m),
         grid=_Grid(grid),
         params=params,
         _d=tuple(_outputs(raw, "_d", k, shape, mu_shape) for k in range(len(params))),
@@ -400,9 +408,11 @@ def _weight(w: ArrayLike, what: str) -> float:
     return float(a)
 
 
-def _sphere(particle: object, what: str) -> Sphere:
-    if not isinstance(particle, Sphere):
-        raise TypeError(f"{what} must be a Sphere, not {type(particle).__name__}")
+def _particle(particle: object, what: str) -> Sphere | LayeredSphere:
+    if not isinstance(particle, (Sphere, LayeredSphere)):
+        raise TypeError(
+            f"{what} must be a Sphere or a LayeredSphere, not {type(particle).__name__}"
+        )
     return particle
 
 
@@ -470,29 +480,30 @@ class Ensemble:
 
     Parameters
     ----------
-    particle : Sphere
-        The particle, with its refractive-index model.
+    particle : Sphere or LayeredSphere
+        The particle, with its refractive-index models.
     size : LogNormal, Gamma, ModifiedGamma, PowerLaw, Monodisperse, Tabulated, MultiModal or list
-        The number size distribution. If its parameters are arrays, the ensemble is a grid
-        of distributions, one per point of their broadcast shape. A list (or tuple) of
-        distributions with scalar parameters, of any kinds, is a batch with one axis, one
-        point per distribution; it is stored as a tuple.
+        The number size distribution of the particle's radius (of a layered sphere's outer
+        radius, its interfaces at fixed fractions of it). If its parameters are arrays, the
+        ensemble is a grid of distributions, one per point of their broadcast shape. A list
+        (or tuple) of distributions with scalar parameters, of any kinds, is a batch with one
+        axis, one point per distribution; it is stored as a tuple.
 
     Raises
     ------
     TypeError
-        If ``particle`` is not a :class:`~scatrs.Sphere`, or ``size`` or an element of a
-        list not a distribution.
+        If ``particle`` is not a :class:`~scatrs.Sphere` or a :class:`~scatrs.LayeredSphere`,
+        or ``size`` or an element of a list not a distribution.
     ValueError
         If the distribution's array-valued parameters do not broadcast together, or an
         element of a list is a grid.
     """
 
-    particle: Sphere
+    particle: Sphere | LayeredSphere
     size: Distribution | tuple[Distribution, ...]
 
     def __post_init__(self) -> None:
-        _sphere(self.particle, "particle")
+        _particle(self.particle, "particle")
         if isinstance(self.size, (list, tuple)):
             sizes = tuple(_check(d, f"size[{i}]") for i, d in enumerate(self.size))
             for i, d in enumerate(sizes):
@@ -575,8 +586,12 @@ class Ensemble:
 Ensemble.compute.__doc__ = _COMPUTE_DOC.format(
     summary="Averages over the ensemble at each wavelength.",
     derivatives='the distribution\'s (``"median_radius"``, …; ``"mode{j}.<name>"`` and '
-    '``"mode{j}.weight"`` for a multimodal one), ``"m_real"`` and ``"m_imag"`` (Re m and Im m '
-    "at each wavelength). Not available for a grid or a list.",
+    '``"mode{j}.weight"`` for a multimodal one), then the particle\'s: ``"m_real"`` and '
+    '``"m_imag"`` (Re m and Im m at each wavelength) of a Sphere; ``"layer{l}.m_real"`` and '
+    '``"layer{l}.m_imag"`` of each layer of a LayeredSphere, then ``"layer{l}.radius_fraction"`` '
+    "of each layer but the last (at a fixed outer radius). Not available for a grid or a list. "
+    "A LayeredSphere's derivatives along its own parameters, and a monodisperse one's radius "
+    "derivative, are validated up to an outer size parameter of 1000.",
     per="wavelength (shared by the points of a grid or a list)",
     returns="Every array of shape ``grid + wavelength.shape`` (``grid`` is "
     ":attr:`grid_shape`). The points of a grid or a list share their size nodes and each "
@@ -655,14 +670,15 @@ class Mixture:
             for (w, e), (particle, _) in zip(self.components, engine, strict=True)
         ]
         raw, params = _single(items, True, wl, derivatives, o)
-        return _result(raw, wl.shape, o, wl, np.stack(ms), {}, params)
+        return _result(raw, wl.shape, o, wl, tuple(ms), {}, params)
 
 
 Mixture.compute.__doc__ = _COMPUTE_DOC.format(
     summary="Averages over the mixture at each wavelength.",
-    derivatives='each component\'s distribution parameters, ``"m_real"`` and ``"m_imag"``, '
-    'prefixed by ``"c{i}."``, and the weights ``"c{i}.weight"``.',
+    derivatives="each component's distribution and particle parameters (as for "
+    ':meth:`Ensemble.compute`), prefixed by ``"c{i}."``, and the weights ``"c{i}.weight"``.',
     per="component and wavelength",
-    returns="Every array of the wavelengths' shape.",
+    returns="Every array of the wavelengths' shape (``refractive_index`` is a tuple of each "
+    "component's).",
     raises="",
 )
diff --git a/python/scatrs/particles.py b/python/scatrs/particles.py
index 66eba42..7079ce2 100644
--- a/python/scatrs/particles.py
+++ b/python/scatrs/particles.py
@@ -3,6 +3,7 @@
 from __future__ import annotations
 
 import numbers
+from collections.abc import Sequence
 from dataclasses import dataclass
 
 import numpy as np
@@ -10,7 +11,7 @@ from numpy.typing import NDArray
 
 from scatrs.refractive import Constant, Function, RefractiveIndex
 
-__all__ = ["Sphere"]
+__all__ = ["LayeredSphere", "Sphere"]
 
 
 def _model(value: object) -> RefractiveIndex:
@@ -68,3 +69,102 @@ class Sphere:
         index there (the wavelengths' shape), which the result reports."""
         m = self.refractive_index.evaluate(wavelength)
         return ("sphere", np.ascontiguousarray(m.reshape(-1))), m
+
+
+def _fractions(value: object, layers: int) -> tuple[float, ...]:
+    """``value`` as the radius fractions of ``layers`` layers.
+
+    Raises
+    ------
+    TypeError
+        If ``value`` is not a sequence of real numbers.
+    ValueError
+        If there are not ``layers - 1`` of them, or they are not nondecreasing in (0, 1].
+    """
+    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
+        raise TypeError(
+            f"radius_fraction must be a sequence of real numbers, not {type(value).__name__}"
+        )
+    out: list[float] = []
+    for i, v in enumerate(value):
+        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
+            raise TypeError(f"radius_fraction[{i}] must be a real number, not {type(v).__name__}")
+        f = float(v)
+        if not 0.0 < f <= 1.0:
+            raise ValueError(f"radius_fraction[{i}] must be in (0, 1], got {f}")
+        if out and f < out[-1]:
+            raise ValueError(
+                f"radius_fraction must be nondecreasing from the core out, but "
+                f"radius_fraction[{i}] = {f} < {out[-1]}"
+            )
+        out.append(f)
+    if len(out) != layers - 1:
+        raise ValueError(
+            f"a layered sphere of {layers} layers needs {layers - 1} radius fractions, "
+            f"got {len(out)}"
+        )
+    return tuple(out)
+
+
+@dataclass(frozen=True)
+class LayeredSphere:
+    """A layered sphere: a core and concentric shells (see :mod:`scatrs.layered`).
+
+    In an :class:`~scatrs.Ensemble` the size distribution gives the outer radius, and each
+    interface stays at a fixed fraction of it. The core's volume fraction is the cube of its
+    radius fraction: a core of a quarter of the particle's volume has
+    ``radius_fraction=[0.25 ** (1 / 3)]``.
+
+    Parameters
+    ----------
+    refractive_index : sequence
+        Each layer's refractive index, the core first: :mod:`scatrs.refractive` models, numbers
+        (:class:`~scatrs.refractive.Constant`) or other callables
+        (:class:`~scatrs.refractive.Function`), as for :class:`Sphere`. Stored as a tuple.
+    radius_fraction : sequence of float
+        The outer radius of each layer but the last, as a fraction of the particle's radius:
+        L − 1 values, nondecreasing in (0, 1] (empty for one layer). Stored as a tuple.
+
+    Raises
+    ------
+    TypeError
+        If ``refractive_index`` is not a sequence of models, numbers or callables, or
+        ``radius_fraction`` not a sequence of real numbers.
+    ValueError
+        If there is no layer, or the fractions are not L − 1 values nondecreasing in (0, 1].
+    """
+
+    refractive_index: tuple[RefractiveIndex, ...]
+    radius_fraction: tuple[float, ...] = ()
+
+    def __post_init__(self) -> None:
+        m = self.refractive_index
+        if isinstance(m, (str, bytes)) or not isinstance(m, (Sequence, np.ndarray)):
+            raise TypeError(
+                "refractive_index must be a sequence with one model, number or callable per "
+                f"layer, not {type(m).__name__}; for a homogeneous sphere use Sphere"
+            )
+        if len(m) == 0:
+            raise ValueError("a layered sphere needs at least one layer")
+        object.__setattr__(self, "refractive_index", tuple(_model(v) for v in m))
+        object.__setattr__(
+            self, "radius_fraction", _fractions(self.radius_fraction, len(self.refractive_index))
+        )
+
+    @classmethod
+    def coated(cls, core: object, shell: object, *, core_fraction: float) -> LayeredSphere:
+        """A core of refractive index ``core`` in a shell of ``shell``, the core's radius
+        ``core_fraction`` of the particle's."""
+        return cls((_model(core), _model(shell)), (core_fraction,))
+
+    @property
+    def layers(self) -> int:
+        """The number of layers L."""
+        return len(self.refractive_index)
+
+    def _engine(self, wavelength: NDArray[np.float64]) -> tuple[tuple, NDArray[np.complex128]]:
+        """The engine's tagged tuple for the particle at ``wavelength``, and its refractive
+        indices there (shape ``(L,) + wavelength.shape``), which the result reports."""
+        m = np.stack([v.evaluate(wavelength) for v in self.refractive_index])
+        fractions = np.array(self.radius_fraction, dtype=np.float64)
+        return ("layered", fractions, np.ascontiguousarray(m.reshape(self.layers, -1))), m
diff --git a/scatrs-py/src/ensemble.rs b/scatrs-py/src/ensemble.rs
index 6aad463..50007fb 100644
--- a/scatrs-py/src/ensemble.rs
+++ b/scatrs-py/src/ensemble.rs
@@ -2,8 +2,9 @@
 //!
 //! Distributions arrive as tagged tuples from `scatrs.ensemble` (`("lognormal", r_g, σ_g)`,
 //! `("tabulated", radii, densities)`, `("multimodal", [(weight, spec), …])`, …), particles as
-//! tagged tuples too (`("sphere", refractive indices)`, one refractive index per wavelength),
-//! and components as `(spec, particle, weight)`. Results
+//! tagged tuples too (`("sphere", m)`, one refractive index per wavelength; `("layered",
+//! fractions, m)`, L − 1 radius fractions and m of shape (L, wavelengths)), and components as
+//! `(spec, particle, weight)`. Results
 //! return as a dict of arrays, one row per wavelength, which `scatrs.ensemble` wraps;
 //! derivatives are selected by their index in `ensemble_parameters`. Arrays are borrowed with
 //! `try_readonly` (see `special.rs`), and the outputs are computed and laid out without the GIL.
@@ -15,7 +16,8 @@ use pyo3::prelude::*;
 use pyo3::types::{PyDict, PyTuple};
 use scatrs::ensemble::{
     Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, Particle,
-    SizeDistribution, compute_batch, compute_mixture, mixture_parameters, parameters,
+    SizeDistribution, compute_batch_with_particle, compute_mixture, mixture_parameters,
+    parameters_with_particle,
 };
 use scatrs::mie::{MAX_GREEK_ORDER, Normalization};
 
@@ -89,6 +91,31 @@ fn particle(spec: &Bound<'_, PyAny>) -> PyResult<Particle> {
             let m = borrow(m, "refractive_index")?;
             Ok(Particle::sphere(m.as_array().iter().copied().collect()))
         }
+        "layered" => {
+            let f = t.get_item(1)?;
+            let f = f.cast::<PyArrayDyn<f64>>().map_err(|_| {
+                PyTypeError::new_err("a layered sphere's radius fractions must be an array")
+            })?;
+            let f = borrow(f, "radius_fraction")?;
+            let m = t.get_item(2)?;
+            let m = m.cast::<PyArrayDyn<Complex64>>().map_err(|_| {
+                PyTypeError::new_err("a layered sphere's refractive indices must be an array")
+            })?;
+            let m = borrow(m, "refractive_index")?;
+            let m = m.as_array();
+            if m.ndim() != 2 {
+                return Err(PyTypeError::new_err(format!(
+                    "a layered sphere's refractive indices must have shape (layers, \
+                     wavelengths), got {} axes",
+                    m.ndim()
+                )));
+            }
+            let layers = m.outer_iter().map(|row| row.iter().copied().collect());
+            Ok(Particle::layered_sphere(
+                f.as_array().iter().copied().collect(),
+                layers.collect(),
+            ))
+        }
         other => Err(PyTypeError::new_err(format!(
             "unknown particle kind {other:?}"
         ))),
@@ -126,7 +153,7 @@ fn all_parameters(py: Python<'_>, cs: &[Component], mixture: bool) -> PyResult<V
     Ok(if mixture {
         mixture_parameters(cs)
     } else {
-        parameters(&cs[0].distribution)
+        parameters_with_particle(&cs[0].distribution, &cs[0].particle)
     })
 }
 
@@ -431,10 +458,7 @@ pub fn ensemble_compute_batch<'py>(
     let err = |e| to_pyerr(py, e);
     let distributions: Vec<SizeDistribution> =
         specs.iter().map(distribution).collect::<PyResult<_>>()?;
-    let m = match particle(&particle_spec)? {
-        Particle::Sphere { refractive_index } => refractive_index,
-        _ => return Err(PyTypeError::new_err("a batch needs spheres")),
-    };
+    let particle = particle(&particle_spec)?;
     let wl: Vec<f64> = borrow(&wavelengths, "wavelength")?
         .as_array()
         .iter()
@@ -455,10 +479,11 @@ pub fn ensemble_compute_batch<'py>(
     let (opts, order) = settings.options(mu, Vec::new(), n, 0).map_err(err)?;
     let arrays = py
         .detach(|| -> scatrs::Result<Arrays> {
-            let results: Vec<EnsembleResult> = compute_batch(&distributions, &wl, &m, &opts)?
-                .into_iter()
-                .flatten()
-                .collect();
+            let results: Vec<EnsembleResult> =
+                compute_batch_with_particle(&distributions, &wl, &particle, &opts)?
+                    .into_iter()
+                    .flatten()
+                    .collect();
             arrays(&results, 0, width(order, &results), n_mu)
         })
         .map_err(err)?;
diff --git a/tests/test_ensemble.py b/tests/test_ensemble.py
index 9af1605..6e53e9a 100644
--- a/tests/test_ensemble.py
+++ b/tests/test_ensemble.py
@@ -124,8 +124,12 @@ def test_mixture_results_carry_each_components_refractive_index():
     a = Ensemble(Sphere(1.45), LOGNORMAL)
     b = Ensemble(Sphere(refractive.Cauchy(1.5, 0.004)), LOGNORMAL)
     r = Mixture([(1.0, a), (1.0, b)]).compute(wl)
-    assert r.refractive_index.shape == (2, 1, 3) and r.c_ext.shape == (1, 3)
+    assert r.c_ext.shape == (1, 3)
+    assert isinstance(r.refractive_index, tuple) and len(r.refractive_index) == 2
+    assert r.refractive_index[0].shape == (1, 3)
     np.testing.assert_array_equal(r.refractive_index[1], b.particle.refractive_index.evaluate(wl))
+    with pytest.raises(ValueError):
+        r.refractive_index[0].flags.writeable = True
     assert dict(r.grid) == {}
 
 
diff --git a/tests/test_ensemble_layered.py b/tests/test_ensemble_layered.py
new file mode 100644
index 0000000..63dbfd2
--- /dev/null
+++ b/tests/test_ensemble_layered.py
@@ -0,0 +1,153 @@
+"""LayeredSphere: construction, ensembles and mixtures of layered spheres, derivatives, errors."""
+
+import copy
+import pickle
+from dataclasses import FrozenInstanceError
+
+import numpy as np
+import pytest
+
+from scatrs import (
+    Ensemble,
+    LayeredSphere,
+    LogNormal,
+    Mixture,
+    Monodisperse,
+    OutsideValidatedDomainError,
+    Sphere,
+    refractive,
+)
+from scatrs.ensemble import SCALARS
+
+SIZE = LogNormal(0.1, 1.5)
+CORE, SHELL = 1.95 + 0.79j, 1.43 + 1e-8j
+
+
+def test_construction_converts_and_stores_tuples():
+    p = LayeredSphere([CORE, refractive.Cauchy(1.5, 0.004)], [0.5])
+    assert p.refractive_index == (refractive.Constant(CORE), refractive.Cauchy(1.5, 0.004))
+    assert p.radius_fraction == (0.5,) and p.layers == 2
+    assert LayeredSphere.coated(CORE, SHELL, core_fraction=0.5) == LayeredSphere(
+        (CORE, SHELL), (0.5,)
+    )
+    assert LayeredSphere([1.5]).radius_fraction == ()
+    assert LayeredSphere(np.array([1.5, 1.33]), np.array([0.4])).radius_fraction == (0.4,)
+    # Frozen and hashable.
+    assert hash(p) == hash(LayeredSphere([CORE, refractive.Cauchy(1.5, 0.004)], (0.5,)))
+    with pytest.raises(FrozenInstanceError):
+        p.radius_fraction = (0.6,)  # type: ignore[misc]
+
+
+@pytest.mark.parametrize(
+    ("m", "f", "error", "match"),
+    [
+        ([], (), ValueError, "at least one layer"),
+        (1.5, (), TypeError, "sequence"),
+        ("1.5", (), TypeError, "sequence"),
+        ([1.5, [1.33]], (0.5,), TypeError, "for values at given wavelengths"),
+        ([1.5, 1.33], (), ValueError, "needs 1 radius fractions"),
+        ([1.5, 1.33], (0.5, 0.6), ValueError, "needs 1 radius fractions"),
+        ([1.5, 1.33], (0.0,), ValueError, r"\(0, 1\]"),
+        ([1.5, 1.33], (1.2,), ValueError, r"\(0, 1\]"),
+        ([1.5, 1.33], (float("nan"),), ValueError, r"\(0, 1\]"),
+        ([1.5, 1.4, 1.33], (0.6, 0.5), ValueError, "nondecreasing"),
+        ([1.5, 1.33], (True,), TypeError, "real number"),
+        ([1.5, 1.33], 0.5, TypeError, "sequence"),
+    ],
+)
+def test_bad_layered_spheres_are_rejected(m, f, error, match):
+    with pytest.raises(error, match=match):
+        LayeredSphere(m, f)
+
+
+def test_one_layer_is_the_sphere_ensemble_bitwise():
+    wl = np.array([0.45, 0.8])
+    model = refractive.Cauchy(1.5, 0.004)
+    kw = {"greek": 10, "mu": [0.3, -0.8], "derivatives": True}
+    a = Ensemble(Sphere(model), SIZE).compute(wl, **kw)
+    b = Ensemble(LayeredSphere([model]), SIZE).compute(wl, **kw)
+    assert b.params == ("median_radius", "width", "layer0.m_real", "layer0.m_imag")
+    for name in SCALARS:
+        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
+        np.testing.assert_array_equal(
+            getattr(a.d("m_imag"), name), getattr(b.d("layer0.m_imag"), name)
+        )
+    np.testing.assert_array_equal(a.greek.beta1, b.greek.beta1)
+    np.testing.assert_array_equal(a.d("m_real").phase.f12, b.d("layer0.m_real").phase.f12)
+    assert b.refractive_index.shape == (1, 2)
+    np.testing.assert_array_equal(b.refractive_index[0], a.refractive_index)
+
+
+def test_derivative_names_and_the_refractive_index():
+    wl = np.array([[0.45, 0.55, 0.8]])
+    p = LayeredSphere([CORE, 1.6 + 0.01j, SHELL], [0.3, 0.8])
+    r = Ensemble(p, SIZE).compute(wl, derivatives=True)
+    assert r.params == (
+        "median_radius",
+        "width",
+        "layer0.m_real",
+        "layer0.m_imag",
+        "layer1.m_real",
+        "layer1.m_imag",
+        "layer2.m_real",
+        "layer2.m_imag",
+        "layer0.radius_fraction",
+        "layer1.radius_fraction",
+    )
+    assert r.c_ext.shape == (1, 3) and r.refractive_index.shape == (3, 1, 3)
+    np.testing.assert_array_equal(r.refractive_index[1], np.full((1, 3), 1.6 + 0.01j))
+
+
+def test_radius_fraction_derivative_matches_a_central_difference():
+    wl, h = 0.55, 1e-5
+
+    def run(f, **kw):
+        p = LayeredSphere.coated(CORE, SHELL, core_fraction=f)
+        return Ensemble(p, SIZE).compute(wl, rtol=1e-11, max_panels=50_000, **kw)
+
+    r = run(0.5, derivatives=["layer0.radius_fraction"])
+    plus, minus = run(0.5 + h), run(0.5 - h)
+    for name in ("c_ext", "c_abs", "g"):
+        fd = (getattr(plus, name) - getattr(minus, name)) / (2 * h)
+        assert getattr(r.d("layer0.radius_fraction"), name) == pytest.approx(fd, rel=1e-6)
+    # A larger black-carbon core absorbs more.
+    assert r.d("layer0.radius_fraction").c_abs > 0
+
+
+def test_mixtures_and_grids_of_layered_spheres():
+    wl = np.array([0.45, 0.55])
+    coated = Ensemble(LayeredSphere.coated(CORE, SHELL, core_fraction=0.5), SIZE)
+    sphere = Ensemble(Sphere(1.45), LogNormal(0.2, 1.4))
+    mix = Mixture([(2.0, sphere), (0.5, coated)])
+    values = mix.compute(wl)
+    a, b = sphere.compute(wl), coated.compute(wl)
+    np.testing.assert_allclose(values.c_ext, 0.8 * a.c_ext + 0.2 * b.c_ext, rtol=1e-14)
+    r = mix.compute(wl, derivatives=["c1.layer0.radius_fraction", "c0.m_real"])
+    assert r.params == ("c1.layer0.radius_fraction", "c0.m_real")
+    assert np.all(r.d("c1.layer0.radius_fraction").c_abs > 0)
+    assert [m.shape for m in r.refractive_index] == [(2,), (2, 2)]
+    # The result pickles and copies read-only.
+    for c in (pickle.loads(pickle.dumps(r)), copy.deepcopy(r)):
+        np.testing.assert_array_equal(c.refractive_index[1], r.refractive_index[1])
+        with pytest.raises(ValueError):
+            c.refractive_index[1].flags.writeable = True
+    # A grid of distributions of layered spheres, against its points.
+    grid = Ensemble(coated.particle, LogNormal(np.array([0.1, 0.3]), 1.5)).compute(wl)
+    for k, rg in enumerate((0.1, 0.3)):
+        alone = Ensemble(coated.particle, LogNormal(rg, 1.5)).compute(wl)
+        np.testing.assert_allclose(grid.c_ext[k], alone.c_ext, atol=2 * alone.error.c_ext.max())
+
+
+def test_the_derivatives_domain_and_wrong_particles():
+    coated = LayeredSphere.coated(CORE, SHELL, core_fraction=0.5)
+    big = Ensemble(coated, Monodisperse(200.0))  # x_L ≈ 2300 at λ = 0.55
+    assert big.compute(0.55).c_ext > 0
+    with pytest.raises(OutsideValidatedDomainError, match="derivatives"):
+        big.compute(0.55, derivatives=["layer0.radius_fraction"])
+    # So does the radius derivative of a monodisperse ensemble, at fixed fractions.
+    with pytest.raises(OutsideValidatedDomainError, match="derivatives"):
+        big.compute(0.55, derivatives=["radius"])
+    with pytest.raises(TypeError, match="Sphere or a LayeredSphere"):
+        Ensemble(1.5, SIZE)  # type: ignore[arg-type]
+    with pytest.raises(ValueError, match="unknown parameter"):
+        Ensemble(coated, SIZE).compute(0.55, derivatives=["m_real"])
````

### Task 4: Documentation

- [ ] Add the example `docs/examples/coated.md` and its toctree entry. Check its numbers by prototype first: the enhancement is 1.52–1.65, and the albedo minimum is near f = 0.87.
- [ ] Update the API page, quick start, limitations, references (Toon & Ackerman 1981, Bond & Bergstrom 2006), the ensembles theory chapter, validation (calibration table), README, index and the Rust overview.
- [ ] Draft the 0.3.0 release notes above v0.2.0's.
- [ ] Add the two layered cases to the calibration test.

The commit (`ab940fc`):

````diff
diff --git a/README.md b/README.md
index 3e77dd1..0f9b899 100644
--- a/README.md
+++ b/README.md
@@ -12,11 +12,12 @@ library for the optical properties of particles.
   amplitudes, phase matrix and Greek coefficients to every order.
 - **Layered spheres:** a core and any number of concentric shells (coated spheres among them),
   with the same outputs and derivatives with respect to every layer.
-- **Ensembles of spheres:** averages over seven kinds of size distribution, external mixtures
-  and batches of distributions, at many wavelengths in parallel, with an error estimate for
-  every output.
-- **Derivatives** of every output: in closed form for spheres, by dual numbers for layered
-  spheres.
+- **Ensembles of spheres and layered spheres:** averages over seven kinds of size
+  distribution, external mixtures and batches of distributions, at many wavelengths in
+  parallel, with an error estimate for every output.
+- **Derivatives** of every output, in closed form: with respect to the size parameter and
+  every layer's refractive index, and for ensembles the distribution parameters, a layered
+  sphere's radius fractions and the mixture weights.
 - **Validated accuracy:** every output has a declared accuracy, checked against independent
   arbitrary-precision reference values across the validated domain, and against published
   results.
diff --git a/docs/api.md b/docs/api.md
index 9701b52..5218ec1 100644
--- a/docs/api.md
+++ b/docs/api.md
@@ -30,6 +30,9 @@ level.
 
 .. autoclass:: scatrs.Sphere
 
+.. autoclass:: scatrs.LayeredSphere
+   :members: coated, layers
+
 .. autoclass:: scatrs.Ensemble
    :members: compute, grid_shape
 
diff --git a/docs/examples/coated.md b/docs/examples/coated.md
new file mode 100644
index 0000000..6bd3414
--- /dev/null
+++ b/docs/examples/coated.md
@@ -0,0 +1,152 @@
+---
+file_format: mystnb
+kernelspec:
+  name: python3
+  display_name: Python 3
+  language: python
+---
+
+# Example: coated black carbon
+
+Soot ages in the atmosphere by gathering a coating of sulfate, organics and water. The coating
+absorbs almost nothing itself, but it focuses light onto the absorbing core, so a coated
+particle absorbs more than its bare core: the *absorption enhancement*. This page computes it
+for single particles with {mod}`scatrs.layered`, then for a size distribution of coated
+particles with {class}`~scatrs.LayeredSphere`, and differentiates the results with respect to
+the coating. See {doc}`/theory/layered` for the method. This page runs whenever the
+documentation is built; its checks are `assert`s, so a failing check fails the build.
+
+The refractive indices are constants here: $1.95 + 0.79i$ for black carbon (Bond & Bergstrom
+2006) and $1.43 + 10^{-8}i$ for sulfate. Radii and wavelengths are in µm.
+
+```{code-cell} python
+import matplotlib.pyplot as plt
+import numpy as np
+
+import scatrs
+from scatrs import layered
+
+bc, sulfate = 1.95 + 0.79j, 1.43 + 1e-8j
+```
+
+## One particle: the absorption enhancement
+
+A black-carbon core of radius 0.05 µm, in sulfate shells of growing thickness, at 550 nm. The
+size parameters of the interfaces go along the last axis, the core's first, and broadcast as in
+{mod}`scatrs.mie`. The absorption cross section is $Q_\text{abs}$ times the geometric cross
+section of the *outer* surface.
+
+```{code-cell} python
+wavelength, r_core = 0.55, 0.05
+k = 2 * np.pi / wavelength
+r_outer = r_core * np.linspace(1, 3, 41)  # the outer radius, from a bare core to 3 times it
+x = np.stack([np.full_like(r_outer, k * r_core), k * r_outer], axis=-1)
+q = layered.efficiencies(x, [bc, sulfate])
+c_abs = q.q_abs * np.pi * r_outer**2
+enhancement = c_abs / c_abs[0]
+print(f"bare core: C_abs = {c_abs[0]:.3e} µm²")
+for ratio in (1.5, 2.0, 3.0):
+    i = np.argmin(abs(r_outer / r_core - ratio))
+    print(f"outer/core radius {ratio}: enhancement {enhancement[i]:.2f}")
+assert np.all(np.diff(enhancement) > 0) and 2 < enhancement[-1] < 2.5
+```
+
+The first point is the bare core itself: a shell of zero thickness changes nothing. A coating
+twice the core's radius nearly doubles the absorption, even though it absorbs almost nothing
+itself.
+
+```{code-cell} python
+fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
+ax.plot(r_outer / r_core, enhancement)
+ax.set(xlabel="outer radius / core radius", ylabel="$C_\\mathrm{abs}$ / bare core's",
+       title="Absorption enhancement at 550 nm")
+plt.show()
+```
+
+## A size distribution of coated particles
+
+In an ensemble, the size distribution gives the *outer* radius, and each interface stays at a
+fixed fraction of it: here the core's radius is 60% of the particle's, so its volume fraction
+is $0.6^3 \approx 0.22$. {meth}`LayeredSphere.coated <scatrs.LayeredSphere.coated>` builds a
+core and one shell. The bare cores alone are an ensemble of spheres whose distribution is the
+coated one's scaled by the fraction: a lognormal of median radius $0.6\,r_g$ and the same width.
+
+```{code-cell} python
+r_g, width, f = 0.08, 1.6, 0.6
+wavelengths = np.array([0.35, 0.45, 0.55, 0.7, 0.87, 1.06])
+coated = scatrs.Ensemble(
+    scatrs.LayeredSphere.coated(bc, sulfate, core_fraction=f),
+    scatrs.LogNormal(median_radius=r_g, width=width),
+)
+cores = scatrs.Ensemble(scatrs.Sphere(bc), scatrs.LogNormal(median_radius=f * r_g, width=width))
+r, bare = coated.compute(wavelengths), cores.compute(wavelengths)
+print(f"{'λ (µm)':>7}{'ω coated':>10}{'ω cores':>9}{'enhancement':>13}")
+for row in zip(wavelengths, r.ssa, bare.ssa, r.c_abs / bare.c_abs, strict=True):
+    print("{:7.2f}{:10.3f}{:9.3f}{:13.2f}".format(*row))
+assert r.converged.all() and bare.converged.all()
+assert np.all(r.c_abs / bare.c_abs > 1.4) and np.all(r.ssa > bare.ssa)
+```
+
+The coated particles both absorb and scatter more than their cores: their single-scattering
+albedo $\omega$ is higher, and their absorption is 1.5 to 1.65 times the cores'. A shell of
+zero thickness (`core_fraction=1`) gives back the bare cores:
+
+```{code-cell} python
+thin = scatrs.Ensemble(
+    scatrs.LayeredSphere.coated(bc, sulfate, core_fraction=1.0),
+    scatrs.LogNormal(median_radius=f * r_g, width=width),
+).compute(wavelengths)
+assert np.allclose(thin.c_abs, bare.c_abs, rtol=1e-12) and np.allclose(thin.g, bare.g, rtol=1e-12)
+```
+
+## Derivatives with respect to the coating
+
+The derivatives with respect to each layer's refractive index and each radius fraction come
+with the values (`derivatives=True` for all of them, or a list of names). At a fixed outer
+radius, a larger core fraction means a thinner coating and more black carbon:
+
+```{code-cell} python
+r = coated.compute(0.55, derivatives=True)
+print(r.params)
+d = r.d("layer0.radius_fraction")
+print(f"d<C_abs>/df = {float(d.c_abs):.4e} µm², dω/df = {float(d.ssa):.4f}")
+
+h = 1e-6
+plus, minus = (
+    scatrs.Ensemble(scatrs.LayeredSphere.coated(bc, sulfate, core_fraction=f + s), coated.size)
+    .compute(0.55, rtol=1e-11)
+    for s in (h, -h)
+)
+fd = (plus.ssa - minus.ssa) / (2 * h)
+print(f"central difference: dω/df = {float(fd):.4f}")
+assert abs(d.ssa - fd) < 1e-6 * abs(fd)
+assert d.c_abs > 0 and d.ssa < 0
+```
+
+The albedo across core fractions, with the tangent the derivative gives at each point. The
+albedo falls as the core grows, to a shallow minimum near $f = 0.87$, where the shell is thin
+and the lensing it gave is mostly gone:
+
+```{code-cell} python
+fractions = np.linspace(0.3, 1.0, 15)
+ssa, slope = [], []
+for frac in fractions:
+    p = scatrs.LayeredSphere.coated(bc, sulfate, core_fraction=frac)
+    e = scatrs.Ensemble(p, coated.size).compute(0.55, derivatives=["layer0.radius_fraction"])
+    ssa.append(float(e.ssa))
+    slope.append(float(e.d("layer0.radius_fraction").ssa))
+ssa, slope = np.array(ssa), np.array(slope)
+
+fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
+ax.plot(fractions, ssa, "o", ms=3)
+for fr, s, ds in zip(fractions, ssa, slope, strict=True):
+    dx = np.array([-0.02, 0.02])
+    ax.plot(fr + dx, s + ds * dx, color="C1", lw=1)
+ax.set(xlabel="core radius fraction $f$", ylabel="single-scattering albedo $\\omega$",
+       title="550 nm, with $d\\omega/df$")
+plt.show()
+assert np.all(np.sign(slope[:-3]) == -1) and slope[-1] > 0
+```
+
+The derivatives are validated for outer size parameters up to 1000 ({doc}`/limitations`); the
+values, up to $10^4$.
diff --git a/docs/index.md b/docs/index.md
index 2b594a9..d6c2661 100644
--- a/docs/index.md
+++ b/docs/index.md
@@ -14,14 +14,14 @@ library for the optical properties of particles. Install it with `pip install sc
 - **Layered spheres** ({mod}`scatrs.layered`): a core and any number of concentric shells,
   coated spheres among them, with the same outputs and derivatives with respect to every
   layer.
-- **Ensembles of spheres** ({class}`scatrs.Ensemble`, {class}`scatrs.Mixture`): averages over
-  seven kinds of size distribution and over external mixtures, at many wavelengths in
-  parallel, with an error estimate for every output, and refractive indices that vary with
+- **Ensembles** of spheres and layered spheres ({class}`scatrs.Ensemble`,
+  {class}`scatrs.Mixture`): averages over seven kinds of size distribution and over external
+  mixtures, at many wavelengths in parallel, with an error estimate for every output, and refractive indices that vary with
   wavelength ({mod}`scatrs.refractive`). Grids of distributions share their Mie solutions, for
   building optics tables.
 - **Derivatives** of every output, in closed form: with respect to a sphere's size parameter
   and refractive index, to every layer's of a layered sphere, and for ensembles, the
-  distribution parameters and the mixture weights.
+  distribution parameters, a layered sphere's radius fractions and the mixture weights.
 - **Validated accuracy.** Every output has a declared accuracy, checked against independent
   reference values with rigorous error bounds across the whole validated domain, and against
   published results. See {doc}`validation/overview`.
@@ -46,6 +46,7 @@ quickstart
 examples/mie
 examples/greek
 examples/ensemble
+examples/coated
 ```
 
 ```{toctree}
diff --git a/docs/limitations.md b/docs/limitations.md
index 185e2bd..94d63c5 100644
--- a/docs/limitations.md
+++ b/docs/limitations.md
@@ -10,8 +10,8 @@ describes what the accuracy figures rest on.
 
 ## Scope and availability
 
-- scatrs computes homogeneous spheres, single or averaged over size distributions and
-  external mixtures, and single layered spheres (core and shells).
+- scatrs computes homogeneous spheres and layered spheres (a core and concentric shells),
+  single or averaged over size distributions and external mixtures.
 - Python exposes only `f64` results. Double-double precision and the `Dual` derivatives of
   the Rust core are available from Rust only.
 - The tests run on Linux (x86_64), macOS (arm64) and Windows (x86_64) for every change. The
@@ -37,6 +37,15 @@ describes what the accuracy figures rest on.
   `OutsideValidatedDomain` unless `allow_unvalidated` is set.
 - **Cost.** About 2.5 times a sphere of the outer size for a coated sphere, and roughly
   1.5 times more per further layer ({doc}`theory/layered`).
+- **Fixed fractions in ensembles.** A size distribution of layered spheres scales the whole
+  particle: each interface stays at a fixed fraction of the outer radius. A coating of fixed
+  thickness across the sizes, and fractions or layer indices as grid axes, are not available.
+- **Ensemble derivatives up to $x_L = 1000$.** The derivatives of an ensemble of layered
+  spheres with respect to their indices and fractions, and the radius derivative of a
+  monodisperse one, need every size node within the derivatives' domain: a distribution that
+  reaches a larger outer size parameter raises `OutsideValidatedDomain` unless
+  `allow_unvalidated` is set. Its values, and its derivatives with respect to the
+  distribution's own parameters, are validated up to $10^4$.
 
 ## Mie scattering
 
diff --git a/docs/plans/next-release-notes.md b/docs/plans/next-release-notes.md
index 2c36525..d9d83d7 100644
--- a/docs/plans/next-release-notes.md
+++ b/docs/plans/next-release-notes.md
@@ -3,7 +3,43 @@
 The GitHub release notes are the changelog (see "Releasing" in `CONTRIBUTING.md`). Paste this
 into the next release's notes, above GitHub's generated list, then delete this file.
 
-## Breaking changes
+## 0.3.0: layered spheres
+
+### Breaking changes
+
+- Rust: `scatrs::ensemble::Component`'s public field `refractive_index` is replaced by
+  `particle`, a `scatrs::ensemble::Particle` (`Particle::Sphere { refractive_index }` or
+  `Particle::LayeredSphere { radius_fractions, refractive_index }`). `Component::new(distribution,
+  refractive_index, weight)` still builds a component of spheres; `Component::with_particle`
+  takes any particle.
+- Python: the `refractive_index` of a `Mixture`'s result is a tuple with each component's array
+  (it was one stacked array, which components of different shapes no longer allow). Indexing
+  it by component works as before.
+
+### Added
+
+- Layered spheres: a core and any number of concentric shells (coated spheres among them).
+  - `scatrs.layered` (`coefficients`, `efficiencies`, `amplitudes`, `phase_matrix`, `greek`,
+    `default_n_max`, `parameters`) and, in Rust, `scatrs::layered::Layered`, with every output
+    of a sphere and closed-form derivatives with respect to each layer's size parameter and
+    refractive index.
+  - `scatrs.LayeredSphere(refractive_index=[...], radius_fraction=[...])` and
+    `LayeredSphere.coated(core, shell, core_fraction=...)`, in `Ensemble`, `Mixture` and grids
+    of distributions. The size distribution gives the outer radius, each interface at a fixed
+    fraction of it. Derivatives: `layer{l}.m_real`, `layer{l}.m_imag` and
+    `layer{l}.radius_fraction`.
+  - Rust: `Particle::layered_sphere`, `compute_with_particle`, `compute_batch_with_particle`,
+    `parameters_with_particle`, and `ParamKind::{LayerRefrReal, LayerRefrImag,
+    RadiusFraction}`.
+  - Validated against independent Arb references (direct interface matching) and dense
+    double-double accuracy maps: outer size parameters 10⁻⁶ to 10⁴ with |m|·x ≤ 3·10⁴, up to 16
+    layers, derivatives up to x = 1000. Compared with scattnlay.
+  - Documentation: the theory chapter "Layered spheres", the validation pages, and the example
+    "Coated black carbon".
+
+## 0.2.0 (the notes of v0.2.0: delete this section once they are on its GitHub release)
+
+### Breaking changes
 
 The functions of `scatrs.ensemble` are replaced by objects. Each removed call and its
 replacement:
@@ -23,7 +59,7 @@ replacement:
 The keyword options, the outputs, their values and the parameter names of `derivatives` are
 unchanged.
 
-## Added
+### Added
 
 - `scatrs.Sphere`, `scatrs.Ensemble` and `scatrs.Mixture`: immutable, hashable, picklable
   descriptions of particles, ensembles and mixtures, each ensemble and mixture with a
@@ -42,7 +78,7 @@ unchanged.
   about 3.7 times faster for x ≥ 100. Python's amplitude and phase-matrix functions use one
   for several spheres.
 
-## Changed
+### Changed
 
 - The Rust crate no longer depends on faer (54 crates): the ensembles' matrix products use
   `matrixmultiply` (2 crates), at the same speed on the machines measured. The crate's lint is
diff --git a/docs/quickstart.md b/docs/quickstart.md
index de392cb..621c366 100644
--- a/docs/quickstart.md
+++ b/docs/quickstart.md
@@ -110,7 +110,11 @@ print("alpha1 at 0.55 µm:", r.greek.alpha1[1, :4])
 ```
 
 A refractive index that varies with wavelength is a model from {mod}`scatrs.refractive`, such
-as a table of measured values ({doc}`theory/refractive-index`). A {class}`~scatrs.Mixture` combines
+as a table of measured values ({doc}`theory/refractive-index`). A
+{class}`~scatrs.LayeredSphere` is a core in concentric shells, such as black carbon coated in
+sulfate, with the core's radius 60% of the particle's:
+`scatrs.LayeredSphere.coated(1.95 + 0.79j, 1.43 + 1e-8j, core_fraction=0.6)`
+({doc}`examples/coated`). A {class}`~scatrs.Mixture` combines
 ensembles of different materials. Distribution parameters given as arrays make a grid of
 distributions, computed at once, as an optics table needs:
 
@@ -128,7 +132,8 @@ print("C_ext of r_g = 0.05 µm, σ_g = 1.6:", t.c_ext[0, 1])
 
 `derivatives=True` adds derivatives of every output, in closed form, at little extra cost:
 with respect to the size parameter and the real and imaginary parts of the refractive index
-for single spheres, and to the distribution parameters and refractive index for ensembles.
+for single spheres, and to the distribution parameters and the particle's refractive indices
+(and a layered sphere's radius fractions) for ensembles.
 
 ```{code-cell} python
 q = mie.efficiencies(x, m, derivatives=True)
diff --git a/docs/references.md b/docs/references.md
index 530d194..2005046 100644
--- a/docs/references.md
+++ b/docs/references.md
@@ -37,6 +37,12 @@
 - O. Peña and U. Pal, "Scattering of electromagnetic radiation by a multilayered sphere",
   *Computer Physics Communications* 180 (2009), 2348–2354. Yang's recursion in the form of
   scattnlay, the code of the comparison.
+- O. B. Toon and T. P. Ackerman, "Algorithms for the calculation of scattering by stratified
+  spheres", *Applied Optics* 20 (1981), 3657–3660. The coated sphere in atmospheric optics,
+  and the stability of its recurrences.
+- T. C. Bond and R. W. Bergstrom, "Light absorption by carbonaceous particles: an
+  investigative review", *Aerosol Science and Technology* 40 (2006), 27–67. The refractive
+  index of black carbon in {doc}`examples/coated`.
 
 **Quadrature**
 
diff --git a/docs/rust-api.md b/docs/rust-api.md
index 669517f..05c1c49 100644
--- a/docs/rust-api.md
+++ b/docs/rust-api.md
@@ -14,7 +14,7 @@ Add it with `cargo add scatrs`. The same reference is on [docs.rs](https://docs.
 |---|---|---|
 | {rust}`scatrs::mie` | {mod}`scatrs.mie` | One sphere: coefficients, efficiencies, amplitudes, phase matrix, Greek coefficients and their derivatives. |
 | {rust}`scatrs::layered` | {mod}`scatrs.layered` | One layered sphere (a core and concentric shells): the same outputs, with derivatives with respect to every layer. |
-| {rust}`scatrs::ensemble` | {class}`scatrs.Ensemble`, {class}`scatrs.Mixture` | Averages over size distributions, mixtures and batches (Python: grids), with error estimates and derivatives. |
+| {rust}`scatrs::ensemble` | {class}`scatrs.Ensemble`, {class}`scatrs.Mixture` | Averages of spheres and layered spheres over size distributions, mixtures and batches (Python: grids), with error estimates and derivatives. |
 | {rust}`scatrs::special` | {mod}`scatrs.special` | Gauss–Legendre quadrature and Wigner d-functions. |
 | {rust}`scatrs::scalar` | | The {rust}`Real <scatrs::scalar::Real>` trait and the double-double and dual-number types. Rust only. |
 | {rust}`scatrs::error` | {mod}`scatrs.errors` | The error type, and the exceptions that mirror it. |
diff --git a/docs/theory/ensembles.md b/docs/theory/ensembles.md
index 98d5fd8..cd50b89 100644
--- a/docs/theory/ensembles.md
+++ b/docs/theory/ensembles.md
@@ -1,17 +1,24 @@
-# Ensembles of spheres
+# Ensembles
 
 This page describes the methods behind {rust}`scatrs::ensemble` (Python: {class}`scatrs.Ensemble`
 and {class}`scatrs.Mixture`): the
 size distributions, their cut-offs, the adaptive size integration, the error estimates, the
 Jacobians and the mixtures. The rustdoc of {rust}`scatrs::ensemble` is the authoritative statement of
-the interface; the single-sphere quantities come from {doc}`mie`.
+the interface; the single-particle quantities come from {doc}`mie` for spheres and from
+{doc}`layered` for layered spheres.
 
 ## Averages
 
 <!-- synced with scatrs/src/ensemble/mod.rs rustdoc -->
 
 Radii and wavelengths are in the same (any) unit, and $x = 2\pi r/\lambda$. Each component of
-an ensemble has one refractive index per wavelength. Every output is a number average,
+an ensemble is a particle given at each wavelength: a sphere, with one refractive index per
+wavelength, or a layered sphere ({rust}`Particle::LayeredSphere
+<scatrs::ensemble::Particle::LayeredSphere>`, Python {class}`scatrs.LayeredSphere`), with one
+per layer and wavelength. The size distribution describes the radius the particle defines: a
+layered sphere's outer radius $r$, with layer $l$'s outer surface at the fixed fraction $f_l$
+of it ($x_l = f_l x$), so a distribution scales the whole particle. Everything below applies to
+either; the text says "sphere" for any particle. Every output is a number average,
 
 $$\langle C\rangle = \frac{\int C(r)\,n(r)\,dr}{\int n(r)\,dr}
 = \frac{\int C(e^t)\,n(e^t)\,e^t\,dt}{\int n(e^t)\,e^t\,dt}, \qquad t = \ln r,$$
@@ -235,13 +242,16 @@ $$S_1(\pm\mu) = G_E u \pm G_O v,\qquad S_2(\pm\mu) = G_E v \pm G_O u$$
 
 at the non-negative nodes $\mu$ (the Gauss–Legendre nodes are exactly symmetric). For the 15
 nodes of a panel these are two real matrix products, $G_E$ and $G_O$ ($n_q/2 \times N$) times
-the nodes' coefficients ($N \times 60$; 120 with the $m$-derivatives $\mathrm{d}a_n/\mathrm{d}m$,
-$\mathrm{d}b_n/\mathrm{d}m$). The products cost $O(N\,n_q)$, like sums sphere by sphere, but
+the nodes' coefficients ($N \times 60$; 60 more for each derivative column:
+$\mathrm{d}a_n/\mathrm{d}m$, $\mathrm{d}b_n/\mathrm{d}m$ of a sphere, of each layer of a
+layered sphere, and $x\,\partial a_n/\partial x_l$, $x\,\partial b_n/\partial x_l$ for each
+of its radius fractions). The products cost $O(N\,n_q)$, like sums sphere by sphere, but
 run far faster, and the tables agree with the per-sphere phase matrix to $10^{-12}$ of its
 largest value (a unit test).
 
 The sums over the nodes are matrix products too: each panel's sums for every part and block
-(values, each $\theta$, $\mathrm{Re}\,m$, $\mathrm{Im}\,m$) are the weights at the nodes
+(values, each $\theta$, each $\mathrm{Re}\,m$, $\mathrm{Im}\,m$ and radius fraction) are the
+weights at the nodes
 (Kronrod and Gauss) times the nodes' values. The projection likewise: a band's matrices
 $\tfrac{2l + 1}{2}w_k d_l(\mu_k)$ times the tables of every mixture (or batch distribution) and
 derivative at once, so the generalized spherical functions are computed once per band.
@@ -288,10 +298,16 @@ has a derivative.
   panels are held fixed within a call.
 - **Monodisperse radius.** $\partial(\pi r^2 Q)/\partial r = 2\pi r\,Q + \pi r^2\,(2\pi/\lambda)
   \,\partial Q/\partial x$, from the single-sphere derivatives with respect to $x$, and likewise
-  for the phase matrix.
-- **$\mathrm{Re}\,m$ and $\mathrm{Im}\,m$** of each component, at each wavelength: from the
-  single-sphere efficiency Jacobian and phase-matrix derivatives, integrated with the same
-  weights ($m$ does not change the density, so no renormalization term).
+  for the phase matrix. For a layered sphere, at fixed fractions, $\partial Q/\partial x =
+  \sum_l f_l\,\partial Q/\partial x_l$ ($f_{L-1} = 1$).
+- **$\mathrm{Re}\,m$ and $\mathrm{Im}\,m$** of each component (of each layer of a layered
+  sphere), at each wavelength: from the single-particle efficiency Jacobian and phase-matrix
+  derivatives, integrated with the same weights ($m$ does not change the density, so no
+  renormalization term).
+- **Radius fractions** of a layered sphere, at a fixed outer radius:
+  $\partial Q/\partial f_l = x\,\partial Q/\partial x_l$, likewise integrated with the same
+  weights. The derivatives with respect to the inner size parameters $x_l$ carry no term of
+  the outputs' scale, which only the outer radius sets.
 - **Weights.** A component $i$ (or mode $j$) enters with the number fraction
   $f = (w_i/\sum w)(v_j/\sum v)$, with $v_j$ the mode weights; its derivatives with respect to
   each weight follow from that product.
@@ -301,13 +317,18 @@ The tests check every scalar output, $\beta_1^l$ up to order 6 and $F_{12}$ at t
 against central differences of the whole computation at rtol $10^{-11}$, for the lognormal and
 gamma parameters, the modified gamma's $b$ and $\gamma$, the power-law exponent, the
 monodisperse radius, $\mathrm{Re}\,m$, $\mathrm{Im}\,m$, a mixture weight, the
-$\mathrm{Re}\,m$ of a mixture's second component, a mode weight and a mode's width.
+$\mathrm{Re}\,m$ of a mixture's second component, a mode weight and a mode's width; and for
+layered spheres $F_{11}$ too, with each layer's $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$ and the
+core fraction of a coated sphere, the middle layer's fraction and $\mathrm{Im}\,m$ of a
+three-layer one, the median radius, the monodisperse radius, and the core fraction of a
+mixture's coated component. An ensemble of one-layer layered spheres is the sphere ensemble
+bitwise, values, estimates and derivatives.
 
 ## Mixtures
 
 {rust}`compute_mixture <scatrs::ensemble::compute_mixture>` (Python {class}`scatrs.Mixture`)
-averages over external mixtures: components, each a distribution, its refractive indices and a
-weight $w_i \ge 0$ (a relative number concentration; their sum finite and positive).
+averages over external mixtures: components, each a distribution, its particle (spheres and
+layered spheres can mix) and a weight $w_i \ge 0$ (a relative number concentration; their sum finite and positive).
 {rust}`compute <scatrs::ensemble::compute>` is a mixture of one component, without the weight
 parameter. Cross sections and moments are averaged with the number fractions $w_i/\sum w$. The phase-matrix
 tables, in $C_\text{sca}$ units, are averaged with the same fractions, so the phase matrices
diff --git a/docs/validation/accuracy.md b/docs/validation/accuracy.md
index ad827f7..f19d4a0 100644
--- a/docs/validation/accuracy.md
+++ b/docs/validation/accuracy.md
@@ -257,7 +257,7 @@ rounding errors of the recurrences at the shell's two surfaces are alike from on
 next, so they add up in the forward sum, and the derivatives' domain ends at 1000.
 
 (ensemble-accuracy)=
-## Ensembles of spheres
+## Ensembles
 
 <!-- synced with scatrs/tests/ensemble.rs -->
 
@@ -269,10 +269,11 @@ solutions' own errors, declared above, are far smaller at the tolerances the ens
 and are not included.
 
 The test `error_estimates_bound_the_true_errors` in `scatrs/tests/ensemble.rs` checks the
-estimates for every change. It computes nine distributions at rtol $10^{-6}$ and again at rtol
-$10^{-11}$ (the reference), with backscattering refined, the Greek coefficients to order 24,
-the phase matrix at $\mu = 1, 0.7, 0, -1$, and the derivatives with respect to every parameter
-of the distribution and to $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$:
+estimates for every change. It computes nine distributions of spheres and two of layered
+spheres at rtol $10^{-6}$ and again at rtol $10^{-11}$ (the reference), with backscattering
+refined, the Greek coefficients to order 24, the phase matrix at $\mu = 1, 0.7, 0, -1$, and the
+derivatives with respect to every parameter of the distribution and of the particle
+($\mathrm{Re}\,m$ and $\mathrm{Im}\,m$ of every layer, and the radius fractions):
 
 | Distribution | $m$ | $\lambda$ |
 |---|---|---|
@@ -285,6 +286,8 @@ of the distribution and to $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$:
 | power law, $\alpha = 3.5$ on $[0.01, 5]$ | $1.5 + 0.01i$ | 0.5 |
 | tabulated, $(0.05, 0.1, 0.4, 1)$, densities $(0, 3, 1, 0.2)$ | $2 + 0.5i$ | 0.4 |
 | multimodal: weight 1000, lognormal $r_g = 0.05$, $\sigma_g = 1.5$; weight 1, lognormal $r_g = 1$, $\sigma_g = 1.4$ | $1.53 + 0.005i$ | 0.55 |
+| lognormal, $r_g = 0.1$, $\sigma_g = 1.6$, coated (core fraction 0.6) | core $1.95 + 0.79i$, shell $1.43 + 10^{-8}i$ | 0.55 |
+| gamma, $a = 0.8$, $b = 0.1$, three layers (fractions 0.3, 0.8) | $1.6 + 0.3i$, $2 + 0.05i$, $1.33 + 10^{-6}i$ | 0.5 |
 
 Both results must converge, and the refined cross sections must be within rtol of the
 reference ($g$ within $2\cdot10^{-6}$). The true errors of the refined outputs
diff --git a/scatrs/src/lib.rs b/scatrs/src/lib.rs
index ac0df4f..eba745d 100644
--- a/scatrs/src/lib.rs
+++ b/scatrs/src/lib.rs
@@ -11,7 +11,7 @@
 //! |---|---|
 //! | [`mie`] | One sphere: Mie coefficients, efficiencies, amplitudes, phase matrix and Greek coefficients, with their derivatives. |
 //! | [`layered`] | One layered sphere, a core and concentric shells: the same outputs, with derivatives with respect to every layer. |
-//! | [`ensemble`] | Averages over size distributions, external mixtures and batches of distributions, with error estimates and derivatives. |
+//! | [`ensemble`] | Averages of spheres and layered spheres over size distributions, external mixtures and batches of distributions, with error estimates and derivatives. |
 //! | [`special`] | Gauss–Legendre quadrature and Wigner d-functions. |
 //! | [`scalar`] | The [`Real`] trait the numerical kernels are generic over, and its double-double and dual-number types. |
 //! | [`error`] | The [`Error`] that every fallible function returns. |
diff --git a/scatrs/tests/ensemble.rs b/scatrs/tests/ensemble.rs
index 844bfad..647d79f 100644
--- a/scatrs/tests/ensemble.rs
+++ b/scatrs/tests/ensemble.rs
@@ -9,7 +9,8 @@ use num_complex::Complex;
 use scatrs::Error;
 use scatrs::ensemble::{
     Component, EnsembleOptions, EnsembleParam, EnsembleResult, GreekOrder, Outputs, ParamKind,
-    SizeDistribution, compute, compute_batch, compute_mixture, mixture_parameters, parameters,
+    Particle, SizeDistribution, compute, compute_batch, compute_mixture, compute_with_particle,
+    mixture_parameters, parameters, parameters_with_particle,
 };
 use scatrs::mie::{GreekCoefficients, Mie, MieOptions, Normalization, PhaseBuffers};
 use scatrs::special::gauss_legendre;
@@ -513,6 +514,29 @@ fn error_estimates_bound_the_true_errors() {
             0.55,
         ),
     ];
+    // And layered spheres: black carbon coated in sulfate, and three layers.
+    let layered = [
+        (
+            lognormal(0.1, 1.6),
+            Particle::layered_sphere(vec![0.6], vec![vec![c(1.95, 0.79)], vec![c(1.43, 1e-8)]]),
+            0.55,
+        ),
+        (
+            SizeDistribution::Gamma {
+                effective_radius: 0.8,
+                effective_variance: 0.1,
+            },
+            Particle::layered_sphere(
+                vec![0.3, 0.8],
+                vec![vec![c(1.6, 0.3)], vec![c(2.0, 0.05)], vec![c(1.33, 1e-6)]],
+            ),
+            0.5,
+        ),
+    ];
+    let cases = cases
+        .into_iter()
+        .map(|(d, m, lambda)| (d, Particle::sphere(vec![m]), lambda))
+        .chain(layered);
     let mu = vec![1.0, 0.7, 0.0, -1.0];
     // The worst ratio of true error to estimate, of refined and of other outputs.
     let worst = std::cell::Cell::new([0.0f64; 2]);
@@ -524,21 +548,21 @@ fn error_estimates_bound_the_true_errors() {
         }
         worst.set(w);
     };
-    for (d, m, lambda) in cases {
-        let params = parameters(&d);
+    for (d, p, lambda) in cases {
+        let params = parameters_with_particle(&d, &p);
         // With backscattering refined, so that every estimate is a bound.
         let opts = EnsembleOptions::new()
             .with_greek(GreekOrder::Order(24))
             .with_mu(mu.clone())
             .with_derivatives(params.clone())
             .with_refine_backscatter(true);
-        let a = one(&d, lambda, m, &opts);
-        let b = one(
-            &d,
-            lambda,
-            m,
-            &opts.clone().with_rtol(1e-11).with_max_panels(50_000),
-        );
+        let run = |o: &EnsembleOptions| {
+            compute_with_particle(&d, &[lambda], &p, o)
+                .unwrap()
+                .remove(0)
+        };
+        let a = run(&opts);
+        let b = run(&opts.clone().with_rtol(1e-11).with_max_panels(50_000));
         assert!(a.converged && b.converged, "{d:?}");
         let check = |x: &Outputs, y: &Outputs, e: &Outputs, what: &str| {
             let (xs, ys, es) = (scalars(x), scalars(y), scalars(e));
````

### Task 5: Benchmark of coated-sphere ensembles

- [ ] Add `w1-coated` and `w1-coated-derivatives` to `benches/ensemble.rs`. The latter allows unvalidated sizes: w1 reaches beyond x_L = 1000.
- [ ] Record the ratios in the ensembles theory chapter.

The commit (`9b66a65`):

````diff
diff --git a/docs/theory/ensembles.md b/docs/theory/ensembles.md
index cd50b89..19639de 100644
--- a/docs/theory/ensembles.md
+++ b/docs/theory/ensembles.md
@@ -389,5 +389,13 @@ $m = 1.33 + 10^{-4}i$, at rtol $10^{-6}$ and the other options default:
 | 0.3 | 4092 | 0.20 s | 0.90 s |
 
 All three converge. A narrow fine-mode aerosol takes milliseconds ({doc}`/examples/ensemble`).
+
+Layered spheres cost what their single-particle solutions cost more ({doc}`layered`): the same
+distribution of coated spheres (a $1.95 + 0.79i$ core of 0.6 of the radius) at the three
+wavelengths, Greek coefficients to order 63 and rtol $10^{-4}$ (`cargo bench -p scatrs --bench
+ensemble`), takes 1.8 times the spheres' time, on about as many panels. Derivatives cost more
+per node, and their refinement can take more panels: here the derivative with respect to the
+weakly absorbing shell's $\mathrm{Im}\,m$, like a sphere's, reaches `max_panels` at every
+wavelength, and with every derivative the coated spheres take 3.5 times the spheres' time.
 {doc}`/validation/sasktran2` compares the speed and accuracy with sasktran2's at matched
 accuracy, for one distribution and for a database grid.
diff --git a/scatrs/benches/ensemble.rs b/scatrs/benches/ensemble.rs
index 22c2fc9..6e8f3dc 100644
--- a/scatrs/benches/ensemble.rs
+++ b/scatrs/benches/ensemble.rs
@@ -4,6 +4,9 @@
 //! * `w1`: one broad lognormal (r_g = 1 µm, σ_g = 2) at 0.3, 0.5 and 1 µm, m = 1.33 + 1e-4i;
 //! * `w1-derivatives`: the same with the derivatives with respect to every parameter;
 //! * `w1-mu`: the same with the phase matrix at 181 angles instead;
+//! * `w1-coated`, `w1-coated-derivatives`: `w1` of coated spheres (a 1.95 + 0.79i core of 0.6
+//!   of the radius, the shell 1.33 + 1e-4i), without and with every derivative (the latter
+//!   reaches beyond the derivatives' validated x_L = 1000, so it allows unvalidated sizes);
 //! * `large`: a lognormal of large spheres (r_g = 5 µm, σ_g = 1.5) at 0.5 µm;
 //! * `batch`: 12 lognormals (r_g in [0.05, 1] µm × σ_g in {1.2, 1.6, 2.0}) at 5 wavelengths in
 //!   [0.28, 1.02] µm, m = 1.43 + 1e-8i, by `compute_batch`.
@@ -19,7 +22,8 @@ use std::hint::black_box;
 use criterion::{Criterion, criterion_group, criterion_main};
 use num_complex::Complex;
 use scatrs::ensemble::{
-    EnsembleOptions, GreekOrder, SizeDistribution, compute, compute_batch, parameters,
+    EnsembleOptions, GreekOrder, Particle, SizeDistribution, compute, compute_batch,
+    compute_with_particle, parameters, parameters_with_particle,
 };
 
 fn ensemble(c: &mut Criterion) {
@@ -41,6 +45,30 @@ fn ensemble(c: &mut Criterion) {
     group.bench_function("w1-derivatives", |b| {
         b.iter(|| compute(black_box(&broad), &wavelengths, &m, &with_derivatives).unwrap())
     });
+    let coated = Particle::layered_sphere(
+        vec![0.6],
+        vec![vec![Complex::new(1.95, 0.79); 3], m.to_vec()],
+    );
+    group.bench_function("w1-coated", |b| {
+        b.iter(|| {
+            compute_with_particle(black_box(&broad), &wavelengths, &coated, &options).unwrap()
+        })
+    });
+    let coated_derivatives = options
+        .clone()
+        .with_derivatives(parameters_with_particle(&broad, &coated))
+        .with_allow_unvalidated(true);
+    group.bench_function("w1-coated-derivatives", |b| {
+        b.iter(|| {
+            compute_with_particle(
+                black_box(&broad),
+                &wavelengths,
+                &coated,
+                &coated_derivatives,
+            )
+            .unwrap()
+        })
+    });
     let mu: Vec<f64> = (0..181)
         .map(|k| (f64::from(k).to_radians()).cos())
         .collect();
````

## Hand-off

- **Derivatives above x_L = 1000.** `docs/plans/phase-3b-followups.md` has the analysis: rounding errors that are coherent across orders, at a thin shell's two surfaces.
- **Not in phase 3** (spec section 1.4):
  - fractions or layer indices as grid axes;
  - a fixed shell thickness across a distribution;
  - the parameters of each layer's refractive-index model (phase 2c-2).
- **Release.** The `Component` field change is breaking, so the next release is 0.3.0. The notes are drafted in `docs/plans/next-release-notes.md`, above v0.2.0's, which were still in the file.
- **Phase 4** (spheroids, T-matrix):
  - shape parameters take the real-direction mechanism of the radius fractions;
  - the engine's `Solver` gains a direct path (spec section 15).
