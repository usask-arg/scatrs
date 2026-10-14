# Phase 1b: Sphere Phase Matrix and Exact Greek Coefficients — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The phase matrix of a sphere at any cosines and its exact Greek expansion coefficients up to any order, with Jacobians, in Rust and Python, validated like phase 1a.

**Architecture:** An O(n) Gauss–Legendre rule (Bogaert's expansions) for `f64` above 100 points; `MieSolution::phase_matrix` from the existing amplitudes; a reusable `GreekProjector` that streams the four Wigner recurrences over its nodes and projects any number of phase-matrix tables (values or derivatives) at once; `MieSolution::greek` choosing N + ⌈(L + 1)/2⌉ nodes so the projection is exact up to rounding. Accuracy follows the 1a model, C·(κ′ + cond)·2⁻⁵², with new outputs and a quadrature term for the Greek coefficients.

**Tech Stack:** Rust 2024 (MSRV 1.85), pyo3/numpy 0.29 (abi3), python-flint (Arb) for references, Sphinx/MyST-NB docs.

**Spec:** `docs/specs/2026-10-13-phase-1b-greek-design.md` (with the changes recorded in its section 9).

## Global Constraints

- MSRV 1.85, edition 2024: no let-chains (`if let … && …`), no APIs newer than 1.85.
- No 32-bit targets (dropped in 8719288).
- `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `ruff check`, `ruff format --check`, `mypy.stubtest scatrs._scatrs` and `sphinx-build -E -W --keep-going` must pass at every commit, as must `cargo test --workspace`, `uv run pytest`, and `uv run --directory reference python -m pytest`.
- Every quoted figure in `docs/` that `tests/test_docs_consistency.py` tracks must change everywhere at once.
- Declared constants are powers of two, at least 4 times the worst C measured (map and Arb).
- No platform libm in anything whose bits are documented as platform-independent (Gauss–Legendre).
- Never send personal data (such as the user's email) in network requests.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## How this plan was made, and how to execute it

Tasks 1–4 were prototyped and verified before this plan was written: each was committed on a
scratch branch with fmt, clippy, the workspace tests, ruff, pytest, stubtest and the docs build
green. Their code below is the exact diff of those commits (generated data files excepted: the
steps regenerate them). Executing a task means applying its diff, regenerating its data, running
its checks, and committing; reviewers then check it against this plan and the spec. Task 5 (the
narrative documentation) was not prototyped and is written from the requirements below.

## Measured results the plan relies on

| What | Measured | Declared |
|---|---|---|
| Asymptotic Gauss–Legendre, n ≥ 101 (DD and Arb, up to 2¹⁷) | node 2.4·2⁻⁵², weight 3.6·2⁻⁵² | 16·2⁻⁵² |
| Asymptotic rule time (Apple M4 Pro) | 1 ms at n = 1000, 13 ms at 20 000, 40 ms at 2¹⁷ | — |
| Phase matrix, C | 7.4 (map), 0.59 (Arb) | 32 |
| Phase-matrix derivatives, C | 22 (map) | 128 |
| Greek coefficients, C | 1.7 (map), 0.72 (Arb), 0.22 (x = 2·10⁴) | 8 |
| Greek derivatives, C | 2.8 (map) | 16 |
| `DoubleDouble` against Arb (units of 2⁻¹⁰⁴) | phase 0.38, Greek 0.35 | 16× the constants |
| Greek map (test profile, 12 cores) | 12 s | — |
| Greek, all orders, `f64` | 0.09 s at x = 10³, 7.6 s at x = 10⁴ | — |

Why a quadrature term: the f64 nodes are the exact Gauss nodes rounded, so the coefficients are
those of a rule with nodes moved by half an ulp. Next to μ = 1 that changes d_L by about
L²·2⁻⁵⁴, and the forward peak (slope ∝ x⁴ in μ, weights ∝ 1/x² there) by about x²·2⁻⁵⁴ of its
size. Without the term the Greek C reached 85 at x = 100 (all orders) and 100 at x = 2·10⁴
(L = 64), growing with L² and x²; with ((L + 1)² + x²)/8 it is at most 1.7 everywhere.
Forcing the double-double rule changes nothing (the rounding of its nodes to `f64` is the same).

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `scatrs/src/special/gauss_legendre_asymptotic.rs` (new) | 1 | Bogaert's expansions, Bessel tables, McMahon |
| `scatrs/src/special/gauss_legendre.rs` | 1 | dispatch (threshold 100), `#[non_exhaustive]`, `gauss_legendre_into`, docs |
| `scatrs/src/mie/phase.rs` (new) | 2 | `Normalization`, `PhaseBuffers`, `phase_matrix[_with_derivatives]` |
| `scatrs/src/mie/greek.rs` (new) | 2 | `GreekCoefficients`, `PhaseTables`, `GreekProjector`, `MAX_GREEK_ORDER`, `greek[_with_derivatives]` |
| `scatrs/tests/common/greek.rs` (new) | 2 | solving one sphere for both outputs; error measures |
| `scatrs/tests/mie_greek.rs` (new) | 2 | identities, exactness, `Dual` oracle, Rayleigh signs, errors |
| `reference/scatrs_reference/generators/mie_greek.py` (new) | 3 | Arb phase matrix and Greek coefficients |
| `scatrs/src/mie/domain.rs` | 3 | four `MieOutput` variants, priors, constants |
| `scatrs/tests/common/check.rs` (new) | 3 | `Worst`, `Parts`, `records` moved out of `mie_reference.rs` |
| `scatrs/tests/mie_greek_reference.rs` (new) | 3 | f64 and DD against Arb |
| `scatrs/tests/mie_greek_accuracy_map.rs` (new) | 3 | map (x ≤ 10³) and ignored large-sphere test |
| `scatrs-py/src/mie.rs`, `python/scatrs/mie.py`, stubs | 4 | `mie_phase_matrix`, `mie_greek`; `phase_matrix`, `greek`, result classes |
| `docs/…` | 1, 3, 5 | GL theory and limitations (1); constants, reference suite (3); Mie theory, example, accuracy, limitations (5) |

---

### Task 1: Gauss–Legendre in O(n) for `f64`

**Files:**
- Create: `scatrs/src/special/gauss_legendre_asymptotic.rs`
- Modify: `scatrs/src/special/gauss_legendre.rs`, `scatrs/src/special/mod.rs`, `scatrs/tests/gauss_legendre.rs`, `python/scatrs/special.py`, `tests/test_special.py`, `tests/test_docs_consistency.py`, `docs/theory/special-functions.md`, `docs/limitations.md`, `docs/examples/quadrature.md`

**Interfaces:**
- Consumes: `DoubleDouble::sin_cos`, `DoubleDouble::cos` (phase 0).
- Produces: `scatrs::special::gauss_legendre_into<T: Real>(nodes: &mut [T], weights: &mut [T]) -> Result<()>`; `GaussLegendre<T>` is `#[non_exhaustive]` (fields `nodes`, `weights` unchanged); `gauss_legendre::<f64>(n)` for n > 100 uses the asymptotic rule (16·2⁻⁵² contract). `pub(super) const ASYMPTOTIC_THRESHOLD: usize = 100`; `Recurrence::{new, root_and_weight}` become `pub(super)` for the asymptotic module's tests.

- [ ] **Step 1: Apply the diff** (new module, dispatch, tests updated to the two-method contract, docs and contracts):

````diff
diff --git a/docs/examples/quadrature.md b/docs/examples/quadrature.md
index 219659b..3315b99 100644
--- a/docs/examples/quadrature.md
+++ b/docs/examples/quadrature.md
@@ -74,9 +74,9 @@ The {doc}`reference suite <../validation/reference-suite>` stores the 1000-point
 with Arb. Each value $v$ is stored as three doubles `hi + lo + lo2`, with a rigorous bound `err`
 on the remainder. For an `f64` value $g$, the cell computes
 $\lvert g - (\text{hi} + \text{lo} + \text{lo2})\rvert + \text{err}$ with rational
-arithmetic. That is a rigorous and nearly tight bound on $\lvert g - v\rvert$. The contract is
-an absolute error of at most $2^{-52}$ for nodes and a relative error of at most $2^{-52}$ for
-weights.
+arithmetic. That is a rigorous and nearly tight bound on $\lvert g - v\rvert$. Above 100
+points the rule comes from asymptotic expansions, and the contract is an absolute error of at
+most $16\cdot2^{-52}$ for nodes and a relative error of at most $16\cdot2^{-52}$ for weights.
 
 For comparison, the cell also measures {func}`numpy.polynomial.legendre.leggauss`, which works
 in `f64` throughout.
@@ -117,16 +117,17 @@ for name, (xs, ws) in {"scatrs": (x, w), f"numpy {np.__version__}": (xn, wn)}.it
     first = ", ".join(f"{e:.3g}" for e in weight_err[:3])
     print(f"{name:>12} {node_err.max():11.3g} {weight_err.max():13.3g}   {first}")
     if name == "scatrs":
-        assert node_err.max() <= 1 and weight_err.max() <= 1
+        assert node_err.max() <= 16 and weight_err.max() <= 16
 ```
 
 Next to $\pm1$ the weights are ill-conditioned in the nodes. A node error $\delta$ changes the
 weight by a relative $2\lvert x\rvert\delta/(1 - x^2)$, and $1 - x^2 \approx 5.8/n^2$ at the
 extreme nodes. That is about $n^2/6$ units of $2^{-52}$ per ulp of node error. So a weight
 computed from an `f64` node loses up to that much, even from the correctly rounded node: about
-$7\cdot10^4$ units at $n = 1000$. scatrs avoids this by computing
-the rule in double-double arithmetic and rounding it once. This matters next to $+1$, where
-the Mie forward-scattering peak is integrated.
+$7\cdot10^4$ units at $n = 1000$. scatrs avoids this: up to 100 points it computes the rule
+in double-double arithmetic and rounds it once, and above it computes each weight from an
+asymptotic expansion in the angle of the node rather than from the node. This matters next to
+$+1$, where the Mie forward-scattering peak is integrated.
 
 numpy's error, $3.7\cdot10^7$ units here, is much larger, and the difference is algorithmic.
 `leggauss` computes the weights as $2/\bigl(n\,P_{n-1}(x)\,P_n'(x)\bigr)$, up to a final
diff --git a/docs/limitations.md b/docs/limitations.md
index cf26795..3308d64 100644
--- a/docs/limitations.md
+++ b/docs/limitations.md
@@ -38,14 +38,13 @@ it, in `docs/plans/phase-0-followups.md`.
 
 ## Gauss–Legendre quadrature
 
-- **Cost.** The method is $O(n^2)$. On an Apple M4 Pro it takes about 11 s at $n$ = 20 000,
-  and an extrapolated 8 minutes at the cap of $2^{17}$ points. Mie quadrature at size parameter
-  $2\cdot10^4$ needs about $4\cdot10^4$ points, which takes about 45 s.
-- **Interruption.** A call cannot be interrupted. In Python, Ctrl+C takes effect only once it
-  returns. Compute a rule once and reuse it.
-- **Planned speed-ups.** Two tunings of the present method measured about 5× faster together
-  during code review (not committed). The real fix is an $O(n)$ method (Bogaert 2014). Faster
-  rules are a priority for phase 1b, and the cap will be raised once the $O(n)$ method lands.
+- **Accuracy above 100 points.** `f64` rules above 100 points come from asymptotic expansions,
+  within $16\cdot2^{-52}$ (measured: 2.4 and 3.6 units of $2^{-52}$ for nodes and weights)
+  rather than the half ulp of the double-double rule rounded once.
+- **Double-double cost.** Double-double rules use the $O(n^2)$ Newton method at every $n$. On
+  an Apple M4 Pro that takes about 11 s at $n$ = 20 000, and an extrapolated 8 minutes at the
+  cap of $2^{17}$ points. A call cannot be interrupted; compute a double-double rule once and
+  reuse it.
 
 ## Builds without FMA
 
diff --git a/docs/theory/special-functions.md b/docs/theory/special-functions.md
index 11f6990..878efd6 100644
--- a/docs/theory/special-functions.md
+++ b/docs/theory/special-functions.md
@@ -21,8 +21,13 @@ $$w_k = \frac{2}{(1 - x_k^2)\,P_n'(x_k)^2}.$$
 
 <!-- synced with scatrs/src/special/gauss_legendre.rs rustdoc -->
 
-`gauss_legendre::<T>(n)` finds the $\lfloor n/2 \rfloor$ positive roots by Newton iteration in
-double-double arithmetic, and mirrors them. For odd $n$ the middle node is 0.
+`gauss_legendre::<T>(n)` uses one of two methods. Up to 100 points, and at every $n$ for
+double-double precision, it finds the roots by Newton iteration in double-double arithmetic.
+Above 100 points, `f64`-precision rules (`f64` and `Dual<f64, N>`) come from asymptotic
+expansions in $O(n)$. Both compute the $\lceil n/2 \rceil$ nodes nearest $+1$ and mirror
+them; for odd $n$ the middle node is 0.
+
+**Newton iteration** (double-double):
 
 - **Start.** Each root starts from Tricomi's approximation
 
@@ -40,25 +45,43 @@ double-double arithmetic, and mirrors them. For odd $n$ the middle node is 0.
   rounding level. Failing that within 100 iterations returns `NotConverged`.
 - **Weight.** The weight is computed in double-double from the root and $P_n'$ there.
 - **Rounding.** The double-double rule is converted once with `Real::from_f64_pair`, so every
-  type gets the same rule:
-  - the `f64` rule is the double-double rule rounded to nearest;
-  - a `Dual` rule is a constant (zero derivatives), whose values are bitwise those of `f64` or
-    `DoubleDouble`.
+  type gets the same rule: the `f64` rule (for $n \le 100$) is the double-double rule rounded
+  to nearest.
+
+**Asymptotic expansions** (Bogaert 2014). With $v = 1/(n + \tfrac12)$, $j_k$ the $k$-th zero
+of the Bessel function $J_0$ and $a_k = j_k v$, the $k$-th node from $+1$ is $\cos\theta_k$
+with
+
+$$\theta_k = a_k + v^2 t_1 + v^4 t_2 + v^6 t_3,$$
+
+and its weight is
+
+$$w_k = \frac{2}{J_1(j_k)^2\,(a_k/\sin a_k)\,\bigl(v^{-2} + W_1 + v^2 W_2 + v^4 W_3\bigr)},$$
+
+where $t_i$ and $W_i$ are polynomials in $\cot a_k$ and $1/a_k$. The coefficients are those of
+FastGaussQuadrature.jl, which keeps fewer terms for larger $n$; scatrs keeps them all. The
+first 20 zeros $j_k$ and the first 10 values $J_1(j_k)^2$ are tabulated (computed with Arb);
+beyond them McMahon's expansion and its counterpart for $J_1(j_k)^2$ take over, already
+within $2\cdot2^{-52}$ where the tables end. The weight comes from the angle, not from the
+node, so it does not inherit the conditioning of $1 - x^2$ described below.
 
-Nodes ascend and the rule is exactly symmetric. The rule uses only IEEE 754 operations and
-fused multiply-adds, with no platform maths library, so its bits are designed not to depend on
-the platform. CI runs the tests on Linux, macOS and Windows.
+`Dual` rules are constants (zero derivatives), whose values are bitwise those of `f64` or
+`DoubleDouble`. Nodes ascend and the rule is exactly symmetric. Both methods use only IEEE 754
+operations and fused multiply-adds, with the double-double sine and cosine instead of the
+platform maths library, so the bits of the rule are designed not to depend on the platform. CI
+runs the tests on Linux, macOS and Windows.
 
-### Why double-double
+### Why not plain `f64`
 
 Next to $\pm1$ the weights are ill-conditioned in the nodes. A node error $\delta$ changes the
 weight by a relative $2\lvert x\rvert\delta/(1 - x^2)$, and at the extreme nodes
 $1 - x^2 \approx 5.8/n^2$. That is about $n^2/6$ units of $2^{-52}$ per ulp of node error. So
 a weight computed from an `f64` node can be off by up to that much, even when the node is
 correctly rounded: about $7\cdot10^4$ units at $n = 1000$. Next to $+1$ is where the Mie
-forward-scattering peak sits. Computed in double-double and rounded once, the `f64` weights are
-instead within about half an ulp. The {doc}`quadrature example <../examples/quadrature>`
-measures this against Arb.
+forward-scattering peak sits. Computed in double-double and rounded once, the `f64` weights of
+the Newton method are within about half an ulp; the asymptotic weights, computed from the
+angle, within a few. The {doc}`quadrature example <../examples/quadrature>` measures this
+against Arb.
 
 ### Accuracy
 
@@ -67,53 +90,57 @@ measures this against Arb.
 The tests check the bounds below against Arb for every node of the full reference rules
 ($n$ = 1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 128, 257, 500 and 1000). They also check them for
 the 8 nodes nearest each of $\pm1$ and the 4 nearest 0 at $n$ = 5000, 20 000 and $2^{17}$, the
-cap. The $n$ = 20 000 rule takes about 11 s, so its check is an ignored test, which CI runs
-weekly. At the cap the test computes only the individual roots. Other values of $n$ are not
-individually checked against Arb.
+cap, and the asymptotic rules against the double-double rules node by node for nine values of
+$n$ from 101 to 4001. The double-double rule at $n$ = 20 000 takes about 11 s, so its check is
+an ignored test, which CI runs weekly. At the cap the test computes only the individual
+double-double roots.
 
 | Type | Nodes: absolute error | Weights: relative error | Measured worst |
 |---|---|---|---|
-| `f64` | $\le 2^{-52}$ | $\le 2^{-52}$ | $0.25\cdot2^{-52}$ for nodes and $0.49\cdot2^{-52}$ for weights: the rounding of the double-double rule |
+| `f64`, $n \le 100$ | $\le 2^{-52}$ | $\le 2^{-52}$ | $0.25\cdot2^{-52}$ for nodes and $0.49\cdot2^{-52}$ for weights: the rounding of the double-double rule |
+| `f64`, $n > 100$ | $\le 16\cdot2^{-52}$ | $\le 16\cdot2^{-52}$ | $2.4\cdot2^{-52}$ for nodes and $3.6\cdot2^{-52}$ for weights |
 | `DoubleDouble` | $\le 2^{-98}$ | $\le 2^{-100}(4 + n^2)$ | $0.51\cdot2^{-106}$ for nodes; about $2^{-85}$, $2^{-82}$ and $2^{-80}$ for weights at $n$ = 5000, 20 000 and $2^{17}$ |
 
 The $n^2$ in the double-double weight bound is the same conditioning of $1 - x^2$, now at
 double-double precision.
 
-The tests also check that Newton finds the right root, whatever its accuracy. By Bruns's
+The tests also check that each method finds the right root, whatever its accuracy. By Bruns's
 inequality, root $j$ lies in its own interval of $\theta$, $(j - \tfrac12)\pi/(n + \tfrac12)
 < \theta_j < j\pi/(n + \tfrac12)$, and these intervals are disjoint. The tests check that
-every root of every rule with $n \le 300$ lies inside its interval, and so do the roots nearest
-$\pm1$ and 0 for over a hundred values of $n$ up to the cap. An ignored test extends the first
-check to $n \le 3000$. None of these checks has found a rule that fails to converge.
+every root of every `f64` rule with $n \le 300$ lies inside its interval, and so does every
+root of a dozen rules up to the cap; for the Newton method, so do the roots nearest $\pm1$ and
+0 for over a hundred values of $n$ up to the cap. An ignored test extends the first check to
+$n \le 3000$. None of these checks has found a rule that fails.
 
 ### Cost and the cap
 
 <!-- synced with scatrs/src/special/gauss_legendre.rs rustdoc -->
 
-The method is $O(n^2)$ in time and $O(n)$ in memory. Each root takes four to seven
-evaluations of the recurrence, about four on average for $n \ge 5000$. On an Apple M4 Pro
-this comes to about $28\,\text{ns}\cdot n^2$ for large $n$. These timings were measured
-during development; the repository has no benchmarks yet.
+The asymptotic method is $O(n)$ in time and memory; the Newton method is $O(n^2)$ in time.
+Each Newton root takes four to seven evaluations of the recurrence, about four on average for
+$n \ge 5000$. On an Apple M4 Pro this comes to about $28\,\text{ns}\cdot n^2$ for large $n$.
+These timings were measured during development; the repository has no benchmarks for them.
 
 | $n$ | 100 | 1000 | 5000 | 20 000 | $2^{17}$ (cap) |
 |---|---|---|---|---|---|
-| Time, measured | 0.8 ms | 32 ms | 0.72 s | 11 s | about 8 min (extrapolated) |
+| `f64`, measured | 0.8 ms | 1 ms | 5 ms | 13 ms | 40 ms |
+| `DoubleDouble`, measured | 0.8 ms | 32 ms | 0.72 s | 11 s | about 8 min (extrapolated) |
 
 - **Without FMA.** x86_64 builds without the `fma` target feature take roughly ten times
-  longer, because every double-double operation then calls a software `mul_add`.
-- **Memory.** A call takes `(32 + 2·size_of::<T>())·n` bytes, 32 bytes per point of
-  recurrence coefficients besides the rule itself: 6.3 MB for `f64` at the cap, and 277 MB
-  for `Dual<DoubleDouble, 64>`.
+  longer in double-double, because every double-double operation then calls a software
+  `mul_add`.
+- **Memory.** A double-double call takes `(32 + 2·size_of::<T>())·n` bytes, 32 bytes per point
+  of recurrence coefficients besides the rule itself: 277 MB for `Dual<DoubleDouble, 64>` at
+  the cap.
 - **No interruption.** A call cannot be interrupted. In Python it releases the GIL, but Ctrl+C
   takes effect only when it returns.
 
 `MAX_GAUSS_LEGENDRE_POINTS` caps $n$ at $2^{17} = 131\,072$; a larger $n$ returns
 `InvalidInput`. That is about three times the roughly $4\cdot10^4$ points that Mie quadrature
-needs at size parameter $2\cdot10^4$. The cap will be raised, never lowered, when an $O(n)$
-method replaces the present one.
+needs at size parameter $2\cdot10^4$. The cap may be raised, never lowered.
 
-Compute a rule once and reuse it. A rule that is exact to degree $D$ is also exact for every
-lower degree.
+A rule that is exact to degree $D$ is also exact for every lower degree, so a double-double
+rule is worth computing once and reusing.
 
 (wigner-d)=
 ## Wigner d-functions
diff --git a/python/scatrs/special.py b/python/scatrs/special.py
index 171eab9..bdcaf9d 100644
--- a/python/scatrs/special.py
+++ b/python/scatrs/special.py
@@ -56,14 +56,14 @@ def gauss_legendre(n: SupportsIndex) -> tuple[NDArray[np.float64], NDArray[np.fl
 
     Notes
     -----
-    The rule is computed in double-double arithmetic and rounded once, so nodes and weights
-    are accurate to about half an ulp, and the rule is exactly symmetric.
-
-    The cost is ``O(n**2)`` time and ``O(n)`` memory: about ``28 ns * n**2`` for large n on an
-    Apple M4 Pro, measured as 32 ms at n = 1000, 0.72 s at n = 5000 and 11 s at n = 20000, and
-    about 8 minutes at the cap; x86-64 builds without FMA are roughly ten times slower. The
-    call releases the GIL, so other Python threads keep running, but it cannot be interrupted:
-    Ctrl+C takes effect only once it returns. Compute a rule once and reuse it.
+    For ``n <= 100`` the rule is computed in double-double arithmetic and rounded once, so
+    nodes and weights are accurate to about half an ulp. Above, it comes from Bogaert's (2014)
+    asymptotic expansions, with node absolute errors and weight relative errors of at most
+    ``16 * 2**-52`` (measured: 2.4 and 3.6 units of ``2**-52``). Either way the weights next to
+    +-1 are as accurate as the others, and the rule is exactly symmetric.
+
+    The cost is ``O(n)``: about 1 ms at n = 1000 and 40 ms at the cap on an Apple M4 Pro. The
+    call releases the GIL, so other Python threads keep running.
     """
     return _scatrs.gauss_legendre(_args.size(n, "n"))
 
diff --git a/scatrs/src/special/gauss_legendre.rs b/scatrs/src/special/gauss_legendre.rs
index 82baae6..d6108f7 100644
--- a/scatrs/src/special/gauss_legendre.rs
+++ b/scatrs/src/special/gauss_legendre.rs
@@ -1,19 +1,24 @@
 //! Gauss–Legendre quadrature on [−1, 1].
 //!
-//! Nodes come from Newton iteration on Pₙ in double-double arithmetic, started from
-//! Tricomi's approximation, and are then rounded to the requested type. Next to ±1 the weights
-//! are ill-conditioned in the nodes: a node error of one ulp changes the weight by about n²/6
-//! units of 2⁻⁵² (relative), so a plain `f64` computation loses that much there; even from the
-//! correctly rounded nodes, the extreme weight at n = 1000 is about 7·10⁴ units off. Mie
-//! forward-scattering peaks sit next to +1. Rounding from double-double keeps `f64` nodes and
-//! weights within about half an ulp. The cost is O(n²); callers that reuse a rule should cache
-//! it.
+//! Two methods. Up to `ASYMPTOTIC_THRESHOLD` points, and for double-double precision at any
+//! size, nodes come from Newton iteration on Pₙ in double-double arithmetic, started from
+//! Tricomi's approximation, and are then rounded to the requested type: O(n²). Above it,
+//! `f64`-precision rules come from Bogaert's (2014) asymptotic expansions of the angles θₖ and
+//! the weights (`gauss_legendre_asymptotic`): O(n).
+//!
+//! Next to ±1 the weights are ill-conditioned in the nodes: a node error of one ulp changes the
+//! weight by about n²/6 units of 2⁻⁵² (relative), so computing a weight from an `f64` node
+//! loses that much there; even from the correctly rounded nodes, the extreme weight at n = 1000
+//! is about 7·10⁴ units off. Mie forward-scattering peaks sit next to +1. Both methods avoid
+//! it: the Newton method computes the weights in double-double, and the expansions compute
+//! them from the angle, not the node.
 
 use crate::scalar::{DoubleDouble as Dd, Real};
 use crate::{Error, Result};
 
 /// Gauss–Legendre nodes (ascending) and weights on [−1, 1].
 #[derive(Debug, Clone, PartialEq)]
+#[non_exhaustive]
 pub struct GaussLegendre<T> {
     /// Nodes in ascending order; `nodes[i] == -nodes[n - 1 - i]`.
     pub nodes: Vec<T>,
@@ -24,31 +29,36 @@ pub struct GaussLegendre<T> {
 /// The largest number of points [`gauss_legendre`] accepts: 2¹⁷ = 131 072.
 ///
 /// Larger `n` (up to `usize::MAX`) returns [`Error::InvalidInput`]. The cap is about three
-/// times the roughly 4·10⁴ points that Mie quadrature needs at size parameter 2·10⁴. It will be
-/// raised when an O(n) method replaces the present O(n²) one (raising it is not a breaking
-/// change, lowering it would be).
+/// times the roughly 4·10⁴ points that Mie quadrature needs at size parameter 2·10⁴. It may be
+/// raised (raising it is not a breaking change, lowering it would be).
 ///
-/// Below the cap a call can still be expensive, and it cannot be interrupted:
+/// For `f64` precision a rule at the cap takes about 40 ms. A double-double rule can be
+/// expensive, and the call cannot be interrupted:
 ///
 /// * time: about 28 ns·n² for large n on an Apple M4 Pro with FMA, so about 8 minutes at the
 ///   cap, and roughly 10 times longer on x86_64 builds without the `fma` target feature, where
 ///   every double-double operation calls a software `mul_add`;
 /// * memory: (32 + 2·`size_of::<T>()`)·n bytes, 32 bytes per point of recurrence coefficients
-///   besides the rule itself: 6.3 MB for `f64` at the cap, 277 MB for `Dual<DoubleDouble, 64>`.
+///   besides the rule itself: 277 MB for `Dual<DoubleDouble, 64>` at the cap.
 pub const MAX_GAUSS_LEGENDRE_POINTS: usize = 1 << 17;
 
+/// Above this many points, `f64`-precision rules come from asymptotic expansions in O(n). At
+/// 100 points the expansions are already within about 2·2⁻⁵² for nodes and weights, and the
+/// Newton rule costs 0.8 ms.
+pub(super) const ASYMPTOTIC_THRESHOLD: usize = 100;
+
 /// Newton steps stop once |Δx| ≤ 2⁻⁹⁰; one further step then reaches rounding level.
 const CONVERGED: f64 = f64::from_bits((1023 - 90) << 52); // 2⁻⁹⁰
 const MAX_ITERATIONS: usize = 100;
 
 /// Recurrence coefficients for Pⱼ = aⱼ·x·Pⱼ₋₁ − bⱼ·Pⱼ₋₂, with aⱼ = (2j − 1)/j and bⱼ = (j − 1)/j.
-struct Recurrence {
+pub(super) struct Recurrence {
     a: Vec<Dd>,
     b: Vec<Dd>,
 }
 
 impl Recurrence {
-    fn new(n: usize) -> Self {
+    pub(super) fn new(n: usize) -> Self {
         let a = (0..=n)
             .map(|j| {
                 if j == 0 {
@@ -111,7 +121,7 @@ impl Recurrence {
     }
 
     /// The root x ≥ 0 of Pₙ that is the (i + 1)-th largest, for i < ⌈n/2⌉, and its weight.
-    fn root_and_weight(&self, n: usize, i: usize) -> Result<(Dd, Dd)> {
+    pub(super) fn root_and_weight(&self, n: usize, i: usize) -> Result<(Dd, Dd)> {
         let (x, dp) = if 2 * i + 1 == n {
             (Dd::ZERO, self.legendre_and_derivative(n, Dd::ZERO).1)
         } else {
@@ -133,21 +143,31 @@ impl Recurrence {
 /// Computes the `n`-point Gauss–Legendre rule, exact for polynomials of degree ≤ 2n − 1.
 ///
 /// Nodes ascend, and the rule is exactly symmetric: `nodes[i] == -nodes[n - 1 - i]` and
-/// `weights[i] == weights[n - 1 - i]`; for odd `n` the middle node is `+0`. The rule is
-/// computed in double-double and converted with [`Real::from_f64_pair`], so every `T` gets the
-/// same rule rounded once: the `f64` rule is the double-double rule rounded to nearest, and
+/// `weights[i] == weights[n - 1 - i]`; for odd `n` the middle node is `+0`. Two methods:
+///
+/// * **Newton (n ≤ 100, and every n for `DoubleDouble` precision).** The rule is computed in
+///   double-double and converted with [`Real::from_f64_pair`], so it is rounded once: the
+///   `f64` rule is the double-double rule rounded to nearest.
+/// * **Asymptotic (n > 100, for `f64` precision: `f64` and `Dual<f64, N>`).** Bogaert's
+///   (2014) expansions of the angle θₖ of each node and of its weight, in powers of
+///   1/(n + ½)², around the zeros of the Bessel function J₀; the weights come from the
+///   angle, not from the node, so they keep the accuracy of the nodes next to ±1.
+///
 /// `Dual` rules are constants (zero derivatives) whose values are bitwise those of `f64` or
-/// `DoubleDouble`. No platform libm is used (the starting guesses come from the double-double
-/// cosine), only IEEE 754 operations and fused multiply-adds, so the bits of the rule do not
-/// depend on the platform.
+/// `DoubleDouble`. No platform libm is used (the starting guesses and the expansions use the
+/// double-double sine and cosine), only IEEE 754 operations and fused multiply-adds, so the
+/// bits of the rule do not depend on the platform.
 ///
 /// # Accuracy
 ///
 /// Verified against Arb for every node of the rules with n ≤ 1000, and for the nodes nearest
-/// to ±1 and to 0 at n = 5000, 20 000 and 2¹⁷ (the cap):
+/// to ±1 and to 0 at n = 5000, 20 000 and 2¹⁷ (the cap), and against the double-double rule
+/// for every node of the rules with n = 101, 102, 128, 150, 257, 500, 1000, 2000 and 4001:
 ///
-/// * `f64`: node absolute error ≤ 2⁻⁵², weight relative error ≤ 2⁻⁵². Measured: at most
-///   0.25·2⁻⁵² and 0.49·2⁻⁵², the rounding of the double-double rule.
+/// * `f64`, n ≤ 100: node absolute error ≤ 2⁻⁵², weight relative error ≤ 2⁻⁵². Measured: at
+///   most 0.25·2⁻⁵² and 0.49·2⁻⁵², the rounding of the double-double rule.
+/// * `f64`, n > 100: node absolute error ≤ 16·2⁻⁵², weight relative error ≤ 16·2⁻⁵². Measured:
+///   at most 2.4·2⁻⁵² and 3.6·2⁻⁵².
 /// * `DoubleDouble`: node absolute error ≤ 2⁻⁹⁸, weight relative error ≤ 2⁻¹⁰⁰·(4 + n²). The n²
 ///   comes from the conditioning of 1 − x² next to ±1: a node error δ changes the weight by
 ///   2|x|δ/(1 − x²) relative, and 1 − x² ≈ 5.8/n² at the extreme nodes. Measured: at most
@@ -155,11 +175,12 @@ impl Recurrence {
 ///
 /// # Cost
 ///
-/// O(n²) time and O(n) memory; see [`MAX_GAUSS_LEGENDRE_POINTS`] for the cost model. Each of the
-/// ⌈n/2⌉ roots takes four to seven evaluations of the three-term recurrence in double-double
-/// (about four on average for n ≥ 5000). Measured on an Apple M4 Pro: 0.8 ms at n = 100,
-/// 32 ms at n = 1000, 0.72 s at n = 5000 and 11 s at n = 20 000. The call cannot be
-/// interrupted.
+/// O(n) memory. Asymptotic: O(n) time, measured on an Apple M4 Pro as 1 ms at n = 1000,
+/// 13 ms at n = 20 000 and 40 ms at the cap. Newton: O(n²) time; see
+/// [`MAX_GAUSS_LEGENDRE_POINTS`] for the cost model. Each of the ⌈n/2⌉ roots takes four to
+/// seven evaluations of the three-term recurrence in double-double (about four on average for
+/// n ≥ 5000). Measured on an Apple M4 Pro: 0.8 ms at n = 100, 32 ms at n = 1000, 0.72 s at
+/// n = 5000 and 11 s at n = 20 000. The call cannot be interrupted.
 ///
 /// # Errors
 ///
@@ -182,6 +203,32 @@ impl Recurrence {
 /// # Ok::<(), scatrs::Error>(())
 /// ```
 pub fn gauss_legendre<T: Real>(n: usize) -> Result<GaussLegendre<T>> {
+    check_n(n)?;
+    let mut nodes = vec![T::zero(); n];
+    let mut weights = vec![T::zero(); n];
+    fill(&mut nodes, &mut weights)?;
+    Ok(GaussLegendre { nodes, weights })
+}
+
+/// [`gauss_legendre`] into caller-owned buffers: the `nodes.len()`-point rule, bitwise the same.
+///
+/// # Errors
+///
+/// As [`gauss_legendre`], and [`Error::InvalidInput`] if `weights.len() != nodes.len()`.
+/// Nothing is written on an input error.
+pub fn gauss_legendre_into<T: Real>(nodes: &mut [T], weights: &mut [T]) -> Result<()> {
+    check_n(nodes.len())?;
+    if weights.len() != nodes.len() {
+        return Err(Error::InvalidInput(format!(
+            "Gauss-Legendre buffers must have one length, got {} nodes and {} weights",
+            nodes.len(),
+            weights.len()
+        )));
+    }
+    fill(nodes, weights)
+}
+
+fn check_n(n: usize) -> Result<()> {
     if n == 0 {
         return Err(Error::InvalidInput(
             "Gauss-Legendre rule needs n >= 1".into(),
@@ -192,9 +239,17 @@ pub fn gauss_legendre<T: Real>(n: usize) -> Result<GaussLegendre<T>> {
             "Gauss-Legendre rule needs n <= {MAX_GAUSS_LEGENDRE_POINTS}, got {n}"
         )));
     }
+    Ok(())
+}
+
+/// The rule of size `nodes.len()` (checked) into `nodes` and `weights` (of one length).
+fn fill<T: Real>(nodes: &mut [T], weights: &mut [T]) -> Result<()> {
+    let n = nodes.len();
+    if n > ASYMPTOTIC_THRESHOLD && T::epsilon().approx_f64() >= 2f64.powi(-60) {
+        super::gauss_legendre_asymptotic::asymptotic_rule(nodes, weights);
+        return Ok(());
+    }
     let rec = Recurrence::new(n);
-    let mut nodes = vec![T::zero(); n];
-    let mut weights = vec![T::zero(); n];
     for i in 0..n.div_ceil(2) {
         let (x, w) = rec.root_and_weight(n, i)?;
         let w = T::from_f64_pair(w.hi(), w.lo());
@@ -204,7 +259,7 @@ pub fn gauss_legendre<T: Real>(n: usize) -> Result<GaussLegendre<T>> {
         weights[i] = w;
         weights[n - 1 - i] = w;
     }
-    Ok(GaussLegendre { nodes, weights })
+    Ok(())
 }
 
 #[cfg(test)]
@@ -261,8 +316,9 @@ mod tests {
         ((got - Dd::new(hi, lo)) - lo2).abs().hi() + err
     }
 
-    /// Single roots of the largest accepted rule against Arb. The whole rule takes minutes,
-    /// but each root costs O(n), so this checks the accuracy contract at the cap.
+    /// Single roots of the largest accepted rule against Arb. The whole double-double rule
+    /// takes minutes, but each root costs O(n), so this checks the double-double contract at
+    /// the cap; the `f64` rule, from the asymptotic expansions, takes milliseconds whole.
     #[test]
     fn roots_at_the_cap_match_arb() {
         let Some((n, indices, nodes, weights)) =
@@ -272,6 +328,7 @@ mod tests {
         };
         assert_eq!(n, MAX_GAUSS_LEGENDRE_POINTS);
         let rec = Recurrence::new(n);
+        let rule = gauss_legendre::<f64>(n).unwrap();
         let u2 = 2f64.powi(-106);
         let (mut worst_x, mut worst_w) = (0.0_f64, 0.0_f64);
         for (k, &idx) in indices.iter().enumerate() {
@@ -283,14 +340,20 @@ mod tests {
             let ew = error(w, weights[k]) / weights[k][0];
             worst_x = worst_x.max(ex / u2);
             worst_w = worst_w.max(ew / u2);
-            // The double-double contract, and the f64 contract after rounding.
+            // The double-double contract, and the f64 contract of the asymptotic rule.
             assert!(ex <= 2f64.powi(-98), "node {idx}: error {ex:e}");
             let wtol = 2f64.powi(-100) * (4.0 + (n * n) as f64);
             assert!(ew <= wtol, "weight {idx}: relative error {ew:e}");
-            let ex64 = error(Dd::from_f64(x.hi()), nodes[k]);
-            let ew64 = error(Dd::from_f64(w.hi()), weights[k]) / weights[k][0];
-            assert!(ex64 <= f64::EPSILON, "node {idx}: f64 error {ex64:e}");
-            assert!(ew64 <= f64::EPSILON, "weight {idx}: f64 error {ew64:e}");
+            let ex64 = error(Dd::from_f64(rule.nodes[idx]), nodes[k]);
+            let ew64 = error(Dd::from_f64(rule.weights[idx]), weights[k]) / weights[k][0];
+            assert!(
+                ex64 <= 16.0 * f64::EPSILON,
+                "node {idx}: f64 error {ex64:e}"
+            );
+            assert!(
+                ew64 <= 16.0 * f64::EPSILON,
+                "weight {idx}: f64 error {ew64:e}"
+            );
         }
         eprintln!(
             "n = {n}: worst node error {worst_x:.3} u², worst weight error {worst_w:.3e} u² \
diff --git a/scatrs/src/special/gauss_legendre_asymptotic.rs b/scatrs/src/special/gauss_legendre_asymptotic.rs
new file mode 100644
index 0000000..9841f60
--- /dev/null
+++ b/scatrs/src/special/gauss_legendre_asymptotic.rs
@@ -0,0 +1,186 @@
+//! Gauss–Legendre rules in O(n) from asymptotic expansions (Bogaert 2014).
+//!
+//! The k-th node from +1 is cos θₖ, with θₖ an expansion in v² = 1/(n + ½)² around
+//! aₖ = jₖ·v (jₖ the k-th zero of J₀), and the weight an expansion in v² around
+//! 2v²/(J₁(jₖ)²·aₖ/sin aₖ), with u = cot aₖ in the coefficients. The coefficients are Bogaert's,
+//! as in FastGaussQuadrature.jl (MIT licence, © 2014 Alex Townsend), which keeps fewer terms
+//! for larger n; this module keeps them all. The Bessel tables were recomputed with Arb, and
+//! the double-double sine and cosine replace the platform's, so the rule is the same on every
+//! platform.
+
+use core::f64::consts::PI;
+
+use crate::scalar::{DoubleDouble as Dd, Real};
+
+/// The first 20 zeros of J₀ (Arb, 300 bits).
+const J0_ZEROS: [f64; 20] = [
+    2.404825557695773,
+    5.520078110286311,
+    8.653727912911013,
+    11.791534439014281,
+    14.930917708487787,
+    18.071063967910924,
+    21.21163662987926,
+    24.352471530749302,
+    27.493479132040257,
+    30.634606468431976,
+    33.77582021357357,
+    36.917098353664045,
+    40.05842576462824,
+    43.19979171317673,
+    46.341188371661815,
+    49.482609897397815,
+    52.624051841115,
+    55.76551075501998,
+    58.90698392608094,
+    62.04846919022717,
+];
+
+/// J₁(j₀,ₖ)² for the first 10 zeros of J₀ (Arb, 300 bits).
+const J1_SQUARED: [f64; 10] = [
+    0.2695141239419169,
+    0.1157801385822037,
+    0.07368635113640822,
+    0.05403757319811628,
+    0.04266142901724309,
+    0.0352421034909961,
+    0.030021070103054673,
+    0.02614739149530809,
+    0.023159121824691393,
+    0.02078382912226786,
+];
+
+/// The k-th zero of J₀ (k from 1): the table, then McMahon's expansion.
+fn j0_zero(k: usize) -> f64 {
+    if k <= J0_ZEROS.len() {
+        J0_ZEROS[k - 1]
+    } else {
+        mcmahon(k)
+    }
+}
+
+/// McMahon's expansion of the k-th zero of J₀ (NIST DLMF 10.21.19).
+fn mcmahon(k: usize) -> f64 {
+    let a = PI * (k as f64 - 0.25);
+    let t = (0.125 / a) * (0.125 / a);
+    a + 0.125 / a
+        * (1.0 + t * (-124.0 / 3.0 + t * (120_928.0 / 15.0 + t * (-401_743_168.0 / 105.0))))
+}
+
+/// J₁(j₀,ₖ)² (k from 1): the table, then its expansion.
+fn j1_squared(k: usize) -> f64 {
+    if k <= J1_SQUARED.len() {
+        J1_SQUARED[k - 1]
+    } else {
+        j1_squared_series(k)
+    }
+}
+
+/// The expansion of J₁(j₀,ₖ)² in 1/(π(k − ¼)) of FastGaussQuadrature.jl.
+fn j1_squared_series(k: usize) -> f64 {
+    let a = PI * (k as f64 - 0.25);
+    let t = 1.0 / (a * a);
+    let p = -7.0 / 24.0
+        + t * (151.0 / 80.0
+            + t * (-172_913.0 / 8064.0
+                + t * (461_797.0 / 1152.0 + t * (-171_497_088_497.0 / 15_206_400.0))));
+    (2.0 + t * t * p) / (PI * a)
+}
+
+/// Writes the ascending nodes and the weights of the n-point rule, n = `nodes.len()` ≥ 2
+/// (`weights.len()` must be n).
+pub(super) fn asymptotic_rule<T: Real>(nodes: &mut [T], weights: &mut [T]) {
+    let n = nodes.len();
+    let v = 1.0 / (n as f64 + 0.5);
+    let v2 = v * v;
+    let half = n.div_ceil(2);
+    for k in 1..=half {
+        let a = j0_zero(k) * v;
+        // The double-double sine and cosine use no platform libm, so the rule is bitwise the
+        // same on every platform.
+        let (sin_a, cos_a) = Dd::from_f64(a).sin_cos();
+        let (sin_a, cos_a) = (sin_a.hi(), cos_a.hi());
+        let u = cos_a / sin_a;
+        let u2 = u * u;
+        let (a2, a3) = (a * a, a * a * a);
+        let a5 = a2 * a3;
+        // Node: θ = a + v²·t₁ + v⁴·t₂ + v⁶·t₃.
+        let t1 = (u - 1.0 / a) / 8.0;
+        let t2 = (6.0 * (1.0 + u2) / a + 25.0 / a3 - u * (31.0 * u2 + 33.0)) / 384.0;
+        let t3 = u * (2595.0 / 15360.0 + u2 * (6350.0 / 15360.0 + u2 * 3779.0 / 15360.0))
+            - 1073.0 / 5120.0 / a5
+            + (1.0 + u2)
+                * (-(31.0 / 1024.0 * u2 + 11.0 / 1024.0) / a + u / 512.0 / a2 - 25.0 / 3072.0 / a3);
+        let theta = a + v2 * (t1 + v2 * (t2 + v2 * t3));
+        // Weight: 2/(J₁(jₖ)²·(a/sin a)·(1/v² + W₁ + v²W₂ + v⁴W₃)).
+        let ua = u * a;
+        let ai = 1.0 / a;
+        let ai2 = ai * ai;
+        let w1 = ((ua - 1.0) * ai2 + 1.0) / 8.0;
+        let w2 = ((-27.0 + u2 * (-84.0 + u2 * -56.0))
+            + ai2 * ((-3.0 * (1.0 - 2.0 * u2) + 6.0 * ua) + ai2 * (81.0 - 31.0 * ua)))
+            / 384.0;
+        let w3 = (153.0 / 1024.0 + u2 * (295.0 / 256.0 + u2 * (187.0 / 96.0 + u2 * 151.0 / 160.0)))
+            + ai * ((-65.0 / 1024.0 + u2 * (-119.0 / 768.0 + u2 * -35.0 / 384.0)) * u
+                + ai * ((5.0 / 512.0 + u2 * (15.0 / 512.0 + u2 * 7.0 / 384.0))
+                    + ai * ((u2 / 512.0 - 13.0 / 1536.0) * u
+                        + ai * ((-7.0 / 384.0 * u2 + 53.0 / 3072.0)
+                            + ai * (3749.0 / 15360.0 * u + ai * (-1125.0 / 1024.0))))));
+        let big_w = 1.0 / v2 + w1 + v2 * (w2 + v2 * w3);
+        let w = 2.0 / (j1_squared(k) * (a / sin_a) * big_w);
+        let x = Dd::from_f64(theta).cos().hi();
+        nodes[n - k] = T::from_f64(x);
+        nodes[k - 1] = T::from_f64(-x);
+        weights[n - k] = T::from_f64(w);
+        weights[k - 1] = T::from_f64(w);
+    }
+    if n % 2 == 1 {
+        nodes[half - 1] = T::zero();
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::special::gauss_legendre::{ASYMPTOTIC_THRESHOLD, Recurrence};
+
+    /// The rule against the double-double Newton rule: node absolute error and weight relative
+    /// error within 16·2⁻⁵² (measured: at most 2.4 and 3.6), from the threshold up.
+    #[test]
+    fn matches_the_double_double_rule() {
+        let n0 = ASYMPTOTIC_THRESHOLD + 1;
+        for n in [n0, n0 + 1, 128, 150, 257, 500, 1000, 2000, 4001] {
+            let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
+            asymptotic_rule(&mut x, &mut w);
+            let rec = Recurrence::new(n);
+            let (mut ex, mut ew) = (0.0f64, 0.0f64);
+            for i in 0..n.div_ceil(2) {
+                let (xd, wd) = rec.root_and_weight(n, i).unwrap();
+                // Root i is the (i + 1)-th largest, at index n − 1 − i.
+                let k = n - 1 - i;
+                ex = ex.max((Dd::from_f64(x[k]) - xd).abs().hi() / f64::EPSILON);
+                ew = ew.max(((Dd::from_f64(w[k]) - wd) / wd).abs().hi() / f64::EPSILON);
+            }
+            assert!(
+                ex <= 16.0 && ew <= 16.0,
+                "n = {n}: node {ex:.2}, weight {ew:.2} eps"
+            );
+        }
+    }
+
+    /// Each expansion is already within 2·2⁻⁵² of the last tabulated values (Arb: at most
+    /// 1.5·2⁻⁵² for the zeros from k = 16, 1.15·2⁻⁵² for J₁² at k = 10; below that, at k = 13
+    /// and k = 9, more than 6·2⁻⁵²), so switching to it after the table loses nothing.
+    #[test]
+    fn the_expansions_take_over_from_the_tables() {
+        for k in 16..=J0_ZEROS.len() {
+            let table = J0_ZEROS[k - 1];
+            let err = (mcmahon(k) - table).abs();
+            assert!(err <= 2.0 * f64::EPSILON * table, "zero {k} of J0");
+        }
+        let k = J1_SQUARED.len();
+        let table = J1_SQUARED[k - 1];
+        let err = (j1_squared_series(k) - table).abs();
+        assert!(err <= 2.0 * f64::EPSILON * table, "J1^2 at zero {k}");
+    }
+}
diff --git a/scatrs/src/special/mod.rs b/scatrs/src/special/mod.rs
index 9a8e7f1..a70d30c 100644
--- a/scatrs/src/special/mod.rs
+++ b/scatrs/src/special/mod.rs
@@ -1,7 +1,10 @@
 //! Special functions and quadrature rules shared by all scattering methods.
 
 mod gauss_legendre;
+mod gauss_legendre_asymptotic;
 mod wigner;
 
-pub use gauss_legendre::{GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre};
+pub use gauss_legendre::{
+    GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre, gauss_legendre_into,
+};
 pub use wigner::{wigner_d, wigner_d_vec};
diff --git a/scatrs/tests/gauss_legendre.rs b/scatrs/tests/gauss_legendre.rs
index ba24b37..95e574b 100644
--- a/scatrs/tests/gauss_legendre.rs
+++ b/scatrs/tests/gauss_legendre.rs
@@ -8,12 +8,27 @@ use std::sync::atomic::{AtomicUsize, Ordering};
 use common::{Record, load};
 use scatrs::Error;
 use scatrs::scalar::{DoubleDouble as Dd, Dual};
-use scatrs::special::{GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre};
+use scatrs::special::{
+    GaussLegendre, MAX_GAUSS_LEGENDRE_POINTS, gauss_legendre, gauss_legendre_into,
+};
 
 const EPS: f64 = f64::EPSILON; // 2^-52
 /// u² = 2^-106.
 const U2: f64 = 1.232_595_164_407_831e-32;
 
+/// From this many points on, the `f64` rule comes from the asymptotic expansions, within
+/// 16·2⁻⁵²; below, it is the double-double rule rounded, within 2⁻⁵².
+const ASYMPTOTIC_FROM: usize = 101;
+
+/// The `f64` contract at n points: node absolute error and weight relative error at most this.
+fn f64_tolerance(n: usize) -> f64 {
+    if n >= ASYMPTOTIC_FROM {
+        16.0 * EPS
+    } else {
+        EPS
+    }
+}
+
 /// The indices into the rule of the values that reference record `r` holds, in the order
 /// stored: every node of the rule, or only those listed in its `indices` input.
 fn indices(r: &Record) -> Vec<usize> {
@@ -27,20 +42,21 @@ fn indices(r: &Record) -> Vec<usize> {
     }
 }
 
-/// Rules above this size take seconds each (the cost is O(n²)), so their records are checked
-/// only by the ignored test `largest_rules_match_arb`.
+/// Double-double rules above this size take seconds each (the cost is O(n²)), so their records
+/// are checked in double-double only by the ignored test `largest_double_double_rules_match_arb`.
 const SLOW_N: usize = 10_000;
 
-/// The reference records whose rules have at most `SLOW_N` points, or more if `slow`.
+/// The reference records whose rules have at most `SLOW_N` points, or more if `slow`, or all
+/// (`None`).
 ///
 /// Records whose case ends in `,roots` are skipped: they hold single roots of the largest
-/// accepted rule, which takes minutes to compute whole, and a unit test in
+/// accepted rule, whose double-double form takes minutes to compute whole, and a unit test in
 /// `src/special/gauss_legendre.rs` checks them root by root.
-fn records(slow: bool) -> Vec<Record> {
+fn records(slow: Option<bool>) -> Vec<Record> {
     load("gauss_legendre.jsonl")
         .into_iter()
         .filter(|r| !r.case.ends_with(",roots"))
-        .filter(|r| (r.input_i64("n") as usize > SLOW_N) == slow)
+        .filter(|r| slow.is_none_or(|s| (r.input_i64("n") as usize > SLOW_N) == s))
         .collect()
 }
 
@@ -57,8 +73,9 @@ fn check_f64(records: &[Record]) -> (f64, f64) {
             let ew = ws.dd_error(k, rule.weights[i], 0.0) / ws.hi[k];
             worst_x = worst_x.max(ex / EPS);
             worst_w = worst_w.max(ew / EPS);
-            assert!(ex <= EPS, "n={n} node {i}: error {ex:e}");
-            assert!(ew <= EPS, "n={n} weight {i}: relative error {ew:e}");
+            let tol = f64_tolerance(n);
+            assert!(ex <= tol, "n={n} node {i}: error {ex:e}");
+            assert!(ew <= tol, "n={n} weight {i}: relative error {ew:e}");
         }
     }
     (worst_x, worst_w)
@@ -93,14 +110,28 @@ fn check_dd(records: &[Record]) -> (f64, f64, f64) {
 }
 
 #[test]
-fn f64_rule_matches_arb_to_half_an_ulp_level() {
-    let (x, w) = check_f64(&records(false));
-    eprintln!("f64, n <= {SLOW_N}: worst node error {x:.3} eps, worst weight error {w:.3} eps");
+fn f64_rule_matches_arb() {
+    let small: Vec<Record> = records(None)
+        .into_iter()
+        .filter(|r| (r.input_i64("n") as usize) < ASYMPTOTIC_FROM)
+        .collect();
+    let (x, w) = check_f64(&small);
+    eprintln!(
+        "f64, n < {ASYMPTOTIC_FROM}: worst node error {x:.3} eps, worst weight error {w:.3} eps"
+    );
+    let large: Vec<Record> = records(None)
+        .into_iter()
+        .filter(|r| (r.input_i64("n") as usize) >= ASYMPTOTIC_FROM)
+        .collect();
+    let (x, w) = check_f64(&large);
+    eprintln!(
+        "f64, n >= {ASYMPTOTIC_FROM}: worst node error {x:.3} eps, worst weight error {w:.3} eps"
+    );
 }
 
 #[test]
 fn double_double_rule_matches_arb() {
-    let (x, w, w_tol) = check_dd(&records(false));
+    let (x, w, w_tol) = check_dd(&records(Some(false)));
     eprintln!(
         "DD, n <= {SLOW_N}: worst node error {x:.3} u², worst weight error {w:.3e} u² \
          ({w_tol:.3e} of its bound); u² = 2^-106"
@@ -108,12 +139,10 @@ fn double_double_rule_matches_arb() {
 }
 
 #[test]
-#[ignore = "computes two 20000-point rules, about 11 s each in release builds"]
-fn largest_rules_match_arb() {
-    let recs = records(true);
+#[ignore = "computes two 20000-point double-double rules, about 11 s each in release builds"]
+fn largest_double_double_rules_match_arb() {
+    let recs = records(Some(true));
     assert!(!recs.is_empty());
-    let (x, w) = check_f64(&recs);
-    eprintln!("f64, n > {SLOW_N}: worst node error {x:.3} eps, worst weight error {w:.3} eps");
     let (x, w, w_tol) = check_dd(&recs);
     eprintln!(
         "DD, n > {SLOW_N}: worst node error {x:.3} u², worst weight error {w:.3e} u² \
@@ -180,10 +209,10 @@ fn check_structure(n: usize, rule: &GaussLegendre<f64>) {
     }
     assert!(rule.weights.iter().all(|&w| w > 0.0));
     // Summed in double-double, so the sum's own error is negligible; each weight is within
-    // eps relative of the exact one, and the exact weights sum to 2.
+    // `f64_tolerance` relative of the exact one, and the exact weights sum to 2.
     let sum = rule.weights.iter().fold(Dd::ZERO, |s, &w| s + w);
     assert!(
-        (sum - 2.0).abs().hi() <= 2.0 * EPS,
+        (sum - 2.0).abs().hi() <= 2.0 * f64_tolerance(n),
         "n={n}: weights sum to {sum:?}"
     );
 }
@@ -232,7 +261,19 @@ fn every_rule_up_to_3000_points_converges_to_the_right_roots() {
 
 #[test]
 fn large_rules_converge_to_the_right_roots() {
-    for n in [511usize, 512, 1023, 1024, 2047, 2048, 4999, 5000] {
+    for n in [
+        511usize,
+        512,
+        1023,
+        1024,
+        2047,
+        2048,
+        4999,
+        5000,
+        20_000,
+        MAX_GAUSS_LEGENDRE_POINTS - 1,
+        MAX_GAUSS_LEGENDRE_POINTS,
+    ] {
         check_structure(n, &gauss_legendre::<f64>(n).unwrap());
     }
 }
@@ -300,8 +341,11 @@ fn every_type_gets_the_same_rule_bit_for_bit() {
         let dd = gauss_legendre::<Dual<Dd, 3>>(n).unwrap();
         let (f, d, df, dd) = (values(&f), values(&d), values(&df), values(&dd));
         for i in 0..2 * n {
-            // The f64 rule is the double-double rule rounded to nearest.
-            assert_eq!(d[i].hi().to_bits(), f[i].to_bits(), "n={n} value {i}");
+            // Below the asymptotic range, the f64 rule is the double-double rule rounded to
+            // nearest.
+            if n < ASYMPTOTIC_FROM {
+                assert_eq!(d[i].hi().to_bits(), f[i].to_bits(), "n={n} value {i}");
+            }
             assert_eq!(df[i].re().to_bits(), f[i].to_bits(), "n={n} value {i}");
             assert_eq!(dd_bits(dd[i].re()), dd_bits(d[i]), "n={n} value {i}");
             assert!(
@@ -315,3 +359,24 @@ fn every_type_gets_the_same_rule_bit_for_bit() {
         }
     }
 }
+
+#[test]
+fn caller_buffers_get_the_same_rule() {
+    for n in [1usize, 7, 100, 101, 1000] {
+        let rule = gauss_legendre::<f64>(n).unwrap();
+        let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
+        gauss_legendre_into(&mut x, &mut w).unwrap();
+        assert_eq!((x, w), (rule.nodes, rule.weights), "n={n}");
+    }
+    let (mut x, mut w) = (vec![5.0; 4], vec![5.0; 3]);
+    assert!(matches!(
+        gauss_legendre_into(&mut x, &mut w),
+        Err(Error::InvalidInput(_))
+    ));
+    assert!(x.iter().chain(&w).all(|&v| v == 5.0), "nothing written");
+    let mut empty: [f64; 0] = [];
+    assert!(matches!(
+        gauss_legendre_into(&mut empty, &mut []),
+        Err(Error::InvalidInput(_))
+    ));
+}
diff --git a/tests/test_docs_consistency.py b/tests/test_docs_consistency.py
index 36929cc..d48f184 100644
--- a/tests/test_docs_consistency.py
+++ b/tests/test_docs_consistency.py
@@ -67,6 +67,7 @@ SCALARS = "docs/theory/scalar-types.md"
 LIMITATIONS = "docs/limitations.md"
 WIGNER_EXAMPLE = "docs/examples/wigner-d.md"
 HG_EXAMPLE = "docs/examples/henyey-greenstein.md"
+QUADRATURE = "docs/examples/quadrature.md"
 REFERENCE_SUITE = "docs/validation/reference-suite.md"
 DOMAIN_RS = "scatrs/src/mie/domain.rs"
 MIE_THEORY = "docs/theory/mie.md"
@@ -82,13 +83,23 @@ CONTRACTS = [
     ("Wigner double-double bound", "(l+1)2^-97", [WIGNER_RS, THEORY]),
     ("Gauss-Legendre cap", "2^17", [GL_RS, SPECIAL_PY, THEORY, LIMITATIONS]),
     ("Gauss-Legendre cap", "131072", [GL_RS, SPECIAL_PY, THEORY]),
-    ("Gauss-Legendre cost model", "28nsn^2", [GL_RS, SPECIAL_PY, THEORY]),
-    ("Gauss-Legendre timing", "0.72s", [GL_RS, SPECIAL_PY, THEORY]),
-    ("Gauss-Legendre timing", "11s", [GL_RS, SPECIAL_PY, THEORY, LIMITATIONS]),
+    ("Gauss-Legendre double-double cost model", "28nsn^2", [GL_RS, THEORY]),
+    ("Gauss-Legendre double-double timing", "0.72s", [GL_RS, THEORY]),
+    ("Gauss-Legendre double-double timing", "11s", [GL_RS, THEORY, LIMITATIONS]),
+    ("Gauss-Legendre f64 timing at the cap", "40ms", [GL_RS, SPECIAL_PY, THEORY]),
+    (
+        "Gauss-Legendre asymptotic bound",
+        "162^-52",
+        [GL_RS, SPECIAL_PY, THEORY, LIMITATIONS, QUADRATURE],
+    ),
+    ("Gauss-Legendre asymptotic measured", "2.42^-52", [GL_RS, THEORY]),
+    ("Gauss-Legendre asymptotic measured", "3.62^-52", [GL_RS, THEORY]),
+    ("Gauss-Legendre asymptotic measured", "2.4and3.6unitsof2^-52", [SPECIAL_PY, LIMITATIONS]),
     ("Wigner cost", "2.6ns", [WIGNER_RS, THEORY]),
     ("Gauss-Legendre f64 node contract", "nodeabsoluteerror≤2^-52", [GL_RS]),
     ("Gauss-Legendre f64 weight contract", "weightrelativeerror≤2^-52", [GL_RS]),
-    ("Gauss-Legendre f64 contract", "|f64|≤2^-52|≤2^-52|", [THEORY]),
+    ("Gauss-Legendre f64 contract", "|f64,n≤100|≤2^-52|≤2^-52|", [THEORY]),
+    ("Gauss-Legendre asymptotic contract", "|f64,n>100|≤162^-52|≤162^-52|", [THEORY]),
     ("Gauss-Legendre double-double weight bound", "2^-100(4+n^2)", [GL_RS, THEORY]),
     ("Python Wigner output limit", "2^28", [SPECIAL_PY, LIMITATIONS]),
     ("Double-double full precision", "2^-969", [DD_RS, SCALARS]),
diff --git a/tests/test_special.py b/tests/test_special.py
index 3340e99..588ff2b 100644
--- a/tests/test_special.py
+++ b/tests/test_special.py
@@ -237,8 +237,10 @@ def _main_thread_stamps_during(call):
 @pytest.mark.parametrize(
     "call",
     [
-        # About 0.1 s on an Apple M4 Pro (28 ns·n²), longer on other machines.
-        pytest.param(lambda: special.gauss_legendre(2000), id="gauss_legendre"),
+        # The largest rule: about 0.04 s on an Apple M4 Pro, longer on other machines.
+        pytest.param(
+            lambda: special.gauss_legendre(special.MAX_GAUSS_LEGENDRE_POINTS), id="gauss_legendre"
+        ),
         # About 2·10⁷ values (160 MB) in the general difference form, the slowest per value.
         pytest.param(
             lambda: special.wigner_d(0, 2, 19_999, np.linspace(0.95, 1, 1000)), id="wigner_d"
````

- [ ] **Step 2: Verify**

Run: `cargo test -p scatrs --release --lib gauss_legendre && cargo test -p scatrs --release --test gauss_legendre -- --nocapture`
Expected: PASS; `matches_the_double_double_rule` and `f64_rule_matches_arb` print worst errors ≤ 2.4 (nodes) and ≤ 3.6 (weights) units of 2⁻⁵² for n ≥ 101.

Run: `cargo clippy --workspace --all-targets -- -D warnings && uv run pytest -q && uv run --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html`
Expected: clean; the executed quadrature example asserts the 16·2⁻⁵² contract at n = 1000.

- [ ] **Step 3: Commit** — `feat(special): O(n) Gauss–Legendre rules for f64 above 100 points`

---

### Task 2: Phase matrix, Greek projector, single-sphere Greek coefficients

**Files:**
- Create: `scatrs/src/mie/phase.rs`, `scatrs/src/mie/greek.rs`, `scatrs/tests/common/greek.rs`, `scatrs/tests/mie_greek.rs`
- Modify: `scatrs/src/mie/mod.rs`, `scatrs/tests/common/mod.rs`

**Interfaces:**
- Consumes: `MieSolution::{amplitudes, amplitudes_with_derivatives, efficiencies, efficiencies_jacobian, n_terms, x}`, `AmplitudeBuffers`, `wigner_d`, `gauss_legendre` (Task 1).
- Produces (all re-exported from `scatrs::mie`):
  - `enum Normalization { Normalized, Unnormalized }` (`#[non_exhaustive]`, `Copy`, `Eq`);
  - `struct PhaseBuffers<'b, T> { f11, f12, f33, f34: &'b mut [T] }`;
  - `MieSolution::phase_matrix(&self, mu: &[T], normalization, out: PhaseBuffers<T>) -> Result<()>`;
  - `MieSolution::phase_matrix_with_derivatives(&self, mu, normalization, values: PhaseBuffers<T>, d: [PhaseBuffers<T>; 3]) -> Result<()>` (d/dx, d/dRe m, d/dIm m);
  - `struct GreekCoefficients<T> { alpha1, alpha2, alpha3, alpha4, beta1, beta2: Vec<T> }` (`#[non_exhaustive]`, `order() -> usize`);
  - `struct PhaseTables<'a, T> { f11, f12, f33, f34: &'a [T] }`;
  - `GreekProjector<T>::{new(order, n_nodes) -> Result<Self>, order(), nodes() -> &[T], project(&self, &[PhaseTables<T>]) -> Result<Vec<GreekCoefficients<T>>>}`;
  - `MieSolution::greek(&self, order: Option<usize>, normalization) -> Result<GreekCoefficients<T>>`;
  - `MieSolution::greek_with_derivatives(&self, order, normalization) -> Result<(GreekCoefficients<T>, [GreekCoefficients<T>; 3])>`;
  - `const MAX_GREEK_ORDER: usize = 2 * MAX_MIE_ORDER` (2²¹).
  - Test helpers: `common::greek::{GreekOutputs, solve_greek, greek_reference, greek_value_measures, greek_derivative_measures, GREEK_NAMES, PHASE_NAMES}`.

- [ ] **Step 1: Apply the diff:**

````diff
diff --git a/scatrs/src/mie/greek.rs b/scatrs/src/mie/greek.rs
new file mode 100644
index 0000000..537f875
--- /dev/null
+++ b/scatrs/src/mie/greek.rs
@@ -0,0 +1,356 @@
+//! Greek (generalized-spherical-function) expansion coefficients of the phase matrix.
+
+use super::{MAX_MIE_ORDER, MieSolution, Normalization, PhaseBuffers};
+use crate::scalar::Real;
+use crate::special::{GaussLegendre, gauss_legendre, wigner_d};
+use crate::{Error, Result};
+
+/// The expansion coefficients of a sphere's phase matrix for l = 0, …, L (Mishchenko, Travis &
+/// Lacis 2002, Eqs. 4.75–4.80, with `wigner_d`'s convention):
+///
+/// F₁₁ = Σ α₁ d⁰⁰, F₂₂ + F₃₃ = Σ (α₂ + α₃) d²², F₂₂ − F₃₃ = Σ (α₂ − α₃) d²⁻², F₄₄ = Σ α₄ d⁰⁰,
+/// F₁₂ = Σ β₁ d⁰², F₃₄ = Σ β₂ d⁰², each dₗ at cos Θ.
+#[derive(Clone, Debug, PartialEq)]
+#[non_exhaustive]
+pub struct GreekCoefficients<T> {
+    /// α₁ (of F₁₁); α₁ at l = 0 is 1 and at l = 1 is 3g when normalized.
+    pub alpha1: Vec<T>,
+    /// α₂.
+    pub alpha2: Vec<T>,
+    /// α₃.
+    pub alpha3: Vec<T>,
+    /// α₄ (of F₄₄).
+    pub alpha4: Vec<T>,
+    /// β₁ (of F₁₂).
+    pub beta1: Vec<T>,
+    /// β₂ (of F₃₄).
+    pub beta2: Vec<T>,
+}
+
+impl<T: Real> GreekCoefficients<T> {
+    fn zeros(len: usize) -> Self {
+        let z = || vec![T::zero(); len];
+        Self {
+            alpha1: z(),
+            alpha2: z(),
+            alpha3: z(),
+            alpha4: z(),
+            beta1: z(),
+            beta2: z(),
+        }
+    }
+
+    /// The highest order L.
+    pub fn order(&self) -> usize {
+        self.alpha1.len() - 1
+    }
+
+    fn arrays_mut(&mut self) -> [&mut Vec<T>; 6] {
+        [
+            &mut self.alpha1,
+            &mut self.alpha2,
+            &mut self.alpha3,
+            &mut self.alpha4,
+            &mut self.beta1,
+            &mut self.beta2,
+        ]
+    }
+
+    fn arrays(&self) -> [&Vec<T>; 6] {
+        [
+            &self.alpha1,
+            &self.alpha2,
+            &self.alpha3,
+            &self.alpha4,
+            &self.beta1,
+            &self.beta2,
+        ]
+    }
+
+    /// Extends every array with zeros to `len` values.
+    fn pad(&mut self, len: usize) {
+        for v in self.arrays_mut() {
+            v.resize(len, T::zero());
+        }
+    }
+
+    /// Applies `f` to every coefficient.
+    fn map(&mut self, f: impl Fn(T) -> T) {
+        for v in self.arrays_mut() {
+            for c in v.iter_mut() {
+                *c = f(*c);
+            }
+        }
+    }
+
+    /// Replaces every coefficient c by f(c, the matching coefficient of `other`).
+    fn zip_map(&mut self, other: &Self, f: impl Fn(T, T) -> T) {
+        for (v, w) in self.arrays_mut().into_iter().zip(other.arrays()) {
+            for (c, &o) in v.iter_mut().zip(w) {
+                *c = f(*c, o);
+            }
+        }
+    }
+}
+
+/// The phase-matrix elements of one table at the projector's nodes: F₁₁, F₁₂, F₃₃, F₃₄ (for a
+/// sphere F₂₂ = F₁₁ and F₄₄ = F₃₃).
+#[derive(Clone, Copy, Debug)]
+pub struct PhaseTables<'a, T> {
+    /// F₁₁.
+    pub f11: &'a [T],
+    /// F₁₂.
+    pub f12: &'a [T],
+    /// F₃₃.
+    pub f33: &'a [T],
+    /// F₃₄.
+    pub f34: &'a [T],
+}
+
+/// Gauss–Legendre quadrature that projects phase-matrix tables onto the Greek coefficients up
+/// to an order L.
+///
+/// With n nodes the projection is exact (up to rounding) for elements that are polynomials of
+/// degree ≤ 2n − 1 − L in μ; a sphere's elements have degree 2N, so n = N + ⌈(L + 1)/2⌉
+/// suffices ([`MieSolution::greek`] chooses it). Build a projector once and reuse it: for
+/// example, sum the elements of many spheres at its nodes and project once.
+#[derive(Clone, Debug)]
+pub struct GreekProjector<T> {
+    order: usize,
+    rule: GaussLegendre<T>,
+}
+
+impl<T: Real> GreekProjector<T> {
+    /// A projector for orders 0, …, `order` with an `n_nodes`-point Gauss–Legendre rule.
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`]; as [`gauss_legendre`].
+    pub fn new(order: usize, n_nodes: usize) -> Result<Self> {
+        check_order(order)?;
+        Ok(Self {
+            order,
+            rule: gauss_legendre(n_nodes)?,
+        })
+    }
+
+    /// The highest order L.
+    pub fn order(&self) -> usize {
+        self.order
+    }
+
+    /// The cosines at which to tabulate the phase matrix (ascending).
+    pub fn nodes(&self) -> &[T] {
+        &self.rule.nodes
+    }
+
+    /// The Greek coefficients of each table: c = (2l + 1)/2 · Σ wₖ F(μₖ) dₗ(μₖ). The tables share
+    /// the generalized spherical functions, which dominate the cost: O(nodes·L) for any number of
+    /// tables.
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if a table's length differs from the number of nodes.
+    pub fn project(&self, tables: &[PhaseTables<'_, T>]) -> Result<Vec<GreekCoefficients<T>>> {
+        let n = self.rule.nodes.len();
+        for t in tables {
+            let lens = [t.f11.len(), t.f12.len(), t.f33.len(), t.f34.len()];
+            if lens.iter().any(|&l| l != n) {
+                return Err(Error::InvalidInput(format!(
+                    "phase-matrix tables must have one value per node ({n}), got {lens:?}"
+                )));
+            }
+        }
+        let len = self.order + 1;
+        let mut out: Vec<_> = tables
+            .iter()
+            .map(|_| GreekCoefficients::zeros(len))
+            .collect();
+        let mut d = [
+            vec![T::zero(); len],
+            vec![T::zero(); len],
+            vec![T::zero(); len],
+            vec![T::zero(); len],
+        ];
+        for (k, (&mu, &w)) in self.rule.nodes.iter().zip(&self.rule.weights).enumerate() {
+            let [d00, d22, d2m2, d02] = &mut d;
+            wigner_d(0, 0, mu, d00)?;
+            wigner_d(2, 2, mu, d22)?;
+            wigner_d(2, -2, mu, d2m2)?;
+            wigner_d(0, 2, mu, d02)?;
+            for (t, c) in tables.iter().zip(&mut out) {
+                let f11 = w * t.f11[k];
+                let f33 = w * t.f33[k];
+                let (sum, diff) = (f11 + f33, f11 - f33);
+                let (f12, f34) = (w * t.f12[k], w * t.f34[k]);
+                for l in 0..len {
+                    c.alpha1[l] += f11 * d00[l];
+                    c.alpha4[l] += f33 * d00[l];
+                    c.alpha2[l] += sum * d22[l]; // α₂ + α₃ for now
+                    c.alpha3[l] += diff * d2m2[l]; // α₂ − α₃ for now
+                    c.beta1[l] += f12 * d02[l];
+                    c.beta2[l] += f34 * d02[l];
+                }
+            }
+        }
+        let half = T::from_f64(0.5);
+        for c in &mut out {
+            for l in 0..len {
+                let s = T::from_f64((2 * l + 1) as f64) * half;
+                let (plus, minus) = (c.alpha2[l], c.alpha3[l]);
+                c.alpha1[l] *= s;
+                c.alpha4[l] *= s;
+                c.alpha2[l] = s * half * (plus + minus);
+                c.alpha3[l] = s * half * (plus - minus);
+                c.beta1[l] *= s;
+                c.beta2[l] *= s;
+            }
+        }
+        Ok(out)
+    }
+}
+
+/// The highest order [`MieSolution::greek`] accepts, 2²¹: twice [`MAX_MIE_ORDER`], since the
+/// coefficients of an N-term sphere vanish above order 2N.
+pub const MAX_GREEK_ORDER: usize = 2 * MAX_MIE_ORDER;
+
+fn check_order(order: usize) -> Result<()> {
+    if order > MAX_GREEK_ORDER {
+        return Err(Error::InvalidInput(format!(
+            "the Greek order must be at most {MAX_GREEK_ORDER}, got {order}"
+        )));
+    }
+    Ok(())
+}
+
+/// Phase-matrix tables over four vectors.
+fn tables<T>(f: &[Vec<T>; 4]) -> PhaseTables<'_, T> {
+    PhaseTables {
+        f11: &f[0],
+        f12: &f[1],
+        f33: &f[2],
+        f34: &f[3],
+    }
+}
+
+/// Phase-matrix buffers over four vectors.
+fn buffers<T>(b: &mut [Vec<T>; 4]) -> PhaseBuffers<'_, T> {
+    let [f11, f12, f33, f34] = b;
+    PhaseBuffers { f11, f12, f33, f34 }
+}
+
+impl<T: Real> MieSolution<'_, T> {
+    /// The order L asked for (default 2N) and the projector for orders up to min(L, 2N): the
+    /// coefficients of an N-term sphere vanish above 2N.
+    fn greek_projector(&self, order: Option<usize>) -> Result<(usize, GreekProjector<T>)> {
+        let n = self.n_terms();
+        let order = order.unwrap_or(2 * n);
+        check_order(order)?;
+        let projected = order.min(2 * n);
+        Ok((
+            order,
+            GreekProjector::new(projected, n + (projected + 1).div_ceil(2))?,
+        ))
+    }
+
+    /// The Greek coefficients up to `order` (default: all, 2N), exact up to rounding: the phase
+    /// matrix at the nodes of an (N + ⌈(L + 1)/2⌉)-point Gauss–Legendre rule, projected. Orders
+    /// above 2N are exactly 0. Normalized coefficients are divided by the projected α₁ at l = 0,
+    /// so that it is exactly 1; unnormalized ones are those times Q_sca.
+    ///
+    /// The cost is the amplitudes at the N + ⌈(L + 1)/2⌉ nodes, O(N²) for L ≈ 2N, and four
+    /// Wigner recurrences up to L at each, O(N·L): for all orders, about 0.1 s at x = 10³ and
+    /// 8 s at x = 10⁴ in `f64` on an Apple M4 Pro.
+    ///
+    /// # Errors
+    ///
+    /// [`Error::InvalidInput`] if `order` > [`MAX_GREEK_ORDER`]; as [`gauss_legendre`] (above
+    /// 2¹⁷ nodes, so for N + ⌈(min(L, 2N) + 1)/2⌉ > 2¹⁷).
+    pub fn greek(
+        &self,
+        order: Option<usize>,
+        normalization: Normalization,
+    ) -> Result<GreekCoefficients<T>> {
+        let (order, projector) = self.greek_projector(order)?;
+        let n = projector.nodes().len();
+        let mut f: [Vec<T>; 4] = core::array::from_fn(|_| vec![T::zero(); n]);
+        self.phase_matrix(
+            projector.nodes(),
+            Normalization::Unnormalized,
+            buffers(&mut f),
+        )?;
+        let mut g = projector.project(&[tables(&f)])?.remove(0);
+        self.normalize(&mut g, normalization);
+        g.pad(order + 1);
+        Ok(g)
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
+        let Some(jac) = self.efficiencies_jacobian() else {
+            return Err(Error::InvalidInput(
+                "Greek derivatives need a solution from Mie::compute_with_derivatives".into(),
+            ));
+        };
+        let (order, projector) = self.greek_projector(order)?;
+        let n = projector.nodes().len();
+        let mut f: [[Vec<T>; 4]; 4] =
+            core::array::from_fn(|_| core::array::from_fn(|_| vec![T::zero(); n]));
+        {
+            let [v, dx, dr, di] = &mut f;
+            self.phase_matrix_with_derivatives(
+                projector.nodes(),
+                Normalization::Unnormalized,
+                buffers(v),
+                [buffers(dx), buffers(dr), buffers(di)],
+            )?;
+        }
+        let projected =
+            projector.project(&[tables(&f[0]), tables(&f[1]), tables(&f[2]), tables(&f[3])])?;
+        let [r, dr_x, dr_re, dr_im]: [GreekCoefficients<T>; 4] =
+            projected.try_into().expect("four tables");
+        // Normalized g = r/r₀: dg = (dr − g·dr₀)/r₀. Unnormalized: Q·g, d = dQ·g + Q·dg.
+        let r0 = r.alpha1[0];
+        let mut g = r;
+        self.normalize(&mut g, Normalization::Normalized);
+        let q = self.efficiencies().q_sca;
+        let dq = [jac.x.q_sca, jac.m_re.q_sca, jac.m_im.q_sca];
+        let mut d = [dr_x, dr_re, dr_im];
+        for (p, dp) in d.iter_mut().enumerate() {
+            let dr0 = dp.alpha1[0];
+            dp.zip_map(&g, |dv, gv| (dv - gv * dr0) / r0);
+            if normalization == Normalization::Unnormalized {
+                dp.zip_map(&g, |dv, gv| dq[p] * gv + q * dv);
+            }
+        }
+        if normalization == Normalization::Unnormalized {
+            g.map(|v| v * q);
+        }
+        g.pad(order + 1);
+        for dp in &mut d {
+            dp.pad(order + 1);
+        }
+        Ok((g, d))
+    }
+
+    /// Divides projected coefficients by their α₁ at l = 0, which is then exactly 1, and
+    /// multiplies them by Q_sca if unnormalized.
+    fn normalize(&self, g: &mut GreekCoefficients<T>, normalization: Normalization) {
+        let r0 = g.alpha1[0];
+        g.map(|v| v / r0);
+        g.alpha1[0] = T::one();
+        if normalization == Normalization::Unnormalized {
+            let q = self.efficiencies().q_sca;
+            g.map(|v| v * q);
+        }
+    }
+}
diff --git a/scatrs/src/mie/mod.rs b/scatrs/src/mie/mod.rs
index 168e634..027c91e 100644
--- a/scatrs/src/mie/mod.rs
+++ b/scatrs/src/mie/mod.rs
@@ -70,7 +70,9 @@
 mod amplitudes;
 mod domain;
 mod efficiencies;
+mod greek;
 mod kernel;
+mod phase;
 
 use num_complex::Complex;
 
@@ -80,6 +82,8 @@ use crate::{Error, Result};
 pub use amplitudes::AmplitudeBuffers;
 pub use domain::{MieOutput, VALIDATED_DOMAIN, ValidatedDomain};
 pub use efficiencies::{Efficiencies, EfficiencyJacobian};
+pub use greek::{GreekCoefficients, GreekProjector, MAX_GREEK_ORDER, PhaseTables};
+pub use phase::{Normalization, PhaseBuffers};
 
 /// The largest number of terms [`Mie::compute`] accepts, 2²⁰.
 ///
diff --git a/scatrs/src/mie/phase.rs b/scatrs/src/mie/phase.rs
new file mode 100644
index 0000000..88d1625
--- /dev/null
+++ b/scatrs/src/mie/phase.rs
@@ -0,0 +1,196 @@
+//! The phase matrix of a sphere.
+
+use num_complex::Complex;
+
+use super::{AmplitudeBuffers, MieSolution};
+use crate::scalar::Real;
+use crate::{Error, Result};
+
+/// How the phase matrix (and the Greek coefficients) are scaled.
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+#[non_exhaustive]
+pub enum Normalization {
+    /// ½∫F₁₁ dμ = 1 (so that the Greek coefficient α₁ at l = 0 is 1).
+    Normalized,
+    /// The normalized form times Q_sca: the phase matrix per unit geometric cross section, the
+    /// quantity that sums over a size distribution.
+    Unnormalized,
+}
+
+/// Output buffers for the independent elements of a sphere's phase matrix, or for their
+/// derivatives, one value per cosine. For a sphere F₂₂ = F₁₁ and F₄₄ = F₃₃.
+#[derive(Debug)]
+pub struct PhaseBuffers<'b, T> {
+    /// F₁₁ = (|S₁|² + |S₂|²)/2, scaled.
+    pub f11: &'b mut [T],
+    /// F₁₂ = (|S₂|² − |S₁|²)/2, scaled.
+    pub f12: &'b mut [T],
+    /// F₃₃ = Re(S₂S₁*), scaled.
+    pub f33: &'b mut [T],
+    /// F₃₄ = Im(S₂S₁*), scaled.
+    pub f34: &'b mut [T],
+}
+
+impl<T> PhaseBuffers<'_, T> {
+    fn check(&self, len: usize, what: &str) -> Result<()> {
+        let lens = [
+            self.f11.len(),
+            self.f12.len(),
+            self.f33.len(),
+            self.f34.len(),
+        ];
+        if lens.iter().any(|&l| l != len) {
+            return Err(Error::InvalidInput(format!(
+                "the {what} buffers must have one value per cosine ({len}), got {lens:?}"
+            )));
+        }
+        Ok(())
+    }
+}
+
+/// (|S₁|² + |S₂|²)/2, (|S₂|² − |S₁|²)/2, Re(S₂S₁*), Im(S₂S₁*).
+fn elements<T: Real>(s1: Complex<T>, s2: Complex<T>) -> [T; 4] {
+    let half = T::from_f64(0.5);
+    let (n1, n2) = (s1.norm_sqr(), s2.norm_sqr());
+    let p = s2 * s1.conj();
+    [half * (n1 + n2), half * (n2 - n1), p.re, p.im]
+}
+
+/// The derivatives of [`elements`] for amplitude derivatives (d1, d2).
+fn element_derivatives<T: Real>(
+    s1: Complex<T>,
+    s2: Complex<T>,
+    d1: Complex<T>,
+    d2: Complex<T>,
+) -> [T; 4] {
+    // d|S|² = 2 Re(S* dS); d(S₂S₁*) = dS₂ S₁* + S₂ dS₁*.
+    let r1 = (s1.conj() * d1).re;
+    let r2 = (s2.conj() * d2).re;
+    let p = d2 * s1.conj() + s2 * d1.conj();
+    [r1 + r2, r2 - r1, p.re, p.im]
+}
+
+impl<T: Real> MieSolution<'_, T> {
+    /// The scale of the phase matrix: 4/(x²Q_sca) for [`Normalization::Normalized`], 4/x² for
+    /// [`Normalization::Unnormalized`].
+    fn phase_scale(&self, normalization: Normalization) -> T {
+        let x = self.x();
+        let four_x2 = T::from_f64(4.0) / (x * x);
+        match normalization {
+            Normalization::Normalized => four_x2 / self.efficiencies().q_sca,
+            Normalization::Unnormalized => four_x2,
+        }
+    }
+
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
+        out.check(mu.len(), "phase-matrix")?;
+        let zero = Complex::new(T::zero(), T::zero());
+        let (mut s1, mut s2) = (vec![zero; mu.len()], vec![zero; mu.len()]);
+        self.amplitudes(
+            mu,
+            AmplitudeBuffers {
+                s1: &mut s1,
+                s2: &mut s2,
+            },
+        )?;
+        let c = self.phase_scale(normalization);
+        for j in 0..mu.len() {
+            let f = elements(s1[j], s2[j]);
+            out.f11[j] = c * f[0];
+            out.f12[j] = c * f[1];
+            out.f33[j] = c * f[2];
+            out.f34[j] = c * f[3];
+        }
+        Ok(())
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
+        let Some(jac) = self.efficiencies_jacobian() else {
+            return Err(Error::InvalidInput(
+                "phase-matrix derivatives need a solution from Mie::compute_with_derivatives"
+                    .into(),
+            ));
+        };
+        values.check(mu.len(), "phase-matrix")?;
+        for buffers in &d {
+            buffers.check(mu.len(), "derivative")?;
+        }
+        let n = mu.len();
+        let zero = Complex::new(T::zero(), T::zero());
+        let mut s: [Vec<Complex<T>>; 6] = core::array::from_fn(|_| vec![zero; n]);
+        {
+            let [s1, s2, dx1, dx2, dm1, dm2] = &mut s;
+            self.amplitudes_with_derivatives(
+                mu,
+                AmplitudeBuffers { s1, s2 },
+                AmplitudeBuffers { s1: dx1, s2: dx2 },
+                AmplitudeBuffers { s1: dm1, s2: dm2 },
+            )?;
+        }
+        // The scale c and its derivatives: c = 4/(x²Q) or 4/x².
+        let x = self.x();
+        let c = self.phase_scale(normalization);
+        let two = T::from_f64(2.0);
+        let dc = match normalization {
+            Normalization::Normalized => {
+                let q = self.efficiencies().q_sca;
+                [
+                    -c * (two / x + jac.x.q_sca / q),
+                    -c * jac.m_re.q_sca / q,
+                    -c * jac.m_im.q_sca / q,
+                ]
+            }
+            Normalization::Unnormalized => [-two * c / x, T::zero(), T::zero()],
+        };
+        let i1 = Complex::new(T::zero(), T::one());
+        let v = values;
+        let [mut dx, mut dr, mut di] = d;
+        for j in 0..n {
+            let (s1, s2) = (s[0][j], s[1][j]);
+            let f = elements(s1, s2);
+            // ∂S/∂x, ∂S/∂(Re m) = dS/dm, ∂S/∂(Im m) = i·dS/dm.
+            let df = [
+                element_derivatives(s1, s2, s[2][j], s[3][j]),
+                element_derivatives(s1, s2, s[4][j], s[5][j]),
+                element_derivatives(s1, s2, i1 * s[4][j], i1 * s[5][j]),
+            ];
+            v.f11[j] = c * f[0];
+            v.f12[j] = c * f[1];
+            v.f33[j] = c * f[2];
+            v.f34[j] = c * f[3];
+            for (p, out) in [&mut dx, &mut dr, &mut di].into_iter().enumerate() {
+                let g = |k: usize| c * df[p][k] + dc[p] * f[k];
+                out.f11[j] = g(0);
+                out.f12[j] = g(1);
+                out.f33[j] = g(2);
+                out.f34[j] = g(3);
+            }
+        }
+        Ok(())
+    }
+}
diff --git a/scatrs/tests/common/greek.rs b/scatrs/tests/common/greek.rs
new file mode 100644
index 0000000..2907283
--- /dev/null
+++ b/scatrs/tests/common/greek.rs
@@ -0,0 +1,272 @@
+//! Shared by the phase-matrix and Greek-coefficient tests: solving one sphere for both, and the
+//! scale (the denominator) and condition number of each error measure of
+//! `scatrs::mie::MieOutput::{PhaseMatrix, PhaseMatrixDerivatives, GreekCoefficients,
+//! GreekDerivatives}`, from double-double solutions.
+
+use num_complex::Complex;
+use scatrs::Real;
+use scatrs::mie::{GreekCoefficients, Mie, MieOptions, Normalization, PhaseBuffers};
+use scatrs::scalar::DoubleDouble as Dd;
+
+use super::mie::Measure;
+
+/// The names of the Greek coefficients, as in the reference records.
+pub const GREEK_NAMES: [&str; 6] = ["alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2"];
+
+/// The names of the phase-matrix elements, as in the reference records.
+pub const PHASE_NAMES: [&str; 4] = ["f11", "f12", "f33", "f34"];
+
+/// Everything the phase-matrix and Greek tests compare, from one solution.
+#[derive(Clone, Debug)]
+pub struct GreekOutputs<T> {
+    /// α₁, α₂, α₃, α₄, β₁, β₂ for l = 0, …, L.
+    pub greek: [Vec<T>; 6],
+    /// Their derivatives with respect to x, Re m and Im m.
+    pub greek_d: [[Vec<T>; 6]; 3],
+    /// F₁₁, F₁₂, F₃₃, F₃₄ at each cosine.
+    pub phase: [Vec<T>; 4],
+    /// Their derivatives with respect to x, Re m and Im m.
+    pub phase_d: [[Vec<T>; 4]; 3],
+}
+
+/// The arrays of `g`, ordered as `GREEK_NAMES`.
+pub fn greek_arrays<T: Clone>(g: &GreekCoefficients<T>) -> [Vec<T>; 6] {
+    [
+        g.alpha1.clone(),
+        g.alpha2.clone(),
+        g.alpha3.clone(),
+        g.alpha4.clone(),
+        g.beta1.clone(),
+        g.beta2.clone(),
+    ]
+}
+
+fn buffers<T>(b: &mut [Vec<T>; 4]) -> PhaseBuffers<'_, T> {
+    let [f11, f12, f33, f34] = b;
+    PhaseBuffers { f11, f12, f33, f34 }
+}
+
+/// Solves (x, m) with derivatives, for the Greek coefficients up to `order` (default 2N) and
+/// the phase matrix at `mu`, inside or outside the domain.
+pub fn solve_greek<T: Real>(
+    x: T,
+    m: Complex<T>,
+    mu: &[T],
+    n_max: Option<usize>,
+    order: Option<usize>,
+    normalization: Normalization,
+) -> GreekOutputs<T> {
+    let mut opts = MieOptions::new().with_allow_unvalidated(true);
+    if let Some(n) = n_max {
+        opts = opts.with_n_max(n);
+    }
+    let mut mie = Mie::<T>::new();
+    let sol = mie
+        .compute_with_derivatives(x, m, &opts)
+        .expect("valid inputs");
+    let (g, dg) = sol
+        .greek_with_derivatives(order, normalization)
+        .expect("valid order");
+    let zeros = || vec![T::zero(); mu.len()];
+    let mut phase: [Vec<T>; 4] = std::array::from_fn(|_| zeros());
+    let mut phase_d: [[Vec<T>; 4]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| zeros()));
+    {
+        let [dx, dr, di] = &mut phase_d;
+        sol.phase_matrix_with_derivatives(
+            mu,
+            normalization,
+            buffers(&mut phase),
+            [buffers(dx), buffers(dr), buffers(di)],
+        )
+        .expect("valid cosines");
+    }
+    GreekOutputs {
+        greek: greek_arrays(&g),
+        greek_d: dg.each_ref().map(greek_arrays),
+        phase,
+        phase_d,
+    }
+}
+
+/// The double-double solution at (x, m) with `n_terms` terms (those of the `f64` solution, so
+/// that both expand the same series), the Greek coefficients up to `order` and the phase
+/// matrix at `mu`.
+pub fn greek_reference(
+    x: f64,
+    m: Complex<f64>,
+    mu: &[f64],
+    n_terms: usize,
+    order: Option<usize>,
+    normalization: Normalization,
+) -> GreekOutputs<Dd> {
+    let mu_dd: Vec<Dd> = mu.iter().map(|&v| Dd::from_f64(v)).collect();
+    solve_greek(
+        Dd::from_f64(x),
+        Complex::new(Dd::from_f64(m.re), Dd::from_f64(m.im)),
+        &mu_dd,
+        Some(n_terms),
+        order,
+        normalization,
+    )
+}
+
+/// The measures of the values at (x, m).
+pub struct GreekValueMeasures {
+    /// α₁, …, β₂, over all orders.
+    pub greek: [Measure; 6],
+    /// F₁₁, F₁₂, F₃₃, F₃₄ at each cosine.
+    pub phase: Vec<[Measure; 4]>,
+}
+
+/// The measures of the derivatives at (x, m).
+pub struct GreekDerivativeMeasures {
+    /// [∂/∂x, ∂/∂Re m, ∂/∂Im m][α₁, …, β₂], over all orders.
+    pub greek: [[Measure; 6]; 3],
+    /// [∂/∂x, ∂/∂Re m, ∂/∂Im m][F₁₁, F₁₂, F₃₃, F₃₄], over all cosines.
+    pub phase: [[Measure; 4]; 3],
+}
+
+/// F̄ = ½∫F₁₁ dμ, which is α₁ at l = 0 (1 normalized, Q_sca unnormalized).
+fn mean_f11(d: &GreekOutputs<Dd>) -> f64 {
+    d.greek[0][0].hi()
+}
+
+/// A = max over l of |α₁|.
+fn alpha1_size(d: &GreekOutputs<Dd>) -> f64 {
+    d.greek[0].iter().map(|v| v.hi().abs()).fold(0.0, f64::max)
+}
+
+/// Componentwise: x·|∂y/∂x| + |Re m|·|∂y/∂(Re m)| + |Im m|·|∂y/∂(Im m)|.
+fn sensitivity(x: f64, m: Complex<f64>, d: [f64; 3]) -> f64 {
+    x * d[0].abs() + m.re.abs() * d[1].abs() + m.im.abs() * d[2].abs()
+}
+
+/// The measures of the values at (x, m), from the double-double solution `d` there.
+pub fn greek_value_measures(x: f64, m: Complex<f64>, d: &GreekOutputs<Dd>) -> GreekValueMeasures {
+    let scale = alpha1_size(d);
+    let greek = std::array::from_fn(|k| {
+        let num = (0..d.greek[k].len())
+            .map(|l| sensitivity(x, m, std::array::from_fn(|p| d.greek_d[p][k][l].hi())))
+            .fold(0.0, f64::max);
+        Measure {
+            scale,
+            condition: num / scale,
+        }
+    });
+    let mean = mean_f11(d);
+    let phase = (0..d.phase[0].len())
+        .map(|j| {
+            let scale = d.phase[0][j].hi().max(mean);
+            std::array::from_fn(|k| Measure {
+                scale,
+                condition: sensitivity(x, m, std::array::from_fn(|p| d.phase_d[p][k][j].hi()))
+                    / scale,
+            })
+        })
+        .collect();
+    GreekValueMeasures { greek, phase }
+}
+
+/// The measures of the derivatives at (x, m), from the double-double solution `d` there (with
+/// `n_terms` terms, the Greek coefficients up to `order` and the phase matrix at `mu`) and
+/// central differences of double-double derivatives, with the steps of
+/// `common::mie::derivative_measures` (2⁻³⁰ x and 2⁻³⁰ |m|, one-sided in Im m at Im m = 0).
+pub fn greek_derivative_measures(
+    x: f64,
+    m: Complex<f64>,
+    mu: &[f64],
+    n_terms: usize,
+    order: usize,
+    normalization: Normalization,
+    d: &GreekOutputs<Dd>,
+) -> GreekDerivativeMeasures {
+    let h = 2f64.powi(-30);
+    let (h_x, h_m) = (x * h, m.norm() * h);
+    let one_sided = m.im < h_m;
+    let at = |x: f64, re: f64, im: f64| {
+        greek_reference(
+            x,
+            Complex::new(re, im),
+            mu,
+            n_terms,
+            Some(order),
+            normalization,
+        )
+    };
+    // [+, −] in x, Re m and Im m, and the width of each difference.
+    let steps = [
+        [at(x + h_x, m.re, m.im), at(x - h_x, m.re, m.im)],
+        [at(x, m.re + h_m, m.im), at(x, m.re - h_m, m.im)],
+        [
+            at(x, m.re, m.im + h_m),
+            if one_sided {
+                d.clone()
+            } else {
+                at(x, m.re, m.im - h_m)
+            },
+        ],
+    ];
+    let widths = [
+        2.0 * h_x,
+        2.0 * h_m,
+        if one_sided { h_m } else { 2.0 * h_m },
+    ];
+    let fd = |q: usize, plus: Dd, minus: Dd| ((plus - minus) / Dd::from_f64(widths[q])).hi();
+    let phase_of_x = 1.0 + m.norm() * x;
+    let p_sizes = [x, m.norm(), m.norm()];
+    let a = alpha1_size(d);
+    let greek = std::array::from_fn(|p| {
+        std::array::from_fn(|k| {
+            let natural = a * phase_of_x / p_sizes[p];
+            let values = &d.greek_d[p][k];
+            let scale = values
+                .iter()
+                .map(|v| v.hi().abs().max(natural))
+                .fold(0.0, f64::max);
+            let num = (0..values.len())
+                .map(|l| {
+                    let g = std::array::from_fn(|q| {
+                        fd(
+                            q,
+                            steps[q][0].greek_d[p][k][l],
+                            steps[q][1].greek_d[p][k][l],
+                        )
+                    });
+                    sensitivity(x, m, g)
+                })
+                .fold(0.0, f64::max);
+            Measure {
+                scale,
+                condition: num / scale,
+            }
+        })
+    });
+    let mean = mean_f11(d);
+    let phase = std::array::from_fn(|p| {
+        std::array::from_fn(|k| {
+            let scale = (0..mu.len())
+                .map(|j| {
+                    let natural = d.phase[0][j].hi().max(mean) * phase_of_x / p_sizes[p];
+                    d.phase_d[p][k][j].hi().abs().max(natural)
+                })
+                .fold(0.0, f64::max);
+            let num = (0..mu.len())
+                .map(|j| {
+                    let g = std::array::from_fn(|q| {
+                        fd(
+                            q,
+                            steps[q][0].phase_d[p][k][j],
+                            steps[q][1].phase_d[p][k][j],
+                        )
+                    });
+                    sensitivity(x, m, g)
+                })
+                .fold(0.0, f64::max);
+            Measure {
+                scale,
+                condition: num / scale,
+            }
+        })
+    });
+    GreekDerivativeMeasures { greek, phase }
+}
diff --git a/scatrs/tests/common/mod.rs b/scatrs/tests/common/mod.rs
index fd425e3..083bf62 100644
--- a/scatrs/tests/common/mod.rs
+++ b/scatrs/tests/common/mod.rs
@@ -1,6 +1,7 @@
 //! Loader for the committed Arb reference data in `reference/data/`.
 #![allow(dead_code)]
 
+pub mod greek;
 pub mod mie;
 
 use std::collections::{BTreeMap, BTreeSet};
diff --git a/scatrs/tests/mie_greek.rs b/scatrs/tests/mie_greek.rs
new file mode 100644
index 0000000..d200640
--- /dev/null
+++ b/scatrs/tests/mie_greek.rs
@@ -0,0 +1,492 @@
+//! The phase matrix and the Greek coefficients of spheres: identities, exactness of the
+//! projection, the `Dual` derivative oracle, and the rejection of bad inputs.
+#![allow(clippy::needless_range_loop)] // parallel arrays are indexed together
+
+mod common;
+
+use common::greek::{GreekOutputs, solve_greek};
+use num_complex::Complex;
+use scatrs::Error;
+use scatrs::mie::{
+    GreekCoefficients, GreekProjector, Mie, MieOptions, Normalization, PhaseBuffers, PhaseTables,
+    ValidatedDomain,
+};
+use scatrs::scalar::Dual;
+use scatrs::special::{gauss_legendre, wigner_d_vec};
+
+const EPS: f64 = f64::EPSILON;
+
+fn c(re: f64, im: f64) -> Complex<f64> {
+    Complex::new(re, im)
+}
+
+/// Spheres from the Rayleigh regime to large, weakly and strongly absorbing, below and above
+/// m = 1.
+const SPHERES: [(f64, Complex<f64>); 7] = [
+    (0.05, Complex::new(1.5, 0.0)),
+    (1.0, Complex::new(1.33, 1e-8)),
+    (3.0, Complex::new(1.0001, 0.0)),
+    (5.0, Complex::new(1.5, 0.01)),
+    (12.0, Complex::new(0.75, 0.0)),
+    (40.0, Complex::new(1.75, 0.44)),
+    (150.0, Complex::new(2.5, 1.0)),
+];
+
+fn bufs(t: &mut [Vec<f64>; 4]) -> PhaseBuffers<'_, f64> {
+    let [f11, f12, f33, f34] = t;
+    PhaseBuffers { f11, f12, f33, f34 }
+}
+
+fn solve(x: f64, m: Complex<f64>, mu: &[f64], normalization: Normalization) -> GreekOutputs<f64> {
+    solve_greek(x, m, mu, None, None, normalization)
+}
+
+/// The sum Σ cₗ d^l_{mn}(μ) over the coefficients `coef`.
+fn expand(coef: &[f64], m: i32, n: i32, mu: f64) -> f64 {
+    let d = wigner_d_vec(m, n, coef.len() - 1, mu).unwrap();
+    coef.iter().zip(&d).map(|(c, d)| c * d).sum()
+}
+
+#[test]
+fn a_sphere_is_a_pure_scatterer() {
+    // For a single sphere F₁₁² = F₁₂² + F₃₃² + F₃₄² (Bohren & Huffman Eq. 4.79), and the
+    // normalized F₁₁ averages to 1 over μ.
+    let mu: Vec<f64> = (0..=40).map(|k| -1.0 + f64::from(k) / 20.0).collect();
+    for (x, m) in SPHERES {
+        let f = solve(x, m, &mu, Normalization::Normalized);
+        for j in 0..mu.len() {
+            let [f11, f12, f33, f34] = [0, 1, 2, 3].map(|k| f.phase[k][j]);
+            let rest = (f12 * f12 + f33 * f33 + f34 * f34).sqrt();
+            assert!(
+                (f11 - rest).abs() <= 1e-12 * f11,
+                "x = {x}, m = {m}, mu = {}: F11 = {f11:e}, |(F12, F33, F34)| = {rest:e}",
+                mu[j]
+            );
+        }
+        let n = 2 * x as usize + 40;
+        let rule = gauss_legendre::<f64>(n).unwrap();
+        let g = solve(x, m, &rule.nodes, Normalization::Normalized);
+        let mean: f64 = rule
+            .weights
+            .iter()
+            .zip(&g.phase[0])
+            .map(|(w, f)| w * f)
+            .sum::<f64>()
+            / 2.0;
+        assert!(
+            (mean - 1.0).abs() <= 1e-12,
+            "x = {x}, m = {m}: mean F11 = {mean}"
+        );
+    }
+}
+
+#[test]
+fn unnormalized_is_normalized_times_q_sca() {
+    let mu = [1.0, 0.5, 0.0, -1.0];
+    for (x, m) in SPHERES {
+        let q = Mie::<f64>::new()
+            .compute(x, m, &MieOptions::new())
+            .unwrap()
+            .efficiencies()
+            .q_sca;
+        let n = solve(x, m, &mu, Normalization::Normalized);
+        let u = solve(x, m, &mu, Normalization::Unnormalized);
+        let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 8.0 * EPS * scale;
+        for k in 0..4 {
+            for j in 0..mu.len() {
+                let s = n.phase[0][j] * q;
+                assert!(
+                    close(u.phase[k][j], n.phase[k][j] * q, s),
+                    "x = {x}: F{k} at {}",
+                    mu[j]
+                );
+            }
+        }
+        for k in 0..6 {
+            let s = n.greek[0].iter().fold(0.0f64, |a, v| a.max(v.abs())) * q;
+            for l in 0..n.greek[k].len() {
+                assert!(
+                    close(u.greek[k][l], n.greek[k][l] * q, s),
+                    "x = {x}: Greek {k} l = {l}"
+                );
+            }
+        }
+        assert_eq!(n.greek[0][0], 1.0, "x = {x}: alpha1 at l = 0");
+    }
+}
+
+#[test]
+fn greek_coefficients_expand_the_phase_matrix() {
+    // α₁ at l = 1 is 3g; the expansions (MTL 2002 Eqs. 4.75–4.80) reproduce the phase matrix.
+    let mu = [1.0, 0.95, 0.3, -0.4, -1.0];
+    for (x, m) in SPHERES {
+        let f = solve(x, m, &mu, Normalization::Normalized);
+        let g = Mie::<f64>::new()
+            .compute(x, m, &MieOptions::new())
+            .unwrap()
+            .efficiencies()
+            .g;
+        let [a1, a2, a3, a4, b1, b2] = &f.greek;
+        assert!(
+            (a1[1] - 3.0 * g).abs() <= 1e-13 * a1[1].abs().max(1e-3),
+            "x = {x}: 3g"
+        );
+        let plus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a + b).collect();
+        let minus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a - b).collect();
+        let scale = a1.iter().fold(0.0f64, |s, v| s.max(v.abs()));
+        for (j, &u) in mu.iter().enumerate() {
+            let (f11, f12, f33, f34) = (f.phase[0][j], f.phase[1][j], f.phase[2][j], f.phase[3][j]);
+            let tol = 1e-12 * scale * a1.len() as f64 * f11.max(1.0);
+            let checks = [
+                ("F11", expand(a1, 0, 0, u), f11),
+                ("F22 + F33", expand(&plus, 2, 2, u), f11 + f33),
+                ("F22 - F33", expand(&minus, 2, -2, u), f11 - f33),
+                ("F44", expand(a4, 0, 0, u), f33),
+                ("F12", expand(b1, 0, 2, u), f12),
+                ("F34", expand(b2, 0, 2, u), f34),
+            ];
+            for (name, sum, value) in checks {
+                assert!(
+                    (sum - value).abs() <= tol,
+                    "x = {x}, m = {m}, mu = {u}: {name} = {value:e}, expansion {sum:e}"
+                );
+            }
+        }
+    }
+}
+
+#[test]
+fn orders_above_2n_are_exactly_zero_and_lower_orders_a_prefix() {
+    let (x, m) = (5.0, c(1.5, 0.01));
+    let mut mie = Mie::<f64>::new();
+    let s = mie.compute(x, m, &MieOptions::new()).unwrap();
+    let n = s.n_terms();
+    let all = s.greek(None, Normalization::Normalized).unwrap();
+    assert_eq!(all.order(), 2 * n);
+    let more = s.greek(Some(2 * n + 7), Normalization::Normalized).unwrap();
+    assert_eq!(more.order(), 2 * n + 7);
+    let fields = |g: &GreekCoefficients<f64>| {
+        [
+            g.alpha1.clone(),
+            g.alpha2.clone(),
+            g.alpha3.clone(),
+            g.alpha4.clone(),
+            g.beta1.clone(),
+            g.beta2.clone(),
+        ]
+    };
+    for (a, b) in fields(&all).iter().zip(fields(&more).iter()) {
+        assert_eq!(&b[..=2 * n], &a[..], "orders up to 2N agree bitwise");
+        assert!(
+            b[2 * n + 1..].iter().all(|&v| v == 0.0),
+            "orders above 2N are 0"
+        );
+    }
+    let low = s.greek(Some(6), Normalization::Normalized).unwrap();
+    assert_eq!(low.order(), 6);
+    for (a, b) in fields(&all).iter().zip(fields(&low).iter()) {
+        for l in 0..=6 {
+            assert!(
+                (a[l] - b[l]).abs() <= 1e-13 * a[l].abs().max(1.0),
+                "order {l}"
+            );
+        }
+    }
+    // β₁, β₂ and the d²² parts start at l = 2.
+    assert_eq!(
+        [all.beta1[0], all.beta1[1], all.beta2[0], all.beta2[1]],
+        [0.0; 4]
+    );
+    assert_eq!(
+        [all.alpha2[0], all.alpha2[1], all.alpha3[0], all.alpha3[1]],
+        [0.0; 4]
+    );
+}
+
+#[test]
+fn the_projector_is_exact_for_polynomials() {
+    // Tables built from known coefficients up to order K, projected with a rule exact for
+    // degree K + L: the coefficients come back, and those above K are 0.
+    let (k_max, order) = (37usize, 45usize);
+    let coef = |s: f64| -> Vec<f64> {
+        (0..=k_max)
+            .map(|l| ((l as f64 + s) * 0.7).sin() / (1.0 + l as f64))
+            .collect()
+    };
+    let mut known = [
+        coef(0.1),
+        coef(0.2),
+        coef(0.3),
+        coef(0.4),
+        coef(0.5),
+        coef(0.6),
+    ];
+    for k in [1, 2, 4, 5] {
+        known[k][0] = 0.0;
+        known[k][1] = 0.0;
+    }
+    let [a1, a2, a3, a4, b1, b2] = &known;
+    let n_nodes = (k_max + order) / 2 + 1;
+    let p = GreekProjector::<f64>::new(order, n_nodes).unwrap();
+    assert_eq!(p.order(), order);
+    let nodes = p.nodes().to_vec();
+    let plus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a + b).collect();
+    let minus: Vec<f64> = a2.iter().zip(a3).map(|(a, b)| a - b).collect();
+    let f11: Vec<f64> = nodes.iter().map(|&u| expand(a1, 0, 0, u)).collect();
+    let f33: Vec<f64> = nodes.iter().map(|&u| expand(a4, 0, 0, u)).collect();
+    // F₂₂ = F₁₁ for a sphere, so α₂ ± α₃ must expand F₁₁ ± F₃₃: build tables that satisfy it by
+    // taking F₁₁ + F₃₃ and F₁₁ − F₃₃ from α₂ ± α₃ instead, with F₁₁ and F₃₃ from those.
+    let sum: Vec<f64> = nodes.iter().map(|&u| expand(&plus, 2, 2, u)).collect();
+    let diff: Vec<f64> = nodes.iter().map(|&u| expand(&minus, 2, -2, u)).collect();
+    let (f11_s, f33_s): (Vec<f64>, Vec<f64>) = sum
+        .iter()
+        .zip(&diff)
+        .map(|(s, d)| ((s + d) / 2.0, (s - d) / 2.0))
+        .unzip();
+    let f12: Vec<f64> = nodes.iter().map(|&u| expand(b1, 0, 2, u)).collect();
+    let f34: Vec<f64> = nodes.iter().map(|&u| expand(b2, 0, 2, u)).collect();
+    let out = p
+        .project(&[
+            PhaseTables {
+                f11: &f11,
+                f12: &f12,
+                f33: &f33,
+                f34: &f34,
+            },
+            PhaseTables {
+                f11: &f11_s,
+                f12: &f12,
+                f33: &f33_s,
+                f34: &f34,
+            },
+        ])
+        .unwrap();
+    let tol = 1e-13;
+    let get = |g: &GreekCoefficients<f64>| {
+        [
+            g.alpha1.clone(),
+            g.alpha2.clone(),
+            g.alpha3.clone(),
+            g.alpha4.clone(),
+            g.beta1.clone(),
+            g.beta2.clone(),
+        ]
+    };
+    let [g1, g2] = [get(&out[0]), get(&out[1])];
+    for l in 0..=order {
+        let want = |k: usize| if l <= k_max { known[k][l] } else { 0.0 };
+        // α₁, α₄, β₁, β₂ from the first table; α₂, α₃ from the second.
+        for k in [0, 3, 4, 5] {
+            assert!(
+                (g1[k][l] - want(k)).abs() <= tol,
+                "table 1, element {k}, l = {l}"
+            );
+        }
+        for k in [1, 2] {
+            assert!(
+                (g2[k][l] - want(k)).abs() <= tol,
+                "table 2, element {k}, l = {l}"
+            );
+        }
+    }
+}
+
+type D3 = Dual<f64, 3>;
+
+/// The closed-form derivatives against `Dual<f64, 3>` (seeds x, Re m, Im m) propagated through
+/// the same computation: the values agree bitwise, and the derivatives within 16·κ′·2⁻⁵² of
+/// the measure scales of `MieOutput::PhaseMatrixDerivatives` and `GreekDerivatives`, with κ′
+/// their a-priori factors.
+#[test]
+fn closed_form_derivatives_agree_with_dual() {
+    let mu = [1.0, 0.6, 0.0, -0.3, -1.0];
+    for normalization in [Normalization::Normalized, Normalization::Unnormalized] {
+        for (x, m) in SPHERES {
+            let f = solve(x, m, &mu, normalization);
+            let xd = D3::new(x, [1.0, 0.0, 0.0]);
+            let md = Complex::new(
+                D3::new(m.re, [0.0, 1.0, 0.0]),
+                D3::new(m.im, [0.0, 0.0, 1.0]),
+            );
+            let mu_d: Vec<D3> = mu.iter().map(|&v| D3::constant(v)).collect();
+            let d = solve_greek(xd, md, &mu_d, None, None, normalization);
+            let near_one = (m.norm() / (m - 1.0).norm()).max(1.0);
+            let kappa = ValidatedDomain::condition_factor(x, m) * near_one;
+            let l = f.greek[0].len() as f64;
+            let bound_f = 16.0 * kappa * EPS;
+            let bound_g = 16.0 * (kappa + (l * l + x * x) / 8.0) * EPS;
+            let phase = 1.0 + m.norm() * x;
+            let sizes = [x, m.norm(), m.norm()];
+            let mean = f.greek[0][0];
+            for k in 0..4 {
+                for j in 0..mu.len() {
+                    assert_eq!(d.phase[k][j].re(), f.phase[k][j], "x = {x}: F{k} value");
+                }
+                for p in 0..3 {
+                    let natural = |j: usize| f.phase[0][j].max(mean) * phase / sizes[p];
+                    let scale = (0..mu.len())
+                        .map(|j| f.phase_d[p][k][j].abs().max(natural(j)))
+                        .fold(0.0, f64::max);
+                    for j in 0..mu.len() {
+                        let err = (d.phase[k][j].eps()[p] - f.phase_d[p][k][j]).abs();
+                        assert!(
+                            err <= bound_f * scale,
+                            "{normalization:?} x = {x}, m = {m}: dF{k}/dp{p} at {}: {err:e}",
+                            mu[j]
+                        );
+                    }
+                }
+            }
+            let a = f.greek[0].iter().fold(0.0f64, |s, v| s.max(v.abs()));
+            for k in 0..6 {
+                for (lv, v) in d.greek[k].iter().enumerate() {
+                    assert_eq!(
+                        v.re(),
+                        f.greek[k][lv],
+                        "x = {x}: Greek {k} value at l = {lv}"
+                    );
+                }
+                for p in 0..3 {
+                    let natural = a * phase / sizes[p];
+                    let scale = f.greek_d[p][k]
+                        .iter()
+                        .map(|v| v.abs().max(natural))
+                        .fold(0.0, f64::max);
+                    for (lv, v) in d.greek[k].iter().enumerate() {
+                        let err = (v.eps()[p] - f.greek_d[p][k][lv]).abs();
+                        assert!(
+                            err <= bound_g * scale,
+                            "{normalization:?} x = {x}, m = {m}: Greek {k} p{p} l = {lv}: {err:e}"
+                        );
+                    }
+                }
+            }
+        }
+    }
+}
+
+#[test]
+fn bad_buffers_tables_and_solutions_are_rejected() {
+    let mut mie = Mie::<f64>::new();
+    let s = mie.compute(2.0, c(1.5, 0.0), &MieOptions::new()).unwrap();
+    let mu = [0.5, 0.0];
+    let mut f: [Vec<f64>; 4] = std::array::from_fn(|_| vec![7.0; 2]);
+    f[2] = vec![7.0; 3];
+    let [f11, f12, f33, f34] = &mut f;
+    let err = s.phase_matrix(
+        &mu,
+        Normalization::Normalized,
+        PhaseBuffers { f11, f12, f33, f34 },
+    );
+    assert!(matches!(err, Err(Error::InvalidInput(_))), "{err:?}");
+    assert!(
+        f.iter().all(|v| v.iter().all(|&x| x == 7.0)),
+        "nothing written"
+    );
+    let mut g: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; 2]);
+    let [f11, f12, f33, f34] = &mut g;
+    let bad = s.phase_matrix(
+        &[1.5],
+        Normalization::Normalized,
+        PhaseBuffers {
+            f11: &mut f11[..1],
+            f12: &mut f12[..1],
+            f33: &mut f33[..1],
+            f34: &mut f34[..1],
+        },
+    );
+    assert!(matches!(bad, Err(Error::InvalidInput(_))), "{bad:?}");
+    // Derivatives need a solution with derivatives.
+    let mut v: [[Vec<f64>; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; 2]));
+    let [a, b, cc, dd] = &mut v;
+    let r = s.phase_matrix_with_derivatives(
+        &mu,
+        Normalization::Normalized,
+        bufs(a),
+        [bufs(b), bufs(cc), bufs(dd)],
+    );
+    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
+    let r = s.greek_with_derivatives(None, Normalization::Normalized);
+    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
+    // Tables of the wrong length.
+    let p = GreekProjector::<f64>::new(4, 6).unwrap();
+    let short = vec![0.0; 5];
+    let r = p.project(&[PhaseTables {
+        f11: &short,
+        f12: &short,
+        f33: &short,
+        f34: &short,
+    }]);
+    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
+    assert!(matches!(
+        GreekProjector::<f64>::new(4, 0),
+        Err(Error::InvalidInput(_))
+    ));
+}
+
+#[test]
+fn derivatives_values_match_the_plain_values_bitwise() {
+    let mu = [1.0, 0.2, -1.0];
+    for (x, m) in SPHERES {
+        let mut mie = Mie::<f64>::new();
+        let plain = mie.compute(x, m, &MieOptions::new()).unwrap();
+        let g = plain.greek(None, Normalization::Normalized).unwrap();
+        let mut f: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; mu.len()]);
+        {
+            let [f11, f12, f33, f34] = &mut f;
+            plain
+                .phase_matrix(
+                    &mu,
+                    Normalization::Normalized,
+                    PhaseBuffers { f11, f12, f33, f34 },
+                )
+                .unwrap();
+        }
+        let both = solve(x, m, &mu, Normalization::Normalized);
+        assert_eq!(both.phase, f, "x = {x}: phase matrix");
+        assert_eq!(both.greek[0], g.alpha1, "x = {x}: alpha1");
+        assert_eq!(both.greek[5], g.beta2, "x = {x}: beta2");
+    }
+}
+
+/// The generic code runs in every scalar type.
+#[test]
+fn double_double_greek_runs() {
+    use scatrs::scalar::DoubleDouble as Dd;
+    let mut mie = Mie::<Dd>::new();
+    let s = mie
+        .compute(
+            Dd::from_f64(2.0),
+            Complex::new(Dd::from_f64(1.5), Dd::from_f64(0.0)),
+            &MieOptions::new(),
+        )
+        .unwrap();
+    let g = s.greek(None, Normalization::Normalized).unwrap();
+    assert_eq!(g.alpha1[0], Dd::from_f64(1.0));
+}
+
+/// The sign conventions, pinned by the Rayleigh limit (x → 0): F₁₁ = ¾(1 + μ²) and
+/// F₁₂ = −¾(1 − μ²) (Bohren & Huffman's F₁₂ = (|S₂|² − |S₁|²)/2 is negative at 90°), so with
+/// `wigner_d`'s d²₀₂ = (√6/4) sin²θ ≥ 0, β₁ at l = 2 is −√6/2.
+#[test]
+fn rayleigh_limit_pins_the_sign_conventions() {
+    let f = solve(1e-4, c(1.5, 0.0), &[0.0], Normalization::Normalized);
+    let expected: [[f64; 4]; 6] = [
+        [1.0, 0.0, 0.5, 0.0],
+        [0.0, 0.0, 3.0, 0.0],
+        [0.0; 4],
+        [0.0, 1.5, 0.0, 0.0],
+        [0.0, 0.0, -(6f64.sqrt()) / 2.0, 0.0],
+        [0.0; 4],
+    ];
+    for (k, row) in expected.iter().enumerate() {
+        for (l, &want) in row.iter().enumerate() {
+            let got = f.greek[k][l];
+            assert!(
+                (got - want).abs() <= 1e-7,
+                "Greek element {k} at l = {l}: {got}"
+            );
+        }
+    }
+    assert!((f.phase[0][0] - 0.75).abs() <= 1e-7 && (f.phase[1][0] + 0.75).abs() <= 1e-7);
+}
````

- [ ] **Step 2: Verify**

Run: `cargo test -p scatrs --test mie_greek`
Expected: 10 tests PASS (pure-scatterer identity and ½∫F₁₁ = 1; unnormalized = normalized·Q_sca; the expansions reproduce F at five cosines; orders above 2N are exactly 0; projector exact for polynomials; `Dual<f64, 3>` oracle for both normalizations; bad buffers and tables rejected without writing; plain and with-derivative values bitwise equal; `DoubleDouble` runs; Rayleigh signs α₁ = (1, 0, ½), α₂² = 3, α₄¹ = 3/2, β₁² = −√6/2).

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo doc -p scatrs --no-deps --document-private-items`
Expected: clean.

- [ ] **Step 3: Commit** — `feat(mie): sphere phase matrix and exact Greek coefficients`

---

### Task 3: Arb references, tolerances and the accuracy map

**Files:**
- Create: `reference/scatrs_reference/generators/mie_greek.py`, `reference/tests/test_mie_greek.py`, `scatrs/tests/common/check.rs`, `scatrs/tests/mie_greek_reference.rs`, `scatrs/tests/mie_greek_accuracy_map.rs`; generated: `reference/data/mie_greek.jsonl`, `docs/validation/data/greek-accuracy-map.json`
- Modify: `reference/scatrs_reference/generators/__init__.py`, `reference/data/manifest.json` (generated), `pyproject.toml`, `scatrs/src/mie/domain.rs`, `scatrs/tests/common/mod.rs`, `scatrs/tests/mie_reference.rs`, `tests/test_docs_consistency.py`, `docs/theory/mie.md`, `docs/validation/reference-suite.md`, `.github/workflows/scheduled.yml`, `CONTRIBUTING.md`

**Interfaces:**
- Consumes: Task 2's API and `common::greek`.
- Produces: `MieOutput::{PhaseMatrix, PhaseMatrixDerivatives, GreekCoefficients { order }, GreekDerivatives { order }}`; `ValidatedDomain::tolerance` priors κ, κ·max(1, |m|/|m − 1|), κ + ((L + 1)² + x²)/8 and κ·max(1, |m|/|m − 1|) + ((L + 1)² + x²)/8; constants 32, 128, 8, 16. `common::check::{Parts, Worst, error, records, refractive_index, size_parameter, DOUBLE_DOUBLE_SLACK}`.

- [ ] **Step 1: Apply the diff** (generated files excluded):

````diff
diff --git a/.github/workflows/scheduled.yml b/.github/workflows/scheduled.yml
index 6b20bde..79a2e08 100644
--- a/.github/workflows/scheduled.yml
+++ b/.github/workflows/scheduled.yml
@@ -83,8 +83,12 @@ jobs:
           persist-credentials: false
       - name: Install Rust
         run: rustup toolchain install stable --profile minimal --no-self-update
-      # The #[ignore] tests, all in tests/gauss_legendre.rs: Gauss–Legendre at n = 20000
-      # against Arb, and every rule up to 3000 points. Building only that test binary saves
-      # fat-LTO builds of the others. --nocapture shows the worst errors they measure.
+      # The #[ignore] tests: in tests/gauss_legendre.rs, the double-double Gauss–Legendre rule
+      # at n = 20000 against Arb and every rule up to 3000 points; in
+      # tests/mie_greek_accuracy_map.rs, the phase matrix and Greek coefficients of spheres up
+      # to x = 2e4. Building only those test binaries saves fat-LTO builds of the others.
+      # --nocapture shows the worst errors they measure.
       - name: Run the ignored tests
-        run: cargo test -p scatrs --release --locked --test gauss_legendre -- --ignored --nocapture
+        run: >-
+          cargo test -p scatrs --release --locked --test gauss_legendre
+          --test mie_greek_accuracy_map -- --ignored --nocapture
diff --git a/CONTRIBUTING.md b/CONTRIBUTING.md
index c0a17a5..9444671 100644
--- a/CONTRIBUTING.md
+++ b/CONTRIBUTING.md
@@ -55,12 +55,20 @@ SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/mie-accuracy-map.json \
     cargo test -p scatrs --release --test mie_accuracy_map -- --nocapture
 ```
 
+`scatrs/tests/mie_greek_accuracy_map.rs` does the same for the phase matrix and the Greek
+coefficients, in about 15 s with all cores; regenerate its map the same way:
+
+```bash
+SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/greek-accuracy-map.json \
+    cargo test -p scatrs --release --test mie_greek_accuracy_map -- --nocapture
+```
+
 Wall-clock benchmarks of the Mie kernel: `cargo bench -p scatrs --bench mie`.
 
 The slow `#[ignore]` tests take minutes, so CI runs them weekly rather than on every change:
 
 ```bash
-cargo test -p scatrs --release --test gauss_legendre -- --ignored
+cargo test -p scatrs --release --test gauss_legendre --test mie_greek_accuracy_map -- --ignored
 ```
 
 ## Continuous integration
diff --git a/docs/theory/mie.md b/docs/theory/mie.md
index 23bb88e..5fec828 100644
--- a/docs/theory/mie.md
+++ b/docs/theory/mie.md
@@ -233,7 +233,8 @@ the amplitudes, with $S_\text{rms} = x\sqrt{Q_\text{sca}}/2$). The two condition
 The constants C are measured, with a margin of at least 4 over the worst case found at every
 Arb reference point and on a dense double-double grid: 16 for the coefficients, efficiencies
 and $g$; 128 for $S_1$, $S_2$; 64 for their derivatives; 256 for the efficiency Jacobian; 512
-for the coefficient derivatives. See {doc}`/validation/accuracy` for the map, and for why
+for the coefficient derivatives; 32 for the phase matrix; 128 for its derivatives; 8 for the
+Greek coefficients; 16 for their derivatives. See {doc}`/validation/accuracy` for the map, and for why
 neither term alone is enough.
 
 ## Reference values
diff --git a/docs/validation/reference-suite.md b/docs/validation/reference-suite.md
index 240996e..f55a531 100644
--- a/docs/validation/reference-suite.md
+++ b/docs/validation/reference-suite.md
@@ -80,6 +80,7 @@ coefficient records are all certified. {doc}`/theory/mie` derives the bound.
 | `gauss_legendre.jsonl` | 17 | see below |
 | `mie.jsonl` | 612 | see below |
 | `mie_derivatives.jsonl` | 160 | see below |
+| `mie_greek.jsonl` | 36 | see below |
 | `wigner_d.jsonl` | 212 | see below |
 
 The edge cases sit at the edges of each function's domain and of its algorithm's branches:
@@ -117,7 +118,16 @@ record holds the efficiencies, $g$, and $S_1$, $S_2$ at 9 angles from 0° to 180
 **`mie_derivatives.jsonl`** holds the same for 80 spheres, with the derivatives of each
 output instead of the values.
 
-The committed data take about 12.8 MB, within the 20 MB budget, which a test enforces.
+**`mie_greek.jsonl`** holds two records per sphere, for 18 spheres from $x = 0.1$ to 200, of
+the $N$-term series with the default `f64` truncation $N$. A `,phase` record holds the
+normalized phase matrix $F_{11}$, $F_{12}$, $F_{33}$, $F_{34}$ at 7 angles from 0° to 180°; a
+`,greek` record holds $\alpha_1, \ldots, \alpha_4, \beta_1, \beta_2$ up to all orders $2N$
+(13 spheres) or a lower order $L$ (5 spheres), normalized so that $\alpha_1^0 = 1$. The
+coefficients come from Gauss–Legendre quadrature on Arb's own Legendre roots, exact for these
+polynomial integrands, and the Wigner d-functions from their three-term recurrence in ball
+arithmetic, so every value is certified.
+
+The committed data take about 13.3 MB, within the 20 MB budget, which a test enforces.
 
 ## How the tests use it
 
diff --git a/pyproject.toml b/pyproject.toml
index 0c2777c..03cc15f 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -79,7 +79,9 @@ allowed-confusables = ["−", "–", "ρ", "×", "′"]
 
 [tool.ruff.lint.per-file-ignores]
 # l is the degree of d^l_{mn}, the notation of the papers the generator follows.
+"reference/scatrs_reference/generators/mie_greek.py" = ["E741"]
 "reference/scatrs_reference/generators/wigner_d.py" = ["E741"]
 "reference/tests/test_generators.py" = ["E741"]
+"reference/tests/test_mie_greek.py" = ["E741"]
 # The edge-case tables concatenate sublists grouped by what they probe.
 "reference/scatrs_reference/generators/dd_elementary.py" = ["RUF005"]
diff --git a/reference/scatrs_reference/generators/__init__.py b/reference/scatrs_reference/generators/__init__.py
index 286f58c..c50e92a 100644
--- a/reference/scatrs_reference/generators/__init__.py
+++ b/reference/scatrs_reference/generators/__init__.py
@@ -11,6 +11,7 @@ from scatrs_reference.generators import (
     dd_trig,
     gauss_legendre,
     mie,
+    mie_greek,
     wigner_d,
 )
 
@@ -26,5 +27,6 @@ GENERATORS: dict[str, Generator] = {
     "gauss_legendre": gauss_legendre.generate,
     "mie": mie.generate,
     "mie_derivatives": mie.generate_derivatives,
+    "mie_greek": mie_greek.generate,
     "wigner_d": wigner_d.generate,
 }
diff --git a/reference/scatrs_reference/generators/mie_greek.py b/reference/scatrs_reference/generators/mie_greek.py
new file mode 100644
index 0000000..ce59770
--- /dev/null
+++ b/reference/scatrs_reference/generators/mie_greek.py
@@ -0,0 +1,212 @@
+"""Phase matrices and Greek coefficients of spheres, from the Arb Mie coefficients.
+
+For each case (x, m, N, L) the generator takes a_n, b_n for n = 1..N from
+`scatrs_reference.generators.mie` (finite computations in ball arithmetic), and computes:
+
+* the phase-matrix elements F₁₁ = (|S₁|² + |S₂|²)/2, F₁₂ = (|S₂|² − |S₁|²)/2, F₃₃ = Re(S₂S₁*),
+  F₃₄ = Im(S₂S₁*) of the N-term series at the cosines ``COSINES``, normalized by 4/(x²Q_sca)
+  with Q_sca the N-term sum;
+* the Greek coefficients α₁…α₄, β₁, β₂ for l = 0..L of the N-term series, by Gauss–Legendre
+  quadrature on Arb's own Legendre roots (``arb.legendre_p_root``) with n_q = N + ⌈(L + 1)/2⌉
+  nodes, which is exact for these polynomial integrands, normalized so that α₁ at l = 0 is 1.
+  The Wigner d-functions d⁰⁰, d⁰², d²², d²⁻² come from the three-term recurrence in l in ball
+  arithmetic, seeded with the explicit sum (`wigner_d.d_ref`).
+
+Everything is a finite computation in ball arithmetic, so every record is certified. The
+precision starts at ``PRECISION_BITS`` and doubles until every stored value's radius is below
+2^-``REL_BITS`` of its scale (the largest |F₁₁| for the phase matrix, the largest |α₁| for the
+Greek coefficients).
+"""
+
+from __future__ import annotations
+
+import math
+
+from flint import acb, arb, ctx
+
+from scatrs_reference.generators import mie
+from scatrs_reference.records import record
+
+GENERATOR = "mie_greek"
+PRECISION_BITS = 320
+REL_BITS = 200
+MAX_PRECISION_BITS = 2560
+
+#: (x, Re m, Im m, L or None for all orders 2N).
+CASES = (
+    (0.1, 1.5, 0.0, None),
+    (1.0, 1.33, 0.0, None),
+    (1.0, 1.5, 0.1, None),
+    (3.0, 0.75, 0.0, None),
+    (5.0, 1.5, 0.01, None),
+    (5.0, 2.5, 1.0, None),
+    (10.0, 1.33, 1e-8, None),
+    (10.0, 0.2, 3.0, None),
+    (25.0, 1.5, 0.001, None),
+    (25.0, 1.5, 0.001, 16),
+    (50.0, 1.33, 0.0, 40),
+    (50.0, 10.0, 10.0, 30),
+    (0.5, 0.05, 4.0, None),
+    (2.0, 1.0001, 0.0, None),
+    (2.0, 1.75, 0.44, None),
+    (20.0, 0.75, 0.0, None),
+    (100.0, 1.5, 0.01, 50),
+    (200.0, 1.33, 0.0, 20),
+)
+
+#: Cosines of the phase-matrix records.
+COSINES = (1.0, 0.9, 0.5, 0.0, -0.5, -0.9, -1.0)
+
+
+def default_n(x: float) -> int:
+    """The f64 default truncation of scatrs, ⌈x + 6x^(1/3) + 3⌉ (exact for these x)."""
+    return math.ceil(x + 6.0 * x ** (1.0 / 3.0) + 3.0)
+
+
+def pi_tau(mu: arb, n: int) -> tuple[list[arb], list[arb]]:
+    """π_k(μ), τ_k(μ) for k = 1..n by the upward recurrences (Bohren & Huffman Eq. 4.47)."""
+    pis, taus = [], []
+    pi_prev, pi = arb(0), arb(1)
+    for k in range(1, n + 1):
+        pis.append(pi)
+        taus.append(k * mu * pi - (k + 1) * pi_prev)
+        pi_prev, pi = pi, ((2 * k + 1) * mu * pi - (k + 1) * pi_prev) / k
+    return pis, taus
+
+
+def elements(s: mie.Solution, mu: arb) -> tuple[arb, arb, arb, arb]:
+    """(|S₁|² + |S₂|²)/2, (|S₂|² − |S₁|²)/2, Re(S₂S₁*), Im(S₂S₁*) of the n_ref-term series."""
+    pis, taus = pi_tau(mu, s.n_ref)
+    s1, s2 = acb(0), acb(0)
+    for k in range(1, s.n_ref + 1):
+        c = arb(2 * k + 1) / (k * (k + 1))
+        a, b = s.a[k - 1], s.b[k - 1]
+        s1 += c * (a * pis[k - 1] + b * taus[k - 1])
+        s2 += c * (a * taus[k - 1] + b * pis[k - 1])
+    n1, n2 = abs(s1) ** 2, abs(s2) ** 2
+    p = s2 * s1.conjugate()
+    return (n1 + n2) / 2, (n2 - n1) / 2, p.real, p.imag
+
+
+def wigner_column(m: int, n: int, x: arb, l_max: int) -> list[arb]:
+    """d^l_{mn}(x) for l = 0..l_max: the explicit sum at l₀ = max(|m|, |n|), then the
+    three-term recurrence (Mishchenko, Travis & Lacis 2002, Eq. B.22)."""
+    l0 = max(abs(m), abs(n))
+    out = [arb(0)] * (l_max + 1)
+    if l_max < l0:
+        return out
+    prev, cur = arb(0), d_ref_ball(l0, m, n, x)
+    out[l0] = cur
+    for l in range(l0, l_max):
+        a = (2 * l + 1) * (l * (l + 1) * x - m * n)
+        b = (l + 1) * (arb(l * l - m * m) * (l * l - n * n)).sqrt() if l > 0 else arb(0)
+        c = l * (arb((l + 1) ** 2 - m * m) * ((l + 1) ** 2 - n * n)).sqrt()
+        if l == 0:
+            # d¹ from d⁰ when l₀ = 0 (only m = n = 0): d¹₀₀ = x d⁰₀₀.
+            nxt = x * cur
+        else:
+            nxt = (a * cur - b * prev) / c
+        prev, cur = cur, nxt
+        out[l + 1] = cur
+    return out
+
+
+def d_ref_ball(l: int, m: int, n: int, x: arb) -> arb:
+    """d^l_{mn}(x) at its lowest order l = max(|m|, |n|), for the four (m, n) used here."""
+    c = ((1 + x) / 2).sqrt()
+    s = ((1 - x) / 2).sqrt()
+    if (m, n) == (0, 0):
+        return arb(1)
+    if (m, n) == (2, 2):
+        return c**4
+    if (m, n) == (2, -2):
+        return s**4
+    if (m, n) == (0, 2):
+        return arb(6).sqrt() / 4 * (1 - x * x)
+    raise ValueError((l, m, n))
+
+
+def greek(s: mie.Solution, order: int) -> dict[str, list[arb]]:
+    """α₁…α₄, β₁, β₂ for l = 0..order, normalized so that α₁ at l = 0 is 1."""
+    n_q = s.n_ref + (order + 2) // 2
+    sums = {k: [arb(0)] * (order + 1) for k in ("a1", "a4", "plus", "minus", "b1", "b2")}
+    for k in range(n_q):
+        mu, w = arb.legendre_p_root(n_q, k, weight=True)
+        f11, f12, f33, f34 = (w * v for v in elements(s, mu))
+        d00 = wigner_column(0, 0, mu, order)
+        d22 = wigner_column(2, 2, mu, order)
+        d2m2 = wigner_column(2, -2, mu, order)
+        d02 = wigner_column(0, 2, mu, order)
+        for l in range(order + 1):
+            sums["a1"][l] += f11 * d00[l]
+            sums["a4"][l] += f33 * d00[l]
+            sums["plus"][l] += (f11 + f33) * d22[l]
+            sums["minus"][l] += (f11 - f33) * d2m2[l]
+            sums["b1"][l] += f12 * d02[l]
+            sums["b2"][l] += f34 * d02[l]
+    r0 = sums["a1"][0]
+    out: dict[str, list[arb]] = {
+        k: [] for k in ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
+    }
+    for l in range(order + 1):
+        # The (2l + 1)/2 factors cancel in the normalization by α₁⁰ = ½·r₀, except for l.
+        f = arb(2 * l + 1) / r0
+        out["alpha1"].append(f * sums["a1"][l])
+        out["alpha4"].append(f * sums["a4"][l])
+        out["alpha2"].append(f * (sums["plus"][l] + sums["minus"][l]) / 2)
+        out["alpha3"].append(f * (sums["plus"][l] - sums["minus"][l]) / 2)
+        out["beta1"].append(f * sums["b1"][l])
+        out["beta2"].append(f * sums["b2"][l])
+    return out
+
+
+def phase_matrix(s: mie.Solution) -> dict[str, list[arb]]:
+    """F₁₁, F₁₂, F₃₃, F₃₄ at ``COSINES``, normalized by 4/(x²Q_sca)."""
+    q_sca = arb(0)
+    for k in range(1, s.n_ref + 1):
+        q_sca += (2 * k + 1) * (abs(s.a[k - 1]) ** 2 + abs(s.b[k - 1]) ** 2)
+    q_sca *= 2 / (s.x * s.x)
+    c = 4 / (s.x * s.x * q_sca)
+    out: dict[str, list[arb]] = {"f11": [], "f12": [], "f33": [], "f34": []}
+    for mu in COSINES:
+        for name, v in zip(out, elements(s, arb(mu)), strict=True):
+            out[name].append(c * v)
+    return out
+
+
+def _tight(outputs: dict[str, list[arb]], scale: arb) -> bool:
+    bound = scale * arb(2) ** -REL_BITS
+    return all(v.is_finite() and v.rad() <= bound for vs in outputs.values() for v in vs)
+
+
+def case_records(x: float, m_re: float, m_im: float, order: int | None) -> list[dict]:
+    n = default_n(x)
+    l_max = 2 * n if order is None else order
+    prec = PRECISION_BITS
+    while True:
+        with ctx.workprec(prec):
+            s = mie.solve(x, m_re, m_im, n)
+            f = phase_matrix(s)
+            g = greek(s, l_max)
+            f_scale = max(abs(v) for v in f["f11"])
+            g_scale = max(abs(v) for v in g["alpha1"])
+            if _tight(f, f_scale) and _tight(g, g_scale):
+                break
+        prec *= 2
+        if prec > MAX_PRECISION_BITS:
+            raise ArithmeticError(
+                f"x={x}, m=({m_re}, {m_im}) needs more than {MAX_PRECISION_BITS} bits"
+            )
+    case = f"x={x!r},m=({m_re!r},{m_im!r}),L={l_max}"
+    inputs = {"x": x, "m_re": m_re, "m_im": m_im, "n_terms": n, "order": l_max}
+    return [
+        record(GENERATOR, case + ",phase", {**inputs, "mu": list(COSINES)}, f),
+        record(GENERATOR, case + ",greek", inputs, g),
+    ]
+
+
+def generate() -> tuple[int, list[dict]]:
+    records: list[dict] = []
+    for case in CASES:
+        records.extend(case_records(*case))
+    return PRECISION_BITS, records
diff --git a/reference/tests/test_mie_greek.py b/reference/tests/test_mie_greek.py
new file mode 100644
index 0000000..f36ad00
--- /dev/null
+++ b/reference/tests/test_mie_greek.py
@@ -0,0 +1,39 @@
+import json
+from pathlib import Path
+
+from flint import arb, ctx
+
+from scatrs_reference.generators import mie_greek, wigner_d
+
+DATA = Path(__file__).resolve().parent.parent / "data"
+
+
+def test_wigner_columns_match_the_explicit_sum():
+    with ctx.workprec(200):
+        for m, n in ((0, 0), (2, 2), (2, -2), (0, 2)):
+            for x in (-0.95, -0.3, 0.0, 0.4, 0.999):
+                col = mie_greek.wigner_column(m, n, arb(x), 30)
+                for l in range(31):
+                    assert abs(col[l] - wigner_d.d_ref(l, m, n, x)) < arb(2) ** -150, (m, n, x, l)
+
+
+def _value(out, i):
+    return out["hi"][i] + out["lo"][i] + out["lo2"][i]
+
+
+def test_records_satisfy_the_identities():
+    records = [json.loads(line) for line in (DATA / "mie_greek.jsonl").read_text().splitlines()]
+    greek = {r["case"].removesuffix(",greek"): r for r in records if r["case"].endswith(",greek")}
+    phase = {r["case"].removesuffix(",phase"): r for r in records if r["case"].endswith(",phase")}
+    assert greek.keys() == phase.keys() and len(greek) == len(mie_greek.CASES)
+    for case, g in greek.items():
+        a1 = g["outputs"]["alpha1"]
+        assert _value(a1, 0) == 1.0, case
+        assert g["label"] == phase[case]["label"] == "certified"
+        assert len(a1["hi"]) == g["inputs"]["order"] + 1
+        if g["inputs"]["order"] == 2 * g["inputs"]["n_terms"]:
+            # All orders: F11(1) is the sum of alpha1, since d00_l(1) = 1.
+            f11 = phase[case]["outputs"]["f11"]
+            assert phase[case]["inputs"]["mu"][0] == 1.0
+            total = sum(_value(a1, i) for i in range(len(a1["hi"])))
+            assert abs(total - _value(f11, 0)) <= 1e-12 * _value(f11, 0), case
diff --git a/scatrs/src/mie/domain.rs b/scatrs/src/mie/domain.rs
index 5567b89..97f0f1d 100644
--- a/scatrs/src/mie/domain.rs
+++ b/scatrs/src/mie/domain.rs
@@ -38,6 +38,26 @@ pub enum MieOutput {
     /// ∂S/∂x and dS/dm: max over μ of |error| / max over μ of
     /// max(|∂S/∂p|, S_rms·(1 + |m|x)/|p|).
     AmplitudeDerivatives,
+    /// F₁₁, F₁₂, F₃₃, F₃₄ at each cosine: |error| / max(F₁₁, F̄), where F̄ = ½∫F₁₁ dμ is 1
+    /// normalized and Q_sca unnormalized, so the deep minima of the angular pattern are measured
+    /// against its mean.
+    PhaseMatrix,
+    /// Each ∂F/∂p of the phase matrix: max over μ of |error| / max over μ of
+    /// max(|∂F/∂p|, max(F₁₁, F̄)·(1 + |m|x)/|p|), with |p| as for
+    /// [`EfficiencyJacobian`](Self::EfficiencyJacobian).
+    PhaseMatrixDerivatives,
+    /// Each of α₁, α₂, α₃, α₄, β₁, β₂ up to `order` L: max over l of |error| / max over l of
+    /// |α₁|.
+    GreekCoefficients {
+        /// The highest order L.
+        order: usize,
+    },
+    /// Each derivative of α₁, …, β₂ with respect to p: max over l of |error| / max over l of
+    /// max(|∂c/∂p|, A·(1 + |m|x)/|p|), with A = max over l of |α₁|.
+    GreekDerivatives {
+        /// The highest order L.
+        order: usize,
+    },
 }
 
 /// The region of inputs whose accuracy has been validated, and the declared tolerances there.
@@ -67,7 +87,14 @@ pub enum MieOutput {
 ///   the efficiency Jacobian (the derivatives of g lose a second factor |m|/|m − 1| near
 ///   m = 1), and 1 + max(1, |m|)·x for the derivatives of the coefficients and amplitudes,
 ///   which lose nothing near m = 1: m − 1 enters the x-derivatives as an exact factor of
-///   m² − 1, and the m-derivatives have no factor that cancels.
+///   m² − 1, and the m-derivatives have no factor that cancels. The phase matrix takes κ and
+///   its derivatives κ·max(1, |m|/|m − 1|), like the efficiencies and their Jacobian (the
+///   normalized phase matrix divides by Q_sca). The Greek coefficients up to order L add
+///   ((L + 1)² + x²)/8 to those: the quadrature nodes are rounded to `f64`, so the
+///   coefficients are those of a rule with nodes moved by up to half an ulp, and next to μ = 1
+///   such a move changes d_L by about L²·2⁻⁵⁴ and the forward peak of the phase matrix, whose
+///   slope in μ grows like x⁴ while the weights there shrink like 1/x², by about x²·2⁻⁵⁴ of
+///   its size.
 /// * C is the output's constant ([`constant`](Self::constant)).
 ///
 /// Neither term alone is a bound: the accuracy map in the documentation shows outputs that
@@ -120,12 +147,18 @@ impl ValidatedDomain {
     /// the measure of [`MieOutput`], where `condition` is the output's condition number (see
     /// the type documentation for it and for κ′).
     pub fn tolerance(&self, output: MieOutput, x: f64, m: Complex<f64>, condition: f64) -> f64 {
+        let kappa = Self::condition_factor(x, m);
+        let quadrature = |order: usize| ((order as f64 + 1.0).powi(2) + x * x) / 8.0;
         let prior = match output {
             MieOutput::CoefficientDerivatives | MieOutput::AmplitudeDerivatives => {
                 1.0 + m.norm().max(1.0) * x
             }
-            MieOutput::EfficiencyJacobian => Self::condition_factor(x, m) * near_one(m),
-            _ => Self::condition_factor(x, m),
+            MieOutput::EfficiencyJacobian | MieOutput::PhaseMatrixDerivatives => {
+                kappa * near_one(m)
+            }
+            MieOutput::GreekCoefficients { order } => kappa + quadrature(order),
+            MieOutput::GreekDerivatives { order } => kappa * near_one(m) + quadrature(order),
+            _ => kappa,
         };
         Self::constant(output) * (prior + condition) * f64::EPSILON
     }
@@ -139,6 +172,10 @@ impl ValidatedDomain {
     /// | derivatives of S₁, S₂ | 64 |
     /// | efficiency Jacobian | 256 |
     /// | coefficient derivatives | 512 |
+    /// | phase matrix | 32 |
+    /// | derivatives of the phase matrix | 128 |
+    /// | Greek coefficients | 8 |
+    /// | derivatives of the Greek coefficients | 16 |
     pub fn constant(output: MieOutput) -> f64 {
         match output {
             MieOutput::Coefficients
@@ -151,6 +188,10 @@ impl ValidatedDomain {
             MieOutput::AmplitudeDerivatives => 64.0,
             MieOutput::EfficiencyJacobian => 256.0,
             MieOutput::CoefficientDerivatives => 512.0,
+            MieOutput::PhaseMatrix => 32.0,
+            MieOutput::PhaseMatrixDerivatives => 128.0,
+            MieOutput::GreekCoefficients { .. } => 8.0,
+            MieOutput::GreekDerivatives { .. } => 16.0,
         }
     }
 }
diff --git a/scatrs/tests/common/check.rs b/scatrs/tests/common/check.rs
new file mode 100644
index 0000000..8ce7cfa
--- /dev/null
+++ b/scatrs/tests/common/check.rs
@@ -0,0 +1,138 @@
+//! Checking computed values against Arb reference values within the declared tolerances
+//! (`scatrs::mie::ValidatedDomain::tolerance`), shared by the Mie reference tests.
+
+use num_complex::Complex;
+use scatrs::mie::{MieOutput, VALIDATED_DOMAIN, ValidatedDomain};
+use scatrs::scalar::DoubleDouble as Dd;
+
+use super::mie::Measure;
+use super::{Record, load};
+
+pub fn size_parameter(r: &Record) -> f64 {
+    r.input_f64("x")
+}
+
+pub fn refractive_index(r: &Record) -> Complex<f64> {
+    Complex::new(r.input_f64("m_re"), r.input_f64("m_im"))
+}
+
+/// A computed value as the unevaluated sum hi + lo.
+pub trait Parts: Copy {
+    fn parts(self) -> (f64, f64);
+}
+
+impl Parts for f64 {
+    fn parts(self) -> (f64, f64) {
+        (self, 0.0)
+    }
+}
+
+impl Parts for Dd {
+    fn parts(self) -> (f64, f64) {
+        (self.hi(), self.lo())
+    }
+}
+
+/// |got − v| for the reference value i of `name`.
+pub fn error<T: Parts>(got: T, r: &Record, name: &str, i: usize) -> f64 {
+    let (hi, lo) = got.parts();
+    r.output(name).dd_error(i, hi, lo)
+}
+
+/// `DoubleDouble` stands in for the exact values in the accuracy map, so it must be accurate
+/// far beyond `f64`, not within the same constants: it may reach 16 times them, in units of
+/// (κ′ + cond)·2⁻¹⁰⁴ (measured: up to 1.5 times, in the amplitude derivatives at x = 2·10⁴).
+/// Its error then moves a C of the map by at most 2⁻⁴⁸ of the declared constant.
+pub const DOUBLE_DOUBLE_SLACK: f64 = 16.0;
+
+/// The largest measured C of one output, and where.
+pub struct Worst {
+    pub what: String,
+    /// C is measured in units of (κ′ + cond)·2⁻⁵²·`unit`: 1 for `f64`, 2⁻⁵² for `DoubleDouble`.
+    pub unit: f64,
+    /// The largest C allowed, as a multiple of the declared constant.
+    pub slack: f64,
+    pub c: f64,
+    pub case: String,
+}
+
+impl Worst {
+    pub fn new(what: &str) -> Self {
+        Self {
+            what: what.to_owned(),
+            unit: 1.0,
+            slack: 1.0,
+            c: 0.0,
+            case: String::new(),
+        }
+    }
+
+    pub fn double_double(what: &str) -> Self {
+        Self {
+            what: format!("DoubleDouble {what}"),
+            unit: f64::EPSILON,
+            slack: DOUBLE_DOUBLE_SLACK,
+            c: 0.0,
+            case: String::new(),
+        }
+    }
+
+    /// Checks the error `err` of `output` at the case of `r` against the declared tolerance in
+    /// the output's measure, times `unit`·`slack`, and records its C.
+    pub fn check(
+        &mut self,
+        output: MieOutput,
+        err: f64,
+        measure: Measure,
+        r: &Record,
+        detail: &str,
+    ) {
+        let (x, m) = (size_parameter(r), refractive_index(r));
+        assert!(
+            VALIDATED_DOMAIN.contains(x, m),
+            "{} is outside the domain",
+            r.case
+        );
+        let tol =
+            VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition) * self.unit * self.slack;
+        assert!(
+            measure.scale > 0.0 && tol.is_finite(),
+            "{} {} {detail}: degenerate measure {measure:?}, tolerance {tol:e}",
+            self.what,
+            r.case
+        );
+        let rel = err / measure.scale;
+        let c = rel / tol * self.slack * ValidatedDomain::constant(output);
+        if c > self.c {
+            self.c = c;
+            self.case = format!("{} {detail}", r.case);
+        }
+        assert!(
+            rel <= tol,
+            "{} {} {detail}: error {rel:.3e} exceeds {tol:.3e} (C = {c:.2}, condition {:.3e})",
+            self.what,
+            r.case,
+            measure.condition
+        );
+    }
+}
+
+impl Drop for Worst {
+    fn drop(&mut self) {
+        if !std::thread::panicking() {
+            eprintln!(
+                "{:<40} worst C = {:8.3} at {}",
+                self.what, self.c, self.case
+            );
+        }
+    }
+}
+
+pub fn records(file: &str, kind: &str) -> Vec<Record> {
+    let records: Vec<Record> = load(file)
+        .into_iter()
+        .filter(|r| r.case.ends_with(kind))
+        .collect();
+    assert!(!records.is_empty(), "no {kind} records in {file}");
+    records
+}
diff --git a/scatrs/tests/common/mod.rs b/scatrs/tests/common/mod.rs
index 083bf62..e8d70fb 100644
--- a/scatrs/tests/common/mod.rs
+++ b/scatrs/tests/common/mod.rs
@@ -1,6 +1,7 @@
 //! Loader for the committed Arb reference data in `reference/data/`.
 #![allow(dead_code)]
 
+pub mod check;
 pub mod greek;
 pub mod mie;
 
diff --git a/scatrs/tests/mie_greek_accuracy_map.rs b/scatrs/tests/mie_greek_accuracy_map.rs
new file mode 100644
index 0000000..43610f5
--- /dev/null
+++ b/scatrs/tests/mie_greek_accuracy_map.rs
@@ -0,0 +1,359 @@
+//! The accuracy map of the phase matrix and the Greek coefficients: `f64` against
+//! `DoubleDouble` on a grid over the validated domain, every output within its declared
+//! tolerance.
+//!
+//! The grid: x = d·10^e for the mantissas `MANTISSAS` and e = −6, …, 3, up to 10³, 46 values,
+//! against the refractive indices `INDICES` of the Mie accuracy map
+//! (`tests/mie_accuracy_map.rs`), with the phase matrix at `COSINES`, normalized. The Greek
+//! coefficients go up to all orders (2N) for x ≤ 160 and to order 100 above. The double-double
+//! Greek coefficients cost O(x²) (amplitudes at about x nodes, and all orders at as many), and
+//! the derivative measures take seven of them per point, so the grid stops at 10³, where the
+//! map takes about 2 CPU-minutes. The ignored test `large_spheres_are_within_the_tolerances`
+//! (about a minute; CI runs it weekly) checks the values at x = 2·10³ (all orders), 10⁴ and
+//! 2·10⁴ (order 64). `DoubleDouble` is within 16 times the declared constants, in units of
+//! (κ′ + cond)·2⁻¹⁰⁴, of Arb (tests/mie_greek_reference.rs), so it stands in for the exact
+//! values.
+//!
+//! Each error is reported as C = error/((κ′ + cond)·2⁻⁵²); the test prints the worst C of each
+//! output. With `SCATRS_ACCURACY_MAP=path` it also writes every C as JSON, which the
+//! documentation plots (docs/validation/accuracy.md):
+//!
+//! ```text
+//! SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/greek-accuracy-map.json \
+//!     cargo test -p scatrs --release --test mie_greek_accuracy_map -- --nocapture
+//! ```
+
+mod common;
+
+use std::fmt::Write as _;
+use std::sync::atomic::{AtomicUsize, Ordering};
+
+use common::greek::{
+    greek_derivative_measures, greek_reference, greek_value_measures, solve_greek,
+};
+use common::mie::Measure;
+use num_complex::Complex;
+use scatrs::mie::{MieOutput, Normalization, VALIDATED_DOMAIN, ValidatedDomain, default_n_max};
+use scatrs::scalar::DoubleDouble as Dd;
+
+/// Five size parameters per decade.
+const MANTISSAS: [&str; 5] = ["1", "1.6", "2.5", "4", "6.4"];
+
+/// The largest x of the grid.
+const X_MAX: f64 = 1e3;
+
+/// Above this x the Greek coefficients go up to `ORDER_ABOVE` only.
+const ALL_ORDERS_UP_TO: f64 = 160.0;
+const ORDER_ABOVE: usize = 100;
+
+/// (Re m, Im m), as in the Mie accuracy map.
+const INDICES: [(f64, f64); 28] = [
+    (1.0001, 0.0),
+    (1.001, 1e-6),
+    (1.01, 0.0),
+    (1.1, 1e-3),
+    (1.33, 0.0),
+    (1.33, 1e-8),
+    (1.33, 1e-4),
+    (1.5, 0.0),
+    (1.5, 1e-2),
+    (1.5, 0.1),
+    (1.5, 1.0),
+    (1.5, 10.0),
+    (1.75, 0.44),
+    (2.0, 0.0),
+    (2.5, 1.0),
+    (3.0, 1e-3),
+    (4.0, 0.01),
+    (5.0, 5.0),
+    (8.0, 0.1),
+    (10.0, 0.0),
+    (10.0, 10.0),
+    (0.5, 2.0),
+    (0.2, 3.0),
+    (0.05, 4.0),
+    (0.75, 0.0),
+    (0.5, 1e-4),
+    (0.3, 0.6),
+    (0.1, 0.0),
+];
+
+/// Cosines through the forward peak, the sides and the glory: exact doubles.
+const COSINES: [f64; 13] = [
+    1.0,
+    0.999_999_046_325_683_6, // 1 − 2⁻²⁰
+    0.999_755_859_375,       // 1 − 2⁻¹²
+    0.996_093_75,            // 1 − 2⁻⁸
+    0.9375,
+    0.75,
+    0.5,
+    0.0,
+    -0.5,
+    -0.75,
+    -0.9375,
+    -0.996_093_75,
+    -1.0,
+];
+
+const NORMALIZATION: Normalization = Normalization::Normalized;
+
+/// The columns of the map.
+const OUTPUTS: [&str; 4] = [
+    "phase_matrix",
+    "phase_matrix_derivatives",
+    "greek",
+    "greek_derivatives",
+];
+
+fn size_parameters() -> Vec<f64> {
+    (-6..=3)
+        .flat_map(|e| MANTISSAS.iter().map(move |d| format!("{d}e{e}")))
+        .map(|s| s.parse::<f64>().expect("a decimal literal"))
+        .filter(|&x| x <= X_MAX)
+        .collect()
+}
+
+fn order_at(x: f64) -> Option<usize> {
+    (x > ALL_ORDERS_UP_TO).then_some(ORDER_ABOVE)
+}
+
+fn diff(a: f64, b: Dd) -> f64 {
+    (Dd::from_f64(a) - b).hi().abs()
+}
+
+/// max over the array of |f − d|.
+fn array_error(f: &[f64], d: &[Dd]) -> f64 {
+    f.iter()
+        .zip(d)
+        .map(|(&a, &b)| diff(a, b))
+        .fold(0.0, f64::max)
+}
+
+/// C = error/((κ′ + cond)·2⁻⁵²) of `output` at (x, m); NaN counts as infinite.
+fn constant_of(output: MieOutput, x: f64, m: Complex<f64>, err: f64, measure: Measure) -> f64 {
+    let tol = VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition);
+    assert!(
+        measure.scale > 0.0 && tol.is_finite(),
+        "{output:?} at x = {x:e}, m = {m}: degenerate measure {measure:?}, tolerance {tol:e}"
+    );
+    let c = err / measure.scale / tol * ValidatedDomain::constant(output);
+    if c.is_nan() { f64::INFINITY } else { c }
+}
+
+/// C of every output at (x, m), ordered as `OUTPUTS`.
+fn point(x: f64, m: Complex<f64>) -> [f64; 4] {
+    let order = order_at(x);
+    let n = default_n_max(x);
+    let f = solve_greek(x, m, &COSINES, None, order, NORMALIZATION);
+    let d = greek_reference(x, m, &COSINES, n, order, NORMALIZATION);
+    let l = f.greek[0].len() - 1;
+    let v = greek_value_measures(x, m, &d);
+    let dm = greek_derivative_measures(x, m, &COSINES, n, l, NORMALIZATION, &d);
+    let mut out = [0.0f64; 4];
+    for j in 0..COSINES.len() {
+        for k in 0..4 {
+            let err = diff(f.phase[k][j], d.phase[k][j]);
+            let c = constant_of(MieOutput::PhaseMatrix, x, m, err, v.phase[j][k]);
+            out[0] = out[0].max(c);
+        }
+    }
+    for p in 0..3 {
+        for k in 0..4 {
+            let err = array_error(&f.phase_d[p][k], &d.phase_d[p][k]);
+            let c = constant_of(MieOutput::PhaseMatrixDerivatives, x, m, err, dm.phase[p][k]);
+            out[1] = out[1].max(c);
+        }
+    }
+    for k in 0..6 {
+        let err = array_error(&f.greek[k], &d.greek[k]);
+        let c = constant_of(
+            MieOutput::GreekCoefficients { order: l },
+            x,
+            m,
+            err,
+            v.greek[k],
+        );
+        out[2] = out[2].max(c);
+    }
+    for p in 0..3 {
+        for k in 0..6 {
+            let err = array_error(&f.greek_d[p][k], &d.greek_d[p][k]);
+            let output = MieOutput::GreekDerivatives { order: l };
+            out[3] = out[3].max(constant_of(output, x, m, err, dm.greek[p][k]));
+        }
+    }
+    out
+}
+
+/// The constant declared for column `o` of `OUTPUTS` (the order does not change it).
+fn declared(o: usize) -> f64 {
+    ValidatedDomain::constant(
+        [
+            MieOutput::PhaseMatrix,
+            MieOutput::PhaseMatrixDerivatives,
+            MieOutput::GreekCoefficients { order: 0 },
+            MieOutput::GreekDerivatives { order: 0 },
+        ][o],
+    )
+}
+
+fn to_json(xs: &[f64], results: &[[f64; 4]]) -> String {
+    let list = |v: Vec<String>| v.join(", ");
+    let mut json = String::from("{\n");
+    let _ = writeln!(
+        json,
+        "  \"x\": [{}],",
+        list(xs.iter().map(|x| format!("{x:e}")).collect())
+    );
+    let indices = INDICES
+        .iter()
+        .map(|(r, i)| format!("[{r:e}, {i:e}]"))
+        .collect();
+    let _ = writeln!(json, "  \"m\": [{}],", list(indices));
+    let constants = (0..OUTPUTS.len())
+        .map(|o| format!("\"{}\": {}", OUTPUTS[o], declared(o)))
+        .collect();
+    let _ = writeln!(json, "  \"declared\": {{{}}},", list(constants));
+    json += "  \"c\": {\n";
+    for (o, name) in OUTPUTS.iter().enumerate() {
+        let rows = (0..xs.len())
+            .map(|i| {
+                let row = (0..INDICES.len())
+                    .map(|j| match results[i * INDICES.len() + j][o] {
+                        c if c.is_finite() => format!("{c:.3e}"),
+                        _ => "null".to_owned(),
+                    })
+                    .collect();
+                format!("[{}]", list(row))
+            })
+            .collect();
+        let comma = if o + 1 < OUTPUTS.len() { "," } else { "" };
+        let _ = writeln!(json, "    \"{name}\": [{}]{comma}", list(rows));
+    }
+    json + "  }\n}\n"
+}
+
+/// Runs `f` on every item on all threads; results in the order of `items`.
+fn parallel<I: Sync, R: Send + Copy + Default>(items: &[I], f: impl Fn(&I) -> R + Sync) -> Vec<R> {
+    let next = AtomicUsize::new(0);
+    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
+    let done: Vec<Vec<(usize, R)>> = std::thread::scope(|scope| {
+        let workers: Vec<_> = (0..threads)
+            .map(|_| {
+                scope.spawn(|| {
+                    let mut mine = Vec::new();
+                    loop {
+                        let k = next.fetch_add(1, Ordering::Relaxed);
+                        let Some(item) = items.get(k) else {
+                            break mine;
+                        };
+                        mine.push((k, f(item)));
+                    }
+                })
+            })
+            .collect();
+        workers
+            .into_iter()
+            .map(|w| w.join().expect("worker"))
+            .collect()
+    });
+    let mut results = vec![R::default(); items.len()];
+    let mut count = 0;
+    for (k, r) in done.into_iter().flatten() {
+        results[k] = r;
+        count += 1;
+    }
+    assert_eq!(count, items.len(), "every item computed");
+    results
+}
+
+#[test]
+fn accuracy_map_is_within_the_declared_tolerances() {
+    let xs = size_parameters();
+    assert_eq!(xs.len(), 46, "size parameters");
+    let points: Vec<(f64, Complex<f64>)> = xs
+        .iter()
+        .flat_map(|&x| INDICES.iter().map(move |&(r, i)| (x, Complex::new(r, i))))
+        .collect();
+    let results = parallel(&points, |&(x, m)| point(x, m));
+    if let Ok(path) = std::env::var("SCATRS_ACCURACY_MAP") {
+        std::fs::write(&path, to_json(&xs, &results)).expect("write the map");
+        eprintln!("wrote {path}");
+    }
+    let mut failures = Vec::new();
+    for (o, name) in OUTPUTS.iter().enumerate() {
+        let (k, worst) = results
+            .iter()
+            .enumerate()
+            .map(|(k, c)| (k, c[o]))
+            .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
+        let (x, m) = points[k];
+        eprintln!(
+            "{name:<26} worst C = {worst:8.3} (declared {}) at x = {x:e}, m = {m}",
+            declared(o)
+        );
+        if worst > declared(o) {
+            failures.push(*name);
+        }
+    }
+    assert!(
+        failures.is_empty(),
+        "outside the declared tolerance: {failures:?}"
+    );
+}
+
+/// The values beyond the grid: the phase matrix and the Greek coefficients within their
+/// tolerances (with condition numbers from the double-double derivatives) at x = 2·10³ up to
+/// all orders and at x = 10⁴ and 2·10⁴ up to order 64.
+#[test]
+#[ignore = "double-double Greek coefficients of large spheres: about a minute with all cores"]
+fn large_spheres_are_within_the_tolerances() {
+    let points = [
+        (2e3, Complex::new(1.33, 0.0), None),
+        (2e3, Complex::new(0.3, 0.6), None),
+        (1e4, Complex::new(1.5, 0.01), Some(64)),
+        (2e4, Complex::new(1.33, 0.0), Some(64)),
+    ];
+    let results = parallel(&points, |&(x, m, order)| {
+        let n = default_n_max(x);
+        let f = solve_greek(x, m, &COSINES, None, order, NORMALIZATION);
+        let d = greek_reference(x, m, &COSINES, n, order, NORMALIZATION);
+        let l = f.greek[0].len() - 1;
+        let v = greek_value_measures(x, m, &d);
+        let mut c = [0.0f64; 2];
+        for j in 0..COSINES.len() {
+            for k in 0..4 {
+                let err = diff(f.phase[k][j], d.phase[k][j]);
+                c[0] = c[0].max(constant_of(
+                    MieOutput::PhaseMatrix,
+                    x,
+                    m,
+                    err,
+                    v.phase[j][k],
+                ));
+            }
+        }
+        for k in 0..6 {
+            let err = array_error(&f.greek[k], &d.greek[k]);
+            let output = MieOutput::GreekCoefficients { order: l };
+            c[1] = c[1].max(constant_of(output, x, m, err, v.greek[k]));
+        }
+        c
+    });
+    for ((x, m, _), c) in points.iter().zip(&results) {
+        eprintln!(
+            "x = {x:e}, m = {m}: phase matrix C = {:.3}, Greek C = {:.3}",
+            c[0], c[1]
+        );
+        assert!(
+            c[0] <= declared(0),
+            "phase matrix at x = {x:e}: C = {}",
+            c[0]
+        );
+        assert!(
+            c[1] <= declared(2),
+            "Greek coefficients at x = {x:e}: C = {}",
+            c[1]
+        );
+    }
+}
diff --git a/scatrs/tests/mie_greek_reference.rs b/scatrs/tests/mie_greek_reference.rs
new file mode 100644
index 0000000..36cc184
--- /dev/null
+++ b/scatrs/tests/mie_greek_reference.rs
@@ -0,0 +1,118 @@
+//! The phase matrix and the Greek coefficients against the Arb reference values in
+//! `mie_greek.jsonl`.
+//!
+//! Every reference point is inside the validated domain, so every error must be within its
+//! declared tolerance, with the scale and condition number of the error measure from
+//! `common::greek`. The records are normalized, of the N-term series with the default `f64`
+//! truncation N. Each test prints the worst C = error/((κ′ + cond)·2⁻⁵²) it measures.
+//! `DoubleDouble` must be within the same tolerances times 16·2⁻⁵², as in
+//! `tests/mie_reference.rs`.
+
+mod common;
+
+use common::Record;
+use common::check::{Parts, Worst, error, records, refractive_index, size_parameter};
+use common::greek::{
+    GREEK_NAMES, GreekOutputs, PHASE_NAMES, greek_reference, greek_value_measures, solve_greek,
+};
+use scatrs::mie::{MieOutput, Normalization, default_n_max};
+
+/// The order L of a record.
+fn order(r: &Record) -> usize {
+    r.input_i64("order") as usize
+}
+
+/// The f64 solution of the case of `r`: the default truncation, which the record states.
+fn solve_f64(r: &Record, mu: &[f64]) -> GreekOutputs<f64> {
+    let (x, m) = (size_parameter(r), refractive_index(r));
+    let n = r.input_i64("n_terms") as usize;
+    assert_eq!(
+        n,
+        default_n_max(x),
+        "{}: not the default truncation",
+        r.case
+    );
+    solve_greek(x, m, mu, None, Some(order(r)), Normalization::Normalized)
+}
+
+/// The double-double solution of the case of `r`, with the same number of terms.
+fn solve_dd(r: &Record, mu: &[f64]) -> GreekOutputs<scatrs::scalar::DoubleDouble> {
+    let (x, m) = (size_parameter(r), refractive_index(r));
+    let n = r.input_i64("n_terms") as usize;
+    greek_reference(x, m, mu, n, Some(order(r)), Normalization::Normalized)
+}
+
+/// The Greek records and the phase-matrix records, paired by case.
+fn cases() -> Vec<(Record, Record)> {
+    let mut phases = records("mie_greek.jsonl", ",phase");
+    records("mie_greek.jsonl", ",greek")
+        .into_iter()
+        .map(|g| {
+            let case = g.case.trim_end_matches(",greek").to_owned();
+            let i = phases
+                .iter()
+                .position(|p| p.case.trim_end_matches(",phase") == case)
+                .expect("a phase record per case");
+            (g, phases.remove(i))
+        })
+        .collect()
+}
+
+/// Checks `got` against the records with `worst`, in the measures of the double-double
+/// solution `d`.
+fn check<T: Parts>(
+    got: &GreekOutputs<T>,
+    d: &GreekOutputs<scatrs::scalar::DoubleDouble>,
+    g: &Record,
+    p: &Record,
+    worst_g: &mut Worst,
+    worst_f: &mut Worst,
+) {
+    let (x, m) = (size_parameter(g), refractive_index(g));
+    let measures = greek_value_measures(x, m, d);
+    let output = MieOutput::GreekCoefficients { order: order(g) };
+    for (k, name) in GREEK_NAMES.into_iter().enumerate() {
+        assert_eq!(got.greek[k].len(), order(g) + 1);
+        let err = (0..got.greek[k].len())
+            .map(|l| error(got.greek[k][l], g, name, l))
+            .fold(0.0, f64::max);
+        worst_g.check(output, err, measures.greek[k], g, name);
+    }
+    let mu = p.input_f64s("mu");
+    for (j, cosine) in mu.iter().enumerate() {
+        for (k, name) in PHASE_NAMES.into_iter().enumerate() {
+            let err = error(got.phase[k][j], p, name, j);
+            let detail = format!("{name} at mu = {cosine}");
+            worst_f.check(
+                MieOutput::PhaseMatrix,
+                err,
+                measures.phase[j][k],
+                p,
+                &detail,
+            );
+        }
+    }
+}
+
+#[test]
+fn phase_matrix_and_greek_coefficients_match_arb() {
+    let mut worst_g = Worst::new("Greek coefficients");
+    let mut worst_f = Worst::new("phase matrix");
+    for (g, p) in cases() {
+        let mu = p.input_f64s("mu");
+        let d = solve_dd(&g, &mu);
+        check(&solve_f64(&g, &mu), &d, &g, &p, &mut worst_g, &mut worst_f);
+    }
+}
+
+/// Every `DoubleDouble` value within 16·C·(κ′ + cond)·2⁻¹⁰⁴.
+#[test]
+fn double_double_phase_matrix_and_greek_coefficients_match_arb() {
+    let mut worst_g = Worst::double_double("Greek coefficients");
+    let mut worst_f = Worst::double_double("phase matrix");
+    for (g, p) in cases() {
+        let mu = p.input_f64s("mu");
+        let d = solve_dd(&g, &mu);
+        check(&d, &d, &g, &p, &mut worst_g, &mut worst_f);
+    }
+}
diff --git a/scatrs/tests/mie_reference.rs b/scatrs/tests/mie_reference.rs
index dea95e4..f64cd30 100644
--- a/scatrs/tests/mie_reference.rs
+++ b/scatrs/tests/mie_reference.rs
@@ -9,15 +9,11 @@
 
 mod common;
 
-use common::mie::{
-    EFFICIENCY_NAMES, Measure, derivative_measures, q_of, reference, value_measures,
-};
-use common::{Record, load};
+use common::Record;
+use common::check::{Parts, Worst, error, records, refractive_index, size_parameter};
+use common::mie::{EFFICIENCY_NAMES, derivative_measures, q_of, reference, value_measures};
 use num_complex::Complex;
-use scatrs::mie::{
-    AmplitudeBuffers, Mie, MieOptions, MieOutput, VALIDATED_DOMAIN, ValidatedDomain, default_n_max,
-};
-use scatrs::scalar::DoubleDouble as Dd;
+use scatrs::mie::{AmplitudeBuffers, Mie, MieOptions, MieOutput, default_n_max};
 
 /// The measures of the efficiencies, in the order of `EFFICIENCY_NAMES`.
 const EFFICIENCY_OUTPUTS: [MieOutput; 5] = [
@@ -28,132 +24,10 @@ const EFFICIENCY_OUTPUTS: [MieOutput; 5] = [
     MieOutput::Asymmetry,
 ];
 
-fn size_parameter(r: &Record) -> f64 {
-    r.input_f64("x")
-}
-
-fn refractive_index(r: &Record) -> Complex<f64> {
-    Complex::new(r.input_f64("m_re"), r.input_f64("m_im"))
-}
-
-/// A computed value as the unevaluated sum hi + lo.
-trait Parts: Copy {
-    fn parts(self) -> (f64, f64);
-}
-
-impl Parts for f64 {
-    fn parts(self) -> (f64, f64) {
-        (self, 0.0)
-    }
-}
-
-impl Parts for Dd {
-    fn parts(self) -> (f64, f64) {
-        (self.hi(), self.lo())
-    }
-}
-
-/// |got − v| for the reference value i of `name`.
-fn error<T: Parts>(got: T, r: &Record, name: &str, i: usize) -> f64 {
-    let (hi, lo) = got.parts();
-    r.output(name).dd_error(i, hi, lo)
-}
-
 fn complex_error<T: Parts>(got: Complex<T>, r: &Record, stem: &str, i: usize) -> f64 {
     error(got.re, r, &format!("{stem}_re"), i).hypot(error(got.im, r, &format!("{stem}_im"), i))
 }
 
-/// `DoubleDouble` stands in for the exact values in the accuracy map, so it must be accurate
-/// far beyond `f64`, not within the same constants: it may reach 16 times them, in units of
-/// (κ′ + cond)·2⁻¹⁰⁴ (measured: up to 1.5 times, in the amplitude derivatives at x = 2·10⁴).
-/// Its error then moves a C of the map by at most 2⁻⁴⁸ of the declared constant.
-const DOUBLE_DOUBLE_SLACK: f64 = 16.0;
-
-/// The largest measured C of one output, and where.
-struct Worst {
-    what: String,
-    /// C is measured in units of (κ′ + cond)·2⁻⁵²·`unit`: 1 for `f64`, 2⁻⁵² for `DoubleDouble`.
-    unit: f64,
-    /// The largest C allowed, as a multiple of the declared constant.
-    slack: f64,
-    c: f64,
-    case: String,
-}
-
-impl Worst {
-    fn new(what: &str) -> Self {
-        Self {
-            what: what.to_owned(),
-            unit: 1.0,
-            slack: 1.0,
-            c: 0.0,
-            case: String::new(),
-        }
-    }
-
-    fn double_double(what: &str) -> Self {
-        Self {
-            what: format!("DoubleDouble {what}"),
-            unit: f64::EPSILON,
-            slack: DOUBLE_DOUBLE_SLACK,
-            c: 0.0,
-            case: String::new(),
-        }
-    }
-
-    /// Checks the error `err` of `output` at the case of `r` against the declared tolerance in
-    /// the output's measure, times `unit`·`slack`, and records its C.
-    fn check(&mut self, output: MieOutput, err: f64, measure: Measure, r: &Record, detail: &str) {
-        let (x, m) = (size_parameter(r), refractive_index(r));
-        assert!(
-            VALIDATED_DOMAIN.contains(x, m),
-            "{} is outside the domain",
-            r.case
-        );
-        let tol =
-            VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition) * self.unit * self.slack;
-        assert!(
-            measure.scale > 0.0 && tol.is_finite(),
-            "{} {} {detail}: degenerate measure {measure:?}, tolerance {tol:e}",
-            self.what,
-            r.case
-        );
-        let rel = err / measure.scale;
-        let c = rel / tol * self.slack * ValidatedDomain::constant(output);
-        if c > self.c {
-            self.c = c;
-            self.case = format!("{} {detail}", r.case);
-        }
-        assert!(
-            rel <= tol,
-            "{} {} {detail}: error {rel:.3e} exceeds {tol:.3e} (C = {c:.2}, condition {:.3e})",
-            self.what,
-            r.case,
-            measure.condition
-        );
-    }
-}
-
-impl Drop for Worst {
-    fn drop(&mut self) {
-        if !std::thread::panicking() {
-            eprintln!(
-                "{:<40} worst C = {:8.3} at {}",
-                self.what, self.c, self.case
-            );
-        }
-    }
-}
-
-fn records(file: &str, kind: &str) -> Vec<Record> {
-    let records: Vec<Record> = load(file)
-        .into_iter()
-        .filter(|r| r.case.ends_with(kind))
-        .collect();
-    assert!(!records.is_empty(), "no {kind} records in {file}");
-    records
-}
-
 /// The stored orders of a coefficient record (1-based).
 fn orders(r: &Record) -> Vec<usize> {
     r.input_i64s("orders").iter().map(|&n| n as usize).collect()
diff --git a/tests/test_docs_consistency.py b/tests/test_docs_consistency.py
index d48f184..9d88e21 100644
--- a/tests/test_docs_consistency.py
+++ b/tests/test_docs_consistency.py
@@ -181,14 +181,16 @@ def _declared_mie_constants() -> dict[str, float]:
     body = source[source.index("pub fn constant(") :]
     body = body[: body.index("\n    }\n")]
     constants = {}
-    for arm in re.finditer(r"((?:\s*\|?\s*MieOutput::\w+)+)\s*=>\s*([\d.]+)", body):
+    variant = r"MieOutput::\w+(?:\s*\{\s*\.\.\s*\})?"
+    for arm in re.finditer(rf"((?:\s*\|?\s*{variant})+)\s*=>\s*([\d.]+)", body):
         for variant in re.findall(r"MieOutput::(\w+)", arm.group(1)):
             constants[variant] = float(arm.group(2))
     return constants
 
 
-def test_mie_constants_match_the_accuracy_map() -> None:
-    columns = {
+# The columns of each accuracy map, by the `MieOutput` variant each measures.
+ACCURACY_MAPS = {
+    "mie-accuracy-map.json": {
         "coefficients": "Coefficients",
         "q_ext": "Extinction",
         "q_sca": "Scattering",
@@ -199,10 +201,24 @@ def test_mie_constants_match_the_accuracy_map() -> None:
         "coefficient_derivatives": "CoefficientDerivatives",
         "efficiency_jacobian": "EfficiencyJacobian",
         "amplitude_derivatives": "AmplitudeDerivatives",
-    }
+    },
+    "greek-accuracy-map.json": {
+        "phase_matrix": "PhaseMatrix",
+        "phase_matrix_derivatives": "PhaseMatrixDerivatives",
+        "greek": "GreekCoefficients",
+        "greek_derivatives": "GreekDerivatives",
+    },
+}
+
+
+@pytest.mark.parametrize("name", sorted(ACCURACY_MAPS))
+def test_mie_constants_match_the_accuracy_map(name: str) -> None:
+    columns = ACCURACY_MAPS[name]
     constants = _declared_mie_constants()
-    assert set(constants) == set(columns.values())
-    data = json.loads((ROOT / "docs/validation/data/mie-accuracy-map.json").read_text())
+    measured = {v for c in ACCURACY_MAPS.values() for v in c.values()}
+    assert set(constants) == measured, "a constant that no map measures, or the reverse"
+    data = json.loads((ROOT / "docs/validation/data" / name).read_text())
+    assert set(data["declared"]) == set(data["c"]) == set(columns)
     for column, variant in columns.items():
         assert data["declared"][column] == constants[variant], column
         # The documented margin: every constant at least 4 times the worst C of the map. A null
@@ -223,6 +239,10 @@ MIE_CONSTANT_GROUPS = [
     (("AmplitudeDerivatives",), "their derivatives", "derivatives of S₁, S₂"),
     (("EfficiencyJacobian",), "the efficiency Jacobian", "efficiency Jacobian"),
     (("CoefficientDerivatives",), "the coefficient derivatives", "coefficient derivatives"),
+    (("PhaseMatrix",), "the phase matrix", "phase matrix"),
+    (("PhaseMatrixDerivatives",), "its derivatives", "derivatives of the phase matrix"),
+    (("GreekCoefficients",), "the Greek coefficients", "Greek coefficients"),
+    (("GreekDerivatives",), "their derivatives", "derivatives of the Greek coefficients"),
 ]
 
 
````

- [ ] **Step 2: Generate the reference data**

Run: `uv run --directory reference python -m scatrs_reference generate --only mie_greek`
Expected: `mie_greek.jsonl: 36 records` (about 1 s); `manifest.json` gains its entry. Then `uv run --directory reference python -m pytest -q`: all pass (including the manifest hash and the two new tests).

- [ ] **Step 3: Generate the accuracy map**

Run: `SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/greek-accuracy-map.json cargo test -p scatrs --release --test mie_greek_accuracy_map -- --nocapture`
Expected: PASS, worst C about 7.4 (phase matrix), 22 (derivatives), 1.7 (Greek), 2.8 (Greek derivatives); `wrote …/greek-accuracy-map.json`.

- [ ] **Step 4: Verify**

Run: `cargo test -p scatrs --release --test mie_greek_reference --test mie_reference -- --nocapture`
Expected: PASS; Greek C ≤ 0.72 and phase C ≤ 0.59 for f64, ≤ 0.38 for double-double; the 1a tests unchanged.

Run: `cargo test -p scatrs --release --test mie_greek_accuracy_map -- --ignored --nocapture`
Expected: PASS in about a minute; Greek C ≤ 0.22 at x = 2·10⁴.

Run: `uv run pytest -q tests/test_docs_consistency.py && uvx --from actionlint-py==1.7.12.25 actionlint`
Expected: PASS (constants match both maps with the 4× margin; the reference-suite table matches the manifest).

- [ ] **Step 5: Commit** — `test(mie): Arb references, tolerances and accuracy map for the phase matrix and Greek coefficients`

---

### Task 4: Python API

**Files:**
- Create: `tests/test_mie_greek.py`
- Modify: `scatrs-py/src/mie.rs`, `scatrs-py/src/lib.rs`, `python/scatrs/mie.py`, `python/scatrs/_scatrs/__init__.pyi`, `tests/test_mie.py`, `tests/test_package.py`

**Interfaces:**
- Consumes: Task 2's API.
- Produces: `_scatrs.mie_phase_matrix(x, m, mu, n_max, derivatives, allow_unvalidated, normalized) -> (f: (4, len, n_mu), d: (3, 4, len, n_mu) | None)`; `_scatrs.mie_greek(x, m, order, n_max, derivatives, allow_unvalidated, normalized) -> (n_terms: (len,), g: (6, len, L + 1), d: (3, 6, len, L + 1) | None)`; `_scatrs.MAX_GREEK_ORDER`. Python: `scatrs.mie.phase_matrix(x, m, mu, *, normalized=True, n_max=None, derivatives=False, allow_unvalidated=False) -> MiePhaseMatrix`, `scatrs.mie.greek(x, m, order=None, *, normalized=True, …) -> MieGreek`, `MAX_GREEK_ORDER`. Both result classes are frozen dataclasses with `.d(param)` for `"x"`, `"m_real"`, `"m_imag"` (`"m"` raises: the outputs are real).

- [ ] **Step 1: Apply the diff:**

````diff
diff --git a/python/scatrs/_scatrs/__init__.pyi b/python/scatrs/_scatrs/__init__.pyi
index 01c389e..6ce5bde 100644
--- a/python/scatrs/_scatrs/__init__.pyi
+++ b/python/scatrs/_scatrs/__init__.pyi
@@ -8,6 +8,7 @@ from . import _testing as _testing
 
 __all__ = [
     "MAX_GAUSS_LEGENDRE_POINTS",
+    "MAX_GREEK_ORDER",
     "MAX_MIE_ORDER",
     "MAX_MIE_VALUES",
     "MAX_WIGNER_D_VALUES",
@@ -17,6 +18,8 @@ __all__ = [
     "mie_coefficients",
     "mie_default_n_max",
     "mie_efficiencies",
+    "mie_greek",
+    "mie_phase_matrix",
     "wigner_d",
 ]
 
@@ -25,6 +28,7 @@ MAX_GAUSS_LEGENDRE_POINTS: Final[int]
 MAX_WIGNER_D_VALUES: Final[int]
 MAX_MIE_ORDER: Final[int]
 MAX_MIE_VALUES: Final[int]
+MAX_GREEK_ORDER: Final[int]
 
 def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
 def wigner_d(m: int, n: int, l_max: int, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
@@ -56,3 +60,21 @@ def mie_amplitudes(
     derivatives: bool,
     allow_unvalidated: bool,
 ) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]: ...
+def mie_phase_matrix(
+    x: NDArray[np.float64],
+    m: NDArray[np.complex128],
+    mu: NDArray[np.float64],
+    n_max: int | None,
+    derivatives: bool,
+    allow_unvalidated: bool,
+    normalized: bool,
+) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
+def mie_greek(
+    x: NDArray[np.float64],
+    m: NDArray[np.complex128],
+    order: int | None,
+    n_max: int | None,
+    derivatives: bool,
+    allow_unvalidated: bool,
+    normalized: bool,
+) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64] | None]: ...
diff --git a/python/scatrs/mie.py b/python/scatrs/mie.py
index b170c74..fee0455 100644
--- a/python/scatrs/mie.py
+++ b/python/scatrs/mie.py
@@ -28,16 +28,21 @@ from numpy.typing import ArrayLike, NDArray
 from scatrs import _args, _scatrs
 
 __all__ = [
+    "MAX_GREEK_ORDER",
     "MAX_MIE_ORDER",
     "MAX_MIE_VALUES",
     "PARAMETERS",
     "MieAmplitudes",
     "MieCoefficients",
     "MieEfficiencies",
+    "MieGreek",
+    "MiePhaseMatrix",
     "amplitudes",
     "coefficients",
     "default_n_max",
     "efficiencies",
+    "greek",
+    "phase_matrix",
 ]
 
 MAX_MIE_ORDER: Final[int] = _scatrs.MAX_MIE_ORDER
@@ -47,6 +52,9 @@ MAX_MIE_VALUES: Final[int] = _scatrs.MAX_MIE_VALUES
 """The most complex values one call returns, counting derivatives: ``2**27`` (2 GiB); twice as
 many real values for :func:`efficiencies`."""
 
+MAX_GREEK_ORDER: Final[int] = _scatrs.MAX_GREEK_ORDER
+"""The largest ``order`` :func:`greek` accepts: ``2**21``, twice :data:`MAX_MIE_ORDER`."""
+
 PARAMETERS: Final = ("x", "m_real", "m_imag")
 """The parameters ``derivatives`` may name."""
 
@@ -240,6 +248,95 @@ class MieAmplitudes:
         return MieAmplitudes(ds1_dm, ds2_dm)
 
 
+def _check_real(params: tuple[str, ...], param: str, what: str) -> int:
+    _check(params, param)
+    if param == "m":
+        raise ValueError(f"{what} are real: use 'm_real' or 'm_imag'")
+    return PARAMETERS.index(param)
+
+
+@dataclass(frozen=True, eq=False)
+class MiePhaseMatrix:
+    """The phase matrix of spheres, arrays of shape ``S + mu.shape`` (Bohren & Huffman 1983,
+    §4.4). For a sphere ``F22 = F11`` and ``F44 = F33``; the other elements are 0.
+
+    Attributes
+    ----------
+    f11, f12, f33, f34 : numpy.ndarray
+        ``(|S1|**2 + |S2|**2)/2``, ``(|S2|**2 - |S1|**2)/2``, ``Re(S2 conj(S1))`` and
+        ``Im(S2 conj(S1))``, times ``4/(x**2 q_sca)`` (normalized: ``F11`` averages to 1 over
+        the sphere of directions) or ``4/x**2`` (unnormalized: per unit geometric cross
+        section, which sums over a size distribution).
+    params : tuple of str
+        The parameters :meth:`d` accepts.
+    """
+
+    f11: NDArray[np.float64]
+    f12: NDArray[np.float64]
+    f33: NDArray[np.float64]
+    f34: NDArray[np.float64]
+    params: tuple[str, ...] = ()
+    _derivatives: NDArray[np.float64] | None = field(default=None, repr=False)
+
+    def d(self, param: str) -> MiePhaseMatrix:
+        """The derivatives with respect to ``param`` (``"x"``, ``"m_real"`` or ``"m_imag"``);
+        normalized derivatives include that of ``q_sca``.
+
+        Raises
+        ------
+        ValueError
+            If the derivative was not requested.
+        """
+        p = _check_real(self.params, param, "phase-matrix elements")
+        assert self._derivatives is not None
+        return MiePhaseMatrix(*self._derivatives[p])
+
+
+@dataclass(frozen=True, eq=False)
+class MieGreek:
+    """Greek (generalized-spherical-function) expansion coefficients of the phase matrix of
+    spheres, arrays of shape ``S + (L + 1,)``; ``alpha1[..., l]`` is the coefficient of order
+    ``l`` (Mishchenko, Travis & Lacis 2002, Eqs. 4.75–4.80):
+    ``F11 = sum alpha1[l] d00_l``, ``F22 + F33 = sum (alpha2 + alpha3)[l] d22_l``,
+    ``F22 - F33 = sum (alpha2 - alpha3)[l] d2-2_l``, ``F44 = sum alpha4[l] d00_l``,
+    ``F12 = sum beta1[l] d02_l``, ``F34 = sum beta2[l] d02_l``, with ``dmn_l`` the Wigner
+    d-functions of :func:`scatrs.special.wigner_d` at the cosine of the scattering angle.
+
+    Attributes
+    ----------
+    alpha1, alpha2, alpha3, alpha4, beta1, beta2 : numpy.ndarray
+        float64. Normalized, ``alpha1[..., 0]`` is 1 and ``alpha1[..., 1]`` is ``3 g``;
+        unnormalized, they are those times ``q_sca``. Orders above a sphere's ``2 n_terms``
+        are exactly 0.
+    n_terms : numpy.ndarray
+        int64, shape ``S``: the number of terms of each sphere.
+    params : tuple of str
+        The parameters :meth:`d` accepts.
+    """
+
+    alpha1: NDArray[np.float64]
+    alpha2: NDArray[np.float64]
+    alpha3: NDArray[np.float64]
+    alpha4: NDArray[np.float64]
+    beta1: NDArray[np.float64]
+    beta2: NDArray[np.float64]
+    n_terms: NDArray[np.int64]
+    params: tuple[str, ...] = ()
+    _derivatives: NDArray[np.float64] | None = field(default=None, repr=False)
+
+    def d(self, param: str) -> MieGreek:
+        """The derivatives with respect to ``param`` (``"x"``, ``"m_real"`` or ``"m_imag"``).
+
+        Raises
+        ------
+        ValueError
+            If the derivative was not requested.
+        """
+        p = _check_real(self.params, param, "Greek coefficients")
+        assert self._derivatives is not None
+        return MieGreek(*self._derivatives[p], self.n_terms)
+
+
 _SHARED_DOC = """
     Parameters
     ----------
@@ -385,3 +482,99 @@ amplitudes.__doc__ = """Amplitude scattering functions S₁ and S₂ of spheres.
     extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n    ",
     extra_raise=", ``mu`` outside [-1, 1] or NaN",
 )
+
+
+def phase_matrix(
+    x: ArrayLike,
+    m: ArrayLike,
+    mu: ArrayLike,
+    *,
+    normalized: bool = True,
+    n_max: SupportsIndex | None = None,
+    derivatives: bool | Sequence[str] = False,
+    allow_unvalidated: bool = False,
+) -> MiePhaseMatrix:
+    params = _parameters(derivatives)
+    xb, mb, shape = _inputs(x, m)
+    mus = _args.real_array(mu, "mu")
+    f, d = _scatrs.mie_phase_matrix(
+        xb,
+        mb,
+        mus.reshape(mus.shape),
+        _n_max(n_max),
+        bool(params),
+        _args.flag(allow_unvalidated, "allow_unvalidated"),
+        _args.flag(normalized, "normalized"),
+    )
+    f, d = _frozen(f), _frozen(d)
+    out = (*shape, *mus.shape)
+    dd = None if d is None else d.reshape((3, 4, *out))
+    return MiePhaseMatrix(*(v.reshape(out) for v in f), params=params, _derivatives=dd)
+
+
+phase_matrix.__doc__ = """The phase matrix F11, F12, F33, F34 of spheres.
+
+    Returns
+    -------
+    MiePhaseMatrix
+        Shape ``S + mu.shape``.
+""" + _SHARED_DOC.format(
+    extra="mu : array_like\n        Cosines of the scattering angle, in [-1, 1].\n"
+    "    normalized : bool, optional\n        Divide by ``q_sca`` (the default), so that F11"
+    " averages to 1.\n    ",
+    extra_raise=", ``mu`` outside [-1, 1] or NaN",
+)
+
+
+def greek(
+    x: ArrayLike,
+    m: ArrayLike,
+    order: SupportsIndex | None = None,
+    *,
+    normalized: bool = True,
+    n_max: SupportsIndex | None = None,
+    derivatives: bool | Sequence[str] = False,
+    allow_unvalidated: bool = False,
+) -> MieGreek:
+    params = _parameters(derivatives)
+    xb, mb, shape = _inputs(x, m)
+    terms, g, d = _scatrs.mie_greek(
+        xb,
+        mb,
+        None if order is None else _args.size(order, "order"),
+        _n_max(n_max),
+        bool(params),
+        _args.flag(allow_unvalidated, "allow_unvalidated"),
+        _args.flag(normalized, "normalized"),
+    )
+    g, d = _frozen(g), _frozen(d)
+    width = g.shape[-1]
+    out = (*shape, width)
+    n_terms = _frozen(terms.astype(np.int64)).reshape(shape)
+    dd = None if d is None else d.reshape((3, 6, *out))
+    return MieGreek(*(v.reshape(out) for v in g), n_terms, params, dd)
+
+
+greek.__doc__ = (
+    """Greek expansion coefficients of the phase matrix of spheres, exact up to
+    rounding: the phase matrix at the nodes of a Gauss–Legendre rule with
+    ``n_terms + ceil((L + 1)/2)`` points, projected onto the Wigner d-functions.
+
+    Returns
+    -------
+    MieGreek
+        Shape ``S + (L + 1,)``.
+"""
+    + _SHARED_DOC.format(
+        extra="order : int, optional\n        The highest order L; by default ``2 n_terms`` of the"
+        " largest sphere, which is\n        all of them. Orders above ``2 n_terms`` are exactly 0."
+        "\n    normalized : bool, optional\n        Normalize ``alpha1[..., 0]`` to 1 (the"
+        " default), or to ``q_sca``.\n    ",
+        extra_raise=", ``order`` above ``MAX_GREEK_ORDER``",
+    )
+    + """
+    The cost per sphere grows like ``n_terms * L`` for the projection (on top of the
+    amplitudes at ``n_terms + L/2`` angles): all orders take about 0.1 s at x = 1e3 and
+    8 s at x = 1e4.
+"""
+)
diff --git a/scatrs-py/src/lib.rs b/scatrs-py/src/lib.rs
index dccf82d..e36e375 100644
--- a/scatrs-py/src/lib.rs
+++ b/scatrs-py/src/lib.rs
@@ -27,6 +27,9 @@ fn _scatrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
     m.add_function(wrap_pyfunction!(mie::mie_efficiencies, m)?)?;
     m.add_function(wrap_pyfunction!(mie::mie_coefficients, m)?)?;
     m.add_function(wrap_pyfunction!(mie::mie_amplitudes, m)?)?;
+    m.add("MAX_GREEK_ORDER", scatrs::mie::MAX_GREEK_ORDER)?;
+    m.add_function(wrap_pyfunction!(mie::mie_phase_matrix, m)?)?;
+    m.add_function(wrap_pyfunction!(mie::mie_greek, m)?)?;
     testing::register(m)?;
     Ok(())
 }
diff --git a/scatrs-py/src/mie.rs b/scatrs-py/src/mie.rs
index 4cf620a..24910ad 100644
--- a/scatrs-py/src/mie.rs
+++ b/scatrs-py/src/mie.rs
@@ -9,16 +9,16 @@
 //! Broadcasting makes outputs far larger than inputs, so every output is bounded by
 //! `MAX_MIE_VALUES` before anything is allocated, and then allocated and filled without the GIL.
 
-use numpy::ndarray::{Array2, Array3, ArrayViewD};
+use numpy::ndarray::{Array2, Array3, Array4, ArrayViewD};
 use numpy::{
-    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods,
+    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayDyn, PyArrayMethods,
     PyReadonlyArrayDyn, PyUntypedArrayMethods,
 };
 use pyo3::exceptions::{PyBufferError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
 use pyo3::prelude::*;
 use scatrs::mie::{
-    AmplitudeBuffers, MAX_MIE_ORDER, MAX_SIZE_PARAMETER, MIN_SIZE_PARAMETER, Mie, MieOptions,
-    MieSolution, n_terms,
+    AmplitudeBuffers, GreekCoefficients, MAX_GREEK_ORDER, MAX_MIE_ORDER, MAX_SIZE_PARAMETER,
+    MIN_SIZE_PARAMETER, Mie, MieOptions, MieSolution, Normalization, PhaseBuffers, n_terms,
 };
 
 use crate::errors::to_pyerr;
@@ -170,6 +170,46 @@ fn complex_values(what: &str, count: Option<usize>, derivatives: bool) -> scatrs
     )
 }
 
+/// `count` real values, at most 2·`MAX_MIE_VALUES` (the bytes of `MAX_MIE_VALUES` complex
+/// values).
+fn real_values(what: &str, count: Option<usize>, derivatives: bool) -> scatrs::Result<usize> {
+    bounded(
+        what,
+        count,
+        "real values",
+        derivatives,
+        2 * MAX_MIE_VALUES,
+        "2 * MAX_MIE_VALUES",
+    )
+}
+
+fn normalization(normalized: bool) -> Normalization {
+    if normalized {
+        Normalization::Normalized
+    } else {
+        Normalization::Unnormalized
+    }
+}
+
+/// The cosines `mu` as a vector, every one in [−1, 1] (checked here too, so that an empty x
+/// does not hide a bad cosine).
+fn cosines(py: Python<'_>, mu: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<Vec<f64>> {
+    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
+    if let Some((j, v)) = mu
+        .iter()
+        .enumerate()
+        .find(|&(_, v)| !(-1.0..=1.0).contains(v))
+    {
+        return Err(to_pyerr(
+            py,
+            invalid(format!(
+                "element {j} of mu: a cosine of the scattering angle must be in [-1, 1], got {v}"
+            )),
+        ));
+    }
+    Ok(mu)
+}
+
 /// A zeroed vector of `len` values, or `MemoryError`.
 fn zeros<T: Clone + Default>(len: usize, what: &str) -> PyResult<Vec<T>> {
     let mut v = Vec::new();
@@ -379,20 +419,7 @@ pub fn mie_amplitudes<'py>(
     let (x, m, mu) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&mu, "mu")?);
     let (xv, mv) = (x.as_array(), m.as_array());
     let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
-    let mu: Vec<f64> = mu.as_array().iter().copied().collect();
-    // The core checks the cosines too; checking here also covers an empty x.
-    if let Some((j, v)) = mu
-        .iter()
-        .enumerate()
-        .find(|&(_, v)| !(-1.0..=1.0).contains(v))
-    {
-        return Err(to_pyerr(
-            py,
-            invalid(format!(
-                "element {j} of mu: a cosine of the scattering angle must be in [-1, 1], got {v}"
-            )),
-        ));
-    }
+    let mu = cosines(py, &mu)?;
     let (len, n_mu) = (xv.len(), mu.len());
     let arrays = if derivatives { 6 } else { 2 };
     // At least one angle, so that an empty mu does not leave the number of spheres (each still
@@ -455,3 +482,197 @@ pub fn mie_amplitudes<'py>(
     };
     Ok((s.into_pyarray(py), d))
 }
+
+type PhaseArrays<'py> = (Bound<'py, PyArray3<f64>>, Option<Bound<'py, PyArray4<f64>>>);
+
+/// F₁₁, F₁₂, F₃₃, F₃₄ (4, len, n_mu) at the cosines `mu`, normalized or not, and with
+/// `derivatives` their derivatives with respect to x, Re m and Im m (3, 4, len, n_mu).
+#[pyfunction]
+#[allow(clippy::too_many_arguments)]
+pub fn mie_phase_matrix<'py>(
+    py: Python<'py>,
+    x: Bound<'py, PyArrayDyn<f64>>,
+    m: Bound<'py, PyArrayDyn<Complex64>>,
+    mu: Bound<'py, PyArrayDyn<f64>>,
+    n_max: Option<usize>,
+    derivatives: bool,
+    allow_unvalidated: bool,
+    normalized: bool,
+) -> PyResult<PhaseArrays<'py>> {
+    let (x, m, mu) = (borrow(&x, "x")?, borrow(&m, "m")?, borrow(&mu, "mu")?);
+    let (xv, mv) = (x.as_array(), m.as_array());
+    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
+    let mu = cosines(py, &mu)?;
+    let (len, n_mu) = (xv.len(), mu.len());
+    let arrays = if derivatives { 16 } else { 4 };
+    // At least one angle, as in `mie_amplitudes`.
+    let count = len
+        .checked_mul(n_mu.max(1))
+        .and_then(|v| v.checked_mul(arrays));
+    real_values("the phase matrix", count, derivatives).map_err(|e| to_pyerr(py, e))?;
+    let per_array = len * n_mu;
+    let norm = normalization(normalized);
+    let (f, d) = py
+        .detach(|| -> Result<_, Failure> {
+            let mut f = zeros::<f64>(4 * per_array, "the phase matrix")?;
+            let mut d = zeros::<f64>(if derivatives { 12 * per_array } else { 0 }, "derivatives")?;
+            let mut mie = Mie::<f64>::new();
+            for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
+                if n_mu == 0 {
+                    // Nothing to compute, but the inputs are still checked.
+                    n_terms(xi, mi, &opts).map_err(at(i))?;
+                    continue;
+                }
+                let sol = solve(&mut mie, i, xi, mi, &opts, derivatives)?;
+                let row = i * n_mu..(i + 1) * n_mu;
+                let values = phase_buffers(&mut f, per_array, row.clone());
+                if derivatives {
+                    let (dx, rest) = d.split_at_mut(4 * per_array);
+                    let (dr, di) = rest.split_at_mut(4 * per_array);
+                    sol.phase_matrix_with_derivatives(
+                        &mu,
+                        norm,
+                        values,
+                        [
+                            phase_buffers(dx, per_array, row.clone()),
+                            phase_buffers(dr, per_array, row.clone()),
+                            phase_buffers(di, per_array, row.clone()),
+                        ],
+                    )?;
+                } else {
+                    sol.phase_matrix(&mu, norm, values)?;
+                }
+            }
+            Ok((f, d))
+        })
+        .map_err(|f| f.into_pyerr(py))?;
+    let f = Array3::from_shape_vec((4, len, n_mu), f).map_err(shape_error)?;
+    let d = if derivatives {
+        let d = Array4::from_shape_vec((3, 4, len, n_mu), d).map_err(shape_error)?;
+        Some(d.into_pyarray(py))
+    } else {
+        None
+    };
+    Ok((f.into_pyarray(py), d))
+}
+
+/// The rows `row` of the four consecutive arrays of `per_array` values in `buf`, as
+/// phase-matrix buffers.
+fn phase_buffers(
+    buf: &mut [f64],
+    per_array: usize,
+    row: std::ops::Range<usize>,
+) -> PhaseBuffers<'_, f64> {
+    let (f11, rest) = buf.split_at_mut(per_array);
+    let (f12, rest) = rest.split_at_mut(per_array);
+    let (f33, f34) = rest.split_at_mut(per_array);
+    PhaseBuffers {
+        f11: &mut f11[row.clone()],
+        f12: &mut f12[row.clone()],
+        f33: &mut f33[row.clone()],
+        f34: &mut f34[row],
+    }
+}
+
+type GreekArrays<'py> = (
+    Bound<'py, PyArray1<u64>>,
+    Bound<'py, PyArray3<f64>>,
+    Option<Bound<'py, PyArray4<f64>>>,
+);
+
+/// The arrays of `g` in the order α₁, α₂, α₃, α₄, β₁, β₂.
+fn greek_arrays(g: &GreekCoefficients<f64>) -> [&[f64]; 6] {
+    [
+        &g.alpha1, &g.alpha2, &g.alpha3, &g.alpha4, &g.beta1, &g.beta2,
+    ]
+}
+
+/// The number of terms of each element (len), the Greek coefficients α₁, α₂, α₃, α₄, β₁, β₂
+/// (6, len, L + 1) up to `order` (by default the largest 2N, each sphere zero-padded above its
+/// own 2N), normalized or not, and with `derivatives` their derivatives with respect to x,
+/// Re m and Im m (3, 6, len, L + 1).
+#[pyfunction]
+#[allow(clippy::too_many_arguments)]
+pub fn mie_greek<'py>(
+    py: Python<'py>,
+    x: Bound<'py, PyArrayDyn<f64>>,
+    m: Bound<'py, PyArrayDyn<Complex64>>,
+    order: Option<usize>,
+    n_max: Option<usize>,
+    derivatives: bool,
+    allow_unvalidated: bool,
+    normalized: bool,
+) -> PyResult<GreekArrays<'py>> {
+    let (x, m) = (borrow(&x, "x")?, borrow(&m, "m")?);
+    let (xv, mv) = (x.as_array(), m.as_array());
+    let opts = prepare(py, &xv, &mv, n_max, allow_unvalidated)?;
+    if let Some(l) = order.filter(|&l| l > MAX_GREEK_ORDER) {
+        return Err(to_pyerr(
+            py,
+            invalid(format!(
+                "the Greek order must be at most {MAX_GREEK_ORDER}, got {l}"
+            )),
+        ));
+    }
+    let len = xv.len();
+    let arrays = if derivatives { 24 } else { 6 };
+    // Every sphere has at least order 0, so this bounds the sizing pass below.
+    real_values(
+        "the Greek coefficients",
+        len.checked_mul(arrays),
+        derivatives,
+    )
+    .map_err(|e| to_pyerr(py, e))?;
+    let norm = normalization(normalized);
+    let (terms, g, d, width) = py
+        .detach(|| -> Result<_, Failure> {
+            // Check every element and find the number of orders, before allocating.
+            let mut most = 0;
+            for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
+                most = most.max(n_terms(xi, mi, &opts).map_err(at(i))?);
+            }
+            let width = order.unwrap_or(2 * most) + 1;
+            let count = len.checked_mul(width).and_then(|v| v.checked_mul(arrays));
+            let per_array = real_values("the Greek coefficients", count, derivatives)? / arrays;
+            let mut terms = zeros::<u64>(len, "the numbers of terms")?;
+            let mut g = zeros::<f64>(6 * per_array, "the Greek coefficients")?;
+            let mut d = zeros::<f64>(if derivatives { 18 * per_array } else { 0 }, "derivatives")?;
+            let mut mie = Mie::<f64>::new();
+            for (i, (&xi, &mi)) in xv.iter().zip(mv.iter()).enumerate() {
+                let s = solve(&mut mie, i, xi, mi, &opts, derivatives)?;
+                // `n_terms` above made the same checks and choice, unless x or m changed since.
+                if order.is_none() && s.n_terms() > most {
+                    return Err(
+                        invalid(format!("element {i}: x or m changed during the call")).into(),
+                    );
+                }
+                terms[i] = s.n_terms() as u64;
+                let row = i * width..(i + 1) * width;
+                let put = |out: &mut [f64], first: usize, c: &GreekCoefficients<f64>| {
+                    for (k, src) in greek_arrays(c).into_iter().enumerate() {
+                        let off = (first + k) * per_array;
+                        out[off + row.start..off + row.end].copy_from_slice(src);
+                    }
+                };
+                if derivatives {
+                    let (values, jac) = s.greek_with_derivatives(Some(width - 1), norm)?;
+                    put(&mut g, 0, &values);
+                    for (p, c) in jac.iter().enumerate() {
+                        put(&mut d, 6 * p, c);
+                    }
+                } else {
+                    put(&mut g, 0, &s.greek(Some(width - 1), norm)?);
+                }
+            }
+            Ok((terms, g, d, width))
+        })
+        .map_err(|f| f.into_pyerr(py))?;
+    let g = Array3::from_shape_vec((6, len, width), g).map_err(shape_error)?;
+    let d = if derivatives {
+        let d = Array4::from_shape_vec((3, 6, len, width), d).map_err(shape_error)?;
+        Some(d.into_pyarray(py))
+    } else {
+        None
+    };
+    Ok((PyArray1::from_vec(py, terms), g.into_pyarray(py), d))
+}
diff --git a/tests/test_mie.py b/tests/test_mie.py
index 353847a..4cbd4a0 100644
--- a/tests/test_mie.py
+++ b/tests/test_mie.py
@@ -14,6 +14,8 @@ CALLS = {
     "efficiencies": lambda x, m, **kw: mie.efficiencies(x, m, **kw),
     "coefficients": lambda x, m, **kw: mie.coefficients(x, m, **kw),
     "amplitudes": lambda x, m, **kw: mie.amplitudes(x, m, 0.5, **kw),
+    "phase_matrix": lambda x, m, **kw: mie.phase_matrix(x, m, 0.5, **kw),
+    "greek": lambda x, m, **kw: mie.greek(x, m, 4, **kw),
 }
 
 
diff --git a/tests/test_mie_greek.py b/tests/test_mie_greek.py
new file mode 100644
index 0000000..76fdce1
--- /dev/null
+++ b/tests/test_mie_greek.py
@@ -0,0 +1,156 @@
+"""scatrs.mie.phase_matrix and scatrs.mie.greek: values, shapes, derivatives and errors (the
+checks shared with the other Mie functions are in test_mie.py)."""
+
+import numpy as np
+import pytest
+
+from scatrs import InvalidInputError, mie, special
+
+
+def test_phase_matrix_is_built_from_the_amplitudes():
+    x, m, mu = 7.0, 1.5 + 0.01j, np.array([1.0, 0.3, -0.8, -1.0])
+    s = mie.amplitudes(x, m, mu)
+    q = mie.efficiencies(x, m)
+    f = mie.phase_matrix(x, m, mu)
+    n1, n2 = np.abs(s.s1) ** 2, np.abs(s.s2) ** 2
+    p = s.s2 * np.conj(s.s1)
+    c = 4 / (x**2 * q.q_sca)
+    for got, expected in [
+        (f.f11, (n1 + n2) / 2), (f.f12, (n2 - n1) / 2), (f.f33, p.real), (f.f34, p.imag)
+    ]:  # fmt: skip
+        np.testing.assert_allclose(got, c * expected, rtol=1e-14, atol=1e-14 * f.f11.max())
+    u = mie.phase_matrix(x, m, mu, normalized=False)
+    np.testing.assert_allclose(u.f11, f.f11 * q.q_sca, rtol=1e-14)
+
+
+def test_normalized_phase_function_averages_to_one():
+    nodes, weights = special.gauss_legendre(200)
+    f = mie.phase_matrix([0.5, 30.0], 1.33, nodes)
+    np.testing.assert_allclose(f.f11 @ weights / 2, 1.0, rtol=1e-13)
+
+
+def test_greek_identities():
+    x, m = np.array([0.3, 5.0, 60.0]), 1.5 + 0.01j
+    g = mie.greek(x, m)
+    q = mie.efficiencies(x, m)
+    np.testing.assert_array_equal(g.alpha1[:, 0], 1.0)
+    np.testing.assert_allclose(g.alpha1[:, 1], 3 * q.g, rtol=1e-13)
+    # F11(mu = 1) = sum of alpha1, since d00_l(1) = 1.
+    f = mie.phase_matrix(x, m, 1.0)
+    np.testing.assert_allclose(g.alpha1.sum(axis=-1), f.f11, rtol=1e-12)
+    u = mie.greek(x, m, normalized=False)
+    np.testing.assert_allclose(u.alpha1[:, 0], q.q_sca, rtol=1e-15)
+    np.testing.assert_allclose(u.beta1, g.beta1 * q.q_sca[:, None], atol=1e-14 * q.q_sca.max())
+
+
+def test_greek_expansion_reproduces_the_phase_matrix():
+    x, m, mu = 4.0, 1.33 + 1e-3j, np.array([0.9, 0.1, -0.6])
+    g, f = mie.greek(x, m), mie.phase_matrix(x, m, mu)
+    lmax = g.alpha1.shape[-1] - 1
+    d00, d22, d2m2, d02 = (
+        special.wigner_d(a, b, lmax, mu) for a, b in [(0, 0), (2, 2), (2, -2), (0, 2)]
+    )
+    np.testing.assert_allclose(d00 @ g.alpha1, f.f11, rtol=1e-12)
+    np.testing.assert_allclose(d22 @ (g.alpha2 + g.alpha3), f.f11 + f.f33, rtol=1e-12, atol=1e-13)
+    np.testing.assert_allclose(d2m2 @ (g.alpha2 - g.alpha3), f.f11 - f.f33, rtol=1e-12, atol=1e-13)
+    np.testing.assert_allclose(d00 @ g.alpha4, f.f33, rtol=1e-12, atol=1e-13)
+    np.testing.assert_allclose(d02 @ g.beta1, f.f12, rtol=1e-12, atol=1e-13)
+    np.testing.assert_allclose(d02 @ g.beta2, f.f34, rtol=1e-12, atol=1e-13)
+
+
+def test_shapes_broadcast_and_greek_is_zero_padded():
+    x = np.array([[0.5], [20.0]])
+    m = np.array([1.33, 1.5 + 0.01j, 2.0])
+    mu = np.linspace(-1, 1, 6).reshape(2, 3)
+    f = mie.phase_matrix(x, m, mu)
+    assert f.f11.shape == (2, 3, 2, 3) and f.f34.dtype == np.float64
+    g = mie.greek(x, m)
+    n = mie.default_n_max(x)[:, 0]
+    assert g.n_terms.shape == (2, 3)
+    assert g.alpha1.shape == (2, 3, 2 * n.max() + 1)
+    assert np.all(g.alpha1[0, :, 2 * n[0] + 1 :] == 0)
+    assert mie.greek(1.0, 1.5, 3).alpha1.shape == (4,)
+    # An order above 2 n_terms pads with exact zeros, as for the smaller spheres of a batch.
+    small = mie.greek(1.0, 1.5)
+    big = mie.greek(1.0, 1.5, 50)
+    np.testing.assert_array_equal(big.alpha1[: small.alpha1.size], small.alpha1)
+    assert np.all(big.alpha1[small.alpha1.size :] == 0)
+
+
+def test_elements_equal_scalar_calls():
+    x = np.array([0.1, 3.0, 40.0])
+    m = np.array([1.33 + 1e-8j, 2.5 + 1j, 0.75])
+    g = mie.greek(x, m, 30)
+    f = mie.phase_matrix(x, m, [0.5, -1.0])
+    for i in range(3):
+        np.testing.assert_array_equal(g.beta2[i], mie.greek(x[i], m[i], 30).beta2)
+        np.testing.assert_array_equal(f.f12[i], mie.phase_matrix(x[i], m[i], [0.5, -1.0]).f12)
+
+
+def _fd(f, h):
+    return (f(+h) - f(-h)) / (2 * h)
+
+
+@pytest.mark.parametrize("normalized", [True, False])
+def test_derivatives_match_finite_differences(normalized):
+    x, m, mu = 6.0, 1.5 + 0.05j, [0.95, 0.2, -1.0]
+    n = int(mie.default_n_max(x)) + 1  # x ± h keeps the number of terms
+    kw = {"n_max": n, "normalized": normalized}
+    f = mie.phase_matrix(x, m, mu, derivatives=True, **kw)
+    g = mie.greek(x, m, 2 * n, derivatives=True, **kw)
+    steps = {
+        "x": (lambda h: (x + h, m), 1e-6 * x),
+        "m_real": (lambda h: (x, m + h), 1e-7),
+        "m_imag": (lambda h: (x, m + 1j * h), 1e-7),
+    }
+    for p, (at, h) in steps.items():
+        for name in ("f11", "f12", "f33", "f34"):
+            fd = _fd(
+                lambda e, at=at, name=name: getattr(mie.phase_matrix(*at(e), mu, **kw), name), h
+            )
+            np.testing.assert_allclose(getattr(f.d(p), name), fd, atol=1e-6 * np.abs(fd).max())
+        for name in ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2"):
+            fd = _fd(lambda e, at=at, name=name: getattr(mie.greek(*at(e), 2 * n, **kw), name), h)
+            got = getattr(g.d(p), name)
+            np.testing.assert_allclose(got, fd, atol=1e-6 * max(np.abs(fd).max(), 1e-3))
+    with pytest.raises(ValueError, match="are real"):
+        f.d("m")
+    with pytest.raises(ValueError, match="are real"):
+        g.d("m")
+    assert g.d("x").n_terms is g.n_terms
+
+
+def test_greek_errors():
+    with pytest.raises(InvalidInputError, match="order must not be negative"):
+        mie.greek(1.0, 1.5, -1)
+    with pytest.raises(InvalidInputError, match="Greek order must be at most 2097152"):
+        mie.greek(1.0, 1.5, mie.MAX_GREEK_ORDER + 1)
+    with pytest.raises(TypeError):
+        mie.greek(1.0, 1.5, 3.0)
+    with pytest.raises(TypeError, match="normalized must be a bool"):
+        mie.greek(1.0, 1.5, normalized=1)
+    with pytest.raises(TypeError, match="normalized must be a bool"):
+        mie.phase_matrix(1.0, 1.5, 0.5, normalized=None)
+    with pytest.raises(InvalidInputError, match="element 0 of mu"):
+        mie.phase_matrix(1.0, 1.5, 1.5)
+    # The output bounds count the orders and the derivatives.
+    with pytest.raises(InvalidInputError, match=r"2 \* MAX_MIE_VALUES"):
+        mie.greek(np.ones(2**12), 1.5, 2**16)
+    with pytest.raises(InvalidInputError, match=r"2 \* MAX_MIE_VALUES"):
+        mie.phase_matrix(np.ones(2**12), 1.5, np.zeros(2**14), derivatives=True)
+
+
+def test_empty_inputs():
+    g = mie.greek(np.zeros(0), 1.5)
+    assert g.alpha1.shape == (0, 1) and g.n_terms.shape == (0,)
+    assert mie.phase_matrix([1.0], 1.5, np.zeros(0)).f11.shape == (1, 0)
+    with pytest.raises(InvalidInputError, match="element 1 of mu"):
+        mie.phase_matrix([], 1.5, [0.5, np.nan])
+
+
+def test_results_are_frozen_and_read_only():
+    g = mie.greek(1.0, 1.5, derivatives=True)
+    f = mie.phase_matrix(1.0, 1.5, [0.5], derivatives=True)
+    for a in (g.alpha1, g.n_terms, g.d("x").beta1, f.f11, f.d("m_imag").f34):
+        with pytest.raises(ValueError, match="read-only"):
+            a[...] = 0
diff --git a/tests/test_package.py b/tests/test_package.py
index 8eec64a..2cbbed2 100644
--- a/tests/test_package.py
+++ b/tests/test_package.py
@@ -67,11 +67,13 @@ def test_limits_are_exposed():
     assert special.MAX_WIGNER_D_VALUES == _scatrs.MAX_WIGNER_D_VALUES == 2**28
     assert mie.MAX_MIE_ORDER == _scatrs.MAX_MIE_ORDER == 2**20
     assert mie.MAX_MIE_VALUES == _scatrs.MAX_MIE_VALUES == 2**27
+    assert mie.MAX_GREEK_ORDER == _scatrs.MAX_GREEK_ORDER == 2**21
 
 
 def test_test_hooks_live_outside_the_extension_namespace():
     assert set(_scatrs.__all__) == {
         "MAX_GAUSS_LEGENDRE_POINTS",
+        "MAX_GREEK_ORDER",
         "MAX_MIE_ORDER",
         "MAX_MIE_VALUES",
         "MAX_WIGNER_D_VALUES",
@@ -81,6 +83,8 @@ def test_test_hooks_live_outside_the_extension_namespace():
         "mie_coefficients",
         "mie_default_n_max",
         "mie_efficiencies",
+        "mie_greek",
+        "mie_phase_matrix",
         "wigner_d",
     }
     assert not hasattr(_scatrs, "_raise_core_error")
````

- [ ] **Step 2: Verify**

Run: `uv run pytest -q && uv run python -m mypy.stubtest scatrs._scatrs && uv run ruff check . && uv run ruff format --check .`
Expected: all pass (the shared invalid-input and element-naming tests now also cover `phase_matrix` and `greek`).

- [ ] **Step 3: Commit** — `feat(python): scatrs.mie.phase_matrix and scatrs.mie.greek`

---

### Task 5: Documentation

**Files:**
- Create: `docs/examples/greek.md` (executed MyST-NB page, like `docs/examples/mie.md`)
- Modify: `docs/theory/mie.md`, `docs/validation/accuracy.md`, `docs/limitations.md`, `docs/index.md`, `docs/api.md` (if the new names need listing), `CHANGELOG.md`, `docs/conventions.md` (if it lists Mie outputs), `docs/specs/2026-10-13-phase-1b-greek-design.md` (already updated with this plan: check only), `tests/test_docs_consistency.py` (new contracts for any figure quoted in more than one place), `docs/plans/phase-0-followups.md` or a new `docs/plans/phase-1b-followups.md`

**Requirements** (facts to state; prose in the style of the existing pages, figures exactly as measured):

1. `docs/theory/mie.md`, new sections after "Closed-form derivatives":
   - **The phase matrix.** F₁₁ = (|S₁|² + |S₂|²)/2, F₁₂ = (|S₂|² − |S₁|²)/2, F₃₃ = Re(S₂S₁*), F₃₄ = Im(S₂S₁*), F₂₂ = F₁₁, F₄₄ = F₃₃ (Bohren & Huffman §4.4); normalized by 4/(x²Q_sca) (½∫F₁₁dμ = 1) or 4/x² (per unit geometric cross section, what sums over sizes); derivatives from the amplitude derivatives with ∂/∂(Im m) = i·d/dm, and for the normalized form the derivative of Q_sca.
   - **Greek coefficients.** The expansion (as in the `GreekCoefficients` rustdoc), with `wigner_d`'s convention (d²₀₂(θ) = (√6/4) sin²θ); c_l = (2l + 1)/2·∫F d_l dμ; each element is a polynomial of degree 2N in μ, so n_q = N + ⌈(L + 1)/2⌉ Gauss–Legendre nodes make the projection exact; orders above 2N vanish (padded with exact zeros); normalization by the projected α₁⁰ (exactly 1), unnormalized = that × Q_sca; streaming (O(n_q·L) time, O(L) memory; storing d at every node would need about 25 GB at x = 10⁴); linearity gives the Jacobians, and `GreekProjector` projects sums over many spheres at once (phase 2).
   - **Sign conventions** table: B&H elements → this expansion; the Rayleigh limit α₁ = (1, 0, ½), α₂² = 3, α₄¹ = 3/2, β₁² = −√6/2 (pinned by `rayleigh_limit_pins_the_sign_conventions`), F₁₂ < 0 at 90° so the degree of linear polarization is −F₁₂/F₁₁. Do not claim other codes' conventions (sasktran2's comparison is deferred).
   - **Accuracy model additions** in "The accuracy model": priors κ (phase matrix), κ·max(1, |m|/|m − 1|) (its derivatives), κ + ((L + 1)² + x²)/8 and κ·max(1, |m|/|m − 1|) + ((L + 1)² + x²)/8 (Greek), with the node-rounding explanation of "Measured results" above (C reached 85 and 100 without the term).
   - **Cost**: Greek all orders 0.09 s at x = 10³ and 7.6 s at x = 10⁴; asymptotic Gauss–Legendre makes the rule negligible.
2. `docs/examples/greek.md` (executed, cells assert their claims): phase function and degree of linear polarization −F₁₂/F₁₁ of a water droplet (x = 10, m = 1.33) against angle; Greek coefficients of the same sphere, α₁⁰ = 1 and α₁¹ = 3g asserted; reconstruct F₁₁ and F₁₂ from the coefficients with `scatrs.special.wigner_d` and assert agreement (≤ 1e-12 relative to max F₁₁); truncation: F₁₁ from the first L orders for a few L against the full expansion at the forward peak (shows why all orders are needed for exact phase functions); a derivative check against central differences. Add the page to the toctree next to `examples/mie`.
3. `docs/validation/accuracy.md`: a section for the phase matrix and Greek coefficients: the grid (46 x up to 10³, 28 indices, 13 cosines, all orders to x = 160 and order 100 above), the worst C per output against the declared constants (read them from `docs/validation/data/greek-accuracy-map.json` in an executed cell or the same plotting code as the Mie map, as the page does), the weekly large-sphere test, and the quadrature term.
4. `docs/limitations.md`, "Mie scattering": the Greek cost (seconds at x = 10⁴ for all orders; the DD reference costs O(x²) so the map stops at 10³ and larger spheres are checked weekly at three sizes); the node-rounding error growing like L² + x² (measured about 2·10⁻⁹ of max α₁ at x = 2·10⁴ with L = 64, and 2·10⁻¹¹ at x = 10³ for all orders); only spheres (no size distributions or mixtures yet: phase 2).
5. `docs/index.md`: status line (phase 1b: phase matrix, Greek coefficients, O(n) Gauss–Legendre). `CHANGELOG.md` [Unreleased]/Added: the four user-visible additions (O(n) GL and `gauss_legendre_into`; phase matrix; Greek coefficients and `GreekProjector`; Python functions) and Changed: `GaussLegendre` is `#[non_exhaustive]`, the `f64` rule above 100 points is within 16·2⁻⁵² instead of half an ulp.
6. Follow-ups file: sasktran2 comparison of Greek coefficients and conventions; speed comparison with sasktran2; evaluating amplitudes in 1 − μ to remove the node-rounding term; caching projectors across spheres (phase 2).

- [ ] **Step 1:** Write the pages above.
- [ ] **Step 2:** Run `uv run --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html` and `uv run pytest -q tests/test_docs_consistency.py`. Expected: build succeeded (every executed cell's asserts pass), contracts pass.
- [ ] **Step 3: Commit** — `docs: phase matrix and Greek coefficients (theory, example, accuracy, limitations)`

---

### Task 6: Final verification

- [ ] Run the full CONTRIBUTING checklist: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build --workspace --lib --examples`, `cargo test --workspace`, `cargo +1.85.0 check --workspace --lib`, `cargo package -p scatrs --allow-dirty`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`, `uv run python -m mypy.stubtest scatrs._scatrs`, `uv run --directory reference python -m pytest`, `uv run --directory reference python -m scatrs_reference generate` then `git status reference/data` (no changes), the docs build.
- [ ] Final whole-branch review; then finish the branch (usask-foundation:finishing-a-development-branch).
