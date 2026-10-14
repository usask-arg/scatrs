# Phase 3b follow-ups

These items came out of phase 3b (the layered-sphere kernel) and were deliberately deferred. Each belongs to the phase or task named in its heading.

## Derivatives (a later phase, before large coated particles need Jacobians)

**Closed-form derivatives through the recursion.**
- **What:** derivatives by the chain rule through the H recursion, with an adjoint sweep per order, O(L) per order. They would replace the dual numbers for L ≥ 2.
- **Why:** two reasons.
  - Dual numbers differentiate every intermediate quantity, and for thin lossless shells some of those are far more sensitive than the result. The derivatives' validated domain therefore ends at x_L = 100, where the values reach 10⁴.
  - The cost misses (1 + 3L)× slightly at x_L = 100 (7.4× for two layers, 32× for ten).
- **Then:** extend `derivative_x_max`, and re-measure both maps and the cost.

## Domain

**Thin lossless shells of high index.**
- **What:** above |m|·x_L = 3·10⁴, thin real shells of index 10 lose up to 10⁹ ulps in either second solution.
- **Why deferred:** only very large, lossless, high-index shells are affected, and they are outside the declared domain.
- **Then:** investigate the narrow whispering-gallery resonances, and whether the condition estimate captures them.

## Python

**Double-double and dual layered results.** As for spheres, Python exposes `f64` results only.
