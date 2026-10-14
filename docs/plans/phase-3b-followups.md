# Phase 3b follow-ups

These items came out of phase 3b (the layered-sphere kernel) and were deliberately deferred. Each belongs to the phase or task named in its heading.

## Domain

**Thin lossless shells of high index.**
- **What:** above |m|·x_L = 3·10⁴, thin real shells of index 10 lose up to 10⁹ ulps in either second solution.
- **Why deferred:** only very large, lossless, high-index shells are affected, and they are outside the declared domain.
- **Then:** investigate the narrow whispering-gallery resonances, and whether the condition estimate captures them.

**Derivatives above x_L = 1000.**
- **What:** the forward amplitude's derivative with respect to a thin air shell's index reaches C ≈ 1700 at x_L = 10⁴. The recurrence errors at the shell's two surfaces are coherent across orders and add up in the forward sum.
- **Why deferred:** 1000 is ten times the dual numbers' domain, and every other output stays within its constant up to 10⁴.
- **Then:** find which recurrence carries the coherent error (P_n's error is 10⁻¹² to 10⁻⁹ relative), and whether pairing the two surfaces' terms removes it.

## Python

**Double-double and dual layered results.** As for spheres, Python exposes `f64` results only.
