# Phase 2c-1 follow-ups

These items came out of phase 2c-1 (the model layer: refractive-index models, `Sphere`,
`Ensemble`, `Mixture` and grids of distributions) and were deliberately deferred. Each belongs
to the phase or task named in its heading.

## API (undecided)
- **Heterogeneous batches.**
  - **What:** 0.1.0's `compute_batch` took any list of distributions: different kinds, or many
    different `Tabulated` tables. A grid broadcasts the numeric parameters of one distribution
    only, so such lists have no replacement; the changelog tells users to compute each
    distribution as its own `Ensemble`, which loses the shared size nodes.
  - **Then:** accept a sequence of distributions as `size=`, mapped to the batch engine
    (`ensemble_compute_batch`) with a leading axis of the results.
  - **Why deferred:** not in 2c-1's scope; for the user to decide whether it is wanted.

## Particles (with `LayeredSphere`)
- **A particle-side hook for engine items.** `Ensemble.compute` assumes one refractive index
  per particle: it builds the engine's item from `particle.refractive_index.evaluate(wl)`. A
  `LayeredSphere` has an index (and a radius fraction) per layer, so the particle should build
  its own engine item, and name its own derivative parameters, through a private method that
  `Ensemble` and `Mixture` call instead.

## Derivatives (phase 2c-2)
- **Derivatives of grids.** `Ensemble.compute` raises `NotImplementedError` for `derivatives`
  of a grid of distributions. The engine's groups already support per-distribution θ blocks;
  the work is the result layout (the grid's axes first, as for the values) and its tests.

## Input checks kept from 0.1.0 (undecided)
- **`Tabulated` flattens and accepts bools.**
  - **What:** `Tabulated` converts its radii and densities with `np.asarray(..., dtype=float)`
    and `ravel()`, so a 2-D table is silently flattened and bools become 0.0 and 1.0, as in
    0.1.0. The numeric parameters reject bools (`TypeError`).
  - **Then:** require one-dimensional real sequences, raising `TypeError` for bools and
    `ValueError` for other shapes.
  - **Why deferred:** a behaviour change from 0.1.0, outside the review fixes.
- **Doubled prefix in a scalar ensemble's engine errors.**
  - **What:** an engine error of a scalar `Ensemble` reads
    "invalid input: component 0: invalid input: …", as in 0.1.0: the engine wraps the error of
    the mixture's single component in a second prefix.
  - **Then:** have the engine (or the binding) drop the component prefix for a single
    component, or the inner "invalid input: ".
  - **Why deferred:** the wording comes from the Rust engine, outside 2c-1's Python scope.
