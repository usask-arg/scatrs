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
