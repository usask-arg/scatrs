# Comparison with sasktran2

scatrs is meant to replace the Mie code of [sasktran2](https://github.com/usask-arg/sasktran2),
whose `MieDatabase` integrates the optical properties of spheres over size distributions for
its radiative transfer. This page compares the two for accuracy and speed on the same machine.
The script `benchmarks/compare.py` produces every number here, and the results it wrote are in
`benchmarks/results/2026-10-13-arm64.json`.

## Method

<!-- synced with benchmarks/compare.py -->

- **sasktran2** 2026.10.1, through `sasktran2.mie.distribution.integrate_mie_cpp` with the
  settings `MieDatabase` uses (31 Gauss points per panel, the size quantile 0.99999) and 64
  coefficients, without the database's netCDF caching. It builds one radius grid, from QUADPACK's
  panels for an extinction integrand at the shortest wavelength, for every wavelength and
  distribution, and projects the phase matrix on a split Gauss–Legendre grid of 128 cosines.
- **scatrs** `ensemble.compute_batch` with the Greek coefficients to order 63 (the same 64
  coefficients) at rtol $3\cdot10^{-3}$, $10^{-3}$, $3\cdot10^{-4}$ and $10^{-4}$.
- **Reference:** scatrs at rtol $10^{-9}$ (converged everywhere).
- **Conventions:** sasktran2 takes $m = n - ik$, and works in nm and m²; its $\beta_1$ and
  $\beta_2$ have the opposite sign to scatrs's, and its $\alpha_1$ to $\alpha_4$ and their
  normalization ($\alpha_1^0 = 1$) are the same ({doc}`/theory/mie`).
- **Errors:** the largest over every distribution and wavelength: relative for
  $C_\text{ext}$, $C_\text{sca}$ and the co-albedo $1 - \omega$, absolute for $\omega$ and the
  normalized Greek coefficients (the tables show $\alpha_1$ and $\beta_1$; $\alpha_2$ to
  $\alpha_4$ and $\beta_2$ are in the results file and behave alike).
- **Matched accuracy:** the loosest rtol at which scatrs's errors of $C_\text{ext}$,
  $C_\text{sca}$ and all six Greek coefficients are at most sasktran2's. $\omega$ and the
  co-albedo are not matched (below).
- **Timing:** the best of 3 calls, on one thread (`RAYON_NUM_THREADS=1` for scatrs,
  `num_threads=1` for sasktran2) and on all threads (sasktran2 `num_threads=0`).

## Results

Apple M4 Pro, 12 logical CPUs, macOS (Darwin 25.6.0); scatrs 0.0.1, sasktran2 2026.10.1;
2026-10-13.

**W1**, one distribution: a lognormal with $r_g = 1$ µm, $\sigma_g = 2$ and
$m = 1.33 + 10^{-4}i$, at $\lambda$ = 0.3, 0.5 and 1 µm (one call).

| | one thread | all threads | $C_\text{ext}$ | $C_\text{sca}$ | $\omega$ | co-albedo | $\alpha_1$ | $\beta_1$ |
|---|---|---|---|---|---|---|---|---|
| sasktran2 | 0.735 s | 0.357 s | 6.8e-05 | 6.7e-05 | 3.5e-05 | 4.6e-03 | 5.1e-02 | 1.4e-03 |
| scatrs rtol 0.003 | 0.153 s | 0.0378 s | 1.5e-03 | 1.6e-03 | 5.4e-05 | 4.4e-03 | 6.3e-02 | 4.8e-02 |
| scatrs rtol 0.001 | 0.34 s | 0.0784 s | 2.4e-04 | 2.3e-04 | 3.6e-05 | 5.7e-03 | 4.5e-03 | 4.0e-03 |
| scatrs rtol 0.0003 (matched) | 0.444 s | 0.0881 s | 6.7e-05 | 5.2e-05 | 3.2e-05 | 4.2e-03 | 1.8e-03 | 6.5e-04 |
| scatrs rtol 0.0001 | 1.06 s | 0.17 s | 3.0e-05 | 1.4e-05 | 1.6e-05 | 3.3e-03 | 9.4e-04 | 2.1e-04 |

**W2**, a database grid: 50 lognormals ($r_g$ at 10 values log-spaced in [0.05, 1] µm,
$\sigma_g$ = 1.2, 1.4, 1.6, 1.8, 2.0), 20 wavelengths in [0.28, 1.02] µm,
$m = 1.43 + 10^{-8}i$.

| | one thread | all threads | $C_\text{ext}$ | $C_\text{sca}$ | $\omega$ | co-albedo | $\alpha_1$ | $\beta_1$ |
|---|---|---|---|---|---|---|---|---|
| sasktran2 | 3.48 s | 0.721 s | 8.5e-04 | 8.5e-04 | 3.4e-07 | 3.5e-01 | 7.1e-02 | 9.0e-03 |
| scatrs rtol 0.003 | 1.47 s | 0.217 s | 8.8e-04 | 8.8e-04 | 9.1e-08 | 1.0e-01 | 2.0e-02 | 1.2e-02 |
| scatrs rtol 0.001 (matched) | 3.16 s | 0.493 s | 4.3e-04 | 4.3e-04 | 4.2e-07 | 6.8e-01 | 5.8e-03 | 3.2e-03 |
| scatrs rtol 0.0003 | 4.94 s | 0.724 s | 1.3e-04 | 1.3e-04 | 8.4e-08 | 8.2e-02 | 2.8e-03 | 9.9e-04 |
| scatrs rtol 0.0001 | 10.5 s | 1.53 s | 7.7e-05 | 7.7e-05 | 9.0e-08 | 8.1e-02 | 1.7e-03 | 4.6e-04 |

At matched accuracy scatrs is faster on both workloads: 1.7 times on one thread and 4.1 times
on all threads for W1, 1.10 and 1.46 times for W2. Its Greek coefficients are then 3 to 12
times more accurate than sasktran2's, whose $\alpha$ errors of 5 to 9% (absolute, normalized)
are consistent with the aliasing of its 128-cosine projection, which is not exact for spheres
of hundreds of terms (main spec, Appendix A), while its cross sections are accurate to
$10^{-3}$. Every scatrs run converged, and its own error estimates are
meant to be relied on ({ref}`ensemble-accuracy`); sasktran2's results come without estimates.

## Notes

- **$\omega$ and the co-albedo.** Both codes integrate $C_\text{sca}$ and $C_\text{ext}$ on
  the same nodes, so their errors largely cancel in $\omega$: sasktran2's $\omega$ errors in W2
  stay below $4\cdot10^{-7}$ by that cancellation, not by resolving the size integral, and
  scatrs's are of the same size. The co-albedo of W2's weak absorbers (about $10^{-6}$) is
  resolved by neither: sasktran2 takes $C_\text{ext} - C_\text{sca}$, and scatrs refines
  $\langle C_\text{abs}\rangle$ only to rtol of $\langle C_\text{ext}\rangle$ ({doc}`/limitations`).
  So neither enters the matched accuracy.
- **What the time buys.** sasktran2's grid is fixed by its settings; scatrs's panels follow the
  requested rtol, so the tables also give its accuracy-against-time curve. scatrs's default rtol,
  $10^{-6}$, is far tighter than these: ask for the accuracy the application needs.
- **Batches.** W2 uses `compute_batch`, which shares the size nodes of the 50 distributions;
  computing them one by one costs about 2.4 times as much on this grid.
- **Reproduce:** `uv run --locked --directory benchmarks python compare.py` (several minutes
  on this machine, mostly the reference).
