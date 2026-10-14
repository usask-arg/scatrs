# Comparison with sasktran2

scatrs is built out of the scattering core of the radiative transfer model
[sasktran2](https://github.com/usask-arg/sasktran2), whose own Mie code, through its
`MieDatabase`, integrates the optical properties of spheres over size distributions. It is a
separate implementation of the same averages as {class}`scatrs.Ensemble`, so it serves both as
an external check of scatrs's results and as a benchmark of its speed. This page compares the two for accuracy and speed on the same machine.
The script `benchmarks/compare.py` produces every number here, and the results it wrote are in
`benchmarks/results/2026-10-13-arm64.json`.

## Method

<!-- synced with benchmarks/compare.py -->

- **sasktran2** 2026.10.1, through `sasktran2.mie.distribution.integrate_mie_cpp` with the
  settings `MieDatabase` uses (31 Gauss points per panel, the size quantile 0.99999) and 64
  coefficients, without the database's netCDF caching. It builds one radius grid, from QUADPACK's
  panels for an extinction integrand at the shortest wavelength, for every wavelength and
  distribution, and projects the phase matrix on a split Gauss–Legendre grid of 128 cosines.
- **scatrs** {class}`~scatrs.Ensemble` with the 50 lognormals as one grid (array-valued
  parameters), with the Greek coefficients to order 63 (the same 64
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

| | one thread | all threads | C_ext | C_sca | ω | co-albedo | α₁ | β₁ |
|---|---|---|---|---|---|---|---|---|
| sasktran2 | 0.728 s | 0.359 s | 6.8e-05 | 6.7e-05 | 3.5e-05 | 4.6e-03 | 5.1e-02 | 1.4e-03 |
| scatrs rtol 0.003 | 0.148 s | 0.0386 s | 1.5e-03 | 1.6e-03 | 5.4e-05 | 4.4e-03 | 6.3e-02 | 4.8e-02 |
| scatrs rtol 0.001 | 0.336 s | 0.0753 s | 2.4e-04 | 2.3e-04 | 3.6e-05 | 5.7e-03 | 4.5e-03 | 4.0e-03 |
| scatrs rtol 0.0003 (matched) | 0.438 s | 0.0838 s | 6.7e-05 | 5.2e-05 | 3.2e-05 | 4.2e-03 | 1.8e-03 | 6.5e-04 |
| scatrs rtol 0.0001 | 1.05 s | 0.154 s | 3.0e-05 | 1.4e-05 | 1.6e-05 | 3.3e-03 | 9.4e-04 | 2.1e-04 |

**W2**, a database grid: 50 lognormals ($r_g$ at 10 values log-spaced in [0.05, 1] µm,
$\sigma_g$ = 1.2, 1.4, 1.6, 1.8, 2.0), 20 wavelengths in [0.28, 1.02] µm,
$m = 1.43 + 10^{-8}i$.

| | one thread | all threads | C_ext | C_sca | ω | co-albedo | α₁ | β₁ |
|---|---|---|---|---|---|---|---|---|
| sasktran2 | 3.52 s | 0.711 s | 8.5e-04 | 8.5e-04 | 3.4e-07 | 3.5e-01 | 7.1e-02 | 9.0e-03 |
| scatrs rtol 0.003 | 1.42 s | 0.267 s | 7.8e-04 | 7.8e-04 | 3.8e-07 | 3.9e-01 | 1.8e-02 | 1.5e-02 |
| scatrs rtol 0.001 | 3 s | 0.553 s | 2.8e-04 | 2.8e-04 | 8.2e-08 | 8.3e-02 | 6.0e-03 | 2.5e-03 |
| scatrs rtol 0.0003 (matched) | 4.71 s | 0.83 s | 1.8e-04 | 1.8e-04 | 9.1e-08 | 8.4e-02 | 3.4e-03 | 1.1e-03 |
| scatrs rtol 0.0001 | 10.2 s | 1.83 s | 1.0e-04 | 1.0e-04 | 1.7e-07 | 1.3e-01 | 1.8e-03 | 6.1e-04 |

On W1, at matched accuracy (rtol $3\cdot10^{-4}$), scatrs is 1.7 times faster than sasktran2 on
one thread and 4.3 times on all threads, with Greek coefficients more accurate: the $\alpha$
29 to 33 times, the $\beta$ 1.8 to 2.2 times.

On W2, the matched rtol is $3\cdot10^{-4}$, where scatrs is slower than sasktran2: 4.71 s against
3.52 s on one thread, 0.83 s against 0.711 s on all threads, with its Greek coefficients 21 to 26
times ($\alpha$) and 5 to 8 times ($\beta$) more accurate. At rtol $10^{-3}$ every scatrs error
but one is smaller than sasktran2's (the $\alpha$ 12 to 14 times), and scatrs is 1.17 times
faster on one thread and 1.29 times on all; the exception is $\beta_2$, $6.8\cdot10^{-3}$
against $6.7\cdot10^{-3}$, which decides the strict matching rule.

sasktran2's $\alpha$ errors of 5 to 9% (absolute, normalized) are consistent with the aliasing
of its 128-cosine projection, which is not exact for spheres of hundreds of terms, while its
cross sections are accurate to $10^{-3}$. Since the reference is scatrs at a tight tolerance,
sasktran2's errors are also the differences between the two codes: the two implementations
agree to within sasktran2's own accuracy. Every scatrs run converged, and
its own error estimates are meant to be relied on ({ref}`ensemble-accuracy`); sasktran2's
results come without estimates.

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
- **Grids.** W2 computes the 50 distributions as one grid, which shares their size nodes;
  computing them one by one cost about 2.5 times as much on this grid (measured separately, one
  thread, rtol $10^{-3}$).
- **Reproduce:** `uv run --locked --directory benchmarks python compare.py` (several minutes
  on this machine, mostly the reference).
