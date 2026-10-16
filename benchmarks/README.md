# Benchmarks

`compare.py` compares scatrs's ensembles with sasktran2's Mie integration
(`sasktran2.mie.distribution.integrate_mie_cpp`, which `MieDatabase` calls) for accuracy and speed,
as published in `docs/validation/sasktran2.md`:

```sh
uv run --locked --directory benchmarks python compare.py
```

It writes `results/<date>-<machine>.json` and prints the tables of the documentation. The
reference integrals (scatrs at rtol 10⁻⁹) take a few minutes on all cores. `--quick` runs a
smaller grid to check the script itself.

## Spheroids against Mishchenko's T-matrix code

`compare_tmatd.py` fetches Mishchenko's extended-precision code for randomly oriented particles
(`tmq.lp.f`, `lpq.f`, `tmq.par.f`) from its page at NASA GISS into `tmatd/` (ignored by git: the
code is publicly available but not redistributed here), checks the SHA-256 of each file,
patches it (input from standard input, 16 printed digits, gfortran's names for its
quad-precision intrinsics, the single-particle radii of its notes), builds it with `gfortran`,
and compares it with `scatrs.tmatrix` for 72 spheroids, accuracy and time. Needs `gfortran`;
about 25 minutes:

    uv run --directory benchmarks python compare_tmatd.py
