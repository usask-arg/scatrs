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
