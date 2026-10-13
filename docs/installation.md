# Installation

Neither wheels nor the Rust crate are published yet, so scatrs is built from source.

## Requirements

- A Rust toolchain, 1.85 or later. The repository's `rust-toolchain.toml` selects the stable
  channel with rustfmt and clippy, so [rustup](https://rustup.rs) installs the right one.
- [uv](https://docs.astral.sh/uv/), which manages the Python environment, or pip.
- Python 3.11 or later. The extension is built for the stable ABI (abi3), so one build serves
  every Python from 3.11 on. The only runtime dependency is numpy 1.26 or later.

## Python package

With uv:

```bash
git clone https://github.com/usask-arg/scatrs
cd scatrs
uv sync         # environment with the development tools; builds the extension (release)
uv run pytest
```

uv installs scatrs in editable mode and rebuilds the extension whenever the Rust sources
change, so `uv run` always uses the current code. (`uv run maturin develop --uv --release`
also works, but is not needed.)

With pip, in a virtual environment:

```bash
python -m pip install ".[test]"   # builds the extension in release mode
python -m pytest
```

Then:

```python
import scatrs

nodes, weights = scatrs.special.gauss_legendre(16)
```

```{note}
Double-double arithmetic, which {func}`~scatrs.special.gauss_legendre` uses internally, relies
on fused multiply-add (FMA). On x86_64 the default target has no FMA, so each FMA becomes a
software call and double-double code runs about ten times slower. Where the CPU has FMA,
build with `RUSTFLAGS="-C target-cpu=native"` (or `-C target-feature=+fma`), for example
`RUSTFLAGS="-C target-cpu=native" uv sync --reinstall-package scatrs`. This changes the speed,
not the results: `f64::mul_add` is correctly rounded with or without hardware FMA. aarch64
targets, Apple silicon included, always have FMA.
```

## Rust crate

Once the crate is published it will be added with `cargo add scatrs`. Until then, use a path
dependency on the `scatrs` directory of a checkout:

```toml
[dependencies]
scatrs = { path = "path/to/checkout/scatrs" }
```

The Rust API reference will be on [docs.rs](https://docs.rs/scatrs) once the crate is
published (it is not yet). Until then, build it locally with:

```bash
cargo doc -p scatrs --no-deps --open
```

The rustdoc is the authoritative statement of each function's accuracy contract and edge-case
behaviour. These pages summarise it.

## Building this documentation

autodoc imports the installed package and the examples run it; `uv run` builds and installs
it first. The examples also read the reference data in `reference/data`, so build from a
checkout:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Every build re-runs the examples and re-reads the API docstrings, even when their sources have
not changed.

Every warning fails the build, and every cross-reference must resolve (nitpicky mode). The
links to the Python and numpy documentation come from inventories that intersphinx downloads.
Sphinx keeps them in the build directory for five days, so rebuilds soon after an online build
also work offline. Otherwise, without network access, the download fails and so does the
build. Set `SCATRS_DOCS_OFFLINE=1` to skip the download:

```bash
SCATRS_DOCS_OFFLINE=1 uv run --extra docs \
    sphinx-build -W --keep-going -b html docs docs/_build/html
```

Offline, references to Python and numpy objects appear as plain text. References within
scatrs are still checked.
