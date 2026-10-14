# Installation

## Python package

```bash
pip install scatrs
```

or, with [uv](https://docs.astral.sh/uv/), `uv add scatrs`. scatrs needs Python 3.11 or later
and numpy 1.26 or later. Wheels are published for Linux (x86_64, aarch64, s390x, ppc64le, and
musl on x86_64 and aarch64), macOS (Apple silicon and Intel) and Windows (x86_64 and arm64).
The extension uses the stable ABI (abi3), so one wheel serves every Python from 3.11 on.
Elsewhere pip builds from the source distribution, which needs a Rust toolchain, 1.85 or
later ([rustup](https://rustup.rs) installs one).

```python
from scatrs import mie

q = mie.efficiencies(10.0, 1.33 + 1e-8j)
print(q.q_ext, q.q_sca, q.g)
```

{doc}`quickstart` continues from there.

```{note}
Double-double arithmetic, which scatrs uses for the Gauss–Legendre rules behind the Greek
coefficients, relies on fused multiply-add (FMA). The x86_64 wheels are built for the baseline
x86_64 target, which has no FMA, so each FMA there becomes a software call and double-double
code runs about ten times slower. Where the CPU has FMA, building from source with
`RUSTFLAGS="-C target-cpu=native"` avoids this, for example
`RUSTFLAGS="-C target-cpu=native" pip install --no-binary scatrs scatrs`. This changes the
speed, not the results: `f64::mul_add` is correctly rounded with or without hardware FMA.
aarch64 targets, Apple silicon included, always have FMA.
```

## Rust crate

```bash
cargo add scatrs
```

The crate needs Rust 1.85 or later. Its API reference is on [docs.rs](https://docs.rs/scatrs)
and in these pages ({doc}`rust-api`). The rustdoc is the authoritative statement of each
function's accuracy contract and edge-case behaviour; these pages summarise it.

## From source

A checkout builds both, and is needed to run the tests or build this documentation (they read
the reference data in `reference/data`). With uv:

```bash
git clone https://github.com/usask-arg/scatrs
cd scatrs
uv sync         # environment with the development tools; builds the extension (release)
uv run pytest
cargo test --workspace
```

uv installs scatrs in editable mode and rebuilds the extension whenever the Rust sources
change, so `uv run` always uses the current code. With pip, in a virtual environment,
`python -m pip install ".[test]"` builds the extension in release mode. The repository's
`rust-toolchain.toml` selects the stable channel with rustfmt and clippy.

## Building this documentation

autodoc imports the installed package and the examples run it; `uv run` builds and installs
it first. The examples also read the reference data in `reference/data`, so build from a
checkout:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Every build re-runs the examples and re-reads the API docstrings, even when their sources have
not changed. It also runs `cargo doc` and includes the Rust API reference.

Every warning fails the build, and every cross-reference must resolve (nitpicky mode). The
links to the Python and numpy documentation come from inventories that intersphinx downloads.
Sphinx keeps them in the build directory for five days, so rebuilds soon after an online build
also work offline. The {doc}`changelog` page downloads the release notes from GitHub,
anonymously or with a token from `SPHINX_GITHUB_CHANGELOG_TOKEN`, `GITHUB_TOKEN`, a git
credential helper or the GitHub CLI; if GitHub refuses an anonymous request (its rate limit),
the page shows a notice instead of the notes. Without network access these downloads fail and so
does the build. Set `SCATRS_DOCS_OFFLINE=1` to skip them:

```bash
SCATRS_DOCS_OFFLINE=1 uv run --extra docs \
    sphinx-build -W --keep-going -b html docs docs/_build/html
```

Offline, references to Python and numpy objects appear as plain text, and the changelog page
links to the GitHub releases. References within scatrs are still checked.
