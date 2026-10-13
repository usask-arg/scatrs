# Contributing to scatrs

## Setup

```bash
uv sync                     # Python environment with the dev tools; builds the extension
uv run pre-commit install   # formatting and lint checks on every commit
```

`uv run` rebuilds the extension whenever the Rust sources change, so `uv run pytest` always
tests the current code.

Run cargo from inside the repository. `.cargo/config.toml` sets
`PYO3_BUILD_EXTENSION_MODULE=1`, which lets cargo build `scatrs-py` without a suitable Python
interpreter, but cargo reads that file only from the directory it is started in and its
parents. `cargo … --manifest-path path/to/scatrs/Cargo.toml` started elsewhere ignores it, and
pyo3's build script then fails if the Python on `PATH` is older than 3.11.

## Checks (the same ones CI runs)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --lib --examples
cargo package -p scatrs          # add --allow-dirty while changes are uncommitted
cargo test --workspace
cargo +1.85.0 test --workspace   # the MSRV; rustup toolchain install 1.85.0
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m mypy.stubtest scatrs._scatrs
uv run --directory reference python -m pytest
uv run --directory reference python -m scatrs_reference generate   # then: git status reference/data
uv run --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html
uvx --from actionlint-py actionlint   # after changing .github/workflows
```

The documentation build runs the examples and downloads the Python and numpy inventories; see
`docs/installation.md` for building it offline.

The slow `#[ignore]` tests take minutes, so CI runs them weekly rather than on every change:

```bash
cargo test -p scatrs --release -- --ignored
```

## Continuous integration

- `.github/workflows/ci.yml`, on every pull request and push to `main`: the lint checks
  (rustfmt, clippy, ruff, actionlint), the examples and the crate package, the Rust tests on
  Linux, macOS and Windows, the MSRV (Rust 1.85.0), the Python tests with Python 3.11 and 3.14
  on the same three systems, the type stubs, the strict documentation build, and the reference
  generator's tests and a regeneration of every reference file.
- `.github/workflows/release.yml`: wheels for every platform and the sdist, on every pull
  request and push to `main`; a tag `v*` also publishes them (see below).
- `.github/workflows/docs.yml`: publishes the documentation from `main` to GitHub Pages.
- `.github/workflows/scheduled.yml`, every Monday: the `#[ignore]` tests on Linux, macOS and
  Windows, the Rust tests on Intel macOS, and a regeneration of every reference file on Linux
  and macOS that fails on any difference from the committed data.

## Reference data

Arb reference values live in `reference/data/` and are produced by `reference/` (python-flint,
pinned). Regenerate them with
`uv run --directory reference python -m scatrs_reference generate [--only NAME]`
and commit the data together with the manifest. Never edit the data by hand. Tolerances in
tests are never loosened without a written justification in the test.

Regeneration gives the same bytes on every platform, because generators draw their inputs
without the platform's maths library, whose results differ between platforms. Use
`nearest_pow10`, `nearest_cos` and `nearest_sin` from `reference/scatrs_reference/records.py`
instead of `10.0 ** u`, `math.cos` and `math.sin`; `reference/tests/test_portable_inputs.py`
enforces this.

## Releasing

1. Set `[workspace.package] version` in `Cargo.toml`. It is the single version source:
   the Python version and the docs `release` are derived from it. For a pre-release use a
   SemVer form that maturin can convert, `X.Y.Z-alpha.N`, `X.Y.Z-beta.N` or `X.Y.Z-rc.N`:
   the Python package and `scatrs.__version__` get its PEP 440 form (`X.Y.ZaN`, `X.Y.ZbN`,
   `X.Y.ZrcN`), which the tests check.
2. Update `CHANGELOG.md`, commit, and tag `vX.Y.Z` (with the Cargo form of the version, such
   as `v0.1.0-alpha.1`).
3. Pushing the tag builds wheels, publishes to PyPI (Trusted Publishing, environment `pypi`)
   and publishes the `scatrs` crate (crates.io Trusted Publishing, environment `crates-io`).
   The very first crates.io release must be published manually with an API token
   (`cargo publish -p scatrs`), because crates.io can only enable Trusted Publishing for a
   crate that already exists; the `crates-io` job of that first tag then fails, as expected.

Before the first public release, decide how the x86_64 wheels handle fused multiply-add: built
for the baseline x86-64 target, as now, they make every `f64::mul_add` a library call, and
double-double code about ten times slower (see "x86_64 builds without FMA" in
`docs/plans/phase-0-followups.md`).
