# Contributing to scatrs

## Setup

```bash
rustup update stable        # CI uses the latest stable Rust (rust-toolchain.toml: "stable")
uv sync                     # Python environment with the dev tools; builds the extension
uv run pre-commit install   # formatting and lint checks on every commit
```

`uv run` rebuilds the extension whenever the Rust sources change, so `uv run pytest` always
tests the current code. Keep stable Rust up to date: CI installs the latest, whose new clippy
lints an older local toolchain does not show.

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
cargo test --workspace
cargo +1.85.0 check --workspace --lib   # the MSRV: rustup toolchain install 1.85.0
cargo package -p scatrs                 # add --allow-dirty while changes are uncommitted
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m mypy.stubtest scatrs._scatrs
uv run --directory reference python -m pytest
uv run --directory reference python -m scatrs_reference generate   # then: git status reference/data
uv run --extra docs sphinx-build -E -W --keep-going -b html docs docs/_build/html
uvx --from actionlint-py==1.7.12.25 actionlint   # after changing .github/workflows
```

The documentation build runs the examples and downloads the Python and numpy inventories; see
`docs/installation.md` for building it offline. It also runs `cargo doc` (warnings are errors)
and publishes the Rust API reference under `rustdoc/` in the site. In the pages, write
``{rust}`scatrs::mie::Mie::compute` `` to link to a Rust item: a path that rustdoc did not
document fails the build.

CI also tests the packaged crate on its own, as `cargo test` on the crates.io sources would:
extract `target/package/scatrs-*.crate` outside the repository and run `cargo test --locked`
there. Without the reference data, the unit tests that read it skip their checks with a
message; `SCATRS_REQUIRE_REFERENCE_DATA=1`, which CI sets everywhere else, makes a missing file
an error.

`scatrs/tests/mie_accuracy_map.rs`, the dense accuracy map of the Mie outputs, takes about
20 s with all cores (it is part of `cargo test`). After a change to the Mie kernel or the
declared tolerances, regenerate the map the documentation plots, and commit it:

```bash
SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/mie-accuracy-map.json \
    cargo test -p scatrs --release --test mie_accuracy_map -- --nocapture
```

`scatrs/tests/mie_greek_accuracy_map.rs` does the same for the phase matrix and the Greek
coefficients, in about 12 s with all cores; regenerate its map the same way:

```bash
SCATRS_ACCURACY_MAP=$PWD/docs/validation/data/greek-accuracy-map.json \
    cargo test -p scatrs --release --test mie_greek_accuracy_map -- --nocapture
```

Wall-clock benchmarks of the Mie kernel: `cargo bench -p scatrs --bench mie`.

The slow `#[ignore]` tests take minutes, so CI runs them weekly rather than on every change:

```bash
cargo test -p scatrs --release --lib --test gauss_legendre --test mie_greek_accuracy_map -- --ignored
```

## Continuous integration

Every action is pinned to a commit SHA, with its version in a comment. Dependabot updates the
actions, `Cargo.lock` and both `uv.lock` files weekly, in one grouped pull request each, and
only to releases at least a week old.

- `.github/workflows/ci.yml`, on every pull request and push to `main`:
  - lint: rustfmt, clippy, the examples, `cargo check` for the big-endian wheel target (s390x;
    32-bit platforms are not supported), the crate package and its tests, ruff and actionlint;
  - the Rust tests on Linux, macOS and Windows, and `cargo check` of the libraries with Rust
    1.85.0 (the MSRV);
  - on each of the three systems, one abi3 wheel built and tested with Python 3.11 and 3.14,
    and on Linux also with Python 3.11 and the oldest supported numpy (1.26);
  - the strict documentation build and the type stubs;
  - the reference generator's tests and a regeneration of every reference file;
  - "All checks passed", which succeeds only if every job above did: the one check to
    require.
- `.github/workflows/release.yml`, on pushes to `main`, tags `v*`, and pull requests that touch
  the packaging (the build configuration, `scatrs-py/`, `python/` or the workflow itself):
  wheels for every platform and the sdist; the wheels are installed and tested on every
  platform with a GitHub runner, and the sdist is built from source and tested on Linux. A tag
  also runs the preflight checks and publishes (see below). On pull requests this workflow is
  advisory, not a required check (it does not run on every pull request), so a pull request
  that breaks a wheel build or test can merge; the run on `main` then catches it.
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

The Mie generators (`mie`, `mie_derivatives`) run their cases in parallel, one process per
CPU; `SCATRS_REFERENCE_WORKERS=1` runs them in one process. The records do not depend on the
number of processes.

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
2. Update `CHANGELOG.md`, commit, merge to `main`, and tag that commit `v` followed by the
   Cargo version, such as `v0.1.0` or `v0.1.0-alpha.1`.
3. Pushing the tag runs the release workflow. Its preflight job checks that the tag is `v` plus
   the Cargo version, that the tagged commit is on `main`, and that `cargo publish --dry-run`
   succeeds. Once the preflight, every build and every wheel test pass, it publishes to PyPI
   (Trusted Publishing, environment `pypi`; versions already there are skipped), then the
   `scatrs` crate (crates.io Trusted Publishing, environment `crates-io`). Each environment
   asks a reviewer to approve.

For the very first crates.io release, see "crates.io" below.

Before the first public release, decide how the x86_64 wheels handle fused multiply-add: built
for the baseline x86-64 target, as now, they make every `f64::mul_add` a library call, and
double-double code about ten times slower (see "x86_64 builds without FMA" in
`docs/plans/phase-0-followups.md`).

## One-time repository setup

Done once, by a maintainer, in the repository settings and on the registries.

- **GitHub Pages.** Settings → Pages → Source: "GitHub Actions". `docs.yml` fails until this is
  set. Pages on a private repository needs a paid plan.
- **Environments** `pypi` and `crates-io` (Settings → Environments). For each, set "Deployment
  branches and tags" to "Selected branches and tags" with the single tag rule `v*` and no
  branches, and add a required reviewer.
- **Tag ruleset.** Settings → Rules → Rulesets: a tag ruleset for `v*` that restricts creation,
  update and deletion to maintainers.
- **PyPI.** Add a pending trusted publisher (Account → Publishing): project `scatrs`, owner
  `usask-arg`, repository `scatrs`, workflow `release.yml`, environment `pypi`. A pending
  publisher does not reserve the name; the first upload does.
- **crates.io.** Trusted Publishing can only be set up for a crate that exists, so the first
  release goes like this:
  1. Push the tag. PyPI publishes; the `crates-io` job then fails, as expected (or its
     reviewer rejects it).
  2. Create a short-lived crates.io API token scoped to publishing new crates, and publish the
     tagged commit by hand: `git switch --detach vX.Y.Z && cargo publish -p scatrs --locked`.
  3. On crates.io, in the crate's settings → Trusted Publishing, add GitHub: owner
     `usask-arg`, repository `scatrs`, workflow `release.yml`, environment `crates-io`.
  4. Revoke the token.

  Alternatively, publish `0.0.1` by hand now (steps 2 to 4), to claim the name before the
  first real release.
- **Scheduled workflows.** GitHub notifies only the person who last edited the `cron` line of
  `scheduled.yml` when a scheduled run fails, and disables scheduled workflows of a public
  repository after 60 days without activity; re-enable them from the Actions tab.
- **Branch protection** for `main`: require the status check "All checks passed" (from
  `ci.yml`), rather than each job.
