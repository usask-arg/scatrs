# scatrs Phase 0 (Foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use usask-foundation:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scatrs workspace with generic scalar types (`f64`, `DoubleDouble`, `Dual`), the first shared special functions (Gauss–Legendre quadrature, Wigner d-functions), an Arb reference-value generator, a minimal Python package, docs, and CI.

**Architecture:**
- **Rust workspace.** The core crate `scatrs` is pure Rust. The `scatrs-py` crate is a pyo3 extension that maturin installs as `scatrs._scatrs`, and the Python sources live in `python/scatrs/`.
- **Generic kernels.** Every numerical kernel is generic over a `Real` trait, which implies `num_traits::Float`. `DoubleDouble` and `Dual<T, N>` implement it, so `num_complex::Complex<T>` works for every `T: Real`.
- **Reference data.** Ground-truth values are produced by an independent Python generator (`reference/`) using python-flint (Arb ball arithmetic). They are committed as JSON Lines and read by the Rust tests.

**Tech Stack:**
- Rust 1.85+ (edition 2024), num-traits 0.2.19, num-complex 0.4.6, thiserror 2.
- proptest, serde_json (feature `float_roundtrip`).
- pyo3 0.29, numpy (crate) 0.29, maturin ≥ 1.8.
- Python ≥ 3.11, numpy ≥ 1.26, python-flint 0.9.0, pytest.
- Sphinx + furo + MyST-NB.
- GitHub Actions.

**Spec:** `docs/specs/2026-10-12-scatrs-design.md` (sections 4, 8, 9.4, 12 and 13 apply to this phase).

**Deferred to later phases:**

| Deferred item | Phase | Reason |
|---|---|---|
| `rayon` dependency | 1 | No parallel code exists yet. |
| Criterion and instruction-count benchmarks | 1 | No hot kernels exist yet. |
| `cargo-fuzz` targets | 1 | The first user-facing numerical kernels arrive in phase 1. |
| Gauss–Kronrod rules | 2 | Used only by size integration. |
| Riccati–Bessel functions and logarithmic derivatives | 1 | Their stability analysis belongs with Mie. |
| Full Wigner D rotations | 4 | Needed only by the T-matrix. |

## Global Constraints

These apply to every task.

**Rust**
- Edition `2024`; `rust-version = "1.85"` (MSRV).
- The core crate depends only on `num-complex`, `num-traits` and `thiserror`. `rayon` arrives in phase 1. Dev-dependencies are unrestricted.
- The core crate declares `#![forbid(unsafe_code)]` and `#![warn(missing_docs)]`.
- No `ndarray` or `pyo3` types appear in the core crate's public API.
- No input to a public library function may cause a panic or a silent NaN. Invalid inputs return `Error::InvalidInput`; failed convergence returns `Error::NotConverged { achieved, requested }`. Tests may panic.
- Results are bitwise deterministic.
- `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must pass at every commit.

**Python**
- Python ≥ 3.11; wheels are `abi3-py311`; numpy ≥ 1.26; pyo3 and numpy crates 0.29.
- Python exceptions: `ScatrsError` is the base class, with `InvalidInputError(ScatrsError, ValueError)`, `NotConvergedError(ScatrsError, ArithmeticError)` (attributes `achieved`, `requested`) and `OutsideValidatedDomainError(ScatrsError, ValueError)`.

**Naming and versions**
- Rust crate `scatrs`, PyPI package `scatrs`, import name `scatrs`, extension module `scatrs._scatrs`, repository `usask-arg/scatrs`, license MIT.
- The single version source is `[workspace.package] version` in the root `Cargo.toml`. Python reads it through maturin (`dynamic = ["version"]`).

**Reference data** (`reference/data/*.jsonl`)
- One JSON object per line, with keys `generator`, `case`, `label`, `inputs`, `outputs`.
- Each output is a set of parallel arrays: `hi`, `lo` (doubles), `dec` (25 significant digits) and `err` (a rigorous bound on |true − (hi + lo)|).
- `label` is `certified` or `high-precision`.
- `manifest.json` records the python-flint and FLINT versions, the precision and the SHA-256 of each file.
- Total committed data must stay at or below 20 MB.

**Tolerances**
- A test tolerance may never be loosened without a written justification in the test comment.
- Accuracy tests print their measured worst error with `eprintln!`. Implementers report those numbers in their task summary.

---

**Clippy contingencies.** Planning checked the code against the default clippy lints. If clippy still flags an intended formula — `suspicious_arithmetic_impl` (for example `%` defined with `-`, `*` and `/`) or `needless_range_loop` in tests — add a targeted `#[allow(...)]` with a one-line reason. Never use a crate-wide allow.

## File Map

| File | Responsibility | Task |
|---|---|---|
| `Cargo.toml` | workspace, shared package metadata, profiles | 1, 11 |
| `rust-toolchain.toml`, `rustfmt.toml`, `.gitignore`, `LICENSE`, `README.md`, `CHANGELOG.md` | repo basics | 1 |
| `scatrs/Cargo.toml` | core crate manifest | 1 |
| `scatrs/src/lib.rs` | crate root, re-exports | 1, 9 |
| `scatrs/src/error.rs` | `Error`, `Result` | 1 |
| `scatrs/src/scalar/mod.rs` | `Real` trait, `f64` impl, re-exports | 1, 3, 7 |
| `scatrs/src/scalar/dd/mod.rs` | `DoubleDouble` type, error-free transforms, arithmetic, rounding | 3 |
| `scatrs/src/scalar/dd/consts.rs` | DD constant tables (checked against Arb) | 3 |
| `scatrs/src/scalar/dd/elementary.rs` | sqrt/exp/log family | 4 |
| `scatrs/src/scalar/dd/trig.rs` | trig family, argument reduction | 5 |
| `scatrs/src/scalar/dd/parse.rs` | decimal parsing | 6 |
| `scatrs/src/scalar/dd/traits.rs` | num-traits + `Real` impls | 6 |
| `scatrs/src/scalar/dual/mod.rs` | `Dual<T, N>` type and arithmetic | 7 |
| `scatrs/src/scalar/dual/traits.rs` | num-traits + `Real` impls for `Dual` | 8 |
| `scatrs/src/special/mod.rs` | special-function module root | 9 |
| `scatrs/src/special/gauss_legendre.rs` | Gauss–Legendre rule | 9 |
| `scatrs/src/special/wigner.rs` | Wigner d-functions | 10 |
| `scatrs/tests/common/mod.rs` | reference-data loader for integration tests | 2 |
| `scatrs/tests/*.rs` | integration tests (one file per topic) | 1–10 |
| `.gitattributes` | force LF on reference data | 2 |
| `reference/pyproject.toml`, `reference/README.md` | generator project | 2 |
| `reference/scatrs_reference/records.py` | certified DD splits, JSON Lines and manifest I/O | 2 |
| `reference/scatrs_reference/__main__.py` | CLI | 2 |
| `reference/scatrs_reference/generators/*.py` | one module per reference file | 2–6, 9, 10 |
| `reference/tests/*.py` | generator tests | 2 |
| `reference/data/*` | committed reference data and manifest | 2–6, 9, 10 |
| `pyproject.toml`, `.python-version` | Python package (maturin) | 11 |
| `scatrs-py/` | pyo3 extension | 11 |
| `python/scatrs/` | Python package sources and stubs | 11 |
| `tests/*.py` | Python package tests | 11 |
| `docs/` | Sphinx site (`docs/specs`, `docs/plans` excluded from the build) | 12 |
| `.github/workflows/*.yml`, `.github/dependabot.yml`, `.pre-commit-config.yaml`, `CONTRIBUTING.md` | CI, release, tooling | 13 |

---

### Task 1: Workspace, `Real` trait, and error type

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`, `.gitignore`, `LICENSE`, `README.md`, `CHANGELOG.md`
- Create: `scatrs/Cargo.toml`, `scatrs/src/lib.rs`, `scatrs/src/error.rs`, `scatrs/src/scalar/mod.rs`
- Test: `scatrs/tests/real_f64.rs` (the error tests are unit tests inside `error.rs`)

**Interfaces:**
- Produces:
  - `scatrs::Error`, an enum with variants `InvalidInput(String)`, `NotConverged { achieved: f64, requested: f64 }` and `OutsideValidatedDomain(String)`. It is `#[non_exhaustive]` and derives `Debug, Clone, PartialEq, thiserror::Error`.
  - `scatrs::Result<T>`.
  - `scatrs::Real`, a trait with supertraits `num_traits::Float + num_traits::FloatConst + Debug + Send + Sync + 'static + AddAssign + SubAssign + MulAssign + DivAssign + Add<f64, Output = Self> + Sub<f64, Output = Self> + Mul<f64, Output = Self> + Div<f64, Output = Self>`, and these methods:
    - `fn from_f64(x: f64) -> Self`
    - `fn from_f64_pair(hi: f64, lo: f64) -> Self`
    - `fn approx_f64(self) -> f64`
  - `impl Real for f64`.
  - `scatrs::scalar` is a public module; `Real` is re-exported at the crate root.

- [ ] **Step 1: Create the repository basics**

`Cargo.toml`:
```toml
[workspace]
members = ["scatrs"]
resolver = "3"

[workspace.package]
version = "0.0.1"
edition = "2024"
rust-version = "1.85"
license = "MIT"
repository = "https://github.com/usask-arg/scatrs"

[workspace.dependencies]
num-complex = "0.4.6"
num-traits = "0.2.19"
thiserror = "2"

[profile.release]
lto = "fat"
codegen-units = 1

# Double-double reference tests are slow without optimisation; keep debug assertions on.
[profile.test]
opt-level = 2
```

`rust-toolchain.toml`:
```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

`rustfmt.toml`:
```toml
edition = "2024"
```

`.gitignore`:
```
/target/
**/__pycache__/
*.so
*.pyd
.venv/
/dist/
/docs/_build/
/docs/jupyter_execute/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.egg-info/
.DS_Store
```

`LICENSE`: the standard MIT license text with the line `Copyright (c) 2026 scatrs contributors`.

`README.md`:
```markdown
# scatrs

Fast, linearized, rigorously validated single-scattering calculations: Lorenz–Mie, layered
spheres and axisymmetric T-matrix, with a Python interface. The name reads as both
"scatterers" and "scat-rs".

**Status:** pre-alpha (phase 0: numerical foundations). Not yet usable for scattering
calculations.

- Design: [`docs/specs/2026-10-12-scatrs-design.md`](docs/specs/2026-10-12-scatrs-design.md)
- Development: see [`CONTRIBUTING.md`](CONTRIBUTING.md)

License: MIT.
```

`CHANGELOG.md`:
```markdown
# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Workspace, `Real` scalar trait and error type.
```

- [ ] **Step 2: Create the core crate manifest**

`scatrs/Cargo.toml`:
```toml
[package]
name = "scatrs"
description = "Fast, linearized, rigorously validated single-scattering calculations (Mie, layered spheres, T-matrix)"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
readme = "../README.md"
keywords = ["scattering", "mie", "t-matrix", "optics", "atmosphere"]
categories = ["science", "mathematics"]

[dependencies]
num-complex.workspace = true
num-traits.workspace = true
thiserror.workspace = true

[dev-dependencies]
proptest = "1"
serde = { version = "1", features = ["derive"] }
# float_roundtrip makes f64 parsing correctly rounded; without it, reference values can be 1 ulp off.
serde_json = { version = "1", features = ["float_roundtrip"] }
```

- [ ] **Step 3: Write the failing tests**

`scatrs/tests/real_f64.rs`:
```rust
use num_complex::Complex;
use scatrs::Real;

fn horner<T: Real>(coeffs: &[f64], x: T) -> T {
    coeffs.iter().rev().fold(T::zero(), |acc, &c| acc * x + c)
}

fn abs2_of_inverse<T: Real>(z: Complex<T>) -> T {
    (Complex::new(T::one(), T::zero()) / z).norm_sqr()
}

#[test]
fn f64_supports_generic_kernels() {
    assert_eq!(horner(&[1.0, 2.0, 3.0], 2.0_f64), 17.0);
}

#[test]
fn complex_over_real_works() {
    let v = abs2_of_inverse(Complex::new(3.0_f64, 4.0));
    assert!((v - 1.0 / 25.0).abs() < 1e-17);
}

#[test]
fn from_f64_pair_rounds_once_for_f64() {
    assert_eq!(<f64 as Real>::from_f64_pair(1.0, 2f64.powi(-60)), 1.0);
    assert_eq!(<f64 as Real>::from_f64_pair(1.0, 0.5), 1.5);
}

#[test]
fn approx_f64_is_identity_for_f64() {
    assert_eq!(Real::approx_f64(0.1_f64), 0.1);
    assert_eq!(<f64 as Real>::from_f64(0.1), 0.1);
}
```

`scatrs/src/error.rs`. Write the tests first, then the type below them in Step 5:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_informative() {
        assert_eq!(
            Error::InvalidInput("n must be >= 1".into()).to_string(),
            "invalid input: n must be >= 1"
        );
        assert_eq!(
            Error::NotConverged { achieved: 1e-3, requested: 1e-6 }.to_string(),
            "not converged: achieved 1e-3, requested 1e-6"
        );
        assert_eq!(
            Error::OutsideValidatedDomain("x > 2e4".into()).to_string(),
            "outside validated domain: x > 2e4"
        );
    }
}
```

- [ ] **Step 4: Run the tests and confirm they fail**

Run: `cargo test -p scatrs`
Expected: a compile error, because `scatrs/src/lib.rs` does not exist yet.

- [ ] **Step 5: Implement**

`scatrs/src/lib.rs`:
```rust
//! Fast, linearized, rigorously validated single-scattering calculations.
//!
//! The design is described in `docs/specs/2026-10-12-scatrs-design.md` in the repository.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod scalar;

pub use error::{Error, Result};
pub use scalar::Real;
```

`scatrs/src/error.rs` (above the test module from Step 3):
```rust
//! Error type shared by every scatrs computation.

/// Errors returned by scatrs computations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An input is outside its mathematically valid range (for example `|x| > 1` for `x = cos θ`).
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// An iterative or truncated computation did not reach the requested accuracy.
    #[error("not converged: achieved {achieved:e}, requested {requested:e}")]
    NotConverged {
        /// Accuracy actually reached (method-specific measure).
        achieved: f64,
        /// Accuracy that was requested.
        requested: f64,
    },
    /// The inputs are valid but outside the domain where accuracy has been validated.
    #[error("outside validated domain: {0}")]
    OutsideValidatedDomain(String),
}

/// Result alias using [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
```

`scatrs/src/scalar/mod.rs`:
```rust
//! Scalar types that every numerical kernel is generic over.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{Float, FloatConst};

/// A real scalar type for generic kernels.
///
/// Implemented for `f64`, `DoubleDouble` (about 32 significant digits) and `Dual<T, N>`
/// (forward-mode derivatives). The `num_traits::Float` supertrait makes
/// `num_complex::Complex<T>` usable for every `T: Real`.
pub trait Real:
    Float
    + FloatConst
    + Debug
    + Send
    + Sync
    + 'static
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// Converts an `f64` exactly.
    fn from_f64(x: f64) -> Self;

    /// Builds the value of the unevaluated sum `hi + lo`.
    ///
    /// Exact for `DoubleDouble`; for `f64` the sum is rounded once.
    fn from_f64_pair(hi: f64, lo: f64) -> Self;

    /// The nearest `f64` to the value, ignoring any derivative parts.
    fn approx_f64(self) -> f64;
}

impl Real for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }

    #[inline]
    fn from_f64_pair(hi: f64, lo: f64) -> Self {
        hi + lo
    }

    #[inline]
    fn approx_f64(self) -> f64 {
        self
    }
}
```

- [ ] **Step 6: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: 5 tests pass (4 in `real_f64.rs`, 1 in `error.rs`); fmt and clippy are clean.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml .gitignore LICENSE README.md CHANGELOG.md scatrs/
git commit -m "feat: workspace, Real scalar trait and error type"
```

---

### Task 2: Arb reference generator skeleton, constants, and Rust loader

**Files:**
- Create: `.gitattributes`, `reference/pyproject.toml`, `reference/README.md`
- Create: `reference/scatrs_reference/__init__.py`, `reference/scatrs_reference/__main__.py`, `reference/scatrs_reference/records.py`
- Create: `reference/scatrs_reference/generators/__init__.py`, `reference/scatrs_reference/generators/constants.py`
- Create: `reference/tests/test_records.py`, `reference/tests/test_generators.py`
- Generate: `reference/data/constants.jsonl`, `reference/data/manifest.json`
- Create: `scatrs/tests/common/mod.rs`, `scatrs/tests/reference_constants.rs`

**Interfaces:**
- Produces (Python):
  - `records.split(v: arb) -> tuple[float, float, float]`
  - `records.nearest_double(v) -> float`
  - `records.upper_double(x) -> float`
  - `records.values(vs) -> dict`
  - `records.record(generator, case, inputs, outputs, label="certified") -> dict`
  - `records.write_jsonl(path, records) -> str` (returns the SHA-256)
  - `records.update_manifest(data_dir, filename, generator, precision_bits, n_records, sha256)`
  - `generators.GENERATORS: dict[str, Callable[[], tuple[int, list[dict]]]]`
- CLI: `uv run --directory reference python -m scatrs_reference generate [--only NAME ...]`
- Produces (Rust tests): module `common` (each test file includes it with `mod common;`) providing:
  - `load(file: &str) -> Vec<Record>`
  - `find(&[Record], case) -> &Record`
  - `Record` fields `generator`, `case`, `label`, `inputs`, `outputs`
  - `Record` methods `output(name) -> &RefValues`, `input_f64`, `input_i64`, `input_f64s`, `input_i64s`, `input_strs`
  - `RefValues { hi: Vec<f64>, lo: Vec<f64>, dec: Vec<String>, err: Vec<f64> }`
- Reference cases in `constants.jsonl` (each has one output, `value`):
  - `PI`, `TAU`, `FRAC_PI_2`, `FRAC_PI_3`, `FRAC_PI_4`, `FRAC_PI_6`, `FRAC_PI_8`, `FRAC_3PI_4`;
  - `FRAC_1_PI`, `FRAC_2_PI`, `FRAC_2_SQRT_PI`, `SQRT_2`, `FRAC_1_SQRT_2`, `E`;
  - `LN_2`, `LN_10`, `LOG2_E`, `LOG10_E`, `LOG10_2`, `LOG2_10`;
  - `INV_FACT_0` … `INV_FACT_30`;
  - `POW10_2POW0` … `POW10_2POW8` (10^(2^i));
  - `FRAC_PI_2_TAIL` (π/2 − hi − lo) and `LN_2_TAIL` (ln 2 − hi − lo).

- [ ] **Step 1: Create the project files**

`.gitattributes`:
```
reference/data/** text eol=lf
```

`reference/pyproject.toml`:
```toml
[project]
name = "scatrs-reference"
version = "0.1.0"
description = "Arb (python-flint) reference-value generator for scatrs validation"
requires-python = ">=3.11"
# Pinned exactly: regenerated data must be byte-identical for the scheduled drift check.
dependencies = ["python-flint==0.9.0"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
package = false
```

`reference/README.md`:
```markdown
# scatrs reference generator

Independent ground truth for scatrs tests, computed with Arb ball arithmetic through
[python-flint](https://python-flint.readthedocs.io). It does not import scatrs.

    uv run --directory reference python -m scatrs_reference generate            # all files
    uv run --directory reference python -m scatrs_reference generate --only constants
    uv run --directory reference python -m pytest

Each value is stored as `hi`/`lo` doubles, a 25-digit decimal `dec`, and a rigorous error
bound `err` on |true - (hi + lo)|. `label` is `certified` when every source of error
(including series truncation) is rigorously bounded, otherwise `high-precision`.
```

`reference/scatrs_reference/__init__.py`:
```python
"""Arb reference-value generator for scatrs."""
```

- [ ] **Step 2: Write the failing generator tests**

`reference/tests/test_records.py`:
```python
import json
import math

import pytest
from flint import arb, ctx

from scatrs_reference.records import (
    nearest_double,
    record,
    split,
    update_manifest,
    upper_double,
    write_jsonl,
)


@pytest.fixture(autouse=True)
def _precision():
    ctx.prec = 400


def test_split_is_normalised_and_certified():
    v = arb.pi()
    hi, lo, err = split(v)
    assert hi == math.pi
    assert hi + lo == hi
    assert err <= 2.0**-105 * abs(hi)
    assert abs(v - arb(hi) - arb(lo)) <= arb(err)


def test_split_of_a_double_is_exact():
    assert split(arb(0.1)) == (0.1, 0.0, 0.0)


def test_nearest_double_and_upper_double():
    third = arb(1) / 3
    assert nearest_double(third) == 1 / 3
    u = upper_double(third)
    assert arb(u) >= third
    assert u == math.nextafter(1 / 3, math.inf)


def test_record_rejects_unknown_label():
    with pytest.raises(ValueError):
        record("g", "c", {}, {}, label="bogus")


def test_record_layout():
    r = record("g", "c", {"n": 3}, {"value": [arb(1) / 3, arb(2)]})
    assert r["label"] == "certified"
    v = r["outputs"]["value"]
    assert set(v) == {"hi", "lo", "dec", "err"}
    assert v["hi"] == [1 / 3, 2.0]
    assert v["dec"][1].startswith("2")


def test_write_jsonl_is_deterministic_and_updates_manifest(tmp_path):
    recs = [record("g", "c", {}, {"value": [arb(1) / 7]})]
    sha1 = write_jsonl(tmp_path / "g.jsonl", recs)
    sha2 = write_jsonl(tmp_path / "g.jsonl", recs)
    assert sha1 == sha2
    assert (tmp_path / "g.jsonl").read_bytes().count(b"\r") == 0
    update_manifest(tmp_path, "g.jsonl", "g", 400, 1, sha1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["files"]["g.jsonl"] == {
        "generator": "g",
        "precision_bits": 400,
        "records": 1,
        "sha256": sha1,
    }
    assert manifest["python_flint"] == "0.9.0"
```

`reference/tests/test_generators.py`:
```python
import hashlib
import json
from pathlib import Path

import pytest

from scatrs_reference.generators import GENERATORS, constants
from scatrs_reference.records import write_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_committed_file_matches_manifest(name):
    manifest = json.loads((DATA / "manifest.json").read_text())
    entry = manifest["files"][f"{name}.jsonl"]
    content = (DATA / f"{name}.jsonl").read_bytes()
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    lines = content.decode().splitlines()
    assert len(lines) == entry["records"]
    for line in lines:
        rec = json.loads(line)
        assert rec["label"] in ("certified", "high-precision")
        for out in rec["outputs"].values():
            assert len(out["hi"]) == len(out["lo"]) == len(out["dec"]) == len(out["err"])


def test_constants_regenerate_identically(tmp_path):
    _, recs = constants.generate()
    sha = write_jsonl(tmp_path / "constants.jsonl", recs)
    assert (tmp_path / "constants.jsonl").read_bytes() == (DATA / "constants.jsonl").read_bytes()
    assert sha == json.loads((DATA / "manifest.json").read_text())["files"]["constants.jsonl"]["sha256"]


def test_constants_values():
    _, recs = constants.generate()
    by_case = {r["case"]: r["outputs"]["value"] for r in recs}
    assert by_case["PI"]["hi"] == [3.141592653589793]
    assert by_case["INV_FACT_3"]["hi"] == [1 / 6]
    assert by_case["FRAC_PI_2_TAIL"]["hi"] == [-1.4973849048591698e-33]
    assert by_case["LN_2_TAIL"]["hi"] == [5.707708438416212e-34]
    assert by_case["POW10_2POW5"]["lo"] == [-5366162204393472.0]
```

- [ ] **Step 3: Run the generator tests and confirm they fail**

Run: `uv run --directory reference python -m pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scatrs_reference.records'`.

- [ ] **Step 4: Implement the generator**

`reference/scatrs_reference/records.py`:
```python
"""Reference records: certified double-double splits of Arb balls, and JSON Lines I/O.

Every stored value v is represented three ways:

* ``hi``/``lo``: two doubles with ``hi == fl(hi + lo)`` whose exact sum approximates v;
* ``dec``: v to 25 significant digits, for humans (tests use ``hi``/``lo``);
* ``err``: a rigorous upper bound on ``|v - (hi + lo)|``, including the Arb ball radius.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import flint
from flint import arb

FORMAT_VERSION = 1
DECIMAL_DIGITS = 25
LABELS = ("certified", "high-precision")


def nearest_double(v: arb) -> float:
    """The double nearest to the midpoint of ``v``."""
    mid = v.mid()
    f = float(mid)
    for cand in (math.nextafter(f, -math.inf), math.nextafter(f, math.inf)):
        if abs(mid - arb(cand)) < abs(mid - arb(f)):
            f = cand
    return f


def upper_double(x: arb) -> float:
    """The smallest double certainly >= every point of ``|x|``."""
    u = x.abs_upper()
    f = float(u)
    while not arb(f) >= u:
        f = math.nextafter(f, math.inf)
    return f


def split(v: arb) -> tuple[float, float, float]:
    """Split ``v`` into ``(hi, lo, err)`` with ``|v - (hi + lo)| <= err`` rigorously."""
    hi = nearest_double(v)
    lo = nearest_double(v - arb(hi))
    s = hi + lo
    if s != hi:  # renormalise (Fast2Sum) so that hi == fl(hi + lo)
        lo = lo - (s - hi)
        hi = s
    err = upper_double(v - arb(hi) - arb(lo))
    return hi, lo, err


def values(vs: Iterable[arb]) -> dict[str, list]:
    """Encode Arb balls as the parallel arrays stored in a record."""
    out: dict[str, list] = {"hi": [], "lo": [], "dec": [], "err": []}
    for v in vs:
        hi, lo, err = split(v)
        out["hi"].append(hi)
        out["lo"].append(lo)
        out["dec"].append(v.str(DECIMAL_DIGITS, radius=False))
        out["err"].append(err)
    return out


def record(
    generator: str,
    case: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, Iterable[arb]],
    label: str = "certified",
) -> dict:
    """Build one reference record."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
    return {
        "generator": generator,
        "case": case,
        "label": label,
        "inputs": dict(inputs),
        "outputs": {name: values(vs) for name, vs in outputs.items()},
    }


def write_jsonl(path: Path, records: Sequence[dict]) -> str:
    """Write records one per line with sorted keys and LF endings; return the SHA-256."""
    text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records)
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def update_manifest(
    data_dir: Path,
    filename: str,
    generator: str,
    precision_bits: int,
    n_records: int,
    sha256: str,
) -> None:
    """Insert or replace the manifest entry for one data file."""
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["format_version"] = FORMAT_VERSION
    manifest["python_flint"] = flint.__version__
    manifest["flint"] = flint.__FLINT_VERSION__
    files = manifest.get("files", {})
    files[filename] = {
        "generator": generator,
        "precision_bits": precision_bits,
        "records": n_records,
        "sha256": sha256,
    }
    manifest["files"] = dict(sorted(files.items()))
    path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
```

`reference/scatrs_reference/generators/constants.py`:
```python
"""Mathematical constants used by DoubleDouble (FloatConst, reduction and series tables)."""

from __future__ import annotations

import math

from flint import arb, ctx, fmpz

from scatrs_reference.records import record, split

PRECISION_BITS = 400


def _constants() -> dict[str, arb]:
    pi, two, ten = arb.pi(), arb(2), arb(10)
    c = {
        "PI": pi,
        "TAU": 2 * pi,
        "FRAC_PI_2": pi / 2,
        "FRAC_PI_3": pi / 3,
        "FRAC_PI_4": pi / 4,
        "FRAC_PI_6": pi / 6,
        "FRAC_PI_8": pi / 8,
        "FRAC_3PI_4": 3 * pi / 4,
        "FRAC_1_PI": 1 / pi,
        "FRAC_2_PI": 2 / pi,
        "FRAC_2_SQRT_PI": 2 / pi.sqrt(),
        "SQRT_2": two.sqrt(),
        "FRAC_1_SQRT_2": 1 / two.sqrt(),
        "E": arb(1).exp(),
        "LN_2": two.log(),
        "LN_10": ten.log(),
        "LOG2_E": 1 / two.log(),
        "LOG10_E": 1 / ten.log(),
        "LOG10_2": two.log() / ten.log(),
        "LOG2_10": ten.log() / two.log(),
    }
    for k in range(31):
        c[f"INV_FACT_{k}"] = 1 / arb(fmpz(math.factorial(k)))
    for i in range(9):
        c[f"POW10_2POW{i}"] = arb(fmpz(10)) ** (2**i)
    for name, base in (("FRAC_PI_2_TAIL", pi / 2), ("LN_2_TAIL", two.log())):
        hi, lo, _ = split(base)
        c[name] = base - arb(hi) - arb(lo)
    return c


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    recs = [record("constants", name, {}, {"value": [v]}) for name, v in _constants().items()]
    return PRECISION_BITS, recs
```

`reference/scatrs_reference/generators/__init__.py`:
```python
"""Registry of reference generators. Each returns ``(precision_bits, records)``."""

from collections.abc import Callable

from scatrs_reference.generators import constants

Generator = Callable[[], tuple[int, list[dict]]]

GENERATORS: dict[str, Generator] = {
    "constants": constants.generate,
}
```

`reference/scatrs_reference/__main__.py`:
```python
"""Command line: ``python -m scatrs_reference generate [--out DIR] [--only NAME ...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scatrs_reference.generators import GENERATORS
from scatrs_reference.records import update_manifest, write_jsonl

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scatrs_reference")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="(re)generate reference data files")
    gen.add_argument("--out", type=Path, default=DEFAULT_OUT)
    gen.add_argument("--only", nargs="+", choices=sorted(GENERATORS))
    args = parser.parse_args(argv)
    for name in args.only or sorted(GENERATORS):
        precision_bits, records = GENERATORS[name]()
        filename = f"{name}.jsonl"
        digest = write_jsonl(args.out / filename, records)
        update_manifest(args.out, filename, name, precision_bits, len(records), digest)
        print(f"{filename}: {len(records)} records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Generate the data and run the generator tests**

Run: `uv run --directory reference python -m scatrs_reference generate && uv run --directory reference python -m pytest -q`
Expected:
- `reference/data/constants.jsonl` (62 records) and `reference/data/manifest.json` exist.
- All tests pass.

If `test_constants_values` fails on a value, stop and report it. The expected values were computed independently with Arb on 2026-10-12.

- [ ] **Step 6: Write the failing Rust loader test**

`scatrs/tests/reference_constants.rs`:
```rust
mod common;

use common::{find, load};

#[test]
fn std_f64_constants_equal_reference_hi_words() {
    use std::f64::consts as c;
    let recs = load("constants.jsonl");
    let pairs = [
        ("PI", c::PI),
        ("TAU", c::TAU),
        ("FRAC_PI_2", c::FRAC_PI_2),
        ("FRAC_PI_3", c::FRAC_PI_3),
        ("FRAC_PI_4", c::FRAC_PI_4),
        ("FRAC_PI_6", c::FRAC_PI_6),
        ("FRAC_PI_8", c::FRAC_PI_8),
        ("FRAC_1_PI", c::FRAC_1_PI),
        ("FRAC_2_PI", c::FRAC_2_PI),
        ("FRAC_2_SQRT_PI", c::FRAC_2_SQRT_PI),
        ("SQRT_2", c::SQRT_2),
        ("FRAC_1_SQRT_2", c::FRAC_1_SQRT_2),
        ("E", c::E),
        ("LN_2", c::LN_2),
        ("LN_10", c::LN_10),
        ("LOG2_E", c::LOG2_E),
        ("LOG10_E", c::LOG10_E),
        ("LOG10_2", c::LOG10_2),
        ("LOG2_10", c::LOG2_10),
    ];
    for (name, value) in pairs {
        let r = find(&recs, name);
        assert_eq!(r.label, "certified", "{name}");
        assert_eq!(r.output("value").hi[0], value, "{name}");
    }
}

#[test]
fn every_constant_is_known_to_double_double_accuracy() {
    for r in load("constants.jsonl") {
        let v = r.output("value");
        assert!(
            v.err[0] <= 2f64.powi(-105) * v.hi[0].abs(),
            "{}: err {:e} too large",
            r.case,
            v.err[0]
        );
    }
}
```

- [ ] **Step 7: Run it and confirm it fails**

Run: `cargo test -p scatrs --test reference_constants`
Expected: a compile error, `file not found for module common`.

- [ ] **Step 8: Implement the loader**

`scatrs/tests/common/mod.rs`:
```rust
//! Loader for the committed Arb reference data in `reference/data/`.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

/// Parallel arrays describing reference values (see `reference/scatrs_reference/records.py`).
#[derive(Debug, Deserialize)]
pub struct RefValues {
    pub hi: Vec<f64>,
    pub lo: Vec<f64>,
    pub dec: Vec<String>,
    pub err: Vec<f64>,
}

/// One reference record.
#[derive(Debug, Deserialize)]
pub struct Record {
    pub generator: String,
    pub case: String,
    pub label: String,
    pub inputs: serde_json::Map<String, serde_json::Value>,
    pub outputs: BTreeMap<String, RefValues>,
}

impl Record {
    pub fn output(&self, name: &str) -> &RefValues {
        self.outputs
            .get(name)
            .unwrap_or_else(|| panic!("record {} has no output {name}", self.case))
    }

    fn input(&self, name: &str) -> &serde_json::Value {
        self.inputs
            .get(name)
            .unwrap_or_else(|| panic!("record {} has no input {name}", self.case))
    }

    pub fn input_f64(&self, name: &str) -> f64 {
        self.input(name).as_f64().expect("number")
    }

    pub fn input_i64(&self, name: &str) -> i64 {
        self.input(name).as_i64().expect("integer")
    }

    pub fn input_f64s(&self, name: &str) -> Vec<f64> {
        self.input(name)
            .as_array()
            .expect("array")
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect()
    }

    pub fn input_i64s(&self, name: &str) -> Vec<i64> {
        self.input(name)
            .as_array()
            .expect("array")
            .iter()
            .map(|v| v.as_i64().expect("integer"))
            .collect()
    }

    pub fn input_strs(&self, name: &str) -> Vec<String> {
        self.input(name)
            .as_array()
            .expect("array")
            .iter()
            .map(|v| v.as_str().expect("string").to_owned())
            .collect()
    }
}

pub fn data_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../reference/data")
        .join(file)
}

pub fn load(file: &str) -> Vec<Record> {
    let text = std::fs::read_to_string(data_path(file))
        .unwrap_or_else(|e| panic!("cannot read reference file {file}: {e}"));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid reference record"))
        .collect()
}

pub fn find<'a>(records: &'a [Record], case: &str) -> &'a Record {
    records
        .iter()
        .find(|r| r.case == case)
        .unwrap_or_else(|| panic!("no reference case {case}"))
}
```

- [ ] **Step 9: Run all tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add .gitattributes reference/ scatrs/tests/common scatrs/tests/reference_constants.rs
git commit -m "feat: Arb reference generator with certified constants and Rust loader"
```

(`reference/uv.lock` is created by uv; commit it.)

---

### Task 3: `DoubleDouble` arithmetic

**Files:**
- Create: `scatrs/src/scalar/dd/mod.rs`, `scatrs/src/scalar/dd/consts.rs`
- Modify: `scatrs/src/scalar/mod.rs` (add `mod dd; pub use dd::DoubleDouble;`)
- Create: `reference/scatrs_reference/generators/dd_arith.py`
- Modify: `reference/scatrs_reference/generators/__init__.py` (register `dd_arith`)
- Generate: `reference/data/dd_arith.jsonl`
- Test: `scatrs/tests/dd_arith.rs`

**Interfaces:**
- Consumes: `common::{load, find, Record}` (Task 2).
- Produces: `scatrs::scalar::DoubleDouble` (`Clone, Copy, Debug, Default, PartialEq, PartialOrd`), with:
  - **Constants:** `ZERO`, `ONE`, `NAN`, `INFINITY`, `NEG_INFINITY`.
  - **Construction and access:** `const fn from_f64(f64)`, `fn new(hi, lo)` (normalises any pair), `const fn hi()`, `const fn lo()`.
  - **Arithmetic methods:** `add_f64`, `add_dd`, `mul_f64`, `mul_dd`, `div_f64`, `div_dd`, `mul_pow2(p: f64)`, `ldexp(i32)`, `recip`.
  - **Sign and rounding:** `abs`, `floor`, `ceil`, `trunc`, `round` (halfway cases away from zero).
  - **Operators:**
    - `+ - * / %` and `-x`;
    - the `*Assign` operators with `Self` and `f64`, and `RemAssign<Self>`;
    - mixed `DoubleDouble ⊕ f64` and `f64 ⊕ DoubleDouble`;
    - `From<f64>`.
- Produces (crate-internal):
  - `pub(crate) fn two_sum`, `fast_two_sum`, `two_prod`;
  - `pub(crate) const fn DoubleDouble::from_parts_unchecked`;
  - in `consts.rs`, the `pub(crate)` constants `PI`, `TAU`, `FRAC_PI_2`, `FRAC_PI_3`, `FRAC_PI_4`, `FRAC_PI_6`, `FRAC_PI_8`, `FRAC_3PI_4`, `FRAC_1_PI`, `FRAC_2_PI`, `FRAC_2_SQRT_PI`, `SQRT_2`, `FRAC_1_SQRT_2`, `E`, `LN_2`, `LN_10`, `LOG2_E`, `LOG10_E`, `LOG10_2`, `LOG2_10` (all `DoubleDouble`);
  - `FRAC_PI_2_TAIL: f64` and `LN_2_TAIL: f64`;
  - `INV_FACT: [DoubleDouble; 31]` and `POW10_2I: [DoubleDouble; 9]`.

Error bounds proved by Joldes, Muller & Popescu (2017), with u = 2⁻⁵³:

| Operation | Relative error bound |
|---|---|
| `add_dd` | 3u² |
| `add_f64` | 2u² |
| `mul_dd` | 4u² |
| `mul_f64` | 2u² |
| `div_dd` | 10u² |
| `div_f64` | 3u² |

- [ ] **Step 1: Add the reference generator**

`reference/scatrs_reference/generators/dd_arith.py`:
```python
"""Exact results of double-double basic operations on random normalised inputs."""

from __future__ import annotations

import math
import random

from flint import arb, ctx

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 1000
SEED = 3
OPS = ("add", "sub", "mul", "div", "add_f64", "mul_f64", "div_f64")


def _random_dd(rng: random.Random) -> tuple[float, float]:
    hi = rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 2.0) * 2.0 ** rng.randint(-60, 60)
    lo = rng.uniform(-0.5, 0.5) * math.ulp(hi)
    return (hi, lo) if hi + lo == hi else (hi, 0.0)


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    rng = random.Random(SEED)
    recs = []
    for op in OPS:
        a_hi, a_lo, b_hi, b_lo, z = [], [], [], [], []
        for i in range(N):
            ah, al = _random_dd(rng)
            if op in ("add", "sub") and i % 10 == 0:
                # Massive cancellation: b = ∓a with a different low word.
                bh = -ah if op == "add" else ah
                bl = rng.uniform(-0.5, 0.5) * math.ulp(bh)
            else:
                bh, bl = _random_dd(rng)
            if op.endswith("_f64"):
                bl = 0.0
            a, b = arb(ah) + arb(al), arb(bh) + arb(bl)
            exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op.removesuffix("_f64")]
            a_hi.append(ah)
            a_lo.append(al)
            b_hi.append(bh)
            b_lo.append(bl)
            z.append(exact)
        recs.append(
            record("dd_arith", op, {"a_hi": a_hi, "a_lo": a_lo, "b_hi": b_hi, "b_lo": b_lo}, {"z": z})
        )
    return PRECISION_BITS, recs
```

Register it in `generators/__init__.py`: add `dd_arith` to the import list and the entry `"dd_arith": dd_arith.generate,` to `GENERATORS`.

Run: `uv run --directory reference python -m scatrs_reference generate --only dd_arith && uv run --directory reference python -m pytest -q`
Expected: `dd_arith.jsonl: 7 records`; the tests pass.

- [ ] **Step 2: Write the failing Rust tests**

`scatrs/tests/dd_arith.rs`:
```rust
#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::{find, load};
use proptest::prelude::*;
use scatrs::scalar::DoubleDouble as Dd;

/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    f64::from_bits(a.to_bits() + 1) - a
}

/// Checks |op(a, b) - exact| + ref_err <= bound * u² * |exact| over every reference case.
fn check(op: &str, bound: f64, f: impl Fn(Dd, Dd, f64) -> Dd) {
    let recs = load("dd_arith.jsonl");
    let r = find(&recs, op);
    let (a_hi, a_lo) = (r.input_f64s("a_hi"), r.input_f64s("a_lo"));
    let (b_hi, b_lo) = (r.input_f64s("b_hi"), r.input_f64s("b_lo"));
    let z = r.output("z");
    let mut worst = 0.0_f64;
    for i in 0..a_hi.len() {
        let a = Dd::new(a_hi[i], a_lo[i]);
        let b = Dd::new(b_hi[i], b_lo[i]);
        let got = f(a, b, b_hi[i]);
        let exact = Dd::new(z.hi[i], z.lo[i]);
        let rel = ((got - exact).abs().hi() + z.err[i]) / exact.abs().hi();
        worst = worst.max(rel / U2);
        assert!(rel <= bound * U2, "{op} case {i}: {:.3} u^2 > {bound} u^2", rel / U2);
    }
    eprintln!("{op}: worst {worst:.3} u^2 (bound {bound} u^2)");
}

#[test]
fn add_within_3u2() {
    check("add", 3.0, |a, b, _| a + b)
}

#[test]
fn sub_within_3u2() {
    check("sub", 3.0, |a, b, _| a - b)
}

#[test]
fn mul_within_4u2() {
    check("mul", 4.0, |a, b, _| a * b)
}

#[test]
fn div_within_10u2() {
    check("div", 10.0, |a, b, _| a / b)
}

#[test]
fn add_f64_within_2u2() {
    check("add_f64", 2.0, |a, _, b| a + b)
}

#[test]
fn mul_f64_within_2u2() {
    check("mul_f64", 2.0, |a, _, b| a * b)
}

#[test]
fn div_f64_within_3u2() {
    check("div_f64", 3.0, |a, _, b| a / b)
}

#[test]
fn special_values() {
    let inf = Dd::INFINITY;
    assert_eq!((inf + 1.0).hi(), f64::INFINITY);
    assert_eq!((inf * Dd::from_f64(2.0)).hi(), f64::INFINITY);
    assert_eq!((Dd::ONE / Dd::ZERO).hi(), f64::INFINITY);
    assert!((Dd::ZERO / Dd::ZERO).hi().is_nan());
    assert!((inf / inf).hi().is_nan());
    assert_eq!((Dd::ONE / inf).hi(), 0.0);
    assert_eq!((Dd::from_f64(1e308) * 10.0).hi(), f64::INFINITY);
    assert_eq!(Dd::NAN.partial_cmp(&Dd::ONE), None);
    assert_eq!((-inf).hi(), f64::NEG_INFINITY);
}

#[test]
fn rounding_functions() {
    let big = Dd::new(9007199254740992.0, 0.5); // 2^53 + 1/2
    assert_eq!(big.floor(), Dd::from_f64(9007199254740992.0));
    assert_eq!(big.ceil(), Dd::new(9007199254740992.0, 1.0));
    let below_one = Dd::new(1.0, -(2f64.powi(-80)));
    assert_eq!(below_one.floor(), Dd::ZERO);
    assert_eq!(below_one.ceil(), Dd::ONE);
    assert_eq!(below_one.trunc(), Dd::ZERO);
    assert_eq!(Dd::new(-1.0, 2f64.powi(-80)).floor(), Dd::from_f64(-1.0));
    assert_eq!(Dd::from_f64(2.5).round(), Dd::from_f64(3.0));
    assert_eq!(Dd::from_f64(-2.5).round(), Dd::from_f64(-3.0));
    assert_eq!(Dd::new(0.5, -(2f64.powi(-60))).round(), Dd::ZERO);
    assert_eq!(Dd::INFINITY.floor(), Dd::INFINITY);
    assert_eq!(Dd::from_f64(7.5) % Dd::from_f64(2.0), Dd::from_f64(1.5));
    assert_eq!(Dd::from_f64(-7.5) % Dd::from_f64(2.0), Dd::from_f64(-1.5));
}

#[test]
fn ldexp_and_recip() {
    assert_eq!(Dd::new(1.0, 2f64.powi(-60)).ldexp(1500).hi(), f64::INFINITY);
    assert_eq!(Dd::new(0.75, 2f64.powi(-60)).ldexp(1024).hi(), 0.75 * 2f64.powi(1023) * 2.0);
    assert_eq!(Dd::new(1.0, 2f64.powi(-60)).ldexp(-10), Dd::new(2f64.powi(-10), 2f64.powi(-70)));
    assert_eq!(Dd::from_f64(4.0).recip(), Dd::from_f64(0.25));
}

fn dd(hi: f64, frac: f64) -> Dd {
    Dd::new(hi, frac * ulp(hi))
}

proptest! {
    #[test]
    fn results_are_normalised(ah in -1e30..1e30f64, af in -0.5..0.5f64, bh in -1e30..1e30f64, bf in -0.5..0.5f64) {
        prop_assume!(bh != 0.0);
        let (a, b) = (dd(ah, af), dd(bh, bf));
        for z in [a + b, a - b, a * b, a / b, a + bh, a * bh, a / bh] {
            prop_assert_eq!(z.hi() + z.lo(), z.hi());
        }
    }

    #[test]
    fn addition_commutes(ah in -1e30..1e30f64, af in -0.5..0.5f64, bh in -1e30..1e30f64, bf in -0.5..0.5f64) {
        let (a, b) = (dd(ah, af), dd(bh, bf));
        prop_assert_eq!(a + b, b + a);
    }

    #[test]
    fn rounding_matches_f64_for_f64_inputs(x in -1e15..1e15f64) {
        let d = Dd::from_f64(x);
        prop_assert_eq!(d.floor().hi(), x.floor());
        prop_assert_eq!(d.ceil().hi(), x.ceil());
        prop_assert_eq!(d.trunc().hi(), x.trunc());
        prop_assert_eq!(d.round().hi(), x.round());
    }
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test dd_arith`
Expected: a compile error, `no DoubleDouble in scalar`.

- [ ] **Step 4: Implement `DoubleDouble`**

In `scatrs/src/scalar/mod.rs`, add after the imports:
```rust
mod dd;

pub use dd::DoubleDouble;
```

`scatrs/src/scalar/dd/mod.rs`:
```rust
//! Double-double arithmetic: a value is the unevaluated sum `hi + lo` of two `f64`s,
//! normalised so that `hi == fl(hi + lo)`, giving about 106 significant bits.
//!
//! Basic operations are the algorithms of Joldes, Muller & Popescu, "Tight and rigorous
//! error bounds for basic building blocks of double-word arithmetic", ACM TOMS 44(2), 2017.
//! With u = 2⁻⁵³ the relative error bounds are: `+`/`-` 3u², `+ f64` 2u², `*` 4u²,
//! `* f64` 2u², `/` 10u², `/ f64` 3u². They rely on a correctly rounded fused multiply-add,
//! which `f64::mul_add` provides on every platform (in software where hardware lacks FMA).
//!
//! Non-finite results are returned as `(hi, 0.0)` so that infinities do not become NaN.
//! Full relative precision requires magnitudes above about 2⁻⁹⁶⁹ ≈ 2·10⁻²⁹².

use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

mod consts;

/// A double-double number: the exact sum `hi + lo` with `hi == fl(hi + lo)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoubleDouble {
    hi: f64,
    lo: f64,
}

/// Error-free sum: `s = fl(a + b)` and `s + e = a + b` exactly.
#[inline]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// Error-free sum, valid when `|a| >= |b|` or `a == 0`.
#[inline]
pub(crate) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Error-free product: `p = fl(a * b)` and `p + e = a * b` exactly.
#[inline]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// `2^e` for `-1022 <= e <= 1023`.
#[inline]
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

impl DoubleDouble {
    /// Zero.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    /// One.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    /// Not a number.
    pub const NAN: Self = Self { hi: f64::NAN, lo: 0.0 };
    /// Positive infinity.
    pub const INFINITY: Self = Self { hi: f64::INFINITY, lo: 0.0 };
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self { hi: f64::NEG_INFINITY, lo: 0.0 };

    /// Converts an `f64` exactly.
    #[inline]
    pub const fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// The exact value `hi + lo` of any two `f64`s, renormalised.
    #[inline]
    pub fn new(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        if !s.is_finite() {
            return Self { hi: s, lo: 0.0 };
        }
        Self { hi: s, lo: e }
    }

    /// Builds from parts already known to be normalised (constant tables).
    #[inline]
    pub(crate) const fn from_parts_unchecked(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// The leading word: the value rounded to `f64`.
    #[inline]
    pub const fn hi(self) -> f64 {
        self.hi
    }

    /// The trailing word.
    #[inline]
    pub const fn lo(self) -> f64 {
        self.lo
    }

    #[inline]
    fn renormalize(hi: f64, lo: f64) -> Self {
        let (h, l) = fast_two_sum(hi, lo);
        Self { hi: h, lo: l }
    }

    /// `self + y` (relative error ≤ 2u²).
    #[inline]
    pub fn add_f64(self, y: f64) -> Self {
        let (sh, sl) = two_sum(self.hi, y);
        if !sh.is_finite() {
            return Self { hi: sh, lo: 0.0 };
        }
        Self::renormalize(sh, self.lo + sl)
    }

    /// `self + y` (relative error ≤ 3u²).
    #[inline]
    pub fn add_dd(self, y: Self) -> Self {
        let (sh, sl) = two_sum(self.hi, y.hi);
        if !sh.is_finite() {
            return Self { hi: sh, lo: 0.0 };
        }
        let (th, tl) = two_sum(self.lo, y.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        let w = tl + vl;
        Self::renormalize(vh, w)
    }

    /// `self * y` (relative error ≤ 2u²).
    #[inline]
    pub fn mul_f64(self, y: f64) -> Self {
        let (ch, cl1) = two_prod(self.hi, y);
        if !ch.is_finite() {
            return Self { hi: ch, lo: 0.0 };
        }
        let cl3 = self.lo.mul_add(y, cl1);
        Self::renormalize(ch, cl3)
    }

    /// `self * y` (relative error ≤ 4u²).
    #[inline]
    pub fn mul_dd(self, y: Self) -> Self {
        let (ch, cl1) = two_prod(self.hi, y.hi);
        if !ch.is_finite() {
            return Self { hi: ch, lo: 0.0 };
        }
        let tl0 = self.lo * y.lo;
        let tl1 = self.hi.mul_add(y.lo, tl0);
        let cl2 = self.lo.mul_add(y.hi, tl1);
        let cl3 = cl1 + cl2;
        Self::renormalize(ch, cl3)
    }

    /// `self / y` (relative error ≤ 3u²).
    #[inline]
    pub fn div_f64(self, y: f64) -> Self {
        let th = self.hi / y;
        if !th.is_finite() || !y.is_finite() {
            return Self { hi: th, lo: 0.0 };
        }
        let (ph, pl) = two_prod(th, y);
        let dh = self.hi - ph;
        let dt = dh - pl;
        let d = dt + self.lo;
        let tl = d / y;
        Self::renormalize(th, tl)
    }

    /// `self / y` (relative error ≤ 10u²).
    #[inline]
    pub fn div_dd(self, y: Self) -> Self {
        let q = self.hi / y.hi;
        if !q.is_finite() || !y.hi.is_finite() {
            return Self { hi: q, lo: 0.0 };
        }
        let th = 1.0 / y.hi;
        let rh = (-y.hi).mul_add(th, 1.0);
        let rl = -y.lo * th;
        let (eh, el) = fast_two_sum(rh, rl);
        let m = Self { hi: eh, lo: el }.mul_f64(th).add_f64(th);
        self.mul_dd(m)
    }

    /// Multiplies by `p`, which must be a power of two (exact unless over/underflow).
    #[inline]
    pub fn mul_pow2(self, p: f64) -> Self {
        Self {
            hi: self.hi * p,
            lo: self.lo * p,
        }
    }

    /// `self * 2^e`.
    pub fn ldexp(self, e: i32) -> Self {
        let mut x = self;
        let mut e = e;
        while e > 1000 {
            x = x.mul_pow2(pow2(1000));
            e -= 1000;
        }
        while e < -1000 {
            x = x.mul_pow2(pow2(-1000));
            e += 1000;
        }
        x.mul_pow2(pow2(e))
    }

    /// `1 / self`.
    #[inline]
    pub fn recip(self) -> Self {
        Self::ONE.div_dd(self)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        if self.hi.is_sign_negative() { -self } else { self }
    }

    /// Largest integer not greater than `self`.
    pub fn floor(self) -> Self {
        if !self.hi.is_finite() {
            return self;
        }
        let f = self.hi.floor();
        if f == self.hi {
            let (h, l) = two_sum(f, self.lo.floor());
            Self { hi: h, lo: l }
        } else {
            Self { hi: f, lo: 0.0 }
        }
    }

    /// Smallest integer not less than `self`.
    pub fn ceil(self) -> Self {
        -(-self).floor()
    }

    /// Integer part (rounds towards zero).
    pub fn trunc(self) -> Self {
        if self.hi.is_sign_negative() { self.ceil() } else { self.floor() }
    }

    /// Nearest integer, rounding half-way cases away from zero (like `f64::round`).
    pub fn round(self) -> Self {
        let t = self.trunc();
        if (self - t).abs() >= Self::from_f64(0.5) {
            if self.hi.is_sign_negative() { t - 1.0 } else { t + 1.0 }
        } else {
            t
        }
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, y: Self) -> Self {
        self.add_dd(y)
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, y: Self) -> Self {
        self.add_dd(-y)
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, y: Self) -> Self {
        self.mul_dd(y)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, y: Self) -> Self {
        self.div_dd(y)
    }
}

impl Rem for DoubleDouble {
    type Output = Self;
    /// `self - y * trunc(self / y)`, with the sign of `self` (like `f64`'s `%`).
    fn rem(self, y: Self) -> Self {
        self - y * (self / y).trunc()
    }
}

impl Add<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, y: f64) -> Self {
        self.add_f64(y)
    }
}

impl Sub<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, y: f64) -> Self {
        self.add_f64(-y)
    }
}

impl Mul<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, y: f64) -> Self {
        self.mul_f64(y)
    }
}

impl Div<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, y: f64) -> Self {
        self.div_f64(y)
    }
}

impl Add<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn add(self, y: DoubleDouble) -> DoubleDouble {
        y.add_f64(self)
    }
}

impl Sub<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn sub(self, y: DoubleDouble) -> DoubleDouble {
        (-y).add_f64(self)
    }
}

impl Mul<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn mul(self, y: DoubleDouble) -> DoubleDouble {
        y.mul_f64(self)
    }
}

impl Div<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    #[inline]
    fn div(self, y: DoubleDouble) -> DoubleDouble {
        DoubleDouble::from_f64(self).div_dd(y)
    }
}

macro_rules! assign_ops {
    ($($trait:ident, $method:ident, $op:tt);* $(;)?) => {$(
        impl $trait for DoubleDouble {
            #[inline]
            fn $method(&mut self, y: Self) {
                *self = *self $op y;
            }
        }
        impl $trait<f64> for DoubleDouble {
            #[inline]
            fn $method(&mut self, y: f64) {
                *self = *self $op y;
            }
        }
    )*};
}

assign_ops!(
    AddAssign, add_assign, +;
    SubAssign, sub_assign, -;
    MulAssign, mul_assign, *;
    DivAssign, div_assign, /;
);

impl RemAssign for DoubleDouble {
    #[inline]
    fn rem_assign(&mut self, y: Self) {
        *self = *self % y;
    }
}

impl From<f64> for DoubleDouble {
    #[inline]
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl PartialOrd for DoubleDouble {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&other.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&other.lo),
            ord => Some(ord),
        }
    }
}
```

`scatrs/src/scalar/dd/consts.rs`. The values below are the certified Arb splits from Task 2; the unit test checks every one of them.
```rust
//! Double-double constants: the certified splits in `reference/data/constants.jsonl`.
// Used by the elementary, trigonometric and FloatConst code added in Tasks 4-6.
#![allow(dead_code)]

use super::DoubleDouble as Dd;

const fn c(hi: f64, lo: f64) -> Dd {
    Dd::from_parts_unchecked(hi, lo)
}

// Leading words use std constants (identical values; avoids clippy::approx_constant).
use core::f64::consts as f;

pub(crate) const PI: Dd = c(f::PI, 1.2246467991473532e-16);
pub(crate) const TAU: Dd = c(f::TAU, 2.4492935982947064e-16);
pub(crate) const FRAC_PI_2: Dd = c(f::FRAC_PI_2, 6.123233995736766e-17);
pub(crate) const FRAC_PI_3: Dd = c(f::FRAC_PI_3, -1.072081766451091e-16);
pub(crate) const FRAC_PI_4: Dd = c(f::FRAC_PI_4, 3.061616997868383e-17);
pub(crate) const FRAC_PI_6: Dd = c(f::FRAC_PI_6, -5.360408832255455e-17);
pub(crate) const FRAC_PI_8: Dd = c(f::FRAC_PI_8, 1.5308084989341915e-17);
pub(crate) const FRAC_3PI_4: Dd = c(2.356194490192345, 9.184850993605148e-17);
pub(crate) const FRAC_1_PI: Dd = c(f::FRAC_1_PI, -1.9678676675182486e-17);
pub(crate) const FRAC_2_PI: Dd = c(f::FRAC_2_PI, -3.935735335036497e-17);
pub(crate) const FRAC_2_SQRT_PI: Dd = c(f::FRAC_2_SQRT_PI, 1.533545961316588e-17);
pub(crate) const SQRT_2: Dd = c(f::SQRT_2, -9.667293313452913e-17);
pub(crate) const FRAC_1_SQRT_2: Dd = c(f::FRAC_1_SQRT_2, -4.833646656726457e-17);
pub(crate) const E: Dd = c(f::E, 1.4456468917292502e-16);
pub(crate) const LN_2: Dd = c(f::LN_2, 2.3190468138462996e-17);
pub(crate) const LN_10: Dd = c(f::LN_10, -2.1707562233822494e-16);
pub(crate) const LOG2_E: Dd = c(f::LOG2_E, 2.0355273740931033e-17);
pub(crate) const LOG10_E: Dd = c(f::LOG10_E, 1.098319650216765e-17);
pub(crate) const LOG10_2: Dd = c(f::LOG10_2, -2.8037281277851704e-18);
pub(crate) const LOG2_10: Dd = c(f::LOG2_10, 1.661617516973592e-16);

/// Third word of π/2: π/2 − FRAC_PI_2.hi − FRAC_PI_2.lo, rounded to `f64`.
pub(crate) const FRAC_PI_2_TAIL: f64 = -1.4973849048591698e-33;
/// Third word of ln 2.
pub(crate) const LN_2_TAIL: f64 = 5.707708438416212e-34;

/// `INV_FACT[k] = 1/k!` for `k = 0..=30`.
pub(crate) const INV_FACT: [Dd; 31] = [
    c(1.0, 0.0),
    c(1.0, 0.0),
    c(0.5, 0.0),
    c(0.16666666666666666, 9.25185853854297e-18),
    c(0.041666666666666664, 2.3129646346357427e-18),
    c(0.008333333333333333, 1.1564823173178714e-19),
    c(0.001388888888888889, -5.300543954373577e-20),
    c(0.0001984126984126984, 1.7209558293420705e-22),
    c(2.48015873015873e-05, 2.1511947866775882e-23),
    c(2.7557319223985893e-06, -1.858393274046472e-22),
    c(2.755731922398589e-07, 2.3767714622250297e-23),
    c(2.505210838544172e-08, -1.448814070935912e-24),
    c(2.08767569878681e-09, -1.20734505911326e-25),
    c(1.6059043836821613e-10, 1.2585294588752098e-26),
    c(1.1470745597729725e-11, 2.0655512752830745e-28),
    c(7.647163731819816e-13, 7.03872877733453e-30),
    c(4.779477332387385e-14, 4.399205485834081e-31),
    c(2.8114572543455206e-15, 1.6508842730861433e-31),
    c(1.5619206968586225e-16, 1.1910679660273754e-32),
    c(8.22063524662433e-18, 2.2141894119604265e-34),
    c(4.110317623312165e-19, 1.4412973378659527e-36),
    c(1.9572941063391263e-20, -1.3643503830087908e-36),
    c(8.896791392450574e-22, -7.911402614872376e-38),
    c(3.868170170630684e-23, -8.843177655482344e-40),
    c(1.6117375710961184e-24, -3.6846573564509766e-41),
    c(6.446950284384474e-26, -1.9330404233703465e-42),
    c(2.4795962632247976e-27, -1.2953730964765229e-43),
    c(9.183689863795546e-29, 1.4303150396787322e-45),
    c(3.279889237069838e-30, 1.5117542744029879e-46),
    c(1.1309962886447716e-31, 1.0498015412959506e-47),
    c(3.7699876288159054e-33, 2.5870347832750324e-49),
];

/// `POW10_2I[i] = 10^(2^i)` for `i = 0..=8`.
pub(crate) const POW10_2I: [Dd; 9] = [
    c(10.0, 0.0),
    c(100.0, 0.0),
    c(10000.0, 0.0),
    c(100000000.0, 0.0),
    c(1e16, 0.0),
    c(1e32, -5366162204393472.0),
    c(1e64, -2.1320419009454396e47),
    c(1e128, -7.51744869165182e111),
    c(1e256, -3.012765990014054e239),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> Vec<serde_json::Value> {
        include_str!("../../../../reference/data/constants.jsonl")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid record"))
            .collect()
    }

    fn hi_lo(records: &[serde_json::Value], case: &str) -> (f64, f64) {
        let r = records
            .iter()
            .find(|r| r["case"] == case)
            .unwrap_or_else(|| panic!("missing {case}"));
        let v = &r["outputs"]["value"];
        (v["hi"][0].as_f64().unwrap(), v["lo"][0].as_f64().unwrap())
    }

    #[test]
    fn constants_match_reference() {
        let recs = reference();
        let table = [
            ("PI", PI),
            ("TAU", TAU),
            ("FRAC_PI_2", FRAC_PI_2),
            ("FRAC_PI_3", FRAC_PI_3),
            ("FRAC_PI_4", FRAC_PI_4),
            ("FRAC_PI_6", FRAC_PI_6),
            ("FRAC_PI_8", FRAC_PI_8),
            ("FRAC_3PI_4", FRAC_3PI_4),
            ("FRAC_1_PI", FRAC_1_PI),
            ("FRAC_2_PI", FRAC_2_PI),
            ("FRAC_2_SQRT_PI", FRAC_2_SQRT_PI),
            ("SQRT_2", SQRT_2),
            ("FRAC_1_SQRT_2", FRAC_1_SQRT_2),
            ("E", E),
            ("LN_2", LN_2),
            ("LN_10", LN_10),
            ("LOG2_E", LOG2_E),
            ("LOG10_E", LOG10_E),
            ("LOG10_2", LOG10_2),
            ("LOG2_10", LOG2_10),
        ];
        for (name, value) in table {
            assert_eq!((value.hi(), value.lo()), hi_lo(&recs, name), "{name}");
        }
        assert_eq!(FRAC_PI_2_TAIL, hi_lo(&recs, "FRAC_PI_2_TAIL").0);
        assert_eq!(LN_2_TAIL, hi_lo(&recs, "LN_2_TAIL").0);
        for (k, value) in INV_FACT.iter().enumerate() {
            let expected = hi_lo(&recs, &format!("INV_FACT_{k}"));
            assert_eq!((value.hi(), value.lo()), expected, "1/{k}!");
        }
        for (i, value) in POW10_2I.iter().enumerate() {
            let expected = hi_lo(&recs, &format!("POW10_2POW{i}"));
            assert_eq!((value.hi(), value.lo()), expected, "10^(2^{i})");
        }
    }
}
```

- [ ] **Step 5: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass. Report the "worst … u^2" lines from `cargo test -p scatrs --test dd_arith -- --nocapture`. Every worst value must sit below its bound.

- [ ] **Step 6: Commit**

```bash
git add scatrs/src/scalar reference/scatrs_reference/generators reference/data scatrs/tests/dd_arith.rs
git commit -m "feat: DoubleDouble arithmetic with certified error-bound tests"
```

---

### Task 4: `DoubleDouble` square root, exponential and logarithm family

**Files:**
- Create: `scatrs/src/scalar/dd/elementary.rs`
- Modify: `scatrs/src/scalar/dd/mod.rs` (add `mod elementary;` below `mod consts;`)
- Create: `reference/scatrs_reference/generators/dd_elementary.py`
- Modify: `reference/scatrs_reference/generators/__init__.py` (register it)
- Generate: `reference/data/dd_elementary.jsonl`
- Test: `scatrs/tests/dd_elementary.rs`

**Interfaces:**
- Consumes: `DoubleDouble` and its helpers; the consts `INV_FACT`, `LN_2`, `LN_2_TAIL`, `LOG2_E`, `LOG10_E` (Task 3).
- Produces these inherent `pub fn`s on `DoubleDouble`:
  - one argument (`self`): `sqrt`, `exp`, `exp_m1`, `ln`, `ln_1p`, `exp2`, `log2`, `log10`, `cbrt`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`;
  - two arguments: `log(self, base: Self)`, `powi(self, n: i32)`, `powf(self, y: Self)`, `hypot(self, other: Self)`.
- Accuracy contract (verified against Arb; measured on exact double-double inputs):

| Functions | Relative error bound |
|---|---|
| `sqrt`, `exp`, `exp_m1`, `ln`, `ln_1p`, `log2`, `log10`, `cbrt`, `tanh`, `asinh`, `acosh`, `atanh`, `hypot` | 2⁻¹⁰⁰ |
| `sinh`, `cosh` | 2⁻⁹⁹ |
| `exp2(x)` | 2⁻¹⁰⁰·max(1, \|x\|) |
| `powf(x, y)` | 2⁻⁹⁹·max(1, \|y ln x\|) |
| `powi(x, n)` | 16(\|n\| + 1)u² |

  Domains:
  - `exp`: x ∈ [−670, 709].
  - Results must have magnitude ≥ 2⁻⁹⁶⁹.

- [ ] **Step 1: Add the reference generator**

`reference/scatrs_reference/generators/dd_elementary.py`:
```python
"""Arb values of double-double elementary functions at random double-double inputs."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from flint import arb, ctx

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 200
SEED = 4


def _dd(rng: random.Random, f: float) -> tuple[float, float]:
    """A normalised double-double with leading word f and a random low word."""
    if f == 0.0:
        return f, 0.0
    lo = rng.uniform(-0.5, 0.5) * math.ulp(f)
    return (f, lo) if f + lo == f else (f, 0.0)


def _logu(rng: random.Random, lo_exp: float, hi_exp: float) -> float:
    return 10.0 ** rng.uniform(lo_exp, hi_exp)


def _signed(rng: random.Random, v: float) -> float:
    return v if rng.random() < 0.5 else -v


def _ln1p_sample(r: random.Random) -> float:
    u = r.random()
    if u < 0.5:
        return _signed(r, _logu(r, -30, -0.7))
    if u < 0.75:
        return _logu(r, -0.6, 6)
    return -_logu(r, -0.6, -0.0005)


SAMPLERS: dict[str, Callable[[random.Random], float]] = {
    "sqrt": lambda r: _logu(r, -300, 300),
    "exp": lambda r: r.uniform(-670.0, 709.0) if r.random() < 0.5 else _signed(r, _logu(r, -30, 0)),
    "exp_m1": lambda r: _signed(r, _logu(r, -30, 1.5)),
    "ln": lambda r: _logu(r, -300, 300) if r.random() < 0.5 else 1.0 + _signed(r, _logu(r, -15, -0.7)),
    "ln_1p": _ln1p_sample,
    "log2": lambda r: _logu(r, -300, 300),
    "log10": lambda r: _logu(r, -300, 300),
    "cbrt": lambda r: _signed(r, _logu(r, -300, 300)),
    "sinh": lambda r: _signed(r, _logu(r, -30, 2.8)),
    "cosh": lambda r: _signed(r, _logu(r, -30, 2.8)),
    "tanh": lambda r: _signed(r, _logu(r, -30, 1.7)),
    "asinh": lambda r: _signed(r, _logu(r, -30, 30)),
    "acosh": lambda r: 1.0 + _logu(r, -15, 30),
    "atanh": lambda r: _signed(r, _logu(r, -30, -0.3))
    if r.random() < 0.5
    else _signed(r, 1.0 - _logu(r, -15, -0.0005)),
    "exp2": lambda r: r.uniform(-960.0, 1020.0),
}

FUNCTIONS: dict[str, Callable[[arb], arb]] = {
    "sqrt": lambda x: x.sqrt(),
    "exp": lambda x: x.exp(),
    "exp_m1": lambda x: x.expm1(),
    "ln": lambda x: x.log(),
    "ln_1p": lambda x: x.log1p(),
    "log2": lambda x: x.log() / arb(2).log(),
    "log10": lambda x: x.log() / arb(10).log(),
    "cbrt": lambda x: x.root(3) if x >= 0 else -((-x).root(3)),
    "sinh": lambda x: x.sinh(),
    "cosh": lambda x: x.cosh(),
    "tanh": lambda x: x.tanh(),
    "asinh": lambda x: x.asinh(),
    "acosh": lambda x: x.acosh(),
    "atanh": lambda x: x.atanh(),
    "exp2": lambda x: arb(2) ** x,
}


def _as_arb(p: tuple[float, float]) -> arb:
    return arb(p[0]) + arb(p[1])


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    rng = random.Random(SEED)
    recs = []
    for name, sample in SAMPLERS.items():
        xs = [_dd(rng, sample(rng)) for _ in range(N)]
        recs.append(
            record(
                "dd_elementary",
                name,
                {"x_hi": [h for h, _ in xs], "x_lo": [lo for _, lo in xs]},
                {"value": [FUNCTIONS[name](_as_arb(p)) for p in xs]},
            )
        )
    xs = [_dd(rng, _signed(rng, _logu(rng, -150, 150))) for _ in range(N)]
    ys = [_dd(rng, _signed(rng, _logu(rng, -150, 150))) for _ in range(N)]
    recs.append(
        record(
            "dd_elementary",
            "hypot",
            {"x_hi": [p[0] for p in xs], "x_lo": [p[1] for p in xs],
             "y_hi": [p[0] for p in ys], "y_lo": [p[1] for p in ys]},
            {"value": [(_as_arb(a) ** 2 + _as_arb(b) ** 2).sqrt() for a, b in zip(xs, ys)]},
        )
    )
    xs = [_dd(rng, _logu(rng, -5, 5)) for _ in range(N)]
    ys = [_dd(rng, rng.uniform(-20.0, 20.0)) for _ in range(N)]
    recs.append(
        record(
            "dd_elementary",
            "powf",
            {"x_hi": [p[0] for p in xs], "x_lo": [p[1] for p in xs],
             "y_hi": [p[0] for p in ys], "y_lo": [p[1] for p in ys]},
            {"value": [_as_arb(a) ** _as_arb(b) for a, b in zip(xs, ys)]},
        )
    )
    xs = [_dd(rng, _signed(rng, _logu(rng, -3, 3))) for _ in range(N)]
    ns = [rng.randint(-100, 100) for _ in range(N)]
    recs.append(
        record(
            "dd_elementary",
            "powi",
            {"x_hi": [p[0] for p in xs], "x_lo": [p[1] for p in xs], "n": ns},
            {"value": [_as_arb(a) ** n for a, n in zip(xs, ns)]},
        )
    )
    return PRECISION_BITS, recs
```

Register `dd_elementary` in `generators/__init__.py`. Then:

Run: `uv run --directory reference python -m scatrs_reference generate --only dd_elementary && uv run --directory reference python -m pytest -q`
Expected: 18 records; the tests pass.

- [ ] **Step 2: Write the failing Rust tests**

`scatrs/tests/dd_elementary.rs`:
```rust
#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::{find, load, Record};
use scatrs::scalar::DoubleDouble as Dd;

/// 2^-100.
const TOL: f64 = 7.888_609_052_210_118e-31;
/// u² = 2^-106.
const U2: f64 = 1.232_595_164_407_831e-32;

fn dd_input(r: &Record, prefix: &str) -> Vec<Dd> {
    let hi = r.input_f64s(&format!("{prefix}_hi"));
    let lo = r.input_f64s(&format!("{prefix}_lo"));
    hi.iter().zip(&lo).map(|(&h, &l)| Dd::new(h, l)).collect()
}

/// Asserts (|got − ref| + ref_err) / |ref| <= tol(i) for every case of `case`.
fn check(case: &str, got: impl Fn(&Record, usize) -> Dd, tol: impl Fn(&Record, usize) -> f64) {
    let recs = load("dd_elementary.jsonl");
    let r = find(&recs, case);
    let v = r.output("value");
    let mut worst = 0.0_f64;
    for i in 0..v.hi.len() {
        let exact = Dd::new(v.hi[i], v.lo[i]);
        let g = got(r, i);
        let rel = ((g - exact).abs().hi() + v.err[i]) / exact.abs().hi();
        let t = tol(r, i);
        worst = worst.max(rel / t);
        assert!(rel <= t, "{case} case {i}: got {g:?}, reference {exact:?}, rel {rel:e} > {t:e}");
    }
    eprintln!("{case}: worst error = {worst:.3} x tolerance");
}

fn unary(case: &str, f: fn(Dd) -> Dd, tol: f64) {
    check(case, |r, i| f(dd_input(r, "x")[i]), |_, _| tol);
}

#[test]
fn sqrt() {
    unary("sqrt", Dd::sqrt, TOL)
}
#[test]
fn exp() {
    unary("exp", Dd::exp, TOL)
}
#[test]
fn exp_m1() {
    unary("exp_m1", Dd::exp_m1, TOL)
}
#[test]
fn ln() {
    unary("ln", Dd::ln, TOL)
}
#[test]
fn ln_1p() {
    unary("ln_1p", Dd::ln_1p, TOL)
}
#[test]
fn log2() {
    unary("log2", Dd::log2, TOL)
}
#[test]
fn log10() {
    unary("log10", Dd::log10, TOL)
}
#[test]
fn cbrt() {
    unary("cbrt", Dd::cbrt, TOL)
}
#[test]
fn sinh() {
    unary("sinh", Dd::sinh, 2.0 * TOL)
}
#[test]
fn cosh() {
    unary("cosh", Dd::cosh, 2.0 * TOL)
}
#[test]
fn tanh() {
    unary("tanh", Dd::tanh, TOL)
}
#[test]
fn asinh() {
    unary("asinh", Dd::asinh, TOL)
}
#[test]
fn acosh() {
    unary("acosh", Dd::acosh, TOL)
}
#[test]
fn atanh() {
    unary("atanh", Dd::atanh, TOL)
}

#[test]
fn exp2() {
    // exp2(x) = exp(x ln 2): rounding x·ln 2 costs a factor |x| (conditioning).
    check("exp2", |r, i| dd_input(r, "x")[i].exp2(), |r, i| {
        TOL * dd_input(r, "x")[i].hi().abs().max(1.0)
    });
}

#[test]
fn hypot() {
    check("hypot", |r, i| dd_input(r, "x")[i].hypot(dd_input(r, "y")[i]), |_, _| TOL);
}

#[test]
fn powf() {
    // exp(y ln x): the error in y·ln x is amplified by |y ln x| (conditioning).
    check("powf", |r, i| dd_input(r, "x")[i].powf(dd_input(r, "y")[i]), |r, i| {
        let (x, y) = (dd_input(r, "x")[i].hi(), dd_input(r, "y")[i].hi());
        2.0 * TOL * (y * x.ln()).abs().max(1.0)
    });
}

#[test]
fn powi() {
    // Binary powering: each squaring doubles the accumulated relative error.
    check("powi", |r, i| dd_input(r, "x")[i].powi(r.input_i64s("n")[i] as i32), |r, i| {
        16.0 * (r.input_i64s("n")[i].unsigned_abs() as f64 + 1.0) * U2
    });
}

#[test]
fn special_values() {
    let d = Dd::from_f64;
    assert!(d(-1.0).sqrt().hi().is_nan());
    assert_eq!(d(0.0).sqrt(), d(0.0));
    assert_eq!(d(f64::INFINITY).sqrt().hi(), f64::INFINITY);
    assert_eq!(d(1000.0).exp().hi(), f64::INFINITY);
    assert_eq!(d(-1000.0).exp(), Dd::ZERO);
    assert!(Dd::NAN.exp().hi().is_nan());
    assert_eq!(d(0.0).exp(), Dd::ONE);
    assert_eq!(d(0.0).ln().hi(), f64::NEG_INFINITY);
    assert!(d(-1.0).ln().hi().is_nan());
    assert_eq!(d(1.0).ln(), Dd::ZERO);
    assert_eq!(d(0.0).exp_m1(), d(0.0));
    assert_eq!(d(f64::NEG_INFINITY).exp_m1(), d(-1.0));
    assert_eq!(d(-1.0).ln_1p().hi(), f64::NEG_INFINITY);
    assert!(d(-2.0).ln_1p().hi().is_nan());
    assert_eq!(d(1.0).atanh().hi(), f64::INFINITY);
    assert!(d(0.5).acosh().hi().is_nan());
    assert_eq!(d(-8.0).cbrt(), d(-2.0));
    assert_eq!(d(3.0).hypot(d(4.0)), d(5.0));
    assert_eq!(d(0.0).powf(d(2.0)), Dd::ZERO);
    assert_eq!(d(-2.0).powf(d(3.0)), d(-8.0));
    assert!(d(-2.0).powf(d(0.5)).hi().is_nan());
    assert_eq!(d(7.0).powi(0), Dd::ONE);
    assert_eq!(d(2.0).powi(-2), d(0.25));
    assert_eq!(d(100.0).log(d(10.0)).hi(), 2.0);
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test dd_elementary`
Expected: compile errors, `no method named sqrt …`.

- [ ] **Step 4: Implement**

Add `mod elementary;` in `scatrs/src/scalar/dd/mod.rs` after `mod consts;`.

`scatrs/src/scalar/dd/elementary.rs`:
```rust
//! Square root, exponential and logarithm family for `DoubleDouble`.
//!
//! * `exp`: Cody–Waite reduction by a three-word ln 2, scaling by 2⁻⁹, a degree-10 Taylor
//!   polynomial for eʳ − 1, then nine squarings in the form (1 + s)² − 1 = 2s + s².
//! * `ln`: one Newton step x₀ + a·e^(−x₀) − 1 from the `f64` logarithm, except near 1,
//!   where it uses the series ln(1 + x) = 2 atanh(x / (2 + x)) on the exact x − 1.
//! * `sqrt`: Karp & Markstein, √a ≈ y + (a − y²)·(1/√a)/2 with y² computed exactly.

use super::consts::{INV_FACT, LN_2, LN_2_TAIL, LOG10_E, LOG2_E};
use super::{two_prod, DoubleDouble as Dd};

/// |x| below which ln(1 + x) uses the atanh series (and ln(a) uses it for |a − 1| below this).
const LN_1P_SERIES_LIMIT: f64 = 0.25;
/// ln(2)/2: below this |x|, exp_m1 avoids the cancellation in exp(x) − 1.
const HALF_LN_2: f64 = 0.34657359027997264;

/// eʳ − 1 for |r| ≤ ln(2)/1024 (truncation below 10⁻³⁴ relative).
fn expm1_kernel(r: Dd) -> Dd {
    let mut p = INV_FACT[10];
    for k in (2..10).rev() {
        p = p * r + INV_FACT[k];
    }
    (p * r + 1.0) * r
}

/// eˣ − 1 for |x| < ln(2)/2: kernel at x/512, then nine squarings of (1 + s).
fn expm1_small(x: Dd) -> Dd {
    let mut s = expm1_kernel(x.mul_pow2(1.0 / 512.0));
    for _ in 0..9 {
        s = s.mul_pow2(2.0) + s * s;
    }
    s
}

/// ln(1 + x) for |x| < LN_1P_SERIES_LIMIT via 2 atanh(x / (2 + x)).
fn ln_1p_series(x: Dd) -> Dd {
    if x.hi() == 0.0 {
        return x;
    }
    let z = x / (x + 2.0);
    let z2 = z * z;
    let mut term = z;
    let mut sum = z;
    let mut k = 1.0;
    loop {
        term *= z2;
        k += 2.0;
        let t = term / k;
        sum += t;
        if t.hi().abs() <= sum.hi().abs() * 1e-34 || k > 121.0 {
            break;
        }
    }
    sum.mul_pow2(2.0)
}

impl Dd {
    /// Square root.
    pub fn sqrt(self) -> Self {
        if self.hi() == 0.0 {
            return self;
        }
        if self.hi() < 0.0 || self.hi().is_nan() {
            return Self::NAN;
        }
        if self.hi().is_infinite() {
            return Self::INFINITY;
        }
        let inv = 1.0 / self.hi().sqrt();
        let y = self.hi() * inv;
        let (p, e) = two_prod(y, y);
        let resid = (self - Dd::from_parts_unchecked(p, e)).hi();
        Dd::new(y, resid * inv * 0.5)
    }

    /// eˣ.
    pub fn exp(self) -> Self {
        let x = self.hi();
        if x.is_nan() {
            return Self::NAN;
        }
        if x > 709.782712893384 {
            return Self::INFINITY;
        }
        if x < -745.1332191019412 {
            return Self::ZERO;
        }
        if x == 0.0 {
            return Self::ONE;
        }
        let m = (x * LOG2_E.hi()).round();
        let (a, b) = two_prod(m, LN_2.hi());
        let (c, d) = two_prod(m, LN_2.lo());
        let r = self - Dd::from_parts_unchecked(a, b) - Dd::from_parts_unchecked(c, d) - m * LN_2_TAIL;
        (expm1_small(r) + 1.0).ldexp(m as i32)
    }

    /// eˣ − 1, accurate near zero.
    pub fn exp_m1(self) -> Self {
        if self.hi().is_nan() {
            return Self::NAN;
        }
        if self.hi().abs() >= HALF_LN_2 {
            return self.exp() - 1.0;
        }
        if self.hi() == 0.0 {
            return self;
        }
        expm1_small(self)
    }

    /// Natural logarithm.
    pub fn ln(self) -> Self {
        let x = self.hi();
        if x.is_nan() || x < 0.0 {
            return Self::NAN;
        }
        if x == 0.0 {
            return Self::NEG_INFINITY;
        }
        if x == f64::INFINITY {
            return Self::INFINITY;
        }
        if (x - 1.0).abs() < LN_1P_SERIES_LIMIT {
            return ln_1p_series(self - 1.0);
        }
        if x < 1e-270 {
            return self.ldexp(1000).ln() - LN_2 * 1000.0;
        }
        if x > 1e270 {
            return self.ldexp(-1000).ln() + LN_2 * 1000.0;
        }
        let x0 = Dd::from_f64(x.ln());
        x0 + self * (-x0).exp() - 1.0
    }

    /// ln(1 + x), accurate near zero.
    pub fn ln_1p(self) -> Self {
        if self.hi().is_nan() || self.hi() < -1.0 {
            return Self::NAN;
        }
        if self.hi().abs() < LN_1P_SERIES_LIMIT {
            ln_1p_series(self)
        } else {
            (self + 1.0).ln()
        }
    }

    /// 2ˣ.
    pub fn exp2(self) -> Self {
        (self * LN_2).exp()
    }

    /// Base-2 logarithm.
    pub fn log2(self) -> Self {
        self.ln() * LOG2_E
    }

    /// Base-10 logarithm.
    pub fn log10(self) -> Self {
        self.ln() * LOG10_E
    }

    /// Logarithm in an arbitrary base.
    pub fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    /// Integer power by binary powering.
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        let mut result = Self::ONE;
        let mut base = self;
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                result *= base;
            }
            e >>= 1;
            if e > 0 {
                base *= base;
            }
        }
        if n < 0 { result.recip() } else { result }
    }

    /// `self` raised to a real power (negative bases only for integer exponents).
    pub fn powf(self, y: Self) -> Self {
        if y.hi() == 0.0 {
            return Self::ONE;
        }
        if self.hi().is_nan() || y.hi().is_nan() {
            return Self::NAN;
        }
        if self.hi() > 0.0 {
            return (y * self.ln()).exp();
        }
        if self.hi() == 0.0 {
            return if y.hi() > 0.0 { Self::ZERO } else { Self::INFINITY };
        }
        if y == y.trunc() && y.hi().abs() < 2_147_483_647.0 {
            return self.powi(y.hi() as i32);
        }
        Self::NAN
    }

    /// Cube root.
    pub fn cbrt(self) -> Self {
        if self.hi() == 0.0 || !self.hi().is_finite() {
            return Dd::from_f64(self.hi().cbrt());
        }
        let y = Dd::from_f64(self.hi().cbrt());
        y + (self - y * y * y) / (y * y * 3.0)
    }

    /// √(self² + other²) without intermediate overflow.
    pub fn hypot(self, other: Self) -> Self {
        let (a, b) = (self.abs(), other.abs());
        if a.hi().is_infinite() || b.hi().is_infinite() {
            return Self::INFINITY;
        }
        if a.hi().is_nan() || b.hi().is_nan() {
            return Self::NAN;
        }
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        if big.hi() == 0.0 {
            return Self::ZERO;
        }
        let r = small / big;
        big * (r * r + 1.0).sqrt()
    }

    /// Hyperbolic sine.
    pub fn sinh(self) -> Self {
        if self.hi().abs() < 0.5 {
            let e = self.exp_m1();
            (e * (e + 2.0) / (e + 1.0)).mul_pow2(0.5)
        } else {
            let ex = self.exp();
            (ex - ex.recip()).mul_pow2(0.5)
        }
    }

    /// Hyperbolic cosine.
    pub fn cosh(self) -> Self {
        let ex = self.abs().exp();
        (ex + ex.recip()).mul_pow2(0.5)
    }

    /// Hyperbolic tangent.
    pub fn tanh(self) -> Self {
        if self.hi().is_nan() {
            return Self::NAN;
        }
        if self.hi().abs() > 40.0 {
            return Dd::from_f64(self.hi().signum());
        }
        let e = self.mul_pow2(2.0).exp_m1();
        e / (e + 2.0)
    }

    /// Inverse hyperbolic sine.
    pub fn asinh(self) -> Self {
        if !self.hi().is_finite() {
            return self;
        }
        let a = self.abs();
        let r = if a.hi() > 1e150 {
            a.ln() + LN_2
        } else {
            let a2 = a * a;
            (a + a2 / ((a2 + 1.0).sqrt() + 1.0)).ln_1p()
        };
        if self.hi() < 0.0 { -r } else { r }
    }

    /// Inverse hyperbolic cosine (`self >= 1`).
    pub fn acosh(self) -> Self {
        if self.hi().is_nan() || self < Self::ONE {
            return Self::NAN;
        }
        if self.hi().is_infinite() {
            return Self::INFINITY;
        }
        if self.hi() > 1e150 {
            return self.ln() + LN_2;
        }
        let t = self - 1.0;
        (t + (t * (self + 1.0)).sqrt()).ln_1p()
    }

    /// Inverse hyperbolic tangent (`|self| <= 1`).
    pub fn atanh(self) -> Self {
        let a = self.abs();
        if self.hi().is_nan() || a > Self::ONE {
            return Self::NAN;
        }
        if a == Self::ONE {
            return Dd::from_f64(f64::INFINITY.copysign(self.hi()));
        }
        (self.mul_pow2(2.0) / (Self::ONE - self)).ln_1p().mul_pow2(0.5)
    }
}
```

- [ ] **Step 5: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass. Run `cargo test -p scatrs --test dd_elementary -- --nocapture` and report the "worst error = … x tolerance" lines.

If a function exceeds its tolerance, do not loosen the tolerance. First check the algorithm against the module-level description and the reference sampler domain. If that doesn't resolve it, stop and report the case.

- [ ] **Step 6: Commit**

```bash
git add scatrs/src/scalar/dd reference/scatrs_reference/generators reference/data scatrs/tests/dd_elementary.rs
git commit -m "feat: DoubleDouble sqrt/exp/log family validated against Arb"
```

---

### Task 5: `DoubleDouble` trigonometric family

**Files:**
- Create: `scatrs/src/scalar/dd/trig.rs`
- Modify: `scatrs/src/scalar/dd/mod.rs` (add `mod trig;`)
- Create: `reference/scatrs_reference/generators/dd_trig.py`
- Modify: `reference/scatrs_reference/generators/__init__.py` (register it)
- Generate: `reference/data/dd_trig.jsonl`
- Test: `scatrs/tests/dd_trig.rs`

**Interfaces:**
- Consumes: `two_prod`, `DoubleDouble`, `INV_FACT`, `PI`, `FRAC_PI_2`, `FRAC_PI_4`, `FRAC_3PI_4`, `FRAC_2_PI`, `FRAC_PI_2_TAIL` (Task 3); `sqrt` and `hypot` (Task 4).
- Produces these inherent `pub fn`s on `DoubleDouble`: `sin`, `cos`, `sin_cos(self) -> (Self, Self)`, `tan`, `atan`, `atan2(self, x: Self)` (with `self` as y), `asin`, `acos`.
- Accuracy contract:
  - `sin` and `cos`: relative error ≤ 2⁻¹⁰⁰ for |x| ≤ π/4, absolute error ≤ 2⁻¹⁰⁰ for |x| ≤ 2⁵⁰, and NaN for |x| > 2⁵⁰.
  - `tan`: absolute error ≤ 2⁻⁹⁹(1 + |tan x|)/|cos x|.
  - `atan`, `atan2`, `asin`, `acos`: relative error ≤ 2⁻⁹⁹.

- [ ] **Step 1: Add the reference generator**

`reference/scatrs_reference/generators/dd_trig.py`:
```python
"""Arb values of double-double trigonometric functions at random double-double inputs."""

from __future__ import annotations

import math
import random

from flint import arb, ctx

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 200
SEED = 5


def _dd(rng: random.Random, f: float) -> tuple[float, float]:
    if f == 0.0:
        return f, 0.0
    lo = rng.uniform(-0.5, 0.5) * math.ulp(f)
    return (f, lo) if f + lo == f else (f, 0.0)


def _logu(rng: random.Random, a: float, b: float) -> float:
    return 10.0 ** rng.uniform(a, b)


def _signed(rng: random.Random, v: float) -> float:
    return v if rng.random() < 0.5 else -v


def _angle(rng: random.Random) -> float:
    u = rng.random()
    if u < 0.3:
        return _signed(rng, _logu(rng, -30, -0.11))  # |x| < π/4: relative accuracy
    if u < 0.6:
        return rng.uniform(-10.0, 10.0)
    if u < 0.85:
        return rng.uniform(-(2.0**20), 2.0**20)
    return rng.uniform(-(2.0**50), 2.0**50)


SAMPLERS = {
    "sin": _angle,
    "cos": _angle,
    "tan": lambda r: _signed(r, _logu(r, -30, -0.11)) if r.random() < 0.4 else r.uniform(-10.0, 10.0),
    "atan": lambda r: _signed(r, _logu(r, -30, 30)),
    "asin": lambda r: _signed(r, _logu(r, -30, 0)),
    "acos": lambda r: _signed(r, _logu(r, -30, 0)),
}

FUNCTIONS = {
    "sin": lambda x: x.sin(),
    "cos": lambda x: x.cos(),
    "tan": lambda x: x.tan(),
    "atan": lambda x: x.atan(),
    "asin": lambda x: x.asin(),
    "acos": lambda x: x.acos(),
}


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    rng = random.Random(SEED)
    recs = []
    for name, sample in SAMPLERS.items():
        xs = [_dd(rng, sample(rng)) for _ in range(N)]
        recs.append(
            record(
                "dd_trig",
                name,
                {"x_hi": [h for h, _ in xs], "x_lo": [lo for _, lo in xs]},
                {"value": [FUNCTIONS[name](arb(h) + arb(lo)) for h, lo in xs]},
            )
        )
    ys = [_dd(rng, _signed(rng, _logu(rng, -30, 30))) for _ in range(N)]
    xs = [_dd(rng, _signed(rng, _logu(rng, -30, 30))) for _ in range(N)]
    recs.append(
        record(
            "dd_trig",
            "atan2",
            {"y_hi": [p[0] for p in ys], "y_lo": [p[1] for p in ys],
             "x_hi": [p[0] for p in xs], "x_lo": [p[1] for p in xs]},
            {"value": [arb.atan2(arb(y[0]) + arb(y[1]), arb(x[0]) + arb(x[1])) for y, x in zip(ys, xs)]},
        )
    )
    return PRECISION_BITS, recs
```

Register `dd_trig`. Then:

Run: `uv run --directory reference python -m scatrs_reference generate --only dd_trig && uv run --directory reference python -m pytest -q`
Expected: 7 records; the tests pass.

- [ ] **Step 2: Write the failing Rust tests**

`scatrs/tests/dd_trig.rs`:
```rust
#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::{find, load, Record};
use scatrs::scalar::DoubleDouble as Dd;

/// 2^-100.
const TOL: f64 = 7.888_609_052_210_118e-31;

fn dd_input(r: &Record, prefix: &str) -> Vec<Dd> {
    let hi = r.input_f64s(&format!("{prefix}_hi"));
    let lo = r.input_f64s(&format!("{prefix}_lo"));
    hi.iter().zip(&lo).map(|(&h, &l)| Dd::new(h, l)).collect()
}

/// Asserts |got − ref| + ref_err <= abs_tol(x, ref) for every case.
fn check(case: &str, got: impl Fn(&Record, usize) -> Dd, abs_tol: impl Fn(&Record, usize, Dd) -> f64) {
    let recs = load("dd_trig.jsonl");
    let r = find(&recs, case);
    let v = r.output("value");
    let mut worst = 0.0_f64;
    for i in 0..v.hi.len() {
        let exact = Dd::new(v.hi[i], v.lo[i]);
        let g = got(r, i);
        let err = (g - exact).abs().hi() + v.err[i];
        let t = abs_tol(r, i, exact);
        worst = worst.max(err / t);
        assert!(err <= t, "{case} case {i}: got {g:?}, reference {exact:?}, error {err:e} > {t:e}");
    }
    eprintln!("{case}: worst error = {worst:.3} x tolerance");
}

/// Relative for |x| <= π/4 (no reduction), absolute otherwise.
fn sin_cos_tol(r: &Record, i: usize, exact: Dd) -> f64 {
    if dd_input(r, "x")[i].hi().abs() <= std::f64::consts::FRAC_PI_4 {
        TOL * exact.abs().hi()
    } else {
        TOL
    }
}

#[test]
fn sin() {
    check("sin", |r, i| dd_input(r, "x")[i].sin(), sin_cos_tol);
}

#[test]
fn cos() {
    check("cos", |r, i| dd_input(r, "x")[i].cos(), sin_cos_tol);
}

#[test]
fn sin_cos_agrees_with_sin_and_cos() {
    let recs = load("dd_trig.jsonl");
    for x in dd_input(find(&recs, "sin"), "x") {
        let (s, c) = x.sin_cos();
        assert_eq!(s, x.sin());
        assert_eq!(c, x.cos());
    }
}

#[test]
fn tan() {
    check("tan", |r, i| dd_input(r, "x")[i].tan(), |r, i, exact| {
        let c = dd_input(r, "x")[i].hi().cos().abs();
        2.0 * TOL * (1.0 + exact.abs().hi()) / c
    });
}

#[test]
fn inverse_functions() {
    for case in ["atan", "asin", "acos"] {
        let f = match case {
            "atan" => Dd::atan,
            "asin" => Dd::asin,
            _ => Dd::acos,
        };
        check(case, |r, i| f(dd_input(r, "x")[i]), |_, _, exact| 2.0 * TOL * exact.abs().hi());
    }
}

#[test]
fn atan2() {
    check(
        "atan2",
        |r, i| dd_input(r, "y")[i].atan2(dd_input(r, "x")[i]),
        |_, _, exact| 2.0 * TOL * exact.abs().hi(),
    );
}

#[test]
fn special_values() {
    let d = Dd::from_f64;
    let pi = Dd::new(std::f64::consts::PI, 1.2246467991473532e-16);
    let half_pi = Dd::new(std::f64::consts::FRAC_PI_2, 6.123233995736766e-17);
    assert_eq!(d(0.0).sin(), d(0.0));
    assert_eq!(d(0.0).cos(), Dd::ONE);
    assert!(Dd::NAN.sin().hi().is_nan());
    assert!(Dd::INFINITY.cos().hi().is_nan());
    assert!(d(2f64.powi(51)).sin().hi().is_nan());
    assert_eq!(d(0.0).atan2(d(-1.0)), pi);
    assert_eq!(d(1.0).atan2(d(0.0)), half_pi);
    let neg_zero = d(-0.0).atan2(d(1.0));
    assert!(neg_zero.hi() == 0.0 && neg_zero.hi().is_sign_negative());
    assert_eq!(d(1.0).asin(), half_pi);
    assert_eq!(d(-1.0).acos(), pi);
    assert!(d(2.0).acos().hi().is_nan());
    assert_eq!(Dd::INFINITY.atan(), half_pi);
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test dd_trig`
Expected: compile errors, `no method named sin`.

- [ ] **Step 4: Implement**

Add `mod trig;` to `scatrs/src/scalar/dd/mod.rs`.

`scatrs/src/scalar/dd/trig.rs`:
```rust
//! Trigonometric functions for `DoubleDouble`.
//!
//! * `sin`/`cos`: Cody–Waite reduction x = k·π/2 + r with a three-word π/2 (exact products
//!   k·P₁ and k·P₂), valid for |x| ≤ 2⁵⁰; Taylor series on |r| ≤ π/4 to order 29.
//! * `atan2`: one Newton step from the `f64` result, on sin z = y/r or cos z = x/r,
//!   whichever is better conditioned. `asin`/`acos`/`atan` reduce to `atan2`.

use super::consts::{FRAC_2_PI, FRAC_3PI_4, FRAC_PI_2, FRAC_PI_2_TAIL, FRAC_PI_4, INV_FACT, PI};
use super::{two_prod, DoubleDouble as Dd};

/// Largest |x| accepted by the argument reduction (2⁵⁰).
const MAX_REDUCIBLE: f64 = 1_125_899_906_842_624.0;

/// Returns (k, r) with x = k·π/2 + r and |r| ≲ π/4, or `None` for NaN, ±∞ and |x| > 2⁵⁰.
fn reduce_pio2(x: Dd) -> Option<(i64, Dd)> {
    if x.hi().is_nan() || x.hi().abs() > MAX_REDUCIBLE {
        return None;
    }
    if x.hi().abs() <= FRAC_PI_4.hi() {
        return Some((0, x));
    }
    let k = (x.hi() * FRAC_2_PI.hi()).round();
    let (a, b) = two_prod(k, FRAC_PI_2.hi());
    let (c, d) = two_prod(k, FRAC_PI_2.lo());
    let r = x - Dd::from_parts_unchecked(a, b) - Dd::from_parts_unchecked(c, d) - k * FRAC_PI_2_TAIL;
    Some((k as i64, r))
}

/// sin r for |r| ≲ π/4: r − r·r²·(1/3! − r²(1/5! − … r²/29!)).
fn sin_taylor(r: Dd) -> Dd {
    let r2 = r * r;
    let mut p = INV_FACT[29];
    for k in (3..=27).rev().step_by(2) {
        p = INV_FACT[k] - r2 * p;
    }
    r - r * r2 * p
}

/// cos r for |r| ≲ π/4: 1 − r²·(1/2! − r²(1/4! − … r²/28!)).
fn cos_taylor(r: Dd) -> Dd {
    let r2 = r * r;
    let mut p = INV_FACT[28];
    for k in (2..=26).rev().step_by(2) {
        p = INV_FACT[k] - r2 * p;
    }
    Dd::ONE - r2 * p
}

/// IEEE special cases of atan2 (a zero or infinite argument), mapped to exact constants.
fn special_atan2(y: f64, x: f64) -> Dd {
    let a = y.atan2(x);
    let exact = [FRAC_PI_4, FRAC_PI_2, FRAC_3PI_4, PI].into_iter().find(|c| c.hi() == a.abs());
    match exact {
        Some(c) if a < 0.0 => -c,
        Some(c) => c,
        None => Dd::from_f64(a), // ±0
    }
}

impl Dd {
    /// Sine (NaN for |x| > 2⁵⁰).
    pub fn sin(self) -> Self {
        let Some((k, r)) = reduce_pio2(self) else {
            return Self::NAN;
        };
        match k.rem_euclid(4) {
            0 => sin_taylor(r),
            1 => cos_taylor(r),
            2 => -sin_taylor(r),
            _ => -cos_taylor(r),
        }
    }

    /// Cosine (NaN for |x| > 2⁵⁰).
    pub fn cos(self) -> Self {
        let Some((k, r)) = reduce_pio2(self) else {
            return Self::NAN;
        };
        match k.rem_euclid(4) {
            0 => cos_taylor(r),
            1 => -sin_taylor(r),
            2 => -cos_taylor(r),
            _ => sin_taylor(r),
        }
    }

    /// `(sin x, cos x)`.
    pub fn sin_cos(self) -> (Self, Self) {
        let Some((k, r)) = reduce_pio2(self) else {
            return (Self::NAN, Self::NAN);
        };
        let (s, c) = (sin_taylor(r), cos_taylor(r));
        match k.rem_euclid(4) {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }

    /// Tangent.
    pub fn tan(self) -> Self {
        let (s, c) = self.sin_cos();
        s / c
    }

    /// Four-quadrant arctangent of `self / x` (`self` is y).
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        if y.hi().is_nan() || x.hi().is_nan() {
            return Self::NAN;
        }
        if !y.hi().is_finite() || !x.hi().is_finite() || y.hi() == 0.0 || x.hi() == 0.0 {
            return special_atan2(y.hi(), x.hi());
        }
        let r = y.hypot(x);
        let (yy, xx) = (y / r, x / r);
        let z = Self::from_f64(y.hi().atan2(x.hi()));
        let (s, c) = z.sin_cos();
        if xx.hi().abs() > yy.hi().abs() {
            z + (yy - s) / c
        } else {
            z - (xx - c) / s
        }
    }

    /// Arctangent.
    pub fn atan(self) -> Self {
        self.atan2(Self::ONE)
    }

    /// Arcsine (`|x| <= 1`).
    pub fn asin(self) -> Self {
        let a = self.abs();
        if self.hi().is_nan() || a > Self::ONE {
            return Self::NAN;
        }
        if a == Self::ONE {
            return if self.hi() > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
        }
        self.atan2(((Self::ONE - self) * (Self::ONE + self)).sqrt())
    }

    /// Arccosine (`|x| <= 1`).
    pub fn acos(self) -> Self {
        if self.hi().is_nan() || self.abs() > Self::ONE {
            return Self::NAN;
        }
        ((Self::ONE - self) * (Self::ONE + self)).sqrt().atan2(self)
    }
}
```

- [ ] **Step 5: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass. Report the worst-error lines from `cargo test -p scatrs --test dd_trig -- --nocapture`. The "do not loosen tolerances" rule from Task 4 applies.

- [ ] **Step 6: Commit**

```bash
git add scatrs/src/scalar/dd reference/scatrs_reference/generators reference/data scatrs/tests/dd_trig.rs
git commit -m "feat: DoubleDouble trigonometric functions validated against Arb"
```

---

### Task 6: `DoubleDouble` as a `Real` (num-traits implementations and decimal parsing)

**Files:**
- Create: `scatrs/src/scalar/dd/parse.rs`, `scatrs/src/scalar/dd/traits.rs`
- Modify: `scatrs/src/scalar/dd/mod.rs`:
  - add `mod parse; mod traits; pub use parse::ParseDoubleDoubleError;`;
  - re-export `ParseDoubleDoubleError` from `scalar/mod.rs` too.
- Modify: `scatrs/src/scalar/dd/consts.rs` (remove `#![allow(dead_code)]`; every constant is now used)
- Create: `reference/scatrs_reference/generators/dd_parse.py`; register it
- Generate: `reference/data/dd_parse.jsonl`
- Test: `scatrs/tests/dd_traits.rs`

**Interfaces:**
- Consumes: everything from Tasks 3–5.
- Produces:
  - `impl Real for DoubleDouble`, plus `Zero`, `One`, `Num` (`FromStrRadixErr = ParseDoubleDoubleError`, radix 10 only), `NumCast`, `ToPrimitive`, `Float` and `FloatConst`.
  - `DoubleDouble::from_decimal_str(&str) -> Result<DoubleDouble, ParseDoubleDoubleError>`.
  - `pub struct ParseDoubleDoubleError` (`Debug, Clone, PartialEq, Eq, Display, std::error::Error`).
- `Float` constants:
  - `epsilon()` = 2⁻¹⁰⁴;
  - `min_positive_value()` = 2⁻⁹⁶⁹, the smallest magnitude with full double-double precision;
  - `max_value()` = `f64::MAX`.
- The rounding, classification and sign methods follow `f64` semantics on the leading word.

- [ ] **Step 1: Add the reference generator**

`reference/scatrs_reference/generators/dd_parse.py`:
```python
"""Exact values of random decimal literals, for DoubleDouble::from_decimal_str."""

from __future__ import annotations

import random
from fractions import Fraction

from flint import arb, ctx, fmpq

from scatrs_reference.records import record

PRECISION_BITS = 600
N = 300
SEED = 6


def _literal(rng: random.Random) -> tuple[str, Fraction]:
    while True:
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 40))).lstrip("0") or "1"
        point = rng.randint(0, len(digits))
        mantissa = digits[:point] + "." + digits[point:] if rng.random() < 0.7 else digits
        frac_digits = len(digits) - point if "." in mantissa else 0
        exponent = rng.randint(-280, 280)
        value = Fraction(int(digits), 10**frac_digits) * Fraction(10) ** exponent
        if Fraction(1, 10**289) < value < Fraction(10**289):
            sign = rng.choice(("", "-", "+"))
            text = f"{sign}{mantissa}e{exponent}"
            return text, -value if sign == "-" else value


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    rng = random.Random(SEED)
    pairs = [_literal(rng) for _ in range(N)]
    values = [arb(fmpq(v.numerator, v.denominator)) for _, v in pairs]
    return PRECISION_BITS, [record("dd_parse", "literals", {"text": [t for t, _ in pairs]}, {"value": values})]
```

Register it. Then:

Run: `uv run --directory reference python -m scatrs_reference generate --only dd_parse && uv run --directory reference python -m pytest -q`
Expected: 1 record; the tests pass.

- [ ] **Step 2: Write the failing Rust tests**

`scatrs/tests/dd_traits.rs`:
```rust
mod common;

use common::{find, load};
use num_complex::Complex;
use num_traits::{Float, FloatConst, Num, NumCast, ToPrimitive};
use proptest::prelude::*;
use scatrs::Real;
use scatrs::scalar::DoubleDouble as Dd;

/// 2^-100.
const TOL: f64 = 7.888_609_052_210_118e-31;

fn horner<T: Real>(coeffs: &[f64], x: T) -> T {
    coeffs.iter().rev().fold(T::zero(), |acc, &c| acc * x + c)
}

#[test]
fn float_const_matches_reference() {
    let recs = load("constants.jsonl");
    let table: [(&str, Dd); 19] = [
        ("PI", Dd::PI()),
        ("TAU", Dd::TAU()),
        ("FRAC_PI_2", Dd::FRAC_PI_2()),
        ("FRAC_PI_3", Dd::FRAC_PI_3()),
        ("FRAC_PI_4", Dd::FRAC_PI_4()),
        ("FRAC_PI_6", Dd::FRAC_PI_6()),
        ("FRAC_PI_8", Dd::FRAC_PI_8()),
        ("FRAC_1_PI", Dd::FRAC_1_PI()),
        ("FRAC_2_PI", Dd::FRAC_2_PI()),
        ("FRAC_2_SQRT_PI", Dd::FRAC_2_SQRT_PI()),
        ("SQRT_2", Dd::SQRT_2()),
        ("FRAC_1_SQRT_2", Dd::FRAC_1_SQRT_2()),
        ("E", Dd::E()),
        ("LN_2", Dd::LN_2()),
        ("LN_10", Dd::LN_10()),
        ("LOG2_E", Dd::LOG2_E()),
        ("LOG10_E", Dd::LOG10_E()),
        ("LOG10_2", Dd::LOG10_2()),
        ("LOG2_10", Dd::LOG2_10()),
    ];
    for (name, value) in table {
        let v = find(&recs, name).output("value");
        assert_eq!((value.hi(), value.lo()), (v.hi[0], v.lo[0]), "{name}");
    }
}

#[test]
fn complex_double_double_arithmetic() {
    let d = Dd::from_f64;
    let a = Complex::new(d(1.0), d(2.0));
    let b = Complex::new(d(3.0), d(-1.0));
    assert_eq!(a * b, Complex::new(d(5.0), d(5.0)));
    let back = (a * b) / b;
    assert!((back - a).norm_sqr().sqrt().hi() <= TOL);
    let minus_one = Complex::new(Dd::ZERO, Dd::PI()).exp();
    assert!((minus_one + Complex::new(Dd::ONE, Dd::ZERO)).norm_sqr().sqrt().hi() <= TOL);
}

#[test]
fn generic_kernel_runs_in_double_double() {
    let x = Dd::from_f64(1.0) / Dd::from_f64(3.0);
    let p = horner(&[1.0, 2.0, 3.0], x);
    let expected = Dd::from_f64(2.0); // 1 + 2/3 + 3/9 = 2
    assert!((p - expected).abs().hi() <= 4.0 * TOL);
    assert_eq!(<Dd as Real>::from_f64_pair(1.0, 2f64.powi(-60)).lo(), 2f64.powi(-60));
    assert_eq!(Real::approx_f64(Dd::new(1.0, 2f64.powi(-60))), 1.0);
}

#[test]
fn decimal_parsing_matches_reference() {
    let recs = load("dd_parse.jsonl");
    let r = find(&recs, "literals");
    let v = r.output("value");
    let mut worst = 0.0_f64;
    for (i, text) in r.input_strs("text").iter().enumerate() {
        let got = Dd::from_decimal_str(text).expect("valid literal");
        let exact = Dd::new(v.hi[i], v.lo[i]);
        let rel = ((got - exact).abs().hi() + v.err[i]) / exact.abs().hi();
        worst = worst.max(rel);
        assert!(rel <= 4.0 * TOL, "{text}: got {got:?}, reference {exact:?}, rel {rel:e}");
    }
    eprintln!("parse: worst relative error {worst:e}");
}

#[test]
fn decimal_parsing_edge_cases() {
    for bad in ["", "abc", "1e", "1.2.3", "--1", "e5", "."] {
        assert!(Dd::from_decimal_str(bad).is_err(), "{bad:?} should not parse");
    }
    assert_eq!(Dd::from_decimal_str("inf").unwrap().hi(), f64::INFINITY);
    assert_eq!(Dd::from_decimal_str("-Infinity").unwrap().hi(), f64::NEG_INFINITY);
    assert!(Dd::from_decimal_str("NaN").unwrap().hi().is_nan());
    assert_eq!(Dd::from_decimal_str("+0.0").unwrap(), Dd::ZERO);
    assert_eq!(Dd::from_decimal_str("1e-400").unwrap(), Dd::ZERO);
    assert_eq!(Dd::from_decimal_str("1e400").unwrap().hi(), f64::INFINITY);
    // Decimal fractions go through a double-double division, so compare with a tolerance.
    let half = Dd::from_decimal_str(".5").unwrap();
    assert!((half - Dd::from_f64(0.5)).abs().hi() <= TOL * 0.5);
    assert!(<Dd as Num>::from_str_radix("10", 2).is_err());
    let two_and_a_half = <Dd as Num>::from_str_radix("2.5", 10).unwrap();
    assert!((two_and_a_half - Dd::from_f64(2.5)).abs().hi() <= TOL * 2.5);
}

#[test]
fn num_cast_and_to_primitive() {
    assert_eq!(<Dd as NumCast>::from(2.5_f64), Some(Dd::from_f64(2.5)));
    let big = <Dd as NumCast>::from(i64::MAX).unwrap();
    assert_eq!(big.to_i64(), Some(i64::MAX));
    assert_eq!(<Dd as NumCast>::from(u64::MAX).unwrap().to_u64(), Some(u64::MAX));
    assert_eq!(Dd::from_f64(-3.7).to_i64(), Some(-3));
    assert_eq!(Dd::from_f64(-1.0).to_u64(), None);
    assert_eq!(Dd::from_f64(1e30).to_i64(), None);
    assert_eq!(Dd::new(1.0, 2f64.powi(-60)).to_f64(), Some(1.0));
}

#[test]
fn float_constants_and_limits() {
    assert_eq!(<Dd as Float>::epsilon().hi(), 2f64.powi(-104));
    assert_eq!(<Dd as Float>::min_positive_value().hi(), 2f64.powi(-969));
    assert_eq!(<Dd as Float>::max_value().hi(), f64::MAX);
    assert!(<Dd as Float>::nan().is_nan());
    assert!(<Dd as Float>::neg_zero().is_sign_negative());
    assert_eq!(Float::max(Dd::NAN, Dd::ONE), Dd::ONE);
    assert_eq!(Float::min(Dd::ONE, Dd::NAN), Dd::ONE);
    let deg = Float::to_degrees(Dd::PI());
    assert!((deg - Dd::from_f64(180.0)).abs().hi() <= 180.0 * TOL);
}

proptest! {
    #[test]
    fn float_semantics_match_f64(x in prop::num::f64::NORMAL | prop::num::f64::ZERO) {
        let d = Dd::from_f64(x);
        prop_assert_eq!(Float::floor(d).hi(), x.floor());
        prop_assert_eq!(Float::ceil(d).hi(), x.ceil());
        prop_assert_eq!(Float::round(d).hi(), x.round());
        prop_assert_eq!(Float::trunc(d).hi(), x.trunc());
        prop_assert_eq!(Float::fract(d).hi(), x.fract());
        prop_assert_eq!(Float::abs(d).hi(), x.abs());
        prop_assert_eq!(Float::signum(d).hi(), x.signum());
        prop_assert_eq!(Float::classify(d), x.classify());
        prop_assert_eq!(Float::is_sign_negative(d), x.is_sign_negative());
        prop_assert_eq!(Float::is_finite(d), x.is_finite());
    }
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test dd_traits`
Expected: compile errors, `the trait Float is not implemented for DoubleDouble`.

- [ ] **Step 4: Implement parsing**

`scatrs/src/scalar/dd/parse.rs`:
```rust
//! Decimal literal parsing for `DoubleDouble`.

use core::fmt;

use super::consts::POW10_2I;
use super::DoubleDouble as Dd;

/// Error returned when a string is not a decimal floating-point literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDoubleDoubleError;

impl fmt::Display for ParseDoubleDoubleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid double-double literal")
    }
}

impl std::error::Error for ParseDoubleDoubleError {}

/// 10^e for 0 <= e <= 511, from the exact/certified table 10^(2^i) (≤ 9 products).
fn pow10(e: u32) -> Dd {
    let mut result = Dd::ONE;
    for (i, p) in POW10_2I.iter().enumerate() {
        if e & (1 << i) != 0 {
            result *= *p;
        }
    }
    result
}

/// v · 10^e, split into steps of at most 300 so intermediate powers stay finite.
fn scale_pow10(v: Dd, e: i64) -> Dd {
    let mut v = v;
    let mut e = e.clamp(-2000, 2000);
    while e > 300 {
        v *= pow10(300);
        e -= 300;
    }
    while e < -300 {
        v /= pow10(300);
        e += 300;
    }
    if e >= 0 { v * pow10(e as u32) } else { v / pow10((-e) as u32) }
}

impl Dd {
    /// Parses `[+-]digits[.digits][(e|E)[+-]digits]`, `inf`, `infinity` or `nan`
    /// (case-insensitive).
    ///
    /// Up to 31 significant digits are used. Within [10⁻²⁸⁹, 10²⁸⁹] the relative error is
    /// below 2⁻⁹⁸.
    pub fn from_decimal_str(s: &str) -> Result<Self, ParseDoubleDoubleError> {
        let t = s.trim();
        let (negative, body) = match t.as_bytes().first() {
            Some(b'-') => (true, &t[1..]),
            Some(b'+') => (false, &t[1..]),
            _ => (false, t),
        };
        let lower = body.to_ascii_lowercase();
        if lower == "inf" || lower == "infinity" {
            return Ok(if negative { Self::NEG_INFINITY } else { Self::INFINITY });
        }
        if lower == "nan" {
            return Ok(Self::NAN);
        }
        let (mantissa, exponent) = match body.find(['e', 'E']) {
            Some(i) => (&body[..i], Some(&body[i + 1..])),
            None => (body, None),
        };
        let mut exp10: i64 = match exponent {
            Some(e) => e.parse().map_err(|_| ParseDoubleDoubleError)?,
            None => 0,
        };
        let mut value = Self::ZERO;
        let mut significant = 0u32;
        let mut digits_seen = false;
        let mut seen_point = false;
        for ch in mantissa.chars() {
            match ch {
                '0'..='9' => {
                    digits_seen = true;
                    let d = f64::from(ch as u8 - b'0');
                    if significant < 31 {
                        if significant > 0 || d != 0.0 {
                            significant += 1;
                        }
                        value = value * 10.0 + d;
                        if seen_point {
                            exp10 -= 1;
                        }
                    } else if !seen_point {
                        exp10 += 1;
                    }
                }
                '.' if !seen_point => seen_point = true,
                _ => return Err(ParseDoubleDoubleError),
            }
        }
        if !digits_seen {
            return Err(ParseDoubleDoubleError);
        }
        let v = if value.hi() == 0.0 { value } else { scale_pow10(value, exp10) };
        Ok(if negative { -v } else { v })
    }
}
```

- [ ] **Step 5: Implement the traits**

`scatrs/src/scalar/dd/traits.rs`:
```rust
//! num-traits and `Real` implementations for `DoubleDouble`.

use core::num::FpCategory;

use num_traits::{Float, FloatConst, Num, NumCast, One, ToPrimitive, Zero};

use super::consts;
use super::{DoubleDouble as Dd, ParseDoubleDoubleError};
use crate::scalar::Real;

/// 2⁻¹⁰⁴: the double-double unit used for `Float::epsilon`.
const EPSILON: f64 = 4.930_380_657_631_324e-32;
/// 2⁻⁹⁶⁹: below this the trailing word becomes subnormal and precision is lost.
const MIN_POSITIVE_FULL_PRECISION: f64 = 2.004_168_360_008_973e-292;

impl Zero for Dd {
    fn zero() -> Self {
        Dd::ZERO
    }
    fn is_zero(&self) -> bool {
        self.hi() == 0.0
    }
}

impl One for Dd {
    fn one() -> Self {
        Dd::ONE
    }
}

impl Num for Dd {
    type FromStrRadixErr = ParseDoubleDoubleError;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err(ParseDoubleDoubleError);
        }
        Dd::from_decimal_str(s)
    }
}

impl ToPrimitive for Dd {
    fn to_i64(&self) -> Option<i64> {
        i64::try_from(trunc_to_i128(*self)?).ok()
    }
    fn to_u64(&self) -> Option<u64> {
        u64::try_from(trunc_to_i128(*self)?).ok()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.hi())
    }
}

/// Integer part as i128 (both words of a truncated value are integers), if it is small enough.
fn trunc_to_i128(x: Dd) -> Option<i128> {
    let t = Dd::trunc(x);
    if !t.hi().is_finite() || t.hi().abs() > 1e30 {
        return None;
    }
    Some(t.hi() as i128 + t.lo() as i128)
}

impl NumCast for Dd {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let f = n.to_f64()?;
        if f.is_finite() && f.abs() >= 9_007_199_254_740_992.0 && f == f.trunc() {
            if let Some(i) = n.to_i64() {
                let hi = i as f64;
                return Some(Dd::new(hi, (i128::from(i) - hi as i128) as f64));
            }
            if let Some(u) = n.to_u64() {
                let hi = u as f64;
                return Some(Dd::new(hi, (i128::from(u) - hi as i128) as f64));
            }
        }
        Some(Dd::from_f64(f))
    }
}

impl Float for Dd {
    fn nan() -> Self {
        Dd::NAN
    }
    fn infinity() -> Self {
        Dd::INFINITY
    }
    fn neg_infinity() -> Self {
        Dd::NEG_INFINITY
    }
    fn neg_zero() -> Self {
        Dd::from_f64(-0.0)
    }
    fn min_value() -> Self {
        Dd::from_f64(f64::MIN)
    }
    fn min_positive_value() -> Self {
        Dd::from_f64(MIN_POSITIVE_FULL_PRECISION)
    }
    fn epsilon() -> Self {
        Dd::from_f64(EPSILON)
    }
    fn max_value() -> Self {
        Dd::from_f64(f64::MAX)
    }
    fn is_nan(self) -> bool {
        self.hi().is_nan()
    }
    fn is_infinite(self) -> bool {
        self.hi().is_infinite()
    }
    fn is_finite(self) -> bool {
        self.hi().is_finite()
    }
    fn is_normal(self) -> bool {
        self.hi().is_normal()
    }
    fn classify(self) -> FpCategory {
        self.hi().classify()
    }
    fn floor(self) -> Self {
        Dd::floor(self)
    }
    fn ceil(self) -> Self {
        Dd::ceil(self)
    }
    fn round(self) -> Self {
        Dd::round(self)
    }
    fn trunc(self) -> Self {
        Dd::trunc(self)
    }
    fn fract(self) -> Self {
        self - Dd::trunc(self)
    }
    fn abs(self) -> Self {
        Dd::abs(self)
    }
    fn signum(self) -> Self {
        if self.hi().is_nan() { Dd::NAN } else { Dd::from_f64(self.hi().signum()) }
    }
    fn is_sign_positive(self) -> bool {
        self.hi().is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.hi().is_sign_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    fn recip(self) -> Self {
        Dd::recip(self)
    }
    fn powi(self, n: i32) -> Self {
        Dd::powi(self, n)
    }
    fn powf(self, n: Self) -> Self {
        Dd::powf(self, n)
    }
    fn sqrt(self) -> Self {
        Dd::sqrt(self)
    }
    fn exp(self) -> Self {
        Dd::exp(self)
    }
    fn exp2(self) -> Self {
        Dd::exp2(self)
    }
    fn ln(self) -> Self {
        Dd::ln(self)
    }
    fn log(self, base: Self) -> Self {
        Dd::log(self, base)
    }
    fn log2(self) -> Self {
        Dd::log2(self)
    }
    fn log10(self) -> Self {
        Dd::log10(self)
    }
    fn max(self, other: Self) -> Self {
        if self.hi().is_nan() || (!other.hi().is_nan() && other > self) { other } else { self }
    }
    fn min(self, other: Self) -> Self {
        if self.hi().is_nan() || (!other.hi().is_nan() && other < self) { other } else { self }
    }
    fn abs_sub(self, other: Self) -> Self {
        if self <= other { Dd::ZERO } else { self - other }
    }
    fn cbrt(self) -> Self {
        Dd::cbrt(self)
    }
    fn hypot(self, other: Self) -> Self {
        Dd::hypot(self, other)
    }
    fn sin(self) -> Self {
        Dd::sin(self)
    }
    fn cos(self) -> Self {
        Dd::cos(self)
    }
    fn tan(self) -> Self {
        Dd::tan(self)
    }
    fn asin(self) -> Self {
        Dd::asin(self)
    }
    fn acos(self) -> Self {
        Dd::acos(self)
    }
    fn atan(self) -> Self {
        Dd::atan(self)
    }
    fn atan2(self, other: Self) -> Self {
        Dd::atan2(self, other)
    }
    fn sin_cos(self) -> (Self, Self) {
        Dd::sin_cos(self)
    }
    fn exp_m1(self) -> Self {
        Dd::exp_m1(self)
    }
    fn ln_1p(self) -> Self {
        Dd::ln_1p(self)
    }
    fn sinh(self) -> Self {
        Dd::sinh(self)
    }
    fn cosh(self) -> Self {
        Dd::cosh(self)
    }
    fn tanh(self) -> Self {
        Dd::tanh(self)
    }
    fn asinh(self) -> Self {
        Dd::asinh(self)
    }
    fn acosh(self) -> Self {
        Dd::acosh(self)
    }
    fn atanh(self) -> Self {
        Dd::atanh(self)
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self.hi())
    }
    fn to_degrees(self) -> Self {
        self * 180.0 / consts::PI
    }
    fn to_radians(self) -> Self {
        self * consts::PI / 180.0
    }
}

impl FloatConst for Dd {
    fn E() -> Self {
        consts::E
    }
    fn FRAC_1_PI() -> Self {
        consts::FRAC_1_PI
    }
    fn FRAC_1_SQRT_2() -> Self {
        consts::FRAC_1_SQRT_2
    }
    fn FRAC_2_PI() -> Self {
        consts::FRAC_2_PI
    }
    fn FRAC_2_SQRT_PI() -> Self {
        consts::FRAC_2_SQRT_PI
    }
    fn FRAC_PI_2() -> Self {
        consts::FRAC_PI_2
    }
    fn FRAC_PI_3() -> Self {
        consts::FRAC_PI_3
    }
    fn FRAC_PI_4() -> Self {
        consts::FRAC_PI_4
    }
    fn FRAC_PI_6() -> Self {
        consts::FRAC_PI_6
    }
    fn FRAC_PI_8() -> Self {
        consts::FRAC_PI_8
    }
    fn LN_10() -> Self {
        consts::LN_10
    }
    fn LN_2() -> Self {
        consts::LN_2
    }
    fn LOG10_E() -> Self {
        consts::LOG10_E
    }
    fn LOG2_E() -> Self {
        consts::LOG2_E
    }
    fn PI() -> Self {
        consts::PI
    }
    fn SQRT_2() -> Self {
        consts::SQRT_2
    }
    fn TAU() -> Self {
        consts::TAU
    }
    fn LOG10_2() -> Self {
        consts::LOG10_2
    }
    fn LOG2_10() -> Self {
        consts::LOG2_10
    }
}

impl Real for Dd {
    fn from_f64(x: f64) -> Self {
        Dd::from_f64(x)
    }
    fn from_f64_pair(hi: f64, lo: f64) -> Self {
        Dd::new(hi, lo)
    }
    fn approx_f64(self) -> f64 {
        self.hi()
    }
}
```

If the compiler reports a missing required `Float` method that is not listed here (num-traits may add some), implement it with `f64` semantics on the leading word, or with the matching inherent method. Do not use `unimplemented!()`.

- [ ] **Step 6: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass, including the constants unit test, which is now built without `allow(dead_code)`.

- [ ] **Step 7: Commit**

```bash
git add scatrs/src/scalar reference/scatrs_reference/generators reference/data scatrs/tests/dd_traits.rs
git commit -m "feat: DoubleDouble implements Real (num-traits Float/FloatConst, decimal parsing)"
```

---

### Task 7: `Dual<T, N>` arithmetic

**Files:**
- Create: `scatrs/src/scalar/dual/mod.rs`
- Modify: `scatrs/src/scalar/mod.rs` (add `mod dual; pub use dual::Dual;`)
- Test: `scatrs/tests/dual_arith.rs`

**Interfaces:**
- Consumes: `Real` (Task 1) and `DoubleDouble` (Task 6).
- Produces: `scatrs::scalar::Dual<T, const N: usize>` (`Clone, Copy, Debug`), for `T: Real`:
  - **Construction:**
    - `fn new(re: T, eps: [T; N]) -> Self`
    - `fn constant(re: T) -> Self`
    - `fn variables(values: [T; N]) -> [Self; N]`, where element i has derivative 1 in direction i and 0 elsewhere.
  - **Access:** `fn re(&self) -> T`, `fn eps(&self) -> &[T; N]`.
  - **Crate-internal:** `pub(crate) fn chain(self, f: T, df: T) -> Self`.
  - **Comparisons:** `PartialEq` and `PartialOrd` compare the value `re` only.
  - **Operators:**
    - `+ - * / %` and `-x` with `Self`;
    - `+ - * /` with `f64`;
    - `*Assign` with `Self` and `f64`, and `RemAssign<Self>`.
  - **num-traits:** `Zero` (`is_zero` looks at `re` only), `One`, `Num` (`FromStrRadixErr = T::FromStrRadixErr`), `NumCast`, `ToPrimitive` (from `re`).

- [ ] **Step 1: Write the failing tests**

`scatrs/tests/dual_arith.rs`:
```rust
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use proptest::prelude::*;
use scatrs::scalar::{DoubleDouble as Dd, Dual};

type D2 = Dual<f64, 2>;

#[test]
fn product_quotient_and_sum_rules() {
    let [x, y] = D2::variables([3.0, 2.0]);
    let f = x * y + x / y - y;
    assert_eq!(f.re(), 3.0 * 2.0 + 1.5 - 2.0);
    assert_eq!(f.eps(), &[2.0 + 0.5, 3.0 - 0.75 - 1.0]);
}

#[test]
fn mixed_f64_operations() {
    let [x, _] = D2::variables([3.0, 2.0]);
    let f = (x * 2.0 + 1.0 - 0.5) / 4.0;
    assert_eq!(f.re(), 1.625);
    assert_eq!(f.eps(), &[0.5, 0.0]);
    let mut g = x;
    g += x;
    g *= 2.0;
    g -= 1.0;
    g /= x;
    assert_eq!(g.re(), 11.0 / 3.0);
    assert!((g.eps()[0] - 1.0 / 9.0).abs() < 1e-16);
}

#[test]
fn negation_and_remainder() {
    let [x] = Dual::<f64, 1>::variables([7.5]);
    assert_eq!((-x).eps(), &[-1.0]);
    let r = x % Dual::constant(2.0);
    assert_eq!(r.re(), 1.5);
    assert_eq!(r.eps(), &[1.0]);
}

#[test]
fn comparisons_use_value_only() {
    assert_eq!(D2::new(1.0, [5.0, 0.0]), D2::new(1.0, [0.0, 0.0]));
    assert!(D2::new(1.0, [9.0, 9.0]) < D2::new(2.0, [0.0, 0.0]));
    assert!(D2::new(0.0, [1.0, 0.0]).is_zero());
}

#[test]
fn num_traits() {
    assert_eq!(D2::one().re(), 1.0);
    assert_eq!(<D2 as Num>::from_str_radix("2.5", 10).unwrap().re(), 2.5);
    assert_eq!(<D2 as NumCast>::from(3_i32).unwrap().eps(), &[0.0, 0.0]);
    assert_eq!(D2::new(-3.7, [1.0, 1.0]).to_i64(), Some(-3));
}

#[test]
fn works_over_double_double() {
    let [x, y] = Dual::<Dd, 2>::variables([Dd::from_f64(3.0), Dd::from_f64(2.0)]);
    let f = x * y;
    assert_eq!(f.eps(), &[Dd::from_f64(2.0), Dd::from_f64(3.0)]);
}

proptest! {
    #[test]
    fn cube_derivative(x in -1e3..1e3f64) {
        let [v] = Dual::<f64, 1>::variables([x]);
        let c = v * v * v;
        let expected = 3.0 * x * x;
        prop_assert!((c.eps()[0] - expected).abs() <= 4.0 * f64::EPSILON * expected.abs());
    }
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test dual_arith`
Expected: a compile error, `no Dual in scalar`.

- [ ] **Step 3: Implement**

In `scatrs/src/scalar/mod.rs`, add `mod dual;` and `pub use dual::Dual;` next to the `dd` lines.

`scatrs/src/scalar/dual/mod.rs`:
```rust
//! Forward-mode dual numbers carrying `N` first-order directional derivatives.

use core::array;
use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

use crate::scalar::Real;

/// A value with `N` first-order derivatives: `re + Σᵢ eps[i]·εᵢ`.
///
/// Comparisons and `is_zero` look only at the value `re`. `Float::is_nan` and
/// `Float::is_finite` also inspect the derivatives, so a NaN derivative is never silent.
#[derive(Clone, Copy, Debug)]
pub struct Dual<T, const N: usize> {
    re: T,
    eps: [T; N],
}

impl<T: Real, const N: usize> Dual<T, N> {
    /// A dual number with the given value and derivatives.
    #[inline]
    pub fn new(re: T, eps: [T; N]) -> Self {
        Self { re, eps }
    }

    /// A constant: all derivatives zero.
    #[inline]
    pub fn constant(re: T) -> Self {
        Self { re, eps: [T::zero(); N] }
    }

    /// Seeds `N` independent variables: element `i` has value `values[i]` and derivative 1
    /// in direction `i` only.
    pub fn variables(values: [T; N]) -> [Self; N] {
        array::from_fn(|i| {
            let mut eps = [T::zero(); N];
            eps[i] = T::one();
            Self { re: values[i], eps }
        })
    }

    /// The value.
    #[inline]
    pub fn re(&self) -> T {
        self.re
    }

    /// The derivatives.
    #[inline]
    pub fn eps(&self) -> &[T; N] {
        &self.eps
    }

    /// Chain rule: value `f`, derivatives scaled by `df` = f′(re).
    #[inline]
    pub(crate) fn chain(self, f: T, df: T) -> Self {
        Self { re: f, eps: self.eps.map(|e| e * df) }
    }
}

impl<T: Real, const N: usize> Neg for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { re: -self.re, eps: self.eps.map(|e| -e) }
    }
}

impl<T: Real, const N: usize> Add for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self { re: self.re + o.re, eps: array::from_fn(|i| self.eps[i] + o.eps[i]) }
    }
}

impl<T: Real, const N: usize> Sub for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self { re: self.re - o.re, eps: array::from_fn(|i| self.eps[i] - o.eps[i]) }
    }
}

impl<T: Real, const N: usize> Mul for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re,
            eps: array::from_fn(|i| self.eps[i] * o.re + self.re * o.eps[i]),
        }
    }
}

impl<T: Real, const N: usize> Div for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = T::one() / o.re;
        let q = self.re * inv;
        Self { re: q, eps: array::from_fn(|i| (self.eps[i] - q * o.eps[i]) * inv) }
    }
}

impl<T: Real, const N: usize> Rem for Dual<T, N> {
    type Output = Self;
    fn rem(self, o: Self) -> Self {
        let t = (self.re / o.re).trunc();
        Self { re: self.re % o.re, eps: array::from_fn(|i| self.eps[i] - t * o.eps[i]) }
    }
}

impl<T: Real, const N: usize> Add<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, y: f64) -> Self {
        Self { re: self.re + y, eps: self.eps }
    }
}

impl<T: Real, const N: usize> Sub<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, y: f64) -> Self {
        Self { re: self.re - y, eps: self.eps }
    }
}

impl<T: Real, const N: usize> Mul<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, y: f64) -> Self {
        Self { re: self.re * y, eps: self.eps.map(|e| e * y) }
    }
}

impl<T: Real, const N: usize> Div<f64> for Dual<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, y: f64) -> Self {
        Self { re: self.re / y, eps: self.eps.map(|e| e / y) }
    }
}

macro_rules! assign_ops {
    ($($trait:ident, $method:ident, $op:tt);* $(;)?) => {$(
        impl<T: Real, const N: usize> $trait for Dual<T, N> {
            #[inline]
            fn $method(&mut self, o: Self) {
                *self = *self $op o;
            }
        }
        impl<T: Real, const N: usize> $trait<f64> for Dual<T, N> {
            #[inline]
            fn $method(&mut self, o: f64) {
                *self = *self $op o;
            }
        }
    )*};
}

assign_ops!(
    AddAssign, add_assign, +;
    SubAssign, sub_assign, -;
    MulAssign, mul_assign, *;
    DivAssign, div_assign, /;
);

impl<T: Real, const N: usize> RemAssign for Dual<T, N> {
    #[inline]
    fn rem_assign(&mut self, o: Self) {
        *self = *self % o;
    }
}

impl<T: Real, const N: usize> PartialEq for Dual<T, N> {
    #[inline]
    fn eq(&self, o: &Self) -> bool {
        self.re == o.re
    }
}

impl<T: Real, const N: usize> PartialOrd for Dual<T, N> {
    #[inline]
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&o.re)
    }
}

impl<T: Real, const N: usize> Zero for Dual<T, N> {
    fn zero() -> Self {
        Self::constant(T::zero())
    }
    fn is_zero(&self) -> bool {
        self.re.is_zero()
    }
}

impl<T: Real, const N: usize> One for Dual<T, N> {
    fn one() -> Self {
        Self::constant(T::one())
    }
}

impl<T: Real, const N: usize> Num for Dual<T, N> {
    type FromStrRadixErr = T::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(Self::constant)
    }
}

impl<T: Real, const N: usize> NumCast for Dual<T, N> {
    fn from<U: ToPrimitive>(n: U) -> Option<Self> {
        <T as NumCast>::from(n).map(Self::constant)
    }
}

impl<T: Real, const N: usize> ToPrimitive for Dual<T, N> {
    fn to_i64(&self) -> Option<i64> {
        self.re.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.re.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        self.re.to_f64()
    }
}
```

Note: `chain` is unused until Task 8. If clippy reports dead code, add `#[allow(dead_code)] // used by the Float impl in Task 8` to `chain`, and remove that attribute in Task 8.

- [ ] **Step 4: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scatrs/src/scalar scatrs/tests/dual_arith.rs
git commit -m "feat: forward-mode Dual numbers (arithmetic and basic num-traits)"
```

---

### Task 8: `Dual` as a `Real` (`Float`, `FloatConst` and the chain rule for every function)

**Files:**
- Create: `scatrs/src/scalar/dual/traits.rs`
- Modify: `scatrs/src/scalar/dual/mod.rs` (add `mod traits;`; remove any `allow(dead_code)` on `chain`)
- Test: `scatrs/tests/dual_functions.rs`

**Interfaces:**
- Consumes: `Dual` (Task 7), `DoubleDouble` implementing `Real` (Task 6).
- Produces:
  - `impl<T: Real, const N: usize> Float for Dual<T, N>` — `is_nan` and `is_finite` also inspect the derivatives;
  - `FloatConst` (constants with zero derivatives);
  - `Real` (`from_f64` and `from_f64_pair` give constants; `approx_f64` returns the value).
- Derivative rules (with x′ the input derivatives):

| Functions | Derivative |
|---|---|
| `floor`, `ceil`, `round`, `trunc`, `signum` | 0 |
| `fract` | x′ |
| `recip` | −x′/x² |
| `powi(n)` | n·xⁿ⁻¹·x′ |
| `powf(y)` | y·xʸ⁻¹·x′ + xʸ·ln x·y′ (the ln term only when y′ ≠ 0) |
| `sqrt` | x′/(2√x) |
| `exp` | eˣ·x′ |
| `exp2` | 2ˣ·ln 2·x′ |
| `ln` | x′/x |
| `log2` | x′/(x ln 2) |
| `log10` | x′/(x ln 10) |
| `log(b)` | ln x / ln b, via the quotient rule |
| `cbrt` | x′/(3·cbrt(x)²) |
| `hypot` | (x x′ + y y′)/h, with derivative 0 when h = 0 |
| `sin` | cos x·x′ |
| `cos` | −sin x·x′ |
| `tan` | (1 + tan² x)·x′ |
| `asin` | x′/√((1 − x)(1 + x)) |
| `acos` | −x′/√((1 − x)(1 + x)) |
| `atan` | x′/(1 + x²) |
| `atan2(y, x)` | (x y′ − y x′)/(x² + y²) |
| `exp_m1` | eˣ·x′ |
| `ln_1p` | x′/(1 + x) |
| `sinh` | cosh x·x′ |
| `cosh` | sinh x·x′ |
| `tanh` | (1 − tanh² x)·x′ |
| `asinh` | x′/√(x² + 1) |
| `acosh` | x′/√((x − 1)(x + 1)) |
| `atanh` | x′/((1 − x)(1 + x)) |
| `max`, `min` | the derivative of the chosen branch |
| `abs` | −x′ when x is negative, x′ otherwise |

- [ ] **Step 1: Write the failing tests**

`scatrs/tests/dual_functions.rs`:
```rust
use num_complex::Complex;
use num_traits::{Float, FloatConst, One, Zero};
use scatrs::Real;
use scatrs::scalar::{DoubleDouble as Dd, Dual};

type D1 = Dual<Dd, 1>;
type D2 = Dual<Dd, 2>;

/// Central difference in double-double: truncation ~2^-60 relative, rounding ~2^-74.
fn central(f: &dyn Fn(Dd) -> Dd, x: Dd) -> Dd {
    let h = Dd::from_f64(x.hi().abs().max(1.0) * 2f64.powi(-30));
    (f(x + h) - f(x - h)) / (h * 2.0)
}

fn assert_close(got: Dd, expected: Dd, rel: f64, what: &str) {
    let err = (got - expected).abs().hi();
    let scale = expected.abs().hi();
    assert!(
        err <= rel * scale || (scale == 0.0 && err == 0.0),
        "{what}: got {got:?}, expected {expected:?}"
    );
}

macro_rules! unary {
    ($name:ident, $method:ident, [$($x:expr),* $(,)?]) => {
        #[test]
        fn $name() {
            for x in [$($x),*] {
                let xd = Dd::from_f64(x);
                let [v] = D1::variables([xd]);
                let got = Float::$method(v);
                assert_eq!(got.re(), Float::$method(xd), "value of {}({x})", stringify!($method));
                let fd = central(&|t| Float::$method(t), xd);
                assert_close(got.eps()[0], fd, 1e-15, &format!("d/dx {}({x})", stringify!($method)));
                // The f64 instantiation agrees with the double-double one.
                let [v64] = Dual::<f64, 1>::variables([x]);
                let g64 = Float::$method(v64).eps()[0];
                let gdd = got.eps()[0].hi();
                assert!((g64 - gdd).abs() <= 1e-13 * gdd.abs(), "f64 path {}({x})", stringify!($method));
            }
        }
    };
}

unary!(sqrt, sqrt, [0.7, 2.3, 1e5]);
unary!(exp, exp, [-2.0, 0.5, 10.0]);
unary!(exp2, exp2, [-3.7, 3.7]);
unary!(exp_m1, exp_m1, [1e-3, 2.0]);
unary!(ln, ln, [0.3, 3.0]);
unary!(ln_1p, ln_1p, [1e-3, 2.0]);
unary!(log2, log2, [3.7]);
unary!(log10, log10, [3.7]);
unary!(cbrt, cbrt, [-8.5, 3.0]);
unary!(sin, sin, [0.4, 2.0, -5.0]);
unary!(cos, cos, [0.4, 2.0, -5.0]);
unary!(tan, tan, [0.4, 2.0]);
unary!(asin, asin, [-0.3, 0.9]);
unary!(acos, acos, [-0.3, 0.9]);
unary!(atan, atan, [3.0, -0.2]);
unary!(sinh, sinh, [0.2, 3.0]);
unary!(cosh, cosh, [0.2, 3.0]);
unary!(tanh, tanh, [0.2, 3.0]);
unary!(asinh, asinh, [-4.0, 0.1]);
unary!(acosh, acosh, [1.5, 10.0]);
unary!(atanh, atanh, [-0.3, 0.9]);
unary!(recip, recip, [-2.5, 0.3]);
unary!(abs, abs, [-2.5, 0.3]);
unary!(fract, fract, [2.25, -1.75]);

#[test]
fn piecewise_constant_functions_have_zero_derivative() {
    let [v] = D1::variables([Dd::from_f64(2.25)]);
    let fs: [fn(D1) -> D1; 5] = [Float::floor, Float::ceil, Float::round, Float::trunc, Float::signum];
    for f in fs {
        assert_eq!(f(v).eps()[0], Dd::ZERO);
    }
}

#[test]
fn integer_powers() {
    let [v] = D1::variables([Dd::from_f64(1.7)]);
    for n in [5, -3, 1, 0] {
        let fd = central(&|t| t.powi(n), Dd::from_f64(1.7));
        assert_close(Float::powi(v, n).eps()[0], fd, 1e-15, &format!("powi {n}"));
    }
}

#[test]
fn binary_functions() {
    let (x0, y0) = (Dd::from_f64(1.7), Dd::from_f64(-0.6));
    let [x, y] = D2::variables([x0, y0]);
    let cases: [(&str, fn(D2, D2) -> D2, fn(Dd, Dd) -> Dd); 5] = [
        ("powf", |a, b| a.powf(b), |a, b| a.powf(b)),
        ("hypot", |a, b| a.hypot(b), |a, b| a.hypot(b)),
        ("atan2", |a, b| a.atan2(b), |a, b| a.atan2(b)),
        ("log", |a, b| a.log(b * b + 2.0), |a, b| a.log(b * b + 2.0)),
        ("mul_add", |a, b| a.mul_add(b, a), |a, b| a.mul_add(b, a)),
    ];
    for (name, fd2, fdd) in cases {
        let got = fd2(x, y);
        assert_close(got.eps()[0], central(&|t| fdd(t, y0), x0), 1e-15, &format!("d/dx {name}"));
        assert_close(got.eps()[1], central(&|t| fdd(x0, t), y0), 1e-15, &format!("d/dy {name}"));
    }
}

#[test]
fn max_min_follow_the_chosen_branch() {
    let [x, y] = D2::variables([Dd::from_f64(1.0), Dd::from_f64(2.0)]);
    assert_eq!(x.max(y).eps(), y.eps());
    assert_eq!(x.min(y).eps(), x.eps());
}

#[test]
fn complex_of_dual_propagates_derivatives() {
    let [x, y] = Dual::<f64, 2>::variables([0.6, -1.3]);
    let z = Complex::new(x, y);
    let one = Complex::new(Dual::<f64, 2>::one(), Dual::zero());
    let f = (one / (z * z)).norm_sqr(); // 1 / (x² + y²)²
    let r2: f64 = 0.6 * 0.6 + 1.3 * 1.3;
    assert!((f.re() - 1.0 / (r2 * r2)).abs() < 1e-15);
    assert!((f.eps()[0] - (-4.0 * 0.6 / (r2 * r2 * r2))).abs() < 1e-14);
    assert!((f.eps()[1] - (-4.0 * -1.3 / (r2 * r2 * r2))).abs() < 1e-14);
    let w = Complex::new(Dual::<f64, 2>::zero(), y).exp(); // e^{iy}
    assert!((w.re.eps()[1] - (-(-1.3f64).sin())).abs() < 1e-15);
}

#[test]
fn constants_nan_detection_and_real() {
    assert_eq!(Dual::<f64, 3>::PI().re(), std::f64::consts::PI);
    assert_eq!(Dual::<f64, 3>::PI().eps(), &[0.0; 3]);
    assert!(Float::is_nan(Dual::<f64, 1>::new(1.0, [f64::NAN])));
    assert!(!Float::is_finite(Dual::<f64, 1>::new(1.0, [f64::INFINITY])));
    let c = <Dual<f64, 2> as Real>::from_f64(1.5);
    assert_eq!((c.re(), c.eps()), (1.5, &[0.0, 0.0]));
    assert_eq!(Real::approx_f64(Dual::<Dd, 1>::new(Dd::new(1.0, 1e-20), [Dd::ONE])), 1.0);
}

#[test]
fn generic_kernel_differentiates() {
    fn horner<T: Real>(coeffs: &[f64], x: T) -> T {
        coeffs.iter().rev().fold(T::zero(), |acc, &c| acc * x + c)
    }
    let [x] = Dual::<f64, 1>::variables([2.0]);
    let p = horner(&[1.0, 2.0, 3.0], x);
    assert_eq!((p.re(), p.eps()[0]), (17.0, 14.0));
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test dual_functions`
Expected: compile errors, `Float is not implemented for Dual`.

- [ ] **Step 3: Implement**

Add `mod traits;` to `scatrs/src/scalar/dual/mod.rs`.

`scatrs/src/scalar/dual/traits.rs`:
```rust
//! `Float`, `FloatConst` and `Real` for [`Dual`]: the chain rule for every function.

use core::array;
use core::num::FpCategory;

use num_traits::{Float, FloatConst, One, Zero};

use super::Dual;
use crate::scalar::Real;

impl<T: Real, const N: usize> Float for Dual<T, N> {
    fn nan() -> Self {
        Self::constant(T::nan())
    }
    fn infinity() -> Self {
        Self::constant(T::infinity())
    }
    fn neg_infinity() -> Self {
        Self::constant(T::neg_infinity())
    }
    fn neg_zero() -> Self {
        Self::constant(T::neg_zero())
    }
    fn min_value() -> Self {
        Self::constant(T::min_value())
    }
    fn min_positive_value() -> Self {
        Self::constant(T::min_positive_value())
    }
    fn epsilon() -> Self {
        Self::constant(T::epsilon())
    }
    fn max_value() -> Self {
        Self::constant(T::max_value())
    }
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.eps.iter().any(|e| e.is_nan())
    }
    fn is_infinite(self) -> bool {
        self.re.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.eps.iter().all(|e| e.is_finite())
    }
    fn is_normal(self) -> bool {
        self.re.is_normal()
    }
    fn classify(self) -> FpCategory {
        self.re.classify()
    }
    fn floor(self) -> Self {
        Self::constant(self.re.floor())
    }
    fn ceil(self) -> Self {
        Self::constant(self.re.ceil())
    }
    fn round(self) -> Self {
        Self::constant(self.re.round())
    }
    fn trunc(self) -> Self {
        Self::constant(self.re.trunc())
    }
    fn fract(self) -> Self {
        Self { re: self.re.fract(), eps: self.eps }
    }
    fn abs(self) -> Self {
        if self.re.is_sign_negative() { -self } else { self }
    }
    fn signum(self) -> Self {
        Self::constant(self.re.signum())
    }
    fn is_sign_positive(self) -> bool {
        self.re.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.re.is_sign_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    fn recip(self) -> Self {
        let r = self.re.recip();
        self.chain(r, -(r * r))
    }
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::one();
        }
        self.chain(self.re.powi(n), self.re.powi(n - 1) * T::from_f64(f64::from(n)))
    }
    fn powf(self, n: Self) -> Self {
        let f = self.re.powf(n.re);
        let df_dx = n.re * self.re.powf(n.re - T::one());
        let df_dy = if n.eps.iter().any(|e| !e.is_zero()) { f * self.re.ln() } else { T::zero() };
        Self { re: f, eps: array::from_fn(|i| self.eps[i] * df_dx + n.eps[i] * df_dy) }
    }
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        self.chain(s, (s + s).recip())
    }
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }
    fn exp2(self) -> Self {
        let e = self.re.exp2();
        self.chain(e, e * T::LN_2())
    }
    fn ln(self) -> Self {
        self.chain(self.re.ln(), self.re.recip())
    }
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    fn log2(self) -> Self {
        self.chain(self.re.log2(), (self.re * T::LN_2()).recip())
    }
    fn log10(self) -> Self {
        self.chain(self.re.log10(), (self.re * T::LN_10()).recip())
    }
    fn max(self, other: Self) -> Self {
        if self.re.is_nan() || (!other.re.is_nan() && other.re > self.re) { other } else { self }
    }
    fn min(self, other: Self) -> Self {
        if self.re.is_nan() || (!other.re.is_nan() && other.re < self.re) { other } else { self }
    }
    fn abs_sub(self, other: Self) -> Self {
        if self.re <= other.re { Self::zero() } else { self - other }
    }
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        self.chain(c, (c * c * T::from_f64(3.0)).recip())
    }
    fn hypot(self, other: Self) -> Self {
        let h = self.re.hypot(other.re);
        if h.is_zero() {
            return Self::constant(h);
        }
        let (a, b) = (self.re / h, other.re / h);
        Self { re: h, eps: array::from_fn(|i| self.eps[i] * a + other.eps[i] * b) }
    }
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(s, c)
    }
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(c, -s)
    }
    fn tan(self) -> Self {
        let t = self.re.tan();
        self.chain(t, T::one() + t * t)
    }
    fn asin(self) -> Self {
        let d = ((T::one() - self.re) * (T::one() + self.re)).sqrt().recip();
        self.chain(self.re.asin(), d)
    }
    fn acos(self) -> Self {
        let d = ((T::one() - self.re) * (T::one() + self.re)).sqrt().recip();
        self.chain(self.re.acos(), -d)
    }
    fn atan(self) -> Self {
        self.chain(self.re.atan(), (T::one() + self.re * self.re).recip())
    }
    fn atan2(self, other: Self) -> Self {
        let (y, x) = (self.re, other.re);
        let r2 = x * x + y * y;
        Self {
            re: y.atan2(x),
            eps: array::from_fn(|i| (x * self.eps[i] - y * other.eps[i]) / r2),
        }
    }
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (self.chain(s, c), self.chain(c, -s))
    }
    fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }
    fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), (T::one() + self.re).recip())
    }
    fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }
    fn tanh(self) -> Self {
        let t = self.re.tanh();
        self.chain(t, T::one() - t * t)
    }
    fn asinh(self) -> Self {
        self.chain(self.re.asinh(), (self.re * self.re + T::one()).sqrt().recip())
    }
    fn acosh(self) -> Self {
        let d = ((self.re - T::one()) * (self.re + T::one())).sqrt().recip();
        self.chain(self.re.acosh(), d)
    }
    fn atanh(self) -> Self {
        let d = ((T::one() - self.re) * (T::one() + self.re)).recip();
        self.chain(self.re.atanh(), d)
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.re.integer_decode()
    }
    fn to_degrees(self) -> Self {
        let f = T::from_f64(180.0) / T::PI();
        self.chain(self.re * f, f)
    }
    fn to_radians(self) -> Self {
        let f = T::PI() / T::from_f64(180.0);
        self.chain(self.re * f, f)
    }
}

macro_rules! float_consts {
    ($($name:ident),* $(,)?) => {
        impl<T: Real, const N: usize> FloatConst for Dual<T, N> {
            $(fn $name() -> Self { Self::constant(T::$name()) })*
        }
    };
}

float_consts!(
    E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4,
    FRAC_PI_6, FRAC_PI_8, LN_10, LN_2, LOG10_E, LOG2_E, PI, SQRT_2, TAU, LOG10_2, LOG2_10,
);

impl<T: Real, const N: usize> Real for Dual<T, N> {
    fn from_f64(x: f64) -> Self {
        Self::constant(T::from_f64(x))
    }
    fn from_f64_pair(hi: f64, lo: f64) -> Self {
        Self::constant(T::from_f64_pair(hi, lo))
    }
    fn approx_f64(self) -> f64 {
        self.re.approx_f64()
    }
}
```

- [ ] **Step 4: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scatrs/src/scalar/dual scatrs/tests/dual_functions.rs
git commit -m "feat: Dual implements Real with chain rules verified by double-double finite differences"
```

---

### Task 9: Gauss–Legendre quadrature

**Files:**
- Create: `scatrs/src/special/mod.rs`, `scatrs/src/special/gauss_legendre.rs`
- Modify: `scatrs/src/lib.rs` (add `pub mod special;`)
- Create: `reference/scatrs_reference/generators/gauss_legendre.py`; register it
- Generate: `reference/data/gauss_legendre.jsonl`
- Test: `scatrs/tests/gauss_legendre.rs`

**Interfaces:**
- Consumes: `Real`, `DoubleDouble`, `Dual`, `Error`, `Result`.
- Produces:
  - `scatrs::special::GaussLegendre<T> { pub nodes: Vec<T>, pub weights: Vec<T> }`, with nodes in ascending order and exact symmetry `nodes[i] == -nodes[n-1-i]`. It derives `Debug, Clone, PartialEq`.
  - `scatrs::special::gauss_legendre<T: Real>(n: usize) -> Result<GaussLegendre<T>>`.
    - `n == 0` returns `Error::InvalidInput`.
    - Newton failure returns `Error::NotConverged`.
    - The rule is computed internally in double-double and then converted with `T::from_f64_pair`.
- Accuracy contract (verified against Arb for n ≤ 1000):
  - `f64`: nodes have absolute error ≤ 2⁻⁵², weights relative error ≤ 2⁻⁵².
  - `DoubleDouble`: nodes have absolute error ≤ 2⁻⁹⁸, weights relative error ≤ 2⁻¹⁰⁰·(4 + n²). The n² comes from the conditioning of 1 − x² next to ±1.

- [ ] **Step 1: Add the reference generator**

`reference/scatrs_reference/generators/gauss_legendre.py`:
```python
"""Gauss–Legendre nodes and weights with certified Arb enclosures (arb.legendre_p_root)."""

from __future__ import annotations

from flint import arb, ctx

from scatrs_reference.records import record

PRECISION_BITS = 300
NS = (1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 128, 257, 500, 1000)


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    recs = []
    for n in NS:
        roots = [arb.legendre_p_root(n, k, weight=True) for k in range(n)]  # descending
        roots.reverse()
        recs.append(
            record(
                "gauss_legendre",
                f"n={n}",
                {"n": n},
                {"nodes": [r for r, _ in roots], "weights": [w for _, w in roots]},
            )
        )
    return PRECISION_BITS, recs
```

Also add this test to `reference/tests/test_generators.py`:
```python
def test_gauss_legendre_nodes_ascend_and_are_symmetric():
    lines = (DATA / "gauss_legendre.jsonl").read_text().splitlines()
    for line in lines:
        rec = json.loads(line)
        hi = rec["outputs"]["nodes"]["hi"]
        assert hi == sorted(hi)
        assert hi == [-x for x in reversed(hi)]
```

Register the generator. Then:

Run: `uv run --directory reference python -m scatrs_reference generate --only gauss_legendre && uv run --directory reference python -m pytest -q`
Expected: 14 records; the tests pass.

- [ ] **Step 2: Write the failing Rust tests**

`scatrs/tests/gauss_legendre.rs`:
```rust
#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::load;
use scatrs::scalar::{DoubleDouble as Dd, Dual};
use scatrs::special::gauss_legendre;
use scatrs::Error;

const EPS: f64 = f64::EPSILON; // 2^-52

#[test]
fn f64_rule_matches_arb_to_half_an_ulp_level() {
    let (mut worst_x, mut worst_w) = (0.0_f64, 0.0_f64);
    for r in load("gauss_legendre.jsonl") {
        let n = r.input_i64("n") as usize;
        let rule = gauss_legendre::<f64>(n).unwrap();
        let (xs, ws) = (r.output("nodes"), r.output("weights"));
        for i in 0..n {
            let ex = (rule.nodes[i] - (xs.hi[i] + xs.lo[i])).abs() + xs.err[i];
            let ew = ((rule.weights[i] - (ws.hi[i] + ws.lo[i])).abs() + ws.err[i]) / ws.hi[i];
            worst_x = worst_x.max(ex / EPS);
            worst_w = worst_w.max(ew / EPS);
            assert!(ex <= EPS, "n={n} node {i}: error {ex:e}");
            assert!(ew <= EPS, "n={n} weight {i}: relative error {ew:e}");
        }
    }
    eprintln!("f64: worst node error {worst_x:.3} eps, worst weight error {worst_w:.3} eps");
}

#[test]
fn double_double_rule_matches_arb() {
    let tol = 2f64.powi(-100);
    for r in load("gauss_legendre.jsonl") {
        let n = r.input_i64("n") as usize;
        let rule = gauss_legendre::<Dd>(n).unwrap();
        let (xs, ws) = (r.output("nodes"), r.output("weights"));
        for i in 0..n {
            let x = Dd::new(xs.hi[i], xs.lo[i]);
            let w = Dd::new(ws.hi[i], ws.lo[i]);
            let ex = (rule.nodes[i] - x).abs().hi() + xs.err[i];
            let ew = ((rule.weights[i] - w).abs().hi() + ws.err[i]) / w.hi();
            assert!(ex <= 4.0 * tol, "n={n} node {i}: error {ex:e}");
            // Weights next to ±1 inherit the n² conditioning of 1 − x².
            let wtol = tol * (4.0 + (n * n) as f64);
            assert!(ew <= wtol, "n={n} weight {i}: relative error {ew:e} > {wtol:e}");
        }
    }
}

#[test]
fn integrates_even_monomials_exactly() {
    for n in [1usize, 2, 5, 16, 64, 257] {
        let rule = gauss_legendre::<f64>(n).unwrap();
        for k in (0..=(2 * n - 1).min(40)).step_by(2) {
            let s: f64 = rule.nodes.iter().zip(&rule.weights).map(|(x, w)| w * x.powi(k as i32)).sum();
            let exact = 2.0 / (k as f64 + 1.0);
            // Sequential summation of n terms: allow 64 eps of accumulated rounding.
            assert!((s - exact).abs() <= 64.0 * EPS, "n={n} k={k}: {s} vs {exact}");
        }
    }
}

#[test]
fn rule_is_symmetric_and_sorted() {
    for n in [1usize, 2, 7, 64] {
        let rule = gauss_legendre::<f64>(n).unwrap();
        for i in 0..n {
            assert_eq!(rule.nodes[i], -rule.nodes[n - 1 - i]);
            assert_eq!(rule.weights[i], rule.weights[n - 1 - i]);
        }
        assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]));
        if n % 2 == 1 {
            assert!(rule.nodes[n / 2] == 0.0 && rule.nodes[n / 2].is_sign_positive());
        }
    }
}

#[test]
fn zero_points_is_invalid() {
    assert!(matches!(gauss_legendre::<f64>(0), Err(Error::InvalidInput(_))));
}

#[test]
fn dual_instantiation_has_zero_derivatives() {
    let rule = gauss_legendre::<Dual<f64, 1>>(4).unwrap();
    assert!(rule.nodes.iter().chain(&rule.weights).all(|v| v.eps()[0] == 0.0));
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test gauss_legendre`
Expected: a compile error, `unresolved import scatrs::special`.

- [ ] **Step 4: Implement**

In `scatrs/src/lib.rs`, add `pub mod special;` after `pub mod scalar;`.

`scatrs/src/special/mod.rs`:
```rust
//! Special functions and quadrature rules shared by all scattering methods.

mod gauss_legendre;

pub use gauss_legendre::{GaussLegendre, gauss_legendre};
```

`scatrs/src/special/gauss_legendre.rs`:
```rust
//! Gauss–Legendre quadrature on [−1, 1].
//!
//! Nodes come from Newton iteration on Pₙ in double-double arithmetic, started from
//! Tricomi's approximation, and are then rounded to the requested type. A plain `f64`
//! computation loses about n² ulps in the weights next to ±1, which are exactly where Mie
//! forward peaks sit. Rounding from double-double keeps `f64` nodes and weights within
//! about half an ulp. The cost is O(n²); callers that reuse a rule should cache it.

use crate::scalar::{DoubleDouble as Dd, Real};
use crate::{Error, Result};

/// Gauss–Legendre nodes (ascending) and weights on [−1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct GaussLegendre<T> {
    /// Nodes in ascending order; `nodes[i] == -nodes[n - 1 - i]`.
    pub nodes: Vec<T>,
    /// Weights; `weights[i] == weights[n - 1 - i]`.
    pub weights: Vec<T>,
}

/// Newton steps stop once |Δx| ≤ 2⁻⁹⁰; one further step then reaches rounding level.
const CONVERGED: f64 = 8.077_935_669_463_161e-28;
const MAX_ITERATIONS: usize = 100;

/// Recurrence coefficients for Pⱼ = aⱼ·x·Pⱼ₋₁ − bⱼ·Pⱼ₋₂, with aⱼ = (2j − 1)/j and bⱼ = (j − 1)/j.
struct Recurrence {
    a: Vec<Dd>,
    b: Vec<Dd>,
}

impl Recurrence {
    fn new(n: usize) -> Self {
        let a = (0..=n)
            .map(|j| if j == 0 { Dd::ZERO } else { Dd::from_f64((2 * j - 1) as f64) / j as f64 })
            .collect();
        let b = (0..=n)
            .map(|j| if j == 0 { Dd::ZERO } else { Dd::from_f64((j - 1) as f64) / j as f64 })
            .collect();
        Self { a, b }
    }

    /// (Pₙ(x), Pₙ′(x)) via (1 − x²)Pₙ′ = n(Pₙ₋₁ − x·Pₙ).
    fn legendre_and_derivative(&self, n: usize, x: Dd) -> (Dd, Dd) {
        let mut p_prev = Dd::ONE;
        let mut p = x;
        for j in 2..=n {
            let next = self.a[j] * x * p - self.b[j] * p_prev;
            p_prev = p;
            p = next;
        }
        let dp = (p_prev - x * p) * n as f64 / ((Dd::ONE - x) * (Dd::ONE + x));
        (p, dp)
    }

    /// Newton iteration for the root near `guess`; returns (root, Pₙ′(root)).
    fn newton(&self, n: usize, guess: f64) -> Result<(Dd, Dd)> {
        let mut x = Dd::from_f64(guess);
        let mut last = f64::INFINITY;
        let mut polish = false;
        for _ in 0..MAX_ITERATIONS {
            let (p, dp) = self.legendre_and_derivative(n, x);
            let dx = p / dp;
            x -= dx;
            if polish {
                let (_, dp) = self.legendre_and_derivative(n, x);
                return Ok((x, dp));
            }
            last = dx.hi().abs();
            polish = last <= CONVERGED;
        }
        Err(Error::NotConverged { achieved: last, requested: CONVERGED })
    }
}

/// Computes the `n`-point Gauss–Legendre rule, exact for polynomials of degree ≤ 2n − 1.
pub fn gauss_legendre<T: Real>(n: usize) -> Result<GaussLegendre<T>> {
    if n == 0 {
        return Err(Error::InvalidInput("Gauss-Legendre rule needs n >= 1".into()));
    }
    let rec = Recurrence::new(n);
    let nf = n as f64;
    let mut nodes = vec![T::zero(); n];
    let mut weights = vec![T::zero(); n];
    for i in 0..n.div_ceil(2) {
        let (x, dp) = if n % 2 == 1 && i == n / 2 {
            (Dd::ZERO, rec.legendre_and_derivative(n, Dd::ZERO).1)
        } else {
            let k = (i + 1) as f64;
            let theta = core::f64::consts::PI * (4.0 * k - 1.0) / (4.0 * nf + 2.0);
            let guess = (1.0 - (nf - 1.0) / (8.0 * nf * nf * nf)) * theta.cos();
            rec.newton(n, guess)?
        };
        let w = Dd::from_f64(2.0) / ((Dd::ONE - x) * (Dd::ONE + x) * dp * dp);
        let w = T::from_f64_pair(w.hi(), w.lo());
        nodes[i] = T::from_f64_pair(-x.hi(), -x.lo());
        nodes[n - 1 - i] = T::from_f64_pair(x.hi(), x.lo());
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    Ok(GaussLegendre { nodes, weights })
}
```

Note on the middle node: for odd n it is assigned `-0.0` first and then overwritten with `+0.0`, so the stored value is `+0.0`.

- [ ] **Step 5: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass. Report the worst node and weight errors from `--nocapture`. During planning, a prototype of this algorithm measured ≤ 0.25 eps for nodes and ≤ 0.49 eps for weights up to n = 1000.

- [ ] **Step 6: Commit**

```bash
git add scatrs/src/lib.rs scatrs/src/special reference/scatrs_reference reference/tests reference/data scatrs/tests/gauss_legendre.rs
git commit -m "feat: Gauss-Legendre rule computed in double-double, certified against Arb"
```

---

### Task 10: Wigner d-functions

**Files:**
- Create: `scatrs/src/special/wigner.rs`
- Modify: `scatrs/src/special/mod.rs` (add `mod wigner; pub use wigner::{wigner_d, wigner_d_vec};`)
- Create: `reference/scatrs_reference/generators/wigner_d.py`; register it
- Generate: `reference/data/wigner_d.jsonl`
- Test: `scatrs/tests/wigner_d.rs`

**Interfaces:**
- Consumes: `Real`, `Error`, `Result`, `gauss_legendre` (used in the tests), `DoubleDouble`, `Dual`.
- Produces:
  - `scatrs::special::wigner_d<T: Real>(m: i32, n: i32, x: T, out: &mut [T]) -> Result<()>`
    - Fills `out[l]` with d^l_{mn}(θ), where x = cos θ, for l = 0..out.len().
    - Entries with l < max(|m|, |n|) are zero.
    - If x is outside [−1, 1] or is NaN, it returns `Error::InvalidInput` with a message containing `-1 <= x <= 1`.
    - An empty `out` is fine.
  - `scatrs::special::wigner_d_vec<T: Real>(m: i32, n: i32, l_max: usize, x: T) -> Result<Vec<T>>`.
- Convention: Mishchenko, Travis & Lacis (2002), Appendix B. It equals the common d^j_{m′m}(β) with m′ = m. Examples:
  - d¹₀₁ = sin θ/√2
  - d¹₁₀ = −sin θ/√2
  - d²₂₂ = ((1 + x)/2)²
- Accuracy contract (absolute error, against Arb at an exact x):
  - `f64`: ≤ 16(l + 1)·2⁻⁵²;
  - `DoubleDouble`: ≤ (l + 1)·2⁻⁹⁷.

- [ ] **Step 1: Add the reference generator**

`reference/scatrs_reference/generators/wigner_d.py`:
```python
"""Wigner d^l_{mn}(θ) from the explicit finite sum, evaluated in Arb at exact x = cos θ.

Convention: d^l_{mn} equals the common d^j_{m'm}(β) with m' = m (Mishchenko, Travis & Lacis
2002, Appendix B). The sum is finite, so every value is certified.
"""

from __future__ import annotations

import math

from flint import arb, ctx, fmpz

from scatrs_reference.records import record

PRECISION_BITS = 800
L_MAX = 100
PAIRS = ((0, 0), (0, 2), (2, 2), (2, -2), (1, 1), (1, -1), (0, 1), (1, 0), (3, -7), (-4, 2), (5, 5), (0, -3), (10, 4))
XS = (-1.0, -0.9999999, -0.5, 0.0, 0.3, 0.8660254037844386, 0.9999999, 1.0)


def _fact(k: int) -> arb:
    return arb(fmpz(math.factorial(k)))


def _pow(base: arb, e: int) -> arb:
    return arb(1) if e == 0 else base**e


def d_ref(l: int, m: int, n: int, x: float) -> arb:
    if l < max(abs(m), abs(n)):
        return arb(0)
    big_x = arb(x)
    c = ((1 + big_x) / 2).sqrt()
    s = ((1 - big_x) / 2).sqrt()
    pref = (_fact(l + m) * _fact(l - m) * _fact(l + n) * _fact(l - n)).sqrt()
    total = arb(0)
    for k in range(0, 2 * l + 1):
        a, b, cc, d = l + n - k, k, m - n + k, l - m - k
        if min(a, b, cc, d) < 0:
            continue
        term = pref / (_fact(a) * _fact(b) * _fact(cc) * _fact(d))
        term *= _pow(c, 2 * l + n - m - 2 * k) * _pow(s, m - n + 2 * k)
        total += -term if (m - n + k) % 2 else term
    return total


def generate() -> tuple[int, list[dict]]:
    ctx.prec = PRECISION_BITS
    recs = []
    cases = [(m, n, x, L_MAX) for (m, n) in PAIRS for x in XS] + [(0, 2, 0.3, 300)]
    for m, n, x, l_max in cases:
        recs.append(
            record(
                "wigner_d",
                f"m={m},n={n},x={x!r},l_max={l_max}",
                {"m": m, "n": n, "x": x, "l_max": l_max},
                {"d": [d_ref(l, m, n, x) for l in range(l_max + 1)]},
            )
        )
    return PRECISION_BITS, recs
```

Register it. Then:

Run: `uv run --directory reference python -m scatrs_reference generate --only wigner_d && uv run --directory reference python -m pytest -q`
Expected: 105 records. This may take a few minutes. Check that `reference/data` stays under 20 MB with `du -sh reference/data`.

- [ ] **Step 2: Write the failing Rust tests**

`scatrs/tests/wigner_d.rs`:
```rust
#![allow(clippy::needless_range_loop)] // parallel reference arrays are indexed together

mod common;

use common::load;
use scatrs::scalar::{DoubleDouble as Dd, Dual};
use scatrs::special::{gauss_legendre, wigner_d, wigner_d_vec};
use scatrs::Error;

const EPS: f64 = f64::EPSILON;

#[test]
fn closed_forms() {
    for x in [-1.0, -0.6, 0.0, 0.25, 0.9, 1.0_f64] {
        let s = (1.0 - x * x).sqrt();
        let close = |a: f64, b: f64| assert!((a - b).abs() <= 4.0 * EPS, "x={x}: {a} vs {b}");
        let d00 = wigner_d_vec(0, 0, 2, x).unwrap();
        close(d00[0], 1.0);
        close(d00[1], x);
        close(d00[2], (3.0 * x * x - 1.0) / 2.0);
        close(wigner_d_vec(1, 1, 1, x).unwrap()[1], (1.0 + x) / 2.0);
        close(wigner_d_vec(1, -1, 1, x).unwrap()[1], (1.0 - x) / 2.0);
        close(wigner_d_vec(0, 1, 1, x).unwrap()[1], s / 2f64.sqrt());
        close(wigner_d_vec(1, 0, 1, x).unwrap()[1], -s / 2f64.sqrt());
        close(wigner_d_vec(2, 2, 2, x).unwrap()[2], ((1.0 + x) / 2.0).powi(2));
        close(wigner_d_vec(2, -2, 2, x).unwrap()[2], ((1.0 - x) / 2.0).powi(2));
        close(wigner_d_vec(0, 2, 2, x).unwrap()[2], 6f64.sqrt() / 4.0 * s * s);
        assert_eq!(wigner_d_vec(2, 2, 2, x).unwrap()[..2], [0.0, 0.0]);
    }
}

#[test]
fn symmetry_relations() {
    for (m, n) in [(1, 3), (-2, 4), (5, -1), (0, 2)] {
        for x in [-0.7, 0.1, 0.95_f64] {
            let a = wigner_d_vec(m, n, 30, x).unwrap();
            let b = wigner_d_vec(n, m, 30, x).unwrap();
            let c = wigner_d_vec(-n, -m, 30, x).unwrap();
            let sign = if (m - n) % 2 == 0 { 1.0 } else { -1.0 };
            for l in 0..=30 {
                // Each side carries up to ~16(l+1) ulps of recurrence error.
                let tol = 32.0 * (l as f64 + 1.0) * EPS;
                assert!((a[l] - sign * b[l]).abs() <= tol);
                assert!((a[l] - c[l]).abs() <= tol);
            }
        }
    }
}

#[test]
fn orthogonality_under_gauss_legendre() {
    let rule = gauss_legendre::<f64>(64).unwrap();
    for (m, n) in [(0, 0), (2, 2), (2, -2), (0, 2), (1, 3)] {
        let table: Vec<Vec<f64>> = rule.nodes.iter().map(|&x| wigner_d_vec(m, n, 40, x).unwrap()).collect();
        let l_min = m.unsigned_abs().max(n.unsigned_abs()) as usize;
        for l in l_min..=40 {
            for k in l_min..=40 {
                let s: f64 = rule.weights.iter().zip(&table).map(|(w, d)| w * d[l] * d[k]).sum();
                let expected = if l == k { 2.0 / (2.0 * l as f64 + 1.0) } else { 0.0 };
                assert!((s - expected).abs() <= 2e-12, "m={m} n={n} l={l} k={k}: {s}");
            }
        }
    }
}

#[test]
fn matches_arb_reference() {
    let mut worst = 0.0_f64;
    for r in load("wigner_d.jsonl") {
        let (m, n) = (r.input_i64("m") as i32, r.input_i64("n") as i32);
        let (x, l_max) = (r.input_f64("x"), r.input_i64("l_max") as usize);
        let d = r.output("d");
        let got = wigner_d_vec(m, n, l_max, x).unwrap();
        let got_dd = wigner_d_vec(m, n, l_max, Dd::from_f64(x)).unwrap();
        for l in 0..=l_max {
            let exact = Dd::new(d.hi[l], d.lo[l]);
            let err = (got[l] - (d.hi[l] + d.lo[l])).abs() + d.err[l];
            let tol = 16.0 * (l as f64 + 1.0) * EPS;
            worst = worst.max(err / ((l as f64 + 1.0) * EPS));
            assert!(err <= tol, "{}: l={l} f64 error {err:e} > {tol:e}", r.case);
            let err_dd = (got_dd[l] - exact).abs().hi() + d.err[l];
            let tol_dd = (l as f64 + 1.0) * 2f64.powi(-97);
            assert!(err_dd <= tol_dd, "{}: l={l} double-double error {err_dd:e}", r.case);
        }
    }
    eprintln!("wigner_d f64: worst error {worst:.2} (l+1) eps");
}

#[test]
fn invalid_arguments() {
    let mut out = [0.0; 4];
    for bad in [1.5, -1.0000001, f64::NAN] {
        match wigner_d(0, 0, bad, &mut out) {
            Err(Error::InvalidInput(msg)) => assert!(msg.contains("-1 <= x <= 1"), "{msg}"),
            other => panic!("expected InvalidInput for x={bad}, got {other:?}"),
        }
    }
    assert!(wigner_d::<f64>(0, 0, 0.5, &mut []).is_ok());
}

#[test]
fn derivative_in_x_matches_legendre_derivative() {
    let x0 = 0.3_f64;
    let [x] = Dual::<f64, 1>::variables([x0]);
    let d = wigner_d_vec(0, 0, 20, x).unwrap();
    let p = wigner_d_vec(0, 0, 20, x0).unwrap();
    for l in 1..=20 {
        // P_l'(x) = l (x P_l − P_{l−1}) / (x² − 1)
        let expected = l as f64 * (x0 * p[l] - p[l - 1]) / (x0 * x0 - 1.0);
        assert!((d[l].eps()[0] - expected).abs() <= 1e-12 * expected.abs().max(1.0), "l={l}");
    }
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cargo test -p scatrs --test wigner_d`
Expected: a compile error, `no wigner_d in special`.

- [ ] **Step 4: Implement**

Update `scatrs/src/special/mod.rs`:
```rust
//! Special functions and quadrature rules shared by all scattering methods.

mod gauss_legendre;
mod wigner;

pub use gauss_legendre::{GaussLegendre, gauss_legendre};
pub use wigner::{wigner_d, wigner_d_vec};
```

`scatrs/src/special/wigner.rs`:
```rust
//! Wigner small-d functions d^l_{mn}(θ).
//!
//! Convention: Mishchenko, Travis & Lacis (2002), *Scattering, Absorption, and Emission of
//! Light by Small Particles*, Appendix B. It equals the common d^j_{m′m}(β) with m′ = m.
//! Examples: d¹₀₁(θ) = sin θ/√2, d¹₁₀(θ) = −sin θ/√2, d²₂₂(θ) = cos⁴(θ/2).
//!
//! Upward recurrence in l (MTL Eq. B.22), starting at l₀ = max(|m|, |n|) with
//! d^{l₀}_{mn} = ξ √C(2l₀, |m − n|) · sin^{|m−n|}(θ/2) · cos^{|m+n|}(θ/2),
//! where ξ = 1 if n ≥ m and (−1)^{m−n} otherwise.

use crate::scalar::Real;
use crate::{Error, Result};

/// Fills `out[l]` with d^l_{mn}(θ) for `l = 0, …, out.len() − 1`, given `x = cos θ`.
///
/// Entries with `l < max(|m|, |n|)` are zero. The `f64` absolute error is about
/// 10·(l + 1) ulps (measured against Arb).
pub fn wigner_d<T: Real>(m: i32, n: i32, x: T, out: &mut [T]) -> Result<()> {
    if !(-T::one()..=T::one()).contains(&x) {
        return Err(Error::InvalidInput(format!(
            "wigner_d needs -1 <= x <= 1, got {:?}",
            x.approx_f64()
        )));
    }
    out.fill(T::zero());
    let (m64, n64) = (i64::from(m), i64::from(n));
    let l_min = m64.unsigned_abs().max(n64.unsigned_abs()) as usize;
    if l_min >= out.len() {
        return Ok(());
    }
    let a = (m64 - n64).unsigned_abs();
    let b = (m64 + n64).unsigned_abs();
    let half_sin = ((T::one() - x) * 0.5).sqrt();
    let half_cos = ((T::one() + x) * 0.5).sqrt();
    // √C(a + b, a) · sin^a · cos^b, interleaving the factors to stay in range.
    let mut v = T::one();
    for k in 1..=a.max(b) {
        if k <= a {
            v = v * (T::from_f64((b + k) as f64) / T::from_f64(k as f64)).sqrt() * half_sin;
        }
        if k <= b {
            v *= half_cos;
        }
    }
    let xi = if n64 >= m64 || (m64 - n64) % 2 == 0 { T::one() } else { -T::one() };
    out[l_min] = xi * v;

    let (mf, nf) = (m as f64, n as f64);
    let mut s = l_min;
    if s == 0 {
        if out.len() > 1 {
            out[1] = x;
        }
        s = 1;
    }
    while s + 1 < out.len() {
        let sf = s as f64;
        let c_prev = if s > l_min {
            T::from_f64(sf * sf - mf * mf).sqrt() * T::from_f64(sf * sf - nf * nf).sqrt() * (sf + 1.0)
        } else {
            T::zero()
        };
        let numer = (x * (sf * (sf + 1.0)) - mf * nf) * (2.0 * sf + 1.0) * out[s] - c_prev * out[s - 1];
        let s1 = sf + 1.0;
        let denom = T::from_f64(s1 * s1 - mf * mf).sqrt() * T::from_f64(s1 * s1 - nf * nf).sqrt() * sf;
        out[s + 1] = numer / denom;
        s += 1;
    }
    Ok(())
}

/// Allocating form of [`wigner_d`]: d^l_{mn}(θ) for `l = 0..=l_max`.
pub fn wigner_d_vec<T: Real>(m: i32, n: i32, l_max: usize, x: T) -> Result<Vec<T>> {
    let len = l_max
        .checked_add(1)
        .ok_or_else(|| Error::InvalidInput("l_max is too large".into()))?;
    let mut out = vec![T::zero(); len];
    wigner_d(m, n, x, &mut out)?;
    Ok(out)
}
```

- [ ] **Step 5: Run the tests and checks**

Run: `cargo fmt --all && cargo test -p scatrs && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all pass. Report the worst error in (l+1)·eps units; planning measured 11.6.

- [ ] **Step 6: Commit**

```bash
git add scatrs/src/special reference/scatrs_reference/generators reference/data scatrs/tests/wigner_d.rs
git commit -m "feat: Wigner d-functions (MTL convention) validated against Arb explicit sums"
```

---

### Task 11: Python package (`scatrs`, with `scatrs._scatrs`)

**Files:**
- Modify: `Cargo.toml` (members: `["scatrs", "scatrs-py"]`)
- Create: `scatrs-py/Cargo.toml`, `scatrs-py/src/lib.rs`, `scatrs-py/src/errors.rs`, `scatrs-py/src/special.rs`
- Create: `pyproject.toml`, `.python-version`
- Create: `python/scatrs/__init__.py`, `python/scatrs/errors.py`, `python/scatrs/special.py`, `python/scatrs/_scatrs.pyi`, `python/scatrs/py.typed`
- Test: `tests/test_package.py`, `tests/test_special.py`, `tests/test_errors.py`

**Interfaces:**
- Consumes: `scatrs::special::{gauss_legendre, wigner_d}`, `scatrs::Error`.
- Produces (Python):
  - `scatrs.__version__`.
  - `scatrs.special.gauss_legendre(n) -> (nodes, weights)`.
  - `scatrs.special.wigner_d(m, n, l_max, x)`, returning shape `np.shape(x) + (l_max + 1,)`.
  - The exceptions in `scatrs.errors`, re-exported at the top level.
- Produces (extension):
  - `scatrs._scatrs.gauss_legendre(n: int)`.
  - `scatrs._scatrs.wigner_d(m: int, n: int, l_max: int, x: 1-D float64 array)`, returning a 2-D array.
  - Both release the GIL while computing.

- [ ] **Step 1: Create the build configuration**

`Cargo.toml`: change `members = ["scatrs"]` to `members = ["scatrs", "scatrs-py"]`.

`scatrs-py/Cargo.toml`:
```toml
[package]
name = "scatrs-py"
description = "Python extension module for scatrs"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
publish = false

[lib]
name = "_scatrs"
crate-type = ["cdylib"]

[dependencies]
scatrs = { path = "../scatrs" }
pyo3 = { version = "0.29", features = ["abi3-py311"] }
numpy = "0.29"

[features]
extension-module = ["pyo3/extension-module"]
```

`pyproject.toml`:
```toml
[build-system]
requires = ["maturin>=1.8,<2.0"]
build-backend = "maturin"

[project]
name = "scatrs"
dynamic = ["version"]
description = "Fast, linearized, rigorously validated single-scattering calculations"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Topic :: Scientific/Engineering :: Physics",
]

[project.optional-dependencies]
xarray = ["xarray>=2024.7"]
test = ["pytest>=8"]
docs = [
    "sphinx>=8",
    "furo>=2024.8",
    "myst-nb>=1.1",
    "numpydoc>=1.8",
    "ipykernel>=6.29",
]

[project.urls]
Documentation = "https://usask-arg.github.io/scatrs/"
Repository = "https://github.com/usask-arg/scatrs"

[dependency-groups]
dev = ["maturin>=1.8,<2.0", "pytest>=8", "mypy>=1.13", "ruff>=0.8", "pre-commit>=4"]

[tool.maturin]
manifest-path = "scatrs-py/Cargo.toml"
python-source = "python"
module-name = "scatrs._scatrs"
features = ["scatrs-py/extension-module"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "NPY", "RUF"]
```

`.python-version`:
```
3.12
```

- [ ] **Step 2: Write the failing Python tests**

`tests/test_package.py`:
```python
from importlib.metadata import version

import scatrs


def test_version_matches_distribution_metadata():
    assert scatrs.__version__ == version("scatrs")


def test_public_names():
    for name in (
        "special",
        "ScatrsError",
        "InvalidInputError",
        "NotConvergedError",
        "OutsideValidatedDomainError",
        "__version__",
    ):
        assert hasattr(scatrs, name)
```

`tests/test_special.py`:
```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scatrs import special


def test_gauss_legendre_matches_numpy():
    x, w = special.gauss_legendre(7)
    xr, wr = np.polynomial.legendre.leggauss(7)
    np.testing.assert_allclose(x, xr, rtol=0, atol=1e-15)
    np.testing.assert_allclose(w, wr, rtol=1e-14)


def test_gauss_legendre_integrates_polynomials_exactly():
    x, w = special.gauss_legendre(257)
    for k in range(0, 40, 2):
        assert abs(np.sum(w * x**k) - 2 / (k + 1)) < 1e-14


def test_wigner_d_00_is_legendre():
    x = np.linspace(-1, 1, 12).reshape(3, 4)
    d = special.wigner_d(0, 0, 10, x)
    assert d.shape == (3, 4, 11)
    for ell in range(11):
        expected = np.polynomial.legendre.Legendre.basis(ell)(x)
        np.testing.assert_allclose(d[..., ell], expected, atol=1e-14)


def test_wigner_d_22_closed_form():
    x = np.array([-0.3, 0.2, 0.9])
    d = special.wigner_d(2, 2, 2, x)
    np.testing.assert_allclose(d[:, 2], ((1 + x) / 2) ** 2, rtol=1e-15)
    np.testing.assert_array_equal(d[:, :2], 0.0)


def test_wigner_d_accepts_scalars_and_non_contiguous_arrays():
    x = np.linspace(-1, 1, 10)[::3]
    np.testing.assert_allclose(special.wigner_d(1, -1, 4, x), special.wigner_d(1, -1, 4, x.copy()))
    assert special.wigner_d(0, 0, 3, 0.5).shape == (4,)


def test_calls_from_threads_agree():
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: special.gauss_legendre(300)[1], range(8)))
    for r in results[1:]:
        np.testing.assert_array_equal(r, results[0])
```

`tests/test_errors.py`:
```python
import pytest

import scatrs
from scatrs import special


def test_invalid_n_raises_invalid_input():
    with pytest.raises(scatrs.InvalidInputError) as info:
        special.gauss_legendre(0)
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, scatrs.ScatrsError)


def test_negative_l_max_raises_invalid_input():
    with pytest.raises(scatrs.InvalidInputError):
        special.wigner_d(0, 0, -1, 0.5)


def test_x_outside_unit_interval_raises_from_rust():
    with pytest.raises(scatrs.InvalidInputError, match="-1 <= x <= 1"):
        special.wigner_d(0, 0, 3, [0.0, 1.5])


def test_not_converged_error_carries_accuracy():
    err = scatrs.NotConvergedError("not converged", achieved=1e-3, requested=1e-6)
    assert (err.achieved, err.requested) == (1e-3, 1e-6)
    assert isinstance(err, ArithmeticError)
```

- [ ] **Step 3: Confirm the tests fail**

Run: `uv sync && uv run pytest -q`
Expected: the build or import fails, because `python/scatrs` and `scatrs-py/src` do not exist yet.

- [ ] **Step 4: Implement the extension**

`scatrs-py/src/lib.rs`:
```rust
//! Python extension module `scatrs._scatrs`.

use pyo3::prelude::*;

mod errors;
mod special;

#[pymodule]
fn _scatrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(special::gauss_legendre, m)?)?;
    m.add_function(wrap_pyfunction!(special::wigner_d, m)?)?;
    Ok(())
}
```

`scatrs-py/src/errors.rs`:
```rust
//! Conversion of `scatrs::Error` into the exception classes defined in `scatrs.errors`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// The matching `scatrs.errors` exception for a core error.
pub fn to_pyerr(py: Python<'_>, err: scatrs::Error) -> PyErr {
    let message = err.to_string();
    let build = || -> PyResult<PyErr> {
        let module = py.import("scatrs.errors")?;
        let instance = match &err {
            scatrs::Error::InvalidInput(_) => module.getattr("InvalidInputError")?.call1((message.clone(),))?,
            scatrs::Error::NotConverged { achieved, requested } => module
                .getattr("NotConvergedError")?
                .call1((message.clone(), *achieved, *requested))?,
            scatrs::Error::OutsideValidatedDomain(_) => {
                module.getattr("OutsideValidatedDomainError")?.call1((message.clone(),))?
            }
            _ => module.getattr("ScatrsError")?.call1((message.clone(),))?,
        };
        Ok(PyErr::from_value(instance))
    };
    build().unwrap_or_else(|import_error| {
        let e = PyRuntimeError::new_err(message);
        e.set_cause(py, Some(import_error));
        e
    })
}
```

`scatrs-py/src/special.rs`:
```rust
//! Bindings for `scatrs::special`.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;

use crate::errors::to_pyerr;

/// Gauss–Legendre nodes and weights on [-1, 1].
#[pyfunction]
pub fn gauss_legendre<'py>(
    py: Python<'py>,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let rule = py
        .detach(|| scatrs::special::gauss_legendre::<f64>(n))
        .map_err(|e| to_pyerr(py, e))?;
    Ok((PyArray1::from_vec(py, rule.nodes), PyArray1::from_vec(py, rule.weights)))
}

/// Wigner d^l_{mn} for l = 0..=l_max at each x = cos θ; shape (len(x), l_max + 1).
#[pyfunction]
pub fn wigner_d<'py>(
    py: Python<'py>,
    m: i32,
    n: i32,
    l_max: usize,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xs = x.as_slice()?;
    let width = l_max
        .checked_add(1)
        .ok_or_else(|| PyOverflowError::new_err("l_max is too large"))?;
    let len = xs
        .len()
        .checked_mul(width)
        .ok_or_else(|| PyOverflowError::new_err("output is too large"))?;
    let values = py
        .detach(|| -> scatrs::Result<Vec<f64>> {
            let mut out = vec![0.0; len];
            for (&xi, row) in xs.iter().zip(out.chunks_exact_mut(width)) {
                scatrs::special::wigner_d(m, n, xi, row)?;
            }
            Ok(out)
        })
        .map_err(|e| to_pyerr(py, e))?;
    PyArray1::from_vec(py, values).reshape([xs.len(), width])
}
```

If `x.as_slice()?` does not compile because numpy 0.29 lacks `From<AsSliceError> for PyErr`, map the error explicitly: `.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?`.

- [ ] **Step 5: Implement the Python package**

`python/scatrs/errors.py`:
```python
"""Exception hierarchy raised by scatrs."""

from __future__ import annotations

__all__ = [
    "InvalidInputError",
    "NotConvergedError",
    "OutsideValidatedDomainError",
    "ScatrsError",
]


class ScatrsError(Exception):
    """Base class for every error raised by scatrs."""


class InvalidInputError(ScatrsError, ValueError):
    """An input is outside its mathematically valid range."""


class NotConvergedError(ScatrsError, ArithmeticError):
    """A computation did not reach the requested accuracy.

    Parameters
    ----------
    message : str
        Description of the failure.
    achieved : float, optional
        Accuracy actually reached.
    requested : float, optional
        Accuracy that was requested.
    """

    def __init__(
        self, message: str, achieved: float | None = None, requested: float | None = None
    ) -> None:
        super().__init__(message)
        self.achieved = achieved
        self.requested = requested


class OutsideValidatedDomainError(ScatrsError, ValueError):
    """Inputs are valid but outside the domain where accuracy has been validated."""
```

`python/scatrs/special.py`:
```python
"""Special functions and quadrature rules (thin wrappers over the Rust core)."""

from __future__ import annotations

import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scatrs import _scatrs
from scatrs.errors import InvalidInputError

__all__ = ["gauss_legendre", "wigner_d"]


def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss–Legendre nodes and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of nodes (at least 1). The rule integrates polynomials of degree up to
        ``2 * n - 1`` exactly.

    Returns
    -------
    nodes : numpy.ndarray
        Nodes in ascending order, shape ``(n,)``.
    weights : numpy.ndarray
        Weights, shape ``(n,)``.

    Notes
    -----
    The rule is computed in double-double arithmetic and rounded, so nodes and weights are
    accurate to about half an ulp. The cost grows as ``n**2``.
    """
    n = operator.index(n)
    if n < 1:
        raise InvalidInputError(f"invalid input: Gauss-Legendre rule needs n >= 1, got {n}")
    return _scatrs.gauss_legendre(n)


def wigner_d(m: int, n: int, l_max: int, x: ArrayLike) -> NDArray[np.float64]:
    r"""Wigner d-functions :math:`d^l_{mn}(\theta)` for :math:`l = 0, \ldots, l_{max}`.

    Uses the convention of Mishchenko, Travis & Lacis (2002), Appendix B, for example
    :math:`d^1_{01}(\theta) = \sin\theta / \sqrt{2}`.

    Parameters
    ----------
    m, n : int
        Orders.
    l_max : int
        Largest degree (at least 0).
    x : array_like
        :math:`\cos\theta`, with every element in [-1, 1].

    Returns
    -------
    numpy.ndarray
        Shape ``np.shape(x) + (l_max + 1,)``. Entries with ``l < max(|m|, |n|)`` are zero.
    """
    m, n, l_max = operator.index(m), operator.index(n), operator.index(l_max)
    if l_max < 0:
        raise InvalidInputError(f"invalid input: l_max must be >= 0, got {l_max}")
    xs = np.ascontiguousarray(x, dtype=np.float64)
    out = _scatrs.wigner_d(m, n, l_max, xs.reshape(-1))
    return out.reshape((*xs.shape, l_max + 1))
```

`python/scatrs/__init__.py`:
```python
"""scatrs: fast, linearized, rigorously validated single-scattering calculations."""

from scatrs import special
from scatrs._scatrs import __version__
from scatrs.errors import (
    InvalidInputError,
    NotConvergedError,
    OutsideValidatedDomainError,
    ScatrsError,
)

__all__ = [
    "InvalidInputError",
    "NotConvergedError",
    "OutsideValidatedDomainError",
    "ScatrsError",
    "__version__",
    "special",
]
```

`python/scatrs/_scatrs.pyi`:
```python
import numpy as np
from numpy.typing import NDArray

__version__: str

def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wigner_d(m: int, n: int, l_max: int, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

`python/scatrs/py.typed`: an empty file.

- [ ] **Step 6: Build and run the tests and checks**

Run:
```bash
uv sync
uv run maturin develop --uv --release
uv run pytest -q
uv run python -m mypy.stubtest scatrs._scatrs
uv run ruff check . && uv run ruff format --check .
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p scatrs
```
Expected: 12 Python tests pass; stubtest reports no errors; ruff, fmt, clippy and the Rust tests are clean.

If `ruff format --check` flags existing files, run `uv run ruff format .` and include those changes in this commit.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock scatrs-py pyproject.toml uv.lock .python-version python tests reference
git commit -m "feat: Python package with gauss_legendre and wigner_d bindings"
```

---

### Task 12: Documentation site

**Files:**
- Create:
  - `docs/conf.py`, `docs/index.md`, `docs/installation.md`, `docs/conventions.md`;
  - `docs/theory/special-functions.md`, `docs/validation/reference-suite.md`;
  - `docs/examples/quadrature.md`, `docs/api.md`.

**Interfaces:**
- Consumes: the installed `scatrs` package (Task 11).
- Produces: a strict Sphinx build. Any warning fails it. `docs/specs` and `docs/plans` are excluded from the build.

- [ ] **Step 1: Write the configuration and pages**

`docs/conf.py`:
```python
"""Sphinx configuration. Build after installing scatrs (``uv run maturin develop --uv``)."""

from importlib.metadata import version as _version

project = "scatrs"
author = "scatrs contributors"
release = _version("scatrs")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

exclude_patterns = ["_build", "jupyter_execute", "specs/**", "plans/**", "**.ipynb_checkpoints"]
source_suffix = {".md": "myst-nb", ".rst": "restructuredtext"}

html_theme = "furo"
html_title = "scatrs"

autodoc_default_options = {"members": True, "undoc-members": False}
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
```

`docs/index.md`:
````markdown
# scatrs

Fast, linearized, rigorously validated single-scattering calculations: Lorenz–Mie, layered
spheres and axisymmetric T-matrix, written in Rust with a Python interface.

```{warning}
Pre-alpha. Phase 0 provides the numerical foundations (scalar types, quadrature, Wigner
d-functions and the reference suite). Scattering calculations arrive in phase 1.
```

```{toctree}
:maxdepth: 2

installation
conventions
examples/quadrature
theory/special-functions
validation/reference-suite
api
```
````

`docs/installation.md`:
````markdown
# Installation

Wheels are not yet published. To build from source you need a Rust toolchain (1.85 or later)
and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/usask-arg/scatrs
cd scatrs
uv sync
uv run maturin develop --uv --release
uv run pytest
```

The Rust crate is used with `cargo add scatrs` once it is published.
````

`docs/conventions.md`:
```markdown
# Conventions

These hold throughout scatrs (design spec §3).

- **Refractive index:** m = n + ik, with Im m ≥ 0 meaning absorption. Negative Im m is
  rejected, never silently flipped.
- **Lengths:** radii and wavelength may use any unit as long as both use the same one.
  Cross sections come out in that unit squared.
- **Phase-matrix normalization:** ½ ∫ F₁₁(μ) dμ = 1.
- **Greek coefficients:** F₁₁(Θ) = Σ α₁,ₗ Pₗ(cos Θ), with α₁,₀ = 1 and α₁,₁ = 3g. The other
  elements follow Hovenier, van der Mee & Domke (2004).
- **Wigner d-functions:** Mishchenko, Travis & Lacis (2002), Appendix B; see
  {doc}`theory/special-functions`.
- **Euler angles:** z-y-z (α, β, γ), as in Mishchenko.
```

`docs/theory/special-functions.md`:
```markdown
# Special functions

## Gauss–Legendre quadrature

The n-point rule integrates polynomials of degree up to 2n − 1 exactly. Nodes are the roots
of Pₙ, found by Newton iteration from Tricomi's approximation

$$x_k \approx \left(1 - \frac{n-1}{8n^3}\right)\cos\frac{(4k-1)\pi}{4n+2},$$

with weights $w_k = 2 / \left[(1 - x_k^2)\,P_n'(x_k)^2\right]$.

A plain `f64` computation of the weights loses about n² ulps next to ±1, because
1 − x² is ill-conditioned there, and that is where Mie forward-scattering peaks sit. scatrs
therefore runs the iteration in double-double arithmetic and rounds the result. Against Arb
for n ≤ 1000, `f64` nodes are within 0.25 ulp and weights within 0.5 ulp.

## Wigner d-functions

scatrs uses the convention of Mishchenko, Travis & Lacis (2002), Appendix B, which matches
the common $d^j_{m'm}(\beta)$ with $m' = m$; for example
$d^1_{01}(\theta) = \sin\theta/\sqrt2$. The functions are computed by the upward recurrence

$$d^{s+1}_{mn} = \frac{(2s+1)\,[s(s+1)x - mn]\,d^s_{mn} - (s+1)\sqrt{s^2-m^2}\sqrt{s^2-n^2}\,d^{s-1}_{mn}}{s\sqrt{(s+1)^2-m^2}\sqrt{(s+1)^2-n^2}},$$

starting from

$$d^{s_0}_{mn} = \xi_{mn}\sqrt{\binom{2s_0}{|m-n|}}\,\sin^{|m-n|}\tfrac\theta2\,\cos^{|m+n|}\tfrac\theta2,
\qquad s_0 = \max(|m|,|n|),$$

where $\xi_{mn} = 1$ for $n \ge m$ and $(-1)^{m-n}$ otherwise. The `f64` absolute error
grows linearly in l, at about 12·(l + 1) ulps in the worst case measured against Arb.
```

`docs/validation/reference-suite.md`:
````markdown
# Reference suite

Ground truth for the tests comes from an independent generator, `reference/`, that uses
[Arb](https://arblib.org) ball arithmetic through python-flint. It never imports scatrs.

Every stored value has three parts:

- `hi`/`lo`: a double-double approximation;
- `dec`: 25 significant digits;
- `err`: a *rigorous* bound on |true − (hi + lo)|.

Records are labelled **certified** when every error source (including any series
truncation) is rigorously bounded, and **high-precision** otherwise. All phase 0 data is
certified.

| File | Contents |
|---|---|
| `constants.jsonl` | π, ln 2, … and the internal tables (1/k!, 10^(2^i), third words) |
| `dd_arith.jsonl` | exact double-double +, −, ×, ÷ results (checked against the proven bounds) |
| `dd_elementary.jsonl`, `dd_trig.jsonl` | elementary and trigonometric functions |
| `dd_parse.jsonl` | exact values of decimal literals |
| `gauss_legendre.jsonl` | certified Legendre roots and weights, n ≤ 1000 |
| `wigner_d.jsonl` | explicit-sum Wigner d values, l ≤ 300 |

To regenerate the data:

```bash
uv run --directory reference python -m scatrs_reference generate
```

A weekly CI job regenerates everything and fails if the bytes change.

**Tolerance policy:** tolerances in tests are never loosened without a written justification.
Tests print their measured worst error.
````

`docs/examples/quadrature.md`:
````markdown
---
file_format: mystnb
kernelspec:
  name: python3
---

# Example: Gauss–Legendre quadrature

```{code-cell} python
import numpy as np
import scatrs

x, w = scatrs.special.gauss_legendre(8)
print(w.sum())                      # 2
print(np.sum(w * x**14), 2 / 15)    # degree 14 <= 2n - 1 = 15: exact
```
````

`docs/api.md`:
````markdown
# API reference

## `scatrs.special`

```{eval-rst}
.. automodule:: scatrs.special
   :members:
```

## Errors

```{eval-rst}
.. automodule:: scatrs.errors
   :members:
   :show-inheritance:
```
````

- [ ] **Step 2: Build strictly**

Run: `uv run --extra docs maturin develop --uv --release && uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: `build succeeded` with no warnings. Fix any warning at its source; do not suppress it.

- [ ] **Step 3: Commit**

```bash
git add docs/conf.py docs/*.md docs/theory docs/validation docs/examples
git commit -m "docs: Sphinx site with conventions, theory, validation and API pages"
```

---

### Task 13: CI, release, scheduled checks and contributor tooling

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.github/workflows/docs.yml`, `.github/workflows/scheduled.yml`
- Create: `.github/dependabot.yml`, `.pre-commit-config.yaml`, `CONTRIBUTING.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: every command used in Tasks 1–12.
- Produces: workflows that lint cleanly under `actionlint`, plus a local dry run of every CI command.

- [ ] **Step 1: Write the workflows**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: pipx run ruff check .
      - run: pipx run ruff format --check .

  rust-test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test -p scatrs

  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@1.85
      - run: cargo check -p scatrs --lib

  python:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.14"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
      - uses: dtolnay/rust-toolchain@stable
      - run: python -m pip install ".[test]"
      - run: python -m pytest -q
      - name: Check type stubs
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.14'
        run: |
          python -m pip install mypy
          python -m mypy.stubtest scatrs._scatrs

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - uses: dtolnay/rust-toolchain@stable
      - run: python -m pip install ".[docs]"
      - run: python -m sphinx -W --keep-going -b html docs docs/_build/html

  reference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v6
      - run: uv run --directory reference python -m pytest -q
```

`.github/workflows/release.yml`: copy `/Users/djz828/dev/pratmo-rust/.github/workflows/release.yml` verbatim, with these changes:
1. Every `python-version:` in the build jobs becomes `"3.11"`. The abi3 floor is 3.11, so one wheel per platform serves every newer Python.
2. Change the header comment to say it was adapted from pratmo-rust's maturin template.
3. Append this job:
```yaml
  crates-io:
    name: Publish scatrs crate
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    needs: [publish]
    environment:
      name: crates-io
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@stable
      - uses: rust-lang/crates-io-auth-action@v1
        id: auth
      - run: cargo publish -p scatrs
        env:
          CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}
```

`.github/workflows/docs.yml`:
```yaml
name: Publish documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - uses: dtolnay/rust-toolchain@stable
      - run: python -m pip install ".[docs]"
      - run: python -m sphinx -W --keep-going -b html docs docs/_build/html
      - uses: actions/upload-pages-artifact@v4
        with:
          path: docs/_build/html

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

`.github/workflows/scheduled.yml`:
```yaml
name: Scheduled checks

on:
  schedule:
    - cron: "17 6 * * 1"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  reference-drift:
    name: Regenerate Arb reference data and check for drift
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v6
      - run: uv run --directory reference python -m scatrs_reference generate
      - run: git diff --exit-code -- reference/data
```

`.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: cargo
    directory: "/"
    schedule:
      interval: weekly
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
  - package-ecosystem: uv
    directory: "/"
    schedule:
      interval: weekly
  - package-ecosystem: uv
    directory: "/reference"
    schedule:
      interval: weekly
```

- [ ] **Step 2: Contributor tooling**

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
        exclude: ^reference/data/
      - id: end-of-file-fixer
        exclude: ^reference/data/
      - id: check-yaml
      - id: check-toml
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --all -- --check
        language: system
        types: [rust]
        pass_filenames: false
```

`CONTRIBUTING.md`:
````markdown
# Contributing to scatrs

## Setup

```bash
uv sync                                 # Python env with dev tools
uv run maturin develop --uv --release   # build the extension into it
uv run pre-commit install
```

## Checks (the same ones CI runs)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p scatrs
uv run pytest
uv run python -m mypy.stubtest scatrs._scatrs
uv run --directory reference python -m pytest
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Reference data

Arb reference values live in `reference/data/` and are produced by `reference/` (python-flint,
pinned). Regenerate them with
`uv run --directory reference python -m scatrs_reference generate [--only NAME]`
and commit the data together with the manifest. Never edit the data by hand. Tolerances in
tests are never loosened without a written justification in the test.

## Releasing

1. Set `[workspace.package] version` in `Cargo.toml`. It is the single version source:
   the Python version and the docs `release` are derived from it.
2. Update `CHANGELOG.md`, commit, and tag `vX.Y.Z`.
3. Pushing the tag builds wheels, publishes to PyPI (Trusted Publishing, environment `pypi`)
   and publishes the `scatrs` crate (crates.io Trusted Publishing, environment `crates-io`).
   The very first crates.io release must be published manually with an API token, because
   crates.io can only enable Trusted Publishing for a crate that already exists.
````

Add these lines to `CHANGELOG.md` under `### Added`:
```markdown
- `DoubleDouble` (~32 digits) and forward-mode `Dual<T, N>` scalar types implementing `Real`.
- Gauss–Legendre quadrature (computed in double-double) and Wigner d-functions.
- Arb reference-value generator and certified reference data.
- Python package `scatrs` with `special.gauss_legendre` and `special.wigner_d`.
- Documentation site, CI on Linux/macOS/Windows, wheel release workflow.
```

- [ ] **Step 3: Lint the workflows and dry-run the CI commands locally**

Run:
```bash
uvx --from actionlint-py actionlint
uvx pre-commit autoupdate
uvx pre-commit run --all-files
rustup toolchain install 1.85.0 --profile minimal
cargo +1.85.0 check -p scatrs --lib
cargo test -p scatrs
uv run pytest -q
uv run --directory reference python -m pytest -q
```
Expected:
- actionlint reports no errors;
- the pre-commit hooks pass (autoupdate may bump the hook revisions in `.pre-commit-config.yaml`; keep the bump);
- the MSRV check compiles;
- all tests pass.

- [ ] **Step 4: Commit**

```bash
git add .github .pre-commit-config.yaml CONTRIBUTING.md CHANGELOG.md
git commit -m "ci: CI on three OSes, wheel/crate release, docs deploy, weekly reference drift check"
```

- [ ] **Step 5: Hand off for remote verification**

Phase 0's exit criterion includes "CI green on 3 OSes", which needs the repository pushed to `usask-arg/scatrs`. That is an outward-facing action, so **ask the user before creating the remote or pushing.**

---

## Phase 0 exit checklist (maps to spec §13)

- [ ] `cargo test -p scatrs` passes locally. The worst-error reports for every reference-backed test are recorded in the task summaries.
- [ ] Special functions match the Arb references: Gauss–Legendre to ≤ 1 eps for f64; Wigner d to ≤ 16(l + 1) eps.
- [ ] The Python package builds, its tests pass, and stubtest is clean.
- [ ] The docs build strictly.
- [ ] actionlint is clean and the MSRV check passes.
- [ ] CI is green on Linux, macOS and Windows (after the user approves pushing).
````
