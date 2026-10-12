//! Loader for the committed Arb reference data in `reference/data/`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

/// Parallel arrays describing reference values (see `reference/scatrs_reference/records.py`).
///
/// Value `i` is `hi[i] + lo[i] + lo2[i]` to within `err[i]` (about 2^-160 relative), and
/// `(hi[i], lo[i])` is a normalised double-double.
#[derive(Debug, Deserialize)]
pub struct RefValues {
    pub hi: Vec<f64>,
    pub lo: Vec<f64>,
    pub lo2: Vec<f64>,
    pub dec: Vec<String>,
    pub err: Vec<f64>,
}

impl RefValues {
    /// `|got - v|` for the double-double `got_hi + got_lo` and the reference value `v` of
    /// case `i`, as `|got - (hi + lo + lo2)| + err`.
    ///
    /// The difference is formed from error-free transformations and summed with
    /// compensation (Ogita, Rump & Oishi's Sum2), so it is as accurate as if computed in
    /// twice the working precision: the result is within about 2^-150 |v| of a rigorous bound,
    /// negligible against double-double errors of order u^2 = 2^-106.
    pub fn dd_error(&self, i: usize, got_hi: f64, got_lo: f64) -> f64 {
        let (s1, e1) = two_sum(got_hi, -self.hi[i]);
        let (s2, e2) = two_sum(got_lo, -self.lo[i]);
        let (mut sum, mut comp) = (0.0, 0.0);
        for t in [s1, s2, e1, e2, -self.lo2[i]] {
            let (s, e) = two_sum(sum, t);
            sum = s;
            comp += e;
        }
        (sum + comp).abs() + self.err[i]
    }
}

/// Error-free sum: `s = fl(a + b)` and `s + e = a + b` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
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

/// Load and validate a reference file: every line must parse, every output must have
/// equal-length `hi`/`lo`/`lo2`/`dec`/`err` arrays, and case names must be unique within the
/// file.
pub fn load(file: &str) -> Vec<Record> {
    let path = data_path(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read reference file {file} ({}): {e}",
            path.display()
        )
    });
    let records: Vec<Record> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("{file}:{}: invalid reference record: {e}", i + 1))
        })
        .collect();
    validate(file, &records);
    records
}

fn validate(file: &str, records: &[Record]) {
    let mut seen = BTreeSet::new();
    for r in records {
        assert!(
            seen.insert(r.case.as_str()),
            "{file}: duplicate case name {}",
            r.case
        );
        for (name, v) in &r.outputs {
            let lens = [
                v.hi.len(),
                v.lo.len(),
                v.lo2.len(),
                v.dec.len(),
                v.err.len(),
            ];
            assert!(
                lens.iter().all(|&n| n == lens[0]),
                "{file}: case {} output {name}: hi/lo/lo2/dec/err lengths differ {lens:?}",
                r.case
            );
        }
    }
}

pub fn find<'a>(records: &'a [Record], case: &str) -> &'a Record {
    records
        .iter()
        .find(|r| r.case == case)
        .unwrap_or_else(|| panic!("no reference case {case}"))
}
