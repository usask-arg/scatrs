//! Loader for the committed Arb reference data in `reference/data/`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
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

/// Load and validate a reference file: every line must parse, every output must have
/// equal-length `hi`/`lo`/`dec`/`err` arrays, and case names must be unique within the file.
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
            assert!(
                v.hi.len() == v.lo.len() && v.hi.len() == v.dec.len() && v.hi.len() == v.err.len(),
                "{file}: case {} output {name}: hi/lo/dec/err lengths differ ({}/{}/{}/{})",
                r.case,
                v.hi.len(),
                v.lo.len(),
                v.dec.len(),
                v.err.len()
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
