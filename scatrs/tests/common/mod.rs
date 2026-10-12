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
