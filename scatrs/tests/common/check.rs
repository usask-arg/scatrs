//! Checking computed values against Arb reference values within the declared tolerances
//! (`scatrs::mie::ValidatedDomain::tolerance`), shared by the Mie reference tests.

use num_complex::Complex;
use scatrs::mie::{MieOutput, VALIDATED_DOMAIN, ValidatedDomain};
use scatrs::scalar::DoubleDouble as Dd;

use super::mie::Measure;
use super::{Record, load};

pub fn size_parameter(r: &Record) -> f64 {
    r.input_f64("x")
}

pub fn refractive_index(r: &Record) -> Complex<f64> {
    Complex::new(r.input_f64("m_re"), r.input_f64("m_im"))
}

/// A computed value as the unevaluated sum hi + lo.
pub trait Parts: Copy {
    fn parts(self) -> (f64, f64);
}

impl Parts for f64 {
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

impl Parts for Dd {
    fn parts(self) -> (f64, f64) {
        (self.hi(), self.lo())
    }
}

/// |got − v| for the reference value i of `name`.
pub fn error<T: Parts>(got: T, r: &Record, name: &str, i: usize) -> f64 {
    let (hi, lo) = got.parts();
    r.output(name).dd_error(i, hi, lo)
}

/// `DoubleDouble` stands in for the exact values in the accuracy map, so it must be accurate
/// far beyond `f64`, not within the same constants: it may reach 16 times them, in units of
/// (κ′ + cond)·2⁻¹⁰⁴ (measured: up to 1.5 times, in the amplitude derivatives at x = 2·10⁴).
/// Its error then moves a C of the map by at most 2⁻⁴⁸ of the declared constant.
pub const DOUBLE_DOUBLE_SLACK: f64 = 16.0;

/// The largest measured C of one output, and where.
pub struct Worst {
    pub what: String,
    /// C is measured in units of (κ′ + cond)·2⁻⁵²·`unit`: 1 for `f64`, 2⁻⁵² for `DoubleDouble`.
    pub unit: f64,
    /// The largest C allowed, as a multiple of the declared constant.
    pub slack: f64,
    pub c: f64,
    pub case: String,
}

impl Worst {
    pub fn new(what: &str) -> Self {
        Self {
            what: what.to_owned(),
            unit: 1.0,
            slack: 1.0,
            c: 0.0,
            case: String::new(),
        }
    }

    pub fn double_double(what: &str) -> Self {
        Self {
            what: format!("DoubleDouble {what}"),
            unit: f64::EPSILON,
            slack: DOUBLE_DOUBLE_SLACK,
            c: 0.0,
            case: String::new(),
        }
    }

    /// Checks the error `err` of `output` at the case of `r` against the declared tolerance in
    /// the output's measure, times `unit`·`slack`, and records its C.
    pub fn check(
        &mut self,
        output: MieOutput,
        err: f64,
        measure: Measure,
        r: &Record,
        detail: &str,
    ) {
        let (x, m) = (size_parameter(r), refractive_index(r));
        assert!(
            VALIDATED_DOMAIN.contains(x, m),
            "{} is outside the domain",
            r.case
        );
        let tol =
            VALIDATED_DOMAIN.tolerance(output, x, m, measure.condition) * self.unit * self.slack;
        assert!(
            measure.scale > 0.0 && tol.is_finite(),
            "{} {} {detail}: degenerate measure {measure:?}, tolerance {tol:e}",
            self.what,
            r.case
        );
        let rel = err / measure.scale;
        let c = rel / tol * self.slack * ValidatedDomain::constant(output);
        if c > self.c {
            self.c = c;
            self.case = format!("{} {detail}", r.case);
        }
        assert!(
            rel <= tol,
            "{} {} {detail}: error {rel:.3e} exceeds {tol:.3e} (C = {c:.2}, condition {:.3e})",
            self.what,
            r.case,
            measure.condition
        );
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            eprintln!(
                "{:<40} worst C = {:8.3} at {}",
                self.what, self.c, self.case
            );
        }
    }
}

pub fn records(file: &str, kind: &str) -> Vec<Record> {
    let records: Vec<Record> = load(file)
        .into_iter()
        .filter(|r| r.case.ends_with(kind))
        .collect();
    assert!(!records.is_empty(), "no {kind} records in {file}");
    records
}

/// max(a, b), NaN if either is NaN: `f64::max` would drop a NaN, and a NaN error or scale must
/// fail the check instead. For folds: `.fold(0.0, max_nan)`.
pub fn max_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}
