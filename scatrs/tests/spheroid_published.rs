//! Randomly oriented spheroids against the published values of Somerville, Auguié & Le Ru,
//! "A new numerically stable implementation of the T-matrix method for electromagnetic
//! scattering by spheroidal particles", JQSRT 123 (2013), 153–168, Tables 1 and 2: Q_sca and
//! Q_ext computed with the T-matrix and its inversion in arbitrary precision, to 15 digits.
//!
//! * Table 2 is model 1 of Kuik, de Haan & Hovenier, "Benchmark results for single scattering by
//!   spheroids", JQSRT 47 (1992), 477–489: a prolate spheroid of aspect ratio 4, x_max = 10.079368,
//!   m = 1.55 + 0.01i.
//! * Table 1 B: a prolate spheroid of aspect ratio 4, x_max = 35, m = 1.5 + 0.02i (about 20 s in
//!   release, so ignored by default).
//! * Table 1 A: a silver nanorod at its dipolar plasmon resonance, aspect ratio 20,
//!   m ≈ 1.02 + 13.4i, outside the validated domain.
//!
//! The paper's efficiencies are cross sections over the orientation-averaged projected area, S/4
//! for a convex particle (Cauchy), and its x_max is k times the long semi-axis; scatrs's are over
//! the cross section of the sphere of equal volume, with x its size parameter. Inside the
//! validated domain both values must be within the declared tolerance C·max(`rtol`, estimate);
//! outside it, within max(`rtol`, estimate), the error estimate's own claim.

use num_complex::Complex;
use scatrs::mie::MieOutput;
use scatrs::tmatrix::{Shape, TMatrix, TMatrixOptions, VALIDATED_DOMAIN};

/// A prolate spheroid of aspect ratio `h` = c/a and maximum size parameter `x_max` = k c, with the
/// paper's Q_sca and Q_ext.
struct Case {
    h: f64,
    x_max: f64,
    m: Complex<f64>,
    q_sca: f64,
    q_ext: f64,
}

impl Case {
    /// scatrs's aspect ratio a/c and equal-volume size parameter.
    fn inputs(&self) -> (f64, f64) {
        let eps = 1.0 / self.h;
        (eps, self.x_max * eps.powf(2.0 / 3.0))
    }

    /// The orientation-averaged projected area S/4 over the equal-volume sphere's cross section:
    /// S = 2πa²(1 + (c/(a e)) arcsin e), e = √(1 − a²/c²).
    fn area_ratio(&self) -> f64 {
        let (a, c) = (1.0, self.h);
        let e = (1.0 - (a / c).powi(2)).sqrt();
        let surface = 2.0 * std::f64::consts::PI * a * a * (1.0 + c / (a * e) * e.asin());
        surface / 4.0 / (std::f64::consts::PI * (a * a * c).powf(2.0 / 3.0))
    }

    /// The relative differences of Q_sca and Q_ext from the paper's, the error estimate and the
    /// solution's N and n_θ.
    fn compare(&self, opts: &TMatrixOptions) -> ([f64; 2], f64, usize, usize) {
        let (eps, x) = self.inputs();
        let mut tm = TMatrix::new();
        let s = tm
            .compute(Shape::Spheroid { aspect_ratio: eps }, x, self.m, opts)
            .unwrap();
        let q = s.efficiencies();
        let ratio = self.area_ratio();
        let rel = |ours: f64, paper: f64| (ours / ratio - paper).abs() / paper;
        (
            [rel(q.q_sca, self.q_sca), rel(q.q_ext, self.q_ext)],
            s.error_estimate(),
            s.n_terms(),
            s.n_theta(),
        )
    }

    /// Inside the validated domain: each difference within the declared tolerance.
    fn check_inside(&self, rtol: f64, name: &str) {
        let (eps, x) = self.inputs();
        assert!(
            VALIDATED_DOMAIN.contains(eps, x, self.m),
            "{name} is not inside"
        );
        let (diff, est, n, nth) = self.compare(&TMatrixOptions::new().with_rtol(rtol));
        for (d, output, label) in [
            (diff[0], MieOutput::Scattering, "Q_sca"),
            (diff[1], MieOutput::Extinction, "Q_ext"),
        ] {
            let tol = VALIDATED_DOMAIN.tolerance(output, rtol, est);
            eprintln!(
                "{name}: {label} differs by {d:.1e}, tolerance {tol:.1e} (N {n}, n_theta {nth}, estimate {est:.1e})"
            );
            assert!(d <= tol, "{name}: {label} differs by {d:.3e} > {tol:.3e}");
        }
    }
}

#[test]
fn kuik_benchmark_model_1() {
    Case {
        h: 4.0,
        x_max: 10.079368,
        m: Complex::new(1.55, 0.01),
        q_sca: 3.21290554203154,
        q_ext: 3.36721292620919,
    }
    .check_inside(1e-12, "Table 2");
}

#[test]
#[ignore = "about 20 s in release"]
fn large_prolate_spheroid() {
    Case {
        h: 4.0,
        x_max: 35.0,
        m: Complex::new(1.5, 0.02),
        q_sca: 1.68179883363743,
        q_ext: 2.3411277033233,
    }
    .check_inside(1e-10, "Table 1 B");
}

#[test]
fn silver_nanorod_at_resonance() {
    // a = 10 nm, c = 200 nm, ε = −318 + 48.5i in a solvent of index 1.33, at 2560 nm.
    let k = 2.0 * std::f64::consts::PI * 1.33 / 2560.0;
    let case = Case {
        h: 20.0,
        x_max: k * 200.0,
        m: Complex::new(-318.0f64, 48.5).sqrt() / 1.33,
        q_sca: 2.91644692910476,
        q_ext: 18.6590264036833,
    };
    let (eps, x) = case.inputs();
    assert!(!VALIDATED_DOMAIN.contains(eps, x, case.m));
    let rtol = 1e-12;
    let opts = TMatrixOptions::new()
        .with_rtol(rtol)
        .with_n_terms(31)
        .with_n_theta(400)
        .with_allow_unvalidated(true);
    let (diff, est, n, nth) = case.compare(&opts);
    let tol = rtol.max(est);
    eprintln!(
        "Table 1 A: Q_sca, Q_ext differ by {:.1e}, {:.1e}; estimate {est:.1e} (N {n}, n_theta {nth})",
        diff[0], diff[1]
    );
    assert!(
        diff[0] <= tol && diff[1] <= tol,
        "{diff:?} against {tol:.3e}"
    );
}
