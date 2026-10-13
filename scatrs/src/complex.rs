//! Complex reciprocal for the scattering kernels.
//!
//! num-complex computes 1/(c + id) as (c − id)/(c² + d²). That overflows once |c| or |d|
//! exceeds about 10¹⁵⁴ (in `f64` and `DoubleDouble` alike) and underflows below about
//! 10⁻¹⁵⁴. Smith's algorithm (Smith 1962) divides by the larger of c and d first, so the
//! intermediate values stay near the size of the result.

use num_complex::Complex;

use crate::scalar::Real;

/// 1/z by Smith's algorithm: correct to a few ulps of |1/z| wherever 1/z is a normal number.
///
/// Infinite or NaN for z = 0, like `1.0 / 0.0`.
#[inline]
pub(crate) fn inv<T: Real>(z: Complex<T>) -> Complex<T> {
    let (c, d) = (z.re, z.im);
    if d.abs() <= c.abs() {
        let r = d / c;
        let t = (c + d * r).recip();
        Complex::new(t, -(r * t))
    } else {
        let r = c / d;
        let t = (c * r + d).recip();
        Complex::new(r * t, -t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::DoubleDouble as Dd;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    /// |got − want| / |want|.
    fn rel(got: Complex<f64>, want: Complex<f64>) -> f64 {
        (got - want).norm() / want.norm()
    }

    #[test]
    fn agrees_with_num_complex_on_ordinary_values() {
        for z in [
            c(3.0, -4.0),
            c(0.125, 8.0),
            c(0.0, -2.0),
            c(1.0, 1.0),
            c(-7.5, 0.25),
        ] {
            assert!(rel(inv(z), z.inv()) <= 2.0 * f64::EPSILON, "{z}");
        }
    }

    #[test]
    fn exact_cases() {
        assert_eq!(inv(c(2.0, 0.0)), c(0.5, -0.0));
        assert_eq!(inv(c(0.0, 4.0)), c(0.0, -0.25));
        assert_eq!(inv(c(-4.0, 0.0)), c(-0.25, 0.0));
    }

    #[test]
    fn no_overflow_or_underflow_where_num_complex_fails() {
        // c² + d² is 25·10⁶⁰⁰ or 25·10⁻⁶⁰⁰, outside the f64 range.
        for scale in [1e300, 1e-300] {
            let z = c(3.0 * scale, 4.0 * scale);
            let want = c(3.0 / 25.0 / scale, -4.0 / 25.0 / scale);
            assert!(rel(inv(z), want) <= 4.0 * f64::EPSILON, "scale {scale:e}");
            assert!(!rel(z.inv(), want).is_finite() || rel(z.inv(), want) > 0.5);
        }
    }

    #[test]
    fn zero_gives_infinity_or_nan_not_a_panic() {
        let w = inv(c(0.0, 0.0));
        assert!(!w.re.is_finite() || !w.im.is_finite());
    }

    /// `n · inv(d)` for `Complex<DoubleDouble>` against the Arb quotients n/d in
    /// `dd_complex.jsonl` (200 random pairs with moduli between 10⁻³⁰ and 10³⁰): within 8u²
    /// of |n/d|, including the rounding of the product.
    #[test]
    fn double_double_reciprocal_matches_arb() {
        let Some(text) = crate::test_data::reference_file("dd_complex.jsonl") else {
            return;
        };
        let rec: serde_json::Value = text
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("valid record"))
            .find(|r| r["case"] == "div")
            .expect("a div record");
        let get = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        };
        let input = |key: &str| get(&rec["inputs"][key]);
        let output = |key: &str, part: &str| get(&rec["outputs"][key][part]);
        let dd = |hi: &[f64], lo: &[f64], i: usize| Dd::new(hi[i], lo[i]);
        let (re_hi, re_lo, im_hi, im_lo) = (
            input("re_hi"),
            input("re_lo"),
            input("im_hi"),
            input("im_lo"),
        );
        let (y_re_hi, y_re_lo, y_im_hi, y_im_lo) = (
            input("y_re_hi"),
            input("y_re_lo"),
            input("y_im_hi"),
            input("y_im_lo"),
        );
        let (q_re_hi, q_re_lo) = (output("re", "hi"), output("re", "lo"));
        let (q_im_hi, q_im_lo) = (output("im", "hi"), output("im", "lo"));
        let u2 = 2f64.powi(-104);
        let mut worst = 0.0f64;
        for i in 0..re_hi.len() {
            let n = Complex::new(dd(&re_hi, &re_lo, i), dd(&im_hi, &im_lo, i));
            let d = Complex::new(dd(&y_re_hi, &y_re_lo, i), dd(&y_im_hi, &y_im_lo, i));
            let want = Complex::new(dd(&q_re_hi, &q_re_lo, i), dd(&q_im_hi, &q_im_lo, i));
            let err = ((n * inv(d) - want).norm() / want.norm()).hi();
            worst = worst.max(err / u2);
            assert!(err <= 8.0 * u2, "case {i}: error {:.2} u²", err / u2);
        }
        eprintln!("Complex<DoubleDouble> n·inv(d): worst error {worst:.2} u² of |n/d|");
    }
}
