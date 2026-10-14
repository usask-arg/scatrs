//! Gauss–Legendre rules in O(n) from asymptotic expansions (Bogaert 2014).
//!
//! The k-th node from +1 is cos θₖ, with θₖ an expansion in v² = 1/(n + ½)² around
//! aₖ = jₖ·v (jₖ the k-th zero of J₀), and the weight an expansion in v² around
//! 2v²/(J₁(jₖ)²·aₖ/sin aₖ), with u = cot aₖ in the coefficients. The coefficients are Bogaert's,
//! as in FastGaussQuadrature.jl (MIT licence, © 2014 Alex Townsend), which keeps fewer terms
//! for larger n; this module keeps them all. The Bessel tables were recomputed with Arb, and
//! the double-double sine and cosine replace the platform's, so the rule is the same on every
//! platform.

use core::f64::consts::PI;

use crate::scalar::{DoubleDouble as Dd, Real};

/// The first 20 zeros of J₀ (Arb, 300 bits).
const J0_ZEROS: [f64; 20] = [
    2.404825557695773,
    5.520078110286311,
    8.653727912911013,
    11.791534439014281,
    14.930917708487787,
    18.071063967910924,
    21.21163662987926,
    24.352471530749302,
    27.493479132040257,
    30.634606468431976,
    33.77582021357357,
    36.917098353664045,
    40.05842576462824,
    43.19979171317673,
    46.341188371661815,
    49.482609897397815,
    52.624051841115,
    55.76551075501998,
    58.90698392608094,
    62.04846919022717,
];

/// J₁(j₀,ₖ)² for the first 10 zeros of J₀ (Arb, 300 bits).
const J1_SQUARED: [f64; 10] = [
    0.2695141239419169,
    0.1157801385822037,
    0.07368635113640822,
    0.05403757319811628,
    0.04266142901724309,
    0.0352421034909961,
    0.030021070103054673,
    0.02614739149530809,
    0.023159121824691393,
    0.02078382912226786,
];

/// The k-th zero of J₀ (k from 1): the table, then McMahon's expansion.
fn j0_zero(k: usize) -> f64 {
    if k <= J0_ZEROS.len() {
        J0_ZEROS[k - 1]
    } else {
        mcmahon(k)
    }
}

/// McMahon's expansion of the k-th zero of J₀ (NIST DLMF 10.21.19).
fn mcmahon(k: usize) -> f64 {
    let a = PI * (k as f64 - 0.25);
    let t = (0.125 / a) * (0.125 / a);
    a + 0.125 / a
        * (1.0 + t * (-124.0 / 3.0 + t * (120_928.0 / 15.0 + t * (-401_743_168.0 / 105.0))))
}

/// J₁(j₀,ₖ)² (k from 1): the table, then its expansion.
fn j1_squared(k: usize) -> f64 {
    if k <= J1_SQUARED.len() {
        J1_SQUARED[k - 1]
    } else {
        j1_squared_series(k)
    }
}

/// The expansion of J₁(j₀,ₖ)² in 1/(π(k − ¼)) of FastGaussQuadrature.jl.
fn j1_squared_series(k: usize) -> f64 {
    let a = PI * (k as f64 - 0.25);
    let t = 1.0 / (a * a);
    let p = -7.0 / 24.0
        + t * (151.0 / 80.0
            + t * (-172_913.0 / 8064.0
                + t * (461_797.0 / 1152.0 + t * (-171_497_088_497.0 / 15_206_400.0))));
    (2.0 + t * t * p) / (PI * a)
}

/// Writes the ascending nodes and the weights of the n-point rule, n = `nodes.len()` ≥ 2
/// (`weights.len()` must be n).
pub(super) fn asymptotic_rule<T: Real>(nodes: &mut [T], weights: &mut [T]) {
    let n = nodes.len();
    let v = 1.0 / (n as f64 + 0.5);
    let v2 = v * v;
    let half = n.div_ceil(2);
    for k in 1..=half {
        let a = j0_zero(k) * v;
        // The double-double sine and cosine use no platform libm, so the rule is bitwise the
        // same on every platform.
        let (sin_a, cos_a) = Dd::from_f64(a).sin_cos();
        let (sin_a, cos_a) = (sin_a.hi(), cos_a.hi());
        let u = cos_a / sin_a;
        let u2 = u * u;
        let (a2, a3) = (a * a, a * a * a);
        let a5 = a2 * a3;
        // Node: θ = a + v²·t₁ + v⁴·t₂ + v⁶·t₃.
        let t1 = (u - 1.0 / a) / 8.0;
        let t2 = (6.0 * (1.0 + u2) / a + 25.0 / a3 - u * (31.0 * u2 + 33.0)) / 384.0;
        let t3 = u * (2595.0 / 15360.0 + u2 * (6350.0 / 15360.0 + u2 * 3779.0 / 15360.0))
            - 1073.0 / 5120.0 / a5
            + (1.0 + u2)
                * (-(31.0 / 1024.0 * u2 + 11.0 / 1024.0) / a + u / 512.0 / a2 - 25.0 / 3072.0 / a3);
        let theta = a + v2 * (t1 + v2 * (t2 + v2 * t3));
        // Weight: 2/(J₁(jₖ)²·(a/sin a)·(1/v² + W₁ + v²W₂ + v⁴W₃)).
        let ua = u * a;
        let ai = 1.0 / a;
        let ai2 = ai * ai;
        let w1 = ((ua - 1.0) * ai2 + 1.0) / 8.0;
        let w2 = ((-27.0 + u2 * (-84.0 + u2 * -56.0))
            + ai2 * ((-3.0 * (1.0 - 2.0 * u2) + 6.0 * ua) + ai2 * (81.0 - 31.0 * ua)))
            / 384.0;
        let w3 = (153.0 / 1024.0 + u2 * (295.0 / 256.0 + u2 * (187.0 / 96.0 + u2 * 151.0 / 160.0)))
            + ai * ((-65.0 / 1024.0 + u2 * (-119.0 / 768.0 + u2 * -35.0 / 384.0)) * u
                + ai * ((5.0 / 512.0 + u2 * (15.0 / 512.0 + u2 * 7.0 / 384.0))
                    + ai * ((u2 / 512.0 - 13.0 / 1536.0) * u
                        + ai * ((-7.0 / 384.0 * u2 + 53.0 / 3072.0)
                            + ai * (3749.0 / 15360.0 * u + ai * (-1125.0 / 1024.0))))));
        let big_w = 1.0 / v2 + w1 + v2 * (w2 + v2 * w3);
        let w = 2.0 / (j1_squared(k) * (a / sin_a) * big_w);
        let x = Dd::from_f64(theta).cos().hi();
        nodes[n - k] = T::from_f64(x);
        nodes[k - 1] = T::from_f64(-x);
        weights[n - k] = T::from_f64(w);
        weights[k - 1] = T::from_f64(w);
    }
    if n % 2 == 1 {
        nodes[half - 1] = T::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::gauss_legendre::{ASYMPTOTIC_THRESHOLD, Recurrence};

    /// The rule against the double-double Newton rule: node absolute error and weight relative
    /// error within 16·2⁻⁵² (measured: at most 2.4 and 3.6), from the threshold up.
    #[test]
    fn matches_the_double_double_rule() {
        let n0 = ASYMPTOTIC_THRESHOLD + 1;
        for n in [n0, n0 + 1, 128, 150, 257, 500, 1000, 2000, 4001] {
            let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
            asymptotic_rule(&mut x, &mut w);
            let rec = Recurrence::new(n);
            let (mut ex, mut ew) = (0.0f64, 0.0f64);
            for i in 0..n.div_ceil(2) {
                let (xd, wd) = rec.root_and_weight(n, i).unwrap();
                // Root i is the (i + 1)-th largest, at index n − 1 − i.
                let k = n - 1 - i;
                ex = ex.max((Dd::from_f64(x[k]) - xd).abs().hi() / f64::EPSILON);
                ew = ew.max(((Dd::from_f64(w[k]) - wd) / wd).abs().hi() / f64::EPSILON);
            }
            assert!(
                ex <= 16.0 && ew <= 16.0,
                "n = {n}: node {ex:.2}, weight {ew:.2} eps"
            );
        }
    }

    /// Each expansion is already within 2·2⁻⁵² of the last tabulated values (Arb: at most
    /// 1.5·2⁻⁵² for the zeros from k = 16, 1.15·2⁻⁵² for J₁² at k = 10; below that, at k = 13
    /// and k = 9, more than 6·2⁻⁵²), so switching to it after the table loses nothing.
    #[test]
    fn the_expansions_take_over_from_the_tables() {
        for k in 16..=J0_ZEROS.len() {
            let table = J0_ZEROS[k - 1];
            let err = (mcmahon(k) - table).abs();
            assert!(err <= 2.0 * f64::EPSILON * table, "zero {k} of J0");
        }
        let k = J1_SQUARED.len();
        let table = J1_SQUARED[k - 1];
        let err = (j1_squared_series(k) - table).abs();
        assert!(err <= 2.0 * f64::EPSILON * table, "J1^2 at zero {k}");
    }
}
