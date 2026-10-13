//! Gauss–Legendre quadrature on [−1, 1].
//!
//! Nodes come from Newton iteration on Pₙ in double-double arithmetic, started from
//! Tricomi's approximation, and are then rounded to the requested type. Next to ±1 the weights
//! are ill-conditioned in the nodes: a node error of one ulp changes the weight by about n²/6
//! units of 2⁻⁵² (relative), so a plain `f64` computation loses that much there; even from the
//! correctly rounded nodes, the extreme weight at n = 1000 is about 7·10⁴ units off. Mie
//! forward-scattering peaks sit next to +1. Rounding from double-double keeps `f64` nodes and
//! weights within about half an ulp. The cost is O(n²); callers that reuse a rule should cache
//! it.

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

/// The largest number of points [`gauss_legendre`] accepts: 2¹⁷ = 131 072.
///
/// Larger `n` (up to `usize::MAX`) returns [`Error::InvalidInput`]. The cap is about three
/// times the roughly 4·10⁴ points that Mie quadrature needs at size parameter 2·10⁴. It will be
/// raised when an O(n) method replaces the present O(n²) one (raising it is not a breaking
/// change, lowering it would be).
///
/// Below the cap a call can still be expensive, and it cannot be interrupted:
///
/// * time: about 28 ns·n² for large n on an Apple M4 Pro with FMA, so about 8 minutes at the
///   cap, and roughly 10 times longer on x86_64 builds without the `fma` target feature, where
///   every double-double operation calls a software `mul_add`;
/// * memory: (32 + 2·`size_of::<T>()`)·n bytes, 32 bytes per point of recurrence coefficients
///   besides the rule itself: 6.3 MB for `f64` at the cap, 277 MB for `Dual<DoubleDouble, 64>`.
pub const MAX_GAUSS_LEGENDRE_POINTS: usize = 1 << 17;

/// Newton steps stop once |Δx| ≤ 2⁻⁹⁰; one further step then reaches rounding level.
const CONVERGED: f64 = f64::from_bits((1023 - 90) << 52); // 2⁻⁹⁰
const MAX_ITERATIONS: usize = 100;

/// Recurrence coefficients for Pⱼ = aⱼ·x·Pⱼ₋₁ − bⱼ·Pⱼ₋₂, with aⱼ = (2j − 1)/j and bⱼ = (j − 1)/j.
struct Recurrence {
    a: Vec<Dd>,
    b: Vec<Dd>,
}

impl Recurrence {
    fn new(n: usize) -> Self {
        let a = (0..=n)
            .map(|j| {
                if j == 0 {
                    Dd::ZERO
                } else {
                    Dd::from_f64((2 * j - 1) as f64) / j as f64
                }
            })
            .collect();
        let b = (0..=n)
            .map(|j| {
                if j == 0 {
                    Dd::ZERO
                } else {
                    Dd::from_f64((j - 1) as f64) / j as f64
                }
            })
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
    ///
    /// After a step |Δx| ≤ 2⁻⁹⁰, one more (polishing) step takes x to rounding level, as
    /// Newton converges quadratically; that step must itself be at most 2⁻⁹⁰, or iteration
    /// goes on. On failure, `achieved` is the last |Δx| (its leading word; +∞ if it was NaN).
    fn newton(&self, n: usize, guess: f64) -> Result<(Dd, Dd)> {
        let mut x = Dd::from_f64(guess);
        let mut last = f64::INFINITY;
        let mut polish = false;
        for _ in 0..MAX_ITERATIONS {
            let (p, dp) = self.legendre_and_derivative(n, x);
            let dx = p / dp;
            x -= dx;
            last = dx.hi().abs();
            let small = last <= CONVERGED;
            if polish && small {
                let (_, dp) = self.legendre_and_derivative(n, x);
                return Ok((x, dp));
            }
            polish = small;
        }
        Err(Error::NotConverged {
            achieved: if last.is_nan() { f64::INFINITY } else { last },
            requested: CONVERGED,
        })
    }

    /// The root x ≥ 0 of Pₙ that is the (i + 1)-th largest, for i < ⌈n/2⌉, and its weight.
    fn root_and_weight(&self, n: usize, i: usize) -> Result<(Dd, Dd)> {
        let (x, dp) = if 2 * i + 1 == n {
            (Dd::ZERO, self.legendre_and_derivative(n, Dd::ZERO).1)
        } else {
            let nf = n as f64;
            let k = (i + 1) as f64;
            let theta = core::f64::consts::PI * (4.0 * k - 1.0) / (4.0 * nf + 2.0);
            // The double-double cosine uses no platform libm, so the starting point is the
            // same on every platform; it matters, since the last bits of a double-double root
            // depend on where Newton starts.
            let cos_theta = Dd::from_f64(theta).cos().hi();
            let guess = (1.0 - (nf - 1.0) / (8.0 * nf * nf * nf)) * cos_theta;
            self.newton(n, guess)?
        };
        let w = Dd::from_f64(2.0) / ((Dd::ONE - x) * (Dd::ONE + x) * dp * dp);
        Ok((x, w))
    }
}

/// Computes the `n`-point Gauss–Legendre rule, exact for polynomials of degree ≤ 2n − 1.
///
/// Nodes ascend, and the rule is exactly symmetric: `nodes[i] == -nodes[n - 1 - i]` and
/// `weights[i] == weights[n - 1 - i]`; for odd `n` the middle node is `+0`. The rule is
/// computed in double-double and converted with [`Real::from_f64_pair`], so every `T` gets the
/// same rule rounded once: the `f64` rule is the double-double rule rounded to nearest, and
/// `Dual` rules are constants (zero derivatives) whose values are bitwise those of `f64` or
/// `DoubleDouble`. No platform libm is used (the starting guesses come from the double-double
/// cosine), only IEEE 754 operations and fused multiply-adds, so the bits of the rule do not
/// depend on the platform.
///
/// # Accuracy
///
/// Verified against Arb for every node of the rules with n ≤ 1000, and for the nodes nearest
/// to ±1 and to 0 at n = 5000, 20 000 and 2¹⁷ (the cap):
///
/// * `f64`: node absolute error ≤ 2⁻⁵², weight relative error ≤ 2⁻⁵². Measured: at most
///   0.25·2⁻⁵² and 0.49·2⁻⁵², the rounding of the double-double rule.
/// * `DoubleDouble`: node absolute error ≤ 2⁻⁹⁸, weight relative error ≤ 2⁻¹⁰⁰·(4 + n²). The n²
///   comes from the conditioning of 1 − x² next to ±1: a node error δ changes the weight by
///   2|x|δ/(1 − x²) relative, and 1 − x² ≈ 5.8/n² at the extreme nodes. Measured: at most
///   0.51·2⁻¹⁰⁶ for nodes; for weights about 2⁻⁸⁵, 2⁻⁸² and 2⁻⁸⁰ at n = 5000, 20 000 and 2¹⁷.
///
/// # Cost
///
/// O(n²) time and O(n) memory; see [`MAX_GAUSS_LEGENDRE_POINTS`] for the cost model. Each of the
/// ⌈n/2⌉ roots takes four to seven evaluations of the three-term recurrence in double-double
/// (about four on average for n ≥ 5000). Measured on an Apple M4 Pro: 0.8 ms at n = 100,
/// 32 ms at n = 1000, 0.72 s at n = 5000 and 11 s at n = 20 000. The call cannot be
/// interrupted.
///
/// # Errors
///
/// * [`Error::InvalidInput`] if `n == 0` or `n > MAX_GAUSS_LEGENDRE_POINTS`.
/// * [`Error::NotConverged`] if a Newton iteration fails to converge, with `achieved` the
///   last Newton step |Δx| (absolute, in x; +∞ if it was NaN) and `requested` 2⁻⁹⁰. The tests
///   have not found any `n` where this happens: they check every root of every rule with
///   n ≤ 300 (and, in an ignored test, n ≤ 3000), and the roots nearest ±1 and 0 for over
///   a hundred values of n up to the cap.
///
/// # Examples
///
/// ```
/// use scatrs::special::gauss_legendre;
///
/// // The 3-point rule integrates x⁴ exactly: ∫ x⁴ dx over [-1, 1] = 2/5.
/// let rule = gauss_legendre::<f64>(3)?;
/// let integral: f64 = rule.nodes.iter().zip(&rule.weights).map(|(x, w)| w * x.powi(4)).sum();
/// assert!((integral - 0.4).abs() < 1e-15);
/// # Ok::<(), scatrs::Error>(())
/// ```
pub fn gauss_legendre<T: Real>(n: usize) -> Result<GaussLegendre<T>> {
    if n == 0 {
        return Err(Error::InvalidInput(
            "Gauss-Legendre rule needs n >= 1".into(),
        ));
    }
    if n > MAX_GAUSS_LEGENDRE_POINTS {
        return Err(Error::InvalidInput(format!(
            "Gauss-Legendre rule needs n <= {MAX_GAUSS_LEGENDRE_POINTS}, got {n}"
        )));
    }
    let rec = Recurrence::new(n);
    let mut nodes = vec![T::zero(); n];
    let mut weights = vec![T::zero(); n];
    for i in 0..n.div_ceil(2) {
        let (x, w) = rec.root_and_weight(n, i)?;
        let w = T::from_f64_pair(w.hi(), w.lo());
        // For odd n the middle node is first set to −0 and then overwritten with +0.
        nodes[i] = T::from_f64_pair(-x.hi(), -x.lo());
        nodes[n - 1 - i] = T::from_f64_pair(x.hi(), x.lo());
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    Ok(GaussLegendre { nodes, weights })
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;
    use std::collections::BTreeSet;

    use super::*;

    /// A reference value: [hi, lo, lo2, err], with |v − (hi + lo + lo2)| ≤ err.
    type RefValue = [f64; 4];

    /// A rule size, node indices, and those nodes and weights.
    type RefRule = (usize, Vec<usize>, Vec<RefValue>, Vec<RefValue>);

    /// The Arb record `case` of `reference/data/gauss_legendre.jsonl` (written by
    /// `reference/scatrs_reference/generators/gauss_legendre.py`): the rule size, the ascending
    /// indices of the nodes it holds, and those nodes and weights. `None` without the reference
    /// data, as in the published crate.
    fn reference(case: &str) -> Option<RefRule> {
        let text = crate::test_data::reference_file("gauss_legendre.jsonl")?;
        let rec = text
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .find(|r| r["case"] == case)
            .unwrap_or_else(|| panic!("no reference case {case}: regenerate the data"));
        let size = |v: &serde_json::Value| v.as_u64().expect("integer") as usize;
        let values = |name: &str| -> Vec<RefValue> {
            let column = |k: &str| -> Vec<f64> {
                let out = &rec["outputs"][name][k];
                out.as_array()
                    .expect("array")
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            };
            let (hi, lo, lo2, err) = (column("hi"), column("lo"), column("lo2"), column("err"));
            (0..hi.len())
                .map(|i| [hi[i], lo[i], lo2[i], err[i]])
                .collect()
        };
        let indices = rec["inputs"]["indices"].as_array().expect("array");
        Some((
            size(&rec["inputs"]["n"]),
            indices.iter().map(size).collect(),
            values("nodes"),
            values("weights"),
        ))
    }

    /// A bound on |got − v|: the reference agrees with `got` to about 100 bits, so the
    /// double-double subtractions cancel almost exactly and add far less than 2⁻¹⁰⁶·|got|.
    fn error(got: Dd, [hi, lo, lo2, err]: RefValue) -> f64 {
        ((got - Dd::new(hi, lo)) - lo2).abs().hi() + err
    }

    /// Single roots of the largest accepted rule against Arb. The whole rule takes minutes,
    /// but each root costs O(n), so this checks the accuracy contract at the cap.
    #[test]
    fn roots_at_the_cap_match_arb() {
        let Some((n, indices, nodes, weights)) =
            reference(&format!("n={MAX_GAUSS_LEGENDRE_POINTS},roots"))
        else {
            return;
        };
        assert_eq!(n, MAX_GAUSS_LEGENDRE_POINTS);
        let rec = Recurrence::new(n);
        let u2 = 2f64.powi(-106);
        let (mut worst_x, mut worst_w) = (0.0_f64, 0.0_f64);
        for (k, &idx) in indices.iter().enumerate() {
            // `gauss_legendre` stores root i (the (i + 1)-th largest) at index n − 1 − i, and its
            // negation at index i.
            let (x, w) = rec.root_and_weight(n, idx.min(n - 1 - idx)).unwrap();
            let x = if idx < n / 2 { -x } else { x };
            let ex = error(x, nodes[k]);
            let ew = error(w, weights[k]) / weights[k][0];
            worst_x = worst_x.max(ex / u2);
            worst_w = worst_w.max(ew / u2);
            // The double-double contract, and the f64 contract after rounding.
            assert!(ex <= 2f64.powi(-98), "node {idx}: error {ex:e}");
            let wtol = 2f64.powi(-100) * (4.0 + (n * n) as f64);
            assert!(ew <= wtol, "weight {idx}: relative error {ew:e}");
            let ex64 = error(Dd::from_f64(x.hi()), nodes[k]);
            let ew64 = error(Dd::from_f64(w.hi()), weights[k]) / weights[k][0];
            assert!(ex64 <= f64::EPSILON, "node {idx}: f64 error {ex64:e}");
            assert!(ew64 <= f64::EPSILON, "weight {idx}: f64 error {ew64:e}");
        }
        eprintln!(
            "n = {n}: worst node error {worst_x:.3} u², worst weight error {worst_w:.3e} u² \
             (u² = 2^-106)"
        );
    }

    /// Newton finds the right root next to ±1 and 0, where Tricomi's guess is least accurate
    /// and the roots are closest together, for n from 1 to the cap: each root lies strictly
    /// inside its own interval of Bruns's inequality (see `tests/gauss_legendre.rs`), whose
    /// bounds are about 1/n² away from the root in x.
    #[test]
    fn roots_next_to_the_ends_and_the_middle_converge_up_to_the_cap() {
        let cap = MAX_GAUSS_LEGENDRE_POINTS;
        let mut ns: BTreeSet<usize> = (1..=64).collect();
        let mut m = 64.0_f64;
        while (m as usize) < cap {
            ns.insert(m as usize);
            m *= 1.1;
        }
        for e in 6..=cap.trailing_zeros() {
            ns.extend(
                [(1 << e) - 1, 1 << e, (1 << e) + 1]
                    .into_iter()
                    .filter(|&n| n <= cap),
            );
        }
        for &n in &ns {
            let rec = Recurrence::new(n);
            let half = n.div_ceil(2);
            let h = n as f64 + 0.5;
            let roots: BTreeSet<usize> = (0..half.min(6))
                .chain(half.saturating_sub(3)..half)
                .collect();
            for i in roots {
                let (x, w) = rec
                    .root_and_weight(n, i)
                    .unwrap_or_else(|e| panic!("n={n} root {i}: {e}"));
                let j = (i + 1) as f64; // root i is the j-th largest
                let (below, above) = ((j * PI / h).cos(), ((j - 0.5) * PI / h).cos());
                assert!(
                    below < x.hi() && x.hi() < above,
                    "n={n} root {i}: {x:?} outside ({below}, {above})"
                );
                assert!(w.hi() > 0.0, "n={n} root {i}: weight {w:?}");
            }
        }
        eprintln!("{} values of n up to {cap}: ok", ns.len());
    }
}
