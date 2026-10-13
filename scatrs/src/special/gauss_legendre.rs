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

/// The largest number of points [`gauss_legendre`] accepts: 2²⁰ = 1 048 576.
///
/// Larger `n` (up to `usize::MAX`) returns [`Error::InvalidInput`], so that no input can
/// overflow a size, exhaust memory or run for days. At the cap the O(n²) construction already
/// takes about 8 hours (28 ns·n² on an Apple M4 Pro), and the cap is still about 25 times the
/// roughly 4·10⁴ points that Mie quadrature needs at size parameter 2·10⁴. The accuracy
/// contract is checked against Arb on single roots at the cap. A faster O(n) method may raise
/// the cap later.
pub const MAX_GAUSS_LEGENDRE_POINTS: usize = 1 << 20;

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
        Err(Error::NotConverged {
            achieved: last,
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
/// to ±1 and to 0 at n = 5000, 20 000 and 2²⁰:
///
/// * `f64`: node absolute error ≤ 2⁻⁵², weight relative error ≤ 2⁻⁵². Measured: at most
///   0.25·2⁻⁵² and 0.49·2⁻⁵², the rounding of the double-double rule.
/// * `DoubleDouble`: node absolute error ≤ 2⁻⁹⁸, weight relative error ≤ 2⁻¹⁰⁰·(4 + n²). The n²
///   comes from the conditioning of 1 − x² next to ±1: a node error δ changes the weight by
///   2|x|δ/(1 − x²) relative, and 1 − x² ≈ 5.8/n² at the extreme nodes. Measured: at most
///   0.51·2⁻¹⁰⁶ for nodes; for weights 2⁻⁸⁵ at n = 5000, 2⁻⁸² at n = 20 000 and 2⁻⁷² at n = 2²⁰.
///
/// # Cost
///
/// O(n²): each of the ⌈n/2⌉ roots takes four to seven evaluations of the three-term recurrence
/// in double-double (about four on average for n ≥ 5000). Measured on an Apple M4 Pro: 0.8 ms
/// at n = 100, 32 ms at n = 1000, 0.72 s at n = 5000 and 11 s at n = 20 000. Memory is O(n):
/// 32 bytes per point for the recurrence coefficients, besides the rule itself.
///
/// # Errors
///
/// * [`Error::InvalidInput`] if `n == 0` or `n > MAX_GAUSS_LEGENDRE_POINTS`.
/// * [`Error::NotConverged`] if a Newton iteration fails to converge. This has not happened for
///   any `n` tried: every n ≤ 3000, and the roots nearest ±1 and 0 for 147 values of n up to
///   2²⁰.
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
    use super::*;

    /// |got − (hi + lo + lo2)|, accurate to far below 2⁻¹⁰⁶·|got|: the reference agrees with
    /// `got` to about 100 bits, so the double-double subtractions cancel almost exactly.
    fn error(got: Dd, [hi, lo, lo2]: [f64; 3]) -> f64 {
        ((got - Dd::new(hi, lo)) - lo2).abs().hi()
    }

    /// Single roots of the largest accepted rule, against Arb at 300 bits (python-flint 0.9.0,
    /// `arb.legendre_p_root(n, k, weight=True)` split into three doubles; the remainders are
    /// below 2⁻¹⁶⁰ relative). The whole rule would take hours, but each root costs O(n).
    #[test]
    fn roots_at_the_cap_match_arb() {
        const N: usize = MAX_GAUSS_LEGENDRE_POINTS;
        type Ref = (usize, [f64; 3], [f64; 3]);
        const REFS: [Ref; 4] = [
            (
                0,
                [
                    0.9999999999973701,
                    1.0312454907615059e-17,
                    6.46912313677422e-34,
                ],
                [
                    6.749136713940452e-12,
                    1.1793618456518766e-28,
                    -9.005883376763423e-45,
                ],
            ),
            (
                1,
                [
                    0.9999999999861433,
                    -1.9043773092809482e-17,
                    -1.4382386576536834e-33,
                ],
                [
                    1.571070557601383e-11,
                    5.7112218541792405e-28,
                    2.2937093581105864e-44,
                ],
            ),
            (
                2,
                [
                    0.9999999999659454,
                    1.2513794199728001e-17,
                    1.7247756829565976e-34,
                ],
                [
                    2.468554407634494e-11,
                    1.5316259107730884e-27,
                    -6.446720764937644e-44,
                ],
            ),
            (
                N / 2 - 1,
                [
                    1.4980273988537132e-6,
                    -9.990635557478403e-23,
                    -5.865200403019905e-39,
                ],
                [
                    2.9960547977051852e-6,
                    -8.559746921672828e-23,
                    -2.931030688665622e-39,
                ],
            ),
        ];
        let rec = Recurrence::new(N);
        let u2 = 2f64.powi(-106);
        for (i, x_ref, w_ref) in REFS {
            let (x, w) = rec.root_and_weight(N, i).unwrap();
            let ex = error(x, x_ref);
            let ew = error(w, w_ref) / w_ref[0];
            eprintln!(
                "n = 2^20, root {i}: node error {:.3} u², weight error {:.3e} u²",
                ex / u2,
                ew / u2
            );
            // The double-double contract, and the f64 contract after rounding.
            assert!(ex <= 2f64.powi(-98), "root {i}: node error {ex:e}");
            let wtol = 2f64.powi(-100) * (4.0 + (N * N) as f64);
            assert!(ew <= wtol, "root {i}: weight error {ew:e}");
            let ex64 = error(Dd::from_f64(x.hi()), x_ref);
            let ew64 = error(Dd::from_f64(w.hi()), w_ref) / w_ref[0];
            assert!(ex64 <= f64::EPSILON && ew64 <= f64::EPSILON, "root {i}");
        }
    }
}
