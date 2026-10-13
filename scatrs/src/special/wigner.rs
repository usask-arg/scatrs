//! Wigner small-d functions d^l_{mn}(θ): see [`wigner_d`] for the convention, the method and
//! the contracts.
//!
//! Implementation: [`three_term`] runs MTL Eq. B.22 ([`ThreeTerm`]); [`difference_form`] and
//! [`general_difference_form`] run Reinsch's difference form for m = n ([`Difference`]) and for
//! any m, n ([`GeneralDifference`]). All three go through [`run`], which carries a separate
//! binary exponent while the values are below 2⁻⁹⁰⁰, so that an underflowing start value
//! ([`start_value`]) does not flush later, normal values to zero.
//!
//! Method selection: a difference form wherever |x| ≥ 0.9, and for m = ±n on the side of the
//! pole where |d| → 1 (x ≥ 0 for m = n, x < 0 for m = −n); the three-term form elsewhere.
//! The three-term form alone loses accuracy like l² next to the poles (measured up to
//! 266·(l+1) ulps at l = 2000 for (0, 0)), which is why the difference forms are used there.

use crate::scalar::Real;
use crate::{Error, Result};

/// Scaled values are kept within [2⁻²⁵⁶, 2²⁵⁶]: one step of the start product or of the
/// recurrence changes a value by far less than 2⁷⁶⁶, so no scaled value under- or overflows.
const SCALED_RANGE_BITS: i64 = 256;
/// The scaled recurrence hands over to unscaled values once they reach 2⁻⁹⁰⁰, comfortably above
/// 2⁻⁹⁶⁹, below which double-double loses precision.
const UNSCALED_FROM_BITS: i64 = -900;
/// The general difference form is used for |x| ≥ this. Below, the three-term form is as
/// accurate (in sweeps up to l = 4000 during code review, not committed) and, in double-double,
/// faster.
const GENERAL_DIFFERENCE_FROM: f64 = 0.9;

/// Fills `out[l]` with d^l_{mn}(θ) for `l = 0, …, out.len() − 1`, given `x = cos θ`.
///
/// Entries with `l < max(|m|, |n|)` are zero.
///
/// Convention: Mishchenko, Travis & Lacis (2002), *Scattering, Absorption, and Emission of
/// Light by Small Particles*, Appendix B. It is the common d^j_{m′m}(β) with (m′, m) renamed
/// (m, n). Examples: d¹₀₁(θ) = sin θ/√2, d¹₁₀(θ) = −sin θ/√2, d²₂₂(θ) = cos⁴(θ/2).
///
/// # Method
///
/// Upward recurrence in l (MTL Eq. B.22), C_s d^{s+1} = A_s d^s − B_s d^{s−1}, starting at
/// l₀ = max(|m|, |n|) with
/// d^{l₀}_{mn} = ξ √C(2l₀, |m − n|) · sin^{|m−n|}(θ/2) · cos^{|m+n|}(θ/2),
/// where ξ = 1 if n ≥ m and (−1)^{m−n} otherwise.
///
/// Next to x = 1 the error of this three-term form grows like l² rather than l, most for m = n
/// (where d^l → 1): there A_s ≈ B_s + C_s, so d^{s+1} ≈ 2d^s − d^{s−1}, and a rounding error at
/// degree k comes back multiplied by about k·ln(l/k). In `f64` it reached 266(l + 1)·2⁻⁵² at
/// l = 2000 for P_l at x = −1 + 2⁻⁵³, 27(l + 1)·2⁻⁵² at x = 1 for (5, 5), and 18(l + 1)·2⁻⁵² at
/// l = 4000 for (35, 34) at x = cos(4.2·10⁻⁴). So, with d^l_{mn}(x) = (−1)^{l+m} d^l_{m,−n}(−x)
/// mapping x = −1 onto x = 1, the recurrence runs in Reinsch's difference form on
/// Δ^s = d^s − d^{s−1}, with the small coefficient K_s = A_s − B_s − C_s computed without
/// cancellation:
///
/// * for m = ±n on the side of the pole where |d| → 1 (x ≥ 0 for m = n, x < 0 for m = −n),
///   where B_s and C_s are integers and K_s = (2s + 1)s(s + 1)(x − 1);
/// * for any other m, n where |x| ≥ 0.9;
///
/// and in the three-term form elsewhere, where its error grows like l.
///
/// The start value can lie far below the `f64` range although later degrees do not: at the
/// Gauss–Legendre node nearest a pole for about 500 points (sin θ ≈ 4.8·10⁻³), d^{150}_{0,150}
/// is about 2⁻¹¹⁵⁷, yet d^l_{0,150} is a normal double from l = 227 on (2⁻⁶⁸⁴ at l = 1000).
/// So while the values are below 2⁻⁹⁰⁰ the recurrence runs on values scaled by a separate
/// binary exponent.
///
/// # Accuracy
///
/// Absolute error against Arb at the exact `x` (the explicit finite sum, or Arb's Legendre
/// functions at very large l; see `reference/scatrs_reference/generators/wigner_d.py`):
///
/// * `f64`: ≤ 16(l + 1)·2⁻⁵²;
/// * `DoubleDouble`: ≤ (l + 1)·2⁻⁹⁷.
///
/// These bounds are measured, not proven. The tests validate them up to l = 2.5·10⁵ for (0, 0)
/// and (0, 2), and for general (m, n) at sampled degrees up to l = 2000 (to l = 4000 for two
/// cases next to the poles, and to l = 3000 for start values below the normal range). The Arb
/// tests check:
///
/// * every degree l ≤ 100 of 17 pairs (m, n), covering all sign combinations, m = ±n and
///   |m|, |n| up to 50, at x = ±1, −1 + 2⁻⁵³, 1 − 3·2⁻⁵³, ±0.9999999, −0.5, 0, 0.3 and cos(π/6);
/// * sampled degrees up to l = 2000 of (0, 0), (2, ±2), (0, 2), (50, −37), (−37, −50) and
///   (7, 150); up to l = 4000 of (35, 34) at x = cos(4.2·10⁻⁴) and (0, 0) at x = −1 + 2⁻⁵³; up
///   to l = 1000–3000 of seven cases whose start value lies below the normal range; and up to
///   l = 2.5·10⁵ of (0, 0) and (0, 2) at seven x covering all three forms.
///
/// The worst measured errors are 0.53(l + 1)·2⁻⁵² and 0.002(l + 1)·2⁻⁹⁷ (at most 6·10⁻⁶ of the
/// double-double bound at l = 2.5·10⁵). Measured during code review (not committed): an
/// adversarial probe of about 4500 Arb cases with |m|, |n| ≤ 1000 and l ≤ 5000 (turning points,
/// the switch at |x| = 0.9, near-diagonal pairs at small lθ) found at most 0.74(l + 1)·2⁻⁵² and
/// 0.0034(l + 1)·2⁻⁹⁷, and probes of general (m, n) reached l = 5000 to 20 000. All integer
/// coefficients are exact in double-double for degrees below about 9·10⁷ (2^26.5) and
/// |m − n| below 2^26.5: for all orders below about 4.7·10⁷ in magnitude.
///
/// While |d^l| grows with l, before its first turning point, the error is also within the bound
/// relative to |d^l| (measured during code review, not committed: up to 1.05(l + 1)·2⁻⁵², at
/// the start value of (1313, 1313) at x = 0.2087); in the oscillatory region beyond, near zeros
/// of d^l, only the absolute bound holds. Values below the normal range are not flushed to
/// zero: where the start value underflows, the recurrence runs on scaled values, so the values
/// that are normal keep that relative accuracy, and smaller ones are rounded once to a
/// subnormal or zero.
/// (`DoubleDouble` has full precision only above 2⁻⁹⁶⁹.)
///
/// # Derivatives
///
/// With `x` a [`Dual`](crate::scalar::Dual) the derivatives are those of the computed values,
/// and the values are bitwise those of the underlying type. For |x| < 1 the derivatives d′ are
/// within (l + 1)·l times the value bound above, times max(|d′|, 1) and divided by 1 − x²
/// (measured: at most 0.02 of that). At x = ±1 a constant `x` (zero derivatives) gives zero
/// derivatives. A nonzero derivative of `x` at x = 1 gives, with a = |m − n|:
///
/// * a = 0 and a = 2: the exact derivative (finite; within 0.004 of the same bound at a = 2);
/// * a = 1: ±∞ or NaN, where the true derivative is infinite (sin(θ/2) = √((1 − x)/2));
/// * a ≥ 3: zero, the exact derivative.
///
/// The same holds at x = −1 with b = |m + n| in place of a. With nested duals
/// (`Dual<Dual<T, M>, N>`), second derivatives at x = ±1 are exact for a = 0 and a = 2, and not
/// finite for a = 1, where they are infinite. For a ≥ 3 the start value is the constant zero, so
/// every derivative beyond the first is zero. That is right for a ≥ 5, since d^l is
/// (1 − x)^(a/2) times a function smooth at x = 1, but silently wrong for a = 3, where the true
/// second derivative is infinite, and for a = 4, where it is finite but nonzero (7.5, 105 and
/// 500.5 for (2, −2) at l = 4, 8 and 12). For |x| < 1 derivatives of every order are those of
/// the computed values.
///
/// # Cost
///
/// O(max(|m|, |n|) + `out.len()`): for the start value, once sin(θ/2) and cos(θ/2) are known,
/// |m − n| divisions and square roots and 2|m − n| + |m + n| ≤ 4·max(|m|, |n|) multiplications;
/// then per degree one division and two square roots (three-term form), one division and no
/// square root (m = ±n difference form), or two divisions and one square root (general
/// difference form).
/// Measured on an Apple M4 Pro over the nodes of the 64-point Gauss–Legendre rule, up to
/// l = 2000, per degree: in `f64` 2.6 ns for (0, 0), 3.1 ns for (2, ±2) and 3.7 ns for (0, 2)
/// and other m ≠ ±n; in `DoubleDouble` 30, 38 and 45 ns.
///
/// # Errors
///
/// [`Error::InvalidInput`] if `x` (for a `Dual`, its value) is NaN or outside [−1, 1], even when
/// `out` is empty; `out` is then left unchanged. Any `m` and `n` are accepted, `i32::MIN`
/// included.
pub fn wigner_d<T: Real>(m: i32, n: i32, x: T, out: &mut [T]) -> Result<()> {
    check_x(x)?;
    let (m, n) = (i64::from(m), i64::from(n));
    let l_min = match usize::try_from(m.unsigned_abs().max(n.unsigned_abs())) {
        Ok(l_min) if l_min < out.len() => l_min,
        _ => {
            out.fill(T::zero());
            return Ok(());
        }
    };
    // Every entry from l_min on is written by the recurrence.
    out[..l_min].fill(T::zero());
    // d^l_{mn}(x) = (−1)^{l+m} d^l_{m,−n}(−x) maps the pole x = −1 onto x = 1.
    let reflect = x < T::zero() && (m == -n || -x >= T::from_f64(GENERAL_DIFFERENCE_FROM));
    let (n, x) = if reflect { (-n, -x) } else { (n, x) };
    if m == n && x >= T::zero() {
        difference_form(m, x, l_min, out);
    } else if x >= T::from_f64(GENERAL_DIFFERENCE_FROM) {
        general_difference_form(m, n, x, l_min, out);
    } else {
        three_term(m, n, x, l_min, out);
    }
    if reflect {
        // l = l_min + i with l + m odd: odd i if l_min + m is even, else even i.
        let first_odd = 1 - (l_min as i64 + m).rem_euclid(2) as usize;
        for v in out[l_min..].iter_mut().skip(first_odd).step_by(2) {
            *v = -*v;
        }
    }
    Ok(())
}

/// Allocating form of [`wigner_d`]: d^l_{mn}(θ) for `l = 0..=l_max`.
///
/// # Errors
///
/// [`Error::InvalidInput`] if `x` is NaN or outside [−1, 1] (checked first), or if the
/// (l_max + 1)·`size_of::<T>()` bytes of the result cannot be allocated: for `l_max ==
/// usize::MAX`, for sizes beyond `isize::MAX` bytes, and where the allocator refuses. On
/// systems that overcommit memory an allocation larger than the memory available can still
/// succeed, and the process is then killed when the values are written; callers taking
/// `l_max` from untrusted input should bound it.
pub fn wigner_d_vec<T: Real>(m: i32, n: i32, l_max: usize, x: T) -> Result<Vec<T>> {
    check_x(x)?;
    let too_large = || {
        Error::InvalidInput(format!(
            "wigner_d_vec cannot allocate l_max + 1 values for l_max = {l_max}"
        ))
    };
    let len = l_max.checked_add(1).ok_or_else(too_large)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| too_large())?;
    // Initialised once (safe code cannot hand uninitialised memory to `wigner_d`), which then
    // writes each entry once more.
    out.resize(len, T::zero());
    wigner_d(m, n, x, &mut out)?;
    Ok(out)
}

fn check_x<T: Real>(x: T) -> Result<()> {
    if (-T::one()..=T::one()).contains(&x) {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "wigner_d needs -1 <= x <= 1, got {:?}",
            x.approx_f64()
        )))
    }
}

/// The three-term recurrence (MTL Eq. B.22) for any m, n with l_min ≥ 1 (m = n = 0 always takes
/// the difference form).
fn three_term<T: Real>(m: i64, n: i64, x: T, l_min: usize, out: &mut [T]) {
    let (a, b) = ((m - n).unsigned_abs(), (m + n).unsigned_abs());
    let (magnitude, exp) = start_value(a, b, x);
    let start = if n >= m || (m - n) % 2 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let mut rec = ThreeTerm {
        x,
        m: m as f64,
        n: n as f64,
        q: T::zero(),
    };
    run(&mut rec, l_min, [T::zero(), start], exp, out);
}

/// The difference form for m = n (|m| = l_min) and x ≥ 0.
fn difference_form<T: Real>(m: i64, x: T, l_min: usize, out: &mut [T]) {
    let b = 2 * m.unsigned_abs();
    let (start, exp) = start_value(0, b, x);
    let w = x - T::one();
    let mut rec = Difference { w, m: m as f64 };
    if l_min == 0 {
        // d⁰ = 1, d¹ = x, Δ¹ = x − 1.
        out[0] = start;
        if out.len() > 1 {
            run(&mut rec, 1, [w, x], exp, out);
        }
    } else {
        // Δ^{l₀} = d^{l₀} − d^{l₀−1} = d^{l₀}.
        run(&mut rec, l_min, [start, start], exp, out);
    }
}

/// The difference form for m ≠ n (l_min ≥ 1) and x ≥ 0.
fn general_difference_form<T: Real>(m: i64, n: i64, x: T, l_min: usize, out: &mut [T]) {
    let (a, b) = ((m - n).unsigned_abs(), (m + n).unsigned_abs());
    let (magnitude, exp) = start_value(a, b, x);
    let start = if n >= m || (m - n) % 2 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let (mf, nf, lf) = (m as f64, n as f64, l_min as f64);
    let mut rec = GeneralDifference {
        w: x - T::one(),
        m: mf,
        n: nf,
        q: T::zero(),
        // √P vanishes at l₀ = max(|m|, |n|), so u = α there.
        u: T::from_f64(lf * (lf + 1.0)) - mf * nf,
    };
    run(&mut rec, l_min, [start, start], exp, out);
}

/// One step of a recurrence whose state is linear in the d's, with d^s in `state[1]`: scaling
/// the state by 2^k scales every later value by 2^k.
trait Step<T> {
    /// Advances the state from degree s to s + 1, for s ≥ 1.
    fn step(&mut self, s: usize, state: &mut [T; 2]);
}

/// Stores d^s, …, d^{out.len() − 1}, starting from degree s with state·2^exp (exp ≤ 0).
fn run<T: Real, R: Step<T>>(
    rec: &mut R,
    mut s: usize,
    mut state: [T; 2],
    mut exp: i64,
    out: &mut [T],
) {
    // Scaled phase: d^s = state[1]·2^exp while the values are below the normal range.
    while exp != 0 {
        out[s] = mul_pow2(state[1], exp);
        if s + 1 == out.len() {
            return;
        }
        rec.step(s, &mut state);
        s += 1;
        let k = log2_floor(state[0].abs().max(state[1].abs()));
        if k + exp >= UNSCALED_FROM_BITS {
            state = state.map(|v| mul_pow2(v, exp));
            exp = 0;
        } else if k.abs() > SCALED_RANGE_BITS {
            state = state.map(|v| mul_pow2(v, -k));
            exp += k;
        }
    }
    out[s] = state[1];
    while s + 1 < out.len() {
        rec.step(s, &mut state);
        s += 1;
        out[s] = state[1];
    }
}

/// MTL Eq. B.22 on the state (d^{s−1}, d^s).
struct ThreeTerm<T> {
    x: T,
    m: f64,
    n: f64,
    /// √(s² − m²)·√(s² − n²) for the current degree s: zero at s = max(|m|, |n|), and otherwise
    /// the denominator factor of the previous step, reused as the next step's coefficient.
    q: T,
}

impl<T: Real> Step<T> for ThreeTerm<T> {
    #[inline]
    fn step(&mut self, s: usize, state: &mut [T; 2]) {
        let [prev, cur] = *state;
        let (sf, mf, nf) = (s as f64, self.m, self.n);
        let c_prev = self.q * (sf + 1.0);
        let numer = (self.x * (sf * (sf + 1.0)) - mf * nf) * (2.0 * sf + 1.0) * cur - c_prev * prev;
        let s1 = sf + 1.0;
        // (s1 − m)(s1 + m) is s1² − m² rounded once, exact below 2⁵³.
        self.q =
            T::from_f64((s1 - mf) * (s1 + mf)).sqrt() * T::from_f64((s1 - nf) * (s1 + nf)).sqrt();
        *state = [cur, numer / (self.q * sf)];
    }
}

/// Reinsch's difference form of MTL Eq. B.22 for m = n, on the state (Δ^s, d^s) with
/// Δ^s = d^s − d^{s−1}. With C_s d^{s+1} = A_s d^s − B_s d^{s−1}, where
/// A_s = (2s + 1)(s(s + 1)x − m²), B_s = (s + 1)(s² − m²) and C_s = s((s + 1)² − m²),
/// A_s − B_s − C_s = (2s + 1)s(s + 1)(x − 1), so
/// Δ^{s+1} = ((2s + 1)s(s + 1)(x − 1)·d^s + B_s Δ^s)/C_s and d^{s+1} = d^s + Δ^{s+1}.
struct Difference<T> {
    /// x − 1, exact for x in [½, 1].
    w: T,
    m: f64,
}

impl<T: Real> Step<T> for Difference<T> {
    #[inline]
    fn step(&mut self, s: usize, state: &mut [T; 2]) {
        let [delta, cur] = *state;
        let (sf, mf) = (s as f64, self.m);
        let s1 = sf + 1.0;
        // Integers formed in T: in f64 they would round above 2⁵³ (s ≈ 2·10⁵), perturbing the
        // ratios B_s/C_s by 2⁻⁵³, far above double-double precision. In double-double they are
        // exact while (s + 1)² < 2⁵³, s below about 9·10⁷; f64 results are unchanged.
        let k = T::from_f64(2.0 * sf + 1.0) * sf * s1;
        let b = T::from_f64(s1) * ((sf - mf) * (sf + mf));
        let c = T::from_f64(sf) * ((s1 - mf) * (s1 + mf));
        // The reciprocal is off the dependency chain through d: 1.6 times faster in f64 than
        // dividing, for one more rounding of the small Δ.
        let next = (self.w * k * cur + delta * b) * (T::one() / c);
        *state = [next, cur + next];
    }
}

/// Reinsch's difference form of MTL Eq. B.22 for any m, n, on the state (Δ^s, d^s). As for
/// [`Difference`], K_s = A_s − B_s − C_s is small next to x = 1 and must be accurate in absolute
/// terms; the rounding of B_s = (s + 1)q_s and C_s = s·q_{s+1}, with q_s = √(s² − m²)·√(s² − n²),
/// only perturbs the small Δ. With α_s = s(s + 1) − mn, u_s = α_s − q_s and v_s = α_s − q_{s+1},
/// K_s = (2s + 1)s(s + 1)(x − 1) + (s + 1)u_s + s·v_s. The differences are formed without
/// cancellation: v_s = (α_s² − q_{s+1}²)/(α_s + q_{s+1}), whose numerator
/// α_s² − ((s + 1)² − m²)((s + 1)² − n²) = (s + 1)²(1 + (m − n)² − 2(s + 1)) + 2(s + 1)mn is
/// computed from integers, and u_{s+1} = v_s + 2(s + 1) since α_{s+1} = α_s + 2(s + 1).
struct GeneralDifference<T> {
    /// x − 1, exact for x in [½, 1].
    w: T,
    m: f64,
    n: f64,
    /// q_s for the current degree s (zero at s = max(|m|, |n|)).
    q: T,
    /// u_s for the current degree s.
    u: T,
}

impl<T: Real> Step<T> for GeneralDifference<T> {
    #[inline]
    fn step(&mut self, s: usize, state: &mut [T; 2]) {
        let [delta, cur] = *state;
        let (sf, mf, nf) = (s as f64, self.m, self.n);
        let s1 = sf + 1.0;
        // One square root of the product (exact in double-double, and in f64 below 2⁵³): the
        // difference form does not need q rounded as in the three-term recurrence.
        let q1 = (T::from_f64((s1 - mf) * (s1 + mf)) * ((s1 - nf) * (s1 + nf))).sqrt();
        let alpha = T::from_f64(sf * s1) - mf * nf;
        let dmn = mf - nf;
        let numer =
            T::from_f64(s1 * s1) * (1.0 + dmn * dmn - 2.0 * s1) + T::from_f64(2.0 * s1) * (mf * nf);
        let v = numer / (alpha + q1);
        // (2s + 1)s(s + 1) in T, as in `Difference`.
        let k = self.w * (T::from_f64(2.0 * sf + 1.0) * sf * s1) + self.u * s1 + v * sf;
        // As in `Difference`, the reciprocal keeps the division off the chain through d.
        let next = (k * cur + delta * (self.q * s1)) * (T::one() / (q1 * sf));
        (self.q, self.u) = (q1, v + 2.0 * s1);
        *state = [next, cur + next];
    }
}

/// √C(a + b, a)·sin^a(θ/2)·cos^b(θ/2) as `(v, e)` with value v·2^e and e ≤ 0.
///
/// The factors are interleaved to stay in range, and the rare products that leave
/// [2⁻²⁵⁶, 2²⁵⁶] are rescaled by exact powers of two, so the result is the plain product
/// bitwise wherever that stays in the normal range. The value is at most 1, so after the final
/// [`fold`] e < 0 only if the value is below 2⁻⁹⁰⁰.
fn start_value<T: Real>(a: u64, b: u64, x: T) -> (T, i64) {
    let half_sin = ((T::one() - x) * 0.5).sqrt();
    let half_cos = ((T::one() + x) * 0.5).sqrt();
    if a >= 2 && half_sin == T::zero() {
        // x = 1 (x = −1 is reflected onto it). The value is +0, as from the product below, but
        // sin^a(θ/2) = ((1 − x)/2)^(a/2) has a finite derivative there, which a product of `a`
        // factors √((1 − x)/2), each zero with an infinite derivative, would make NaN for a
        // `Dual`. For a = 2 the start value is √C(b + 2, 2)·(1 − x)/2·cos^b(θ/2) exactly (the
        // binomial in T, which neither overflows nor loses double-double precision); for a ≥ 3
        // its first derivative is zero.
        if a > 2 {
            return (T::zero(), 0);
        }
        let (b1, b2) = (b as f64 + 1.0, b as f64 + 2.0);
        let mut v = (T::one() - x) * 0.5 * (T::from_f64(b1) * b2 * 0.5).sqrt();
        for _ in 0..b {
            v *= half_cos;
        }
        return (v, 0);
    }
    let (low, high) = (pow2(-SCALED_RANGE_BITS), pow2(SCALED_RANGE_BITS));
    let mut v = T::one();
    let mut e = 0;
    for k in 1..=a.max(b) {
        if k <= a {
            v = v * (T::from_f64((b + k) as f64) / T::from_f64(k as f64)).sqrt() * half_sin;
        }
        if k <= b {
            v *= half_cos;
        }
        let size = v.abs();
        if (size < T::from_f64(low) && size != T::zero()) || size > T::from_f64(high) {
            let k = log2_floor(v);
            (v, e) = (mul_pow2(v, -k), e + k);
        }
    }
    fold(v, e)
}

/// Moves the exponent into the value where v·2^e is at least 2⁻⁹⁰⁰ (or v is zero).
fn fold<T: Real>(v: T, e: i64) -> (T, i64) {
    if e == 0 || v == T::zero() {
        (v, 0)
    } else if log2_floor(v) + e >= UNSCALED_FROM_BITS {
        (mul_pow2(v, e), 0)
    } else {
        (v, e)
    }
}

/// 2^k for −1022 ≤ k ≤ 1023.
fn pow2(k: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// ⌊log₂|v|⌋ of the leading `f64` of a nonzero finite `v` (subnormals included); 0 for zero and
/// non-finite values.
fn log2_floor<T: Real>(v: T) -> i64 {
    let v = v.approx_f64();
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    let biased = ((v.to_bits() >> 52) & 0x7ff) as i64;
    if biased == 0 {
        // Subnormal: scale into the normal range first.
        return log2_floor(v * pow2(64)) - 64;
    }
    biased - 1023
}

/// v·2^e for a finite `v` and a result of magnitude below 2¹⁰²³: exact while the result is
/// normal, and otherwise rounded once to a subnormal or zero (for `f64`; a double-double's low
/// word is rounded already below 2⁻⁹⁶⁹).
fn mul_pow2<T: Real>(v: T, e: i64) -> T {
    if v == T::zero() || e == 0 {
        return v;
    }
    // Scale v to [1, 2) (exactly, in at most two steps for a subnormal v) and then to the target
    // exponent t; below the normal range stop at 2⁻¹⁰²² first, so that only the last
    // multiplication rounds. Results below 2⁻¹⁰⁸³ round to zero, as does v·2⁻¹⁰²²·2⁻⁶².
    let ev = log2_floor(v);
    let mut unit = v;
    let mut k = -ev;
    while k != 0 {
        let step = k.clamp(-1022, 1023);
        unit *= pow2(step);
        k -= step;
    }
    let t = ev + e;
    if t >= -1022 {
        unit * pow2(t)
    } else {
        unit * pow2(-1022) * pow2((t + 1022).max(-62))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::DoubleDouble as Dd;

    /// v·2^e for f64 v, rounded to nearest-even, from the integer significand.
    fn reference_mul_pow2(v: f64, e: i64) -> f64 {
        let bits = v.to_bits();
        let (biased, frac) = (((bits >> 52) & 0x7ff) as i64, bits & ((1 << 52) - 1));
        let (sig, exp) = if biased == 0 {
            (frac, -1074)
        } else {
            (frac | (1 << 52), biased - 1075)
        };
        let target = exp + e; // value = sig·2^target
        // 2^k for −1074 ≤ k ≤ 1023, from its bits (`powi` underflows 1/2^k).
        let exact_pow2 = |k: i64| {
            if k >= -1022 {
                f64::from_bits(((k + 1023) as u64) << 52)
            } else {
                f64::from_bits(1 << (k + 1074))
            }
        };
        let mag = if target >= -1074 {
            // A multiple of 2⁻¹⁰⁷⁴ with a 53-bit significand: exact.
            sig as f64 * exact_pow2(target.min(1023 - 52))
        } else {
            let shift = (-1074 - target) as u32;
            let q = if shift >= 64 { 0 } else { sig >> shift };
            let rem = if shift >= 64 { sig } else { sig - (q << shift) };
            let half = if shift >= 65 {
                u64::MAX
            } else {
                1u64 << (shift - 1)
            };
            let q = if rem > half || (rem == half && q % 2 == 1) {
                q + 1
            } else {
                q
            };
            q as f64 * exact_pow2(-1074)
        };
        mag.copysign(v)
    }

    #[test]
    fn mul_pow2_is_exact_or_rounds_once() {
        let values = [
            1.0,
            -1.5,
            1.0 + f64::EPSILON,
            1.75 - f64::EPSILON,
            3.0e-300,
            2f64.powi(-1070) * 1.25,
            f64::MIN_POSITIVE,
            -5e-324,
            1.0e200,
        ];
        for v in values {
            for e in [
                -1200, -1130, -1100, -1080, -1075, -1074, -1060, -1023, -1000, -300, -1, 1, 300,
            ] {
                let expected = reference_mul_pow2(v, e);
                if expected.abs() >= 2f64.powi(1023) {
                    continue;
                }
                let got = mul_pow2(v, e);
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{v:e}·2^{e}: {got:e} vs {expected:e}"
                );
            }
        }
        // Double-double: exact while the low word stays above the subnormal range.
        let x = Dd::new(1.0, 2f64.powi(-60));
        let y = mul_pow2(x, -500);
        assert_eq!((y.hi(), y.lo()), (2f64.powi(-500), 2f64.powi(-560)));
        let z = mul_pow2(y, 500);
        assert_eq!(z, x);
    }

    #[test]
    fn log2_floor_handles_every_finite_double() {
        assert_eq!(log2_floor(1.0), 0);
        assert_eq!(log2_floor(1.999), 0);
        assert_eq!(log2_floor(-2.0), 1);
        assert_eq!(log2_floor(f64::MIN_POSITIVE), -1022);
        assert_eq!(log2_floor(5e-324), -1074);
        assert_eq!(log2_floor(3.0 * 5e-324), -1073);
        assert_eq!(log2_floor(f64::MAX), 1023);
        assert_eq!(log2_floor(0.0), 0);
        assert_eq!(log2_floor(f64::INFINITY), 0);
    }

    /// The numerator of v_s, α_s² − ((s + 1)² − m²)((s + 1)² − n²), in exact integers.
    #[test]
    fn difference_form_numerator_identity() {
        for m in -12..=12_i128 {
            for n in -12..=12_i128 {
                let l_min = m.abs().max(n.abs());
                for s in l_min.max(1)..l_min + 40 {
                    let s1 = s + 1;
                    let alpha = s * s1 - m * n;
                    let q2 = (s1 * s1 - m * m) * (s1 * s1 - n * n);
                    let formula = s1 * s1 * (1 + (m - n) * (m - n) - 2 * s1) + 2 * s1 * m * n;
                    assert_eq!(alpha * alpha - q2, formula, "m={m} n={n} s={s}");
                    // α_{s+1} = α_s + 2(s + 1).
                    assert_eq!((s + 1) * (s + 2) - m * n, alpha + 2 * s1);
                }
            }
        }
    }
}
