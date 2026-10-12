//! The truncated remainder `x - y * trunc(x / y)`, computed exactly and rounded once.
//!
//! The partial remainders of a long division by a double-double can need more than the
//! 107 bits a double-double holds, so they are kept as floating-point expansions: exact,
//! nonoverlapping sums of `f64`s (Shewchuk, "Adaptive precision floating-point arithmetic
//! and fast robust geometric predicates", Discrete Comput. Geom. 18, 1997).

use super::{DoubleDouble, fast_two_sum, two_prod, two_sum};

/// An exact sum of nonzero, nonoverlapping `f64`s stored in order of increasing magnitude.
struct Expansion(Vec<f64>);

impl Expansion {
    fn of(x: DoubleDouble) -> Self {
        let mut v = Vec::with_capacity(16);
        for c in [x.lo, x.hi] {
            if c != 0.0 {
                v.push(c);
            }
        }
        Self(v)
    }

    /// Adds `b` exactly (Shewchuk's Grow-Expansion with zero elimination).
    fn add(&mut self, b: f64) {
        let mut q = b;
        let mut k = 0;
        for i in 0..self.0.len() {
            let (s, h) = two_sum(q, self.0[i]);
            q = s;
            if h != 0.0 {
                self.0[k] = h;
                k += 1;
            }
        }
        self.0.truncate(k);
        if q != 0.0 {
            self.0.push(q);
        }
    }

    fn add_dd(&mut self, y: DoubleDouble) {
        self.add(y.lo);
        self.add(y.hi);
    }

    fn sub_dd(&mut self, y: DoubleDouble) {
        self.add(-y.lo);
        self.add(-y.hi);
    }

    /// Subtracts `n * y` exactly, for an integer `n` with `|n| < 2^53` and `|n * y| <= MAX`.
    /// The products are exact even when subnormal, as `n` is an integer.
    fn sub_multiple(&mut self, n: f64, y: DoubleDouble) {
        let (p1, e1) = two_prod(n, y.hi);
        let (p2, e2) = two_prod(n, y.lo);
        for t in [e2, p2, e1, p1] {
            self.add(-t);
        }
        self.compress();
    }

    /// Shewchuk's Compress: the same value with no more components, nonadjacent, and with the
    /// largest component within one ulp of the sum. Works in place.
    fn compress(&mut self) {
        let e = &mut self.0;
        let m = e.len();
        if m < 2 {
            return;
        }
        // Top-down pass; the outputs fill e[bottom..m], above the inputs still to be read.
        let mut bottom = m - 1;
        let mut q = e[m - 1];
        for i in (0..m - 1).rev() {
            let (s, t) = fast_two_sum(q, e[i]);
            if t != 0.0 {
                e[bottom] = s;
                bottom -= 1;
                q = t;
            } else {
                q = s;
            }
        }
        e[bottom] = q;
        // Bottom-up pass; the outputs fill e[0..top], below the inputs still to be read.
        let mut top = 0;
        for i in bottom + 1..m {
            let (s, t) = fast_two_sum(e[i], q);
            if t != 0.0 {
                e[top] = t;
                top += 1;
            }
            q = s;
        }
        e[top] = q;
        e.truncate(top + 1);
        e.retain(|&c| c != 0.0); // only a zero total leaves a zero component
    }

    fn is_negative(&self) -> bool {
        // The largest component of a nonoverlapping expansion carries the sign of the sum.
        self.0.last().is_some_and(|&c| c < 0.0)
    }

    /// The sum rounded to `f64`, within a few ulps.
    fn approx(&self) -> f64 {
        self.0.iter().fold(0.0, |s, &c| s + c)
    }

    /// The sum rounded to a double-double: with a compressed expansion, the error is about
    /// one rounding of the low word, below u²·|sum|·(1 + O(u)).
    fn into_dd(mut self) -> DoubleDouble {
        self.compress();
        self.0
            .iter()
            .fold(DoubleDouble::ZERO, |acc, &c| acc.add_f64(c))
    }
}

/// `floor(log2 |v|)` for finite nonzero `v`, including subnormals.
fn exponent(v: f64) -> i32 {
    let bits = v.to_bits() & !(1 << 63);
    match (bits >> 52) as i32 {
        0 => -1011 - bits.leading_zeros() as i32,
        biased => biased - 1023,
    }
}

/// The largest double-double below `y > 0`, give or take one step of the low word.
fn just_below(y: DoubleDouble) -> DoubleDouble {
    let lo = if y.lo == 0.0 {
        -f64::from_bits(1) // the smallest subnormal
    } else if y.lo > 0.0 {
        f64::from_bits(y.lo.to_bits() - 1)
    } else {
        f64::from_bits(y.lo.to_bits() + 1)
    };
    DoubleDouble::new(y.hi, lo)
}

/// `x - y * trunc(x / y)` with the sign of `x` and magnitude below `|y|`, like `f64`'s `%`.
pub(super) fn rem(x: DoubleDouble, y: DoubleDouble) -> DoubleDouble {
    if !x.hi.is_finite() || y.hi.is_nan() || y.hi == 0.0 {
        return DoubleDouble::NAN;
    }
    if y.hi.is_infinite() || x.hi == 0.0 {
        return x;
    }
    let (xa, ya) = (x.abs(), y.abs());
    if xa < ya {
        return x;
    }
    let mut r = Expansion::of(xa);
    // Long division: while the quotient exceeds 2^52, remove about 50 of its leading bits
    // at a time by subtracting an exact multiple of ya * 2^k. Each step lowers the exponent
    // of the partial remainder by at least 49, so there are at most about 43 steps.
    loop {
        let a = r.approx();
        if a == 0.0 {
            break; // r is exactly zero
        }
        let k = exponent(a) - exponent(ya.hi) - 51;
        if k <= 0 {
            break;
        }
        let ys = ya.ldexp(k); // exact: scaling up, and |ys| <= |r| * 2^-50
        // |a / ys.hi| lies in [2^50, 2^52); shrink it slightly so that n * ys.hi <= |a|.
        let n = (a / ys.hi * (1.0 - 2f64.powi(-50))).trunc();
        r.sub_multiple(n, ys);
    }
    // Now |r| < 2^53 * ya: one more quotient step, then exact corrections into [0, ya).
    let mut n = (r.approx() / ya.hi).trunc();
    if !(n * ya.hi).is_finite() {
        n -= n.signum(); // |x| is within an ulp of MAX
    }
    r.sub_multiple(n, ya);
    while r.is_negative() {
        r.add_dd(ya);
    }
    loop {
        r.sub_dd(ya);
        if r.is_negative() {
            r.add_dd(ya);
            break;
        }
    }
    // r is exact and 0 <= r < ya; the final rounding may reach ya, so step back below it.
    let mut z = r.into_dd();
    if z >= ya {
        z = just_below(ya);
    }
    if x.hi.is_sign_negative() { -z } else { z }
}
