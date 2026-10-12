//! Decimal literal parsing for `DoubleDouble`.
//!
//! The significant digits are accumulated exactly in a `u128`: up to 38 of them (10³⁸ < 2¹²⁷),
//! with any further digits rounding the 38th, half up (a relative change below 10⁻³⁷). Trailing
//! zeros move into the decimal exponent, leaving the value m·10ᵉ, which is formed as follows:
//!
//! * e ≥ 0 and m·10ᵉ < 2¹²⁸: the integer, converted once (exactly below 2¹⁰⁶);
//! * |e| ≤ 22: one multiplication or division by the exact `f64` power 10^|e|;
//! * otherwise m is multiplied by the exact `f64` power 10^(e mod 16) and then by the certified
//!   double-double powers 10^(2ⁱ), i = 4..8 (`POW10_2I`), for the set bits of e, in increasing
//!   order; or divided by the product of the same factors (in two divisions, by 10^(|e| − 256)
//!   and by 10²⁵⁶, beyond 10³⁰⁸).
//!
//! Error bound (u = 2⁻⁵³, first order): converting m costs at most u², an `f64` product or
//! quotient 2u² or 3u², a double-double product 4u², a double-double quotient 10u², and each
//! inexact certified power (10⁶⁴, 10¹²⁸, 10²⁵⁶) u². With at most four products (exponents up
//! to 308 set at most four of the bits 4..8), the multiplications cost at most 21u², a division
//! by up to 10³⁰⁸ 29u², and the two divisions beyond it 31u² (10^(|e| − 256) takes at most two
//! products). Results below 2⁻⁹⁶⁸ add at most 1.5·2⁻¹⁰⁷⁴ from the subnormal trailing word,
//! 3u² relative at 2⁻⁹⁶⁹. Every result in [2⁻⁹⁶⁹, MAX] is therefore within 34u² < 2⁻¹⁰⁰ relative
//! of the literal.
//!
//! Range ends. A product of leading words can overflow although the double-double result would
//! not (within an ulp of MAX); such results are recomputed at half scale. Results below 2⁻⁹⁶⁸ are
//! recomputed at scale 2¹²⁸ and scaled back keeping the leading word correctly rounded: a
//! subnormal result is rounded once using both words, and otherwise the trailing word is
//! rounded to the subnormal grid without being allowed to move the leading word. So the leading
//! word is the correctly rounded `f64` throughout, except within the error bound of a rounding
//! boundary.

use core::fmt;
use core::str::FromStr;

use super::consts::POW10_2I;
use super::{DoubleDouble as Dd, from_u128, pow2, scale_down_tiny};

/// Error returned when a string is not a decimal floating-point literal.
///
/// It carries no detail yet; `#[non_exhaustive]` leaves room to add some (such as the kind of
/// error) without a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParseDoubleDoubleError;

impl ParseDoubleDoubleError {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl fmt::Display for ParseDoubleDoubleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid double-double literal")
    }
}

impl std::error::Error for ParseDoubleDoubleError {}

/// Significant digits kept exactly; 10³⁸ (38 nines rounded up) still fits a `u128`.
const MAX_DIGITS: u32 = 38;
/// The largest exponent with an exact `f64` power of ten.
const MAX_EXACT_POW10: u32 = 22;
/// `POW10_F64[k] = 10^k`, exact in `f64` for k ≤ 22.
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];
/// m·10ᵉ with m ≥ 1 overflows for e > 308 (10³⁰⁹ > MAX).
const OVERFLOW_EXPONENT: i64 = 308;
/// m·10⁻ᵏ with m ≤ 10³⁸ is below half the smallest subnormal (2⁻¹⁰⁷⁵ ≈ 2.5·10⁻³²⁴) for k > 362.
const UNDERFLOW_EXPONENT: i64 = 362;
/// The largest power of ten below MAX; larger divisors are applied in two steps.
const MAX_POW10: u32 = 308;
/// Below 2⁻⁹⁶⁸ a result's trailing word may be subnormal.
const FULL_PRECISION_MIN: f64 = f64::from_bits((1023 - 968) << 52);
/// Binary exponent of the scale for results below `FULL_PRECISION_MIN`.
const TINY_SCALE_EXPONENT: i32 = 128;

impl Dd {
    /// Parses a decimal literal: an optional sign, then `inf`, `infinity` or `nan` in any case,
    /// or `digits[.digits][(e|E)[+-]digits]`, where either digit string before the exponent may
    /// be empty but not both. This is the grammar of `f64::from_str`, except that leading and
    /// trailing whitespace is ignored; `str::parse` (`FromStr`) does not ignore it, and accepts
    /// exactly the strings that `f64` accepts.
    ///
    /// Relative error ≤ 2⁻¹⁰⁰ for results in [2⁻⁹⁶⁹, MAX]; exact for integers below 2¹⁰⁶ and for
    /// doubles whose significant digits form an integer below 2⁵³ with at most 22 of them after
    /// the decimal point (such as 0.5 or 2.5).
    ///
    /// The leading word is the correctly rounded `f64` (barring values within the error bound of
    /// a rounding boundary), across the whole range: as for `f64::from_str`, values that round
    /// beyond MAX give ±∞, subnormal results are rounded once, values below half the smallest
    /// subnormal give ±0, and zero, NaN and infinite results keep the literal's sign (`"-0"` is
    /// −0). Digits beyond the 38th significant one round the 38th, half up.
    pub fn from_decimal_str(s: &str) -> Result<Self, ParseDoubleDoubleError> {
        parse_literal(s.trim().as_bytes())
    }
}

impl FromStr for Dd {
    type Err = ParseDoubleDoubleError;

    /// [`DoubleDouble::from_decimal_str`](Dd::from_decimal_str), but strict like `f64`:
    /// surrounding whitespace is an error, so generic `T: FromStr` code (and parsing a
    /// `Complex<DoubleDouble>`) accepts exactly the strings it accepts for `f64`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_literal(s.as_bytes())
    }
}

/// `[+-](inf|infinity|nan|digits[.digits][(e|E)[+-]digits])`, without surrounding whitespace.
fn parse_literal(t: &[u8]) -> Result<Dd, ParseDoubleDoubleError> {
    let (negative, body) = match t.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        _ => (false, t),
    };
    let magnitude = if body.eq_ignore_ascii_case(b"inf") || body.eq_ignore_ascii_case(b"infinity") {
        Dd::INFINITY
    } else if body.eq_ignore_ascii_case(b"nan") {
        Dd::NAN
    } else {
        let (m, e) = parse_decimal(body).ok_or(ParseDoubleDoubleError::new())?;
        from_decimal(m, e)
    };
    Ok(if negative { -magnitude } else { magnitude })
}

/// The significand m (at most 38 digits) and exponent e of `digits[.digits][(e|E)[+-]digits]`:
/// m·10ᵉ is the literal's value rounded to 38 significant digits, half up.
fn parse_decimal(s: &[u8]) -> Option<(u128, i64)> {
    let mut m: u128 = 0;
    let mut kept = 0;
    let mut exp: i64 = 0;
    let mut round_up = None; // whether the first dropped digit is 5 or more
    let mut any_digit = false;
    let mut seen_point = false;
    let mut i = 0;
    while let Some(&c) = s.get(i) {
        match c {
            b'0'..=b'9' => {
                any_digit = true;
                let d = c - b'0';
                if kept < MAX_DIGITS {
                    if m != 0 || d != 0 {
                        m = m * 10 + u128::from(d); // < 10^38: no overflow
                        kept += 1;
                    }
                    if seen_point {
                        exp -= 1;
                    }
                } else {
                    round_up.get_or_insert(d >= 5);
                    if !seen_point {
                        exp += 1;
                    }
                }
            }
            b'.' if !seen_point => seen_point = true,
            b'e' | b'E' => break,
            _ => return None,
        }
        i += 1;
    }
    if !any_digit {
        return None;
    }
    if i < s.len() {
        exp = exp.saturating_add(parse_exponent(&s[i + 1..])?);
    }
    if round_up == Some(true) {
        m += 1; // at most 10^38
    }
    Some((m, exp))
}

/// `[+-]digits`, saturating at ±(2⁶³ − 1): any exponent that large over- or underflows.
fn parse_exponent(s: &[u8]) -> Option<i64> {
    let (negative, digits) = match s.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut e: i64 = 0;
    for &c in digits {
        if !c.is_ascii_digit() {
            return None;
        }
        e = e.saturating_mul(10).saturating_add(i64::from(c - b'0'));
    }
    Some(if negative { -e } else { e })
}

/// m·10ᵉ for m ≤ 10³⁸.
fn from_decimal(mut m: u128, mut e: i64) -> Dd {
    if m == 0 {
        return Dd::ZERO;
    }
    while m % 10 == 0 {
        m /= 10;
        e = e.saturating_add(1);
    }
    if e >= 0 {
        if e > OVERFLOW_EXPONENT {
            return Dd::INFINITY;
        }
        let e = e as u32;
        if let Some(n) = 10u128.checked_pow(e).and_then(|p| m.checked_mul(p)) {
            return from_u128(n);
        }
        let v = mul_pow10(from_u128(m), e);
        if v.hi.is_infinite() {
            return mul_pow10_halved(m, e);
        }
        v
    } else {
        if e < -UNDERFLOW_EXPONENT {
            return Dd::ZERO;
        }
        let k = e.unsigned_abs() as u32;
        let v = div_pow10(from_u128(m), k);
        if v.hi < FULL_PRECISION_MIN {
            return tiny_quotient(m, k);
        }
        v
    }
}

/// v·10ᵉ for e ≤ 308, multiplying by increasing factors (so no intermediate overflows early).
fn mul_pow10(v: Dd, e: u32) -> Dd {
    if e <= MAX_EXACT_POW10 {
        return v * POW10_F64[e as usize];
    }
    let mut v = v * POW10_F64[(e & 15) as usize];
    for (i, p) in POW10_2I.iter().enumerate().skip(4) {
        if e & (1 << i) != 0 {
            v *= *p;
        }
    }
    v
}

/// v / 10ᵏ for k ≤ 362.
fn div_pow10(v: Dd, k: u32) -> Dd {
    if k <= MAX_EXACT_POW10 {
        v / POW10_F64[k as usize]
    } else if k <= MAX_POW10 {
        v / mul_pow10(Dd::ONE, k)
    } else {
        v / mul_pow10(Dd::ONE, k - 256) / POW10_2I[8]
    }
}

/// m·10ᵉ when the product of the leading words overflowed: at half scale every product of
/// leading words stays finite, and the final doubling overflows exactly when the result rounds
/// beyond MAX.
#[cold]
#[inline(never)]
fn mul_pow10_halved(m: u128, e: u32) -> Dd {
    mul_pow10(from_u128(m).mul_pow2(0.5), e).mul_pow2(2.0)
}

/// m·10⁻ᵏ below 2⁻⁹⁶⁸: computed at scale 2¹²⁸ (with full precision down to 2⁻¹⁰⁹⁶, below which
/// the result is zero anyway), then scaled back keeping the leading word correctly rounded.
#[cold]
#[inline(never)]
fn tiny_quotient(m: u128, k: u32) -> Dd {
    let v = div_pow10(from_u128(m).mul_pow2(pow2(TINY_SCALE_EXPONENT)), k);
    scale_down_tiny(v, TINY_SCALE_EXPONENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_of_ten_are_exact() {
        for (k, &p) in POW10_F64.iter().enumerate() {
            assert_eq!(p as u128, 10u128.pow(k as u32), "10^{k}");
        }
    }
}
