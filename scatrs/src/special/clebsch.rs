//! Clebsch–Gordan coefficients C^{J M}_{ja ma; jb mb} (Condon–Shortley phases) for every J at
//! once, by the Schulten–Gordon recursion of the 3j symbol (Schulten & Gordon 1975):
//! f(J) = (J ja jb; −M ma mb) satisfies
//!
//!   J A(J+1) f(J+1) + B(J) f(J) + (J+1) A(J) f(J−1) = 0,
//!   A(J) = √((J² − (ja − jb)²)((ja + jb + 1)² − J²)(J² − M²)),
//!   B(J) = −(2J + 1)[ja(ja+1)(−M) − jb(jb+1)(−M) − J(J+1)(mb − ma)],
//!
//! run backward from J_max = ja + jb and forward from J_min = max(|ja − jb|, |M|), each stable
//! where the solution grows, joined where the backward one peaks, normalized by
//! Σ (2J + 1) f² = 1 and signed by the closed form at J_max, sgn = (−1)^{ja − jb + M}. Then
//! C^{J M} = (−1)^{ja − jb + M} √(2J + 1) f(J).
#![allow(dead_code)] // used by the T-matrix kernel, whose API lands later in phase 4a

use crate::scalar::Real;

fn int<T: Real>(k: i64) -> T {
    T::from_f64(k as f64)
}

/// The recursion coefficients A(J), B(J).
fn ab<T: Real>(j: i64, ja: i64, jb: i64, ma: i64, mb: i64) -> (T, T) {
    let m = ma + mb;
    let a2 =
        (j * j - (ja - jb) * (ja - jb)) * ((ja + jb + 1) * (ja + jb + 1) - j * j) * (j * j - m * m);
    let a = if a2 > 0 {
        int::<T>(a2).sqrt()
    } else {
        T::zero()
    };
    let b = -int::<T>(2 * j + 1)
        * (int::<T>(ja * (ja + 1)) * int::<T>(-m)
            - int::<T>(jb * (jb + 1)) * int::<T>(-m)
            - int::<T>(j * (j + 1)) * int::<T>(mb - ma));
    (a, b)
}

/// C^{J, ma+mb}_{ja ma; jb mb} for J = J_min, …, ja + jb, with J_min (the first value's J);
/// empty if |ma| > ja or |mb| > jb.
pub(crate) fn clebsch_gordan<T: Real>(
    ja: usize,
    ma: i64,
    jb: usize,
    mb: i64,
    out: &mut Vec<T>,
) -> usize {
    out.clear();
    let (ja, jb) = (ja as i64, jb as i64);
    let m = ma + mb;
    if ma.abs() > ja || mb.abs() > jb {
        return 0;
    }
    let j_min = (ja - jb).abs().max(m.abs());
    let j_max = ja + jb;
    if j_min > j_max {
        return j_min as usize;
    }
    let len = (j_max - j_min + 1) as usize;
    let idx = |j: i64| (j - j_min) as usize;
    // Backward, from f(J_max) = 1.
    let mut back = vec![T::zero(); len];
    back[idx(j_max)] = T::one();
    for j in (j_min + 1..=j_max).rev() {
        let (a_j, b_j) = ab::<T>(j, ja, jb, ma, mb);
        let (a_j1, _) = ab::<T>(j + 1, ja, jb, ma, mb);
        let next = if j < j_max {
            back[idx(j + 1)]
        } else {
            T::zero()
        };
        let den = int::<T>(j + 1) * a_j;
        back[idx(j - 1)] = if den == T::zero() {
            T::zero()
        } else {
            -(int::<T>(j) * a_j1 * next + b_j * back[idx(j)]) / den
        };
    }
    // Where the backward solution peaks.
    let peak = (0..len)
        .max_by(|&p, &q| {
            back[p]
                .abs()
                .partial_cmp(&back[q].abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut f = back.clone();
    if peak > 0 && j_min > 0 {
        // Forward, from f(J_min) = 1, up to the peak; scaled to meet the backward one there.
        let mut fwd = vec![T::zero(); peak + 1];
        fwd[0] = T::one();
        for p in 0..peak {
            let j = j_min + p as i64;
            let (a_j, b_j) = ab::<T>(j, ja, jb, ma, mb);
            let (a_j1, _) = ab::<T>(j + 1, ja, jb, ma, mb);
            let prev = if p > 0 { fwd[p - 1] } else { T::zero() };
            let den = int::<T>(j) * a_j1;
            fwd[p + 1] = if den == T::zero() {
                T::zero()
            } else {
                -(b_j * fwd[p] + int::<T>(j + 1) * a_j * prev) / den
            };
        }
        if fwd[peak] != T::zero() {
            let scale = back[peak] / fwd[peak];
            for p in 0..peak {
                f[p] = fwd[p] * scale;
            }
        }
    }
    let norm: T = (0..len).fold(T::zero(), |acc, p| {
        acc + int::<T>(2 * (j_min + p as i64) + 1) * f[p] * f[p]
    });
    let mut scale = norm.sqrt().recip();
    // sgn f(J_max) = (−1)^{ja − jb + M}.
    let want_negative = (ja - jb + m).rem_euclid(2) == 1;
    if (f[len - 1] < T::zero()) != want_negative {
        scale = -scale;
    }
    let phase = if (ja - jb + m).rem_euclid(2) == 1 {
        -T::one()
    } else {
        T::one()
    };
    for (p, &fp) in f.iter().enumerate() {
        let j = j_min + p as i64;
        out.push(phase * int::<T>(2 * j + 1).sqrt() * fp * scale);
    }
    j_min as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(n: i64) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }

    /// Racah's formula, for small arguments.
    fn racah(j1: i64, m1: i64, j2: i64, m2: i64, j: i64, m: i64) -> f64 {
        if m1 + m2 != m
            || j < (j1 - j2).abs()
            || j > j1 + j2
            || m1.abs() > j1
            || m2.abs() > j2
            || m.abs() > j
        {
            return 0.0;
        }
        let pre = ((2 * j + 1) as f64 * fact(j + j1 - j2) * fact(j - j1 + j2) * fact(j1 + j2 - j)
            / fact(j1 + j2 + j + 1))
        .sqrt()
            * (fact(j + m)
                * fact(j - m)
                * fact(j1 - m1)
                * fact(j1 + m1)
                * fact(j2 - m2)
                * fact(j2 + m2))
            .sqrt();
        let mut sum = 0.0;
        for k in 0..=(j1 + j2 + j) {
            let terms = [
                k,
                j1 + j2 - j - k,
                j1 - m1 - k,
                j2 + m2 - k,
                j - j2 + m1 + k,
                j - j1 - m2 + k,
            ];
            if terms.iter().any(|&t| t < 0) {
                continue;
            }
            let den: f64 = terms.iter().map(|&t| fact(t)).product();
            sum += if k % 2 == 0 { 1.0 } else { -1.0 } / den;
        }
        pre * sum
    }

    #[test]
    fn matches_racah_for_small_arguments() {
        let mut out = Vec::new();
        let mut worst = 0.0f64;
        for ja in 0..7usize {
            for jb in 0..7usize {
                for ma in -(ja as i64)..=ja as i64 {
                    for mb in -(jb as i64)..=jb as i64 {
                        let j_min = clebsch_gordan::<f64>(ja, ma, jb, mb, &mut out);
                        for (p, &v) in out.iter().enumerate() {
                            let j = (j_min + p) as i64;
                            let want = racah(ja as i64, ma, jb as i64, mb, j, ma + mb);
                            worst = worst.max((v - want).abs());
                        }
                    }
                }
            }
        }
        assert!(worst < 1e-13, "worst {worst:e}");
    }

    /// Orthonormality at large arguments: Σ_{ma} C^{J,M}_{ja ma; jb M−ma} C^{J',M}_{…} = δ_{JJ'}.
    #[test]
    fn orthonormal_at_large_arguments() {
        let (ja, jb, m) = (60usize, 45usize, 7i64);
        let mut table: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut out = Vec::new();
        for ma in -(ja as i64)..=ja as i64 {
            let mb = m - ma;
            if mb.abs() > jb as i64 {
                continue;
            }
            let j_min = clebsch_gordan::<f64>(ja, ma, jb, mb, &mut out);
            table.push((j_min, out.clone()));
        }
        let get = |t: &(usize, Vec<f64>), j: usize| {
            if j >= t.0 {
                t.1.get(j - t.0).copied().unwrap_or(0.0)
            } else {
                0.0
            }
        };
        let mut worst = 0.0f64;
        for j in [15usize, 30, 60, 90, 105] {
            for j2 in [15usize, 30, 105] {
                let s: f64 = table.iter().map(|t| get(t, j) * get(t, j2)).sum();
                let want = if j == j2 { 1.0 } else { 0.0 };
                worst = worst.max((s - want).abs());
            }
        }
        assert!(worst < 1e-12, "worst {worst:e}");
    }
}
