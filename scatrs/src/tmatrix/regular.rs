//! The regularized yₙ part of the EBCM integrals.
//!
//! Q = P + Y, with P the integrals of the regular jₙ part of hₙ = jₙ + i yₙ and Y those of i yₙ:
//! per element Y = 4π γₙ γ_k Σ w Σ_t a_t(θ) R_t(x), a sum of eight radial products
//! R_t = x^e Y_t(x) J_t(s x) (Y_t ∈ {yₙ, [x yₙ]'/x, yₙ/x}, J_t ∈ {j_k, [z j_k]'/z, j_k/z}, e = 2 or 4)
//! with angular factors a_t (see `angular_factors`).
//!
//! For a spheroid the terms of power ≤ 0 of every R_t's Laurent series in x integrate, together, to
//! exactly zero in every element with n ≠ k, and those of power < 0 alone when |n − k| ≥ 3
//! (research/tmatrix, findings 11–12): they are the electrostatic limit, whose outgoing and regular
//! multipoles pair to zero off the diagonal on any surface. They carry the catastrophic
//! cancellation of the plain integrals, up to 10¹⁶ at x = 15, ε = 4. Each element is therefore
//! accumulated in three exact forms, plain, minus the powers < 0, minus the powers ≤ 0, each with
//! the sum of the magnitudes of what it adds, and the form with the smallest sum is kept.
//!
//! At each node the kept part R⁺ of a product is evaluated either by its own series (the positive
//! powers) or as R − R_{≤0} from the Bessel values, whichever adds less in magnitude.
//!
//! The series are normalized at x_min = k r_min (ξ = x/x_min ≥ 1): yₙ(x) = Σ_l a_l x^{2l−n−1}
//! (a₀ = −(2n − 1)!!, a_l = −a_{l−1}/(2l(2l − 2n − 1))), scaled by σₙ: Ã_l = σₙ a_l x_min^{2l−n−1},
//! bounded; j_k(z) = Σ_i b_i z^{k+2i}, scaled by c_k: B̃_i = c_k b_i (s x_min)^{k+2i}, bounded by
//! (x_min/x_max)^k.

use num_complex::Complex;

use super::functions::Scales;
use crate::mie::kernel::{int, real};
use crate::scalar::Real;

/// The y-kinds and j-kinds of the radial products, and e.
const TYPES: [(Kind, Kind, i64); 8] = [
    (Kind::Plain, Kind::Plain, 2),
    (Kind::Plain, Kind::Deriv, 2),
    (Kind::Deriv, Kind::Plain, 2),
    (Kind::Deriv, Kind::Deriv, 2),
    (Kind::Plain, Kind::Over, 4),
    (Kind::Deriv, Kind::Over, 4),
    (Kind::Over, Kind::Plain, 4),
    (Kind::Over, Kind::Deriv, 4),
];

/// f, [z f]'/z or f/z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Plain,
    Deriv,
    Over,
}

/// The series terms at one node, for every order: the y-part Y_l = Ã_l ξ^{a_l} (with the kind's
/// coefficient) and the j-part Taylor terms t_i = B̃_i ξ^{b_i}, as (exponent of the first term,
/// values), with the prefix and suffix sums of the j-part (and of their magnitudes).
pub(crate) struct NodeSeries<T> {
    /// y[n][kind] = (a₀ exponent, terms Y_l).
    y: Vec<[(i64, Vec<T>); 3]>,
    /// j[k][kind] = (b₀ exponent, terms, prefix, prefix |·|, suffix, suffix |·|).
    j: Vec<[JTerms<T>; 3]>,
}

#[derive(Clone, Debug)]
struct JTerms<T> {
    b0: i64,
    head: Vec<Complex<T>>,
    head_abs: Vec<T>,
    tail: Vec<Complex<T>>,
    tail_abs: Vec<T>,
}

/// The regularized products of one pair at one node: R_full is supplied; the forms and their
/// magnitude sums.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Forms<T> {
    /// R⁺ minus the powers ≤ 0, and minus the powers < 0; their magnitude sums (the plain form's
    /// is |R_full|).
    pub(crate) le0: Complex<T>,
    pub(crate) le0_abs: T,
    pub(crate) lt0: Complex<T>,
    pub(crate) lt0_abs: T,
}

/// The coefficients Ã_l (per order n) and B̃_i (per order k), node-independent.
pub(crate) struct Series<T> {
    x_min: T,
    s: Complex<T>,
    a: Vec<Vec<T>>,
    b: Vec<Vec<Complex<T>>>,
}

impl<T: Real> Series<T> {
    /// The series up to the terms negligible (below ε/2⁸ of the largest) on x ≤ x_max.
    pub(crate) fn new(x_min: T, x_max: T, s: Complex<T>, n: usize, scales: &Scales<T>) -> Self {
        let mut a = vec![Vec::new(); n + 1];
        let x2 = x_min * x_min;
        let xi_max = x_max / x_min;
        let xi2_max = xi_max * xi_max;
        let tol = T::epsilon() * T::from_f64(1.0 / 256.0);
        let half = T::from_f64(0.5);
        for (nn, slot) in a.iter_mut().enumerate().skip(1) {
            // Ã₀ = −Π_{i=1}^{n} (2i − 1)/(x_min |r_i(x_min)|) (|h₀(x_min)| = 1/x_min).
            let mut a0 = -T::one();
            for i in 1..=nn {
                a0 *= int::<T>(2 * i - 1) / (x_min * scales.h_ratio[i]);
            }
            // Terms until they are past their peak and negligible at ξ_max (the term of l at
            // ξ_max is Ã_l ξ_max^{2l−n−1}; track it relative to Ã₀ ξ_max^{−n−1}).
            let mut v = vec![a0];
            let (mut at_max, mut peak) = (a0.abs(), a0.abs());
            for l in 1.. {
                let prev = v[l - 1];
                let den = int::<T>(2 * l) * (int::<T>(2 * l) - int::<T>(2 * nn + 1));
                let next = -(prev * x2) / den;
                v.push(next);
                let ratio = (x2 * xi2_max / den).abs();
                at_max *= ratio;
                peak = peak.max(at_max);
                if 2 * l > 2 * nn + 1 && ratio < half && at_max < tol * peak {
                    break;
                }
            }
            *slot = v;
        }
        let sx = s.scale(x_min);
        let sx2 = sx * sx;
        let mut b = vec![Vec::new(); n + 1];
        for (kk, slot) in b.iter_mut().enumerate() {
            // B̃₀ = Π_{i=1}^{k} s x_min/max(2i + 1, |z_max|) · |z_max| e^{−|Im z_max|} / 1 (j₀'s
            // b₀ = 1), the c_k of `Scales`.
            let mut b0 = real(scales.e_ratio[0].recip() * (-scales.im_z_max).exp());
            for i in 1..=kk {
                b0 = b0 * sx.scale((int::<T>(2 * i + 1) * scales.e_ratio[i]).recip());
            }
            let mut v = vec![b0];
            let (mut at_max, mut peak) = (b0.norm(), b0.norm());
            for i in 1.. {
                let den = int::<T>(2 * i) * int::<T>(2 * i + 2 * kk + 1);
                let prev = v[i - 1];
                v.push(-(prev * sx2).scale(den.recip()));
                let ratio = sx2.norm() * xi2_max / den;
                at_max *= ratio;
                peak = peak.max(at_max);
                if ratio < half && at_max < tol * peak {
                    break;
                }
            }
            *slot = v;
        }
        Self { x_min, s, a, b }
    }

    /// The per-node terms at ξ = x/x_min.
    pub(crate) fn at(&self, xi: T) -> NodeSeries<T> {
        let n = self.a.len() - 1;
        let inv_xmin = self.x_min.recip();
        let inv_sx = (self.s.scale(self.x_min)).inv();
        let mut y = Vec::with_capacity(n + 1);
        for (nn, a) in self.a.iter().enumerate() {
            if nn == 0 {
                y.push(core::array::from_fn(|_| (0, Vec::new())));
                continue;
            }
            let e0 = -(nn as i64) - 1;
            let start = xi.powi(e0 as i32);
            let xi2 = xi * xi;
            let mut plain = Vec::with_capacity(a.len());
            let mut deriv = Vec::with_capacity(a.len());
            let mut over = Vec::with_capacity(a.len());
            let mut p = start;
            for (l, &al) in a.iter().enumerate() {
                let v = al * p;
                plain.push(v);
                // [x y]'/x: exponent one less, coefficient (2l − n); y/x: exponent one less.
                let v1 = v / xi * inv_xmin;
                deriv.push(v1 * int::<T>(2 * l) - v1 * int::<T>(nn));
                over.push(v1);
                p *= xi2;
            }
            y.push([(e0, plain), (e0 - 1, deriv), (e0 - 1, over)]);
        }
        let mut j = Vec::with_capacity(n + 1);
        for (kk, b) in self.b.iter().enumerate() {
            let e0 = kk as i64;
            let xi2 = xi * xi;
            let mut p = xi.powi(e0 as i32);
            let mut plain = Vec::with_capacity(b.len());
            let mut deriv = Vec::with_capacity(b.len());
            let mut over = Vec::with_capacity(b.len());
            for (i, &bi) in b.iter().enumerate() {
                let v = bi.scale(p);
                plain.push(v);
                let v1 = (v * inv_sx).scale(xi.recip());
                deriv.push(v1.scale(int::<T>(kk + 1 + 2 * i)));
                over.push(v1);
                p *= xi2;
            }
            j.push([sums(e0, &plain), sums(e0 - 1, &deriv), sums(e0 - 1, &over)]);
        }
        NodeSeries { y, j }
    }
}

fn sums<T: Real>(b0: i64, terms: &[Complex<T>]) -> JTerms<T> {
    let len = terms.len();
    let zero = real(T::zero());
    let mut head = Vec::with_capacity(len);
    let mut head_abs = Vec::with_capacity(len);
    let (mut acc, mut acc_abs) = (zero, T::zero());
    for &t in terms {
        acc = acc + t;
        acc_abs += t.norm();
        head.push(acc);
        head_abs.push(acc_abs);
    }
    let mut tail = vec![zero; len + 1];
    let mut tail_abs = vec![T::zero(); len + 1];
    for i in (0..len).rev() {
        tail[i] = tail[i + 1] + terms[i];
        tail_abs[i] = tail_abs[i + 1] + terms[i].norm();
    }
    JTerms {
        b0,
        head,
        head_abs,
        tail,
        tail_abs,
    }
}

impl<T: Real> NodeSeries<T> {
    /// The forms of the product of type t for orders (n, k) at this node; `x_e` = (x², x⁴) at the
    /// node.
    pub(crate) fn forms(
        &self,
        t: usize,
        n: usize,
        k: usize,
        full: Complex<T>,
        x_e: [T; 2],
    ) -> Forms<T> {
        let (yk, jk, e) = TYPES[t];
        let (a0, ys) = &self.y[n][kind_index(yk)];
        let js = &self.j[k][kind_index(jk)];
        let xe = if e == 2 { x_e[0] } else { x_e[1] };
        // For each l: the largest i with a₀ + 2l + b₀ + 2i + e ≤ cut.
        let calc = |cut: i64| -> (Complex<T>, T, Complex<T>, T) {
            let (mut neg, mut neg_abs) = (real(T::zero()), T::zero());
            let (mut pos, mut pos_abs) = (real(T::zero()), T::zero());
            let len = js.head.len() as i64;
            for (l, &yl) in ys.iter().enumerate() {
                let base = a0 + 2 * l as i64 + js.b0 + e;
                // p(i) = base + 2i ≤ cut ⇔ i ≤ (cut − base)/2.
                let i_max = if cut >= base { (cut - base) / 2 } else { -1 };
                let yv = yl * xe;
                if i_max >= 0 {
                    let i = i_max.min(len - 1) as usize;
                    neg = neg + js.head[i].scale(yv);
                    neg_abs += js.head_abs[i] * yv.abs();
                }
                let from = (i_max + 1).clamp(0, len) as usize;
                pos = pos + js.tail[from].scale(yv);
                pos_abs += js.tail_abs[from] * yv.abs();
            }
            (neg, neg_abs, pos, pos_abs)
        };
        let pick = |cut: i64| -> (Complex<T>, T) {
            let (neg, neg_abs, pos, pos_abs) = calc(cut);
            // R⁺ by its series, or as R − R_{≤cut}: whichever adds less.
            let sub_abs = full.norm() + neg_abs;
            if pos_abs <= sub_abs {
                (pos, pos_abs)
            } else {
                (full - neg, sub_abs)
            }
        };
        let (le0, le0_abs) = pick(0);
        let (lt0, lt0_abs) = pick(-1);
        Forms {
            le0,
            le0_abs,
            lt0,
            lt0_abs,
        }
    }
}

fn kind_index(k: Kind) -> usize {
    match k {
        Kind::Plain => 0,
        Kind::Deriv => 1,
        Kind::Over => 2,
    }
}

/// The eight angular factors of element (row type rt, order n; column type ct, order k) at a
/// node, for the products of `TYPES` (with s on the first column field, and ν' = −β sin θ cos θ on
/// the ν terms, whose x² is in e): rows (A₁, A₂) = (Mₙ, Nₙ) | (Nₙ, Mₙ), columns (B₁, B₂) =
/// (s N_k, M_k) | (s M_k, N_k), and n̂·(A × B) with the y-part components
/// Mₙ = (0, i yₙ pₙ, −yₙ τₙ), Nₙ = (n(n+1) (yₙ/x) dₙ, Dyₙ τₙ, i Dyₙ pₙ), pₙ = −πₙ.
#[allow(clippy::too_many_arguments)]
pub(crate) fn angular_factors<T: Real>(
    rt: usize,
    ct: usize,
    s: Complex<T>,
    nu: T,
    n: usize,
    k: usize,
    (dn, pin, tn): (T, T, T),
    (dk, pik, tk): (T, T, T),
) -> [Complex<T>; 8] {
    let i1 = Complex::new(T::zero(), T::one());
    let pn = -pin;
    let nn1 = int::<T>(n * (n + 1));
    let kk1 = int::<T>(k * (k + 1));
    let zero = real(T::zero());
    // n̂·(A × B) of each (A, B) as coefficients of the TYPES.
    let mm = [
        i1.scale(tn * pik - pn * tk),
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    ];
    let mn = [
        zero,
        real(tn * tk - pn * pik),
        zero,
        zero,
        real(-(nu * tn * kk1 * dk)),
        zero,
        zero,
        zero,
    ];
    let nm = [
        zero,
        zero,
        real(pn * pik - tn * tk),
        zero,
        zero,
        zero,
        real(nu * nn1 * dn * tk),
        zero,
    ];
    let nnv = [
        zero,
        zero,
        zero,
        i1.scale(tn * pik - pn * tk),
        zero,
        i1.scale(nu * pn * kk1 * dk),
        zero,
        i1.scale(-(nu * nn1 * dn * pik)),
    ];
    let combine = |a: [Complex<T>; 8], b: [Complex<T>; 8]| -> [Complex<T>; 8] {
        core::array::from_fn(|t| s * a[t] + b[t])
    };
    match (rt, ct) {
        (0, 0) => combine(mn, nm),
        (0, _) => combine(mm, nnv),
        (_, 0) => combine(nnv, mm),
        _ => combine(nm, mn),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::DoubleDouble as Dd;
    use crate::tmatrix::functions::Radial;

    /// The whole series reproduce the scaled Bessel factors on [x_min, x_max] to the rounding of
    /// their largest term, in double-double (so none is truncated).
    #[test]
    fn series_reproduce_the_functions() {
        let x_min = Dd::from(9.0);
        let x_max = Dd::from(36.0);
        let s = Complex::new(Dd::from(1.5), Dd::from(0.01));
        let n = 40;
        let scales = Scales::new(x_min, s.scale(x_max), n);
        let series = Series::new(x_min, x_max, s, n, &scales);
        let mut rad = Radial::new();
        let tol = <Dd as num_traits::Float>::epsilon() * Dd::from(64.0);
        for xi in [1.0, 1.7, 2.5, 4.0] {
            rad.compute(x_min * Dd::from(xi), s, n, &scales);
            let ns = series.at(Dd::from(xi));
            for nn in [1usize, 10, 40] {
                for (kind, v) in [(0, rad.h[nn].im), (1, rad.dh[nn].im), (2, rad.h_x[nn].im)] {
                    let terms = &ns.y[nn][kind].1;
                    let sum = terms.iter().fold(Dd::from(0.0), |a, &b| a + b);
                    let mag = terms.iter().fold(Dd::from(0.0), |a, &b| a + b.abs());
                    assert!((sum - v).abs() <= tol * mag, "y n {nn} kind {kind} ξ {xi}");
                }
            }
            for kk in [1usize, 2, 40] {
                for (kind, v) in [(0, rad.jz[kk]), (1, rad.djz[kk]), (2, rad.jz_z[kk])] {
                    let js = &ns.j[kk][kind];
                    assert!(
                        (js.tail[0] - v).norm() <= tol * js.tail_abs[0],
                        "j k {kk} kind {kind} ξ {xi}"
                    );
                }
            }
        }
    }
}
