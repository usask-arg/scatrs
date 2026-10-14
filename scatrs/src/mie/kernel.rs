//! The Mie recurrences: coefficients, their derivatives and the absorption sums.
//!
//! See the module documentation of [`crate::mie`] for the method. Notation: u_k(ρ) =
//! ψ_k(ρ)/ψ_{k−1}(ρ), D_n(ρ) = (n + 1)/ρ − u_{n+1}(ρ), R_n = ξ_{n−1}/ξ_n, T_n = ψ_n/ξ_n, and
//! Δ_a = A ξ_n − ξ_{n−1}, Δ_b = B ξ_n − ξ_{n−1} with A = D_n(mx)/m + n/x and
//! B = m D_n(mx) + n/x, so that a_n = (A ψ_n − ψ_{n−1})/Δ_a and b_n = (B ψ_n − ψ_{n−1})/Δ_b.
//!
//! A layered sphere has the same outer step with D_n(mx) replaced by the effective
//! log-derivatives Hₐ and H_b at its outer surface (`Outer`), which `crate::layered` computes.

use num_complex::Complex;

use crate::complex::inv;
use crate::scalar::Real;

/// One coefficient's effective log-derivative H at the outer surface of a layered sphere, at
/// one order: as the correction δ = H − D_n(mx) to the outer layer's own log-derivative, which
/// keeps the ratio region's numerators free of cancellation, or as H itself, near the poles of
/// D_n(mx) where D_n(mx) and δ would cancel.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Side<T> {
    Delta(Complex<T>),
    Direct(Complex<T>),
}

/// Hₐ and H_b at one order.
pub(crate) type Outer<T> = (Side<T>, Side<T>);

#[derive(Clone, Debug)]
pub(crate) struct State<T: Real> {
    pub(crate) x: T,
    pub(crate) m: Complex<T>,
    pub(crate) a: Vec<Complex<T>>,
    pub(crate) b: Vec<Complex<T>>,
    pub(super) derivatives: bool,
    pub(crate) da_dx: Vec<Complex<T>>,
    pub(crate) da_dm: Vec<Complex<T>>,
    pub(crate) db_dx: Vec<Complex<T>>,
    pub(crate) db_dm: Vec<Complex<T>>,
    /// (x²/2) Q_abs, and its derivatives with respect to x, Re m and Im m (of the sum alone,
    /// without the factor 2/x²).
    pub(crate) abs_sum: T,
    pub(crate) abs_sum_jacobian: [T; 3],
    /// u_k(mx) for k = 1, …, N + 1 (index k).
    u_z: Vec<Complex<T>>,
    /// u_k(x) for k = n_s + 1, …, N + 1 (index k).
    u_x: Vec<T>,
    /// u_k(mx) for real m, before conversion into `u_z`.
    u_real: Vec<T>,
}

pub(crate) fn real<T: Real>(v: T) -> Complex<T> {
    Complex::new(v, T::zero())
}

pub(crate) fn int<T: Real>(k: usize) -> T {
    // Exact: the orders are below 2^53.
    T::from_f64(k as f64)
}

/// Steps from the start of a downward ratio recurrence to the first stored value: enough for
/// the start value to be forgotten to rounding level (each step shrinks it at least sixfold).
fn margin<T: Real>() -> usize {
    let bits = -T::epsilon().approx_f64().log2();
    (bits / 2.0).ceil() as usize + 4
}

/// The first index of a downward ratio recurrence that must reach `n_top`: above both `n_top`
/// and 1.5|ρ| by the margin.
fn start_index<T: Real>(n_top: usize, rho_abs: f64) -> usize {
    let above = (1.5 * rho_abs).ceil();
    let above = if above < n_top as f64 {
        n_top
    } else {
        above as usize
    };
    above + margin::<T>()
}

/// u_k(z) for k = 1, …, n_top + 1 into `out` (index k), complex z.
pub(crate) fn ratios_complex<T: Real>(out: &mut Vec<Complex<T>>, z: Complex<T>, n_top: usize) {
    let inv_z = inv(z);
    let start = start_index::<T>(n_top + 1, z.norm().approx_f64());
    out.clear();
    out.resize(n_top + 2, real(T::zero()));
    let mut u = real(T::zero());
    for k in (1..=start).rev() {
        let c = inv_z.scale(int(2 * k + 1));
        let mut den = c - u;
        if den.re == T::zero() && den.im == T::zero() {
            // Only at an exact zero of ψ_{k−1}: nudge by a rounding error, as Lentz's
            // method does, so that the pole of D_{k−1} stays finite.
            den = real(T::epsilon() * (c.norm() + u.norm()));
        }
        u = inv(den);
        if k <= n_top + 1 {
            out[k] = u;
        }
    }
}

/// u_k(ρ) for k = from, …, n_top + 1 (real ρ), returned at index k of a vector.
pub(crate) fn ratios_real<T: Real>(out: &mut Vec<T>, rho: T, from: usize, n_top: usize) {
    let inv_rho = rho.recip();
    let start = start_index::<T>(n_top + 1, rho.approx_f64().abs());
    out.clear();
    out.resize(n_top + 2, T::zero());
    let mut u = T::zero();
    for k in (from.max(1)..=start).rev() {
        let c = inv_rho * int::<T>(2 * k + 1);
        let mut den = c - u;
        if den == T::zero() {
            den = T::epsilon() * (c.abs() + u.abs());
        }
        u = den.recip();
        if k <= n_top + 1 {
            out[k] = u;
        }
    }
}

/// u_k(z) for z = m·x and k = 1, …, n_top + 1 into `out` (index k): by the real recurrence
/// (through `scratch`) when Im m is zero (with no derivative seeded in it), as for D_n(mx).
pub(crate) fn ratios<T: Real>(
    out: &mut Vec<Complex<T>>,
    scratch: &mut Vec<T>,
    m: Complex<T>,
    x: T,
    n_top: usize,
) {
    let z = m.scale(x);
    if m.im.is_zero() {
        ratios_real(scratch, z.re, 1, n_top);
        out.clear();
        out.extend(scratch.iter().map(|&u| real(u)));
    } else {
        ratios_complex(out, z, n_top);
    }
}

impl<T: Real> State<T> {
    pub(crate) fn new() -> Self {
        Self {
            x: T::zero(),
            m: real(T::zero()),
            a: Vec::new(),
            b: Vec::new(),
            derivatives: false,
            da_dx: Vec::new(),
            da_dm: Vec::new(),
            db_dx: Vec::new(),
            db_dm: Vec::new(),
            abs_sum: T::zero(),
            abs_sum_jacobian: [T::zero(); 3],
            u_z: Vec::new(),
            u_x: Vec::new(),
            u_real: Vec::new(),
        }
    }

    /// Computes a_n, b_n for n = 1, …, `n` (and their derivatives if asked), and the
    /// absorption sum.
    pub(crate) fn solve(&mut self, x: T, m: Complex<T>, n: usize, derivatives: bool) {
        self.prepare(x, m, n);
        self.finish(x, m, n, derivatives, None);
    }

    /// The first step of [`solve`](Self::solve): u_k(mx) for k = 1, …, `n` + 1, which
    /// [`u_z`](Self::u_z) then returns.
    pub(crate) fn prepare(&mut self, x: T, m: Complex<T>, n: usize) {
        // D_n(mx). For real m (with no derivative seeded in Im m) the real recurrence is used:
        // twice as fast, and for m = 1 exactly it reproduces u_k(x) bit for bit.
        ratios(&mut self.u_z, &mut self.u_real, m, x, n);
    }

    /// u_k(mx) for k = 1, …, N + 1 (index k), after [`prepare`](Self::prepare).
    pub(crate) fn u_z(&self) -> &[Complex<T>] {
        &self.u_z
    }

    /// The rest of [`solve`](Self::solve), after [`prepare`](Self::prepare) with the same
    /// x, m and n. With `outer`, the effective log-derivatives of a layered sphere at each
    /// order replace D_n(mx); derivatives are then not available.
    pub(crate) fn finish(
        &mut self,
        x: T,
        m: Complex<T>,
        n: usize,
        derivatives: bool,
        outer: Option<&[Outer<T>]>,
    ) {
        debug_assert!(!(derivatives && outer.is_some()));
        let one = T::one();
        let two = T::from_f64(2.0);
        let i1 = Complex::new(T::zero(), one);
        self.x = x;
        self.m = m;
        self.derivatives = derivatives;
        let z = m.scale(x);
        let xf = x.approx_f64();
        // Orders up to n_s use the upward ψ and χ recurrences; above, the ratios.
        let n_s = if xf >= 1.0 {
            (xf.floor() as usize).min(n)
        } else {
            0
        };
        if n_s < n {
            ratios_real(&mut self.u_x, x, n_s + 1, n);
        }
        for v in [
            &mut self.a,
            &mut self.b,
            &mut self.da_dx,
            &mut self.da_dm,
            &mut self.db_dx,
            &mut self.db_dm,
        ] {
            v.clear();
        }
        let inv_x = x.recip();
        let inv_z = inv(z);
        let inv_m = inv(m);
        let m2 = m * m;
        let inv_m2 = inv_m * inv_m;
        // (m − 1)(m + 1): m − 1 is exact for m near 1, unlike m² − 1.
        let m2_minus_1 = (m - real(one)) * (m + real(one));
        let c_a = -(m2_minus_1 * inv_m2); // 1/m² − 1
        let (sin_x, cos_x) = x.sin_cos();
        let (mut psi_prev, mut psi) = (cos_x, sin_x); // ψ₋₁, ψ₀
        let (mut chi_prev, mut chi) = (-sin_x, cos_x); // χ₋₁, χ₀
        // Ratio-region state: T_n, 1/ξ_n and R_n, starting at n = 0 when n_s = 0.
        let mut t = real(T::zero());
        let mut inv_xi = real(T::zero());
        let mut r = real(T::zero());
        if n_s == 0 {
            let xi0 = Complex::new(sin_x, -cos_x);
            t = real(sin_x) * inv(xi0);
            inv_xi = Complex::new(sin_x, cos_x); // 1/ξ₀ = i e^(−ix)
            r = i1; // ξ₋₁/ξ₀
        }
        let mut abs_sum = T::zero();
        let mut jac = [T::zero(); 3];
        for k in 1..=n {
            let kf: T = int(k);
            let k_x = kf * inv_x;
            let d = inv_z.scale(int(k + 1)) - self.u_z[k + 1];
            let side = outer.map(|o| o[k - 1]);
            let h = |s: Option<Side<T>>| match s {
                None => d,
                Some(Side::Delta(delta)) => d + delta,
                Some(Side::Direct(h)) => h,
            };
            let d_m = h(side.map(|s| s.0)) * inv_m;
            let md = h(side.map(|s| s.1)) * m;
            let big_a = d_m + real(k_x);
            let big_b = md + real(k_x);
            // a_n, b_n, 1/Δ_a, 1/Δ_b, ξ_n/Δ and ξ_{n−1}/Δ.
            let (an, bn, inv_da, inv_db, g_a, g_b, h_a, h_b);
            if k <= n_s {
                let c: T = int::<T>(2 * k - 1) * inv_x;
                let psi_next = c * psi - psi_prev;
                psi_prev = psi;
                psi = psi_next;
                let chi_next = c * chi - chi_prev;
                chi_prev = chi;
                chi = chi_next;
                let xi = Complex::new(psi, -chi);
                let xi_prev = Complex::new(psi_prev, -chi_prev);
                inv_da = inv(big_a * xi - xi_prev);
                inv_db = inv(big_b * xi - xi_prev);
                match side {
                    None => {
                        an = (big_a.scale(psi) - real(psi_prev)) * inv_da;
                        bn = (big_b.scale(psi) - real(psi_prev)) * inv_db;
                    }
                    Some((sa, sb)) => {
                        // A layered sphere: the outer layer's own numerator, plus δ/m·ψ_n and
                        // m δ·ψ_n, so that δ (the inner layers' share, which can be far smaller
                        // than D_n(mx)) is not rounded away; the own numerator vanishes exactly
                        // for m = 1 (D_n(x) + n/x = ψ_{n−1}/ψ_n), where it would otherwise leave
                        // a rounding error of the size of ψ_{n−1}.
                        let own = |f: Complex<T>| {
                            if m == real(one) {
                                real(T::zero())
                            } else {
                                (f + real(k_x)).scale(psi) - real(psi_prev)
                            }
                        };
                        an = match sa {
                            Side::Delta(delta) => {
                                (own(d * inv_m) + (delta * inv_m).scale(psi)) * inv_da
                            }
                            Side::Direct(_) => (big_a.scale(psi) - real(psi_prev)) * inv_da,
                        };
                        bn = match sb {
                            Side::Delta(delta) => (own(d * m) + (m * delta).scale(psi)) * inv_db,
                            Side::Direct(_) => (big_b.scale(psi) - real(psi_prev)) * inv_db,
                        };
                    }
                }
                g_a = xi * inv_da;
                g_b = xi * inv_db;
                h_a = xi_prev * inv_da;
                h_b = xi_prev * inv_db;
                if k == n_s {
                    inv_xi = inv(xi);
                    t = real(psi) * inv_xi;
                    r = xi_prev * inv_xi;
                }
            } else {
                r = inv(real(int::<T>(2 * k - 1) * inv_x) - r);
                t = (t * r).scale(self.u_x[k]);
                inv_xi = inv_xi * r;
                let u_z1 = self.u_z[k + 1];
                let u_x1 = real(self.u_x[k + 1]);
                // D/m − D_n(x) and m D − D_n(x), without the cancelling (n + 1)/x terms; for a
                // layered sphere, plus δ/m and m δ, or H/m − D_n(x) and m H − D_n(x) directly.
                let mut num_a = c_a.scale(int::<T>(k + 1) * inv_x) + u_x1 - u_z1 * inv_m;
                let mut num_b = u_x1 - m * u_z1;
                if let Some((sa, sb)) = side {
                    let d_x = real(int::<T>(k + 1) * inv_x) - u_x1;
                    num_a = match sa {
                        Side::Delta(delta) => num_a + delta * inv_m,
                        Side::Direct(_) => d_m - d_x,
                    };
                    num_b = match sb {
                        Side::Delta(delta) => num_b + m * delta,
                        Side::Direct(_) => md - d_x,
                    };
                }
                let ia = inv(big_a - r);
                let ib = inv(big_b - r);
                an = t * num_a * ia;
                bn = t * num_b * ib;
                inv_da = inv_xi * ia;
                inv_db = inv_xi * ib;
                g_a = ia;
                g_b = ib;
                h_a = r * ia;
                h_b = r * ib;
            }
            self.a.push(an);
            self.b.push(bn);
            let na = inv_da.norm_sqr();
            let nb = inv_db.norm_sqr();
            let weight: T = int(2 * k + 1);
            // Re a_n − |a_n|² and Re b_n − |b_n|².
            abs_sum += weight * (-(d_m.im * na) - md.im * nb);
            if derivatives {
                // D′ = n(n + 1)/z² − 1 − D², D + zD′ and D − zD′ in terms of u = u_{n+1}(z),
                // with D = (n + 1)/z − u: the terms in 1/z² and 1/z that cancel at small z
                // (D + zD′ = O(z)) are removed exactly.
                let u = self.u_z[k + 1];
                let one_u2 = real(one) + u * u;
                let dp = (u.scale(two) * inv_z - inv_z * inv_z).scale(int(k + 1)) - one_u2;
                let d_plus = u.scale(weight) - z * one_u2; // D + zD′
                let d_minus = d.scale(two) - d_plus; // D − zD′
                // As a product of two `T`s: k(k + 1) overflows a 32-bit usize for k ≥ 65536.
                let kk1: T = int::<T>(k) * int::<T>(k + 1);
                let ida2 = inv_da * inv_da;
                let idb2 = inv_db * inv_db;
                self.da_dm.push(i1 * d_minus * inv_m2 * ida2);
                self.db_dm.push(-(i1 * d_plus * idb2));
                self.da_dx
                    .push(i1 * m2_minus_1 * (d * d + real(kk1 * inv_x * inv_x)) * inv_m2 * ida2);
                self.db_dx.push(i1 * m2_minus_1 * idb2);
                // Derivatives of −Im(f)/|Δ|² with f = D/m or m D: for a parameter p,
                // −[Im(∂f/∂p) − 2 Im(f) Re(∂Δ/∂p / Δ)]/|Δ|².
                let fa_m = -(d_minus * inv_m2); // d(D/m)/dm = dA/dm
                let fb_m = d_plus; // d(mD)/dm = dB/dm
                let k_x2 = k_x * inv_x;
                let la_x = g_a * (dp - real(k_x2) - big_a.scale(k_x) + real(one)) + h_a * d_m;
                let lb_x = g_b * (m2 * dp - real(k_x2) - big_b.scale(k_x) + real(one)) + h_b * md;
                let la_m = fa_m * g_a;
                let lb_m = fb_m * g_b;
                let term = |f: Complex<T>, df: Complex<T>, l: Complex<T>, n2: T| {
                    -((df.im - two * f.im * l.re) * n2)
                };
                jac[0] += weight * (term(d_m, dp, la_x, na) + term(md, m2 * dp, lb_x, nb));
                jac[1] += weight * (term(d_m, fa_m, la_m, na) + term(md, fb_m, lb_m, nb));
                jac[2] += weight
                    * (term(d_m, i1 * fa_m, i1 * la_m, na) + term(md, i1 * fb_m, i1 * lb_m, nb));
            }
        }
        self.abs_sum = abs_sum;
        self.abs_sum_jacobian = jac;
    }
}
