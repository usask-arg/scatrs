//! The layered-sphere recursion: the effective log-derivatives Hₐ and H_b at the outer surface of
//! each layer, from the core outward (Yang 2003; Peña & Pal 2009), in the pole-free form of the
//! theory chapter, then the homogeneous outer step of [`crate::mie`] with them.
//!
//! Notation (layers l = 0, …, L − 1 from the core; z₁ = m_l x_{l−1} and z₂ = m_l x_l for
//! l ≥ 1): u_n = ψ_n/ψ_{n−1}, D⁽¹⁾_n = (n + 1)/z − u_{n+1}, w_n = ξ_n/ξ_{n−1},
//! D⁽³⁾_n = (n + 1)/z − w_{n+1}, Π_n = ψ_n ξ_n, P_n = [ψ_n/ξ_n](z₁)·ξ_n(z₂)², and for layer l
//! G₁ = m_l H − m_{l−1} D⁽¹⁾_n(z₁), G₂ = m_l H − m_{l−1} D⁽³⁾_n(z₁) for Hₐ (m_l and m_{l−1}
//! exchanged for H_b), with
//!
//! H = D⁽³⁾_n(z₂) − i G₂/(Π_n(z₂) G₂ − P_n G₁) = D⁽¹⁾_n(z₂) + δ,
//! δ = −i P_n G₁/[Π_n(z₂)(Π_n(z₂) G₂ − P_n G₁)].

use num_complex::Complex;

use crate::complex::{exp_m1, inv};
use crate::mie::kernel::{Outer, Side, State, int, ratios, real};
use crate::scalar::Real;

/// Π_n(z) and w_{n+1}(z) of one argument, advanced order by order from Π₀ = (1 − e^(2iz))/2
/// and w₁ = 1/z − i by Π_n = Π_{n−1} u_n w_n and w_{n+1} = (2n + 1)/z − 1/w_n (the upward ξ
/// recurrence, stable for Im z ≥ 0).
#[derive(Clone, Copy, Debug)]
struct Arg<T> {
    inv_z: Complex<T>,
    pi: Complex<T>,
    /// w_{n+1} after `step(n)`.
    w: Complex<T>,
    /// w_n after `step(n)`.
    w_n: Complex<T>,
}

/// 2iz.
fn two_i<T: Real>(z: Complex<T>) -> Complex<T> {
    Complex::new(-(z.im * 2.0), z.re * 2.0)
}

impl<T: Real> Arg<T> {
    fn new(z: Complex<T>) -> Self {
        let inv_z = inv(z);
        Self {
            inv_z,
            pi: -exp_m1(two_i(z)).scale(T::from_f64(0.5)),
            w: inv_z - Complex::new(T::zero(), T::one()),
            w_n: real(T::zero()),
        }
    }

    /// From order n − 1 to n, with u_n(z).
    fn step(&mut self, n: usize, u_n: Complex<T>) {
        self.w_n = self.w;
        self.pi = self.pi * u_n * self.w;
        self.w = self.inv_z.scale(int(2 * n + 1)) - inv(self.w);
    }

    /// D⁽¹⁾_n(z), with u_{n+1}(z).
    fn d1(&self, n: usize, u_next: Complex<T>) -> Complex<T> {
        self.inv_z.scale(int(n + 1)) - u_next
    }

    /// D⁽³⁾_n(z), after `step(n)`.
    fn d3(&self, n: usize) -> Complex<T> {
        self.inv_z.scale(int(n + 1)) - self.w
    }
}

/// |re| + |im|.
fn norm1<T: Real>(z: Complex<T>) -> T {
    z.re.abs() + z.im.abs()
}

/// The quantities of one layer's outer step at one order.
struct Step<T> {
    d1i: Complex<T>,
    d3i: Complex<T>,
    d1o: Complex<T>,
    d3o: Complex<T>,
    pio: Complex<T>,
    p: Complex<T>,
}

/// H at the outer surface of a layer from H at its inner surface (`h`), with `mi` the layer's
/// index and `mo` the inner one's (exchanged for H_b): as D⁽¹⁾_n(z₂) + δ unless the two cancel
/// (by more than a factor 4), near the poles of D⁽¹⁾_n(z₂), where it is H in the pole-free form.
fn side<T: Real>(
    mi: Complex<T>,
    mo: Complex<T>,
    h: Complex<T>,
    s: &Step<T>,
) -> (Side<T>, Complex<T>) {
    let i1 = Complex::new(T::zero(), T::one());
    let mh = mi * h;
    let g1 = mh - mo * s.d1i;
    let g2 = mh - mo * s.d3i;
    let inv_den = inv(s.pio * g2 - s.p * g1);
    let delta = -(i1 * s.p * g1 * inv_den * inv(s.pio));
    let hd = s.d1o + delta;
    if norm1(hd) * 4.0 >= norm1(s.d1o).max(norm1(delta)) {
        (Side::Delta(delta), hd)
    } else {
        let hp = s.d3o - i1 * g2 * inv_den;
        (Side::Direct(hp), hp)
    }
}

/// A workspace for the recursion and the outer step.
#[derive(Clone, Debug)]
pub(super) struct Kernel<T: Real> {
    pub(super) mie: State<T>,
    /// u_k of the core's argument m₀x₀, then for each layer l ≥ 1 of m_l x_{l−1} and, but for
    /// the outermost (whose ratios `mie` holds), of m_l x_l.
    u: Vec<Vec<Complex<T>>>,
    scratch: Vec<T>,
    inner: Vec<Arg<T>>,
    outer_args: Vec<Arg<T>>,
    p: Vec<Complex<T>>,
    outer: Vec<Outer<T>>,
}

impl<T: Real> Kernel<T> {
    pub(super) fn new() -> Self {
        Self {
            mie: State::new(),
            u: Vec::new(),
            scratch: Vec::new(),
            inner: Vec::new(),
            outer_args: Vec::new(),
            p: Vec::new(),
            outer: Vec::new(),
        }
    }

    /// a_n, b_n for n = 1, …, `n` and the absorption sum of the layered sphere `x`, `m`
    /// (validated, L ≥ 2) into `self.mie`.
    pub(super) fn solve(&mut self, x: &[T], m: &[Complex<T>], n: usize) {
        let layers = x.len();
        debug_assert!(layers >= 2 && m.len() == layers);
        let last = layers - 1;
        self.mie.prepare(x[last], m[last], n);
        self.u.resize_with(2 * last, Vec::new);
        ratios(&mut self.u[0], &mut self.scratch, m[0], x[0], n);
        for l in 1..layers {
            ratios(&mut self.u[2 * l - 1], &mut self.scratch, m[l], x[l - 1], n);
            if l < last {
                ratios(&mut self.u[2 * l], &mut self.scratch, m[l], x[l], n);
            }
        }
        self.inner.clear();
        self.outer_args.clear();
        self.p.clear();
        for l in 1..layers {
            let z1 = m[l].scale(x[l - 1]);
            self.inner.push(Arg::new(z1));
            self.outer_args.push(Arg::new(m[l].scale(x[l])));
            // P₀ = e^(2i m_l (x_l − x_{l−1}))·(1 − e^(2iz₁))/2.
            let dx = x[l] - x[l - 1];
            let (a, b) = (-(m[l].im * dx * 2.0), m[l].re * dx * 2.0);
            let (sin_b, cos_b) = b.sin_cos();
            let e = Complex::new(cos_b, sin_b).scale(a.exp());
            self.p
                .push(-(e * exp_m1(two_i(z1))).scale(T::from_f64(0.5)));
        }
        let core = Arg::new(m[0].scale(x[0]));
        self.outer.clear();
        let u_z = self.mie.u_z();
        for k in 1..=n {
            let u_of = |l: usize| -> &[Complex<T>] { if l == last { u_z } else { &self.u[2 * l] } };
            for l in 1..layers {
                let (u1, u2) = (&self.u[2 * l - 1], u_of(l));
                let (ai, ao) = (&mut self.inner[l - 1], &mut self.outer_args[l - 1]);
                ai.step(k, u1[k]);
                ao.step(k, u2[k]);
                self.p[l - 1] = self.p[l - 1] * u1[k] * inv(ai.w_n) * ao.w_n * ao.w_n;
            }
            let mut ha = core.d1(k, self.u[0][k + 1]);
            let mut hb = ha;
            let mut sides = (Side::Delta(real(T::zero())), Side::Delta(real(T::zero())));
            for l in 1..layers {
                let (ai, ao) = (&self.inner[l - 1], &self.outer_args[l - 1]);
                let s = Step {
                    d1i: ai.d1(k, self.u[2 * l - 1][k + 1]),
                    d3i: ai.d3(k),
                    d1o: ao.d1(k, u_of(l)[k + 1]),
                    d3o: ao.d3(k),
                    pio: ao.pi,
                    p: self.p[l - 1],
                };
                let (sa, a) = side(m[l], m[l - 1], ha, &s);
                let (sb, b) = side(m[l - 1], m[l], hb, &s);
                (ha, hb, sides) = (a, b, (sa, sb));
            }
            self.outer.push(sides);
        }
        self.mie
            .finish(x[last], m[last], n, false, Some(&self.outer));
        // A lossless particle absorbs nothing; H is real then in exact arithmetic, but its
        // complex intermediates leave a rounding-level imaginary part.
        if m.iter().all(|m| m.im.is_zero()) {
            self.mie.abs_sum = T::zero();
        }
    }
}
