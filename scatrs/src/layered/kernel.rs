//! The layered-sphere recursion: the effective log-derivatives Hₐ and H_b at the outer surface of
//! each layer, from the core outward (Yang 2003; Peña & Pal 2009), in the pole-free form of the
//! theory chapter, then the homogeneous outer step of [`crate::mie`] with them.
//!
//! Notation (layers l = 0, …, L − 1 from the core; z₁ = m_l x_{l−1} and z₂ = m_l x_l for
//! l ≥ 1): ζ is a second solution of the Riccati–Bessel equation besides ψ, with Wronskian
//! W = ψζ′ − ψ′ζ; u_n = ψ_n/ψ_{n−1}, D_n = (n + 1)/z − u_{n+1}, w_n = ζ_n/ζ_{n−1},
//! D^ζ_n = (n + 1)/z − w_{n+1}, Π_n = ψ_n ζ_n, P_n = [ψ_n/ζ_n](z₁)·ζ_n(z₂)², and for layer l
//! G₁ = m_l H − m_{l−1} D_n(z₁), G₂ = m_l H − m_{l−1} D^ζ_n(z₁) for Hₐ (m_l and m_{l−1}
//! exchanged for H_b), with
//!
//! H = D^ζ_n(z₂) − W G₂/(Π_n(z₂) G₂ − P_n G₁) = D_n(z₂) + δ,
//! δ = −W P_n G₁/[Π_n(z₂)(Π_n(z₂) G₂ − P_n G₁)].
//!
//! For ζ = ξ (W = i) these are Yang's formulas; ξ keeps the decay of the inner layers'
//! contribution through an absorbing layer explicit (|P| ≤ |Π| e^(−2 Im(z₂ − z₁))). For an
//! optically thin layer (Im m_l·x_l < ½) ζ = χ (W = −1) instead: χ is real for real z, as ψ is,
//! so every quantity of the layer stays real for a real index, and a weak absorption appears
//! as imaginary parts of relative accuracy, where ξ's complex values would bury it in their
//! rounding (the error of Q_abs grew like 2⁻⁵²/Im m).

use num_complex::Complex;

use crate::complex::{exp_m1, inv};
use crate::mie::kernel::{Outer, Side, State, int, ratios, real};
use crate::scalar::Real;

/// Π_n(z) and w_{n+1}(z) of one argument, advanced order by order by Π_n = Π_{n−1} u_n w_n
/// and w_{n+1} = (2n + 1)/z − 1/w_n (the upward recurrence of ζ, stable for Im z ≥ 0), from
/// Π₀ = (1 − e^(2iz))/2 and w₁ = 1/z − i for ζ = ξ, or Π₀ = sin z cos z and w₁ = 1/z + tan z
/// for ζ = χ (χ₀ = cos z, χ₋₁ = −sin z).
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
    fn new(z: Complex<T>, chi: bool) -> Self {
        let inv_z = inv(z);
        let (pi, w) = if chi {
            let (s, c) = (z.sin(), z.cos());
            (s * c, inv_z + s * inv(c))
        } else {
            (
                -exp_m1(two_i(z)).scale(T::from_f64(0.5)),
                inv_z - Complex::new(T::zero(), T::one()),
            )
        };
        Self {
            inv_z,
            pi,
            w,
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

    /// D^ζ_n(z), after `step(n)`.
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
    /// u_{n+1} of the inner layer's outer argument m_{l−1}x_{l−1}, and of z₁ = m_l x_{l−1}.
    u_prev: Complex<T>,
    u1: Complex<T>,
    d1i: Complex<T>,
    d3i: Complex<T>,
    d1o: Complex<T>,
    d3o: Complex<T>,
    pio: Complex<T>,
    p: Complex<T>,
    /// The Wronskian W of ψ and the layer's ζ.
    w: Complex<T>,
}

/// H at the outer surface of a layer from H at its inner surface (`h`, as `prev` represents it),
/// with `mi` the layer's index and `mo` the inner one's (exchanged for H_b), and `lead` the
/// (n + 1)/x_{l−1} terms of G₁ (see below): as D⁽¹⁾_n(z₂) + δ unless the two cancel (by more than
/// a factor 4), near the poles of D⁽¹⁾_n(z₂), where it is H in the pole-free form.
///
/// G₁ = mi H − mo D⁽¹⁾_n(z₁) cancels at small x_{l−1} (both terms are about (n + 1)/x_{l−1}, as in
/// the sphere's numerators). With H = D⁽¹⁾_n(m_{l−1}x_{l−1}) + δ and D⁽¹⁾ = (n + 1)/z − u_{n+1},
/// G₁ = (n + 1)/x_{l−1}·(mi/m_{l−1} − mo/m_l) − mi u_{n+1}(m_{l−1}x_{l−1}) + mo u_{n+1}(z₁) + mi δ,
/// whose first term is (n + 1)/x_{l−1}·(m_l² − m_{l−1}²)/(m_l m_{l−1}) for Hₐ and 0 for H_b: the
/// cancelling terms are removed exactly.
fn side<T: Real>(
    mi: Complex<T>,
    mo: Complex<T>,
    prev: Side<T>,
    h: Complex<T>,
    lead: Complex<T>,
    s: &Step<T>,
) -> (Side<T>, Complex<T>) {
    let mh = mi * h;
    let g1 = match prev {
        Side::Delta(delta) => lead - mi * s.u_prev + mo * s.u1 + mi * delta,
        Side::Direct(_) => mh - mo * s.d1i,
    };
    let g2 = mh - mo * s.d3i;
    let inv_den = inv(s.pio * g2 - s.p * g1);
    let delta = -(s.w * s.p * g1 * inv_den * inv(s.pio));
    let hd = s.d1o + delta;
    if norm1(hd) * 4.0 >= norm1(s.d1o).max(norm1(delta)) {
        (Side::Delta(delta), hd)
    } else {
        let hp = s.d3o - s.w * g2 * inv_den;
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
    /// (m_l² − m_{l−1}²)/(m_l m_{l−1}) for each layer l ≥ 1.
    lead: Vec<Complex<T>>,
    /// Whether layer l ≥ 1 takes ζ = χ (else ξ).
    chi: Vec<bool>,
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
            lead: Vec::new(),
            chi: Vec::new(),
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
        self.lead.clear();
        self.chi.clear();
        for l in 1..layers {
            let chi = m[l].im.approx_f64() * x[l].approx_f64() < 0.5;
            self.chi.push(chi);
            // (m_l − m_{l−1})(m_l + m_{l−1}): exactly 0 for equal indices.
            let (a, b) = (m[l], m[l - 1]);
            self.lead.push((a - b) * (a + b) * inv(a * b));
            let (z1, z2) = (m[l].scale(x[l - 1]), m[l].scale(x[l]));
            self.inner.push(Arg::new(z1, chi));
            self.outer_args.push(Arg::new(z2, chi));
            let p0 = if chi {
                // P₀ = tan z₁ cos² z₂.
                let c2 = z2.cos();
                z1.sin() * inv(z1.cos()) * c2 * c2
            } else {
                // P₀ = e^(2i m_l (x_l − x_{l−1}))·(1 − e^(2iz₁))/2.
                let dx = x[l] - x[l - 1];
                let (a, b) = (-(m[l].im * dx * 2.0), m[l].re * dx * 2.0);
                let (sin_b, cos_b) = b.sin_cos();
                let e = Complex::new(cos_b, sin_b).scale(a.exp());
                -(e * exp_m1(two_i(z1))).scale(T::from_f64(0.5))
            };
            self.p.push(p0);
        }
        // Only its D_n is used.
        let core = Arg::new(m[0].scale(x[0]), false);
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
            let zero = Side::Delta(real(T::zero()));
            let mut sides = (zero, zero);
            for l in 1..layers {
                let (ai, ao) = (&self.inner[l - 1], &self.outer_args[l - 1]);
                let u_prev = if l == 1 {
                    &self.u[0]
                } else {
                    &self.u[2 * (l - 1)]
                };
                let s = Step {
                    u_prev: u_prev[k + 1],
                    u1: self.u[2 * l - 1][k + 1],
                    d1i: ai.d1(k, self.u[2 * l - 1][k + 1]),
                    d3i: ai.d3(k),
                    d1o: ao.d1(k, u_of(l)[k + 1]),
                    d3o: ao.d3(k),
                    pio: ao.pi,
                    p: self.p[l - 1],
                    w: if self.chi[l - 1] {
                        real(-T::one())
                    } else {
                        Complex::new(T::zero(), T::one())
                    },
                };
                let lead = self.lead[l - 1].scale(int::<T>(k + 1) / x[l - 1]);
                let (sa, a) = side(m[l], m[l - 1], sides.0, ha, lead, &s);
                let (sb, b) = side(m[l - 1], m[l], sides.1, hb, real(T::zero()), &s);
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
