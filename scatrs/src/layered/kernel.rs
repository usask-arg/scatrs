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
use crate::mie::kernel::{Outer, OuterDerivs, Side, State, int, ratios, real};
use crate::scalar::Real;

/// Π_n(z) and w_{n+1}(z) of one argument, advanced order by order by Π_n = Π_{n−1} u_n w_n
/// and w_{n+1} = (2n + 1)/z − 1/w_n (the upward recurrence of ζ, stable for Im z ≥ 0), from
/// Π₀ = (1 − e^(2iz))/2 and w₁ = 1/z − i for ζ = ξ, or Π₀ = sin z cos z and w₁ = 1/z + tan z
/// for ζ = χ (χ₀ = cos z, χ₋₁ = −sin z).
#[derive(Clone, Copy, Debug)]
struct Arg<T> {
    z: Complex<T>,
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
            z,
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

    /// D′_n(z) = n(n + 1)/z² − 1 − D_n², with u = u_{n+1}(z), in the sphere's form without the
    /// terms in 1/z² that cancel: (n + 1)(2u/z − 1/z²) − (1 + u²).
    fn d1_prime(&self, n: usize, u: Complex<T>) -> Complex<T> {
        let iz = self.inv_z;
        (u.scale(T::from_f64(2.0)) * iz - iz * iz).scale(int(n + 1)) - (real(T::one()) + u * u)
    }

    /// D_n + zD′_n, with u = u_{n+1}(z): (2n + 1)u − z(1 + u²), which is O(z) at small z.
    fn d1_plus(&self, n: usize, u: Complex<T>) -> Complex<T> {
        u.scale(int(2 * n + 1)) - self.z * (real(T::one()) + u * u)
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
    /// Whether to compute the derivatives' sensitivities, and for them: the order, u_{n+1}(z₁),
    /// u_{n+1}(z₂), S = ζ_n(z₂)/ζ_n(z₁), and the arguments.
    derivatives: bool,
    n: usize,
    u2: Complex<T>,
    s: Complex<T>,
    a1: Arg<T>,
    a2: Arg<T>,
}

/// The sensitivities of a layer's H at one order, for the closed-form derivatives. H is the
/// value at z₂ of the logarithmic derivative y of the layer's field, which solves the Riccati
/// equation y′ = V − y², V = n(n + 1)/z² − 1, from y(z₁) = t (the inner layer's H times
/// mi/mo). Hence ∂H/∂t = R², R = f(z₁)/f(z₂) the ratio of the field at the two surfaces
/// (= −mo W S/(Π_n(z₂)G₂ − P_n G₁)), ∂H/∂z₂ = V(z₂) − H² and ∂H/∂z₁ = −(V(z₁) − t²)R².
#[derive(Clone, Copy, Debug)]
pub(super) struct Sens<T> {
    /// H itself.
    hv: Complex<T>,
    r2: Complex<T>,
    /// m_l·∂H/∂m_l through the inner surface, over R²: σt − z₁(V(z₁) − t²), with σ = +1 for Hₐ
    /// and −1 for H_b (t ∝ m_l^σ). Its two terms are both about (n + 1)/z₁ and cancel for H_b; with
    /// t = D + e (D = D_n(z₁), e = G₁/mo) it is σD − zD′ + e(σ + z(2D + e)), whose first term is
    /// −(D + zD′) for H_b, free of the cancellation in the sphere's form.
    em: Complex<T>,
    /// H′ = V(z₂) − H².
    hp: Complex<T>,
    /// H + z₂H′.
    hplus: Complex<T>,
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
    lead_is_a: bool,
) -> (Side<T>, Complex<T>, Option<Sens<T>>) {
    let mh = mi * h;
    let g1 = match prev {
        Side::Delta(delta) => lead - mi * s.u_prev + mo * s.u1 + mi * delta,
        Side::Direct(_) => mh - mo * s.d1i,
    };
    let g2 = mh - mo * s.d3i;
    let inv_den = inv(s.pio * g2 - s.p * g1);
    let delta = -(s.w * s.p * g1 * inv_den * inv(s.pio));
    let hd = s.d1o + delta;
    let (out, h_out) = if norm1(hd) * 4.0 >= norm1(s.d1o).max(norm1(delta)) {
        (Side::Delta(delta), hd)
    } else {
        let hp = s.d3o - s.w * g2 * inv_den;
        (Side::Direct(hp), hp)
    };
    if !s.derivatives {
        return (out, h_out, None);
    }
    let two = T::from_f64(2.0);
    let inv_mo = inv(mo);
    let t = mh * inv_mo;
    let r = -(mo * s.w * s.s * inv_den);
    let e = g1 * inv_mo;
    let sigma = if lead_is_a { T::one() } else { -T::one() };
    let d1 = s.d1i;
    let z1 = s.a1.z;
    let own = if lead_is_a {
        d1.scale(two) - s.a1.d1_plus(s.n, s.u1) // D − zD′
    } else {
        -s.a1.d1_plus(s.n, s.u1) // −(D + zD′)
    };
    let em = if norm1(t) * 4.0 >= norm1(d1).max(norm1(e)) {
        own + e * (real(sigma) + z1 * (d1.scale(two) + e))
    } else {
        // Near a pole of D_n(z₁), where the terms of that form cancel instead (D ≈ −e ≫ t).
        let iz = s.a1.inv_z;
        let v = real(int::<T>(s.n) * int::<T>(s.n + 1)) * iz * iz - real(T::one());
        t.scale(sigma) - z1 * (v - t * t)
    };
    let (hp, hplus) = match out {
        // H = D + δ: H′ = D′ − δ(2D + δ) and H + zH′ = (D + zD′) + δ(1 − z(2D + δ)), each
        // free of the cancellation at small z.
        Side::Delta(delta) => {
            let d2 = s.d1o.scale(two) + delta;
            (
                s.a2.d1_prime(s.n, s.u2) - delta * d2,
                s.a2.d1_plus(s.n, s.u2) + delta * (real(T::one()) - s.a2.z * d2),
            )
        }
        // Near a pole of D_n(z₂), where nothing cancels.
        Side::Direct(hv) => {
            let iz = s.a2.inv_z;
            let v = real(int::<T>(s.n) * int::<T>(s.n + 1)) * iz * iz - real(T::one());
            let hp = v - hv * hv;
            (hp, hv + s.a2.z * hp)
        }
    };
    let sens = Sens {
        hv: h_out,
        r2: r * r,
        em,
        hp,
        hplus,
    };
    (out, h_out, Some(sens))
}

/// The derivatives of one inner layer's (or with `outer`, the last layer's) coefficients and
/// absorption sum, as `Layered` stores them.
#[derive(Clone, Debug)]
pub(super) struct LayerGrad<T> {
    pub(super) da_dx: Vec<Complex<T>>,
    pub(super) da_dm: Vec<Complex<T>>,
    pub(super) db_dx: Vec<Complex<T>>,
    pub(super) db_dm: Vec<Complex<T>>,
    pub(super) abs_sum: [T; 3],
}

impl<T: Real> LayerGrad<T> {
    fn new() -> Self {
        Self {
            da_dx: Vec::new(),
            da_dm: Vec::new(),
            db_dx: Vec::new(),
            db_dm: Vec::new(),
            abs_sum: [T::zero(); 3],
        }
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
    /// For the derivatives: S = ζ_n(z₂)/ζ_n(z₁) of each layer l ≥ 1, each layer's sensitivities
    /// at the current order (Hₐ, H_b), the outer layer's quantities per order, and dH/dx_l,
    /// dH/dm_l of every layer per order (for Hₐ then H_b).
    s: Vec<Complex<T>>,
    sens: Vec<(Sens<T>, Sens<T>)>,
    outer_derivs: Vec<OuterDerivs<T>>,
    dh: Vec<[Vec<Complex<T>>; 4]>,
    /// The derivatives of every layer but the last (whose `mie` holds).
    pub(super) grads: Vec<LayerGrad<T>>,
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
            s: Vec::new(),
            sens: Vec::new(),
            outer_derivs: Vec::new(),
            dh: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// a_n, b_n for n = 1, …, `n` and the absorption sum of the layered sphere `x`, `m`
    /// (validated, L ≥ 2) into `self.mie`; with `derivatives`, their derivatives with respect
    /// to the last layer's x and m into `self.mie` too and to every other layer's into
    /// `self.grads`, in closed form.
    pub(super) fn solve(&mut self, x: &[T], m: &[Complex<T>], n: usize, derivatives: bool) {
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
        self.s.clear();
        if derivatives {
            for l in 1..layers {
                let (z1, z2) = (m[l].scale(x[l - 1]), m[l].scale(x[l]));
                // S₀ = ζ₀(z₂)/ζ₀(z₁): cos z₂/cos z₁, or e^(i m_l (x_l − x_{l−1})).
                let s0 = if self.chi[l - 1] {
                    z2.cos() * inv(z1.cos())
                } else {
                    let dx = x[l] - x[l - 1];
                    let (sin_b, cos_b) = (m[l].re * dx).sin_cos();
                    Complex::new(cos_b, sin_b).scale((-(m[l].im * dx)).exp())
                };
                self.s.push(s0);
            }
            self.dh
                .resize_with(layers, || core::array::from_fn(|_| Vec::new()));
            self.dh.truncate(layers);
            for d in &mut self.dh {
                for v in d.iter_mut() {
                    v.clear();
                }
            }
            self.outer_derivs.clear();
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
                if derivatives {
                    self.s[l - 1] = self.s[l - 1] * ao.w_n * inv(ai.w_n);
                }
            }
            self.sens.clear();
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
                    derivatives,
                    n: k,
                    u2: u_of(l)[k + 1],
                    s: if derivatives {
                        self.s[l - 1]
                    } else {
                        real(T::zero())
                    },
                    a1: *ai,
                    a2: *ao,
                };
                let lead = self.lead[l - 1].scale(int::<T>(k + 1) / x[l - 1]);
                let (sa, a, ea) = side(m[l], m[l - 1], sides.0, ha, lead, &s, true);
                let (sb, b, eb) = side(m[l - 1], m[l], sides.1, hb, real(T::zero()), &s, false);
                (ha, hb, sides) = (a, b, (sa, sb));
                if let (Some(ea), Some(eb)) = (ea, eb) {
                    self.sens.push((ea, eb));
                }
            }
            self.outer.push(sides);
            if derivatives {
                let u0 = self.u[0][k + 1];
                let core_h = (core.d1(k, u0), core.d1_plus(k, u0));
                let extra = Self::adjoint(&self.sens, &mut self.dh, x, m, k, core_h);
                let (la, lb) = self.sens[last - 1];
                self.outer_derivs.push(OuterDerivs {
                    hp: [la.hp, lb.hp],
                    hplus: [la.hplus, lb.hplus],
                    extra_m: extra,
                });
            }
        }
        let od = derivatives.then_some(&self.outer_derivs[..]);
        self.mie
            .finish(x[last], m[last], n, derivatives, Some(&self.outer), od);
        if derivatives {
            self.inner_grads(m[last], n);
        }
        // A lossless particle absorbs nothing; H is real then in exact arithmetic, but its
        // complex intermediates leave a rounding-level imaginary part.
        if m.iter().all(|m| m.im.is_zero()) {
            self.mie.abs_sum = T::zero();
            // And it stays lossless as any x or Re m changes.
            self.mie.abs_sum_jacobian[0] = T::zero();
            self.mie.abs_sum_jacobian[1] = T::zero();
            for g in &mut self.grads {
                g.abs_sum[0] = T::zero();
                g.abs_sum[1] = T::zero();
            }
        }
    }

    /// The adjoint sweep at order `n`, from the sensitivities of every layer: dH/dx_l and dH/dm_l
    /// of the outer H into `dh` for every layer but the last, and for the last its index's
    /// derivative through its inner surface, which it returns (for Hₐ, H_b). `core` is
    /// (D_n, D_n + zD′_n) of the core's argument.
    ///
    /// An interface x_{l−1} moves both the inner layer's outer surface and layer l's inner one,
    /// whose terms cancel at leading order for H_b. Together, per unit of the outer H's
    /// sensitivity λ_l R_l² to layer l's starting value, they are the closed forms
    /// J = m_l(m_l² − m_{l−1}²)[n(n + 1)/(x² m_{l−1}² m_l²) + H²/m_{l−1}²] for Hₐ and
    /// (m_l² − m_{l−1}²)/m_l for H_b (H the inner layer's, x = x_{l−1}): the terms in n(n + 1)/x² and
    /// H² cancel exactly. Likewise the inner layer's index, through its outer surface and t_l:
    /// −(m_l/m_{l−1}²)(H − zH′) for Hₐ and (H + zH′)/m_l for H_b, z = m_{l−1}x_{l−1}.
    fn adjoint(
        sens: &[(Sens<T>, Sens<T>)],
        dh: &mut [[Vec<Complex<T>>; 4]],
        x: &[T],
        m: &[Complex<T>],
        n: usize,
        core: (Complex<T>, Complex<T>),
    ) -> [Complex<T>; 2] {
        let layers = x.len();
        let last = layers - 1;
        let two = T::from_f64(2.0);
        let nn1 = int::<T>(n) * int::<T>(n + 1);
        let mut extra = [real(T::zero()); 2];
        for side in 0..2 {
            let mut dx = vec![real(T::zero()); layers];
            let mut dm = vec![real(T::zero()); layers];
            let mut lambda = real(T::one());
            for l in (1..layers).rev() {
                let e = if side == 0 {
                    sens[l - 1].0
                } else {
                    sens[l - 1].1
                };
                let (ml, mp) = (m[l], m[l - 1]);
                // Layer l's index through its inner surface and t (over R²: em/m_l).
                dm[l] = dm[l] + lambda * e.r2 * e.em * inv(ml);
                // The inner layer's H at the interface, and H + zH′ there.
                let (h, hplus) = if l == 1 {
                    core
                } else {
                    let p = if side == 0 {
                        sens[l - 2].0
                    } else {
                        sens[l - 2].1
                    };
                    (p.hv, p.hplus)
                };
                let c = (ml - mp) * (ml + mp);
                let w = lambda * e.r2;
                let inv_mp = inv(mp);
                let (jump, k) = if side == 0 {
                    let inv_x = x[l - 1].recip();
                    let j = ml
                        * c
                        * (real(nn1 * inv_x * inv_x) * inv(mp * mp * ml * ml)
                            + h * h * inv_mp * inv_mp);
                    // H − zH′ = 2H − (H + zH′).
                    (j, -(ml * inv_mp * inv_mp * (h.scale(two) - hplus)))
                } else {
                    let inv_ml = inv(ml);
                    (c * inv_ml, hplus * inv_ml)
                };
                dx[l - 1] = dx[l - 1] + w * jump;
                dm[l - 1] = dm[l - 1] + w * k;
                let (mi, mo) = if side == 0 { (ml, mp) } else { (mp, ml) };
                lambda = w * mi * inv(mo);
            }
            extra[side] = dm[last];
            for l in 0..last {
                dh[l][2 * side].push(dx[l]);
                dh[l][2 * side + 1].push(dm[l]);
            }
        }
        extra
    }

    /// The derivatives of the coefficients and the absorption sum with respect to every layer
    /// but the last, from `self.dh` and what the outer step stored: with B = Hₐ/m (m H_b),
    /// ∂a/∂p = −i(∂B/∂p)/Δₐ², and the absorption term −Im(B)/|Δ|² changes by
    /// −[Im(∂B) − 2 Im(B) Re(∂B·ξ_n/Δ)]/|Δ|².
    fn inner_grads(&mut self, m: Complex<T>, n: usize) {
        let last = self.dh.len() - 1;
        let i1 = Complex::new(T::zero(), T::one());
        let two = T::from_f64(2.0);
        let inv_m = inv(m);
        self.grads.resize_with(last, LayerGrad::new);
        self.grads.truncate(last);
        let term = |f: Complex<T>, df: Complex<T>, l: Complex<T>, n2: T| {
            -((df.im - two * f.im * l.re) * n2)
        };
        for (l, g) in self.grads.iter_mut().enumerate() {
            for v in [&mut g.da_dx, &mut g.da_dm, &mut g.db_dx, &mut g.db_dm] {
                v.clear();
            }
            let mut jac = [T::zero(); 3];
            let dh = &self.dh[l];
            #[allow(clippy::needless_range_loop)] // `dh`'s four arrays and `stored` are parallel
            for k in 0..n {
                let [inv_da, inv_db, g_a, g_b, b_a, b_b] = self.mie.stored[k];
                let (ida2, idb2) = (inv_da * inv_da, inv_db * inv_db);
                let (na, nb) = (inv_da.norm_sqr(), inv_db.norm_sqr());
                let ba_x = dh[0][k] * inv_m;
                let ba_m = dh[1][k] * inv_m;
                let bb_x = m * dh[2][k];
                let bb_m = m * dh[3][k];
                g.da_dx.push(-(i1 * ba_x * ida2));
                g.da_dm.push(-(i1 * ba_m * ida2));
                g.db_dx.push(-(i1 * bb_x * idb2));
                g.db_dm.push(-(i1 * bb_m * idb2));
                let weight: T = int(2 * k + 3);
                let pair = |da: Complex<T>, db: Complex<T>| {
                    term(b_a, da, da * g_a, na) + term(b_b, db, db * g_b, nb)
                };
                jac[0] += weight * pair(ba_x, bb_x);
                jac[1] += weight * pair(ba_m, bb_m);
                jac[2] += weight * pair(i1 * ba_m, i1 * bb_m);
            }
            g.abs_sum = jac;
        }
    }
}
