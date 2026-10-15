//! The functions the EBCM integrands are built from, at one quadrature node: the outgoing and
//! regular radial functions of the medium (x = k r(θ)), the regular radial functions of the
//! particle (z = s x), and the normalized angular functions of one azimuthal mode.
//!
//! The radial functions span hundreds of orders of magnitude over the orders and the surface
//! (hₙ(x) grows like (2n − 1)!!/x^{n+1}), so they are kept scaled: the medium's of order n by
//! σₙ = 1/|hₙ(x_min)|, x_min = k r_min, and the particle's of order k by c_k = 1/E_k, a smooth
//! envelope of |j_k| at z_max = s k r_max. Every scaled value stays of order 1 or below over the
//! surface. The scaling is undone on T (`super::kernel`).

use num_complex::Complex;

use crate::complex::inv;
use crate::mie::kernel::{int, ratios_complex, ratios_real, real};
use crate::scalar::Real;

/// The scale factors of the radial functions: σₙ = 1/|hₙ(x_min)| through the moduli of the
/// upward ratios hₙ/hₙ₋₁ at x_min, and c_k = 1/E_k with E_k = e^{|Im z_max|}/|z_max| ·
/// Π_{i ≤ k} min(1, |z_max|/(2i + 1)).
#[derive(Clone, Debug)]
pub(crate) struct Scales<T> {
    /// |hₙ(x_min)|/|hₙ₋₁(x_min)| for n = 1, …, N + 1 (index n), and |h₀(x_min)| at index 0.
    pub(crate) h_ratio: Vec<T>,
    /// E_k/E_{k−1} for k = 1, …, N + 1 (index k), and E₀ at index 0 (without the e^{|Im z_max|}).
    pub(crate) e_ratio: Vec<T>,
    /// |Im z_max|.
    pub(crate) im_z_max: T,
}

impl<T: Real> Scales<T> {
    pub(crate) fn new(x_min: T, z_max: Complex<T>, n: usize) -> Self {
        let mut h_ratio = Vec::with_capacity(n + 2);
        h_ratio.push(x_min.recip());
        // h₁/h₀ = 1/x − i, h_{n+1}/hₙ = (2n + 1)/x − hₙ₋₁/hₙ.
        let inv_x = x_min.recip();
        let mut r = Complex::new(inv_x, -T::one());
        h_ratio.push(r.norm());
        for k in 1..=n {
            r = real(inv_x * int::<T>(2 * k + 1)) - inv(r);
            h_ratio.push(r.norm());
        }
        let za = z_max.norm();
        let mut e_ratio = Vec::with_capacity(n + 2);
        e_ratio.push(za.recip());
        for k in 1..=n + 1 {
            e_ratio.push((za / int::<T>(2 * k + 1)).min(T::one()));
        }
        Self {
            h_ratio,
            e_ratio,
            im_z_max: z_max.im.abs(),
        }
    }
}

/// The scaled radial functions of one node.
#[derive(Clone, Debug)]
pub(crate) struct Radial<T> {
    /// σₙ hₙ(x), σₙ [x hₙ(x)]'/x and σₙ hₙ(x)/x (index n = 0, …, N).
    pub(crate) h: Vec<Complex<T>>,
    pub(crate) dh: Vec<Complex<T>>,
    pub(crate) h_x: Vec<Complex<T>>,
    /// σₙ jₙ(x), σₙ [x jₙ]'/x, σₙ jₙ/x.
    pub(crate) j: Vec<Complex<T>>,
    pub(crate) dj: Vec<Complex<T>>,
    pub(crate) j_x: Vec<Complex<T>>,
    /// c_k j_k(z), c_k [z j_k]'/z, c_k j_k/z.
    pub(crate) jz: Vec<Complex<T>>,
    pub(crate) djz: Vec<Complex<T>>,
    pub(crate) jz_z: Vec<Complex<T>>,
    u_x: Vec<T>,
    u_z: Vec<Complex<T>>,
}

impl<T: Real> Radial<T> {
    pub(crate) fn new() -> Self {
        Self {
            h: Vec::new(),
            dh: Vec::new(),
            h_x: Vec::new(),
            j: Vec::new(),
            dj: Vec::new(),
            j_x: Vec::new(),
            jz: Vec::new(),
            djz: Vec::new(),
            jz_z: Vec::new(),
            u_x: Vec::new(),
            u_z: Vec::new(),
        }
    }

    /// The functions at x (real) and z = s x, orders 0, …, n.
    pub(crate) fn compute(&mut self, x: T, s: Complex<T>, n: usize, scales: &Scales<T>) {
        let z = s.scale(x);
        let inv_x = x.recip();
        let inv_z = inv(z);
        for v in [
            &mut self.h,
            &mut self.dh,
            &mut self.h_x,
            &mut self.j,
            &mut self.dj,
            &mut self.j_x,
            &mut self.jz,
            &mut self.djz,
            &mut self.jz_z,
        ] {
            v.clear();
        }
        // Outgoing: h₀ = −i e^{ix}/x, hₙ = hₙ₋₁ rₙ with the upward ratios; scaled by σₙ, so
        // σₙ hₙ = (σₙ₋₁ hₙ₋₁)·rₙ/|rₙ(x_min)|.
        let (sx, cx) = x.sin_cos();
        let h0 = Complex::new(sx, -cx).scale(inv_x);
        let mut hs = h0.scale(scales.h_ratio[0].recip());
        let mut prev = Complex::new(cx, sx).scale(inv_x * scales.h_ratio[0].recip());
        // h₋₁ = e^{ix}/x (h_{−1} = j_{−1} + i y_{−1} = cos x/x + i sin x/x); σ₋₁ := σ₀.
        let mut r = Complex::new(inv_x, -T::one());
        for k in 0..=n {
            if k > 0 {
                r = if k == 1 {
                    Complex::new(inv_x, -T::one())
                } else {
                    real(inv_x * int::<T>(2 * k - 1)) - inv(r)
                };
                prev = hs.scale(scales.h_ratio[k].recip());
                hs = (hs * r).scale(scales.h_ratio[k].recip());
            }
            self.h.push(hs);
            self.h_x.push(hs.scale(inv_x));
            // [x hₖ]'/x = hₖ₋₁ − k hₖ/x, both scaled by σₖ.
            self.dh.push(prev - hs.scale(inv_x * int::<T>(k)));
        }
        // Regular jₙ(x) = j₀ Π u_i with the downward ratios u_i = jᵢ/jᵢ₋₁, scaled by σₙ.
        ratios_real(&mut self.u_x, x, 1, n);
        let j0 = sx * inv_x;
        let mut js = j0 / scales.h_ratio[0];
        let mut jprev = cx * inv_x / scales.h_ratio[0]; // j₋₁ = cos x/x
        for k in 0..=n {
            if k > 0 {
                jprev = js / scales.h_ratio[k];
                js = js * self.u_x[k] / scales.h_ratio[k];
            }
            self.j.push(real(js));
            self.j_x.push(real(js * inv_x));
            self.dj.push(real(jprev - js * inv_x * int::<T>(k)));
        }
        // Particle: j_k(z) = j₀(z) Π u_i(z), scaled by c_k; j₀(z) e^{−|Im z_max|} without
        // overflow: sin z = (e^{iz} − e^{−iz})/(2i).
        ratios_complex(&mut self.u_z, z, n);
        let sin_scaled = sin_scaled(z, scales.im_z_max);
        let cos_scaled = cos_scaled(z, scales.im_z_max);
        let mut jz = sin_scaled * inv_z.scale(scales.e_ratio[0].recip());
        let mut jzprev = cos_scaled * inv_z.scale(scales.e_ratio[0].recip());
        for k in 0..=n {
            if k > 0 {
                jzprev = jz.scale(scales.e_ratio[k].recip());
                jz = (jz * self.u_z[k]).scale(scales.e_ratio[k].recip());
            }
            self.jz.push(jz);
            self.jz_z.push(jz * inv_z);
            self.djz.push(jzprev - jz * inv_z.scale(int::<T>(k)));
        }
    }
}

/// e^{iz}, e^{−iz} times e^{−a}, a ≥ |Im z|: with exponents ≤ 0.
fn exps_scaled<T: Real>(z: Complex<T>, a: T) -> (Complex<T>, Complex<T>) {
    let (s, c) = z.re.sin_cos();
    let plus = Complex::new(c, s).scale((-z.im - a).exp());
    let minus = Complex::new(c, -s).scale((z.im - a).exp());
    (plus, minus)
}

/// sin z · e^{−a}.
fn sin_scaled<T: Real>(z: Complex<T>, a: T) -> Complex<T> {
    let (p, m) = exps_scaled(z, a);
    // (e^{iz} − e^{−iz})/(2i) = −i (p − m)/2.
    let d = p - m;
    Complex::new(d.im, -d.re).scale(T::from_f64(0.5))
}

/// cos z · e^{−a}.
fn cos_scaled<T: Real>(z: Complex<T>, a: T) -> Complex<T> {
    let (p, m) = exps_scaled(z, a);
    (p + m).scale(T::from_f64(0.5))
}

/// The normalized angular functions of mode m at cos θ = `ct`, sin θ = `st` > 0, for orders
/// 0, …, n: d_{mn} = √((n−m)!/(n+m)!) Pₙᵐ(cos θ) (Mishchenko's dₙ, with his sign), πₙ = m dₙ/sin θ
/// and τₙ = d dₙ/dθ.
#[derive(Clone, Debug)]
pub(crate) struct Angular<T> {
    pub(crate) d: Vec<T>,
    pub(crate) pi: Vec<T>,
    pub(crate) tau: Vec<T>,
}

impl<T: Real> Angular<T> {
    pub(crate) fn new() -> Self {
        Self {
            d: Vec::new(),
            pi: Vec::new(),
            tau: Vec::new(),
        }
    }

    pub(crate) fn compute(&mut self, m: usize, n: usize, ct: T, st: T) {
        let zero = T::zero();
        self.d.clear();
        self.d.resize(n + 1, zero);
        self.pi.clear();
        self.pi.resize(n + 1, zero);
        self.tau.clear();
        self.tau.resize(n + 1, zero);
        if m > n {
            return;
        }
        let mut am = T::one();
        for k in 1..=m {
            am *= (int::<T>(2 * k - 1) / int::<T>(2 * k)).sqrt();
        }
        self.d[m] = am * st.powi(m as i32);
        if m < n {
            self.d[m + 1] = int::<T>(2 * m + 1).sqrt() * ct * self.d[m];
        }
        for k in m + 1..n {
            let a = int::<T>(k * k - m * m).sqrt();
            let b = int::<T>((k + 1) * (k + 1) - m * m).sqrt();
            self.d[k + 1] = (int::<T>(2 * k + 1) * ct * self.d[k] - a * self.d[k - 1]) / b;
        }
        let inv_st = st.recip();
        for k in m.max(1)..=n {
            let prev = if k > m { self.d[k - 1] } else { zero };
            let a = int::<T>(k * k - m * m).sqrt();
            self.tau[k] = (int::<T>(k) * ct * self.d[k] - a * prev) * inv_st;
            self.pi[k] = int::<T>(m) * self.d[k] * inv_st;
        }
    }
}
