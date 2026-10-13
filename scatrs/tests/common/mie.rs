//! Shared by the Mie tests: solving one sphere in any scalar type, and the scale (the
//! denominator) and condition number of each error measure of `scatrs::mie::MieOutput`, from
//! double-double solutions (see `scatrs::mie::ValidatedDomain`).

use num_complex::Complex;
use scatrs::Real;
use scatrs::mie::{AmplitudeBuffers, Efficiencies, Mie, MieOptions};
use scatrs::scalar::DoubleDouble as Dd;

/// Everything the Mie tests compare, from one solution.
#[derive(Clone, Debug)]
pub struct Outputs<T> {
    pub a: Vec<Complex<T>>,
    pub b: Vec<Complex<T>>,
    /// ∂a/∂x, da/dm, ∂b/∂x, db/dm.
    pub d: [Vec<Complex<T>>; 4],
    pub q: Efficiencies<T>,
    /// ∂/∂x, ∂/∂Re m, ∂/∂Im m.
    pub jac: [Efficiencies<T>; 3],
    /// S₁, S₂, ∂S₁/∂x, ∂S₂/∂x, dS₁/dm, dS₂/dm at each cosine.
    pub s: [Vec<Complex<T>>; 6],
}

/// Solves (x, m) with derivatives and amplitudes at `mu`, inside or outside the domain.
pub fn solve<T: Real>(x: T, m: Complex<T>, mu: &[T], n_max: Option<usize>) -> Outputs<T> {
    let mut opts = MieOptions::new().with_allow_unvalidated(true);
    if let Some(n) = n_max {
        opts = opts.with_n_max(n);
    }
    let mut mie = Mie::<T>::new();
    let sol = mie
        .compute_with_derivatives(x, m, &opts)
        .expect("valid inputs");
    let d = sol.derivatives().expect("computed");
    let jac = sol.efficiencies_jacobian().expect("computed");
    let zero = Complex::new(T::zero(), T::zero());
    let mut s: [Vec<Complex<T>>; 6] = std::array::from_fn(|_| vec![zero; mu.len()]);
    {
        let [s1, s2, dx1, dx2, dm1, dm2] = &mut s;
        sol.amplitudes_with_derivatives(
            mu,
            AmplitudeBuffers { s1, s2 },
            AmplitudeBuffers { s1: dx1, s2: dx2 },
            AmplitudeBuffers { s1: dm1, s2: dm2 },
        )
        .expect("valid cosines");
    }
    Outputs {
        a: sol.a().to_vec(),
        b: sol.b().to_vec(),
        d: [
            d.da_dx.to_vec(),
            d.da_dm.to_vec(),
            d.db_dx.to_vec(),
            d.db_dm.to_vec(),
        ],
        q: sol.efficiencies(),
        jac: [jac.x, jac.m_re, jac.m_im],
        s,
    }
}

/// Efficiency k of `e`: Q_ext, Q_sca, Q_abs, Q_back, g.
pub fn q_of<T: Copy>(e: &Efficiencies<T>, k: usize) -> T {
    [e.q_ext, e.q_sca, e.q_abs, e.q_back, e.g][k]
}

pub const EFFICIENCY_NAMES: [&str; 5] = ["q_ext", "q_sca", "q_abs", "q_back", "g"];

/// The denominator and the condition number of one error measure.
#[derive(Clone, Copy, Debug)]
pub struct Measure {
    pub scale: f64,
    pub condition: f64,
}

/// The measures of the values at (x, m).
pub struct ValueMeasures {
    /// aₙ, bₙ, over the orders asked for.
    pub coefficients: [Measure; 2],
    pub efficiencies: [Measure; 5],
    /// S₁, S₂ at each cosine.
    pub amplitudes: Vec<[Measure; 2]>,
}

/// The measures of the derivatives at (x, m).
pub struct DerivativeMeasures {
    /// ∂a/∂x, da/dm, ∂b/∂x, db/dm, over the orders asked for.
    pub coefficients: [Measure; 4],
    /// [∂/∂x, ∂/∂Re m, ∂/∂Im m][Q_ext, Q_sca, Q_abs, Q_back, g].
    pub jacobian: [[Measure; 5]; 3],
    /// [∂/∂x, d/dm][S₁, S₂], over all cosines.
    pub amplitudes: [[Measure; 2]; 2],
}

fn cnorm(z: Complex<Dd>) -> f64 {
    z.re.hi().hypot(z.im.hi())
}

/// The floor of the asymmetry measure, 10⁻³ min(1, x²).
pub fn g_floor(x: f64) -> f64 {
    1e-3 * x.powi(2).min(1.0)
}

/// The double-double solution at (x, m) and cosines `mu`, with the default truncation.
pub fn reference(x: f64, m: Complex<f64>, mu: &[f64]) -> Outputs<Dd> {
    let mu_dd: Vec<Dd> = mu.iter().map(|&v| Dd::from_f64(v)).collect();
    solve(
        Dd::from_f64(x),
        Complex::new(Dd::from_f64(m.re), Dd::from_f64(m.im)),
        &mu_dd,
        None,
    )
}

/// max over `orders` (1-based) of |values| and of x·|d_dx| + |m|·|d_dm|, as a measure.
fn array_measure(
    orders: &[usize],
    values: &[Complex<Dd>],
    x: f64,
    abs_m: f64,
    d_dx: &dyn Fn(usize) -> f64,
    d_dm: &dyn Fn(usize) -> f64,
) -> Measure {
    let scale = orders
        .iter()
        .map(|&n| cnorm(values[n - 1]))
        .fold(0.0, f64::max);
    let num = orders
        .iter()
        .map(|&n| x * d_dx(n - 1) + abs_m * d_dm(n - 1))
        .fold(0.0, f64::max);
    Measure {
        scale,
        condition: num / scale,
    }
}

/// The sizes of Q_ext, Q_sca, Q_abs (Q_ext when Q_abs = 0), Q_back and g in their measures.
fn efficiency_sizes(d: &Outputs<Dd>, x: f64) -> [f64; 5] {
    let q = |k: usize| q_of(&d.q, k).hi();
    let q_abs = if q(2) > 0.0 { q(2) } else { q(0) };
    [
        q(0),
        q(1),
        q_abs,
        q(3).max(q(1)),
        q(4).abs().max(g_floor(x)),
    ]
}

/// The measures of the values at (x, m), from the double-double solution `d` there (whose
/// cosines are those of the amplitudes), over the coefficient orders `orders`.
pub fn value_measures(x: f64, m: Complex<f64>, d: &Outputs<Dd>, orders: &[usize]) -> ValueMeasures {
    let abs_m = m.norm();
    // Holomorphic outputs (coefficients, amplitudes): |m|·|dy/dm|, which rounding m bounds up
    // to a factor √2. Real outputs: componentwise, since rounding m perturbs Re m by at most
    // u·|Re m| and Im m by at most u·|Im m| (a real m stays real).
    let cond = |dx: f64, dm: f64, scale: f64| (x * dx.abs() + abs_m * dm) / scale;
    let cond_real = |dx: f64, dr: f64, di: f64, scale: f64| {
        (x * dx.abs() + m.re.abs() * dr.abs() + m.im.abs() * di.abs()) / scale
    };
    let coefficients = [
        array_measure(orders, &d.a, x, abs_m, &|k| cnorm(d.d[0][k]), &|k| {
            cnorm(d.d[1][k])
        }),
        array_measure(orders, &d.b, x, abs_m, &|k| cnorm(d.d[2][k]), &|k| {
            cnorm(d.d[3][k])
        }),
    ];
    let sizes = efficiency_sizes(d, x);
    let efficiencies = std::array::from_fn(|k| {
        let (jx, jr, ji) = (
            q_of(&d.jac[0], k).hi(),
            q_of(&d.jac[1], k).hi(),
            q_of(&d.jac[2], k).hi(),
        );
        Measure {
            scale: sizes[k],
            condition: cond_real(jx, jr, ji, sizes[k]),
        }
    });
    let s_rms = x * q_of(&d.q, 1).hi().sqrt() / 2.0;
    let amplitudes = (0..d.s[0].len())
        .map(|j| {
            std::array::from_fn(|i| {
                let scale = cnorm(d.s[i][j]).max(s_rms);
                Measure {
                    scale,
                    condition: cond(cnorm(d.s[2 + i][j]), cnorm(d.s[4 + i][j]), scale),
                }
            })
        })
        .collect();
    ValueMeasures {
        coefficients,
        efficiencies,
        amplitudes,
    }
}

/// The measures of the derivatives at (x, m), from the double-double solution `d` there and
/// central differences of double-double derivatives, over the coefficient orders `orders` and
/// the cosines `mu`.
///
/// The steps are 2⁻³⁰ x and 2⁻³⁰ |m|, one-sided in Im m at Im m = 0. Their relative truncation
/// error is about (2⁻³⁰ |m|x)²/6 (10⁻⁸ at |m|x = 3·10⁵; first order, about 10⁻⁴, for the
/// one-sided difference, which only real outputs use and multiply by Im m = 0), and near a
/// resonance narrower than the step the quotient is no derivative at all. That is enough for
/// a condition number, which only has to be right to within a small factor.
pub fn derivative_measures(
    x: f64,
    m: Complex<f64>,
    mu: &[f64],
    d: &Outputs<Dd>,
    orders: &[usize],
) -> DerivativeMeasures {
    let mu_dd: Vec<Dd> = mu.iter().map(|&v| Dd::from_f64(v)).collect();
    let at = |x: Dd, re: Dd, im: Dd| solve(x, Complex::new(re, im), &mu_dd, None);
    let (xd, rd, id) = (Dd::from_f64(x), Dd::from_f64(m.re), Dd::from_f64(m.im));
    let h_x = xd * Dd::from_f64(2f64.powi(-30));
    let h_m = Dd::from_f64(m.norm() * 2f64.powi(-30));
    let (px, nx) = (at(xd + h_x, rd, id), at(xd - h_x, rd, id));
    let (pr, nr) = (at(xd, rd + h_m, id), at(xd, rd - h_m, id));
    let one_sided = m.im < m.norm() * 2f64.powi(-30);
    let pi = at(xd, rd, id + h_m);
    let ni = if one_sided {
        d.clone()
    } else {
        at(xd, rd, id - h_m)
    };
    let two = Dd::from_f64(2.0);
    // ∂/∂x, ∂/∂Re m (= d/dm for holomorphic quantities) and ∂/∂Im m.
    let gx = |p: Dd, n: Dd| ((p - n) / (two * h_x)).hi();
    let gr = |p: Dd, n: Dd| ((p - n) / (two * h_m)).hi();
    let gi = |p: Dd, n: Dd| ((p - n) / if one_sided { h_m } else { two * h_m }).hi();
    let cx = |p: Complex<Dd>, n: Complex<Dd>| gx(p.re, n.re).hypot(gx(p.im, n.im));
    let cr = |p: Complex<Dd>, n: Complex<Dd>| gr(p.re, n.re).hypot(gr(p.im, n.im));
    let (abs_m, phase) = (m.norm(), 1.0 + m.norm() * x);
    let coefficients = std::array::from_fn(|i| {
        array_measure(
            orders,
            &d.d[i],
            x,
            abs_m,
            &|k| cx(px.d[i][k], nx.d[i][k]),
            &|k| cr(pr.d[i][k], nr.d[i][k]),
        )
    });
    // The natural size of a derivative: Q·(1 + |m|x)/|p|, with Q_ext for Q_abs.
    let mut sizes = efficiency_sizes(d, x);
    sizes[2] = sizes[0];
    let p_sizes = [x, abs_m, abs_m];
    let jacobian = std::array::from_fn(|p| {
        std::array::from_fn(|k| {
            let value = q_of(&d.jac[p], k).hi();
            let scale = value.abs().max(sizes[k] * phase / p_sizes[p]);
            let dx = gx(q_of(&px.jac[p], k), q_of(&nx.jac[p], k));
            let dr = gr(q_of(&pr.jac[p], k), q_of(&nr.jac[p], k));
            let di = gi(q_of(&pi.jac[p], k), q_of(&ni.jac[p], k));
            Measure {
                scale,
                condition: (x * dx.abs() + m.re.abs() * dr.abs() + m.im.abs() * di.abs()) / scale,
            }
        })
    });
    let s_rms = x * q_of(&d.q, 1).hi().sqrt() / 2.0;
    let amplitudes = std::array::from_fn(|p| {
        std::array::from_fn(|i| {
            let k = 2 + 2 * p + i;
            let natural = s_rms * phase / [x, abs_m][p];
            let scale = d.s[k]
                .iter()
                .map(|&v| cnorm(v).max(natural))
                .fold(0.0, f64::max);
            let num = (0..mu.len())
                .map(|j| x * cx(px.s[k][j], nx.s[k][j]) + abs_m * cr(pr.s[k][j], nr.s[k][j]))
                .fold(0.0, f64::max);
            Measure {
                scale,
                condition: num / scale,
            }
        })
    });
    DerivativeMeasures {
        coefficients,
        jacobian,
        amplitudes,
    }
}
