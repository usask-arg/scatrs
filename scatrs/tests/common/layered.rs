//! Shared by the layered-sphere tests: solving one layered sphere in any scalar type, and the
//! scale (the denominator) and condition number of each error measure of
//! `scatrs::mie::MieOutput`, from double-double solutions (see
//! `scatrs::layered::ValidatedDomain`): as for a sphere (`common::mie`), with the condition
//! numbers summed over every layer's x_l and m_l.

use num_complex::Complex;
use scatrs::Real;
use scatrs::layered::Layered;
use scatrs::mie::{AmplitudeBuffers, Efficiencies, MieOptions};
use scatrs::scalar::DoubleDouble as Dd;

#[allow(unused_imports)] // each test binary uses some of them
pub use super::mie::{EFFICIENCY_NAMES, Measure, g_floor, q_of};

/// Everything the layered tests compare, from one solution.
#[derive(Clone, Debug)]
pub struct Outputs<T> {
    pub a: Vec<Complex<T>>,
    pub b: Vec<Complex<T>>,
    /// Per layer: ∂a/∂x_l, da/dm_l, ∂b/∂x_l, db/dm_l.
    pub d: Vec<[Vec<Complex<T>>; 4]>,
    pub q: Efficiencies<T>,
    /// Per layer: ∂/∂x_l, ∂/∂Re m_l, ∂/∂Im m_l.
    pub jac: Vec<[Efficiencies<T>; 3]>,
    /// S₁, S₂ at each cosine.
    pub s: [Vec<Complex<T>>; 2],
    /// Per layer: ∂S₁/∂x_l, ∂S₂/∂x_l, dS₁/dm_l, dS₂/dm_l at each cosine.
    pub ds: Vec<[Vec<Complex<T>>; 4]>,
}

/// Solves the layered sphere with derivatives and amplitudes at `mu`, inside or outside the
/// domain.
pub fn solve<T: Real>(x: &[T], m: &[Complex<T>], mu: &[T], n_max: Option<usize>) -> Outputs<T> {
    let mut opts = MieOptions::new().with_allow_unvalidated(true);
    if let Some(n) = n_max {
        opts = opts.with_n_max(n);
    }
    let mut lay = Layered::<T>::new();
    let sol = lay
        .compute_with_derivatives(x, m, &opts)
        .expect("valid inputs");
    let zero = Complex::new(T::zero(), T::zero());
    let layers = x.len();
    let mut s: [Vec<Complex<T>>; 2] = std::array::from_fn(|_| vec![zero; mu.len()]);
    let mut ds = Vec::with_capacity(layers);
    for l in 0..layers {
        let mut v: [Vec<Complex<T>>; 4] = std::array::from_fn(|_| vec![zero; mu.len()]);
        {
            let [s1, s2] = &mut s;
            let [dx1, dx2, dm1, dm2] = &mut v;
            sol.amplitudes_with_derivatives(
                mu,
                l,
                AmplitudeBuffers { s1, s2 },
                AmplitudeBuffers { s1: dx1, s2: dx2 },
                AmplitudeBuffers { s1: dm1, s2: dm2 },
            )
            .expect("valid cosines");
        }
        ds.push(v);
    }
    let jac = sol.efficiencies_jacobian().expect("computed");
    Outputs {
        a: sol.a().to_vec(),
        b: sol.b().to_vec(),
        d: (0..layers)
            .map(|l| {
                let d = sol.derivatives(l).expect("computed");
                [
                    d.da_dx.to_vec(),
                    d.da_dm.to_vec(),
                    d.db_dx.to_vec(),
                    d.db_dm.to_vec(),
                ]
            })
            .collect(),
        q: sol.efficiencies(),
        jac: jac.iter().map(|j| [j.x, j.m_re, j.m_im]).collect(),
        s,
        ds,
    }
}

fn dd(v: f64) -> Dd {
    Dd::from_f64(v)
}

fn dd_inputs(x: &[f64], m: &[Complex<f64>]) -> (Vec<Dd>, Vec<Complex<Dd>>) {
    (
        x.iter().map(|&v| dd(v)).collect(),
        m.iter().map(|v| Complex::new(dd(v.re), dd(v.im))).collect(),
    )
}

/// The double-double solution at (x, m) and cosines `mu`, with `n_max` terms (default: the
/// f64 default, which the references were checked with).
pub fn reference(x: &[f64], m: &[Complex<f64>], mu: &[f64], n_max: Option<usize>) -> Outputs<Dd> {
    let mu_dd: Vec<Dd> = mu.iter().map(|&v| dd(v)).collect();
    let (xd, md) = dd_inputs(x, m);
    solve(&xd, &md, &mu_dd, n_max)
}

pub fn cnorm(z: Complex<Dd>) -> f64 {
    z.re.hi().hypot(z.im.hi())
}

/// The measures of the values.
pub struct ValueMeasures {
    /// aₙ, bₙ, over the orders asked for.
    pub coefficients: [Measure; 2],
    pub efficiencies: [Measure; 5],
    /// S₁, S₂ at each cosine.
    pub amplitudes: Vec<[Measure; 2]>,
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

/// The measures of the values of the layered sphere (x, m), from its double-double solution
/// `d`, over the coefficient orders `orders` (1-based).
pub fn value_measures(
    x: &[f64],
    m: &[Complex<f64>],
    d: &Outputs<Dd>,
    orders: &[usize],
) -> ValueMeasures {
    let layers = x.len();
    let x_l = x[layers - 1];
    // Holomorphic outputs: Σ_l x_l|∂y/∂x_l| + |m_l||dy/dm_l|; real outputs componentwise.
    let holo = |dx: &dyn Fn(usize) -> f64, dm: &dyn Fn(usize) -> f64| -> f64 {
        (0..layers)
            .map(|l| x[l] * dx(l) + m[l].norm() * dm(l))
            .sum()
    };
    let coefficient = |values: &[Complex<Dd>], i: usize| {
        let scale = orders
            .iter()
            .map(|&n| cnorm(values[n - 1]))
            .fold(0.0, f64::max);
        let num = orders
            .iter()
            .map(|&n| {
                holo(&|l| cnorm(d.d[l][i][n - 1]), &|l| {
                    cnorm(d.d[l][i + 1][n - 1])
                })
            })
            .fold(0.0, f64::max);
        Measure {
            scale,
            condition: num / scale,
        }
    };
    let coefficients = [coefficient(&d.a, 0), coefficient(&d.b, 2)];
    let sizes = efficiency_sizes(d, x_l);
    let efficiencies = std::array::from_fn(|k| {
        let num: f64 = (0..layers)
            .map(|l| {
                let j = &d.jac[l];
                x[l] * q_of(&j[0], k).hi().abs()
                    + m[l].re.abs() * q_of(&j[1], k).hi().abs()
                    + m[l].im.abs() * q_of(&j[2], k).hi().abs()
            })
            .sum();
        Measure {
            scale: sizes[k],
            condition: num / sizes[k],
        }
    });
    let s_rms = x_l * q_of(&d.q, 1).hi().sqrt() / 2.0;
    let amplitudes = (0..d.s[0].len())
        .map(|j| {
            std::array::from_fn(|i| {
                let scale = cnorm(d.s[i][j]).max(s_rms);
                let num = holo(&|l| cnorm(d.ds[l][i][j]), &|l| cnorm(d.ds[l][2 + i][j]));
                Measure {
                    scale,
                    condition: num / scale,
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

/// The measures of the derivatives.
pub struct DerivativeMeasures {
    /// Per layer: ∂a/∂x_l, da/dm_l, ∂b/∂x_l, db/dm_l over the orders asked for.
    pub coefficients: Vec<[Measure; 4]>,
    /// Per layer: [∂/∂x_l, ∂/∂Re m_l, ∂/∂Im m_l][Q_ext, Q_sca, Q_abs, Q_back, g].
    pub jacobian: Vec<[[Measure; 5]; 3]>,
    /// Per layer: [∂/∂x_l, d/dm_l][S₁, S₂], over all cosines.
    pub amplitudes: Vec<[[Measure; 2]; 2]>,
}

/// The measures of the derivatives of the layered sphere (x, m), from its double-double
/// solution `d` and central differences of double-double derivatives (steps 2⁻³⁰ x_l and
/// 2⁻³⁰ |m_l|, one-sided in Im m_l at Im m_l = 0, as `common::mie::derivative_measures`), over
/// the coefficient orders `orders` and the cosines `mu` (those of `d`). A derivative's scale is
/// at least its natural size, the
/// value's times (1 + |m|x_L)/|p| (|m| the largest index, |p| = x_l or |m_l|): a derivative can
/// vanish identically (with respect to the index of a shell of zero thickness).
pub fn derivative_measures(
    x: &[f64],
    m: &[Complex<f64>],
    mu: &[f64],
    d: &Outputs<Dd>,
    orders: &[usize],
) -> DerivativeMeasures {
    let mu_dd: Vec<Dd> = mu.iter().map(|&v| dd(v)).collect();
    let layers = x.len();
    let x_l = x[layers - 1];
    let n = d.a.len();
    let big = m.iter().map(|v| v.norm()).fold(1.0, f64::max);
    let phase = 1.0 + big * x_l;
    let (xd, md) = dd_inputs(x, m);
    // at[l][p] = (plus, minus, divisor) for parameter p of layer l.
    let mut steps = Vec::with_capacity(layers);
    for l in 0..layers {
        let mut per = Vec::with_capacity(3);
        for p in 0..3 {
            let h = if p == 0 {
                x[l] * 2f64.powi(-30)
            } else {
                m[l].norm() * 2f64.powi(-30)
            };
            // One-sided where a step would make Im m negative or move x_l past a neighbour (a
            // shell of zero thickness).
            let up = p != 0 || l + 1 == layers || x[l] + h <= x[l + 1];
            let down = match p {
                0 => l == 0 || x[l] - h >= x[l - 1],
                1 => true,
                _ => m[l].im >= h,
            };
            let shifted = |sign: f64| {
                let (mut xs, mut ms) = (xd.clone(), md.clone());
                let s = dd(sign * h);
                match p {
                    0 => xs[l] += s,
                    1 => ms[l].re += s,
                    _ => ms[l].im += s,
                }
                solve(&xs, &ms, &mu_dd, Some(n))
            };
            per.push(match (up, down) {
                (true, true) => (shifted(1.0), shifted(-1.0), 2.0 * h),
                (true, false) => (shifted(1.0), d.clone(), h),
                (false, true) => (d.clone(), shifted(-1.0), h),
                (false, false) => unreachable!("a step is always possible one way"),
            });
        }
        steps.push(per);
    }
    let g = |pl: Dd, mi: Dd, div: f64| ((pl - mi) / dd(div)).hi();
    let cg = |pl: Complex<Dd>, mi: Complex<Dd>, div: f64| {
        g(pl.re, mi.re, div).hypot(g(pl.im, mi.im, div))
    };
    let a_size = orders
        .iter()
        .map(|&k| cnorm(d.a[k - 1]).max(cnorm(d.b[k - 1])))
        .fold(0.0, f64::max);
    let coefficients = (0..layers)
        .map(|l| {
            std::array::from_fn(|i| {
                let p_size = if i % 2 == 0 { x[l] } else { m[l].norm() };
                let scale = orders
                    .iter()
                    .map(|&k| cnorm(d.d[l][i][k - 1]))
                    .fold(a_size * phase / p_size, f64::max);
                let num = orders
                    .iter()
                    .map(|&k| {
                        (0..layers)
                            .map(|q| {
                                let (px, nx, dx) = &steps[q][0];
                                let (pr, nr, dr) = &steps[q][1];
                                x[q] * cg(px.d[l][i][k - 1], nx.d[l][i][k - 1], *dx)
                                    + m[q].norm() * cg(pr.d[l][i][k - 1], nr.d[l][i][k - 1], *dr)
                            })
                            .sum::<f64>()
                    })
                    .fold(0.0, f64::max);
                Measure {
                    scale,
                    condition: num / scale,
                }
            })
        })
        .collect();
    let mut sizes = efficiency_sizes(d, x_l);
    sizes[2] = sizes[0];
    let jacobian = (0..layers)
        .map(|l| {
            std::array::from_fn(|p| {
                let p_size = if p == 0 { x[l] } else { m[l].norm() };
                std::array::from_fn(|k| {
                    let value = q_of(&d.jac[l][p], k).hi();
                    let scale = value.abs().max(sizes[k] * phase / p_size);
                    let num: f64 = (0..layers)
                        .map(|q| {
                            let parts = [x[q], m[q].re.abs(), m[q].im.abs()];
                            (0..3)
                                .map(|r| {
                                    let (pl, mi, div) = &steps[q][r];
                                    parts[r]
                                        * g(q_of(&pl.jac[l][p], k), q_of(&mi.jac[l][p], k), *div)
                                            .abs()
                                })
                                .sum::<f64>()
                        })
                        .sum();
                    Measure {
                        scale,
                        condition: num / scale,
                    }
                })
            })
        })
        .collect();
    let s_rms = x_l * q_of(&d.q, 1).hi().sqrt() / 2.0;
    let amplitudes = (0..layers)
        .map(|l| {
            std::array::from_fn(|p| {
                std::array::from_fn(|i| {
                    let k = 2 * p + i;
                    let p_size = if p == 0 { x[l] } else { m[l].norm() };
                    let natural = s_rms * phase / p_size;
                    let scale = d.ds[l][k]
                        .iter()
                        .map(|&v| cnorm(v).max(natural))
                        .fold(0.0, f64::max);
                    let num = (0..mu.len())
                        .map(|j| {
                            (0..layers)
                                .map(|q| {
                                    let (px, nx, dx) = &steps[q][0];
                                    let (pr, nr, dr) = &steps[q][1];
                                    x[q] * cg(px.ds[l][k][j], nx.ds[l][k][j], *dx)
                                        + m[q].norm() * cg(pr.ds[l][k][j], nr.ds[l][k][j], *dr)
                                })
                                .sum::<f64>()
                        })
                        .fold(0.0, f64::max);
                    Measure {
                        scale,
                        condition: num / scale,
                    }
                })
            })
        })
        .collect();
    DerivativeMeasures {
        coefficients,
        jacobian,
        amplitudes,
    }
}
