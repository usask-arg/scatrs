//! Shared by the phase-matrix and Greek-coefficient tests: solving one sphere for both, and the
//! scale (the denominator) and condition number of each error measure of
//! `scatrs::mie::MieOutput::{PhaseMatrix, PhaseMatrixDerivatives, GreekCoefficients,
//! GreekDerivatives}`, from double-double solutions.

use num_complex::Complex;
use scatrs::Real;
use scatrs::mie::{GreekCoefficients, Mie, MieOptions, Normalization, PhaseBuffers};
use scatrs::scalar::DoubleDouble as Dd;

use super::check::max_nan;
use super::mie::Measure;

/// The names of the Greek coefficients, as in the reference records.
pub const GREEK_NAMES: [&str; 6] = ["alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2"];

/// The names of the phase-matrix elements, as in the reference records.
pub const PHASE_NAMES: [&str; 4] = ["f11", "f12", "f33", "f34"];

/// Everything the phase-matrix and Greek tests compare, from one solution.
#[derive(Clone, Debug)]
pub struct GreekOutputs<T> {
    /// α₁, α₂, α₃, α₄, β₁, β₂ for l = 0, …, L.
    pub greek: [Vec<T>; 6],
    /// Their derivatives with respect to x, Re m and Im m.
    pub greek_d: [[Vec<T>; 6]; 3],
    /// F₁₁, F₁₂, F₃₃, F₃₄ at each cosine.
    pub phase: [Vec<T>; 4],
    /// Their derivatives with respect to x, Re m and Im m.
    pub phase_d: [[Vec<T>; 4]; 3],
}

/// The arrays of `g`, ordered as `GREEK_NAMES`.
pub fn greek_arrays<T: Clone>(g: &GreekCoefficients<T>) -> [Vec<T>; 6] {
    [
        g.alpha1.clone(),
        g.alpha2.clone(),
        g.alpha3.clone(),
        g.alpha4.clone(),
        g.beta1.clone(),
        g.beta2.clone(),
    ]
}

fn buffers<T>(b: &mut [Vec<T>; 4]) -> PhaseBuffers<'_, T> {
    let [f11, f12, f33, f34] = b;
    PhaseBuffers { f11, f12, f33, f34 }
}

/// Solves (x, m) with derivatives, for the Greek coefficients up to `order` (default 2N) and
/// the phase matrix at `mu`, inside or outside the domain.
pub fn solve_greek<T: Real>(
    x: T,
    m: Complex<T>,
    mu: &[T],
    n_max: Option<usize>,
    order: Option<usize>,
    normalization: Normalization,
) -> GreekOutputs<T> {
    let mut opts = MieOptions::new().with_allow_unvalidated(true);
    if let Some(n) = n_max {
        opts = opts.with_n_max(n);
    }
    let mut mie = Mie::<T>::new();
    let sol = mie
        .compute_with_derivatives(x, m, &opts)
        .expect("valid inputs");
    let (g, dg) = sol
        .greek_with_derivatives(order, normalization)
        .expect("valid order");
    let zeros = || vec![T::zero(); mu.len()];
    let mut phase: [Vec<T>; 4] = std::array::from_fn(|_| zeros());
    let mut phase_d: [[Vec<T>; 4]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| zeros()));
    {
        let [dx, dr, di] = &mut phase_d;
        sol.phase_matrix_with_derivatives(
            mu,
            normalization,
            buffers(&mut phase),
            [buffers(dx), buffers(dr), buffers(di)],
        )
        .expect("valid cosines");
    }
    GreekOutputs {
        greek: greek_arrays(&g),
        greek_d: dg.each_ref().map(greek_arrays),
        phase,
        phase_d,
    }
}

/// The double-double solution at (x, m) with `n_terms` terms (those of the `f64` solution, so
/// that both expand the same series), the Greek coefficients up to `order` and the phase
/// matrix at `mu`.
pub fn greek_reference(
    x: f64,
    m: Complex<f64>,
    mu: &[f64],
    n_terms: usize,
    order: Option<usize>,
    normalization: Normalization,
) -> GreekOutputs<Dd> {
    let mu_dd: Vec<Dd> = mu.iter().map(|&v| Dd::from_f64(v)).collect();
    solve_greek(
        Dd::from_f64(x),
        Complex::new(Dd::from_f64(m.re), Dd::from_f64(m.im)),
        &mu_dd,
        Some(n_terms),
        order,
        normalization,
    )
}

/// The measures of the values at (x, m).
pub struct GreekValueMeasures {
    /// α₁, …, β₂, over all orders.
    pub greek: [Measure; 6],
    /// F₁₁, F₁₂, F₃₃, F₃₄ at each cosine.
    pub phase: Vec<[Measure; 4]>,
}

/// The measures of the derivatives at (x, m).
pub struct GreekDerivativeMeasures {
    /// [∂/∂x, ∂/∂Re m, ∂/∂Im m][α₁, …, β₂], over all orders.
    pub greek: [[Measure; 6]; 3],
    /// [∂/∂x, ∂/∂Re m, ∂/∂Im m][F₁₁, F₁₂, F₃₃, F₃₄], over all cosines.
    pub phase: [[Measure; 4]; 3],
}

/// F̄ = ½∫F₁₁ dμ, which is α₁ at l = 0 (1 normalized, Q_sca unnormalized).
fn mean_f11(d: &GreekOutputs<Dd>) -> f64 {
    d.greek[0][0].hi()
}

/// A = max over l of |α₁|.
fn alpha1_size(d: &GreekOutputs<Dd>) -> f64 {
    d.greek[0].iter().map(|v| v.hi().abs()).fold(0.0, max_nan)
}

/// Componentwise: x·|∂y/∂x| + |Re m|·|∂y/∂(Re m)| + |Im m|·|∂y/∂(Im m)|.
fn sensitivity(x: f64, m: Complex<f64>, d: [f64; 3]) -> f64 {
    x * d[0].abs() + m.re.abs() * d[1].abs() + m.im.abs() * d[2].abs()
}

/// The measures of the values at (x, m), from the double-double solution `d` there.
pub fn greek_value_measures(x: f64, m: Complex<f64>, d: &GreekOutputs<Dd>) -> GreekValueMeasures {
    let scale = alpha1_size(d);
    let greek = std::array::from_fn(|k| {
        let num = (0..d.greek[k].len())
            .map(|l| sensitivity(x, m, std::array::from_fn(|p| d.greek_d[p][k][l].hi())))
            .fold(0.0, max_nan);
        Measure {
            scale,
            condition: num / scale,
        }
    });
    let mean = mean_f11(d);
    let phase = (0..d.phase[0].len())
        .map(|j| {
            let scale = d.phase[0][j].hi().max(mean);
            std::array::from_fn(|k| Measure {
                scale,
                condition: sensitivity(x, m, std::array::from_fn(|p| d.phase_d[p][k][j].hi()))
                    / scale,
            })
        })
        .collect();
    GreekValueMeasures { greek, phase }
}

/// The measures of the derivatives at (x, m), from the double-double solution `d` there (with
/// `n_terms` terms, the Greek coefficients up to `order` and the phase matrix at `mu`) and
/// central differences of double-double derivatives, with the steps of
/// `common::mie::derivative_measures` (2⁻³⁰ x and 2⁻³⁰ |m|, one-sided in Im m at Im m = 0).
pub fn greek_derivative_measures(
    x: f64,
    m: Complex<f64>,
    mu: &[f64],
    n_terms: usize,
    order: usize,
    normalization: Normalization,
    d: &GreekOutputs<Dd>,
) -> GreekDerivativeMeasures {
    let h = 2f64.powi(-30);
    let (h_x, h_m) = (x * h, m.norm() * h);
    let one_sided = m.im < h_m;
    let at = |x: f64, re: f64, im: f64| {
        greek_reference(
            x,
            Complex::new(re, im),
            mu,
            n_terms,
            Some(order),
            normalization,
        )
    };
    // [+, −] in x, Re m and Im m, and the width of each difference.
    let steps = [
        [at(x + h_x, m.re, m.im), at(x - h_x, m.re, m.im)],
        [at(x, m.re + h_m, m.im), at(x, m.re - h_m, m.im)],
        [
            at(x, m.re, m.im + h_m),
            if one_sided {
                d.clone()
            } else {
                at(x, m.re, m.im - h_m)
            },
        ],
    ];
    let widths = [
        2.0 * h_x,
        2.0 * h_m,
        if one_sided { h_m } else { 2.0 * h_m },
    ];
    let fd = |q: usize, plus: Dd, minus: Dd| ((plus - minus) / Dd::from_f64(widths[q])).hi();
    let phase_of_x = 1.0 + m.norm() * x;
    let p_sizes = [x, m.norm(), m.norm()];
    let a = alpha1_size(d);
    let greek = std::array::from_fn(|p| {
        std::array::from_fn(|k| {
            let natural = a * phase_of_x / p_sizes[p];
            let values = &d.greek_d[p][k];
            let scale = values
                .iter()
                .map(|v| v.hi().abs().max(natural))
                .fold(0.0, max_nan);
            let num = (0..values.len())
                .map(|l| {
                    let g = std::array::from_fn(|q| {
                        fd(
                            q,
                            steps[q][0].greek_d[p][k][l],
                            steps[q][1].greek_d[p][k][l],
                        )
                    });
                    sensitivity(x, m, g)
                })
                .fold(0.0, max_nan);
            Measure {
                scale,
                condition: num / scale,
            }
        })
    });
    let mean = mean_f11(d);
    let phase = std::array::from_fn(|p| {
        std::array::from_fn(|k| {
            let scale = (0..mu.len())
                .map(|j| {
                    let natural = d.phase[0][j].hi().max(mean) * phase_of_x / p_sizes[p];
                    d.phase_d[p][k][j].hi().abs().max(natural)
                })
                .fold(0.0, max_nan);
            let num = (0..mu.len())
                .map(|j| {
                    let g = std::array::from_fn(|q| {
                        fd(
                            q,
                            steps[q][0].phase_d[p][k][j],
                            steps[q][1].phase_d[p][k][j],
                        )
                    });
                    sensitivity(x, m, g)
                })
                .fold(0.0, max_nan);
            Measure {
                scale,
                condition: num / scale,
            }
        })
    });
    GreekDerivativeMeasures { greek, phase }
}
