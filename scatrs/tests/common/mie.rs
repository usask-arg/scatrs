//! Shared by the Mie tests: solving one sphere in any scalar type.

use num_complex::Complex;
use scatrs::Real;
use scatrs::mie::{AmplitudeBuffers, Efficiencies, Mie, MieOptions};

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
