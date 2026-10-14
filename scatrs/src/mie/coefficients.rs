//! The coefficients aₙ, bₙ of a particle, from which every output of [`MieSolution`] follows:
//! the efficiencies, amplitudes, phase matrix and Greek coefficients are implemented once on a
//! [`Coefficients`] view, and their derivatives on a [`Group`] of coefficient derivatives.
//!
//! A sphere has one group, with respect to x and m.

use num_complex::Complex;

use super::{CoefficientDerivatives, MieSolution};
use crate::scalar::Real;

/// The coefficients aₙ, bₙ (n = 1, …, N at index n − 1) of a particle of size parameter `x`,
/// which scales every output, and the absorption sum (x²/2)·Q_abs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Coefficients<'a, T> {
    pub(crate) x: T,
    pub(crate) a: &'a [Complex<T>],
    pub(crate) b: &'a [Complex<T>],
    pub(crate) abs_sum: T,
}

/// The derivatives of the coefficients with respect to a real size parameter and a refractive
/// index, in which they are holomorphic, and those of the absorption sum with respect to the
/// size parameter, Re m and Im m. `outer` says whether the size parameter is
/// [`Coefficients::x`], which scales the outputs (a sphere's, or a layered sphere's outer
/// layer's), so that the derivatives include those of the scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Group<'a, T> {
    pub(crate) d: CoefficientDerivatives<'a, T>,
    pub(crate) abs_sum: [T; 3],
    pub(crate) outer: bool,
}

impl<T: Real> Coefficients<'_, T> {
    /// The number of terms N.
    pub(crate) fn n_terms(&self) -> usize {
        self.a.len()
    }
}

impl<'a, T: Real> MieSolution<'a, T> {
    /// The sphere's coefficients.
    pub(crate) fn coefficients(&self) -> Coefficients<'a, T> {
        let s = self.state;
        Coefficients {
            x: s.x,
            a: &s.a,
            b: &s.b,
            abs_sum: s.abs_sum,
        }
    }

    /// The derivatives with respect to x and m, if computed.
    pub(crate) fn group(&self) -> Option<Group<'a, T>> {
        self.derivatives().map(|d| Group {
            d,
            abs_sum: self.state.abs_sum_jacobian,
            outer: true,
        })
    }
}
