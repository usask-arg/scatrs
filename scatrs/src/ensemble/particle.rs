//! The particles of a component as the engine solves them: at each wavelength, a particle of
//! size parameter x gives its coefficients and, on request, their derivative groups.
//!
//! The per-sphere quantities are differentiated along the particle's directions: the real and
//! imaginary parts of the refractive index of each group, in order (Re m and Im m for a
//! sphere).

use num_complex::Complex;

use super::Particle;
use crate::Result;
use crate::mie::{Coefficients, Group, Mie, MieOptions, n_terms};

impl Particle {
    /// The particle at wavelength `i`.
    pub(super) fn optics(&self, i: usize) -> Optics {
        match self {
            Particle::Sphere { refractive_index } => Optics::Sphere(refractive_index[i]),
        }
    }

    /// The number of wavelengths it is given at.
    pub(super) fn wavelengths(&self) -> usize {
        match self {
            Particle::Sphere { refractive_index } => refractive_index.len(),
        }
    }

    /// The number of derivative directions: two per group.
    pub(super) fn directions(&self) -> usize {
        match self {
            Particle::Sphere { .. } => 2,
        }
    }
}

/// A component's particle at one wavelength.
#[derive(Clone, Copy, Debug)]
pub(super) enum Optics {
    /// A homogeneous sphere of refractive index m.
    Sphere(Complex<f64>),
}

impl Optics {
    /// The number of terms of the particle of size parameter `x`, after the input checks of
    /// [`Solver::solve`].
    pub(super) fn n_terms(self, x: f64, opts: &MieOptions) -> Result<usize> {
        match self {
            Optics::Sphere(m) => n_terms(x, m, opts),
        }
    }
}

/// One particle's solution: its coefficients and, if computed, their derivative groups.
pub(super) struct Solved<'a> {
    pub(super) c: Coefficients<'a, f64>,
    pub(super) groups: Vec<Group<'a, f64>>,
}

/// A per-thread workspace for solving particles.
pub(super) struct Solver {
    mie: Mie<f64>,
}

impl Solver {
    pub(super) fn new() -> Self {
        Self { mie: Mie::new() }
    }

    /// The particle `optics` of size parameter `x`, with its derivative groups if
    /// `derivatives`.
    pub(super) fn solve(
        &mut self,
        optics: Optics,
        x: f64,
        opts: &MieOptions,
        derivatives: bool,
    ) -> Result<Solved<'_>> {
        match optics {
            Optics::Sphere(m) => {
                let s = if derivatives {
                    self.mie.compute_with_derivatives(x, m, opts)
                } else {
                    self.mie.compute(x, m, opts)
                }?;
                Ok(Solved {
                    c: s.coefficients(),
                    groups: s.group().into_iter().collect(),
                })
            }
        }
    }
}
