//! The particles of a component as the engine solves them: at each wavelength, a particle of
//! size parameter x gives its coefficients and, on request, their derivative groups.
//!
//! The per-sphere quantities are differentiated along the particle's directions: the real and
//! imaginary parts of the refractive index of each group, in order (Re m and Im m for a
//! sphere; of each layer for a layered sphere), then the particle's real parameters (the radius
//! fraction of each layer of a layered sphere but the last).

use num_complex::Complex;

use super::Particle;
use crate::Result;
use crate::layered::{self, Layered};
use crate::mie::{Coefficients, Efficiencies, EfficiencyJacobian, Group, Mie, MieOptions, n_terms};

impl Particle {
    /// The particle at wavelength `i`.
    pub(super) fn optics(&self, i: usize) -> Optics<'_> {
        match self {
            Particle::Sphere { refractive_index } => Optics::Sphere(refractive_index[i]),
            Particle::LayeredSphere {
                radius_fractions,
                refractive_index,
            } => Optics::Layered {
                fractions: radius_fractions,
                m: refractive_index,
                i,
            },
        }
    }

    /// The number of wavelengths it is given at (of the core, for a layered sphere; see
    /// [`validate`](Self::validate)).
    pub(super) fn wavelengths(&self) -> usize {
        match self {
            Particle::Sphere { refractive_index } => refractive_index.len(),
            Particle::LayeredSphere {
                refractive_index, ..
            } => refractive_index.first().map_or(0, Vec::len),
        }
    }

    /// Checks the particle's own description: a layered sphere's layers and fractions.
    pub(super) fn validate(&self) -> core::result::Result<(), String> {
        let Particle::LayeredSphere {
            radius_fractions: f,
            refractive_index: m,
        } = self
        else {
            return Ok(());
        };
        let layers = m.len();
        if layers == 0 || layers > layered::MAX_LAYERS {
            return Err(format!(
                "a layered sphere needs 1 to {} layers, got {layers}",
                layered::MAX_LAYERS
            ));
        }
        if f.len() + 1 != layers {
            return Err(format!(
                "a layered sphere of {layers} layers needs {} radius fractions, got {}",
                layers - 1,
                f.len()
            ));
        }
        if let Some((l, v)) = f
            .iter()
            .enumerate()
            .find(|&(_, v)| !(v.is_finite() && *v > 0.0 && *v <= 1.0))
        {
            return Err(format!("radius fraction {l} must be in (0, 1], got {v}"));
        }
        if let Some(l) = (1..f.len()).find(|&l| f[l] < f[l - 1]) {
            return Err(format!(
                "the radius fractions must be nondecreasing from the core out, but fraction {l} \
                 is {} < {}",
                f[l],
                f[l - 1]
            ));
        }
        if let Some(l) = (1..layers).find(|&l| m[l].len() != m[0].len()) {
            return Err(format!(
                "layer {l} has {} refractive indices and the core {}: one per wavelength",
                m[l].len(),
                m[0].len()
            ));
        }
        Ok(())
    }

    /// The number of derivative groups (a size parameter and a refractive index each).
    pub(super) fn groups(&self) -> usize {
        match self {
            Particle::Sphere { .. } => 1,
            Particle::LayeredSphere {
                refractive_index, ..
            } => refractive_index.len(),
        }
    }

    /// The number of real directions after the groups' (the radius fractions).
    pub(super) fn reals(&self) -> usize {
        match self {
            Particle::Sphere { .. } => 0,
            Particle::LayeredSphere {
                radius_fractions, ..
            } => radius_fractions.len(),
        }
    }

    /// The number of derivative directions: two per group, then the real ones.
    pub(super) fn directions(&self) -> usize {
        2 * self.groups() + self.reals()
    }
}

/// A component's particle at one wavelength.
#[derive(Clone, Copy, Debug)]
pub(super) enum Optics<'a> {
    /// A homogeneous sphere of refractive index m.
    Sphere(Complex<f64>),
    /// A layered sphere: its radius fractions, and its layers' refractive indices at
    /// wavelength `i` (`m[l][i]`).
    Layered {
        fractions: &'a [f64],
        m: &'a [Vec<Complex<f64>>],
        i: usize,
    },
}

impl Optics<'_> {
    /// The number of terms of the particle of size parameter `x`, after the input checks of
    /// [`Solver::solve`].
    pub(super) fn n_terms(self, x: f64, opts: &MieOptions) -> Result<usize> {
        match self {
            Optics::Sphere(m) => n_terms(x, m, opts),
            Optics::Layered { .. } => {
                let (mut xs, mut ms) = (Vec::new(), Vec::new());
                self.layers(x, &mut xs, &mut ms);
                layered::n_terms(&xs, &ms, opts)
            }
        }
    }

    /// A layered sphere's interface size parameters at outer size parameter `x` (x_l = f_l·x,
    /// and x itself last) and its layers' indices, into `xs` and `ms`.
    fn layers(self, x: f64, xs: &mut Vec<f64>, ms: &mut Vec<Complex<f64>>) {
        let Optics::Layered { fractions, m, i } = self else {
            unreachable!("a layered sphere")
        };
        xs.clear();
        xs.extend(fractions.iter().map(|f| f * x));
        xs.push(x);
        ms.clear();
        ms.extend(m.iter().map(|layer| layer[i]));
    }
}

/// One particle's solution: its coefficients and, if computed, their derivative groups, with
/// the radius fractions of a layered sphere (empty for a sphere).
pub(super) struct Solved<'a> {
    pub(super) c: Coefficients<'a, f64>,
    pub(super) groups: Vec<Group<'a, f64>>,
    pub(super) fractions: &'a [f64],
}

/// `e` times `s`.
fn scaled(e: &Efficiencies<f64>, s: f64) -> Efficiencies<f64> {
    Efficiencies {
        q_ext: s * e.q_ext,
        q_sca: s * e.q_sca,
        q_abs: s * e.q_abs,
        q_back: s * e.q_back,
        g: s * e.g,
    }
}

impl Solved<'_> {
    /// The derivative of the efficiencies along direction `k`, from the groups' Jacobians
    /// `jacs`: Re m or Im m of group k/2 for k below twice the number of groups, and above,
    /// the radius fraction j = k − 2·groups, x_L·∂/∂x_j at a fixed outer size x_L.
    pub(super) fn direction(
        &self,
        jacs: &[EfficiencyJacobian<f64>],
        k: usize,
    ) -> Efficiencies<f64> {
        let groups = self.groups.len();
        if k < 2 * groups {
            let j = &jacs[k / 2];
            if k % 2 == 0 { j.m_re } else { j.m_im }
        } else {
            scaled(&jacs[k - 2 * groups].x, self.c.x)
        }
    }

    /// The derivative of the efficiencies with respect to the outer size parameter, at fixed
    /// radius fractions: Σ_l f_l·∂/∂x_l (∂/∂x for a sphere).
    pub(super) fn size_derivative(&self, jacs: &[EfficiencyJacobian<f64>]) -> Efficiencies<f64> {
        let (outer, inner) = jacs.split_last().expect("a group");
        let mut d = outer.x;
        for (f, j) in self.fractions.iter().zip(inner) {
            let e = scaled(&j.x, *f);
            d.q_ext += e.q_ext;
            d.q_sca += e.q_sca;
            d.q_abs += e.q_abs;
            d.q_back += e.q_back;
            d.g += e.g;
        }
        d
    }
}

/// A per-thread workspace for solving particles.
pub(super) struct Solver {
    mie: Mie<f64>,
    layered: Layered<f64>,
    xs: Vec<f64>,
    ms: Vec<Complex<f64>>,
}

impl Solver {
    pub(super) fn new() -> Self {
        Self {
            mie: Mie::new(),
            layered: Layered::new(),
            xs: Vec::new(),
            ms: Vec::new(),
        }
    }

    /// The particle `optics` of (outer) size parameter `x`, with its derivative groups if
    /// `derivatives`.
    pub(super) fn solve<'a>(
        &'a mut self,
        optics: Optics<'a>,
        x: f64,
        opts: &MieOptions,
        derivatives: bool,
    ) -> Result<Solved<'a>> {
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
                    fractions: &[],
                })
            }
            Optics::Layered { fractions, .. } => {
                optics.layers(x, &mut self.xs, &mut self.ms);
                let s = if derivatives {
                    self.layered
                        .compute_with_derivatives(&self.xs, &self.ms, opts)
                } else {
                    self.layered.compute(&self.xs, &self.ms, opts)
                }?;
                Ok(Solved {
                    c: s.coefficients(),
                    groups: (0..self.xs.len()).filter_map(|l| s.group(l)).collect(),
                    fractions,
                })
            }
        }
    }
}
