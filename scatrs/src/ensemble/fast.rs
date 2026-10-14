//! The phase matrix of a panel's spheres at a projector band's Gauss nodes, as matrix products
//! against tables of πₙ and τₙ that every sphere shares.
//!
//! S₁ = Σ wₙ(aₙπₙ + bₙτₙ) and S₂ = Σ wₙ(aₙτₙ + bₙπₙ), wₙ = (2n + 1)/(n(n + 1)). Since
//! πₙ(−μ) = (−1)ⁿ⁺¹πₙ(μ) and τₙ(−μ) = (−1)ⁿτₙ(μ), with G_E holding wₙπₙ for odd n and wₙτₙ for
//! even n, G_O the other two, uₙ = aₙ (odd n) or bₙ (even n) and vₙ the other:
//! S₁(±μ) = G_E·u ± G_O·v and S₂(±μ) = G_E·v ± G_O·u at the non-negative nodes μ. The Gauss
//! nodes are exactly symmetric, so the products at h = ⌈n_q/2⌉ nodes give all n_q.

use super::matrix::{Mat, matmul, product};
use num_complex::Complex;

use crate::Result;
use crate::mie::{
    Coefficients as MieCoefficients, GreekProjector, Group, element_derivatives, elements,
    for_each_order,
};
use crate::special::wigner_d;

/// G_E and G_O of one band (h × N, one column per order).
pub(crate) struct BandTables {
    q: usize,
    ge: Mat,
    go: Mat,
}

impl BandTables {
    /// The tables of orders 1..=`n` at the non-negative half of the ascending, symmetric
    /// `nodes`.
    pub(crate) fn new(nodes: &[f64], n: usize) -> Self {
        let q = nodes.len();
        let lo = q / 2;
        let h = q - lo;
        let mut ge = Mat::zeros(h, n);
        let mut go = Mat::zeros(h, n);
        for (i, &mu) in nodes[lo..].iter().enumerate() {
            for_each_order(mu, n, |k, pi, tau| {
                let (e, o) = if k % 2 == 1 { (pi, tau) } else { (tau, pi) };
                ge[(i, k - 1)] = e;
                go[(i, k - 1)] = o;
            });
        }
        Self { q, ge, go }
    }

    fn orders(&self) -> usize {
        self.ge.ncols()
    }
}

/// The right-hand sides of a panel's spheres (4 columns each: u and v, real and imaginary
/// parts; 4 more per derivative group: du/dm, dv/dm) and the products.
pub(crate) struct PanelProducts {
    per: usize,
    x: Mat,
    ce: Mat,
    co: Mat,
}

impl PanelProducts {
    /// Room for `spheres` spheres with `groups` derivative groups each, at the band's orders
    /// and half nodes.
    pub(crate) fn new(tables: &BandTables, spheres: usize, groups: usize) -> Self {
        let per = 4 + 4 * groups;
        let (h, n) = (tables.ge.nrows(), tables.orders());
        Self {
            per,
            x: Mat::zeros(n, per * spheres),
            ce: Mat::zeros(h, per * spheres),
            co: Mat::zeros(h, per * spheres),
        }
    }

    /// Whether this fits `tables` and `spheres` spheres with `groups` derivative groups.
    pub(crate) fn fits(&self, tables: &BandTables, spheres: usize, groups: usize) -> bool {
        self.x.nrows() == tables.orders()
            && self.ce.nrows() == tables.ge.nrows()
            && self.per == 4 + 4 * groups
            && self.x.ncols() == self.per * spheres
    }

    /// The number of derivative groups.
    fn groups(&self) -> usize {
        (self.per - 4) / 4
    }

    /// Sphere `k`'s coefficients (zero beyond its number of terms) and daₙ/dm, dbₙ/dm of each
    /// of the first `self.groups()` of `groups`.
    pub(crate) fn set(
        &mut self,
        k: usize,
        c: &MieCoefficients<'_, f64>,
        groups: &[Group<'_, f64>],
    ) {
        let (a, b) = (c.a, c.b);
        let groups = &groups[..self.groups()];
        let base = k * self.per;
        let n = self.x.nrows();
        let mut put = |row: usize, col: usize, u: Complex<f64>, v: Complex<f64>| {
            self.x[(row, col)] = u.re;
            self.x[(row, col + 1)] = u.im;
            self.x[(row, col + 2)] = v.re;
            self.x[(row, col + 3)] = v.im;
        };
        // Order row + 1: u = a for odd orders (even rows), b for even ones.
        let uv = |row: usize, a: Complex<f64>, b: Complex<f64>| {
            if row % 2 == 0 { (a, b) } else { (b, a) }
        };
        for row in 0..n {
            if row < a.len() {
                let (u, v) = uv(row, a[row], b[row]);
                put(row, base, u, v);
                for (i, g) in groups.iter().enumerate() {
                    let (du, dv) = uv(row, g.d.da_dm[row], g.d.db_dm[row]);
                    put(row, base + 4 + 4 * i, du, dv);
                }
            } else {
                for i in 0..=groups.len() {
                    put(row, base + 4 * i, Complex::default(), Complex::default());
                }
            }
        }
    }

    /// The products for every sphere set so far, whose coefficients vanish beyond `terms`
    /// orders (the band's tables may have more).
    pub(crate) fn multiply(&mut self, tables: &BandTables, terms: usize) {
        let terms = terms.min(tables.orders());
        matmul(&mut self.ce, &tables.ge, &self.x, terms);
        matmul(&mut self.co, &tables.go, &self.x, terms);
    }

    /// Sphere `k`'s phase matrix times `scale` at every node of the band (F₁₁, F₁₂, F₃₃, F₃₄,
    /// each n_q values, into `out`), and its derivatives with respect to Re m and Im m of each
    /// group into `d` (Re m, Im m of the first group, then of the next; empty for none).
    pub(crate) fn phase(
        &self,
        tables: &BandTables,
        k: usize,
        scale: f64,
        out: &mut [f64],
        d: &mut [&mut [f64]],
    ) {
        let q = tables.q;
        let lo = q / 2;
        let base = k * self.per;
        let c = |m: &Mat, i: usize, col: usize| Complex::new(m[(i, col)], m[(i, col + 1)]);
        let i1 = Complex::new(0.0, 1.0);
        for i in 0..self.ce.nrows() {
            let j = lo + i;
            let (eu, ev) = (c(&self.ce, i, base), c(&self.ce, i, base + 2));
            let (ou, ov) = (c(&self.co, i, base), c(&self.co, i, base + 2));
            let mut write = |node: usize, sign: f64| {
                let (s1, s2) = (eu + ov * sign, ev + ou * sign);
                let f = elements(s1, s2);
                for (e, v) in f.into_iter().enumerate() {
                    out[e * q + node] = scale * v;
                }
                for (g, pair) in d.chunks_exact_mut(2).enumerate() {
                    let col = base + 4 + 4 * g;
                    let (deu, dev) = (c(&self.ce, i, col), c(&self.ce, i, col + 2));
                    let (dou, dov) = (c(&self.co, i, col), c(&self.co, i, col + 2));
                    let (d1, d2) = (deu + dov * sign, dev + dou * sign);
                    let fr = element_derivatives(s1, s2, d1, d2);
                    let fi = element_derivatives(s1, s2, i1 * d1, i1 * d2);
                    let [dre, dim] = pair else { unreachable!() };
                    for e in 0..4 {
                        dre[e * q + node] = scale * fr[e];
                        dim[e * q + node] = scale * fi[e];
                    }
                }
            };
            write(j, 1.0);
            let mirror = q - 1 - j;
            if mirror != j {
                write(mirror, -1.0);
            }
        }
    }
}

/// A band's projection onto the Greek coefficients as matrix products: the matrices
/// (2l + 1)/2·wₖ·dₗ(μₖ) (n_q × (L + 1)) of d₀₀, d₂₂, d₂,₋₂ and d₀₂, as `GreekProjector::project`
/// sums them, and their absolute values for the bounds of `GreekProjector::project_bounds`.
pub(crate) struct BandProjection {
    d: [Mat; 4],
    abs: [Mat; 4],
}

/// The six coefficient arrays of many tables, a row per table: α₁, α₂, α₃, α₄, β₁, β₂.
pub(crate) type Coefficients = [Mat; 6];

impl BandProjection {
    pub(crate) fn new(projector: &GreekProjector<f64>) -> Result<Self> {
        let (nodes, weights) = (projector.nodes(), projector.weights());
        let len = projector.order() + 1;
        let mut d: [Mat; 4] = core::array::from_fn(|_| Mat::zeros(nodes.len(), len));
        let mut buf = vec![0.0; len];
        for (k, (&mu, &w)) in nodes.iter().zip(weights).enumerate() {
            for (i, (m, n)) in [(0, 0), (2, 2), (2, -2), (0, 2)].into_iter().enumerate() {
                wigner_d(m, n, mu, &mut buf)?;
                for (l, &v) in buf.iter().enumerate() {
                    d[i][(k, l)] = (2 * l + 1) as f64 * 0.5 * w * v;
                }
            }
        }
        let abs = core::array::from_fn(|i| d[i].map(f64::abs));
        Ok(Self { d, abs })
    }

    /// The coefficients of the tables, whose elements are the rows of `f` (F₁₁, F₁₂, F₃₃, F₃₄,
    /// each tables × n_q).
    pub(crate) fn project(&self, f: &[Mat; 4]) -> Coefficients {
        let [f11, f12, f33, f34] = f;
        let sum = f11.zip_map(f33, |a, b| a + b);
        let diff = f11.zip_map(f33, |a, b| a - b);
        let [d00, d22, d2m2, d02] = &self.d;
        let plus = product(&sum, d22);
        let minus = product(&diff, d2m2);
        [
            product(f11, d00),
            plus.zip_map(&minus, |p, m| (p + m) * 0.5),
            plus.zip_map(&minus, |p, m| (p - m) * 0.5),
            product(f33, d00),
            product(f12, d02),
            product(f34, d02),
        ]
    }

    /// Bounds on the coefficients' changes for tables of bounds on the elements' changes (as
    /// `project`): α₂ and α₃ share the half sum of the bounds of α₂ ± α₃.
    pub(crate) fn bounds(&self, e: &[Mat; 4]) -> Coefficients {
        let [e11, e12, e33, e34] = e;
        let sum = e11.zip_map(e33, |a, b| a + b);
        let [d00, d22, d2m2, d02] = &self.abs;
        let both = product(&sum, d22).zip_map(&product(&sum, d2m2), |p, m| (p + m) * 0.5);
        [
            product(e11, d00),
            both.clone(),
            both,
            product(e33, d00),
            product(e12, d02),
            product(e34, d02),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mie::{Mie, MieOptions, Normalization, PhaseBuffers};
    use crate::special::gauss_legendre;

    fn bufs(t: &mut [f64], q: usize) -> PhaseBuffers<'_, f64> {
        let (f11, rest) = t.split_at_mut(q);
        let (f12, rest) = rest.split_at_mut(q);
        let (f33, f34) = rest.split_at_mut(q);
        PhaseBuffers { f11, f12, f33, f34 }
    }

    /// The products give each sphere's phase matrix and its m-derivatives at every node, as
    /// the per-sphere evaluation does, for spheres of fewer orders than the band too.
    #[test]
    fn products_match_the_per_sphere_phase_matrix() {
        let cases = [
            (0.3, Complex::new(1.33, 0.0)),
            (7.0, Complex::new(1.5, 0.01)),
            (40.0, Complex::new(1.45, 1e-4)),
            (95.0, Complex::new(2.0, 0.5)),
        ];
        let opts = MieOptions::new();
        let mut mie = Mie::<f64>::new();
        let n_max = cases
            .iter()
            .map(|&(x, m)| crate::mie::n_terms(x, m, &opts).unwrap())
            .max()
            .unwrap();
        for q in [n_max + 7, n_max + 8] {
            let nodes = gauss_legendre::<f64>(q).unwrap().nodes;
            let tables = BandTables::new(&nodes, n_max);
            for derivatives in [false, true] {
                let mut p = PanelProducts::new(&tables, cases.len(), usize::from(derivatives));
                for (k, &(x, m)) in cases.iter().enumerate() {
                    let s = if derivatives {
                        mie.compute_with_derivatives(x, m, &opts).unwrap()
                    } else {
                        mie.compute(x, m, &opts).unwrap()
                    };
                    let groups: Vec<_> = s.group().into_iter().collect();
                    p.set(k, &s.coefficients(), &groups);
                }
                p.multiply(&tables, n_max);
                for (k, &(x, m)) in cases.iter().enumerate() {
                    let s = mie.compute_with_derivatives(x, m, &opts).unwrap();
                    let mut want = vec![vec![0.0; 4 * q]; 4];
                    {
                        let [v, dx, dre, dim] = &mut want[..] else {
                            unreachable!()
                        };
                        s.phase_matrix_with_derivatives(
                            &nodes,
                            Normalization::Unnormalized,
                            bufs(v, q),
                            [bufs(dx, q), bufs(dre, q), bufs(dim, q)],
                        )
                        .unwrap();
                    }
                    let scale = 4.0 / (x * x);
                    let mut got = vec![0.0; 4 * q];
                    let (mut gre, mut gim) = (vec![0.0; 4 * q], vec![0.0; 4 * q]);
                    let mut d: Vec<&mut [f64]> = if derivatives {
                        vec![&mut gre[..], &mut gim[..]]
                    } else {
                        Vec::new()
                    };
                    p.phase(&tables, k, scale, &mut got, &mut d);
                    let check = |got: &[f64], want: &[f64], what: &str| {
                        let size = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                        for (j, (g, w)) in got.iter().zip(want).enumerate() {
                            assert!(
                                (g - w).abs() <= 1e-12 * size,
                                "x {x}, {what} [{j}]: {g} vs {w}"
                            );
                        }
                    };
                    check(&got, &want[0], "values");
                    if derivatives {
                        check(&gre, &want[2], "d/dRe m");
                        check(&gim, &want[3], "d/dIm m");
                    }
                }
            }
        }
    }
}
