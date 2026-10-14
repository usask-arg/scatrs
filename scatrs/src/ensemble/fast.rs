//! The phase matrix of a panel's spheres at a projector band's Gauss nodes, as matrix products
//! against tables of πₙ and τₙ that every sphere shares.
//!
//! S₁ = Σ wₙ(aₙπₙ + bₙτₙ) and S₂ = Σ wₙ(aₙτₙ + bₙπₙ), wₙ = (2n + 1)/(n(n + 1)). Since
//! πₙ(−μ) = (−1)ⁿ⁺¹πₙ(μ) and τₙ(−μ) = (−1)ⁿτₙ(μ), with G_E holding wₙπₙ for odd n and wₙτₙ for
//! even n, G_O the other two, uₙ = aₙ (odd n) or bₙ (even n) and vₙ the other:
//! S₁(±μ) = G_E·u ± G_O·v and S₂(±μ) = G_E·v ± G_O·u at the non-negative nodes μ. The Gauss
//! nodes are exactly symmetric, so the products at h = ⌈n_q/2⌉ nodes give all n_q.

use faer::{Accum, Mat, Par};
use num_complex::Complex;

use crate::mie::{MieSolution, element_derivatives, elements, for_each_order};

/// G_E and G_O of one band (h × N, one column per order).
pub(crate) struct BandTables {
    q: usize,
    ge: Mat<f64>,
    go: Mat<f64>,
}

impl BandTables {
    /// The tables of orders 1..=`n` at the non-negative half of the ascending, symmetric
    /// `nodes`.
    pub(crate) fn new(nodes: &[f64], n: usize) -> Self {
        let q = nodes.len();
        let lo = q / 2;
        let h = q - lo;
        let mut ge = Mat::<f64>::zeros(h, n);
        let mut go = Mat::<f64>::zeros(h, n);
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
/// parts; 4 more with derivatives: du/dm, dv/dm) and the products.
pub(crate) struct PanelProducts {
    per: usize,
    x: Mat<f64>,
    ce: Mat<f64>,
    co: Mat<f64>,
}

impl PanelProducts {
    /// Room for `spheres` spheres of at most `n` orders, at `h` half nodes.
    pub(crate) fn new(tables: &BandTables, spheres: usize, derivatives: bool) -> Self {
        let per = if derivatives { 8 } else { 4 };
        let (h, n) = (tables.ge.nrows(), tables.orders());
        Self {
            per,
            x: Mat::zeros(n, per * spheres),
            ce: Mat::zeros(h, per * spheres),
            co: Mat::zeros(h, per * spheres),
        }
    }

    /// Whether this fits `tables` and `spheres` spheres.
    pub(crate) fn fits(&self, tables: &BandTables, spheres: usize, derivatives: bool) -> bool {
        self.x.nrows() == tables.orders()
            && self.ce.nrows() == tables.ge.nrows()
            && self.per == if derivatives { 8 } else { 4 }
            && self.x.ncols() == self.per * spheres
    }

    /// Sphere `k`'s coefficients (zero beyond its number of terms); with derivatives, also
    /// daₙ/dm and dbₙ/dm.
    pub(crate) fn set(&mut self, k: usize, s: &MieSolution<'_, f64>) {
        let (a, b) = (s.a(), s.b());
        let d = if self.per == 8 { s.derivatives() } else { None };
        let base = k * self.per;
        let n = self.x.nrows();
        for row in 0..n {
            let (u, v, du, dv) = if row < a.len() {
                let (dan, dbn) = d
                    .as_ref()
                    .map_or((Complex::default(), Complex::default()), |d| {
                        (d.da_dm[row], d.db_dm[row])
                    });
                // Order row + 1: u = a for odd orders (even rows), b for even ones.
                if row % 2 == 0 {
                    (a[row], b[row], dan, dbn)
                } else {
                    (b[row], a[row], dbn, dan)
                }
            } else {
                Default::default()
            };
            self.x[(row, base)] = u.re;
            self.x[(row, base + 1)] = u.im;
            self.x[(row, base + 2)] = v.re;
            self.x[(row, base + 3)] = v.im;
            if self.per == 8 {
                self.x[(row, base + 4)] = du.re;
                self.x[(row, base + 5)] = du.im;
                self.x[(row, base + 6)] = dv.re;
                self.x[(row, base + 7)] = dv.im;
            }
        }
    }

    /// The products for every sphere set so far.
    pub(crate) fn multiply(&mut self, tables: &BandTables) {
        faer::linalg::matmul::matmul(
            self.ce.as_mut(),
            Accum::Replace,
            tables.ge.as_ref(),
            self.x.as_ref(),
            1.0,
            Par::Seq,
        );
        faer::linalg::matmul::matmul(
            self.co.as_mut(),
            Accum::Replace,
            tables.go.as_ref(),
            self.x.as_ref(),
            1.0,
            Par::Seq,
        );
    }

    /// Sphere `k`'s phase matrix times `scale` at every node of the band (F₁₁, F₁₂, F₃₃, F₃₄,
    /// each n_q values, into `out`), and with derivatives its derivatives with respect to Re m
    /// and Im m into `d`.
    pub(crate) fn phase(
        &self,
        tables: &BandTables,
        k: usize,
        scale: f64,
        out: &mut [f64],
        mut d: Option<[&mut [f64]; 2]>,
    ) {
        let q = tables.q;
        let lo = q / 2;
        let base = k * self.per;
        let c = |m: &Mat<f64>, i: usize, col: usize| Complex::new(m[(i, col)], m[(i, col + 1)]);
        let i1 = Complex::new(0.0, 1.0);
        for i in 0..self.ce.nrows() {
            let j = lo + i;
            let (eu, ev) = (c(&self.ce, i, base), c(&self.ce, i, base + 2));
            let (ou, ov) = (c(&self.co, i, base), c(&self.co, i, base + 2));
            let derivs = (self.per == 8).then(|| {
                (
                    c(&self.ce, i, base + 4),
                    c(&self.ce, i, base + 6),
                    c(&self.co, i, base + 4),
                    c(&self.co, i, base + 6),
                )
            });
            let mut write = |node: usize, sign: f64| {
                let (s1, s2) = (eu + ov * sign, ev + ou * sign);
                let f = elements(s1, s2);
                for (e, v) in f.into_iter().enumerate() {
                    out[e * q + node] = scale * v;
                }
                if let (Some([dre, dim]), Some((deu, dev, dou, dov))) = (d.as_mut(), derivs) {
                    let (d1, d2) = (deu + dov * sign, dev + dou * sign);
                    let fr = element_derivatives(s1, s2, d1, d2);
                    let fi = element_derivatives(s1, s2, i1 * d1, i1 * d2);
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
                let mut p = PanelProducts::new(&tables, cases.len(), derivatives);
                for (k, &(x, m)) in cases.iter().enumerate() {
                    let s = if derivatives {
                        mie.compute_with_derivatives(x, m, &opts).unwrap()
                    } else {
                        mie.compute(x, m, &opts).unwrap()
                    };
                    p.set(k, &s);
                }
                p.multiply(&tables);
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
                    let d = derivatives.then_some([&mut gre[..], &mut gim[..]]);
                    p.phase(&tables, k, scale, &mut got, d);
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
