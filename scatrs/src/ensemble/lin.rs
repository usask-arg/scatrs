//! Values with derivatives and error estimates, propagated to first order through the
//! ratios that turn integrals into ensemble outputs.

/// A value, its error estimate, its derivatives with respect to the requested parameters,
/// and their error estimates.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Lin {
    pub(crate) v: f64,
    pub(crate) e: f64,
    pub(crate) d: Vec<f64>,
    pub(crate) de: Vec<f64>,
}

impl Lin {
    /// An exact value without derivatives (`n` parameters).
    pub(crate) fn constant(v: f64, n: usize) -> Self {
        Self {
            v,
            e: 0.0,
            d: vec![0.0; n],
            de: vec![0.0; n],
        }
    }

    /// `self`/`b`: the quotient rule, and the first-order error bounds
    /// e(a/b) ≈ (e_a + |a/b| e_b)/|b| and
    /// e(∂(a/b)) ≈ (e_∂a + |a/b| e_∂b + |∂(a/b)| e_b + e(a/b) |∂b|)/|b|.
    pub(crate) fn div(&self, b: &Self) -> Self {
        let v = self.v / b.v;
        let e = (self.e + v.abs() * b.e) / b.v.abs();
        let d: Vec<f64> = self
            .d
            .iter()
            .zip(&b.d)
            .map(|(da, db)| (da - v * db) / b.v)
            .collect();
        let de = (0..d.len())
            .map(|k| {
                (self.de[k] + v.abs() * b.de[k] + d[k].abs() * b.e + e * b.d[k].abs()) / b.v.abs()
            })
            .collect();
        Self { v, e, d, de }
    }

    /// Σ cᵢ·`terms`ᵢ for exact constants cᵢ, the error estimates added as bounds (`n`
    /// parameters).
    pub(crate) fn combine(terms: &[(f64, &Self)], n: usize) -> Self {
        let mut out = Self::constant(0.0, n);
        for &(c, l) in terms {
            out.v += c * l.v;
            out.e += c.abs() * l.e;
            for k in 0..n {
                out.d[k] += c * l.d[k];
                out.de[k] += c.abs() * l.de[k];
            }
        }
        out
    }

    /// `c`·`self` for an exact constant `c`.
    pub(crate) fn scale(&self, c: f64) -> Self {
        Self {
            v: c * self.v,
            e: c.abs() * self.e,
            d: self.d.iter().map(|x| c * x).collect(),
            de: self.de.iter().map(|x| c.abs() * x).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quotient_rule_matches_finite_differences() {
        // a(θ) = 3 + 2θ, b(θ) = 5 − θ at θ = 0.
        let a = Lin {
            v: 3.0,
            e: 1e-3,
            d: vec![2.0],
            de: vec![1e-4],
        };
        let b = Lin {
            v: 5.0,
            e: 2e-3,
            d: vec![-1.0],
            de: vec![0.0],
        };
        let q = a.div(&b);
        let h = 1e-6;
        let fd = ((3.0 + 2.0 * h) / (5.0 - h) - (3.0 - 2.0 * h) / (5.0 + h)) / (2.0 * h);
        assert!((q.d[0] - fd).abs() < 1e-9);
        assert!((q.v - 0.6).abs() < 1e-15);
        // Perturbing a and b by their errors moves a/b by the bound, to first order.
        let worst = ((3.0f64 + 1e-3) / (5.0 - 2e-3) - 0.6).abs();
        assert!(worst <= q.e * (1.0 + 1e-3) && worst >= q.e * (1.0 - 1e-3));
        assert_eq!(a.scale(2.0).d, vec![4.0]);
        let c = Lin::combine(&[(2.0, &a), (-1.0, &b)], 1);
        assert_eq!((c.v, c.d[0]), (1.0, 5.0));
        assert!((c.e - 4e-3).abs() < 1e-18 && (c.de[0] - 2e-4).abs() < 1e-18);
    }
}
