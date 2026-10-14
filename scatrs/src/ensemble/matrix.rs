//! Dense column-major matrices of `f64` and their products, for the ensemble's matrix
//! products.
//!
//! The products are `matrixmultiply`'s packed, register-blocked kernels, which choose AVX2 and
//! FMA (x86-64) or NEON (aarch64) at run time.

use std::ops::{Index, IndexMut};

/// A column-major matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Mat {
    nrows: usize,
    ncols: usize,
    /// The columns, one after another: always `nrows * ncols` values.
    data: Vec<f64>,
}

impl Mat {
    /// The `nrows` × `ncols` zero matrix.
    pub(crate) fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: vec![0.0; nrows * ncols],
        }
    }

    /// The `nrows` × `ncols` matrix of `f(i, j)`.
    pub(crate) fn from_fn(
        nrows: usize,
        ncols: usize,
        mut f: impl FnMut(usize, usize) -> f64,
    ) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(i, j));
            }
        }
        Self { nrows, ncols, data }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn ncols(&self) -> usize {
        self.ncols
    }

    /// `f(self[i, j], other[i, j])` elementwise.
    pub(crate) fn zip_map(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        assert_eq!((self.nrows, self.ncols), (other.nrows, other.ncols));
        Self {
            nrows: self.nrows,
            ncols: self.ncols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    /// `f(self[i, j])` elementwise.
    pub(crate) fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            nrows: self.nrows,
            ncols: self.ncols,
            data: self.data.iter().map(|&a| f(a)).collect(),
        }
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        debug_assert!(i < self.nrows && j < self.ncols);
        &self.data[i + j * self.nrows]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        debug_assert!(i < self.nrows && j < self.ncols);
        &mut self.data[i + j * self.nrows]
    }
}

/// `dst` = A·B for the first `k` columns of `a` and the first `k` rows of `b`, with `dst`
/// resized to `a.nrows()` × `b.ncols()` if its shape differs.
pub(crate) fn matmul(dst: &mut Mat, a: &Mat, b: &Mat, k: usize) {
    assert!(
        k <= a.ncols && k <= b.nrows,
        "matmul: k = {k} is out of range"
    );
    let (m, n) = (a.nrows, b.ncols);
    if dst.nrows != m || dst.ncols != n {
        *dst = Mat::zeros(m, n);
    }
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        dst.data.fill(0.0);
        return;
    }
    // The bounds of every element dgemm touches: A[i, p] at i + p·m, B[p, j] at p + j·b.nrows
    // and C[i, j] at i + j·m, for i < m, p < k, j < n.
    assert!(m + (k - 1) * m <= a.data.len());
    assert!(k + (n - 1) * b.nrows <= b.data.len());
    assert!(m + (n - 1) * m <= dst.data.len());
    let (lda, ldb) = (m as isize, b.nrows as isize);
    // SAFETY: the pointers are valid for the strided m × k, k × n and m × n matrices (the
    // assertions above), C does not alias A or B (it is borrowed mutably), and with β = 0 C is
    // only written.
    #[allow(unsafe_code)]
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            a.data.as_ptr(),
            1,
            lda,
            b.data.as_ptr(),
            1,
            ldb,
            0.0,
            dst.data.as_mut_ptr(),
            1,
            lda,
        );
    }
}

/// A·B as a new matrix.
pub(crate) fn product(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.ncols, b.nrows, "product: the shapes do not match");
    let mut out = Mat::default();
    matmul(&mut out, a, b, a.ncols);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The product matches the naive sum, with a partial inner dimension, a resized and a
    /// reused destination, and empty shapes.
    #[test]
    fn matches_the_naive_product() {
        let mut seed = 1u64;
        let mut rand = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut c = Mat::default();
        for (m, n, k) in [
            (1, 1, 1),
            (7, 3, 5),
            (9, 5, 257),
            (130, 13, 600),
            (300, 61, 40),
            (3, 2100, 15),
            (5, 7, 0),
            (0, 4, 3),
            (4, 0, 3),
        ] {
            let a = Mat::from_fn(m, k + 3, |_, _| rand());
            let b = Mat::from_fn(k + 2, n, |_, _| rand());
            matmul(&mut c, &a, &b, k);
            assert_eq!((c.nrows(), c.ncols()), (m, n));
            for i in 0..m {
                for j in 0..n {
                    let want: f64 = (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum();
                    let size: f64 = (0..k).map(|p| (a[(i, p)] * b[(p, j)]).abs()).sum();
                    assert!(
                        (c[(i, j)] - want).abs() <= 1e-14 * size,
                        "{m}×{n}×{k} [{i}, {j}]: {} vs {want}",
                        c[(i, j)]
                    );
                }
            }
        }
    }
}
