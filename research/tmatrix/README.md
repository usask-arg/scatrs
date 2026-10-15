# Where the EBCM for spheroids loses precision

A laboratory for the design of the spheroid T-matrix (phase 4): an EBCM in mpmath that builds Q
and P = RgQ of one azimuthal mode from their definition, as surface integrals of cross products of
vector spherical wave functions, at any working precision. At 53 bits every operation rounds as
in f64 while the special functions stay correctly rounded, so the losses measured there are the
method's own.

Run with `uv run --no-project --with mpmath python <script> <args>`. Arguments are
`x eps s_re s_im m nmax ntheta`: equal-volume size parameter, aspect ratio a/c (> 1 oblate),
relative index, azimuthal mode, truncation and Gauss–Legendre points in cos θ.

| Script | Checks or measures |
|---|---|
| `ebcm.py` | the EBCM itself |
| `check_sphere.py` | a sphere gives T¹¹ = −bₙ, T²² = −aₙ (Bohren & Huffman) to 30 digits, off-diagonal 10⁻³² |
| `check_spheroid.py`, `check_recip.py` | ⟨C_ext⟩ = ⟨C_sca⟩ for a lossless spheroid (2·10⁻¹²); reciprocity converges with N (10⁻¹⁸ at N = 20) |
| `precision.py` | plain 53-bit EBCM (B); exact elements solved at 53 bits (C); rounded elements solved exactly (D); cond and Skeel cond of Q |
| `cancel.py` | the cancellation inside each Q integral (Σ\|terms\| / \|Q_nk\|) |
| `columns.py` | Skeel(Q) and the f64 floor after a change of internal basis (QR of the stacked [P; Q]) |
| `basis.py` | 53-bit integration with that basis sampled exactly at the nodes |
| `sensitivity.py` | T's sensitivity to 2⁻⁵³ errors in the sampled internal, regular and outgoing functions |
| `floor.py` | the f64 floor (C) and Skeel(Q) over a grid; results in `floor-grid.txt` |
| `subtract.py` | the exact identity Q = Q̃ + diag(α) Q⁽ᵏ¹⁾ with regularized test functions |
| `sources.py` | null-field equations tested with distributed sources (NFM-DS) |
| `mixed.py` | which rows or columns of P and Q carry the f64 floor |
| `refine.py` | mixed-precision iterative refinement |

## Findings (mode m = 1, s = 1.5 + 0.01i unless stated)

1. **The plain EBCM loses its digits in the Q integrals, not in the solve.**
   - Plain f64 gives a T error of 10⁻¹⁰ at x = 10, ε = 2, and 4·10⁻⁹ at x = 20. At x = 15, ε = 4 it fails completely (error ~10²).
   - The exact elements solved in f64 give 1.3·10⁻¹⁵, 1.4·10⁻¹³ and 3·10⁻¹¹ for the same cases.
   - P is accurate in f64 throughout.
2. **The cause is exact cancellation.**
   - For a spheroid, 1/r(θ)² is a polynomial in cos²θ.
   - In the outgoing pairings ⟨h_n(kr) …, j_k(skr) …⟩, the singular (negative-power Laurent) terms of the product therefore multiply a polynomial in cos θ of degree below |n − k|, and integrate to exactly zero against the Legendre functions.
   - The cancellation reaches 10⁸ at x = 10, ε = 2 (`cancel.py`), worst for the highest outgoing order against low internal orders, mostly in the radial term.
   - Somerville, Auguié & Le Ru (2012, 2013) remove these terms analytically per pair (n, k): SMARTIES's "modified Bessel products".
3. **The cancellation amplifies any pointwise error in either factor** (`sensitivity.py`).
   - Errors of 2⁻⁵³ in the sampled internal functions alone give T errors of 4·10⁻¹³ (x = 10, ε = 2), 3·10⁻¹¹ (x = 20) and 1.2 (x = 15, ε = 4). Errors in the outgoing functions do the same.
   - Errors in the regular functions (P's rows) give 3·10⁻¹⁷.
   - So the vanishing terms must be removed symbolically. This makes the element errors independent, and the f64 limit is then the Skeel floor of point 4.
4. **The f64 floor** (exact elements rounded to f64, then an f64 solve) is about Skeel(Q)·2⁻⁵³ (`floor-grid.txt`):
   - ε = 2: 10⁻¹⁵ at x = 10, 10⁻¹³ at x = 20, 10⁻¹¹ at 30, 10⁻⁹ at 40, 6·10⁻⁸ at 50;
   - ε = 3: 10⁻¹¹ at x = 20, 9·10⁻⁹ at 30;
   - ε = 1/3: 4·10⁻¹¹ at x = 15, 1.2·10⁻⁶ at 25;
   - ε = 5: 5·10⁻¹³ at x = 10, 4·10⁻¹⁰ at 15;
   - s = 1.33, x = 30, ε = 2: 10⁻¹². s = 3 + 0.1i, x = 15, ε = 2: 10⁻¹².

   It reaches about 10⁻⁸ near |s|·x_max ≈ 60–95, depending on the shape, where x_max = k·max(a, c).
5. **That floor is the conditioning of the internal basis, not of the problem.**
   - The spherical internal functions j_k(skr) are nearly dependent on the spheroid's surface, because the high orders all concentrate near r_max.
   - T = −P Q⁻¹ is unchanged by any column transform. With the orthonormalizing one, Skeel drops from 8·10² to 14 (x = 10, ε = 2), from 4·10⁴ to 22 (x = 20) and from 8·10⁷ to 19 (x = 15, ε = 4). The f64 floor drops to 6·10⁻¹⁷.
   - It cannot be exploited in f64 with spherical waves (`basis.py`). Integrating in the new basis needs the symbolic cancellation of point 2, which is defined per spherical pair. A numerically sampled basis falls under point 3: 9·10⁻¹¹ at x = 20, and 4·10⁻² at x = 15, ε = 4.
   - Exploiting it needs an analytic well-conditioned basis with its own cancellation-free integrals. Spheroidal wave functions (separation of variables), or distributed sources (NFM-DS), are separate methods.
   - With elements in double-double the transform does work. Q built in double-double, transformed and rounded to f64, then solved in f64, is at the rounding level.

6. **An exact regularization for any axisymmetric shape** (`subtract.py`).
   - Every element is Q_nk = −i B(X_n, Y_k), with the reciprocity form B(X, Y) = ∫_S n̂·(X × ∇×Y − Y × ∇×X).
   - By the divergence theorem, B(G, Y) = 0 for any k₁-field G regular in the particle. It is surface-independent, and diagonal in the order, for the outgoing k₁-waves.
   - So Q = Q̃ + diag(α) Q⁽ᵏ¹⁾ exactly. Here Q̃ uses X̃_n = X_n(kr) − α_n X_n(k₁r), and Q⁽ᵏ¹⁾ is evaluated on a sphere. Verified to the quadrature error; Q⁽ᵏ¹⁾'s off-diagonal is 10⁻⁴⁹.
   - α_n = s^{n+1} removes the leading singular term of h_n. The rest of the Laurent tail keeps a factor (1 − s^{2l}), which is small only when n ≫ k r_min.
   - Fitted α (least squares over the surface) cuts the cancellation 2–6× (1.2·10⁸ → 2·10⁷ at x = 10, ε = 2): modest.
7. **The f64 floor is carried by the high outgoing-order rows, not the columns** (`mixed.py`).
   - Rows n ≥ N/2 exact, the rest rounded: 1.4·10⁻¹³ → 5·10⁻¹⁶ (x = 20, ε = 2), 3.7·10⁻¹⁰ → 1.7·10⁻¹² (x = 15, ε = 5), 1.2·10⁻⁶ → 4.6·10⁻¹³ (x = 25, ε = 1/3).
   - Rows n ≥ N/4 exact gives the rounding level in all three. Exact high columns barely help.
   - On the surface, those rows are the evanescent multipoles r^{−n−1}Pₙ. They concentrate at r_min and are nearly dependent on any non-spherical surface.
8. **Distributed sources trade the cancellation for redundancy** (`sources.py`, prolate, sources on the focal segment).
   - T = P Q_D⁻¹ S_D, with S_D surface-independent (−I for origin tests; one origin source reproduces the EBCM to 10⁻⁴⁸).
   - The Q_D cancellation falls: 10⁶ → 6·10² at x = 5, and 1.7·10¹⁰ → 5·10³ at x = 10.
   - But multipoles at neighbouring centres are nearly dependent: Skeel 87 → 6·10⁶, and 4·10³ → 7·10¹². The f64 floor is worse.
   - Test fields both cancellation-free and well-conditioned would be singular exactly on the focal set: outgoing spheroidal waves, i.e. the spheroidal basis (separation of variables).
9. **An f64 factorization carries the same floor as f64 elements.** Backward stability means it perturbs every element by about 2⁻⁵³|Q|. So beating the floor needs the residuals in higher precision, not only the elements.
10. **Mixed-precision iterative refinement beats the floor** (`refine.py`).
    - Q is factorized in f64. The residual of X Q = −P is evaluated with double-double elements and arithmetic, O(N²) per sweep. Corrections use the f64 factors.
    - It converges when Skeel(Q)·2⁻⁵³ < 1.
    - Results, as f64 floor → refined:

      | case | rows n ≥ N/4 in double-double, 2 sweeps | all rows in double-double, 3–4 sweeps |
      |---|---|---|
      | x = 25, ε = 1/3 | 1.2·10⁻⁶ → 7·10⁻¹⁷ | 1.2·10⁻²² |
      | x = 15, ε = 5 | 3.7·10⁻¹⁰ → 4·10⁻¹⁶ | 4·10⁻²⁶ |
      | x = 50, ε = 2 | 5.9·10⁻⁸ → 4·10⁻¹⁷ | 4·10⁻²⁴ |

    - The O(N³) factorization stays in f64. Double-double is needed only for the integrals of the upper three quarters of the rows, and for the residuals.
