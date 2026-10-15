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
| `terms.py`, `check_terms.py` | Q's integrands as scalar terms, generated from the vector components (reproduce Q to 10⁻¹⁸) |
| `laurent.py`, `laurent_q.py`, `laurent_rule2.py` | the Laurent series of the y_n products, and which groups of their terms integrate to zero |
| `regular.py`, `cancel_reg.py` | the regularized integrals at 53 bits, and the cancellation left per element; `cancel-reg-x15-eps4.txt` |
| `amplitude.py`, `check_amplitude.py` | fixed-orientation amplitudes from T (a sphere gives Bohren & Huffman's i·S₁, i·S₂), the brute-force orientation average |
| `ranks.py` | random orientation by the rank decomposition of T, against the brute-force average |
| `where_cancel.py` | where an element's cancellation happens: among the terms at a node, across the nodes, between the term types after integration, between the ν terms and the others |
| `partial_head.py` | subtracting only the Laurent powers p ≤ p_cut: which cuts are exact, and what each leaves |
| `cancel_shapes.py` | the plain cancellation for a spheroid, a finite cylinder and a Chebyshev particle; with `y` or `h`, of the shell integrals of a coated particle |

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

11. **Which terms integrate to zero, exactly** (`laurent_q.py`, `laurent_rule2.py`).
    - In every element with n ≠ k, in every block, the terms of power ≤ 0 of the y_n products' Laurent series integrate to exactly zero, as a group. Checked on 2192 elements (oblate and prolate, ε from 1/4 to 5, m = 0–3), to 10⁻³⁵.
    - Those terms are the electrostatic limit: the outgoing and regular harmonic multipoles, whose pairing is surface-independent and vanishes for n ≠ k, as on a sphere.
    - The negative powers alone vanish only for |n − k| ≥ 3. The positive powers do not vanish: the group p ≤ |n − k| − 2 is non-zero in about 15% of elements, by 10⁻⁵ to 10⁻².
    - An earlier per-J analysis (since removed) used hand-derived integrands with sign errors; `terms.py` replaces it, generated from the verified vector components.
12. **Three exact forms per element, the cleanest kept** (`cancel_reg.py`).
    - The forms: plain; minus the negative powers (|n − k| ≥ 3); minus the powers ≤ 0 (n ≠ k).
    - Subtracting can create large terms where the y_n product is small (the constant term at large x), so the form with the smallest Σ|terms| is chosen per element.
    - Worst cancellation, plain → best: 9.6·10⁷ → 2.6·10⁶ (x = 10, ε = 2), and **6·10¹⁶ → 1.7·10⁵ (x = 15, ε = 4)**, where plain f64 fails completely.
    - What remains is the positive powers: smooth functions with small integrals against high-degree angular factors, a loss no exact identity removes.
13. **Random orientation in O(N⁴)** (`ranks.py`).
    - For an axisymmetric particle the rank-s components of T rotate as D^s_{q0}(R). So A_{pσ}(Θ; R) = Σ_{s,q} D^s_{q0}(R) Φ_{sq}(Θ), and ⟨A A'*⟩ = Σ Φ Φ'*/(2s + 1).
    - It matches the brute-force orientation average to 10⁻¹⁵ in all six elements (x = 2, ε = 1.8, s = 1.5 + 0.02i).
    - T^{ij}_{−m} = ±T^{ij}_m (+ for 11 and 22, − for 12 and 21) exactly, and the plane wave's helicity coefficients are 2 i^{n−1} √(π(2n + 1)).
14. **The cancellation left after the removal is across the nodes, and no identity between the terms reaches it** (`where_cancel.py`, `partial_head.py`; x = 15, ε = 1/3, N = 42, rows 40–42 against columns 2–14).
    - At a node the terms barely cancel: Σ|t| / |Σt| is 1.1 plain and 1.5 after the removal. The loss is across the nodes.
    - Plain, it is inside each term's own integral (up to 6·10⁹): the electrostatic head. The ν terms against the others, the split the integration by parts of Somerville et al. (2011) acts on, cancel by 20–10⁴ plain and by about 3 after the removal, so that identity has nothing left to remove.
    - Every cut p ≤ p_cut above the lowest two is an exact form, but each leaves a remainder of up to 10¹⁷ (the Laurent terms are large and alternate); only the complete head (p ≤ 0) leaves a small one, so the kernel's three forms are the best of this family.
    - The head's coefficients near p = 0 are the leading y_n coefficient times Taylor coefficients of j_k at large |s r| (about 10¹⁴ here): large at every node, cancelled at the poles by the positive powers. The loss does not follow the shape's Bernstein ellipse alone (ε = 1/2 loses more than ε = 1/3 at x = 15).
15. **Cylinders and Chebyshev particles cancel less; the shell integrals of a coated spheroid as much** (`cancel_shapes.py`, x = 8, N = 26, s = 1.5 + 0.02i, worst over the columns of row 26).
    - Plain: spheroid ε = 1/2 5·10⁹, ε = 0.74 8·10¹²; cylinder (half-length/radius √3, the quadrature split at the edges) 1.4·10⁷; Chebyshev (T₄, δ = 0.15) 2·10⁴. Doubling the nodes changes none of them. Somerville et al. (2012) found no cancellation of the electrostatic kind for the last two at x_max = 0.5; at larger sizes the cylinder keeps a loss like a spheroid's after the removal.
    - With outgoing internal functions h_k(s r), the integrals a coated particle's shell adds, a spheroid loses 1.5·10⁹ (ε = 1/2) and 5·10⁹ (ε = 2) at row 26; the y_n·y_k half alone 1.3–2·10⁹. Its j_k half is the plain integral the removal treats; the y_n·y_k half would need its own analysis.
16. **For elongated or flattened spheroids the quadrature, not the precision, sets the cost** (the kernel, fixed N = 12, x_max = 1, s = 1.5 + 0.02i).
    - r(cos θ) is singular at cos θ = ±1/e for a prolate spheroid and at ±i/√(ε² − 1) for an oblate one. Both give the Bernstein ellipse ρ = √((h + 1)/(h − 1)), h = max(ε, 1/ε), so Gauss–Legendre needs about ln(1/tol)/ln((h + 1)/(h − 1)) nodes on [−1, 1] whatever N is.
    - Measured, half-range nodes → relative error of Q_ext, prolate and oblate alike: h = 10, 40 → 6·10⁻⁶, 80 → 2·10⁻¹²; h = 30, 160 → 5·10⁻⁸, 320 → 10⁻¹⁵; h = 100, 640 → 10⁻⁹, 1280 → 10⁻¹⁴. Per node on [−1, 1] the error falls by 1.205, 1.061 and 1.018, against (h + 1)/(h − 1) = 1.222, 1.069 and 1.020.
    - With those nodes the kernel converges at h = 30 and 100 up to x_max = 10, for s = 1.5 + 0.02i and for the metal s = √(−4 + 0.5i), within 10⁻¹² of double-double; N = 40 for the dielectric, 60 for the metal, as the extra multipoles of Somerville et al. (2015).
    - The driver's node floor (`driver::shape_theta`) follows from it: the nodes for 10⁻⁴·rtol at that rate, so that 2N nodes no longer leave the quadrature, not the orders, unconverged.
