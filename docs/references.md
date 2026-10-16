# References

**Mie scattering**

- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*,
  Wiley (1983), Chapter 4. The conventions, the series and the recurrences of $\pi_n$ and
  $\tau_n$.
- W. J. Wiscombe, *Mie Scattering Calculations: Advances in Technique and Fast,
  Vector-Speed Computer Codes*, NCAR Technical Note NCAR/TN-140+STR (1979, revised 1996), and
  "Improved Mie scattering algorithms", *Applied Optics* 19 (1980), 1505–1509. The MIEV0
  baseline, its truncation rule and its published test cases.
- W. Gautschi, "Computational aspects of three-term recurrence relations", *SIAM Review* 9
  (1967), 24–82. Minimal solutions and Pincherle's theorem, behind the enclosure of the
  logarithmic derivatives.
- W. J. Lentz, "Generating Bessel functions in Mie scattering calculations using continued
  fractions", *Applied Optics* 15 (1976), 668–671. The continued fraction for $D_n$, which
  scatrs replaces by a contracting downward recurrence, and its guard against a zero
  denominator, which scatrs keeps.
- R. L. Smith, "Algorithm 116: Complex division", *Communications of the ACM* 5 (1962), 435.
  The complex reciprocal.
- I. Gargantini and P. Henrici, "Circular arithmetic and the determination of polynomial
  zeros", *Numerische Mathematik* 18 (1972), 305–320. Disk arithmetic, used by the reference
  generator.

**Layered spheres**

- A. L. Aden and M. Kerker, "Scattering of electromagnetic waves from two concentric spheres",
  *Journal of Applied Physics* 22 (1951), 1242–1246. The coated sphere, by direct matching at
  the interfaces, as the reference generator and the textbook method of the comparison do.
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*,
  Wiley (1983), Chapter 5 and Section 8.1. The electrostatic polarizability of a coated
  sphere (the Rayleigh limit checked in the external validation) and the coated sphere's
  coefficients.
- W. Yang, "Improved recursive algorithm for light scattering by a multilayered sphere",
  *Applied Optics* 42 (2003), 1710–1720. The recursion for the effective logarithmic
  derivatives.
- O. Peña and U. Pal, "Scattering of electromagnetic radiation by a multilayered sphere",
  *Computer Physics Communications* 180 (2009), 2348–2354. Yang's recursion in the form of
  scattnlay, the code of the comparison.
- O. B. Toon and T. P. Ackerman, "Algorithms for the calculation of scattering by stratified
  spheres", *Applied Optics* 20 (1981), 3657–3660. The coated sphere in atmospheric optics,
  and the stability of its recurrences.
- T. C. Bond and R. W. Bergstrom, "Light absorption by carbonaceous particles: an
  investigative review", *Aerosol Science and Technology* 40 (2006), 27–67. The refractive
  index of black carbon in {doc}`examples/coated`.

**Size distributions and refractive indices**

- J. E. Hansen and L. D. Travis, "Light scattering in planetary atmospheres", *Space Science
  Reviews* 16 (1974), 527–610. The effective radius and effective variance, and the gamma
  distribution of `Gamma`.
- D. Deirmendjian, *Electromagnetic Scattering on Spherical Polydispersions*, Elsevier (1969).
  The modified gamma distribution of `ModifiedGamma`.
- I. H. Malitson, "Interspecimen comparison of the refractive index of fused silica",
  *Journal of the Optical Society of America* 55 (1965), 1205–1209. The Sellmeier
  coefficients of fused silica in {doc}`theory/refractive-index` and
  {doc}`examples/distributions`.
- C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles*,
  Wiley (1983), Chapter 9. The Lorentz oscillator model of an absorption band in
  {doc}`examples/distributions`.

**Quadrature**

- F. G. Tricomi, "Sugli zeri dei polinomi sferici ed ultrasferici", *Annali di Matematica Pura
  ed Applicata* (4) 31 (1950), as cited in N. Hale and A. Townsend, *SIAM Journal on
  Scientific Computing* 35 (2013). The starting approximation of the Gauss–Legendre nodes.
- G. Szegő, *Orthogonal Polynomials*, 4th ed., American Mathematical Society Colloquium
  Publications 23 (1975), Theorem 6.21.2. Bruns's inequality for the zeros of $P_n$, used by
  the tests.
- I. Bogaert, "Iteration-free computation of Gauss–Legendre quadrature nodes and weights",
  *SIAM Journal on Scientific Computing* 36(3) (2014), A1008–A1026. The $O(n)$ method
  of the `f64` rules above 100 points.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (2013), Theorem
  19.3. The error bound for Gauss quadrature of analytic functions, used in
  {doc}`utilities/legendre-moments` and for the nodes of spheroids ({doc}`theory/spheroids`).
  The theorem is stated for rules with $n + 1$ points.

**Spheroids and the T-matrix method**

- P. C. Waterman, "Symmetry, unitarity, and geometry in electromagnetic scattering",
  *Physical Review D* 3 (1971), 825–839. The extended boundary condition method.
- M. I. Mishchenko, "Light scattering by randomly oriented axially symmetric particles",
  *Journal of the Optical Society of America A* 8 (1991), 871–882. The orientation averages
  from the T-matrix, by its rank decomposition.
- M. I. Mishchenko and L. D. Travis, "T-matrix computations of light scattering by large
  spheroidal particles", *Optics Communications* 109 (1994), 16–21. The loss of precision of
  the EBCM for spheroids, and the extended-precision code `tmatd` of the comparison.
- M. I. Mishchenko, L. D. Travis and A. A. Lacis, *Scattering, Absorption, and Emission of
  Light by Small Particles*, Cambridge University Press (2002), Chapter 5. The T-matrix of an
  axisymmetric particle (§5.8), its conventions and the convergence procedure.
- W. R. C. Somerville, B. Auguié and E. C. Le Ru, "Severe loss of precision in calculations
  of T-matrix integrals", *Journal of Quantitative Spectroscopy and Radiative Transfer* 113
  (2012), 524–535; "A new numerically stable implementation of the T-matrix method for
  electromagnetic scattering by spheroidal particles", 123 (2013), 153–168; and "Accurate and
  convergent T-matrix calculations of light scattering by spheroids", 160 (2015), 29–35. The
  cancellations in the integrals for spheroids, an algorithm that avoids them (with the
  arbitrary-precision values {doc}`validation/external` compares), and its convergence.
- F. Kuik, J. F. de Haan and J. W. Hovenier, "Benchmark results for single scattering by
  spheroids", *Journal of Quantitative Spectroscopy and Radiative Transfer* 47 (1992),
  477–489. The benchmark spheroid whose arbitrary-precision values {doc}`validation/external`
  compares.
- W. R. C. Somerville, B. Auguié and E. C. Le Ru, "SMARTIES: User-friendly codes for fast and
  accurate calculations of light scattering by spheroids", *Journal of Quantitative
  Spectroscopy and Radiative Transfer* 174 (2016), 39–55. The code, and the range it reaches,
  compared in {doc}`validation/external`.
- M. R. A. Majić, L. Pratley, D. Schebarchov, W. R. C. Somerville, B. Auguié and E. C. Le Ru,
  "Approximate T matrix and optical properties of spheroidal particles to third order with
  respect to size parameter", *Physical Review A* 99 (2019), 013853. The long-wavelength
  expansion of the T-matrix that the tests check.
- M. R. A. Majić, F. Gray, B. Auguié and E. C. Le Ru, "Electrostatic limit of the T-matrix for
  electromagnetic scattering: Exact results for spheroidal particles", *Journal of Quantitative
  Spectroscopy and Radiative Transfer* 200 (2017), 50–58. The electrostatic limit of the
  EBCM matrices.
- O. Dubovik, A. Sinyuk, T. Lapyonok, B. N. Holben, M. Mishchenko, P. Yang, T. F. Eck,
  H. Volten, O. Muñoz, B. Veihelmann, W. J. van der Zande, J.-F. Leon, M. Sorokin and
  I. Slutsker, "Application of spheroid models to account for aerosol particle nonsphericity
  in remote sensing of desert dust", *Journal of Geophysical Research* 111 (2006), D11208.
  Spheroids as the model of mineral dust in {doc}`examples/spheroids`.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, SIAM (2002),
  Chapters 12 and 15. Iterative refinement in mixed precision, and condition estimation.

**Wigner d-functions and polarized radiative transfer**

- M. I. Mishchenko, L. D. Travis and A. A. Lacis, *Scattering, Absorption, and Emission of
  Light by Small Particles*, Cambridge University Press (2002), Appendix B. The convention of
  the Wigner d-functions, the recurrence of Eq. B.22, and the expansion of the phase matrix in
  Greek coefficients (Eqs. 4.75–4.80) that scatrs follows.
- J. Stoer and R. Bulirsch, *Introduction to Numerical Analysis*, 3rd ed., Springer (2002),
  the section on the algorithms of Goertzel and Reinsch. Reinsch's difference form, the idea
  behind the difference forms that scatrs uses next to the poles.
- J. W. Hovenier, C. van der Mee and H. Domke, *Transfer of Polarized Light in Planetary
  Atmospheres: Basic Concepts and Practical Methods*, Kluwer Academic Publishers (2004). The
  expansion of the phase matrix in generalized spherical functions, behind the Greek
  coefficients.
- L. G. Henyey and J. L. Greenstein, "Diffuse radiation in the Galaxy", *Astrophysical
  Journal* 93 (1941), 70–83. The Henyey–Greenstein phase function.

**Arithmetic**

- M. Joldes, J.-M. Muller and V. Popescu, "Tight and rigorous error bounds for basic building
  blocks of double-word arithmetic", *ACM Transactions on Mathematical Software* 44(2) (2017).
  The double-double algorithms and their error bounds.
- J. R. Shewchuk, "Adaptive precision floating-point arithmetic and fast robust geometric
  predicates", *Discrete & Computational Geometry* 18 (1997), 305–363. The floating-point
  expansions of the double-double remainder.
- T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product", *SIAM Journal on
  Scientific Computing* 26(6) (2005), 1955–1988. The compensated sum that the tests use to
  measure errors against the reference values.
- J. Revels, M. Lubin and T. Papamarkou, "Forward-mode automatic differentiation in Julia",
  arXiv:1607.07892 (2016). ForwardDiff.jl, whose dual numbers made the same choice of strict
  zero tests (in version 1.0).

**Reference values**

- F. Johansson, "Arb: efficient arbitrary-precision midpoint-radius interval arithmetic", *IEEE
  Transactions on Computers* 66(8) (2017), 1281–1292. The ball arithmetic behind the reference
  suite, used through [python-flint](https://python-flint.readthedocs.io).
