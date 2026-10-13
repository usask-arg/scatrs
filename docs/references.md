# References

**Quadrature**

- F. G. Tricomi, "Sugli zeri dei polinomi sferici ed ultrasferici", *Annali di Matematica Pura
  ed Applicata* (4) 31 (1950), as cited in N. Hale and A. Townsend, *SIAM Journal on
  Scientific Computing* 35 (2013). The starting approximation of the Gauss–Legendre nodes.
- G. Szegő, *Orthogonal Polynomials*, 4th ed., American Mathematical Society Colloquium
  Publications 23 (1975), Theorem 6.21.2. Bruns's inequality for the zeros of $P_n$, used by
  the tests.
- I. Bogaert, "Iteration-free computation of Gauss–Legendre quadrature nodes and weights",
  *SIAM Journal on Scientific Computing* 36(3) (2014), A1008–A1026. The planned $O(n)$
  method.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (2013), Theorem
  19.3. The error bound for Gauss quadrature of analytic functions, used in
  {doc}`examples/henyey-greenstein`. The theorem is stated for rules with $n + 1$ points.

**Wigner d-functions and polarized radiative transfer**

- M. I. Mishchenko, L. D. Travis and A. A. Lacis, *Scattering, Absorption, and Emission of
  Light by Small Particles*, Cambridge University Press (2002), Appendix B. The convention of
  the Wigner d-functions, and the recurrence of Eq. B.22.
- J. Stoer and R. Bulirsch, *Introduction to Numerical Analysis*, 3rd ed., Springer (2002),
  the section on the algorithms of Goertzel and Reinsch. Reinsch's difference form, the idea
  behind the difference forms that scatrs uses next to the poles.
- J. W. Hovenier, C. van der Mee and H. Domke, *Transfer of Polarized Light in Planetary
  Atmospheres: Basic Concepts and Practical Methods*, Kluwer Academic Publishers (2004). The
  convention for the Greek coefficients.
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
