# Spheroids

<!-- synced with scatrs/src/tmatrix/ -->

{mod}`scatrs.tmatrix` computes the optical properties of spheroids averaged uniformly over
their orientations, the case of randomly oriented particles in a cloud or an aerosol layer.
It uses the T-matrix method, with the T-matrix from the extended boundary condition method
(EBCM; Waterman 1971). This chapter describes the method, the three changes that make it
accurate in double precision where the classical method is not, and how accurate and costly
the results are.

## Geometry and conventions

A spheroid has semi-axes $a$, $a$ and $c$, with $c$ along its symmetry axis, and aspect ratio
$\varepsilon = a/c$: oblate for $\varepsilon > 1$, prolate for $\varepsilon < 1$, a sphere at
$\varepsilon = 1$. Its surface is

$$
r(\theta) = \frac{ac}{\sqrt{c^2\sin^2\theta + a^2\cos^2\theta}} .
$$

Its size is the size parameter $x = k r_v$ of the sphere of equal volume, $r_v^3 = a^2 c$,
with $k = 2\pi/\lambda$. That sphere's geometric cross section $\pi r_v^2$ normalizes the
efficiencies and the unnormalized phase matrix and Greek coefficients, so that a spheroid and
a sphere of the same volume compare directly. The longest semi-axis has the size parameter
$x_{\max} = k\max(a, c)$, which sets the number of orders. The other conventions are those of
{doc}`conventions`.

## The T-matrix

The T-matrix relates the coefficients of the scattered field, in outgoing vector spherical
wave functions $\mathbf{M}_{mn}$ and $\mathbf{N}_{mn}$, to those of the incident field, in
regular ones. For an axisymmetric particle in its own frame it is diagonal in the azimuthal
order $m$: each mode $m = 0, \dots, N$ has its own matrix $T^m$ over the orders
$n, k = \max(m, 1), \dots, N$, with four blocks ($\mathbf{M}$ and $\mathbf{N}$ in and out).

The EBCM gives $T^m = -P^m (Q^m)^{-1}$, with the elements of $Q^m$ and $P^m$ surface integrals
of cross products of the wave functions,

$$
Q_{nk} = -i \oint_S \hat{\mathbf n}\cdot\bigl[\mathbf X_n \times \nabla\times\mathbf Y_k
  - \mathbf Y_k \times \nabla\times\mathbf X_n\bigr]\, dS ,
$$

where $\mathbf X_n$ is an outgoing wave of the medium for $Q$ and a regular one for $P$, and
$\mathbf Y_k$ a regular wave inside the particle (Mishchenko, Travis & Lacis 2002, §5.8). The
integral over $\phi$ is exact, and the one over $\theta$ uses Gauss–Legendre quadrature with
$2n_\theta$ nodes in $\cos\theta$. A spheroid is symmetric under reflection in its equatorial
plane: half of the elements vanish, each mode splits into two independent systems of half the
size, and the integrands of the others are even in $\cos\theta$, so only the $n_\theta$ nodes
with $\cos\theta > 0$ are evaluated.

For a sphere $T^m$ is diagonal, $T^{11} = -b_n$ and $T^{22} = -a_n$, the Mie coefficients;
scatrs checks this identity, and that every averaged output of an aspect ratio of 1 equals
{mod}`scatrs.mie`'s.

## Why the classical method loses precision

The outgoing waves contain the spherical Bessel functions of the second kind,
$y_n(kr) \sim -(2n-1)!!/(kr)^{n+1}$ for $kr < n$. On a non-spherical surface the integrand of
an element $Q_{nk}$ with $n$ much larger than $k$ contains terms as large as
$(n/kr_{\min})^n$, which the integration over $\theta$ must cancel down to a result many
orders of magnitude smaller. In double precision this cancellation destroys the element: at
$x = 15$ and $\varepsilon = 4$ the classical integrals lose up to sixteen digits, and the
T-matrix is wrong in its leading digit. This is the well-known loss of precision of the EBCM
for elongated or flattened particles, usually addressed by computing in extended precision
(Mishchenko & Travis 1994). Somerville, Auguié & Le Ru (2012) traced it, for spheroids, to
cancellations in these integrals, and gave an algorithm that avoids them (2013), implemented in
their code SMARTIES (2016).

scatrs removes the cancellation in its own way. Expand the radial factors of each integrand, the
products of $y_n$ (or its derivative) with $j_k(m k r)$, in powers of $kr$. For a spheroid
$1/r^2 = \alpha + \beta\cos^2\theta$, so every term of non-positive power is a polynomial in
$\cos\theta$ times an angular function, the products of the associated Legendre functions
of orders $n$ and $k$. Those polynomials have low degree, and the Legendre functions are
orthogonal over the whole range of $\cos\theta$, so the terms of negative power integrate to
exactly zero when $|n - k| \ge 3$, and the terms of power at most 0 (the electrostatic limit of
the integral) integrate to exactly zero as a group whenever $n \ne k$. This was verified to 35
digits on more than 2000 elements. It rests on $1/r^2$ being a polynomial in $\cos\theta$,
which holds for spheroids, not for other shapes. These are exactly the terms that cancel. scatrs therefore accumulates each element in three exact forms (the
plain integrand, without the terms of negative power, and without those of power at most 0)
together with the sum of the magnitudes of what each adds, and keeps the form with the
smallest sum. At each node the kept part is evaluated either from its own series or as the
function minus the removed terms, whichever adds less in magnitude. The series are summed
until negligible at the longest axis, so both routes agree to rounding.

With this change the elements carry little cancellation, and at $x = 15$, $\varepsilon = 4$
the T-matrix error in double precision falls from order 1 to $3\cdot10^{-4}$.

## The conditioning floor and mixed-precision refinement

Even with accurate elements, a double-precision solve leaves an error of about
$\mathrm{cond}(Q)\cdot 2^{-53}$, carried by the rows of high outgoing order. On a
non-spherical surface these are the evanescent multipoles $r^{-n-1}P_n^m(\cos\theta)$, which
concentrate near the shortest axis and are nearly dependent. Rounding their elements to double
precision perturbs $T$ by up to $10^{-8}$ near $|m|\,x_{\max} \approx 60$, and much more for
prolate spheroids, whose elements in these rows keep a cancellation that the electrostatic
removal does not reach.

scatrs estimates the error of every mode and refines the modes that need it:

1. **Estimate.** Each element's rounding is bounded by the unit roundoff times the magnitudes
   it was summed from. To first order an error $\delta Q$, $\delta P$ moves $T$ by
   $-(\delta P + T\,\delta Q)\,Q^{-1}$. scatrs applies this to random-signed perturbations at
   the bounds and random-signed vectors, with the factors already computed, which costs
   $O(K^2)$ for a mode of size $K$. The estimate is split by row: the share of the error that
   each row's rounding causes.
2. **Refine.** If the estimate exceeds a tenth of the tolerance, the rows whose share is not
   negligible (from the lowest such row up, usually the upper three quarters of the orders) are
   integrated again in double-double arithmetic (about 32 digits; {doc}`scalar-types`). Only
   what needs the extra digits is recomputed: of the three forms of each element, the one the
   double-precision pass kept, and of the regular part $P$, only the rows where it is not
   negligible next to the outgoing part. The mixed matrix is factorized in double precision, and the solution
   of $X Q = -P$ is corrected with residuals $-P - XQ$ computed in double-double (iterative
   refinement in mixed precision; Higham 2002, §12). Each sweep gains the factor
   $\mathrm{cond}(Q)\cdot 2^{-53}$, and two sweeps usually reach double-double accuracy.
3. **Double-double.** If refinement still leaves an estimate above the tolerance, the whole
   T-matrix is computed in double-double.

The paths taken for each mode are reported (`precision_paths`). For example, at $x = 15$ and
$\varepsilon = 1/3$ the double-precision T-matrix is wrong by $0.15$, and the refined one is
within $3\cdot10^{-14}$ of the double-double one after two sweeps.

## Convergence

The number of orders $N$ and of nodes $n_\theta$ are chosen as in Mishchenko's `tmatd`, with a
floor on $n_\theta$ set by the shape:

1. $N$ starts at $\lceil x_{\max} + 4x_{\max}^{1/3}\rceil + 2$ and grows by an eighth
   (at least 2), with $n_\theta = \max(2N, n_\text{shape})$, until the contributions of mode 0 to
   $\langle C_{\mathrm{ext}}\rangle$ and $\langle C_{\mathrm{sca}}\rangle$ change by less than
   `rtol` (mode 0 involves every order, and the other modes converge faster). Each step's
   error estimate counts in the change.
2. $n_\theta$ then grows by a tenth (at least 2) until they settle again.
3. The modes are solved at the final $N$ and $n_\theta$, up to a cutoff: mode $m$ involves only
   the orders $n \ge m$, so the modes above the highest order whose elements in mode 0 reach
   `rtol`/10 of the largest element of $T$ are left out (after Mishchenko's NMAX1). The cutoff
   compares amplitudes, not contributions to the cross sections: the phase matrix is bilinear
   in $T$, so a mode whose share of $\langle C_{\mathrm{sca}}\rangle$ is $\delta$ moves it by
   about $\sqrt\delta$. The size of the modes left out is estimated from the decay of the last
   two modes computed and added to the error estimate; if it exceeds `rtol`/10, every mode is
   solved.
4. If Greek coefficients up to an order must settle, one more step in $N$ and $n_\theta$ with
   every mode must leave them, and the cross sections, within `rtol`.

The floor $n_\text{shape}$ comes from the shape, not from $N$. The surface $r(\cos\theta)$ of a
spheroid is singular at $\cos\theta = \pm 1/e$ for a prolate spheroid ($e$ its eccentricity)
and at $\pm i/\sqrt{\varepsilon^2 - 1}$ for an oblate one, close to the range of integration when
the spheroid is elongated or flattened. The Bernstein ellipse through these points is the same
for $\varepsilon$ and $1/\varepsilon$, $\rho^2 = (h + 1)/(h - 1)$ with
$h = \max(\varepsilon, 1/\varepsilon)$, so the error of Gauss–Legendre quadrature in $\cos\theta$
falls by that factor per node on $[-1, 1]$ (Trefethen 2013, Theorem 19.3), whatever $N$. At a
fixed $N$ the measured errors fall by 1.205, 1.061 and 1.018 per node at $h$ = 10, 30 and 100,
against 1.222, 1.069 and 1.020, prolate and oblate alike. scatrs
starts with the nodes for $10^{-4}$`rtol` at that rate,
$n_\text{shape} = \lceil \ln(10^4/\mathtt{rtol}) / (2\ln\rho^2)\rceil$ in $0 < \theta < \pi/2$:
35 at $h = 5$ and `rtol` = $10^{-8}$, below $2N$ for most of the validated domain, but 208 at
$h = 30$ and 691 at $h = 100$, where $2N$ nodes would leave the quadrature, not the orders,
unconverged.

$N$ and $n_\theta$ have caps; reaching one raises `NotConverged` unless the unconverged result
is asked for. The cap on $N$ is twice Wiscombe's estimate at $|m|\,x_{\max}$, plus 20: the
internal field needs about $|m|\,x_{\max}$ orders on the surface, so a metal such as
$m = \sqrt{-4 + 0.5i}$ at $x_{\max} = 20$ needs $N \approx 100$, beyond twice the external
estimate. The result's error estimate is the last change of the loop (with the modes left
out), or the solve's estimate if larger.

## Random orientation

The orientation averages follow from the rank decomposition of $T$ (Mishchenko 1991). In the
helicity basis $(\mathbf M \pm \mathbf N)/\sqrt2$, the T-matrix between orders $n$ and $n'$
splits into components of rank $s = |n - n'|, \dots, n + n'$ by the Clebsch–Gordan
coefficients, and rotating the particle multiplies each by a Wigner function. The average over
orientations of a product of two amplitudes is then a sum over $s$ of products of
rank-$s$ amplitudes divided by $2s + 1$. scatrs evaluates the amplitudes for each helicity at
the nodes of a Gauss–Legendre rule in the scattering angle, forms the orientation-averaged
phase matrix, which has six independent elements ($F_{11}$, $F_{12}$, $F_{22}$, $F_{33}$,
$F_{34}$, $F_{44}$), and projects it onto the generalized spherical functions. The elements
are polynomials of degree $2N$ in $\cos\Theta$, so the projection with
$N + \lceil (L+1)/2 \rceil$ nodes is exact, and the coefficients vanish above $L = 2N$. The
cost is $O(N^4)$. The cross sections come from $T$ directly,

$$
\langle C_{\mathrm{ext}}\rangle = -\frac{2\pi}{k^2}\sum_{m=0}^{N} (2 - \delta_{m0})
  \sum_n \mathrm{Re}\bigl(T^{m,11}_{nn} + T^{m,22}_{nn}\bigr),
\qquad
\langle C_{\mathrm{sca}}\rangle = \frac{2\pi}{k^2}\sum_{m=0}^{N} (2 - \delta_{m0})
  \sum_{n,k}\sum_{i,j=1}^{2}\bigl|T^{m,ij}_{nk}\bigr|^2,
$$

$\langle C_{\mathrm{abs}}\rangle$ is their difference, the asymmetry parameter is
$\alpha_1(1)/3$, and the backscattering cross section is $4\pi F_{11}(180^\circ)$ from the
Greek expansion. The phase matrix at any angle also comes from the Greek expansion.

These formulas were checked against a direct numerical average over orientations of the
amplitude matrix, to $10^{-15}$.

## Derivatives

The derivatives with respect to $x$, $\varepsilon$, $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$ come
from one more solve at the converged $N$ and $n_\theta$: the elements are computed in dual
numbers carrying four derivatives (their values are bitwise the double-precision ones), and

$$
\frac{\partial T}{\partial p} = -\Bigl(\frac{\partial P}{\partial p}
  + T\,\frac{\partial Q}{\partial p}\Bigr) Q^{-1}
$$

reuses the factors of the solve, refined in double-double in the modes where $T$ is. The
averages are computed in the same dual numbers, so every output has its derivatives. All four
together cost 4 to 8 times the values.

## Accuracy and cost

The results are compared with an EBCM computed from its definition, with no simplification of
the integrands, in Arb ball arithmetic ({doc}`reference-suite`), and with the external codes
of {doc}`/validation/external`. Inside the validated domain every output is within a small
multiple of the larger of `rtol` and the error estimate ({doc}`/validation/accuracy`).

The cost grows like $N^4$, with $N \approx x_{\max}$, and steeply for extreme aspect ratios,
where refinement engages. At `rtol = 1e-8` on one core of an Apple M4 Pro, an oblate dust
spheroid ($m = 1.53 + 0.008i$) with $\varepsilon = 2$ takes 22 ms at $x = 5$ and 2.1 s at
$x = 15$, and with $\varepsilon = 5$ at $x = 12$, 3.6 s. Refined modes cost far more than
double-precision ones: the prolate $\varepsilon = 1/5$ at $x = 12$, near the edge of the
validated domain, takes 72 s. The quadrature runs in parallel, so on several cores these times
fall several-fold; the results do not depend on the number of threads.
