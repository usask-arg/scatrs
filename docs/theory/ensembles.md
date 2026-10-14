# Ensembles of spheres

This page describes the methods behind `scatrs::ensemble` (Python: {mod}`scatrs.ensemble`): the
size distributions, their cut-offs, the adaptive size integration, the error estimates, the
Jacobians and the mixtures. The rustdoc of `scatrs::ensemble` is the authoritative statement of
the interface; the single-sphere quantities come from {doc}`mie`.

## Averages

<!-- synced with scatrs/src/ensemble/mod.rs rustdoc -->

Radii and wavelengths are in the same (any) unit, and $x = 2\pi r/\lambda$. Each component of
an ensemble has one refractive index per wavelength. Every output is a number average over the
integration range $[r_\text{lo}, r_\text{hi}]$ of the distribution (below):

$$\langle C\rangle = \frac{\int C(r)\,n(r)\,dr}{\int n(r)\,dr}
= \frac{\int C(e^t)\,n(e^t)\,e^t\,dt}{\int n(e^t)\,e^t\,dt}, \qquad t = \ln r,$$

so cross sections are per particle, in the square of the unit of the radii. The outputs at each
wavelength are

- $\langle C_\text{ext}\rangle$, $\langle C_\text{sca}\rangle$ and $\langle C_\text{abs}\rangle$,
  the last from $Q_\text{abs}$, which phase 1a computes without cancellation;
- the single-scattering albedo $\omega = \langle C_\text{sca}\rangle/\langle C_\text{ext}\rangle$
  and the asymmetry parameter $g = \langle C_\text{sca}\,g\rangle/\langle C_\text{sca}\rangle$;
- $\langle C_\text{back}\rangle$, Bohren & Huffman's backscattering cross section ($4\pi$ times
  $d\sigma/d\Omega$ at 180°), the lidar ratio $4\pi\langle C_\text{ext}\rangle/\langle
  C_\text{back}\rangle$, and the linear depolarization ratio, exactly 0 for spheres;
- the moments $r_\text{eff} = \langle r^3\rangle/\langle r^2\rangle$, $\langle\pi r^2\rangle$
  and $\langle 4\pi r^3/3\rangle$;
- on request, the Greek coefficients up to an order $L$ or to every order, and the phase matrix
  ($F_{11}$, $F_{12}$, $F_{33}$, $F_{34}$) at given cosines, normalized
  ($\alpha_1^0 = 1$, $\tfrac12\int F_{11}\,d\mu = 1$) or per particle ($\alpha_1^0 =
  \langle C_\text{sca}\rangle$, the phase matrix times $\langle C_\text{sca}\rangle$).

Each output comes with an error estimate of the same shape and, on request, derivatives with
their own error estimates. The relative tolerance rtol of the size integration is in $[10^{-13},
0.1]$ (default $10^{-6}$). Every size node must be in the validated domain of the Mie solution
({doc}`mie`) unless `allow_unvalidated` is set; otherwise the call fails with
`OutsideValidatedDomain`, naming the wavelength and the radius. The tails of broad distributions
can push $x$ past $2\cdot10^4$. A result also says whether the integration `converged`, how many
panels it used, and its integration ranges.


## Size distributions

<!-- synced with scatrs/src/ensemble/distribution.rs rustdoc -->

| Distribution | Number density $n(r) \propto$ | Parameters |
|---|---|---|
| `LogNormal` | $\exp\bigl(-(\ln r - \ln r_g)^2/(2\ln^2\sigma_g)\bigr)/r$ | `median_radius` $r_g > 0$, `width` $\sigma_g > 1$ |
| `Gamma` (Hansen & Travis) | $r^{(1-3b)/b}\exp\bigl(-r/(ab)\bigr)$ | `effective_radius` $a > 0$, `effective_variance` $0 < b < \tfrac12$ |
| `ModifiedGamma` (Deirmendjian) | $r^\alpha\exp(-b\,r^\gamma)$ | `alpha` $> -1$, `b` $> 0$, `gamma` $> 0$ |
| `PowerLaw` | $r^{-\alpha}$ on $[r_\text{min}, r_\text{max}]$ | `exponent` $\alpha$ (the limits `min_radius`, `max_radius` are fixed) |
| `Monodisperse` | $\delta(r - r_0)$ | `radius` $r_0 > 0$ |
| `Tabulated` | linear between the points $(r_i, n_i)$, 0 outside | none (radii increasing, densities $\ge 0$) |
| `MultiModal` | $\sum_j w_j\,n_j(r)$, each mode normalized | each mode's parameters, and its weight $w_j > 0$ |

The parameter names are those of the Rust variants' fields and of the Python dataclasses; the
derivatives use them too (`"median_radius"`, and `"mode0.width"` for a mode). The modes of a
multimodal distribution cannot themselves be multimodal. Invalid parameters (non-finite values,
$\sigma_g \le 1$, $b$ outside $(0, \tfrac12)$, $r_\text{min} \ge r_\text{max}$, a table that does
not increase or has negative densities, mode weights that are not positive) are
`InvalidInput`.

### Numeric normalization

The engine never uses a normalization constant. Each distribution provides an unnormalized
density $u(r; \theta)$ and its exact gradient $\partial u/\partial\theta$, and every average is
the ratio $\int C\,u\,dr \big/ \int u\,dr$ over the integration range. So every distribution
is normalized over its cut-off range (the renormalization the cut-offs need), and no $\Gamma$
functions are evaluated.

To keep powers such as $r^\alpha$ from overflowing or underflowing, $u$ is scaled to about 1 at
a reference radius $r_\text{ref}$: $r_g$ for the lognormal, the mode $(1 - 3b)\,a$ of the gamma
distribution (or $a$ when $b \ge \tfrac13$), $\bigl(\max(\alpha, 1)/(b\gamma)\bigr)^{1/\gamma}$
for the modified gamma, and $r_\text{min}$ for the power law. The gradient is the partial
derivative with $r_\text{ref}$ held fixed. That choice does not affect any derivative: the
scaling multiplies $u$ by a factor $K(\theta)$ independent of $r$, and in


$$\frac{\partial}{\partial\theta}\frac{\int C\,u\,dr}{\int u\,dr}
= \frac{\int C\,\partial_\theta u\,dr - \langle C\rangle\int\partial_\theta u\,dr}{\int u\,dr}$$

the terms in $K'(\theta)$ cancel. Closed-form moments $\langle r^k\rangle$ exist for every
distribution except `Tabulated` (`SizeDistribution::moment`); the tests use them.

## Cut-offs

<!-- synced with scatrs/src/ensemble/range.rs rustdoc -->

`PowerLaw` and `Tabulated` are integrated over their support, and `Monodisperse` is the single
sphere, without integration. The others have unbounded support, and their range starts at a
characteristic radius ($r_g$, the effective radius $a$, or the modified gamma's reference
radius) and marches outward in steps of ½ in $\ln r$ on each side. Each step's mass
$s = \int r^k u\,dr$ is computed for every moment $r^k$ tracked. A side stops, after at least
two steps, when for every moment the last step's mass and the geometric bound on the rest,
$s/(1 - q)$ in all with $q < 1$ the ratio of the last two steps' masses, is at most rtol/10 of
the mass found so far. A side that needs more than 400 steps is `InvalidInput`: the
distribution is too broad.

- **The Mie range** bounds the number ($k = 0$) and area ($k = 2$) tails. Renormalizing over the
  range moves every per-particle average by the omitted number fraction, so the number tail
  must be bounded; the area tail bounds the extinction of large spheres.
- **The moment range** also bounds the volume tail ($k = 3$). Bounding it for the Mie range
  pushed the range of a lognormal with $\sigma_g = 2$ to $x \approx 5000$, so $\langle
  r^2\rangle$ and $\langle r^3\rangle$ are integrated separately, without Mie solutions, on this
  wider range: adaptive G7–K15 panels (below) refined until each moment, and each of its
  derivatives, meets rtol. These moments do not depend on the wavelength.
- **The extension rule.** The area tail is not conservative when the bulk of the distribution
  is in the Rayleigh regime and the tail is not: there $C_\text{ext}$ grows like $r^3$ to
  $r^6$, faster than the area. So once pass 1 (below) meets its targets, the range is extended
  by a further step of ½ in $\ln r$ at any end where the extinction integrand per unit $\ln r$
  at the end node, times ½, exceeds rtol/10 of the integral of $C_\text{ext}$, and pass 1
  continues; at most 100 extensions per side.

The omitted tails, at most rtol/10 of each moment bounded, move an average by at most about
$2\cdot\text{rtol}/10$ of its value, a truncation term that every error estimate includes
(below). The integration ranges are held fixed when differentiating.

## Size integration

<!-- synced with scatrs/src/ensemble/engine.rs rustdoc -->

Per wavelength and per component (and per mode of a multimodal distribution, which is
integrated as a component of its own), in $t = \ln r$:

- **Rule.** Gauss–Kronrod G7–K15 panels, with QUADPACK's nodes and weights. The seven Gauss
  nodes are among the fifteen Kronrod nodes, so every integral gets $\lvert K_{15} -
  G_7\rvert$ on every panel without another Mie solution.
- **Initial panels.** At most ½ wide in $t$ and at most 20 wide in size parameter (the broad
  interference structure of the efficiencies has a period of about $\pi/\lvert m - 1\rvert$ in
  $x$), with the radii of a `Tabulated` distribution as panel edges. If that would make more
  than `max_panels` panels, the range is split into `max_panels` equal panels in $t$ instead.

(ensemble-pass-1)=
### Pass 1: refinement

Pass 1 evaluates only the efficiencies, $O(N)$ per radius for $N$ terms (and, with
`refine_order`, each sphere's Greek coefficients). It integrates, and refines on:

| Integral | Target |
|---|---|
| $\int u$ (the normalization) | rtol of itself |
| $\int C_\text{ext}u$, $\int C_\text{sca}u$ | rtol of itself |
| $\int C_\text{sca}g\,u$ | rtol of $\int C_\text{sca}u$ |
| for each requested distribution parameter $\theta$, each of the above with $\partial_\theta u$ in place of $u$ | rtol of the integral of its absolute value |
| with derivatives with respect to $m$: $\int \partial_m C\,u$ for $C_\text{ext}$, $C_\text{sca}$, $C_\text{sca}g$ | rtol of the integral of its absolute value |
| with `refine_backscatter`: $\int C_\text{back}u$ (and its derivatives) | rtol of the larger of itself and $\int C_\text{sca}u$ |
| with `refine_order` $L_r$ (Rust only): $\int \pi r^2\alpha^l u$ for every Greek coefficient up to $L_r$ | rtol of $\int C_\text{sca}u$ |

The derivatives with respect to $m$ are refined because, without it, their error estimates fell
short of their true errors by a factor of 3. The targets of derivatives are relative to the
integrals of their absolute values because derivatives can cancel to near 0. $\langle
C_\text{back}\rangle$ is refined only on request: backscattering by large, weakly absorbing
spheres is a sum of narrow glory ripples (hence the target relative to $\langle
C_\text{sca}\rangle$, as for single spheres), and refining on it can take far more panels: for a
lognormal with $r_g = 1$, $\sigma_g = 2$ and $m = 1.33 + 10^{-4}i$ at $\lambda = 0.5$, 2847
panels without it, and unconverged at 4096 with it. Without it, $\langle C_\text{back}\rangle$
is integrated on the panels the other quantities chose, and its estimate is indicative (see
{ref}`ensemble-accuracy`).


Pass 1 works in rounds. Each round evaluates the new panels, sums every integral and its
estimate $\sum_\text{panels}\lvert K_{15} - G_7\rvert$, and stops if every estimate meets its
target (then applies the extension rule). Otherwise, every panel whose estimate of a quantity
still above its target exceeds its share of that target (the target divided by the number of
panels) is halved in $t$, worst first, as many as `max_panels` (default 4096) allows; panels
narrower than $10^{-10}$ are not split. If no panel can be split, the integration stops with
`converged = false`. That is not an error: the values are returned with their estimates.

### Pass 2: outputs

At the K15 nodes of the final panels, pass 2 computes every output once: the efficiencies,
$Q_\text{abs}$ and $Q_\text{back}$, the phase matrix at the requested cosines, the tables of the
Greek coefficients (below) and, if requested, the derivatives of all of them with respect to
$m$. Each is integrated with the Kronrod and the Gauss weights, so every output has its
$\lvert K_{15} - G_7\rvert$ estimate.

### Greek coefficients and projector bands

The Greek coefficients of the ensemble are those of the averaged phase matrix. Phase 1b's
`GreekProjector` computes them exactly from tables of a phase matrix at its Gauss–Legendre
nodes: for a sphere of $N$ terms, coefficients up to order $L$ need
$n_q = N + \lceil(L + 1)/2\rceil$ nodes. The tables $\pi r^2 F$ ($C_\text{sca}$ times the
normalized phase matrix) are integrated over the size distribution at those nodes and
projected once.

One shared projector, with $n_q$ set by the largest $N$, made every small sphere pay for the
largest: a sphere's amplitudes cost $O(N\,n_q)$, and a broad distribution took 20 s per
wavelength. So the nodes are grouped into bands by their number of terms: $N$ belongs to the
band of the first value $\ge N$ of the sequence $1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 24, \ldots$
(each $\lceil 1.25\times\rceil$ the last). A band with largest $N_b$ has its own projector, to
order $\min(L, 2N_b)$ with $N_b + \lceil(\min(L, 2N_b) + 1)/2\rceil$ nodes, which is exact for
every sphere in it (a sphere's coefficients vanish above $2N$). The coefficients of the
ensemble are the sum of the bands' projections. With `GreekOrder::All` (Python
`greek="all"`), $L = 2N_\text{max}$ for the largest $N$ at any node.

## Error estimates

Every output has an estimate of the error of the size integration and of the cut-offs, of the
same shape. The Mie solutions' own rounding errors, bounded separately ({doc}`mie`), are not
included; at the tolerances the ensembles accept they are far smaller.

- **Integrals.** $E = \sum_\text{panels}\lvert K_{15} - G_7\rvert$, the usual estimate of
  the error of the $K_{15}$ sum, which is pessimistic once the panels resolve the integrand.
  It is not a proof: it relies on the panels resolving the integrand, which pass 1 ensures for
  the refined quantities only.
- **Averages.** For $v = S/S_0$, with $S_0 = \int u$, to first order
  $e_v = (E_S + \lvert v\rvert E_0)/S_0 + 2\cdot\text{rtol}/10\cdot\lvert v\rvert$, the last
  term the cut-offs' (for distributions with unbounded support only).
- **Greek coefficients.** The per-node sums of the panels' $\lvert K_{15} - G_7\rvert$ tables,
  $\lvert\Delta T(\mu_k)\rvert$, are projected with absolute values:
  $\tfrac{2l + 1}{2}\sum_k w_k\lvert\Delta T(\mu_k)\rvert\lvert d_l(\mu_k)\rvert$ for
  $\alpha_1$, $\alpha_4$, $\beta_1$ and $\beta_2$ (with the d-function of each), and half the
  sum of the bounds of $\alpha_2 \pm \alpha_3$ for $\alpha_2$ and $\alpha_3$
  (`GreekProjector::project_bounds`). By the triangle inequality this bounds the error of the
  projection, given the table estimates. It is a loose bound: within a panel the band of a node
  changes with its radius, so each band's table is a step function of $r$, and its
  $\lvert K_{15} - G_7\rvert$ is of the order of the panel's whole contribution. In
  {doc}`/examples/ensemble` the estimate of the unnormalized $\alpha_1^0$, which equals
  $\langle C_\text{sca}\rangle$, is more than $10^3$ times that of $\langle C_\text{sca}\rangle$.
- **Ratios.** $\omega$, $g$, the lidar ratio, $r_\text{eff}$ and the normalized forms are
  propagated to first order: $e(a/b) = (e_a + \lvert a/b\rvert e_b)/\lvert b\rvert$, and for
  the derivatives $e(\partial(a/b)) = (e_{\partial a} + \lvert a/b\rvert e_{\partial b} +
  \lvert\partial(a/b)\rvert e_b + e(a/b)\lvert\partial b\rvert)/\lvert b\rvert$.
- **Mixtures** add the components' estimates with their weights.

A monodisperse component is exact: its estimates are 0. {ref}`ensemble-accuracy` measures how
well the estimates bound the true errors.

## Jacobians

Only the requested parameters are differentiated; every output, its error estimate included,
has a derivative.

- **Distribution parameters.** $\partial_\theta\int C\,u$ through $\partial_\theta u$ at the
  same nodes, with the renormalization by the quotient rule above. The integration range and
  panels are held fixed within a call.
- **Monodisperse radius.** $\partial(\pi r^2 Q)/\partial r = 2\pi r\,Q + \pi r^2\,(2\pi/\lambda)
  \,\partial Q/\partial x$, from the single-sphere derivatives with respect to $x$, and likewise
  for the phase matrix.
- **$\mathrm{Re}\,m$ and $\mathrm{Im}\,m$** of each component, at each wavelength: from the
  single-sphere efficiency Jacobian and phase-matrix derivatives, integrated with the same
  weights ($m$ does not change the density, so no renormalization term).
- **Weights.** A component $i$ (or mode $j$) enters with the number fraction
  $f = (w_i/\sum w)(v_j/\sum v)$, with $v_j$ the mode weights; its derivatives with respect to
  each weight follow from that product.
- **Normalized and ratio outputs** by the quotient rule, as for the error estimates.

The tests check every scalar output and $\beta_1^l$ up to order 6 against central differences
of the whole computation at rtol $10^{-11}$, for the lognormal and gamma parameters, the
monodisperse radius, $\mathrm{Re}\,m$, $\mathrm{Im}\,m$, a mixture weight and a mode weight.

## Mixtures

`compute_mixture` averages over external mixtures: components, each a distribution, its
refractive indices and a weight $w_i \ge 0$ (a relative number concentration; not all 0).
`compute` is a mixture of one component, without the weight parameter. Cross sections and
moments are averaged with the number fractions $w_i/\sum w$. The phase-matrix tables, in
$C_\text{sca}$ units, are averaged with the same fractions, so the phase matrices are
averaged with $w_i\langle C_\text{sca}\rangle_i$, before one projection of all components. A
multimodal distribution is a mixture of its modes with one refractive index.

## Parallelism and determinism

Wavelengths, the panels of each pass-1 round and chunks of four pass-2 panels run in parallel
on the current rayon pool (in Rust, a caller can `install` its own); each task owns its Mie
workspace. Refinement decisions depend only on the values, and the sums are reduced in a fixed
order, so results are bitwise identical for any number of threads (tested with pools of 1 and
4). Python calls release the GIL.

## Cost

Pass 1, and pass 2 without the Greek coefficients, are cheap. The Greek coefficients are not:
the exact coefficients need each node's amplitudes at about $N + L/2$ cosines, $O(N^2)$ for
large spheres. Measured on an Apple M4 Pro for one wavelength, a lognormal with $r_g = 1$,
$\sigma_g = 2$ and $m = 1.33 + 10^{-4}i$, at rtol $10^{-6}$:


| $\lambda$ | panels | cross sections, $g$ and moments | with Greek coefficients to order 32 |
|---|---|---|---|
| 1 | 1553 | 0.04 s | 1.1 s |
| 0.5 | 2847 | 0.11 s | 5.9 s |
| 0.3 | 4096 | 0.23 s | 22.6 s |

A narrow fine-mode aerosol takes milliseconds ({doc}`/examples/ensemble`). Clebsch–Gordan Greek
coefficients ($O(N\,L)$ per sphere) and $\pi_n$, $\tau_n$ tables shared within a band are
planned to reduce the cost of pass 2; see {doc}`/limitations`.
