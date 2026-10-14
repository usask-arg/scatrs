# Ensembles of spheres

This page describes the methods behind {rust}`scatrs::ensemble` (Python: {class}`scatrs.Ensemble`
and {class}`scatrs.Mixture`): the
size distributions, their cut-offs, the adaptive size integration, the error estimates, the
Jacobians and the mixtures. The rustdoc of {rust}`scatrs::ensemble` is the authoritative statement of
the interface; the single-sphere quantities come from {doc}`mie`.

## Averages

<!-- synced with scatrs/src/ensemble/mod.rs rustdoc -->

Radii and wavelengths are in the same (any) unit, and $x = 2\pi r/\lambda$. Each component of
an ensemble has one refractive index per wavelength. Every output is a number average,

$$\langle C\rangle = \frac{\int C(r)\,n(r)\,dr}{\int n(r)\,dr}
= \frac{\int C(e^t)\,n(e^t)\,e^t\,dt}{\int n(e^t)\,e^t\,dt}, \qquad t = \ln r,$$

with the numerator over the Mie range $[r_\text{lo}, r_\text{hi}]$ of the distribution and the
denominator over its wider moment range (below), so cross sections are per particle, in the
square of the unit of the radii. The outputs at each wavelength are

- $\langle C_\text{ext}\rangle$, $\langle C_\text{sca}\rangle$ and $\langle C_\text{abs}\rangle$,
  the last from $Q_\text{abs}$, which {doc}`mie` computes without cancellation;
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
can push $x$ past $2\cdot10^4$; a Mie range that reaches $x > 10^6$ (`MAX_SIZE_PARAMETER`, the
largest the Mie solution accepts) is `InvalidInput` before any panel is evaluated. So are
a derivative parameter requested twice, mixture or mode weights with a non-finite sum,
`refine_order` above `MAX_REFINE_ORDER` = 1024, and a zero or non-finite $\langle
C_\text{ext}\rangle$, $\langle C_\text{sca}\rangle$ or $\langle C_\text{back}\rangle$ (for
$m = 1$, say), whose ratios are undefined. A result also says whether the integration
`converged`, how many panels it used, and its Mie ranges.


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
the ratio $S/Z$ of $S = \int C\,u\,dr$ over the Mie range and $Z = \int u\,dr$ over the moment
range (below). So every distribution is normalized over its cut-off range (the renormalization
the cut-offs need), and no $\Gamma$ functions are evaluated.

To keep powers such as $r^\alpha$ from overflowing or underflowing, $u$ is scaled to about 1 at
a reference radius $r_\text{ref}$: $r_g$ for the lognormal, the mode $(1 - 3b)\,a$ of the gamma
distribution (or $a$ when $b \ge \tfrac13$), $\bigl(\max(\alpha, 1)/(b\gamma)\bigr)^{1/\gamma}$
for the modified gamma, and for the power law the limit where its density is largest,
$r_\text{min}$ for $\alpha \ge 0$ and $r_\text{max}$ otherwise. The gradient is the partial
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
radius) and marches outward on each side in steps of $h_\text{max} = \min(\tfrac12,
\tfrac12 w)$ in $\ln r$, with $w$ the width of the density's peak in $\ln r$: $\ln\sigma_g$
for the lognormal, $\sqrt{b/(1 - 2b)}$ for the gamma distribution, $1/\sqrt{\alpha\gamma}$ for
the modified gamma with $\alpha > 0$ (else $1/\gamma$), and $\infty$ for finite support. Each
step's mass $s = \int r^k u\,dr$ is computed for every moment $r^k$ tracked. A side stops,
after at least two steps, when for every moment the last step's mass and the geometric bound on
the rest, $s/(1 - q)$ in all with $q < 1$ the ratio of the last two steps' masses, is at most
$\epsilon$ of the mass found so far. A side that needs more than 400 steps is `InvalidInput`:
the distribution is too broad.

- **The Mie range** bounds only the area ($k = 2$) tail, at $\epsilon = \text{rtol}/10$: it
  bounds the extinction of large spheres, and the cross sections of small spheres fall faster
  than $r^2$. The tiny spheres of a long number tail need no Mie solutions.
- **The moment range** bounds the number ($k = 0$), area and volume ($k = 3$) tails, at
  $\epsilon = \max(10^{-3}\,\text{rtol}, 10^{-16})$: without Mie solutions a far smaller tail
  is cheap, and the derivatives weight the tails by about $(\partial\ln u/\partial\theta)^2$.
  On this range, the normalization $Z = \int u\,dr$ and the moments $\langle r^2\rangle$ and
  $\langle r^3\rangle$ (so $r_\text{eff}$, the mean area and the mean volume) are integrated
  separately, without Mie solutions: adaptive G7–K15 panels (below) refined until each
  integral, and each of its derivatives, meets rtol. They do not depend on the wavelength.
- **The extension rule.** The area tail is not conservative for every integrand: when the bulk
  of the distribution is in the Rayleigh regime and the tail is not, $C_\text{ext}$ grows like
  $r^3$ to $r^6$, faster than the area, and the derivatives carry the extra weight of
  $\partial\ln u/\partial\theta$. So once pass 1 (below) meets its targets, the range is
  extended by $e = \min(\tfrac12, w)$ in $\ln r$ at any end where a refined integrand (a value
  or any derivative block) per unit $\ln r$ at the end node, times $e/\tfrac12$, exceeds a tenth
  of its target, and pass 1 continues. The θ-derivative integrals can cancel, so for their
  blocks the target here is $\text{rtol}\cdot\max(\lvert\int\partial_\theta u\,C\,dr\rvert,
  10^{-3}A)$, with $A$ the absolute integral of their pass-1 target. At most 100 extensions per
  side; an extension that would exceed `max_panels` ends the integration with
  `converged = false`.

The omitted tails, at most rtol/10 of the area and less of the number, move an average by at
most about $2\cdot\text{rtol}/10$ of its value, a truncation term that every error estimate
includes (below). The integration ranges are held fixed when differentiating.

## Size integration

<!-- synced with scatrs/src/ensemble/engine.rs rustdoc -->

Per wavelength and per component (and per mode of a multimodal distribution, which is
integrated as a component of its own), in $t = \ln r$:

- **Rule.** Gauss–Kronrod G7–K15 panels, with QUADPACK's nodes and weights. The seven Gauss
  nodes are among the fifteen Kronrod nodes, so every integral gets $\lvert K_{15} -
  G_7\rvert$ on every panel without another Mie solution.
- **Initial panels.** At most $h_\text{max}$ wide in $t$ (the cut-offs' step: ½, or half the
  peak's width) and at most 20 wide in size parameter (the broad interference structure of the
  efficiencies has a period of about $\pi/\lvert m - 1\rvert$ in $x$), with the radii of a
  `Tabulated` distribution as panel edges. If that would make more than `max_panels` panels,
  the range is split into about `max_panels` equal panels in $t$ instead, at least one between
  consecutive edges.

(ensemble-pass-1)=
### Pass 1: refinement

Pass 1 evaluates only the efficiencies, $O(N)$ per radius for $N$ terms (and, with
`refine_order`, each sphere's Greek coefficients). It integrates, and refines on:

| Integral | Target |
|---|---|
| $\int C_\text{ext}u$, $\int C_\text{sca}u$ | rtol of itself |
| $\int C_\text{abs}u$ | rtol of $\int C_\text{ext}u$ |
| $\int C_\text{sca}g\,u$ | rtol of $\int C_\text{sca}u$ |
| for each requested distribution parameter $\theta$, each of the above with $\partial_\theta u$ in place of $u$ | rtol of $\int\lvert\partial_\theta u\,C\rvert\,dr$ ($C_\text{ext}$'s for $C_\text{abs}$) |
| with derivatives with respect to $m$: $\int \partial_m C\,u$ for $C_\text{ext}$, $C_\text{sca}$, $C_\text{sca}g$, $C_\text{abs}$ | rtol of $\int\lvert u\,\partial_m C\rvert\,dr$ ($C_\text{ext}$'s for $C_\text{abs}$) |
| with `refine_backscatter`: $\int C_\text{back}u$ (and its derivatives) | rtol of the larger of itself and $\int C_\text{sca}u$ (derivatives: as above) |
| with `refine_order` $L_r \le$ `MAX_REFINE_ORDER` = 1024 (Rust only): $\int \pi r^2\alpha^l u$ for every Greek coefficient up to $L_r$ | rtol of $\int C_\text{sca}u$ (derivatives: as above) |

The targets of derivatives are relative to the
integrals of their absolute values because derivatives can cancel to near 0. $\langle
C_\text{abs}\rangle$ is refined to rtol of $\langle C_\text{ext}\rangle$, the accuracy of
$C_\text{ext} - C_\text{sca}$, and its derivatives against the matching $C_\text{ext}$
integrals: relative to $C_\text{abs}$ itself, the absorption resonances of weakly absorbing
large spheres took the whole panel budget (for a lognormal with $r_g = 1$, $\sigma_g = 2$ and
$m = 1.33 + 10^{-4}i$ at $\lambda = 0.5$, unconverged at 4096 panels, against 2630 converged).
$\langle C_\text{back}\rangle$ is refined only on request: backscattering by large, weakly
absorbing spheres is a sum of narrow glory ripples (hence the target relative to $\langle
C_\text{sca}\rangle$, as for single spheres), and refining on it can take far more panels: for
the same lognormal, unconverged at 4096 panels with it. Without it, $\langle
C_\text{back}\rangle$ is integrated on the panels the other quantities chose, and its estimate
is indicative (see {ref}`ensemble-accuracy`).


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

The Greek coefficients of the ensemble are those of the averaged phase matrix. The
`GreekProjector` of the Mie module computes them exactly from tables of a phase matrix at its Gauss–Legendre
nodes: for a sphere of $N$ terms, coefficients up to order $L$ need
$n_q = N + \lceil(L + 1)/2\rceil$ nodes. The tables $\pi r^2 F$ ($C_\text{sca}$ times the
normalized phase matrix) are integrated over the size distribution at those nodes and
projected once.

One shared projector, with $n_q$ set by the largest $N$, would make every small sphere pay for
the largest: a sphere's amplitudes cost $O(N\,n_q)$. So the panels are grouped into bands by
the largest number of terms $N$ at their nodes: $N$ belongs to the band of the first value
$\ge N$ of the sequence $1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 24, \ldots$ (each $\lceil
1.25\times\rceil$ the last). A band with largest $N_b$ has its own projector, to order $L_b =
\min(L, 2N_b)$ with $n_q = N_b + \lceil(L_b + 1)/2\rceil$ nodes, which is exact for every sphere
in it (a sphere's coefficients vanish above $2N$). Every node of a panel fills the tables of the
panel's band, so each band's tables are smooth within a panel and their $\lvert K_{15} -
G_7\rvert$ measures the integration error. The coefficients of the ensemble are the sum of the
bands' projections. With `GreekOrder::All` (Python `greek="all"`), $L = 2N_\text{max}$ for the
largest $N$ at any node.

### The tables as matrix products

A band's tables need $S_1$ and $S_2$ of every node at the band's $n_q$ cosines, the bulk of the
cost of ensembles with Greek coefficients. With $w_n = (2n + 1)/(n(n + 1))$,
$S_1 = \sum_n w_n(a_n\pi_n + b_n\tau_n)$ and $S_2 = \sum_n w_n(a_n\tau_n + b_n\pi_n)$. The
functions $w_n\pi_n$ and $w_n\tau_n$ at a band's nodes are the same for every sphere, so each
band tabulates them once, at its non-negative nodes. Since $\pi_n(-\mu) =
(-1)^{n+1}\pi_n(\mu)$ and $\tau_n(-\mu) = (-1)^n\tau_n(\mu)$, with $G_E$ the table of
$w_n\pi_n$ for odd $n$ and $w_n\tau_n$ for even $n$, $G_O$ the other two, $u_n = a_n$ (odd $n$)
or $b_n$ (even $n$) and $v_n$ the other coefficient,

$$S_1(\pm\mu) = G_E u \pm G_O v,\qquad S_2(\pm\mu) = G_E v \pm G_O u$$

at the non-negative nodes $\mu$ (the Gauss–Legendre nodes are exactly symmetric). For the 15
nodes of a panel these are two real matrix products, $G_E$ and $G_O$ ($n_q/2 \times N$) times
the nodes' coefficients ($N \times 60$; 120 with the $m$-derivatives $\mathrm{d}a_n/\mathrm{d}m$,
$\mathrm{d}b_n/\mathrm{d}m$). The products cost $O(N\,n_q)$, like sums sphere by sphere, but
run far faster, and the tables agree with the per-sphere phase matrix to $10^{-12}$ of its
largest value (a unit test).

The sums over the nodes are matrix products too: each panel's sums for every part and block
(values, each $\theta$, $\mathrm{Re}\,m$, $\mathrm{Im}\,m$) are the weights at the nodes
(Kronrod and Gauss) times the nodes' values. The projection likewise: a band's matrices
$\tfrac{2l + 1}{2}w_k d_l(\mu_k)$ times the tables of every mixture (or batch distribution) and
derivative at once, so the generalized spherical functions are computed once per band.

## Error estimates

Every output has an estimate of the error of the size integration and of the cut-offs, of the
same shape. The Mie solutions' own rounding errors, bounded separately ({doc}`mie`), are not
included; at the tolerances the ensembles accept they are far smaller.

- **Integrals.** $E = \sum_\text{panels}\lvert K_{15} - G_7\rvert$, the usual estimate of
  the error of the $K_{15}$ sum, which is pessimistic once the panels resolve the integrand.
  It is not a proof: it relies on the panels resolving the integrand, which pass 1 ensures for
  the refined quantities only.
- **Averages.** For $v = S/Z$, with $Z = \int u$ from the moment integration and $E_Z$ its
  estimate, to first order $e_v = (E_S + \lvert v\rvert E_Z)/Z + 2\cdot\text{rtol}/10\cdot
  \lvert v\rvert$, the last term the cut-offs' (for distributions with unbounded support only).
- **Greek coefficients.** The per-node sums of the panels' $\lvert K_{15} - G_7\rvert$ tables,
  $\lvert\Delta T(\mu_k)\rvert$, are projected with absolute values:
  $\tfrac{2l + 1}{2}\sum_k w_k\lvert\Delta T(\mu_k)\rvert\lvert d_l(\mu_k)\rvert$ for
  $\alpha_1$, $\alpha_4$, $\beta_1$ and $\beta_2$ (with the d-function of each), and half the
  sum of the bounds of $\alpha_2 \pm \alpha_3$ for $\alpha_2$ and $\alpha_3$
  (`GreekProjector::project_bounds`). By the triangle inequality this bounds the error of the
  projection, given the table estimates. Since the bands are chosen per panel, it is about as
  tight as the estimates of the integrals: in {doc}`/examples/ensemble` the estimate of the
  unnormalized $\alpha_1^0$, which equals $\langle C_\text{sca}\rangle$, is less than twice
  that of $\langle C_\text{sca}\rangle$.
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

The tests check every scalar output, $\beta_1^l$ up to order 6 and $F_{12}$ at two cosines
against central differences of the whole computation at rtol $10^{-11}$, for the lognormal and
gamma parameters, the modified gamma's $b$ and $\gamma$, the power-law exponent, the
monodisperse radius, $\mathrm{Re}\,m$, $\mathrm{Im}\,m$, a mixture weight, the
$\mathrm{Re}\,m$ of a mixture's second component, a mode weight and a mode's width.

## Mixtures

{rust}`compute_mixture <scatrs::ensemble::compute_mixture>` (Python {class}`scatrs.Mixture`)
averages over external mixtures: components, each a distribution, its refractive indices and a
weight $w_i \ge 0$ (a relative number concentration; their sum finite and positive).
{rust}`compute <scatrs::ensemble::compute>` is a mixture of one component, without the weight
parameter. Cross sections and moments are averaged with the number fractions $w_i/\sum w$. The phase-matrix
tables, in $C_\text{sca}$ units, are averaged with the same fractions, so the phase matrices
are averaged with $w_i\langle C_\text{sca}\rangle_i$, before one projection of all components.
A multimodal distribution is a mixture of its modes with one refractive index.

## Batches

{rust}`compute_batch <scatrs::ensemble::compute_batch>` averages over each of many distributions of one material, as an optics
database needs: a grid of median radii and widths, say. In Python, an {class}`scatrs.Ensemble`
whose distribution has array-valued parameters is such a batch: its parameters broadcast
together into a grid, flattened in C order, and the results have the grid's axes before the
wavelengths'. So is an ensemble whose `size` is a list of distributions of any kinds (many
`Tabulated` tables, say), with one leading axis. The distributions share their size nodes: one set of panels per wavelength covers the
union of their Mie ranges, with the ends of finite supports and `Tabulated` radii as panel
edges, and is refined until every distribution meets its own targets of pass 1 (a panel splits
if it exceeds its share of any distribution's target) and extended until every distribution's
integrands are negligible at both ends. Each distribution's largest initial step (half its
peak's width) applies within its own range, and the extensions at an end take the widths of
the distributions that want them, so a narrow member does not refine the whole range. Each node's Mie solution and tables then serve every distribution, weighted by each
density. The normalizations and moments are integrated per distribution, without Mie
solutions, and monodisperse distributions on their own.

Each distribution has its own error estimates and `converged` flag. A distribution in a batch is
at least as refined as alone, so its values differ from `compute`'s within the error estimates
(a test checks both against references at rtol $10^{-10}$). A batch computes values only: the
Jacobians come from `compute`. For the database grid of {doc}`/validation/sasktran2` (rtol
$10^{-3}$, Greek coefficients to order 63, one thread, measured separately), the batch costs
about 1.6 times its broadest distribution alone, against 4 times for the 50 distributions one
by one. Memory grows with the distributions: each holds its sums over every band's tables, about
2.5 MB per distribution and wavelength in flight for the grid's shortest wavelength (2 GB for
800 distributions on one thread), and the sums of the panels in flight are capped at about
64 MB.

## Parallelism and determinism

Wavelengths, the panels of each pass-1 round and chunks of four pass-2 panels run in parallel
on the current rayon pool (in Rust, a caller can `install` its own); each task owns its Mie
workspace, and the matrix products run on one thread within a task. Refinement decisions depend
only on the values, and the sums are reduced in a fixed order (the chunks' sums in windows of
32 chunks, which bounds the memory of large batches), so results are bitwise identical for any
number of threads (tested with pools of 1 to 5, mixtures and batches included). Errors are too: the parallel tasks all finish, and the first error in wavelength, part and
panel order is returned. Python calls release the GIL.

## Cost

Pass 1, and pass 2 without the Greek coefficients, are cheap. The Greek coefficients cost
more: the exact coefficients need each node's amplitudes at about $N + L/2$ cosines,
$O(N^2)$ for large spheres, as matrix products (above). Measured on an Apple M4 Pro
(12 cores) per wavelength (best of 3), a lognormal with $r_g = 1$, $\sigma_g = 2$ and
$m = 1.33 + 10^{-4}i$, at rtol $10^{-6}$ and the other options default:

| $\lambda$ | panels | cross sections, $g$ and moments | with Greek coefficients to order 32 |
|---|---|---|---|
| 1 | 1461 | 0.037 s | 0.093 s |
| 0.5 | 2630 | 0.11 s | 0.31 s |
| 0.3 | 4092 | 0.20 s | 0.90 s |

All three converge. A narrow fine-mode aerosol takes milliseconds ({doc}`/examples/ensemble`).
{doc}`/validation/sasktran2` compares the speed and accuracy with sasktran2's at matched
accuracy, for one distribution and for a database grid.
