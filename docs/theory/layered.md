# Layered spheres

<!-- synced with scatrs/src/layered/kernel.rs and scatrs/src/mie/kernel.rs -->

A layered sphere is a core with any number of concentric shells. Its scattering has the same
form as a homogeneous sphere's: coefficients $a_n$ and $b_n$, and from them every output of
{doc}`mie` (efficiencies, amplitudes, phase matrix and Greek coefficients), computed by the
same code. Only the coefficients differ. This chapter describes how scatrs computes them, why
each choice was made, and how accurate the results are. A coated sphere is the case of two
layers.

## Geometry and conventions

The layers are numbered from the core, $l = 0, \dots, L - 1$. Layer $l$ lies between the radii
$r_{l-1}$ and $r_l$ (with $r_{-1} = 0$) and has the refractive index $m_l$ relative to the
medium. Its outer surface has the size parameter $x_l = 2\pi r_l/\lambda$, so
$0 < x_0 \le x_1 \le \dots \le x_{L-1}$; a layer of zero thickness is allowed. The outer size
parameter $x_L = x_{L-1}$ normalizes the efficiencies and the phase matrix, and sets the
number of terms. The conventions are those of {doc}`conventions`: time dependence
$e^{-i\omega t}$, $\mathrm{Im}\,m \ge 0$ for absorption, and the Riccati–Bessel functions
$\psi_n$, $\chi_n$ and $\xi_n = \psi_n - i\chi_n$.

In layer $l$ the field of order $n$ is a combination of $\psi_n(m_l k r)$ and a second
solution. Matching the tangential fields at each interface carries the information of the
inner layers outward. At the outer surface it becomes two effective logarithmic derivatives,
$H_a$ and $H_b$, and the coefficients are the homogeneous sphere's,

$$
a_n = \frac{(H_a/m_{L-1} + n/x_L)\,\psi_n(x_L) - \psi_{n-1}(x_L)}
           {(H_a/m_{L-1} + n/x_L)\,\xi_n(x_L) - \xi_{n-1}(x_L)},
\qquad
b_n = \frac{(m_{L-1} H_b + n/x_L)\,\psi_n(x_L) - \psi_{n-1}(x_L)}
           {(m_{L-1} H_b + n/x_L)\,\xi_n(x_L) - \xi_{n-1}(x_L)},
$$

with $H_a$ and $H_b$ in place of the sphere's $D_n(mx)$. A single layer is a sphere, and
scatrs then calls the sphere's code itself, so that its results are bitwise those of
{mod}`scatrs.mie`.

## The recursion

With $D_n = \psi_n'/\psi_n$, a second solution $\zeta_n$ with $D^\zeta_n = \zeta_n'/\zeta_n$ and
Wronskian $W = \psi_n\zeta_n' - \psi_n'\zeta_n$, and for layer $l$ the arguments
$z_1 = m_l x_{l-1}$ and $z_2 = m_l x_l$, the effective logarithmic derivative at the outer
surface of layer $l$ follows from the one at its inner surface (Yang 2003; Peña & Pal 2009).
Starting from the core's $H^{(0)} = D_n(m_0 x_0)$,

$$
G_1 = m_l H^{(l-1)} - m_{l-1} D_n(z_1), \qquad G_2 = m_l H^{(l-1)} - m_{l-1} D^\zeta_n(z_1)
$$

for $H_a$ (with $m_l$ and $m_{l-1}$ exchanged for $H_b$), and with
$Q_n = [\psi_n/\zeta_n](z_1) \,/\, [\psi_n/\zeta_n](z_2)$,

$$
H^{(l)} = \frac{G_2 D_n(z_2) - Q_n G_1 D^\zeta_n(z_2)}{G_2 - Q_n G_1}.
$$

For $\zeta = \xi$ this is Yang's formula. Written this way it is not usable near the real zeros
of $\psi_n(z_2)$, where $Q_n$ and $D_n(z_2)$ have poles but $H^{(l)}$ does not. scatrs uses two
equivalent forms instead. With $\Pi_n = \psi_n\zeta_n$ and $P_n = Q_n\Pi_n(z_2)$,

$$
H^{(l)} = D^\zeta_n(z_2) - \frac{W G_2}{\Pi_n(z_2) G_2 - P_n G_1}
        = D_n(z_2) + \delta, \qquad
\delta = -\frac{W P_n G_1}{\Pi_n(z_2)\,\bigl(\Pi_n(z_2) G_2 - P_n G_1\bigr)}.
$$

The first form has no pole where $\psi_n(z_2)$ vanishes. The second separates the inner layers'
contribution $\delta$ from the layer's own $D_n(z_2)$. That separation matters at the outer
surface, where the homogeneous sphere's numerators are free of cancellation only in terms of
$D_n$ (below). scatrs takes the second form at every order unless $D_n(z_2)$ and $\delta$
cancel, by more than a factor of 4, which happens only near the poles of $D_n(z_2)$, and the
first form there. The two forms were checked against a direct solution of the interface
conditions in 60-digit arithmetic, to $10^{-40}$.

The auxiliary functions come from recurrences in $n$ that stay finite:

- $D_n$, through $u_n = \psi_n/\psi_{n-1}$, from the sphere's contracting downward recurrence,
  for every argument;
- $w_n = \zeta_n/\zeta_{n-1}$ from the upward recurrence of $\zeta$,
  $w_{n+1} = (2n + 1)/z - 1/w_n$, which is stable for $\mathrm{Im}\,z \ge 0$, and
  $D^\zeta_n = (n + 1)/z - w_{n+1}$;
- $\Pi_n = \Pi_{n-1} u_n w_n$ and $P_n = P_{n-1}\,[u_n(z_1)/w_n(z_1)]\,w_n(z_2)^2$, ratios
  that neither overflow nor underflow. $P_n$ falls off like $(x_{l-1}/x_l)^{2n+1}$ at high
  orders, and like $e^{-2\,\mathrm{Im}\,m_l (x_l - x_{l-1})}$ through an absorbing layer: the
  layers inside a thick absorber become invisible, as they should.

The starting values $\Pi_0$ and $P_0$ involve $1 - e^{2iz}$, which loses digits for small $z$
unless it is computed as $-\operatorname{expm1}(2iz)$; scatrs has a complex `expm1` for this.

## The second solution

The formulas hold for any second solution $\zeta$, and the choice matters for absorption.

- $\zeta = \xi$ ($W = i$) makes the decay through an absorbing layer explicit: $|P_n| \le
  |\Pi_n|\,e^{-2\,\mathrm{Im}(z_2 - z_1)}$. With $\zeta = \chi$ that decay would appear only
  as a difference of nearly equal numbers.
- $\zeta = \chi$ ($W = -1$) is real for a real argument, as $\psi$ is. Every quantity of a layer
  with a real index is then real, and a weak absorption shows up as imaginary parts of
  relative accuracy. With $\xi$, whose values are complex even for a real argument, the
  absorption is a small part of large complex numbers, and their rounding buries it: the
  relative error of $Q_\text{abs}$ grew like $2^{-52}/\mathrm{Im}\,m$, to $10^{-5}$ at
  $\mathrm{Im}\,m = 10^{-12}$.

scatrs takes $\chi$ for a layer that is optically thin, $\mathrm{Im}\,m_l\,x_l < \tfrac12$, and
$\xi$ otherwise. On a grid of weakly absorbing particles the error of $Q_\text{abs}$ dropped to
rounding level, from $10^{-13}$ to $10^{-5}$ before, and an absorbing core in a clear shell
went from $7\cdot10^{-9}$ to $2\cdot10^{-16}$.

## Cancellation-free numerators

For small spheres the sphere's numerators would cancel catastrophically: $D_n(mx)/m$ and
$D_n(x)$ are both about $(n + 1)/x$, and their difference is far smaller. The sphere's code
removes the cancelling terms exactly ({doc}`mie`). A layered sphere has two more places where
the same terms meet, and scatrs removes them there too.

- **At the outer surface.** With $H = D_n(z_2) + \delta$, the numerators are the sphere's plus
  $\delta/m$ (or $m\delta$), so the sphere's cancellation-free forms carry over. Where the
  first form of $H$ is used, near poles of $D_n(z_2)$, nothing cancels. In the orders below
  $x_L$, $\delta\,\psi_n(x_L)$ is added to the outer layer's own numerator rather than to
  $D_n$ first, since $\delta$ can be far smaller than $D_n$. That own numerator is exactly
  0 for an outer layer of index 1 (an air shell), as it is in exact arithmetic.
- **In $G_1$.** $m_l H^{(l-1)}$ and $m_{l-1} D_n(z_1)$ are both about $(n + 1)/x_{l-1}$. With
  $D_n = (n + 1)/z - u_{n+1}$ and $H^{(l-1)} = D_n(m_{l-1}x_{l-1}) + \delta^{(l-1)}$,
  $$
  G_1 = \frac{n + 1}{x_{l-1}}\Bigl(\frac{m_l}{m_{l-1}} - \frac{m_{l-1}}{m_l}\Bigr)
        - m_l u_{n+1}(m_{l-1}x_{l-1}) + m_{l-1} u_{n+1}(m_l x_{l-1}) + m_l\delta^{(l-1)}
  $$
  for $H_a$; for $H_b$ the first term is exactly 0. Computed directly, $G_1$ of $b_n$ lost
  about four digits for a core of $x_0 = 10^{-9}$.

Layers with the same index then give $G_1 = 0$ exactly and $\delta = 0$, so their results are
bitwise those of the homogeneous sphere of the outer radius. The tests check this.

## Absorption

The sphere's identity $\mathrm{Re}\,a_n - |a_n|^2 = -\mathrm{Im}\,A/|\Delta_a|^2$ holds for any
complex $A$, so $Q_\text{abs}$ keeps its per-order form with $A = H_a/m_{L-1} + n/x_L$
(likewise for $b_n$), without the cancellation of $Q_\text{ext} - Q_\text{sca}$, and
$Q_\text{ext} = Q_\text{sca} + Q_\text{abs}$. A particle with real indices in every layer
absorbs nothing; scatrs sets its $Q_\text{abs}$ to exactly 0, as for a real sphere.

## Derivatives

The derivatives with respect to each layer's $x_l$, $\mathrm{Re}\,m_l$ and $\mathrm{Im}\,m_l$
are in closed form. The coefficients are holomorphic in each $m_l$: the derivative with respect
to $\mathrm{Re}\,m_l$ is the complex derivative $d/dm_l$, and the one with respect to
$\mathrm{Im}\,m_l$ is $i\,d/dm_l$. The absorption sum is not holomorphic; its derivative with
respect to $\mathrm{Im}\,m_l$ follows from the coefficients',

$$
\frac{\partial}{\partial\,\mathrm{Im}\,m_l}\bigl(\mathrm{Re}\,a_n - |a_n|^2\bigr)
  = 2\,\mathrm{Im}\bigl(\bar a_n a_n'\bigr) - \mathrm{Im}\,a_n',
$$

with $a_n' = da_n/dm_l$, likewise for $b_n$.

Inside layer $l$ the logarithmic derivative $y = f'/f$ of the layer's field $f$ (of order $n$)
solves the Riccati equation

$$
y' = V - y^2, \qquad V(z) = \frac{n(n + 1)}{z^2} - 1,
$$

from $y(z_1) = t = m_l H^{(l-1)}/m_{l-1}$ at the inner surface (for $H_a$; the indices are
exchanged for $H_b$) to $y(z_2) = H^{(l)}$. Its sensitivities follow without differentiating
the recursion:

$$
\frac{\partial H^{(l)}}{\partial t} = R^2, \qquad
\frac{\partial H^{(l)}}{\partial z_2} = V(z_2) - \bigl(H^{(l)}\bigr)^2, \qquad
\frac{\partial H^{(l)}}{\partial z_1} = -\bigl(V(z_1) - t^2\bigr)R^2,
$$

with $R = f(z_1)/f(z_2)$ the ratio of the field at the two surfaces,

$$
R = -\frac{m_{l-1} W S_n}{\Pi_n(z_2) G_2 - P_n G_1}, \qquad
S_n = \frac{\zeta_n(z_2)}{\zeta_n(z_1)} = S_{n-1}\,\frac{w_n(z_2)}{w_n(z_1)},
$$

which reuses the recursion's quantities. At each order a sweep from the outer layer inward
chains these sensitivities to give the outer $H_a$ and $H_b$'s derivatives with respect to
every layer's $x_l$ and $m_l$; the outer step then differentiates as the sphere's closed forms
do, with $H$ in place of $D_n(mx)$. All $3L$ derivatives cost one pass, and the values are
bitwise those computed without derivatives.

The sweep's terms need the same care as the values, for the same reasons:

- **At an interface.** $x_{l-1}$ moves both the outer surface of layer $l - 1$ and the inner
  surface of layer $l$. For $H_b$ the two terms cancel at leading order at small $x_{l-1}$.
  Together they are, per unit of the outer $H$'s sensitivity to $t$,
  $m_l(m_l^2 - m_{l-1}^2)\bigl[n(n + 1)/(x_{l-1}^2 m_{l-1}^2 m_l^2) + H^2/m_{l-1}^2\bigr]$ for
  $H_a$ and $(m_l^2 - m_{l-1}^2)/m_l$ for $H_b$ ($H$ the inner layer's), which vanish exactly
  for equal indices.
- **A layer's own index** enters through $z_1$ and $t$ as
  $R^2\bigl[\pm t - z_1\bigl(V(z_1) - t^2\bigr)\bigr]$, whose two terms are both about
  $(n + 1)/z_1$ at small $z_1$. Written with $t = D_n(z_1) + e$ it is
  $\pm D_n - z_1 D_n' + e\bigl(\pm 1 + z_1(2D_n + e)\bigr)$, free of that cancellation; near a
  pole of $D_n(z_1)$, where $D_n$ and $e$ cancel instead, scatrs keeps the first form.
- **The derivative of $H$** at $z_2$, $V - H^2$, is
  $D_n'(z_2) - \delta\bigl(2D_n(z_2) + \delta\bigr)$ in the form $H = D_n(z_2) + \delta$.

The derivatives' validated domain ends at $x_L = 1000$ ({doc}`/validation/accuracy`). For thin
lossless shells around large particles, the rounding errors of the recurrences at a shell's
two surfaces are independent of each other and alike from one order to the next. In the
forward-scattering amplitude's derivative with respect to the shell's index they add up instead
of averaging out. That derivative reaches about $1.7$ times its declared constant at
$x_L = 10^4$, for an air shell of a thousandth of the radius.

## Truncation

The series is truncated at the sphere's $N = \lceil x_L + 6x_L^{1/3} + 3\rceil$ terms, for the
outer size parameter. The reference values sum to $\lceil x_L + 18x_L^{1/3} + 32\rceil$ terms,
so the accuracy maps measure the truncation error too, including for high-index cores whose
$|m_0| x_0$ exceeds $x_L$.

## The accuracy model

The accuracy is declared as for spheres ({doc}`mie`): every output is within
$C\,(\kappa' + \mathrm{cond})\,2^{-52}$ of the exact value. Here cond sums the sensitivities
to every layer's parameters. $\kappa'$ takes the largest index and the largest near-unity factor
$|m_l|/|m_l - 1|$ of any layer whose index is not exactly 1:

$$
\kappa = \bigl(1 + \max_l \max(1, |m_l|)\, x_L\bigr)\,\max\Bigl(1, \max_{l:\,m_l \ne 1} \frac{|m_l|}{|m_l - 1|}\Bigr).
$$

The derivatives of the coefficients and the amplitudes take $\kappa$ itself, near-unity factor
included, since they are built from the recursion's $H$ and share its losses near $m = 1$; a
sphere's closed forms do not. The constants $C$ were measured on two accuracy maps and at the Arb references, each at
least four times the worst case found, and differ from the sphere's where the layered kernel
needs more ({rust}`scatrs::layered::ValidatedDomain::constant`). The validated domain:

- $10^{-6} \le x_L \le 10^4$, and $|m_l|\,x_L \le 3\cdot10^4$ for every layer. Above this,
  the narrow resonances of thin lossless shells of high index cost the coefficients up to
  $10^9$ ulps;
- every layer's index in the sphere's domain ($0 < \mathrm{Re}\,m \le 10$,
  $0 \le \mathrm{Im}\,m \le 10$, $|m| \ge 0.1$), and at most 16 layers;
- the derivatives up to $x_L = 1000$.

## Reference values

The Arb reference values ({doc}`reference-suite`) do not use this algorithm. The generator
computes $\psi_n$ and $\xi_n$ of every argument themselves, in ball arithmetic, and matches the
fields at each interface directly, a form that would lose digits in double precision but is
exact in Arb, where the working precision grows until every value is enclosed tightly. The
references therefore check the recursion and its reformulations, not only their arithmetic.

## Cost

Each layer adds downward recurrences for $D_n$ at its two arguments, and the upward
recurrences of $\Pi_n$, $w_n$ and $P_n$. Measured on one thread of an Apple M4 Pro
(`cargo bench -p scatrs --bench layered`), the efficiencies of a coated sphere cost about 2.8
to 3.3 times those of a homogeneous sphere of the same outer size: 0.92 µs at $x_L = 1$, 9.6 µs
at 100 and 73 µs at 1000. A three-layer particle costs 4.3 to 5.5 times a sphere, a ten-layer
one 15 to 21 times. Every derivative ($3L$ of them) costs 2.3 to 3.3 times the values, for two,
three or ten layers alike: 2.5 µs for a coated sphere at $x_L = 1$, 30 µs at 100 and 243 µs at
1000.
