# Homogeneous spheres (Lorenz–Mie)

This page derives the methods behind `scatrs::mie` (Python: `scatrs.mie`): the series
and its truncation, the recurrences, the cancellation-free forms of the absorption and the
derivatives, the accuracy model, and the rigorous reference values. The rustdoc of
`scatrs::mie` is the authoritative statement of the contracts; the figures here repeat it.

## Conventions and the series

<!-- synced with scatrs/src/mie/mod.rs rustdoc -->

scatrs follows Bohren & Huffman (1983), with time dependence $e^{-i\omega t}$: the refractive
index relative to the medium is $m = n + ik$ with $k \ge 0$ for absorbing spheres, and the size
parameter is $x = 2\pi r/\lambda$. Codes that use $e^{+i\omega t}$, such as Wiscombe's MIEV0,
report the complex conjugates of $a_n$, $b_n$, $S_1$ and $S_2$.

With the Riccati–Bessel functions $\psi_n(\rho) = \rho j_n(\rho)$,
$\chi_n(\rho) = -\rho y_n(\rho)$, $\xi_n = \psi_n - i\chi_n$, and the logarithmic derivative
$D_n = \psi_n'/\psi_n$, the coefficients are

$$a_n = \frac{A_n\psi_n(x) - \psi_{n-1}(x)}{A_n\xi_n(x) - \xi_{n-1}(x)}, \qquad
b_n = \frac{B_n\psi_n(x) - \psi_{n-1}(x)}{B_n\xi_n(x) - \xi_{n-1}(x)},$$

where $A_n = D_n(mx)/m + n/x$ and $B_n = m\,D_n(mx) + n/x$. Call the denominators
$\Delta_a$ and $\Delta_b$. Two identities, valid for all $n$ and real $x$, carry the rest of
this page:

$$\text{(Wronskian)}\quad \psi_n\xi_n' - \psi_n'\xi_n = i, \qquad
\text{(Casoratian)}\quad \psi_n\chi_{n-1} - \psi_{n-1}\chi_n = -1 .$$

The efficiencies (cross sections divided by $\pi r^2$), the asymmetry parameter and the
amplitudes are

$$Q_\text{sca} = \frac{2}{x^2}\sum_n (2n+1)\bigl(\lvert a_n\rvert^2 + \lvert b_n\rvert^2\bigr),
\qquad Q_\text{ext} = \frac{2}{x^2}\sum_n (2n+1)\,\mathrm{Re}(a_n + b_n),$$

$$Q_\text{back} = \frac{1}{x^2}\Bigl\lvert\sum_n (2n+1)(-1)^n(a_n - b_n)\Bigr\rvert^2, \qquad
g = \frac{4}{x^2 Q_\text{sca}}\sum_n\Bigl[\tfrac{n(n+2)}{n+1}\mathrm{Re}\bigl(a_n a_{n+1}^* +
b_n b_{n+1}^*\bigr) + \tfrac{2n+1}{n(n+1)}\mathrm{Re}(a_n b_n^*)\Bigr],$$

$$S_1(\mu) = \sum_n \frac{2n+1}{n(n+1)}(a_n\pi_n + b_n\tau_n), \qquad
S_2(\mu) = \sum_n \frac{2n+1}{n(n+1)}(a_n\tau_n + b_n\pi_n),$$

with $\pi_n = P_n'(\mu)$ and $\tau_n = n(n+1)P_n(\mu) - \mu\pi_n$, computed by the upward
recurrences of Bohren & Huffman (Eq. 4.47).

## Truncation

<!-- synced with scatrs/src/mie/mod.rs rustdoc -->

`f64` keeps `ceil(x + 6 x**(1/3) + 3)` terms (`default_n_max`), and `DoubleDouble`
`ceil(x + 12 x**(1/3) + 6)`. Wiscombe's $\lceil x + 4.05x^{1/3} + 2\rceil$ (MIEV0, and
Bohren & Huffman's BHMIE) left truncation errors up to about $10^{-10}$ in $Q_\text{ext}$ and
$10^{-6}$ in $Q_\text{back}$ against the Arb references; the constants above were chosen from
the same measurements so that truncation errors are below rounding errors in each type.

Beyond $n \approx x$ the terms decay like $\exp\bigl(-\tfrac23(2t)^{3/2}\bigr)$ with
$n = x + t\,x^{1/3}$ — except at the whispering-gallery resonances of spheres with
$\mathrm{Re}\,m > 1$ and small $\mathrm{Im}\,m$, at orders $x < n < \mathrm{Re}(m)\,x$, where a
single $\lvert a_n\rvert$ can approach 1. Resonances above $N$ are exponentially narrow, so a
size parameter must fall within an exponentially small window to meet one; but no truncation
rule of this form can exclude them. The rule is therefore validated by the accuracy maps
rather than proved, and `n_max` (Python `n_max=`) adds terms where a calculation must
resolve them. The reference values, by contrast, are infinite-series values with a proven
bound on the tail (below).

## Logarithmic derivatives

Let $u_k(\rho) = \psi_k(\rho)/\psi_{k-1}(\rho)$. The recurrence
$\psi_{k-1} + \psi_{k+1} = \tfrac{2k+1}{\rho}\psi_k$ gives

$$u_k = \frac{1}{c_k - u_{k+1}}, \qquad c_k = \frac{2k+1}{\rho}, \qquad
D_n(\rho) = \frac{n+1}{\rho} - u_{n+1}(\rho).$$

**Lemma 1 (enclosure).** If $\lvert c_j\rvert \ge 2$ for all $j > k$, then
$\lvert u_j\rvert \le 1$ for all $j > k$, and if also $\lvert c_k\rvert > 1$, then
$\lvert u_k\rvert \le 1/(\lvert c_k\rvert - 1)$.

*Proof.* The map $T_j(u) = 1/(c_j - u)$ sends the closed unit disk into itself when
$\lvert c_j\rvert \ge 2$, since $\lvert c_j - u\rvert \ge \lvert c_j\rvert - 1 \ge 1$. $\psi$ is
the minimal solution of the recurrence as $k \to \infty$ ($j_k(\rho) \sim
\rho^k/(2k+1)!!$ while $y_k(\rho) \sim (2k-1)!!/\rho^{k+1}$), so by Pincherle's theorem
(Gautschi 1967) $u_j$ is the limit of the approximants $T_j \circ T_{j+1} \circ \cdots \circ
T_{j+K}(0)$, all of which lie in the disk; so does the limit. Then
$\lvert u_k\rvert = 1/\lvert c_k - u_{k+1}\rvert \le 1/(\lvert c_k\rvert - 1)$. ∎

scatrs runs the recurrence downward from $u = 0$ at an index
$k_0 = \max(N + 1, \lceil 1.5\lvert\rho\rvert\rceil)$ plus a margin of half the bits of the
type, plus 4. Above $1.5\lvert\rho\rvert$, $\lvert c_k\rvert \ge 3$, so both the true value
and the iterate stay in the disk of radius $\tfrac12$, where $T_k$ contracts distances by
at least $(\lvert c_k\rvert - \tfrac12)^2 \ge 6.25$: the start value is forgotten to rounding
level before the first stored order. There is no continued fraction to converge, so no
convergence test and no failure mode. An exactly zero denominator (at an exact zero of
$\psi_{k-1}$, a pole of $D_{k-1}$) is replaced by a rounding-level value, as in Lentz's
method. For real $m$ the real recurrence is used: twice as fast, and for $m = 1$ it reproduces
$u_k(x)$ bit for bit, so $N_a = N_b = 0$ (below) and $a_n = b_n = 0$ exactly at the orders
computed from ratios, $n > \lfloor x\rfloor$ (all orders for $x < 1$). The orders
$n \le \lfloor x\rfloor$ come from $\psi_n$ and $\chi_n$ and are only rounding-level.

## Riccati–Bessel functions: a hybrid scheme

The upward recurrence for $\psi_n(x)$ starts with $\psi_1 = \sin x/x - \cos x$, which cancels
catastrophically for small $x$ (relative error about $\varepsilon/x^2$, so $2\cdot10^{-4}$ at
$x = 10^{-6}$), and for $n > x$ the upward recurrence of $\psi_n$, the minimal solution, is
unstable. MIEV0 handles small spheres with a separate small-particle expansion, accurate to
about $10^{-8}$. scatrs uses one scheme for every $x$:

- for $n \le n_s = \lfloor x\rfloor$ (only when $x \ge 1$): $\psi_n$ and $\chi_n$ by upward
  recurrence, where both are of order one, and $a_n$, $b_n$ from the formulas above;
- for $n > n_s$: the ratios $u_n(x)$ (downward, as above; $\psi_n(x)$ has no zeros for
  $n > x$, so $u_n(x) > 0$), $R_n = \xi_{n-1}/\xi_n$ upward by
  $1/R_n = (2n-1)/x - R_{n-1}$ (stable: $\lvert\xi_n\rvert$ increases with $n$),
  $T_n = \psi_n/\xi_n = T_{n-1}\,u_n(x)\,R_n$ and $1/\xi_n = R_n/\xi_{n-1}$, and

$$a_n = T_n\frac{N_a}{A_n - R_n}, \quad b_n = T_n\frac{N_b}{B_n - R_n}, \qquad
N_a = \frac{n+1}{x}\Bigl(\frac{1}{m^2} - 1\Bigr) + u_{n+1}(x) - \frac{u_{n+1}(mx)}{m}, \quad
N_b = u_{n+1}(x) - m\,u_{n+1}(mx).$$

$N_a = D_n(mx)/m - D_n(x)$ and $N_b = m D_n(mx) - D_n(x)$ (substitute
$D_n = (n+1)/\rho - u_{n+1}$): the terms of size $n/x$ that cancel are removed exactly. Without
this, $b_n$ lost all its digits at $x = 10^{-6}$. $T_n$ and $1/\xi_n$ underflow gracefully to 0
as $n$ grows, so any `n_max` is safe.

The small-particle expansion is not needed, and the accuracy maps confirm the scheme across
the domain. The coefficient $1/m^2 - 1 = -(m-1)(m+1)/m^2$ is formed from $m - 1$, which is
exact for $m$ near 1.

## Absorption without cancellation

<!-- synced with scatrs/src/mie/efficiencies.rs rustdoc -->

**Proposition 1.** $\mathrm{Re}\,a_n - \lvert a_n\rvert^2 = -\mathrm{Im}\bigl(D_n(mx)/m\bigr)
/\lvert\Delta_a\rvert^2$, and $\mathrm{Re}\,b_n - \lvert b_n\rvert^2 =
-\mathrm{Im}\bigl(m D_n(mx)\bigr)/\lvert\Delta_b\rvert^2$.

*Proof.* $\mathrm{Re}\,a - \lvert a\rvert^2 = \lvert a\rvert^2(\mathrm{Re}(1/a) - 1)$, and
$1/a_n - 1 = (\Delta_a - \text{numerator})/\text{numerator} = -i(A_n\chi_n -
\chi_{n-1})/(A_n\psi_n - \psi_{n-1})$ since $\xi = \psi - i\chi$. So
$\mathrm{Re}(1/a_n) - 1 = \mathrm{Im} f(A_n)$ with $f(A) = (A\chi_n - \chi_{n-1})/(A\psi_n -
\psi_{n-1})$, a Möbius map with real coefficients, for which
$\mathrm{Im} f(A) = \det\cdot\,\mathrm{Im} A/\lvert A\psi_n - \psi_{n-1}\rvert^2$ with
$\det = \psi_n\chi_{n-1} - \psi_{n-1}\chi_n = -1$ (the Casoratian). Multiplying by
$\lvert a_n\rvert^2 = \lvert\text{numerator}\rvert^2/\lvert\Delta_a\rvert^2$ and using
$\mathrm{Im}A_n = \mathrm{Im}(D_n/m)$ gives the result; $b_n$ likewise. ∎

So scatrs sums $Q_\text{abs} = \frac{2}{x^2}\sum(2n+1)\bigl[-\mathrm{Im}(D_n/m)/
\lvert\Delta_a\rvert^2 - \mathrm{Im}(mD_n)/\lvert\Delta_b\rvert^2\bigr]$ directly: accurate
relative to itself however small, exactly 0 for real $m$, and $Q_\text{ext} = Q_\text{sca} +
Q_\text{abs}$. Computing $Q_\text{abs}$ as $Q_\text{ext} - Q_\text{sca}$ loses about
$\log_{10}(Q_\text{ext}/Q_\text{abs})$ digits — 8 digits of the co-albedo at $k = 10^{-8}$ —
and the usual $Q_\text{ext}$ sum of $\mathrm{Re}(a_n + b_n)$ is itself inaccurate for small
non-absorbing spheres, where $\mathrm{Re}\,a_n = \lvert a_n\rvert^2 \ll \lvert a_n\rvert$.
The optical theorem $Q_\text{ext} = (4/x^2)\,\mathrm{Re}\,S(0)$ remains a test.

## Closed-form derivatives

<!-- synced with scatrs/src/mie/mod.rs rustdoc -->

**Proposition 2.** With $z = mx$, $D = D_n(z)$ and $D' = n(n+1)/z^2 - 1 - D^2$,

$$\frac{\partial a_n}{\partial m} = \frac{i\,(D - zD')}{m^2\Delta_a^2}, \quad
\frac{\partial b_n}{\partial m} = \frac{-i\,(D + zD')}{\Delta_b^2}, \quad
\frac{\partial a_n}{\partial x} = \frac{i\,(m^2-1)\bigl(n(n+1)/x^2 + D^2\bigr)}{m^2\Delta_a^2},
\quad \frac{\partial b_n}{\partial x} = \frac{i\,(m^2-1)}{\Delta_b^2}.$$

*Proof (for $\partial a_n/\partial m$; the others are alike).* Write
$a_n = (m\psi_n' - D\psi_n)/(m\xi_n' - D\xi_n)$ (Bohren & Huffman's form divided by
$\psi_n(mx)$), whose denominator is $-m\Delta_a$. Only $D$ and the explicit $m$ depend on
$m$, with $\partial D/\partial m = xD'$ (the Riccati equation of $D$). The quotient rule gives
the numerator $(D - mxD')(\psi_n\xi_n' - \psi_n'\xi_n) = i(D - zD')$ by the Wronskian; all
other terms cancel. For $\partial/\partial x$, $\psi_n'' = (n(n+1)/x^2 - 1)\psi_n$ and
$\partial D/\partial x = mD'$ reduce the numerator to
$(m^2 - 1)(n(n+1)/x^2 + D^2)$ times the Wronskian. ∎

Each costs a few complex operations per order on top of the values. $a_n$ and $b_n$ are
holomorphic in $m$, so $\partial/\partial(\mathrm{Re}\,m) = d/dm$ and
$\partial/\partial(\mathrm{Im}\,m) = i\,d/dm$. At small $z$ the leading terms of $D + zD'$
cancel (it is $O(z)$ while $D \approx (n+1)/z$); scatrs evaluates, with $u = u_{n+1}(z)$,

$$D' = (n+1)\Bigl(\frac{2u}{z} - \frac{1}{z^2}\Bigr) - (1 + u^2), \qquad
D + zD' = (2n+1)u - z(1 + u^2), \qquad D - zD' = 2D - (D + zD').$$

The Jacobian of $Q_\text{abs}$ differentiates the terms of Proposition 1:
$\partial_p\bigl(-\mathrm{Im} f/\lvert\Delta\rvert^2\bigr) = -\bigl[\mathrm{Im}\,\partial_p f
- 2\,\mathrm{Im} f\,\mathrm{Re}(\partial_p\Delta/\Delta)\bigr]/\lvert\Delta\rvert^2$ with
$f = D/m$ or $mD$, $\partial\Delta_a/\partial x = \xi_n(D' - n/x^2 - nA_n/x + 1) +
\xi_{n-1}D/m$ and $\partial\Delta_a/\partial m = \xi_n\,\partial A_n/\partial m$ (and
likewise for $b_n$). The other Jacobians follow by the chain and product rules. `Dual`
numbers, propagated through the same recurrences, are the test oracle: their values agree
bitwise with `f64`, and their derivatives with the closed forms to within the rounding
errors of the two routes.

(mie-accuracy-model)=
## The accuracy model

<!-- synced with scatrs/src/mie/domain.rs rustdoc -->

Inside the validated domain ($10^{-6} \le x \le 2\cdot10^4$, $0 < \mathrm{Re}\,m \le 10$,
$0 \le \mathrm{Im}\,m \le 10$) every `f64` output is within

$$C\,(\kappa' + \mathrm{cond})\,2^{-52}$$

of the exact value for the exact inputs, in the error measure of its output
(`MieOutput`; for example $\lvert\text{error}\rvert/\max(\lvert S\rvert, S_\text{rms})$ for
the amplitudes, with $S_\text{rms} = x\sqrt{Q_\text{sca}}/2$). The two condition terms are:

- **cond**, the output's relative condition number with respect to $x$ and $m$, with $s$ the
  denominator of the measure: $(x\lvert\partial y/\partial x\rvert + \lvert m\rvert\,
  \lvert dy/dm\rvert)/s$ for the holomorphic $a_n$, $b_n$, $S_1$, $S_2$ and their derivatives,
  and componentwise, $(x\lvert\partial y/\partial x\rvert + \lvert\mathrm{Re}\,m\rvert\,
  \lvert\partial y/\partial\,\mathrm{Re}\,m\rvert + \lvert\mathrm{Im}\,m\rvert\,
  \lvert\partial y/\partial\,\mathrm{Im}\,m\rvert)/s$, for the real efficiencies, $g$ and
  their Jacobian. Rounding $x$ and $m$ to `f64` (each part of $m$ to within $2^{-53}$ of
  itself, so a real $m$ stays real) already perturbs the output by up to about
  $\mathrm{cond}\cdot2^{-53}$, so no algorithm can avoid it. It is large near the sharp
  resonances of weakly absorbing spheres. A normwise cond for the real outputs would be
  vacuous: $\partial Q_\text{ext}/\partial\,\mathrm{Im}\,m$ is huge when the absorption is
  small, but rounding cannot change $\mathrm{Im}\,m$ by more than $2^{-53}\lvert\mathrm{Im}\,
  m\rvert$.
- **κ′**, an a-priori factor for what cond does not cover: the cancellation in the series (the
  alternating sum of $Q_\text{back}$ can be far smaller than its terms) and the separate losses
  of each order near $m = 1$, where $a_n, b_n \propto m - 1$. It is $\kappa(x, m) =
  (1 + \lvert m\rvert x)\max(1, \lvert m\rvert/\lvert m - 1\rvert)$ for the values;
  $\kappa\max(1, \lvert m\rvert/\lvert m - 1\rvert)$ for the efficiency Jacobian, whose
  derivatives of $g$ lose a second such factor near $m = 1$; and $1 + \lvert m\rvert x$ for the
  derivatives of the coefficients and amplitudes, which lose nothing near $m = 1$: $m - 1$
  enters the $x$-derivatives as an exact factor of $m^2 - 1 = (m - 1)(m + 1)$, and the
  $m$-derivatives have no factor that cancels.

The constants C are measured, with a margin of at least 4 over the worst case found at every
Arb reference point and on a dense double-double grid: 16 for the coefficients, efficiencies
and $g$; 64 for $S_1$, $S_2$ and their derivatives; 256 for the efficiency Jacobian; 512 for
the coefficient derivatives. See {doc}`/validation/accuracy` for the map, and for why
neither term alone is enough.

## Reference values

<!-- synced with reference/scatrs_reference/generators/mie.py docstrings -->

The generator `reference/scatrs_reference/generators/mie.py` computes every value in Arb ball
arithmetic, at the exact double inputs, without Arb's Bessel functions (which lose almost all
their bits at orders in the thousands). It uses the recurrences above, made rigorous:

- $u_k(\rho)$ is seeded by Lemma 1, with the ball $\lvert u\rvert \le 1$, at
  $k_0 \ge 2\lvert\rho\rvert$; every step with $k \ge 2\lvert\rho\rvert$ shrinks the
  uncertainty at least ninefold.
- Complex recurrences use **disk arithmetic** (`scatrs_reference/disk.py`): Arb's complex
  balls are rectangles, which a recurrence that multiplies by numbers of modulus near 1
  rotates and Arb must enclose in larger rectangles — 1.5 times larger per step in the Mie
  recurrences, useless after a few hundred orders. A disk (exact centre, rigorous radius) does
  not grow under rotation.
- $\pi_n$, $\tau_n$ come from Arb's Legendre polynomials, which are accurate at any order.

The precision starts at 320 bits and doubles until every stored value's radius is below
$2^{-200}$ of its scale. Coefficients and their derivatives are finite computations, so
**certified**. The series are summed to $N_\text{ref}$ terms and certified by the following
bound; where that would need more than 60 000 terms ($\lvert m\rvert x$ above about
$6\cdot10^4$) they are summed to $\lceil x + 18x^{1/3} + 32\rceil$ terms and labelled
**high-precision**. Series of derivatives have no tail bound yet and are high-precision.

### Series tail bound

**Proposition 3.** Let $2N + 3 \ge 2\lvert m\rvert x$ and $N \ge 1.5x - \tfrac12$, and
suppose $\lvert R_N\rvert \le 1$ (checked in the computation). With

$$U_z = \frac{1}{(2N+3)/\lvert mx\rvert - 1}, \quad U_x = \frac{1}{(2N+3)/x - 1}, \quad
\bar R = \frac{1}{(2N+1)/x - 1}, \quad q = U_x\bar R,$$

every $n > N$ satisfies $\lvert a_n\rvert + \lvert b_n\rvert \le
\lvert T_N\rvert(F_a + F_b)\,q^{\,n-N}$, where

$$F_a = \frac{(N+2)c_1 + e_1}{(N+1)c_2 - e_2} + \frac{c_1}{c_2}, \qquad
F_b = \frac{U_x + \lvert m\rvert U_z}{(2N+3)/x - \lvert m\rvert U_z - \bar R},$$

$c_1 = \lvert 1/m^2 - 1\rvert/x$, $e_1 = U_z/\lvert m\rvert + U_x$,
$c_2 = \lvert 1 + 1/m^2\rvert/x$, $e_2 = 1/(\lvert m\rvert^2x) + U_z/\lvert m\rvert + \bar R$,
whenever both denominators are positive.

*Proof.* By Lemma 1 (with $\lvert c_j\rvert = (2j+1)/\lvert\rho\rvert \ge 2$ for $j \ge N + 1$),
$\lvert u_k(mx)\rvert \le U_z$ and $0 < u_k(x) \le U_x$ for $k > N$. From
$1/R_n = (2n-1)/x - R_{n-1}$ and $\lvert R_N\rvert \le 1$, induction gives
$\lvert R_n\rvert \le \bar R \le \tfrac12$ for $n > N$. So $\lvert T_{n+1}\rvert =
\lvert T_n\rvert\,u_{n+1}(x)\,\lvert R_{n+1}\rvert \le q\lvert T_n\rvert$. In
$a_n = T_n N_a/(A_n - R_n)$, the forms of $N_a$ and of
$A_n - R_n = (n + (n+1)/m^2)/x - u_{n+1}(mx)/m - R_n$ give $\lvert N_a\rvert \le (n+1)c_1 + e_1$
and $\lvert A_n - R_n\rvert \ge nc_2 - e_2$; the ratio is a Möbius function of $n$, monotone on
$n \ge N + 1$, so it is at most the sum of its value at $N + 1$ and its limit. Likewise
$\lvert N_b\rvert \le U_x + \lvert m\rvert U_z$ and $\lvert B_n - R_n\rvert \ge
(2n+1)/x - \lvert m\rvert U_z - \bar R$. ∎

Each series tail is then a sum of a polynomial weight times a geometric sequence, bounded by
its first term over $1 - q\rho$, where $\rho$ is the weight's ratio at $N + 1$ (the weights
are products of factors $an + b$, whose ratios decrease). For $S_1$, $S_2$ the weights use
$\lvert\pi_n\rvert \le n(n+1)/2$ and the crude $\lvert\tau_n\rvert \le n^2(n+1)^2/4$
(polynomial factors cost little against $q^{\,n-N}$). From $P_n' = \sum_k (2k+1)P_k$ over
$k = n-1, n-3, \ldots$, whose weights sum to $n(n+1)/2$, and $\lvert P_k\rvert \le 1$:
$\lvert P_n'\rvert \le n(n+1)/2$. The same sum for $P_n'' = \sum_k (2k+1)P_k'$, with
$\lvert P_k'\rvert \le (n-1)n/2$ for $k \le n - 1$, gives $\lvert P_n''\rvert \le
(n-1)n^2(n+1)/4$; and $\tau_n = \mu\pi_n - (1 - \mu^2)P_n''$, so
$\lvert\tau_n\rvert \le n(n+1)/2 + (n-1)n^2(n+1)/4 \le n^2(n+1)^2/4$. The bound is added to
each value's error; the generator's tests check that it dominates the next terms and that a
longer sum lies inside the certified ball.

## Cost

<!-- synced with scatrs/benches/mie.rs -->

$O(N + \lvert m\rvert x)$ per sphere: the $D_n(mx)$ recurrence starts at about
$1.5\lvert m\rvert x$. Measured with `cargo bench -p scatrs --bench mie` on an Apple M4 Pro
($m = 1.5 + 0.01i$): 0.54 µs at $x = 0.1$, 1.1 µs at $x = 10$, 36 µs at $x = 1000$ and 345 µs
at $x = 10^4$ for the efficiencies (about 34 ns per term); with the full
$(x, \mathrm{Re}\,m, \mathrm{Im}\,m)$ Jacobian 1.3 to 1.8 times that; amplitudes at 100 angles
add about 4.5 ns per angle and term.
