# Special functions

This page describes the methods behind {rust}`scatrs::special` (Python: {mod}`scatrs.special`)
and the accuracy each one is validated to. The rustdoc of each function,
{rust}`gauss_legendre <scatrs::special::gauss_legendre>` and
{rust}`wigner_d <scatrs::special::wigner_d>`, is the authoritative statement of its contract. The figures below come from it, and from the test output, which
prints the worst error each accuracy test measures.

(gauss-legendre)=
## Gauss–Legendre quadrature

The $n$-point rule

$$\int_{-1}^{1} f(x)\,dx \approx \sum_{k=1}^{n} w_k\,f(x_k)$$

is exact for polynomials of degree up to $2n - 1$. Its nodes $x_k$ are the roots of the
Legendre polynomial $P_n$, and its weights are

$$w_k = \frac{2}{(1 - x_k^2)\,P_n'(x_k)^2}.$$

### Method

<!-- synced with scatrs/src/special/gauss_legendre.rs rustdoc -->

`gauss_legendre::<T>(n)` uses one of two methods. Up to 100 points, and at every $n$ for
double-double precision, it finds the roots by Newton iteration in double-double arithmetic.
Above 100 points, `f64`-precision rules (`f64` and `Dual<f64, N>`) come from asymptotic
expansions in $O(n)$. Both compute the $\lceil n/2 \rceil$ nodes nearest $+1$ and mirror
them; for odd $n$ the middle node is 0.

**Newton iteration** (double-double):

- **Start.** Each root starts from Tricomi's approximation

  $$x_k \approx \left(1 - \frac{n-1}{8n^3}\right)\cos\frac{(4k-1)\pi}{4n+2}.$$

  The cosine is the double-double one, rounded to `f64`, rather than the platform's. The last
  bits of a double-double root depend on where Newton starts, so this keeps the rule the same
  on every platform.
- **Newton step.** Each step evaluates $P_n$ by the three-term recurrence
  $P_j = a_j\,x\,P_{j-1} - b_j\,P_{j-2}$, with $a_j = (2j-1)/j$ and $b_j = (j-1)/j$
  precomputed in double-double. It then gets the derivative from
  $(1 - x^2)\,P_n'(x) = n\,\bigl(P_{n-1}(x) - x\,P_n(x)\bigr)$.
- **Stopping rule.** Iteration stops when two successive steps are both at most $2^{-90}$. The
  second of them is a polishing step: Newton converges quadratically, so it takes the root to
  rounding level. Failing that within 100 iterations returns `NotConverged`.
- **Weight.** The weight is computed in double-double from the root and $P_n'$ there.
- **Rounding.** The double-double rule is converted once with `Real::from_f64_pair`, so every
  type gets the same rule: the `f64` rule (for $n \le 100$) is the double-double rule rounded
  to nearest.

**Asymptotic expansions** (Bogaert 2014). With $v = 1/(n + \tfrac12)$, $j_k$ the $k$-th zero
of the Bessel function $J_0$ and $a_k = j_k v$, the $k$-th node from $+1$ is $\cos\theta_k$
with

$$\theta_k = a_k + v^2 t_1 + v^4 t_2 + v^6 t_3,$$

and its weight is

$$w_k = \frac{2}{J_1(j_k)^2\,(a_k/\sin a_k)\,\bigl(v^{-2} + W_1 + v^2 W_2 + v^4 W_3\bigr)},$$

where $t_i$ and $W_i$ are polynomials in $\cot a_k$ and $1/a_k$. The coefficients are those of
FastGaussQuadrature.jl, which keeps fewer terms for larger $n$; scatrs keeps them all. The
first 20 zeros $j_k$ and the first 10 values $J_1(j_k)^2$ are tabulated (computed with Arb);
beyond them McMahon's expansion and its counterpart for $J_1(j_k)^2$ take over, already
within $2\cdot2^{-52}$ where the tables end. The weight comes from the angle, not from the
node, so it does not inherit the conditioning of $1 - x^2$ described below.

`Dual` rules are constants (zero derivatives), whose values are bitwise those of `f64` or
`DoubleDouble`. Nodes ascend and the rule is exactly symmetric. Both methods use only IEEE 754
operations and fused multiply-adds, with the double-double sine and cosine instead of the
platform maths library, so the bits of the rule are designed not to depend on the platform. CI
runs the tests on Linux, macOS and Windows.

### Why not plain `f64`

Next to $\pm1$ the weights are ill-conditioned in the nodes. A node error $\delta$ changes the
weight by a relative $2\lvert x\rvert\delta/(1 - x^2)$, and at the extreme nodes
$1 - x^2 \approx 5.8/n^2$. That is about $n^2/6$ units of $2^{-52}$ per ulp of node error. So
a weight computed from an `f64` node can be off by up to that much, even when the node is
correctly rounded: about $7\cdot10^4$ units at $n = 1000$. Next to $+1$ is where the Mie
forward-scattering peak sits. Computed in double-double and rounded once, the `f64` weights of
the Newton method are within about half an ulp; the asymptotic weights, computed from the
angle, within a few. {doc}`/utilities/quadrature` measures this against Arb.

### Accuracy

<!-- synced with scatrs/src/special/gauss_legendre.rs rustdoc -->

The tests check the bounds below against Arb for every node of the full reference rules
($n$ = 1, 2, 3, 4, 5, 8, 16, 33, 64, 100, 128, 257, 500 and 1000). They also check them for
the 8 nodes nearest each of $\pm1$ and the 4 nearest 0 at $n$ = 5000, 20 000 and $2^{17}$, the
cap, and the asymptotic rules against the double-double rules node by node for eleven values
of $n$ from 101 to 4001, and, in an ignored test, for every $n$ from 101 to 3000. The
double-double rule at $n$ = 20 000 takes about 11 s, so its check is an ignored test too; CI
runs the ignored tests weekly. At the cap the test computes only the individual
double-double roots.

| Type | Nodes: absolute error | Weights: relative error | Measured worst |
|---|---|---|---|
| `f64`, $n \le 100$ | $\le 2^{-52}$ | $\le 2^{-52}$ | $0.25\cdot2^{-52}$ for nodes and $0.46\cdot2^{-52}$ for weights: the rounding of the double-double rule |
| `f64`, $n > 100$ | $\le 32\cdot2^{-52}$ | $\le 32\cdot2^{-52}$ | $2.8\cdot2^{-52}$ for nodes ($n$ = 2958) and $4.0\cdot2^{-52}$ for weights ($n$ = 1033) |
| `DoubleDouble` | $\le 2^{-98}$ | $\le 2^{-100}(4 + n^2)$ | $0.51\cdot2^{-106}$ for nodes; about $2^{-85}$, $2^{-82}$ and $2^{-80}$ for weights at $n$ = 5000, 20 000 and $2^{17}$ |

The $n^2$ in the double-double weight bound is the same conditioning of $1 - x^2$, now at
double-double precision.

The tests also check that each method finds the right root, whatever its accuracy. By Bruns's
inequality, root $j$ lies in its own interval of $\theta$, $(j - \tfrac12)\pi/(n + \tfrac12)
< \theta_j < j\pi/(n + \tfrac12)$, and these intervals are disjoint. The tests check that
every root of every `f64` rule with $n \le 300$ lies inside its interval, and so does every
root of eleven rules up to the cap; for the Newton method, so do the roots nearest $\pm1$ and
0 for over a hundred values of $n$ up to the cap. A further test extends the first check to
$n \le 3000$. None of these checks has found a rule that fails.

### Cost and the cap

<!-- synced with scatrs/src/special/gauss_legendre.rs rustdoc -->

The asymptotic method is $O(n)$ in time and memory; the Newton method is $O(n^2)$ in time.
Each Newton root takes four to seven evaluations of the recurrence, about four on average for
$n \ge 5000$. On an Apple M4 Pro this comes to about $28\,\text{ns}\cdot n^2$ for large $n$.
These timings were measured during development; the repository has no benchmarks for them.

| $n$ | 100 | 1000 | 5000 | 20 000 | $2^{17}$ (cap) |
|---|---|---|---|---|---|
| `f64`, measured | 0.8 ms | 0.3 ms | 1.5 ms | 6 ms | 40 ms |
| `DoubleDouble`, measured | 0.8 ms | 32 ms | 0.72 s | 11 s | about 8 min (extrapolated) |

- **Without FMA.** x86_64 builds without the `fma` target feature take roughly ten times
  longer in double-double, because every double-double operation then calls a software
  `mul_add`.
- **Memory.** A double-double call takes `(32 + 2·size_of::<T>())·n` bytes, 32 bytes per point
  of recurrence coefficients besides the rule itself: 277 MB for `Dual<DoubleDouble, 64>` at
  the cap.
- **No interruption.** A call cannot be interrupted. In Python it releases the GIL, but Ctrl+C
  takes effect only when it returns.

`MAX_GAUSS_LEGENDRE_POINTS` caps $n$ at $2^{17} = 131\,072$; a larger $n$ returns
`InvalidInput`. That is about three times the roughly $4\cdot10^4$ points that Mie quadrature
needs at size parameter $2\cdot10^4$. The cap may be raised, never lowered.

A rule that is exact to degree $D$ is also exact for every lower degree, so a double-double
rule is worth computing once and reusing.

(wigner-d)=
## Wigner d-functions

### Convention

<!-- synced with scatrs/src/special/wigner.rs rustdoc -->

The functions $d^l_{mn}(\theta)$ follow Mishchenko, Travis & Lacis (2002), *Scattering,
Absorption, and Emission of Light by Small Particles*, Appendix B (MTL). They are the common
$d^j_{m'm}(\beta)$ with $(m', m)$ renamed $(m, n)$; for example
$d^1_{01}(\theta) = \sin\theta/\sqrt2$. They are
evaluated at $x = \cos\theta$ for every degree from 0 to $l_\text{max}$, and are zero for
$l < l_0 = \max(\lvert m\rvert, \lvert n\rvert)$.

### Start value

The recurrences start at $l_0$ from

$$d^{l_0}_{mn} = \xi_{mn}\sqrt{\binom{2l_0}{\lvert m-n\rvert}}\,
\sin^{\lvert m-n\rvert}\frac\theta2\,\cos^{\lvert m+n\rvert}\frac\theta2,$$

where $\xi_{mn} = 1$ for $n \ge m$ and $(-1)^{m-n}$ otherwise.

### Three-term recurrence

The upward recurrence of MTL Eq. B.22 is $C_s\,d^{s+1} = A_s\,d^s - B_s\,d^{s-1}$, with

$$\begin{aligned}
A_s &= (2s+1)\,\bigl[s(s+1)\,x - mn\bigr], \\
B_s &= (s+1)\,q_s, \qquad q_s = \sqrt{s^2-m^2}\,\sqrt{s^2-n^2}, \\
C_s &= s\,q_{s+1}.
\end{aligned}$$

Away from the poles its absolute error grows like $l$.

### Difference forms next to the poles

Next to $x = 1$, $A_s \approx B_s + C_s$, so $d^{s+1} \approx 2d^s - d^{s-1}$. A rounding error
made at degree $k$ then returns multiplied by about $k\ln(l/k)$, and the error grows like $l^2$
rather than $l$. It is worst for $m = n$, where $d^l \to 1$. Measured `f64` errors of the
three-term form include:

- $266(l+1)\cdot2^{-52}$ at $l = 2000$ for $P_l$ at $x = -1 + 2^{-53}$;
- $27(l+1)\cdot2^{-52}$ at $x = 1$ for $(m, n) = (5, 5)$;
- $18(l+1)\cdot2^{-52}$ at $l = 4000$ for $(35, 34)$ at $x = \cos(4.2\cdot10^{-4})$.

So near the poles scatrs runs the recurrence in Reinsch's difference form instead. It works
with $\Delta^s = d^s - d^{s-1}$:

$$\Delta^{s+1} = \frac{K_s\,d^s + B_s\,\Delta^s}{C_s}, \qquad d^{s+1} = d^s + \Delta^{s+1},
\qquad K_s = A_s - B_s - C_s .$$

$K_s$ is small next to $x = 1$ and must be accurate in absolute terms, so it is computed
without cancellation. The rounding of $B_s$ and $C_s$ perturbs only the small $\Delta$.

- **$m = n$.** $B_s = (s+1)(s^2 - m^2)$ and $C_s = s\bigl((s+1)^2 - m^2\bigr)$ are integers, and
  $K_s = (2s+1)\,s\,(s+1)\,(x - 1)$, where $x - 1$ is exact for $x \in [\tfrac12, 1]$.
- **Any $m$, $n$.** With $\alpha_s = s(s+1) - mn$, $u_s = \alpha_s - q_s$ and
  $v_s = \alpha_s - q_{s+1}$,

  $$K_s = (2s+1)\,s\,(s+1)\,(x - 1) + (s+1)\,u_s + s\,v_s .$$

  The differences are formed without cancellation:

  $$v_s = \frac{\alpha_s^2 - q_{s+1}^2}{\alpha_s + q_{s+1}}, \qquad
  \alpha_s^2 - q_{s+1}^2 = (s+1)^2\bigl(1 + (m-n)^2 - 2(s+1)\bigr) + 2(s+1)\,mn .$$

  The numerator comes from integers, and $u_{s+1} = v_s + 2(s+1)$.

The integer coefficients are formed in the working type. They are exact in double-double for
degrees $s$ below about $9\cdot10^7$ ($2^{26.5}$) and for $\lvert m - n\rvert$ below $2^{26.5}$,
so for $\lvert m\rvert, \lvert n\rvert$ below about $4.7\cdot10^7$ when the orders have
opposite signs. In `f64` they would round from $s \approx 2\cdot10^5$ on, too coarse for
double-double results.

The reflection $d^l_{mn}(x) = (-1)^{l+m}\,d^l_{m,-n}(-x)$ maps the pole $x = -1$ onto $x = 1$.
The method is chosen as follows:

| Region | Form |
|---|---|
| $m = n$ with $x \ge 0$, and $m = -n$ with $x < 0$ (reflected): the side of the pole where $\lvert d\rvert \to 1$ | difference form for $m = n$ |
| any other $m$, $n$ with $\lvert x\rvert \ge 0.9$ ($x \le -0.9$ reflected) | general difference form |
| everywhere else | three-term recurrence |

So $P_l$ ($m = n = 0$) always takes the difference form. The general difference form starts
at $\lvert x\rvert = 0.9$ because below that the three-term form is as accurate and faster in
double-double (in one-off sweeps up to $l = 4000$).

### Start values below the floating-point range

The start value can lie far below the `f64` range although later degrees do not. At the
Gauss–Legendre node nearest a pole for about 500 points ($\sin\theta \approx 4.8\cdot10^{-3}$),
$d^{150}_{0,150} \approx 2^{-1157}$. Yet $d^l_{0,150}$ is a normal double from $l = 227$ on, and
is $2^{-684}$ at $l = 1000$. So while the values are below $2^{-900}$, the recurrence runs on
values scaled by a separate binary exponent, kept within $[2^{-256}, 2^{256}]$. Values that
end up normal therefore meet the accuracy bound relative to their size (for `DoubleDouble`,
values above $2^{-969}$). Smaller ones are rounded once to a
subnormal or to zero, never flushed early. The reference suite has seven such cases,
covering all three forms and the reflection.

### Accuracy

<!-- synced with scatrs/src/special/wigner.rs rustdoc -->

The absolute error against Arb, at the exact `x`, is within:

| Type | Bound | Measured worst |
|---|---|---|
| `f64` | $16(l+1)\cdot2^{-52}$ | $0.53(l+1)\cdot2^{-52}$ |
| `DoubleDouble` | $(l+1)\cdot2^{-97}$ | $0.002(l+1)\cdot2^{-97}$ |

```{important}
These bounds are measured, not proven. The tests check them against the Arb reference values
for:

- every degree $l \le 100$ of 17 pairs $(m, n)$, covering all sign combinations, $m = \pm n$,
  and $\lvert m\rvert, \lvert n\rvert$ up to 50, at $x = \pm1$, $-1 + 2^{-53}$,
  $1 - 3\cdot2^{-53}$, $\pm0.9999999$, $-0.5$, $0$, $0.3$ and $\cos(\pi/6)$;
- sampled degrees up to $l = 2000$ of $(0, 0)$, $(2, \pm2)$, $(0, 2)$, $(50, -37)$,
  $(-37, -50)$ and $(7, 150)$;
- sampled degrees up to $l = 4000$ next to the poles, for $(35, 34)$ at
  $x = \cos(4.2\cdot10^{-4})$ and $(0, 0)$ at $x = -1 + 2^{-53}$;
- sampled degrees up to $l$ = 1000–3000 of the seven cases whose start value lies below the
  floating-point range (see above);
- sampled degrees up to $l = 2.5\cdot10^5$ of $(0, 0)$ and $(0, 2)$, at seven values of $x$
  that cover all three forms.

For general $(m, n)$ the tests therefore reach $l = 2000$, $l = 3000$ in two of the underflow
cases, and $l = 4000$ only in the two cases next to the poles.

Beyond the tests, a one-off adversarial search of about 4500 Arb cases, with $\lvert m\rvert, \lvert n\rvert \le 1000$ and
$l \le 5000$, covered turning points, the switch at $\lvert x\rvert = 0.9$, and near-diagonal
pairs at small $l\theta$. The worst errors it found were $0.74(l+1)\cdot2^{-52}$ and
$0.0034(l+1)\cdot2^{-97}$. Other one-off checks of general $(m, n)$ reached $l$ = 5000–20 000.
```

**Relative accuracy.** While $\lvert d^l\rvert$ grows with $l$, before its first turning
point, the error is also within the bound relative to $\lvert d^l\rvert$. The largest found
in one-off measurements is $1.05(l+1)\cdot2^{-52}$, at the start value of
$(1313, 1313)$ at $x = 0.2087$. Beyond that
point, in the oscillatory region near the zeros of $d^l$, only the absolute bound holds.

### Derivatives

<!-- synced with scatrs/src/special/wigner.rs rustdoc -->

With $x$ a `Dual`, the derivatives are those of the computed values, and the values are
bitwise those of the underlying type.

**Inside the interval.** For $\lvert x\rvert < 1$ the derivatives $d'$ are within
$(l+1)\,l$ times the value bound, times $\max(\lvert d'\rvert, 1)$, divided by $1 - x^2$. The
measured worst is 0.02 of that.

**At the poles.** At $x = \pm1$ a constant `x` (zero derivatives) gives zero derivatives. With
a nonzero derivative of `x` at $x = 1$, the result depends on $a = \lvert m - n\rvert$:

| $a$ | Derivative at $x = 1$ |
|---|---|
| 0 or 2 | finite and correct. For $a = 0$ the tests check it to within $(l+1)\,l$ times the value bound, times $\max(\lvert d'\rvert, 1)$. For $a = 2$ they check it against a finite-difference estimate, itself accurate to about $2^{-48}$: within that bound plus $2^{-48}$, and the `f64` error measured at most 0.004 of it |
| 1 | $\pm\infty$ or NaN, where the true derivative is infinite, since $\sin(\theta/2) = \sqrt{(1-x)/2}$ |
| 3 or more | zero, which is the exact derivative |

The same holds at $x = -1$ with $b = \lvert m + n\rvert$ in place of $a$. The tests check this
for every $l \le 30$ and $\lvert m\rvert, \lvert n\rvert \le 4$.

**Second derivatives.** With nested duals (`Dual<Dual<T, M>, N>`), second derivatives at
$x = \pm1$ are right for $a = 0$ and $a = 2$; for $a = 1$ they are not finite, as the true
value is infinite. For $a \ge 3$ the start value is the constant zero, so every derivative
beyond the first is zero. That is right for $a \ge 5$, since $d^l$ is $(1 - x)^{a/2}$ times a
function smooth at $x = 1$, but silently wrong for $a = 3$ and $a = 4$. The true second
derivative is infinite for $a = 3$. For $a = 4$ it is finite but nonzero: 7.5, 105 and 500.5
for $(2, -2)$ at $l$ = 4, 8 and 12. For
$\lvert x\rvert < 1$ derivatives of every order are those of the computed values.

### Cost

<!-- synced with scatrs/src/special/wigner.rs rustdoc -->

The start value costs $O(\max(\lvert m\rvert, \lvert n\rvert))$. Once $\sin(\theta/2)$ and
$\cos(\theta/2)$ are known, it takes $\lvert m - n\rvert$ divisions and square roots, and
$2\lvert m - n\rvert + \lvert m + n\rvert$ multiplications, at most
$4\max(\lvert m\rvert, \lvert n\rvert)$. Each
further degree then costs:

| Form | Divisions | Square roots |
|---|---|---|
| three-term | 1 | 2 |
| difference form, $m = \pm n$ | 1 | 0 |
| general difference form | 2 | 1 |

Measured during development on an Apple M4 Pro, over the nodes of the 64-point
Gauss–Legendre rule up to $l = 2000$, the time per degree is:

| Orders | `f64` | `DoubleDouble` |
|---|---|---|
| $(0, 0)$ | 2.6 ns | 30 ns |
| $(2, \pm2)$ | 3.1 ns | 38 ns |
| $(0, 2)$, and other $m \ne \pm n$ | 3.7 ns | 45 ns |

The Python function returns at most `MAX_WIGNER_D_VALUES` ($2^{28}$) values per call.
