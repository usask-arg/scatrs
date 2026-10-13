---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Example: Legendre moments of a phase function

Radiative-transfer solvers take a phase function as its Legendre expansion
$p(\mu) = \sum_l \alpha_l P_l(\mu)$, normalised so that $\tfrac12\int_{-1}^{1} p\,d\mu = 1$
(see {doc}`../conventions`). The moments are

$$\alpha_l = \frac{2l+1}{2}\int_{-1}^{1} p(\mu)\,P_l(\mu)\,d\mu ,$$

which a Gauss–Legendre rule and $P_l = d^l_{00}$ compute directly. The Henyey–Greenstein phase
function

$$p(\mu) = \frac{1 - g^2}{(1 + g^2 - 2g\mu)^{3/2}}$$

has the exact moments $\alpha_l = (2l+1)\,g^l$, so it makes a good check.

```{code-cell} python
import math

import numpy as np

from scatrs.special import gauss_legendre, wigner_d

g, n, l_max = 0.7, 200, 30
mu, w = gauss_legendre(n)
# (1 - g)^2 + 2g(1 - mu) has no cancellation, unlike 1 + g^2 - 2g mu next to mu = 1.
p = (1 - g * g) / ((1 - g) ** 2 + 2 * g * (1 - mu)) ** 1.5
legendre = wigner_d(0, 0, l_max, mu)  # P_l(mu), shape (n, l_max + 1)
l = np.arange(l_max + 1)
alpha = (2 * l + 1) / 2 * ((w * p) @ legendre)
exact = (2 * l + 1) * g**l
for k in (0, 1, 2, 5, 10, 20, 30):
    print(f"l = {k:2d}: alpha = {alpha[k]:.15f}, (2l + 1) g^l = {exact[k]:.15f}")
```

## How accurate is that?

Two errors add up.

**Quadrature.** $p\,P_l$ is analytic in the Bernstein ellipse $E_\rho$ for every
$\rho < 1/g$, because the singularity of $p$ lies at $\mu = (g + 1/g)/2$. On $E_\rho$:

- $\lvert P_l\rvert \le \rho^l$;
- $\lvert p\rvert \le (1 - g^2)/c^{3/2}$, with $c = 1 + g^2 - g(\rho + 1/\rho) > 0$, a lower bound
  on $\operatorname{Re}(1 + g^2 - 2g\mu)$ there.

An $n$-point Gauss rule then integrates $p\,P_l$ to within
$\frac{64}{15}\,M\rho^{-2n}/(\rho^2 - 1)$, where $M$ bounds $\lvert p\,P_l\rvert$ on $E_\rho$
(Trefethen, *Approximation Theory and Approximation Practice*, Theorem 19.3). The cell takes
the best $\rho$ on a grid.

**Rounding.** With $\epsilon = 2^{-52}$:

- each $P_l$ is within the contract, $16(l+1)\epsilon$, of the true value;
- $p$ is within about $5\epsilon$ (relative) and each weight within $\epsilon$;
- the products and the sum of $n$ terms add up to about $(n + 2)\epsilon$ relative to
  $\sum w\,p\,\lvert P_l\rvert \le 2$.

So the error of $\alpha_l$ is at most
$\frac{2l+1}{2}\bigl(2\cdot16(l+1)\epsilon + 2(n + 8)\epsilon + Q_l\bigr)$, where $Q_l$ is the
quadrature bound. The term $2(n + 8)\epsilon$ also covers rounding in $(2l+1)g^l$.

```{code-cell} python
eps = 2.0**-52


def quadrature_bound(l, n, g):
    """Smallest Trefethen bound for a Gauss rule on p P_l, over a grid of ellipses."""
    best = math.inf
    for rho in np.linspace(1.001, 1 / g, 400, endpoint=False):
        c = 1 + g * g - g * (rho + 1 / rho)
        m = (1 - g * g) / c**1.5 * rho**l
        best = min(best, 64 / 15 * m * rho ** (-2 * n) / (rho * rho - 1))
    return best


tolerance = np.array(
    [(2 * k + 1) / 2 * (32 * (k + 1) * eps + 2 * (n + 8) * eps + quadrature_bound(k, n, g))
     for k in l]
)
error = np.abs(alpha - exact)
print(f"largest error {error.max():.1e}, largest fraction of its tolerance "
      f"{np.max(error / tolerance):.3f}")
assert np.all(error <= tolerance)
```

The quadrature bound is negligible here: with 200 points the rule is exact to rounding for
these moments. Fewer points, or $g$ closer to 1, make the forward peak sharper and the bound
larger. A rule with too few points shows up as a large `quadrature_bound`.
