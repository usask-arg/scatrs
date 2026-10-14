---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Example: Gauss–Legendre quadrature

{func}`scatrs.special.gauss_legendre` returns the nodes and weights of the $n$-point
Gauss–Legendre rule on $[-1, 1]$. This page runs whenever the documentation is built. The
cells check their claims with `assert`, so a failing check fails the build.

```{code-cell} python
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

import scatrs
from scatrs.special import gauss_legendre

nodes, weights = gauss_legendre(8)
print(nodes)
print(weights)
```

## Exactness

The $n$-point rule integrates every polynomial of degree up to $2n - 1$ exactly. With $n = 8$
the monomials $x^k$ for $k \le 15$ come out right to rounding level. For $x^{16}$ the rule
is short by exactly the Gauss–Legendre remainder,
$\frac{2^{2n+1}(n!)^4}{(2n+1)\,((2n)!)^2} \approx 4.7\cdot10^{-5}$.

The tolerance follows from the accuracy contract, with $\epsilon = 2^{-52}$. The weights
contribute at most $2\epsilon$, and the nodes about another $2\epsilon$ (as
$k\sum w_i\lvert x_i\rvert^{k-1} \approx 2$). Rounding `x**k` and the products adds a few
$\epsilon$ more. The sums use {func}`math.fsum`, so summation adds almost nothing.

```{code-cell} python
n = 8
x, w = gauss_legendre(n)
eps = np.finfo(float).eps
tolerance = 8 * eps
for k in range(2 * n + 1):
    exact = 2 / (k + 1) if k % 2 == 0 else 0.0
    error = math.fsum(w * x**k) - exact
    print(f"degree {k:2d}: error {error: .3e}")
    if k <= 2 * n - 1:
        assert abs(error) <= tolerance

f = math.factorial
remainder = 2 ** (2 * n + 1) * f(n) ** 4 / ((2 * n + 1) * f(2 * n) ** 2)
print(f"remainder for degree {2 * n}: {remainder:.3e}")
assert abs(error + remainder) <= tolerance
```

The rule is exactly symmetric, and its weights sum to 2. The contract allows $2\epsilon$ for
the weights and the rounding of the sum adds at most $\epsilon$; for this rule the sum is
exactly 2.

```{code-cell} python
assert np.array_equal(x, -x[::-1]) and np.array_equal(w, w[::-1])
print(math.fsum(w) - 2)
assert abs(math.fsum(w) - 2) <= 3 * eps
```

## Accuracy against Arb

The {doc}`reference suite <../validation/reference-suite>` stores the 1000-point rule computed
with Arb. Each value $v$ is stored as three doubles `hi + lo + lo2`, with a rigorous bound `err`
on the remainder. For an `f64` value $g$, the cell computes
$\lvert g - (\text{hi} + \text{lo} + \text{lo2})\rvert + \text{err}$ with rational
arithmetic. That is a rigorous and nearly tight bound on $\lvert g - v\rvert$. Above 100
points the rule comes from asymptotic expansions, and the contract is an absolute error of at
most $32\cdot2^{-52}$ for nodes and a relative error of at most $32\cdot2^{-52}$ for weights.

For comparison, the cell also measures {func}`numpy.polynomial.legendre.leggauss`, which works
in `f64` throughout.

```{code-cell} python
data = next(
    p / "reference" / "data"
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "reference" / "data").is_dir()
)
with open(data / "gauss_legendre.jsonl") as f:
    record = next(r for r in map(json.loads, f) if r["case"] == "n=1000")


def reference(values, i):
    """hi + lo + lo2 of reference value i, as an exact fraction."""
    return Fraction(values["hi"][i]) + Fraction(values["lo"][i]) + Fraction(values["lo2"][i])


def errors(got, values):
    """|got - v| for each reference value v, as a rigorous upper bound, in units of 2**-52."""
    return np.array(
        [
            float((abs(Fraction(g) - reference(values, i)) + Fraction(values["err"][i])) * 2**52)
            for i, g in enumerate(got)
        ]
    )


ref_nodes, ref_weights = record["outputs"]["nodes"], record["outputs"]["weights"]
x, w = gauss_legendre(1000)
xn, wn = np.polynomial.legendre.leggauss(1000)
print("Errors in units of 2**-52: absolute for nodes, relative for weights.")
print(f"{'':>12} {'worst node':>11} {'worst weight':>13}   weights at the 3 nodes nearest -1")
for name, (xs, ws) in {"scatrs": (x, w), f"numpy {np.__version__}": (xn, wn)}.items():
    node_err = errors(xs, ref_nodes)
    weight_err = errors(ws, ref_weights) / np.array(ref_weights["hi"])
    first = ", ".join(f"{e:.3g}" for e in weight_err[:3])
    print(f"{name:>12} {node_err.max():11.3g} {weight_err.max():13.3g}   {first}")
    if name == "scatrs":
        assert node_err.max() <= 32 and weight_err.max() <= 32
```

Next to $\pm1$ the weights are ill-conditioned in the nodes. A node error $\delta$ changes the
weight by a relative $2\lvert x\rvert\delta/(1 - x^2)$, and $1 - x^2 \approx 5.8/n^2$ at the
extreme nodes. That is about $n^2/6$ units of $2^{-52}$ per ulp of node error. So a weight
computed from an `f64` node loses up to that much, even from the correctly rounded node: about
$7\cdot10^4$ units at $n = 1000$. scatrs avoids this: up to 100 points it computes the rule
in double-double arithmetic and rounds it once, and above it computes each weight from an
asymptotic expansion in the angle of the node rather than from the node. This matters next to
$+1$, where the Mie forward-scattering peak is integrated.

numpy's error, $3.7\cdot10^7$ units here, is much larger, and the difference is algorithmic.
`leggauss` computes the weights as $2/\bigl(n\,P_{n-1}(x)\,P_n'(x)\bigr)$, up to a final
normalisation. That equals the weight only at an exact root, and it is $(n+2)/2$ times as
sensitive to the node, so its error grows like $n^3$. The next cell evaluates both formulas
exactly, in rational arithmetic, at the correctly rounded node nearest $+1$:

```{code-cell} python
n = 1000
xq = Fraction(ref_nodes["hi"][-1])  # the double nearest to the node
p_prev, p = Fraction(1), xq  # P_{n-1} and P_n by Bonnet's recurrence, exactly
for j in range(2, n + 1):
    p_prev, p = p, ((2 * j - 1) * xq * p - (j - 1) * p_prev) / j
dp = n * (p_prev - xq * p) / (1 - xq**2)  # P_n'(x)
exact = reference(ref_weights, n - 1)
formulas = {
    "2 / ((1 - x^2) P_n'(x)^2)": 2 / ((1 - xq**2) * dp**2),
    "2 / (n P_(n-1)(x) P_n'(x))": Fraction(2, n) / (p_prev * dp),
}
error = {name: float(abs(v - exact) / exact * 2**52) for name, v in formulas.items()}
for name, e in error.items():
    print(f"{name:>27}: weight error {e:.3g} units of 2**-52")
textbook, leggauss_form = error.values()
print(f"ratio {leggauss_form / textbook:.4f}; (n + 2) / 2 = {(n + 2) / 2}")
assert textbook <= n**2 / 6
assert math.isclose(leggauss_form / textbook, (n + 2) / 2, rel_tol=1e-4)
```

See {ref}`gauss-legendre` for the method.

## Invalid input

$n$ must be a positive integer of at most
{data}`~scatrs.special.MAX_GAUSS_LEGENDRE_POINTS` ($2^{17}$).

```{code-cell} python
for bad in (0, scatrs.special.MAX_GAUSS_LEGENDRE_POINTS + 1, 8.0):
    try:
        gauss_legendre(bad)
    except (scatrs.InvalidInputError, TypeError) as e:
        print(f"{bad!r}: {type(e).__name__}: {e}")
    else:
        raise AssertionError(f"gauss_legendre({bad!r}) should have raised")
```
