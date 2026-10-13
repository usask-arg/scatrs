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
with Arb. Each value is stored as three doubles `hi + lo + lo2`, with a rigorous bound `err` on
the remainder. With rational arithmetic, $\lvert g - (\text{hi} + \text{lo} + \text{lo2})\rvert
+ \text{err}$ is then a rigorous and nearly tight bound on the error of each `f64` value $g$. The contract is an absolute error of at most $2^{-52}$ for nodes and a
relative error of at most $2^{-52}$ for weights.

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


def errors(got, values):
    """|got - v| for each reference value v, as a rigorous upper bound, in units of 2**-52."""
    out = []
    for g, hi, lo, lo2, err in zip(
        got, values["hi"], values["lo"], values["lo2"], values["err"], strict=True
    ):
        v = Fraction(hi) + Fraction(lo) + Fraction(lo2)
        out.append(float((abs(Fraction(g) - v) + Fraction(err)) * 2**52))
    return np.array(out)


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
        assert node_err.max() <= 1 and weight_err.max() <= 1
```

The weights next to $\pm1$ are where an `f64` computation loses accuracy. There
$1 - x^2 \approx 5.8/n^2$ amplifies the error of the node, and that is where the Mie
forward-scattering peak is integrated. scatrs computes the rule in double-double arithmetic
and rounds it once, so its `f64` weights stay within half an ulp. See
{ref}`gauss-legendre`.

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
