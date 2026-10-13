---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Example: Wigner d-functions

{func}`scatrs.special.wigner_d` returns $d^l_{mn}(\theta)$ for every degree
$l = 0, \ldots, l_\text{max}$ at each $x = \cos\theta$, with the degree as the last axis. The
convention is that of Mishchenko, Travis & Lacis (2002); see {doc}`../conventions`.

Every check below uses the accuracy contract, an absolute error of at most
$16(l+1)\cdot2^{-52}$ per value, rather than an ad hoc tolerance. Each check fails the
documentation build if it does not hold.

```{code-cell} python
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from scatrs.special import gauss_legendre, wigner_d

eps = 2.0**-52
x = np.linspace(-1, 1, 201)
d = wigner_d(2, -1, 10, x)
print(d.shape)  # np.shape(x) + (l_max + 1,)
assert np.all(d[:, :2] == 0)  # zero below l = max(|m|, |n|)
```

## Closed forms

These are $d^1_{01} = \sin\theta/\sqrt2$, $d^2_{22} = \cos^4(\theta/2)$ and
$d^l_{00} = P_l(\cos\theta)$, written in terms of $x$. The last two are rational in $x$, so
the cell computes them exactly with fractions. The square root in the first one is evaluated in
`f64`, which adds up to an ulp.

```{code-cell} python
def bound(l):
    return 16 * (l + 1) * eps


def exact_error(got, value):
    """|got - value| for a double got and an exact rational value."""
    return float(abs(Fraction(got) - value))


d101 = wigner_d(0, 1, 1, x)[:, 1]
assert np.all(np.abs(d101 - np.sqrt((1 - x) * (1 + x) / 2)) <= bound(1) + eps)

d222 = wigner_d(2, 2, 2, x)[:, 2]
for xi, di in zip(x, d222, strict=True):
    assert exact_error(di, ((1 + Fraction(xi)) / 2) ** 2) <= bound(2)

l_max = 20
legendre = wigner_d(0, 0, l_max, x)
for xi, row in zip(x, legendre, strict=True):
    xf = Fraction(xi)
    p = [Fraction(1), xf]  # P_0 and P_1; then (j + 1) P_{j+1} = (2j + 1) x P_j - j P_{j-1}
    for j in range(1, l_max):
        p.append(((2 * j + 1) * xf * p[j] - j * p[j - 1]) / (j + 1))
    assert all(exact_error(row[l], p[l]) <= bound(l) for l in range(l_max + 1))
print("closed forms: ok")
```

## Symmetries

The functions satisfy

$$d^l_{mn}(x) = (-1)^{m-n}\,d^l_{nm}(x) = d^l_{-n,-m}(x), \qquad
d^l_{mn}(-x) = (-1)^{l+m}\,d^l_{m,-n}(x).$$

scatrs computes each side separately. For $x < 0$ it uses the last identity itself in some
cases (near $x = -1$, and for $m = -n$), to reflect the computation onto $x > 0$. Each side is within the contract, so they must agree to within
twice the bound:

```{code-cell} python
l_max = 30
l = np.arange(l_max + 1)
worst = 0.0
for m in range(-5, 6):
    for n in range(-5, 6):
        d = wigner_d(m, n, l_max, x)
        swapped = (-1.0) ** (m - n) * wigner_d(n, m, l_max, x)
        negated = wigner_d(-n, -m, l_max, x)
        reflected = (-1.0) ** (l + m) * wigner_d(m, -n, l_max, -x)
        for other in (swapped, negated, reflected):
            worst = max(worst, np.max(np.abs(d - other) / (2 * bound(l))))
print(f"worst disagreement: {worst:.3f} of twice the bound")
assert worst <= 1
```

## Unitarity

For each $m$ and $l$, $\sum_{n=-l}^{l} \bigl(d^l_{mn}\bigr)^2 = 1$:

```{code-cell} python
l_max = 25
l = np.arange(l_max + 1)
for m in range(-l_max, l_max + 1):
    total = sum(wigner_d(m, n, l_max, x) ** 2 for n in range(-l_max, l_max + 1))
    valid = l >= abs(m)
    tolerance = (32 * (l + 1) * np.sqrt(2 * l + 1) + 2 * l + 1) * eps
    assert np.all(np.abs(total[:, valid] - 1) <= tolerance[valid]), m
print("unitarity: ok")
```

## Orthogonality under Gauss–Legendre quadrature

$\int_{-1}^{1} d^l_{mn}(x)\,d^k_{mn}(x)\,dx = \frac{2}{2l+1}\,\delta_{lk}$. The product is a
polynomial of degree $l + k$, so a Gauss–Legendre rule with $l_\text{max} + 1$ points
integrates it exactly, up to rounding. Every value is within `bound(l_max)` and has
$\lvert d\rvert \le 1$, and the weights sum to 2, so the values contribute at most
$4\,\text{bound}(l_\text{max})$ to the error of each integral. Rounding in the weights and in
the sums of $l_\text{max} + 1$ terms adds well under $4(l_\text{max} + 1)\cdot2^{-52}$.

```{code-cell} python
l_max = 40
nodes, weights = gauss_legendre(l_max + 1)
tolerance = 4 * bound(l_max) + 4 * (l_max + 1) * eps
for m, n in [(0, 0), (2, 2), (2, -2), (0, 2), (1, 3), (-7, 4)]:
    d = wigner_d(m, n, l_max, nodes)  # shape (l_max + 1 nodes, l_max + 1 degrees)
    gram = d.T @ (weights[:, None] * d)
    l0 = max(abs(m), abs(n))
    expected = np.diag(2 / (2 * np.arange(l_max + 1) + 1.0))
    error = np.max(np.abs(gram - expected)[l0:, l0:])
    print(f"(m, n) = {(m, n)}: largest error {error:.1e} (tolerance {tolerance:.1e})")
    assert error <= tolerance
```

## Accuracy against Arb next to a pole

Next to $x = \pm1$ the textbook three-term recurrence loses accuracy like $l^2$, so scatrs
switches to a difference form there (see {ref}`wigner-d`). The reference suite holds $P_l$ at
$x = -1 + 2^{-53}$ for 11 degrees up to $l = 4000$, computed with Arb. The cell bounds the
error of scatrs rigorously, with rational arithmetic. It does the same for a plain `f64`
evaluation of Bonnet's
recurrence $(l+1)P_{l+1} = (2l+1)\,x\,P_l - l\,P_{l-1}$, which is the three-term recurrence
for $m = n = 0$.

```{code-cell} python
data = next(
    p / "reference" / "data"
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "reference" / "data").is_dir()
)
with open(data / "wigner_d.jsonl") as f:
    case = "m=0,n=0,x=-0.9999999999999999,l_max=4000,partial"
    record = next(r for r in map(json.loads, f) if r["case"] == case)
x0, l_max = record["inputs"]["x"], record["inputs"]["l_max"]

plain = [1.0, x0]
for l in range(1, l_max):
    plain.append(((2 * l + 1) * x0 * plain[l] - l * plain[l - 1]) / (l + 1))
got = wigner_d(0, 0, l_max, x0)

ref = record["outputs"]["d"]
print("Errors in units of (l + 1) * 2**-52; the contract allows 16.")
print(f"{'l':>5} {'P_l (Arb)':>28} {'scatrs':>8} {'plain':>8}")
for i, l in enumerate(record["inputs"]["ls"]):
    v = Fraction(ref["hi"][i]) + Fraction(ref["lo"][i]) + Fraction(ref["lo2"][i])
    error, plain_error = (
        float(abs(Fraction(g) - v) + Fraction(ref["err"][i])) for g in (got[l], plain[l])
    )
    unit = (l + 1) * eps
    print(f"{l:5d} {ref['dec'][i]:>28} {error / unit:8.2g} {plain_error / unit:8.3g}")
    assert error <= bound(l)
assert plain_error > bound(l_max)  # the plain recurrence breaks the contract at l = 4000
```
