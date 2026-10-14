---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Accuracy of the Mie outputs

The accuracy of every Mie output is declared in the code, as a tolerance relative to the
conditioning of the problem (see {ref}`mie-accuracy-model`), and enforced by two tests in
continuous integration on Linux, macOS and Windows:

- `scatrs/tests/mie_reference.rs` compares every output with the Arb reference values
  ({doc}`reference-suite`): 306 spheres for the values, 80 for the derivatives.
- `scatrs/tests/mie_accuracy_map.rs` compares `f64` with `DoubleDouble` on a dense grid: 104
  size parameters from $10^{-6}$ to $2\cdot10^4$ against 28 refractive indices, with the
  amplitudes at 41 angles. `DoubleDouble` is itself checked against Arb, on every output:
  within 16 times the declared constants in units of $2^{-104}$, so its own error moves no
  value of the map by more than $2^{-48}$ of the declared constant.

This page plots the map. Each value is

$$C = \frac{\text{error}}{(\kappa' + \mathrm{cond})\,2^{-52}},$$

the constant the declared tolerance $C\,(\kappa' + \mathrm{cond})\,2^{-52}$ would need at that
point; the declared constants are at least 4 times the largest value found.

```{code-cell} python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(name):
    """An accuracy map: its size parameters, refractive indices and output names."""
    data = json.loads(Path("data", name).read_text())
    return data, np.array(data["x"]), [complex(*p) for p in data["m"]], list(data["c"])


def worst_c(data, names):
    """Print the worst C of each output against its declared constant; check the margin of 4."""
    print(f"{'output':<26}{'worst C':>10}{'declared':>10}")
    for name in names:
        # null (a non-finite C) becomes NaN, which fails the check below.
        worst = np.max(np.array(data["c"][name], dtype=float))
        print(f"{name:<26}{worst:10.2f}{data['declared'][name]:10.0f}")
        assert 4 * worst <= data["declared"][name], f"{name}: margin below 4"


def plot_map(data, x, m, names, rows, height):
    """log10 C of each output: size parameter across, refractive index down."""
    fig, axes = plt.subplots(
        rows, 2, figsize=(11, height), sharex=True, sharey=True, layout="constrained"
    )
    labels = [f"{z.real:g}{z.imag:+g}i" for z in m]
    for ax, name in zip(axes.flat, names, strict=True):
        c = np.array(data["c"][name], dtype=float).T
        with np.errstate(divide="ignore"):
            image = ax.pcolormesh(
                np.arange(len(x) + 1), np.arange(len(m) + 1), np.log10(c),
                vmin=-3, vmax=np.log10(data["declared"][name]), cmap="viridis",
            )
        ax.set_title(f"{name} (declared C = {data['declared'][name]:g})")
        fig.colorbar(image, ax=ax, label="log10 C")
    ticks = np.arange(0, len(x), 10)
    for ax in axes[-1]:
        ax.set_xticks(ticks + 0.5, [f"{x[i]:.0e}" for i in ticks], rotation=60)
        ax.set_xlabel("size parameter x")
    for ax in axes[:, 0]:
        ax.set_yticks(np.arange(len(m)) + 0.5, labels, fontsize=7)
    plt.show()


def by_index(data, x, m, names):
    """The worst C of each output over all size parameters, for each refractive index."""
    print(f"{'m':<14}" + "".join(f"{n[:10]:>11}" for n in names))
    for j, z in enumerate(m):
        row = [max(data["c"][name][i][j] for i in range(len(x))) for name in names]
        print(f"{z.real:g}{z.imag:+g}i".ljust(14) + "".join(f"{v:11.2f}" for v in row))


data, x, m, names = load("mie-accuracy-map.json")
print(f"{len(x)} size parameters × {len(m)} refractive indices")
worst_c(data, names)
```

## The map

Each panel shows $\log_{10} C$ for one output: size parameter across, refractive index down.
White cells are exact (for example $Q_\text{abs}$ of real $m$, which is exactly 0).

```{code-cell} python
plot_map(data, x, m, names, rows=5, height=16)
```

## Why the tolerance has two condition terms

A tolerance proportional to the a-priori factor κ′ alone (for the values $\kappa(x, m) = (1 +
\max(1, \lvert m\rvert)\,x)\max(1, \lvert m\rvert/\lvert m - 1\rvert)$) fails where the outputs
are more sensitive to $x$ and $m$ than $\max(1, \lvert m\rvert)\,x$ suggests. That is mostly
near the resonances of weakly absorbing and lossless spheres, which are arbitrarily sharp: on
this grid, κ′ alone would need $C = 360$ for the coefficients ($x = 1250$, $m = 2$), 3700 and
4000 for the derivatives of the coefficients and amplitudes ($x = 2500$, $m = 1.33 +
10^{-8}i$), and 7600 for the efficiency Jacobian ($x = 1.6\cdot10^4$, $m = 10$) — and more on a
denser grid, which lands closer to sharper resonances. But not only there: $g$ of a small,
strongly absorbing sphere ($x = 0.125$, $m = 5 + 5i$), with no resonance, would need 98. The
relative condition number, computed from double-double derivatives (and, for the derivatives,
from central differences of them), measures that sensitivity at each point.

The condition number alone is not enough either: the summed series lose accuracy to
cancellation that the inputs' conditioning does not show. Under cond alone, backscattering by
a metallic sphere ($m = 0.05 + 4i$, $x = 2500$), whose alternating series is far smaller than
its terms, would need $C \approx 6\cdot10^5$; the amplitudes $10^5$, $g$ and $Q_\text{ext}$
$6\cdot10^4$ and $5\cdot10^4$; and the efficiency Jacobian of large absorbing spheres, whose
derivatives barely vary with $x$ and $m$, $10^{17}$. With both terms every value stays below
$C = 3$ except the amplitudes (below 19), and every derivative below 70.

## By refractive index

The worst $C$ of each output over all size parameters, for each refractive index of the grid.
The values stay within a few units everywhere except the amplitudes, which reach about 19 at
$\lvert m\rvert < 1$ and 9 for metal-like spheres ($m = 0.5 + 2i$): their error grows like $x$
whatever $m$. Among the derivatives, the coefficient derivatives of high-index spheres with
some absorption ($m = 8 + 0.1i$, $3 + 0.001i$) and the efficiency Jacobian of lossless water
spheres ($m = 1.33$) reach a few tens.

```{code-cell} python
by_index(data, x, m, names)
```

(greek-accuracy)=
## The phase matrix and the Greek coefficients

<!-- synced with scatrs/tests/mie_greek_accuracy_map.rs -->

The phase matrix and the Greek coefficients follow the same model, with their own error
measures and priors κ′, and a quadrature term for the Greek coefficients
({ref}`mie-accuracy-model`). Two more tests enforce them in CI:

- `scatrs/tests/mie_greek_reference.rs` compares them with the Arb reference values
  ({doc}`reference-suite`): 18 spheres from $x = 0.1$ to 200, where the worst $C$ is 0.59 for
  the phase matrix and 0.61 for the Greek coefficients.
- `scatrs/tests/mie_greek_accuracy_map.rs` compares `f64` with `DoubleDouble` on a grid of 46
  size parameters from $10^{-6}$ to $10^3$ against the 28 refractive indices of the Mie map,
  with the normalized phase matrix at 13 cosines (through the forward peak, the sides and the
  glory) and the Greek coefficients up to all orders for $x \le 160$ and to order 100 above,
  each with its derivatives.

The double-double Greek coefficients cost $O(x^2)$ (the amplitudes at about $x$ nodes, and
about $x$ orders at each), and the derivative measures take seven of them per point, so the
map stops at $x = 10^3$. A test marked `#[ignore]`, which CI runs weekly, checks the values
beyond it: at $x = 2\cdot10^3$ up to all orders ($m = 1.33$ and $0.3 + 0.6i$), and at
$x = 10^4$ ($m = 1.5 + 0.01i$) and $2\cdot10^4$ ($m = 1.33$) up to order 64. There the Greek
coefficients reach $C = 0.22$ at $x = 2\cdot10^4$.

The worst values of the map are 7.4 for the phase matrix ($x = 10^3$, $m = 0.5 +
10^{-4}i$), 22 for its derivatives ($x = 10^{-3}$, same $m$), 1.7 for the Greek coefficients
($x = 1$, $m = 1.0001$) and 2.8 for their derivatives ($x = 6.4\cdot10^{-6}$, $m = 0.1$),
against declared constants of 32, 128, 8 and 16.

```{code-cell} python
greek, gx, gm, gnames = load("greek-accuracy-map.json")
print(f"{len(gx)} size parameters × {len(gm)} refractive indices")
worst_c(greek, gnames)
# The worst values quoted above.
quoted = {
    "phase_matrix": 7.4, "phase_matrix_derivatives": 22, "greek": 1.7, "greek_derivatives": 2.8,
}
for name, c in quoted.items():
    assert round(np.max(greek["c"][name]), 1 if c < 10 else 0) == c, name
```

The quadrature term matters most at large $x$ and $L$. Without it, the Greek coefficients
would need $C = 85$ at $x = 100$ (all orders) and $C = 100$ at $x = 2\cdot10^4$ (to order
64), and more for larger spheres and orders, since the error of the rounded nodes grows like
$L^2$ and $x^2$. With it, $C$ is at most 1.7 on the map. The phase matrix at given cosines has
no quadrature and needs no such term.

```{code-cell} python
plot_map(greek, gx, gm, gnames, rows=2, height=7)
```

By refractive index, the phase matrix is least accurate for $\lvert m\rvert < 1$ and for
strongly absorbing spheres, like the amplitudes it is computed from; the Greek coefficients
stay within 1.7 everywhere, largest next to $m = 1$.

```{code-cell} python
by_index(greek, gx, gm, gnames)
```
