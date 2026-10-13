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

data = json.loads(Path("data/mie-accuracy-map.json").read_text())
x = np.array(data["x"])
m = [complex(*p) for p in data["m"]]
names = list(data["c"])
print(f"{len(x)} size parameters × {len(m)} refractive indices")
print(f"{'output':<26}{'worst C':>10}{'declared':>10}")
for name in names:
    # null (a non-finite C) becomes NaN, which fails the check below.
    worst = np.max(np.array(data["c"][name], dtype=float))
    print(f"{name:<26}{worst:10.2f}{data['declared'][name]:10.0f}")
    assert 4 * worst <= data["declared"][name], f"{name}: margin below 4"
```

## The map

Each panel shows $\log_{10} C$ for one output: size parameter across, refractive index down.
White cells are exact (for example $Q_\text{abs}$ of real $m$, which is exactly 0).

```{code-cell} python
fig, axes = plt.subplots(5, 2, figsize=(11, 16), sharex=True, sharey=True, layout="constrained")
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
print(f"{'m':<14}" + "".join(f"{n[:10]:>11}" for n in names))
for j, z in enumerate(m):
    row = [max(data["c"][name][i][j] for i in range(len(x))) for name in names]
    print(f"{z.real:g}{z.imag:+g}i".ljust(14) + "".join(f"{v:11.2f}" for v in row))
```
