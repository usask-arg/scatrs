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
  size parameters from $10^{-6}$ to $2\cdot10^4$ against 24 refractive indices, with the
  amplitudes at 41 angles. `DoubleDouble` is itself checked against Arb.

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
    worst = np.max(data["c"][name])
    print(f"{name:<26}{worst:10.2f}{data['declared'][name]:10.0f}")
    assert worst <= data["declared"][name]
```

## The map

Each panel shows $\log_{10} C$ for one output: size parameter across, refractive index down.
White cells are exact (for example $Q_\text{abs}$ of real $m$, which is exactly 0).

```{code-cell} python
fig, axes = plt.subplots(5, 2, figsize=(11, 16), sharex=True, sharey=True, layout="constrained")
labels = [f"{z.real:g}{z.imag:+g}i" for z in m]
for ax, name in zip(axes.flat, names, strict=True):
    c = np.array(data["c"][name]).T
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

A tolerance proportional to the a-priori factor $\kappa(x, m) = (1 + \lvert m\rvert x)
\max(1, \lvert m\rvert/\lvert m - 1\rvert)$ alone holds for absorbing spheres, but not for
weakly absorbing or lossless ones: their resonances are arbitrarily sharp, so the outputs
near a resonance are far more sensitive to $x$ and $m$ than $\lvert m\rvert x$ suggests. On a
first version of this grid the constants needed under $\kappa$ alone reached 360 for the
coefficients and 7600 for the efficiency Jacobian, all at real $m$ — and grew as the grid was
refined, since a denser grid lands closer to sharper resonances. The relative condition
number, computed from double-double derivatives (and, for the derivatives, from central
differences of them), measures that sensitivity at each point.

The condition number alone is not enough either: the summed series lose accuracy to
cancellation that the inputs' conditioning does not show. Backscattering by a metallic sphere
($m = 0.05 + 4i$, $x = 2500$) needed $C \approx 6600$ under cond alone: its alternating
series is about $x$ times smaller than its terms. With both terms every output stays below
$C = 10$ for the values and below 60 for the derivatives.

## By refractive index

The worst $C$ of each output over all size parameters, for each refractive index of the grid.
Absorbing spheres stay close to 1; the larger values belong to weakly absorbing and lossless
spheres near sharp resonances, and to the derivatives.

```{code-cell} python
print(f"{'m':<14}" + "".join(f"{n[:10]:>11}" for n in names))
for j, z in enumerate(m):
    row = [max(data["c"][name][i][j] for i in range(len(x))) for name in names]
    print(f"{z.real:g}{z.imag:+g}i".ljust(14) + "".join(f"{v:11.2f}" for v in row))
```
