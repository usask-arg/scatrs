---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Refractive-index models

A {class}`~scatrs.Sphere` takes its refractive index as a model of the wavelength, from
{mod}`scatrs.refractive`. {meth}`Ensemble.compute <scatrs.Ensemble.compute>` evaluates the
model once at the requested wavelengths; the values used are in the result's
`refractive_index`. scatrs has no database of materials: the data or coefficients come from
the user.

## Conventions

- $m = n + ik$ relative to the surrounding medium, with $k \ge 0$ for absorbing particles
  ({doc}`conventions`). Codes and tables that write $m = n - ik$ need the complex conjugate.
- Wavelengths are in the unit of the radii, and the coefficients of {class}`~scatrs.refractive.Cauchy`
  and {class}`~scatrs.refractive.Sellmeier` are in that unit too. Published coefficients
  usually assume micrometres: with radii in nanometres, convert them (Cauchy's $B$ scales by
  $10^6$ and $C$ by $10^{12}$; Sellmeier's $C_i$ by $10^6$).
- There is no extrapolation. A model evaluated where it has no valid value (a wavelength
  outside a table, a Sellmeier law with $n^2 \le 0$, a callable returning $k < 0$ or a value
  that is not finite) raises {class}`~scatrs.errors.InvalidInputError` naming the wavelength.

## Models

| Model | $m(\lambda)$ | Parameters |
|---|---|---|
| `Constant(m)` | $m$ | `n`, `k` |
| `Table(wavelength, m, interpolation="linear")` | $n$ and $k$ linear in $\lambda$ between the nodes; `"log_k"` interpolates $\ln k$ instead | none |
| `Cauchy(A, B=0, C=0, k=0)` | $n = A + B/\lambda^2 + C/\lambda^4$, constant $k$ | `A`, `B`, `C`, `k` |
| `Sellmeier(B, C, k=0)` | $n^2 = 1 + \sum_i B_i\lambda^2/(\lambda^2 - C_i)$, constant $k$ | `B0`, …, `C0`, …, `k` |
| `Function(f)` | `f(wavelength)` | none |

A number given to {class}`~scatrs.Sphere` becomes a `Constant`, and any other callable a
`Function`. An array is not accepted: values of $m$ mean something only together with their
wavelengths, which a `Table` holds.

A `Table` needs at least two nodes, at strictly increasing wavelengths. Linear interpolation
suits smooth $n$ and $k$; absorption that changes by decades between nodes is better
interpolated in $\ln k$ (`"log_k"`, which needs $k > 0$ at every node). The table is data: it
has no parameters.

Each model lists its parameters in `parameters`, and `derivatives(wavelength)` gives
$\partial m/\partial\theta$ for each. The derivatives of an ensemble with respect to
`"m_real"` and `"m_imag"` are those with respect to $\mathrm{Re}\,m$ and $\mathrm{Im}\,m$ at
each wavelength, whatever model produced $m$.

## Example

Fused silica from its Sellmeier coefficients (Malitson, 1965, $\lambda$ in µm), and an
absorbing material from a short table, interpolated in $\ln k$:

```{code-cell} python
import numpy as np

import scatrs
from scatrs import refractive

silica = refractive.Sellmeier(
    B=[0.6961663, 0.4079426, 0.8974794],
    C=[0.0684043**2, 0.1162414**2, 9.896161**2],
)
dust = refractive.Table(
    [0.3, 0.5, 1.0, 2.0],
    [1.55 + 5e-3j, 1.53 + 1.5e-3j, 1.52 + 8e-4j, 1.50 + 2e-3j],
    interpolation="log_k",
)
wavelength = np.array([0.35, 0.55, 1.6])
print("silica:", silica.evaluate(wavelength).real)
print("dust:  ", dust.evaluate(wavelength))
assert abs(silica.evaluate(0.5893) - 1.4584) < 1e-4  # the sodium D line

r = scatrs.Ensemble(scatrs.Sphere(dust), scatrs.LogNormal(0.5, 1.8)).compute(wavelength)
print("C_ext:", r.c_ext, "µm²; m used:", r.refractive_index)
assert np.array_equal(r.refractive_index, dust.evaluate(wavelength))
```
