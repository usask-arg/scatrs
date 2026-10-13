---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Example: Mie scattering by spheres

{mod}`scatrs.mie` computes the Mie coefficients, efficiencies and amplitudes of homogeneous
spheres, vectorized over size parameter and refractive index, with derivatives. This page runs
whenever the documentation is built; its checks are `assert`s, so a failing check fails the
build.

```{code-cell} python
import numpy as np

from scatrs import mie

# Water droplets in visible light, from Rayleigh to geometric optics.
x = np.geomspace(0.01, 1000, 9)
q = mie.efficiencies(x, 1.33 + 1e-8j)
for xi, qe, w, g in zip(x, q.q_ext, q.q_sca / q.q_ext, q.g, strict=True):
    print(f"x = {xi:8.2f}   Q_ext = {qe:.6f}   single-scattering albedo = {w:.10f}   g = {g:+.4f}")
```

## Extinction paradox and the optical theorem

Large spheres remove twice their geometric cross section ($Q_\text{ext} \to 2$), and
$Q_\text{ext}$ equals $(4/x^2)\,\mathrm{Re}\,S(0)$, the optical theorem, which scatrs does not
use to compute it.

```{code-cell} python
q = mie.efficiencies(1e4, 1.5 + 0.01j)
s = mie.amplitudes(1e4, 1.5 + 0.01j, 1.0)
print(q.q_ext, 4 / 1e4**2 * s.s1.real)
assert abs(q.q_ext - 2) < 0.01
assert abs(4 / 1e4**2 * s.s1.real / q.q_ext - 1) < 1e-11
```

## Weak absorption without cancellation

For weakly absorbing particles, $Q_\text{abs} = Q_\text{ext} - Q_\text{sca}$ cancels:
here about 8 digits. scatrs computes $Q_\text{abs}$ from an exact per-term identity instead
({doc}`/theory/mie`), so the co-albedo $1 - \omega = Q_\text{abs}/Q_\text{ext}$ keeps full
precision. Absorption is linear in $k$ for small $k$, which the direct value shows to all its
digits, and the difference does not.

```{code-cell} python
for k in (1e-6, 1e-8, 1e-10):
    q = mie.efficiencies(20.0, 1.33 + 1j * k)
    print(f"k = {k:.0e}: Q_abs/k direct {q.q_abs / k:.15f}, "
          f"by difference {(q.q_ext - q.q_sca) / k:.15f}")
q6 = mie.efficiencies(20.0, 1.33 + 1e-6j)
q10 = mie.efficiencies(20.0, 1.33 + 1e-10j)
assert abs((q10.q_abs / 1e-10) / (q6.q_abs / 1e-6) - 1) < 1e-4  # linear in k
```

## The phase function

$F_{11} = (\lvert S_1\rvert^2 + \lvert S_2\rvert^2)/2$, normalized so that
$\tfrac12\int_{-1}^1 F_{11}\,d\mu = 1$ once divided by $x^2 Q_\text{sca}/4$. $F_{11}$ is a
polynomial of degree $2N$ in $\mu$, so a Gauss–Legendre rule with $N + 1$ points integrates it
exactly.

```{code-cell} python
from scatrs.special import gauss_legendre

x0, m0 = 30.0, 1.5 + 0.001j
n = int(mie.default_n_max(x0))
mu, w = gauss_legendre(n + 1)
s = mie.amplitudes(x0, m0, mu)
q = mie.efficiencies(x0, m0)
f11 = (abs(s.s1) ** 2 + abs(s.s2) ** 2) / 2 / (x0**2 * q.q_sca / 4)
print("normalization:", 0.5 * np.sum(w * f11))
print("asymmetry parameter:", 0.5 * np.sum(w * f11 * mu), q.g)
assert abs(0.5 * np.sum(w * f11) - 1) < 1e-13
assert abs(0.5 * np.sum(w * f11 * mu) - q.g) < 1e-13
```

## Derivatives

`derivatives=True` returns the derivatives with respect to the size parameter and the real
and imaginary parts of the refractive index, in closed form, for about 1.3 to 1.8 times the
cost of the values. Here they are checked against central differences.

```{code-cell} python
x0, m0 = 12.0, 1.45 + 0.02j
q = mie.efficiencies(x0, m0, derivatives=True)
h = 1e-6
for name, dq, at in [
    ("x", q.d("x"), lambda e: mie.efficiencies(x0 + e, m0)),
    ("m_real", q.d("m_real"), lambda e: mie.efficiencies(x0, m0 + e)),
    ("m_imag", q.d("m_imag"), lambda e: mie.efficiencies(x0, m0 + 1j * e)),
]:
    fd = (at(h).q_ext - at(-h).q_ext) / (2 * h)
    print(f"dQ_ext/d{name}: closed form {dq.q_ext:+.10f}, finite difference {fd:+.10f}")
    assert abs(dq.q_ext - fd) < 1e-7 * max(1, abs(fd))
```
