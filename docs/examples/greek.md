---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Example: phase matrix and Greek coefficients

{func}`scatrs.mie.phase_matrix` returns the phase matrix of spheres at any scattering angles,
and {func}`scatrs.mie.greek` its expansion in Wigner d-functions, the Greek coefficients that
polarized radiative-transfer codes take, exact up to rounding and up to every order. Both are
vectorized over size parameter and refractive index, with derivatives. See
{doc}`/theory/mie` for the definitions and the sign conventions. This page runs whenever the
documentation is built; its checks are `assert`s, so a failing check fails the build.

## Phase function and polarization

A water droplet with $x = 10$ and $m = 1.33$. The phase function $F_{11}$ is normalized so
that $\tfrac12\int_{-1}^1 F_{11}\,d\mu = 1$, and the degree of linear polarization of
scattered unpolarized light is $-F_{12}/F_{11}$: positive when the light is polarized
perpendicular to the scattering plane. $F_{33}/F_{11}$ and $F_{34}/F_{11}$ complete the phase
matrix of a sphere ($F_{22} = F_{11}$ and $F_{44} = F_{33}$).

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

from scatrs import mie
from scatrs.special import gauss_legendre, wigner_d

x, m = 10.0, 1.33
angles = np.linspace(0, 180, 721)
mu = np.cos(np.radians(angles))
f = mie.phase_matrix(x, m, mu)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout="constrained")
axes[0, 0].semilogy(angles, f.f11)
axes[0, 0].set(ylabel="$F_{11}$")
for ax, ratio, label in [
    (axes[0, 1], -f.f12 / f.f11, "$-F_{12}/F_{11}$"),
    (axes[1, 0], f.f33 / f.f11, "$F_{33}/F_{11}$"),
    (axes[1, 1], f.f34 / f.f11, "$F_{34}/F_{11}$"),
]:
    ax.plot(angles, ratio)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(ylabel=label, ylim=(-1.05, 1.05))
for ax in axes[1]:
    ax.set(xlabel="scattering angle (°)", xticks=range(0, 181, 30))
fig.suptitle(f"$x$ = {x:g}, $m$ = {m}")
plt.show()
```

The phase function is a polynomial of degree $2N$ in $\mu$, so a Gauss–Legendre rule with
$N + 1$ points integrates it exactly. In the forward and backward directions
$\lvert S_1\rvert = \lvert S_2\rvert$, so $F_{12} = 0$ and the light stays unpolarized. And
since a single sphere scatters coherently, its elements satisfy
$F_{11}^2 = F_{12}^2 + F_{33}^2 + F_{34}^2$ at every angle.

```{code-cell} python
n = int(mie.default_n_max(x))
nodes, weights = gauss_legendre(n + 1)
mean = 0.5 * np.sum(weights * mie.phase_matrix(x, m, nodes).f11)
print("normalization:", mean)
assert abs(mean - 1) < 1e-13
# Each end measured against its own F11 (the forward peak is far larger than the backward).
assert abs(f.f12[0]) < 1e-13 * f.f11[0] and abs(f.f12[-1]) < 1e-13 * f.f11[-1]
# A single sphere's phase matrix is a pure Mueller matrix: F11² = F12² + F33² + F34².
assert np.allclose(f.f12**2 + f.f33**2 + f.f34**2, f.f11**2, rtol=1e-12, atol=0)
```

## Greek coefficients

By default {func}`~scatrs.mie.greek` returns every order up to $2N$, above which the
coefficients are exactly 0. Normalized, $\alpha_1^0 = 1$ and $\alpha_1^1 = 3g$, with $g$ the
asymmetry parameter.

```{code-cell} python
g = mie.greek(x, m)
q = mie.efficiencies(x, m)
L = g.alpha1.shape[-1] - 1
print(f"N = {g.n_terms}, L = {L}")
names = ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
print(f"{'l':>3}" + "".join(f"{name:>12}" for name in names))
for l in range(8):
    row = (g.alpha1[l], g.alpha2[l], g.alpha3[l], g.alpha4[l], g.beta1[l], g.beta2[l])
    print(f"{l:3d}" + "".join(f"{c:12.6f}" for c in row))
print("3g =", 3 * q.g)
assert L == 2 * g.n_terms

fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
for name in names:
    ax.plot(getattr(g, name), label=f"$\\{name[:-1]}_{name[-1]}$")
ax.axhline(0, color="0.6", lw=0.8)
ax.set(xlabel="order $l$", ylabel="coefficient", title=f"$x$ = {x:g}, $m$ = {m}, $L = 2N$ = {L}")
ax.legend(ncols=2)
plt.show()
assert g.alpha1[0] == 1
assert abs(g.alpha1[1] - 3 * q.g) < 1e-13 * 3 * q.g
```

The expansion gives the phase matrix back. Here $F_{11}$ and $F_{12}$ are summed from the
coefficients with {func}`scatrs.special.wigner_d` and compared with
{func}`~scatrs.mie.phase_matrix` at every angle of the figure above.

```{code-cell} python
f11 = wigner_d(0, 0, L, mu) @ g.alpha1
f12 = wigner_d(0, 2, L, mu) @ g.beta1
err11 = np.max(np.abs(f11 - f.f11)) / np.max(f.f11)
err12 = np.max(np.abs(f12 - f.f12)) / np.max(f.f11)
print(f"largest difference relative to max F11: F11 {err11:.1e}, F12 {err12:.1e}")
assert err11 < 1e-12 and err12 < 1e-12
```

## Truncation

The forward peak needs high orders. Summed up to order $L$, the expansion of $F_{11}$ in the
forward direction ($d^l_{00}(0°) = 1$, so the sum is $\sum_{l \le L}\alpha_1^l$) misses most of
the peak at low orders, and reaches rounding level only at about $L = 36$ here, more than
three times the size parameter. Truncating keeps the normalization ($\alpha_1^0$ is unchanged)
but not the peak; so {func}`~scatrs.mie.greek` returns all orders unless asked for fewer.

```{code-cell} python
peak = f.f11[0]
print(f"F11 in the forward direction: {peak:.12f}")
for order in (8, 16, 24, 32, 36, L):
    partial = np.sum(g.alpha1[: order + 1])
    print(f"L = {order:2d}: {partial:16.12f}, relative error {abs(partial / peak - 1):.1e}")
assert abs(np.sum(g.alpha1[:17]) / peak - 1) > 0.1
assert abs(np.sum(g.alpha1[:37]) / peak - 1) < 1e-13
assert abs(np.sum(g.alpha1) / peak - 1) < 1e-12
```

Summed at every angle, the truncated expansions oscillate about the phase function and can go
negative, while the full expansion reproduces it.

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
ax1.semilogy(angles, f.f11, color="k", lw=2.5, alpha=0.3, label="phase_matrix")
for order in (8, 16, 20, L):
    partial = wigner_d(0, 0, order, mu) @ g.alpha1[: order + 1]
    ax1.semilogy(angles, np.where(partial > 0, partial, np.nan), lw=1, label=f"$L$ = {order}")
ax1.set(xlabel="scattering angle (°)", ylabel="$F_{11}$", xticks=range(0, 181, 30),
        title="Expansion summed to order $L$ (negative values not shown)")
ax1.legend(loc="upper right")
orders = np.arange(L + 1)
ax2.semilogy(orders, abs(np.cumsum(g.alpha1) / peak - 1) + 1e-17, ".-")
ax2.set(xlabel="order $L$", ylabel="relative error", title="Forward peak, $\\Theta = 0°$")
plt.show()
```

## Derivatives

`derivatives=True` returns the derivatives with respect to the size parameter and the real
and imaginary parts of the refractive index, from those of the amplitudes, projected like the
values. Here the derivatives of $\alpha_1^1$ (the asymmetry term), $\beta_1^2$ and $\beta_1^{20}$
are checked against central differences, in $x$ and $\mathrm{Re}\,m$ (this $m$ is real, and a central difference in
$\mathrm{Im}\,m$ would need a negative imaginary part, which is rejected).

```{code-cell} python
g = mie.greek(x, m, derivatives=True)
h = 1e-6
for name, at in [
    ("x", lambda e: mie.greek(x + e, m, L)),
    ("m_real", lambda e: mie.greek(x, m + e, L)),
]:
    plus, minus = at(h), at(-h)
    for coef, l in [("alpha1", 1), ("beta1", 2), ("beta1", 20)]:
        fd = (getattr(plus, coef)[l] - getattr(minus, coef)[l]) / (2 * h)
        exact = getattr(g.d(name), coef)[l]
        print(f"d {coef}[{l}] / d{name}: closed form {exact:+.10f}, difference {fd:+.10f}")
        # The difference's rounding error is about eps * |coefficient| / h, 5e-10 here, and
        # its truncation error about h**2 times the third derivative: 1e-7 bounds both.
        assert abs(exact - fd) < 1e-7 * max(1, abs(fd))
```
