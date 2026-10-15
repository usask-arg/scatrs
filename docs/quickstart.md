---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Quick start

scatrs computes the optical properties of spherical particles: single spheres with
{mod}`scatrs.mie`, and size distributions and mixtures of them with {class}`~scatrs.Ensemble` and
{class}`~scatrs.Mixture`.
This page shows the main calls and the conventions to know before using the results. See
{doc}`installation` to install the package.

## Conventions at a glance

These are the points that most often trip up a comparison with another code. The full list is
in {doc}`/theory/conventions`.

| Quantity | Convention |
|---|---|
| Refractive index | $m = n + ik$, with $k \ge 0$ for absorbing particles (Bohren & Huffman). Codes that write $m = n - ik$, such as MIEV0 and sasktran2, need the complex conjugate. A negative imaginary part raises {class}`~scatrs.errors.InvalidInputError`. |
| Size | {mod}`scatrs.mie` takes the size parameter $x = 2\pi r/\lambda$. Ensembles take radii and wavelengths in any unit, the same unit for both. $\lambda$ and $m$ are relative to the surrounding medium. |
| Cross sections | {mod}`scatrs.mie` returns efficiencies $Q = C/(\pi r^2)$. Ensembles return cross sections per particle, in the square of the length unit. |
| Angles | Functions of angle take $\mu = \cos\Theta$, with $\Theta$ the scattering angle. |
| Phase matrix | Normalized so that $\tfrac12\int_{-1}^{1} F_{11}\,d\mu = 1$. The degree of linear polarization is $-F_{12}/F_{11}$. |
| Greek coefficients | The expansion of Mishchenko, Travis & Lacis (2002), normalized so that $\alpha_1^0 = 1$, and then $\alpha_1^1 = 3g$. $\beta_1$ and $\beta_2$ have the opposite sign to sasktran2's. |
| Amplitudes | $S_1$, $S_2$ of Bohren & Huffman, time dependence $e^{-i\omega t}$: codes that use $e^{+i\omega t}$, such as MIEV0, report their complex conjugates. |
| Validated domain | $10^{-6} \le x \le 2\cdot10^4$, $0 < \mathrm{Re}\,m \le 10$, $0 \le \mathrm{Im}\,m \le 10$, $\lvert m\rvert \ge 0.1$. Outside it, calls raise {class}`~scatrs.errors.OutsideValidatedDomainError` unless `allow_unvalidated=True`. |

## One sphere

A water droplet of radius 0.5 µm in green light. The functions take NumPy arrays (or scalars),
which broadcast against each other.

```{code-cell} python
import numpy as np

from scatrs import mie

radius, wavelength = 0.5, 0.55  # µm: any unit, the same for both
x = 2 * np.pi * radius / wavelength
m = 1.33 + 1e-8j

q = mie.efficiencies(x, m)
print(f"x = {x:.3f}")
print(f"Q_ext = {q.q_ext:.6f}, Q_sca = {q.q_sca:.6f}, Q_abs = {q.q_abs:.3e}")
print(f"single-scattering albedo = {q.q_sca / q.q_ext:.10f}, g = {q.g:.6f}")
print(f"C_ext = {q.q_ext * np.pi * radius**2:.6f} µm²")
```

## Angular scattering

{func}`~scatrs.mie.phase_matrix` returns $F_{11}$, $F_{12}$, $F_{33}$ and $F_{34}$ at the
given cosines.

```{code-cell} python
import matplotlib.pyplot as plt

angle = np.linspace(0, 180, 721)
f = mie.phase_matrix(x, m, np.cos(np.radians(angle)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
ax1.semilogy(angle, f.f11)
ax1.set(xlabel="scattering angle (°)", ylabel="$F_{11}$", title="Phase function")
ax2.plot(angle, -f.f12 / f.f11)
ax2.set(xlabel="scattering angle (°)", ylabel="$-F_{12}/F_{11}$", title="Degree of linear polarization")
for ax in (ax1, ax2):
    ax.set_xticks(range(0, 181, 30))
plt.show()
```

## Coefficients for radiative transfer

{func}`~scatrs.mie.greek` expands the phase matrix in Wigner d-functions. The coefficients
are exact up to rounding, and by default every nonzero order is returned.

```{code-cell} python
g = mie.greek(x, m)
print(f"{g.alpha1.shape[-1]} orders")
print("alpha1[:4] =", g.alpha1[:4])
print("beta1[:4]  =", g.beta1[:4])
assert g.alpha1[0] == 1 and abs(g.alpha1[1] - 3 * q.g) < 1e-12
```

## A size distribution

An {class}`~scatrs.Ensemble` is a particle and a size distribution;
{meth}`~scatrs.Ensemble.compute` averages over the sizes at each wavelength. Every output comes
with an error estimate, and `converged` says whether the size integration met its tolerance
(`rtol`, by default $10^{-6}$).

```{code-cell} python
import scatrs

sulfate = scatrs.Ensemble(
    scatrs.Sphere(refractive_index=1.43 + 1e-8j),
    scatrs.LogNormal(median_radius=0.08, width=1.6),  # µm
)
wavelengths = np.array([0.35, 0.55, 1.0])  # µm
r = sulfate.compute(wavelengths, greek=32)

for i, lam in enumerate(wavelengths):
    print(f"λ = {lam} µm: C_ext = {r.c_ext[i]:.4e} µm² (± {r.error.c_ext[i]:.0e}), "
          f"ω = {r.ssa[i]:.8f}, g = {r.g[i]:.4f}")
print("converged:", r.converged)
print("alpha1 at 0.55 µm:", r.greek.alpha1[1, :4])
```

A refractive index that varies with wavelength is a model from {mod}`scatrs.refractive`, such
as a table of measured values ({doc}`theory/refractive-index`). A
{class}`~scatrs.LayeredSphere` is a core in concentric shells, such as black carbon coated in
sulfate, with the core's radius 60% of the particle's:
`scatrs.LayeredSphere.coated(1.95 + 0.79j, 1.43 + 1e-8j, core_fraction=0.6)`
({doc}`examples/coated`). A {class}`~scatrs.Mixture` combines
ensembles of different materials. Distribution parameters given as arrays make a grid of
distributions, computed at once, as an optics table needs:

```{code-cell} python
table = scatrs.Ensemble(
    scatrs.Sphere(1.43 + 1e-8j),
    scatrs.LogNormal(median_radius=np.geomspace(0.05, 0.5, 4)[:, None], width=[1.4, 1.6, 1.8]),
)
t = table.compute(wavelengths)
print(t)
print("C_ext of r_g = 0.05 µm, σ_g = 1.6:", t.c_ext[0, 1])
```

## Derivatives

`derivatives=True` adds derivatives of every output, in closed form, at little extra cost:
with respect to the size parameter and the real and imaginary parts of the refractive index
for single spheres, and to the distribution parameters and the particle's refractive indices
(and a layered sphere's radius fractions) for ensembles.

```{code-cell} python
q = mie.efficiencies(x, m, derivatives=True)
print("dQ_ext/dx       =", q.d("x").q_ext)
print("dQ_ext/d(Re m)  =", q.d("m_real").q_ext)

r = sulfate.compute(0.55, derivatives=["median_radius"])
print("dC_ext/dr_g =", r.d("median_radius").c_ext, "µm")
```

## Next steps

- The examples go further: {doc}`examples/mie`, {doc}`examples/greek` and
  {doc}`examples/ensemble`.
- {doc}`validation/overview` explains how the results are checked, and how far they can be
  trusted.
- {doc}`api` documents every function and option.
