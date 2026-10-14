---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Example: an aerosol size distribution

{mod}`scatrs.ensemble` averages the optical properties of spheres over a size distribution, at
many wavelengths, with an error estimate for every output and, on request, derivatives. See
{doc}`/theory/ensembles` for the method. This page runs whenever the documentation is built;
its checks are `assert`s, so a failing check fails the build.

## A sulfate aerosol across the spectrum

A lognormal distribution of sulfate-like droplets: median radius $r_g = 0.08$ µm, width
$\sigma_g = 1.6$, refractive index $m = 1.43 + 10^{-8}i$ at every wavelength. Radii and
wavelengths share a unit (µm here), so the cross sections are in µm² per particle.

```{code-cell} python
import numpy as np

from scatrs import ensemble

sulfate = ensemble.LogNormal(median_radius=0.08, width=1.6)
m = 1.43 + 1e-8j
wavelength = np.array([0.35, 0.40, 0.45, 0.50, 0.55, 0.67, 0.87, 1.0])
r = ensemble.compute(sulfate, wavelength, m)
print(f"{'λ (µm)':>7}{'C_ext (µm²)':>13}{'1 - ω':>10}{'g':>8}{'panels':>8}")
for row in zip(wavelength, r.c_ext, 1 - r.ssa, r.g, r.panels, strict=True):
    print("{:7.2f}{:13.4e}{:10.1e}{:8.4f}{:8d}".format(*row))
assert r.converged.all()
assert np.all(np.diff(r.c_ext) < 0) and np.all(np.diff(r.g) < 0)
assert np.all(1 - r.ssa < 1e-6)
```

Extinction falls and the scattering becomes less forward-peaked as the wavelength grows past
the particles. The Ångström exponent $-d\ln\langle C_\text{ext}\rangle/d\ln\lambda$, here from
neighbouring wavelengths, rises from about 1.5 in the ultraviolet towards the Rayleigh limit
of 4 as the particles become small compared with the wavelength.

```{code-cell} python
angstrom = -np.diff(np.log(r.c_ext)) / np.diff(np.log(wavelength))
mid = np.sqrt(wavelength[1:] * wavelength[:-1])
for lam, a in zip(mid, angstrom, strict=True):
    print(f"λ ≈ {lam:.2f} µm: Ångström exponent {a:.2f}")
assert np.all(np.diff(angstrom) > 0) and 1 < angstrom[0] and angstrom[-1] < 4
```

The moments do not depend on the wavelength. The effective radius of a lognormal is
$r_g\exp\bigl(\tfrac52\ln^2\sigma_g\bigr)$; the integrated one, over the cut-off range, agrees
with it within its error estimate, which includes the cut-offs.

```{code-cell} python
closed_form = 0.08 * np.exp(2.5 * np.log(1.6) ** 2)
print(f"r_eff = {r.effective_radius[0]:.10f} µm, closed form {closed_form:.10f} µm, "
      f"estimate {r.error.effective_radius[0]:.1e} µm")
assert np.all(abs(r.effective_radius - closed_form) <= r.error.effective_radius)
```

## Error estimates and convergence

Every output has an error estimate (`r.error`), and `r.converged` says whether every refined
quantity met the tolerance `rtol` (default $10^{-6}$) within `max_panels` panels. Here the
default result is compared with one at `rtol=1e-11`: the true errors of the refined outputs
are within `rtol` and within their estimates. The backscattering is not refined by default, so
its estimate is only indicative ({ref}`ensemble-accuracy`); `refine_backscatter=True` refines
it too.

```{code-cell} python
tight = ensemble.compute(sulfate, wavelength, m, rtol=1e-11, max_panels=50_000)
print(f"{'output':<10}{'largest error/estimate':>24}{'largest relative error':>24}")
for name in ("c_ext", "c_sca", "g", "ssa", "c_back", "lidar_ratio"):
    value, exact, estimate = getattr(r, name), getattr(tight, name), getattr(r.error, name)
    error = abs(value - exact)
    print(f"{name:<10}{np.max(error / estimate):24.2e}{np.max(error / abs(exact)):24.2e}")
    if name in ("c_ext", "c_sca", "g", "ssa"):
        assert np.all(error <= estimate)
    if name in ("c_ext", "c_sca"):
        assert np.all(error <= 1e-6 * exact)
```

## Greek coefficients

`greek=` asks for the Greek coefficients of the averaged phase matrix, here to order 32, the
first few at 0.55 µm. Normalized, $\alpha_1^0 = 1$ and $\alpha_1^1 = 3g$, with $g$ from the
asymmetry parameters of the spheres.

```{code-cell} python
g = ensemble.compute(sulfate, 0.55, m, greek=32)
names = ("alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2")
print(f"{'l':>3}" + "".join(f"{name:>11}" for name in names))
for l in range(6):
    print(f"{l:3d}" + "".join(f"{getattr(g.greek, name)[l]:11.5f}" for name in names))
print("3g =", 3 * g.g)
assert g.greek.alpha1[0] == 1
assert abs(g.greek.alpha1[1] - 3 * g.g) < 1e-13 * 3 * g.g
```

Their error estimates are bounds, but loose ones: the size nodes are projected in bands by
their number of terms, and a band's table jumps where a node leaves it, which the per-panel
$\lvert K_{15} - G_7\rvert$ takes for integration error ({doc}`/theory/ensembles`). Unnormalized
(`normalized=False`), $\alpha_1^0$ is $\langle C_\text{sca}\rangle$ itself, yet its estimate is
more than $10^3$ times that of $\langle C_\text{sca}\rangle$.

```{code-cell} python
u = ensemble.compute(sulfate, 0.55, m, greek=32, normalized=False)
exact = ensemble.compute(sulfate, 0.55, m, greek=32, normalized=False, rtol=1e-11, max_panels=50_000)
print(f"alpha1[0] = {u.greek.alpha1[0]:.12e}, c_sca = {u.c_sca:.12e}")
print(f"estimates: alpha1[0] {u.error.greek.alpha1[0]:.1e}, c_sca {u.error.c_sca:.1e}; "
      f"true error {abs(u.greek.alpha1[0] - exact.greek.alpha1[0]):.1e}")
assert abs(u.greek.alpha1[0] / u.c_sca - 1) < 1e-13
assert u.error.greek.alpha1[0] > 1e3 * u.error.c_sca
assert np.all(abs(u.greek.alpha1 - exact.greek.alpha1) <= u.error.greek.alpha1)
```

## Derivatives

`derivatives=` names the parameters to differentiate with respect to (or `True` for all):
the distribution's (`"median_radius"`, `"width"`), `"m_real"` and `"m_imag"`. `r.d(name)` holds
the derivatives of every output, and `r.d_error(name)` their estimates. Here
$\partial\langle C_\text{ext}\rangle/\partial r_g$ and $\partial g/\partial r_g$ at 0.55 µm
are checked against central differences, all at `rtol=1e-11` so that the integration error
does not swamp the difference.

```{code-cell} python
def run(median_radius, **kwargs):
    return ensemble.compute(
        ensemble.LogNormal(median_radius, 1.6), 0.55, m, rtol=1e-11, max_panels=50_000, **kwargs
    )


d = run(0.08, derivatives=["median_radius"])
h = 1e-4 * 0.08
plus, minus = run(0.08 + h), run(0.08 - h)
for name in ("c_ext", "g"):
    exact = getattr(d.d("median_radius"), name)
    fd = (getattr(plus, name) - getattr(minus, name)) / (2 * h)
    print(f"d {name} / d median_radius: {exact:.10f}, central difference {fd:.10f}, "
          f"relative difference {abs(exact / fd - 1):.0e}")
    assert abs(exact - fd) < 1e-7 * abs(fd)
```

## A mixture with a coarse mode

{func}`~scatrs.ensemble.compute_mixture` averages over an external mixture of components, each
a distribution, a refractive index and a weight, its relative number concentration. Here the
sulfate is mixed with one coarse, weakly absorbing particle per thousand ($r_g = 0.4$ µm,
$\sigma_g = 1.8$, $m = 1.50 + 10^{-3}i$). Cross sections average with the number fractions
$w_i/\sum w$; the phase matrix, and so the Greek coefficients, with $w_i\langle
C_\text{sca}\rangle_i$.

```{code-cell} python
coarse = ensemble.LogNormal(median_radius=0.4, width=1.8)
weights = (1.0, 1e-3)
components = [
    ensemble.Component(sulfate, m, weights[0]),
    ensemble.Component(coarse, 1.50 + 1e-3j, weights[1]),
]
lam = np.array([0.35, 0.55, 1.0])
mix = ensemble.compute_mixture(components, lam, greek=16, derivatives=["c1.weight"])
fine = ensemble.compute(sulfate, lam, m, greek=16)
big = ensemble.compute(coarse, lam, 1.50 + 1e-3j, greek=16)
f = np.array(weights) / sum(weights)
print(f"{'λ (µm)':>7}{'C_ext (µm²)':>13}{'ω':>9}{'g':>8}{'coarse share of C_ext':>23}")
for i in range(len(lam)):
    share = f[1] * big.c_ext[i] / mix.c_ext[i]
    print(f"{lam[i]:7.2f}{mix.c_ext[i]:13.4e}{mix.ssa[i]:9.5f}{mix.g[i]:8.4f}{share:23.3f}")
c_sca = f[0] * fine.c_sca + f[1] * big.c_sca
alpha1 = (
    f[0] * fine.c_sca[:, None] * fine.greek.alpha1 + f[1] * big.c_sca[:, None] * big.greek.alpha1
) / c_sca[:, None]
assert np.allclose(mix.c_ext, f[0] * fine.c_ext + f[1] * big.c_ext, rtol=1e-13, atol=0)
assert np.allclose(mix.ssa, c_sca / mix.c_ext, rtol=1e-13, atol=0)
assert np.allclose(mix.greek.alpha1, alpha1, rtol=0, atol=1e-13)
assert mix.converged.all()
```

The derivative with respect to a weight follows from the number fractions:
$\partial\langle C_\text{ext}\rangle/\partial w_1 = (\langle C_\text{ext}\rangle_1 -
\langle C_\text{ext}\rangle)/\sum w$, where $\langle C_\text{ext}\rangle_1$ is the coarse mode's.
The coarse particles dominate it: per unit of $w_1$, $\langle C_\text{ext}\rangle$ changes by
about 40 to 330 times its value.

```{code-cell} python
dw = mix.d("c1.weight").c_ext
print("d C_ext / d c1.weight:", dw, "relative to C_ext:", dw / mix.c_ext)
assert np.all((dw / mix.c_ext > 39) & (dw / mix.c_ext < 340))
assert np.allclose(dw, (big.c_ext - mix.c_ext) / sum(weights), rtol=1e-12, atol=0)
```
