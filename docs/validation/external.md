---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# External validation

The internal validation ({doc}`accuracy`) shows that scatrs evaluates the Mie equations
accurately. This page checks that it evaluates the right equations, in the conventions
users expect, against results that do not depend on scatrs's code: the published test cases of
another Mie code, and the limits of known physics. The page runs whenever the documentation is
built; its checks are `assert`s, so a failing check fails the build.

## Wiscombe's MIEV0 test cases

Wiscombe's MIEV0 (Wiscombe 1979, 1980) is the most widely used Mie code, and its documentation
publishes test cases that most Mie codes since have been checked against: $Q_\text{ext}$,
$Q_\text{sca}$, and the amplitudes $S_1$ and $S_2$ at seven angles from 0° to 180°, to seven
significant digits. The cases span small and large spheres, refractive indices below 1, weak
and strong absorption, and a metal-like $m = 10 + 10i$.

MIEV0 uses time dependence $e^{+i\omega t}$: its refractive index is written $n - ik$, and its
amplitudes are the complex conjugates of scatrs's ({doc}`/theory/conventions`). The values
are those of test cases 5 to 19, as tabulated in the test suite of sasktran2, which replaces a
few backward-angle amplitudes at $x = 10^4$ (cases 11, 16 and 19) with values from another
code; those are left out here.

```{code-cell} python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scatrs import mie

data = json.loads(Path("data", "miev0-test-cases.json").read_text())
mu = np.cos(np.radians(data["angles_deg"]))


def half_unit(printed):
    """Half a unit in the last printed digit of a number such as "2.101321E+00" or "0.093923"."""
    mantissa, _, exponent = printed.upper().partition("E")
    decimals = len(mantissa.partition(".")[2])
    return 0.5 * 10.0 ** (int(exponent or 0) - decimals)


rows = []
for case in data["cases"]:
    x, m = case["x"], complex(*case["m"])
    q = mie.efficiencies(x, m)
    s = mie.amplitudes(x, m, mu)
    # Amplitude differences relative to the amplitudes' size: the larger of |S| and its
    # root-mean-square value over all directions, x √Q_sca / 2.
    scale_rms = x * np.sqrt(q.q_sca) / 2
    worst_s = max(
        abs(got - complex(*want).conjugate()) / max(abs(complex(*want)), scale_rms)
        for name, values in (("s1", s.s1), ("s2", s.s2))
        for got, want in zip(values, case[name], strict=True)
        if want is not None
    )
    # Q differences in units of the last printed digit.
    worst_q = max(abs(getattr(q, k) - float(case[k])) / (2 * half_unit(case[k]))
                  for k in ("q_ext", "q_sca") if k in case)
    rows.append((case["case"], x, m, q.q_ext, q.q_sca, worst_q, worst_s))

print(f"{'case':>4} {'x':>8} {'m':>12} {'Q_ext':>10} {'Q_sca':>10} "
      f"{'Q: last digits':>15} {'S: relative':>12}")
for c, x, m, qe, qs, dq, ds in rows:
    print(f"{c:4d} {x:8g} {f'{m.real:g}+{m.imag:g}i':>12} {qe:10.7f} {qs:10.7f} {dq:15.2f} {ds:12.1e}")
assert max(r[5] for r in rows) < 1, "a Q differs by a unit in its last printed digit"
assert max(r[6] for r in rows) < 1e-6, "an amplitude differs in its 7th digit"
```

Every published $Q$ is reproduced to within one unit in its last printed digit (the column
"Q: last digits"), and every amplitude to within $10^{-6}$ of the amplitudes' size. All but
two cases agree to within half a unit in the seventh digit, the rounding of the published
values. The two exceptions, cases 5 and 6 at $x \approx 0.1$, differ by less than $10^{-6}$;
there the declared accuracy of scatrs's amplitudes is better than $10^{-12}$ ({doc}`accuracy`),
so the differences come from the published values.

```{code-cell} python
cases = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
ax.bar(np.arange(len(rows)), [r[6] for r in rows], color="C0")
ax.axhline(5e-7, color="0.4", ls="--", lw=0.8)
ax.text(len(rows) - 0.5, 6e-7, "half a unit in the 7th digit", ha="right", color="0.3")
ax.set_xticks(np.arange(len(rows)), [f"{c}\n{r[1]:g}" for c, r in zip(cases, rows, strict=True)],
              fontsize=7)
ax.set(yscale="log", ylim=(1e-8, 3e-6), xlabel="MIEV0 test case and its size parameter",
       ylabel="largest relative difference of $S_1$, $S_2$")
plt.show()
```

## Rayleigh scattering

Spheres much smaller than the wavelength scatter as dipoles. Their scattering efficiency is
$Q_\text{sca} = \tfrac83 x^4 \lvert (m^2 - 1)/(m^2 + 2)\rvert^2$, with corrections of order
$x^2$, and their normalized phase matrix has the closed form $F_{11} = \tfrac34(1 + \mu^2)$,
$F_{12} = -\tfrac34(1 - \mu^2)$, $F_{33} = \tfrac32\mu$, $F_{34} = 0$. Its Greek coefficients
(Mishchenko, Travis & Lacis 2002) are $\alpha_1 = (1, 0, \tfrac12)$, $\alpha_2^2 = 3$,
$\alpha_4^1 = \tfrac32$ and $\beta_1^2 = -\sqrt6/2$, all others zero. These fix the sign
convention of every element: a sign error in any of them would show here.

```{code-cell} python
x_small = 1e-4
for m in (1.33, 1.5 + 0.1j, 0.5 + 3j, 10 + 10j):
    q = mie.efficiencies(x_small, m)
    rayleigh = 8 / 3 * x_small**4 * abs((m * m - 1) / (m * m + 2)) ** 2
    print(f"m = {m}: Q_sca / Rayleigh - 1 = {q.q_sca / rayleigh - 1:+.1e}")
    assert abs(q.q_sca / rayleigh - 1) < 1e-6

mu_check = np.linspace(-1, 1, 11)
f = mie.phase_matrix(x_small, 1.5, mu_check)
for got, want in [
    (f.f11, 0.75 * (1 + mu_check**2)),
    (f.f12, -0.75 * (1 - mu_check**2)),
    (f.f33, 1.5 * mu_check),
    (f.f34, 0 * mu_check),
]:
    assert np.allclose(got, want, rtol=0, atol=1e-7)

g = mie.greek(x_small, 1.5, 4)
expected = {
    "alpha1": [1, 0, 0.5, 0, 0], "alpha2": [0, 0, 3, 0, 0], "alpha3": [0, 0, 0, 0, 0],
    "alpha4": [0, 1.5, 0, 0, 0], "beta1": [0, 0, -np.sqrt(6) / 2, 0, 0], "beta2": [0, 0, 0, 0, 0],
}
for name, want in expected.items():
    print(f"{name}: {np.round(getattr(g, name), 6) + 0.0}")  # + 0.0: no negative zeros
    assert np.allclose(getattr(g, name), want, rtol=0, atol=1e-7), name
```

As $x$ grows, $Q_\text{sca}$ leaves the Rayleigh law by a relative amount proportional to
$x^2$:

```{code-cell} python
xs = np.geomspace(1e-4, 1, 200)
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
for m in (1.33 + 0j, 1.5 + 0.1j, 0.5 + 3j):
    q = mie.efficiencies(xs, m)
    rayleigh = 8 / 3 * xs**4 * abs((m * m - 1) / (m * m + 2)) ** 2
    ax.loglog(xs, abs(q.q_sca / rayleigh - 1), label=f"$m$ = {m.real:g} + {m.imag:g}i")
ax.loglog(xs, xs**2, ":", color="0.5", label="$x^2$")
ax.set(xlabel="size parameter $x$", ylabel="$|Q_\\mathrm{sca}/Q_\\mathrm{Rayleigh} - 1|$")
ax.legend()
plt.show()
```

## Large spheres and the optical theorem

A sphere much larger than the wavelength removes twice its geometric cross section, half by
absorption or reflection and refraction and half by diffraction (the extinction paradox), so
$Q_\text{ext} \to 2$. And for every sphere, the extinction equals the forward amplitude by the
optical theorem, $Q_\text{ext} = (4/x^2)\,\mathrm{Re}\,S(0°)$. scatrs computes $Q_\text{ext}$
from the Mie coefficients directly, not through $S(0°)$, so the two are independent results.

```{code-cell} python
xs = np.geomspace(1, 2e4, 300)
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
for m in (1.33 + 1e-3j, 1.5 + 0.1j, 0.5 + 3j):
    q = mie.efficiencies(xs, m)
    s = mie.amplitudes(xs, m, 1.0)
    theorem = 4 / xs**2 * s.s1.real
    assert np.allclose(theorem, q.q_ext, rtol=1e-10, atol=0)
    ax.semilogx(xs, q.q_ext, label=f"$m$ = {m.real:g} + {m.imag:g}i")
    assert abs(q.q_ext[-1] - 2) < 0.02
ax.axhline(2, color="0.5", ls="--", lw=0.8)
ax.set(xlabel="size parameter $x$", ylabel="$Q_\\mathrm{ext}$", ylim=(0, 4.5))
ax.legend()
plt.show()
```

A non-absorbing sphere conserves energy: $Q_\text{abs}$ is exactly zero, and
$Q_\text{ext} = Q_\text{sca}$.

```{code-cell} python
q = mie.efficiencies(np.geomspace(1e-3, 1e4, 50), 1.33)
assert np.all(q.q_abs == 0)
assert np.allclose(q.q_ext, q.q_sca, rtol=1e-13, atol=0)
```

## Size distributions

For averages over size distributions, {doc}`sasktran2` compares scatrs with the Mie
integration of the sasktran2 radiative transfer model, for accuracy and speed.
