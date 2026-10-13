# Conventions

## In phase 0

**Wigner d-functions.** scatrs uses the convention of Mishchenko, Travis & Lacis (2002),
*Scattering, Absorption, and Emission of Light by Small Particles*, Appendix B. It is the
common $d^j_{m'm}(\beta)$ with $(m', m)$ renamed $(m, n)$. For example

$$d^1_{01}(\theta) = \frac{\sin\theta}{\sqrt2}, \qquad
d^1_{10}(\theta) = -\frac{\sin\theta}{\sqrt2}, \qquad
d^2_{22}(\theta) = \cos^4\frac\theta2 .$$

The functions take $x = \cos\theta \in [-1, 1]$ and return every degree
$l = 0, \ldots, l_\text{max}$. Entries with $l < \max(\lvert m\rvert, \lvert n\rvert)$ are
zero. In Python the degree is the last axis: {func}`~scatrs.special.wigner_d` returns an array
of shape `np.shape(x) + (l_max + 1,)`. See {doc}`theory/special-functions`.

**Quadrature.** Gauss–Legendre rules are on $[-1, 1]$. Nodes ascend and the rule is exactly
symmetric: `nodes[i] == -nodes[n - 1 - i]` and `weights[i] == weights[n - 1 - i]`. For odd
$n$ the middle node is $+0$.

**Errors.** The Rust core returns one error type, `scatrs::Error`, and Python raises a matching
exception. The Python exceptions are also subclasses of the standard exception for the same
kind of failure:

| Rust `scatrs::Error` | Python exception | Meaning |
|---|---|---|
| `InvalidInput` | {class}`~scatrs.errors.InvalidInputError` (also `ValueError`) | an argument is outside its domain or a documented limit |
| `NotConverged { achieved, requested }` | {class}`~scatrs.errors.NotConvergedError` (also `ArithmeticError`) | an iteration missed the accuracy it promises |
| `OutsideValidatedDomain` | {class}`~scatrs.errors.OutsideValidatedDomainError` (also `ValueError`) | valid inputs outside the domain where accuracy has been validated |

All three derive from {class}`~scatrs.errors.ScatrsError`. No phase 0 function returns
`OutsideValidatedDomain` yet. Python raises the standard exceptions where Python itself would:
`TypeError` for a wrong argument type, for example. See {mod}`scatrs.errors`.

Kernels such as `gauss_legendre` and `wigner_d` check their inputs and return an error rather
than panic or return NaN. The scalar types instead follow `f64`: a function outside its domain
returns NaN, as `f64::sqrt(-1.0)` does.

**Mixed arithmetic in Rust.** Generic code over `T: Real` combines `T` with `f64` with the
`f64` on the right: `x * 2.0`, not `2.0 * x`. For `Complex<T>`, write
`z.scale(T::from_f64(2.0))`.

**Derivatives.** `Dual<T, N>` carries $N$ first-order derivatives alongside a value. The value
is always computed bit for bit as `T` would compute it; `sin` and `cos` are the one
exception (see below). The conventions that affect results are:

- Comparisons (`==`, `<`, …) look only at the value. `is_zero` and `is_one` are strict: every
  derivative must be zero too.
- A derivative component that is exactly zero is a *strong zero*. It stays zero even when
  multiplied by an infinite or NaN factor.
- At points where a function is not differentiable, scatrs makes a fixed choice. For example,
  `hypot` at the origin has derivative 0 and `atan2` at the origin has NaN derivatives.
- On Apple targets the `f64` values of `sin` and `cos` come from one fused `sincos` call.

{doc}`theory/scalar-types` gives the reasons and the main rules; the rustdoc of `Dual` lists
them all.

**Determinism.** Each phase 0 kernel runs on one thread, and a given build gives the same bits
for the same inputs every time. Gauss–Legendre rules are designed to be bitwise identical
across platforms too, because they use no platform maths library. Some results do use the
platform's library and can differ between platforms in the last bits, within their accuracy
bounds:

- `f64` elementary functions such as `exp` and `sin`, including those inside `Dual<f64, N>`;
- the double-double functions that start from an `f64` result (`ln`, `cbrt` and `atan2`), and
  those computed from them, such as `powf`, `asin` and `acos`;
- `Dual<f64, N>` `sin` and `cos` on Apple targets.

## Planned for the scattering kernels

The design specification fixes these conventions for the scattering kernels. No phase 0 code
uses them yet.

- **Refractive index.** $m = n + ik$, where $\operatorname{Im} m \ge 0$ means absorbing (the
  Bohren–Huffman and Mishchenko convention). Negative $\operatorname{Im} m$ is rejected with
  `InvalidInput`; it is never silently flipped.
- **Lengths.** Radii and wavelength may be in any unit, provided both use the same one. Cross
  sections are returned in that unit squared.
- **Phase-matrix normalization.** $\tfrac12 \int_{-1}^{1} F_{11}(\mu)\,d\mu = 1$.
- **Greek coefficients.** $F_{11}(\Theta) = \sum_l \alpha_{1,l} P_l(\cos\Theta)$, with
  $\alpha_{1,0} = 1$ and $\alpha_{1,1} = 3g$. The other elements follow Hovenier, van der Mee
  & Domke (2004):
  - $\alpha_2 \pm \alpha_3$ expand against $d^l_{2,\pm2}$;
  - $\alpha_4$ expands against $P_l$;
  - $\beta_1$ and $\beta_2$ expand against $d^l_{0,2}$.
- **Random orientation.** The phase matrix has six independent elements: $F_{11}$, $F_{12}$,
  $F_{22}$, $F_{33}$, $F_{34}$ and $F_{44}$. This assumes the ensemble is macroscopically
  isotropic and mirror-symmetric.
- **Fixed orientation.** Euler angles $(\alpha, \beta, \gamma)$ use the z-y-z convention.
  Incidence and scattering directions $(\theta, \varphi)$ are in the laboratory frame. Both
  follow Mishchenko.
- **Shapes.**
  - The spheroid aspect ratio $\varepsilon$ is the equatorial semi-axis divided by the polar
    one: $\varepsilon > 1$ is oblate.
  - The cylinder aspect ratio is the diameter divided by the length.
  - A Chebyshev particle is $r(\theta) = r_0(1 + \varepsilon\cos n\theta)$, with integer
    $n \ge 2$ and $\lvert\varepsilon\rvert < 1$.
- **Size of non-spherical particles.** The radius of the sphere with the same volume, or the
  same surface area, as the user selects.
