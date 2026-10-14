# scatrs

Fast, linearized, rigorously validated single-scattering calculations for spherical
particles, written in Rust with a Python interface. The name reads as both "scatterers" and
"scat-rs".

scatrs is built out of the scattering core of the
[sasktran2](https://github.com/usask-arg/sasktran2) radiative transfer model, as a standalone
library for the optical properties of particles.

- **Mie scattering by homogeneous spheres:** coefficients, efficiencies, asymmetry parameter,
  amplitudes, phase matrix and Greek coefficients to every order.
- **Ensembles of spheres:** averages over seven kinds of size distribution, external mixtures
  and batches of distributions, at many wavelengths in parallel, with an error estimate for
  every output.
- **Derivatives** of every output, in closed form.
- **Validated accuracy:** every output has a declared accuracy, checked against independent
  arbitrary-precision reference values across the validated domain, and against published
  results.

## Installation

```bash
pip install scatrs
```

```bash
cargo add scatrs
```

## Example

```python
import numpy as np
import scatrs
from scatrs import mie

# One water droplet: efficiencies, asymmetry parameter and the phase matrix.
q = mie.efficiencies(10.0, 1.33 + 1e-8j)  # size parameter, refractive index n + ik
f = mie.phase_matrix(10.0, 1.33 + 1e-8j, np.cos(np.radians([0, 90, 180])))

# A lognormal aerosol at three wavelengths (µm), with Greek coefficients to order 32.
sulfate = scatrs.Ensemble(
    scatrs.Sphere(refractive_index=1.43 + 1e-8j),
    scatrs.LogNormal(median_radius=0.08, width=1.6),
)
r = sulfate.compute([0.35, 0.55, 1.0], greek=32)
print(r.c_ext, r.ssa, r.g, r.greek.alpha1[:, :4])
```

```rust
use num_complex::Complex;
use scatrs::mie::{Mie, MieOptions};

let mut mie = Mie::<f64>::new();
let sphere = mie.compute(10.0, Complex::new(1.33, 1e-8), &MieOptions::new())?;
println!("Q_ext = {}", sphere.efficiencies().q_ext);
```

## Documentation

[usask-arg.github.io/scatrs](https://usask-arg.github.io/scatrs/): a quick start, examples, the
validation of the results, the theory behind every method, and the Python and Rust API
references. The Rust API is also on [docs.rs](https://docs.rs/scatrs).

## Contributing

See [`CONTRIBUTING.md`](https://github.com/usask-arg/scatrs/blob/main/CONTRIBUTING.md).

## License

MIT.
