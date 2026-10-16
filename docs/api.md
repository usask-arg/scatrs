# Python API

The Python package returns `float64` and `complex128` results; double-double values are
available from Rust ({doc}`rust-api`). Both APIs follow the {doc}`/theory/conventions`.

## `scatrs`

```{eval-rst}
.. automodule:: scatrs
   :no-members:

.. py:data:: __version__
   :type: str

   The installed package's version: the version of the Rust workspace, in PEP 440 form (the
   pre-release ``0.1.0-alpha.1``, for example, is ``0.1.0a1``).
```

The package also re-exports the exceptions of {mod}`scatrs.errors` and the modules
{mod}`scatrs.mie`, {mod}`scatrs.layered`, {mod}`scatrs.tmatrix`, {mod}`scatrs.ensemble`, {mod}`scatrs.particles`, {mod}`scatrs.refractive`,
{mod}`scatrs.size` and {mod}`scatrs.special`. The classes below are defined in
{mod}`scatrs.particles`, {mod}`scatrs.size` and {mod}`scatrs.ensemble`, and used from the top
level.

### Particles, ensembles and mixtures

```{eval-rst}
.. automodule:: scatrs.particles
   :no-members:

.. autoclass:: scatrs.Sphere

.. autoclass:: scatrs.LayeredSphere
   :members: coated, layers

.. autoclass:: scatrs.Spheroid

.. autoclass:: scatrs.Ensemble
   :members: compute, grid_shape

.. autoclass:: scatrs.Mixture
   :members: compute
```

### Size distributions

```{eval-rst}
.. automodule:: scatrs.size
   :no-members:

.. autoclass:: scatrs.LogNormal
.. autoclass:: scatrs.Gamma
.. autoclass:: scatrs.ModifiedGamma
.. autoclass:: scatrs.PowerLaw
.. autoclass:: scatrs.Monodisperse
.. autoclass:: scatrs.Tabulated
.. autoclass:: scatrs.MultiModal
```

### Results

```{eval-rst}
.. autoclass:: scatrs.EnsembleResult
   :members: d, d_error

.. autoclass:: scatrs.EnsembleOutputs

.. autoclass:: scatrs.EnsembleGreek

.. autoclass:: scatrs.EnsemblePhase

.. autodata:: scatrs.ensemble.SCALARS
```

## `scatrs.ensemble`

```{eval-rst}
.. automodule:: scatrs.ensemble
   :no-members:
```

## `scatrs.refractive`

```{eval-rst}
.. automodule:: scatrs.refractive
   :members:
```

## `scatrs.mie`

```{eval-rst}
.. automodule:: scatrs.mie
   :members:
```

## `scatrs.layered`

```{eval-rst}
.. automodule:: scatrs.layered
   :members:
```

## `scatrs.tmatrix`

```{eval-rst}
.. automodule:: scatrs.tmatrix
   :members:
```

## `scatrs.special`

```{eval-rst}
.. automodule:: scatrs.special
   :members:
```

## `scatrs.errors`

```{eval-rst}
.. automodule:: scatrs.errors
   :members:
   :show-inheritance:
```
