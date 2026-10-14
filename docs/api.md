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
{mod}`scatrs.mie`, {mod}`scatrs.ensemble` and {mod}`scatrs.special`.

## `scatrs.mie`

```{eval-rst}
.. automodule:: scatrs.mie
   :members:
```

## `scatrs.ensemble`

```{eval-rst}
.. automodule:: scatrs.ensemble
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
