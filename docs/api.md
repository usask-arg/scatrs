# API reference

This is the Python API. In phase 0 it returns `float64` results only: double-double values
and derivatives are available from Rust. The Rust API reference will be on
[docs.rs](https://docs.rs/scatrs) once the crate is published; see {doc}`installation` to
build it locally. Both APIs follow the {doc}`conventions`.

## `scatrs`

```{eval-rst}
.. automodule:: scatrs
   :no-members:

.. py:data:: __version__
   :type: str

   The installed package's version: the version of the Rust workspace, in PEP 440 form (the
   pre-release ``0.1.0-alpha.1``, for example, is ``0.1.0a1``).
```

The package also re-exports the exceptions of {mod}`scatrs.errors` and the module
{mod}`scatrs.special`.

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
