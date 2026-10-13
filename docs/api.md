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

   The package version, which is the version of the Rust workspace.
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
