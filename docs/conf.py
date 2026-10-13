"""Sphinx configuration for the scatrs documentation.

autodoc imports scatrs and the examples run it. ``uv run`` installs it, rebuilding the extension
whenever the Rust sources have changed::

    uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html

Intersphinx downloads the Python and numpy inventories. Without network access that download
fails with a warning, which ``-W`` makes fatal: set ``SCATRS_DOCS_OFFLINE=1`` to skip it.
References to Python and numpy objects are then rendered as plain text, while references to
scatrs objects are still checked.
"""

import builtins
import os
import sys
from importlib.metadata import version as _version

project = "scatrs"
author = "scatrs contributors"
copyright = "2026, scatrs contributors"
release = _version("scatrs")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

# The design records are not part of the site; the build output and executed notebooks are not
# sources.
exclude_patterns = ["_build", "jupyter_execute", "specs/**", "plans/**", "**.ipynb_checkpoints"]
source_suffix = {".md": "myst-nb", ".rst": "restructuredtext"}

# Every cross-reference must resolve.
nitpicky = True

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
# Run every example on every build (see `_always_reread` below), and fail the build if a cell
# raises or writes to stderr.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_output_stderr = "warn"

# No "edit this page" links (furo's source_repository options) until the repository is public.
html_theme = "furo"
html_title = "scatrs"

autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_member_order = "bysource"
# Types are given in the numpydoc "Parameters" and "Returns" sections, and linked from there.
# Signature annotations such as NDArray[np.float64] would render as private numpy paths
# (numpy._typing._array_like.NDArray), and numpy documents float64 and NDArray as attribute
# and data entries, which the class references of annotations do not resolve to.
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "array_like": ":term:`numpy:array_like`",
    "ScatrsError": "scatrs.errors.ScatrsError",
    "InvalidInputError": "scatrs.errors.InvalidInputError",
    "NotConvergedError": "scatrs.errors.NotConvergedError",
    "OutsideValidatedDomainError": "scatrs.errors.OutsideValidatedDomainError",
}
numpydoc_xref_ignore = {"optional", "of"}

OFFLINE = os.environ.get("SCATRS_DOCS_OFFLINE", "") not in ("", "0")

intersphinx_mapping = (
    {}
    if OFFLINE
    else {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
)
intersphinx_timeout = 30


# Python targets that the python and numpy inventories resolve: names under numpy or a
# standard-library module, builtins, and names with an explicit inventory prefix.
_EXTERNAL_MODULES = frozenset({"numpy", "np", *sys.stdlib_module_names})


def _is_external(target):
    if target.startswith(("python:", "numpy:")):
        return True
    target = target.lstrip("~.")
    head = target.split(".")[0]
    return head in _EXTERNAL_MODULES or (head == target and hasattr(builtins, target))


def _external_reference_as_text(app, env, node, contnode):
    """Offline: render a reference that intersphinx would resolve as plain text.

    Anything else, such as a misspelt scatrs name (relative or not), stays unresolved and
    fails the build as usual.
    """
    domain, target = node.get("refdomain"), node.get("reftarget", "")
    if domain == "py" and _is_external(target):
        return contnode
    if domain == "std" and node.get("reftype") == "term" and target.startswith("numpy:"):
        return contnode
    return None


def _always_reread(app, env, added, changed, removed):
    """Re-read the executed examples and the autodoc page on every build, not only on -E.

    Sphinx otherwise skips unchanged sources, so an incremental build would neither re-run the
    examples against the current extension nor pick up changed docstrings.
    """
    return [doc for doc in env.found_docs if doc.startswith("examples/") or doc == "api"]


def setup(app):
    app.connect("env-get-outdated", _always_reread)
    if OFFLINE:
        app.connect("missing-reference", _external_reference_as_text)
