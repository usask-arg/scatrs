"""Sphinx configuration for the scatrs documentation.

autodoc imports scatrs and the examples run it, so install it first::

    uv run --extra docs maturin develop --uv --release
    uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html

Intersphinx downloads the Python and numpy inventories. Without network access that download
fails with a warning, which ``-W`` makes fatal: set ``SCATRS_DOCS_OFFLINE=1`` to skip it.
References to Python and numpy objects are then rendered as plain text, while references to
scatrs objects are still checked.
"""

import os
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
# Run every example on every build, and fail the build if a cell raises.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

html_theme = "furo"
html_title = "scatrs"
html_theme_options = {
    "source_repository": "https://github.com/usask-arg/scatrs/",
    "source_branch": "main",
    "source_directory": "docs/",
}

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
numpydoc_xref_ignore = {"optional"}

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


def _external_reference_as_text(app, env, node, contnode):
    """Offline: resolve a reference intersphinx would have resolved to its plain text."""
    domain, target = node.get("refdomain"), node.get("reftarget", "")
    if domain == "py" and target.split(".")[0] != "scatrs":
        return contnode
    if domain == "std" and node.get("reftype") == "term" and target.startswith("numpy:"):
        return contnode
    return None


def setup(app):
    if OFFLINE:
        app.connect("missing-reference", _external_reference_as_text)
