"""Sphinx configuration for the scatrs documentation.

autodoc imports scatrs and the examples run it. ``uv run`` installs it, rebuilding the extension
whenever the Rust sources have changed::

    uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html

The HTML build also runs ``cargo doc`` and publishes the Rust API reference under ``rustdoc/``;
the ``{rust}`` role links to it (see ``_rust_role``).

Intersphinx downloads the Python and numpy inventories, and the changelog page the GitHub
release notes (sphinx-github-changelog: anonymously, or with the token in
``SPHINX_GITHUB_CHANGELOG_TOKEN``, ``GITHUB_TOKEN``, a git credential helper or the GitHub CLI).
Without network access those downloads fail, which ``-W`` makes fatal: set
``SCATRS_DOCS_OFFLINE=1`` to skip them. References to Python and numpy objects are then
rendered as plain text, while references to scatrs objects are still checked, and the changelog
page links to the releases instead.
"""

import builtins
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import version as _version
from pathlib import Path
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective, SphinxRole

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

# The design records are not part of the site; the build output, the executed notebooks and the
# rustdoc additions are not sources.
exclude_patterns = [
    "_build",
    "_rustdoc",
    "jupyter_execute",
    "specs/**",
    "plans/**",
    "**.ipynb_checkpoints",
]
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

html_theme = "furo"
html_title = "scatrs"
# "View source" and "edit this page" links to the repository.
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
numpydoc_xref_ignore = {"optional", "of"}

OFFLINE = os.environ.get("SCATRS_DOCS_OFFLINE", "") not in ("", "0")
# sphinx-github-changelog may ask git's credential helper for a token: never let git prompt.
os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")

intersphinx_mapping = (
    {}
    if OFFLINE
    else {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
)
intersphinx_timeout = 30

# The changelog page shows the GitHub release notes; offline, `_OfflineChangelog` links to them.
if not OFFLINE:
    extensions.append("sphinx_github_changelog")


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
    # Glossary terms and labels of the inventories (numpydoc links `bool` and `sequence` to
    # python:bltin-boolean-values and python:sequence, `array_like` to numpy:array_like).
    if (
        domain == "std"
        and node.get("reftype") in ("term", "ref")
        and target.startswith(("numpy:", "python:"))
    ):
        return contnode
    return None


def _always_reread(app, env, added, changed, removed):
    """Re-read the executed pages and the autodoc page on every build, not only on -E.

    Sphinx otherwise skips unchanged sources, so an incremental build would neither re-run the
    examples against the current extension nor pick up changed docstrings.
    """
    executed = ("examples/", "utilities/", "validation/", "quickstart", "api")
    return [doc for doc in env.found_docs if doc.startswith(executed)]


# The Rust API reference: rustdoc of the public API, built in a target directory of its own so
# that private items and the pages of removed items never reach the site.
ROOT = Path(__file__).resolve().parent.parent
RUSTDOC = ROOT / "target" / "site-doc" / "doc"
RUSTDOC_OUT = "rustdoc"  # its directory in the HTML output


def _build_rustdoc(app):
    """Run rustdoc (warnings are errors) before any page is read, so that the {rust} role can
    check its targets."""
    if app.builder.format != "html":
        return
    shutil.rmtree(RUSTDOC, ignore_errors=True)
    command = ["cargo", "doc", "-p", "scatrs", "--no-deps", "--locked"]
    command += ["--target-dir", str(RUSTDOC.parent)]
    # A link back to this site on every page; the encoded form allows spaces in the path.
    back_link = Path(__file__).resolve().parent / "_rustdoc" / "back-link.html"
    flags = ["-D", "warnings", "--html-after-content", str(back_link)]
    env = {**os.environ, "CARGO_ENCODED_RUSTDOCFLAGS": "\x1f".join(flags)}
    env.pop("RUSTDOCFLAGS", None)
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise ExtensionError(f"cargo doc failed:\n{result.stderr}")


def _copy_rustdoc(app, exception):
    if exception is None and app.builder.format == "html":
        shutil.copytree(RUSTDOC, Path(app.outdir) / RUSTDOC_OUT, dirs_exist_ok=True)


class _RustRole(SphinxRole):
    """``{rust}`scatrs::mie::Mie::compute``` links a Rust path to its rustdoc page.

    Each part of the path is a module (a directory), then an item (a page such as
    ``struct.Mie.html``), then at most one member (an anchor such as ``#method.compute``). A
    path that rustdoc did not document fails the build. ``{rust}`text <path>``` sets the text.
    """

    def run(self):
        match = re.fullmatch(r"(.+?)\s*<(.+)>", self.text)
        text, path = (match.group(1), match.group(2)) if match else (self.text, self.text)
        target = self._resolve(path)
        if target is None:
            msg = self.inliner.reporter.error(
                f"Rust item not in the rustdoc: {path}", line=self.lineno
            )
            return [nodes.problematic(self.rawtext, self.rawtext, msg)], [msg]
        depth = self.env.docname.count("/")
        uri = "../" * depth + f"{RUSTDOC_OUT}/{target}"
        return [nodes.reference(self.rawtext, "", nodes.literal(text, text), refuri=uri)], []

    @staticmethod
    def _resolve(path):
        parts = path.split("::")
        here = RUSTDOC / parts[0]
        for i, part in enumerate(parts[1:], start=1):
            if (here / part / "index.html").is_file():
                here = here / part
                continue
            pages = list(here.glob(f"*.{part}.html"))
            if len(pages) != 1 or len(parts) > i + 2:
                return None
            page = pages[0].relative_to(RUSTDOC).as_posix()
            if len(parts) == i + 1:
                return page
            html = pages[0].read_text(encoding="utf-8")
            for kind in ("method", "variant", "structfield", "tymethod", "associatedconstant"):
                if f'id="{kind}.{parts[-1]}"' in html:
                    return f"{page}#{kind}.{parts[-1]}"
            return None
        index = here / "index.html"
        return index.relative_to(RUSTDOC).as_posix() if index.is_file() else None


class _OfflineChangelog(SphinxDirective):
    """The ``changelog`` directive without network access: a link to the releases."""

    option_spec: ClassVar = {
        "changelog-url": directives.uri,
        "github": directives.uri,
        "pypi": directives.uri,
    }

    def run(self):
        url = self.options.get("changelog-url") or self.options["github"]
        par = nodes.paragraph()
        par += nodes.Text("The release notes are on ")
        par += nodes.reference("", "GitHub", refuri=url)
        par += nodes.Text(" (not downloaded: SCATRS_DOCS_OFFLINE is set).")
        return [par]


def setup(app):
    app.connect("env-get-outdated", _always_reread)
    app.connect("builder-inited", _build_rustdoc)
    app.connect("build-finished", _copy_rustdoc)
    app.add_role("rust", _RustRole())
    if OFFLINE:
        app.connect("missing-reference", _external_reference_as_text)
        app.add_directive("changelog", _OfflineChangelog)
