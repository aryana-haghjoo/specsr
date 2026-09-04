"""Sphinx configuration for the specsr documentation."""

from __future__ import annotations

import importlib.metadata

project = "specsr"
author = "Aryana Haghjoo"
copyright = "2026, Aryana Haghjoo"

try:
    release = importlib.metadata.version("specsr")
except importlib.metadata.PackageNotFoundError:  # docs built from a bare checkout
    release = "0.1.0.dev0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# Generate stub pages for the API reference automatically.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Heavy/optional imports must not block a docs build.
autodoc_mock_imports = [
    "torch",
    "wandb",
    "huggingface_hub",
    "ppxf",
    "vorbin",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render "Attributes" as :ivar: fields. Without this, napoleon emits
# .. attribute:: directives that collide with the annotations autodoc already
# documents on a dataclass, producing duplicate-description warnings.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    # docs.pytorch.org directly: pytorch.org/docs/stable redirects there, and
    # the redirect costs a round trip on every build.
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# Sphinx sets no connect timeout by default, so an inventory host that accepts
# no connections hangs the build until the OS gives up -- docs.scipy.org went
# down on 2026-09-03 and cost two minutes per attempt before failing.
intersphinx_timeout = 10

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_title = f"specsr {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/aryana-haghjoo/specsr/",
    "source_branch": "main",
    "source_directory": "docs/",
}


def setup(app):
    """Stop a third party's downtime from failing our docs build.

    The build runs with ``-W``. When an inventory host is unreachable,
    intersphinx emits "failed to reach any of the inventories" *without* a
    warning type, so ``suppress_warnings`` cannot target it -- which hands every
    project we cross-reference a veto over deploying our documentation.
    docs.scipy.org going down on 2026-09-03 failed the 1.1.0 docs deployment
    twice with nothing wrong in this repository.

    A reference into a missing inventory degrades to plain text. That is a much
    better outcome than not publishing, so this drops the record before
    warnings-as-errors sees it. Every other warning, intersphinx's included,
    is still an error.
    """
    import logging as _logging

    class _UnreachableInventory(_logging.Filter):
        def filter(self, record):
            return "failed to reach any of the inventories" not in record.getMessage()

    for _name in list(_logging.root.manager.loggerDict):
        if "intersphinx" in _name:
            _logging.getLogger(_name).addFilter(_UnreachableInventory())
