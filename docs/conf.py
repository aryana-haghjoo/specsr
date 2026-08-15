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
    "torch": ("https://pytorch.org/docs/stable", None),
}

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
