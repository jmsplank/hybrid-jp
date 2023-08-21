# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Docs location
import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(root))
sys.path.append(str(root / "scripts"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hybrid-jp"
copyright = "2023, James Plank"
author = "James Plank"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/jmsplank/hybrid-jp",
    "use_repository_button": True,
}
html_logo = "_static/logo.jpeg"
html_favicon = "_static/logo.png"
html_title = "hybrid-jp"
