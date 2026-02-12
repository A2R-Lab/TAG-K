# Configuration file for Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the project root to sys.path so that autodoc can find the package.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "online_estimators"
copyright = "2025, A2R Lab"
author = "Shuo Sha, Anupam Bhakta, Zhenyuan Jiang, Kevin Qiu, Ishaan Mahajan, Gabriel Bravo, Brian Plancher"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc ------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# -- Options for Napoleon (Google/NumPy style docstrings) ---------------------
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx --------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output ---------------------------------------------------
html_baseurl = "http://a2r-lab.org/TAG-K/docs/"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 3,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/A2R-Lab/TAG-K",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": False,
    "show_prev_next": True,
}
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_title = "online_estimators"
html_short_title = "online_estimators"
html_show_sourcelink = False

# -- Custom CSS ----------------------------------------------------------------
html_css_files = [
    "custom.css",
]
