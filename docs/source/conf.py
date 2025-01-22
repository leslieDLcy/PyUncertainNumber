# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "PyUncertainNumber"
author = "(Leslie) Yu Chen & Ioanna Ioannou & Scott Ferson"
copyright = f"{datetime.datetime.now().year}, (Leslie) Yu Chen"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    #   "myst_nb",
    "myst_parser",
    # 'sphinx.ext.autodoc',
    "autoapi.extension",
    "sphinx_inline_tabs",
    "sphinx.ext.napoleon",
    "sphinx-prompt",
    "sphinx_copybutton",
]

autoapi_dirs = ["../../src"]  # location to parse for API reference

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# def setup(app):
#     app.add_css_file("custom.css")  # may also be an URL


html_static_path = ["_static"]
html_theme = "furo"
html_title = " "
html_logo = "_static/UNlogo3.png"

myst_enable_extensions = [
    "dollarmath",
    "html_admonition",
]

### LaTeX settings ###
# f = open("latex-styling.tex", "r+")
# PREAMBLE = f.read()

# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #'papersize': 'a4paper',
#     # The font size ('10pt', '11pt' or '12pt').
#     #'pointsize': '10pt',
#     # Additional stuff for the LaTeX preamble.
#     "preamble": PREAMBLE
# }

latex_elements = {
    # Additional stuff for the LaTeX preamble.
    # "preamble": r"\usepackage{etoc}",
    "preamble": "\setcounter{tocdepth}{1}",
    "fncychap": "\\usepackage[Conny]{fncychap}",
    "extraclassoptions": "openany,oneside",
}
