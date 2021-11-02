# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# -- Project information -----------------------------------------------------

project = "yolort"
copyright = "2020-2021, yolort community"
author = "Zhiqiang Wang"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_logo = "_static/yolort_logo.png"

mathjax_path = "mathjax/tex-chtml.js"


def autodoc_skip_member(app, what, name, obj, skip, options):
    if name == "training":
        return True
    if name in {"predict_shift_from_features", "forward", "extra_repr"} and not obj.__doc__:
        return True
    # print(app, what, name, obj, skip, options)
    return None  # defer


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


autodoc_inherit_docstrings = False

nbsphinx_requirejs_path = ""
nbsphinx_execute = "never"

nbsphinx_epilog = """
View this document as a notebook:
https://github.com/zhiqwang/yolov5-rt-stack/blob/main/{{ env.doc2path(env.docname, base=None) }}

----
"""

nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

html_theme = "insipid"

html_context = {
    "display_github": True,
    "github_user": "zhiqwang",
    "github_repo": "yolov5-rt-stack",
}
html_theme_options = {
    "left_buttons": [],
    "right_buttons": ["repo-button.html", ],
}
