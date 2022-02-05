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
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# -- Project information -----------------------------------------------------

project = "yolort"
copyright = f"{datetime.now().year}, yolort team"
author = "Zhiqiang Wang, Shiquan Yu, Fidan Kharrasov"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
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
html_logo = "_static/yolort_logo_icon.png"

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

html_theme = "sphinx_material"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "YOLOv5 Runtime Stack",
    # Set you GA account ID to enable tracking
    # "google_analytics_account": "UA-XXXXX",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://zhiqwang.com/yolov5-rt-stack",
    # Set the color and the accent color
    "color_primary": "blue",
    "color_accent": "light-blue",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/zhiqwang/yolov5-rt-stack/",
    "repo_name": "yolort",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 3,
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
    # Text to appear at the top of the home page in a "hero" div.
    "heroes": {
        # We can have heroes for the home pages of training and inferencing in future.
        "index": "A runtime stack for object detection on specialized accelerators."
    },
}

# Disable show source link.
html_show_sourcelink = False

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}
