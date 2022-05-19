# Configuration file for the Sphinx documentation builder.
import os

# -- Project information -----------------------------------------------------

project = 'TQL'
copyright = '2022, de Leon'
author = 'jpdeleon'

release = '0.1'
version = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    # for processing notebooks
    "nbsphinx",
    # for readthedocs action
    "rtds_action",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# See https://github.com/rodluger/starry/blob/master/docs/conf.py

html_theme = 'sphinx_rtd_theme'
html_theme_options = {"display_version": True}
html_last_updated_fmt = "%Y %b %d at %H:%M:%S UTC"
html_show_sourcelink = False

# autodocs
autocass_content = "both"
autosummary_generate = True
autodoc_docstring_signature = True

# Add a heading to notebooks
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note:: This tutorial was generated from a Jupyter notebook than can be downloaded 
          `here <https://github.com/jpdeleon/tql/blob/master/{{ docname }}>`_.
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

# nbsphinx
nbsphinx_execute = "never"
napoleon_use_ivar = True

# -- rtds_action settings ----------------------------------------------------
# see additional instructions in https://github.com/dfm/rtds-action
rtds_action_github_repo = "jpdeleon/tql"
rtds_action_path = "notebooks"
rtds_action_artifact_prefix = "notebooks-for-"
rtds_action_github_token = os.environ.get("GITHUB_TOKEN", "")
rtds_action_error_if_missing = True

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'