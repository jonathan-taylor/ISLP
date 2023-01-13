# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'ISLP'
copyright = '2023, ISLP authors'
author = 'Jonathan Taylor'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'texext.math_dollar',
    'numpydoc',
    'nbsphinx'
]

graphviz_dot = '/opt/homebrew/bin/dot'
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'pytorch_lightning': ('https://pytorch-lightning.readthedocs.io/en/latest/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
