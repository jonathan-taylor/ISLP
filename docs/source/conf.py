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
    'myst_nb'
]

graphviz_dot = '/opt/homebrew/bin/dot'
numpydoc_class_members_toctree = False
nb_execution_mode = "cache"
nb_execution_timeout = 60*3 #*100

#nb_kernel_rgx_aliases = {'python3': "islp_test"}

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

html_theme = "sphinx_book_theme" 
html_theme_options = {
    "repository_url": "https://github.com/intro-stat-learning/ISLP.git",
    "use_repository_button": True,
}
html_title = "Introduction to Statistical Learning (Python)"
html_logo = "logo.png"

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
