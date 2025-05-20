# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime

# Add project root to path so we can import the package
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Unlook SDK'
copyright = f'{datetime.datetime.now().year}, Supernova Industries'
author = 'Supernova Industries'

# The version info for the project
from unlook import __version__
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.githubpages',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.inheritance_diagram',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinx_design',
    'sphinx_inline_tabs',
    'sphinx_gallery.gen_gallery',
    'myst_parser',
    'nbsphinx',
    'sphinx_notfound_page',
]

# Configure napoleon for parsing Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Configure sphinx-copybutton
copybutton_prompt_text = ">>> |\\\\$ |\\\\[\\\\d*\\\\]: |In \\\\[\\\\d*\\\\]: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"

# Configure sphinx-gallery
sphinx_gallery_conf = {
    'examples_dirs': '../examples',  # path to example scripts
    'gallery_dirs': 'auto_examples',  # path to gallery generated output
    'filename_pattern': '/example_',  # Include only example files with this pattern
    'ignore_pattern': '/figure_',  # Exclude files with this pattern
}

templates_path = ['_templates']
exclude_patterns = []

# -- MyST Parser configuration -----------------------------------------------
# Enable Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/unlook_logo.png'
html_favicon = '_static/favicon.ico'

# Theme options
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'open3d': ('http://www.open3d.org/docs/', None),
    'opencv': ('https://docs.opencv.org/4.x/', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Autodoc options ---------------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
autoclass_content = 'both'

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = 'never'  # Do not execute notebooks during the build

# -- notfound configuration -------------------------------------------------
notfound_context = {
    'title': 'Page Not Found',
    'body': '''
    <h1>Page Not Found</h1>
    <p>Sorry, we couldn't find that page.</p>
    <p>Try using the search box or go to the <a href="/">homepage</a>.</p>
    ''',
}

# Add custom CSS
def setup(app):
    app.add_css_file('custom.css')