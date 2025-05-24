# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime

# -- Path setup --------------------------------------------------------------

# Add project root to path so we can import the package
sys.path.insert(0, os.path.abspath('../..'))

# Check if we're on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    # Mock heavy imports that might fail on ReadTheDocs
    import unittest.mock as mock
    
    # List of modules to mock - these are typically hardware-dependent or heavy dependencies
    MOCK_MODULES = [
        'cv2',
        'open3d',
        'mediapipe',
        'torch',
        'torchvision',
        'tensorflow',
        'cupy',
        'RPi',
        'RPi.GPIO',
        'picamera2',
        'smbus2',
        'adafruit_circuitpython_busdevice',
        'adafruit_circuitpython_register',
        'AS1170',
        'pymeshlab',
        'trimesh',
        'h5py',
        'pyzmq',
        'zmq',
        'netifaces',
        'zeroconf',
    ]
    
    # Create mock modules
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = mock.MagicMock()

# -- Project information -----------------------------------------------------

project = 'Unlook SDK'
copyright = f'{datetime.datetime.now().year}, Supernova Industries'
author = 'Supernova Industries'

# The version info - handle import errors gracefully
try:
    from unlook import __version__
    version = __version__
    release = __version__
except ImportError:
    version = '0.1.0'
    release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'myst_parser',
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

# Configure sphinx-copybutton
copybutton_prompt_text = ">>> |\\$ "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

templates_path = ['_templates']
exclude_patterns = []

# -- MyST Parser configuration -----------------------------------------------
# Enable Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST configuration
myst_enable_extensions = [
    "deflist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Only add logo/favicon if files exist
logo_path = os.path.join(os.path.dirname(__file__), '_static', 'unlook_logo.png')
favicon_path = os.path.join(os.path.dirname(__file__), '_static', 'favicon.ico')

if os.path.exists(logo_path):
    html_logo = '_static/unlook_logo.png'
if os.path.exists(favicon_path):
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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for todo extension ----------------------------------------------

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

# Mock return values for specific attributes
if on_rtd:
    autodoc_mock_imports = MOCK_MODULES
else:
    autodoc_mock_imports = []

# Add custom CSS only if file exists
def setup(app):
    css_file = os.path.join(app.srcdir, '_static', 'custom.css')
    if os.path.exists(css_file):
        app.add_css_file('custom.css')