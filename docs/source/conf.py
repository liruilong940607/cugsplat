import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'cugsplat'
copyright = '2025, Ruilong Li'
author = 'Ruilong Li'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'breathe'
]

# Breathe configuration
breathe_projects = {
    "cugsplat": "../../build/docs/doxygen/xml"
}
breathe_default_project = "cugsplat"

# Theme
html_theme = 'sphinx_rtd_theme'

# Source files
source_suffix = '.rst'
master_doc = 'index' 