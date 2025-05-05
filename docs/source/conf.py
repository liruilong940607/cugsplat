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
    'breathe',
    'exhale'
]

# Breathe configuration
breathe_projects = {
    "cugsplat": "../build/xml"
}
breathe_default_project = "cugsplat"

# Exhale configuration
exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "cugsplat API",
    "doxygenStripFromPath":  "..",
    "createTreeView":        True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    """
INPUT = ../../cugsplat/include
EXTRACT_ALL = YES
EXTRACT_PRIVATE = YES
EXTRACT_STATIC = YES
RECURSIVE = YES
GENERATE_XML = YES
XML_OUTPUT = ../build/xml
ENABLE_PREPROCESSING = YES
MACRO_EXPANSION = YES
EXPAND_ONLY_PREDEF = YES
PREDEFINED += __host__= __device__=
    """
}

# Theme
html_theme = 'sphinx_rtd_theme'

# Source files
source_suffix = '.rst'
master_doc = 'index' 