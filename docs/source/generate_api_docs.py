#!/usr/bin/env python3

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def get_namespace_functions(xml_file):
    """Extract functions from a namespace XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        functions = []
        
        # Find all memberdef elements that are functions
        for member in root.findall('.//memberdef[@kind="function"]'):
            name = member.find('name')
            if name is not None:
                # Get the brief description
                brief = member.find('briefdescription/para')
                brief_text = brief.text if brief is not None else ""
                functions.append((name.text, brief_text))
        return functions
    except Exception as e:
        print(f"Warning: Could not parse functions from {xml_file}: {e}")
    return []

def has_namespace_members(xml_file):
    """Check if a namespace has any members."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Check for any memberdef elements
        return len(root.findall('.//memberdef')) > 0
    except Exception as e:
        print(f"Warning: Could not parse namespace XML file {xml_file}: {e}")
        return False

def generate_rst_files(doxygen_xml_dir, output_dir):
    """Generate RST files for all namespaces."""
    # Parse the index.xml file
    index_xml = os.path.join(doxygen_xml_dir, 'index.xml')
    if not os.path.exists(index_xml):
        print(f"Error: Doxygen XML file not found at {index_xml}")
        print("Please make sure Doxygen has been run and generated XML output")
        sys.exit(1)
        
    try:
        tree = ET.parse(index_xml)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)

    # Find all namespaces
    namespaces = root.findall('.//compound[@kind="namespace"]')
    if not namespaces:
        print("Warning: No namespaces found in Doxygen XML output")
        sys.exit(1)
    
    # List to store all generated RST files
    generated_files = []
    
    for ns in namespaces:
        ns_name = ns.find('name').text
        ns_refid = ns.get('refid')
        
        # Check if namespace has any members
        ns_file = os.path.join(doxygen_xml_dir, f"{ns_refid}.xml")
        if not has_namespace_members(ns_file):
            print(f"Skipping empty namespace: {ns_name}")
            continue
            
        # Create RST content
        rst_content = f"""{ns_name}
{'=' * len(ns_name)}

.. doxygennamespace:: {ns_name}
   :project: tinyrend
   :members:
   :undoc-members:
   :content-only:
"""
        
        # Write to file
        output_file = os.path.join(output_dir, f"{ns_name.replace('::', '_')}.rst")
        with open(output_file, 'w') as f:
            f.write(rst_content)
        print(f"Generated {output_file}")
        generated_files.append(ns_name.replace('::', '_'))

    # Generate apis.rst
    apis_content = """API Documentation
==================

.. toctree::
   :maxdepth: 2
   :caption: API Reference

"""
    # Sort the files alphabetically
    generated_files.sort()
    for file in generated_files:
        apis_content += f"   api/{file}\n"

    # Write apis.rst
    apis_file = os.path.join(os.path.dirname(output_dir), 'apis.rst')
    with open(apis_file, 'w') as f:
        f.write(apis_content)
    print(f"Generated {apis_file}")

if __name__ == "__main__":
    # Get paths from environment variables or use defaults
    script_dir = os.path.dirname(os.path.abspath(__file__))
    doxygen_xml_dir = os.getenv('DOXYGEN_XML_DIR', 
                               os.path.join(script_dir, 
                                          '../../build/docs/doxygen/xml'))
    output_dir = os.getenv('API_DOCS_DIR', 
                          os.path.join(script_dir, 'api'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using Doxygen XML directory: {doxygen_xml_dir}")
    print(f"Using API docs output directory: {output_dir}")
    
    generate_rst_files(doxygen_xml_dir, output_dir) 