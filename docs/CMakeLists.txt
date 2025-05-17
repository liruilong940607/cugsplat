find_package(Doxygen REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Find sphinx-build
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sphinx; print(sphinx.__path__[0])"
    OUTPUT_VARIABLE SPHINX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Configure Doxygen
set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/../curend/include)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

# Create output directories
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}/xml)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Add custom target for documentation
add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/source/generate_api_docs.py
            DOXYGEN_XML_DIR=${DOXYGEN_OUTPUT_DIR}/xml
            API_DOCS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/source/api
    COMMAND ${Python3_EXECUTABLE} -m sphinx -b html
            ${CMAKE_CURRENT_SOURCE_DIR}/source
            ${CMAKE_BINARY_DIR}/docs/html
    WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIR}
    COMMENT "Generating documentation"
    VERBATIM
) 