cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gsplat LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Header-only core
add_library(gsplat INTERFACE)
target_include_directories(gsplat INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/gsplat
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
)

# relax restrictions on glm constexpr functions so they can be called in device code.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

function(build_cuda_executables_from DIR SUBDIR_NAME)
    file(GLOB CU_FILES  "${DIR}/*.cu")
    file(GLOB CPP_FILES "${DIR}/*.cpp")

    set(SRC_FILES ${CU_FILES} ${CPP_FILES})

    foreach(FILE_PATH ${SRC_FILES})
        file(READ "${FILE_PATH}" FILE_CONTENTS)
        string(FIND "${FILE_CONTENTS}" "int main" MAIN_FOUND)
        if (MAIN_FOUND EQUAL -1)
            message(STATUS "Skipping ${FILE_PATH} (no main function)")
            continue()
        endif()

        get_filename_component(FILE_NAME_WE ${FILE_PATH} NAME_WE)

        # Unique internal target name (e.g., benchmarks_tmp)
        set(TARGET_NAME "${SUBDIR_NAME}_${FILE_NAME_WE}")

        # Runtime binary name = just "tmp", output to subfolder
        set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/${SUBDIR_NAME}")

        add_executable(${TARGET_NAME} ${FILE_PATH})
        target_link_libraries(${TARGET_NAME} PRIVATE gsplat)
        target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBRARIES})

        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
            OUTPUT_NAME "${FILE_NAME_WE}"  # <-- controls the final binary name (no prefix)
        )

        message(STATUS "Added target: ${TARGET_NAME} → ${OUTPUT_DIR}/${FILE_NAME_WE}")
    endforeach()
endfunction()

# Register folders
# build_cuda_executables_from(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks benchmarks)
# build_cuda_executables_from(${CMAKE_CURRENT_SOURCE_DIR}/examples   examples)
build_cuda_executables_from(${CMAKE_CURRENT_SOURCE_DIR}/tests      tests)
