cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gsplat LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Header-only core
add_library(gsplat INTERFACE)
target_include_directories(gsplat INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/gsplat/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
)

# relax restrictions on glm constexpr functions so they can be called in device code.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

function(build_cuda_executables_recursive BASE_DIR PREFIX)
    # 1) grab all .cu/.cpp under BASE_DIR (recursively)
    file(GLOB_RECURSE ALL_SRCS
        "${BASE_DIR}/*.cu"
        "${BASE_DIR}/*.cpp"
    )

    foreach(SRC ${ALL_SRCS})
        # 2) get path relative to BASE_DIR, e.g. "foo/bar/mytest.cpp"
        file(RELATIVE_PATH REL_PATH "${BASE_DIR}" "${SRC}")

        # 3) split into directory part ("foo/bar") and file stem ("mytest")
        get_filename_component(REL_DIR  "${REL_PATH}" PATH)
        get_filename_component(NAME_WE  "${REL_PATH}" NAME_WE)

        # 4) sanitize REL_DIR → replace "/" with "_"
        if(REL_DIR)
            string(REPLACE "/" "_" DIR_NAME "${REL_DIR}")
            set(TARGET_NAME "${PREFIX}_${DIR_NAME}_${NAME_WE}")
            set(OUT_DIR    "${CMAKE_BINARY_DIR}/${PREFIX}/${REL_DIR}")
        else()
            set(TARGET_NAME "${PREFIX}_${NAME_WE}")
            set(OUT_DIR    "${CMAKE_BINARY_DIR}/${PREFIX}")
        endif()

        add_executable(${TARGET_NAME} "${SRC}")
        target_link_libraries(${TARGET_NAME}
            PRIVATE gsplat
            PRIVATE ${TORCH_LIBRARIES}
        )
        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${OUT_DIR}"
            OUTPUT_NAME               "${NAME_WE}"
        )
        message(STATUS 
            "  ↳ ${TARGET_NAME} → ${OUT_DIR}/${NAME_WE}"
        )
    endforeach()
endfunction()

# finally, invoke it on your tests folder with “tests” as the prefix
build_cuda_executables_recursive(
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    tests
)