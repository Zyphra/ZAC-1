cmake_minimum_required(VERSION 3.18)
project(zac-cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- CUDA target (H100 = sm_90). Enable with -DGGML_CUDA=ON ---
if (GGML_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "" FORCE)
    endif()
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE) # CUDA-graph replay for real-time decode
endif()

# --- ggml (upstream submodule, pinned to a pristine SHA) ---
# We carry local ggml fixes as patches under patches/ rather than forking the
# submodule. They are (re)applied idempotently at configure time: the reverse
# --check skips an already-patched tree, so re-running cmake is a no-op. If a
# future submodule bump makes a patch no longer apply, configure fails loudly.
find_package(Git REQUIRED)
file(GLOB GGML_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT GGML_PATCHES)
foreach(_patch ${GGML_PATCHES})
    get_filename_component(_name ${_patch} NAME)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/ggml apply --reverse --check ${_patch}
        RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
    if (_applied EQUAL 0)
        message(STATUS "ggml patch already applied: ${_name}")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/ggml apply ${_patch}
            RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if (_rc EQUAL 0)
            message(STATUS "ggml patch applied: ${_name}")
        else()
            message(FATAL_ERROR "Failed to apply ggml patch ${_name}:\n${_err}")
        endif()
    endif()
endforeach()

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

# The CPU decoder calls Accelerate's sgemm directly (src/zac.cpp) so each conv accumulates its taps,
# bias and residual into one output buffer. Every target compiling zac.cpp uses zac_use_accelerate().
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()
function(zac_use_accelerate target)
    if (APPLE)
        target_link_libraries(${target} PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endfunction()

# Decoder, encoder and semantic teacher.
add_library(zac STATIC src/zac.cpp src/zac-encoder.cpp src/ssl-encoder.cpp)
target_include_directories(zac PUBLIC src PRIVATE vendor)
target_link_libraries(zac PUBLIC ggml)
zac_use_accelerate(zac)

# Audio input (ffmpeg) and WAV output.
add_library(zac-io STATIC src/audio-decode.cpp src/wav.cpp)
target_include_directories(zac-io PUBLIC src)

add_executable(zac-cli src/zac-cli.cpp)
target_link_libraries(zac-cli PRIVATE zac zac-io)
add_executable(zac-encoder-cli src/zac-encoder-cli.cpp)
target_link_libraries(zac-encoder-cli PRIVATE zac zac-io)
add_executable(ssl-encoder-cli src/ssl-encoder-cli.cpp)
target_link_libraries(ssl-encoder-cli PRIVATE zac zac-io)

enable_testing()
add_subdirectory(tests)
