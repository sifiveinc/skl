# Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.23)

# This option may affect the call to `project` and thus must come first.
option(SKL_ENABLE_CLANG_TIDY "Enable linting with clang-tidy" ON)

project(skl
  VERSION 3.0.0
  LANGUAGES C
)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

# Set up clang-tidy
if(SKL_ENABLE_CLANG_TIDY)
    find_program(SKL_CLANG_TIDY_EXE NAMES
        riscv64-unknown-elf-clang-tidy
        riscv64-unknown-linux-gnu-clang-tidy
    )
    set(SKL_CLANG_TIDY_EXE "${SKL_CLANG_TIDY_EXE}" CACHE FILEPATH "Path to clang-tidy")
    if(SKL_CLANG_TIDY_EXE)
        set(CMAKE_C_CLANG_TIDY "${SKL_CLANG_TIDY_EXE}" CACHE STRING "clang-tidy command")
    endif()
else()
    unset(CMAKE_C_CLANG_TIDY CACHE)
endif()

# Build the library itself

set(SKL_LIBRARY skl)

add_library(${SKL_LIBRARY} STATIC)

# Try to compile SKL with a recent standard with GNU extensions.
set_target_properties(${SKL_LIBRARY}
    PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
)
# But do require at least C99.
target_compile_features(${SKL_LIBRARY} PUBLIC c_std_99)

set(SKL_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wimplicit-fallthrough
    -Werror
    ${SKL_COMPILE_OPTIONS}
)

target_compile_options(${SKL_LIBRARY} PRIVATE ${SKL_COMPILE_OPTIONS})

target_sources(${SKL_LIBRARY} PUBLIC FILE_SET HEADERS
  BASE_DIRS
    include
  FILES
    include/skl.h
)

file(GLOB_RECURSE SKL_SRC_H LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS "src/*.h")
target_sources(${SKL_LIBRARY} PUBLIC FILE_SET source_headers TYPE HEADERS
  BASE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:skl>
  FILES
    ${SKL_SRC_H}
)

# Build the reference library for functional testing
add_subdirectory(ref)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_riscv_extensions.cmake)

function(add_skl_src SRC REQUIRED_EXTENSIONS)
  # Add the source file SRC to the SKL_LIBRARY if all RISC-V
  # extensions in REQUIRED_EXTENSIONS are present in RISCV_EXTENSIONS.
  foreach(EXT ${REQUIRED_EXTENSIONS})
    assert_extension_valid(${EXT})
    if(NOT EXT IN_LIST RISCV_EXTENSIONS)
      return()
    endif()
  endforeach()

  target_sources(${SKL_LIBRARY} PRIVATE ${SRC})
endfunction()

add_skl_src(src/cvt/cvt_f4_f8_zvfofp4min.c "zvfofp4min")
add_skl_src(src/cvt/cvt_f8_bf16_zvfofp8min.c "zvfofp8min")
add_skl_src(src/cvt/cvt_f32_f8_zvfofp8min.c "zvfofp8min")
add_skl_src(src/cvt/cvt_bf16_f8_zvfofp8min_zvfbfmin.c "zvfofp8min;zvfbfmin")

add_skl_src(src/depthwise_conv2d/depthwise_conv2d_f16_f16_f16_zvfh.c "zvfh")
add_skl_src(src/depthwise_conv2d/depthwise_conv2d_f32_f32_f32_zve32f.c "zve32f")
add_skl_src(src/depthwise_conv2d/depthwise_conv2d_i8_i8_i32_zve32x.c "zve32x")

add_skl_src(src/exp/exp_bf16_xsfvfbfa.c "xsfvfbfa")
add_skl_src(src/exp/exp_bf16_xsfvfbfexp16e.c "xsfvfbfexp16e")
add_skl_src(src/exp/exp_bf16_xsfvfexpa_zvfbfmin.c "xsfvfexpa;zvfbfmin")
add_skl_src(src/exp/exp_bf16_zve32f.c "zve32f")
add_skl_src(src/exp/exp_bf16_zvfbfmin.c "zvfbfmin")
add_skl_src(src/exp/exp_f16_xsfvfexp16e.c "xsfvfexp16e")
add_skl_src(src/exp/exp_f16_xsfvfexpa_zvfh.c "xsfvfexpa;zvfh")
add_skl_src(src/exp/exp_f16_zvfh.c "zvfh")
add_skl_src(src/exp/exp_f32_xsfvfexp32e.c "xsfvfexp32e")
add_skl_src(src/exp/exp_f32_xsfvfexpa.c "xsfvfexpa")
add_skl_src(src/exp/exp_f32_zve32f.c "zve32f")

add_skl_src(src/gelu/gelu_f32_zve32f.c "zve32f")

add_skl_src(src/gemm/rvv/gemm_bf16_bf16_f32_zvfbfwma.c "zvfbfwma")
add_skl_src(src/gemm/rvv/gemm_bf16_bf16_f32_zvfbfwma_x390.c "zvfbfwma")
add_skl_src(src/gemm/rvv/gemm_f16_f16_f16_zvfh_x390.c "zvfh")
add_skl_src(src/gemm/rvv/gemm_f16_f16_f32_zvfh.c "zvfh")
add_skl_src(src/gemm/rvv/gemm_f16_f16_f32_zvfh_x390.c "zvfh")
add_skl_src(src/gemm/rvv/gemm_f32_f32_f32_zve32f.c "zve32f")
add_skl_src(src/gemm/rvv/gemm_f32_f32_f32_zve32f_x390.c "zve32f")
add_skl_src(src/gemm/rvv/gemm_f64_f64_f64_zve64d.c "zve64d")
add_skl_src(src/gemm/rvv/gemm_f64_f64_f64_zve64d_x390.c "zve64d")
add_skl_src(src/gemm/rvv/gemm_i8_i8_i32_zve32x.c "zve32x")
add_skl_src(src/gemm/rvv/gemm_i8_i8_i32_zve32x_x390.c "zve32x")
add_skl_src(src/gemm/xsfmm/gemm_bf16c_bf16_f32_xsfmm32a16f.c "xsfmm32a16f")
add_skl_src(src/gemm/xsfmm/gemm_f16c_f16_f32_xsfmm32a16f.c "xsfmm32a16f")
add_skl_src(src/gemm/xsfmm/gemm_f32c_f32_f32_xsfmm32a32f.c "xsfmm32a32f")
add_skl_src(src/gemm/xsfmm/gemm_f8e4m3c_f8e4m3_f32_xsfmm32a8f.c "xsfmm32a8f")
add_skl_src(src/gemm/xsfmm/gemm_f8e5m2c_f8e5m2_f32_xsfmm32a8f.c "xsfmm32a8f")
add_skl_src(src/gemm/xsfmm/gemm_i8c_i8_i32_xsfmm32a8i.c "xsfmm32a8i")
add_skl_src(src/gemm/xsfvqdotq/gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq.c "xsfvqdotq")

add_skl_src(src/logistic/logistic_bf16_xsfvfbfa.c "xsfvfbfa")
add_skl_src(src/logistic/logistic_bf16_xsfvfbfexp16e_xsfvfbfa.c "xsfvfbfexp16e;xsfvfbfa")
add_skl_src(src/logistic/logistic_bf16_xsfvfexp32e.c "xsfvfexp32e")
add_skl_src(src/logistic/logistic_bf16_xsfvfexpa.c "xsfvfexpa")
add_skl_src(src/logistic/logistic_bf16_zve32f.c "zve32f")
add_skl_src(src/logistic/logistic_f16_xsfvfexp16e.c "xsfvfexp16e")
add_skl_src(src/logistic/logistic_f16_zvfh.c "zvfh")
add_skl_src(src/logistic/logistic_f32_xsfvfexp32e.c "xsfvfexp32e")
add_skl_src(src/logistic/logistic_f32_xsfvfexpa.c "xsfvfexpa")
add_skl_src(src/logistic/logistic_f32_zve32f.c "zve32f")

add_skl_src(src/pack/pack_e16_zve32x.c "zve32x")
add_skl_src(src/pack/pack_e32_zve32x.c "zve32x")
add_skl_src(src/pack/pack_e8_zve32x.c "zve32x")
add_skl_src(src/pack/pack_e16_xsfmmbase.c "xsfmmbase")
add_skl_src(src/pack/pack_e32_xsfmmbase.c "xsfmmbase")
add_skl_src(src/pack/pack_e8_xsfmmbase.c "xsfmmbase")

add_skl_src(src/silu/silu_f16_xsfvfexp16e.c "xsfvfexp16e")
add_skl_src(src/silu/silu_f16_xsfvfexpa_zvfh.c "xsfvfexpa;zvfh")
add_skl_src(src/silu/silu_f16_zvfh.c "zvfh")
add_skl_src(src/silu/silu_f32_xsfvfexp32e.c "xsfvfexp32e")
add_skl_src(src/silu/silu_f32_xsfvfexpa.c "xsfvfexpa")
add_skl_src(src/silu/silu_f32_zve32f.c "zve32f")

add_skl_src(src/softmax/softmax_bf16_xsfvfbfa.c "xsfvfbfa")
add_skl_src(src/softmax/softmax_bf16_xsfvfbfexp16e_xsfvfbfa.c "xsfvfbfexp16e;xsfvfbfa")
add_skl_src(src/softmax/softmax_bf16_xsfvfbfexp16e_zvfbfmin.c "xsfvfbfexp16e;zvfbfmin")
add_skl_src(src/softmax/softmax_bf16_xsfvfexp32e_zvfbfmin.c "xsfvfexp32e;zvfbfmin")
add_skl_src(src/softmax/softmax_bf16_xsfvfexpa_xsfvfbfa.c "xsfvfexpa;xsfvfbfa")
add_skl_src(src/softmax/softmax_bf16_xsfvfexpa_zvfbfmin.c "xsfvfexpa;zvfbfmin")
add_skl_src(src/softmax/softmax_bf16_zve32f.c "zve32f")
add_skl_src(src/softmax/softmax_bf16_zvfbfmin.c "zvfbfmin")
add_skl_src(src/softmax/softmax_f16_xsfvfexp16e.c "xsfvfexp16e")
add_skl_src(src/softmax/softmax_f16_xsfvfexpa_zvfh.c "xsfvfexpa;zvfh")
add_skl_src(src/softmax/softmax_f16_zvfh.c "zvfh")
add_skl_src(src/softmax/softmax_f32_xsfvfexp32e.c "xsfvfexp32e")
add_skl_src(src/softmax/softmax_f32_xsfvfexpa.c "xsfvfexpa")
add_skl_src(src/softmax/softmax_f32_zve32f.c "zve32f")

# Optionally build library tests

if(NOT DEFINED SKL_TESTER)
  set(SKL_TESTER "qemu-riscv64")
endif()
if(NOT DEFINED SKL_TESTER_OPTIONS)
  set(SKL_TESTER_OPTIONS ${SKL_QEMU_OPTIONS})
endif()

set(SKL_TEST_LIBS skl-ref m)

set(SKL_ENABLE_TESTS OFF CACHE STRING "Enable building of validation tests.")
message(STATUS "SKL_ENABLE_TESTS: ${SKL_ENABLE_TESTS}")

set(SKL_ENABLE_BENCHMARKS OFF CACHE STRING "Enable building of performance benchmarks.")
message(STATUS "SKL_ENABLE_BENCHMARKS: ${SKL_ENABLE_BENCHMARKS}")

if(SKL_ENABLE_TESTS OR SKL_ENABLE_BENCHMARKS)
  enable_testing()
  add_subdirectory(test)
endif()

# Configure libskl installation

include(GNUInstallDirs)

install(
  TARGETS ${SKL_LIBRARY}
  EXPORT SklTargets
  FILE_SET HEADERS
  FILE_SET source_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/skl
)

install(
  FILES LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/skl
)

install(
  EXPORT SklTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Skl
  NAMESPACE Skl::
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SklConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    cmake/SklConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SklConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Skl
)

# Configure libskl packaging

set(CPACK_PACKAGE_VENDOR "SiFive, Inc")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_GENERATOR "TGZ;STGZ" CACHE STRING
    "List of CPack binary generators to use.")
set(CPACK_SOURCE_GENERATOR TGZ CACHE STRING
    "List of CPack source generators to use.")
set(CPACK_SOURCE_IGNORE_FILES
  ".git*"
  build/
  ".*~$"    # Emacs backup files
  ".*sw[a-p]$"  # Vim swap files
  "*.orig$"  # Git mergetool backup files
)
set(CPACK_VERBATIM_VARIABLES YES)
if(NOT CPack_CMake_INCLUDED STREQUAL 1)
  include(CPack)
endif()
