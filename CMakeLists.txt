# Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# This option may affect the call to `project` and thus must come first.
option(SKL_ENABLE_CLANG_TIDY "Enable linting with clang-tidy" ON)

project(SKL)

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

set(SKL_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wimplicit-fallthrough
    -Werror
    ${SKL_COMPILE_OPTIONS}
)

target_include_directories(${SKL_LIBRARY} PUBLIC include)
target_compile_options(${SKL_LIBRARY} PRIVATE ${SKL_COMPILE_OPTIONS})

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
add_skl_src(src/gemm/rvv/gemm_f16_f16_f16_zvfh_x390.c "zvfh")
add_skl_src(src/gemm/rvv/gemm_f16_f16_f32_zvfh_x390.c "zvfh")
add_skl_src(src/gemm/rvv/gemm_f32_f32_f32_zve32f_x390.c "zve32f")
add_skl_src(src/gemm/rvv/gemm_f64_f64_f64_zve64d_x390.c "zve64d")
add_skl_src(src/gemm/rvv/gemm_i8_i8_i32_zve32x_x390.c "zve32x")
add_skl_src(src/gemm/xsfmm/gemm_a1b01_bf16c_bf16_f32_xsfmm32a16f.c "xsfmm32a16f")
add_skl_src(src/gemm/xsfmm/gemm_a1b01_f16c_f16_f32_xsfmm32a16f.c "xsfmm32a16f")
add_skl_src(src/gemm/xsfmm/gemm_a1b01_f32c_f32_f32_xsfmm32a32f.c "xsfmm32a32f")
add_skl_src(src/gemm/xsfmm/gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f.c "xsfmm32a8f")
add_skl_src(src/gemm/xsfmm/gemm_a1b01_f8e5m2c_f8e5m2_f32_xsfmm32a8f.c "xsfmm32a8f")
add_skl_src(src/gemm/xsfmm/gemm_a1b01_i8c_i8_i32_xsfmm32a8i.c "xsfmm32a8i")
add_skl_src(src/gemm/xsfvqdotq/gemm_i8rcp_i8pc_i32_xsfvqdotq.c "xsfvqdotq")
add_skl_src(src/gemm/xsfvqdotq/pack_b_xsfvqdotq.c "zve32x")

add_skl_src(src/logistic/logistic_f16_xsfvfexp16e.c "xsfvfexp16e")
add_skl_src(src/logistic/logistic_f16_zvfh.c "zvfh")
add_skl_src(src/logistic/logistic_f32_xsfvfexp32e.c "xsfvfexp32e")
add_skl_src(src/logistic/logistic_f32_xsfvfexpa.c "xsfvfexpa")
add_skl_src(src/logistic/logistic_f32_zve32f.c "zve32f")

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

add_skl_src(src/transpose/rvv/transpose_e16_zve32x.c "zve32x")
add_skl_src(src/transpose/rvv/transpose_e32_zve32x.c "zve32x")
add_skl_src(src/transpose/rvv/transpose_e8_zve32x.c "zve32x")
add_skl_src(src/transpose/xsfmm/transpose_e16_xsfmmbase.c "xsfmmbase")
add_skl_src(src/transpose/xsfmm/transpose_e32_xsfmmbase.c "xsfmmbase")
add_skl_src(src/transpose/xsfmm/transpose_e8_xsfmmbase.c "xsfmmbase")

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

# --- Legacy ---

function(skl_add_test_legacy TEST_NAME TEST_SRC OPTS)
  add_executable(test-${TEST_NAME} test/${TEST_SRC})
  target_link_libraries(test-${TEST_NAME}
    PRIVATE
      ${SKL_LIBRARY}
      ${SKL_TEST_LIBS}
  )
  target_compile_options(test-${TEST_NAME} PRIVATE
    ${SKL_COMPILE_OPTIONS}
    ${SKL_TEST_COMPILE_OPTIONS}
    ${OPTS}
  )
  target_compile_definitions(test-${TEST_NAME} PRIVATE SKL_TEST_NAME=${TEST_NAME})
  if (SKL_BUILD_TESTS)
    target_compile_definitions(test-${TEST_NAME} PRIVATE ENABLE_TEST)
  endif()
  if (SKL_BUILD_BENCHMARKS)
    target_compile_definitions(test-${TEST_NAME} PRIVATE ENABLE_BENCHMARK)
  endif()
  target_include_directories(test-${TEST_NAME} PRIVATE test src)
  add_test(
    NAME test-${TEST_NAME}
    COMMAND
    ${SKL_TESTER} ${SKL_TESTER_OPTIONS} $<TARGET_FILE:test-${TEST_NAME}>
  )
endfunction()

if (SKL_BUILD_TESTS OR SKL_BUILD_BENCHMARKS)
  enable_testing()

  if("zve32x" IN_LIST RISCV_EXTENSIONS)
    skl_add_test_legacy(skl_gemm_i8_i8_i32_zve32x_x390_wrapper gemm/gemm_i8rc_i8rc_i32rc.c
      "-DM=32;-DN=32;-DK=32;-DALPHA=1;-DBETA=0")
    skl_add_test_legacy(skl_pack_b_i8_xsfvqdotq gemm/pack_b_i8_xsfvqdotq.c "-DK=126;-DN=128")
  endif()
  if("zvfh" IN_LIST RISCV_EXTENSIONS)
    skl_add_test_legacy(exp_f16 exp/exp_f16.c "-DRUN_ZVFH;-DRUN_VFEXPA;-DRUN_VFEXP")
    skl_add_test_legacy(logistic_f16_zvfh logistic/logistic_f16.c "-DRUN_ALL")
    skl_add_test_legacy(silu_f16_zvfh silu/silu_f16.c "-DRUN_ALL")
  endif()
  skl_add_test_legacy(softmax_f16 softmax/softmax_f16.c "-DRUN_ALL;-DBETA=0.5")
  skl_add_test_legacy(softmax_bf16 softmax/softmax_bf16.c "-DRUN_ALL;-DBETA=0.5")
  if("zve32f" IN_LIST RISCV_EXTENSIONS)
    skl_add_test_legacy(logistic_f32_zve32f logistic/logistic_f32.c "-DRUN_ALL")
    skl_add_test_legacy(gelu_f32 gelu/gelu_f32.c "")
    skl_add_test_legacy(silu_f32_zve32f silu/silu_f32.c "-DRUN_ALL")
  endif()
  if("xsfmm32a8f" IN_LIST RISCV_EXTENSIONS)
    skl_add_test_legacy(skl_gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f_wrapper gemm/gemm_f8e4m3rc_f8e4m3rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test_legacy(skl_gemm_a1b01_f8e4m3pc_f8e4m3cp_f32rcp_xsfmm32a8f_wrapper gemm/gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
    skl_add_test_legacy(skl_gemm_a1b01_f8e5m2c_f8e5m2_f32_xsfmm32a8f_wrapper gemm/gemm_f8e5m2rc_f8e5m2rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test_legacy(skl_gemm_a1b01_f8e5m2pc_f8e5m2cp_f32rcp_xsfmm32a8f_wrapper gemm/gemm_f8e5m2rcprc_f8e5m2rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
  endif()
  if("xsfmm32a8i" IN_LIST RISCV_EXTENSIONS)
    skl_add_test_legacy(skl_gemm_a1b01_i8c_i8_i32_xsfmm32a8i_wrapper gemm/gemm_i8rc_i8rc_i32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1;-DBETA=0")
    skl_add_test_legacy(skl_gemm_a1b01_i8pc_i8cp_i32rcp_xsfmm32a8i_wrapper gemm/gemm_i8rcprc_i8rcprc_i32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1;-DBETA=0")
  endif()
  if("xsfvqdotq" IN_LIST RISCV_EXTENSIONS)
    skl_add_test_legacy(skl_gemm_i8_i8pc_i32_xsfvqdotq gemm/gemm_i8_i8pc_i32_xsfvqdotq.c "-DM=128;-DN=128;-DK=256;-DALPHA=1;-DBETA=1")
    skl_add_test_legacy(skl_gemm_i8rcp_i8pc_i32_xsfvqdotq_wrapper gemm/gemm_i8rcprc_i8rcprc_i32rcprc.c
      "-DM0=1;-DN0=1;-DK0=4;-DM1=128;-DN1=128;-DK1=8;-DCSA1=(2 * K0);-DRSA1=(K1 * CSA1);-DRSB0=1;-DCSB0=0;-DALPHA=1;-DBETA=0;")
  endif()
endif()
