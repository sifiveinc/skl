cmake_minimum_required(VERSION 3.21)

project(SKL)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
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

target_include_directories(${SKL_LIBRARY} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SKL_SCALAR_SRCS
  src/depthwise_conv2d/depthwise_conv2d_f32_f32_f32_scalar.c
  src/depthwise_conv2d/depthwise_conv2d_i8_i8_i32_scalar.c
  src/gemm/scalar/gemm_bf16rc_bf16rc_f32rc_scalar.c
  src/gemm/scalar/gemm_f16rc_f16rc_f16rc_scalar.c
  src/gemm/scalar/gemm_f16rc_f16rc_f32rc_scalar.c
  src/gemm/scalar/gemm_f32rc_f32rc_f32rc_scalar.c
  src/gemm/scalar/gemm_f64rc_f64rc_f64rc_scalar.c
  src/gemm/scalar/gemm_i8rc_i8rc_i32rc_scalar.c
  src/gemm/scalar/gemm_bf16rcprc_bf16rcprc_f32rcprc_scalar.c
  src/gemm/scalar/gemm_f16rcprc_f16rcprc_f32rcprc_scalar.c
  src/gemm/scalar/gemm_f32rcprc_f32rcprc_f32rcprc_scalar.c
  src/gemm/scalar/gemm_f64rcprc_f64rcprc_f64rcprc_scalar.c
  src/gemm/scalar/gemm_i8rcprc_i8rcprc_i32rcprc_scalar.c
  src/logistic/logistic_f16_scalar.c
  src/logistic/logistic_f32_scalar.c
  src/softmax/softmax_f16_scalar.c
  src/softmax/softmax_bf16_scalar.c
  src/softmax/softmax_f32_scalar.c
  src/transpose/scalar/transpose_e8_scalar.c
  src/transpose/scalar/transpose_e16_scalar.c
  src/transpose/scalar/transpose_e32_scalar.c
)

set(SKL_ZFH_SRCS
  src/depthwise_conv2d/depthwise_conv2d_f16_f16_f16_scalar.c
)

set(SKL_ZVFH_SRCS
  src/depthwise_conv2d/depthwise_conv2d_f16_f16_f16_zvfh.c
  src/exp/exp_f16_zvfh.c
  src/gemm/rvv/gemm_f16_f16_f16_zvfh_x390.c
  src/gemm/rvv/gemm_f16_f16_f32_zvfh_x390.c
  src/logistic/logistic_f16_zvfh.c
  src/softmax/softmax_f16_zvfh.c
)

set(SKL_ZVFBFWMA_SRCS
  src/gemm/rvv/gemm_bf16_bf16_f32_zvfbfwma.c
)

set(SKL_ZVE32F_SRCS
  src/depthwise_conv2d/depthwise_conv2d_f32_f32_f32_zve32f.c
  src/exp/exp_bf16_zve32f.c
  src/exp/exp_f32_zve32f.c
  src/gemm/rvv/gemm_f32_f32_f32_zve32f_x390.c
  src/logistic/logistic_f32_zve32f.c
  src/softmax/softmax_bf16_zve32f.c
  src/softmax/softmax_f32_zve32f.c
  src/gelu/gelu_f32_zve32f.c
)

set(SKL_ZVE32X_SRCS
  src/gemm/rvv/gemm_i8_i8_i32_zve32x_x390.c
  src/gemm/xsfvqdotq/pack_b_xsfvqdotq.c
  src/depthwise_conv2d/depthwise_conv2d_i8_i8_i32_zve32x.c
  src/transpose/rvv/transpose_e8_zve32x.c
  src/transpose/rvv/transpose_e16_zve32x.c
  src/transpose/rvv/transpose_e32_zve32x.c
)

set(SKL_XSFMM_SRCS
  src/transpose/xsfmm/transpose_e8_xsfmmbase.c
  src/transpose/xsfmm/transpose_e16_xsfmmbase.c
  src/transpose/xsfmm/transpose_e32_xsfmmbase.c
)

set (SKL_XSFMM32A8F_SRCS
  src/gemm/xsfmm/gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f.c
  src/gemm/xsfmm/gemm_a1b01_f8e5m2c_f8e5m2_f32_xsfmm32a8f.c
)

set(SKL_XSFMM32A8I_SRCS
  src/gemm/xsfmm/gemm_a1b01_i8c_i8_i32_xsfmm32a8i.c
)

set(SKL_XSFMM32A16F_SRCS
  src/gemm/xsfmm/gemm_a1b01_bf16c_bf16_f32_xsfmm32a16f.c
  src/gemm/xsfmm/gemm_a1b01_f16c_f16_f32_xsfmm32a16f.c
)

set(SKL_XSFMM32A32F_SRCS
  src/gemm/xsfmm/gemm_a1b01_f32c_f32_f32_xsfmm32a32f.c
)

set(SKL_XSFVFBFA_SRCS
  src/exp/exp_bf16_xsfvfbfa.c
  src/softmax/softmax_bf16_xsfvfbfa.c
)

set(SKL_XSFVFEXPA_ZVFH_SRCS
  src/exp/exp_f16_xsfvfexpa_zvfh.c
  src/softmax/softmax_f16_xsfvfexpa_zvfh.c
)

set(SKL_XSFVFEXPA_XSFVFBFA_SRCS
  src/softmax/softmax_bf16_xsfvfexpa_xsfvfbfa.c
)

set(SKL_XSFVFEXPA_ZVFBFMIN_SRCS
  src/exp/exp_bf16_xsfvfexpa_zvfbfmin.c
  src/softmax/softmax_bf16_xsfvfexpa_zvfbfmin.c
)

set(SKL_XSFVFEXPA_SRCS
  src/exp/exp_f32_xsfvfexpa.c
  src/logistic/logistic_f32_xsfvfexpa.c
  src/softmax/softmax_f32_xsfvfexpa.c
)

set(SKL_XSFVFEXP16E_SRCS
  src/exp/exp_f16_xsfvfexp16e.c
  src/logistic/logistic_f16_xsfvfexp16e.c
  src/softmax/softmax_f16_xsfvfexp16e.c
)

set(SKL_XSFVFBFEXP16E_SRCS
  src/exp/exp_bf16_xsfvfbfexp16e.c
)

set(SKL_XSFVFBFEXP16E_XSFVFBFA_SRCS
  src/softmax/softmax_bf16_xsfvfbfexp16e_xsfvfbfa.c
)

set(SKL_XSFVFBFEXP16E_ZVFBFMIN_SRCS
  src/softmax/softmax_bf16_xsfvfbfexp16e_zvfbfmin.c
)

set(SKL_XSFVFEXP32E_SRCS
  src/exp/exp_f32_xsfvfexp32e.c
  src/softmax/softmax_f32_xsfvfexp32e.c
)

set(SKL_XSFVFEXP32E_ZVFBFMIN_SRCS
  src/softmax/softmax_bf16_xsfvfexp32e_zvfbfmin.c
)

set(SKL_ZVE64D_SRCS
  src/gemm/rvv/gemm_f64_f64_f64_zve64d_x390.c
)

set(SKL_ZVFBFMIN_SRCS
  src/exp/exp_bf16_zvfbfmin.c
  src/softmax/softmax_bf16_zvfbfmin.c
)

set(SKL_ZVFOFP8MIN_SRCS
  src/cvt/cvt_zvfofp8min.c
)

set(SKL_ZVFOFP8MIN_ZVFBFMIN_SRCS
  src/cvt/cvt_zvfofp8min_zvfbfmin.c
)

set(SKL_ZVFOFP4MIN_SRCS
  src/cvt/cvt_zvfofp4min.c
)
set(SKL_XSFVQDOTQ_SRCS
  src/gemm/xsfvqdotq/gemm_i8_i8pc_i32_xsfvqdotq.c
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-riscv-extensions.cmake)

if(RISCV_ZFH)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZFH_SRCS})
endif()

if(RISCV_ZVFH)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVFH_SRCS})
endif()

if(RISCV_ZVFBFWMA)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVFBFWMA_SRCS})
endif()

if(RISCV_ZVE32F)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVE32F_SRCS})
endif()

if(RISCV_ZVE64D)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVE64D_SRCS})
endif()

if(RISCV_ZVE32X)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVE32X_SRCS})
endif()

if(RISCV_XSFMM)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFMM_SRCS})
endif()

if(RISCV_XSFMM32A8F)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFMM32A8F_SRCS})
endif()

if(RISCV_XSFMM32A8I)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFMM32A8I_SRCS})
endif()

if(RISCV_XSFMM32A16F)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFMM32A16F_SRCS})
endif()

if(RISCV_XSFMM32A32F)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFMM32A32F_SRCS})
endif()

if(RISCV_XSFVFBFA)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFBFA_SRCS})
endif()

if(RISCV_ZVFH AND RISCV_XSFVFEXPA)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXPA_ZVFH_SRCS})
endif()

if(RISCV_XSFVFEXPA AND RISCV_XSFVFBFA)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXPA_XSFVFBFA_SRCS})
endif()

if(RISCV_XSFVFEXPA AND RISCV_ZVFBFMIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXPA_ZVFBFMIN_SRCS})
endif()

if(RISCV_XSFVFEXPA)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXPA_SRCS})
endif()

if(RISCV_XSFVFEXP16E)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXP16E_SRCS})
endif()

if(RISCV_XSFVFBFEXP16E)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFBFEXP16E_SRCS})
endif()

if(RISCV_XSFVFBFEXP16E AND RISCV_XSFVFBFA)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFBFEXP16E_XSFVFBFA_SRCS})
endif()

if(RISCV_XSFVFBFEXP16E AND RISCV_ZVFBFMIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFBFEXP16E_ZVFBFMIN_SRCS})
endif()

if(RISCV_XSFVFEXP32E)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXP32E_SRCS})
endif()

if(RISCV_XSFVFEXP32E AND RISCV_ZVFBFMIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVFEXP32E_ZVFBFMIN_SRCS})
endif()

if(RISCV_ZVFBFMIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVFBFMIN_SRCS})
endif()

if(RISCV_ZVFOFP8MIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVFOFP8MIN_SRCS})
endif()

if(RISCV_ZVFOFP8MIN AND RISCV_ZVFBFMIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVFOFP8MIN_ZVFBFMIN_SRCS})
endif()

if(RISCV_ZVFOFP4MIN)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_ZVFOFP4MIN_SRCS})
endif()

if(RISCV_XSFVQDOTQ)
  target_sources(${SKL_LIBRARY} PRIVATE ${SKL_XSFVQDOTQ_SRCS})
endif()

target_sources(${SKL_LIBRARY} PRIVATE ${SKL_SCALAR_SRCS})
target_compile_options(${SKL_LIBRARY} PRIVATE ${SKL_COMPILE_OPTIONS})

# Optionally build library tests

if(NOT DEFINED SKL_TESTER)
  set(SKL_TESTER "qemu-riscv64")
endif()
if(NOT DEFINED SKL_TESTER_OPTIONS)
  set(SKL_TESTER_OPTIONS ${SKL_QEMU_OPTIONS})
endif()

set(SKL_TEST_LIBS m)

function(skl_add_test TEST_NAME TEST_SRC OPTS)
  add_executable(test-${TEST_NAME} test/${TEST_SRC})
  target_link_libraries(test-${TEST_NAME} PRIVATE ${SKL_LIBRARY} ${SKL_TEST_LIBS})
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

function(skl_add_example EXAMPLE_NAME EXAMPLE_SRCS OPTS)
  # Convert single source file to list for backward compatibility
  if(NOT "${EXAMPLE_SRCS}" MATCHES ";")
    set(EXAMPLE_SRCS_LIST "example/${EXAMPLE_SRCS}")
  else()
    # Handle multiple source files
    set(EXAMPLE_SRCS_LIST "")
    foreach(SRC ${EXAMPLE_SRCS})
      list(APPEND EXAMPLE_SRCS_LIST "example/${SRC}")
    endforeach()
  endif()

  add_executable(example-${EXAMPLE_NAME} ${EXAMPLE_SRCS_LIST})
  target_link_libraries(example-${EXAMPLE_NAME} PRIVATE ${SKL_LIBRARY} ${SKL_TEST_LIBS})
  target_compile_options(example-${EXAMPLE_NAME} PRIVATE
    ${SKL_COMPILE_OPTIONS}
    ${SKL_TEST_COMPILE_OPTIONS}
    ${OPTS}
  )
  target_compile_definitions(example-${EXAMPLE_NAME} PRIVATE SKL_EXAMPLE_NAME=${EXAMPLE_NAME})
  if (SKL_BUILD_TESTS)
    target_compile_definitions(example-${EXAMPLE_NAME} PRIVATE ENABLE_TEST)
  endif()
  if (SKL_BUILD_BENCHMARKS)
    target_compile_definitions(example-${EXAMPLE_NAME} PRIVATE ENABLE_BENCHMARK)
  endif()
  target_include_directories(example-${EXAMPLE_NAME} PRIVATE test src example)
  add_test(
    NAME example-${EXAMPLE_NAME}
    COMMAND
    ${SKL_TESTER} ${SKL_TESTER_OPTIONS} $<TARGET_FILE:example-${EXAMPLE_NAME}>
  )
endfunction()

if (SKL_BUILD_TESTS OR SKL_BUILD_BENCHMARKS)
  enable_testing()

  # Examples
  if (RISCV_ZVE32F)
    skl_add_example(skl_conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390 "conv2d/main.c;conv2d/conv2d_nhwc_f32.c" "-DTEST_IM2ROW_GEMM")
    skl_add_example(skl_conv2d_io_nhwc_filter_hwio_direct_gemm_f32_f32_f32_zve32f_x390 "conv2d/main.c;conv2d/conv2d_nhwc_f32.c" "")
  endif()

  if(RISCV_ZVE32X)
    skl_add_test(skl_depthwise_conv2d_i8_i8_i32_zve32x depthwise_conv2d/depthwise_conv2d_i8_i8_i32.c "")
    skl_add_test(skl_gemm_i8_i8_i32_zve32x_x390_wrapper gemm/gemm_i8rc_i8rc_i32rc.c
      "-DM=32;-DN=32;-DK=32;-DALPHA=1;-DBETA=0")
    skl_add_test(skl_transpose_e8_zve32x transpose/transpose_e8.c "")
    skl_add_test(skl_transpose_e16_zve32x transpose/transpose_e16.c "")
    skl_add_test(skl_transpose_e32_zve32x transpose/transpose_e32.c "")
    skl_add_test(skl_pack_b_i8_xsfvqdotq gemm/pack_b_i8_xsfvqdotq.c "-DK=126;-DN=128")
  endif()
  if(RISCV_ZVFH)
    skl_add_test(skl_depthwise_conv2d_f16_f16_f16_zvfh depthwise_conv2d/depthwise_conv2d_f16_f16_f16.c "")
    skl_add_test(exp_f16 exp/exp_f16.c "-DRUN_ZVFH;-DRUN_VFEXPA;-DRUN_VFEXP")
    skl_add_test(skl_gemm_f16_f16_f16_zvfh_x390_wrapper gemm/gemm_f16rc_f16rc_f16rc.c
      "-DM=32;-DN=32;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_f16_f16_f32_zvfh_x390_wrapper gemm/gemm_f16rc_f16rc_f32rc.c
      "-DM=32;-DN=32;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(logistic_f16 logistic/logistic_f16.c "-DRUN_ALL")
  endif()
  if(RISCV_ZVFBFWMA)
    skl_add_test(skl_gemm_bf16_bf16_f32_zvfbfwma_wrapper gemm/gemm_bf16rc_bf16rc_f32rc.c
      "-DM=4;-DN=16;-DK=128;-DALPHA=1.f;-DBETA=0.f")
  endif()
  skl_add_test(softmax_f16 softmax/softmax_f16.c "-DRUN_ALL;-DBETA=0.5")
  skl_add_test(softmax_bf16 softmax/softmax_bf16.c "-DRUN_ALL;-DBETA=0.5")
  skl_add_test(softmax_f32 softmax/softmax_f32.c "-DRUN_ALL;-DBETA=0.5")
  skl_add_test(softmax_2d_f32 softmax/softmax_2d_f32.c
    "-DRUN_ALL;-DM=64;-DN=128;-DBETA=0.5")
  if (RISCV_ZVE32F)
    skl_add_test(skl_depthwise_conv2d_f32_f32_f32_zve32f depthwise_conv2d/depthwise_conv2d_f32_f32_f32.c "")
    skl_add_test(exp_f32 exp/exp_f32.c "-DRUN_ZVE32F;-DRUN_VFEXPA;-DRUN_VFEXP")
    skl_add_test(exp_bf16 exp/exp_bf16.c "-DRUN_ZVE32F;-DRUN_VFBFA;-DRUN_VFEXP;-DRUN_VFEXPA")
    skl_add_test(skl_gemm_f32_f32_f32_zve32f_x390_wrapper gemm/gemm_f32rc_f32rc_f32rc.c
      "-DM=32;-DN=32;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(logistic_f32_zve32f logistic/logistic_f32.c "-DRUN_ALL")
    skl_add_test(gelu_f32 gelu/gelu_f32.c "")
  endif()
  if(RISCV_ZVE64D)
    skl_add_test(skl_gemm_f64_f64_f64_zve64d_x390_wrapper gemm/gemm_f64rc_f64rc_f64rc.c
      "-DM=32;-DN=32;-DK=32;-DALPHA=1.;-DBETA=0.")
  endif()
  if(RISCV_XSFMM32A8F)
    skl_add_test(skl_gemm_a1b01_f8e4m3c_f8e4m3_f32_xsfmm32a8f_wrapper gemm/gemm_f8e4m3rc_f8e4m3rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_f8e4m3pc_f8e4m3cp_f32rcp_xsfmm32a8f_wrapper gemm/gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_f8e5m2c_f8e5m2_f32_xsfmm32a8f_wrapper gemm/gemm_f8e5m2rc_f8e5m2rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_f8e5m2pc_f8e5m2cp_f32rcp_xsfmm32a8f_wrapper gemm/gemm_f8e5m2rcprc_f8e5m2rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
  endif()
  if(RISCV_XSFMM32A8I)
    skl_add_test(skl_gemm_a1b01_i8c_i8_i32_xsfmm32a8i_wrapper gemm/gemm_i8rc_i8rc_i32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1;-DBETA=0")
    skl_add_test(skl_gemm_a1b01_i8pc_i8cp_i32rcp_xsfmm32a8i_wrapper gemm/gemm_i8rcprc_i8rcprc_i32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1;-DBETA=0")
  endif()
  if(RISCV_XSFMM32A16F)
    skl_add_test(skl_gemm_a1b01_bf16c_bf16_f32_xsfmm32a16f_wrapper gemm/gemm_bf16rc_bf16rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_f16c_f16_f32_xsfmm32a16f_wrapper gemm/gemm_f16rc_f16rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f_wrapper gemm/gemm_bf16rcprc_bf16rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_f16pc_f16cp_f32rcp_xsfmm32a16f_wrapper gemm/gemm_f16rcprc_f16rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
  endif()
  if(RISCV_XSFMM32A32F)
    skl_add_test(skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f_wrapper gemm/gemm_f32rc_f32rc_f32rc.c
      "-DRSA=1;-DCSA=128;-DM=128;-DM=128;-DN=128;-DK=32;-DALPHA=1.f;-DBETA=0.f")
    skl_add_test(skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f_wrapper gemm/gemm_f32rcprc_f32rcprc_f32rcprc.c
      "-DM0=${XSFMM_TE};-DN0=${XSFMM_TE};-DK0=1;-DM1=2;-DN1=2;-DK1=32;-DRSA0=1;-DCSA0=0;-DRSB0=0;-DCSB0=1;-DRSB1=(K0 * N0);-DCSB1=(K0 * N0 * K1);-DALPHA=1.f;-DBETA=0.f")
  endif()
  if(RISCV_XSFMM)
    skl_add_test(skl_transpose_e8_xsfmmbase transpose/transpose_e8.c "")
    skl_add_test(skl_transpose_e16_xsfmmbase transpose/transpose_e16.c "")
    skl_add_test(skl_transpose_e32_xsfmmbase transpose/transpose_e32.c "")
  endif()
  if(RISCV_ZVFOFP8MIN)
    #For SCALING_TEST_MODE, 1: no scaling, 2: scaling, 3: both
    #For SATURATION_TEST_MODE, 1: no saturation, 2: saturation, 3: both
    skl_add_test(skl_cvt_zvfofp8min cvt/cvt_ofp_zvfofp8min.c
      "-DRUN_F32_E4M3;-DRUN_F32_E5M2;-DRUN_E4M3_BF16;-DRUN_E5M2_BF16;-DSCALING_TEST_MODE=3;-DSATURATION_TEST_MODE=3")
  endif()
  if(RISCV_ZVFOFP8MIN AND RISCV_ZVFBFMIN)
    #For SCALING_TEST_MODE, 1: no scaling, 2: scaling, 3: both
    #For SATURATION_TEST_MODE, 1: no saturation, 2: saturation, 3: both
    skl_add_test(skl_cvt_zvfofp8min_zvfbfmin cvt/cvt_ofp_zvfofp8min_zvfbfmin.c
      "-DRUN_BF16_E4M3;-DRUN_BF16_E5M2;-DSCALING_TEST_MODE=3;-DSATURATION_TEST_MODE=3")
  endif()
  if(RISCV_ZVFOFP4MIN)
    skl_add_test(skl_cvt_zvfofp4min cvt/cvt_ofp_zvfofp4min.c "")
  endif()
  if(RISCV_XSFVQDOTQ)
    skl_add_test(skl_gemm_i8_i8pc_i32_xsfvqdotq gemm/gemm_i8_i8pc_i32_xsfvqdotq.c "-DM=128;-DN=128;-DK=256;-DALPHA=1;-DBETA=1")
    skl_add_test(skl_gemm_i8_i8pc_i32_xsfvqdotq_wrapper gemm/gemm_i8rcprc_i8rcprc_i32rcprc.c
      "-DM0=1;-DN0=1;-DK0=4;-DM1=12;-DN1=128;-DK1=8;-DRSB0=1;-DCSB0=0;-DALPHA=1;-DBETA=0;")
    skl_add_test(skl_gemm_i8_i8pc_i32_xsfvqdotq_x390_clp_wrapper gemm/gemm_i8rcprc_i8rcprc_i32rcprc.c
      "-DM0=1;-DN0=1;-DK0=4;-DM1=12;-DN1=128;-DK1=8;-DRSB0=1;-DCSB0=0;-DALPHA=1;-DBETA=0;")
  endif()
endif()
