
# Set default configuration
set(SKL_TEST_LOG_LEVEL SKL_TEST_LOG_INFO CACHE STRING "Test log level")
message(STATUS "SKL_TEST_LOG_LEVEL: ${SKL_TEST_LOG_LEVEL}")

set(SKL_ENABLE_TESTS ON CACHE STRING "Enabling of verification testing.")
message(STATUS "SKL_ENABLE_TESTS: ${SKL_ENABLE_TESTS}")

set(SKL_ENABLE_BENCHMARKS ON CACHE STRING "Enabling of performance testing.")
message(STATUS "SKL_ENABLE_BENCHMARKS: ${SKL_ENABLE_BENCHMARKS}")


add_library(skl-test-driver OBJECT skl-test-driver.c)
target_compile_definitions(
  skl-test-driver
  PRIVATE SKL_TEST_LOG_LEVEL=${SKL_TEST_LOG_LEVEL}
)
target_compile_options(
  skl-test-driver
  PRIVATE ${SKL_COMPILE_OPTIONS} ${SKL_TEST_COMPILE_OPTIONS}
)

function(skl_add_test TEST_NAME HARNESS_SRC TEST_SRC REQUIRED_EXTENSIONS OPTS)
  # Only add the test if all extensions in REQUIRED_EXTENSIONS are present.
  foreach(EXT ${REQUIRED_EXTENSIONS})
    assert_extension_valid(${EXT})
    if(NOT EXT IN_LIST RISCV_EXTENSIONS)
      return()
    endif()
  endforeach()

  add_executable(test-${TEST_NAME} ${HARNESS_SRC} ${TEST_SRC})
  target_link_libraries(
    test-${TEST_NAME}
    PRIVATE ${SKL_LIBRARY} ${SKL_TEST_LIBS} skl-test-driver
  )
  target_compile_options(
    test-${TEST_NAME}
    PRIVATE ${SKL_COMPILE_OPTIONS} ${SKL_TEST_COMPILE_OPTIONS} ${OPTS}
  )
  target_include_directories(test-${TEST_NAME} PRIVATE . ../src)

  if(SKL_ENABLE_TESTS)
    target_compile_definitions(test-${TEST_NAME} PRIVATE SKL_ENABLE_TESTS)
  endif()
  if(SKL_ENABLE_BENCHMARKS)
    target_compile_definitions(test-${TEST_NAME} PRIVATE SKL_ENABLE_BENCHMARKS)
  endif()

  add_test(
    NAME test-${TEST_NAME}
    COMMAND
      ${SKL_TESTER} ${SKL_TESTER_OPTIONS} $<TARGET_FILE:test-${TEST_NAME}>
  )
endfunction()

skl_add_test(
  skl_depthwise_conv2d_f16_f16_f16_zvfh
  depthwise_conv2d/depthwise_conv2d_f16_f16_f16.c
  depthwise_conv2d/rvv/depthwise_conv2d_f16_f16_f16_zvfh.c
  "zvfh"
  ""
)
skl_add_test(
  skl_depthwise_conv2d_f32_f32_f32_zve32f
  depthwise_conv2d/depthwise_conv2d_f32_f32_f32.c
  depthwise_conv2d/rvv/depthwise_conv2d_f32_f32_f32_zve32f.c
  "zve32f"
  ""
)
skl_add_test(
  skl_depthwise_conv2d_i8_i8_i32_zve32x
  depthwise_conv2d/depthwise_conv2d_i8_i8_i32.c
  depthwise_conv2d/rvv/depthwise_conv2d_i8_i8_i32_zve32x.c
  "zve32x"
  ""
)
skl_add_test(
  skl_cvt_bf16_f8_zvfofp8min_zvfbfmin
  cvt/cvt_bf16_f8.c
  cvt/cvt_bf16_f8_zvfofp8min_zvfbfmin.c
  "zvfofp8min;zvfbfmin"
  ""
)
skl_add_test(
  skl_cvt_f32_f8_zvfofp8min
  cvt/cvt_f32_f8.c
  cvt/cvt_f32_f8_zvfofp8min.c
  "zvfofp8min"
  ""
)
skl_add_test(
  skl_cvt_f4_f8_zvfofp4min
  cvt/cvt_f4_f8.c
  cvt/cvt_f4_f8_zvfofp4min.c
  "zvfofp4min"
  ""
)
skl_add_test(
  skl_cvt_f8_bf16_zvfofp8min
  cvt/cvt_f8_bf16.c
  cvt/cvt_f8_bf16_zvfofp8min.c
  "zvfofp8min"
  ""
)
skl_add_test(
  skl_exp_f32_xsfvfexp32e
  elementwise/unary_f32.c
  exp/exp_f32_xsfvfexp32e.c
  "xsfvfexp32e"
  ""
)
skl_add_test(
  skl_exp_f32_xsfvfexpa
  elementwise/unary_f32.c
  exp/exp_f32_xsfvfexpa.c
  "xsfvfexpa"
  ""
)
skl_add_test(
  skl_exp_f32_zve32f
  elementwise/unary_f32.c
  exp/exp_f32_zve32f.c
  "zve32f"
  ""
)
skl_add_test(
  skl_gemm_f32_f32_f32_zve32f_x390
  gemm/gemm_f32rcprc_f32rcprc_f32rcprc.c
  gemm/rvv/skl_gemm_f32_f32_f32_zve32f_x390.c
  "zve32f"
  ""
)
skl_add_test(
  skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f
  gemm/gemm_f32rcprc_f32rcprc_f32rcprc.c
  gemm/xsfmm/skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f.c
  "xsfmm32a32f"
  ""
)
skl_add_test(
  skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f
  gemm/gemm_f32rcprc_f32rcprc_f32rcprc.c
  gemm/xsfmm/skl_gemm_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f.c
  "xsfmm32a32f"
  ""
)
skl_add_test(
  skl_transpose_e8_xsfmmbase
  transpose/transpose_e8.c
  transpose/xsfmmbase/skl_transpose_e8_xsfmmbase.c
  "xsfmmbase"
  ""
)
skl_add_test(
  skl_transpose_e8_zve32x
  transpose/transpose_e8.c
  transpose/rvv/skl_transpose_e8_zve32x.c
  "zve32x"
  ""
)
