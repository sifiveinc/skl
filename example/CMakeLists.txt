# Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

function(skl_add_example EXAMPLE_NAME EXAMPLE_SRCS REQUIRED_EXTENSIONS OPTS)
  # Only add the example if all extensions in REQUIRED_EXTENSIONS
  # are present in RISCV_EXTENSIONS.
  foreach(EXT ${REQUIRED_EXTENSIONS})
    assert_extension_valid(${EXT})
    if(NOT EXT IN_LIST RISCV_EXTENSIONS)
      return()
    endif()
  endforeach()

  add_executable(example-${EXAMPLE_NAME} ${EXAMPLE_SRCS})
  target_link_libraries(example-${EXAMPLE_NAME}
    PRIVATE
      ${SKL_LIBRARY}
      ${SKL_TEST_LIBS}
  )
  target_compile_options(example-${EXAMPLE_NAME}
    PRIVATE
      ${SKL_COMPILE_OPTIONS}
      ${SKL_TEST_COMPILE_OPTIONS}
      ${OPTS}
  )
  target_compile_definitions(example-${EXAMPLE_NAME}
    PRIVATE
      SKL_EXAMPLE_NAME=${EXAMPLE_NAME}
  )

  if (SKL_ENABLE_TESTS)
    target_compile_definitions(example-${EXAMPLE_NAME} PRIVATE SKL_ENABLE_TESTS)
  endif()
  if (SKL_ENABLE_BENCHMARKS)
    target_compile_definitions(example-${EXAMPLE_NAME} PRIVATE SKL_ENABLE_BENCHMARKS)
  endif()
  target_include_directories(example-${EXAMPLE_NAME}
    PRIVATE
      ${SKL_SOURCE_DIR}/test
      ${SKL_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(
    NAME example-${EXAMPLE_NAME}
    COMMAND
      ${SKL_TESTER} ${SKL_TESTER_OPTIONS} $<TARGET_FILE:example-${EXAMPLE_NAME}>
  )
endfunction()

skl_add_example(
  skl_conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390
  "conv2d/main.c;conv2d/conv2d_nhwc_f32.c"
  "zve32f"
  "-DTEST_IM2ROW_GEMM"
)
skl_add_example(
  skl_conv2d_io_nhwc_filter_hwio_direct_gemm_f32_f32_f32_zve32f_x390
  "conv2d/main.c;conv2d/conv2d_nhwc_f32.c"
  "zve32f"
  ""
)
