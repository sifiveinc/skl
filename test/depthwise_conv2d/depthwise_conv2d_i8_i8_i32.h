// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  // Test function pointers for various steps
  // *** This field must be placed first within this struct ***
  skl_test_steps_t steps;

  // Configurable parameters (arguments to depthwise_conv2d function)
  size_t input_height, input_width, input_channel;
  size_t filter_height, filter_width;
  size_t output_height, output_width, output_channel;

  size_t input_row_stride, input_col_stride;
  size_t filter_row_stride, filter_col_stride;
  size_t output_row_stride, output_col_stride;

  size_t depth_multiplier;
  size_t stride_height, stride_width;
  size_t dilation_height_factor, dilation_width_factor;

  int32_t input_zero_point;

  // Buffer generation settings for input, filter, output
  SKL_TEST_BUFFER(int8_t) input, filter;
  SKL_TEST_BUFFER(int32_t) output;

  // Indicate whether to test the specialized kernel
  uint32_t use_specialization;

  // Derived parameters & buffers (private to the test harness)
  struct {
    int32_t *ref_output;
  } ctx;
} depthwise_conv2d_i8_i8_i32_t;

#define DEPTHWISE_CONV2D_I8_I8_I32_DEFAULTS                                    \
  .input = {.min = -128, .max = 127, .mode = SKL_TEST_RANDOM},                 \
  .filter = {.min = -128, .max = 127, .mode = SKL_TEST_RANDOM},                \
  .output = {.min = -1000, .max = 1000, .mode = SKL_TEST_RANDOM}

void depthwise_conv2d_i8_i8_i32_init(skl_test_t *t);
void depthwise_conv2d_i8_i8_i32_verify(skl_test_t *t);
void depthwise_conv2d_i8_i8_i32_report(skl_test_t *t);
void depthwise_conv2d_i8_i8_i32_cleanup(skl_test_t *t);
