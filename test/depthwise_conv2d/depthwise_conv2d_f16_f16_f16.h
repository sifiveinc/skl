// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#pragma once

#include "skl-test-driver.h"
#include <stddef.h>

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

  // Buffer generation settings for input, filter, output
  SKL_TEST_BUFFER(_Float16) input, filter, output;

  // Indicate whether to test the specialized kernel
  uint32_t use_specialization;

  // Derived parameters & buffers (private to the test harness)
  struct {
    double *input_abs;    // Absolute values of input (for bound computation)
    double *filter_abs;   // Absolute values of filter (for bound computation)
    _Float16 *ref_output; // Reference output
    double *bound;        // Per-element error bound
  } ctx;
} depthwise_conv2d_f16_f16_f16_t;

#define DEPTHWISE_CONV2D_F16_F16_F16_DEFAULTS                                  \
  .input = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},               \
  .filter = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},              \
  .output = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM}

void depthwise_conv2d_f16_f16_f16_init(skl_test_t *t);
void depthwise_conv2d_f16_f16_f16_verify(skl_test_t *t);
void depthwise_conv2d_f16_f16_f16_report(skl_test_t *t);
void depthwise_conv2d_f16_f16_f16_cleanup(skl_test_t *t);
