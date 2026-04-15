// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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

  // Derived parameters & buffers (private to the test harness)
  struct {
    _Float16 *ref_output;
  } ctx;

  uint32_t use_specialization;
} depthwise_conv2d_f16_f16_f16_t;

#define DEPTHWISE_CONV2D_F16_F16_F16_DEFAULTS                                  \
  .input = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},               \
  .filter = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM},              \
  .output = {.min = -1.0f, .max = 1.0f, .mode = SKL_TEST_RANDOM}

void depthwise_conv2d_f16_f16_f16_init(skl_test_t *t);
void depthwise_conv2d_f16_f16_f16_verify(skl_test_t *t);
void depthwise_conv2d_f16_f16_f16_report(skl_test_t *t);
void depthwise_conv2d_f16_f16_f16_cleanup(skl_test_t *t);
