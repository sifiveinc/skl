// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the depthwise_conv2d_i8_i8_i32 test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/skl_depthwise_conv2d_i8_i8_i32_zve32x.c).
 */

#include "depthwise_conv2d_i8_i8_i32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void depthwise_conv2d_i8_i8_i32_init(skl_test_t *t) {
  depthwise_conv2d_i8_i8_i32_t *h = (depthwise_conv2d_i8_i8_i32_t *)t->harness;

  h->input.len = h->input_height * h->input_row_stride;
  h->filter.len = h->filter_height * h->filter_row_stride;
  h->output.len = h->output_height * h->output_row_stride;

  SKL_TEST_BUF_CREATE(t, int8_t, &h->input);
  SKL_TEST_BUF_CREATE(t, int8_t, &h->filter);
  SKL_TEST_BUF_CREATE(t, int32_t, &h->output);

  if (h->steps.verify && h->output.len) {
    h->ctx.ref_output = malloc(h->output.len * sizeof(int32_t));

    // Copy original `at` contents into ref to check for clobbered data later.
    memcpy(h->ctx.ref_output, h->output.data, h->output.len * sizeof(int32_t));
  }
}

void depthwise_conv2d_i8_i8_i32_verify(skl_test_t *t) {
  depthwise_conv2d_i8_i8_i32_t *h = (depthwise_conv2d_i8_i8_i32_t *)t->harness;

  const int8_t *input = h->input.data;
  const int8_t *filter = h->filter.data;
  int32_t *output = h->output.data;
  int32_t *ref_output = h->ctx.ref_output;

  // Compute reference value
  skl_depthwise_conv2d_i8hwc_i8hwim_i32hwc_ref(
      ref_output, input, filter, h->input_height, h->input_width,
      h->input_channel, h->filter_height, h->filter_width, h->output_height,
      h->output_width, h->output_channel, h->depth_multiplier, h->stride_height,
      h->stride_width, h->dilation_height_factor, h->dilation_width_factor,
      h->input_row_stride, h->input_col_stride, h->filter_row_stride,
      h->filter_col_stride, h->output_row_stride, h->output_col_stride,
      h->input_zero_point);

  // Verify result
  for (size_t i = 0; i < h->output_height; ++i) {
    for (size_t j = 0; j < h->output_width; ++j) {
      for (size_t k = 0; k < h->output_channel; ++k) {
        size_t idx = i * h->output_row_stride + j * h->output_col_stride + k;
        if (output[idx] != ref_output[idx]) {
          SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                       "position [%zu, %zu, %zu]: %hhi != ref %hhi\n", i, j, k,
                       output[idx], ref_output[idx]);
          t->status.verify_status = SKL_TEST_FAIL;
          return;
        }
      }
    }
  }
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Test pass\n");
}

void depthwise_conv2d_i8_i8_i32_report(skl_test_t *t) {
  depthwise_conv2d_i8_i8_i32_t *h = (depthwise_conv2d_i8_i8_i32_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("Input: %zu x %zu x %zu\n", h->input_height, h->input_width,
       h->input_channel);
  INFO("Filter: %zu x %zu x %zu\n", h->filter_height, h->filter_width,
       h->output_height);
  INFO("Output: %zu x %zu x %zu\n", h->output_height, h->output_width,
       h->output_channel);
  INFO("Input Stride (Row, Col): (%zu, %zu)\n", h->input_row_stride,
       h->input_col_stride);
  INFO("Filter Stride (Row, Col): (%zu, %zu)\n", h->filter_row_stride,
       h->filter_col_stride);
  INFO("Output Stride (Row, Col): (%zu, %zu)\n", h->output_row_stride,
       h->output_col_stride);
  INFO("Depth Multiplier: %zu\n", h->depth_multiplier);
  INFO("Stride: %zu x %zu\n", h->stride_height, h->stride_width);
  INFO("Dilation: %zu x %zu\n", h->dilation_height_factor,
       h->dilation_width_factor);
  INFO("Input Zero Point: %d\n", h->input_zero_point);

  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
#undef INFO
}

void depthwise_conv2d_i8_i8_i32_cleanup(skl_test_t *t) {
  depthwise_conv2d_i8_i8_i32_t *h = (depthwise_conv2d_i8_i8_i32_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->input);
  SKL_TEST_BUF_FREE(t, &h->filter);
  SKL_TEST_BUF_FREE(t, &h->output);

  if (h->steps.verify && h->output.len) {
    free(h->ctx.ref_output);
  }
}
