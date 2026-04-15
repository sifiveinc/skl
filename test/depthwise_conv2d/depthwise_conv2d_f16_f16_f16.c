// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the depthwise_conv2d_f16_f16_f16 test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/depthwise_conv2d_f16_f16_f16_zvfh.c).
 */

#include "depthwise_conv2d_f16_f16_f16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h" // NOLINT(misc-include-cleaner)
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__riscv_zvfh) || !defined(__riscv_zfh)
#error This file requires the Zvfh and Zfh extension
#endif

void depthwise_conv2d_f16_f16_f16_init(skl_test_t *t) {
  depthwise_conv2d_f16_f16_f16_t *h =
      (depthwise_conv2d_f16_f16_f16_t *)t->harness;

  h->input.len = h->input_height * h->input_row_stride;
  h->filter.len = h->filter_height * h->filter_row_stride;
  h->output.len = h->output_height * h->output_row_stride;

  SKL_TEST_BUF_CREATE(t, _Float16, &h->input);
  SKL_TEST_BUF_CREATE(t, _Float16, &h->filter);
  SKL_TEST_BUF_CREATE(t, _Float16, &h->output);

  if (h->steps.verify && h->output.len) {
    h->ctx.ref_output = malloc(h->output.len * sizeof(_Float16));

    // Copy original output contents into ref to check for clobbered data later.
    memcpy(h->ctx.ref_output, h->output.data, h->output.len * sizeof(_Float16));
  }
}

static int fp_eq(float result, float golden, float relative_error) {
  if ((isnan(result) && isnan(golden)) || (isinf(result) && isinf(golden))) {
    return 1;
  }
  // if near zero, do absolute error instead.
  float abs_error =
      relative_error *
      ((fabsf(result) > relative_error) ? fabsf(result) : relative_error);
  return (fabsf(golden - result) <= abs_error);
}

void depthwise_conv2d_f16_f16_f16_verify(skl_test_t *t) {
  depthwise_conv2d_f16_f16_f16_t *h =
      (depthwise_conv2d_f16_f16_f16_t *)t->harness;

  const _Float16 *input = h->input.data;
  const _Float16 *filter = h->filter.data;
  _Float16 *output = h->output.data;
  _Float16 *ref_output = h->ctx.ref_output;

  // Compute reference value
  skl_depthwise_conv2d_hwc_f16_f16_f16_ref(
      ref_output, input, filter, h->input_height, h->input_width,
      h->input_channel, h->filter_height, h->filter_width, h->output_height,
      h->output_width, h->output_channel, h->depth_multiplier, h->stride_height,
      h->stride_width, h->dilation_height_factor, h->dilation_width_factor,
      h->input_row_stride, h->input_col_stride, h->filter_row_stride,
      h->filter_col_stride, h->output_row_stride, h->output_col_stride);

  // Verify result with 1e-3 relative error tolerance (cast to float for
  // comparison)
  for (size_t i = 0; i < h->output_height; ++i) {
    for (size_t j = 0; j < h->output_width; ++j) {
      for (size_t k = 0; k < h->output_channel; ++k) {
        size_t idx = i * h->output_row_stride + j * h->output_col_stride + k;
        if (!fp_eq((float)output[idx], (float)ref_output[idx], 1e-3f)) {
          SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                       "position [%zu, %zu, %zu]: %f != ref %f\n", i, j, k,
                       (float)output[idx], (float)ref_output[idx]);
          t->status.verify_status = SKL_TEST_FAIL;
          return;
        }
      }
    }
  }
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Test pass\n");
}

void depthwise_conv2d_f16_f16_f16_report(skl_test_t *t) {
  depthwise_conv2d_f16_f16_f16_t *h =
      (depthwise_conv2d_f16_f16_f16_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("Input: %zu x %zu x %zu\n", h->input_height, h->input_width,
       h->input_channel);
  INFO("Filter: %zu x %zu x %zu\n", h->filter_height, h->filter_width,
       h->input_channel);
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

  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
#undef INFO
}

void depthwise_conv2d_f16_f16_f16_cleanup(skl_test_t *t) {
  depthwise_conv2d_f16_f16_f16_t *h =
      (depthwise_conv2d_f16_f16_f16_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->input);
  SKL_TEST_BUF_FREE(t, &h->filter);
  SKL_TEST_BUF_FREE(t, &h->output);

  if (h->steps.verify && h->output.len) {
    free(h->ctx.ref_output);
  }
}
