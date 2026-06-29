// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the depthwise_conv2d_f16_f16_f16 test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/depthwise_conv2d_f16_f16_f16_zvfh.c).
 */

#include "depthwise_conv2d_f16_f16_f16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
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

  if (h->steps.verify) {
    // Only allocate if lengths are non-zero to avoid malloc(0)
    h->ctx.input_abs =
        h->input.len > 0 ? malloc(h->input.len * sizeof(double)) : NULL;
    h->ctx.filter_abs =
        h->filter.len > 0 ? malloc(h->filter.len * sizeof(double)) : NULL;
    h->ctx.ref_output =
        h->output.len > 0 ? malloc(h->output.len * sizeof(_Float16)) : NULL;
    h->ctx.bound =
        h->output.len > 0 ? malloc(h->output.len * sizeof(double)) : NULL;
  }
}

void depthwise_conv2d_f16_f16_f16_verify(skl_test_t *t) {
  /* Compute the reference output and error bounds. */
  depthwise_conv2d_f16_f16_f16_t *h =
      (depthwise_conv2d_f16_f16_f16_t *)t->harness;

  //
  // Compute the error bound array for comparing test vs reference results.
  //
  // For depthwise convolution, each output element is computed as:
  //     output[oh,ow,oc] = Σ input[ih,iw,ic] × filter[fh,fw,fc]
  //
  // where K = filter_height × filter_width is the number of multiply-accumulate
  // operations per output element.
  //
  // Let u = 2^-P be the maximum relative roundoff error for a floating-point
  // type with P-1 mantissa bits.
  //
  // The error between computed and exact results is bounded by:
  //     ((1 + u)^K - 1) * Σ |input[ih,iw,ic]| × |filter[fh,fw,fc]|
  //
  // Since both test and reference results have roundoff errors, we double this
  // bound using the triangle inequality to get the final comparison threshold.
  //

  // Convert inputs and filters to absolute values (in double precision)
  for (size_t i = 0; i < h->input.len; ++i) {
    h->ctx.input_abs[i] = fabs((double)h->input.data[i]);
  }
  for (size_t i = 0; i < h->filter.len; ++i) {
    h->ctx.filter_abs[i] = fabs((double)h->filter.data[i]);
  }

  const int P = 11; // 10 bits of mantissa for float16 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  const size_t K = h->filter_height * h->filter_width; // Operations per output
  // Compute 2 * ((1 + u)^K - 1) by change of base formula:
  const double roundoff_scaling = 2.0 * expm1((double)K * log1p(u));

  // Compute bound = roundoff_scaling * (|input| ⊗ |filter|)
  // where ⊗ denotes depthwise convolution
  skl_depthwise_conv2d_f64hwc_f64hwim_f64hwc_ref(
      h->ctx.bound, h->ctx.input_abs, h->ctx.filter_abs, h->input_height,
      h->input_width, h->input_channel, h->filter_height, h->filter_width,
      h->output_height, h->output_width, h->output_channel, h->depth_multiplier,
      h->stride_height, h->stride_width, h->dilation_height_factor,
      h->dilation_width_factor, h->input_row_stride, h->input_col_stride,
      h->filter_row_stride, h->filter_col_stride, h->output_row_stride,
      h->output_col_stride);

  // Scale the bound by the roundoff factor
  for (size_t i = 0; i < h->output.len; ++i) {
    h->ctx.bound[i] *= roundoff_scaling;
  }

  // Compute the reference result
  skl_depthwise_conv2d_f16hwc_f16hwim_f16hwc_ref(
      h->ctx.ref_output, h->input.data, h->filter.data, h->input_height,
      h->input_width, h->input_channel, h->filter_height, h->filter_width,
      h->output_height, h->output_width, h->output_channel, h->depth_multiplier,
      h->stride_height, h->stride_width, h->dilation_height_factor,
      h->dilation_width_factor, h->input_row_stride, h->input_col_stride,
      h->filter_row_stride, h->filter_col_stride, h->output_row_stride,
      h->output_col_stride);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < h->output_height; ++i) {
    for (size_t j = 0; j < h->output_width; ++j) {
      for (size_t k = 0; k < h->output_channel; ++k) {
        size_t idx = i * h->output_row_stride + j * h->output_col_stride + k;
        if (fabs((double)h->output.data[idx] - (double)h->ctx.ref_output[idx]) >
            h->ctx.bound[idx]) {
          SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                       "position [%zu, %zu, %zu]: %f != ref %f [bound = %g]\n",
                       i, j, k, (float)h->output.data[idx],
                       (float)h->ctx.ref_output[idx], h->ctx.bound[idx]);
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

  if (h->steps.verify) {
    if (h->input.len) {
      free(h->ctx.input_abs);
    }
    if (h->filter.len) {
      free(h->ctx.filter_abs);
    }
    if (h->output.len) {
      free(h->ctx.ref_output);
      free(h->ctx.bound);
    }
  }
}
