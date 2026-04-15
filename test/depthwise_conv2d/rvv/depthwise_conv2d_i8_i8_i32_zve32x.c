// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv2d/depthwise_conv2d_i8_i8_i32_zve32x.h"
#include "depthwise_conv2d/depthwise_conv2d_i8_i8_i32.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_zve32x)
#error "This file requires the Zve32x extension."
#endif

/**
 * @brief Test cases for transpose with Zve32x extension.
 *
 * This test uses the depthwise_conv2d_i8_i8_i32 harness.
 */

#define TEST                                                                   \
  DEPTHWISE_CONV2D_I8_I8_I32_DEFAULTS,                                         \
      .steps = {                                                               \
          .init = depthwise_conv2d_i8_i8_i32_init,                             \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = depthwise_conv2d_i8_i8_i32_verify,                         \
          .report = NULL,                                                      \
          .cleanup = depthwise_conv2d_i8_i8_i32_cleanup,                       \
  }

#define BENCH                                                                  \
  DEPTHWISE_CONV2D_I8_I8_I32_DEFAULTS,                                         \
      .steps = {                                                               \
          .init = depthwise_conv2d_i8_i8_i32_init,                             \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = depthwise_conv2d_i8_i8_i32_report,                         \
          .cleanup = depthwise_conv2d_i8_i8_i32_cleanup,                       \
  }

static void execute(skl_test_t *t);

// clang-format off
depthwise_conv2d_i8_i8_i32_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
  // Benchmark tests
{BENCH, 
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 14, .output_width = 14, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },
#endif

#ifdef SKL_ENABLE_TESTS
  // Verification tests

  /* depth_multiplier = 1, dilation = 1, stride = 1 */
  /* Test generic kernel */
  {TEST, .use_specialization = 0,
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 14, .output_width = 14, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },
  /* Test specialized 3x3 kernel */
  {TEST, .use_specialization = 1,
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 14, .output_width = 14, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },

  /* depth_multiplier = 1, dilation = 1, stride = 2 */
  {TEST, .use_specialization = 0,
   .input_height = 15, .input_width = 15, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 7, .output_width = 7, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 2, .stride_width = 2, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },
  {TEST, .use_specialization = 1,
   .input_height = 15, .input_width = 15, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 7, .output_width = 7, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 2, .stride_width = 2, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },

  /* depth_multiplier = 1, dilation = 2, stride = 1 */
  {TEST, .use_specialization = 0,
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 12, .output_width = 12, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 2, .dilation_width_factor = 2,
   .input_zero_point = 0
  },
  {TEST, .use_specialization = 1,
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 12, .output_width = 12, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 2, .dilation_width_factor = 2,
   .input_zero_point = 0
  },

  /* depth_multiplier = 1, dilation = 2, stride = 2 */
  {TEST, .use_specialization = 0,
   .input_height = 15, .input_width = 15, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 6, .output_width = 6, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 2, .stride_width = 2, 
   .dilation_height_factor = 2, .dilation_width_factor = 2,
   .input_zero_point = 0
  },
  {TEST, .use_specialization = 1,
   .input_height = 15, .input_width = 15, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 6, .output_width = 6, .output_channel = 512, 
   .depth_multiplier = 1, .stride_height = 2, .stride_width = 2, 
   .dilation_height_factor = 2, .dilation_width_factor = 2,
   .input_zero_point = 0
  },

  /* depth_multiplier = 2, dilation = 1, stride = 1 */
  {TEST, .use_specialization = 0,
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 14, .output_width = 14, .output_channel = 1024, 
   .depth_multiplier = 2, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },

  /* depth_multiplier = 2, dilation = 1, stride = 2 */
  {TEST, .use_specialization = 0,
   .input_height = 15, .input_width = 15, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 7, .output_width = 7, .output_channel = 1024, 
   .depth_multiplier = 2, .stride_height = 2, .stride_width = 2, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },

  /* depth_multiplier = 2, dilation = 2, stride = 1 */
  {TEST, .use_specialization = 0,
   .input_height = 16, .input_width = 16, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 12, .output_width = 12, .output_channel = 1024, 
   .depth_multiplier = 2, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 2, .dilation_width_factor = 2,
   .input_zero_point = 0
  },

  /* depth_multiplier = 2, dilation = 2, stride = 2 */
  {TEST, .use_specialization = 0,
   .input_height = 15, .input_width = 15, .input_channel = 512, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 6, .output_width = 6, .output_channel = 1024, 
   .depth_multiplier = 2, .stride_height = 2, .stride_width = 2, 
   .dilation_height_factor = 2, .dilation_width_factor = 2,
   .input_zero_point = 0
  },

  /* depth_multiplier = 1, dilation = 1, stride = 1, channel < vlmax */
  /* Test generic kernel */
  {TEST, .use_specialization = 0,
   .input_height = 16, .input_width = 16, .input_channel = 15, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 14, .output_width = 14, .output_channel = 15, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },
  /* Test specialized 3x3 kernel */
  {TEST, .use_specialization = 1,
   .input_height = 16, .input_width = 16, .input_channel = 15, 
   .filter_height = 3, .filter_width = 3, 
   .output_height = 14, .output_width = 14, .output_channel = 15, 
   .depth_multiplier = 1, .stride_height = 1, .stride_width = 1, 
   .dilation_height_factor = 1, .dilation_width_factor = 1,
   .input_zero_point = 0
  },
#endif
};
// clang-format on

static skl_test_suite_t suite = {
    .name = "skl_depthwise_conv2d_i8_i8_i32_zve32x",
    .num_tests = sizeof(tests) / sizeof(tests[0]),
    .test_size = sizeof(depthwise_conv2d_i8_i8_i32_t),
    .tests = tests};

static void execute(skl_test_t *t) {
  const depthwise_conv2d_i8_i8_i32_t *h =
      (depthwise_conv2d_i8_i8_i32_t *)t->harness;

  const int8_t *input = h->input.data;
  const int8_t *filter = h->filter.data;
  int32_t *output = h->output.data;

  if (h->use_specialization && h->filter_height == 3 && h->filter_width == 3 &&
      h->depth_multiplier == 1) {
    skl_depthwise_conv2d_vc_f3x3_sn_dn_m1_in_hwc_i8_i8_i32_zve32x(
        output, input, filter, h->input_height, h->input_width,
        h->input_channel, h->output_height, h->output_width, h->output_channel,
        h->stride_height, h->stride_width, h->dilation_height_factor,
        h->dilation_width_factor, h->input_row_stride, h->input_col_stride,
        h->filter_row_stride, h->filter_col_stride, h->output_row_stride,
        h->output_col_stride, h->input_zero_point);
  } else {
    skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_i8_i8_i32_zve32x(
        output, input, filter, h->input_height, h->input_width,
        h->input_channel, h->filter_height, h->filter_width, h->output_height,
        h->output_width, h->output_channel, h->depth_multiplier,
        h->stride_height, h->stride_width, h->dilation_height_factor,
        h->dilation_width_factor, h->input_row_stride, h->input_col_stride,
        h->filter_row_stride, h->filter_col_stride, h->output_row_stride,
        h->output_col_stride, h->input_zero_point);
  }
}

int main(void) {
  // Set default row and column strides
  for (size_t i = 0; i < suite.num_tests; i++) {
    tests[i].input_col_stride = tests[i].input_col_stride
                                    ? tests[i].input_col_stride
                                    : tests[i].input_channel;
    tests[i].filter_col_stride =
        tests[i].filter_col_stride
            ? tests[i].filter_col_stride
            : tests[i].input_channel * tests[i].depth_multiplier;
    tests[i].output_col_stride = tests[i].output_col_stride
                                     ? tests[i].output_col_stride
                                     : tests[i].output_channel;
    tests[i].input_row_stride =
        tests[i].input_row_stride
            ? tests[i].input_row_stride
            : tests[i].input_width * tests[i].input_channel;
    tests[i].filter_row_stride = tests[i].filter_row_stride
                                     ? tests[i].filter_row_stride
                                     : tests[i].filter_width *
                                           tests[i].input_channel *
                                           tests[i].depth_multiplier;
    tests[i].output_row_stride =
        tests[i].output_row_stride
            ? tests[i].output_row_stride
            : tests[i].output_width * tests[i].output_channel;
  }

  return skl_test_driver_run_suite(&suite);
}
