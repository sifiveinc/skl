// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zfh)
#error This file requires the Zfh extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_depthwise_conv2d_hwc_f16_f16_f16_scalar(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride) {
  (void)input_height;
  (void)input_width;
  (void)output_channel;

  for (size_t out_y = 0; out_y < output_height; ++out_y) {
    for (size_t out_x = 0; out_x < output_width; ++out_x) {
      for (size_t ic = 0; ic < input_channel; ++ic) {
        for (size_t m = 0; m < depth_multiplier; ++m) {
          const size_t oc = m + ic * depth_multiplier;
          const size_t in_x_origin = out_x * stride_width;
          const size_t in_y_origin = out_y * stride_height;

          _Float16 acc = 0;

          for (size_t filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (size_t filter_x = 0; filter_x < filter_width; ++filter_x) {
              const size_t in_x =
                  in_x_origin + dilation_width_factor * filter_x;
              const size_t in_y =
                  in_y_origin + dilation_height_factor * filter_y;

              const _Float16 input_value =
                  input[in_y * input_row_stride + in_x * input_col_stride + ic];
              const _Float16 filter_value =
                  filter[filter_y * filter_row_stride +
                         filter_x * filter_col_stride + oc];

              acc += filter_value * input_value;
            }
          }

          output[out_y * output_row_stride + out_x * output_col_stride + oc] =
              acc;
        }
      }
    }
  }
}
