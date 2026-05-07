// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_depthwise_conv2d_hwc_i8_i8_i32_ref(
    int32_t *output, const int8_t *input, const int8_t *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    int32_t input_zero_point) {
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

          int32_t acc = 0;

          for (size_t filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (size_t filter_x = 0; filter_x < filter_width; ++filter_x) {
              const size_t in_x =
                  in_x_origin + dilation_width_factor * filter_x;
              const size_t in_y =
                  in_y_origin + dilation_height_factor * filter_y;

              const int8_t input_value =
                  input[in_y * input_row_stride + in_x * input_col_stride + ic];
              const int8_t filter_value =
                  filter[filter_y * filter_row_stride +
                         filter_x * filter_col_stride + oc];

              acc += filter_value * (input_value - input_zero_point);
            }
          }

          output[out_y * output_row_stride + out_x * output_col_stride + oc] =
              acc;
        }
      }
    }
  }
}
