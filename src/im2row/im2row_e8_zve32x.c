// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>

#include "./im2row_hwc_zve32x.h"

#include "skl-common.h"

SKL_FUNC void skl_im2row_hwc_e8_zve32x(
    uint8_t *output, const uint8_t *input, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte) {
  skl_im2row_hwc_zve32x(
      output, input, sizeof(uint8_t), input_height, input_width, input_channel,
      input_height_stride, input_width_stride, filter_height, filter_width,
      output_height, output_width, padding_width, padding_height, stride_width,
      stride_height, dilation_width, dilation_height, zero_byte);
}
