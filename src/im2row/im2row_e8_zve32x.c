// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>

#include "./im2row_hwc_zve32x.h"

#include "skl-common.h"

SKL_FUNC void skl_im2row_hwc_e8_zve32x(
    uint8_t *out, const uint8_t *in_batch, int32_t in_h_origin,
    int32_t in_w_origin, int32_t in_c_origin, size_t input_height,
    size_t input_width, size_t filter_height, size_t filter_width,
    size_t patch_channel, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_height_stride,
    size_t input_width_stride, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements) {
  skl_im2row_hwc_zve32x(
      out, in_batch, sizeof(uint8_t), in_h_origin, in_w_origin, in_c_origin,
      input_height, input_width, filter_height, filter_width, patch_channel,
      dilation_height_factor, dilation_width_factor, input_height_stride,
      input_width_stride, zero_byte, patch_begin_coord, patch_elements);
}

SKL_FUNC void skl_im2row_d1_full_patch_hwc_e8_zve32x(
    uint8_t *out, const uint8_t *in_batch, int32_t in_h_origin,
    int32_t in_w_origin, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t input_height_stride, size_t input_width_stride,
    unsigned char zero_byte) {
  skl_im2row_d1_full_patch_hwc_zve32x(
      out, in_batch, sizeof(uint8_t), in_h_origin, in_w_origin, input_height,
      input_width, input_channel, filter_height, filter_width,
      input_height_stride, input_width_stride, zero_byte);
}
