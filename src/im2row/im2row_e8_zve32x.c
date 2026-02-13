// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>

#include "./im2row_hwc_zve32x.h"

#include "skl-common.h"

SKL_FUNC void skl_im2row_hwc_e8_zve32x(
    uint8_t *out, const uint8_t *in_batch, int32_t in_w_origin,
    int32_t in_h_origin, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t dilation_width_factor, size_t dilation_height_factor,
    unsigned char zero_byte, const size_t patch_begin_coord[3],
    size_t patch_elements) {
  skl_im2row_hwc_zve32x(out, in_batch, sizeof(uint8_t), in_w_origin,
                        in_h_origin, input_height, input_width, input_channel,
                        filter_height, filter_width, dilation_width_factor,
                        dilation_height_factor, zero_byte, patch_begin_coord,
                        patch_elements);
}
