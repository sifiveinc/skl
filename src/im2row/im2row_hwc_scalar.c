// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "./im2row_utils.h"

#include "skl-common.h"

SKL_FUNC void skl_im2row_generic_hwc(
    void *out, const void *in_batch, size_t element_size, int32_t in_w_origin,
    int32_t in_h_origin, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t dilation_width_factor, size_t dilation_height_factor,
    unsigned char zero_byte, const size_t patch_begin_coord[3],
    size_t patch_elements) {
  const size_t patch_dims[3] = {filter_height, filter_width, input_channel};

  size_t current_indices[3];
  current_indices[0] = patch_begin_coord[0];
  current_indices[1] = patch_begin_coord[1];
  current_indices[2] = patch_begin_coord[2];

  const char *in_batch_bytes = (const char *)in_batch;
  char *out_bytes = (char *)out;

  // Pre-compute common factors
  const size_t input_width_stride = input_width * input_channel;

  size_t dst_byte_offset = 0;

  size_t iteration = 0;
  do {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * current_indices[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * current_indices[1]);
    const int32_t in_c = (int32_t)current_indices[2];

    skl_im2row_hwc_slice_patch(out_bytes + dst_byte_offset, in_batch_bytes,
                               element_size, in_h, in_w, in_c, input_height,
                               input_width, input_width_stride, input_channel,
                               zero_byte, element_size, memcpy, memset);
    dst_byte_offset += element_size;
    iteration++;
  } while (skl_im2row_patch_element_next(&patch_dims[0], &current_indices[0],
                                         3) == 0 &&
           iteration < patch_elements);
}
