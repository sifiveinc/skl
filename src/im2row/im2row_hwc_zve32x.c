// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h> // TODO(pattyl): Remove it and add rvv memcpy and memset

#include "./im2row_utils.h"

#include "skl-common.h"

SKL_FUNC void skl_im2row_hwc_zve32x(
    void *out, const void *in_batch, size_t element_size, int32_t in_h_origin,
    int32_t in_w_origin, int32_t in_c_origin, size_t input_height,
    size_t input_width, size_t filter_height, size_t filter_width,
    size_t patch_channel, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_height_stride,
    size_t input_width_stride, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements) {
  const size_t patch_dims[3] = {filter_height, filter_width, patch_channel};

  // NOLINTBEGIN(readability-avoid-nested-conditional-operator)
  const size_t head =
      patch_begin_coord[2]
          ? ((patch_elements < (patch_dims[2] - patch_begin_coord[2]))
                 ? patch_elements
                 : (patch_dims[2] - patch_begin_coord[2]))
          : 0;
  // NOLINTEND(readability-avoid-nested-conditional-operator)

  const size_t tail = (patch_elements - head) % patch_dims[2];
  const size_t multiples = (patch_elements - head - tail) / patch_dims[2];

  size_t current_indices[2];
  current_indices[0] = patch_begin_coord[0];
  current_indices[1] = patch_begin_coord[1];

  const char *in_bytes =
      (const char *)in_batch + in_c_origin * input_width_stride * element_size;
  char *out_bytes = (char *)out;

  int is_end = 0;

  size_t dst_byte_offset = 0;

  if (head) {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * patch_begin_coord[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * patch_begin_coord[1]);
    const int32_t in_c = (int32_t)patch_begin_coord[2];

    char *dst = out_bytes + dst_byte_offset;
    const size_t head_bytes = head * element_size;
    skl_im2row_hwc_slice_patch(dst, in_bytes, element_size, in_h, in_w, in_c,
                               input_height, input_width, input_height_stride,
                               input_width_stride, zero_byte, head_bytes,
                               memcpy, memset);
    dst_byte_offset += head_bytes;
    is_end =
        skl_im2row_patch_element_next(&patch_dims[0], &current_indices[0], 2);
  }

  size_t iteration = 0;
  const size_t chunk_bytes = patch_dims[2] * element_size;
  while (is_end == 0 && iteration < multiples) {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * current_indices[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * current_indices[1]);

    char *dst = out_bytes + dst_byte_offset;
    skl_im2row_hwc_slice_patch(dst, in_bytes, element_size, in_h, in_w, 0,
                               input_height, input_width, input_height_stride,
                               input_width_stride, zero_byte, chunk_bytes,
                               memcpy, memset);
    dst_byte_offset += chunk_bytes;
    is_end =
        skl_im2row_patch_element_next(&patch_dims[0], &current_indices[0], 2);
    iteration++;
  }

  if (tail) {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * current_indices[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * current_indices[1]);

    char *dst = out_bytes + dst_byte_offset;
    const size_t tail_bytes = tail * element_size;
    skl_im2row_hwc_slice_patch(dst, in_bytes, element_size, in_h, in_w, 0,
                               input_height, input_width, input_height_stride,
                               input_width_stride, zero_byte, tail_bytes,
                               memcpy, memset);
  }
}
