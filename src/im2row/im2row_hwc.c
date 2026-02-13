// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "skl-common.h"

/**
 * @brief Increment multi-dimensional indices with carry propagation
 *
 * @param dims Array of dimension sizes
 * @param indices Current indices to increment (modified in-place)
 * @param ndims Number of dimensions
 * @return 0 if increment successful, non-zero if all dimensions exhausted
 */
SKL_FUNC_PRIVATE __attribute__((always_inline)) inline int
next(size_t const *dims, size_t *indices, size_t ndims) {
  int is_end = 0;
  size_t carry = 1;
  for (int32_t i = (int32_t)ndims - 1; i >= 0; --i) {
    size_t next_idx = indices[i] + carry;
    if (next_idx == dims[i]) {
      indices[i] = 0;
    } else {
      indices[i] = next_idx;
      carry = 0;
      break;
    }
  }

  if (carry == 1) {
    is_end = 0;
  }

  return is_end;
}

/**
 * @brief Helper function to extract data from a single spatial position (HWC
 * layout)
 *
 * @param dst Destination buffer
 * @param in_batch_tile_bytes Input tensor data as byte array (HWC layout)
 * @param element_size Size in bytes of the input tensor's primitive data type
 * @param in_h Input height coordinate
 * @param in_w Input width coordinate
 * @param in_c Input channel offset
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_width_x_channel Pre-computed input_width * input_channel
 * @param input_channel Number of input channels
 * @param zero_byte Padding byte value
 * @param copy_bytes Number of bytes to copy
 */
SKL_FUNC_PRIVATE __attribute__((always_inline)) inline void
extract_spatial_position_hwc(char *dst, const char *in_batch_tile_bytes,
                             size_t element_size, int32_t in_h, int32_t in_w,
                             int32_t in_c, size_t input_height,
                             size_t input_width, size_t input_width_x_channel,
                             size_t input_channel, unsigned char zero_byte,
                             size_t copy_bytes) {
  if ((in_h >= 0) && (in_h < (int32_t)input_height) && (in_w >= 0) &&
      (in_w < (int32_t)input_width)) {
    const size_t src_byte_offset =
        (in_h * input_width_x_channel + in_w * input_channel + in_c) *
        element_size;
    const char *src = in_batch_tile_bytes + src_byte_offset;
    memcpy(dst, src, copy_bytes);
  } else {
    memset(dst, (int)zero_byte, copy_bytes);
  }
}

SKL_FUNC void skl_im2row_generic_hwc(
    void *im2row_tile, const void *in_batch_tile, size_t element_size,
    int32_t in_w_origin, int32_t in_h_origin, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t dilation_width_factor,
    size_t dilation_height_factor, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements) {
  const size_t patch_dims[3] = {filter_height, filter_width, input_channel};

  size_t current_indices[3];
  memcpy(current_indices, patch_begin_coord, sizeof(current_indices));

  const char *in_batch_tile_bytes = (const char *)in_batch_tile;
  char *im2row_tile_bytes = (char *)im2row_tile;

  // Pre-compute common factors
  const size_t input_width_x_channel = input_width * input_channel;

  size_t dst_byte_offset = 0;

  size_t iteration = 0;
  do {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * current_indices[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * current_indices[1]);
    const int32_t in_c = (int32_t)current_indices[2];

    extract_spatial_position_hwc(
        im2row_tile_bytes + dst_byte_offset, in_batch_tile_bytes, element_size,
        in_h, in_w, in_c, input_height, input_width, input_width_x_channel,
        input_channel, zero_byte, element_size);
    dst_byte_offset += element_size;
    iteration++;
  } while (next(&patch_dims[0], &current_indices[0], 3) == 0 &&
           iteration < patch_elements);
}

SKL_FUNC void
skl_im2row_hwc(void *im2row_tile, const void *in_batch_tile,
               size_t element_size, int32_t in_w_origin, int32_t in_h_origin,
               size_t input_height, size_t input_width, size_t input_channel,
               size_t filter_height, size_t filter_width,
               size_t dilation_width_factor, size_t dilation_height_factor,
               unsigned char zero_byte, const size_t patch_begin_coord[3],
               size_t patch_elements) {
  const size_t patch_dims[3] = {filter_height, filter_width, input_channel};

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
  memcpy(current_indices, patch_begin_coord, sizeof(size_t) * 2);

  const char *in_batch_tile_bytes = (const char *)in_batch_tile;
  char *im2row_tile_bytes = (char *)im2row_tile;

  // Pre-compute common factors
  const size_t input_width_x_channel = input_width * input_channel;

  int is_end = 0;

  size_t dst_byte_offset = 0;

  if (head) {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * patch_begin_coord[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * patch_begin_coord[1]);
    const int32_t in_c = (int32_t)patch_begin_coord[2];

    char *dst = im2row_tile_bytes + dst_byte_offset;
    const size_t head_bytes = head * element_size;
    extract_spatial_position_hwc(dst, in_batch_tile_bytes, element_size, in_h,
                                 in_w, in_c, input_height, input_width,
                                 input_width_x_channel, input_channel,
                                 zero_byte, head_bytes);
    dst_byte_offset += head_bytes;
    is_end = next(&patch_dims[0], &current_indices[0], 2);
  }

  size_t iteration = 0;
  const size_t chunk_bytes = patch_dims[2] * element_size;
  while (is_end == 0 && iteration < multiples) {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * current_indices[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * current_indices[1]);

    char *dst = im2row_tile_bytes + dst_byte_offset;
    extract_spatial_position_hwc(dst, in_batch_tile_bytes, element_size, in_h,
                                 in_w, 0, input_height, input_width,
                                 input_width_x_channel, input_channel,
                                 zero_byte, chunk_bytes);
    dst_byte_offset += chunk_bytes;
    is_end = next(&patch_dims[0], &current_indices[0], 2);
    iteration++;
  }

  if (tail) {
    const int32_t in_h =
        in_h_origin + (int32_t)(dilation_height_factor * current_indices[0]);
    const int32_t in_w =
        in_w_origin + (int32_t)(dilation_width_factor * current_indices[1]);

    char *dst = im2row_tile_bytes + dst_byte_offset;
    const size_t tail_bytes = tail * element_size;
    extract_spatial_position_hwc(dst, in_batch_tile_bytes, element_size, in_h,
                                 in_w, 0, input_height, input_width,
                                 input_width_x_channel, input_channel,
                                 zero_byte, tail_bytes);
  }
}
