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

  const size_t patch_dim2 = patch_dims[2];
  const size_t patch_begin_coord2 = patch_begin_coord[2];

  // NOLINTBEGIN(readability-avoid-nested-conditional-operator)
  const size_t head =
      patch_begin_coord2 ? ((patch_elements < (patch_dim2 - patch_begin_coord2))
                                ? patch_elements
                                : (patch_dim2 - patch_begin_coord2))
                         : 0;
  // NOLINTEND(readability-avoid-nested-conditional-operator)

  const size_t tail = (patch_elements - head) % patch_dim2;
  const size_t multiples = (patch_elements - head - tail) / patch_dim2;

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
    const int32_t in_c = (int32_t)patch_begin_coord2;

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
  const size_t chunk_bytes = patch_dim2 * element_size;
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

/**
 * @brief Internal implementation for optimized full patch extraction
 * (dilation=1)
 *
 * Low-level implementation function that performs the actual memory operations
 * for non-dilated patch extraction. Handles padding calculations and bulk
 * memory copying with optimal performance. Used internally by
 * skl_im2row_d1_full_patch_hwc_zve32x(...).
 *
 * @param out Output matrix row buffer
 * @param in Pointer to valid input data region
 * @param input_tile_height Height of valid input data within the patch
 * @param input_tile_width Width of valid input data within the patch
 * @param input_channel Number of input channels (patch channel)
 * @param top_padding Number of padding rows at top of patch
 * @param left_padding Number of padding columns at left of patch
 * @param right_padding Number of padding columns at right of patch
 * @param bottom_padding Number of padding rows at bottom of patch
 * @param input_height_stride Stride between input rows in elements
 * @param filter_height_stride Stride between output rows in elements
 * @param zero_byte Byte value for padding
 * @param element_size Size in bytes of data type
 *
 * @note Optimized for bulk memory operations and minimal branching
 * @note Handles all padding scenarios efficiently
 */
SKL_FUNC_PRIVATE void skl_im2row_d1_full_patch_hwc_zve32x_internal(
    void *out, const void *in, size_t input_tile_height,
    size_t input_tile_width, size_t input_channel, size_t top_padding,
    size_t left_padding, size_t right_padding, size_t bottom_padding,
    size_t input_height_stride, size_t filter_height_stride, int zero_byte,
    size_t element_size, skl_memcpy memcpy_f, skl_memset memset_f) {
  const size_t input_channel_bytes = input_channel * element_size;
  const size_t inner_row_bytes = input_tile_width * input_channel_bytes;

  const size_t input_height_stride_bytes = input_height_stride * element_size;
  const size_t filter_height_stride_bytes = filter_height_stride * element_size;

  size_t out_col_byte_offset =
      (top_padding * filter_height_stride + left_padding * input_channel) *
      element_size;
  size_t in_byte_offset = 0;

  char *src = (char *)in;
  char *dst = (char *)out;

  // Write out zeroes to the elements representing the top rows of the input
  // patch that are off the edge of the input image.
  if (top_padding > 0) {
    const size_t top_row_bytes = (top_padding * filter_height_stride_bytes);
    memset_f(dst, zero_byte, top_row_bytes);
  }

  // If the patch is on the interior of the input image horizontally, just copy
  // over the rows sequentially, otherwise add zero padding at the start or end.
  if ((left_padding == 0) && (right_padding == 0)) {
    for (size_t ih = 0; ih < input_tile_height; ++ih) {
      memcpy_f(dst + out_col_byte_offset, src + in_byte_offset,
               inner_row_bytes);
      out_col_byte_offset += filter_height_stride_bytes;
      in_byte_offset += input_height_stride_bytes;
    }
  } else {
    for (size_t ih = 0; ih < input_tile_height; ++ih) {
      if (left_padding > 0) {
        const size_t left_start =
            (out_col_byte_offset - (left_padding * input_channel_bytes));
        memset_f(dst + left_start, zero_byte,
                 left_padding * input_channel_bytes);
      }
      memcpy_f(dst + out_col_byte_offset, src + in_byte_offset,
               inner_row_bytes);
      if (right_padding > 0) {
        const size_t right_start = out_col_byte_offset + inner_row_bytes;
        memset_f(dst + right_start, zero_byte,
                 right_padding * input_channel_bytes);
      }
      out_col_byte_offset += filter_height_stride_bytes;
      in_byte_offset += input_height_stride_bytes;
    }
  }

  // If the bottom of the patch falls off the input image, pad the values
  // representing those input rows with zeroes.
  if (bottom_padding > 0) {
    const size_t bottom_row_bytes = bottom_padding * filter_height_stride_bytes;

    const size_t bottom_start =
        (top_padding + input_tile_height) * filter_height_stride_bytes;
    memset_f(dst + bottom_start, zero_byte, bottom_row_bytes);
  }
}

SKL_FUNC void skl_im2row_d1_full_patch_hwc_zve32x(
    void *out, const void *in_batch, size_t element_size, int32_t in_h_origin,
    int32_t in_w_origin, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t input_height_stride, size_t input_width_stride,
    unsigned char zero_byte) {
  const int32_t in_w_ungated_end = in_w_origin + (int32_t)filter_width;
  const size_t in_w_end = (in_w_ungated_end > (int32_t)input_width)
                              ? input_width
                              : (size_t)in_w_ungated_end;

  const int32_t in_h_ungated_end = in_h_origin + (int32_t)filter_height;
  const size_t in_h_end = (in_h_ungated_end > (int32_t)input_height)
                              ? input_height
                              : (size_t)in_h_ungated_end;

  // If the patch is off the edge of the input image, skip writing those
  // rows and columns from the patch into the output array.
  const size_t in_h_start = (in_h_origin > 0) ? in_h_origin : 0;
  const size_t in_w_start = (in_w_origin > 0) ? in_w_origin : 0;

  const size_t in_offset =
      in_h_start * input_height_stride + in_w_start * input_width_stride;
  const char *in = (char *)in_batch + in_offset * element_size;

  const size_t filter_height_stride = filter_width * input_channel;

  // Express all of the calculations as padding around the input patch.
  const size_t ih_offset = (in_h_origin > 0) ? 0 : -in_h_origin;
  const size_t iw_offset = (in_w_origin > 0) ? 0 : -in_w_origin;

  const size_t top_padding = ih_offset;
  const size_t bottom_padding = (size_t)in_h_ungated_end - in_h_end;
  const size_t left_padding = iw_offset;
  const size_t right_padding = (size_t)in_w_ungated_end - in_w_end;

  const size_t input_tile_height = in_h_end - in_h_start;
  const size_t input_tile_width =
      ((filter_width - iw_offset) > (input_width - in_w_start))
          ? (input_width - in_w_start)
          : (filter_width - iw_offset);

  skl_im2row_d1_full_patch_hwc_zve32x_internal(
      out, in, input_tile_height, input_tile_width, input_channel, top_padding,
      left_padding, right_padding, bottom_padding, input_height_stride,
      filter_height_stride, zero_byte, element_size, memcpy, memset);
}
