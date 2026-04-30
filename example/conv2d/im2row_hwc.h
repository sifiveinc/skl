// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

/**
 * @file im2row_hwc.h
 * @brief Im2Row functions for HWC layout convolution patches into matrix rows
 *
 * This header provides generic data type support for im2row preprocessing
 * specifically for HWC (Height, Width, Channels) tensor layout, enabling
 * efficient convolution-to-GEMM transformation for any primitive type.
 * All functions use void pointers and byte-based addressing for type
 * flexibility.
 *
 * @note Im2row preprocessing is only required when:
 *       - Any stride > 1, OR
 *       - Any filter dimension > 1, OR
 *       - Any dilation > 1
 *       For 1x1 filters with stride=1 and dilation=1, direct GEMM is more
 * efficient.
 *
 * @note Input tensor layout: HWC (Height, Width, Channels) - channels are
 * contiguous
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Increment multi-dimensional indices with carry propagation
 *
 * @param dims Array of dimension sizes
 * @param indices Current indices to increment (modified in-place)
 * @param ndims Number of dimensions
 * @return 0 if increment successful, non-zero if all dimensions exhausted
 */
__attribute__((always_inline)) inline int next(size_t const *dims,
                                               size_t *indices, size_t ndims) {
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
 * @param in_h Input height coordinate
 * @param in_w Input width coordinate
 * @param in_c Input channel offset
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_width_x_channel Pre-computed input_width * input_channel
 * @param input_channel Number of input channels
 * @param in_batch_tile_bytes Input tensor data as byte array (HWC layout)
 * @param dst Destination buffer
 * @param zero_byte Padding byte value
 * @param copy_bytes Number of bytes to copy
 * @param element_size Size in bytes of the input tensor's primitive data type
 */
__attribute__((always_inline)) inline void extract_spatial_position_hwc(
    int32_t in_h, int32_t in_w, int32_t in_c, size_t input_height,
    size_t input_width, size_t input_width_x_channel, size_t input_channel,
    const char *in_batch_tile_bytes, char *dst, int zero_byte,
    size_t copy_bytes, size_t element_size) {
  if ((in_h >= 0) && (in_h < (int32_t)input_height) && (in_w >= 0) &&
      (in_w < (int32_t)input_width)) {
    const size_t src_byte_offset =
        (in_h * input_width_x_channel + in_w * input_channel + in_c) *
        element_size;
    const char *src = in_batch_tile_bytes + src_byte_offset;
    memcpy(dst, src, copy_bytes);
  } else {
    memset(dst, zero_byte, copy_bytes);
  }
}

/**
 * @brief Generic patch extraction with element-by-element processing
 *
 * Extracts a convolution patch from input tensor and converts it to a matrix
 * row. Processes elements individually with full bounds checking and padding
 * support.
 *
 * @param in_w_origin Left coordinate of patch in input tensor
 * @param in_h_origin Top coordinate of patch in input tensor
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param dilation_width_factor Width dilation factor for dilated convolution
 * @param dilation_height_factor Height dilation factor for dilated convolution
 * @param in_batch_tile Pointer to the start of input tensor for current batch
 * (NHWC layout)
 * @param im2row_tile Output matrix row buffer (void* for generic type support)
 * @param zero_byte Byte value used for out-of-bounds padding
 * @param patch_begin_coord Starting coordinates [h, w, c] within the patch
 * @param patch_elements Number of elements to extract from patch
 * @param element_size Size in bytes of the input tensor's primitive data type
 * (e.g., sizeof(float))
 *
 * @note Uses element-by-element processing with memcpy/memset for type safety
 * @note Supports arbitrary patch starting coordinates and partial extraction
 * @note Requires HWC layout: channels are contiguous in memory for optimal
 * performance
 */
__attribute__((always_inline)) inline void extract_patch_to_row_generic_hwc(
    int32_t in_w_origin, int32_t in_h_origin, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t dilation_width_factor,
    size_t dilation_height_factor, const void *in_batch_tile, void *im2row_tile,
    int zero_byte, const size_t patch_begin_coord[3], size_t patch_elements,
    size_t element_size) {
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
    const int32_t in_c = current_indices[2];

    extract_spatial_position_hwc(
        in_h, in_w, in_c, input_height, input_width, input_width_x_channel,
        input_channel, in_batch_tile_bytes, im2row_tile_bytes + dst_byte_offset,
        zero_byte, element_size, element_size);
    dst_byte_offset += element_size;
    iteration++;
  } while (next(&patch_dims[0], &current_indices[0], 3) == 0 &&
           iteration < patch_elements);
}

/**
 * @brief Optimized patch extraction with bulk memory operations
 *
 * Extracts a convolution patch from input tensor using optimized bulk copying.
 * Uses three-phase processing (head, middle, tail) to maximize memcpy
 * efficiency while maintaining support for partial patches and arbitrary
 * starting coordinates.
 *
 * @param in_w_origin Left coordinate of patch in input tensor
 * @param in_h_origin Top coordinate of patch in input tensor
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param dilation_width_factor Width dilation factor for dilated convolution
 * @param dilation_height_factor Height dilation factor for dilated convolution
 * @param in_batch_tile Pointer to the start of input tensor for current batch
 * (NHWC layout)
 * @param im2row_tile Output matrix row buffer (void* for generic type support)
 * @param zero_byte Byte value used for out-of-bounds padding
 * @param patch_begin_coord Starting coordinates [h, w, c] within the patch
 * @param patch_elements Number of elements to extract from patch
 * @param element_size Size in bytes of the input tensor's primitive data type
 * (e.g., sizeof(float))
 *
 * @note Optimized for channel-contiguous memory layout
 * @note Uses bulk memcpy operations when possible for better performance
 * @note Head: partial channel at start, Middle: full channels, Tail: partial
 * channel at end
 *
 * @par Usage Examples:
 * @code
 * // Float32 convolution (HWC layout)
 * extract_patch_to_row_hwc(in_w_origin, in_h_origin, input_height, input_width,
 *                          input_channel, filter_height, filter_width,
 *                          dilation_width, dilation_height, in_batch_tile,
 *                          im2row_tile, 0, patch_begin_coord,
 *                          k_len, sizeof(float));
 *
 * // Int8 quantized convolution (HWC layout)
 * extract_patch_to_row_hwc(in_w_origin, in_h_origin, input_height, input_width,
 *                          input_channel, filter_height, filter_width,
 *                          dilation_width, dilation_height, in_batch_tile,
 *                          im2row_tile, 0, patch_begin_coord,
 *                          k_len, sizeof(int8_t));
 * @endcode
 */
__attribute__((always_inline)) inline void extract_patch_to_row_hwc(
    int32_t in_w_origin, int32_t in_h_origin, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t dilation_width_factor,
    size_t dilation_height_factor, const void *in_batch_tile, void *im2row_tile,
    int zero_byte, const size_t patch_begin_coord[3], size_t patch_elements,
    size_t element_size) {
  const size_t patch_dims[3] = {filter_height, filter_width, input_channel};

  const size_t head =
      patch_begin_coord[2]
          ? ((patch_elements < (patch_dims[2] - patch_begin_coord[2]))
                 ? patch_elements
                 : (patch_dims[2] - patch_begin_coord[2]))
          : 0;
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
    const int32_t in_c = patch_begin_coord[2];

    char *dst = im2row_tile_bytes + dst_byte_offset;
    const size_t head_bytes = head * element_size;
    extract_spatial_position_hwc(in_h, in_w, in_c, input_height, input_width,
                                 input_width_x_channel, input_channel,
                                 in_batch_tile_bytes, dst, zero_byte,
                                 head_bytes, element_size);
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
    extract_spatial_position_hwc(in_h, in_w, 0, input_height, input_width,
                                 input_width_x_channel, input_channel,
                                 in_batch_tile_bytes, dst, zero_byte,
                                 chunk_bytes, element_size);
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
    extract_spatial_position_hwc(in_h, in_w, 0, input_height, input_width,
                                 input_width_x_channel, input_channel,
                                 in_batch_tile_bytes, dst, zero_byte,
                                 tail_bytes, element_size);
  }
}

#ifdef __cplusplus
}
#endif
