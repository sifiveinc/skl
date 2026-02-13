// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
__attribute__((always_inline)) inline int
skl_im2row_patch_element_next(size_t const *dims, size_t *indices,
                              size_t ndims) {
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
 * @param out Destination buffer
 * @param in_batch Input tensor data as byte array (HWC layout)
 * @param element_size Size in bytes of the input tensor's primitive data type
 * @param in_h Input height coordinate of the slice
 * @param in_w Input width coordinate of the slice
 * @param in_c Input channel offset of the slice
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_width_stride Width's stride of input tensor
 * @param input_channel Number of input channels
 * @param zero_byte Padding byte value
 * @param copy_bytes Number of bytes to copy
 */
__attribute__((always_inline)) inline void
skl_im2row_hwc_slice_patch(char *out, const char *in_batch, size_t element_size,
                           int32_t in_h, int32_t in_w, int32_t in_c,
                           size_t input_height, size_t input_width,
                           size_t input_width_stride, size_t input_channel,
                           unsigned char zero_byte, size_t copy_bytes) {
  if ((in_h >= 0) && (in_h < (int32_t)input_height) && (in_w >= 0) &&
      (in_w < (int32_t)input_width)) {
    const size_t src_byte_offset =
        (in_h * input_width_stride + in_w * input_channel + in_c) *
        element_size;
    const char *src = in_batch + src_byte_offset;
    memcpy(out, src, copy_bytes);
  } else {
    memset(out, (int)zero_byte, copy_bytes);
  }
}

#ifdef __cplusplus
}
#endif
