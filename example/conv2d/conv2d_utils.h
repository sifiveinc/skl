// Copyright (c) 2025 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Determines if Im2Row preprocessing is required for convolution-to-GEMM
 * transformation
 *
 * This function analyzes convolution parameters to determine whether Im2Row
 * preprocessing is required or if direct GEMM transformation can be used.
 * Direct GEMM is only possible for 1x1 convolutions with stride=1, dilation=1,
 * and no padding.
 *
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param stride_height Stride factor along height dimension
 * @param stride_width Stride factor along width dimension
 * @param dilation_height Dilation factor along height dimension
 * @param dilation_width Dilation factor along width dimension
 * @param padding_height Padding along height dimension
 * @param padding_width Padding along width dimension
 * @return true if Im2Row preprocessing is required, false if direct GEMM can be
 * used
 *
 * @note Returns false only when all conditions are met:
 *       - filter_height == 1 AND filter_width == 1
 *       - stride_height == 1 AND stride_width == 1
 *       - dilation_height == 1 AND dilation_width == 1
 *       - padding_height == 0 AND padding_width == 0
 *
 * @note Used by main.c to automatically select between direct GEMM and
 * Im2Row+GEMM transformations
 */
__attribute__((always_inline)) inline bool
is_conv2d_to_gemm_im2row_required(size_t filter_height, size_t filter_width,
                                  size_t stride_height, size_t stride_width,
                                  size_t dilation_height, size_t dilation_width,
                                  size_t padding_height, size_t padding_width) {
  return (filter_height != 1) || (filter_width != 1) || (stride_height != 1) ||
         (stride_width != 1) || (dilation_height != 1) ||
         (dilation_width != 1) || (padding_height != 0) || (padding_width != 0);
}

#ifdef __cplusplus
}
#endif
