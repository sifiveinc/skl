// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file im2row_hwc_scalar.h
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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert input tensor to im2row matrix for convolution-to-GEMM
 * transformation
 *
 * Processes all output positions for a single batch, generating the complete
 * im2row matrix in one call. Uses element-by-element processing with full
 * bounds checking and padding support. Supports any primitive data type through
 * void pointers and byte-based addressing.
 *
 * @param output Point to the buffer which stores the output im2row matrix
 * (size: output_height × output_width × filter_height × filter_width ×
 * input_channel elements)
 * @param input Point to the start of input tensor for current batch (HWC
 * layout)
 * @param element_size Size in bytes of the input tensor's primitive data type
 * (e.g., sizeof(float))
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param input_height_stride Height's stride of input tensor (for standard HWC:
 * input_width × input_channel)
 * @param input_width_stride Width's stride of input tensor (for standard HWC:
 * input_channel)
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param output_height Height of output tensor (number of output rows)
 * @param output_width Width of output tensor (number of output columns)
 * @param padding_width Padding applied to left and right of input
 * @param padding_height Padding applied to top and bottom of input
 * @param stride_width Horizontal stride for convolution
 * @param stride_height Vertical stride for convolution
 * @param dilation_width Width dilation factor for dilated convolution
 * @param dilation_height Height dilation factor for dilated convolution
 * @param zero_byte Byte value used for out-of-bounds padding
 *
 * @note Uses element-by-element processing with memcpy/memset for type safety
 * @note Processes all output positions: output_height × output_width patches
 * @note Requires HWC layout: channels are contiguous in memory for optimal
 * performance
 */
void skl_im2row_generic_hwc(
    void *output, const void *input, size_t element_size, size_t input_height,
    size_t input_width, size_t input_channel, size_t input_height_stride,
    size_t input_width_stride, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height, unsigned char zero_byte);

#ifdef __cplusplus
}
#endif
