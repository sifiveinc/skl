// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic patch extraction with element-by-element processing
 *
 * Extracts a convolution patch from input tensor and converts it to a matrix
 * row. Processes elements individually with full bounds checking and padding
 * support.
 *
 * @param im2row_tile Output matrix row buffer (void* for generic type support)
 * @param in_batch_tile Pointer to the start of input tensor for current batch
 * (NHWC layout)
 * @param element_size Size in bytes of the input tensor's primitive data type
 * (e.g., sizeof(float))
 * @param in_w_origin Left coordinate of patch in input tensor
 * @param in_h_origin Top coordinate of patch in input tensor
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param dilation_width_factor Width dilation factor for dilated convolution
 * @param dilation_height_factor Height dilation factor for dilated convolution
 * @param zero_byte Byte value used for out-of-bounds padding
 * @param patch_begin_coord Starting coordinates [h, w, c] within the patch
 * @param patch_elements Number of elements to extract from patch
 *
 * @note Uses element-by-element processing with memcpy/memset for type safety
 * @note Supports arbitrary patch starting coordinates and partial extraction
 * @note Requires HWC layout: channels are contiguous in memory for optimal
 * performance
 */
void skl_im2row_generic_hwc(
    void *im2row_tile, const void *in_batch_tile, size_t element_size,
    int32_t in_w_origin, int32_t in_h_origin, size_t input_height,
    size_t input_width, size_t input_channel, size_t filter_height,
    size_t filter_width, size_t dilation_width_factor,
    size_t dilation_height_factor, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements);

/**
 * @brief Optimized patch extraction with bulk memory operations
 *
 * Extracts a convolution patch from input tensor using optimized bulk copying.
 * Uses three-phase processing (head, middle, tail) to maximize memcpy
 * efficiency while maintaining support for partial patches and arbitrary
 * starting coordinates.
 *
 * @param im2row_tile Output matrix row buffer (void* for generic type support)
 * @param in_batch_tile Pointer to the start of input tensor for current batch
 * (NHWC layout)
 * @param element_size Size in bytes of the input tensor's primitive data type
 * (e.g., sizeof(float))
 * @param in_w_origin Left coordinate of patch in input tensor
 * @param in_h_origin Top coordinate of patch in input tensor
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param dilation_width_factor Width dilation factor for dilated convolution
 * @param dilation_height_factor Height dilation factor for dilated convolution
 * @param zero_byte Byte value used for out-of-bounds padding
 * @param patch_begin_coord Starting coordinates [h, w, c] within the patch
 * @param patch_elements Number of elements to extract from patch
 *
 * @note Optimized for channel-contiguous memory layout
 * @note Uses bulk memcpy operations when possible for better performance
 * @note Head: partial channel at start, Middle: full channels, Tail: partial
 * channel at end
 *
 * @par Usage Examples:
 * @code
 * // Float32 convolution (HWC layout)
 * skl_im2row_hwc(im2row_tile, in_batch_tile, sizeof(float),
 *                in_w_origin, in_h_origin, input_height, input_width,
 *                input_channel, filter_height, filter_width,
 *                dilation_width, dilation_height,
 *                0, patch_begin_coord, k_len);
 *
 * // Int8 quantized convolution (HWC layout)
 * skl_im2row_hwc(im2row_tile, in_batch_tile, sizeof(int8_t),
 *                in_w_origin, in_h_origin, input_height, input_width,
 *                input_channel, filter_height, filter_width,
 *                dilation_width, dilation_height,
 *                0, patch_begin_coord, k_len);
 * @endcode
 */
void skl_im2row_hwc(void *im2row_tile, const void *in_batch_tile,
                    size_t element_size, int32_t in_w_origin,
                    int32_t in_h_origin, size_t input_height,
                    size_t input_width, size_t input_channel,
                    size_t filter_height, size_t filter_width,
                    size_t dilation_width_factor, size_t dilation_height_factor,
                    unsigned char zero_byte, const size_t patch_begin_coord[3],
                    size_t patch_elements);

#ifdef __cplusplus
}
#endif
