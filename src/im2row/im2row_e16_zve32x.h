// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file im2row_e16_zve32x.h
 * @brief Im2Row functions for HWC layout convolution patches into matrix rows
 *
 * This header provides support for im2row preprocessing
 * specifically for HWC (Height, Width, Channels) tensor layout, enabling
 * efficient convolution-to-GEMM transformation for any e16 type.
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

#if !defined(__riscv_zve32x)
#error This file requires the Zve16x extension
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optimized patch extraction with bulk memory operations
 *
 * Extracts a convolution patch from input tensor using optimized bulk copying.
 * Uses three-phase processing (head, middle, tail) to maximize memcpy
 * efficiency while maintaining support for partial patches and arbitrary
 * starting coordinates.
 *
 * @param out Point to the buffer which stores a row of output matrix
 * @param in_batch Point to the start of input tensor for current batch
 * (NHWC layout)
 * @param in_h_origin Top coordinate of patch in input tensor
 * @param in_w_origin Left coordinate of patch in input tensor
 * @param in_c_origin Channel cooridnate of patch in input tensor
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param patch_channel Number of patch channels
 * @param dilation_height_factor Height dilation factor for dilated convolution
 * @param dilation_width_factor Width dilation factor for dilated convolution
 * @param input_height_stride Height's stride of input tensor
 * @param input_width_stride Width's stride of input tensor
 * @param zero_byte Byte value used for out-of-bounds padding
 * @param patch_begin_coord Starting coordinates [h, w, c] within the patch
 * @param patch_elements Number of elements to extract from patch
 *
 * @note Optimized for channel-contiguous memory layout
 * @note Uses bulk memcpy operations when possible for better performance
 * @note Head: partial channel at start, Middle: full channels, Tail: partial
 * channel at end
 *
 */
void skl_im2row_hwc_e16_zve32x(
    uint16_t *out, const uint16_t *in_batch, int32_t in_h_origin,
    int32_t in_w_origin, int32_t in_c_origin, size_t input_height,
    size_t input_width, size_t filter_height, size_t filter_width,
    size_t patch_channel, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_height_stride,
    size_t input_width_stride, unsigned char zero_byte,
    const size_t patch_begin_coord[3], size_t patch_elements);

#ifdef __cplusplus
}
#endif
