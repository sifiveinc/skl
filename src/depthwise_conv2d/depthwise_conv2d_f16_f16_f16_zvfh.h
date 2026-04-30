// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

/**
 * @file depthwise_conv2d_f16_f16_f16_zvfh.h
 * @brief RVV-optimized float16 depthwise convolution 2D kernels.
 *
 * This header provides RVV-optimized implementations of depthwise convolution
 * 2D operations for IEEE 754 binary16 (_Float16) input/filter/output data.
 * Depthwise convolution is a specialized convolution where each input channel
 * is convolved with a dedicated filter, reducing computational cost compared
 * to standard convolution operations.
 *
 * The RVV implementations vectorize along the channel dimension for improved
 * performance and support configurable stride, dilation, and depth multiplier
 * parameters. Both generic and specialized (3x3 filter) variants are provided.
 *
 * The input and output tensors of all functions are formatted in HWC
 * (Height-Width-Channel) data layout, while the filter tensor is formatted in
 * HWIM (Height-Width-Input Channel-Depth Multiplier) layout.
 *
 * @note No padding support (must be handled separately)
 * @note Requires RISC-V Zvfh vector extension
 */

#include <stddef.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV float16 depthwise convolution 2D.
 *
 * @param output - Pointer to output tensor (HWC layout).
 * @param input - Pointer to input tensor (HWC layout).
 * @param filter - Pointer to filter tensor (HWIM layout).
 * @param input_height - Input height dimension.
 * @param input_width - Input width dimension.
 * @param input_channel - Input channel dimension.
 * @param filter_height - Filter height dimension.
 * @param filter_width - Filter width dimension.
 * @param output_height - Output height dimension.
 * @param output_width - Output width dimension.
 * @param output_channel - Output channel dimension.
 * @param depth_multiplier - Number of filters applied to each input channel.
 * @param stride_height - Vertical stride.
 * @param stride_width - Horizontal stride.
 * @param dilation_height_factor - Vertical dilation factor.
 * @param dilation_width_factor - Horizontal dilation factor.
 * @param input_row_stride - Input's row stride in elements.
 * @param input_col_stride - Input's column stride in elements.
 * @param filter_row_stride - Filter's row stride in elements.
 * @param filter_col_stride - Filter's column stride in elements.
 * @param output_row_stride - Output's row stride in elements.
 * @param output_col_stride - Output's column stride in elements.
 *
 * This generic RVV implementation defines the semantics of all optimized
 * float16 depthwise convolution kernels where input and output are in HWC data
 * layout, and vectorizes along the channel dimension.
 */
void skl_depthwise_conv2d_vc_fnxn_sn_dn_mn_in_hwc_f16_f16_f16_zvfh(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);

/**
 * @brief RVV float16 depthwise convolution 2D optimized for 3x3 filters.
 *
 * @param output - Pointer to output tensor (HWC layout).
 * @param input - Pointer to input tensor (HWC layout).
 * @param filter - Pointer to filter tensor (HWIM layout).
 * @param input_height - Input height dimension.
 * @param input_width - Input width dimension.
 * @param input_channel - Input channel dimension.
 * @param output_height - Output height dimension.
 * @param output_width - Output width dimension.
 * @param output_channel - Output channel dimension.
 * @param depth_multiplier - Number of filters applied to each input channel.
 * @param stride_height - Vertical stride.
 * @param stride_width - Horizontal stride.
 * @param dilation_height_factor - Vertical dilation factor.
 * @param dilation_width_factor - Horizontal dilation factor.
 * @param input_row_stride - Input's row stride in elements.
 * @param input_col_stride - Input's column stride in elements.
 * @param filter_row_stride - Filter's row stride in elements.
 * @param filter_col_stride - Filter's column stride in elements.
 * @param output_row_stride - Output's row stride in elements.
 * @param output_col_stride - Output's column stride in elements.
 *
 * This specialized implementation is optimized for 3x3 filters, providing
 * better performance than the generic implementation for this common case. It
 * vectorizes along the channel dimension with the same layout assumptions as
 * those in the generic implementation.
 *
 * Equivalent to calling the generic function with filter_height = 3 and
 * filter_width = 3.
 *
 * @note Filter dimensions must be exactly 3x3.
 */
void skl_depthwise_conv2d_vc_f3x3_sn_dn_mn_in_hwc_f16_f16_f16_zvfh(
    _Float16 *output, const _Float16 *input, const _Float16 *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t depth_multiplier, size_t stride_height, size_t stride_width,
    size_t dilation_height_factor, size_t dilation_width_factor,
    size_t input_row_stride, size_t input_col_stride, size_t filter_row_stride,
    size_t filter_col_stride, size_t output_row_stride,
    size_t output_col_stride);

#if defined(__cplusplus)
}
#endif
