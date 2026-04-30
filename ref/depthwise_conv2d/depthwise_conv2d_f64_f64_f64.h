// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file depthwise_conv2d_f64_f64_f64.h
 * @brief Reference float64 depthwise convolution 2D kernels.
 *
 * This header provides reference implementations of depthwise convolution 2D
 * operations for float64 input/filter/output data. Depthwise convolution is a
 * specialized convolution where each input channel is convolved with a
 * dedicated filter, reducing computational cost compared to standard
 * convolution operations.
 *
 * The implementation provides the reference semantics for all optimized
 * depthwise convolution kernels and supports configurable stride, dilation, and
 * depth multiplier parameters. This implementation is primarily intended for
 * API documentation and correctness verification rather than performance.
 *
 * The input and output tensors of the function are formatted in HWC
 * (Height-Width-Channel) data layout, while the filter tensor is formatted in
 * HWIM (Height-Width-Input Channel-Depth Multiplier) layout.
 */

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference implementation of depthwise convolution with float64.
 *
 * This function is primarily used for computing error bounds in verification,
 * not for performance testing.
 *
 * @param output - Output tensor pointer (HWC layout).
 * @param input - Input tensor pointer (HWC layout).
 * @param filter - Filter tensor pointer (HWIM layout).
 * @param input_height - Input height.
 * @param input_width - Input width.
 * @param input_channel - Number of input channels.
 * @param filter_height - Filter height.
 * @param filter_width - Filter width.
 * @param output_height - Output height.
 * @param output_width - Output width.
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
 * @note This function is for API documentation and test purposes only, and
 * should not be used to obtain good performance.
 */
void skl_depthwise_conv2d_hwc_f64_f64_f64_ref(
    double *output, const double *input, const double *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride);

#if defined(__cplusplus)
}
#endif
