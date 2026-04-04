// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file depthwise_conv2d_i8_i8_i32.h
 * @brief Scalar int8 depthwise convolution 2D kernels with int32 accumulator.
 *
 * This header provides scalar reference implementations of depthwise
 * convolution 2D operations for int8 input/filter data with int32 output
 * accumulation. Depthwise convolution is a specialized convolution where each
 * input channel is convolved with a dedicated filter, reducing computational
 * cost compared to standard convolution operations.
 *
 * The scalar implementation provides the reference semantics for all optimized
 * depthwise convolution kernels and supports configurable stride, dilation, and
 * depth multiplier parameters. This implementation is primarily intended for
 * API documentation and correctness verification rather than performance.
 *
 * The input and output tensors of the scalar function are formated in HWC
 * (Height-Width-Channel) data layout, while the filter tensor is formated in
 * HWIM (Height-Width-Input Channel-Depth Multiplier) layout.
 *
 * @note Filter's zero-point must be 0 (symmetric quantization)
 * @note No padding support (must be handled separately)
 * @note No requantization (post-processing must be handled separately)
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Scalar int8 depthwise convolution 2D with int32 accumulator.
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
 * @param input_zero_point - Input's zero-point for quantization.
 *
 * This generic scalar implementation defines the semantics of all optimized
 * int8 depthwise convolution kernels with int32 accumulators that use HWC
 * data layout.
 *
 * @note This function is for API documentation and test purposes only, and
 * should not be used to obtain good performance.
 */
void skl_depthwise_conv2d_hwc_i8_i8_i32_scalar(
    int32_t *output, const int8_t *input, const int8_t *filter,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t depth_multiplier,
    size_t stride_height, size_t stride_width, size_t dilation_height_factor,
    size_t dilation_width_factor, size_t input_row_stride,
    size_t input_col_stride, size_t filter_row_stride, size_t filter_col_stride,
    size_t output_row_stride, size_t output_col_stride,
    int32_t input_zero_point);

#if defined(__cplusplus)
}
#endif
