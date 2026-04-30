// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#pragma once

/**
 * @file conv2d_nhwc_f32.h
 * @brief 2D Convolution implementations for NHWC layout with float32 data type
 *
 * This header provides various implementations of 2D convolution optimized for
 * NHWC (Batch, Height, Width, Channels) tensor layout using float32 precision.
 * Includes adaptive implementations that automatically select optimal
 * algorithms based on convolution parameters.
 *
 * @note All functions assume:
 *       - Input/Output: NHWC layout (channels are contiguous)
 *       - Filter: HWIO layout (Height, Width, Input Channels, Output Channels)
 *       - Data type: float32 (32-bit floating point)
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scalar reference implementation of 2D convolution
 *
 * Naive implementation for correctness verification and testing.
 * Uses nested loops without any optimization.
 *
 * @param output Output tensor buffer [batches, output_height, output_width,
 * output_channel]
 * @param input Input tensor buffer [batches, input_height, input_width,
 * input_channel]
 * @param filter Filter tensor buffer [filter_height, filter_width,
 * input_channel, output_channel]
 * @param batches Number of batches
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param output_height Height of output tensor
 * @param output_width Width of output tensor
 * @param output_channel Number of output channels
 * @param padding_width Padding along width dimension
 * @param padding_height Padding along height dimension
 * @param stride_width Stride along width dimension
 * @param stride_height Stride along height dimension
 * @param dilation_width Dilation factor along width dimension
 * @param dilation_height Dilation factor along height dimension
 */
void conv2d_io_nhwc_filter_hwio_f32_f32_f32_ref(
    float *output, const float *input, const float *filter, size_t batches,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height);

/**
 * @brief Im2row-based convolution using RISC-V Vector GEMM kernel
 *
 * Converts convolution to matrix multiplication using im2row preprocessing.
 * Uses optimized RISC-V Vector GEMM kernel for computation.
 *
 * @param output Output tensor buffer [batches, output_height, output_width,
 * output_channel]
 * @param im2row Im2Row tensor buffer [batches * output_height * output_width,
 * filter_height * filter_width * input_channel]
 * @param input Input tensor buffer [batches, input_height, input_width,
 * input_channel]
 * @param filter Filter tensor buffer [filter_height, filter_width,
 * input_channel, output_channel]
 * @param batches Number of batches
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter
 * @param filter_width Width of convolution filter
 * @param output_height Height of output tensor
 * @param output_width Width of output tensor
 * @param output_channel Number of output channels
 * @param padding_width Padding along width dimension
 * @param padding_height Padding along height dimension
 * @param stride_width Stride along width dimension
 * @param stride_height Stride along height dimension
 * @param dilation_width Dilation factor along width dimension
 * @param dilation_height Dilation factor along height dimension
 *
 * @note Matrix dimensions: M = batch × output_height × output_width,
 *                          N = output_channels,
 *                          K = filter_height × filter_width × input_channels
 * @note Uses skl_gemm_f32_f32_f32_zve32f_x390 RISC-V Vector kernel
 */
void conv2d_io_nhwc_filter_hwio_im2row_gemm_f32_f32_f32_zve32f_x390(
    float *output, float *im2row, const float *input, const float *filter,
    size_t batches, size_t input_height, size_t input_width,
    size_t input_channel, size_t filter_height, size_t filter_width,
    size_t output_height, size_t output_width, size_t output_channel,
    size_t padding_width, size_t padding_height, size_t stride_width,
    size_t stride_height, size_t dilation_width, size_t dilation_height);

/**
 * @brief Direct GEMM implementation for 1x1 convolution
 *
 * Optimized implementation for 1x1 filters with stride=1 and dilation=1.
 * Treats convolution as direct matrix multiplication without im2row
 * preprocessing.
 *
 * @param output Output tensor buffer [batches, output_height, output_width,
 * output_channel]
 * @param input Input tensor buffer [batches, input_height, input_width,
 * input_channel]
 * @param filter Filter tensor buffer [1, 1, input_channel, output_channel]
 * @param batches Number of batches
 * @param input_height Height of input tensor
 * @param input_width Width of input tensor
 * @param input_channel Number of input channels
 * @param filter_height Height of convolution filter (must be 1)
 * @param filter_width Width of convolution filter (must be 1)
 * @param output_height Height of output tensor
 * @param output_width Width of output tensor
 * @param output_channel Number of output channels
 * @param padding_width Padding along width dimension (must be 0)
 * @param padding_height Padding along height dimension (must be 0)
 * @param stride_width Stride along width dimension (must be 1)
 * @param stride_height Stride along height dimension (must be 1)
 * @param dilation_width Dilation factor along width dimension (must be 1)
 * @param dilation_height Dilation factor along height dimension (must be 1)
 *
 * @note Asserts that filter_height=1, filter_width=1, stride=1, dilation=1,
 * padding=0
 * @note Matrix dimensions: M = batch × height × width, N = output_channels, K =
 * input_channels
 */
void conv2d_1x1_direct_gemm_f32_f32_f32_zve32f_x390(
    float *output, const float *input, const float *filter, size_t batches,
    size_t input_height, size_t input_width, size_t input_channel,
    size_t filter_height, size_t filter_width, size_t output_height,
    size_t output_width, size_t output_channel, size_t padding_width,
    size_t padding_height, size_t stride_width, size_t stride_height,
    size_t dilation_width, size_t dilation_height);

#ifdef __cplusplus
}
#endif
