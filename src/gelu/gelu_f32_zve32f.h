// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Fast vector FP32 GELU function.
 *
 * @param dst - Array of output elements.
 * @param src - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the GELU activation function with the following error
 * tolerances:
 *
 * - Results for inputs x ≥ -0 are accurate to 37020.11 ulp.
 * - Results for inputs x ≥ -1 are accurate to 93836.86 ulp.
 * - Results for inputs x ≥ -2 are accurate to 540096.2 ulp.
 * - Results for inputs x < -2 are accurate to 1.6778e7 ulp.
 * - Smallest result for x < -2 is -0x1.1f3d08p-22 at -0x1.7efc0cp+1
 * - Zero results for x < -2 are unsigned.
 * - The result for x == -infty is NaN.
 *
 * The implementation uses a polynomial of degree 9.  Its accuracy is
 * generally significantly better than "sigmoid" style approximations
 * while being faster.
 */
void skl_gelu_p9_f32_zve32f(float *dst, const float *src, size_t n);

/**
 * @brief Vector FP32 GELU function.
 *
 * @param dst - Array of output elements.
 * @param src - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the GELU activation function with the following error
 * tolerances:
 *
 * - Results for inputs x ≥ -0 are accurate to 2298.53 ulp.
 * - Results for inputs x ≥ -1 are accurate to 6809.19 ulp.
 * - Results for inputs x ≥ -2 are accurate to 69076.5 ulp.
 * - Results for inputs x < -2 are accurate to 1.678e7 ulp.
 * - Smallest result for x < -2 is -0x1.0417a8p-19 at -0x1.ce62f0p+1
 * - Zero results for x < -2 are unsigned.
 * - The result for x == -infty is NaN.
 *
 * The implementation uses a polynomial of degree 13.  Its accuracy is
 * generally better than "tanh" approximations while being
 * considerably faster.
 */
void skl_gelu_p13_f32_zve32f(float *dst, const float *src, size_t n);

/**
 * @brief Vector FP32 GELU function.
 *
 * @param dst - Array of output elements.
 * @param src - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the GELU activation function with the following error
 * tolerances:
 *
 * - Results for inputs x ≥ -0 are accurate to 238.76 ulp.
 * - Results for inputs x ≥ -1 are accurate to 238.76 ulp.
 * - Results for inputs x ≥ -2 are accurate to 2731.2 ulp.
 * - Results for inputs x < -2 are accurate to 1.68e7 ulp.
 * - Smallest result for x < -2 is -0x1.e3bc70p-19 at -0x1.01fe1ap+2
 * - Zero results for x < -2 are unsigned.
 * - The result for x == -infty is NaN.
 *
 * The implementation uses a polynomial of degree 17.  Its accuracy is
 * better than "tanh" approximations while being faster.
 */
void skl_gelu_p17_f32_zve32f(float *dst, const float *src, size_t n);

/**
 * @brief Accurate vector FP32 GELU function.
 *
 * @param dst - Array of output elements.
 * @param src - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the GELU activation function with the following error
 * tolerances:
 *
 * - Results for inputs x ≥ -0 are accurate to 3.1136 ulp.
 * - Results for inputs x ≥ -1 are accurate to 4.8028 ulp.
 * - Results for inputs x ≥ -2 are accurate to 61.920 ulp.
 * - Results for inputs x < -2 are accurate to 1.68e7 ulp.
 * - Smallest result for x < -2 is -0x1.446654p-21 at -0x1.446654p+2
 * - Zero results for x < -2 are unsigned.
 * - The result for x == -infty is NaN.
 *
 * The implementation uses a rational polynomial.  Its accuracy is
 * better than the other polynomial-based variants but is slower.
 */
void skl_gelu_rat_f32_zve32f(float *dst, const float *src, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
