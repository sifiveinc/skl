// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Zve32f-based FP32 sigmoid with optional fused multiplications.
 *
 * @param out - Array of output elements.
 * @param beta - Scaling factor for the exponential part.
 * @param x - Array of input elements. If NULL, the function computes nothing.
 * @param y - Optional array. If not NULL, the result is multiplied by y.
 * @param up - Optional array. If not NULL, the result is multiplied by
 *             (up + delta).
 * @param delta - Bias added to up before multiplying.
 * @param n - Number of elements to process.
 *
 * Computes, elementwise:
 *   out = logistic(beta * x)
 *   if (y)  out *= y
 *   if (up) out *= (up + delta)
 * where
 *   logistic(z) = 1 / (1 + e^(-z))
 */
void skl_sigmoid_f32_zve32f(float *out, float beta, const float *x,
                            const float *y, const float *up, float delta,
                            size_t n);

/**
 * @brief zve32f-based FP32 logistic: out = logistic(x).
 *
 * Convenience wrapper around skl_sigmoid_f32_zve32f.
 */
void skl_logistic_f32_zve32f(float *out, const float *x, size_t n);

/**
 * @brief zve32f-based FP32 SiLU: out = x * logistic(x).
 *
 * Convenience wrapper around skl_sigmoid_f32_zve32f.
 */
void skl_silu_f32_zve32f(float *out, const float *x, size_t n);

/**
 * @brief zve32f-based FP32 Swish: out = x * logistic(beta * x).
 *
 * Swish generalizes SiLU with an arbitrary beta (SiLU fixes beta = 1).
 * Convenience wrapper around skl_sigmoid_f32_zve32f.
 */
void skl_swish_f32_zve32f(float *out, float beta, const float *x, size_t n);

/**
 * @brief zve32f-based FP32 GLU: out = x * logistic(y).
 *
 * Convenience wrapper around skl_sigmoid_f32_zve32f.
 */
void skl_glu_f32_zve32f(float *out, const float *x, const float *y, size_t n);

/**
 * @brief zve32f-based FP32 SwiGLU: out = silu(gate) * (up + delta).
 *
 * Convenience wrapper around skl_sigmoid_f32_zve32f.
 */
void skl_swiglu_f32_zve32f(float *out, const float *gate, const float *up,
                           float delta, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
