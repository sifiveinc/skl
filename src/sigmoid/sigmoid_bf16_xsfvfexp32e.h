// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Xsfvfexp32e-based BF16 sigmoid with optional fused multiplications.
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
 *   out = sigmoid(beta * x)
 *   if (y)  out *= y
 *   if (up) out *= (up + delta)
 * where
 *   sigmoid(z) = 1 / (1 + e^(-z))
 *
 * @note
 * The result for beta * x == -infty is NaN.
 */
void skl_sigmoid_bf16_xsfvfexp32e(__bf16 *out, __bf16 beta, const __bf16 *x,
                                  const __bf16 *y, const __bf16 *up,
                                  __bf16 delta, size_t n);

/**
 * @brief xsfvfexp32e-based BF16 logistic: out = sigmoid(x).
 *
 * Convenience wrapper around skl_sigmoid_bf16_xsfvfexp32e.
 */
void skl_logistic_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *x, size_t n);

/**
 * @brief xsfvfexp32e-based BF16 SiLU: out = x * sigmoid(x).
 *
 * Convenience wrapper around skl_sigmoid_bf16_xsfvfexp32e.
 */
void skl_silu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *x, size_t n);

/**
 * @brief xsfvfexp32e-based BF16 Swish: out = x * sigmoid(beta * x).
 *
 * Swish generalizes SiLU with an arbitrary beta (SiLU fixes beta = 1).
 * Convenience wrapper around skl_sigmoid_bf16_xsfvfexp32e.
 */
void skl_swish_bf16_xsfvfexp32e(__bf16 *out, __bf16 beta, const __bf16 *x,
                                size_t n);

/**
 * @brief xsfvfexp32e-based BF16 GLU: out = x * sigmoid(y).
 *
 * Convenience wrapper around skl_sigmoid_bf16_xsfvfexp32e.
 */
void skl_glu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *x, const __bf16 *y,
                              size_t n);

/**
 * @brief xsfvfexp32e-based BF16 SwiGLU: out = silu(gate) * (up + delta).
 *
 * Convenience wrapper around skl_sigmoid_bf16_xsfvfexp32e.
 */
void skl_swiglu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *gate,
                                 const __bf16 *up, __bf16 delta, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
