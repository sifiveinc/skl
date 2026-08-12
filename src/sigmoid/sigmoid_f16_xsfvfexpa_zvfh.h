// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Xsfvfexpa-based FP16 sigmoid with optional fused multiplications.
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
void skl_sigmoid_f16_xsfvfexpa_zvfh(_Float16 *out, _Float16 beta,
                                    const _Float16 *x, const _Float16 *y,
                                    const _Float16 *up, _Float16 delta,
                                    size_t n);

/**
 * @brief xsfvfexpa_zvfh-based FP16 logistic: out = logistic(x).
 *
 * Convenience wrapper around skl_sigmoid_f16_xsfvfexpa_zvfh.
 */
void skl_logistic_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *x,
                                     size_t n);

/**
 * @brief xsfvfexpa_zvfh-based FP16 SiLU: out = x * logistic(x).
 *
 * Convenience wrapper around skl_sigmoid_f16_xsfvfexpa_zvfh.
 */
void skl_silu_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *x, size_t n);

/**
 * @brief xsfvfexpa_zvfh-based FP16 Swish: out = x * logistic(beta * x).
 *
 * Swish generalizes SiLU with an arbitrary beta (SiLU fixes beta = 1).
 * Convenience wrapper around skl_sigmoid_f16_xsfvfexpa_zvfh.
 */
void skl_swish_f16_xsfvfexpa_zvfh(_Float16 *out, _Float16 beta,
                                  const _Float16 *x, size_t n);

/**
 * @brief xsfvfexpa_zvfh-based FP16 GLU: out = x * logistic(y).
 *
 * Convenience wrapper around skl_sigmoid_f16_xsfvfexpa_zvfh.
 */
void skl_glu_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *x,
                                const _Float16 *y, size_t n);

/**
 * @brief xsfvfexpa_zvfh-based FP16 SwiGLU: out = silu(gate) * (up + delta).
 *
 * Convenience wrapper around skl_sigmoid_f16_xsfvfexpa_zvfh.
 */
void skl_swiglu_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *gate,
                                   const _Float16 *up, _Float16 delta,
                                   size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
