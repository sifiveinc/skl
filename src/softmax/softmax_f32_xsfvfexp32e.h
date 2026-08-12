// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief F32 softmax function accelerated with Xsfvfexp32e.
 *
 * @param dst - Array of output elements.
 * @param src - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_f32_ref(dst, src, beta, n);
 * ```
 *
 * Exploits the SiFive vector floating-point exponential function
 * instruction to compute the e^x part of softmax.
 */
void skl_softmax_f32_xsfvfexp32e(float *dst, const float *src, float beta,
                                 size_t n);

/**
 * @brief F32 2D stable softmax, reducing rows.
 *
 * @param s - Pointer to output matrix S.
 * @param rss - Row stride of S (stride between rows) in elements.
 * @param a - Pointer to input matrix A.
 * @param rsa - Row stride of A (stride between rows) in elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param m - Number of rows in S and A.
 * @param n - Number of columns in S and A.
 *
 * Both S and A are unit-stride row-major matrices.
 *
 * This function uses SiFive's vector floating-point exponential
 * function instruction to compute the e^x part of softmax.
 *
 * @note Input A and output S matrices may only overlap if S == A.
 */
void skl_softmax_f32r_xsfvfexp32e(float *s, size_t rss, const float *a,
                                  size_t rsa, float beta, size_t m, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
