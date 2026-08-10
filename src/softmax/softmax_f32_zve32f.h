// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief F32 softmax function.
 *
 * @param[out] dst - Array of output elements.
 * @param[in]  src - Array of input elements.
 * @param[in]  beta - Scaling factor for exponential function arguments.
 * @param[in]  n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_f32_ref(dst, src, beta, n);
 * ```
 */
void skl_softmax_f32_zve32f(float *dst, const float *src, float beta, size_t n);


/**
 * @brief F32 2D stable softmax, reducing rows.
 *
 * @param[out] s - Pointer to output matrix S.
 * @param[in] rss - Row stride of S (stride between rows) in elements.
 * @param[in] a - Pointer to input matrix A.
 * @param[in] rsa - Row stride of A (stride between rows) in elements.
 * @param[in] beta - Scaling factor for exponential function arguments.
 * @param[in] m - Number of rows in S and A, and size of arrays MAX and SUM
 * @param[in] n - Number of columns in S and A.
 *
 * Both S and A are unit-stride row-major matrices.
 *
 * Computes row-wise softmax equivalent to:
 * ```
 * for (i = 0; i < m; i++)
 *   skl_softmax_f32_ref(s + i * rss, a + i * rsa, beta, n);
 * ```
 *
 * @note Input A and output S matrices may only overlap if S == A.
 */
void skl_softmax_f32r_zve32f(float *s, size_t rss, const float *a, size_t rsa,
                             float beta, size_t m, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
