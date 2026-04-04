// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference 1D unit-strided stable softmax.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function as defined by the following algorithm:
 * ```
 *    x_max  := max(x_i for i < n)
 *    x_i    := e^(beta * (x_i - x_max))
 *    x_sum  := sum(x_i)
 *    x_i    := x_i / x_sum
 * ```
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_softmax_f32_ref(float *pDst, const float *pSrc, float beta,
                            size_t n);

/**
 * @brief Reference 2D stable softmax, reducing rows.
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
 * @note Input A and output S matrices may only overlap if S == A.
 *
 * @note
 * This function is for API documentation purposes only, and should
 * not be used for performance applications.
 */
void skl_softmax_2d_f32_ref(float *s, size_t rss, const float *a, size_t rsa,
                               float beta, size_t m, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
