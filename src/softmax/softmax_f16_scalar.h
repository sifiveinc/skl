// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Scalar 1D unit-strided stable softmax.
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
void skl_softmax_f16_scalar(_Float16 *pDst, const _Float16 *pSrc, _Float16 beta,
                            size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
