// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference BFloat16 1D unit-strided softmax.
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
void skl_softmax_bf16_ref(__bf16 *pDst, const __bf16 *pSrc, const __bf16 beta,
                          const size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
