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
 * @brief FP32 RMS normalization function.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param pWeight - Array of weight elements (optional, can be NULL).
 * @param epsilon - Small value to avoid division by zero.
 * @param n - Number of elements to process.
 *
 * Computes the RMS normalization equivalent to:
 * ```
 * skl_rmsnorm_f32_ref(pDst, pSrc, pWeight, epsilon, n);
 * ```
 */
void skl_rmsnorm_f32_zve32f(float *pDst, const float *pSrc, const float *pWeight,
                            size_t rsc,
                            float epsilon, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
