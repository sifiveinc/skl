// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief FP16 RMS normalization function.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param pWeight - Array of weight elements (optional, can be NULL).
 * @param epsilon - Small value to avoid division by zero.
 * @param n - Number of elements to process.
 *
 * Computes the RMS normalization equivalent to:
 * ```
 * skl_rmsnorm_f16_ref(pDst, pSrc, pWeight, epsilon, n);
 * ```
 */
void skl_rmsnorm_f16_zvfh(const _Float16 *pDst, const _Float16 *pSrc,
                          const _Float16 *pWeight, _Float16 epsilon, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
