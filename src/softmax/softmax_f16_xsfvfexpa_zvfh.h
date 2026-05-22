// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector FP16 softmax function accelerated with Xsfvfexpa.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_f16_ref(pDst, pSrc, beta, n);
 * ```
 *
 * Exploits the SiFive vector floating-point exponential approximation
 * instruction to compute the e^x part of softmax.
 */
void skl_softmax_f16_xsfvfexpa_zvfh(_Float16 *pDst, const _Float16 *pSrc,
                                    _Float16 beta, size_t n);
#if defined(__cplusplus)
} // extern "C"
#endif
