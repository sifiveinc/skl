// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfbfmin)
#error This file requires the Xsfvfexpa and Zvfbfmin extensions
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector BFloat16 softmax function using the Xsfvfexpa and
 * Zvfbfmin extensions.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_bf16_ref(pDst, pSrc, beta, n);
 * ```
 */
void skl_softmax_bf16_xsfvfexpa_zvfbfmin(__bf16 *pDst, const __bf16 *pSrc,
                                         const __bf16 beta, const size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
