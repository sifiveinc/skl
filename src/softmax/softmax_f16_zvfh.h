// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector FP16 softmax function.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to the calling:
 * ```
 * skl_softmax_f16_ref(pDst, pSrc, beta, n);
 * ```
 *
 * @note
 * Will perform best when `n >= __riscv_vsetvlmax_e16m8()`.
 */
void skl_softmax_f16_zvfh(_Float16 *pDst, const _Float16 *pSrc,
                          const _Float16 beta, const size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
