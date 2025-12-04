// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zvfbfmin)
#error This file requires the Zvfbfmin extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector BFloat16 softmax function using the Zvfbfmin extension.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to the scalar call:
 * ```
 * skl_softmax_bf16_scalar(pDst, pSrc, beta, n);
 * ```
 */
void skl_softmax_bf16_zvfbfmin(__bf16 *pDst, const __bf16 *pSrc,
                               const __bf16 beta, const size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
