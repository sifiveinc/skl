// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector FP16 softmax function accelerated with Xsfvfexp16e.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to the scalar call:
 * ```
 * skl_softmax_f16_ref(pDst, pSrc, beta, n);
 * ```
 *
 * Exploits the SiFive vector floating-point exponential function
 * instruction to compute the e^x part of softmax.
 */
void skl_softmax_f16_xsfvfexp16e(_Float16 *pDst, const _Float16 *pSrc,
                                 const _Float16 beta, const size_t n);
#if defined(__cplusplus)
} // extern "C"
#endif
