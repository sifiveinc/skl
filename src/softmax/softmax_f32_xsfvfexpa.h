// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector FP32 softmax function accelerated with Xsfvfexpa.
 *
 * @param pDst - Array of output elements.
 * @param pSrc - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to the scalar call:
 * ```
 * skl_softmax_f32_scalar(pDst, pSrc, beta, n);
 * ```
 *
 * Exploits the SiFive vector floating-point exponential approximation
 * instruction to compute the e^x part of softmax.
 */
void skl_softmax_f32_xsfvfexpa(float *pDst, const float *pSrc, const float beta,
                               const size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
