// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#if !defined(__riscv_xsfvfexp32e) || !defined(__riscv_zvfbfmin)
#error This file requires the Xsfvfexp32e and Zvfbfmin extensions
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector BFloat16 softmax function using the Xsfvfexp32e and
 * Zvfbfmin extensions.
 *
 * @param[out] dst - Array of output elements.
 * @param[in] src - Array of input elements.
 * @param[in] beta - Scaling factor for exponential function arguments.
 * @param[in] n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_bf16_ref(dst, src, beta, n);
 * ```
 *
 * This function uses the SiFive vector floating-point exponential
 * function instruction to compute the e^x part of softmax.
 */
void skl_softmax_bf16_xsfvfexp32e_zvfbfmin(__bf16 *dst, const __bf16 *src,
                                           __bf16 beta, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
