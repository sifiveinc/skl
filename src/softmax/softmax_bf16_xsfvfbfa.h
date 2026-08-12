// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Vector BFloat16 softmax function using the Xsfvfbfa extension.
 *
 * @param dst - Array of output elements.
 * @param src - Array of input elements.
 * @param beta - Scaling factor for exponential function arguments.
 * @param n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_bf16_ref(dst, src, beta, n);
 * ```
 */
void skl_softmax_bf16_xsfvfbfa(__bf16 *dst, const __bf16 *src, __bf16 beta,
                               size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
