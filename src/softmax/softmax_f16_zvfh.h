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
 * @brief Vector F16 softmax function.
 *
 * @param[out] dst - Array of output elements.
 * @param[in] src - Array of input elements.
 * @param[in] beta - Scaling factor for exponential function arguments.
 * @param[in] n - Number of elements to process.
 *
 * Computes the softmax function equivalent to calling:
 * ```
 * skl_softmax_f16_ref(dst, src, beta, n);
 * ```
 *
 * @note
 * Will perform best when `n >= __riscv_vsetvlmax_e16m8()`.
 */
void skl_softmax_f16_zvfh(_Float16 *dst, const _Float16 *src, _Float16 beta,
                          size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
