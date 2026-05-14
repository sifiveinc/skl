// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief BF16 logistic using Zve32f.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the logistic function as defined by:
 * logistic(x) = 1 / (1 + e^(-x))
 *
 * Results are accurate to less than 1.82 ulp.
 */
void skl_logistic_2u_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
