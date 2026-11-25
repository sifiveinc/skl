// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Correctly-rounded BF16 square root function.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of BF16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = sqrt(in[i])` for 0 ≤ i < n.
 *
 * Results are correctly-rounded in round-to-nearest-even mode.
 */
void skl_sqrt_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);

/**
 * @brief 1-ULP BF16 square root function.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of BF16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = sqrt(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 1.836 ulp.
 */
void skl_sqrt_2u_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
