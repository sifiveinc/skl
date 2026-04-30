// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief 1-ULP float16 exponential function.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 0.926 ulp.
 */
void skl_exp_1u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
