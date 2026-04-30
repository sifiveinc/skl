// Copyright (c) 2025 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief 1-ULP float16 exponential function using Xsfvfexpa.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 1.0 ulp.
 */
void skl_exp_1u_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *in, size_t n);

/**
 * @brief Fast float16 exponential function using Xsfvfexpa.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n
 *
 * Results for inputs in[i] ≥ -0x1.37p3 are accurate to less than
 * 1.132 ulp.  Results for lesser inputs eventually vanish to zero
 * after a quicker than usual underflow.  That is:
 *   1. in[i] < -0x1.37p3  =>  0 ≤ out[i] < exp(in[i]), and
 *   2. in[i] == -inf  =>  out[i] == 0.
 */
void skl_exp_1p132ugen37P3s0_f16_xsfvfexpa_zvfh(_Float16 *out,
                                                const _Float16 *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
