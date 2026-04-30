// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief 1-ULP float32 exponential function using Xsfvfexpa.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float32 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 1.0 ulp.
 */
void skl_exp_1u_f32_xsfvfexpa(float *out, const float *in, size_t n);

/**
 * @brief Fast float32 exponential function using Xsfvfexpa.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float32 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n
 *
 * Results for inputs in[i] ≥ -0x1.5d639ep6 are accurate to less than
 * 1.0002 ulp.  Results for lesser inputs eventually vanish to zero
 * after a quicker than usual underflow.  That is:
 *   1. in[i] < -0x1.5d639ep6  =>  0 ≤ out[i] < exp(in[i]), and
 *   2. in[i] == -inf  =>  out[i] == 0.
 */
void skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa(float *out, const float *in,
                                                size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
