// Copyright (c) 2025 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Fast float16 exponential function using Xsfvfexp16e.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results for inputs |x|<  8 are accurate to 1.022 ulp.
 * Results for inputs  x ≤ -8 are zero.
 * Results for inputs  x ≥ +8 are infinite.
 */
void skl_exp_1p022u0alt8ainf_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in,
                                             size_t n);

/**
 * @brief General float16 exponential function using Xsfvfexp16e.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 3.16 ulp.
 */
void skl_exp_3p16u_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
