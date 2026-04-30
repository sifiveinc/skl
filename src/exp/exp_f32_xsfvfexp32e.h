// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Fast float32 exponential function using Xsfvfexp32e.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float32 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results for inputs |x|<  64 are accurate to 2.398 ulp.
 * Results for inputs  x ≤ -64 are zero.
 * Results for inputs  x ≥ +64 are infinite.
 */
void skl_exp_2p398u0alt64ainf_f32_xsfvfexp32e(float *out, const float *in,
                                              size_t n);

/**
 * @brief 5.3-ULP float32 exponential function.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float32 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 5.32 ulp.
 */
void skl_exp_5p32u_f32_xsfvfexp32e(float *out, const float *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
