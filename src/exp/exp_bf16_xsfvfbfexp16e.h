// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_xsfvfbfexp16e)
#error This file requires the Xsfvfbfexp16e extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief 1-ULP BFloat16 exponential function using Xsfvfbfexp16e.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of bfloat16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results for inputs |x|<  64 are accurate to exactly 1.0 ulp.
 * Results for inputs  x ≤ -64 are zero.
 * Results for inputs  x ≥ +64 are infinite.
 */
void skl_exp_1u0alt64ainf_bf16_xsfvfbfexp16e(__bf16 *out, const __bf16 *in,
                                             size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
