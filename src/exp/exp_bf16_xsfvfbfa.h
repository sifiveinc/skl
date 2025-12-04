// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief 1-ULP BFloat16 exponential function using Xsfvfbfa.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of bfloat16 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 0.921 ulp.
 */
void skl_exp_1u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
