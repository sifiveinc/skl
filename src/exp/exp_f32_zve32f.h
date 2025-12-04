// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief 1-ULP float32 exponential function.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float32 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 0.927 ulp.
 */
void skl_exp_1u_f32_zve32f(float *out, const float *in, size_t n);

#if defined(__cplusplus)
} // extern "C"
#endif
