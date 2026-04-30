// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Xsfvfexpa-based 52-ULP FP32 Sigmoid Linear Unit.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the SiLU function as defined by:
 * silu(x) = x / (1 + e^(-x))

 * @note
 * The result for x == -infty is NaN.
 */
void skl_silu_52u_f32_xsfvfexpa(float *out, const float *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
