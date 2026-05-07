// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FP32 logistic accelerated with Xsfvfexp32e.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the logistic function as defined by:
 * logistic(x) = 1 / (1 + e^(-x))
 *
 * using the SiFive vector floating-point exponential function
 * instruction to compute the `e^(-x)` component.
 *
 * Results are accurate to less than 4.5168 ulp.
 *
 * @note
 * The result for NaN inputs is 0.
 */
void skl_logistic_5u_f32_xsfvfexp32e(float *out, const float *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
