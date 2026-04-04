// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scalar 52-ULP FP32 Sigmoid Linear Unit.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the SiLU function as defined by:
 * silu(x) = x / (1 + e^(-x))

 * @note
 * The result for x == -infty is NaN.
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_silu_52u_f32_ref(float *out, const float *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
