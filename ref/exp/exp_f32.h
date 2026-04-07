// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reference FP32 Exponential function.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the exponential function e^x.

 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_exp_f32_ref(float *out, const float *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
