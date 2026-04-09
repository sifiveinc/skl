// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reference 3-ULP FP32 logistic.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the logistic function as defined by:
 * logistic(x) = 1 / (1 + e^(-x))
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_logistic_3u_f32_ref(float *out, const float *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
