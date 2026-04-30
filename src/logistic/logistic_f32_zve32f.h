// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RVV-based 3-ULP FP32 logistic.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the logistic function as defined by:
 * logistic(x) = 1 / (1 + e^(-x))
 */
void skl_logistic_3u_f32_zve32f(float *out, const float *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
