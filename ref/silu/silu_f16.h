// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reference 1-ULP FP16 Sigmoid Linear Unit.
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
void skl_silu_1u_f16_ref(_Float16 *out, const _Float16 *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
