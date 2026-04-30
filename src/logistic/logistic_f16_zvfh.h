// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RVV-based 3-ULP FP16 logistic.
 *
 * @param out - Array of output elements.
 * @param in - Array of input elements.
 * @param n - Number of elements to process.
 *
 * Computes the logistic function as defined by:
 * logistic(x) = 1 / (1 + e^(-x))
 */
void skl_logistic_3u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif
