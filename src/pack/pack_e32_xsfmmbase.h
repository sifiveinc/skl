// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief TE x 1 matrix packing kernel for 32-bit matrices using SiFive's Xsfmm
 * matrix engine.
 *
 * @param m - Num. rows in input matrix.
 * @param n - Num. columns in input matrix.
 * @param src - Pointer to input matrix.
 * @param rs - Stride between rows of input matrix.
 * @param dst - Pointer to packed output matrix.
 * @param rs1 - Row stride between blocks of output matrix.
 * @param cs1 - Column stride between blocks of output matrix.
 * @param pad - Padding value.
 *
 * Equivalent to scalar call:
 * ```
 * skl_pack_e32rc_e32rcprc_ref(
 *     m, n,               // m, n,
 *     src, rs, 1,         // src, rs, cs
 *     te, 1,              // m0, n0,
 *     dst, 1, 0, rs1, cs1 // dst, rs0, cs0, rs1, cs1
 *     pad                 // pad
 * );
 * ```
 *
 * This function is intended to provide packing functionality for all 32-bit
 * datatypes by way of type-punning through the input/output array pointers and
 * of the padding value.
 */
void skl_pack_e32_e32rcptex1c_xsfmmbase(size_t m, size_t n, const uint32_t *src,
                                        size_t rs, uint32_t *dst, size_t rs1,
                                        size_t cs1, uint32_t pad);

/**
 * @brief Xsfmm matrix transposition for 32-bit matrices.
 *
 * @param m - Number of rows in A and columns in A^T.
 * @param n - Number of columns in A and rows in A^T.
 * @param a - Pointer to input matrix A.
 * @param rsa - Stride between rows of A in elements.
 * @param at - Pointer to output matrix A^T.
 * @param rsat - Stride between rows of A^T in elements.
 *
 * Both A and A^T must be row-major.
 *
 * This function is intended to provide transpose functionality for all 32-bit
 * datatypes by way of type-punning through the input/output array pointers.
 *
 * @note Input and output matrices must not overlap.
 */
void skl_transpose_e32_xsfmmbase(size_t m, size_t n,
                                 const uint32_t *SKL_RESTRICT a, size_t rsa,
                                 uint32_t *SKL_RESTRICT at, size_t rsat);

#if defined(__cplusplus)
} // extern "C"
#endif
