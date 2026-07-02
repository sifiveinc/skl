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
 * @brief TE x 1 matrix packing for 32-bit matrices using SiFive's Xsfmm matrix
 * engine.
 *
 * @param m - Number of rows in row-major matrix A.
 * @param n - Number of columns in A.
 * @param a - Pointer to input matrix A.
 * @param rsa - Row stride of A (stride between rows) in elements.
 * @param a_pack - Pointer to packed output matrix A_pack.
 * @param rsa1 - Row stride between blocks of A_pack in elements.
 * @param padding_value - Padding value.
 *
 * Equivalent to scalar call:
 * ```
 * skl_pack_e32_scalar(
 *     m, n,                  // m, n,
 *     a, rsa, 1,             // a, rsa, csa
 *     te, 1,                 // m0, n0,
 *     a_pack, 1, 0, rsa1, te // a_pack, rsa0, csa0, rsa1, csa1
 *     padding_value          // padding_value
 * );
 * ```
 *
 * This function is intended to provide packing functionality for all 32-bit
 * datatypes by way of type-punning through the input/output array pointers and
 * of the padding value.
 */
void skl_pack_tex1c_e32_xsfmmbase(size_t m, size_t n, const uint32_t *a,
                                  size_t rsa, uint32_t *a_pack, size_t rsa1,
                                  uint32_t padding_value);

/**
 * @brief Xsfmm matrix transposition for 32-bit matrices.
 *
 * @param m - Number of rows in A and columns in A^T.
 * @param n - Number of columns in A and rows in A^T.
 * @param a - Pointer to input matrix A.
 * @param rsa - Row stride of A (stride between rows) in elements.
 * @param at - Pointer to output matrix A^T.
 * @param rsat - Row stride of A^T (stride between rows) in elements.
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
