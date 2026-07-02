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

void skl_pack_texte_e32_e32rcpc_xsfmmbase(size_t m, size_t n, const uint32_t *a,
                                          size_t rsa, uint32_t *a_pack,
                                          size_t csa0, size_t rsa1, size_t csa1,
                                          bool pad_right, bool pad_bottom,
                                          uint32_t padding_value);

#if defined(__cplusplus)
} // extern "C"
#endif
