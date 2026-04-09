// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

#include "skl-common.h"
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV matrix transposition for 32-bit matrices.
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
void skl_transpose_e32_zve32x(size_t m, size_t n,
                              const uint32_t *SKL_RESTRICT a, size_t rsa,
                              uint32_t *SKL_RESTRICT at, size_t rsat);

#if defined(__cplusplus)
} // extern "C"
#endif
