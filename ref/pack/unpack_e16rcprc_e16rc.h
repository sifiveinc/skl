// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#include "skl-common.h"
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference implementation for unpacking an 16-bit 4D blocked matrix
 * into a 2D matrix.
 *
 * @param m0 - Num. rows in a block of input matrix
 * @param n0 - Num. columns in a block of input matrix
 * @param src - Packed input matrix [ceil(m/m0) x ceil(n/n0) x (m0 x n0)]
 * @param rs0 - Row stride within a block of input matrix
 * @param cs0 - Column stride within a block of input matrix
 * @param rs1 - Row stride between blocks of input matrix
 * @param cs1 - Column stride between blocks of input matrix
 * @param m - Num. rows in output matrix
 * @param n - Num. columns in output matrix
 * @param dst - Output matrix
 * @param rs - Stride between rows of output matrix
 * @param cs - Stride between columns of output matrix
 *
 * Transforms a blocked 4D layout back into a 2D matrix.
 *
 * Input: Blocked layout with m1 × n1 blocks, where each block is m0 × n0
 * elements
 *   - m1 = ⌈m / m0⌉ = (m + m0 - 1) / m0  (number of block-rows)
 *   - n1 = ⌈n / n0⌉ = (n + n0 - 1) / n0  (number of block-columns)
 *
 * Output: 2D matrix of dimensions m × n
 *
 * Extracts only the valid m × n elements from the blocked layout. When m or n
 * is not a multiple of the block dimensions, elements beyond the valid range
 * in incomplete blocks are ignored.
 *
 * @note Input and output matrices must not overlap.
 */
void skl_unpack_e16rcprc_e16rc_ref(size_t m0, size_t n0,
                                   const uint16_t *SKL_RESTRICT src, size_t rs0,
                                   size_t cs0, size_t rs1, size_t cs1, size_t m,
                                   size_t n, uint16_t *SKL_RESTRICT dst,
                                   size_t rs, size_t cs);

#if defined(__cplusplus)
} // extern "C"
#endif
