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
 * @brief Reference implementation for packing an 32-bit 2D matrix into a 4D
 * blocked matrix layout.
 *
 * @param m - Num. rows in input matrix
 * @param n - Num. columns in input matrix
 * @param src - Input matrix
 * @param rs - Stride between rows of input matrix
 * @param cs - Stride between columns of input matrix
 * @param m0 - Num. rows in a block of output matrix
 * @param n0 - Num. columns in a block of output matrix
 * @param dst - Packed output matrix [ceil(m/m0) x ceil(n/n0) x (m0 x n0)]
 * @param rs0 - Row stride within a block of output matrix
 * @param cs0 - Column stride within a block of output matrix
 * @param rs1 - Row stride between blocks of output matrix
 * @param cs1 - Column stride between blocks of output matrix
 * @param pad - Value to insert for padded elements (usually 0)
 *
 * Transforms a 2D matrix into a blocked 4D layout.
 *
 * The output layout consists of m1 × n1 blocks, where:
 *   - m1 = ceil(m / m0) = (m + m0 - 1) / m0  (number of block-rows)
 *   - n1 = ceil(n / n0) = (n + n0 - 1) / n0  (number of block-columns)
 *
 * Each block has dimensions m0 × n0. If m is not a multiple of m0 or n is not
 * a multiple of n0, the incomplete blocks are padded with the specified padding
 * value.
 */
void skl_pack_e32rc_e32rcprc_ref(size_t m, size_t n,
                                 const uint32_t *SKL_RESTRICT src, size_t rs,
                                 size_t cs, size_t m0, size_t n0,
                                 uint32_t *SKL_RESTRICT dst, size_t rs0,
                                 size_t cs0, size_t rs1, size_t cs1,
                                 uint32_t pad);

#if defined(__cplusplus)
} // extern "C"
#endif
