// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

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
 * @brief RVV-based 8-bit general matrix pack kernel (2D to blocked 4D layout).
 *
 * @param m - Num. rows in input matrix
 * @param n - Num. columns in input matrix
 * @param src - Input matrix
 * @param rs - Row stride of input matrix
 * @param cs - Column stride of input matrix
 * @param m0 - Num. rows in a block of the input matrix
 * @param n0 - Num. columns in a block of the input matrix
 * @param dst - Output packed matrix [m1 x n1 x (m0 x n0)]
 * @param rs0 - Row stride within a block of the output matrix
 * @param cs0 - Column stride within a block of the output matrix
 * @param rs1 - Row stride between blocks of the output matrix
 * @param cs1 - Column stride between blocks of the output matrix
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
 *
 * @note Input and output matrices must not overlap.
 */
void skl_pack_e8rc_e8rcprc_zve32x(size_t m, size_t n,
                                  const uint8_t *SKL_RESTRICT src, size_t rs,
                                  size_t cs, size_t m0, size_t n0,
                                  uint8_t *SKL_RESTRICT dst, size_t rs0,
                                  size_t cs0, size_t rs1, size_t cs1,
                                  uint8_t pad);

#if defined(__cplusplus)
} // extern "C"
#endif
