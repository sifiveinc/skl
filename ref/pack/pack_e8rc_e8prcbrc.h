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
 * @brief Reference implementation for packing an 8-bit 2D matrix into a 4D
 * blocked matrix layout.
 *
 * @param m - Number of rows in the input matrix
 * @param n - Number of columns in the input matrix
 * @param src - Pointer to the input matrix
 * @param rs - Row stride of the input matrix (in elements)
 * @param cs - Column stride of the input matrix (in elements)
 * @param m0 - Number of rows per block (intra-block row dimension)
 * @param n0 - Number of columns per block (intra-block column dimension)
 * @param dst - Pointer to the output blocked matrix
 * @param rs0 - Row stride within each block (intra-block row stride)
 * @param cs0 - Column stride within each block (intra-block column stride)
 * @param rs1 - Row stride between blocks (inter-block row stride)
 * @param cs1 - Column stride between blocks (inter-block column stride)
 * @param padding_value - Value to use for padding
 *
 * Transforms a 2D matrix into a blocked 4D layout.
 *
 * The output layout consists of m1 × n1 blocks, where:
 *   - m1 = ceil(m / m0) = (m + m0 - 1) / m0  (number of row blocks)
 *   - n1 = ceil(n / n0) = (n + n0 - 1) / n0  (number of column blocks)
 *
 * Each block has dimensions m0 × n0. If m is not a multiple of m0 or n is not
 * a multiple of n0, the incomplete blocks are padded with the specified padding
 * value.
 */
void skl_pack_e8rc_e8rcprc_ref(
    size_t m,             // Num. rows in input matrix
    size_t n,             // Num. columns in input matrix
    const uint8_t *src,   // Input matrix
    size_t rs,            // Row stride of input matrix
    size_t cs,            // Column stride of input matrix
    size_t m0,            // Num. rows in a block of the input matrix
    size_t n0,            // Num. columns in a block of the input matrix
    uint8_t *dst,         // Output packed matrix
    size_t rs0,           // Row stride within a block of the output matrix
    size_t cs0,           // Column stride within a block of the output matrix
    size_t rs1,           // Row stride between blocks of the output matrix
    size_t cs1,           // Column stride between blocks of the output matrix
    uint8_t padding_value // Value to use for padding
);

#if defined(__cplusplus)
} // extern "C"
#endif
