// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV-based 8-bit general matrix pack kernel (2D to blocked 4D layout).
 *
 * @param m      Number of rows in the input matrix
 * @param n      Number of columns in the input matrix
 * @param src    Pointer to the input matrix
 * @param rs     Row stride of the input matrix (in elements)
 * @param cs     Column stride of the input matrix (in elements)
 * @param m0     Number of rows per block (intra-block row dimension)
 * @param n0     Number of columns per block (intra-block column dimension)
 * @param dst    Pointer to the output blocked matrix
 * @param rs0    Row stride within each block (intra-block row stride)
 * @param cs0    Column stride within each block (intra-block column stride)
 * @param rs1    Row stride between blocks (inter-block row stride)
 * @param cs1    Column stride between blocks (inter-block column stride)
 *
 * Transforms a 2D matrix into a blocked 4D layout.
 *
 * The output layout consists of m1 × n1 blocks, where:
 *   - m1 = ceil(m / m0) = (m + m0 - 1) / m0  (number of row blocks)
 *   - n1 = ceil(n / n0) = (n + n0 - 1) / n0  (number of column blocks)
 *
 * Each block has dimensions m0 × n0. If m is not a multiple of m0 or n is not
 * a multiple of n0, the incomplete blocks are padded with zeros.
 */
void skl_pack_e8rc_e8rcbrc_zve32x(size_t m, size_t n, const uint8_t *src,
                                  size_t rs, size_t cs, size_t m0, size_t n0,
                                  uint8_t *dst, size_t rs0, size_t cs0,
                                  size_t rs1, size_t cs1);

#if defined(__cplusplus)
} // extern "C"
#endif
