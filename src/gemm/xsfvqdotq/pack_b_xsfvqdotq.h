// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#if __STDC_VERSION__ < 202311L
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Pack matrix B for use with xsfvqdotq GEMM kernel.
 *
 * @param k - Number of rows in matrix B
 * @param n - Number of columns in matrix B
 * @param b - Pointer to input matrix B in row-major format
 * @param rsb - Stride between rows of matrix B in elements
 * @param b_pack - Packed output matrix of ((k + 3) / 4) * rsb1 bytes
 * @param rsb1 - Row stride between blocks of packed matrix B in elements.
 *
 * Packs matrix B into a format optimized for the SiFive xsfvqdotq extension.
 * The packing operation reorganizes the matrix data to enable efficient vector
 * quad widening 4D dot product operations. The function processes the matrix
 * in groups of 4 rows at a time to interleave the data in a format suitable for
 * the xsfvqdotq instructions.
 *
 * The packed format groups 4 consecutive rows together and stores them in an
 * interleaved pattern that allows the GEMM kernel to efficiently compute matrix
 * multiplications. When k is not a multiple of 4, the remaining rows are
 * zero-padded to complete the 4-row groups.
 *
 * @note
 * The output buffer b_pack must be large enough to hold the padded data.
 * The minimum size is ((k + 3) / 4) * 4 * n bytes.
 *
 * @note
 * This function is designed to work with
 * skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq().
 */
void skl_pack_b_i8_xsfvqdotq(size_t k, size_t n, const int8_t *b, size_t rsb,
                             int8_t *b_pack, size_t rsb1);

#if defined(__cplusplus)
} // extern "C"
#endif
