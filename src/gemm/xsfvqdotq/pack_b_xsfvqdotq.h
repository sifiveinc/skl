// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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
 * @brief Pack matrix B for use with xsfvqdotq GEMM kernel.
 *
 * @param k - Number of rows in matrix B
 * @param n - Number of columns in matrix B
 * @param b - Pointer to input matrix B in row-major format
 * @param rsb - Stride between rows of matrix B in elements
 * @param b_pack - Packed output matrix of ((k + 3) / 4) * 4 * n bytes
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
 * The minimum size is ((k + 3) / 4) * 4 * n bytes, and must be 4-byte aligned.
 *
 * @note
 * This function is designed to work with skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq().
 */
void skl_pack_b_i8_xsfvqdotq(size_t k, size_t n, const int8_t *b, size_t rsb,
                             int8_t *b_pack);

#if defined(__cplusplus)
} // extern "C"
#endif
