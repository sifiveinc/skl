// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#if !defined(__riscv_xsfvqdotq)
#error This source file requires compiler support for the Xsfvqdotq extension.
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Int8 GEMM kernel using RISC-V xsfvqdotq extension with packed matrix
 * B.
 *
 * @param m - Number of rows in matrix A and matrix C
 * @param n - Number of columns in matrix B and matrix C
 * @param k - Number of columns in matrix A and rows in matrix B
 * @param a - Pointer to matrix A in row-major format
 * @param rsa - Stride between rows of matrix A in elements
 * @param b_pack - Pointer to packed matrix B
 * @param rsb_pack - Stride between rows of packed matrix B in elements
 * @param c - Pointer to matrix C in row-major format
 * @param rsc - Stride between rows of matrix C in elements
 * @param accum - Determines if output matrix is accumulated or overwritten
 *
 * Computes `C = A * B_pack + C` (if accum=true) or `C = A * B_pack` (if
 * accum=false) for int8 matrices A and packed B with int32 output matrix C.
 * This kernel uses the SiFive xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data.
 *
 * The kernel automatically dispatches to optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * Equivalent to scalar call:
 * ```
 * // assume b_pack is pre-allocated and has size ((k + 3) / 4) * 4 * n bytes
 * skl_pack_b_i8_xsfvqdotq(k, n, b, n, b_pack);
 * skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq(m, n, k, a, rsa, b_pack, rsb_pack, c,
 *                                     rsc, 0);
 * Equivalent to scalar call:
 * skl_gemm_i8_i32_scalar(m, n, k, 1, a, rsa, b, n, 0, c, rsc);
 * ```
 *
 * @note
 * Matrix B must be pre-packed using skl_pack_b_i8_xsfvqdotq(). Matrix A is used
 * directly in row-major format without requiring pre-packing.
 */
void skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq(size_t m, size_t n, size_t k,
                                         const int8_t *a, size_t rsa,
                                         const int8_t *b_pack, size_t rsb_pack,
                                         int32_t *c, size_t rsc, bool accum);
#if defined(__cplusplus)
} // extern "C"
#endif
