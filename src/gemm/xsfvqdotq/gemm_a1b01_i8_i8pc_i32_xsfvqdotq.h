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
 * @param m - Number of rows in matrix A and matrix C.
 * @param n - Number of columns in matrix B and matrix C.
 * @param k - Number of columns in matrix A and rows in matrix B.
 * @param a - Pointer to matrix A in row-major format.
 * @param rsa - Stride between rows of matrix A in elements.
 * @param b_pack - Pointer to packed matrix B.
 * @param rsb1 - Row stride between blocks of packed matrix B in elements.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 * @param accum - Determines if output matrix is accumulated or overwritten.
 *
 * Computes `C = A * B_pack + C` (if `accum == true`) or `C = A * B_pack` (if
 * `accum == false`) for int8 matrices A and packed B with int32 output matrix
 * C. This kernel uses the SiFive xsfvqdotq extension for vector quad widening
 * 4D dot product operations to achieve high performance on 8-bit integer data.
 *
 * The kernel automatically dispatches to optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * When k % 4 == 0, equivalent to scalar call:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_scalar(
 *     1, 1, 4, m, n, k / 4,   // m0, n0, k0, m1, n1, k1
 *     1,                      // alpha
 *     a, 0, 1, rsa, 4,        // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 1, 0, rsb1, 4,  // b_pack, rsb0, csb0, rsb1, csb1
 *     accum ? 1 : 0,          // beta
 *     c, 0, 0, rsc, 1         // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * Assuming b_pack is pre-allocated and has size ((k + 3) / 4) * rsb1 bytes,
 * ```
 * skl_pack_b_i8_xsfvqdotq(k, n, b, rsb, b_pack, rsb1);
 * skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq(m, n, k, a, rsa, b_pack, rsb1, c, rsc,
 *                                      accum);
 * ```
 * is equivalent to scalar call:
 * ```
 * skl_gemm_i8_i32_scalar(m, n, k, 1, a, rsa, b, rsb, accum ? 1 : 0, c, rsc);
 * ```
 *
 * @note
 * Matrix B must be pre-packed using skl_pack_b_i8_xsfvqdotq(). Matrix A is used
 * directly in row-major format without requiring pre-packing. Performance will
 * be best when A is 4-byte-aligned and rsa is a multiple of 4.
 */
void skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq(size_t m, size_t n, size_t k,
                                          const int8_t *a, size_t rsa,
                                          const int8_t *b_pack, size_t rsb1,
                                          int32_t *c, size_t rsc, bool accum);
#if defined(__cplusplus)
} // extern "C"
#endif
