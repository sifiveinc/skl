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
 * @brief Xsfvqdotq int8 A * B_pack matrix-matrix multiplication with row-major
 * A, packed B_pack, and row-major int32 output.
 *
 * @param m - Number of rows in matrix A and matrix C.
 * @param n - Number of columns in matrix B and matrix C.
 * @param k - Number of columns in matrix A and rows in matrix B.
 * @param alpha - Scalar multiplier for A * B product.
 * @param a - Pointer to matrix A in row-major format.
 * @param rsa - Stride between rows of matrix A in elements.
 * @param b_pack - Pointer to packed matrix B_pack.
 * @param rsb1 - Row stride between blocks of packed matrix B_pack in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 *
 * Computes `C = alpha * A * B_pack + beta * C` for int8 matrix A, packed int8
 * matrix B_pack, and int32 output matrix C.
 *
 * When k % 4 == 0, equivalent to scalar call:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_scalar(
 *     1, 1, 4, m, n, k / 4,   // m0, n0, k0, m1, n1, k1
 *     alpha,                  // alpha
 *     a, 0, 1, rsa, 4,        // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 1, 0, rsb1, 4,  // b_pack, rsb0, csb0, rsb1, csb1
 *     beta,                   // beta
 *     c, 0, 0, rsc, 1         // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * If B is a k x n int8 row-major matrix with row stride rsb and b_pack is
 * pre-allocated with size ((k + 3) / 4) * rsb1 bytes, then
 * ```
 * skl_pack_b_i8_xsfvqdotq(k, n, b, rsb, b_pack, rsb1);
 * skl_gemm_i8_i8pc_i32_xsfvqdotq(m, n, k, alpha, a, rsa, b_pack, rsb1, beta, c,
 *                                rsc);
 * ```
 * is equivalent to scalar call:
 * ```
 * skl_gemm_i8_i32_scalar(m, n, k, alpha, a, rsa, b, rsb, beta, c, rsc);
 * ```
 *
 * This kernel uses the SiFive Xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data. The
 * kernel automatically dispatches to optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * @note
 * Matrix B_pack must be pre-packed using skl_pack_b_i8_xsfvqdotq(). Matrix A is
 * used directly in row-major format without requiring pre-packing. Performance
 * will be best when A is 4-byte-aligned and rsa is a multiple of 4.
 */
void skl_gemm_i8_i8pc_i32_xsfvqdotq(size_t m, size_t n, size_t k, int32_t alpha,
                                    const int8_t *a, size_t rsa,
                                    const int8_t *b_pack, size_t rsb1,
                                    int32_t beta, int32_t *c, size_t rsc);

/**
 * @brief Int8 GEMM tuned for Core Local Port memory on SiFive's X390.
 *
 * @param m - Number of rows in matrix A and matrix C.
 * @param n - Number of columns in matrix B and matrix C.
 * @param k - Number of columns in matrix A and rows in matrix B.
 * @param alpha - Scalar multiplier for A * B product.
 * @param a - Pointer to matrix A in row-major format.
 * @param rsa - Stride between rows of matrix A in elements.
 * @param b_pack - Pointer to packed matrix B_pack.
 * @param rsb1 - Row stride between blocks of packed matrix B_pack in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 *
 * Computes `C = alpha * A * B_pack + beta * C` for int8 matrix A, packed int8
 * matrix B_pack, and int32 output matrix C.
 *
 * When k % 4 == 0, equivalent to scalar call:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_scalar(
 *     1, 1, 4, m, n, k / 4,   // m0, n0, k0, m1, n1, k1
 *     alpha,                  // alpha
 *     a, 0, 1, rsa, 4,        // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 1, 0, rsb1, 4,  // b_pack, rsb0, csb0, rsb1, csb1
 *     beta,                   // beta
 *     c, 0, 0, rsc, 1         // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * If B is a k x n int8 row-major matrix with row stride rsb and b_pack is
 * pre-allocated with size ((k + 3) / 4) * rsb1 bytes, then
 * ```
 * skl_pack_b_i8_xsfvqdotq(k, n, b, rsb, b_pack, rsb1);
 * skl_gemm_i8_i8pc_i32_xsfvqdotq_x390_clp(m, n, k, alpha, a, rsa, b_pack, rsb1,
 *                                         beta, c, rsc);
 * ```
 * is equivalent to scalar call:
 * ```
 * skl_gemm_i8_i32_scalar(m, n, k, alpha, a, rsa, b, rsb, beta, c, rsc);
 * ```
 *
 * This kernel uses the SiFive Xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data. The
 * kernel automatically dispatches to optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * This version of skl_gemm_i8_i8pc_i32_xsfvqdotq has been tuned for
 * improved performance on X390 when all of the following conditions are met:
 * 1) the matrices are allocated in CLP memory,
 * 2) m >= 6, and
 * 3) A is 4-byte-aligned and rsa is a multiple 4.
 * If matrices are not allocated in the CLP address space, please use
 * skl_gemm_i8_i8pc_i32_xsfvqdotq instead.
 *
 * @note
 * Matrix B_pack must be pre-packed using skl_pack_b_i8_xsfvqdotq(). Matrix A is
 * used directly in row-major format without requiring pre-packing. Performance
 * will be best when A is 4-byte-aligned and rsa is a multiple of 4.
 */
void skl_gemm_i8_i8pc_i32_xsfvqdotq_x390_clp(size_t m, size_t n, size_t k,
                                             int32_t alpha, const int8_t *a,
                                             size_t rsa, const int8_t *b_pack,
                                             size_t rsb1, int32_t beta,
                                             int32_t *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
