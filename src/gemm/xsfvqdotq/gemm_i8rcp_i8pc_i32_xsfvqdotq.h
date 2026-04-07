// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#if !defined(__riscv_xsfvqdotq)
#error This source file requires compiler support for the Xsfvqdotq extension.
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Xsfvqdotq int8 GEMM with int32 accumulator.
 *
 * @param m - Number of block-rows in A_pack and rows in C.
 * @param n - Number of block-columns in B_pack and columns in C.
 * @param k - Inner product length in elements.
 * @param alpha - Scalar multiplier for A_pack * B_pack product.
 * @param a_pack - Pointer to packed matrix A_pack (m0 = 1, k0 = 4, csa0 = 1).
 * @param rsa1 - Row stride between blocks of A_pack in elements.
 * @param csa1 - Column stride between blocks of A_pack in elements.
 * @param b_pack - Pointer to packed matrix B_pack (k0 = 4, n0 = 1, rsb0 = 1,
 *                 csb1 = 4).
 * @param rsb1 - Row stride between blocks of B_pack in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 *
 * Computes `C = alpha * A_pack * B_pack + beta * C` for packed int8 matrices
 * A_pack and B_pack and int32 output matrix C.
 *
 * When k % 4 == 0, equivalent to calling:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
 *     1, 1, 4, m, n, k / 4,     // m0, n0, k0, m1, n1, k1
 *     alpha,                    // alpha
 *     a_pack, 0, 1, rsa1, csa1, // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 1, 0, rsb1, 4,    // b_pack, rsb0, csb0, rsb1, csb1
 *     beta,                     // beta
 *     c, 0, 0, rsc, 1           // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * Suppose
 * ```
 * void skl_pack_a_i8_xsfvqdotq(size_t m, size_t k, const int8_t *a, size_t rsa,
 *                              int8_t *a_pack, size_t rsa1, size_t csa1);
 * ```
 * is a specialization of the general packing kernel (csa = 1, m0 = 1, k0 = 4,
 * rsa0 = 0, csa0 = 1) which packs an m x k row-major matrix a into a_pack.
 *
 * If A is an m x k int8 row-major matrix with row stride rsa and a_pack is
 * pre-allocated with size rsa1 >= csa1 ? m * rsa1 : ((k + 3) / 4) * csa1 bytes,
 * and if B is a k x n int8 row-major matrix with row stride rsb and b_pack is
 * pre-allocated with size ((k + 3) / 4) * rsb1 bytes, then
 * ```
 * skl_pack_a_i8_xsfvqdotq(m, k, a, rsa, a_pack, rsa1, csa1);
 * skl_pack_b_i8_xsfvqdotq(k, n, b, rsb, b_pack, rsb1);
 * skl_gemm_i8rcp_i8pc_i32_xsfvqdotq(m, n, k, alpha, a_pack, rsa1, csa1, b_pack,
 *                                   rsb1, beta, c, rsc);
 * ```
 * is equivalent to calling:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
 *     1, 1, 1, m, n, k,    // m0, n0, k0, m1, n1, k1
 *     alpha,               // alpha
 *     a, 0, 0, rsa, 1,     // a, rsa0, csa0, rsa1, csa1
 *     b, 0, 0, rsb, 1,     // b, rsb0, csb0, rsb1, csb1
 *     beta,                // beta
 *     c, 0, 0, rsc, 1      // c, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * This kernel uses the SiFive Xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data. The
 * kernel automatically dispatches to optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * @note
 * If A is in row-major format, it can be used directly without pre-packing by
 * setting rsa1 = rsa and csa1 = 4. Matrix B_pack must be pre-packed using
 * skl_pack_b_i8_xsfvqdotq(). Performance will be best when A_pack is
 * 4-byte-aligned and rsa1 and csa1 are multiples of 4.
 */
void skl_gemm_i8rcp_i8pc_i32_xsfvqdotq(size_t m, size_t n, size_t k,
                                       int32_t alpha, const int8_t *a_pack,
                                       size_t rsa1, size_t csa1,
                                       const int8_t *b_pack, size_t rsb1,
                                       int32_t beta, int32_t *c, size_t rsc);

/**
 * @brief Int8 GEMM tuned for Core Local Port memory on SiFive's X390.
 *
 * @param m - Number of block-rows in A_pack and rows in C.
 * @param n - Number of block-columns in B_pack and columns in C.
 * @param k - Inner product length in elements.
 * @param alpha - Scalar multiplier for A_pack * B_pack product.
 * @param a_pack - Pointer to packed matrix A_pack (m0 = 1, k0 = 4, csa0 = 1).
 * @param rsa1 - Row stride between blocks of A_pack in elements.
 * @param csa1 - Column stride between blocks of A_pack in elements.
 * @param b_pack - Pointer to packed matrix B_pack (k0 = 4, n0 = 1, rsb0 = 1,
 *                 csb1 = 4).
 * @param rsb1 - Row stride between blocks of B_pack in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 *
 * Computes `C = alpha * A_pack * B_pack + beta * C` for packed int8 matrices
 * A_pack and B_pack and int32 output matrix C.
 *
 * When k % 4 == 0, equivalent to calling:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
 *     1, 1, 4, m, n, k / 4,     // m0, n0, k0, m1, n1, k1
 *     alpha,                    // alpha
 *     a_pack, 0, 1, rsa1, csa1, // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 1, 0, rsb1, 4,    // b_pack, rsb0, csb0, rsb1, csb1
 *     beta,                     // beta
 *     c, 0, 0, rsc, 1           // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * Suppose
 * ```
 * void skl_pack_a_i8_xsfvqdotq(size_t m, size_t k, const int8_t *a, size_t rsa,
 *                              int8_t *a_pack, size_t rsa1, size_t csa1);
 * ```
 * is a specialization of the general packing kernel (csa = 1, m0 = 1, k0 = 4,
 * rsa0 = 0, csa0 = 1) which packs an m x k row-major matrix a into a_pack.
 *
 * If A is an m x k int8 row-major matrix with row stride rsa and a_pack is
 * pre-allocated with size rsa1 >= csa1 ? m * rsa1 : ((k + 3) / 4) * csa1 bytes,
 * and if B is a k x n int8 row-major matrix with row stride rsb and b_pack is
 * pre-allocated with size ((k + 3) / 4) * rsb1 bytes, then
 * ```
 * skl_pack_a_i8_xsfvqdotq(m, k, a, rsa, a_pack, rsa1, csa1);
 * skl_pack_b_i8_xsfvqdotq(k, n, b, rsb, b_pack, rsb1);
 * skl_gemm_i8rcp_i8pc_i32_xsfvqdotq_x390_clp(m, n, k, alpha, a_pack, rsa1,
 *                                            csa1, b_pack, rsb1, beta, c, rsc);
 * ```
 * is equivalent to calling:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
 *     1, 1, 1, m, n, k,    // m0, n0, k0, m1, n1, k1
 *     alpha,               // alpha
 *     a, 0, 0, rsa, 1,     // a, rsa0, csa0, rsa1, csa1
 *     b, 0, 0, rsb, 1,     // b, rsb0, csb0, rsb1, csb1
 *     beta,                // beta
 *     c, 0, 0, rsc, 1      // c, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * This kernel uses the SiFive Xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data. The
 * kernel automatically dispatches to optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * This version of skl_gemm_i8rcp_i8pc_i32_xsfvqdotq has been tuned for
 * improved performance on X390 when all of the following conditions are met:
 * 1) the matrices are allocated in CLP memory,
 * 2) m >= 6, and
 * 3) A_pack is 4-byte-aligned and rsa1 and csa1 are multiples of 4.
 * If matrices are not allocated in the CLP address space, please use
 * skl_gemm_i8rcp_i8pc_i32_xsfvqdotq instead.
 *
 * @note
 * If A is in row-major format, it can be used directly without pre-packing by
 * setting rsa1 = rsa and csa1 = 4. Matrix B_pack must be pre-packed using
 * skl_pack_b_i8_xsfvqdotq(). Performance will be best when A_pack is
 * 4-byte-aligned and rsa1 and csa1 are multiples of 4.
 */
void skl_gemm_i8rcp_i8pc_i32_xsfvqdotq_x390_clp(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a_pack,
    size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta,
    int32_t *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
