// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

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
 * @param m - Number of rows in A_pack and C.
 * @param n - Number of columns in B_pack and C.
 * @param k1 - Number of block-columns in A_pack, block-rows in B_pack.
 * @param alpha - Scalar multiplier for A_pack * B_pack.
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
 * Equivalent to:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
 *     1, 1, 4, m, n, k1,        // m0, n0, k0, m1, n1, k1
 *     alpha,                    // alpha
 *     a_pack, 0, 1, rsa1, csa1, // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 1, 0, rsb1, 4,    // b_pack, rsb0, csb0, rsb1, csb1
 *     beta,                     // beta
 *     c, 0, 0, rsc, 1           // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * This kernel uses the SiFive Xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data.
 * When A_pack satisfies certain alignment requirements (it is 4-byte aligned
 * and rsa1 and csa1 are multiples of 4), the kernel automatically dispatches to
 * optimized internal implementations:
 * - For m=1: uses an internal GEMV kernel
 * - For m>1: uses tiled GEMM kernels
 *
 * @note
 * If A_pack does not meet the alignment requirements stated above, the kernel
 * falls back to an unaligned 1xm4 implementation. This implementation is
 * unlikely to get good performance since it must handle misaligned loads from
 * A_pack.
 *
 * @note
 * The simplest way to pack a row-major A matrix into an A_pack that meets the
 * alignment requirements is to copy it into a 4-byte-aligned buffer, but insert
 * padding between the rows so that the row stride is a multiple of 4. Then the
 * kernel can be called by setting rsa1 to the row stride and csa1 = 4.
 */
void skl_gemm_i8rcp_i8pc_i32_xsfvqdotq(size_t m, size_t n, size_t k1,
                                       int32_t alpha, const int8_t *a_pack,
                                       size_t rsa1, size_t csa1,
                                       const int8_t *b_pack, size_t rsb1,
                                       int32_t beta, int32_t *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
