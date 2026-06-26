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
 * @param m - Number of rows in A and C.
 * @param n - Number of columns in B and C.
 * @param k1 - Number of block-columns in A, block-rows in B.
 * @param alpha - Scalar multiplier for A * B.
 * @param a - Pointer to packed matrix A (m0 = 1, k0 = 4, csa0 = 1).
 * @param rsa1 - Row stride between blocks of A in elements.
 * @param csa1 - Column stride between blocks of A in elements.
 * @param b - Pointer to packed matrix B (k0 = 4, n0 = 1, rsb0 = 1,
 *            csb1 = 4).
 * @param rsb1 - Row stride between blocks of B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for packed int8 matrices
 * A and B and int32 output matrix C.
 *
 * Equivalent to:
 * ```
 * skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
 *     1, 1, 4, m, n, k1,   // m0, n0, k0, m1, n1, k1
 *     alpha,               // alpha
 *     a, 0, 1, rsa1, csa1, // a, rsa0, csa0, rsa1, csa1
 *     b, 1, 0, rsb1, 4,    // b, rsb0, csb0, rsb1, csb1
 *     beta,                // beta
 *     c, 0, 0, rsc, 1      // c, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * This kernel uses the SiFive Xsfvqdotq extension for vector quad widening 4D
 * dot product operations to achieve high performance on 8-bit integer data.
 * Works best when A is 4-byte aligned and rsa1 and csa1 are multiples of 4.
 *
 * @note
 * If A does not meet the alignment requirements stated above, the kernel
 * falls back to an unaligned implementation that is unlikely to get good
 * performance since it must handle misaligned loads from A.
 *
 * @note
 * The simplest way to pack a row-major A matrix into an A that meets the
 * alignment requirements is to copy it into a 4-byte-aligned buffer, but insert
 * padding between the rows so that the row stride is a multiple of 4. Then the
 * kernel can be called by setting rsa1 to the row stride and csa1 = 4.
 */
void skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(size_t m, size_t n, size_t k1,
                                             int32_t alpha, const int8_t *a,
                                             size_t rsa1, size_t csa1,
                                             const int8_t *b, size_t rsb1,
                                             int32_t beta, int32_t *c,
                                             size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
