// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zvqwbdota8i)
#error This source file requires compiler support for the Zvqwbdota8i extension.
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Zvqwbdota8i quad-widening int8 GEMM.
 *
 * @param m - Number of rows in A and C.
 * @param n - Number of columns in B and C.
 * @param k - Number of columns in A, rows in B.
 * @param alpha - Scalar multiplier for A * B.
 * @param a - Pointer to matrix A in row-major format.
 * @param rsa - Stride between rows of matrix A in elements.
 * @param b - Pointer to matrix B in column-major format.
 * @param csb - Stride between columns of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C in row-major format.
 * @param rsc - Stride between rows of matrix C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for int8 row-major matrix A, int8
 * column-major matrix B, and int32 row-major matrix C.
 *
 * Equivalent to:
 * ```
 * skl_gemm_i8rc_i8rc_i32rc_ref(
 *     m, n, k,   // m, n, k
 *     alpha,     // alpha
 *     a, rsa, 1, // a, rsa, csa
 *     b, 1, csb, // b, rsb, csb
 *     beta,      // beta
 *     c, rsc, 1  // c, rsc, csc
 * );
 * ```
 *
 * This kernel uses the Zvqwbdota8i extension for vector quad widening batched
 * dot product operations to achieve high performance on 8-bit integer data.
 */
void skl_gemm_i8_i8c_i32_zvqwbdota8i(size_t m, size_t n, size_t k,
                                     int32_t alpha, const int8_t *a, size_t rsa,
                                     const int8_t *b, size_t csb, int32_t beta,
                                     int32_t *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
