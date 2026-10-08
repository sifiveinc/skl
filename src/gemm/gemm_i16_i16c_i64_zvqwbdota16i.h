// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_zvqwbdota16i)
#error This source file requires compiler support for the Zvqwbdota16i extension.
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Zvqwbdota16i quad-widening int16 GEMM.
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
 * Computes `C = alpha * A * B + beta * C` for int16 row-major matrix A, int16
 * column-major matrix B, and int64 row-major matrix C.
 *
 * Equivalent to:
 * ```
 * skl_gemm_i16rcprc_i16rcprc_i64rcprc_ref(
 *     1, 1, 1,         // m0, n0, k0
 *     m, n, k,         // m1, n1, k1
 *     alpha,           // alpha
 *     a, 1, 1, rsa, 1, // a, rsa0, csa0, rsa1, csa1
 *     b, 1, 1, 1, csb, // b, rsb0, csb0, rsb1, csb1
 *     beta,            // beta
 *     c, 1, 1, rsc, 1  // c, rsc0, csc0, rsc1, csc1
 * );
 * ```
 *
 * This kernel uses the Zvqwbdota16i extension for vector quad widening batched
 * dot product operations to achieve high performance on 16-bit integer data.
 */
void skl_gemm_i16_i16c_i64_zvqwbdota16i(size_t m, size_t n, size_t k,
                                        int64_t alpha, const int16_t *a,
                                        size_t rsa, const int16_t *b,
                                        size_t csb, int64_t beta, int64_t *c,
                                        size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
