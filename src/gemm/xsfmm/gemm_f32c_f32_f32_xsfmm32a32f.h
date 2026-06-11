// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfmm32a32f)
#error This file requires the Xsfmm32a32f extension
#endif

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Xsfmm float32 A * B matrix-matrix multiplication (SGEMM) for
 * column-major A and row-major B.
 *
 * @param m - Number of rows in matrices A and C.
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scaling factor for A * B.
 * @param a - Pointer to matrix A.
 * @param csa - Column stride of matrix A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param beta - Scaling factor for C.
 * @param c - Pointer to matrix C.
 * @param rsc - Row stride of matrix C in elements.
 * @param accum - Determines if output matrix is incremented or overwritten.
 *
 * Computes `C = alpha * A * B + beta * C` for float32 column-major matrix A and
 * float32 row-major matrices B and C.
 *
 * Equivalent to calling:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_ref(
 *     m, n, k,       // m, n, k
 *     alpha,         // alpha
 *     a, 1, csa,     // a, rsa, csa
 *     b, rsb, 1,     // b, rsb, csb
 *     beta,          // beta
 *     c, rsc, 1      // c, rsc, csc
 * );
 * ```
 *
 * @note
 * In memory, a column-major matrix A with column stride `csa` is identical to
 * its matrix transpose A^T stored in row-major order with row stride `csa`. So,
 * to compute A * B for row-major A and B, callers should first compute A^T in
 * row-major order, and then call this function with A^T, using its row stride
 * as `csa`.
 **/
void skl_gemm_f32c_f32_f32_xsfmm32a32f(size_t m, size_t n, size_t k,
                                       float alpha, const float *a, size_t csa,
                                       const float *b, size_t rsb, float beta,
                                       float *c, size_t rsc);

/**
 * @brief Xsfmm float32 A * B packed matrix-matrix multiplication.
 *
 * @param m1 - Number of block-rows in A and C.
 * @param n1 - Number of block-columns in B and C.
 * @param k - Number of columns in A and rows in B.
 * @param alpha - Scaling factor for A * B.
 * @param a_pack - Pointer to matrix A.
 * @param rsa1 - Row stride between blocks of A in elements.
 * @param b_pack - Pointer to matrix B.
 * @param csb1 - Column stride between blocks of B in elements.
 * @param beta - Scaling factor for C.
 * @param c_pack - Pointer to matrix C.
 * @param rsc1 - Row stride between blocks of C in elements.
 * @param csc1 - Column stride between blocks of C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for packed float32 matrices A, B, and
 * C.
 *
 * Equivalent to calling:
 * ```
 * skl_gemm_f32rcprc_f32rcprc_f32rcprc_ref(
 *     ETE, ETE, 1, m1, n1, k,    // m0, n0, k0, m1, n1, k1
 *     alpha,                     // alpha
 *     a_pack, 1, 0, rsa1, ETE,   // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 0, 1, ETE, csb1,   // b_pack, rsb0, csb0, rsb1, csb1
 *     beta,                      // beta
 *     c_pack, ETE, 1, rsc1, csc1 // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 */
void skl_gemm_f32rcptex1c_f32rcp1xte_f32rcptexte_xsfmm32a32f(
    size_t m1, size_t n1, size_t k, float alpha, const float *a, size_t rsa1,
    size_t csa1, const float *b, size_t rsb1, size_t csb1, float beta, float *c,
    size_t rsc0, size_t rsc1, size_t csc1);

#if defined(__cplusplus)
} // extern "C"
#endif
