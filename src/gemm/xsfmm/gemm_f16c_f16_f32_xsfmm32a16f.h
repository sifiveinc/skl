// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__riscv_xsfmm32a16f)
#error This file requires the Xsfmm32a16f extension
#endif

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Xsfmm widening float16 GEMM.
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
 * Computes `C = alpha * A * B + beta * C` for float16 column-major matrix A,
 * float16 row-major matrix B, and float32 row-major matrix C.
 *
 * Equivalent to calling:
 * ```
 * skl_gemm_f16rc_f16rc_f32rc_ref(
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
void skl_gemm_f16c_f16_f32_xsfmm32a16f(size_t m, size_t n, size_t k,
                                       float alpha, const _Float16 *a,
                                       size_t csa, const _Float16 *b,
                                       size_t rsb, float beta, float *c,
                                       size_t rsc);

/**
 * @brief Xsfmm packed widening float16 GEMM.
 *
 * @param m1 - Number of block-rows in A and C.
 * @param n1 - Number of block-columns in B and C.
 * @param k - Number of columns in A and rows in B.
 * @param alpha - Scaling factor for A * B.
 * @param a - Pointer to packed matrix A.
 * @param rsa1 - Row stride between blocks of A in elements.
 * @param csa1 - Column stride between blocks of A in elements.
 * @param b - Pointer to packed matrix B.
 * @param rsb1 - Row stride between blocks of B in elements.
 * @param csb1 - Column stride between blocks of B in elements.
 * @param beta - Scaling factor for C.
 * @param c - Pointer to packed matrix C.
 * @param rsc0 - Row stride within a block of C in elements.
 * @param csc0 - Column stride within a block of C in elements.
 * @param rsc1 - Row stride between blocks of C in elements.
 * @param csc1 - Column stride between blocks of C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for packed float16 matrices A and B,
 * and packed float32 matrix C.
 *
 * Equivalent to calling:
 * ```
 * skl_gemm_f16rcprc_f16rcprc_f32rcprc_ref(
 *     ETE, ETE, 1, m1, n1, k,   // m0, n0, k0, m1, n1, k1
 *     alpha,                    // alpha
 *     a, 1, 0, rsa1, csa1,      // a, rsa0, csa0, rsa1, csa1
 *     b, 0, 1, rsb1, csb1,      // b, rsb0, csb0, rsb1, csb1
 *     beta,                     // beta
 *     c, rsc0, csc0, rsc1, csc1 // c, rsc0, csc0, rsc1, csc1
 * );
 * ```
 */
void skl_gemm_f16rcptex1c_f16rcp1xte_f32rcptexterc_xsfmm32a16f(
    size_t m1, size_t n1, size_t k, float alpha, const _Float16 *a, size_t rsa1,
    size_t csa1, const _Float16 *b, size_t rsb1, size_t csb1, float beta,
    float *c, size_t rsc0, size_t csc0, size_t rsc1, size_t csc1);

#if defined(__cplusplus)
} // extern "C"
#endif
