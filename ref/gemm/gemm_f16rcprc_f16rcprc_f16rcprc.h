// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "skl-common.h"
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference packed float16 GEMM.
 *
 * @param m0 - Number of rows in each block of matrices A and C.
 * @param n0 - Number of columns in each block of matrices B and C.
 * @param k0 - Number of columns in each block of A and rows in each block of B.
 * @param m1 - Number of rows in A and C as block matrices.
 * @param n1 - Number of columns in B and C as block matrices.
 * @param k1 - Number of columns in A and rows in B as block matrices.
 * @param alpha - Scalar multiplier for A * B product.
 * @param a - Pointer to matrix A.
 * @param rsa0 - Row stride within each block of A in elements.
 * @param csa0 - Column stride within each block of A in elements.
 * @param rsa1 - Row stride between blocks of A in elements.
 * @param csa1 - Column stride between blocks of A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb0 - Row stride within each block of B in elements.
 * @param csb0 - Column stride within each block of B in elements.
 * @param rsb1 - Row stride between blocks of B in elements.
 * @param csb1 - Column stride between blocks of B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C.
 * @param rsc0 - Row stride within each block of C in elements.
 * @param csc0 - Column stride within each block of C in elements.
 * @param rsc1 - Row stride between blocks of C in elements.
 * @param csc1 - Column stride between blocks of C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for packed matrices A, B, and C.
 * This generic GEMM function defines the semantics of all optimized packed
 * float16 GEMM kernels in SKL.
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_gemm_f16rcprc_f16rcprc_f16rcprc_ref(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    _Float16 alpha, const _Float16 *a, size_t rsa0, size_t csa0, size_t rsa1,
    size_t csa1, const _Float16 *b, size_t rsb0, size_t csb0, size_t rsb1,
    size_t csb1, _Float16 beta, _Float16 *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1);

#if defined(__cplusplus)
} // extern "C"
#endif
