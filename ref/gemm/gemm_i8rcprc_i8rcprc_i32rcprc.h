// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "skl-common.h"
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference int8 packed matrix-matrix multiplication with int32
 * accumulator.
 *
 * @param m0 - Number of rows in each block of matrices A and C.
 * @param n0 - Number of columns in each block of matrices B and C.
 * @param k0 - Number of columns in each block of A and rows in each block of B.
 * @param m1 - Number of rows in A and C as block matrices.
 * @param n1 - Number of columns in B and C as block matrices.
 * @param k1 - Number of columns in A and rows in B as block matrices.
 * @param alpha - Scalar multiplier for A * B product.
 * @param a_pack - Pointer to matrix A.
 * @param rsa0 - Row stride within each block of A in elements.
 * @param csa0 - Column stride within each block of A in elements.
 * @param rsa1 - Row stride between blocks of A in elements.
 * @param csa1 - Column stride between blocks of A in elements.
 * @param b_pack - Pointer to matrix B.
 * @param rsb0 - Row stride within each block of B in elements.
 * @param csb0 - Column stride within each block of B in elements.
 * @param rsb1 - Row stride between blocks of B in elements.
 * @param csb1 - Column stride between blocks of B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c_pack - Pointer to matrix C.
 * @param rsc0 - Row stride within each block of C in elements.
 * @param csc0 - Column stride within each block of C in elements.
 * @param rsc1 - Row stride between blocks of C in elements.
 * @param csc1 - Column stride between blocks of C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for packed matrices A, B, and C.
 * This generic GEMM function defines the semantics of all optimized int8 packed
 * GEMM kernels with int32 accumulators in SKL.
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    int32_t alpha, const int8_t *a_pack, size_t rsa0, size_t csa0, size_t rsa1,
    size_t csa1, const int8_t *b_pack, size_t rsb0, size_t csb0, size_t rsb1,
    size_t csb1, int32_t beta, int32_t *c_pack, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1);

#if defined(__cplusplus)
} // extern "C"
#endif
