// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "skl-common.h"
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Reference float16 matrix-matrix multiplication (HGEMM).
 *
 * @param m - Number of rows in matrices A and C.
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scalar multiplier for A * B product.
 * @param a - Pointer to matrix A.
 * @param rsa - Row stride of matrix A in elements.
 * @param csa - Column stride of matrix A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param csb - Column stride of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C.
 * @param rsc - Row stride of matrix C in elements.
 * @param csc - Column stride of matrix C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for matrices A, B, and C.
 * This generic GEMM function defines the semantics of all optimized FP16 GEMM
 * kernels in SKL.
 *
 * Matrices may be in row-major or column-major order, depending on the strides.
 * For row-major matrices, the column stride is 1, and the row stride is the
 * leading dimension. For column-major matrices, the reverse is true.
 *
 * @note
 * This function is for API documentation purposes only, and should not be used
 * for performance applications.
 */
void skl_gemm_f16rc_f16rc_f16rc_ref(size_t m, size_t n, size_t k,
                                    _Float16 alpha, const _Float16 *a,
                                    size_t rsa, size_t csa, const _Float16 *b,
                                    size_t rsb, size_t csb, _Float16 beta,
                                    _Float16 *c, size_t rsc, size_t csc);

#if defined(__cplusplus)
} // extern "C"
#endif
