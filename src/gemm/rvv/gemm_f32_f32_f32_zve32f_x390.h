// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV float32 matrix-matrix multiplication (SGEMM) for row-major
 * matrices, tuned for X390.
 *
 * @param m - Number of rows in matrices A and C.
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scalar multiplier for A*B product.
 * @param a - Pointer to matrix A.
 * @param rsa - Row stride of matrix A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C.
 * @param rsc - Row stride of matrix C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for FP32 row-major matrices.
 *
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 */
void skl_gemm_f32_f32_f32_zve32f_x390(size_t m, size_t n, size_t k, float alpha,
                                      const float *a, size_t rsa,
                                      const float *b, size_t rsb, float beta,
                                      float *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
