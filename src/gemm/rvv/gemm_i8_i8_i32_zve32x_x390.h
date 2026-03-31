// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zve32x) || __riscv_zve32x < 1000000
#error This file requires the RISC-V Zve32x extension, version 1000000.
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV int8 matrix-matrix multiplication with int32 output for row-major
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
 * Computes `C = alpha * A * B + beta * C` for int8 row-major matrices A and B
 * and int32 output matrix C.
 *
 * Functionally equivalent to scalar call:
 * ```
 * skl_gemm_i8rc_i8rc_i32rc_scalar(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 */
void skl_gemm_i8_i8_i32_zve32x_x390(size_t m, size_t n, size_t k, int32_t alpha,
                                    const int8_t *a, size_t rsa,
                                    const int8_t *b, size_t rsb, int32_t beta,
                                    int32_t *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
