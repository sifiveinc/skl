// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zvfh) || __riscv_zvfh < 1000000
#error This file requires the RISC-V Zvfh extension, version 1000000.
#endif

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV float16 matrix-matrix multiplication with float32 output for
 * row-major matrices, tuned for X390.
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
 * Computes `C = alpha * A * B + beta * C` for FP16 row-major matrices A and B
 * and FP32 row-major output matrix C.
 *
 * Functionally equivalent to scalar call:
 * ```
 * skl_gemm_f16rc_f16rc_f32rc_scalar(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 */
void skl_gemm_f16_f16_f32_zvfh_x390(size_t m, size_t n, size_t k, float alpha,
                                    const _Float16 *a, size_t rsa,
                                    const _Float16 *b, size_t rsb, float beta,
                                    float *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
