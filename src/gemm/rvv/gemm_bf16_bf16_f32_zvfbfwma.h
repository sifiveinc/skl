// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(__riscv_zvfbfwma) || __riscv_zvfbfwma < 1000000
#error This file requires the RISC-V Zvfbfwma extension, version 1000000.
#endif

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief RVV bfloat16 matrix-matrix multiplication with float32 output for
 * row-major matrices.
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
 * Computes `C = alpha * A * B + beta * C` for BF16 row-major matrices A and B
 * and FP32 row-major output matrix C.
 *
 * Functionally equivalent to scalar call:
 * ```
 * skl_gemm_bf16rc_bf16rc_f32rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 */
void skl_gemm_bf16_bf16_f32_zvfbfwma(size_t m, size_t n, size_t k, float alpha,
                                     const __bf16 *a, size_t rsa,
                                     const __bf16 *b, size_t rsb, float beta,
                                     float *c, size_t rsc);

#if defined(__cplusplus)
} // extern "C"
#endif
