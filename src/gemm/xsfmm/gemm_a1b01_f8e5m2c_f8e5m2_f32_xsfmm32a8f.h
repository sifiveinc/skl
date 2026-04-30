// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#pragma once

#if !defined(__riscv_xsfmm32a8f)
#error This file requires the Xsfmm32a8f extension
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Xsfmm OFP8 E5M2 A * B matrix-matrix multiplication with float32 output
 * for column-major A and row-major B.
 *
 * @param m - Number of rows in matrices A and C.
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param a - Pointer to matrix A.
 * @param csa - Column stride of matrix A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param c - Pointer to matrix C.
 * @param rsc - Row stride of matrix C in elements.
 * @param accum - Determines if output matrix is incremented or overwritten.
 *
 * Computes `C = A * B` (if `accum == false`) or `C += A * B` (if `accum ==
 * true`) for OFP8 E5M2 column-major matrix A, OFP8 E5M2 row-major matrix B, and
 * FP32 row-major matrix C. The entries of A and B are type-punned as 8-bit
 * unsigned integers.
 *
 * Equivalent to calling:
 * ```
 * skl_gemm_f8e5m2rc_f8e5m2rc_f32rc_ref(
 *     m, n, k,       // m, n, k
 *     1,             // alpha
 *     a, 1, csa,     // a, rsa, csa
 *     b, rsb, 1,     // b, rsb, csb
 *     accum ? 1 : 0, // beta
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
void skl_gemm_a1b01_f8e5m2c_f8e5m2_f32_xsfmm32a8f(size_t m, size_t n, size_t k,
                                                  const uint8_t *a, size_t csa,
                                                  const uint8_t *b, size_t rsb,
                                                  float *c, size_t rsc,
                                                  bool accum);

/**
 * @brief Xsfmm OFP8 E5M2 A * B packed matrix-matrix multiplication with float32
 * output.
 *
 * @param m1 - Number of rows in A and C as block matrices.
 * @param n1 - Number of columns in B and C as block matrices.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param a_pack - Pointer to matrix A.
 * @param rsa1 - Row stride between blocks of A in elements.
 * @param b_pack - Pointer to matrix B.
 * @param csb1 - Column stride between blocks of B in elements.
 * @param c_pack - Pointer to matrix C.
 * @param rsc1 - Row stride between blocks of C in elements.
 * @param csc1 - Column stride between blocks of C in elements.
 * @param accum - Determines if output matrix is incremented or overwritten.
 *
 * Computes `C = A * B` (if `accum == false`) or `C += A * B` (if `accum ==
 * true`) for packed OFP8 E5M2 matrices A and B and packed FP32 matrix C. The
 * entries of A and B are type-punned as 8-bit unsigned integers.
 *
 * Equivalent to calling:
 * ```
 * skl_gemm_f8e5m2rcprc_f8e5m2rcprc_f32rcprc_ref(
 *     TE, TE, 1, m1, n1, k,     // m0, n0, k0, m1, n1, k1
 *     1,                        // alpha
 *     a_pack, 1, 0, rsa1, TE,   // a_pack, rsa0, csa0, rsa1, csa1
 *     b_pack, 0, 1, TE, csb1,   // b_pack, rsb0, csb0, rsb1, csb1
 *     accum ? 1 : 0,            // beta
 *     c_pack, TE, 1, rsc1, csc1 // c_pack, rsc0, csc0, rsc1, csc1
 * );
 * ```
 */
void skl_gemm_a1b01_f8e5m2pc_f8e5m2cp_f32rcp_xsfmm32a8f(
    size_t m1, size_t n1, size_t k, const uint8_t *a_pack, size_t rsa1,
    const uint8_t *b_pack, size_t csb1, float *c_pack, size_t rsc1, size_t csc1,
    bool accum);

#if defined(__cplusplus)
} // extern "C"
#endif
