// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve64d)
#error This file requires the Zve64d extension
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

/**
 * @brief RVV float64 matrix-matrix multiplication (DGEMM) for row-major
 * matrices.
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
 * Computes `C = alpha * A * B + beta * C` for FP64 row-major matrices.
 *
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f64rc_f64rc_f64rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 * Uses a 4 x LMUL=4 x 1 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e64m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_f64_f64_f64_zve64d(
    size_t m, size_t n, size_t k, double alpha, const double *a, size_t rsa,
    const double *b, size_t rsb, double beta, double *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;

  double a00;
  double a10;
  double a20;
  double a30;
  vfloat64m4_t b00;
  vfloat64m4_t acc00;
  vfloat64m4_t acc10;
  vfloat64m4_t acc20;
  vfloat64m4_t acc30;
  vfloat64m4_t c00;
  vfloat64m4_t c10;
  vfloat64m4_t c20;
  vfloat64m4_t c30;

  double a0;
  vfloat64m8_t b0;
  vfloat64m8_t acc0;
  vfloat64m8_t c0;

  if (k == 0) {
    for (ii = 0; ii < m; ii++) {
      for (jj = 0; jj < n; jj += jj_vl) {
        jj_vl = __riscv_vsetvl_e64m8(n - jj);
        c0 = __riscv_vle64_v_f64m8(c + (ii + 0) * rsc + jj, jj_vl);
        c0 = __riscv_vfmul_vf_f64m8(c0, beta, jj_vl);
        __riscv_vse64_v_f64m8(c + (ii + 0) * rsc + jj, c0, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 4) <= m; ii += 4) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e64m4(n - jj);

      a00 = a[(ii + 0) * rsa + 0];
      a10 = a[(ii + 1) * rsa + 0];
      a20 = a[(ii + 2) * rsa + 0];
      a30 = a[(ii + 3) * rsa + 0];
      b00 = __riscv_vle64_v_f64m4(b + 0 * rsb + jj, jj_vl);
      acc00 = __riscv_vfmul_vf_f64m4(b00, a00, jj_vl);
      acc10 = __riscv_vfmul_vf_f64m4(b00, a10, jj_vl);
      acc20 = __riscv_vfmul_vf_f64m4(b00, a20, jj_vl);
      acc30 = __riscv_vfmul_vf_f64m4(b00, a30, jj_vl);

      for (kk = 1; kk < k; kk++) {
        a00 = a[(ii + 0) * rsa + kk];
        a10 = a[(ii + 1) * rsa + kk];
        a20 = a[(ii + 2) * rsa + kk];
        a30 = a[(ii + 3) * rsa + kk];
        b00 = __riscv_vle64_v_f64m4(b + kk * rsb + jj, jj_vl);
        acc00 = __riscv_vfmacc_vf_f64m4(acc00, a00, b00, jj_vl);
        acc10 = __riscv_vfmacc_vf_f64m4(acc10, a10, b00, jj_vl);
        acc20 = __riscv_vfmacc_vf_f64m4(acc20, a20, b00, jj_vl);
        acc30 = __riscv_vfmacc_vf_f64m4(acc30, a30, b00, jj_vl);
      }

      c00 = __riscv_vle64_v_f64m4(c + (ii + 0) * rsc + jj, jj_vl);
      c00 = __riscv_vfmul_vf_f64m4(c00, beta, jj_vl);
      c00 = __riscv_vfmacc_vf_f64m4(c00, alpha, acc00, jj_vl);
      __riscv_vse64_v_f64m4(c + (ii + 0) * rsc + jj, c00, jj_vl);

      c10 = __riscv_vle64_v_f64m4(c + (ii + 1) * rsc + jj, jj_vl);
      c10 = __riscv_vfmul_vf_f64m4(c10, beta, jj_vl);
      c10 = __riscv_vfmacc_vf_f64m4(c10, alpha, acc10, jj_vl);
      __riscv_vse64_v_f64m4(c + (ii + 1) * rsc + jj, c10, jj_vl);

      c20 = __riscv_vle64_v_f64m4(c + (ii + 2) * rsc + jj, jj_vl);
      c20 = __riscv_vfmul_vf_f64m4(c20, beta, jj_vl);
      c20 = __riscv_vfmacc_vf_f64m4(c20, alpha, acc20, jj_vl);
      __riscv_vse64_v_f64m4(c + (ii + 2) * rsc + jj, c20, jj_vl);

      c30 = __riscv_vle64_v_f64m4(c + (ii + 3) * rsc + jj, jj_vl);
      c30 = __riscv_vfmul_vf_f64m4(c30, beta, jj_vl);
      c30 = __riscv_vfmacc_vf_f64m4(c30, alpha, acc30, jj_vl);
      __riscv_vse64_v_f64m4(c + (ii + 3) * rsc + jj, c30, jj_vl);
    }
  }

  for (; ii < m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e64m8(n - jj);

      a0 = a[(ii + 0) * rsa + 0];
      b0 = __riscv_vle64_v_f64m8(b + 0 * rsb + jj, jj_vl);
      acc0 = __riscv_vfmul_vf_f64m8(b0, a0, jj_vl);

      for (kk = 1; kk < k; kk++) {
        a0 = a[(ii + 0) * rsa + kk];
        b0 = __riscv_vle64_v_f64m8(b + kk * rsb + jj, jj_vl);
        acc0 = __riscv_vfmacc_vf_f64m8(acc0, a0, b0, jj_vl);
      }

      c0 = __riscv_vle64_v_f64m8(c + (ii + 0) * rsc + jj, jj_vl);
      c0 = __riscv_vfmul_vf_f64m8(c0, beta, jj_vl);
      c0 = __riscv_vfmacc_vf_f64m8(c0, alpha, acc0, jj_vl);
      __riscv_vse64_v_f64m8(c + (ii + 0) * rsc + jj, c0, jj_vl);
    }
  }
}

SKL_FUNC void skl_gemm_f64_f64_f64_zve64d(size_t m, size_t n, size_t k,
                                          double alpha, const double *a,
                                          size_t rsa, const double *b,
                                          size_t rsb, double beta, double *c,
                                          size_t rsc) {
  skl_gemm_4xm4x1_f64_f64_f64_zve64d(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                     rsc);
}
