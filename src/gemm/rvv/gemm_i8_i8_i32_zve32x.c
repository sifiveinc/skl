// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

/**
 * @brief RVV int8 matrix-matrix multiplication with int32 output for row-major
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
 * Computes `C = alpha * A * B + beta * C` for int8 row-major matrices A and B
 * and int32 output matrix C.
 *
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_i8rc_i8rc_i32rc_ref(
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
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e32m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_i8_i8_i32_zve32x(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a, size_t rsa,
    const int8_t *b, size_t rsb, int32_t beta, int32_t *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;

  int8_t a00;
  int8_t a10;
  int8_t a20;
  int8_t a30;
  vint8m1_t b00;
  vint16m2_t b00w;
  vint32m4_t acc00;
  vint32m4_t acc10;
  vint32m4_t acc20;
  vint32m4_t acc30;
  vint32m4_t c00;
  vint32m4_t c10;
  vint32m4_t c20;
  vint32m4_t c30;

  int8_t a0;
  vint8m2_t b0;
  vint16m4_t b0w;
  vint32m8_t acc0;
  vint32m8_t c0;

  if (k == 0) {
    for (ii = 0; ii < m; ii++) {
      for (jj = 0; jj < n; jj += jj_vl) {
        jj_vl = __riscv_vsetvl_e32m8(n - jj);
        c0 = __riscv_vle32_v_i32m8(c + (ii + 0) * rsc + jj, jj_vl);
        c0 = __riscv_vmul_vx_i32m8(c0, beta, jj_vl);
        __riscv_vse32_v_i32m8(c + (ii + 0) * rsc + jj, c0, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 4) <= m; ii += 4) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e32m4(n - jj);

      a00 = a[(ii + 0) * rsa + 0];
      a10 = a[(ii + 1) * rsa + 0];
      a20 = a[(ii + 2) * rsa + 0];
      a30 = a[(ii + 3) * rsa + 0];
      b00 = __riscv_vle8_v_i8m1(b + 0 * rsb + jj, jj_vl);
      b00w = __riscv_vwcvt_x_x_v_i16m2(b00, jj_vl);
      acc00 = __riscv_vwmul_vx_i32m4(b00w, (int16_t)a00, jj_vl);
      acc10 = __riscv_vwmul_vx_i32m4(b00w, (int16_t)a10, jj_vl);
      acc20 = __riscv_vwmul_vx_i32m4(b00w, (int16_t)a20, jj_vl);
      acc30 = __riscv_vwmul_vx_i32m4(b00w, (int16_t)a30, jj_vl);

      for (kk = 1; kk < k; kk++) {
        a00 = a[(ii + 0) * rsa + kk];
        a10 = a[(ii + 1) * rsa + kk];
        a20 = a[(ii + 2) * rsa + kk];
        a30 = a[(ii + 3) * rsa + kk];
        b00 = __riscv_vle8_v_i8m1(b + kk * rsb + jj, jj_vl);
        b00w = __riscv_vwcvt_x_x_v_i16m2(b00, jj_vl);
        acc00 = __riscv_vwmacc_vx_i32m4(acc00, (int16_t)a00, b00w, jj_vl);
        acc10 = __riscv_vwmacc_vx_i32m4(acc10, (int16_t)a10, b00w, jj_vl);
        acc20 = __riscv_vwmacc_vx_i32m4(acc20, (int16_t)a20, b00w, jj_vl);
        acc30 = __riscv_vwmacc_vx_i32m4(acc30, (int16_t)a30, b00w, jj_vl);
      }

      c00 = __riscv_vle32_v_i32m4(c + (ii + 0) * rsc + jj, jj_vl);
      c00 = __riscv_vmul_vx_i32m4(c00, beta, jj_vl);
      c00 = __riscv_vmacc_vx_i32m4(c00, alpha, acc00, jj_vl);
      __riscv_vse32_v_i32m4(c + (ii + 0) * rsc + jj, c00, jj_vl);

      c10 = __riscv_vle32_v_i32m4(c + (ii + 1) * rsc + jj, jj_vl);
      c10 = __riscv_vmul_vx_i32m4(c10, beta, jj_vl);
      c10 = __riscv_vmacc_vx_i32m4(c10, alpha, acc10, jj_vl);
      __riscv_vse32_v_i32m4(c + (ii + 1) * rsc + jj, c10, jj_vl);

      c20 = __riscv_vle32_v_i32m4(c + (ii + 2) * rsc + jj, jj_vl);
      c20 = __riscv_vmul_vx_i32m4(c20, beta, jj_vl);
      c20 = __riscv_vmacc_vx_i32m4(c20, alpha, acc20, jj_vl);
      __riscv_vse32_v_i32m4(c + (ii + 2) * rsc + jj, c20, jj_vl);

      c30 = __riscv_vle32_v_i32m4(c + (ii + 3) * rsc + jj, jj_vl);
      c30 = __riscv_vmul_vx_i32m4(c30, beta, jj_vl);
      c30 = __riscv_vmacc_vx_i32m4(c30, alpha, acc30, jj_vl);
      __riscv_vse32_v_i32m4(c + (ii + 3) * rsc + jj, c30, jj_vl);
    }
  }

  for (; ii < m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e32m8(n - jj);

      a0 = a[(ii + 0) * rsa + 0];
      b0 = __riscv_vle8_v_i8m2(b + 0 * rsb + jj, jj_vl);
      b0w = __riscv_vwcvt_x_x_v_i16m4(b0, jj_vl);
      acc0 = __riscv_vwmul_vx_i32m8(b0w, (int16_t)a0, jj_vl);

      for (kk = 1; kk < k; kk++) {
        a0 = a[(ii + 0) * rsa + kk];
        b0 = __riscv_vle8_v_i8m2(b + kk * rsb + jj, jj_vl);
        b0w = __riscv_vwcvt_x_x_v_i16m4(b0, jj_vl);
        acc0 = __riscv_vwmacc_vx_i32m8(acc0, (int16_t)a0, b0w, jj_vl);
      }

      c0 = __riscv_vle32_v_i32m8(c + (ii + 0) * rsc + jj, jj_vl);
      c0 = __riscv_vmul_vx_i32m8(c0, beta, jj_vl);
      c0 = __riscv_vmacc_vx_i32m8(c0, alpha, acc0, jj_vl);
      __riscv_vse32_v_i32m8(c + (ii + 0) * rsc + jj, c0, jj_vl);
    }
  }
}

SKL_FUNC void skl_gemm_i8_i8_i32_zve32x(size_t m, size_t n, size_t k,
                                        int32_t alpha, const int8_t *a,
                                        size_t rsa, const int8_t *b, size_t rsb,
                                        int32_t beta, int32_t *c, size_t rsc) {
  skl_gemm_4xm4x1_i8_i8_i32_zve32x(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                   rsc);
}
