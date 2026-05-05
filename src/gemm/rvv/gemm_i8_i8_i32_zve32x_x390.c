// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32x) || __riscv_zve32x < 1000000
#error This file requires the RISC-V Zve32x extension, version 1000000.
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

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
 * Uses a 4 x LMUL=4 x 2 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e32m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x2_i8_i8_i32_zve32x_x390(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a, size_t rsa,
    const int8_t *b, size_t rsb, int32_t beta, int32_t *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk_init;
  size_t kk;
  size_t kk0;
  size_t ii0;
  int8_t a00_0;
  int8_t a01_0;
  int8_t a02;
  int8_t a03;
  vint8m1_t b0;
  vint16m2_t prod0_0;
  vint16m2_t prod1_0;
  vint16m2_t prod2;
  vint16m2_t prod3;
  vint32m4_t acc0;
  vint32m4_t acc1;
  vint32m4_t acc2;
  vint32m4_t acc3;
  vint32m4_t c00;
  vint32m4_t c01;
  vint32m4_t c02;
  vint32m4_t c03;
  int8_t a000;
  int8_t a001;
  int8_t a002;
  int8_t a003;
  int8_t a010;
  int8_t a011;
  int8_t a012;
  int8_t a013;
  vint8m1_t b00;
  vint8m1_t b01;
  vint16m2_t prod00;
  vint16m2_t prod01;
  vint16m2_t prod02;
  vint16m2_t prod03;
  vint16m2_t prod10;
  vint16m2_t prod11;
  vint16m2_t prod12;
  vint16m2_t prod13;
  int8_t a0;
  vint16m2_t prod;
  vint32m4_t acc;
  vint32m4_t c0;
  int8_t a00_1;
  int8_t a01_1;
  vint16m2_t prod0_1;
  vint16m2_t prod1_1;
  if (k == 0) {
    for (ii = 0; ii < m; ii++) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m4(n - jj);
        c0 = __riscv_vle32_v_i32m4(c + (((ii + 0) * rsc) + jj), jj_vl);
        c0 = __riscv_vmul_vx_i32m4(c0, beta, jj_vl);
        __riscv_vse32_v_i32m4(c + (((ii + 0) * rsc) + jj), c0, jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e8m1(n - jj);
      for (kk_init = 0; ((kk_init + 1) <= k) && (kk_init < (0 + (1 * 1)));
           kk_init = kk_init + 1) {
        a00_0 = a[((ii + 0) * rsa) + ((kk_init + 0) * 1)];
        a01_0 = a[((ii + 1) * rsa) + ((kk_init + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk_init + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk_init + 0) * 1)];
        b0 = __riscv_vle8_v_i8m1(b + (((kk_init + 0) * rsb) + ((jj + 0) * 1)),
                                 jj_vl);
        prod0_0 = __riscv_vwmul_vx_i16m2(b0, a00_0, jj_vl);
        prod1_0 = __riscv_vwmul_vx_i16m2(b0, a01_0, jj_vl);
        prod2 = __riscv_vwmul_vx_i16m2(b0, a02, jj_vl);
        prod3 = __riscv_vwmul_vx_i16m2(b0, a03, jj_vl);
        acc0 = __riscv_vwcvt_x_x_v_i32m4(prod0_0, jj_vl);
        acc1 = __riscv_vwcvt_x_x_v_i32m4(prod1_0, jj_vl);
        acc2 = __riscv_vwcvt_x_x_v_i32m4(prod2, jj_vl);
        acc3 = __riscv_vwcvt_x_x_v_i32m4(prod3, jj_vl);
      }
      for (kk = kk_init; (kk + 2) <= k; kk = kk + 2) {
        a000 = a[((ii + 0) * rsa) + ((kk + 0) * 1)];
        a001 = a[((ii + 1) * rsa) + ((kk + 0) * 1)];
        a002 = a[((ii + 2) * rsa) + ((kk + 0) * 1)];
        a003 = a[((ii + 3) * rsa) + ((kk + 0) * 1)];
        a010 = a[((ii + 0) * rsa) + ((kk + 1) * 1)];
        a011 = a[((ii + 1) * rsa) + ((kk + 1) * 1)];
        a012 = a[((ii + 2) * rsa) + ((kk + 1) * 1)];
        a013 = a[((ii + 3) * rsa) + ((kk + 1) * 1)];
        b00 =
            __riscv_vle8_v_i8m1(b + (((kk + 0) * rsb) + ((jj + 0) * 1)), jj_vl);
        b01 =
            __riscv_vle8_v_i8m1(b + (((kk + 1) * rsb) + ((jj + 0) * 1)), jj_vl);
        prod00 = __riscv_vwmul_vx_i16m2(b00, a000, jj_vl);
        prod01 = __riscv_vwmul_vx_i16m2(b00, a001, jj_vl);
        prod02 = __riscv_vwmul_vx_i16m2(b00, a002, jj_vl);
        prod03 = __riscv_vwmul_vx_i16m2(b00, a003, jj_vl);
        prod10 = __riscv_vwmul_vx_i16m2(b01, a010, jj_vl);
        prod11 = __riscv_vwmul_vx_i16m2(b01, a011, jj_vl);
        prod12 = __riscv_vwmul_vx_i16m2(b01, a012, jj_vl);
        prod13 = __riscv_vwmul_vx_i16m2(b01, a013, jj_vl);
        acc0 = __riscv_vwadd_wv_i32m4(acc0, prod00, jj_vl);
        acc1 = __riscv_vwadd_wv_i32m4(acc1, prod01, jj_vl);
        acc2 = __riscv_vwadd_wv_i32m4(acc2, prod02, jj_vl);
        acc3 = __riscv_vwadd_wv_i32m4(acc3, prod03, jj_vl);
        acc0 = __riscv_vwadd_wv_i32m4(acc0, prod10, jj_vl);
        acc1 = __riscv_vwadd_wv_i32m4(acc1, prod11, jj_vl);
        acc2 = __riscv_vwadd_wv_i32m4(acc2, prod12, jj_vl);
        acc3 = __riscv_vwadd_wv_i32m4(acc3, prod13, jj_vl);
      }
      for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        a00_0 = a[((ii + 0) * rsa) + ((kk0 + 0) * 1)];
        a01_0 = a[((ii + 1) * rsa) + ((kk0 + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk0 + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk0 + 0) * 1)];
        b0 = __riscv_vle8_v_i8m1(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1)),
                                 jj_vl);
        prod0_0 = __riscv_vwmul_vx_i16m2(b0, a00_0, jj_vl);
        prod1_0 = __riscv_vwmul_vx_i16m2(b0, a01_0, jj_vl);
        prod2 = __riscv_vwmul_vx_i16m2(b0, a02, jj_vl);
        prod3 = __riscv_vwmul_vx_i16m2(b0, a03, jj_vl);
        acc0 = __riscv_vwadd_wv_i32m4(acc0, prod0_0, jj_vl);
        acc1 = __riscv_vwadd_wv_i32m4(acc1, prod1_0, jj_vl);
        acc2 = __riscv_vwadd_wv_i32m4(acc2, prod2, jj_vl);
        acc3 = __riscv_vwadd_wv_i32m4(acc3, prod3, jj_vl);
      }
      c00 = __riscv_vle32_v_i32m4(c + (((ii + 0) * rsc) + jj), jj_vl);
      c01 = __riscv_vle32_v_i32m4(c + (((ii + 1) * rsc) + jj), jj_vl);
      c02 = __riscv_vle32_v_i32m4(c + (((ii + 2) * rsc) + jj), jj_vl);
      c03 = __riscv_vle32_v_i32m4(c + (((ii + 3) * rsc) + jj), jj_vl);
      c00 = __riscv_vmul_vx_i32m4(c00, beta, jj_vl);
      c00 = __riscv_vmacc_vx_i32m4(c00, alpha, acc0, jj_vl);
      __riscv_vse32_v_i32m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c00,
                            jj_vl);
      c01 = __riscv_vmul_vx_i32m4(c01, beta, jj_vl);
      c01 = __riscv_vmacc_vx_i32m4(c01, alpha, acc1, jj_vl);
      __riscv_vse32_v_i32m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), c01,
                            jj_vl);
      c02 = __riscv_vmul_vx_i32m4(c02, beta, jj_vl);
      c02 = __riscv_vmacc_vx_i32m4(c02, alpha, acc2, jj_vl);
      __riscv_vse32_v_i32m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), c02,
                            jj_vl);
      c03 = __riscv_vmul_vx_i32m4(c03, beta, jj_vl);
      c03 = __riscv_vmacc_vx_i32m4(c03, alpha, acc3, jj_vl);
      __riscv_vse32_v_i32m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), c03,
                            jj_vl);
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e8m1(n - jj);
      for (kk_init = 0; ((kk_init + 1) <= k) && (kk_init < (0 + (1 * 1)));
           kk_init = kk_init + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk_init + 0) * 1)];
        b0 = __riscv_vle8_v_i8m1(b + (((kk_init + 0) * rsb) + ((jj + 0) * 1)),
                                 jj_vl);
        prod = __riscv_vwmul_vx_i16m2(b0, a0, jj_vl);
        acc = __riscv_vwcvt_x_x_v_i32m4(prod, jj_vl);
      }
      for (kk = kk_init; (kk + 2) <= k; kk = kk + 2) {
        a00_1 = a[((ii0 + 0) * rsa) + ((kk + 0) * 1)];
        a01_1 = a[((ii0 + 0) * rsa) + ((kk + 1) * 1)];
        b00 =
            __riscv_vle8_v_i8m1(b + (((kk + 0) * rsb) + ((jj + 0) * 1)), jj_vl);
        b01 =
            __riscv_vle8_v_i8m1(b + (((kk + 1) * rsb) + ((jj + 0) * 1)), jj_vl);
        prod0_1 = __riscv_vwmul_vx_i16m2(b00, a00_1, jj_vl);
        prod1_1 = __riscv_vwmul_vx_i16m2(b01, a01_1, jj_vl);
        acc = __riscv_vwadd_wv_i32m4(acc, prod0_1, jj_vl);
        acc = __riscv_vwadd_wv_i32m4(acc, prod1_1, jj_vl);
      }
      for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk0 + 0) * 1)];
        b0 = __riscv_vle8_v_i8m1(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1)),
                                 jj_vl);
        prod = __riscv_vwmul_vx_i16m2(b0, a0, jj_vl);
        acc = __riscv_vwadd_wv_i32m4(acc, prod, jj_vl);
      }
      c0 = __riscv_vle32_v_i32m4(c + (((ii0 + 0) * rsc) + jj), jj_vl);
      c0 = __riscv_vmul_vx_i32m4(c0, beta, jj_vl);
      c0 = __riscv_vmacc_vx_i32m4(c0, alpha, acc, jj_vl);
      __riscv_vse32_v_i32m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)), c0,
                            jj_vl);
    }
  }
}

SKL_FUNC void skl_gemm_i8_i8_i32_zve32x_x390(size_t m, size_t n, size_t k,
                                             int32_t alpha, const int8_t *a,
                                             size_t rsa, const int8_t *b,
                                             size_t rsb, int32_t beta,
                                             int32_t *c, size_t rsc) {
  skl_gemm_4xm4x2_i8_i8_i32_zve32x_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                        rsc);
}
