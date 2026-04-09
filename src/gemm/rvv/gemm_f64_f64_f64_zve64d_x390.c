// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve64d) || __riscv_zve64d < 1000000
#error This file requires the RISC-V zve64d extension, version 1000000.
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

/**
 * @brief RVV float64 matrix-matrix multiplication (DGEMM) for row-major
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
 * Computes `C = alpha * A * B + beta * C` for F64 row-major matrices.
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
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_f64_f64_f64_zve64d_x390(
    size_t m, size_t n, size_t k, double alpha, const double *a, size_t rsa,
    const double *b, size_t rsb, double beta, double *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk_peel;
  size_t kk;
  size_t ii0;
  double _alpha;
  double _beta;
  double a00;
  double a01;
  double a02;
  double a03;
  vfloat64m4_t b0;
  vfloat64m4_t acc0;
  vfloat64m4_t acc1;
  vfloat64m4_t acc2;
  vfloat64m4_t acc3;
  vfloat64m4_t c00;
  vfloat64m4_t c01;
  vfloat64m4_t c02;
  vfloat64m4_t c03;
  double a0;
  vfloat64m4_t acc;
  vfloat64m4_t c0;
  _alpha = alpha;
  _beta = beta;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e64m4(n - jj);
        c0 = __riscv_vle64_v_f64m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)),
                                   jj_vl);
        c0 = __riscv_vfmul_vf_f64m4(c0, _beta, jj_vl);
        __riscv_vse64_v_f64m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c0,
                              jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e64m4(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        b0 = __riscv_vle64_v_f64m4(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        skl_instruction_schedule_barrier();
        a00 = a[((ii + 0) * rsa) + ((kk_peel + 0) * 1)];
        a01 = a[((ii + 1) * rsa) + ((kk_peel + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk_peel + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk_peel + 0) * 1)];
        // Try to schedule all loads before vfmuls:
        __asm__ volatile(""
                         : "+vr"(b0), "+r"(a00), "+r"(a01), "+r"(a02),
                           "+r"(a03));
        acc0 = __riscv_vfmul_vf_f64m4(b0, a00, jj_vl);
        acc1 = __riscv_vfmul_vf_f64m4(b0, a01, jj_vl);
        acc2 = __riscv_vfmul_vf_f64m4(b0, a02, jj_vl);
        acc3 = __riscv_vfmul_vf_f64m4(b0, a03, jj_vl);
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        b0 = __riscv_vle64_v_f64m4(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        skl_instruction_schedule_barrier();
        a00 = a[((ii + 0) * rsa) + ((kk + 0) * 1)];
        a01 = a[((ii + 1) * rsa) + ((kk + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk + 0) * 1)];
        // Try to schedule all loads before vfmaccs:
        __asm__ volatile(""
                         : "+vr"(b0), "+r"(a00), "+r"(a01), "+r"(a02),
                           "+r"(a03));
        acc0 = __riscv_vfmacc_vf_f64m4(acc0, a00, b0, jj_vl);
        acc1 = __riscv_vfmacc_vf_f64m4(acc1, a01, b0, jj_vl);
        acc2 = __riscv_vfmacc_vf_f64m4(acc2, a02, b0, jj_vl);
        acc3 = __riscv_vfmacc_vf_f64m4(acc3, a03, b0, jj_vl);
      }
      c00 =
          __riscv_vle64_v_f64m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), jj_vl);
      c00 = __riscv_vfmul_vf_f64m4(c00, _beta, jj_vl);
      c00 = __riscv_vfmacc_vf_f64m4(c00, _alpha, acc0, jj_vl);
      __riscv_vse64_v_f64m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c00,
                            jj_vl);
      c01 =
          __riscv_vle64_v_f64m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), jj_vl);
      c01 = __riscv_vfmul_vf_f64m4(c01, _beta, jj_vl);
      c01 = __riscv_vfmacc_vf_f64m4(c01, _alpha, acc1, jj_vl);
      __riscv_vse64_v_f64m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), c01,
                            jj_vl);
      c02 =
          __riscv_vle64_v_f64m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), jj_vl);
      c02 = __riscv_vfmul_vf_f64m4(c02, _beta, jj_vl);
      c02 = __riscv_vfmacc_vf_f64m4(c02, _alpha, acc2, jj_vl);
      __riscv_vse64_v_f64m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), c02,
                            jj_vl);
      c03 =
          __riscv_vle64_v_f64m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), jj_vl);
      c03 = __riscv_vfmul_vf_f64m4(c03, _beta, jj_vl);
      c03 = __riscv_vfmacc_vf_f64m4(c03, _alpha, acc3, jj_vl);
      __riscv_vse64_v_f64m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), c03,
                            jj_vl);
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e64m4(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk_peel + 0) * 1)];
        b0 = __riscv_vle64_v_f64m4(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc = __riscv_vfmul_vf_f64m4(b0, a0, jj_vl);
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk + 0) * 1)];
        b0 = __riscv_vle64_v_f64m4(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc = __riscv_vfmacc_vf_f64m4(acc, a0, b0, jj_vl);
      }
      c0 = __riscv_vle64_v_f64m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)),
                                 jj_vl);
      c0 = __riscv_vfmul_vf_f64m4(c0, _beta, jj_vl);
      c0 = __riscv_vfmacc_vf_f64m4(c0, _alpha, acc, jj_vl);
      __riscv_vse64_v_f64m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)), c0,
                            jj_vl);
    }
  }
}

SKL_FUNC void skl_gemm_f64_f64_f64_zve64d_x390(size_t m, size_t n, size_t k,
                                               double alpha, const double *a,
                                               size_t rsa, const double *b,
                                               size_t rsb, double beta,
                                               double *c, size_t rsc) {
  skl_gemm_4xm4x1_f64_f64_f64_zve64d_x390(m, n, k, alpha, a, rsa, b, rsb, beta,
                                          c, rsc);
}
