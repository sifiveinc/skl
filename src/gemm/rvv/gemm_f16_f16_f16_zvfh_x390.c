// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zvfh) || __riscv_zvfh < 1000000
#error This file requires the RISC-V zvfh extension, version 1000000.
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

/**
 * @brief RVV float16 matrix-matrix multiplication (HGEMM) for row-major
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
 * Computes `C = alpha * A * B + beta * C` for FP16 row-major matrices.
 *
 * Functionally equivalent to scalar call:
 * ```
 * skl_gemm_f16rc_f16rc_f16rc_scalar(
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
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e16m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_f16_f16_f16_zvfh_x390(
    size_t m, size_t n, size_t k, _Float16 alpha, const _Float16 *a, size_t rsa,
    const _Float16 *b, size_t rsb, _Float16 beta, _Float16 *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk_peel;
  size_t kk;
  size_t ii0;
  _Float16 _alpha;
  _Float16 _beta;
  _Float16 a00;
  _Float16 a01;
  _Float16 a02;
  _Float16 a03;
  vfloat16m4_t b0;
  vfloat16m4_t acc0;
  vfloat16m4_t acc1;
  vfloat16m4_t acc2;
  vfloat16m4_t acc3;
  _Float16 a000;
  _Float16 a010;
  _Float16 a020;
  _Float16 a030;
  vfloat16m4_t b00;
  vfloat16m4_t c00;
  vfloat16m4_t c01;
  vfloat16m4_t c02;
  vfloat16m4_t c03;
  _Float16 a0;
  vfloat16m4_t acc;
  vfloat16m4_t c0;
  _alpha = alpha;
  _beta = beta;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e16m4(n - jj);
        c0 = __riscv_vle16_v_f16m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)),
                                   jj_vl);
        c0 = __riscv_vfmul_vf_f16m4(c0, _beta, jj_vl);
        __riscv_vse16_v_f16m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c0,
                              jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        b0 = __riscv_vle16_v_f16m4(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        skl_instruction_schedule_barrier();
        a00 = a[((ii + 0) * rsa) + ((kk_peel + 0) * 1)];
        a01 = a[((ii + 1) * rsa) + ((kk_peel + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk_peel + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk_peel + 0) * 1)];
        acc0 = __riscv_vfmul_vf_f16m4(b0, a00, jj_vl);
        acc1 = __riscv_vfmul_vf_f16m4(b0, a01, jj_vl);
        acc2 = __riscv_vfmul_vf_f16m4(b0, a02, jj_vl);
        acc3 = __riscv_vfmul_vf_f16m4(b0, a03, jj_vl);
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        b00 = __riscv_vle16_v_f16m4(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        skl_instruction_schedule_barrier();
        a000 = a[((ii + 0) * rsa) + ((kk + 0) * 1)];
        a010 = a[((ii + 1) * rsa) + ((kk + 0) * 1)];
        a020 = a[((ii + 2) * rsa) + ((kk + 0) * 1)];
        a030 = a[((ii + 3) * rsa) + ((kk + 0) * 1)];
        acc0 = __riscv_vfmacc_vf_f16m4(acc0, a000, b00, jj_vl);
        acc1 = __riscv_vfmacc_vf_f16m4(acc1, a010, b00, jj_vl);
        acc2 = __riscv_vfmacc_vf_f16m4(acc2, a020, b00, jj_vl);
        acc3 = __riscv_vfmacc_vf_f16m4(acc3, a030, b00, jj_vl);
      }
      c00 =
          __riscv_vle16_v_f16m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), jj_vl);
      c01 =
          __riscv_vle16_v_f16m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), jj_vl);
      c02 =
          __riscv_vle16_v_f16m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), jj_vl);
      c03 =
          __riscv_vle16_v_f16m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), jj_vl);
      c00 = __riscv_vfmul_vf_f16m4(c00, _beta, jj_vl);
      c01 = __riscv_vfmul_vf_f16m4(c01, _beta, jj_vl);
      c02 = __riscv_vfmul_vf_f16m4(c02, _beta, jj_vl);
      c03 = __riscv_vfmul_vf_f16m4(c03, _beta, jj_vl);
      c00 = __riscv_vfmacc_vf_f16m4(c00, _alpha, acc0, jj_vl);
      c01 = __riscv_vfmacc_vf_f16m4(c01, _alpha, acc1, jj_vl);
      c02 = __riscv_vfmacc_vf_f16m4(c02, _alpha, acc2, jj_vl);
      c03 = __riscv_vfmacc_vf_f16m4(c03, _alpha, acc3, jj_vl);
      __riscv_vse16_v_f16m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c00,
                            jj_vl);
      __riscv_vse16_v_f16m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), c01,
                            jj_vl);
      __riscv_vse16_v_f16m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), c02,
                            jj_vl);
      __riscv_vse16_v_f16m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), c03,
                            jj_vl);
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk_peel + 0) * 1)];
        b0 = __riscv_vle16_v_f16m4(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc = __riscv_vfmul_vf_f16m4(b0, a0, jj_vl);
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk + 0) * 1)];
        b0 = __riscv_vle16_v_f16m4(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc = __riscv_vfmacc_vf_f16m4(acc, a0, b0, jj_vl);
      }
      c0 = __riscv_vle16_v_f16m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)),
                                 jj_vl);
      c0 = __riscv_vfmul_vf_f16m4(c0, _beta, jj_vl);
      c0 = __riscv_vfmacc_vf_f16m4(c0, _alpha, acc, jj_vl);
      __riscv_vse16_v_f16m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)), c0,
                            jj_vl);
    }
  }
}

SKL_FUNC void skl_gemm_f16_f16_f16_zvfh_x390(size_t m, size_t n, size_t k,
                                             _Float16 alpha, const _Float16 *a,
                                             size_t rsa, const _Float16 *b,
                                             size_t rsb, _Float16 beta,
                                             _Float16 *c, size_t rsc) {
  skl_gemm_4xm4x1_f16_f16_f16_zvfh_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                        rsc);
}
