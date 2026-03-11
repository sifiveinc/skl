// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zvfbfwma)
#error This file requires the RISC-V Zvfbfwma extension.
#endif

#if !defined(__riscv_zve32f)
#error This file requires the RISC-V Zve32f extension.
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

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
 * skl_gemm_bf16rc_bf16rc_f32rc_scalar(
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
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_bf16_bf16_f32_zvfbfwma(
    size_t m, size_t n, size_t k, float alpha, const __bf16 *a, size_t rsa,
    const __bf16 *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  size_t ii0;
  __bf16 a00;
  __bf16 a01;
  __bf16 a02;
  __bf16 a03;
  vbfloat16m2_t b0;
  vfloat32m4_t acc0;
  vfloat32m4_t acc1;
  vfloat32m4_t acc2;
  vfloat32m4_t acc3;
  vfloat32m4_t c00;
  vfloat32m4_t c01;
  vfloat32m4_t c02;
  vfloat32m4_t c03;
  __bf16 a0;
  vfloat32m4_t acc;
  vfloat32m4_t c0;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m4(n - jj);
        c0 = __riscv_vle32_v_f32m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)),
                                   jj_vl);
        c0 = __riscv_vfmul_vf_f32m4(c0, beta, jj_vl);
        __riscv_vse32_v_f32m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c0,
                              jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m2(n - jj);
      float zero = 0.f;
      acc0 = __riscv_vfmv_v_f_f32m4(zero, jj_vl);
      acc1 = __riscv_vfmv_v_f_f32m4(zero, jj_vl);
      acc2 = __riscv_vfmv_v_f_f32m4(zero, jj_vl);
      acc3 = __riscv_vfmv_v_f_f32m4(zero, jj_vl);
      for (kk = 0; (kk + 1) <= k; kk = kk + 1) {
        a00 = a[((ii + 0) * rsa) + ((kk + 0) * 1)];
        a01 = a[((ii + 1) * rsa) + ((kk + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk + 0) * 1)];
        b0 = __riscv_vle16_v_bf16m2(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        acc0 = __riscv_vfwmaccbf16_vf_f32m4(acc0, a00, b0, jj_vl);
        acc1 = __riscv_vfwmaccbf16_vf_f32m4(acc1, a01, b0, jj_vl);
        acc2 = __riscv_vfwmaccbf16_vf_f32m4(acc2, a02, b0, jj_vl);
        acc3 = __riscv_vfwmaccbf16_vf_f32m4(acc3, a03, b0, jj_vl);
      }
      c00 =
          __riscv_vle32_v_f32m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), jj_vl);
      c01 =
          __riscv_vle32_v_f32m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), jj_vl);
      c02 =
          __riscv_vle32_v_f32m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), jj_vl);
      c03 =
          __riscv_vle32_v_f32m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), jj_vl);
      c00 = __riscv_vfmul_vf_f32m4(c00, beta, jj_vl);
      c00 = __riscv_vfmacc_vf_f32m4(c00, alpha, acc0, jj_vl);
      __riscv_vse32_v_f32m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c00,
                            jj_vl);
      c01 = __riscv_vfmul_vf_f32m4(c01, beta, jj_vl);
      c01 = __riscv_vfmacc_vf_f32m4(c01, alpha, acc1, jj_vl);
      __riscv_vse32_v_f32m4(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), c01,
                            jj_vl);
      c02 = __riscv_vfmul_vf_f32m4(c02, beta, jj_vl);
      c02 = __riscv_vfmacc_vf_f32m4(c02, alpha, acc2, jj_vl);
      __riscv_vse32_v_f32m4(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), c02,
                            jj_vl);
      c03 = __riscv_vfmul_vf_f32m4(c03, beta, jj_vl);
      c03 = __riscv_vfmacc_vf_f32m4(c03, alpha, acc3, jj_vl);
      __riscv_vse32_v_f32m4(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), c03,
                            jj_vl);
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m2(n - jj);
      float zero = 0.f;
      acc = __riscv_vfmv_v_f_f32m4(zero, jj_vl);
      for (kk = 0; (kk + 1) <= k; kk = kk + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk + 0) * 1)];
        b0 = __riscv_vle16_v_bf16m2(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        acc = __riscv_vfwmaccbf16_vf_f32m4(acc, a0, b0, jj_vl);
      }
      c0 = __riscv_vle32_v_f32m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)),
                                 jj_vl);
      c0 = __riscv_vfmul_vf_f32m4(c0, beta, jj_vl);
      c0 = __riscv_vfmacc_vf_f32m4(c0, alpha, acc, jj_vl);
      __riscv_vse32_v_f32m4(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)), c0,
                            jj_vl);
    }
  }
}

SKL_FUNC void skl_gemm_bf16_bf16_f32_zvfbfwma(size_t m, size_t n, size_t k,
                                              float alpha, const __bf16 *a,
                                              size_t rsa, const __bf16 *b,
                                              size_t rsb, float beta, float *c,
                                              size_t rsc) {
  skl_gemm_4xm4x1_bf16_bf16_f32_zvfbfwma(m, n, k, alpha, a, rsa, b, rsb, beta,
                                         c, rsc);
}
