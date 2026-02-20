// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

/**
 * @brief RVV float32 vector-matrix multiplication for row-major B and
 * unit-stride vectors, tuned for X390.
 *
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scalar multiplier for A*B product.
 * @param a - Pointer to vector A.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to vector C.
 *
 * Computes `C = alpha * A * B + beta * C` for FP32 unit-stride vector A,
 * row-major matrix B, and output unit-stride vector C.
 *
 * Functionally equivalent to scalar call:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_scalar(
 *     1, n, k,
 *     alpha,
 *     a, 1, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, 1, 1
 * );
 * ```
 * Uses a 1 x 2*LMUL=8 x 2 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `n >= 2*__riscv_vsetvlmax_e32m8()`.
 */
SKL_FUNC_PRIVATE void
skl_gemm_1x2m8x2_f32_f32_f32_zve32f_x390(size_t n, size_t k, float alpha,
                                         const float *a, const float *b,
                                         size_t rsb, float beta, float *c) {
  size_t jj_vl;
  size_t jj_vl_0;
  size_t ii;
  size_t jj;
  size_t kk_peel;
  size_t kk;
  size_t kk0;
  float alpha0;
  float beta0;
  float a0;
  vfloat32m8_t b00_0;
  vfloat32m8_t b01_0;
  vfloat32m8_t acc0;
  vfloat32m8_t acc1;
  float a00;
  vfloat32m8_t b000;
  vfloat32m8_t b001;
  float a01;
  vfloat32m8_t c00;
  vfloat32m8_t c01;
  vfloat32m8_t c0;
  alpha0 = alpha;
  beta0 = beta;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= 1; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m1(n - jj);
        c0 =
            __riscv_vle32_v_f32m8(c + (((ii + 0) * 1) + ((jj + 0) * 1)), jj_vl);
        c0 = __riscv_vfmul_vf_f32m8(c0, beta0, jj_vl);
        __riscv_vse32_v_f32m8(c + (((ii + 0) * 1) + ((jj + 0) * 1)), c0, jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 1) <= 1; ii = ii + 1) {
    for (jj = 0; jj < n; jj = jj + (jj_vl + jj_vl_0)) {
      jj_vl = __riscv_vsetvl_e32m8(n - jj);
      jj_vl_0 = __riscv_vsetvl_e32m8((n - jj) - jj_vl);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a0], 0(%[a_load]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b00_0], (%[b_load]) \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b01_0], (%[b_load_0]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vfmul.vf %[acc0], %[b00_0], %[a0] \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vfmul.vf %[acc1], %[b01_0], %[a0] \n\t"
            : [a0] "=&f"(a0), [b00_0] "=&vr"(b00_0), [b01_0] "=&vr"(b01_0),
              [acc0] "=&vr"(acc0), [acc1] "=vr"(acc1)
            : [a_load] "r"(a + (((ii + 0) * 1) + ((kk_peel + 0) * 1))),
              [jj_vl_in] "r"(jj_vl),
              [b_load] "r"(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1))),
              [jj_vl_0_in] "r"(jj_vl_0),
              [b_load_0] "r"(b + (((kk_peel + 0) * rsb) + ((jj + jj_vl) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk = kk_peel; (kk + 2) <= k; kk = kk + 2) {
        __asm__ volatile(
            "\n\t"
            "flw %[a00], 0(%[a_load_0]) \n\t"
            "flw %[a01], 4(%[a_load_0]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b000], (%[b_load_1]) \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b001], (%[b_load_2]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b000] \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc1], %[a00], %[b001] \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b000], (%[b_load_3]) \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b001], (%[b_load_4]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc0], %[a01], %[b000] \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc1], %[a01], %[b001] \n\t"
            : [a00] "=&f"(a00), [b000] "=&vr"(b000), [b001] "=&vr"(b001),
              [acc0] "+&vr"(acc0), [acc1] "+&vr"(acc1), [a01] "=&f"(a01)
            : [a_load_0] "r"(a + (((ii + 0) * 1) + ((kk + 0) * 1))),
              [jj_vl_in] "r"(jj_vl),
              [b_load_1] "r"(b + (((kk + 0) * rsb) + ((jj + 0) * 1))),
              [jj_vl_0_in] "r"(jj_vl_0),
              [b_load_2] "r"(b + (((kk + 0) * rsb) + ((jj + jj_vl) * 1))),
              [b_load_3] "r"(b + (((kk + 1) * rsb) + ((jj + 0) * 1))),
              [b_load_4] "r"(b + (((kk + 1) * rsb) + ((jj + jj_vl) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a0], 0(%[a_load_2]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b00_0], (%[b_load_5]) \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vle32.v %[b01_0], (%[b_load_6]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc0], %[a0], %[b00_0] \n\t"
            "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc1], %[a0], %[b01_0] \n\t"
            : [a0] "=&f"(a0), [b00_0] "=&vr"(b00_0), [b01_0] "=&vr"(b01_0),
              [acc0] "+&vr"(acc0), [acc1] "+vr"(acc1)
            : [a_load_2] "r"(a + (((ii + 0) * 1) + ((kk0 + 0) * 1))),
              [jj_vl_in] "r"(jj_vl),
              [b_load_5] "r"(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1))),
              [jj_vl_0_in] "r"(jj_vl_0),
              [b_load_6] "r"(b + (((kk0 + 0) * rsb) + ((jj + jj_vl) * 1)))
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
          "vle32.v %[c00], (%[c_load]) \n\t"
          "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
          "vle32.v %[c01], (%[c_load_0]) \n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
          "vfmul.vf %[c00], %[c00], %[beta0] \n\t"
          "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
          "vfmul.vf %[c01], %[c01], %[beta0] \n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
          "vfmacc.vf %[c00], %[alpha0], %[acc0] \n\t"
          "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
          "vfmacc.vf %[c01], %[alpha0], %[acc1] \n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
          "vse32.v %[c00], (%[c_store]) \n\t"
          "vsetvli zero, %[jj_vl_0_in], e32, m8, ta, ma \n\t"
          "vse32.v %[c01], (%[c_store_0]) \n\t"
          : [c00] "=&vr"(c00), [c01] "=&vr"(c01)
          : [jj_vl_in] "r"(jj_vl),
            [c_load] "r"(c + (((ii + 0) * 1) + ((jj + 0) * 1))),
            [jj_vl_0_in] "r"(jj_vl_0),
            [c_load_0] "r"(c + (((ii + 0) * 1) + ((jj + jj_vl) * 1))),
            [beta0] "f"(beta0), [alpha0] "f"(alpha0), [acc0] "vr"(acc0),
            [acc1] "vr"(acc1),
            [c_store] "r"(c + (((ii + 0) * 1) + ((jj + 0) * 1))),
            [c_store_0] "r"(c + (((ii + 0) * 1) + ((jj + jj_vl) * 1)))
          : "vtype", "vl", "memory");
    }
  }
}

/**
 * @brief RVV float32 matrix-matrix multiplication (SGEMM) for row-major
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
 * Computes `C = alpha * A * B + beta * C` for FP32 row-major matrices.
 *
 * Functionally equivalent to the scalar call:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_scalar(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 * Uses an 8 x LMUL=1 x 4 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 8` and `n <= 2*__riscv_vsetvlmax_e32m1()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_8xm1x4_f32_f32_f32_zve32f_x390(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    const float *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk_peel;
  size_t kk;
  size_t kk0;
  size_t ii0;
  float _alpha;
  float _beta;
  float a00;
  float a01;
  float a02;
  float a03;
  float a04;
  float a05;
  float a06;
  float a07;
  vfloat32m1_t b0;
  vfloat32m1_t acc0;
  vfloat32m1_t acc1;
  vfloat32m1_t acc2;
  vfloat32m1_t acc3;
  vfloat32m1_t acc4;
  vfloat32m1_t acc5;
  vfloat32m1_t acc6;
  vfloat32m1_t acc7;
  float a000;
  float a001;
  float a002;
  float a003;
  float a010;
  float a011;
  float a012;
  float a013;
  float a020;
  float a021;
  float a022;
  float a023;
  float a030;
  float a031;
  float a032;
  float a033;
  float a040;
  float a041;
  float a042;
  float a043;
  float a050;
  float a051;
  float a052;
  float a053;
  float a060;
  float a061;
  float a062;
  float a063;
  float a070;
  float a071;
  float a072;
  float a073;
  vfloat32m1_t b00;
  vfloat32m1_t b01;
  vfloat32m1_t b02;
  vfloat32m1_t b03;
  vfloat32m1_t c00;
  vfloat32m1_t c01;
  vfloat32m1_t c02;
  vfloat32m1_t c03;
  vfloat32m1_t c04;
  vfloat32m1_t c05;
  vfloat32m1_t c06;
  vfloat32m1_t c07;
  float a0;
  vfloat32m1_t acc;
  vfloat32m1_t c0;
  _alpha = alpha;
  _beta = beta;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m1(n - jj);
        c0 = __riscv_vle32_v_f32m1(c + (((ii + 0) * rsc) + ((jj + 0) * 1)),
                                   jj_vl);
        c0 = __riscv_vfmul_vf_f32m1(c0, _beta, jj_vl);
        __riscv_vse32_v_f32m1(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c0,
                              jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 8) <= m; ii = ii + 8) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m1(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        a00 = a[((ii + 0) * rsa) + ((kk_peel + 0) * 1)];
        a01 = a[((ii + 1) * rsa) + ((kk_peel + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk_peel + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk_peel + 0) * 1)];
        a04 = a[((ii + 4) * rsa) + ((kk_peel + 0) * 1)];
        a05 = a[((ii + 5) * rsa) + ((kk_peel + 0) * 1)];
        a06 = a[((ii + 6) * rsa) + ((kk_peel + 0) * 1)];
        a07 = a[((ii + 7) * rsa) + ((kk_peel + 0) * 1)];
        b0 = __riscv_vle32_v_f32m1(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc0 = __riscv_vfmul_vf_f32m1(b0, a00, jj_vl);
        acc1 = __riscv_vfmul_vf_f32m1(b0, a01, jj_vl);
        acc2 = __riscv_vfmul_vf_f32m1(b0, a02, jj_vl);
        acc3 = __riscv_vfmul_vf_f32m1(b0, a03, jj_vl);
        acc4 = __riscv_vfmul_vf_f32m1(b0, a04, jj_vl);
        acc5 = __riscv_vfmul_vf_f32m1(b0, a05, jj_vl);
        acc6 = __riscv_vfmul_vf_f32m1(b0, a06, jj_vl);
        acc7 = __riscv_vfmul_vf_f32m1(b0, a07, jj_vl);
      }
      for (kk = kk_peel; (kk + 4) <= k; kk = kk + 4) {
        a000 = a[((ii + 0) * rsa) + ((kk + 0) * 1)];
        a001 = a[((ii + 0) * rsa) + ((kk + 1) * 1)];
        a002 = a[((ii + 0) * rsa) + ((kk + 2) * 1)];
        a003 = a[((ii + 0) * rsa) + ((kk + 3) * 1)];
        a010 = a[((ii + 1) * rsa) + ((kk + 0) * 1)];
        a011 = a[((ii + 1) * rsa) + ((kk + 1) * 1)];
        a012 = a[((ii + 1) * rsa) + ((kk + 2) * 1)];
        a013 = a[((ii + 1) * rsa) + ((kk + 3) * 1)];
        a020 = a[((ii + 2) * rsa) + ((kk + 0) * 1)];
        a021 = a[((ii + 2) * rsa) + ((kk + 1) * 1)];
        a022 = a[((ii + 2) * rsa) + ((kk + 2) * 1)];
        a023 = a[((ii + 2) * rsa) + ((kk + 3) * 1)];
        a030 = a[((ii + 3) * rsa) + ((kk + 0) * 1)];
        a031 = a[((ii + 3) * rsa) + ((kk + 1) * 1)];
        a032 = a[((ii + 3) * rsa) + ((kk + 2) * 1)];
        a033 = a[((ii + 3) * rsa) + ((kk + 3) * 1)];
        a040 = a[((ii + 4) * rsa) + ((kk + 0) * 1)];
        a041 = a[((ii + 4) * rsa) + ((kk + 1) * 1)];
        a042 = a[((ii + 4) * rsa) + ((kk + 2) * 1)];
        a043 = a[((ii + 4) * rsa) + ((kk + 3) * 1)];
        a050 = a[((ii + 5) * rsa) + ((kk + 0) * 1)];
        a051 = a[((ii + 5) * rsa) + ((kk + 1) * 1)];
        a052 = a[((ii + 5) * rsa) + ((kk + 2) * 1)];
        a053 = a[((ii + 5) * rsa) + ((kk + 3) * 1)];
        a060 = a[((ii + 6) * rsa) + ((kk + 0) * 1)];
        a061 = a[((ii + 6) * rsa) + ((kk + 1) * 1)];
        a062 = a[((ii + 6) * rsa) + ((kk + 2) * 1)];
        a063 = a[((ii + 6) * rsa) + ((kk + 3) * 1)];
        a070 = a[((ii + 7) * rsa) + ((kk + 0) * 1)];
        a071 = a[((ii + 7) * rsa) + ((kk + 1) * 1)];
        a072 = a[((ii + 7) * rsa) + ((kk + 2) * 1)];
        a073 = a[((ii + 7) * rsa) + ((kk + 3) * 1)];
        skl_instruction_schedule_barrier();
        b00 = __riscv_vle32_v_f32m1(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        b01 = __riscv_vle32_v_f32m1(b + (((kk + 1) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        b02 = __riscv_vle32_v_f32m1(b + (((kk + 2) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        b03 = __riscv_vle32_v_f32m1(b + (((kk + 3) * rsb) + ((jj + 0) * 1)),
                                    jj_vl);
        acc0 = __riscv_vfmacc_vf_f32m1(acc0, a000, b00, jj_vl);
        acc1 = __riscv_vfmacc_vf_f32m1(acc1, a010, b00, jj_vl);
        acc2 = __riscv_vfmacc_vf_f32m1(acc2, a020, b00, jj_vl);
        acc3 = __riscv_vfmacc_vf_f32m1(acc3, a030, b00, jj_vl);
        acc4 = __riscv_vfmacc_vf_f32m1(acc4, a040, b00, jj_vl);
        acc5 = __riscv_vfmacc_vf_f32m1(acc5, a050, b00, jj_vl);
        acc6 = __riscv_vfmacc_vf_f32m1(acc6, a060, b00, jj_vl);
        acc7 = __riscv_vfmacc_vf_f32m1(acc7, a070, b00, jj_vl);
        acc0 = __riscv_vfmacc_vf_f32m1(acc0, a001, b01, jj_vl);
        acc1 = __riscv_vfmacc_vf_f32m1(acc1, a011, b01, jj_vl);
        acc2 = __riscv_vfmacc_vf_f32m1(acc2, a021, b01, jj_vl);
        acc3 = __riscv_vfmacc_vf_f32m1(acc3, a031, b01, jj_vl);
        acc4 = __riscv_vfmacc_vf_f32m1(acc4, a041, b01, jj_vl);
        acc5 = __riscv_vfmacc_vf_f32m1(acc5, a051, b01, jj_vl);
        acc6 = __riscv_vfmacc_vf_f32m1(acc6, a061, b01, jj_vl);
        acc7 = __riscv_vfmacc_vf_f32m1(acc7, a071, b01, jj_vl);
        acc0 = __riscv_vfmacc_vf_f32m1(acc0, a002, b02, jj_vl);
        acc1 = __riscv_vfmacc_vf_f32m1(acc1, a012, b02, jj_vl);
        acc2 = __riscv_vfmacc_vf_f32m1(acc2, a022, b02, jj_vl);
        acc3 = __riscv_vfmacc_vf_f32m1(acc3, a032, b02, jj_vl);
        acc4 = __riscv_vfmacc_vf_f32m1(acc4, a042, b02, jj_vl);
        acc5 = __riscv_vfmacc_vf_f32m1(acc5, a052, b02, jj_vl);
        acc6 = __riscv_vfmacc_vf_f32m1(acc6, a062, b02, jj_vl);
        acc7 = __riscv_vfmacc_vf_f32m1(acc7, a072, b02, jj_vl);
        acc0 = __riscv_vfmacc_vf_f32m1(acc0, a003, b03, jj_vl);
        acc1 = __riscv_vfmacc_vf_f32m1(acc1, a013, b03, jj_vl);
        acc2 = __riscv_vfmacc_vf_f32m1(acc2, a023, b03, jj_vl);
        acc3 = __riscv_vfmacc_vf_f32m1(acc3, a033, b03, jj_vl);
        acc4 = __riscv_vfmacc_vf_f32m1(acc4, a043, b03, jj_vl);
        acc5 = __riscv_vfmacc_vf_f32m1(acc5, a053, b03, jj_vl);
        acc6 = __riscv_vfmacc_vf_f32m1(acc6, a063, b03, jj_vl);
        acc7 = __riscv_vfmacc_vf_f32m1(acc7, a073, b03, jj_vl);
      }
      for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        a00 = a[((ii + 0) * rsa) + ((kk0 + 0) * 1)];
        a01 = a[((ii + 1) * rsa) + ((kk0 + 0) * 1)];
        a02 = a[((ii + 2) * rsa) + ((kk0 + 0) * 1)];
        a03 = a[((ii + 3) * rsa) + ((kk0 + 0) * 1)];
        a04 = a[((ii + 4) * rsa) + ((kk0 + 0) * 1)];
        a05 = a[((ii + 5) * rsa) + ((kk0 + 0) * 1)];
        a06 = a[((ii + 6) * rsa) + ((kk0 + 0) * 1)];
        a07 = a[((ii + 7) * rsa) + ((kk0 + 0) * 1)];
        b0 = __riscv_vle32_v_f32m1(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc0 = __riscv_vfmacc_vf_f32m1(acc0, a00, b0, jj_vl);
        acc1 = __riscv_vfmacc_vf_f32m1(acc1, a01, b0, jj_vl);
        acc2 = __riscv_vfmacc_vf_f32m1(acc2, a02, b0, jj_vl);
        acc3 = __riscv_vfmacc_vf_f32m1(acc3, a03, b0, jj_vl);
        acc4 = __riscv_vfmacc_vf_f32m1(acc4, a04, b0, jj_vl);
        acc5 = __riscv_vfmacc_vf_f32m1(acc5, a05, b0, jj_vl);
        acc6 = __riscv_vfmacc_vf_f32m1(acc6, a06, b0, jj_vl);
        acc7 = __riscv_vfmacc_vf_f32m1(acc7, a07, b0, jj_vl);
      }
      c00 =
          __riscv_vle32_v_f32m1(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), jj_vl);
      c01 =
          __riscv_vle32_v_f32m1(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), jj_vl);
      c02 =
          __riscv_vle32_v_f32m1(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), jj_vl);
      c03 =
          __riscv_vle32_v_f32m1(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), jj_vl);
      c04 =
          __riscv_vle32_v_f32m1(c + (((ii + 4) * rsc) + ((jj + 0) * 1)), jj_vl);
      c05 =
          __riscv_vle32_v_f32m1(c + (((ii + 5) * rsc) + ((jj + 0) * 1)), jj_vl);
      c06 =
          __riscv_vle32_v_f32m1(c + (((ii + 6) * rsc) + ((jj + 0) * 1)), jj_vl);
      c07 =
          __riscv_vle32_v_f32m1(c + (((ii + 7) * rsc) + ((jj + 0) * 1)), jj_vl);
      c00 = __riscv_vfmul_vf_f32m1(c00, _beta, jj_vl);
      c01 = __riscv_vfmul_vf_f32m1(c01, _beta, jj_vl);
      c02 = __riscv_vfmul_vf_f32m1(c02, _beta, jj_vl);
      c03 = __riscv_vfmul_vf_f32m1(c03, _beta, jj_vl);
      c04 = __riscv_vfmul_vf_f32m1(c04, _beta, jj_vl);
      c05 = __riscv_vfmul_vf_f32m1(c05, _beta, jj_vl);
      c06 = __riscv_vfmul_vf_f32m1(c06, _beta, jj_vl);
      c07 = __riscv_vfmul_vf_f32m1(c07, _beta, jj_vl);
      c00 = __riscv_vfmacc_vf_f32m1(c00, _alpha, acc0, jj_vl);
      c01 = __riscv_vfmacc_vf_f32m1(c01, _alpha, acc1, jj_vl);
      c02 = __riscv_vfmacc_vf_f32m1(c02, _alpha, acc2, jj_vl);
      c03 = __riscv_vfmacc_vf_f32m1(c03, _alpha, acc3, jj_vl);
      c04 = __riscv_vfmacc_vf_f32m1(c04, _alpha, acc4, jj_vl);
      c05 = __riscv_vfmacc_vf_f32m1(c05, _alpha, acc5, jj_vl);
      c06 = __riscv_vfmacc_vf_f32m1(c06, _alpha, acc6, jj_vl);
      c07 = __riscv_vfmacc_vf_f32m1(c07, _alpha, acc7, jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c00,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 1) * rsc) + ((jj + 0) * 1)), c01,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 2) * rsc) + ((jj + 0) * 1)), c02,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 3) * rsc) + ((jj + 0) * 1)), c03,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 4) * rsc) + ((jj + 0) * 1)), c04,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 5) * rsc) + ((jj + 0) * 1)), c05,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 6) * rsc) + ((jj + 0) * 1)), c06,
                            jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii + 7) * rsc) + ((jj + 0) * 1)), c07,
                            jj_vl);
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m1(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk_peel + 0) * 1)];
        b0 = __riscv_vle32_v_f32m1(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc = __riscv_vfmul_vf_f32m1(b0, a0, jj_vl);
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        a0 = a[((ii0 + 0) * rsa) + ((kk + 0) * 1)];
        b0 = __riscv_vle32_v_f32m1(b + (((kk + 0) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a0, b0, jj_vl);
      }
      c0 = __riscv_vle32_v_f32m1(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)),
                                 jj_vl);
      c0 = __riscv_vfmul_vf_f32m1(c0, _beta, jj_vl);
      c0 = __riscv_vfmacc_vf_f32m1(c0, _alpha, acc, jj_vl);
      __riscv_vse32_v_f32m1(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)), c0,
                            jj_vl);
    }
  }
}

/**
 * @brief RVV float32 matrix-matrix multiplication (SGEMM) for row-major
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
 * Computes `C = alpha * A * B + beta * C` for FP32 row-major matrices.
 *
 * Functionally equivalent to the scalar call:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_scalar(
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
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_f32_f32_f32_zve32f_x390(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    const float *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl = __riscv_vsetvlmax_e32m4();
  size_t ii;
  size_t jj;
  size_t kk_peel;
  size_t kk;
  size_t ii0;
  float alpha0;
  float beta0;
  vfloat32m4_t b0;
  vfloat32m4_t c00;
  vfloat32m4_t c01;
  vfloat32m4_t c02;
  vfloat32m4_t c03;
  float a0;
  vfloat32m4_t acc;
  vfloat32m4_t c0;
  alpha0 = alpha;
  beta0 = beta;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m4(n - jj);
        c0 = __riscv_vle32_v_f32m4(c + (((ii + 0) * rsc) + jj), jj_vl);
        c0 = __riscv_vfmul_vf_f32m4(c0, beta0, jj_vl);
        __riscv_vse32_v_f32m4(c + (((ii + 0) * rsc) + jj), c0, jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m4(n - jj);
      size_t jj_vl_m1_0 = __riscv_vsetvl_e32m1(jj_vl);
      size_t jj_vl_m1_1 = __riscv_vsetvl_e32m1(jj_vl - jj_vl_m1_0);
      size_t jj_vl_m1_2 = __riscv_vsetvl_e32m1(jj_vl - jj_vl_m1_0 - jj_vl_m1_1);
      size_t jj_vl_m1_3 =
          __riscv_vsetvl_e32m1(jj_vl - jj_vl_m1_0 - jj_vl_m1_1 - jj_vl_m1_2);

      vfloat32m4_t acc0 = __riscv_vundefined_f32m4();
      vfloat32m1_t acc00 = __riscv_vget_v_f32m4_f32m1(acc0, 0);
      vfloat32m1_t acc01 = __riscv_vget_v_f32m4_f32m1(acc0, 1);
      vfloat32m1_t acc02 = __riscv_vget_v_f32m4_f32m1(acc0, 2);
      vfloat32m1_t acc03 = __riscv_vget_v_f32m4_f32m1(acc0, 3);
      vfloat32m4_t acc1 = __riscv_vundefined_f32m4();
      vfloat32m1_t acc10 = __riscv_vget_v_f32m4_f32m1(acc1, 0);
      vfloat32m1_t acc11 = __riscv_vget_v_f32m4_f32m1(acc1, 1);
      vfloat32m1_t acc12 = __riscv_vget_v_f32m4_f32m1(acc1, 2);
      vfloat32m1_t acc13 = __riscv_vget_v_f32m4_f32m1(acc1, 3);
      vfloat32m4_t acc2 = __riscv_vundefined_f32m4();
      vfloat32m1_t acc20 = __riscv_vget_v_f32m4_f32m1(acc2, 0);
      vfloat32m1_t acc21 = __riscv_vget_v_f32m4_f32m1(acc2, 1);
      vfloat32m1_t acc22 = __riscv_vget_v_f32m4_f32m1(acc2, 2);
      vfloat32m1_t acc23 = __riscv_vget_v_f32m4_f32m1(acc2, 3);
      vfloat32m4_t acc3 = __riscv_vundefined_f32m4();
      vfloat32m1_t acc30 = __riscv_vget_v_f32m4_f32m1(acc3, 0);
      vfloat32m1_t acc31 = __riscv_vget_v_f32m4_f32m1(acc3, 1);
      vfloat32m1_t acc32 = __riscv_vget_v_f32m4_f32m1(acc3, 2);
      vfloat32m1_t acc33 = __riscv_vget_v_f32m4_f32m1(acc3, 3);

      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        size_t vl_max;
        __asm__ volatile(
            "\n\t"
            "vsetvli zero, %[jj_vl_m4], e32, m4, ta, ma \n\t"
            "vle32.v v4, (%[b_load]) \n\t"
            "vsetvli %[vl_max], zero, e32, m1, ta, ma \n\t"
            "vlse32.v v8, (%[a_load]), zero \n\t"
            "vlse32.v v9, (%[a_load_0]), zero \n\t"
            "vlse32.v v10, (%[a_load_1]), zero \n\t"
            "vlse32.v v11, (%[a_load_2]), zero \n\t"
            "vsetvli zero, %[jj_vl_m1_0], e32, m1, ta, ma \n\t"
            "vfmul.vv %[acc00], v4, v8 \n\t"
            "vfmul.vv %[acc10], v4, v9 \n\t"
            "vfmul.vv %[acc20], v4, v10 \n\t"
            "vfmul.vv %[acc30], v4, v11 \n\t"
            "vsetvli zero, %[jj_vl_m1_1], e32, m1, ta, ma \n\t"
            "vfmul.vv %[acc01], v5, v8 \n\t"
            "vfmul.vv %[acc11], v5, v9 \n\t"
            "vfmul.vv %[acc21], v5, v10 \n\t"
            "vfmul.vv %[acc31], v5, v11 \n\t"
            "vsetvli zero, %[jj_vl_m1_2], e32, m1, ta, ma \n\t"
            "vfmul.vv %[acc02], v6, v8 \n\t"
            "vfmul.vv %[acc12], v6, v9 \n\t"
            "vfmul.vv %[acc22], v6, v10 \n\t"
            "vfmul.vv %[acc32], v6, v11 \n\t"
            "vsetvli zero, %[jj_vl_m1_3], e32, m1, ta, ma \n\t"
            "vfmul.vv %[acc03], v7, v8 \n\t"
            "vfmul.vv %[acc13], v7, v9 \n\t"
            "vfmul.vv %[acc23], v7, v10 \n\t"
            "vfmul.vv %[acc33], v7, v11 \n\t"
            :
            [vl_max] "=r"(vl_max), [acc00] "+&vr"(acc00), [acc01] "+&vr"(acc01),
            [acc02] "+&vr"(acc02), [acc03] "+&vr"(acc03), [acc10] "+&vr"(acc10),
            [acc11] "+&vr"(acc11), [acc12] "+&vr"(acc12), [acc13] "+&vr"(acc13),
            [acc20] "+&vr"(acc20), [acc21] "+&vr"(acc21), [acc22] "+&vr"(acc22),
            [acc23] "+&vr"(acc23), [acc30] "+&vr"(acc30), [acc31] "+&vr"(acc31),
            [acc32] "+&vr"(acc32), [acc33] "+vr"(acc33)
            : [a_load] "r"(a + (((ii + 0) * rsa) + ((kk_peel + 0) * 1))),
              [a_load_0] "r"(a + (((ii + 1) * rsa) + ((kk_peel + 0) * 1))),
              [a_load_1] "r"(a + (((ii + 2) * rsa) + ((kk_peel + 0) * 1))),
              [a_load_2] "r"(a + (((ii + 3) * rsa) + ((kk_peel + 0) * 1))),
              [jj_vl_m4] "r"(jj_vl), [jj_vl_m1_0] "r"(jj_vl_m1_0),
              [jj_vl_m1_1] "r"(jj_vl_m1_1), [jj_vl_m1_2] "r"(jj_vl_m1_2),
              [jj_vl_m1_3] "r"(jj_vl_m1_3),
              [b_load] "r"(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11");
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        size_t vl_max;
        __asm__ volatile(
            "\n\t"
            "vsetvli zero, %[jj_vl_m4], e32, m4, ta, ma \n\t"
            "vle32.v v4, (%[b_load_0]) \n\t"
            "vsetvli %[vl_max], zero, e32, m1, ta, ma \n\t"
            "vlse32.v v8, (%[a_load_3]), zero \n\t"
            "vlse32.v v9, (%[a_load_4]), zero \n\t"
            "vlse32.v v10, (%[a_load_5]), zero \n\t"
            "vlse32.v v11, (%[a_load_6]), zero \n\t"
            "vsetvli zero, %[jj_vl_m1_0], e32, m1, ta, ma \n\t"
            "vfmacc.vv %[acc00], v8, v4 \n\t"
            "vfmacc.vv %[acc10], v9, v4 \n\t"
            "vfmacc.vv %[acc20], v10, v4 \n\t"
            "vfmacc.vv %[acc30], v11, v4 \n\t"
            "vsetvli zero, %[jj_vl_m1_1], e32, m1, ta, ma \n\t"
            "vfmacc.vv %[acc01], v8, v5 \n\t"
            "vfmacc.vv %[acc11], v9, v5 \n\t"
            "vfmacc.vv %[acc21], v10, v5 \n\t"
            "vfmacc.vv %[acc31], v11, v5 \n\t"
            "vsetvli zero, %[jj_vl_m1_2], e32, m1, ta, ma \n\t"
            "vfmacc.vv %[acc02], v8, v6 \n\t"
            "vfmacc.vv %[acc12], v9, v6 \n\t"
            "vfmacc.vv %[acc22], v10, v6 \n\t"
            "vfmacc.vv %[acc32], v11, v6 \n\t"
            "vsetvli zero, %[jj_vl_m1_3], e32, m1, ta, ma \n\t"
            "vfmacc.vv %[acc03], v8, v7 \n\t"
            "vfmacc.vv %[acc13], v9, v7 \n\t"
            "vfmacc.vv %[acc23], v10, v7 \n\t"
            "vfmacc.vv %[acc33], v11, v7 \n\t"
            :
            [vl_max] "=r"(vl_max), [acc00] "+&vr"(acc00), [acc01] "+&vr"(acc01),
            [acc02] "+&vr"(acc02), [acc03] "+&vr"(acc03), [acc10] "+&vr"(acc10),
            [acc11] "+&vr"(acc11), [acc12] "+&vr"(acc12), [acc13] "+&vr"(acc13),
            [acc20] "+&vr"(acc20), [acc21] "+&vr"(acc21), [acc22] "+&vr"(acc22),
            [acc23] "+&vr"(acc23), [acc30] "+&vr"(acc30), [acc31] "+&vr"(acc31),
            [acc32] "+&vr"(acc32), [acc33] "+vr"(acc33)
            : [a_load_3] "r"(a + (((ii + 0) * rsa) + ((kk + 0) * 1))),
              [a_load_4] "r"(a + (((ii + 1) * rsa) + ((kk + 0) * 1))),
              [a_load_5] "r"(a + (((ii + 2) * rsa) + ((kk + 0) * 1))),
              [a_load_6] "r"(a + (((ii + 3) * rsa) + ((kk + 0) * 1))),
              [jj_vl_m4] "r"(n - jj), [jj_vl_m1_0] "r"(jj_vl_m1_0),
              [jj_vl_m1_1] "r"(jj_vl_m1_1), [jj_vl_m1_2] "r"(jj_vl_m1_2),
              [jj_vl_m1_3] "r"(jj_vl_m1_3),
              [b_load_0] "r"(b + (((kk + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11");
      }
      acc0 = __riscv_vset_v_f32m1_f32m4(acc0, 0, acc00);
      acc0 = __riscv_vset_v_f32m1_f32m4(acc0, 1, acc01);
      acc0 = __riscv_vset_v_f32m1_f32m4(acc0, 2, acc02);
      acc0 = __riscv_vset_v_f32m1_f32m4(acc0, 3, acc03);
      acc1 = __riscv_vset_v_f32m1_f32m4(acc1, 0, acc10);
      acc1 = __riscv_vset_v_f32m1_f32m4(acc1, 1, acc11);
      acc1 = __riscv_vset_v_f32m1_f32m4(acc1, 2, acc12);
      acc1 = __riscv_vset_v_f32m1_f32m4(acc1, 3, acc13);
      acc2 = __riscv_vset_v_f32m1_f32m4(acc2, 0, acc20);
      acc2 = __riscv_vset_v_f32m1_f32m4(acc2, 1, acc21);
      acc2 = __riscv_vset_v_f32m1_f32m4(acc2, 2, acc22);
      acc2 = __riscv_vset_v_f32m1_f32m4(acc2, 3, acc23);
      acc3 = __riscv_vset_v_f32m1_f32m4(acc3, 0, acc30);
      acc3 = __riscv_vset_v_f32m1_f32m4(acc3, 1, acc31);
      acc3 = __riscv_vset_v_f32m1_f32m4(acc3, 2, acc32);
      acc3 = __riscv_vset_v_f32m1_f32m4(acc3, 3, acc33);
      __asm__ volatile(
          "\n\t"
          "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
          "vle32.v %[c00], (%[c_load]) \n\t"
          "vle32.v %[c01], (%[c_load_0]) \n\t"
          "vle32.v %[c02], (%[c_load_1]) \n\t"
          "vle32.v %[c03], (%[c_load_2]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta0] \n\t"
          "vfmacc.vf %[c00], %[alpha0], %[acc0] \n\t"
          "vse32.v %[c00], (%[c_store]) \n\t"
          "vfmul.vf %[c01], %[c01], %[beta0] \n\t"
          "vfmacc.vf %[c01], %[alpha0], %[acc1] \n\t"
          "vse32.v %[c01], (%[c_store_0]) \n\t"
          "vfmul.vf %[c02], %[c02], %[beta0] \n\t"
          "vfmacc.vf %[c02], %[alpha0], %[acc2] \n\t"
          "vse32.v %[c02], (%[c_store_1]) \n\t"
          "vfmul.vf %[c03], %[c03], %[beta0] \n\t"
          "vfmacc.vf %[c03], %[alpha0], %[acc3] \n\t"
          "vse32.v %[c03], (%[c_store_2]) \n\t"
          : [jj_vl_out] "=&r"(jj_vl), [c00] "=&vr"(c00), [c01] "=&vr"(c01),
            [c02] "=&vr"(c02), [c03] "=&vr"(c03)
          : [jj_vl_in] "r"(n - jj),
            [c_load] "r"(c + (((ii + 0) * rsc) + ((jj + 0) * 1))),
            [c_load_0] "r"(c + (((ii + 1) * rsc) + ((jj + 0) * 1))),
            [c_load_1] "r"(c + (((ii + 2) * rsc) + ((jj + 0) * 1))),
            [c_load_2] "r"(c + (((ii + 3) * rsc) + ((jj + 0) * 1))),
            [beta0] "f"(beta0), [alpha0] "f"(alpha0), [acc0] "vr"(acc0),
            [acc1] "vr"(acc1), [acc2] "vr"(acc2), [acc3] "vr"(acc3),
            [c_store] "r"(c + (((ii + 0) * rsc) + ((jj + 0) * 1))),
            [c_store_0] "r"(c + (((ii + 1) * rsc) + ((jj + 0) * 1))),
            [c_store_1] "r"(c + (((ii + 2) * rsc) + ((jj + 0) * 1))),
            [c_store_2] "r"(c + (((ii + 3) * rsc) + ((jj + 0) * 1)))
          : "vtype", "vl", "memory");
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a0], 0(%[a_load_7]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load_1]) \n\t"
            "vfmul.vf %[acc], %[b0], %[a0] \n\t"
            : [a0] "=&f"(a0), [jj_vl_out] "=&r"(jj_vl), [b0] "=&vr"(b0),
              [acc] "=vr"(acc)
            : [a_load_7] "r"(a + (((ii0 + 0) * rsa) + ((kk_peel + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_1] "r"(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a0], 0(%[a_load_8]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load_2]) \n\t"
            "vfmacc.vf %[acc], %[a0], %[b0] \n\t"
            : [a0] "=&f"(a0), [jj_vl_out] "=&r"(jj_vl), [b0] "=&vr"(b0),
              [acc] "+vr"(acc)
            : [a_load_8] "r"(a + (((ii0 + 0) * rsa) + ((kk + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_2] "r"(b + (((kk + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
          "vle32.v %[c0], (%[c_load_3]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta0] \n\t"
          "vfmacc.vf %[c0], %[alpha0], %[acc] \n\t"
          "vse32.v %[c0], (%[c_store_3]) \n\t"
          : [jj_vl_out] "=&r"(jj_vl), [c0] "=&vr"(c0)
          : [jj_vl_in] "r"(n - jj),
            [c_load_3] "r"(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1))),
            [beta0] "f"(beta0), [alpha0] "f"(alpha0), [acc] "vr"(acc),
            [c_store_3] "r"(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1)))
          : "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_gemm_f32_f32_f32_zve32f_x390(size_t m, size_t n, size_t k,
                                               float alpha, const float *a,
                                               size_t rsa, const float *b,
                                               size_t rsb, float beta, float *c,
                                               size_t rsc) {
  if (m == 1) {
    skl_gemm_1x2m8x2_f32_f32_f32_zve32f_x390(n, k, alpha, a, b, rsb, beta, c);
    return;
  }
  if (n <= __riscv_vsetvlmax_e32m2()) {
    skl_gemm_8xm1x4_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb,
                                            beta, c, rsc);
    return;
  }
  skl_gemm_4xm4x1_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb, beta,
                                          c, rsc);
}
