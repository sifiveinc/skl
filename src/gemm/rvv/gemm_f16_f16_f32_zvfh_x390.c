// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zvfh) || __riscv_zvfh < 1000000
#error This file requires the RISC-V Zvfh extension, version 1000000.
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

/**
 * @brief RVV float16 vector-matrix multiply with float32 output for row-major
 * matrices, tuned for X390.
 *
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scalar multiplier for A*B product.
 * @param a - Pointer to vector A.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to matrix C.
 *
 * Computes `C = alpha * A * B + beta * C` for FP16 unit-stride vector A and
 * row-major matrix B and FP32 unit-stride output vector C.
 * Functionally equivalent to scalar call:
 * ```
 * skl_gemm_f16rc_f16rc_f32rc_scalar(
 *     1, n, k,
 *     alpha,
 *     a, 1, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, 1, 1
 * );
 * ```
 *
 * Uses a 1 x LMUL=8 x 16 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `n >= __riscv_vsetvlmax_e32m8()`.
 */
SKL_FUNC_PRIVATE void
skl_gemm_1xm8x16_f16_f16_f32_zvfh_x390(size_t n, size_t k, float alpha,
                                       const _Float16 *a, const _Float16 *b,
                                       size_t rsb, float beta, float *c) {
  size_t jj_vl;
  size_t jj;
  size_t kk_peel;
  size_t kk;
  size_t kk0;
  float _alpha;
  float _beta;
  _Float16 a0;
  vfloat16m4_t b0;
  vfloat32m8_t acc;
  _Float16 a00;
  _Float16 a01;
  _Float16 a02;
  _Float16 a03;
  _Float16 a04;
  _Float16 a05;
  _Float16 a06;
  _Float16 a07;
  _Float16 a08;
  _Float16 a09;
  _Float16 a010;
  _Float16 a011;
  _Float16 a012;
  _Float16 a013;
  _Float16 a014;
  _Float16 a015;
  vfloat16m4_t b00;
  vfloat16m4_t b01;
  vfloat16m4_t b02;
  vfloat16m4_t b03;
  vfloat16m4_t b04;
  vfloat16m4_t b05;
  vfloat16m4_t b06;
  vfloat16m4_t b07;
  vfloat16m4_t b08;
  vfloat16m4_t b09;
  vfloat16m4_t b010;
  vfloat16m4_t b011;
  vfloat16m4_t b012;
  vfloat16m4_t b013;
  vfloat16m4_t b014;
  vfloat16m4_t b015;
  vfloat32m8_t c0;
  _alpha = alpha;
  _beta = beta;
  if (k == 0) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m8(n - jj);
      c0 = __riscv_vle32_v_f32m8(c + ((jj + 0) * 1), jj_vl);
      c0 = __riscv_vfmul_vf_f32m8(c0, _beta, jj_vl);
      __riscv_vse32_v_f32m8(c + ((jj + 0) * 1), c0, jj_vl);
    }
    return;
  }
  for (jj = 0; jj < n; jj = jj + jj_vl) {
    jj_vl = __riscv_vsetvl_e16m4(n - jj);
    for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
         kk_peel = kk_peel + 1) {
      a0 = a[(kk_peel + 0) * 1];
      b0 = __riscv_vle16_v_f16m4(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)),
                                 jj_vl);
      acc = __riscv_vfwmul_vf_f32m8(b0, a0, jj_vl);
    }
    for (kk = kk_peel; (kk + 16) <= k; kk = kk + 16) {
      a00 = a[(kk + 0) * 1];
      a01 = a[(kk + 1) * 1];
      a02 = a[(kk + 2) * 1];
      a03 = a[(kk + 3) * 1];
      a04 = a[(kk + 4) * 1];
      a05 = a[(kk + 5) * 1];
      a06 = a[(kk + 6) * 1];
      a07 = a[(kk + 7) * 1];
      a08 = a[(kk + 8) * 1];
      a09 = a[(kk + 9) * 1];
      a010 = a[(kk + 10) * 1];
      a011 = a[(kk + 11) * 1];
      a012 = a[(kk + 12) * 1];
      a013 = a[(kk + 13) * 1];
      a014 = a[(kk + 14) * 1];
      a015 = a[(kk + 15) * 1];
      skl_instruction_schedule_barrier();
      b00 =
          __riscv_vle16_v_f16m4(b + (((kk + 0) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a00, b00, jj_vl);
      b01 =
          __riscv_vle16_v_f16m4(b + (((kk + 1) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a01, b01, jj_vl);
      b02 =
          __riscv_vle16_v_f16m4(b + (((kk + 2) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a02, b02, jj_vl);
      b03 =
          __riscv_vle16_v_f16m4(b + (((kk + 3) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a03, b03, jj_vl);
      b04 =
          __riscv_vle16_v_f16m4(b + (((kk + 4) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a04, b04, jj_vl);
      b05 =
          __riscv_vle16_v_f16m4(b + (((kk + 5) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a05, b05, jj_vl);
      b06 =
          __riscv_vle16_v_f16m4(b + (((kk + 6) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a06, b06, jj_vl);
      b07 =
          __riscv_vle16_v_f16m4(b + (((kk + 7) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a07, b07, jj_vl);
      b08 =
          __riscv_vle16_v_f16m4(b + (((kk + 8) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a08, b08, jj_vl);
      b09 =
          __riscv_vle16_v_f16m4(b + (((kk + 9) * rsb) + ((jj + 0) * 1)), jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a09, b09, jj_vl);
      b010 = __riscv_vle16_v_f16m4(b + (((kk + 10) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a010, b010, jj_vl);
      b011 = __riscv_vle16_v_f16m4(b + (((kk + 11) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a011, b011, jj_vl);
      b012 = __riscv_vle16_v_f16m4(b + (((kk + 12) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a012, b012, jj_vl);
      b013 = __riscv_vle16_v_f16m4(b + (((kk + 13) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a013, b013, jj_vl);
      b014 = __riscv_vle16_v_f16m4(b + (((kk + 14) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a014, b014, jj_vl);
      b015 = __riscv_vle16_v_f16m4(b + (((kk + 15) * rsb) + ((jj + 0) * 1)),
                                   jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a015, b015, jj_vl);
    }
    for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
      a0 = a[(kk0 + 0) * 1];
      b0 = __riscv_vle16_v_f16m4(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1)),
                                 jj_vl);
      acc = __riscv_vfwmacc_vf_f32m8(acc, a0, b0, jj_vl);
    }
    c0 = __riscv_vle32_v_f32m8(c + ((jj + 0) * 1), jj_vl);
    c0 = __riscv_vfmul_vf_f32m8(c0, _beta, jj_vl);
    c0 = __riscv_vfmacc_vf_f32m8(c0, _alpha, acc, jj_vl);
    __riscv_vse32_v_f32m8(c + ((jj + 0) * 1), c0, jj_vl);
  }
}

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
 * Uses a 4 x LMUL=4 x 1 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e32m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x1_f16_f16_f32_zvfh_x390(
    size_t m, size_t n, size_t k, float alpha, const _Float16 *a, size_t rsa,
    const _Float16 *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t ii;
  size_t jj;
  size_t kk;
  size_t jj_vl;
  vfloat32m8_t acc0;
  vfloat32m8_t acc1;
  vfloat16m4_t b00;
  _Float16 a00;
  _Float16 a10;
  _Float16 a01;
  _Float16 a11;
  _Float16 a02;
  _Float16 a12;
  _Float16 a03;
  _Float16 a13;
  _Float16 a04;
  _Float16 a14;
  _Float16 a05;
  _Float16 a15;
  _Float16 a06;
  _Float16 a16;
  _Float16 a07;
  _Float16 a17;
  vfloat32m8_t c00;
  vfloat32m8_t c01;

  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m8(n - jj);
        c00 = __riscv_vle32_v_f32m8(c + ii * rsc + jj, jj_vl);
        c00 = __riscv_vfmul_vf_f32m8(c00, beta, jj_vl);
        __riscv_vse32_v_f32m8(c + ii * rsc + jj, c00, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 2) <= m; ii += 2) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "flh %[a00], 0(%[a_addr_0]) \n\t"
          "flh %[a01], 0(%[a_addr_1]) \n\t"
          "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
          "vle16.v %[b00], (%[b_addr]) \n\t"
          "vfwmul.vf %[acc0], %[b00], %[a00] \n\t"
          "vfwmul.vf %[acc1], %[b00], %[a01] \n\t"
          : [a00] "=&f"(a00),
            [a01] "=&f"(a01),
            [b00] "=&vr"(b00),
            [acc0] "=&vr"(acc0),
            [acc1] "=vr"(acc1)
          : [jj_vl_in] "r"(jj_vl),
            [a_addr_0] "r"(a + (ii + 0) * rsa),
            [a_addr_1] "r"(a + (ii + 1) * rsa),
            [b_addr] "r"(b + jj),
            [rsa2] "r" (rsa * sizeof(_Float16))
          : "vtype", "vl", "memory"
          // clang-format on
      );

      for (kk = 1; (kk + 8) <= k; kk += 8) {
        const _Float16 *b_addr = b + (kk + 0) * rsb + jj;
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"

            "flh %[a00],  0(%[a_addr_0]) \n\t"
            "flh %[a10],  0(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"

            "flh %[a01],  2(%[a_addr_0]) \n\t"
            "flh %[a11],  2(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a01], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a11], %[b00] \n\t"

            "flh %[a02],  4(%[a_addr_0]) \n\t"
            "flh %[a12],  4(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a02], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a12], %[b00] \n\t"

            "flh %[a03],  6(%[a_addr_0]) \n\t"
            "flh %[a13],  6(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a03], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a13], %[b00] \n\t"

            "flh %[a04],  8(%[a_addr_0]) \n\t"
            "flh %[a14],  8(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a04], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a14], %[b00] \n\t"

            "flh %[a05], 10(%[a_addr_0]) \n\t"
            "flh %[a15], 10(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a05], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a15], %[b00] \n\t"

            "flh %[a06], 12(%[a_addr_0]) \n\t"
            "flh %[a16], 12(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            "vfwmacc.vf %[acc0], %[a06], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a16], %[b00] \n\t"

            "flh %[a07], 14(%[a_addr_0]) \n\t"
            "flh %[a17], 14(%[a_addr_1]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "vfwmacc.vf %[acc0], %[a07], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a17], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [a01] "=&f"(a01),
              [a02] "=&f"(a02),
              [a03] "=&f"(a03),
              [a04] "=&f"(a04),
              [a05] "=&f"(a05),
              [a06] "=&f"(a06),
              [a07] "=&f"(a07),
              [a10] "=&f"(a10),
              [a11] "=&f"(a11),
              [a12] "=&f"(a12),
              [a13] "=&f"(a13),
              [a14] "=&f"(a14),
              [a15] "=&f"(a15),
              [a16] "=&f"(a16),
              [a17] "=&f"(a17),
              [b00] "=&vr"(b00),
              [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1),
              [b_addr] "+&r"(b_addr)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr_0]  "r"(a + (ii + 0) * rsa + kk),
              [a_addr_1]  "r"(a + (ii + 1) * rsa + kk),
              [rsb2] "r"(rsb * sizeof(_Float16))
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      for (; (kk + 1) <= k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "flh %[a00], 0(%[a_addr_0]) \n\t"
            "flh %[a10], 0(%[a_addr_1]) \n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [a10] "=&f"(a10),
              [b00] "=&vr"(b00),
              [acc0] "+&vr"(acc0),
              [acc1] "+vr"(acc1)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr_0] "r"(a + (ii + 0) * rsa + kk),
              [a_addr_1] "r"(a + (ii + 1) * rsa + kk),
              [b_addr] "r"(b + (kk + 0) * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
          "vle32.v %[c00], (%[c_addr_0]) \n\t"
          "vle32.v %[c01], (%[c_addr_1]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmul.vf %[c01], %[c01], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vfmacc.vf %[c01], %[alpha], %[acc1] \n\t"
          "vse32.v %[c00], (%[c_addr_0]) \n\t"
          "vse32.v %[c01], (%[c_addr_1]) \n\t"
          : [c00] "=&vr"(c00),
            [c01] "=&vr"(c01)
          : [jj_vl_in] "r"(jj_vl),
            [c_addr_0] "r"(c + (ii + 0) * rsc + jj),
            [c_addr_1] "r"(c + (ii + 1) * rsc + jj),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc0] "vr"(acc0),
            [acc1] "vr"(acc1)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }

  for (; (ii + 1) <= m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "flh %[a00], 0(%[a_addr]) \n\t"
          "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
          "vle16.v %[b00], (%[b_addr]) \n\t"
          "vfwmul.vf %[acc0], %[b00], %[a00] \n\t"
          : [a00] "=&f"(a00),
            [b00] "=&vr"(b00),
            [acc0] "=vr"(acc0)
          : [jj_vl_in] "r"(jj_vl),
            [a_addr] "r"(a + ii * rsa),
            [b_addr] "r"(b + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      for (kk = 1; (kk + 1) <= k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "flh %[a00], 0(%[a_addr]) \n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [b00] "=&vr"(b00),
              [acc0] "+vr"(acc0)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr] "r"(a + ii * rsa + kk),
              [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"
          "vle32.v %[c00], (%[c_addr]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vse32.v %[c00], (%[c_addr]) \n\t"
          : [c00] "=&vr"(c00)
          : [jj_vl_in] "r"(jj_vl),
            [c_addr] "r"(c + ii * rsc + jj),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc0] "vr"(acc0)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }
}

SKL_FUNC void skl_gemm_f16_f16_f32_zvfh_x390(size_t m, size_t n, size_t k,
                                             float alpha, const _Float16 *a,
                                             size_t rsa, const _Float16 *b,
                                             size_t rsb, float beta, float *c,
                                             size_t rsc) {
  if (m == 1) {
    skl_gemm_1xm8x16_f16_f16_f32_zvfh_x390(n, k, alpha, a, b, rsb, beta, c);
    return;
  }
  skl_gemm_4xm4x1_f16_f16_f32_zvfh_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                        rsc);
}
