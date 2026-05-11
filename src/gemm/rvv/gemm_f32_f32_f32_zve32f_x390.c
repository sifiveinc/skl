// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

#if defined(__riscv_zihintntl)
#define NTL_P1 "ntl.p1 \n\t"
#else
#define NTL_P1
#endif

/**
 * @brief RVV float32 matrix-matrix multiplication (SGEMM) for row-major
 * matrices, tuned for X390.
 *
 * @param m - Number of rows in matrices A and C.
 * @param n - Number of columns in matrices B and C.
 * @param k - Number of columns in A and rows in B (inner dimension).
 * @param alpha - Scalar multiplier for A*B product.
 * @param a - Pointer to vector A.
 * @param rsa - Row stride of matrix A in elements.
 * @param b - Pointer to matrix B.
 * @param rsb - Row stride of matrix B in elements.
 * @param beta - Scalar multiplier for matrix C.
 * @param c - Pointer to vector C.
 * @param rsc - Row stride of matrix C in elements.
 *
 * Computes `C = alpha * A * B + beta * C` for FP32 row-major matrices.
 *
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 * Uses a 1 x LMUL=8 x 3 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m == 1` and `n >= __riscv_vsetvlmax_e32m8()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_1xm8x3_f32_f32_f32_zve32f_x390(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    const float *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  float a00;
  float a01;
  float a02;
  vfloat32m8_t b00;
  vfloat32m8_t b10;
  vfloat32m8_t b20;
  vfloat32m8_t acc0;
  vfloat32m8_t c00;

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

  for (ii = 0; ii < m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e32m8(n - jj);

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m8, ta, ma \n\t"

          "flw %[a00], 0(%[a_addr]) \n\t"
          "vle32.v %[b00], (%[b_addr]) \n\t"
          "vfmul.vf %[acc0], %[b00], %[a00] \n\t"
          : [a00] "=&f"(a00),
            [b00] "=&vr"(b00),
            [acc0] "=&vr"(acc0)
          : [jj_vl] "r"(jj_vl),
            [a_addr] "r"(a + ii * rsa + 0),
            [b_addr] "r"(b + 0 * rsb + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      const size_t preload_distance = 3;
      const size_t kk_unroll_degree = 12;
      kk = 1;

      if (kk + kk_unroll_degree + preload_distance <= k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m8, ta, ma \n\t"

            "flw %[a00], 0(%[a_addr]) \n\t"
            "flw %[a01], 4(%[a_addr]) \n\t"
            "flw %[a02], 8(%[a_addr]) \n\t"
            "vle32.v %[b00], (%[b_addr_0]) \n\t"
            "vle32.v %[b10], (%[b_addr_1]) \n\t"
            "vle32.v %[b20], (%[b_addr_2]) \n\t"
            : [a00] "=&f"(a00),
              [a01] "=&f"(a01),
              [a02] "=&f"(a02),
              [b00] "=&vr"(b00),
              [b10] "=&vr"(b10),
              [b20] "=vr"(b20)
            : [jj_vl] "r"(jj_vl),
              [a_addr] "r"(a + ii * rsa + kk),
              [b_addr_0] "r"(b + (kk + 0) * rsb + jj),
              [b_addr_1] "r"(b + (kk + 1) * rsb + jj),
              [b_addr_2] "r"(b + (kk + 2) * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      const float *b_addr = b + (kk + preload_distance + 0) * rsb + jj;

      for (; (kk + kk_unroll_degree + preload_distance) <= k;
           kk += kk_unroll_degree) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m8, ta, ma \n\t"

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a00],  0(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vle32.v %[b10], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a01],  4(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vle32.v %[b20], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a02],  8(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a00], 12(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vle32.v %[b10], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a01], 16(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vle32.v %[b20], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a02], 20(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a00], 24(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vle32.v %[b10], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a01], 28(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vle32.v %[b20], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a02], 32(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a00], 36(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vle32.v %[b10], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a01], 40(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vle32.v %[b20], (%[b_addr]) \n\t"
            NTL_P1
            "flw %[a02], 44(%[a_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t"
            : [a00] "+&f"(a00),
              [a01] "+&f"(a01),
              [a02] "+&f"(a02),
              [b00] "+&vr"(b00),
              [b10] "+&vr"(b10),
              [b20] "+&vr"(b20),
              [acc0] "+&vr"(acc0),
              [b_addr] "+&r"(b_addr)
            : [jj_vl] "r"(jj_vl),
              [a_addr] "r"(a + ii * rsa + kk + preload_distance),
              [rsb4] "r" (rsb * sizeof(float))
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      if (1 + kk_unroll_degree + preload_distance <= k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m8, ta, ma \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            : [acc0] "+&vr"(acc0)
            : [jj_vl] "r"(jj_vl),
              [a00] "f"(a00),
              [a01] "f"(a01),
              [a02] "f"(a02),
              [b00] "vr"(b00),
              [b10] "vr"(b10),
              [b20] "vr"(b20)
            : "vtype", "vl"
            // clang-format on
        );
        kk += preload_distance;
      }

      for (; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m8, ta, ma \n\t"
            "flw %[a00], 0(%[a_addr]) \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [b00] "=&vr"(b00),
              [acc0] "+&vr"(acc0)
            : [jj_vl] "r"(jj_vl),
              [a_addr] "r"(a + ii * rsa + kk),
              [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m8, ta, ma \n\t"

          "vle32.v %[c00], (%[c_addr]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vse32.v %[c00], (%[c_addr]) \n\t"
          : [c00] "=&vr"(c00)
          : [jj_vl] "r"(jj_vl),
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
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_ref(
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
  size_t kk;
  float a00;
  float a10;
  float a20;
  float a30;
  float a40;
  float a50;
  float a60;
  float a70;
  float a01;
  float a11;
  float a21;
  float a31;
  float a41;
  float a51;
  float a61;
  float a71;
  float a02;
  float a12;
  float a22;
  float a32;
  float a42;
  float a52;
  float a62;
  float a72;
  float a03;
  float a13;
  float a23;
  float a33;
  float a43;
  float a53;
  float a63;
  float a73;
  vfloat32m2_t acc0;
  vfloat32m2_t acc1;
  vfloat32m2_t acc2;
  vfloat32m2_t acc3;
  vfloat32m2_t acc4;
  vfloat32m2_t acc5;
  vfloat32m2_t acc6;
  vfloat32m2_t acc7;
  vfloat32m2_t b00;
  vfloat32m2_t b10;
  vfloat32m2_t b20;
  vfloat32m2_t b30;
  vfloat32m2_t c00;
  vfloat32m2_t c10;
  vfloat32m2_t c20;
  vfloat32m2_t c30;
  vfloat32m2_t c40;
  vfloat32m2_t c50;
  vfloat32m2_t c60;
  vfloat32m2_t c70;

  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m8(n - jj);
        vfloat32m8_t c0m8 = __riscv_vle32_v_f32m8(c + ii * rsc + jj, jj_vl);
        c0m8 = __riscv_vfmul_vf_f32m8(c0m8, beta, jj_vl);
        __riscv_vse32_v_f32m8(c + ii * rsc + jj, c0m8, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 8) <= m; ii = ii + 8) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m2(n - jj);

      const float *a_addr0 = a + (ii + 0) * rsa;
      const float *a_addr1 = a + (ii + 1) * rsa;
      const float *a_addr2 = a + (ii + 2) * rsa;
      const float *a_addr3 = a + (ii + 3) * rsa;
      const float *a_addr4 = a + (ii + 4) * rsa;
      const float *a_addr5 = a + (ii + 5) * rsa;
      const float *a_addr6 = a + (ii + 6) * rsa;
      const float *a_addr7 = a + (ii + 7) * rsa;

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"
          "vle32.v %[b00], (%[b_load]) \n\t"

          "flw %[a00], 0(%[a_addr0]) \n\t"
          "vfmul.vf %[acc0], %[b00], %[a00] \n\t"
          "addi %[a_addr0], %[a_addr0], 4 \n\t"

          "flw %[a10], 0(%[a_addr1]) \n\t"
          "vfmul.vf %[acc1], %[b00], %[a10] \n\t"
          "addi %[a_addr1], %[a_addr1], 4 \n\t"

          "flw %[a20], 0(%[a_addr2]) \n\t"
          "vfmul.vf %[acc2], %[b00], %[a20] \n\t"
          "addi %[a_addr2], %[a_addr2], 4 \n\t"

          "flw %[a30], 0(%[a_addr3]) \n\t"
          "vfmul.vf %[acc3], %[b00], %[a30] \n\t"
          "addi %[a_addr3], %[a_addr3], 4 \n\t"

          "flw %[a40], 0(%[a_addr4]) \n\t"
          "vfmul.vf %[acc4], %[b00], %[a40] \n\t"
          "addi %[a_addr4], %[a_addr4], 4 \n\t"

          "flw %[a50], 0(%[a_addr5]) \n\t"
          "vfmul.vf %[acc5], %[b00], %[a50] \n\t"
          "addi %[a_addr5], %[a_addr5], 4 \n\t"

          "flw %[a60], 0(%[a_addr6]) \n\t"
          "vfmul.vf %[acc6], %[b00], %[a60] \n\t"
          "addi %[a_addr6], %[a_addr6], 4 \n\t"

          "flw %[a70], 0(%[a_addr7]) \n\t"
          "vfmul.vf %[acc7], %[b00], %[a70] \n\t"
          "addi %[a_addr7], %[a_addr7], 4 \n\t"
          : [a00] "=&f"(a00),
            [a10] "=&f"(a10),
            [a20] "=&f"(a20),
            [a30] "=&f"(a30),
            [a40] "=&f"(a40),
            [a50] "=&f"(a50),
            [a60] "=&f"(a60),
            [a70] "=&f"(a70),
            [b00] "=&vr"(b00),
            [acc0] "=vr"(acc0),
            [acc1] "=vr"(acc1),
            [acc2] "=vr"(acc2),
            [acc3] "=vr"(acc3),
            [acc4] "=vr"(acc4),
            [acc5] "=vr"(acc5),
            [acc6] "=vr"(acc6),
            [acc7] "=vr"(acc7),
            [a_addr0] "+&r"(a_addr0),
            [a_addr1] "+&r"(a_addr1),
            [a_addr2] "+&r"(a_addr2),
            [a_addr3] "+&r"(a_addr3),
            [a_addr4] "+&r"(a_addr4),
            [a_addr5] "+&r"(a_addr5),
            [a_addr6] "+&r"(a_addr6),
            [a_addr7] "+&r"(a_addr7)
          : [jj_vl] "r"(jj_vl),
            [b_load] "r"(b + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      kk = 1;
      const float *b_addr0 = b + (kk + 0) * rsb + jj;
      const float *b_addr1 = b + (kk + 1) * rsb + jj;
      const float *b_addr2 = b + (kk + 2) * rsb + jj;
      const float *b_addr3 = b + (kk + 3) * rsb + jj;

      const size_t preload_distance = 4;
      const size_t kk_unroll_degree = 12;

      if (kk + kk_unroll_degree + preload_distance <= k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"
            "vle32.v %[b00], (%[b_addr0]) \n\t"
            "vle32.v %[b10], (%[b_addr1]) \n\t"
            "vle32.v %[b20], (%[b_addr2]) \n\t"
            "vle32.v %[b30], (%[b_addr3]) \n\t"

            "flw %[a00],  0(%[a_addr0]) \n\t"
            "flw %[a10],  0(%[a_addr1]) \n\t"
            "flw %[a20],  0(%[a_addr2]) \n\t"
            "flw %[a30],  0(%[a_addr3]) \n\t"
            "flw %[a40],  0(%[a_addr4]) \n\t"
            "flw %[a50],  0(%[a_addr5]) \n\t"
            "flw %[a60],  0(%[a_addr6]) \n\t"
            "flw %[a70],  0(%[a_addr7]) \n\t"

            "flw %[a01],  4(%[a_addr0]) \n\t"
            "flw %[a11],  4(%[a_addr1]) \n\t"
            "flw %[a21],  4(%[a_addr2]) \n\t"
            "flw %[a31],  4(%[a_addr3]) \n\t"
            "flw %[a41],  4(%[a_addr4]) \n\t"
            "flw %[a51],  4(%[a_addr5]) \n\t"
            "flw %[a61],  4(%[a_addr6]) \n\t"
            "flw %[a71],  4(%[a_addr7]) \n\t"

            "flw %[a02],  8(%[a_addr0]) \n\t"
            "flw %[a12],  8(%[a_addr1]) \n\t"
            "flw %[a22],  8(%[a_addr2]) \n\t"
            "flw %[a32],  8(%[a_addr3]) \n\t"
            "flw %[a42],  8(%[a_addr4]) \n\t"
            "flw %[a52],  8(%[a_addr5]) \n\t"
            "flw %[a62],  8(%[a_addr6]) \n\t"
            "flw %[a72],  8(%[a_addr7]) \n\t"

            "flw %[a03], 12(%[a_addr0]) \n\t"
            "flw %[a13], 12(%[a_addr1]) \n\t"
            "flw %[a23], 12(%[a_addr2]) \n\t"
            "flw %[a33], 12(%[a_addr3]) \n\t"
            "flw %[a43], 12(%[a_addr4]) \n\t"
            "flw %[a53], 12(%[a_addr5]) \n\t"
            "flw %[a63], 12(%[a_addr6]) \n\t"
            "flw %[a73], 12(%[a_addr7]) \n\t"

            "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"
            "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"
            "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"
            "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"
            "add %[a_addr4], %[a_addr4], %[a_inc] \n\t"
            "add %[a_addr5], %[a_addr5], %[a_inc] \n\t"
            "add %[a_addr6], %[a_addr6], %[a_inc] \n\t"
            "add %[a_addr7], %[a_addr7], %[a_inc] \n\t"

            "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
            "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"
            "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"
            "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

            : [a00] "=f"(a00),
              [a10] "=f"(a10),
              [a20] "=f"(a20),
              [a30] "=f"(a30),
              [a40] "=f"(a40),
              [a50] "=f"(a50),
              [a60] "=f"(a60),
              [a70] "=f"(a70),

              [a01] "=f"(a01),
              [a11] "=f"(a11),
              [a21] "=f"(a21),
              [a31] "=f"(a31),
              [a41] "=f"(a41),
              [a51] "=f"(a51),
              [a61] "=f"(a61),
              [a71] "=f"(a71),

              [a02] "=f"(a02),
              [a12] "=f"(a12),
              [a22] "=f"(a22),
              [a32] "=f"(a32),
              [a42] "=f"(a42),
              [a52] "=f"(a52),
              [a62] "=f"(a62),
              [a72] "=f"(a72),

              [a03] "=f"(a03),
              [a13] "=f"(a13),
              [a23] "=f"(a23),
              [a33] "=f"(a33),
              [a43] "=f"(a43),
              [a53] "=f"(a53),
              [a63] "=f"(a63),
              [a73] "=f"(a73),

              [b00] "=vr"(b00),
              [b10] "=vr"(b10),
              [b20] "=vr"(b20),
              [b30] "=vr"(b30),

              [a_addr0] "+&r"(a_addr0),
              [a_addr1] "+&r"(a_addr1),
              [a_addr2] "+&r"(a_addr2),
              [a_addr3] "+&r"(a_addr3),
              [a_addr4] "+&r"(a_addr4),
              [a_addr5] "+&r"(a_addr5),
              [a_addr6] "+&r"(a_addr6),
              [a_addr7] "+&r"(a_addr7),

              [b_addr0] "+&r"(b_addr0),
              [b_addr1] "+&r"(b_addr1),
              [b_addr2] "+&r"(b_addr2),
              [b_addr3] "+&r"(b_addr3)
            : [a_inc] "r"(sizeof(float) * preload_distance),
              [b_inc] "r"(sizeof(float) * preload_distance * rsb),
              [jj_vl] "r"(jj_vl)
            : "vtype", "vl", "memory"
            // clang-format on
        );

        for (; kk + kk_unroll_degree + preload_distance <= k;
             kk += kk_unroll_degree) {
          __asm__ volatile(
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
              // clang-format off
              "\n\t"
              "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"

              "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
              "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
              "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
              "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
              "vfmacc.vf %[acc4], %[a40], %[b00] \n\t"
              "vfmacc.vf %[acc5], %[a50], %[b00] \n\t"
              "vfmacc.vf %[acc6], %[a60], %[b00] \n\t"
              "vfmacc.vf %[acc7], %[a70], %[b00] \n\t"
              "flw %[a00],  0(%[a_addr0]) \n\t"
              "flw %[a10],  0(%[a_addr1]) \n\t"
              "flw %[a20],  0(%[a_addr2]) \n\t"
              "flw %[a30],  0(%[a_addr3]) \n\t"
              "flw %[a40],  0(%[a_addr4]) \n\t"
              "flw %[a50],  0(%[a_addr5]) \n\t"
              "flw %[a60],  0(%[a_addr6]) \n\t"
              "flw %[a70],  0(%[a_addr7]) \n\t"
              "vle32.v %[b00], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
              "vfmacc.vf %[acc1], %[a11], %[b10] \n\t"
              "vfmacc.vf %[acc2], %[a21], %[b10] \n\t"
              "vfmacc.vf %[acc3], %[a31], %[b10] \n\t"
              "vfmacc.vf %[acc4], %[a41], %[b10] \n\t"
              "vfmacc.vf %[acc5], %[a51], %[b10] \n\t"
              "vfmacc.vf %[acc6], %[a61], %[b10] \n\t"
              "vfmacc.vf %[acc7], %[a71], %[b10] \n\t"
              "flw %[a01],  4(%[a_addr0]) \n\t"
              "flw %[a11],  4(%[a_addr1]) \n\t"
              "flw %[a21],  4(%[a_addr2]) \n\t"
              "flw %[a31],  4(%[a_addr3]) \n\t"
              "flw %[a41],  4(%[a_addr4]) \n\t"
              "flw %[a51],  4(%[a_addr5]) \n\t"
              "flw %[a61],  4(%[a_addr6]) \n\t"
              "flw %[a71],  4(%[a_addr7]) \n\t"
              "vle32.v %[b10], (%[b_addr1]) \n\t"
              "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
              "vfmacc.vf %[acc1], %[a12], %[b20] \n\t"
              "vfmacc.vf %[acc2], %[a22], %[b20] \n\t"
              "vfmacc.vf %[acc3], %[a32], %[b20] \n\t"
              "vfmacc.vf %[acc4], %[a42], %[b20] \n\t"
              "vfmacc.vf %[acc5], %[a52], %[b20] \n\t"
              "vfmacc.vf %[acc6], %[a62], %[b20] \n\t"
              "vfmacc.vf %[acc7], %[a72], %[b20] \n\t"
              "flw %[a02],  8(%[a_addr0]) \n\t"
              "flw %[a12],  8(%[a_addr1]) \n\t"
              "flw %[a22],  8(%[a_addr2]) \n\t"
              "flw %[a32],  8(%[a_addr3]) \n\t"
              "flw %[a42],  8(%[a_addr4]) \n\t"
              "flw %[a52],  8(%[a_addr5]) \n\t"
              "flw %[a62],  8(%[a_addr6]) \n\t"
              "flw %[a72],  8(%[a_addr7]) \n\t"
              "vle32.v %[b20], (%[b_addr2]) \n\t"
              "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a03], %[b30] \n\t"
              "vfmacc.vf %[acc1], %[a13], %[b30] \n\t"
              "vfmacc.vf %[acc2], %[a23], %[b30] \n\t"
              "vfmacc.vf %[acc3], %[a33], %[b30] \n\t"
              "vfmacc.vf %[acc4], %[a43], %[b30] \n\t"
              "vfmacc.vf %[acc5], %[a53], %[b30] \n\t"
              "vfmacc.vf %[acc6], %[a63], %[b30] \n\t"
              "vfmacc.vf %[acc7], %[a73], %[b30] \n\t"
              "flw %[a03], 12(%[a_addr0]) \n\t"
              "flw %[a13], 12(%[a_addr1]) \n\t"
              "flw %[a23], 12(%[a_addr2]) \n\t"
              "flw %[a33], 12(%[a_addr3]) \n\t"
              "flw %[a43], 12(%[a_addr4]) \n\t"
              "flw %[a53], 12(%[a_addr5]) \n\t"
              "flw %[a63], 12(%[a_addr6]) \n\t"
              "flw %[a73], 12(%[a_addr7]) \n\t"
              "vle32.v %[b30], (%[b_addr3]) \n\t"
              "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
              "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
              "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
              "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
              "vfmacc.vf %[acc4], %[a40], %[b00] \n\t"
              "vfmacc.vf %[acc5], %[a50], %[b00] \n\t"
              "vfmacc.vf %[acc6], %[a60], %[b00] \n\t"
              "vfmacc.vf %[acc7], %[a70], %[b00] \n\t"
              "flw %[a00], 16(%[a_addr0]) \n\t"
              "flw %[a10], 16(%[a_addr1]) \n\t"
              "flw %[a20], 16(%[a_addr2]) \n\t"
              "flw %[a30], 16(%[a_addr3]) \n\t"
              "flw %[a40], 16(%[a_addr4]) \n\t"
              "flw %[a50], 16(%[a_addr5]) \n\t"
              "flw %[a60], 16(%[a_addr6]) \n\t"
              "flw %[a70], 16(%[a_addr7]) \n\t"
              "vle32.v %[b00], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
              "vfmacc.vf %[acc1], %[a11], %[b10] \n\t"
              "vfmacc.vf %[acc2], %[a21], %[b10] \n\t"
              "vfmacc.vf %[acc3], %[a31], %[b10] \n\t"
              "vfmacc.vf %[acc4], %[a41], %[b10] \n\t"
              "vfmacc.vf %[acc5], %[a51], %[b10] \n\t"
              "vfmacc.vf %[acc6], %[a61], %[b10] \n\t"
              "vfmacc.vf %[acc7], %[a71], %[b10] \n\t"
              "flw %[a01], 20(%[a_addr0]) \n\t"
              "flw %[a11], 20(%[a_addr1]) \n\t"
              "flw %[a21], 20(%[a_addr2]) \n\t"
              "flw %[a31], 20(%[a_addr3]) \n\t"
              "flw %[a41], 20(%[a_addr4]) \n\t"
              "flw %[a51], 20(%[a_addr5]) \n\t"
              "flw %[a61], 20(%[a_addr6]) \n\t"
              "flw %[a71], 20(%[a_addr7]) \n\t"
              "vle32.v %[b10], (%[b_addr1]) \n\t"
              "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
              "vfmacc.vf %[acc1], %[a12], %[b20] \n\t"
              "vfmacc.vf %[acc2], %[a22], %[b20] \n\t"
              "vfmacc.vf %[acc3], %[a32], %[b20] \n\t"
              "vfmacc.vf %[acc4], %[a42], %[b20] \n\t"
              "vfmacc.vf %[acc5], %[a52], %[b20] \n\t"
              "vfmacc.vf %[acc6], %[a62], %[b20] \n\t"
              "vfmacc.vf %[acc7], %[a72], %[b20] \n\t"
              "flw %[a02], 24(%[a_addr0]) \n\t"
              "flw %[a12], 24(%[a_addr1]) \n\t"
              "flw %[a22], 24(%[a_addr2]) \n\t"
              "flw %[a32], 24(%[a_addr3]) \n\t"
              "flw %[a42], 24(%[a_addr4]) \n\t"
              "flw %[a52], 24(%[a_addr5]) \n\t"
              "flw %[a62], 24(%[a_addr6]) \n\t"
              "flw %[a72], 24(%[a_addr7]) \n\t"
              "vle32.v %[b20], (%[b_addr2]) \n\t"
              "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a03], %[b30] \n\t"
              "vfmacc.vf %[acc1], %[a13], %[b30] \n\t"
              "vfmacc.vf %[acc2], %[a23], %[b30] \n\t"
              "vfmacc.vf %[acc3], %[a33], %[b30] \n\t"
              "vfmacc.vf %[acc4], %[a43], %[b30] \n\t"
              "vfmacc.vf %[acc5], %[a53], %[b30] \n\t"
              "vfmacc.vf %[acc6], %[a63], %[b30] \n\t"
              "vfmacc.vf %[acc7], %[a73], %[b30] \n\t"
              "flw %[a03], 28(%[a_addr0]) \n\t"
              "flw %[a13], 28(%[a_addr1]) \n\t"
              "flw %[a23], 28(%[a_addr2]) \n\t"
              "flw %[a33], 28(%[a_addr3]) \n\t"
              "flw %[a43], 28(%[a_addr4]) \n\t"
              "flw %[a53], 28(%[a_addr5]) \n\t"
              "flw %[a63], 28(%[a_addr6]) \n\t"
              "flw %[a73], 28(%[a_addr7]) \n\t"
              "vle32.v %[b30], (%[b_addr3]) \n\t"
              "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
              "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
              "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
              "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
              "vfmacc.vf %[acc4], %[a40], %[b00] \n\t"
              "vfmacc.vf %[acc5], %[a50], %[b00] \n\t"
              "vfmacc.vf %[acc6], %[a60], %[b00] \n\t"
              "vfmacc.vf %[acc7], %[a70], %[b00] \n\t"
              "flw %[a00], 32(%[a_addr0]) \n\t"
              "flw %[a10], 32(%[a_addr1]) \n\t"
              "flw %[a20], 32(%[a_addr2]) \n\t"
              "flw %[a30], 32(%[a_addr3]) \n\t"
              "flw %[a40], 32(%[a_addr4]) \n\t"
              "flw %[a50], 32(%[a_addr5]) \n\t"
              "flw %[a60], 32(%[a_addr6]) \n\t"
              "flw %[a70], 32(%[a_addr7]) \n\t"
              "vle32.v %[b00], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
              "vfmacc.vf %[acc1], %[a11], %[b10] \n\t"
              "vfmacc.vf %[acc2], %[a21], %[b10] \n\t"
              "vfmacc.vf %[acc3], %[a31], %[b10] \n\t"
              "vfmacc.vf %[acc4], %[a41], %[b10] \n\t"
              "vfmacc.vf %[acc5], %[a51], %[b10] \n\t"
              "vfmacc.vf %[acc6], %[a61], %[b10] \n\t"
              "vfmacc.vf %[acc7], %[a71], %[b10] \n\t"
              "flw %[a01], 36(%[a_addr0]) \n\t"
              "flw %[a11], 36(%[a_addr1]) \n\t"
              "flw %[a21], 36(%[a_addr2]) \n\t"
              "flw %[a31], 36(%[a_addr3]) \n\t"
              "flw %[a41], 36(%[a_addr4]) \n\t"
              "flw %[a51], 36(%[a_addr5]) \n\t"
              "flw %[a61], 36(%[a_addr6]) \n\t"
              "flw %[a71], 36(%[a_addr7]) \n\t"
              "vle32.v %[b10], (%[b_addr1]) \n\t"
              "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
              "vfmacc.vf %[acc1], %[a12], %[b20] \n\t"
              "vfmacc.vf %[acc2], %[a22], %[b20] \n\t"
              "vfmacc.vf %[acc3], %[a32], %[b20] \n\t"
              "vfmacc.vf %[acc4], %[a42], %[b20] \n\t"
              "vfmacc.vf %[acc5], %[a52], %[b20] \n\t"
              "vfmacc.vf %[acc6], %[a62], %[b20] \n\t"
              "vfmacc.vf %[acc7], %[a72], %[b20] \n\t"
              "flw %[a02], 40(%[a_addr0]) \n\t"
              "flw %[a12], 40(%[a_addr1]) \n\t"
              "flw %[a22], 40(%[a_addr2]) \n\t"
              "flw %[a32], 40(%[a_addr3]) \n\t"
              "flw %[a42], 40(%[a_addr4]) \n\t"
              "flw %[a52], 40(%[a_addr5]) \n\t"
              "flw %[a62], 40(%[a_addr6]) \n\t"
              "flw %[a72], 40(%[a_addr7]) \n\t"
              "vle32.v %[b20], (%[b_addr2]) \n\t"
              "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

              "vfmacc.vf %[acc0], %[a03], %[b30] \n\t"
              "vfmacc.vf %[acc1], %[a13], %[b30] \n\t"
              "vfmacc.vf %[acc2], %[a23], %[b30] \n\t"
              "vfmacc.vf %[acc3], %[a33], %[b30] \n\t"
              "vfmacc.vf %[acc4], %[a43], %[b30] \n\t"
              "vfmacc.vf %[acc5], %[a53], %[b30] \n\t"
              "vfmacc.vf %[acc6], %[a63], %[b30] \n\t"
              "vfmacc.vf %[acc7], %[a73], %[b30] \n\t"
              "flw %[a03], 44(%[a_addr0]) \n\t"
              "flw %[a13], 44(%[a_addr1]) \n\t"
              "flw %[a23], 44(%[a_addr2]) \n\t"
              "flw %[a33], 44(%[a_addr3]) \n\t"
              "flw %[a43], 44(%[a_addr4]) \n\t"
              "flw %[a53], 44(%[a_addr5]) \n\t"
              "flw %[a63], 44(%[a_addr6]) \n\t"
              "flw %[a73], 44(%[a_addr7]) \n\t"
              "vle32.v %[b30], (%[b_addr3]) \n\t"
              "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

              "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"
              "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"
              "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"
              "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"
              "add %[a_addr4], %[a_addr4], %[a_inc] \n\t"
              "add %[a_addr5], %[a_addr5], %[a_inc] \n\t"
              "add %[a_addr6], %[a_addr6], %[a_inc] \n\t"
              "add %[a_addr7], %[a_addr7], %[a_inc] \n\t"
#pragma clang diagnostic pop
              : [a00] "+&f"(a00),
                [a10] "+&f"(a10),
                [a20] "+&f"(a20),
                [a30] "+&f"(a30),
                [a40] "+&f"(a40),
                [a50] "+&f"(a50),
                [a60] "+&f"(a60),
                [a70] "+&f"(a70),
                [a01] "+&f"(a01),
                [a11] "+&f"(a11),
                [a21] "+&f"(a21),
                [a31] "+&f"(a31),
                [a41] "+&f"(a41),
                [a51] "+&f"(a51),
                [a61] "+&f"(a61),
                [a71] "+&f"(a71),
                [a02] "+&f"(a02),
                [a12] "+&f"(a12),
                [a22] "+&f"(a22),
                [a32] "+&f"(a32),
                [a42] "+&f"(a42),
                [a52] "+&f"(a52),
                [a62] "+&f"(a62),
                [a72] "+&f"(a72),
                [a03] "+&f"(a03),
                [a13] "+&f"(a13),
                [a23] "+&f"(a23),
                [a33] "+&f"(a33),
                [a43] "+&f"(a43),
                [a53] "+&f"(a53),
                [a63] "+&f"(a63),
                [a73] "+&f"(a73),

                [b00] "+&vr"(b00),
                [b10] "+&vr"(b10),
                [b20] "+&vr"(b20),
                [b30] "+&vr"(b30),

                [acc0] "+&vr"(acc0),
                [acc1] "+&vr"(acc1),
                [acc2] "+&vr"(acc2),
                [acc3] "+&vr"(acc3),
                [acc4] "+&vr"(acc4),
                [acc5] "+&vr"(acc5),
                [acc6] "+&vr"(acc6),
                [acc7] "+&vr"(acc7),

                [a_addr0] "+&r"(a_addr0),
                [a_addr1] "+&r"(a_addr1),
                [a_addr2] "+&r"(a_addr2),
                [a_addr3] "+&r"(a_addr3),
                [a_addr4] "+&r"(a_addr4),
                [a_addr5] "+&r"(a_addr5),
                [a_addr6] "+&r"(a_addr6),
                [a_addr7] "+&r"(a_addr7),

                [b_addr0] "+&r"(b_addr0),
                [b_addr1] "+&r"(b_addr1),
                [b_addr2] "+&r"(b_addr2),
                [b_addr3] "+&r"(b_addr3)
              : [a_inc] "r" (sizeof(float) * kk_unroll_degree),
                [b_inc] "r"(sizeof(float) * rsb * preload_distance),
                [jj_vl] "r"(jj_vl)
              : "vtype", "vl", "memory"
              // clang-format on
          );
        }

        __asm__ volatile(
            // clang-format off
            "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
            "vfmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfmacc.vf %[acc5], %[a50], %[b00] \n\t"
            "vfmacc.vf %[acc6], %[a60], %[b00] \n\t"
            "vfmacc.vf %[acc7], %[a70], %[b00] \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfmacc.vf %[acc1], %[a11], %[b10] \n\t"
            "vfmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfmacc.vf %[acc3], %[a31], %[b10] \n\t"
            "vfmacc.vf %[acc4], %[a41], %[b10] \n\t"
            "vfmacc.vf %[acc5], %[a51], %[b10] \n\t"
            "vfmacc.vf %[acc6], %[a61], %[b10] \n\t"
            "vfmacc.vf %[acc7], %[a71], %[b10] \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfmacc.vf %[acc1], %[a12], %[b20] \n\t"
            "vfmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfmacc.vf %[acc3], %[a32], %[b20] \n\t"
            "vfmacc.vf %[acc4], %[a42], %[b20] \n\t"
            "vfmacc.vf %[acc5], %[a52], %[b20] \n\t"
            "vfmacc.vf %[acc6], %[a62], %[b20] \n\t"
            "vfmacc.vf %[acc7], %[a72], %[b20] \n\t"

            "vfmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfmacc.vf %[acc1], %[a13], %[b30] \n\t"
            "vfmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfmacc.vf %[acc3], %[a33], %[b30] \n\t"
            "vfmacc.vf %[acc4], %[a43], %[b30] \n\t"
            "vfmacc.vf %[acc5], %[a53], %[b30] \n\t"
            "vfmacc.vf %[acc6], %[a63], %[b30] \n\t"
            "vfmacc.vf %[acc7], %[a73], %[b30] \n\t"

            : [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1),
              [acc2] "+&vr"(acc2),
              [acc3] "+&vr"(acc3),
              [acc4] "+&vr"(acc4),
              [acc5] "+&vr"(acc5),
              [acc6] "+&vr"(acc6),
              [acc7] "+&vr"(acc7)
            : [jj_vl] "r"(jj_vl),
              [a00] "f"(a00),
              [a10] "f"(a10),
              [a20] "f"(a20),
              [a30] "f"(a30),
              [a40] "f"(a40),
              [a50] "f"(a50),
              [a60] "f"(a60),
              [a70] "f"(a70),

              [a01] "f"(a01),
              [a11] "f"(a11),
              [a21] "f"(a21),
              [a31] "f"(a31),
              [a41] "f"(a41),
              [a51] "f"(a51),
              [a61] "f"(a61),
              [a71] "f"(a71),

              [a02] "f"(a02),
              [a12] "f"(a12),
              [a22] "f"(a22),
              [a32] "f"(a32),
              [a42] "f"(a42),
              [a52] "f"(a52),
              [a62] "f"(a62),
              [a72] "f"(a72),

              [a03] "f"(a03),
              [a13] "f"(a13),
              [a23] "f"(a23),
              [a33] "f"(a33),
              [a43] "f"(a43),
              [a53] "f"(a53),
              [a63] "f"(a63),
              [a73] "f"(a73),

              [b00] "vr"(b00),
              [b10] "vr"(b10),
              [b20] "vr"(b20),
              [b30] "vr"(b30)
            : "vtype", "vl", "memory"
            // clang-format on
        );

        kk += preload_distance;
      }
      for (; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"

            "vle32.v %[b00], (%[b_addr0]) \n\t"
            "flw %[a00], 0(%[a_addr0]) \n\t"
            "flw %[a10], 0(%[a_addr1]) \n\t"
            "flw %[a20], 0(%[a_addr2]) \n\t"
            "flw %[a30], 0(%[a_addr3]) \n\t"
            "flw %[a40], 0(%[a_addr4]) \n\t"
            "flw %[a50], 0(%[a_addr5]) \n\t"
            "flw %[a60], 0(%[a_addr6]) \n\t"
            "flw %[a70], 0(%[a_addr7]) \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
            "vfmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfmacc.vf %[acc5], %[a50], %[b00] \n\t"
            "vfmacc.vf %[acc6], %[a60], %[b00] \n\t"
            "vfmacc.vf %[acc7], %[a70], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [a10] "=&f"(a10),
              [a20] "=&f"(a20),
              [a30] "=&f"(a30),
              [a40] "=&f"(a40),
              [a50] "=&f"(a50),
              [a60] "=&f"(a60),
              [a70] "=&f"(a70),
              [b00] "=&vr"(b00),
              [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1),
              [acc2] "+&vr"(acc2),
              [acc3] "+&vr"(acc3),
              [acc4] "+&vr"(acc4),
              [acc5] "+&vr"(acc5),
              [acc6] "+&vr"(acc6),
              [acc7] "+vr"(acc7)
            : [jj_vl] "r"(jj_vl),
              [a_addr0] "r"(a + (ii + 0) * rsa + kk),
              [a_addr1] "r"(a + (ii + 1) * rsa + kk),
              [a_addr2] "r"(a + (ii + 2) * rsa + kk),
              [a_addr3] "r"(a + (ii + 3) * rsa + kk),
              [a_addr4] "r"(a + (ii + 4) * rsa + kk),
              [a_addr5] "r"(a + (ii + 5) * rsa + kk),
              [a_addr6] "r"(a + (ii + 6) * rsa + kk),
              [a_addr7] "r"(a + (ii + 7) * rsa + kk),
              [b_addr0] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"
          "vle32.v %[c00], (%[c_addr0]) \n\t"
          "vle32.v %[c10], (%[c_addr1]) \n\t"
          "vle32.v %[c20], (%[c_addr2]) \n\t"
          "vle32.v %[c30], (%[c_addr3]) \n\t"
          "vle32.v %[c40], (%[c_addr4]) \n\t"
          "vle32.v %[c50], (%[c_addr5]) \n\t"
          "vle32.v %[c60], (%[c_addr6]) \n\t"
          "vle32.v %[c70], (%[c_addr7]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmul.vf %[c10], %[c10], %[beta] \n\t"
          "vfmul.vf %[c20], %[c20], %[beta] \n\t"
          "vfmul.vf %[c30], %[c30], %[beta] \n\t"
          "vfmul.vf %[c40], %[c40], %[beta] \n\t"
          "vfmul.vf %[c50], %[c50], %[beta] \n\t"
          "vfmul.vf %[c60], %[c60], %[beta] \n\t"
          "vfmul.vf %[c70], %[c70], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vfmacc.vf %[c10], %[alpha], %[acc1] \n\t"
          "vfmacc.vf %[c20], %[alpha], %[acc2] \n\t"
          "vfmacc.vf %[c30], %[alpha], %[acc3] \n\t"
          "vfmacc.vf %[c40], %[alpha], %[acc4] \n\t"
          "vfmacc.vf %[c50], %[alpha], %[acc5] \n\t"
          "vfmacc.vf %[c60], %[alpha], %[acc6] \n\t"
          "vfmacc.vf %[c70], %[alpha], %[acc7] \n\t"
          "vse32.v %[c00], (%[c_addr0]) \n\t"
          "vse32.v %[c10], (%[c_addr1]) \n\t"
          "vse32.v %[c20], (%[c_addr2]) \n\t"
          "vse32.v %[c30], (%[c_addr3]) \n\t"
          "vse32.v %[c40], (%[c_addr4]) \n\t"
          "vse32.v %[c50], (%[c_addr5]) \n\t"
          "vse32.v %[c60], (%[c_addr6]) \n\t"
          "vse32.v %[c70], (%[c_addr7]) \n\t"
          : [c00] "=&vr"(c00),
            [c10] "=&vr"(c10),
            [c20] "=&vr"(c20),
            [c30] "=&vr"(c30),
            [c40] "=&vr"(c40),
            [c50] "=&vr"(c50),
            [c60] "=&vr"(c60),
            [c70] "=&vr"(c70)
          : [jj_vl] "r"(jj_vl),
            [c_addr0] "r"(c + (ii + 0) * rsc + jj),
            [c_addr1] "r"(c + (ii + 1) * rsc + jj),
            [c_addr2] "r"(c + (ii + 2) * rsc + jj),
            [c_addr3] "r"(c + (ii + 3) * rsc + jj),
            [c_addr4] "r"(c + (ii + 4) * rsc + jj),
            [c_addr5] "r"(c + (ii + 5) * rsc + jj),
            [c_addr6] "r"(c + (ii + 6) * rsc + jj),
            [c_addr7] "r"(c + (ii + 7) * rsc + jj),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc0] "vr"(acc0),
            [acc1] "vr"(acc1),
            [acc2] "vr"(acc2),
            [acc3] "vr"(acc3),
            [acc4] "vr"(acc4),
            [acc5] "vr"(acc5),
            [acc6] "vr"(acc6),
            [acc7] "vr"(acc7)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }
  for (; ii < m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e32m2(n - jj);
      __asm__ volatile(
          // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"
            "flw %[a00], 0(%[a_addr]) \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            "vfmul.vf %[acc0], %[b00], %[a00] \n\t"
            : [a00] "=&f"(a00),
              [b00] "=&vr"(b00),
              [acc0] "=vr"(acc0)
            : [jj_vl] "r"(jj_vl),
              [a_addr] "r"(a + ii * rsa),
              [b_addr] "r"(b + jj)
            : "vtype", "vl", "memory"
          // clang-format on
      );
      for (kk = 1; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"
            "flw %[a00], 0(%[a_addr]) \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [b00] "=&vr"(b00),
              [acc0] "+vr"(acc0)
            : [jj_vl] "r"(jj_vl),
              [a_addr] "r"(a + ii * rsa + kk),
              [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m2, ta, ma \n\t"
          "vle32.v %[c00], (%[c_addr]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vse32.v %[c00], (%[c_addr]) \n\t"
          : [c00] "=&vr"(c00)
          : [jj_vl] "r"(jj_vl),
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
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f32rc_f32rc_f32rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 * Uses a 4 x LMUL=4 x 4 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e32m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x4_f32_f32_f32_zve32f_x390(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    const float *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  size_t kk0;
  size_t ii0;
  vfloat32m4_t b0;
  vfloat32m4_t acc0;
  vfloat32m4_t acc1;
  vfloat32m4_t acc2;
  vfloat32m4_t acc3;
  float a00;
  float a01;
  float a02;
  float a03;
  float a10;
  float a11;
  float a12;
  float a13;
  float a20;
  float a21;
  float a22;
  float a23;
  float a30;
  float a31;
  float a32;
  float a33;
  vfloat32m4_t b00;
  vfloat32m4_t b10;
  vfloat32m4_t b20;
  vfloat32m4_t b30;
  vfloat32m4_t c00;
  vfloat32m4_t c10;
  vfloat32m4_t c20;
  vfloat32m4_t c30;
  float a0;
  vfloat32m4_t acc;
  vfloat32m4_t c0;

  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m4(n - jj);
        c0 = __riscv_vle32_v_f32m4(c + (((ii + 0) * rsc) + jj), jj_vl);
        c0 = __riscv_vfmul_vf_f32m4(c0, beta, jj_vl);
        __riscv_vse32_v_f32m4(c + (((ii + 0) * rsc) + jj), c0, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m4(n - jj);
      if (1 + 4 <= k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"

            // Initialize accs
            "vle32.v %[b00], (%[b_load_0]) \n\t"
            NTL_P1
            "flw %[a00], 0(%[a_addr_0]) \n\t"
            NTL_P1
            "flw %[a10], 0(%[a_addr_1]) \n\t"
            NTL_P1
            "flw %[a20], 0(%[a_addr_2]) \n\t"
            NTL_P1
            "flw %[a30], 0(%[a_addr_3]) \n\t"
            "vfmul.vf %[acc0], %[b00], %[a00] \n\t"
            "vfmul.vf %[acc1], %[b00], %[a10] \n\t"
            "vfmul.vf %[acc2], %[b00], %[a20] \n\t"
            "vfmul.vf %[acc3], %[b00], %[a30] \n\t"

            // Initialize a's and b's for first K iteration
            "vle32.v %[b00], (%[b_load_1]) \n\t"
            "vle32.v %[b10], (%[b_load_2]) \n\t"
            "vle32.v %[b20], (%[b_load_3]) \n\t"
            "vle32.v %[b30], (%[b_load_4]) \n\t"

            "flw %[a00],  4(%[a_addr_0]) \n\t"
            "flw %[a10],  4(%[a_addr_1]) \n\t"
            "flw %[a20],  4(%[a_addr_2]) \n\t"
            "flw %[a30],  4(%[a_addr_3]) \n\t"

            "flw %[a01],  8(%[a_addr_0]) \n\t"
            "flw %[a11],  8(%[a_addr_1]) \n\t"
            "flw %[a21],  8(%[a_addr_2]) \n\t"
            "flw %[a31],  8(%[a_addr_3]) \n\t"

            "flw %[a02], 12(%[a_addr_0]) \n\t"
            "flw %[a12], 12(%[a_addr_1]) \n\t"
            "flw %[a22], 12(%[a_addr_2]) \n\t"
            "flw %[a32], 12(%[a_addr_3]) \n\t"

            "flw %[a03], 16(%[a_addr_0]) \n\t"
            "flw %[a13], 16(%[a_addr_1]) \n\t"
            "flw %[a23], 16(%[a_addr_2]) \n\t"
            "flw %[a33], 16(%[a_addr_3]) \n\t"
            // clang-format on
            : [b00] "=&vr"(b00), [b10] "=&vr"(b10), [b20] "=&vr"(b20),
              [b30] "=&vr"(b30), [acc0] "=&vr"(acc0), [acc1] "=&vr"(acc1),
              [acc2] "=&vr"(acc2), [acc3] "=&vr"(acc3), [a00] "=&f"(a00),
              [a10] "=&f"(a10), [a20] "=&f"(a20), [a30] "=&f"(a30),
              [a01] "=&f"(a01), [a11] "=&f"(a11), [a21] "=&f"(a21),
              [a31] "=&f"(a31), [a02] "=&f"(a02), [a12] "=&f"(a12),
              [a22] "=&f"(a22), [a32] "=&f"(a32), [a03] "=&f"(a03),
              [a13] "=&f"(a13), [a23] "=&f"(a23), [a33] "=&f"(a33)
            :
            [a_addr_0] "r"(a + (ii + 0) * rsa + 0),
            [a_addr_1] "r"(a + (ii + 1) * rsa + 0),
            [a_addr_2] "r"(a + (ii + 2) * rsa + 0),
            [a_addr_3] "r"(a + (ii + 3) * rsa + 0),
            [b_load_0] "r"(b + 0 * rsb + jj), [b_load_1] "r"(b + 1 * rsb + jj),
            [b_load_2] "r"(b + 2 * rsb + jj), [b_load_3] "r"(b + 3 * rsb + jj),
            [b_load_4] "r"(b + 4 * rsb + jj), [jj_vl_in] "r"(jj_vl)
            : "vtype", "vl", "memory");
      } else {
        __asm__ volatile(
            // Just initialize accumulators and let fixup loop below handle any
            // remaining iterations.
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b00], (%[b_load]) \n\t"
            "flw %[a00], 0(%[a_addr_0]) \n\t"
            "flw %[a10], 0(%[a_addr_1]) \n\t"
            "flw %[a20], 0(%[a_addr_2]) \n\t"
            "flw %[a30], 0(%[a_addr_3]) \n\t"
            "vfmul.vf %[acc0], %[b00], %[a00] \n\t"
            "vfmul.vf %[acc1], %[b00], %[a10] \n\t"
            "vfmul.vf %[acc2], %[b00], %[a20] \n\t"
            "vfmul.vf %[acc3], %[b00], %[a30] \n\t"
            : [a00] "=&f"(a00), [a10] "=&f"(a10), [a20] "=&f"(a20),
              [a30] "=&f"(a30), [b00] "=&vr"(b00), [acc0] "=&vr"(acc0),
              [acc1] "=&vr"(acc1), [acc2] "=&vr"(acc2), [acc3] "=vr"(acc3)
            : [a_addr_0] "r"(a + (ii + 0) * rsa + 0),
              [a_addr_1] "r"(a + (ii + 1) * rsa + 0),
              [a_addr_2] "r"(a + (ii + 2) * rsa + 0),
              [a_addr_3] "r"(a + (ii + 3) * rsa + 0), [b_load] "r"(b + jj),
              [jj_vl_in] "r"(jj_vl)
            : "vtype", "vl", "memory");
      }
      for (kk = 1; (kk + 2UL * 4UL) < k; kk = kk + 4) {
        const float *a_addr_1;
        const float *a_addr_2;
        const float *a_addr_3;
        const float *b_addr = b + (kk + 4) * rsb + jj;
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"

            "add %[a_addr_1], %[a_addr_0], %[rsa4] \n\t" // a + (ii + 1) * rsa + kk
            "add %[a_addr_2], %[a_addr_1], %[rsa4] \n\t" // a + (ii + 2) * rsa + kk
            "add %[a_addr_3], %[a_addr_2], %[rsa4] \n\t" // a + (ii + 3) * rsa + kk

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
            NTL_P1
            "flw %[a00], 16(%[a_addr_0]) \n\t"
            NTL_P1
            "flw %[a10], 16(%[a_addr_1]) \n\t"
            NTL_P1
            "flw %[a20], 16(%[a_addr_2]) \n\t"
            NTL_P1
            "flw %[a30], 16(%[a_addr_3]) \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfmacc.vf %[acc1], %[a11], %[b10] \n\t"
            "vle32.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t" // b + (kk + 4 + 1) * rsb + jj
            "vfmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfmacc.vf %[acc3], %[a31], %[b10] \n\t"
            NTL_P1
            "flw %[a01], 20(%[a_addr_0]) \n\t"
            NTL_P1
            "flw %[a11], 20(%[a_addr_1]) \n\t"
            NTL_P1
            "flw %[a21], 20(%[a_addr_2]) \n\t"
            NTL_P1
            "flw %[a31], 20(%[a_addr_3]) \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfmacc.vf %[acc1], %[a12], %[b20] \n\t"
            "vle32.v %[b10], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t" // b + (kk + 4 + 2) * rsb + jj
            "vfmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfmacc.vf %[acc3], %[a32], %[b20] \n\t"
            NTL_P1
            "flw %[a02], 24(%[a_addr_0]) \n\t"
            NTL_P1
            "flw %[a12], 24(%[a_addr_1]) \n\t"
            NTL_P1
            "flw %[a22], 24(%[a_addr_2]) \n\t"
            NTL_P1
            "flw %[a32], 24(%[a_addr_3]) \n\t"

            "vfmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfmacc.vf %[acc1], %[a13], %[b30] \n\t"
            "vle32.v %[b20], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb4] \n\t" // b + (kk + 4 + 3) * rsb + jj
            "vfmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfmacc.vf %[acc3], %[a33], %[b30] \n\t"
            NTL_P1
            "flw %[a03], 28(%[a_addr_0]) \n\t"
            NTL_P1
            "flw %[a13], 28(%[a_addr_1]) \n\t"
            NTL_P1
            "flw %[a23], 28(%[a_addr_2]) \n\t"
            NTL_P1
            "flw %[a33], 28(%[a_addr_3]) \n\t"

            "vle32.v %[b30], (%[b_addr]) \n\t"
            // clang-format on
            : [a00] "+&f"(a00), [a10] "+&f"(a10), [a20] "+&f"(a20),
              [a30] "+&f"(a30), [a01] "+&f"(a01), [a11] "+&f"(a11),
              [a21] "+&f"(a21), [a31] "+&f"(a31), [a02] "+&f"(a02),
              [a12] "+&f"(a12), [a22] "+&f"(a22), [a32] "+&f"(a32),
              [a03] "+&f"(a03), [a13] "+&f"(a13), [a23] "+&f"(a23),
              [a33] "+&f"(a33), [b00] "+&vr"(b00), [b10] "+&vr"(b10),
              [b20] "+&vr"(b20), [b30] "+&vr"(b30), [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1), [acc2] "+&vr"(acc2), [acc3] "+&vr"(acc3),
              [a_addr_1] "=&r"(a_addr_1), [a_addr_2] "=&r"(a_addr_2),
              [a_addr_3] "=&r"(a_addr_3), [b_addr] "+&r"(b_addr)
            : [a_addr_0] "r"(a + ii * rsa + kk),
              [rsa4] "r"(rsa * sizeof(float)), [rsb4] "r"(rsb * sizeof(float)),
              [jj_vl_in] "r"(jj_vl)
            : "vtype", "vl", "memory");
      }

      if (1 + 4 <= k) {
        __asm__ volatile(
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"

            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"

            "vfmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfmacc.vf %[acc1], %[a11], %[b10] \n\t"
            "vfmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfmacc.vf %[acc3], %[a31], %[b10] \n\t"

            "vfmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfmacc.vf %[acc1], %[a12], %[b20] \n\t"
            "vfmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfmacc.vf %[acc3], %[a32], %[b20] \n\t"

            "vfmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfmacc.vf %[acc1], %[a13], %[b30] \n\t"
            "vfmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfmacc.vf %[acc3], %[a33], %[b30] \n\t"
            : [acc0] "+&vr"(acc0), [acc1] "+&vr"(acc1), [acc2] "+&vr"(acc2),
              [acc3] "+&vr"(acc3)
            : [b00] "vr"(b00), [b10] "vr"(b10), [b20] "vr"(b20),
              [b30] "vr"(b30), [a00] "f"(a00), [a10] "f"(a10), [a20] "f"(a20),
              [a30] "f"(a30), [a01] "f"(a01), [a11] "f"(a11), [a21] "f"(a21),
              [a31] "f"(a31), [a02] "f"(a02), [a12] "f"(a12), [a22] "f"(a22),
              [a32] "f"(a32), [a03] "f"(a03), [a13] "f"(a13), [a23] "f"(a23),
              [a33] "f"(a33), [jj_vl_in] "r"(jj_vl)
            : "vtype", "vl", "memory");
        kk += 4;
      }

      for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        __asm__ volatile(
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b00], (%[b_load]) \n\t"
            "flw %[a00], 0(%[a_load_0]) \n\t"
            "flw %[a10], 0(%[a_load_1]) \n\t"
            "flw %[a20], 0(%[a_load_2]) \n\t"
            "flw %[a30], 0(%[a_load_3]) \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a30], %[b00] \n\t"
            : [a00] "=&f"(a00), [a10] "=&f"(a10), [a20] "=&f"(a20),
              [a30] "=&f"(a30), [b00] "=&vr"(b00), [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1), [acc2] "+&vr"(acc2), [acc3] "+vr"(acc3)
            : [a_load_0] "r"(a + (ii + 0) * rsa + kk0),
              [a_load_1] "r"(a + (ii + 1) * rsa + kk0),
              [a_load_2] "r"(a + (ii + 2) * rsa + kk0),
              [a_load_3] "r"(a + (ii + 3) * rsa + kk0),
              [b_load] "r"(b + kk0 * rsb + jj), [jj_vl_in] "r"(jj_vl)
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"

          "vle32.v %[c00], (%[c_addr_0]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vse32.v %[c00], (%[c_addr_0]) \n\t"

          "vle32.v %[c10], (%[c_addr_1]) \n\t"
          "vfmul.vf %[c10], %[c10], %[beta] \n\t"
          "vfmacc.vf %[c10], %[alpha], %[acc1] \n\t"
          "vse32.v %[c10], (%[c_addr_1]) \n\t"

          "vle32.v %[c20], (%[c_addr_2]) \n\t"
          "vfmul.vf %[c20], %[c20], %[beta] \n\t"
          "vfmacc.vf %[c20], %[alpha], %[acc2] \n\t"
          "vse32.v %[c20], (%[c_addr_2]) \n\t"

          "vle32.v %[c30], (%[c_addr_3]) \n\t"
          "vfmul.vf %[c30], %[c30], %[beta] \n\t"
          "vfmacc.vf %[c30], %[alpha], %[acc3] \n\t"
          "vse32.v %[c30], (%[c_addr_3]) \n\t"
          : [c00] "=&vr"(c00), [c10] "=&vr"(c10), [c20] "=&vr"(c20),
            [c30] "=&vr"(c30)
          : [jj_vl_in] "r"(jj_vl), [c_addr_0] "r"(c + (ii + 0) * rsc + jj),
            [c_addr_1] "r"(c + (ii + 1) * rsc + jj),
            [c_addr_2] "r"(c + (ii + 2) * rsc + jj),
            [c_addr_3] "r"(c + (ii + 3) * rsc + jj), [beta] "f"(beta),
            [alpha] "f"(alpha), [acc0] "vr"(acc0), [acc1] "vr"(acc1),
            [acc2] "vr"(acc2), [acc3] "vr"(acc3)
          : "vtype", "vl", "memory");
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e32m4(n - jj);
      __asm__ volatile("\n\t"
                       "flw %[a0], 0(%[a_load]) \n\t"
                       "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"
                       "vle32.v %[b0], (%[b_load]) \n\t"
                       "vfmul.vf %[acc], %[b0], %[a0] \n\t"
                       : [a0] "=&f"(a0), [b0] "=&vr"(b0), [acc] "=vr"(acc)
                       : [jj_vl_in] "r"(jj_vl), [a_load] "r"(a + ii0 * rsa),
                         [b_load] "r"(b + jj)
                       : "vtype", "vl", "memory");
      for (kk = 1; (kk + 1) <= k; kk = kk + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a0], 0(%[a_load]) \n\t"
            "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load]) \n\t"
            "vfmacc.vf %[acc], %[a0], %[b0] \n\t"
            : [a0] "=&f"(a0), [b0] "=&vr"(b0), [acc] "+vr"(acc)
            : [jj_vl_in] "r"(jj_vl), [a_load] "r"(a + ii0 * rsa + kk),
              [b_load] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"
          "vle32.v %[c0], (%[c_addr]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc] \n\t"
          "vse32.v %[c0], (%[c_addr]) \n\t"
          : [c0] "=&vr"(c0)
          : [jj_vl_in] "r"(jj_vl), [c_addr] "r"(c + ii0 * rsc + jj),
            [beta] "f"(beta), [alpha] "f"(alpha), [acc] "vr"(acc)
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
    skl_gemm_1xm8x3_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb,
                                            beta, c, rsc);
    return;
  }
  if (n <= __riscv_vsetvlmax_e32m2()) {
    skl_gemm_8xm1x4_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb,
                                            beta, c, rsc);
    return;
  }
  skl_gemm_4xm4x4_f32_f32_f32_zve32f_x390(m, n, k, alpha, a, rsa, b, rsb, beta,
                                          c, rsc);
}

#undef NTL_P1
