// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve64d) || __riscv_zve64d < 1000000
#error This file requires the RISC-V zve64d extension, version 1000000.
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
 * Uses a 4 x LMUL=4 x 12 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e64m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x12_f64_f64_f64_zve64d_x390(
    size_t m, size_t n, size_t k, double alpha, const double *a, size_t rsa,
    const double *b, size_t rsb, double beta, double *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  vfloat64m4_t acc00;
  vfloat64m4_t acc10;
  vfloat64m4_t acc20;
  vfloat64m4_t acc30;
  double a00;
  double a01;
  double a02;
  double a03;
  double a10;
  double a11;
  double a12;
  double a13;
  double a20;
  double a21;
  double a22;
  double a23;
  double a30;
  double a31;
  double a32;
  double a33;
  vfloat64m4_t b00;
  vfloat64m4_t b10;
  vfloat64m4_t b20;
  vfloat64m4_t b30;
  vfloat64m4_t c00;
  vfloat64m4_t c10;
  vfloat64m4_t c20;
  vfloat64m4_t c30;

  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e64m8(n - jj);
        vfloat64m8_t c00m8 =
            __riscv_vle64_v_f64m8(c + (((ii + 0) * rsc) + jj), jj_vl);
        c00m8 = __riscv_vfmul_vf_f64m8(c00m8, beta, jj_vl);
        __riscv_vse64_v_f64m8(c + (((ii + 0) * rsc) + jj), c00m8, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; ii + 4 <= m; ii += 4) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e64m4(n - jj);

      const double *a_addr0 = a + (ii + 0) * rsa + 0;
      const double *a_addr1 = a + (ii + 1) * rsa + 0;
      const double *a_addr2 = a + (ii + 2) * rsa + 0;
      const double *a_addr3 = a + (ii + 3) * rsa + 0;

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"
          "vle64.v %[b00], (%[b_addr0]) \n\t"
          "fld %[a00], 0(%[a_addr0]) \n\t"
          "fld %[a10], 0(%[a_addr1]) \n\t"
          "fld %[a20], 0(%[a_addr2]) \n\t"
          "fld %[a30], 0(%[a_addr3]) \n\t"
          "vfmul.vf %[acc00], %[b00], %[a00] \n\t"
          "vfmul.vf %[acc10], %[b00], %[a10] \n\t"
          "vfmul.vf %[acc20], %[b00], %[a20] \n\t"
          "vfmul.vf %[acc30], %[b00], %[a30] \n\t"

          "addi %[a_addr0], %[a_addr0], 8 \n\t"
          "addi %[a_addr1], %[a_addr1], 8 \n\t"
          "addi %[a_addr2], %[a_addr2], 8 \n\t"
          "addi %[a_addr3], %[a_addr3], 8 \n\t"
          : [a_addr0] "+&r" (a_addr0),
            [a_addr1] "+&r" (a_addr1),
            [a_addr2] "+&r" (a_addr2),
            [a_addr3] "+&r" (a_addr3),
            [a00] "=&f"(a00),
            [a10] "=&f"(a10),
            [a20] "=&f"(a20),
            [a30] "=&f"(a30),
            [b00] "=&vr"(b00),
            [acc00] "=&vr"(acc00),
            [acc10] "=&vr"(acc10),
            [acc20] "=&vr"(acc20),
            [acc30] "=vr"(acc30)
          : [b_addr0] "r"(b + jj),
            [jj_vl] "r"(jj_vl)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      kk = 1;
      const size_t preload_distance = 4;
      const size_t kk_unroll_degree = 12;

      if (kk + preload_distance + kk_unroll_degree <= k) {
        const double *b_addr0 = b + (kk + 0) * rsb + jj;
        const double *b_addr1 = b + (kk + 1) * rsb + jj;
        const double *b_addr2 = b + (kk + 2) * rsb + jj;
        const double *b_addr3 = b + (kk + 3) * rsb + jj;

        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

            "fld %[a00],  0(%[a_addr0]) \n\t"
            "fld %[a01],  8(%[a_addr0]) \n\t"
            "fld %[a02], 16(%[a_addr0]) \n\t"
            "fld %[a03], 24(%[a_addr0]) \n\t"
            "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"

            "vle64.v %[b00], (%[b_addr0]) \n\t"
            "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

            "fld %[a10],  0(%[a_addr1]) \n\t"
            "fld %[a11],  8(%[a_addr1]) \n\t"
            "fld %[a12], 16(%[a_addr1]) \n\t"
            "fld %[a13], 24(%[a_addr1]) \n\t"
            "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"

            "vle64.v %[b10], (%[b_addr1]) \n\t"
            "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

            "fld %[a20],  0(%[a_addr2]) \n\t"
            "fld %[a21],  8(%[a_addr2]) \n\t"
            "fld %[a22], 16(%[a_addr2]) \n\t"
            "fld %[a23], 24(%[a_addr2]) \n\t"
            "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"

            "vle64.v %[b20], (%[b_addr2]) \n\t"
            "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

            "fld %[a30],  0(%[a_addr3]) \n\t"
            "fld %[a31],  8(%[a_addr3]) \n\t"
            "fld %[a32], 16(%[a_addr3]) \n\t"
            "fld %[a33], 24(%[a_addr3]) \n\t"
            "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"

            "vle64.v %[b30], (%[b_addr3]) \n\t"
            "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

            : [b00] "=&vr"(b00),
              [b10] "=&vr"(b10),
              [b20] "=&vr"(b20),
              [b30] "=&vr"(b30),
              [a00] "=&f"(a00),
              [a10] "=&f"(a10),
              [a20] "=&f"(a20),
              [a30] "=&f"(a30),
              [a01] "=&f"(a01),
              [a11] "=&f"(a11),
              [a21] "=&f"(a21),
              [a31] "=&f"(a31),
              [a02] "=&f"(a02),
              [a12] "=&f"(a12),
              [a22] "=&f"(a22),
              [a32] "=&f"(a32),
              [a03] "=&f"(a03),
              [a13] "=&f"(a13),
              [a23] "=&f"(a23),
              [a33] "=&f"(a33),
              [a_addr0] "+&r" (a_addr0),
              [a_addr1] "+&r" (a_addr1),
              [a_addr2] "+&r" (a_addr2),
              [a_addr3] "+&r" (a_addr3),
              [b_addr0] "+&r" (b_addr0),
              [b_addr1] "+&r" (b_addr1),
              [b_addr2] "+&r" (b_addr2),
              [b_addr3] "+&r" (b_addr3)
            : [jj_vl] "r"(jj_vl),
              [a_inc] "r"(sizeof(double) * preload_distance),
              [b_inc] "r"(sizeof(double) * preload_distance * rsb)
            : "vtype", "vl", "memory"
            // clang-format on
        );

        for (; kk + preload_distance + kk_unroll_degree < k;
             kk += kk_unroll_degree) {
          __asm__ volatile(
          // clang-format off
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
             "\n\t"
             "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

             "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
             "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
             NTL_P1
             "fld %[a00],  0(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a10],  0(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
             "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
             NTL_P1
             "fld %[a20],  0(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a30],  0(%[a_addr3]) \n\t"
             "vle64.v %[b00], (%[b_addr0]) \n\t"
             "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
             "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
             NTL_P1
             "fld %[a01],  8(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a11],  8(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
             "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"
             NTL_P1
             "fld %[a21],  8(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a31],  8(%[a_addr3]) \n\t"
             "vle64.v %[b10], (%[b_addr1]) \n\t"
             "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
             "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
             NTL_P1
             "fld %[a02], 16(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a12], 16(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
             "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"
             NTL_P1
             "fld %[a22], 16(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a32], 16(%[a_addr3]) \n\t"
             "vle64.v %[b20], (%[b_addr2]) \n\t"
             "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
             "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
             NTL_P1
             "fld %[a03], 24(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a13], 24(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
             "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
             NTL_P1
             "fld %[a23], 24(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a33], 24(%[a_addr3]) \n\t"
             "vle64.v %[b30], (%[b_addr3]) \n\t"
             "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
             "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
             NTL_P1
             "fld %[a00], 32(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a10], 32(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
             "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
             NTL_P1
             "fld %[a20], 32(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a30], 32(%[a_addr3]) \n\t"
             "vle64.v %[b00], (%[b_addr0]) \n\t"
             "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
             "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
             NTL_P1
             "fld %[a01], 40(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a11], 40(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
             "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"
             NTL_P1
             "fld %[a21], 40(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a31], 40(%[a_addr3]) \n\t"
             "vle64.v %[b10], (%[b_addr1]) \n\t"
             "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
             "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
             NTL_P1
             "fld %[a02], 48(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a12], 48(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
             "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"
             NTL_P1
             "fld %[a22], 48(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a32], 48(%[a_addr3]) \n\t"
             "vle64.v %[b20], (%[b_addr2]) \n\t"
             "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
             "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
             NTL_P1
             "fld %[a03], 56(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a13], 56(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
             "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
             NTL_P1
             "fld %[a23], 56(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a33], 56(%[a_addr3]) \n\t"
             "vle64.v %[b30], (%[b_addr3]) \n\t"
             "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
             "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
             NTL_P1
             "fld %[a00], 64(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a10], 64(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
             "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
             NTL_P1
             "fld %[a20], 64(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a30], 64(%[a_addr3]) \n\t"
             "vle64.v %[b00], (%[b_addr0]) \n\t"
             "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
             "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
             NTL_P1
             "fld %[a01], 72(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a11], 72(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
             "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"
             NTL_P1
             "fld %[a21], 72(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a31], 72(%[a_addr3]) \n\t"
             "vle64.v %[b10], (%[b_addr1]) \n\t"
             "add %[b_addr1], %[b_addr1], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
             "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
             NTL_P1
             "fld %[a02], 80(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a12], 80(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
             "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"
             NTL_P1
             "fld %[a22], 80(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a32], 80(%[a_addr3]) \n\t"
             "vle64.v %[b20], (%[b_addr2]) \n\t"
             "add %[b_addr2], %[b_addr2], %[b_inc] \n\t"

             "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
             "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
             NTL_P1
             "fld %[a03], 88(%[a_addr0]) \n\t"
             NTL_P1
             "fld %[a13], 88(%[a_addr1]) \n\t"
             "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
             "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
             NTL_P1
             "fld %[a23], 88(%[a_addr2]) \n\t"
             NTL_P1
             "fld %[a33], 88(%[a_addr3]) \n\t"
             "vle64.v %[b30], (%[b_addr3]) \n\t"
             "add %[b_addr3], %[b_addr3], %[b_inc] \n\t"

             "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"
             "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"
             "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"
             "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"
#pragma clang diagnostic pop
             : [a00] "+&f"(a00),
               [a10] "+&f"(a10),
               [a20] "+&f"(a20),
               [a30] "+&f"(a30),
               [a01] "+&f"(a01),
               [a11] "+&f"(a11),
               [a21] "+&f"(a21),
               [a31] "+&f"(a31),
               [a02] "+&f"(a02),
               [a12] "+&f"(a12),
               [a22] "+&f"(a22),
               [a32] "+&f"(a32),
               [a03] "+&f"(a03),
               [a13] "+&f"(a13),
               [a23] "+&f"(a23),
               [a33] "+&f"(a33),
               [b00] "+&vr"(b00),
               [b10] "+&vr"(b10),
               [b20] "+&vr"(b20),
               [b30] "+&vr"(b30),
               [acc00] "+&vr"(acc00),
               [acc10] "+&vr"(acc10),
               [acc20] "+&vr"(acc20),
               [acc30] "+&vr"(acc30),
               [a_addr0] "+&r"(a_addr0),
               [a_addr1] "+&r"(a_addr1),
               [a_addr2] "+&r"(a_addr2),
               [a_addr3] "+&r"(a_addr3),
               [b_addr0] "+&r"(b_addr0),
               [b_addr1] "+&r"(b_addr1),
               [b_addr2] "+&r"(b_addr2),
               [b_addr3] "+&r"(b_addr3)
             : [a_inc] "r"(sizeof(double) * kk_unroll_degree),
               [b_inc] "r"(sizeof(double) * preload_distance * rsb),
               [jj_vl] "r"(jj_vl)
             : "vtype", "vl", "memory"
              // clang-format on
          );
        }

        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

            "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"

            "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
            "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
            "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
            "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"

            "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
            "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
            "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
            "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"

            "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
            "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
            "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
            "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
            : [acc00] "+&vr"(acc00),
              [acc10] "+&vr"(acc10),
              [acc20] "+&vr"(acc20),
              [acc30] "+&vr"(acc30)
            : [b00] "vr"(b00),
              [b10] "vr"(b10),
              [b20] "vr"(b20),
              [b30] "vr"(b30),
              [a00] "f"(a00),
              [a10] "f"(a10),
              [a20] "f"(a20),
              [a30] "f"(a30),
              [a01] "f"(a01),
              [a11] "f"(a11),
              [a21] "f"(a21),
              [a31] "f"(a31),
              [a02] "f"(a02),
              [a12] "f"(a12),
              [a22] "f"(a22),
              [a32] "f"(a32),
              [a03] "f"(a03),
              [a13] "f"(a13),
              [a23] "f"(a23),
              [a33] "f"(a33),
              [jj_vl] "r"(jj_vl)
            : "vtype", "vl"
            // clang-format on
        );
        kk += preload_distance;
      }

      for (; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"
            "vle64.v %[b00], (%[b_addr0]) \n\t"
            "fld %[a00], 0(%[a_addr0]) \n\t"
            "fld %[a10], 0(%[a_addr1]) \n\t"
            "fld %[a20], 0(%[a_addr2]) \n\t"
            "fld %[a30], 0(%[a_addr3]) \n\t"
            "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
            "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
            "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
            "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [a10] "=&f"(a10),
              [a20] "=&f"(a20),
              [a30] "=&f"(a30),
              [b00] "=&vr"(b00),
              [acc00] "+&vr"(acc00),
              [acc10] "+&vr"(acc10),
              [acc20] "+&vr"(acc20),
              [acc30] "+vr"(acc30)
            : [a_addr0] "r"(a + (ii + 0) * rsa + kk),
              [a_addr1] "r"(a + (ii + 1) * rsa + kk),
              [a_addr2] "r"(a + (ii + 2) * rsa + kk),
              [a_addr3] "r"(a + (ii + 3) * rsa + kk),
              [b_addr0] "r"(b + kk * rsb + jj),
              [jj_vl] "r"(jj_vl)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

          "vle64.v %[c00], (%[c_addr0]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc00] \n\t"
          "vse64.v %[c00], (%[c_addr0]) \n\t"

          "vle64.v %[c10], (%[c_addr1]) \n\t"
          "vfmul.vf %[c10], %[c10], %[beta] \n\t"
          "vfmacc.vf %[c10], %[alpha], %[acc10] \n\t"
          "vse64.v %[c10], (%[c_addr1]) \n\t"

          "vle64.v %[c20], (%[c_addr2]) \n\t"
          "vfmul.vf %[c20], %[c20], %[beta] \n\t"
          "vfmacc.vf %[c20], %[alpha], %[acc20] \n\t"
          "vse64.v %[c20], (%[c_addr2]) \n\t"

          "vle64.v %[c30], (%[c_addr3]) \n\t"
          "vfmul.vf %[c30], %[c30], %[beta] \n\t"
          "vfmacc.vf %[c30], %[alpha], %[acc30] \n\t"
          "vse64.v %[c30], (%[c_addr3]) \n\t"
          : [c00] "=&vr"(c00),
            [c10] "=&vr"(c10),
            [c20] "=&vr"(c20),
            [c30] "=&vr"(c30)
          : [jj_vl] "r"(jj_vl),
            [c_addr0] "r"(c + (ii + 0) * rsc + jj),
            [c_addr1] "r"(c + (ii + 1) * rsc + jj),
            [c_addr2] "r"(c + (ii + 2) * rsc + jj),
            [c_addr3] "r"(c + (ii + 3) * rsc + jj),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc00] "vr"(acc00),
            [acc10] "vr"(acc10),
            [acc20] "vr"(acc20),
            [acc30] "vr"(acc30)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }
  for (; ii < m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e64m4(n - jj);
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

          "fld %[a00], 0(%[a_addr0]) \n\t"
          "vle64.v %[b00], (%[b_addr0]) \n\t"
          "vfmul.vf %[acc00], %[b00], %[a00] \n\t"
          : [a00] "=&f"(a00),
            [b00] "=&vr"(b00),
            [acc00] "=vr"(acc00)
          : [jj_vl] "r"(jj_vl),
            [a_addr0] "r"(a + ii * rsa),
            [b_addr0] "r"(b + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      for (kk = 1; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

            "fld %[a00], 0(%[a_addr0]) \n\t"
            "vle64.v %[b00], (%[b_addr0]) \n\t"
            "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [b00] "=&vr"(b00),
              [acc00] "+vr"(acc00)
            : [jj_vl] "r"(jj_vl),
              [a_addr0] "r"(a + ii * rsa + kk),
              [b_addr0] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e64, m4, ta, ma \n\t"

          "vle64.v %[c00], (%[c_addr0]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc00] \n\t"
          "vse64.v %[c00], (%[c_addr0]) \n\t"
          : [c00] "=&vr"(c00)
          : [jj_vl] "r"(jj_vl),
            [c_addr0] "r"(c + ii * rsc + jj),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc00] "vr"(acc00)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }
}

SKL_FUNC void skl_gemm_f64_f64_f64_zve64d_x390(size_t m, size_t n, size_t k,
                                               double alpha, const double *a,
                                               size_t rsa, const double *b,
                                               size_t rsb, double beta,
                                               double *c, size_t rsc) {
  skl_gemm_4xm4x12_f64_f64_f64_zve64d_x390(m, n, k, alpha, a, rsa, b, rsb, beta,
                                           c, rsc);
}

#undef NTL_P1
