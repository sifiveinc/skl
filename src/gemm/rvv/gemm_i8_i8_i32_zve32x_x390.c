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

#if defined(__riscv_zihintntl)
#define NTL_P1 "ntl.p1 \n\t"
#else
#define NTL_P1
#endif

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
 * Uses a 4 x LMUL=4 x 4 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e32m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x4_i8_i8_i32_zve32x_x390(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a, size_t rsa,
    const int8_t *b, size_t rsb, int32_t beta, int32_t *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  int32_t a00;
  int32_t a10;
  int32_t a20;
  int32_t a30;
  int32_t a01;
  int32_t a11;
  int32_t a21;
  int32_t a31;
  int32_t a02;
  int32_t a12;
  int32_t a22;
  int32_t a32;
  int32_t a03;
  int32_t a13;
  int32_t a23;
  int32_t a33;
  vint8m1_t b00;
  vint8m1_t b10;
  vint8m1_t b20;
  vint8m1_t b30;
  vint16m2_t b00w;
  vint16m2_t b10w;
  vint16m2_t b20w;
  vint16m2_t b30w;
  vint32m4_t acc00;
  vint32m4_t acc10;
  vint32m4_t acc20;
  vint32m4_t acc30;
  vint32m4_t c00;
  vint32m4_t c10;
  vint32m4_t c20;
  vint32m4_t c30;

  if (k == 0) {
    for (ii = 0; ii < m; ii++) {
      for (jj = 0; jj < n; jj += jj_vl) {
        jj_vl = __riscv_vsetvl_e32m8(n - jj);
        vint32m8_t c0m8 = __riscv_vle32_v_i32m8(c + ii * rsc + jj, jj_vl);
        c0m8 = __riscv_vmul_vx_i32m8(c0m8, beta, jj_vl);
        __riscv_vse32_v_i32m8(c + ii * rsc + jj, c0m8, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; ii + 4 <= m; ii += 4) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e8m1(n - jj);

      const int8_t *a_addr0 = a + (ii + 0) * rsa;
      const int8_t *a_addr1 = a + (ii + 1) * rsa;
      const int8_t *a_addr2 = a + (ii + 2) * rsa;
      const int8_t *a_addr3 = a + (ii + 3) * rsa;
      const int8_t *b_addr0 = b + jj;

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "lb %[a00], 0(%[a_addr0]) \n\t"
          "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"
          "lb %[a10], 0(%[a_addr1]) \n\t"
          "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"
          "lb %[a20], 0(%[a_addr2]) \n\t"
          "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"
          "lb %[a30], 0(%[a_addr3]) \n\t"
          "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"

          "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
          "vle8.v %[b00], (%[b_addr0]) \n\t"
          "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
          "vwcvt.x.x.v %[b00w], %[b00] \n\t"

          "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
          "vwmul.vx %[acc00], %[b00w], %[a00] \n\t"
          "vwmul.vx %[acc10], %[b00w], %[a10] \n\t"
          "vwmul.vx %[acc20], %[b00w], %[a20] \n\t"
          "vwmul.vx %[acc30], %[b00w], %[a30] \n\t"
          : [a_addr0] "+&r" (a_addr0),
            [a_addr1] "+&r" (a_addr1),
            [a_addr2] "+&r" (a_addr2),
            [a_addr3] "+&r" (a_addr3),
            [b_addr0] "+&r" (b_addr0),
            [a00] "=&r" (a00),
            [a10] "=&r" (a10),
            [a20] "=&r" (a20),
            [a30] "=&r" (a30),
            [b00] "=&vr" (b00),
            [b00w] "=&vr" (b00w),
            [acc00] "=&vr" (acc00),
            [acc10] "=&vr" (acc10),
            [acc20] "=&vr" (acc20),
            [acc30] "=vr" (acc30)
          : [a_inc] "r" (sizeof(int8_t)),
            [b_inc] "r" (sizeof(int8_t) * rsb),
            [jj_vl] "r" (jj_vl)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      kk = 1;
      const size_t kk_unroll_degree = 4;
      const size_t preload_distance = 4;

      if (kk_unroll_degree + preload_distance < k) {

        __asm__ volatile(
            // clang-format off
            "\n\t"
            "lb %[a00], 0(%[a_addr0]) \n\t"
            "lb %[a10], 0(%[a_addr1]) \n\t"
            "lb %[a20], 0(%[a_addr2]) \n\t"
            "lb %[a30], 0(%[a_addr3]) \n\t"

            "lb %[a01], 1(%[a_addr0]) \n\t"
            "lb %[a11], 1(%[a_addr1]) \n\t"
            "lb %[a21], 1(%[a_addr2]) \n\t"
            "lb %[a31], 1(%[a_addr3]) \n\t"

            "lb %[a02], 2(%[a_addr0]) \n\t"
            "lb %[a12], 2(%[a_addr1]) \n\t"
            "lb %[a22], 2(%[a_addr2]) \n\t"
            "lb %[a32], 2(%[a_addr3]) \n\t"

            "lb %[a03], 3(%[a_addr0]) \n\t"
            "lb %[a13], 3(%[a_addr1]) \n\t"
            "lb %[a23], 3(%[a_addr2]) \n\t"
            "lb %[a33], 3(%[a_addr3]) \n\t"

            "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"
            "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"
            "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"
            "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"

            "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"

            "vle8.v %[b00], (%[b_addr0]) \n\t"
            "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
            "vwcvt.x.x.v %[b00w], %[b00] \n\t"

            "vle8.v %[b10], (%[b_addr0]) \n\t"
            "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
            "vwcvt.x.x.v %[b10w], %[b10] \n\t"

            "vle8.v %[b20], (%[b_addr0]) \n\t"
            "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
            "vwcvt.x.x.v %[b20w], %[b20] \n\t"

            "vle8.v %[b30], (%[b_addr0]) \n\t"
            "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
            "vwcvt.x.x.v %[b30w], %[b30] \n\t"
            : [a_addr0] "+&r" (a_addr0),
              [a_addr1] "+&r" (a_addr1),
              [a_addr2] "+&r" (a_addr2),
              [a_addr3] "+&r" (a_addr3),
              [b_addr0] "+&r" (b_addr0),
              [a00] "=&r" (a00),
              [a10] "=&r" (a10),
              [a20] "=&r" (a20),
              [a30] "=&r" (a30),
              [a01] "=&r" (a01),
              [a11] "=&r" (a11),
              [a21] "=&r" (a21),
              [a31] "=&r" (a31),
              [a02] "=&r" (a02),
              [a12] "=&r" (a12),
              [a22] "=&r" (a22),
              [a32] "=&r" (a32),
              [a03] "=&r" (a03),
              [a13] "=&r" (a13),
              [a23] "=&r" (a23),
              [a33] "=&r" (a33),
              [b00] "=&vr" (b00),
              [b10] "=&vr" (b10),
              [b20] "=&vr" (b20),
              [b30] "=&vr" (b30),
              [b00w] "=&vr" (b00w),
              [b10w] "=&vr" (b10w),
              [b20w] "=&vr" (b20w),
              [b30w] "=vr" (b30w)
            : [jj_vl] "r" (jj_vl),
              [a_inc] "r" (sizeof(int8_t) * preload_distance),
              [b_inc] "r" (sizeof(int8_t) * rsb)
            : "vtype", "vl", "memory"
            // clang-format on
        );

        for (; kk + kk_unroll_degree + preload_distance <= k;
             kk += kk_unroll_degree) {

          __asm__ volatile(
              // clang-format off
              "\n\t"
              "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
              "vwmacc.vx %[acc00], %[a00], %[b00w] \n\t"
              "vwmacc.vx %[acc10], %[a10], %[b00w] \n\t"
              NTL_P1
              "lb %[a00], 0(%[a_addr0]) \n\t"
              NTL_P1
              "lb %[a10], 0(%[a_addr1]) \n\t"
              "vwmacc.vx %[acc20], %[a20], %[b00w] \n\t"
              "vwmacc.vx %[acc30], %[a30], %[b00w] \n\t"
              NTL_P1
              "lb %[a20], 0(%[a_addr2]) \n\t"
              NTL_P1
              "lb %[a30], 0(%[a_addr3]) \n\t"

              "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
              "vle8.v %[b00], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
              "vwcvt.x.x.v %[b00w], %[b00] \n\t"


              "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
              "vwmacc.vx %[acc00], %[a01], %[b10w] \n\t"
              "vwmacc.vx %[acc10], %[a11], %[b10w] \n\t"
              NTL_P1
              "lb %[a01], 1(%[a_addr0]) \n\t"
              NTL_P1
              "lb %[a11], 1(%[a_addr1]) \n\t"
              "vwmacc.vx %[acc20], %[a21], %[b10w] \n\t"
              "vwmacc.vx %[acc30], %[a31], %[b10w] \n\t"
              NTL_P1
              "lb %[a21], 1(%[a_addr2]) \n\t"
              NTL_P1
              "lb %[a31], 1(%[a_addr3]) \n\t"

              "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
              "vle8.v %[b10], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
              "vwcvt.x.x.v %[b10w], %[b10] \n\t"


              "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
              "vwmacc.vx %[acc00], %[a02], %[b20w] \n\t"
              "vwmacc.vx %[acc10], %[a12], %[b20w] \n\t"
              NTL_P1
              "lb %[a02], 2(%[a_addr0]) \n\t"
              NTL_P1
              "lb %[a12], 2(%[a_addr1]) \n\t"
              "vwmacc.vx %[acc20], %[a22], %[b20w] \n\t"
              "vwmacc.vx %[acc30], %[a32], %[b20w] \n\t"
              NTL_P1
              "lb %[a22], 2(%[a_addr2]) \n\t"
              NTL_P1
              "lb %[a32], 2(%[a_addr3]) \n\t"

              "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
              "vle8.v %[b20], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
              "vwcvt.x.x.v %[b20w], %[b20] \n\t"


              "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
              "vwmacc.vx %[acc00], %[a03], %[b30w] \n\t"
              "vwmacc.vx %[acc10], %[a13], %[b30w] \n\t"
              NTL_P1
              "lb %[a03], 3(%[a_addr0]) \n\t"
              NTL_P1
              "lb %[a13], 3(%[a_addr1]) \n\t"
              "vwmacc.vx %[acc20], %[a23], %[b30w] \n\t"
              "vwmacc.vx %[acc30], %[a33], %[b30w] \n\t"
              NTL_P1
              "lb %[a23], 3(%[a_addr2]) \n\t"
              NTL_P1
              "lb %[a33], 3(%[a_addr3]) \n\t"

              "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
              "vle8.v %[b30], (%[b_addr0]) \n\t"
              "add %[b_addr0], %[b_addr0], %[b_inc] \n\t"
              "vwcvt.x.x.v %[b30w], %[b30] \n\t"

              "add %[a_addr0], %[a_addr0], %[a_inc] \n\t"
              "add %[a_addr1], %[a_addr1], %[a_inc] \n\t"
              "add %[a_addr2], %[a_addr2], %[a_inc] \n\t"
              "add %[a_addr3], %[a_addr3], %[a_inc] \n\t"
              : [a_addr0] "+&r" (a_addr0),
                [a_addr1] "+&r" (a_addr1),
                [a_addr2] "+&r" (a_addr2),
                [a_addr3] "+&r" (a_addr3),
                [b_addr0] "+&r" (b_addr0),
                [a00] "+&r" (a00),
                [a10] "+&r" (a10),
                [a20] "+&r" (a20),
                [a30] "+&r" (a30),
                [a01] "+&r" (a01),
                [a11] "+&r" (a11),
                [a21] "+&r" (a21),
                [a31] "+&r" (a31),
                [a02] "+&r" (a02),
                [a12] "+&r" (a12),
                [a22] "+&r" (a22),
                [a32] "+&r" (a32),
                [a03] "+&r" (a03),
                [a13] "+&r" (a13),
                [a23] "+&r" (a23),
                [a33] "+&r" (a33),
                [b00] "=&vr" (b00),
                [b10] "=&vr" (b10),
                [b20] "=&vr" (b20),
                [b30] "=&vr" (b30),
                [b00w] "+&vr" (b00w),
                [b10w] "+&vr" (b10w),
                [b20w] "+&vr" (b20w),
                [b30w] "+&vr" (b30w),
                [acc00] "+&vr" (acc00),
                [acc10] "+&vr" (acc10),
                [acc20] "+&vr" (acc20),
                [acc30] "+&vr" (acc30)
              : [jj_vl] "r" (jj_vl),
                [a_inc] "r" (sizeof(int8_t) * kk_unroll_degree),
                [b_inc] "r" (sizeof(int8_t) * rsb)
              : "vtype", "vl", "memory"
              // clang-format on
          );
        }

        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"

            "vwmacc.vx %[acc00], %[a00], %[b00w] \n\t"
            "vwmacc.vx %[acc10], %[a10], %[b00w] \n\t"
            "vwmacc.vx %[acc20], %[a20], %[b00w] \n\t"
            "vwmacc.vx %[acc30], %[a30], %[b00w] \n\t"

            "vwmacc.vx %[acc00], %[a01], %[b10w] \n\t"
            "vwmacc.vx %[acc10], %[a11], %[b10w] \n\t"
            "vwmacc.vx %[acc20], %[a21], %[b10w] \n\t"
            "vwmacc.vx %[acc30], %[a31], %[b10w] \n\t"

            "vwmacc.vx %[acc00], %[a02], %[b20w] \n\t"
            "vwmacc.vx %[acc10], %[a12], %[b20w] \n\t"
            "vwmacc.vx %[acc20], %[a22], %[b20w] \n\t"
            "vwmacc.vx %[acc30], %[a32], %[b20w] \n\t"

            "vwmacc.vx %[acc00], %[a03], %[b30w] \n\t"
            "vwmacc.vx %[acc10], %[a13], %[b30w] \n\t"
            "vwmacc.vx %[acc20], %[a23], %[b30w] \n\t"
            "vwmacc.vx %[acc30], %[a33], %[b30w] \n\t"
            : [acc00] "+&vr" (acc00),
              [acc10] "+&vr" (acc10),
              [acc20] "+&vr" (acc20),
              [acc30] "+&vr" (acc30)
            : [jj_vl] "r" (jj_vl),
              [a00] "r" (a00),
              [a10] "r" (a10),
              [a20] "r" (a20),
              [a30] "r" (a30),
              [a01] "r" (a01),
              [a11] "r" (a11),
              [a21] "r" (a21),
              [a31] "r" (a31),
              [a02] "r" (a02),
              [a12] "r" (a12),
              [a22] "r" (a22),
              [a32] "r" (a32),
              [a03] "r" (a03),
              [a13] "r" (a13),
              [a23] "r" (a23),
              [a33] "r" (a33),
              [b00w] "vr" (b00w),
              [b10w] "vr" (b10w),
              [b20w] "vr" (b20w),
              [b30w] "vr" (b30w)
            : "vtype", "vl"
            // clang-format on
        );

        kk += preload_distance;
      }

      for (; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "lb %[a00], 0(%[a_addr0]) \n\t"
            "lb %[a10], 0(%[a_addr1]) \n\t"
            "lb %[a20], 0(%[a_addr2]) \n\t"
            "lb %[a30], 0(%[a_addr3]) \n\t"

            "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
            "vle8.v %[b00], (%[b_addr0]) \n\t"
            "vwcvt.x.x.v %[b00w], %[b00] \n\t"

            "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
            "vwmacc.vx %[acc00], %[a00], %[b00w] \n\t"
            "vwmacc.vx %[acc10], %[a10], %[b00w] \n\t"
            "vwmacc.vx %[acc20], %[a20], %[b00w] \n\t"
            "vwmacc.vx %[acc30], %[a30], %[b00w] \n\t"
            : [a00] "=&r" (a00),
              [a10] "=&r" (a10),
              [a20] "=&r" (a20),
              [a30] "=&r" (a30),
              [b00] "=&vr" (b00),
              [b00w] "=&vr" (b00w),
              [acc00] "+&vr" (acc00),
              [acc10] "+&vr" (acc10),
              [acc20] "+&vr" (acc20),
              [acc30] "+vr" (acc30)
            : [jj_vl] "r" (jj_vl),
              [a_addr0] "r" (a + (ii + 0) * rsa + kk),
              [a_addr1] "r" (a + (ii + 1) * rsa + kk),
              [a_addr2] "r" (a + (ii + 2) * rsa + kk),
              [a_addr3] "r" (a + (ii + 3) * rsa + kk),
              [b_addr0] "r" (b + (kk + 0) * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      int32_t *c_addr = c + (ii + 0) * rsc + jj;
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m4, ta, ma \n\t"

          "vle32.v %[c00], (%[c_addr]) \n\t"
          "vmul.vx %[c00], %[c00], %[beta] \n\t"
          "vmacc.vx %[c00], %[alpha], %[acc00] \n\t"
          "vse32.v %[c00], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[c_inc] \n\t"

          "vle32.v %[c10], (%[c_addr]) \n\t"
          "vmul.vx %[c10], %[c10], %[beta] \n\t"
          "vmacc.vx %[c10], %[alpha], %[acc10] \n\t"
          "vse32.v %[c10], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[c_inc] \n\t"

          "vle32.v %[c20], (%[c_addr]) \n\t"
          "vmul.vx %[c20], %[c20], %[beta] \n\t"
          "vmacc.vx %[c20], %[alpha], %[acc20] \n\t"
          "vse32.v %[c20], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[c_inc] \n\t"

          "vle32.v %[c30], (%[c_addr]) \n\t"
          "vmul.vx %[c30], %[c30], %[beta] \n\t"
          "vmacc.vx %[c30], %[alpha], %[acc30] \n\t"
          "vse32.v %[c30], (%[c_addr]) \n\t"
          : [c00] "=&vr" (c00),
            [c10] "=&vr" (c10),
            [c20] "=&vr" (c20),
            [c30] "=&vr" (c30),
            [c_addr] "+&r" (c_addr)
          : [jj_vl] "r" (jj_vl),
            [acc00] "vr" (acc00),
            [acc10] "vr" (acc10),
            [acc20] "vr" (acc20),
            [acc30] "vr" (acc30),
            [alpha] "r" (alpha),
            [beta] "r" (beta),
            [c_inc] "r" (sizeof(int32_t) * rsc)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }

  for (; ii < m; ii++) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e8m1(n - jj);

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "lb %[a00], 0(%[a_addr0]) \n\t"

          "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
          "vle8.v %[b00], (%[b_addr0]) \n\t"
          "vwcvt.x.x.v %[b00w], %[b00] \n\t"

          "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
          "vwmul.vx %[acc00], %[b00w], %[a00] \n\t"
          : [a00] "=&r" (a00),
            [b00] "=&vr" (b00),
            [b00w] "=&vr" (b00w),
            [acc00] "=vr" (acc00)
          : [jj_vl] "r" (jj_vl),
            [a_addr0] "r" (a + ii * rsa),
            [b_addr0] "r" (b + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      for (kk = 1; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "lb %[a00], 0(%[a_addr0]) \n\t"

            "vsetvli zero, %[jj_vl], e8, m1, ta, ma \n\t"
            "vle8.v %[b00], (%[b_addr0]) \n\t"
            "vwcvt.x.x.v %[b00w], %[b00] \n\t"

            "vsetvli zero, %[jj_vl], e16, m2, ta, ma \n\t"
            "vwmacc.vx %[acc00], %[a00], %[b00w] \n\t"
            : [a00] "=&r" (a00),
              [b00] "=&vr" (b00),
              [b00w] "=&vr" (b00w),
              [acc00] "+vr" (acc00)
            : [jj_vl] "r" (jj_vl),
              [a_addr0] "r" (a + ii * rsa + kk),
              [b_addr0] "r" (b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e32, m4, ta, ma \n\t"
          "vle32.v %[c00], (%[c_addr]) \n\t"
          "vmul.vx %[c00], %[c00], %[beta] \n\t"
          "vmacc.vx %[c00], %[alpha], %[acc00] \n\t"
          "vse32.v %[c00], (%[c_addr]) \n\t"
          : [c00] "=&vr" (c00)
          : [jj_vl] "r" (jj_vl),
            [acc00] "vr" (acc00),
            [alpha] "r" (alpha),
            [beta] "r" (beta),
            [c_addr] "r" (c + ii * rsc + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }
}

SKL_FUNC void skl_gemm_i8_i8_i32_zve32x_x390(size_t m, size_t n, size_t k,
                                             int32_t alpha, const int8_t *a,
                                             size_t rsa, const int8_t *b,
                                             size_t rsb, int32_t beta,
                                             int32_t *c, size_t rsc) {
  skl_gemm_4xm4x4_i8_i8_i32_zve32x_x390(m, n, k, alpha, a, rsa, b, rsb, beta, c,
                                        rsc);
}

#undef NTL_P1
