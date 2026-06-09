// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfh) || __riscv_zvfh < 1000000
#error This file requires the RISC-V zvfh extension, version 1000000.
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
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f16rc_f16rc_f16rc_ref(
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
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e16m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x12_f16_f16_f16_zvfh_x390(
    size_t m, size_t n, size_t k, _Float16 alpha, const _Float16 *a, size_t rsa,
    const _Float16 *b, size_t rsb, _Float16 beta, _Float16 *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  vfloat16m4_t acc00;
  vfloat16m4_t acc10;
  vfloat16m4_t acc20;
  vfloat16m4_t acc30;
  _Float16 a00;
  _Float16 a01;
  _Float16 a02;
  _Float16 a03;
  _Float16 a10;
  _Float16 a11;
  _Float16 a12;
  _Float16 a13;
  _Float16 a20;
  _Float16 a21;
  _Float16 a22;
  _Float16 a23;
  _Float16 a30;
  _Float16 a31;
  _Float16 a32;
  _Float16 a33;
  vfloat16m4_t b00;
  vfloat16m4_t b10;
  vfloat16m4_t b20;
  vfloat16m4_t b30;
  vfloat16m4_t c00;
  vfloat16m4_t c10;
  vfloat16m4_t c20;
  vfloat16m4_t c30;

  if (k == 0) {
    for (ii = 0; ii < m; ii++) {
      for (jj = 0; jj < n; jj += jj_vl) {
        jj_vl = __riscv_vsetvl_e16m8(n - jj);
        vfloat16m8_t c0m8 = __riscv_vle16_v_f16m8(c + ii * rsc + jj, jj_vl);
        c0m8 = __riscv_vfmul_vf_f16m8(c0m8, beta, jj_vl);
        __riscv_vse16_v_f16m8(c + ii * rsc + jj, c0m8, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; ii + 4 <= m; ii += 4) {
    for (jj = 0; jj < n; jj += jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);

      const _Float16 *a_addr0 = a + (ii + 0) * rsa;
      const _Float16 *a_addr1 = a + (ii + 1) * rsa;
      const _Float16 *a_addr2 = a + (ii + 2) * rsa;
      const _Float16 *a_addr3 = a + (ii + 3) * rsa;
      const _Float16 *b_addr = b + jj;

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"
          "vle16.v %[b00], (%[b_addr]) \n\t"
          "add %[b_addr], %[b_addr], %[b_inc] \n\t"

          "flh %[a00], 0(%[a_addr0]) \n\t"
          "addi %[a_addr0], %[a_addr0], 2 \n\t"
          "vfmul.vf %[acc00], %[b00], %[a00] \n\t"

          "flh %[a10], 0(%[a_addr1]) \n\t"
          "addi %[a_addr1], %[a_addr1], 2 \n\t"
          "vfmul.vf %[acc10], %[b00], %[a10] \n\t"

          "flh %[a20], 0(%[a_addr2]) \n\t"
          "addi %[a_addr2], %[a_addr2], 2 \n\t"
          "vfmul.vf %[acc20], %[b00], %[a20] \n\t"

          "flh %[a30], 0(%[a_addr3]) \n\t"
          "addi %[a_addr3], %[a_addr3], 2 \n\t"
          "vfmul.vf %[acc30], %[b00], %[a30] \n\t"
          : [a_addr0] "+&r" (a_addr0),
            [a_addr1] "+&r" (a_addr1),
            [a_addr2] "+&r" (a_addr2),
            [a_addr3] "+&r" (a_addr3),
            [b_addr] "+&r" (b_addr),
            [a00] "=&f"(a00),
            [a10] "=&f"(a10),
            [a20] "=&f"(a20),
            [a30] "=&f"(a30),
            [b00] "=&vr"(b00),
            [b10] "=&vr"(b10),
            [b20] "=&vr"(b20),
            [b30] "=&vr"(b30),
            [acc00] "=&vr"(acc00),
            [acc10] "=&vr"(acc10),
            [acc20] "=&vr"(acc20),
            [acc30] "=vr"(acc30)
          : [jj_vl] "r" (jj_vl),
            [b_inc] "r" (sizeof(_Float16) * rsb)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      const size_t kk_unroll_degree = 12;
      const size_t preload_distance = 4;
      kk = 1;

      if (kk + kk_unroll_degree + preload_distance <= k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"

            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[b_inc] \n\t"

            "vle16.v %[b10], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[b_inc] \n\t"

            "vle16.v %[b20], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[b_inc] \n\t"

            "vle16.v %[b30], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[b_inc] \n\t"

            "flh %[a00], 0(%[a_addr0]) \n\t"
            "flh %[a01], 2(%[a_addr0]) \n\t"
            "flh %[a02], 4(%[a_addr0]) \n\t"
            "flh %[a03], 6(%[a_addr0]) \n\t"
            "addi %[a_addr0], %[a_addr0], 8 \n\t"

            "flh %[a10], 0(%[a_addr1]) \n\t"
            "flh %[a11], 2(%[a_addr1]) \n\t"
            "flh %[a12], 4(%[a_addr1]) \n\t"
            "flh %[a13], 6(%[a_addr1]) \n\t"
            "addi %[a_addr1], %[a_addr1], 8 \n\t"

            "flh %[a20], 0(%[a_addr2]) \n\t"
            "flh %[a21], 2(%[a_addr2]) \n\t"
            "flh %[a22], 4(%[a_addr2]) \n\t"
            "flh %[a23], 6(%[a_addr2]) \n\t"
            "addi %[a_addr2], %[a_addr2], 8 \n\t"

            "flh %[a30], 0(%[a_addr3]) \n\t"
            "flh %[a31], 2(%[a_addr3]) \n\t"
            "flh %[a32], 4(%[a_addr3]) \n\t"
            "flh %[a33], 6(%[a_addr3]) \n\t"
            "addi %[a_addr3], %[a_addr3], 8 \n\t"
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

              [acc00] "=&vr"(acc00),
              [acc10] "=&vr"(acc10),
              [acc20] "=&vr"(acc20),
              [acc30] "=&vr"(acc30),

              [a_addr0] "+&r" (a_addr0),
              [a_addr1] "+&r" (a_addr1),
              [a_addr2] "+&r" (a_addr2),
              [a_addr3] "+&r" (a_addr3),
              [b_addr] "+&r" (b_addr)
            : [jj_vl] "r"(jj_vl),
              [b_inc] "r"(sizeof(_Float16) * rsb)
            : "vtype", "vl", "memory"
            // clang-format on
        );

        for (; kk + kk_unroll_degree + preload_distance < k;
             kk += kk_unroll_degree) {
          __asm__ volatile(
          // clang-format off
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
              "\n\t"
              "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"

              "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
              "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
              NTL_P1
              "flh %[a00],  0(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a10],  0(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
              "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
              NTL_P1
              "flh %[a20],  0(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a30],  0(%[a_addr3]) \n\t"
              "vle16.v %[b00], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
              "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
              NTL_P1
              "flh %[a01],  2(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a11],  2(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
              "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"
              NTL_P1
              "flh %[a21],  2(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a31],  2(%[a_addr3]) \n\t"
              "vle16.v %[b10], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
              "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
              NTL_P1
              "flh %[a02],  4(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a12],  4(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
              "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"
              NTL_P1
              "flh %[a22],  4(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a32],  4(%[a_addr3]) \n\t"
              "vle16.v %[b20], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
              "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
              NTL_P1
              "flh %[a03],  6(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a13],  6(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
              "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
              NTL_P1
              "flh %[a23],  6(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a33],  6(%[a_addr3]) \n\t"
              "vle16.v %[b30], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
              "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
              NTL_P1
              "flh %[a00],  8(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a10],  8(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
              "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
              NTL_P1
              "flh %[a20],  8(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a30],  8(%[a_addr3]) \n\t"
              "vle16.v %[b00], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
              "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
              NTL_P1
              "flh %[a01], 10(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a11], 10(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
              "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"
              NTL_P1
              "flh %[a21], 10(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a31], 10(%[a_addr3]) \n\t"
              "vle16.v %[b10], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
              "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
              NTL_P1
              "flh %[a02], 12(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a12], 12(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
              "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"
              NTL_P1
              "flh %[a22], 12(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a32], 12(%[a_addr3]) \n\t"
              "vle16.v %[b20], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
              "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
              NTL_P1
              "flh %[a03], 14(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a13], 14(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
              "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
              NTL_P1
              "flh %[a23], 14(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a33], 14(%[a_addr3]) \n\t"
              "vle16.v %[b30], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
              "vfmacc.vf %[acc10], %[a10], %[b00] \n\t"
              NTL_P1
              "flh %[a00], 16(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a10], 16(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a20], %[b00] \n\t"
              "vfmacc.vf %[acc30], %[a30], %[b00] \n\t"
              NTL_P1
              "flh %[a20], 16(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a30], 16(%[a_addr3]) \n\t"
              "vle16.v %[b00], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a01], %[b10] \n\t"
              "vfmacc.vf %[acc10], %[a11], %[b10] \n\t"
              NTL_P1
              "flh %[a01], 18(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a11], 18(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a21], %[b10] \n\t"
              "vfmacc.vf %[acc30], %[a31], %[b10] \n\t"
              NTL_P1
              "flh %[a21], 18(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a31], 18(%[a_addr3]) \n\t"
              "vle16.v %[b10], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a02], %[b20] \n\t"
              "vfmacc.vf %[acc10], %[a12], %[b20] \n\t"
              NTL_P1
              "flh %[a02], 20(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a12], 20(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a22], %[b20] \n\t"
              "vfmacc.vf %[acc30], %[a32], %[b20] \n\t"
              NTL_P1
              "flh %[a22], 20(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a32], 20(%[a_addr3]) \n\t"
              "vle16.v %[b20], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

              "vfmacc.vf %[acc00], %[a03], %[b30] \n\t"
              "vfmacc.vf %[acc10], %[a13], %[b30] \n\t"
              NTL_P1
              "flh %[a03], 22(%[a_addr0]) \n\t"
              NTL_P1
              "flh %[a13], 22(%[a_addr1]) \n\t"
              "vfmacc.vf %[acc20], %[a23], %[b30] \n\t"
              "vfmacc.vf %[acc30], %[a33], %[b30] \n\t"
              NTL_P1
              "flh %[a23], 22(%[a_addr2]) \n\t"
              NTL_P1
              "flh %[a33], 22(%[a_addr3]) \n\t"
              "vle16.v %[b30], (%[b_addr]) \n\t"
              "add %[b_addr], %[b_addr], %[b_inc] \n\t"

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
                [b_addr] "+&r"(b_addr)

              : [a_inc] "r"(sizeof(_Float16) * kk_unroll_degree),
                [b_inc] "r"(sizeof(_Float16) * rsb),
                [jj_vl] "r"(jj_vl)
              : "vtype", "vl", "memory"
              // clang-format on
          );
        }

        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"

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
            "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "flh %[a00], 0(%[a_addr0]) \n\t"
            "flh %[a10], 0(%[a_addr1]) \n\t"
            "flh %[a20], 0(%[a_addr2]) \n\t"
            "flh %[a30], 0(%[a_addr3]) \n\t"
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
              [b_addr] "r"(b + kk * rsb + jj),
              [jj_vl] "r"(jj_vl)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"

          "vle16.v %[c00], (%[c_addr0]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc00] \n\t"
          "vse16.v %[c00], (%[c_addr0]) \n\t"

          "vle16.v %[c10], (%[c_addr1]) \n\t"
          "vfmul.vf %[c10], %[c10], %[beta] \n\t"
          "vfmacc.vf %[c10], %[alpha], %[acc10] \n\t"
          "vse16.v %[c10], (%[c_addr1]) \n\t"

          "vle16.v %[c20], (%[c_addr2]) \n\t"
          "vfmul.vf %[c20], %[c20], %[beta] \n\t"
          "vfmacc.vf %[c20], %[alpha], %[acc20] \n\t"
          "vse16.v %[c20], (%[c_addr2]) \n\t"

          "vle16.v %[c30], (%[c_addr3]) \n\t"
          "vfmul.vf %[c30], %[c30], %[beta] \n\t"
          "vfmacc.vf %[c30], %[alpha], %[acc30] \n\t"
          "vse16.v %[c30], (%[c_addr3]) \n\t"
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
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"
          "flh %[a00], 0(%[a_addr0]) \n\t"
          "vle16.v %[b00], (%[b_addr]) \n\t"
          "vfmul.vf %[acc00], %[b00], %[a00] \n\t"
          : [a00] "=&f"(a00),
            [b00] "=&vr"(b00),
            [acc00] "=vr"(acc00)
          : [jj_vl] "r"(jj_vl),
            [a_addr0] "r"(a + ii * rsa),
            [b_addr] "r"(b + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );
      for (kk = 1; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"
            "flh %[a00], 0(%[a_addr0]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "vfmacc.vf %[acc00], %[a00], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [b00] "=&vr"(b00),
              [acc00] "+vr"(acc00)
            : [jj_vl] "r"(jj_vl),
              [a_addr0] "r"(a + ii * rsa + kk),
              [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl], e16, m4, ta, ma \n\t"
          "vle16.v %[c00], (%[c_addr0]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc00] \n\t"
          "vse16.v %[c00], (%[c_addr0]) \n\t"
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

SKL_FUNC void skl_gemm_f16_f16_f16_zvfh_x390(size_t m, size_t n, size_t k,
                                             _Float16 alpha, const _Float16 *a,
                                             size_t rsa, const _Float16 *b,
                                             size_t rsb, _Float16 beta,
                                             _Float16 *c, size_t rsc) {
  skl_gemm_4xm4x12_f16_f16_f16_zvfh_x390(m, n, k, alpha, a, rsa, b, rsb, beta,
                                         c, rsc);
}

#undef NTL_P1
