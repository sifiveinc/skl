// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfh) || __riscv_zvfh < 1000000
#error This file requires the RISC-V Zvfh extension, version 1000000.
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
 * @brief RVV float16 matrix-matrix multiply with float32 output for row-major
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
 * Computes `C = alpha * A * B + beta * C` for FP16 row-major matrices A and
 * B and FP32 row-major matrix C.
 *
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f16rc_f16rc_f32rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 *
 * Uses a 1 x LMUL=8 x 18 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m = 1` and `n >= __riscv_vsetvlmax_e32m8()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_1xm8x18_f16_f16_f32_zvfh_x390(
    size_t m, size_t n, size_t k, float alpha, const _Float16 *a, size_t rsa,
    const _Float16 *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  _Float16 a0;
  _Float16 a1;
  _Float16 a2;
  _Float16 a3;
  _Float16 a4;
  _Float16 a5;
  vfloat16m4_t b0;
  vfloat16m4_t b1;
  vfloat16m4_t b2;
  vfloat16m4_t b3;
  vfloat16m4_t b4;
  vfloat16m4_t b5;
  vfloat32m8_t acc;
  vfloat32m8_t c0;

  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m8(n - jj);
        c0 = __riscv_vle32_v_f32m8(c + ii * rsc + jj, jj_vl);
        c0 = __riscv_vfmul_vf_f32m8(c0, beta, jj_vl);
        __riscv_vse32_v_f32m8(c + ii * rsc + jj, c0, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);

      const size_t kk_unroll_degree = 18;
      const size_t preload_distance = 6;

      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"

          "flh %[a0], 0(%[a_addr]) \n\t"
          "vle16.v %[b0], (%[b_addr]) \n\t"
          "vfwmul.vf %[acc], %[b0], %[a0] \n\t"
          : [a0] "=&f"(a0),
            [b0] "=&vr"(b0),
            [acc] "=vr"(acc)
          : [jj_vl_in] "r"(jj_vl),
            [a_addr] "r"(a + ii * rsa),
            [b_addr] "r"(b + jj)
          : "vtype", "vl", "memory"
          // clang-format on
      );

      kk = 1;

      if (kk_unroll_degree + preload_distance < k) {
        // Load initial values for kk loop below.
        const _Float16 *b_addr = b + kk * rsb + jj;
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"

            "flh %[a0],  0(%[a_addr]) \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "flh %[a1],  2(%[a_addr]) \n\t"
            "vle16.v %[b1], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "flh %[a2],  4(%[a_addr]) \n\t"
            "vle16.v %[b2], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "flh %[a3],  6(%[a_addr]) \n\t"
            "vle16.v %[b3], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "flh %[a4],  8(%[a_addr]) \n\t"
            "vle16.v %[b4], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "flh %[a5], 10(%[a_addr]) \n\t"
            "vle16.v %[b5], (%[b_addr]) \n\t"
            : [a0] "=&f" (a0),
              [a1] "=&f" (a1),
              [a2] "=&f" (a2),
              [a3] "=&f" (a3),
              [a4] "=&f" (a4),
              [a5] "=&f" (a5),
              [b0] "=&vr" (b0),
              [b1] "=&vr" (b1),
              [b2] "=&vr" (b2),
              [b3] "=&vr" (b3),
              [b4] "=&vr" (b4),
              [b5] "=vr" (b5),
              [b_addr] "+&r" (b_addr)
            : [jj_vl_in] "r" (jj_vl),
              [a_addr] "r" (a + ii * rsa + kk),
              [rsb2] "r" (rsb * sizeof(_Float16))
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      for (; (kk + kk_unroll_degree + preload_distance) <= k;
           kk += kk_unroll_degree) {
        const size_t offset = preload_distance;
        const _Float16 *b_addr = b + (kk + offset) * rsb + jj;
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"

            "vfwmacc.vf %[acc], %[a0], %[b0] \n\t"
            NTL_P1
            "flh %[a0],  0(%[a_addr]) \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a1], %[b1] \n\t"
            NTL_P1
            "flh %[a1],  2(%[a_addr]) \n\t"
            "vle16.v %[b1], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a2], %[b2] \n\t"
            NTL_P1
            "flh %[a2],  4(%[a_addr]) \n\t"
            "vle16.v %[b2], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a3], %[b3] \n\t"
            NTL_P1
            "flh %[a3],  6(%[a_addr]) \n\t"
            "vle16.v %[b3], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a4], %[b4] \n\t"
            NTL_P1
            "flh %[a4],  8(%[a_addr]) \n\t"
            "vle16.v %[b4], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a5], %[b5] \n\t"
            NTL_P1
            "flh %[a5], 10(%[a_addr]) \n\t"
            "vle16.v %[b5], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a0], %[b0] \n\t"
            NTL_P1
            "flh %[a0], 12(%[a_addr]) \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a1], %[b1] \n\t"
            NTL_P1
            "flh %[a1], 14(%[a_addr]) \n\t"
            "vle16.v %[b1], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a2], %[b2] \n\t"
            NTL_P1
            "flh %[a2], 16(%[a_addr]) \n\t"
            "vle16.v %[b2], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a3], %[b3] \n\t"
            NTL_P1
            "flh %[a3], 18(%[a_addr]) \n\t"
            "vle16.v %[b3], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a4], %[b4] \n\t"
            NTL_P1
            "flh %[a4], 20(%[a_addr]) \n\t"
            "vle16.v %[b4], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a5], %[b5] \n\t"
            NTL_P1
            "flh %[a5], 22(%[a_addr]) \n\t"
            "vle16.v %[b5], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a0], %[b0] \n\t"
            NTL_P1
            "flh %[a0], 24(%[a_addr]) \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a1], %[b1] \n\t"
            NTL_P1
            "flh %[a1], 26(%[a_addr]) \n\t"
            "vle16.v %[b1], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a2], %[b2] \n\t"
            NTL_P1
            "flh %[a2], 28(%[a_addr]) \n\t"
            "vle16.v %[b2], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a3], %[b3] \n\t"
            NTL_P1
            "flh %[a3], 30(%[a_addr]) \n\t"
            "vle16.v %[b3], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a4], %[b4] \n\t"
            NTL_P1
            "flh %[a4], 32(%[a_addr]) \n\t"
            "vle16.v %[b4], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc], %[a5], %[b5] \n\t"
            NTL_P1
            "flh %[a5], 34(%[a_addr]) \n\t"
            "vle16.v %[b5], (%[b_addr]) \n\t"
            : [a0] "+&f"(a0),
              [a1] "+&f"(a1),
              [a2] "+&f"(a2),
              [a3] "+&f"(a3),
              [a4] "+&f"(a4),
              [a5] "+&f"(a5),
              [b0] "+&vr"(b0),
              [b1] "+&vr"(b1),
              [b2] "+&vr"(b2),
              [b3] "+&vr"(b3),
              [b4] "+&vr"(b4),
              [b5] "+&vr"(b5),
              [acc] "+&vr"(acc),
              [b_addr] "+&r"(b_addr)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr] "r"(a + ii  * rsa + kk + offset),
              [rsb2] "r" (rsb * sizeof(_Float16))
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      if (kk_unroll_degree + preload_distance < k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
            "vfwmacc.vf %[acc], %[a0], %[b0] \n\t"
            "vfwmacc.vf %[acc], %[a1], %[b1] \n\t"
            "vfwmacc.vf %[acc], %[a2], %[b2] \n\t"
            "vfwmacc.vf %[acc], %[a3], %[b3] \n\t"
            "vfwmacc.vf %[acc], %[a4], %[b4] \n\t"
            "vfwmacc.vf %[acc], %[a5], %[b5] \n\t"
            : [acc] "+&vr"(acc)
            : [jj_vl_in] "r" (jj_vl),
              [a0] "f"(a0),
              [a1] "f"(a1),
              [a2] "f"(a2),
              [a3] "f"(a3),
              [a4] "f"(a4),
              [a5] "f"(a5),
              [b0] "vr"(b0),
              [b1] "vr"(b1),
              [b2] "vr"(b2),
              [b3] "vr"(b3),
              [b4] "vr"(b4),
              [b5] "vr"(b5)
            : "vtype", "vl"
            // clang-format on
        );
        kk += preload_distance;
      }

      for (; (kk + 1) <= k; kk = kk + 1) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"

            "flh %[a0], 0(%[a_addr]) \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "vfwmacc.vf %[acc], %[a0], %[b0] \n\t"
            : [a0] "=&f"(a0),
              [b0] "=&vr"(b0),
              [acc] "+vr"(acc)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr] "r"(a + ii  * rsa + kk),
              [b_addr] "r"(b + kk  * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m8, ta, ma \n\t"

          "vle32.v %[c0], (%[c_addr]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc] \n\t"
          "vse32.v %[c0], (%[c_addr]) \n\t"
          : [c0] "=&vr"(c0)
          : [jj_vl_in] "r"(jj_vl),
            [c_addr] "r"(c + ii * rsc + jj),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc] "vr"(acc)
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
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
 * Functionally equivalent to calling:
 * ```
 * skl_gemm_f16rc_f16rc_f32rc_ref(
 *     m, n, k,
 *     alpha,
 *     a, rsa, 1,
 *     b, rsb, 1,
 *     beta,
 *     c, rsc, 1
 * );
 * ```
 * Uses a 6 x LMUL=4 x 12 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 6` and `n >= __riscv_vsetvlmax_e32m4()`.
 */
SKL_FUNC_PRIVATE void skl_gemm_6xm4x12_f16_f16_f32_zvfh_x390(
    size_t m, size_t n, size_t k, float alpha, const _Float16 *a, size_t rsa,
    const _Float16 *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk;
  size_t ii0;
  size_t kk_peel;
  _Float16 a00;
  _Float16 a10;
  _Float16 a20;
  _Float16 a30;
  _Float16 a40;
  _Float16 a50;
  _Float16 a01;
  _Float16 a11;
  _Float16 a21;
  _Float16 a31;
  _Float16 a41;
  _Float16 a51;
  _Float16 a02;
  _Float16 a12;
  _Float16 a22;
  _Float16 a32;
  _Float16 a42;
  _Float16 a52;
  _Float16 a03;
  _Float16 a13;
  _Float16 a23;
  _Float16 a33;
  _Float16 a43;
  _Float16 a53;
  vfloat16m2_t b00;
  vfloat16m2_t b10;
  vfloat16m2_t b20;
  vfloat16m2_t b30;
  vfloat32m4_t acc0;
  vfloat32m4_t acc1;
  vfloat32m4_t acc2;
  vfloat32m4_t acc3;
  vfloat32m4_t acc4;
  vfloat32m4_t acc5;
  _Float16 a0;
  vfloat16m2_t b0;
  vfloat32m4_t acc;
  vfloat32m4_t c0;
  vfloat32m4_t c1;

  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e32m4(n - jj);
        c0 = __riscv_vle32_v_f32m4(c + ii * rsc + jj, jj_vl);
        c0 = __riscv_vfmul_vf_f32m4(c0, beta, jj_vl);
        __riscv_vse32_v_f32m4(c + ii * rsc + jj, c0, jj_vl);
      }
    }
    return;
  }

  for (ii = 0; (ii + 6) <= m; ii = ii + 6) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m2(n - jj);

      {
        const _Float16 *a_addr = a + ii * rsa;
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"

            "flh %[a00], 0(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "flh %[a10], 0(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "vfwmul.vf %[acc0], %[b00], %[a00] \n\t"
            "vfwmul.vf %[acc1], %[b00], %[a10] \n\t"
            "flh %[a20], 0(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "flh %[a30], 0(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "vfwmul.vf %[acc2], %[b00], %[a20] \n\t"
            "vfwmul.vf %[acc3], %[b00], %[a30] \n\t"
            "flh %[a40], 0(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "flh %[a50], 0(%[a_addr]) \n\t"
            "vfwmul.vf %[acc4], %[b00], %[a40] \n\t"
            "vfwmul.vf %[acc5], %[b00], %[a50] \n\t"
            : [a00] "=&f"(a00),
              [a10] "=&f"(a10),
              [a20] "=&f"(a20),
              [a30] "=&f"(a30),
              [a40] "=&f"(a40),
              [a50] "=&f"(a50),
              [a_addr] "+&r"(a_addr),
              [b00] "=&vr"(b00),
              [acc0] "=&vr"(acc0),
              [acc1] "=&vr"(acc1),
              [acc2] "=&vr"(acc2),
              [acc3] "=&vr"(acc3),
              [acc4] "=&vr"(acc4),
              [acc5] "=vr"(acc5)
            : [jj_vl_in] "r"(jj_vl),
              [rsa2] "r" (rsa * sizeof(_Float16)),
              [b_addr] "r"(b + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      const size_t kk_unroll_degree = 12;
      const size_t preload_distance = 4;

      if (kk_unroll_degree + preload_distance < k) {
        const size_t offset = 1;
        const _Float16 *a_addr = a + ii * rsa + offset;
        const _Float16 *b_addr = b + offset * rsb + jj;
        __asm__ volatile(
            // clang-format off
            "\n\t"

            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"

            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            NTL_P1
            "flh %[a00], 0(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a01], 2(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a02], 4(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a03], 6(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            NTL_P1
            "flh %[a10], 0(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a11], 2(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a12], 4(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a13], 6(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "vle16.v %[b10], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            NTL_P1
            "flh %[a20], 0(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a21], 2(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a22], 4(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a23], 6(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            NTL_P1
            "flh %[a30], 0(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a31], 2(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a32], 4(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a33], 6(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            "vle16.v %[b20], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"
            NTL_P1
            "flh %[a40], 0(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a41], 2(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a42], 4(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a43], 6(%[a_addr]) \n\t"
            "add %[a_addr], %[a_addr], %[rsa2] \n\t"
            NTL_P1
            "flh %[a50], 0(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a51], 2(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a52], 4(%[a_addr]) \n\t"
            NTL_P1
            "flh %[a53], 6(%[a_addr]) \n\t"
            "vle16.v %[b30], (%[b_addr]) \n\t"

          : [a00] "=&f"(a00),
            [a10] "=&f"(a10),
            [a20] "=&f"(a20),
            [a30] "=&f"(a30),
            [a40] "=&f"(a40),
            [a50] "=&f"(a50),
            [a01] "=&f"(a01),
            [a11] "=&f"(a11),
            [a21] "=&f"(a21),
            [a31] "=&f"(a31),
            [a41] "=&f"(a41),
            [a51] "=&f"(a51),
            [a02] "=&f"(a02),
            [a12] "=&f"(a12),
            [a22] "=&f"(a22),
            [a32] "=&f"(a32),
            [a42] "=&f"(a42),
            [a52] "=&f"(a52),
            [a03] "=&f"(a03),
            [a13] "=&f"(a13),
            [a23] "=&f"(a23),
            [a33] "=&f"(a33),
            [a43] "=&f"(a43),
            [a53] "=&f"(a53),
            [a_addr] "+&r"(a_addr),
            [b00] "=&vr"(b00),
            [b10] "=&vr"(b10),
            [b20] "=&vr"(b20),
            [b30] "=vr"(b30),
            [b_addr] "+&r"(b_addr)
          : [jj_vl_in] "r"(jj_vl),
            [rsa2] "r" (rsa * sizeof(_Float16)),
            [rsb2] "r" (rsb * sizeof(_Float16))
          : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      for (kk = 1; (kk + kk_unroll_degree + preload_distance) <= k;
           kk += kk_unroll_degree) {
        const size_t offset = preload_distance;
        const _Float16 *b_addr = b + (kk + offset) * rsb + jj;
        const _Float16 *a_addr_0 = a + (ii + 0) * rsa + kk + offset;
        const _Float16 *a_addr_1;
        const _Float16 *a_addr_2;
        const _Float16 *a_addr_3;
        const _Float16 *a_addr_4;
        const _Float16 *a_addr_5;
        __asm__ volatile(
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
            // clang-format off
            "\n\t"
            "add %[a_addr_1], %[a_addr_0], %[rsa2] \n\t"
            "add %[a_addr_2], %[a_addr_1], %[rsa2] \n\t"
            "add %[a_addr_3], %[a_addr_2], %[rsa2] \n\t"
            "add %[a_addr_4], %[a_addr_3], %[rsa2] \n\t"
            "add %[a_addr_5], %[a_addr_4], %[rsa2] \n\t"

            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"

            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"
            NTL_P1
            "flh %[a00],  0(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a10],  0(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfwmacc.vf %[acc3], %[a30], %[b00] \n\t"
            NTL_P1
            "flh %[a20],  0(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a30],  0(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfwmacc.vf %[acc5], %[a50], %[b00] \n\t"
            NTL_P1
            "flh %[a40],  0(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a50],  0(%[a_addr_5]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfwmacc.vf %[acc1], %[a11], %[b10] \n\t"
            NTL_P1
            "flh %[a01],  2(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a11],  2(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfwmacc.vf %[acc3], %[a31], %[b10] \n\t"
            NTL_P1
            "flh %[a21],  2(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a31],  2(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a41], %[b10] \n\t"
            "vfwmacc.vf %[acc5], %[a51], %[b10] \n\t"
            NTL_P1
            "flh %[a41],  2(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a51],  2(%[a_addr_5]) \n\t"
            "vle16.v %[b10], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfwmacc.vf %[acc1], %[a12], %[b20] \n\t"
            NTL_P1
            "flh %[a02],  4(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a12],  4(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfwmacc.vf %[acc3], %[a32], %[b20] \n\t"
            NTL_P1
            "flh %[a22],  4(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a32],  4(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a42], %[b20] \n\t"
            "vfwmacc.vf %[acc5], %[a52], %[b20] \n\t"
            NTL_P1
            "flh %[a42],  4(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a52],  4(%[a_addr_5]) \n\t"
            "vle16.v %[b20], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfwmacc.vf %[acc1], %[a13], %[b30] \n\t"
            NTL_P1
            "flh %[a03],  6(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a13],  6(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfwmacc.vf %[acc3], %[a33], %[b30] \n\t"
            NTL_P1
            "flh %[a23],  6(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a33],  6(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a43], %[b30] \n\t"
            "vfwmacc.vf %[acc5], %[a53], %[b30] \n\t"
            NTL_P1
            "flh %[a43],  6(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a53],  6(%[a_addr_5]) \n\t"
            "vle16.v %[b30], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"
            NTL_P1
            "flh %[a00],  8(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a10],  8(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfwmacc.vf %[acc3], %[a30], %[b00] \n\t"
            NTL_P1
            "flh %[a20],  8(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a30],  8(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfwmacc.vf %[acc5], %[a50], %[b00] \n\t"
            NTL_P1
            "flh %[a40],  8(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a50],  8(%[a_addr_5]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfwmacc.vf %[acc1], %[a11], %[b10] \n\t"
            NTL_P1
            "flh %[a01], 10(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a11], 10(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfwmacc.vf %[acc3], %[a31], %[b10] \n\t"
            NTL_P1
            "flh %[a21], 10(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a31], 10(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a41], %[b10] \n\t"
            "vfwmacc.vf %[acc5], %[a51], %[b10] \n\t"
            NTL_P1
            "flh %[a41], 10(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a51], 10(%[a_addr_5]) \n\t"
            "vle16.v %[b10], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfwmacc.vf %[acc1], %[a12], %[b20] \n\t"
            NTL_P1
            "flh %[a02], 12(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a12], 12(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfwmacc.vf %[acc3], %[a32], %[b20] \n\t"
            NTL_P1
            "flh %[a22], 12(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a32], 12(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a42], %[b20] \n\t"
            "vfwmacc.vf %[acc5], %[a52], %[b20] \n\t"
            NTL_P1
            "flh %[a42], 12(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a52], 12(%[a_addr_5]) \n\t"
            "vle16.v %[b20], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfwmacc.vf %[acc1], %[a13], %[b30] \n\t"
            NTL_P1
            "flh %[a03], 14(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a13], 14(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfwmacc.vf %[acc3], %[a33], %[b30] \n\t"
            NTL_P1
            "flh %[a23], 14(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a33], 14(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a43], %[b30] \n\t"
            "vfwmacc.vf %[acc5], %[a53], %[b30] \n\t"
            NTL_P1
            "flh %[a43], 14(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a53], 14(%[a_addr_5]) \n\t"
            "vle16.v %[b30], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"
            NTL_P1
            "flh %[a00], 16(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a10], 16(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfwmacc.vf %[acc3], %[a30], %[b00] \n\t"
            NTL_P1
            "flh %[a20], 16(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a30], 16(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfwmacc.vf %[acc5], %[a50], %[b00] \n\t"
            NTL_P1
            "flh %[a40], 16(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a50], 16(%[a_addr_5]) \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfwmacc.vf %[acc1], %[a11], %[b10] \n\t"
            NTL_P1
            "flh %[a01], 18(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a11], 18(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfwmacc.vf %[acc3], %[a31], %[b10] \n\t"
            NTL_P1
            "flh %[a21], 18(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a31], 18(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a41], %[b10] \n\t"
            "vfwmacc.vf %[acc5], %[a51], %[b10] \n\t"
            NTL_P1
            "flh %[a41], 18(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a51], 18(%[a_addr_5]) \n\t"
            "vle16.v %[b10], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfwmacc.vf %[acc1], %[a12], %[b20] \n\t"
            NTL_P1
            "flh %[a02], 20(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a12], 20(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfwmacc.vf %[acc3], %[a32], %[b20] \n\t"
            NTL_P1
            "flh %[a22], 20(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a32], 20(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a42], %[b20] \n\t"
            "vfwmacc.vf %[acc5], %[a52], %[b20] \n\t"
            NTL_P1
            "flh %[a42], 20(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a52], 20(%[a_addr_5]) \n\t"
            "vle16.v %[b20], (%[b_addr]) \n\t"
            "add %[b_addr], %[b_addr], %[rsb2] \n\t"

            "vfwmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfwmacc.vf %[acc1], %[a13], %[b30] \n\t"
            NTL_P1
            "flh %[a03], 22(%[a_addr_0]) \n\t"
            NTL_P1
            "flh %[a13], 22(%[a_addr_1]) \n\t"
            "vfwmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfwmacc.vf %[acc3], %[a33], %[b30] \n\t"
            NTL_P1
            "flh %[a23], 22(%[a_addr_2]) \n\t"
            NTL_P1
            "flh %[a33], 22(%[a_addr_3]) \n\t"
            "vfwmacc.vf %[acc4], %[a43], %[b30] \n\t"
            "vfwmacc.vf %[acc5], %[a53], %[b30] \n\t"
            NTL_P1
            "flh %[a43], 22(%[a_addr_4]) \n\t"
            NTL_P1
            "flh %[a53], 22(%[a_addr_5]) \n\t"
            "vle16.v %[b30], (%[b_addr]) \n\t"
#pragma clang diagnostic pop
            : [a00] "+&f"(a00),
              [a10] "+&f"(a10),
              [a20] "+&f"(a20),
              [a30] "+&f"(a30),
              [a40] "+&f"(a40),
              [a50] "+&f"(a50),
              [a01] "+&f"(a01),
              [a11] "+&f"(a11),
              [a21] "+&f"(a21),
              [a31] "+&f"(a31),
              [a41] "+&f"(a41),
              [a51] "+&f"(a51),
              [a02] "+&f"(a02),
              [a12] "+&f"(a12),
              [a22] "+&f"(a22),
              [a32] "+&f"(a32),
              [a42] "+&f"(a42),
              [a52] "+&f"(a52),
              [a03] "+&f"(a03),
              [a13] "+&f"(a13),
              [a23] "+&f"(a23),
              [a33] "+&f"(a33),
              [a43] "+&f"(a43),
              [a53] "+&f"(a53),
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
              [b_addr] "+&r"(b_addr),
              [a_addr_1] "=&r" (a_addr_1),
              [a_addr_2] "=&r" (a_addr_2),
              [a_addr_3] "=&r" (a_addr_3),
              [a_addr_4] "=&r" (a_addr_4),
              [a_addr_5] "=&r" (a_addr_5)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr_0] "r" (a_addr_0),
              [rsa2] "r" (rsa * sizeof(_Float16)),
              [rsb2] "r" (rsb * sizeof(_Float16))
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      if (kk_unroll_degree + preload_distance < k) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"

            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfwmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfwmacc.vf %[acc3], %[a30], %[b00] \n\t"
            "vfwmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfwmacc.vf %[acc5], %[a50], %[b00] \n\t"

            "vfwmacc.vf %[acc0], %[a01], %[b10] \n\t"
            "vfwmacc.vf %[acc1], %[a11], %[b10] \n\t"
            "vfwmacc.vf %[acc2], %[a21], %[b10] \n\t"
            "vfwmacc.vf %[acc3], %[a31], %[b10] \n\t"
            "vfwmacc.vf %[acc4], %[a41], %[b10] \n\t"
            "vfwmacc.vf %[acc5], %[a51], %[b10] \n\t"

            "vfwmacc.vf %[acc0], %[a02], %[b20] \n\t"
            "vfwmacc.vf %[acc1], %[a12], %[b20] \n\t"
            "vfwmacc.vf %[acc2], %[a22], %[b20] \n\t"
            "vfwmacc.vf %[acc3], %[a32], %[b20] \n\t"
            "vfwmacc.vf %[acc4], %[a42], %[b20] \n\t"
            "vfwmacc.vf %[acc5], %[a52], %[b20] \n\t"

            "vfwmacc.vf %[acc0], %[a03], %[b30] \n\t"
            "vfwmacc.vf %[acc1], %[a13], %[b30] \n\t"
            "vfwmacc.vf %[acc2], %[a23], %[b30] \n\t"
            "vfwmacc.vf %[acc3], %[a33], %[b30] \n\t"
            "vfwmacc.vf %[acc4], %[a43], %[b30] \n\t"
            "vfwmacc.vf %[acc5], %[a53], %[b30] \n\t"
            : [acc0] "+&vr" (acc0),
              [acc1] "+&vr" (acc1),
              [acc2] "+&vr" (acc2),
              [acc3] "+&vr" (acc3),
              [acc4] "+&vr" (acc4),
              [acc5] "+&vr" (acc5)
            : [jj_vl_in] "r"(jj_vl),
              [b00] "vr" (b00),
              [b10] "vr" (b10),
              [b20] "vr" (b20),
              [b30] "vr" (b30),
              [a00] "f"(a00),
              [a10] "f"(a10),
              [a20] "f"(a20),
              [a30] "f"(a30),
              [a40] "f"(a40),
              [a50] "f"(a50),
              [a01] "f"(a01),
              [a11] "f"(a11),
              [a21] "f"(a21),
              [a31] "f"(a31),
              [a41] "f"(a41),
              [a51] "f"(a51),
              [a02] "f"(a02),
              [a12] "f"(a12),
              [a22] "f"(a22),
              [a32] "f"(a32),
              [a42] "f"(a42),
              [a52] "f"(a52),
              [a03] "f"(a03),
              [a13] "f"(a13),
              [a23] "f"(a23),
              [a33] "f"(a33),
              [a43] "f"(a43),
              [a53] "f"(a53)
            : "vtype", "vl", "memory"
            // clang-format on
        );
        kk += preload_distance;
      }

      for (; kk < k; kk++) {
        __asm__ volatile(
            // clang-format off
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"
            "vle16.v %[b00], (%[b_addr]) \n\t"
            "flh %[a00], 0(%[a_addr_0]) \n\t"
            "flh %[a10], 0(%[a_addr_1]) \n\t"
            "flh %[a20], 0(%[a_addr_2]) \n\t"
            "flh %[a30], 0(%[a_addr_3]) \n\t"
            "flh %[a40], 0(%[a_addr_4]) \n\t"
            "flh %[a50], 0(%[a_addr_5]) \n\t"
            "vfwmacc.vf %[acc0], %[a00], %[b00] \n\t"
            "vfwmacc.vf %[acc1], %[a10], %[b00] \n\t"
            "vfwmacc.vf %[acc2], %[a20], %[b00] \n\t"
            "vfwmacc.vf %[acc3], %[a30], %[b00] \n\t"
            "vfwmacc.vf %[acc4], %[a40], %[b00] \n\t"
            "vfwmacc.vf %[acc5], %[a50], %[b00] \n\t"
            : [a00] "=&f"(a00),
              [a10] "=&f"(a10),
              [a20] "=&f"(a20),
              [a30] "=&f"(a30),
              [a40] "=&f"(a40),
              [a50] "=&f"(a50),
              [b00] "=&vr"(b00),
              [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1),
              [acc2] "+&vr"(acc2),
              [acc3] "+&vr"(acc3),
              [acc4] "+&vr"(acc4),
              [acc5] "+vr"(acc5)
            : [jj_vl_in] "r"(jj_vl),
              [a_addr_0] "r"(a + (ii + 0) * rsa + kk),
              [a_addr_1] "r"(a + (ii + 1) * rsa + kk),
              [a_addr_2] "r"(a + (ii + 2) * rsa + kk),
              [a_addr_3] "r"(a + (ii + 3) * rsa + kk),
              [a_addr_4] "r"(a + (ii + 4) * rsa + kk),
              [a_addr_5] "r"(a + (ii + 5) * rsa + kk),
              [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory"
            // clang-format on
        );
      }

      float *c_addr = c + ii * rsc + jj;
      __asm__ volatile(
          // clang-format off
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"

          "vle32.v %[c0], (%[c_addr]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc0] \n\t"
          "vse32.v %[c0], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[rsc4] \n\t"

          "vle32.v %[c1], (%[c_addr]) \n\t"
          "vfmul.vf %[c1], %[c1], %[beta] \n\t"
          "vfmacc.vf %[c1], %[alpha], %[acc1] \n\t"
          "vse32.v %[c1], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[rsc4] \n\t"

          "vle32.v %[c0], (%[c_addr]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc2] \n\t"
          "vse32.v %[c0], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[rsc4] \n\t"

          "vle32.v %[c1], (%[c_addr]) \n\t"
          "vfmul.vf %[c1], %[c1], %[beta] \n\t"
          "vfmacc.vf %[c1], %[alpha], %[acc3] \n\t"
          "vse32.v %[c1], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[rsc4] \n\t"

          "vle32.v %[c0], (%[c_addr]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc4] \n\t"
          "vse32.v %[c0], (%[c_addr]) \n\t"
          "add %[c_addr], %[c_addr], %[rsc4] \n\t"

          "vle32.v %[c1], (%[c_addr]) \n\t"
          "vfmul.vf %[c1], %[c1], %[beta] \n\t"
          "vfmacc.vf %[c1], %[alpha], %[acc5] \n\t"
          "vse32.v %[c1], (%[c_addr]) \n\t"
          : [c0] "=&vr"(c0),
            [c1] "=&vr"(c1),
            [c_addr] "+&r"(c_addr)
          : [jj_vl_in] "r"(jj_vl),
            [beta] "f"(beta),
            [alpha] "f"(alpha),
            [acc0] "vr"(acc0),
            [acc1] "vr"(acc1),
            [acc2] "vr"(acc2),
            [acc3] "vr"(acc3),
            [acc4] "vr"(acc4),
            [acc5] "vr"(acc5),
            [rsc4] "r"(rsc * sizeof(float))
          : "vtype", "vl", "memory"
          // clang-format on
      );
    }
  }

  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m2(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        __asm__ volatile(
            "\n\t"
            "flh %[a0], 0(%[a_addr_83]) \n\t"
            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"
            "vle16.v %[b0], (%[b_addr_13]) \n\t"
            "vfwmul.vf %[acc], %[b0], %[a0] \n\t"
            : [a0] "=&f"(a0), [b0] "=&vr"(b0), [acc] "=vr"(acc)
            : [a_addr_83] "r"(a + (((ii0 + 0) * rsa) + ((kk_peel + 0) * 1))),
              [jj_vl_in] "r"(jj_vl),
              [b_addr_13] "r"(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        __asm__ volatile(
            "\n\t"
            "flh %[a0], 0(%[a_addr_84]) \n\t"
            "vsetvli zero, %[jj_vl_in], e16, m2, ta, ma \n\t"
            "vle16.v %[b0], (%[b_addr_14]) \n\t"
            "vfwmacc.vf %[acc], %[a0], %[b0] \n\t"
            : [a0] "=&f"(a0), [b0] "=&vr"(b0), [acc] "+vr"(acc)
            : [a_addr_84] "r"(a + (((ii0 + 0) * rsa) + ((kk + 0) * 1))),
              [jj_vl_in] "r"(jj_vl),
              [b_addr_14] "r"(b + (((kk + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e32, m4, ta, ma \n\t"
          "vle32.v %[c0], (%[c_addr_5]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc] \n\t"
          "vse32.v %[c0], (%[c_addr_5]) \n\t"
          : [c0] "=&vr"(c0)
          : [jj_vl_in] "r"(jj_vl),
            [c_addr_5] "r"(c + (((ii0 + 0) * rsc) + ((jj + 0) * 1))),
            [beta] "f"(beta), [alpha] "f"(alpha), [acc] "vr"(acc)
          : "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_gemm_f16_f16_f32_zvfh_x390(size_t m, size_t n, size_t k,
                                             float alpha, const _Float16 *a,
                                             size_t rsa, const _Float16 *b,
                                             size_t rsb, float beta, float *c,
                                             size_t rsc) {
  if (m == 1) {
    skl_gemm_1xm8x18_f16_f16_f32_zvfh_x390(m, n, k, alpha, a, rsc, b, rsb, beta,
                                           c, rsc);
    return;
  }
  skl_gemm_6xm4x12_f16_f16_f32_zvfh_x390(m, n, k, alpha, a, rsa, b, rsb, beta,
                                         c, rsc);
}

#undef NTL_P1
