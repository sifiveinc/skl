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
  _Float16 a00;
  _Float16 a01;
  _Float16 a02;
  _Float16 a03;
  vfloat16m4_t b0;
  vfloat16m4_t acc0;
  vfloat16m4_t acc1;
  vfloat16m4_t acc2;
  vfloat16m4_t acc3;
  vfloat16m4_t c00;
  vfloat16m4_t c01;
  vfloat16m4_t c02;
  vfloat16m4_t c03;
  _Float16 a0;
  vfloat16m4_t acc;
  vfloat16m4_t c0;
  if (k == 0) {
    for (ii = 0; (ii + 1) <= m; ii = ii + 1) {
      for (jj = 0; jj < n; jj = jj + jj_vl) {
        jj_vl = __riscv_vsetvl_e16m4(n - jj);
        c0 = __riscv_vle16_v_f16m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)),
                                   jj_vl);
        c0 = __riscv_vfmul_vf_f16m4(c0, beta, jj_vl);
        __riscv_vse16_v_f16m4(c + (((ii + 0) * rsc) + ((jj + 0) * 1)), c0,
                              jj_vl);
      }
    }
    return;
  }
  for (ii = 0; (ii + 4) <= m; ii = ii + 4) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      const _Float16 *a_addr0 = a + (ii + 0) * rsa;
      const _Float16 *a_addr1 = a + (ii + 1) * rsa;
      const _Float16 *a_addr2 = a + (ii + 2) * rsa;
      const _Float16 *a_addr3 = a + (ii + 3) * rsa;
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
          "vle16.v %[b0], (%[b_load]) \n\t"
          "flh %[a00], 0(%[a_addr0]) \n\t"
          "addi %[a_addr0], %[a_addr0], 2 \n\t"
          "flh %[a01], 0(%[a_addr1]) \n\t"
          "addi %[a_addr1], %[a_addr1], 2 \n\t"
          "flh %[a02], 0(%[a_addr2]) \n\t"
          "addi %[a_addr2], %[a_addr2], 2 \n\t"
          "flh %[a03], 0(%[a_addr3]) \n\t"
          "addi %[a_addr3], %[a_addr3], 2 \n\t"
          "vfmul.vf %[acc0], %[b0], %[a00] \n\t"
          "vfmul.vf %[acc1], %[b0], %[a01] \n\t"
          "vfmul.vf %[acc2], %[b0], %[a02] \n\t"
          "vfmul.vf %[acc3], %[b0], %[a03] \n\t"
          : [a00] "=&f"(a00), [a01] "=&f"(a01), [a02] "=&f"(a02),
            [a03] "=&f"(a03), [b0] "=&vr"(b0), [acc0] "=&vr"(acc0),
            [acc1] "=&vr"(acc1), [acc2] "=&vr"(acc2), [acc3] "=vr"(acc3),
            [a_addr0] "+&r"(a_addr0), [a_addr1] "+&r"(a_addr1),
            [a_addr2] "+&r"(a_addr2), [a_addr3] "+&r"(a_addr3)
          : [jj_vl_in] "r"(jj_vl), [b_load] "r"(b + jj)
          : "vtype", "vl", "memory");
      for (kk = 1; (kk + 1) <= k; kk = kk + 1) {
        __asm__ volatile(
            "\n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "flh %[a00], 0(%[a_addr0]) \n\t"
            "addi %[a_addr0], %[a_addr0], 2 \n\t"
            "flh %[a01], 0(%[a_addr1]) \n\t"
            "addi %[a_addr1], %[a_addr1], 2 \n\t"
            "flh %[a02], 0(%[a_addr2]) \n\t"
            "addi %[a_addr2], %[a_addr2], 2 \n\t"
            "flh %[a03], 0(%[a_addr3]) \n\t"
            "addi %[a_addr3], %[a_addr3], 2 \n\t"
            "vfmacc.vf %[acc0], %[a00], %[b0] \n\t"
            "vfmacc.vf %[acc1], %[a01], %[b0] \n\t"
            "vfmacc.vf %[acc2], %[a02], %[b0] \n\t"
            "vfmacc.vf %[acc3], %[a03], %[b0] \n\t"
            : [a00] "=&f"(a00), [a01] "=&f"(a01), [a02] "=&f"(a02),
              [a03] "=&f"(a03), [b0] "=&vr"(b0), [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1), [acc2] "+&vr"(acc2), [acc3] "+vr"(acc3),
              [a_addr0] "+&r"(a_addr0), [a_addr1] "+&r"(a_addr1),
              [a_addr2] "+&r"(a_addr2), [a_addr3] "+&r"(a_addr3)
            : [jj_vl_in] "r"(jj_vl), [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
          "vle16.v %[c00], (%[c_addr0]) \n\t"
          "vle16.v %[c01], (%[c_addr1]) \n\t"
          "vle16.v %[c02], (%[c_addr2]) \n\t"
          "vle16.v %[c03], (%[c_addr3]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta] \n\t"
          "vfmul.vf %[c01], %[c01], %[beta] \n\t"
          "vfmul.vf %[c02], %[c02], %[beta] \n\t"
          "vfmul.vf %[c03], %[c03], %[beta] \n\t"
          "vfmacc.vf %[c00], %[alpha], %[acc0] \n\t"
          "vfmacc.vf %[c01], %[alpha], %[acc1] \n\t"
          "vfmacc.vf %[c02], %[alpha], %[acc2] \n\t"
          "vfmacc.vf %[c03], %[alpha], %[acc3] \n\t"
          "vse16.v %[c00], (%[c_addr0]) \n\t"
          "vse16.v %[c01], (%[c_addr1]) \n\t"
          "vse16.v %[c02], (%[c_addr2]) \n\t"
          "vse16.v %[c03], (%[c_addr3]) \n\t"
          : [c00] "=&vr"(c00), [c01] "=&vr"(c01), [c02] "=&vr"(c02),
            [c03] "=&vr"(c03)
          : [jj_vl_in] "r"(jj_vl), [c_addr0] "r"(c + (ii + 0) * rsc + jj),
            [c_addr1] "r"(c + (ii + 1) * rsc + jj),
            [c_addr2] "r"(c + (ii + 2) * rsc + jj),
            [c_addr3] "r"(c + (ii + 3) * rsc + jj), [beta] "f"(beta),
            [alpha] "f"(alpha), [acc0] "vr"(acc0), [acc1] "vr"(acc1),
            [acc2] "vr"(acc2), [acc3] "vr"(acc3)
          : "vtype", "vl", "memory");
    }
  }
  for (ii0 = ii; (ii0 + 1) <= m; ii0 = ii0 + 1) {
    for (jj = 0; jj < n; jj = jj + jj_vl) {
      jj_vl = __riscv_vsetvl_e16m4(n - jj);
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        __asm__ volatile(
            "\n\t"
            "flh %[a0], 0(%[a_addr]) \n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "vfmul.vf %[acc], %[b0], %[a0] \n\t"
            : [a0] "=&f"(a0), [b0] "=&vr"(b0), [acc] "=vr"(acc)
            : [a_addr] "r"(a + ii0 * rsa + kk_peel), [jj_vl_in] "r"(jj_vl),
              [b_addr] "r"(b + kk_peel * rsb + jj)
            : "vtype", "vl", "memory");
      }
      for (kk = kk_peel; (kk + 1) <= k; kk = kk + 1) {
        __asm__ volatile(
            "\n\t"
            "flh %[a0], 0(%[a_addr]) \n\t"
            "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
            "vle16.v %[b0], (%[b_addr]) \n\t"
            "vfmacc.vf %[acc], %[a0], %[b0] \n\t"
            : [a0] "=&f"(a0), [b0] "=&vr"(b0), [acc] "+vr"(acc)
            : [a_addr] "r"(a + ii0 * rsa + kk), [jj_vl_in] "r"(jj_vl),
              [b_addr] "r"(b + kk * rsb + jj)
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli zero, %[jj_vl_in], e16, m4, ta, ma \n\t"
          "vle16.v %[c0], (%[c_addr]) \n\t"
          "vfmul.vf %[c0], %[c0], %[beta] \n\t"
          "vfmacc.vf %[c0], %[alpha], %[acc] \n\t"
          "vse16.v %[c0], (%[c_addr]) \n\t"
          : [c0] "=&vr"(c0)
          : [jj_vl_in] "r"(jj_vl), [c_addr] "r"(c + ii0 * rsc + jj),
            [beta] "f"(beta), [alpha] "f"(alpha), [acc] "vr"(acc)
          : "vtype", "vl", "memory");
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
