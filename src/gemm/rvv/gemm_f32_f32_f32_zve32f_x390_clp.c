// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

/**
 * @brief RVV float32 matrix-matrix multiplication (SGEMM) for row-major
 * matrices, tuned for X390 and core local port allocated matrices.
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
 * Uses a 4 x LMUL=4 x 2 register tile. Vectorized across the N dimension.
 *
 * @note
 * Works best when `m >= 4` and `n >= __riscv_vsetvlmax_e32m4()`, and `a`
 * resides in core local port memory.
 */
SKL_FUNC_PRIVATE void skl_gemm_4xm4x2_f32_f32_f32_zve32f_x390_clp(
    size_t m, size_t n, size_t k, float alpha, const float *a, size_t rsa,
    const float *b, size_t rsb, float beta, float *c, size_t rsc) {
  size_t jj_vl;
  size_t ii;
  size_t jj;
  size_t kk_peel;
  size_t kk_start;
  size_t kk;
  size_t kk_drain;
  size_t kk0;
  size_t ii0;
  float alpha0;
  float beta0;
  float a00_0;
  float a01_0;
  float a02;
  float a03;
  vfloat32m4_t b0;
  vfloat32m4_t acc0;
  vfloat32m4_t acc1;
  vfloat32m4_t acc2;
  vfloat32m4_t acc3;
  float a000;
  float a001;
  float a002;
  float a003;
  float a010;
  float a011;
  float a012;
  float a013;
  vfloat32m4_t b00;
  vfloat32m4_t b01;
  vfloat32m4_t c00;
  vfloat32m4_t c01;
  vfloat32m4_t c02;
  vfloat32m4_t c03;
  float a0;
  vfloat32m4_t acc;
  float a00_1;
  float a01_1;
  vfloat32m4_t c0;
  alpha0 = alpha;
  beta0 = beta;
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
      for (kk_peel = 0; ((kk_peel + 1) <= k) && (kk_peel < (0 + (1 * 1)));
           kk_peel = kk_peel + 1) {
        __asm__ volatile(
            "\n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load]) \n\t"
            "flw %[a00_0], 0(%[a_load]) \n\t"
            "vfmul.vf %[acc0], %[b0], %[a00_0] \n\t"
            "flw %[a01_0], 0(%[a_load_0]) \n\t"
            "vfmul.vf %[acc1], %[b0], %[a01_0] \n\t"
            "flw %[a02], 0(%[a_load_1]) \n\t"
            "vfmul.vf %[acc2], %[b0], %[a02] \n\t"
            "flw %[a03], 0(%[a_load_2]) \n\t"
            "vfmul.vf %[acc3], %[b0], %[a03] \n\t"
            : [a00_0] "=&f"(a00_0), [a01_0] "=&f"(a01_0), [a02] "=&f"(a02),
              [a03] "=&f"(a03), [jj_vl_out] "=&r"(jj_vl), [b0] "=&vr"(b0),
              [acc0] "=&vr"(acc0), [acc1] "=&vr"(acc1), [acc2] "=&vr"(acc2),
              [acc3] "=vr"(acc3)
            : [a_load] "r"(a + (((ii + 0) * rsa) + ((kk_peel + 0) * 1))),
              [a_load_0] "r"(a + (((ii + 1) * rsa) + ((kk_peel + 0) * 1))),
              [a_load_1] "r"(a + (((ii + 2) * rsa) + ((kk_peel + 0) * 1))),
              [a_load_2] "r"(a + (((ii + 3) * rsa) + ((kk_peel + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load] "r"(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk_start = kk_peel;
           ((kk_start + 2) <= k) && (kk_start < (kk_peel + (2UL * 1UL)));
           kk_start = kk_start + 2) {
        __asm__ volatile(
            "\n\t"
            "flw %[a000], 0(%[a_load_3]) \n\t"
            "flw %[a001], 0(%[a_load_4]) \n\t"
            "flw %[a002], 0(%[a_load_5]) \n\t"
            "flw %[a003], 0(%[a_load_6]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b00], (%[b_load_0]) \n\t"
            "flw %[a010], 4(%[a_load_3]) \n\t"
            "flw %[a011], 4(%[a_load_4]) \n\t"
            "flw %[a012], 4(%[a_load_5]) \n\t"
            "flw %[a013], 4(%[a_load_6]) \n\t"
            "vle32.v %[b01], (%[b_load_1]) \n\t"
            : [a000] "=&f"(a000), [a001] "=&f"(a001), [a002] "=&f"(a002),
              [a003] "=&f"(a003), [a010] "=&f"(a010), [a011] "=&f"(a011),
              [a012] "=&f"(a012), [a013] "=&f"(a013), [jj_vl_out] "=&r"(jj_vl),
              [b00] "=&vr"(b00), [b01] "=vr"(b01)
            : [a_load_3] "r"(a + (((ii + 0) * rsa) + ((kk_start + 0) * 1))),
              [a_load_4] "r"(a + (((ii + 1) * rsa) + ((kk_start + 0) * 1))),
              [a_load_5] "r"(a + (((ii + 2) * rsa) + ((kk_start + 0) * 1))),
              [a_load_6] "r"(a + (((ii + 3) * rsa) + ((kk_start + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_0] "r"(b + (((kk_start + 0) * rsb) + ((jj + 0) * 1))),
              [b_load_1] "r"(b + (((kk_start + 1) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk = kk_peel; (kk + (2UL * 1UL)) < k; kk = kk + 2) {
        __asm__ volatile(
            "\n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vfmacc.vf %[acc0], %[a000], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a001], %[b00] \n\t"
            "flw %[a000], 0(%[a_load_11]) \n\t"
            "vfmacc.vf %[acc2], %[a002], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a003], %[b00] \n\t"
            "flw %[a001], 0(%[a_load_12]) \n\t"
            "vfmacc.vf %[acc0], %[a010], %[b01] \n\t"
            "flw %[a002], 0(%[a_load_13]) \n\t"
            "vfmacc.vf %[acc1], %[a011], %[b01] \n\t"
            "vle32.v %[b00], (%[b_load_2]) \n\t"
            "vfmacc.vf %[acc2], %[a012], %[b01] \n\t"
            "vfmacc.vf %[acc3], %[a013], %[b01] \n\t"
            "flw %[a003], 0(%[a_load_14]) \n\t"
            "vle32.v %[b01], (%[b_load_3]) \n\t"
            "flw %[a010], 4(%[a_load_11]) \n\t"
            "flw %[a011], 4(%[a_load_12]) \n\t"
            "flw %[a012], 4(%[a_load_13]) \n\t"
            "flw %[a013], 4(%[a_load_14]) \n\t"
            : [jj_vl_out] "=&r"(jj_vl), [acc0] "+&vr"(acc0), [a000] "+&f"(a000),
              [b00] "+&vr"(b00), [acc1] "+&vr"(acc1), [a001] "+&f"(a001),
              [acc2] "+&vr"(acc2), [a002] "+&f"(a002), [acc3] "+&vr"(acc3),
              [a003] "+&f"(a003), [a010] "+&f"(a010), [b01] "+vr"(b01),
              [a011] "+&f"(a011), [a012] "+&f"(a012), [a013] "+&f"(a013)
            : [jj_vl_in] "r"(n - jj),
              [a_load_11] "r"(a + (((ii + 0) * rsa) + ((kk + 2) * 1))),
              [a_load_12] "r"(a + (((ii + 1) * rsa) + ((kk + 2) * 1))),
              [a_load_13] "r"(a + (((ii + 2) * rsa) + ((kk + 2) * 1))),
              [a_load_14] "r"(a + (((ii + 3) * rsa) + ((kk + 2) * 1))),
              [b_load_2] "r"(b + (((kk + 2) * rsb) + ((jj + 0) * 1))),
              [b_load_3] "r"(b + (((kk + 3) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk_drain = kk;
           ((kk_drain + 2) <= k) && (kk_drain < (kk + (2UL * 1UL)));
           kk_drain = kk_drain + 2) {
        __asm__ volatile(
            "\n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vfmacc.vf %[acc0], %[a000], %[b00] \n\t"
            "vfmacc.vf %[acc1], %[a001], %[b00] \n\t"
            "vfmacc.vf %[acc2], %[a002], %[b00] \n\t"
            "vfmacc.vf %[acc3], %[a003], %[b00] \n\t"
            "vfmacc.vf %[acc0], %[a010], %[b01] \n\t"
            "vfmacc.vf %[acc1], %[a011], %[b01] \n\t"
            "vfmacc.vf %[acc2], %[a012], %[b01] \n\t"
            "vfmacc.vf %[acc3], %[a013], %[b01] \n\t"
            : [jj_vl_out] "=&r"(jj_vl), [acc0] "+&vr"(acc0),
              [acc1] "+&vr"(acc1), [acc2] "+&vr"(acc2), [acc3] "+&vr"(acc3)
            : [jj_vl_in] "r"(n - jj), [a000] "f"(a000), [b00] "vr"(b00),
              [a001] "f"(a001), [a002] "f"(a002), [a003] "f"(a003),
              [a010] "f"(a010), [b01] "vr"(b01), [a011] "f"(a011),
              [a012] "f"(a012), [a013] "f"(a013)
            : "vtype", "vl");
      }
      for (kk0 = kk_drain; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a00_0], 0(%[a_load_19]) \n\t"
            "flw %[a01_0], 0(%[a_load_20]) \n\t"
            "flw %[a02], 0(%[a_load_21]) \n\t"
            "flw %[a03], 0(%[a_load_22]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load_4]) \n\t"
            "vfmacc.vf %[acc0], %[a00_0], %[b0] \n\t"
            "vfmacc.vf %[acc1], %[a01_0], %[b0] \n\t"
            "vfmacc.vf %[acc2], %[a02], %[b0] \n\t"
            "vfmacc.vf %[acc3], %[a03], %[b0] \n\t"
            : [a00_0] "=&f"(a00_0), [a01_0] "=&f"(a01_0), [a02] "=&f"(a02),
              [a03] "=&f"(a03), [jj_vl_out] "=&r"(jj_vl), [b0] "=&vr"(b0),
              [acc0] "+&vr"(acc0), [acc1] "+&vr"(acc1), [acc2] "+&vr"(acc2),
              [acc3] "+vr"(acc3)
            : [a_load_19] "r"(a + (((ii + 0) * rsa) + ((kk0 + 0) * 1))),
              [a_load_20] "r"(a + (((ii + 1) * rsa) + ((kk0 + 0) * 1))),
              [a_load_21] "r"(a + (((ii + 2) * rsa) + ((kk0 + 0) * 1))),
              [a_load_22] "r"(a + (((ii + 3) * rsa) + ((kk0 + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_4] "r"(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      __asm__ volatile(
          "\n\t"
          "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
          "vle32.v %[c00], (%[c_load]) \n\t"
          "vle32.v %[c01], (%[c_load_0]) \n\t"
          "vle32.v %[c02], (%[c_load_1]) \n\t"
          "vle32.v %[c03], (%[c_load_2]) \n\t"
          "vfmul.vf %[c00], %[c00], %[beta0] \n\t"
          "vfmul.vf %[c01], %[c01], %[beta0] \n\t"
          "vfmul.vf %[c02], %[c02], %[beta0] \n\t"
          "vfmul.vf %[c03], %[c03], %[beta0] \n\t"
          "vfmacc.vf %[c00], %[alpha0], %[acc0] \n\t"
          "vfmacc.vf %[c01], %[alpha0], %[acc1] \n\t"
          "vfmacc.vf %[c02], %[alpha0], %[acc2] \n\t"
          "vfmacc.vf %[c03], %[alpha0], %[acc3] \n\t"
          "vse32.v %[c00], (%[c_store]) \n\t"
          "vse32.v %[c01], (%[c_store_0]) \n\t"
          "vse32.v %[c02], (%[c_store_1]) \n\t"
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
            "flw %[a0], 0(%[a_load_23]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load_5]) \n\t"
            "vfmul.vf %[acc], %[b0], %[a0] \n\t"
            : [a0] "=&f"(a0), [jj_vl_out] "=&r"(jj_vl), [b0] "=&vr"(b0),
              [acc] "=vr"(acc)
            : [a_load_23] "r"(a + (((ii0 + 0) * rsa) + ((kk_peel + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_5] "r"(b + (((kk_peel + 0) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk = kk_peel; (kk + 2) <= k; kk = kk + 2) {
        __asm__ volatile(
            "\n\t"
            "flw %[a00_1], 0(%[a_load_24]) \n\t"
            "flw %[a01_1], 4(%[a_load_24]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b00], (%[b_load_6]) \n\t"
            "vfmacc.vf %[acc], %[a00_1], %[b00] \n\t"
            "vle32.v %[b01], (%[b_load_7]) \n\t"
            "vfmacc.vf %[acc], %[a01_1], %[b01] \n\t"
            : [a00_1] "=&f"(a00_1), [a01_1] "=&f"(a01_1),
              [jj_vl_out] "=&r"(jj_vl), [b00] "=&vr"(b00), [b01] "=&vr"(b01),
              [acc] "+&vr"(acc)
            : [a_load_24] "r"(a + (((ii0 + 0) * rsa) + ((kk + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_6] "r"(b + (((kk + 0) * rsb) + ((jj + 0) * 1))),
              [b_load_7] "r"(b + (((kk + 1) * rsb) + ((jj + 0) * 1)))
            : "vtype", "vl", "memory");
      }
      for (kk0 = kk; (kk0 + 1) <= k; kk0 = kk0 + 1) {
        __asm__ volatile(
            "\n\t"
            "flw %[a0], 0(%[a_load_26]) \n\t"
            "vsetvli %[jj_vl_out], %[jj_vl_in], e32, m4, ta, ma \n\t"
            "vle32.v %[b0], (%[b_load_8]) \n\t"
            "vfmacc.vf %[acc], %[a0], %[b0] \n\t"
            : [a0] "=&f"(a0), [jj_vl_out] "=&r"(jj_vl), [b0] "=&vr"(b0),
              [acc] "+vr"(acc)
            : [a_load_26] "r"(a + (((ii0 + 0) * rsa) + ((kk0 + 0) * 1))),
              [jj_vl_in] "r"(n - jj),
              [b_load_8] "r"(b + (((kk0 + 0) * rsb) + ((jj + 0) * 1)))
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

SKL_FUNC void skl_gemm_f32_f32_f32_zve32f_x390_clp(size_t m, size_t n, size_t k,
                                                   float alpha, const float *a,
                                                   size_t rsa, const float *b,
                                                   size_t rsb, float beta,
                                                   float *c, size_t rsc) {
  skl_gemm_4xm4x2_f32_f32_f32_zve32f_x390_clp(m, n, k, alpha, a, rsa, b, rsb,
                                              beta, c, rsc);
}
