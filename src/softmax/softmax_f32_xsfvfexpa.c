// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f32_xsfvfexpa(float *pDst, const float *pSrc,
                                        const float beta, const size_t n) {
  size_t vl = __riscv_vsetvl_e32m8(n);
  vfloat32m8_t vmax = __riscv_vle32_v_f32m8(pSrc, vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc + i, vl);
    vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, vx, vl);
  }
  vfloat32m1_t vfirst = __riscv_vlmul_trunc_v_f32m8_f32m1(vmax);
  vfirst = __riscv_vfredmax_vs_f32m8_f32m1(vmax, vfirst, n);
  float max = __riscv_vfmv_f_s_f32m1_f32(vfirst);

  vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, n);
  for (size_t i = 0; i < n; i += vl) {
    /* Calculate sum += exp(beta * (x - max)) */

    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_f32m8(vx, max, vl);
    if (beta != 1.0f)
      vx = __riscv_vfmul_vf_f32m8(vx, beta, vl);

    /* Clamp inputs: */
    const float xmin = -0x1.6018dep6f;
    vx = __riscv_vfmax_vf_f32m8(vx, xmin, vl);
    /* Range reduction: */
    const float R = 0x1.715476p0f;  // 1/log(2)
    const float Q = 0x1.003f80p17f; // 2^(24-6-1) + B
    const float C = 0x1.62e43p-1f;  // log(2)
    vfloat32m8_t vQ = __riscv_vfmv_v_f_f32m8(Q, vl);
    vfloat32m8_t vv = __riscv_vfmacc_vf_f32m8(vQ, R, vx, vl);
    vfloat32m8_t vz = __riscv_vfsub_vf_f32m8(vv, Q, vl);
    vfloat32m8_t vs = __riscv_vfnmsac_vf_f32m8(vx, C, vz, vl);
    /* Approximate exp(s): */
    const float c2 = 0x1.ff2a6p-2f;
    vfloat32m8_t vp = __riscv_vfmul_vf_f32m8(vs, c2, vl);
    vp = __riscv_vfmadd_vv_f32m8(vp, vs, vs, vl);
    /* Reconstruction */
    vfloat32m8_t vf = __riscv_sf_vfexpa_v_f32m8(vv, vl);
    vfloat32m8_t vy = __riscv_vfmadd_vv_f32m8(vp, vf, vf, vl);

    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, vy, vl);
    __riscv_vse32_v_f32m8(pDst + i, vy, vl);
  }
  vfloat32m1_t ssum = __riscv_vfmv_s_f_f32m1(0.0f, n);
  ssum = __riscv_vfredusum_vs_f32m8_f32m1(vsum, ssum, n);
  float sum = __riscv_vfmv_f_s_f32m1_f32(ssum);
  float recip_sum = 1.0f / sum;

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pDst + i, vl);
    vx = __riscv_vfmul_vf_f32m8(vx, recip_sum, vl);
    __riscv_vse32_v_f32m8(pDst + i, vx, vl);
  }
}

SKL_FUNC void skl_softmax_2d_f32_xsfvfexpa(float *s, const size_t rss,
                                           const float *a, const size_t rsa,
                                           const float beta, const size_t m,
                                           const size_t n) {
  for (size_t i = 0; i < m; ++i) {
    skl_softmax_f32_xsfvfexpa(s + i * rss, a + i * rsa, beta, n);
  }
}
