// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f32_zve32f(float *pDst, const float *pSrc,
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

  /* Vector accumulator.  It is likely spilled by the compiler early
     in the loop, so keep a full m8 vector to simplify. */
  vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, n);
  for (size_t i = 0; i < n; i += vl) {
    /* Calculate sum += exp(beta * (x - max)) */
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc + i, vl);
    const float c4 = 0x1.573c7ep-5f;
    vfloat32m8_t vp = __riscv_vfmv_v_f_f32m8(c4, vl);

    vx = __riscv_vfsub_vf_f32m8(vx, max, vl);
    if (beta != 1.0f)
      vx = __riscv_vfmul_vf_f32m8(vx, beta, vl);

    /* Clamp inputs: */
    const float xmin = -0x1.5ebb86p6f;
    vx = __riscv_vfmax_vf_f32m8(vx, xmin, vl);
    /* Range reduction: */
    const float rln2 = 0x1.715476p0f; // 1/log(2)
    const float ln2 = 0x1.62e43p-1f;  // log(2)
    vfloat32m8_t vv = __riscv_vfmul_vf_f32m8(vx, rln2, vl);
    vint16m4_t vq = __riscv_vfncvt_x_f_w_i16m4(vv, vl);
    vfloat32m8_t vz = __riscv_vfwcvt_f_x_v_f32m8(vq, vl);
    vfloat32m8_t vs = __riscv_vfnmsac_vf_f32m8(vx, ln2, vz, vl);
    /* Approximate exp(s): */
    const float c1 = 0x1.fffff0p-1f;
    const float c2 = 0x1.fffdbep-2f;
    const float c3 = 0x1.555cf6p-3f;
    const float c5 = 0x1.0eb7d0p-7f;
    vp = __riscv_vfmacc_vf_f32m8(vp, c5, vs, vl);
    vfloat32m8_t vc = __riscv_vfmv_v_f_f32m8(c3, vl);
    vp = __riscv_vfmadd_vv_f32m8(vp, vs, vc, vl);
    vc = __riscv_vfmv_v_f_f32m8(c2, vl);
    vp = __riscv_vfmadd_vv_f32m8(vp, vs, vc, vl);
    vc = __riscv_vfmv_v_f_f32m8(c1, vl);
    vp = __riscv_vfmadd_vv_f32m8(vp, vs, vc, vl);
    /* Reconstruction */
    vq = __riscv_vsll_vx_i16m4(vq, 8, vl);
    vfloat32m8_t vt = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    vint32m8_t vQ = __riscv_vreinterpret_v_f32m8_i32m8(vt);
    vQ = __riscv_vwmaccus_vx_i32m8(vQ, 0x8000, vq, vl);
    vfloat32m8_t vf = __riscv_vreinterpret_v_i32m8_f32m8(vQ);
    vfloat32m8_t vy = __riscv_vfmul_vv_f32m8(vs, vf, vl);
    vy = __riscv_vfmadd_vv_f32m8(vy, vp, vf, vl); /* exp(vx) */

    /* Accumulate and Store */
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

SKL_FUNC void skl_softmax_2d_f32_zve32f(float *s, const size_t rss,
                                        const float *a, const size_t rsa,
                                        const float beta, const size_t m,
                                        const size_t n) {
  for (size_t i = 0; i < m; ++i) {
    skl_softmax_f32_zve32f(s + i * rss, a + i * rsa, beta, n);
  }
}
