// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

/* Implementation is unrolled x2 for additional vector ILP */
SKL_FUNC void skl_softmax_f16_zvfh(_Float16 *pDst, const _Float16 *pSrc,
                                   const _Float16 beta, const size_t n) {
  size_t vl = __riscv_vsetvl_e16m8(n);
  vfloat16m8_t vmax = __riscv_vle16_v_f16m8(pSrc, vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(pSrc + i, vl);
    vmax = __riscv_vfmax_vv_f16m8_tu(vmax, vmax, vx, vl);
  }
  vfloat16m1_t vfirst = __riscv_vlmul_trunc_v_f16m8_f16m1(vmax);
  vfirst = __riscv_vfredmax_vs_f16m8_f16m1(vmax, vfirst, n);
  _Float16 max = __riscv_vfmv_f_s_f16m1_f16(vfirst);

  const size_t vlmax4 = __riscv_vsetvlmax_e16m4();
  const size_t vlmax8 = __riscv_vsetvlmax_e16m8();
  vfloat16m4_t vsum = __riscv_vfmv_v_f_f16m4(0, n >= vlmax4 ? vlmax4 : n);
  for (size_t i = 0; i < n; i += vl) {
    /* Calculate sum += exp(beta * (x - max)) */

    size_t vl0;
    size_t vl1;
    size_t avl = n - i;
    vl = __riscv_vsetvl_e16m8(avl >= vlmax8 ? vlmax8 : avl);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(pSrc + i, vl);
    const _Float16 c2 = 0x1.024p-1f16;
    vfloat16m8_t vp = __riscv_vfmv_v_f_f16m8(c2, vl);

    vl0 = __riscv_vsetvl_e16m4(vl >= vlmax4 ? vlmax4 : vl);
    vl1 = vl - vl0;
    vfloat16m4_t vx0 = __riscv_vget_v_f16m8_f16m4(vx, 0);
    vfloat16m4_t vx1 = __riscv_vget_v_f16m8_f16m4(vx, 1);

    /* We have vl0 >= vl1, so use vl0, until sum accumulation. */
    vx0 = __riscv_vfsub_vf_f16m4(vx0, max, vl0);
    vx1 = __riscv_vfsub_vf_f16m4(vx1, max, vl0);
    if (beta != 1.0f16) {
      vx0 = __riscv_vfmul_vf_f16m4(vx0, beta, vl0);
      vx1 = __riscv_vfmul_vf_f16m4(vx1, beta, vl0);
    }

    /* Clamp */
    const _Float16 xmin = -0x1.4c8p3f16;
    vx0 = __riscv_vfmax_vf_f16m4(vx0, xmin, vl0);
    vx1 = __riscv_vfmax_vf_f16m4(vx1, xmin, vl0);
    /* Reduce */
    const _Float16 R = 0x1.714p0f16;  // 1/log(2)
    const _Float16 C1 = 0x1.64p-1f16; // log(2)
    const _Float16 C2 = -0x1.1bcp-9f16;
    vfloat16m4_t vv0 = __riscv_vfmul_vf_f16m4(vx0, R, vl0);
    vfloat16m4_t vv1 = __riscv_vfmul_vf_f16m4(vx1, R, vl0);
    vint8m2_t vq0 = __riscv_vfncvt_x_f_w_i8m2(vv0, vl0);
    vint8m2_t vq1 = __riscv_vfncvt_x_f_w_i8m2(vv1, vl0);
    vfloat16m4_t vz0 = __riscv_vfwcvt_f_x_v_f16m4(vq0, vl0);
    vfloat16m4_t vz1 = __riscv_vfwcvt_f_x_v_f16m4(vq1, vl0);
    vfloat16m4_t vs0 = __riscv_vfnmsac_vf_f16m4(vx0, C1, vz0, vl0);
    vfloat16m4_t vs1 = __riscv_vfnmsac_vf_f16m4(vx1, C1, vz1, vl0);
    vs0 = __riscv_vfnmsac_vf_f16m4(vs0, C2, vz0, vl0);
    vs1 = __riscv_vfnmsac_vf_f16m4(vs1, C2, vz1, vl0);
    /* Approximate exp(s) */
    const _Float16 c1 = 0x1p0f16;
    const _Float16 c3 = 0x1.574p-3f16;
    vfloat16m4_t vp0 = __riscv_vget_v_f16m8_f16m4(vp, 0);
    vfloat16m4_t vp1 = __riscv_vget_v_f16m8_f16m4(vp, 1);
    vp0 = __riscv_vfmacc_vf_f16m4(vp0, c3, vs0, vl0);
    vp1 = __riscv_vfmacc_vf_f16m4(vp1, c3, vs1, vl0);
    vfloat16m4_t vc = __riscv_vfmv_v_f_f16m4(c1, vl0);
    vp0 = __riscv_vfmadd_vv_f16m4(vp0, vs0, vc, vl0);
    vp1 = __riscv_vfmadd_vv_f16m4(vp1, vs1, vc, vl0);
    vp0 = __riscv_vfmadd_vv_f16m4(vp0, vs0, vc, vl0);
    vp1 = __riscv_vfmadd_vv_f16m4(vp1, vs1, vc, vl0);
    /* Assemble */
    vq0 = __riscv_vsll_vx_i8m2(vq0, 3, vl0);
    vq1 = __riscv_vsll_vx_i8m2(vq1, 3, vl0);
    vint16m4_t vQ0 = __riscv_vreinterpret_v_f16m4_i16m4(vp0);
    vint16m4_t vQ1 = __riscv_vreinterpret_v_f16m4_i16m4(vp1);
    vQ0 = __riscv_vwmaccus_vx_i16m4(vQ0, 0x80, vq0, vl0);
    vQ1 = __riscv_vwmaccus_vx_i16m4(vQ1, 0x80, vq1, vl0);
    vfloat16m4_t vy0 = __riscv_vreinterpret_v_i16m4_f16m4(vQ0);
    vfloat16m4_t vy1 = __riscv_vreinterpret_v_i16m4_f16m4(vQ1);
    /* Accumulate */
    vsum = __riscv_vfadd_vv_f16m4_tu(vsum, vsum, vy0, vl0);
    vsum = __riscv_vfadd_vv_f16m4_tu(vsum, vsum, vy1, vl1);
    /* Store */
    vfloat16m8_t vy = __riscv_vundefined_f16m8();
    vy = __riscv_vset_v_f16m4_f16m8(vy, 0, vy0);
    vy = __riscv_vset_v_f16m4_f16m8(vy, 1, vy1);
    __riscv_vse16_v_f16m8(pDst + i, vy, vl);
  }
  vfloat16m1_t ssum = __riscv_vfmv_s_f_f16m1(0, n);
  ssum = __riscv_vfredusum_vs_f16m4_f16m1(vsum, ssum, n >= vlmax4 ? vlmax4 : n);
  _Float16 sum = __riscv_vfmv_f_s_f16m1_f16(ssum);
  _Float16 recip_sum = 1 / sum;

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(pDst + i, vl);
    vx = __riscv_vfmul_vf_f16m8(vx, recip_sum, vl);
    __riscv_vse16_v_f16m8(pDst + i, vx, vl);
  }
}
