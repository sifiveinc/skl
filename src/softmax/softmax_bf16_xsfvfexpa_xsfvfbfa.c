// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfexpa and Xsfvfbfa extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_bf16_xsfvfexpa_xsfvfbfa(__bf16 *pDst,
                                                  const __bf16 *pSrc,
                                                  const __bf16 beta,
                                                  const size_t n) {
  size_t vl = __riscv_vsetvl_e16m4(n);
  vbfloat16m4_t vmax = __riscv_vle16_v_bf16m4(pSrc, vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(pSrc + i, vl);
    vmax = __riscv_vfmax_vv_bf16m4_tu(vmax, vmax, vx, vl);
  }
  vfloat32m8_t vMAX = __riscv_vfwcvt_f_f_v_bf16m4_f32m8(vmax, vl);
  vfloat32m1_t sMAX = __riscv_vlmul_trunc_v_f32m8_f32m1(vMAX);
  sMAX = __riscv_vfredmax_vs_f32m8_f32m1(vMAX, sMAX, n);
  __bf16 max = (__bf16)__riscv_vfmv_f_s_f32m1_f32(sMAX);

  size_t vlmax4 = __riscv_vsetvlmax_e16m4();
  vbfloat16m8_t vsum = __riscv_vfmv_v_f_bf16m8(0, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(pSrc + i, vl);
    /* 0. Clamp and Scale */
    const float LB = -0x1.6p6f; /* round_down(-126.5 * log(2)) */
    vx = __riscv_vfsub_vf_bf16m8(vx, max, vl);
    vx = __riscv_vfmax_vf_bf16m8(vx, (__bf16)(LB / beta), vl);

    vbfloat16m4_t vx0 = __riscv_vget_v_bf16m8_bf16m4(vx, 0);
    vbfloat16m4_t vx1 = __riscv_vget_v_bf16m8_bf16m4(vx, 1);
    size_t vl0 = __riscv_vsetvl_e16m4(vl >= vlmax4 ? vlmax4 : vl);
    vfloat32m8_t A0 = __riscv_vfwmul_vf_bf16m4_f32m8(vx0, beta, vl0);
    vfloat32m8_t A1 = __riscv_vfwmul_vf_bf16m4_f32m8(vx1, beta, vl0);
    /* 1. Reduce & Evaluate */
    const float R = 0x1.715476p0f;
    vfloat32m8_t Q = __riscv_vfmv_v_f_f32m8(0x1.003f80p17f, vl0);
    vfloat32m8_t V0 = __riscv_vfmadd_vf_f32m8(A0, R, Q, vl0);
    vfloat32m8_t V1 = __riscv_vfmadd_vf_f32m8(A1, R, Q, vl0);
    vfloat32m8_t F0 = __riscv_sf_vfexpa_v_f32m8(V0, vl0);
    vfloat32m8_t F1 = __riscv_sf_vfexpa_v_f32m8(V1, vl0);
    /* 2. Narrow */
    vbfloat16m4_t vy0 = __riscv_vfncvt_f_f_w_bf16m4(F0, vl0);
    vbfloat16m4_t vy1 = __riscv_vfncvt_f_f_w_bf16m4(F1, vl0);
    vbfloat16m8_t vy = __riscv_vundefined_bf16m8();
    vy = __riscv_vset_v_bf16m4_bf16m8(vy, 0, vy0);
    vy = __riscv_vset_v_bf16m4_bf16m8(vy, 1, vy1);
    /* 3. Accumulate & Store */
    vsum = __riscv_vfadd_vv_bf16m8_tu(vsum, vsum, vy, vl);
    __riscv_vse16_v_bf16m8(pDst + i, vy, vl);
  }
  size_t vln = __riscv_vsetvl_e16m8(n);
  size_t vl0 = __riscv_vsetvl_e16m4(n >= vlmax4 ? vlmax4 : n);
  size_t vl1 = vln - vl0;
  vbfloat16m4_t vsum0 = __riscv_vget_v_bf16m8_bf16m4(vsum, 0);
  vbfloat16m4_t vsum1 = __riscv_vget_v_bf16m8_bf16m4(vsum, 1);
  vsum0 = __riscv_vfadd_vv_bf16m4_tu(vsum0, vsum0, vsum1, vl1);

  vfloat32m8_t vSUM = __riscv_vfwcvt_f_f_v_bf16m4_f32m8(vsum0, vl0);
  vfloat32m1_t ssum = __riscv_vfmv_s_f_f32m1(0.0f, vl0);
  ssum = __riscv_vfredusum_vs_f32m8_f32m1(vSUM, ssum, vl0);
  float sum = __riscv_vfmv_f_s_f32m1_f32(ssum);
  __bf16 recip_sum = (__bf16)(1.0f / sum);

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(pDst + i, vl);
    vx = __riscv_vfmul_vf_bf16m8(vx, recip_sum, vl);
    __riscv_vse16_v_bf16m8(pDst + i, vx, vl);
  }
}
