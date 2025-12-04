// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_bf16_xsfvfbfa(__bf16 *pDst, const __bf16 *pSrc,
                                        const __bf16 beta, const size_t n) {
  size_t vl = __riscv_vsetvl_e32m4(n);
  vbfloat16m4_t vmax = __riscv_vle16_v_bf16m4(pSrc, vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(pSrc + i, vl);
    vmax = __riscv_vfmax_vv_bf16m4_tu(vmax, vmax, vx, vl);
  }
  /* Xsfvfbfa does not have bf16 reductions, so reduce in f32 */
  vfloat32m8_t vMAX = __riscv_vfwcvtbf16_f_f_v_f32m8(vmax, vl);
  vfloat32m1_t sMAX = __riscv_vlmul_trunc_v_f32m8_f32m1(vMAX);
  sMAX = __riscv_vfredmax_vs_f32m8_f32m1(vMAX, sMAX, n);
  __bf16 max = (__bf16)__riscv_vfmv_f_s_f32m1_f32(sMAX);

  vbfloat16m8_t vsum = __riscv_vfmv_v_f_bf16m8(0, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_bf16m8(vx, max, vl);
    vx = __riscv_vfmul_vf_bf16m8(vx, beta, vl);
    /* 0. Clamp */
    const __bf16 LB = (__bf16)-0x1.6p6f; /* round_down(-126.5 * log(2)) */
    vx = __riscv_vfmax_vf_bf16m8(vx, LB, vl);
    /* 1. Reduce */
    vbfloat16m8_t v = __riscv_vfmul_vf_bf16m8(vx, (__bf16)0x1.72p0, vl);
    vint8m4_t q = __riscv_vfncvt_x_f_w_bf16m8(v, vl);
    vbfloat16m8_t z = __riscv_vfwcvt_bf_x_v_bf16m8(q, vl);
    vbfloat16m8_t s = __riscv_vfnmsac_vf_bf16m8(vx, (__bf16)0x1.6p-1, z, vl);
    s = __riscv_vfnmsac_vf_bf16m8(s, (__bf16)0x1.72p-8, z, vl);
    /* 2. Approximate exp(s) */
    vbfloat16m8_t c = __riscv_vfmv_v_f_bf16m8((__bf16)0x1.06p-1, vl);
    vbfloat16m8_t u = __riscv_vfmacc_vf_bf16m8(c, (__bf16)0x1.56p-3, s, vl);
    c = __riscv_vfmv_v_f_bf16m8((__bf16)0x1p0, vl);
    u = __riscv_vfmadd_vv_bf16m8(u, s, c, vl);
    u = __riscv_vfmadd_vv_bf16m8(u, s, c, vl);
    /* 3. Assemble */
    vint16m8_t t = __riscv_vreinterpret_v_bf16m8_i16m8(u);
    t = __riscv_vwmaccus_vx_i16m8(t, 1 << 7, q, vl);
    u = __riscv_vreinterpret_v_i16m8_bf16m8(t);
    /* 4. Accumulate & Store */
    vsum = __riscv_vfadd_vv_bf16m8_tu(vsum, vsum, u, vl);
    __riscv_vse16_v_bf16m8(pDst + i, u, vl);
  }
  size_t vlmax4 = __riscv_vsetvlmax_e16m4();
  size_t vln = __riscv_vsetvl_e16m8(n);
  size_t vl0 = __riscv_vsetvl_e16m4(n >= vlmax4 ? vlmax4 : n);
  size_t vl1 = vln - vl0;
  vbfloat16m4_t vsum0 = __riscv_vget_v_bf16m8_bf16m4(vsum, 0);
  vbfloat16m4_t vsum1 = __riscv_vget_v_bf16m8_bf16m4(vsum, 1);
  vsum0 = __riscv_vfadd_vv_bf16m4_tu(vsum0, vsum0, vsum1, vl1);

  vfloat32m8_t vSUM = __riscv_vfwcvtbf16_f_f_v_f32m8(vsum0, vl0);
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
