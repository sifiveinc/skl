// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfbfmin)
#error This file requires the Zvfbfmin extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_bf16_zvfbfmin(__bf16 *pDst, const __bf16 *pSrc,
                                        const __bf16 beta, const size_t n) {
  size_t vl = __riscv_vsetvl_e16m4(n);
  vfloat32m8_t vmax =
      __riscv_vfwcvtbf16_f_f_v_f32m8(__riscv_vle16_v_bf16m4(pSrc, vl), vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(pSrc + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, va, vl);
  }
  vfloat32m1_t vfirst = __riscv_vlmul_trunc_v_f32m8_f32m1(vmax);
  vfirst = __riscv_vfredmax_vs_f32m8_f32m1(vmax, vfirst, n);
  float max = __riscv_vfmv_f_s_f32m1_f32(vfirst);

  vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(pSrc + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    va = __riscv_vfsub_vf_f32m8(va, max, vl);
    if (beta != (__bf16)1.0)
      va = __riscv_vfmul_vf_f32m8(va, (float)beta, vl);
    /* 0. Clamp */
    vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(va, va, vl);
    const float LB = -0x1.6p6f; /* round_down(-126.5 * log(2)) */
    va = __riscv_vfmax_vf_f32m8_mu(nn, va, va, LB, vl);
    /* 1. Reduce */
    const float R = 0x1.715476p+0f; /* 1/log(2) */
    const float C = 0x1.62e430p-1f; /* log(2) */
    vfloat32m8_t v = __riscv_vfmul_vf_f32m8(va, R, vl);
    vint16m4_t q = __riscv_vmv_v_x_i16m4(0, vl);
    q = __riscv_vfncvt_x_f_w_i16m4_mu(nn, q, v, vl);
    vfloat32m8_t z = __riscv_vfwcvt_f_x_v_f32m8(q, vl);
    vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(va, C, z, vl);
    /* 2. Approximate exp(s) */
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(0x1.03cdeep0f, vl);
    vfloat32m8_t u = __riscv_vfmacc_vf_f32m8(c, 0x1.fc2b5ap-2f, s, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.001d0ap0f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c, vl);
    /* 3. Assemble */
    q = __riscv_vsll_vx_i16m4(q, 8, vl);
    vint32m8_t j = __riscv_vreinterpret_v_f32m8_i32m8(u);
    j = __riscv_vwmaccus_vx_i32m8(j, 0x8000, q, vl);
    u = __riscv_vreinterpret_v_i32m8_f32m8(j);
    /* 4. Narrow & Store */
    vbfloat16m4_t vy = __riscv_vfncvtbf16_f_f_w_bf16m4(u, vl);
    __riscv_vse16_v_bf16m4(pDst + i, vy, vl);
    /* 5. Accumulate */
    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, u, vl);
  }
  vfloat32m1_t ssum = __riscv_vfmv_s_f_f32m1(0.0f, n);
  ssum = __riscv_vfredusum_vs_f32m8_f32m1(vsum, ssum, n);
  float sum = __riscv_vfmv_f_s_f32m1_f32(ssum);
  float recip_sum = 1.0f / sum;

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(pDst + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    va = __riscv_vfmul_vf_f32m8(va, recip_sum, vl);
    vx = __riscv_vfncvtbf16_f_f_w_bf16m4(va, vl);
    __riscv_vse16_v_bf16m4(pDst + i, vx, vl);
  }
}
