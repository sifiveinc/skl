// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfbfmin)
#error This file requires the Xsfvfexpa and Zvfbfmin extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_bf16_xsfvfexpa_zvfbfmin(__bf16 *dst,
                                                  const __bf16 *src,
                                                  const __bf16 beta,
                                                  const size_t n) {
  size_t vl = __riscv_vsetvl_e16m4(n);
  vfloat32m8_t vmax =
      __riscv_vfwcvtbf16_f_f_v_f32m8(__riscv_vle16_v_bf16m4(src, vl), vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(src + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, va, vl);
  }
  vfloat32m1_t vfirst = __riscv_vlmul_trunc_v_f32m8_f32m1(vmax);
  vfirst = __riscv_vfredmax_vs_f32m8_f32m1(vmax, vfirst, n);
  float max = __riscv_vfmv_f_s_f32m1_f32(vfirst);

  vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(src + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    va = __riscv_vfsub_vf_f32m8(va, max, vl);
    if (beta != (__bf16)1.0)
      va = __riscv_vfmul_vf_f32m8(va, (float)beta, vl);
    /* 0. Clamp */
    const float LB = -0x1.6p6f; /* round_down(-126.5 * log(2)) */
    va = __riscv_vfmax_vf_f32m8(va, LB, vl);
    /* 1. Reduce & Evaluate */
    const float R = 0x1.715476p0f;
    vfloat32m8_t Q = __riscv_vfmv_v_f_f32m8(0x1.003f80p17f, vl);
    vfloat32m8_t q = __riscv_vfmadd_vf_f32m8(va, R, Q, vl);
    vfloat32m8_t f = __riscv_sf_vfexpa_v_f32m8(q, vl);
    /* 2. Narrow */
    vx = __riscv_vfncvtbf16_f_f_w_bf16m4(f, vl);
    /* 3. Accumulate & Store */
    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, f, vl);
    __riscv_vse16_v_bf16m4(dst + i, vx, vl);
  }
  vfloat32m1_t ssum = __riscv_vfmv_s_f_f32m1(0.0f, n);
  ssum = __riscv_vfredusum_vs_f32m8_f32m1(vsum, ssum, n);
  float sum = __riscv_vfmv_f_s_f32m1_f32(ssum);
  float recip_sum = 1.0f / sum;

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(dst + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    va = __riscv_vfmul_vf_f32m8(va, recip_sum, vl);
    vx = __riscv_vfncvtbf16_f_f_w_bf16m4(va, vl);
    __riscv_vse16_v_bf16m4(dst + i, vx, vl);
  }
}
