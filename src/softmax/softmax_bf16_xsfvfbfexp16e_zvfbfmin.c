// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfbfexp16e) || !defined(__riscv_zvfbfmin)
#error This file requires the Xsfvfbfexp16e and Zvfbfmin extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_bf16_xsfvfbfexp16e_zvfbfmin(__bf16 *pDst,
                                                      const __bf16 *pSrc,
                                                      const __bf16 beta,
                                                      const size_t n) {
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
    vx = __riscv_vfncvtbf16_f_f_w_bf16m4(va, vl);
    vx = __riscv_sf_vfexp_v_bf16m4(vx, vl);
    va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, va, vl);
    __riscv_vse16_v_bf16m4(pDst + i, vx, vl);
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
