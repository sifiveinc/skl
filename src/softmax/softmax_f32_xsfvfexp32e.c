// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f32_xsfvfexp32e(float *pDst, const float *pSrc,
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
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_f32m8(vx, max, vl);
    vx = __riscv_vfmul_vf_f32m8(vx, beta, vl);
    vx = __riscv_sf_vfexp_v_f32m8(vx, vl);
    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, vx, vl);
    __riscv_vse32_v_f32m8(pDst + i, vx, vl);
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
