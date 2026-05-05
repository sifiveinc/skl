// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfexp16e) || !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfexp16e and Xsfvfbfa extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

/** BF16 1D unit-stride softmax using the Xsfvfbfexp16e and Xsfvfbfa
 *  extensions.
 *
 * Exploits the SiFive vector floating-point exponential function
 * instruction to compute the e^x part of softmax.
 *
 * Equivalent to:
 * skl_softmax_bf16_ref(pDst, pSrc, beta, n);
 */
SKL_FUNC void skl_softmax_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *pDst,
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

  vbfloat16m8_t vsum = __riscv_vfmv_v_f_bf16m8(0, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_bf16m8(vx, max, vl);
    if (beta != (__bf16)1.0)
      vx = __riscv_vfmul_vf_bf16m8(vx, beta, vl);
    vx = __riscv_sf_vfexp_v_bf16m8(vx, vl);
    vsum = __riscv_vfadd_vv_bf16m8_tu(vsum, vsum, vx, vl);
    __riscv_vse16_v_bf16m8(pDst + i, vx, vl);
  }
  size_t vlmax4 = __riscv_vsetvlmax_e16m4();
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
