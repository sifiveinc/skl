// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f16_xsfvfexp16e(_Float16 *pDst, const _Float16 *pSrc,
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

  vfloat16m8_t vsum = __riscv_vfmv_v_f_f16m8(0, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_f16m8(vx, max, vl);
    if (beta != 1.0f16)
      vx = __riscv_vfmul_vf_f16m8(vx, beta, vl);
    vx = __riscv_sf_vfexp_v_f16m8(vx, vl);
    vsum = __riscv_vfadd_vv_f16m8_tu(vsum, vsum, vx, vl);
    __riscv_vse16_v_f16m8(pDst + i, vx, vl);
  }
  vfloat16m1_t ssum = __riscv_vfmv_s_f_f16m1(0, n);
  ssum = __riscv_vfredusum_vs_f16m8_f16m1(vsum, ssum, n);
  _Float16 sum = __riscv_vfmv_f_s_f16m1_f16(ssum);
  _Float16 recip_sum = 1 / sum;

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(pDst + i, vl);
    vx = __riscv_vfmul_vf_f16m8(vx, recip_sum, vl);
    __riscv_vse16_v_f16m8(pDst + i, vx, vl);
  }
}
