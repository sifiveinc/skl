// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_softmax_f16_xsfvfexpa_zvfh(_Float16 *pDst,
                                             const _Float16 *pSrc,
                                             const _Float16 beta,
                                             const size_t n) {
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
    /* Calculate sum += exp(beta * (x - max)) */

    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_f16m8(vx, max, vl);
    if (beta != 1.0f16)
      vx = __riscv_vfmul_vf_f16m8(vx, beta, vl);

    /* 0. Clamp */
    const _Float16 xmin = -0x1.4c8p+3f16;
    vx = __riscv_vfmax_vf_f16m8(vx, xmin, vl);
    /* 1. Reduce */
    const _Float16 R = 0x1.714p0f16;    /* 1/log(2) */
    const _Float16 Q = 0x1.78p5f16;     /* 2^(11-5-1) + B */
    const _Float16 C1 = +0x1.800p-1f16; /* ~log(2) */
    const _Float16 C2 = -0x1.d1cp-5f16; /* log(2) - C1 */
    vfloat16m8_t vQ = __riscv_vfmv_v_f_f16m8(Q, vl);
    vfloat16m8_t vv = __riscv_vfmacc_vf_f16m8(vQ, R, vx, vl);
    vfloat16m8_t vz = __riscv_vfsub_vf_f16m8(vv, Q, vl);
    vfloat16m8_t vs = __riscv_vfnmsac_vf_f16m8(vx, C1, vz, vl);
    vs = __riscv_vfnmsac_vf_f16m8(vs, C2, vz, vl);
    /* 2. Assemble with expm1(s) ~ s */
    vfloat16m8_t vf = __riscv_sf_vfexpa_v_f16m8(vv, vl);
    vfloat16m8_t vy = __riscv_vfmadd_vv_f16m8(vs, vf, vf, vl);
    /* 3. Accumulate and Store */
    vsum = __riscv_vfadd_vv_f16m8_tu(vsum, vsum, vy, vl);
    __riscv_vse16_v_f16m8(pDst + i, vy, vl);
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
