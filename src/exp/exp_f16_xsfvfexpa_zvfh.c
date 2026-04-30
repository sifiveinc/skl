// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *in,
                                            size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(in + i, vl);
    /* 0. Clamp */
    vbool2_t nn = __riscv_vmfeq_vv_f16m8_b2(vx, vx, vl);
    const _Float16 LB = -0x1.158p4f16; /* round_down(log(0x1p-25)) */
    const _Float16 UB = +0x1.630p3f16; /* round_up(log(0x1.ffep15)) */
    vx = __riscv_vfmax_vf_f16m8_mu(nn, vx, vx, LB, vl);
    vx = __riscv_vfmin_vf_f16m8_mu(nn, vx, vx, UB, vl);
    /* 1. Reduce */
    vfloat16m8_t R = __riscv_vfmv_v_f_f16m8(0x1.714p0f16, vl);
    vfloat16m8_t Q = __riscv_vfmv_v_f_f16m8(0x1.d00p5f16, vl);
    vbool2_t m = __riscv_vmfgt_vf_f16m8_b2(vx, 0.0f16, vl);
    Q = __riscv_vfmerge_vfm_f16m8(Q, 0x1.200p5f16, m, vl);
    vfloat16m8_t v = __riscv_vfmadd_vv_f16m8(R, vx, Q, vl);
    vfloat16m8_t z = __riscv_vfsub_vv_f16m8(v, Q, vl);
    const _Float16 C1 = +0x1.800p-1f16; /* round(log(2),11-5-4) */
    const _Float16 C2 = -0x1.d1cp-5f16; /* log(2) - C1 */
    vfloat16m8_t s = __riscv_vfnmsac_vf_f16m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f16m8(s, C2, z, vl);
    /* 2. Assemble */
    vfloat16m8_t f = __riscv_sf_vfexpa_v_f16m8(v, vl);
    vfloat16m8_t e = __riscv_vfmadd_vv_f16m8(s, f, f, vl);
    /* 3. Rescale */
    vfloat16m8_t r = __riscv_vfmv_v_f_f16m8(0x1p-11f16, vl);
    r = __riscv_vfmerge_vfm_f16m8(r, 0x1p+11f16, m, vl);
    e = __riscv_vfmul_vv_f16m8(e, r, vl);
    /* 4. Store */
    __riscv_vse16_v_f16m8(out + i, e, vl);
  }
}

SKL_FUNC void skl_exp_1p132ugen37P3s0_f16_xsfvfexpa_zvfh(_Float16 *out,
                                                         const _Float16 *in,
                                                         size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(in + i, vl);
    /* 0. Clamp */
    vbool2_t nn = __riscv_vmfeq_vv_f16m8_b2(vx, vx, vl);
    const _Float16 LB = -0x1.4ccp3f16;
    const _Float16 UB = +0x1.630p3f16; /* round_up(log(0x1.ffep15)) */
    vx = __riscv_vfmax_vf_f16m8_mu(nn, vx, vx, LB, vl);
    vx = __riscv_vfmin_vf_f16m8_mu(nn, vx, vx, UB, vl);
    /* 1. Reduce */
    const _Float16 R = 0x1.710p0f16; /* ~1/ln(2), attenuate for large x */
    const _Float16 Q = 0x1.780p5f16; /* 2^(p-5-1) + B */
    vfloat16m8_t q = __riscv_vfmv_v_f_f16m8(Q, vl);
    vfloat16m8_t v = __riscv_vfmacc_vf_f16m8(q, R, vx, vl);
    vfloat16m8_t z = __riscv_vfsub_vf_f16m8(v, Q, vl);
    const _Float16 C1 = +0x1.800p-1f16; /* round(log(2),p-5-3) */
    const _Float16 C2 = -0x1.d1cp-5f16; /* log(2) - C1 */
    vfloat16m8_t s = __riscv_vfnmsac_vf_f16m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f16m8(s, C2, z, vl);
    /* 2. Approximate via P(s) = s and assemble */
    vfloat16m8_t f = __riscv_sf_vfexpa_v_f16m8(v, vl);
    vfloat16m8_t e = __riscv_vfmadd_vv_f16m8(s, f, f, vl);
    /* 3. Store */
    __riscv_vse16_v_f16m8(out + i, e, vl);
  }
}
