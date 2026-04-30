// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n) {
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
    const _Float16 R = 0x1.714p0f16;    /* 1/log(2) */
    const _Float16 C1 = 0x1.63p-1f16;   /* log(2) */
    const _Float16 C2 = -0x1.bdp-13f16; /* log(2) - C1 */
    vfloat16m8_t v = __riscv_vfmul_vf_f16m8(vx, R, vl);
    vint8m4_t q = __riscv_vmv_v_x_i8m4(0, vl);
    q = __riscv_vfncvt_x_f_w_i8m4_mu(nn, q, v, vl);
    vfloat16m8_t z = __riscv_vfwcvt_f_x_v_f16m8(q, vl);
    vfloat16m8_t s = __riscv_vfnmsac_vf_f16m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f16m8(s, C2, z, vl);
    /* 2. Approximate exp(s) */
    vfloat16m8_t c = __riscv_vfmv_v_f_f16m8(0x1.024p-1f16, vl);
    vfloat16m8_t u = __riscv_vfmacc_vf_f16m8(c, 0x1.574p-3f16, s, vl);
    c = __riscv_vfmv_v_f_f16m8(1.0f16, vl);
    u = __riscv_vfmadd_vv_f16m8(u, s, c, vl);
    u = __riscv_vfmadd_vv_f16m8(u, s, c, vl);
    /* 3. Assemble */
    vint16m8_t r = __riscv_vreinterpret_v_f16m8_i16m8(c);
    vint16m8_t t = __riscv_vreinterpret_v_f16m8_i16m8(u);
    vint8m4_t j = __riscv_vand_vx_i8m4(q, -2, vl);
    vint8m4_t k = __riscv_vsll_vx_i8m4(q, 3, vl);
    k = __riscv_vnmsac(k, 1 << 2, j, vl);
    j = __riscv_vsll(j, 2, vl);
    r = __riscv_vwmaccus_vx_i16m8(r, 1 << 7, j, vl);
    t = __riscv_vwmaccus_vx_i16m8(t, 1 << 7, k, vl);
    u = __riscv_vreinterpret_v_i16m8_f16m8(t);
    v = __riscv_vreinterpret_v_i16m8_f16m8(r);
    u = __riscv_vfmul_vv_f16m8(u, v, vl);
    /* 4. Store */
    __riscv_vse16_v_f16m8(out + i, u, vl);
  }
}
