// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in,
                                       size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(in + i, vl);
    /* 0. Clamp */
    vbool4_t nn = __riscv_vmfeq_vv_bf16m4_b4(vx, vx, vl);
    const __bf16 LB = (__bf16)-0x1.74p6; /* round_down(log(0x1p-134)) */
    const __bf16 UB = (__bf16)+0x1.64p6; /* round_up(log(0x1.ffp127)) */
    vx = __riscv_vfmax_vf_bf16m4_mu(nn, vx, vx, LB, vl);
    vx = __riscv_vfmin_vf_bf16m4_mu(nn, vx, vx, UB, vl);
    /* 1. Reduce */
    vbfloat16m4_t v = __riscv_vfmul_vf_bf16m4(vx, (__bf16)0x1.72p0, vl);
    vint8m2_t j = __riscv_vmv_v_x_i8m2(0, vl);
    j = __riscv_vfncvt_x_f_w_bf16m4_mu(nn, j, v, vl);
    j = __riscv_vmax_vx_i8m2(j, -126, vl);
    vbfloat16m4_t w = __riscv_vfwcvt_bf_x_v_bf16m4(j, vl);
    v = __riscv_vfsub_vv_bf16m4(v, w, vl);
    vint8m2_t k = __riscv_vmv_v_x_i8m2(0, vl);
    k = __riscv_vfncvt_x_f_w_bf16m4_mu(nn, k, v, vl);
    vbfloat16m4_t y = __riscv_vfwcvt_bf_x_v_bf16m4(k, vl);
    vbfloat16m4_t z = __riscv_vfadd_vv_bf16m4(w, y, vl);
    vbfloat16m4_t s = __riscv_vfnmsac_vf_bf16m4(vx, (__bf16)0x1.6p-1, z, vl);
    s = __riscv_vfnmsac_vf_bf16m4(s, (__bf16)0x1.72p-8, z, vl);
    /* 2. Approximate exp(s) */
    vbfloat16m4_t c = __riscv_vfmv_v_f_bf16m4((__bf16)0x1.06p-1, vl);
    vbfloat16m4_t u = __riscv_vfmacc_vf_bf16m4(c, (__bf16)0x1.56p-3, s, vl);
    c = __riscv_vfmv_v_f_bf16m4((__bf16)0x1p0, vl);
    u = __riscv_vfmadd_vv_bf16m4(u, s, c, vl);
    u = __riscv_vfmadd_vv_bf16m4(u, s, c, vl);
    /* 3. Assemble */
    vint16m4_t r = __riscv_vreinterpret_v_bf16m4_i16m4(c);
    vint16m4_t t = __riscv_vreinterpret_v_bf16m4_i16m4(u);
    r = __riscv_vwmaccus_vx_i16m4(r, 1 << 7, j, vl);
    t = __riscv_vwmaccus_vx_i16m4(t, 1 << 7, k, vl);
    u = __riscv_vreinterpret_v_i16m4_bf16m4(t);
    v = __riscv_vreinterpret_v_i16m4_bf16m4(r);
    u = __riscv_vfmul_vv_bf16m4(u, v, vl);
    /* 4. Store */
    __riscv_vse16_v_bf16m4(out + i, u, vl);
  }
}
