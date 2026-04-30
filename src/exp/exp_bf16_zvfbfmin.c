// Copyright (c) 2025 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfbfmin)
#error This file requires the Zvfbfmin extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_bf16_zvfbfmin(__bf16 *out, const __bf16 *in,
                                       size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(in + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    /* 0. Clamp */
    vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(va, va, vl);
    const float LB = -0x1.9fe36ap6f; /* round_down(log(0x1p-150)) */
    const float UB = +0x1.62e430p6f; /* round_up(log(0x1.ffffffp127)) */
    va = __riscv_vfmax_vf_f32m8_mu(nn, va, va, LB, vl);
    va = __riscv_vfmin_vf_f32m8_mu(nn, va, va, UB, vl);
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
    c = __riscv_vfmv_v_f_f32m8(0x1p0f, vl);
    vint32m8_t j = __riscv_vreinterpret_v_f32m8_i32m8(c);
    vint32m8_t k = __riscv_vreinterpret_v_f32m8_i32m8(u);
    vint16m4_t r = __riscv_vand_vx_i16m4(q, -2, vl);
    q = __riscv_vsll_vx_i16m4(q, 8, vl);
    q = __riscv_vnmsac_vx_i16m4(q, 0x80, r, vl);
    r = __riscv_vsll_vx_i16m4(r, 7, vl);
    j = __riscv_vwmaccus_vx_i32m8(j, 0x8000, q, vl);
    k = __riscv_vwmaccus_vx_i32m8(k, 0x8000, r, vl);
    u = __riscv_vreinterpret_v_i32m8_f32m8(k);
    vfloat32m8_t a = __riscv_vreinterpret_v_i32m8_f32m8(j);
    u = __riscv_vfmul_vv_f32m8(u, a, vl);
    /* 4. Narrow */
    vbfloat16m4_t vy = __riscv_vfncvtbf16_f_f_w_bf16m4(u, vl);
    /* 5. Store */
    __riscv_vse16_v_bf16m4(out + i, vy, vl);
  }
}
