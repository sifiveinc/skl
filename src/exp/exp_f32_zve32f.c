// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_f32_zve32f(float *out, const float *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);
    // 0. Clamp
    vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(vx, vx, vl);
    const float LB = -0x1.9fe36ap6f; // round_down(log(0x1p-150))
    const float UB = +0x1.62e430p6f; // round_up(log(0x1.ffffffp127))
    vx = __riscv_vfmax_vf_f32m8_mu(nn, vx, vx, LB, vl);
    vx = __riscv_vfmin_vf_f32m8_mu(nn, vx, vx, UB, vl);
    // 1. Reduce
    const float R = 0x1.715476p0f;    // 1/log(2)
    const float C1 = 0x1.62e400p-1f;  // log(2)
    const float C2 = 0x1.7f7d1cp-20f; // log(2) - C1
    vfloat32m8_t v = __riscv_vfmul_vf_f32m8(vx, R, vl);
    vint16m4_t q = __riscv_vmv_v_x_i16m4(0, vl);
    q = __riscv_vfncvt_x_f_w_i16m4_mu(nn, q, v, vl);
    vfloat32m8_t z = __riscv_vfwcvt_f_x_v_f32m8(q, vl);
    vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f32m8(s, C2, z, vl);
    // 2. Approximate exp(s)
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(0x1.123bccp-7f, vl);
    vfloat32m8_t u = __riscv_vfmacc_vf_f32m8(c, 0x1.6850e4p-10f, s, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.555b98p-5f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.55548ep-3f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.fffff8p-2f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c, vl);
    c = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c, vl);
    // 3. Assemble
    vint32m8_t r = __riscv_vreinterpret_v_f32m8_i32m8(c);
    vint32m8_t t = __riscv_vreinterpret_v_f32m8_i32m8(u);
    vint16m4_t j = __riscv_vand_vx_i16m4(q, -2, vl);
    vint16m4_t k = __riscv_vsll_vx_i16m4(q, 8, vl);
    k = __riscv_vnmsac(k, 1 << 7, j, vl);
    j = __riscv_vsll(j, 7, vl);
    r = __riscv_vwmaccus_vx_i32m8(r, 1 << 15, j, vl);
    t = __riscv_vwmaccus_vx_i32m8(t, 1 << 15, k, vl);
    u = __riscv_vreinterpret_v_i32m8_f32m8(t);
    v = __riscv_vreinterpret_v_i32m8_f32m8(r);
    u = __riscv_vfmul_vv_f32m8(u, v, vl);
    // 4. Store
    __riscv_vse32_v_f32m8(out + i, u, vl);
  }
}
