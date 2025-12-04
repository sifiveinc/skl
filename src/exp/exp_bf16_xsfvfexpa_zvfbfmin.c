// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfbfmin)
#error This file requires the Xsfvfexpa and Zvfbfmin extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_bf16_xsfvfexpa_zvfbfmin(__bf16 *out, const __bf16 *in,
                                                 size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t vx = __riscv_vle16_v_bf16m4(in + i, vl);
    vfloat32m8_t va = __riscv_vfwcvtbf16_f_f_v_f32m8(vx, vl);
    /* 0. Clamp */
    vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(va, va, vl);
    const float LB = -0x1.74p6f; /* round_down(log(0x1p-134)) */
    const float UB = +0x1.64p6f; /* round_up(log(0x1.ffp127)) */
    va = __riscv_vfmax_vf_f32m8_mu(nn, va, va, LB, vl);
    va = __riscv_vfmin_vf_f32m8_mu(nn, va, va, UB, vl);
    /* 1. Reduce */
    vfloat32m8_t R = __riscv_vfmv_v_f_f32m8(0x1.715476p-1f, vl);
    vfloat32m8_t Q = __riscv_vfmv_v_f_f32m8(0x1.003f80p17f, vl);
    vfloat32m8_t v = __riscv_vfmadd_vv_f32m8(R, va, Q, vl);
    vfloat32m8_t z = __riscv_vfsub_vv_f32m8(v, Q, vl);
    vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(va, 0x1.62e43p0f, z, vl);
    /* 2. Assemble */
    vfloat32m8_t f = __riscv_sf_vfexpa_v_f32m8(v, vl);
    vfloat32m8_t g = __riscv_vfmadd_vv_f32m8(s, f, f, vl);
    vfloat32m8_t e = __riscv_vfmul_vv_f32m8(f, g, vl);
    /* 3. Narrow */
    vbfloat16m4_t vy = __riscv_vfncvtbf16_f_f_w_bf16m4(e, vl);
    /* 4. Store */
    __riscv_vse16_v_bf16m4(out + i, vy, vl);
  }
}
