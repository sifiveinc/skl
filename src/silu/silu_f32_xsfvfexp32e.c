// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include "skl-common.h"

#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_silu_52u_f32_xsfvfexp32e(float *out, const float *in,
                                           size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m8();
  const vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    /* 0. Load */
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);

    /* 1. Approximate exp(-|vx|) */
    vfloat32m8_t ex = __riscv_vfmul_vf_f32m8(vx, -0.5f, vl);
    ex = __riscv_vfsgnj_vf_f32m8(ex, -0.5f, vl);
    ex = __riscv_sf_vfexp_v_f32m8(ex, vl);
    ex = __riscv_vfmul_vv_f32m8(ex, ex, vl);

    /* 2. Approximate 1 / (1 + exp(-|vx|)) */
    const vfloat32m8_t d = __riscv_vfadd_vf_f32m8(ex, 1, vl);
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);
    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - x * r)
    t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - x * r)

    /* 3. Calculate quotient */
    const vbool4_t m = __riscv_vmflt_vf_f32m8_b4(vx, 0, vl);
    vfloat32m8_t vy = __riscv_vfmul_vv_f32m8_mu(m, r, r, ex, vl);

    /* 4. Multiply */
    vy = __riscv_vfmul_vv_f32m8(vy, vx, vl);

    /* 5. Store */
    __riscv_vse32_v_f32m8(out + i, vy, vl);
  }
}
