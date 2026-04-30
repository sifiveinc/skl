// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

#include "skl-common.h"

#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_logistic_5u_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in,
                                              size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e16m8();
  const vfloat16m8_t one = __riscv_vfmv_v_f_f16m8(1, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    /* 0. Load */
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(in + i, vl);

    /* 1. Approximate exp(-|vx|) */
    vfloat16m8_t ex = __riscv_vfmul_vf_f16m8(vx, -0.5f16, vl);
    ex = __riscv_vfsgnj_vf_f16m8(ex, -0.5f16, vl);
    ex = __riscv_sf_vfexp_v_f16m8(ex, vl);
    ex = __riscv_vfmul_vv_f16m8(ex, ex, vl);

    /* 2. Approximate 1 / (1 + exp(-|vx|)) */
    const vfloat16m8_t d = __riscv_vfadd_vf_f16m8(ex, 1, vl);
    vfloat16m8_t r = __riscv_vfrec7_v_f16m8(d, vl);
    vfloat16m8_t t = __riscv_vfnmsub_vv_f16m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f16m8(r, t, r, vl); // r + r * (1 - x * r)

    /* 3. Calculate quotient */
    const vbool2_t m = __riscv_vmflt_vf_f16m8_b2(vx, 0, vl);
    vfloat16m8_t vy = __riscv_vfmul_vv_f16m8_mu(m, r, r, ex, vl);

    /* 4. Store */
    __riscv_vse16_v_f16m8(out + i, vy, vl);
  }
}
