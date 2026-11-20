// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sqrt_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m8_t h = __riscv_vfmv_v_f_f32m8(0.5f, vl);
    vfloat16m4_t x = __riscv_vle16_v_f16m4(in + i, vl);

    vfloat16m4_t y = __riscv_vfrsqrt7_v_f16m4(x, vl);
    vfloat32m8_t c = __riscv_vfwmul_vv_f32m8(x, y, vl);
    vfloat32m8_t u = __riscv_vfwmul_vf_f32m8(y, 0.5f16, vl);

    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(c, u, h, vl);
    c = __riscv_vfmadd_vv_f32m8(c, t, c, vl);
    u = __riscv_vfmadd_vv_f32m8(u, t, u, vl);

    t = __riscv_vfnmsub_vv_f32m8(c, u, h, vl);
    c = __riscv_vfmadd_vv_f32m8(c, t, c, vl);

    // Handle special values
    vbool4_t m = __riscv_vmfeq_vv_f32m8_b4(t, t, vl);
    y = __riscv_vfrec7_v_f16m4(y, vl);
    vfloat16m4_t s = __riscv_vfncvt_f_f_w_f16m4_mu(m, y, c, vl);

    // 4. Store
    __riscv_vse16_v_f16m4(out + i, s, vl);
  }
}

SKL_FUNC void skl_sqrt_1u_f16_zvfh(_Float16 *out, const _Float16 *in,
                                   size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat16m4_t x = __riscv_vle16_v_f16m4(in + i, vl);
    vfloat16m4_t y = __riscv_vfrsqrt7_v_f16m4(x, vl), z = y;
    vfloat16m4_t o = __riscv_vfmv_v_f_f16m4(1.0f16, vl);

    // Refinement, ~10.35 bits
    vfloat16m4_t s = __riscv_vfmul_vv_f16m4(x, y, vl);
    vfloat16m4_t u = __riscv_vfmul_vf_f16m4(s, 0.5f16, vl);
    y = __riscv_vfnmsub_vv_f16m4(y, s, o, vl); /* y = 1 - xy² */
    s = __riscv_vfmacc_vv_f16m4(s, u, y, vl);  /* xy + (0.5 xy) (1 - xy²) */

    // Handle special values: u is NaN if x≤0, inf, NaN
    vbool4_t m = __riscv_vmfne_vv_f16m4_b4(u, u, vl);
    s = __riscv_vfrec7_v_f16m4_mu(m, s, z, vl);

    // 4. Store
    __riscv_vse16_v_f16m4(out + i, s, vl);
  }
}
