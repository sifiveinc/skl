// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sqrt_f32_zve32f(float *out, const float *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t h = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t o = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t x = __riscv_vle32_v_f32m4(in + i, vl);

    vfloat32m4_t y = __riscv_vfrsqrt7_v_f32m4(x, vl), z = y;
    vfloat32m4_t s = __riscv_vfmul_vv_f32m4(x, y, vl);
    vfloat32m4_t u = __riscv_vfmul_vf_f32m4(y, 0.5f, vl);

    vfloat32m4_t t = __riscv_vfnmsub_vv_f32m4(s, u, h, vl);
    s = __riscv_vfmadd_vv_f32m4(s, t, s, vl);
    u = __riscv_vfmadd_vv_f32m4(u, t, u, vl);

    t = __riscv_vfnmsub_vv_f32m4(s, u, h, vl);
    y = __riscv_vfmul_vf_f32m4(u, 2.0f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, t, y, vl);

    vbool8_t m = __riscv_vmfeq_vv_f32m4_b8(t, t, vl);
    z = __riscv_vfrec7_v_f32m4(z, vl);
    s = __riscv_vfmul_vv_f32m4(x, y, vl);
    x = __riscv_vfnmsub(x, y, s, vl);
    u = __riscv_vfmul_vf_f32m4_mu(m, z, s, 0.5f, vl);
    t = __riscv_vfnmsub(y, s, o, vl);
    t = __riscv_vfnmsac_vv_f32m4(t, x, y, vl);
    s = __riscv_vfmadd_vv_f32m4_mu(m, u, t, s, vl);

    // 4. Store
    __riscv_vse32_v_f32m4(out + i, s, vl);
  }
}

SKL_FUNC void skl_sqrt_1u_f32_zve32f(float *out, const float *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t x = __riscv_vle32_v_f32m4(in + i, vl);
    vfloat32m4_t y = __riscv_vfrsqrt7_v_f32m4(x, vl), z = y;
    vfloat32m4_t o = __riscv_vfmv_v_f_f32m4(1.0f, vl);

    // First refinement, ~14 bits
    vfloat32m4_t s = __riscv_vfmul_vv_f32m4(x, y, vl);
    vfloat32m4_t u = __riscv_vfmul_vf_f32m4(y, 0.5f, vl);
    vfloat32m4_t t = __riscv_vfnmsub_vv_f32m4(s, y, o, vl);
    y = __riscv_vfmacc_vv_f32m4(y, t, u, vl);

    // Second refinement, ~23.3 bits
    vbool8_t m =
        __riscv_vmfeq_vv_f32m4_b8(t, t, vl); /* t NaN if x≤0, inf, NaN */
    z = __riscv_vfrec7_v_f32m4(z, vl);
    s = __riscv_vfmul_vv_f32m4(x, y, vl);
    u = __riscv_vfmul_vf_f32m4_mu(m, z, s, 0.5f, vl);
    t = __riscv_vfnmsub(y, s, o, vl);
    s = __riscv_vfmadd_vv_f32m4_mu(m, u, t, s, vl); /* xy + 0.5xy (1-xy²) */

    // 4. Store
    __riscv_vse32_v_f32m4(out + i, s, vl);
  }
}
