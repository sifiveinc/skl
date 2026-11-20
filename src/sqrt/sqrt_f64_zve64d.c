// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve64d)
#error This file requires the Zve64d extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sqrt_f64_zve64d(double *out, const double *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t h = __riscv_vfmv_v_f_f64m4(0.5, vl);
    vfloat64m4_t o = __riscv_vfmv_v_f_f64m4(1.0, vl);
    vfloat64m4_t x = __riscv_vle64_v_f64m4(in + i, vl);

    vfloat64m4_t y = __riscv_vfrsqrt7_v_f64m4(x, vl), z = y;
    vfloat64m4_t s = __riscv_vfmul_vv_f64m4(x, y, vl);
    vfloat64m4_t u = __riscv_vfmul_vf_f64m4(y, 0.5, vl);

    vfloat64m4_t t = __riscv_vfnmsub_vv_f64m4(s, u, h, vl);
    s = __riscv_vfmadd_vv_f64m4(s, t, s, vl);
    u = __riscv_vfmadd_vv_f64m4(u, t, u, vl);

    t = __riscv_vfnmsub_vv_f64m4(s, u, h, vl);
    s = __riscv_vfmadd_vv_f64m4(s, t, s, vl);
    u = __riscv_vfmadd_vv_f64m4(u, t, u, vl);

    t = __riscv_vfnmsub_vv_f64m4(s, u, h, vl);
    y = __riscv_vfmul_vf_f64m4(u, 2.0, vl);
    y = __riscv_vfmadd_vv_f64m4(y, t, y, vl);

    vbool16_t m = __riscv_vmfeq_vv_f64m4_b16(t, t, vl);
    z = __riscv_vfrec7_v_f64m4(z, vl);
    s = __riscv_vfmul_vv_f64m4(x, y, vl);
    x = __riscv_vfnmsub(x, y, s, vl);
    u = __riscv_vfmul_vf_f64m4_mu(m, z, s, 0.5, vl);
    t = __riscv_vfnmsub(y, s, o, vl);
    t = __riscv_vfnmsac_vv_f64m4(t, x, y, vl);
    s = __riscv_vfmadd_vv_f64m4_mu(m, u, t, s, vl);

    // 4. Store
    __riscv_vse64_v_f64m4(out + i, s, vl);
  }
}

SKL_FUNC void skl_sqrt_1u_f64_zve64d(double *out, const double *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t x = __riscv_vle64_v_f64m4(in + i, vl);
    vfloat64m4_t y = __riscv_vfrsqrt7_v_f64m4(x, vl), z = y;
    double a = 0x1.8p-2;
    vfloat64m4_t b = __riscv_vfmv_v_f_f64m4(0x1.4p0, vl);
    vfloat64m4_t c = __riscv_vfmv_v_f_f64m4(0x1.ep0, vl);
    vfloat64m4_t o = __riscv_vfmv_v_f_f64m4(1.0, vl);

    // First refinement, ~21 bits
    vfloat64m4_t s = __riscv_vfmul_vv_f64m4(x, y, vl);
    vfloat64m4_t t = __riscv_vfmul_vv_f64m4(s, y, vl);
    vfloat64m4_t u = __riscv_vfnmsac_vf_f64m4(b, a, t, vl);
    t = __riscv_vfnmsub_vv_f64m4(t, u, c, vl);
    y = __riscv_vfmul_vv_f64m4(y, t, vl);

    // Second refinement, ~42 bits
    s = __riscv_vfmul_vv_f64m4(x, y, vl);
    u = __riscv_vfmul_vf_f64m4(y, 0.5, vl);
    t = __riscv_vfnmsub_vv_f64m4(s, y, o, vl);
    y = __riscv_vfmacc_vv_f64m4(y, t, u, vl);

    // Third refinement, ~52.4 bits
    vbool16_t m =
        __riscv_vmfeq_vv_f64m4_b16(t, t, vl); /* t NaN if x≤0, inf, NaN */
    z = __riscv_vfrec7_v_f64m4(z, vl);
    s = __riscv_vfmul_vv_f64m4(x, y, vl);
    u = __riscv_vfmul_vf_f64m4_mu(m, z, s, 0.5, vl);
    t = __riscv_vfnmsub(y, s, o, vl);
    s = __riscv_vfmadd_vv_f64m4_mu(m, u, t, s, vl); /* xy + 0.5xy (1-xy²) */

    // 4. Store
    __riscv_vse64_v_f64m4(out + i, s, vl);
  }
}
