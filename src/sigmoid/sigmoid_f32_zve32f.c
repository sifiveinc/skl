// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

SKL_FUNC void skl_sigmoid_f32_zve32f(float *out, float beta, const float *x,
                                     const float *y, const float *up,
                                     float delta, size_t n) {
  if (!x)
    return;

  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    /* 0. Load and scale by beta */
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
    if (beta != 1)
      vx = __riscv_vfmul_vf_f32m8(vx, beta, vl);

    /* 1. Observe, Orient, & Clamp */
    const vbool4_t m = __riscv_vmflt_vf_f32m8_b4(vx, 0, vl);
    vx = __riscv_vfsgnj_vf_f32m8(vx, -1, vl);
    vx = __riscv_vfmax_vf_f32m8(vx, -0x1.9fe36ap6f, vl);

    /* 2. Reduce x ~ z ln2 + s */
    const float R = 0x1.715476p0f;    /* 1/ln2 */
    const float C1 = 0x1.62e4p-1f;    /* round(log(2), 24-8, RN) */
    const float C2 = 0x1.7f7d1cp-20f; /* ln2 - C1 */
    const vfloat32m8_t v = __riscv_vfmul_vf_f32m8(vx, R, vl);
    const vint16m4_t zi = __riscv_vfncvt_x_f_w_i16m4(v, vl);
    const vfloat32m8_t z = __riscv_vfwcvt_f_x_v_f32m8(zi, vl);
    vfloat32m8_t s =
        __riscv_vfnmsac_vf_f32m8(vx, C1, z, vl); /* s = x - C1 * z */
    const vfloat32m8_t c5 = __riscv_vfmv_v_f_f32m8(0x1.123bccp-7f, vl);
    s = __riscv_vfnmsac_vf_f32m8(s, C2, z, vl); /* s = s - C2 * z */

    /* 3. Approximate */
    const float c6 = 0x1.6850e4p-10f;
    vfloat32m8_t u = __riscv_vfmacc_vf_f32m8(c5, c6, s, vl);
    vfloat32m8_t c4 = __riscv_vfmv_v_f_f32m8(0x1.555b98p-5f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c4, vl);
    vfloat32m8_t c3 = __riscv_vfmv_v_f_f32m8(0x1.55548ep-3f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c3, vl);
    vfloat32m8_t c2 = __riscv_vfmv_v_f_f32m8(0x1.fffff8p-2f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c2, vl);
    vfloat32m8_t c01 = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c01, vl);
    u = __riscv_vfmadd_vv_f32m8(u, s, c01, vl);

    /* 4. Reconstruct e ~ u*2^(z+24) */
    vint32m8_t k = __riscv_vreinterpret_v_f32m8_i32m8(u);
    const int16_t exp_ = 24 << 8;
    vint16m4_t vexp = __riscv_vlse16_v_i16m4(&exp_, 0, vl);
    vexp = __riscv_vmadd_vx_i16m4(zi, 1 << 8, vexp, vl);
    k = __riscv_vwmaccus_vx_i32m8(k, 1 << 15, vexp, vl);
    vfloat32m8_t e = __riscv_vreinterpret_v_i32m8_f32m8(k);

    /* 5. Re-scale */
    e = __riscv_vfmul_vf_f32m8(e, 0x1p-24f, vl); /* UF happens here */

    /* 6. Reciprocate denominator */
    const vfloat32m8_t d = __riscv_vfadd_vf_f32m8(e, 1, vl);
    const vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1, vl);
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);
    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl);
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);
    t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl); /* 1 - x * r */
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    /* r + r * (1 - x * r) */

    /* 7. Calculate quotient */
    vfloat32m8_t q = __riscv_vfmul_vv_f32m8_mu(m, r, r, e, vl);

    /* 8. Optionally multiply by y */
    if (y) {
      const vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
      q = __riscv_vfmul_vv_f32m8(q, vy, vl);
    }

    /* 9. Optionally multiply by (up + delta) */
    if (up) {
      vfloat32m8_t vu = __riscv_vle32_v_f32m8(up + i, vl);
      vu = __riscv_vfadd_vf_f32m8(vu, delta, vl);
      q = __riscv_vfmul_vv_f32m8(q, vu, vl);
    }

    /* 10. Store */
    __riscv_vse32_v_f32m8(out + i, q, vl);
  }
}

SKL_FUNC void skl_logistic_f32_zve32f(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_zve32f(out, 1.f, x, NULL, NULL, 0.f, n);
}

SKL_FUNC void skl_silu_f32_zve32f(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_zve32f(out, 1.f, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swish_f32_zve32f(float *out, float beta, const float *x,
                                   size_t n) {
  skl_sigmoid_f32_zve32f(out, beta, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_glu_f32_zve32f(float *out, const float *x, const float *y,
                                 size_t n) {
  skl_sigmoid_f32_zve32f(out, 1.f, y, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swiglu_f32_zve32f(float *out, const float *gate,
                                    const float *up, float delta, size_t n) {
  skl_sigmoid_f32_zve32f(out, 1.f, gate, gate, up, delta, n);
}
