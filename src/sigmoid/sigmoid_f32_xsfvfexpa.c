// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sigmoid_f32_xsfvfexpa(float *out, float beta, const float *x,
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
    const float R = 0x1.715476p0f;     /* 1/ln2 */
    const float Q = 0x1.004b80p17f;    /* 2^(24-6-1) + B + p */
    const float C1 = 0x1.63p-1f;       /* round(ln2, 24-6-7, RN) */
    const float C2 = -0x1.bd0106p-13f; /* ln2 - C1 */
    const vfloat32m8_t v =
        __riscv_vfmacc_vf_f32m8(__riscv_vfmv_v_f_f32m8(Q, vl), R, vx, vl);
    const vfloat32m8_t f = __riscv_sf_vfexpa_v_f32m8(v, vl);
    const vfloat32m8_t z = __riscv_vfsub_vf_f32m8(v, Q, vl);
    vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f32m8(s, C2, z, vl);

    /* 3. Approximate */
    const float a3 = 0x1.5558f4p-3f;
    const float a2 = 0x1.fff92cp-2f;
    vfloat32m8_t p =
        __riscv_vfmacc_vf_f32m8(__riscv_vfmv_v_f_f32m8(a2, vl), a3, s, vl);
    p = __riscv_vfmadd_vv_f32m8(p, __riscv_vfmul_vv_f32m8(s, s, vl), s, vl);
    vfloat32m8_t e = __riscv_vfmadd_vv_f32m8(p, f, f, vl);

    /* 4. Re-scale */
    e = __riscv_vfmul_vf_f32m8(e, 0x1p-24f, vl); /* UF happens here */

    /* 5. Reciprocate denominator */
    const vfloat32m8_t d = __riscv_vfadd_vf_f32m8(e, 1, vl);
    /* Since we know that e is in [0;1] and d is in [1;2], we can skip the usual
     * scaling and refinement checks that are required in `div`. */
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);

    /* 6. Refine r */
    const vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1, vl);
    vfloat32m8_t t = __riscv_vfnmsac_vv_f32m8(one, d, r, vl); /* 1 - d * r */
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    /* r + r * (1 - d * r) */
    t = __riscv_vfnmsac_vv_f32m8(one, d, r, vl); /* 1 - d * r */
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    /* r + r * (1 - d * r) */

    /* 7. Calculate quotient */
    vfloat32m8_t res = __riscv_vfmul_vv_f32m8_mu(m, r, r, e, vl);

    /* 8. Optionally multiply by y */
    if (y) {
      const vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
      res = __riscv_vfmul_vv_f32m8(res, vy, vl);
    }

    /* 9. Optionally multiply by (up + delta) */
    if (up) {
      vfloat32m8_t vu = __riscv_vle32_v_f32m8(up + i, vl);
      vu = __riscv_vfadd_vf_f32m8(vu, delta, vl);
      res = __riscv_vfmul_vv_f32m8(res, vu, vl);
    }

    /* 10. Store */
    __riscv_vse32_v_f32m8(out + i, res, vl);
  }
}

SKL_FUNC void skl_logistic_f32_xsfvfexpa(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_xsfvfexpa(out, 1.f, x, NULL, NULL, 0.f, n);
}

SKL_FUNC void skl_silu_f32_xsfvfexpa(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_xsfvfexpa(out, 1.f, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swish_f32_xsfvfexpa(float *out, float beta, const float *x,
                                      size_t n) {
  skl_sigmoid_f32_xsfvfexpa(out, beta, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_glu_f32_xsfvfexpa(float *out, const float *x, const float *y,
                                    size_t n) {
  skl_sigmoid_f32_xsfvfexpa(out, 1.f, y, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swiglu_f32_xsfvfexpa(float *out, const float *gate,
                                       const float *up, float delta, size_t n) {
  skl_sigmoid_f32_xsfvfexpa(out, 1.f, gate, gate, up, delta, n);
}
