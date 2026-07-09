// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include "skl-common.h"

#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sigmoid_f32_xsfvfexp32e(float *out, float beta,
                                          const float *x, const float *y,
                                          const float *up, float delta,
                                          size_t n) {
  if (!x)
    return;

  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m8();
  const vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    /* 0. Load and scale by beta */
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
    if (beta != 1)
      vx = __riscv_vfmul_vf_f32m8(vx, beta, vl);

    /* 1. Approximate exp(-|beta x|). Halve the argument to stay in range,
     *    then square to recover the full exponential. Keeping the sign
     *    negative saves a scalar register. */
    vfloat32m8_t ex = __riscv_vfmul_vf_f32m8(vx, -0.5f, vl);
    ex = __riscv_vfsgnj_vf_f32m8(ex, -0.5f, vl);
    ex = __riscv_sf_vfexp_v_f32m8(ex, vl);
    ex = __riscv_vfmul_vv_f32m8(ex, ex, vl);

    /* 2. Approximate 1 / (1 + exp(-|beta x|)) */
    const vfloat32m8_t d = __riscv_vfadd_vf_f32m8(ex, 1, vl);
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);
    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - x * r)
    t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - x * r)

    /* 3. Calculate quotient */
    const vbool4_t m = __riscv_vmflt_vf_f32m8_b4(vx, 0, vl);
    vfloat32m8_t res = __riscv_vfmul_vv_f32m8_mu(m, r, r, ex, vl);

    /* 4. Optionally multiply by y */
    if (y) {
      const vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
      res = __riscv_vfmul_vv_f32m8(res, vy, vl);
    }

    /* 5. Optionally multiply by (up + delta) */
    if (up) {
      vfloat32m8_t vu = __riscv_vle32_v_f32m8(up + i, vl);
      vu = __riscv_vfadd_vf_f32m8(vu, delta, vl);
      res = __riscv_vfmul_vv_f32m8(res, vu, vl);
    }

    /* 6. Store */
    __riscv_vse32_v_f32m8(out + i, res, vl);
  }
}

SKL_FUNC void skl_logistic_f32_xsfvfexp32e(float *out, const float *x,
                                           size_t n) {
  skl_sigmoid_f32_xsfvfexp32e(out, 1.f, x, NULL, NULL, 0.f, n);
}

SKL_FUNC void skl_silu_f32_xsfvfexp32e(float *out, const float *x, size_t n) {
  skl_sigmoid_f32_xsfvfexp32e(out, 1.f, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swish_f32_xsfvfexp32e(float *out, float beta, const float *x,
                                        size_t n) {
  skl_sigmoid_f32_xsfvfexp32e(out, beta, x, x, NULL, 0.f, n);
}

SKL_FUNC void skl_glu_f32_xsfvfexp32e(float *out, const float *x,
                                      const float *y, size_t n) {
  skl_sigmoid_f32_xsfvfexp32e(out, 1.f, y, x, NULL, 0.f, n);
}

SKL_FUNC void skl_swiglu_f32_xsfvfexp32e(float *out, const float *gate,
                                         const float *up, float delta,
                                         size_t n) {
  skl_sigmoid_f32_xsfvfexp32e(out, 1.f, gate, gate, up, delta, n);
}
