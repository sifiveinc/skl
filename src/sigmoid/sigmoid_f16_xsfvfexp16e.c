// Copyright (c) 2026 SiFive, Inc. All rights reserved.
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

SKL_FUNC void skl_sigmoid_f16_xsfvfexp16e(_Float16 *out, _Float16 beta,
                                          const _Float16 *x, const _Float16 *y,
                                          const _Float16 *up, _Float16 delta,
                                          size_t n) {
  if (!x)
    return;

  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e16m8();
  const vfloat16m8_t one = __riscv_vfmv_v_f_f16m8(1, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    /* 0. Load */
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(x + i, vl);
    const vbool2_t m = __riscv_vmflt_vf_f16m8_b2(vx, 0, vl);

    /* 1. Exponentiate */
    vfloat16m8_t e = __riscv_vfmul_vf_f16m8(vx, -0.5f16 * beta, vl);
    e = __riscv_vfsgnj_vf_f16m8(e, -0.5f16, vl);
    e = __riscv_sf_vfexp_v_f16m8(e, vl);
    e = __riscv_vfmul_vv_f16m8(e, e, vl);

    /* 2. Reciprocate denominator */
    const vfloat16m8_t d = __riscv_vfadd_vf_f16m8(e, 1, vl);
    vfloat16m8_t r = __riscv_vfrec7_v_f16m8(d, vl);
    vfloat16m8_t t = __riscv_vfnmsub_vv_f16m8(d, r, one, vl); /* 1 - d * r */
    r = __riscv_vfmadd_vv_f16m8(r, t, r, vl); /* r + r * (1 - d * r) */

    /* 3. Calculate quotient */
    vfloat16m8_t q = __riscv_vfmul_vv_f16m8_mu(m, r, r, e, vl);

    /* 4. Optionally multiply by y */
    if (y) {
      const vfloat16m8_t vy = __riscv_vle16_v_f16m8(y + i, vl);
      q = __riscv_vfmul_vv_f16m8(q, vy, vl);
    }

    /* 5. Optionally multiply by (up + delta) */
    if (up) {
      vfloat16m8_t vu = __riscv_vle16_v_f16m8(up + i, vl);
      vu = __riscv_vfadd_vf_f16m8(vu, delta, vl);
      q = __riscv_vfmul_vv_f16m8(q, vu, vl);
    }

    /* 6. Store */
    __riscv_vse16_v_f16m8(out + i, q, vl);
  }
}

SKL_FUNC void skl_logistic_f16_xsfvfexp16e(_Float16 *out, const _Float16 *x,
                                           size_t n) {
  skl_sigmoid_f16_xsfvfexp16e(out, (_Float16)1, x, NULL, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_silu_f16_xsfvfexp16e(_Float16 *out, const _Float16 *x,
                                       size_t n) {
  skl_sigmoid_f16_xsfvfexp16e(out, (_Float16)1, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swish_f16_xsfvfexp16e(_Float16 *out, _Float16 beta,
                                        const _Float16 *x, size_t n) {
  skl_sigmoid_f16_xsfvfexp16e(out, beta, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_glu_f16_xsfvfexp16e(_Float16 *out, const _Float16 *x,
                                      const _Float16 *y, size_t n) {
  skl_sigmoid_f16_xsfvfexp16e(out, (_Float16)1, y, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swiglu_f16_xsfvfexp16e(_Float16 *out, const _Float16 *gate,
                                         const _Float16 *up, _Float16 delta,
                                         size_t n) {
  skl_sigmoid_f16_xsfvfexp16e(out, (_Float16)1, gate, gate, up, delta, n);
}
