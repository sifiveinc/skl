// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfexp16e) || !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfexp16e and Xsfvfbfa extensions
#endif

#include "skl-common.h"

#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, __bf16 beta,
                                                      const __bf16 *x,
                                                      const __bf16 *y,
                                                      const __bf16 *up,
                                                      __bf16 delta, size_t n) {
  if (!x)
    return;

  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(x + i, vl);
    if (beta != 1)
      vx = __riscv_vfmul_vf_bf16m8(vx, beta, vl);

    /* 0. Observe, Orient, & Squash */
    vbool2_t m = __riscv_vmflt_vf_bf16m8_b2(vx, 0, vl);
    const __bf16 H = (__bf16)-0x1p-1;
    vx = __riscv_vfsgnjn_vf_bf16m8(vx, H, vl);
    vx = __riscv_vfmul_vf_bf16m8(vx, H, vl);
    /* 1. Exponentiate */
    vbfloat16m8_t f = __riscv_sf_vfexp_v_bf16m8(vx, vl);
    vbfloat16m8_t e = __riscv_vfmul_vv_bf16m8(f, f, vl);
    /* 2. Reciprocate denominator */
    vbfloat16m8_t d = __riscv_vfadd_vf_bf16m8(e, (__bf16)1.0, vl);
    vbfloat16m8_t r = __riscv_vfrec7_v_bf16m8(d, vl);
    /* 3. Quotient */
    vbfloat16m8_t q = __riscv_vfmul_vv_bf16m8_mu(m, r, r, e, vl);
    /* 4. Optionally multiply by y */
    if (y) {
      vbfloat16m8_t vy = __riscv_vle16_v_bf16m8(y + i, vl);
      q = __riscv_vfmul_vv_bf16m8(q, vy, vl);
    }
    /* 5. Optionally multiply by (up + delta) */
    if (up) {
      vbfloat16m8_t vu = __riscv_vle16_v_bf16m8(up + i, vl);
      vu = __riscv_vfadd_vf_bf16m8(vu, delta, vl);
      q = __riscv_vfmul_vv_bf16m8(q, vu, vl);
    }
    /* 6. Store */
    __riscv_vse16_v_bf16m8(out + i, q, vl);
  }
}

SKL_FUNC void skl_logistic_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out,
                                                       const __bf16 *x,
                                                       size_t n) {
  skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(out, (__bf16)1, x, NULL, NULL,
                                          (__bf16)0, n);
}

SKL_FUNC void skl_silu_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, const __bf16 *x,
                                                   size_t n) {
  skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(out, (__bf16)1, x, x, NULL, (__bf16)0,
                                          n);
}

SKL_FUNC void skl_swish_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, __bf16 beta,
                                                    const __bf16 *x, size_t n) {
  skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(out, beta, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_glu_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, const __bf16 *x,
                                                  const __bf16 *y, size_t n) {
  skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(out, (__bf16)1, y, x, NULL, (__bf16)0,
                                          n);
}

SKL_FUNC void skl_swiglu_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out,
                                                     const __bf16 *gate,
                                                     const __bf16 *up,
                                                     __bf16 delta, size_t n) {
  skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(out, (__bf16)1, gate, gate, up, delta,
                                          n);
}
