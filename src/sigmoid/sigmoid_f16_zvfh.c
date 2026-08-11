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
#include <stdint.h>

SKL_FUNC void skl_sigmoid_f16_zvfh(_Float16 *out, _Float16 beta,
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
    /* 0. Load and scale by beta */
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(x + i, vl);
    if (beta != 1)
      vx = __riscv_vfmul_vf_f16m8(vx, beta, vl);

    /* 1. Observe, Orient, & Clamp */
    const vbool2_t m = __riscv_vmflt_vf_f16m8_b2(vx, 0, vl);
    vx = __riscv_vfsgnj_vf_f16m8(vx, -0x1.158p4f16, vl);
    __asm volatile("" ::"vr"(m)); /* Prevent reordering the above */
    vx = __riscv_vfmax_vf_f16m8(vx, -0x1.158p4f16, vl);

    /* 2. Reduce x ~ z ln2 + s */
    const _Float16 R = 0x1.714p0f16;    /* 1/ln2 */
    const _Float16 C1 = 0x1.63p-1f16;   /* ln2 */
    const _Float16 C2 = -0x1.bdp-13f16; /* ln2 - C1 */
    const vfloat16m8_t v = __riscv_vfmul_vf_f16m8(vx, R, vl);
    const vint8m4_t zi = __riscv_vfncvt_x_f_w_i8m4(v, vl);
    const vfloat16m8_t z = __riscv_vfwcvt_f_x_v_f16m8(zi, vl);
    vfloat16m8_t s =
        __riscv_vfnmsac_vf_f16m8(vx, C1, z, vl); /* s = x - C1 * z */
    s = __riscv_vfnmsub_vf_f16m8(z, C2, s, vl);  /* s = s - C2 * z */

    /* 3. Approximate */
    const vfloat16m8_t c2 = __riscv_vfmv_v_f_f16m8(0x1.024p-1f16, vl);
    vfloat16m8_t u = __riscv_vfmacc_vf_f16m8(c2, 0x1.574p-3f16, s, vl);
    vfloat16m8_t c01 = __riscv_vfmv_v_f_f16m8(1, vl);
    u = __riscv_vfmadd_vv_f16m8(u, s, c01, vl);
    u = __riscv_vfmadd_vv_f16m8(u, s, c01, vl);

    /* 4. Reconstruct e ~ u*2^(z+11) */
    vint16m8_t j = __riscv_vreinterpret_v_f16m8_i16m8(u);
    const int8_t exp_ = 11 << 3;
    vint8m4_t vexp = __riscv_vlse8_v_i8m4(&exp_, 0, vl);
    vexp = __riscv_vmadd_vx_i8m4(zi, 1 << 3, vexp, vl);
    j = __riscv_vwmaccus_vx_i16m8(j, 1 << 7, vexp, vl);
    vfloat16m8_t e = __riscv_vreinterpret_v_i16m8_f16m8(j);

    /* 5. Re-scale */
    vfloat16m8_t ex =
        __riscv_vfmul_vf_f16m8(e, 0x1p-11f16, vl); /* UF happens here */

    /* 6. Reciprocate denominator */
    const vfloat16m8_t d = __riscv_vfadd_vf_f16m8(ex, 1, vl);
    vfloat16m8_t r = __riscv_vfrec7_v_f16m8(d, vl);
    vfloat16m8_t t = __riscv_vfnmsub_vv_f16m8(d, r, one, vl); /* 1 - x * r */
    r = __riscv_vfmadd_vv_f16m8(r, t, r, vl); /* r + r * (1 - x * r) */

    /* 7. Calculate quotient */
    vfloat16m8_t q = __riscv_vfmul_vv_f16m8_mu(m, r, r, ex, vl);

    /* 8. Optionally multiply by y */
    if (y) {
      const vfloat16m8_t vy = __riscv_vle16_v_f16m8(y + i, vl);
      q = __riscv_vfmul_vv_f16m8(q, vy, vl);
    }

    /* 9. Optionally multiply by (up + delta) */
    if (up) {
      vfloat16m8_t vu = __riscv_vle16_v_f16m8(up + i, vl);
      vu = __riscv_vfadd_vf_f16m8(vu, delta, vl);
      q = __riscv_vfmul_vv_f16m8(q, vu, vl);
    }

    /* 10. Store */
    __riscv_vse16_v_f16m8(out + i, q, vl);
  }
}

SKL_FUNC void skl_logistic_f16_zvfh(_Float16 *out, const _Float16 *x,
                                    size_t n) {
  skl_sigmoid_f16_zvfh(out, (_Float16)1, x, NULL, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_silu_f16_zvfh(_Float16 *out, const _Float16 *x, size_t n) {
  skl_sigmoid_f16_zvfh(out, (_Float16)1, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swish_f16_zvfh(_Float16 *out, _Float16 beta,
                                 const _Float16 *x, size_t n) {
  skl_sigmoid_f16_zvfh(out, beta, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_glu_f16_zvfh(_Float16 *out, const _Float16 *x,
                               const _Float16 *y, size_t n) {
  skl_sigmoid_f16_zvfh(out, (_Float16)1, y, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swiglu_f16_zvfh(_Float16 *out, const _Float16 *gate,
                                  const _Float16 *up, _Float16 delta,
                                  size_t n) {
  skl_sigmoid_f16_zvfh(out, (_Float16)1, gate, gate, up, delta, n);
}
