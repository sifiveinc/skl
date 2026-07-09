// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sigmoid_bf16_xsfvfbfa(__bf16 *out, __bf16 beta,
                                        const __bf16 *x, const __bf16 *y,
                                        const __bf16 *up, __bf16 delta,
                                        size_t n) {
  if (!x)
    return;

  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(x + i, vl);
    if (beta != 1)
      vx = __riscv_vfmul_vf_bf16m8(vx, beta, vl);

    /* 0. Observe, Orient, & Clamp */
    vbool2_t m = __riscv_vmflt_vf_bf16m8_b2(vx, 0, vl);
    const __bf16 B = (__bf16)-0x1.74p6;
    vx = __riscv_vfsgnj_vf_bf16m8(vx, B, vl);
    vx = __riscv_vfmax_vf_bf16m8(vx, B, vl);
    /* 1. Reduce x ~ z ln2 + s */
    const __bf16 R = (__bf16)-0x1.72p0f;  /* -1/ln2 */
    const __bf16 C1 = (__bf16)0x1.60p-1f; /* ln2 */
    const __bf16 C2 = (__bf16)0x1.72p-8f; /* ln2 - C1 */
    vbfloat16m8_t v = __riscv_vfmul_vf_bf16m8(vx, R, vl);
    vuint8m4_t k = __riscv_vfncvt_xu_f_w_bf16m8_u8m4(v, vl);
    vbfloat16m8_t z = __riscv_vfwcvt_f_xu_v_bf16m8(k, vl);
    vbfloat16m8_t s = __riscv_vfmacc_vf_bf16m8(vx, C1, z, vl);
    s = __riscv_vfmacc_vf_bf16m8(s, C2, z, vl);
    /* 2. Approximate exp(s)/2⁸ */
    vbfloat16m8_t c = __riscv_vfmv_v_f_bf16m8((__bf16)0x1.0ap-9, vl);
    vbfloat16m8_t u = __riscv_vfmacc_vf_bf16m8(c, (__bf16)0x1.5cp-11, s, vl);
    c = __riscv_vfmv_v_f_bf16m8((__bf16)0x1p-8, vl);
    u = __riscv_vfmadd_vv_bf16m8(u, s, c, vl);
    u = __riscv_vfmadd_vv_bf16m8(u, s, c, vl);
    /* 3. Assemble scaling factor */
    vuint16m8_t t = __riscv_vreinterpret_v_bf16m8_u16m8(c);
    t = __riscv_vwmaccu_vx_u16m8(t, 1 << 7, k, vl);
    vbfloat16m8_t f = __riscv_vreinterpret_v_u16m8_bf16m8(t);
    /* 4. Reciprocate denominator */
    vbfloat16m8_t d = __riscv_vfadd_vv_bf16m8(f, u, vl);
    vbfloat16m8_t r = __riscv_vfrec7_v_bf16m8(d, vl);
    /* 5. Reconstruct */
    vbfloat16m8_t nn = __riscv_vmerge_vvm_bf16m8(f, u, m, vl);
    vbfloat16m8_t q = __riscv_vfmul_vv_bf16m8(nn, r, vl);
    /* 6. Optionally multiply by y */
    if (y) {
      vbfloat16m8_t yv = __riscv_vle16_v_bf16m8(y + i, vl);
      q = __riscv_vfmul_vv_bf16m8(q, yv, vl);
    }
    /* 7. Optionally multiply by (up + delta) */
    if (up) {
      vbfloat16m8_t uv = __riscv_vle16_v_bf16m8(up + i, vl);
      uv = __riscv_vfadd_vf_bf16m8(uv, delta, vl);
      q = __riscv_vfmul_vv_bf16m8(q, uv, vl);
    }
    /* 8. Store */
    __riscv_vse16_v_bf16m8(out + i, q, vl);
  }
}

SKL_FUNC void skl_logistic_bf16_xsfvfbfa(__bf16 *out, const __bf16 *x,
                                         size_t n) {
  skl_sigmoid_bf16_xsfvfbfa(out, (__bf16)1, x, NULL, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_silu_bf16_xsfvfbfa(__bf16 *out, const __bf16 *x, size_t n) {
  skl_sigmoid_bf16_xsfvfbfa(out, (__bf16)1, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_swish_bf16_xsfvfbfa(__bf16 *out, __bf16 beta, const __bf16 *x,
                                      size_t n) {
  skl_sigmoid_bf16_xsfvfbfa(out, beta, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_glu_bf16_xsfvfbfa(__bf16 *out, const __bf16 *x,
                                    const __bf16 *y, size_t n) {
  skl_sigmoid_bf16_xsfvfbfa(out, (__bf16)1, y, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_swiglu_bf16_xsfvfbfa(__bf16 *out, const __bf16 *gate,
                                       const __bf16 *up, __bf16 delta,
                                       size_t n) {
  skl_sigmoid_bf16_xsfvfbfa(out, (__bf16)1, gate, gate, up, delta, n);
}
