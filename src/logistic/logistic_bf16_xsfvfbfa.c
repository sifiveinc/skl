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

SKL_FUNC void skl_logistic_3u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in,
                                            size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t x = __riscv_vle16_v_bf16m8(in + i, vl);
    /* 0. Observe, Orient, & Clamp */
    vbool2_t m = __riscv_vmflt_vf_bf16m8_b2(x, 0, vl);
    const __bf16 B = (__bf16)-0x1.74p6;
    x = __riscv_vfsgnj_vf_bf16m8(x, B, vl);
    x = __riscv_vfmax_vf_bf16m8(x, B, vl);
    /* 1. Reduce x ~ z ln2 + s */
    const __bf16 R = (__bf16)-0x1.72p0f;  /* -1/ln2 */
    const __bf16 C1 = (__bf16)0x1.60p-1f; /* ln2 */
    const __bf16 C2 = (__bf16)0x1.72p-8f; /* ln2 - C1 */
    vbfloat16m8_t v = __riscv_vfmul_vf_bf16m8(x, R, vl);
    vuint8m4_t k = __riscv_vfncvt_xu_f_w_bf16m8_u8m4(v, vl);
    vbfloat16m8_t z = __riscv_vfwcvt_f_xu_v_bf16m8(k, vl);
    vbfloat16m8_t s = __riscv_vfmacc_vf_bf16m8(x, C1, z, vl);
    s = __riscv_vfmacc_vf_bf16m8(s, C2, z, vl);
    /* 2. Approximate exp(s)/2⁸ */
    vbfloat16m8_t c = __riscv_vfmv_v_f_bf16m8((__bf16)0x1.04p-9, vl);
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
    /* 5. Reconstruct & Store */
    vbfloat16m8_t n = __riscv_vmerge_vvm_bf16m8(u, f, m, vl);
    vbfloat16m8_t q = __riscv_vfmul_vv_bf16m8(n, r, vl);
    __riscv_vse16_v_bf16m8(out + i, q, vl);
  }
}
