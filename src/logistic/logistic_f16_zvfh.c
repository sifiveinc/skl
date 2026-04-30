// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

#include "skl-common.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

SKL_FUNC void skl_logistic_3u_f16_zvfh(_Float16 *out, const _Float16 *in,
                                       size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e16m8();
  const vfloat16m8_t one = __riscv_vfmv_v_f_f16m8(1, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    /* 0. Load */
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(in + i, vl);

    /* 1. Approximate exp(-|vx|) */
    vfloat16m8_t nx = __riscv_vfsgnj_vf_f16m8(vx, -0x1.158p4f16, vl);

    /* 2. Clamp inputs to lower bound */
    vbool2_t neg = __riscv_vmflt_vf_f16m8_b2(vx, 0, vl);
    __asm volatile("" ::"vr"(neg)); // Prevent reordering the above
    vbool2_t nn = __riscv_vmfeq_vv_f16m8_b2(nx, nx, vl); // Propagate NaN inputs
    nx = __riscv_vfmax_vf_f16m8_mu(nn, nx, nx, -0x1.158p4f16, vl);

    /* 3. Reduction */
    const _Float16 r_ln2 = 0x1.714p0f16; // round(1/log(2), HP, RN);
    const vfloat16m8_t v = __riscv_vfmul_vf_f16m8(nx, r_ln2, vl);
    const int8_t offset = -11;
    const vint8m4_t voffset = __riscv_vlse8_v_i8m4(&offset, 0, vl);
    const vint8m4_t q = __riscv_vfncvt_x_f_w_i8m4_mu(nn, voffset, v, vl);
    const vfloat16m8_t z = __riscv_vfwcvt_f_x_v_f16m8(q, vl);

    const _Float16 l2h = 0x1.63p-1f16;   // round(log(2), HP, RN);
    const _Float16 l2l = -0x1.bdp-13f16; // round(log(2) - ln2h, HP, RN);
    vfloat16m8_t s =
        __riscv_vfnmsac_vf_f16m8(nx, l2h, z, vl); // s = x - l2h * z
    s = __riscv_vfnmsub_vf_f16m8(z, l2l, s, vl);  // s = s - l2l * z

    /* 4. Approximation: e^s */
    const vfloat16m8_t c2 = __riscv_vfmv_v_f_f16m8(0x1.024p-1f16, vl);
    vfloat16m8_t u = __riscv_vfmacc_vf_f16m8(c2, 0x1.574p-3f16, s, vl);
    vfloat16m8_t c01 = __riscv_vfmv_v_f_f16m8(1, vl);
    u = __riscv_vfmadd_vv_f16m8(u, s, c01, vl);
    u = __riscv_vfmadd_vv_f16m8(u, s, c01, vl);

    /* 5. Reconstruction: compute e = u*2^(q+11) */
    vint16m8_t j = __riscv_vreinterpret_v_f16m8_i16m8(u);
    const int8_t exp_ = 11 << 3;
    vint8m4_t vexp = __riscv_vlse8_v_i8m4(&exp_, 0, vl);
    vexp = __riscv_vmadd_vx_i8m4(q, 1 << 3, vexp, vl);
    j = __riscv_vwmaccus_vx_i16m8(j, 1 << 7, vexp, vl);
    vfloat16m8_t e = __riscv_vreinterpret_v_i16m8_f16m8(j);

    /* 6. Re-scaling */
    vfloat16m8_t ex =
        __riscv_vfmul_vf_f16m8(e, 0x1p-11f16, vl); // UF happens here

    /* 7. Approximate 1 / (1 + exp(-|vx|)) */
    const vfloat16m8_t d = __riscv_vfadd_vf_f16m8(ex, 1, vl);
    vfloat16m8_t r = __riscv_vfrec7_v_f16m8(d, vl);
    vfloat16m8_t t = __riscv_vfnmsub_vv_f16m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f16m8(r, t, r, vl); // r + r * (1 - x * r)

    /* 8. Calculate quotient */
    vfloat16m8_t vy = __riscv_vfmul_vv_f16m8_mu(neg, r, r, ex, vl);

    /* 9. Store */
    __riscv_vse16_v_f16m8(out + i, vy, vl);
  }
}
