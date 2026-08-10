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
#include <stdint.h>

SKL_FUNC_PRIVATE vfloat32m8_t
skl_sigmoid_bf16_xsfvfexp32e_vfwcvt_f_x_v_f32m8(vint16m4_t i, size_t vl) {
#if defined(__riscv_zvfbfmin)
  vbfloat16m4_t b = __riscv_vreinterpret_v_i16m4_bf16m4(i);
  return __riscv_vfwcvtbf16_f_f_v_f32m8(b, vl);
#elif defined(__riscv_zvbb)
  vuint16m4_t u = __riscv_vreinterpret_v_i16m4_u16m4(i);
  vuint32m8_t U = __riscv_vwsll_vx_u32m8(u, 16, vl);
  return __riscv_vreinterpret_v_u32m8_f32m8(U);
#else
  vint32m8_t I = __riscv_vsll_vx_i32m8(__riscv_vsext_vf2_i32m8(i, vl), 16, vl);
  return __riscv_vreinterpret_v_i32m8_f32m8(I);
#endif
}

SKL_FUNC_PRIVATE vint16m4_t
skl_sigmoid_bf16_xsfvfexp32e_vfncvt_x_f_w_bf16m4(vfloat32m8_t X, size_t vl) {
#if defined(__riscv_zvfbfmin)
  vbfloat16m4_t b = __riscv_vfncvtbf16_f_f_w_bf16m4(X, vl);
  return __riscv_vreinterpret_v_bf16m4_i16m4(b);
#else
  vint32m8_t I = __riscv_vreinterpret_v_f32m8_i32m8(X);
  vint16m4_t i = __riscv_vnclip_wx_i16m4(I, 16, __RISCV_VXRM_RNE, vl);
  return i;
#endif
}

SKL_FUNC void skl_sigmoid_bf16_xsfvfexp32e(__bf16 *out, __bf16 beta,
                                           const __bf16 *x, const __bf16 *y,
                                           const __bf16 *up, __bf16 delta,
                                           size_t n) {
  if (!x)
    return;

  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vint16m4_t vx = __riscv_vle16_v_i16m4((int16_t *)x + i, vl);
    vfloat32m8_t a;
    vbool4_t m;
    /* 0. Observe, Orient, Convert, & Squash */
    if (beta != 1) {
      a = skl_sigmoid_bf16_xsfvfexp32e_vfwcvt_f_x_v_f32m8(vx, vl);
      a = __riscv_vfmul_vf_f32m8(a, (float)beta * 0.5f, vl);
      m = __riscv_vmflt_vf_f32m8_b4(a, 0, vl);
      a = __riscv_vfsgnj_vf_f32m8(a, -0x1.74p6f, vl);
      a = __riscv_vfmax_vf_f32m8(a, -0x1.74p6f, vl);
    } else {
      m = __riscv_vmslt_vx_i16m4_b4(vx, 0, vl);
      vx = __riscv_vor_vx_i16m4(vx, (int16_t)0x8000, vl); /* copysign(x,-1) */
      a = skl_sigmoid_bf16_xsfvfexp32e_vfwcvt_f_x_v_f32m8(vx, vl);
      a = __riscv_vfmul_vf_f32m8(a, 0.5f, vl);
    }
    /* 2. Exponentiate */
    vfloat32m8_t f = __riscv_sf_vfexp_v_f32m8(a, vl);
    vfloat32m8_t e = __riscv_vfmul_vv_f32m8(f, f, vl);
    /* 3. Reciprocate denominator */
    vfloat32m8_t d = __riscv_vfadd_vf_f32m8(e, 1.0f, vl);
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);
    /* 4. Quotient */
    vfloat32m8_t q = __riscv_vfmul_vv_f32m8_mu(m, r, r, e, vl);
    /* 5. Optionally multiply by y */
    if (y) {
      vint16m4_t yi = __riscv_vle16_v_i16m4((int16_t *)y + i, vl);
      vfloat32m8_t yf = skl_sigmoid_bf16_xsfvfexp32e_vfwcvt_f_x_v_f32m8(yi, vl);
      q = __riscv_vfmul_vv_f32m8(q, yf, vl);
    }
    /* 6. Optionally multiply by (up + delta) */
    if (up) {
      vint16m4_t ui = __riscv_vle16_v_i16m4((int16_t *)up + i, vl);
      vfloat32m8_t uf = skl_sigmoid_bf16_xsfvfexp32e_vfwcvt_f_x_v_f32m8(ui, vl);
      uf = __riscv_vfadd_vf_f32m8(uf, (float)delta, vl);
      q = __riscv_vfmul_vv_f32m8(q, uf, vl);
    }
    /* 7. Narrow & Store */
    vint16m4_t b = skl_sigmoid_bf16_xsfvfexp32e_vfncvt_x_f_w_bf16m4(q, vl);
    __riscv_vse16_v_i16m4((int16_t *)out + i, b, vl);
  }
}

SKL_FUNC void skl_logistic_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *x,
                                            size_t n) {
  skl_sigmoid_bf16_xsfvfexp32e(out, (__bf16)1, x, NULL, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_silu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *x,
                                        size_t n) {
  skl_sigmoid_bf16_xsfvfexp32e(out, (__bf16)1, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_swish_bf16_xsfvfexp32e(__bf16 *out, __bf16 beta,
                                         const __bf16 *x, size_t n) {
  skl_sigmoid_bf16_xsfvfexp32e(out, beta, x, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_glu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *x,
                                       const __bf16 *y, size_t n) {
  skl_sigmoid_bf16_xsfvfexp32e(out, (__bf16)1, y, x, NULL, (__bf16)0, n);
}

SKL_FUNC void skl_swiglu_bf16_xsfvfexp32e(__bf16 *out, const __bf16 *gate,
                                          const __bf16 *up, __bf16 delta,
                                          size_t n) {
  skl_sigmoid_bf16_xsfvfexp32e(out, (__bf16)1, gate, gate, up, delta, n);
}
