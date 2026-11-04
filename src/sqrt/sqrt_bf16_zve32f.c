// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

SKL_FUNC_PRIVATE vfloat32m8_t skl_vfwcvt_f_x_v_f32m8(vint16m4_t i, size_t vl) {
#if defined(__riscv_zvbb)
  vuint16m4_t u = __riscv_vreinterpret_v_i16m4_u16m4(i);
  vuint32m8_t U = __riscv_vwsll_vx_u32m8(u, 16, vl);
  return __riscv_vreinterpret_v_u32m8_f32m8(U);
#else
  vint32m8_t I = __riscv_vsll_vx_i32m8(__riscv_vsext_vf2_i32m8(i, vl), 16, vl);
  return __riscv_vreinterpret_v_i32m8_f32m8(I);
#endif
}

SKL_FUNC_PRIVATE vint16m4_t skl_vfncvt_x_f_w_bf16m4(vfloat32m8_t X, size_t vl) {
  vint32m8_t I = __riscv_vreinterpret_v_f32m8_i32m8(X);
  vint16m4_t i = __riscv_vnclip_wx_i16m4(I, 16, __RISCV_VXRM_RNE, vl);
  return i;
}

SKL_FUNC void skl_sqrt_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vint16m4_t b = __riscv_vle16_v_i16m4((int16_t *)(in + i), vl);
    vfloat32m8_t x = skl_vfwcvt_f_x_v_f32m8(b, vl);
    vfloat32m8_t y = __riscv_vfrsqrt7_v_f32m8(x, vl);

    // Single refinement
    vfloat32m8_t c = __riscv_vfmul_vv_f32m8(x, y, vl);
    vfloat32m8_t w = __riscv_vfmul_vf_f32m8(y, 0.5f, vl);
    vfloat32m8_t h = __riscv_vfmv_v_f_f32m8(0.5f, vl);
    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(c, w, h, vl);
    vfloat32m8_t s = __riscv_vfmadd_vv_f32m8(c, t, c, vl);

    // Handle special values
    vbool4_t m = __riscv_vmfne_vv_f32m8_b4(t, t, vl);
    s = __riscv_vfrec7_v_f32m8_mu(m, s, y, vl);

    // Narrow and Store
    b = skl_vfncvt_x_f_w_bf16m4(s, vl);
    __riscv_vse16_v_i16m4((int16_t *)(out + i), b, vl);
  }
}

SKL_FUNC void skl_sqrt_2u_bf16_zve32f(__bf16 *out, const __bf16 *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vint16m4_t b = __riscv_vle16_v_i16m4((int16_t *)(in + i), vl);
    vfloat32m8_t x = skl_vfwcvt_f_x_v_f32m8(b, vl);
    vfloat32m8_t y = __riscv_vfrsqrt7_v_f32m8(x, vl);
    vfloat32m8_t s = __riscv_vfmul_vv_f32m8(x, y, vl);

    // Handle special values
    vbool4_t m = __riscv_vmfne_vv_f32m8_b4(s, s, vl);
    s = __riscv_vfrec7_v_f32m8_mu(m, s, y, vl);

    // Narrow and Store
    b = skl_vfncvt_x_f_w_bf16m4(s, vl);
    __riscv_vse16_v_i16m4((int16_t *)(out + i), b, vl);
  }
}
