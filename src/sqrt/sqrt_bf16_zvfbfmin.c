// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfbfmin)
#error This file requires the Zvfbfmin extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sqrt_bf16_zvfbfmin(__bf16 *out, const __bf16 *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t b = __riscv_vle16_v_bf16m4(in + i, vl);
    vfloat32m8_t x = __riscv_vfwcvtbf16_f_f_v_f32m8(b, vl);
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
    b = __riscv_vfncvtbf16_f_f_w_bf16m4(s, vl);
    __riscv_vse16_v_bf16m4(out + i, b, vl);
  }
}

SKL_FUNC void skl_sqrt_2u_bf16_zvfbfmin(__bf16 *out, const __bf16 *in,
                                        size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m4(n - i);
    vbfloat16m4_t b = __riscv_vle16_v_bf16m4(in + i, vl);
    vfloat32m8_t x = __riscv_vfwcvtbf16_f_f_v_f32m8(b, vl);
    vfloat32m8_t y = __riscv_vfrsqrt7_v_f32m8(x, vl);
    vfloat32m8_t s = __riscv_vfmul_vv_f32m8(x, y, vl);

    // Handle special values
    vbool4_t m = __riscv_vmfne_vv_f32m8_b4(s, s, vl);
    s = __riscv_vfrec7_v_f32m8_mu(m, s, y, vl);

    // Narrow and Store
    b = __riscv_vfncvtbf16_f_f_w_bf16m4(s, vl);
    __riscv_vse16_v_bf16m4(out + i, b, vl);
  }
}
