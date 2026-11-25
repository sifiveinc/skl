// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfa)
#error This file requires the Xsfvfbfa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_sqrt_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m8_t h = __riscv_vfmv_v_f_f32m8(0.5f, vl);
    vbfloat16m4_t x = __riscv_vle16_v_bf16m4(in + i, vl);

    vbfloat16m4_t y = __riscv_vfrsqrt7_v_bf16m4(x, vl);
    vfloat32m8_t c = __riscv_vfwmul_vv_bf16m4_f32m8(x, y, vl);
    vfloat32m8_t u = __riscv_vfwmul_vf_bf16m4_f32m8(y, (__bf16)0.5, vl);

    vbool4_t m = __riscv_vmfeq_vv_f32m8_b4(c, c, vl);
    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(c, u, h, vl);
    c = __riscv_vfmadd_vv_f32m8(c, t, c, vl); /* c ~ sqrt(x) to ~13.4 bits */

    // Handle special values
    y = __riscv_vfrec7_v_bf16m4(y, vl);
    vbfloat16m4_t s = __riscv_vfncvt_f_f_w_bf16m4_mu(m, y, c, vl);

    // 4. Store
    __riscv_vse16_v_bf16m4(out + i, s, vl);
  }
}

SKL_FUNC void skl_sqrt_1u_bf16_xsfvfbfa(__bf16 *out, const __bf16 *in,
                                        size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vbfloat16m4_t x = __riscv_vle16_v_bf16m4(in + i, vl);
    vbfloat16m4_t y = __riscv_vfrsqrt7_v_bf16m4(x, vl);
    vbfloat16m4_t o = __riscv_vfmv_v_f_bf16m4((__bf16)1.0, vl);

    // Refinement, ~7.49 bits
    vbfloat16m4_t s = __riscv_vfmul_vv_bf16m4(x, y, vl);
    vbfloat16m4_t u = __riscv_vfmul_vf_bf16m4(s, (__bf16)0.5, vl);
    vbfloat16m4_t t = __riscv_vfnmsub_vv_bf16m4(y, s, o, vl); /* t = 1 - xy² */
    s = __riscv_vfmacc_vv_bf16m4(s, u, t, vl); /* xy + (0.5 xy) (1 - xy²) */

    // Handle special values
    vbool4_t m =
        __riscv_vmfne_vv_bf16m4_b4(u, u, vl); /* u NaN if x≤0, inf, NaN */
    s = __riscv_vfrec7_v_bf16m4_mu(m, s, y, vl);

    // 4. Store
    __riscv_vse16_v_bf16m4(out + i, s, vl);
  }
}
