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

SKL_FUNC void skl_logistic_5u_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, const __bf16 *in,
                                                          size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t x = __riscv_vle16_v_bf16m8(in + i, vl);
    /* 0. Observe, Orient, & Squash */
    vbool2_t m = __riscv_vmflt_vf_bf16m8_b2(x, 0, vl);
    const __bf16 H = (__bf16)-0x1p-1;
    x = __riscv_vfsgnjn_vf_bf16m8(x, H, vl);
    x = __riscv_vfmul_vf_bf16m8(x, H, vl);
    /* 1. Exponentiate */
    vbfloat16m8_t f = __riscv_sf_vfexp_v_bf16m8(x, vl);
    vbfloat16m8_t e = __riscv_vfmul_vv_bf16m8(f, f, vl);
    /* 2. Reciprocate denominator */
    vbfloat16m8_t d = __riscv_vfadd_vf_bf16m8(e, (__bf16)1.0, vl);
    vbfloat16m8_t r = __riscv_vfrec7_v_bf16m8(d, vl);
    /* 3. Quotient & Store */
    vbfloat16m8_t q = __riscv_vfmul_vv_bf16m8_mu(m, r, r, e, vl);
    __riscv_vse16_v_bf16m8(out + i, q, vl);
  }
}
