// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
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

SKL_FUNC void skl_exp_2p398u0alt64ainf_f32_xsfvfexp32e(float *out,
                                                       const float *in,
                                                       size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);
    vfloat32m8_t vy = __riscv_sf_vfexp_v_f32m8(vx, vl);
    __riscv_vse32_v_f32m8(out + i, vy, vl);
  }
}

SKL_FUNC void skl_exp_5p32u_f32_xsfvfexp32e(float *out, const float *in,
                                            size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);
    const float S = 0x1.3b28dcp-1f;
    vfloat32m8_t a = __riscv_vfmul_vf_f32m8(vx, S, vl);
    vfloat32m8_t e = __riscv_vfsub_vv_f32m8(vx, a, vl);
    vfloat32m8_t f = __riscv_sf_vfexp_v_f32m8(a, vl);
    vfloat32m8_t g = __riscv_sf_vfexp_v_f32m8(e, vl);
    vbool4_t m = __riscv_vmfeq_vv_f32m8_b4(e, e, vl);
    vfloat32m8_t vy = __riscv_vfmul_vv_f32m8_mu(m, f, f, g, vl);
    __riscv_vse32_v_f32m8(out + i, vy, vl);
  }
}
