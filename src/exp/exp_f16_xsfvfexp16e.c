// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp16e)
#error This file requires the Xsfvfexp16e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1p022u0alt8ainf_f16_xsfvfexp16e(_Float16 *out,
                                                      const _Float16 *in,
                                                      size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(in + i, vl);
    vfloat16m8_t vy = __riscv_sf_vfexp_v_f16m8(vx, vl);
    __riscv_vse16_v_f16m8(out + i, vy, vl);
  }
}

SKL_FUNC void skl_exp_3p16u_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in,
                                            size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(in + i, vl);
    vfloat16m8_t a = __riscv_vfmul_vf_f16m8(vx, 0.5f16, vl);
    vfloat16m8_t f = __riscv_sf_vfexp_v_f16m8(a, vl);
    vfloat16m8_t vy = __riscv_vfmul_vv_f16m8(f, f, vl);
    __riscv_vse16_v_f16m8(out + i, vy, vl);
  }
}
