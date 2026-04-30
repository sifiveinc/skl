// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfbfexp16e)
#error This file requires the Xsfvfbfexp16e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u0alt64ainf_bf16_xsfvfbfexp16e(__bf16 *out,
                                                      const __bf16 *in,
                                                      size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    vbfloat16m8_t vx = __riscv_vle16_v_bf16m8(in + i, vl);
    vbfloat16m8_t vy = __riscv_sf_vfexp_v_bf16m8(vx, vl);
    __riscv_vse16_v_bf16m8(out + i, vy, vl);
  }
}
