// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_XSFMM_NEW
SKL_FUNC void skl_transpose_e32_xsfmmbase(size_t m, size_t n,
                                          const uint32_t *SKL_RESTRICT a,
                                          size_t rsa, uint32_t *SKL_RESTRICT at,
                                          size_t rsat) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t kShiftCol = 24;
  const size_t kInc = 1;
  size_t tm = 0;
  size_t tn = 0;
  for (size_t i = 0; i < m; i += tm) {
    for (size_t j = 0; j < n; j += tn) {
      size_t mt0_load = 0;
      size_t mt0_store = (size_t)(1) << kShiftCol;

      const uint32_t *SKL_RESTRICT a_load = a + i * rsa + j;
      uint32_t *SKL_RESTRICT at_store = at + j * rsat + i;

      size_t k = 0;
      __asm__ volatile(
          "sf.vsettnt %[tn], %[avl_n], e32, w1\n"
          "sf.vsettm %[tm], %[avl_m]\n"

          "0:\n"
          "sf.vlte32 %[mt0_load], (%[a])\n"
          "add %[a], %[a], %[sa]\n"
          "add %[mt0_load], %[mt0_load], %[kInc]\n"
          "addi %[k], %[k], 1\n"
          "bltu %[k], %[tm], 0b\n"

          "li %[k], 0\n"
          "sf.vsettn x0, %[tm]\n"
          "1:\n"
          "sf.vste32 %[mt0_store], (%[at])\n"
          "add %[at], %[at], %[sat]\n"
          "add %[mt0_store], %[mt0_store], %[kInc]\n"
          "addi %[k], %[k], 1\n"
          "bltu %[k], %[tn], 1b\n"
          : [tm] "=&r"(tm), [tn] "=&r"(tn), [mt0_load] "+&r"(mt0_load),
            [mt0_store] "+&r"(mt0_store), [a] "+&r"(a_load),
            [at] "+&r"(at_store), [k] "+&r"(k)
          : [avl_m] "r"(m - i), [avl_n] "r"(n - j),
            [sa] "r"(rsa * sizeof(uint32_t)),
            [sat] "r"(rsat * sizeof(uint32_t)), [kInc] "rI"(kInc)
          : "vtype", "vl", "memory");
    }
  }
  __asm__ volatile("sf.vtdiscard");
}
