// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

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

// first prototype: one te x n row

// kernel: store a0, load a1
// assume: tm1 == tn0
SKL_XSFMM_INOUT
SKL_FUNC void
skl_transpose_s1l1_e32_xsfmmbase(size_t tm0, size_t tn0, size_t tn1,
                                 size_t tss0, size_t tss1,
                                 uint32_t *SKL_RESTRICT a0t, size_t rsat,
                                 const uint32_t *SKL_RESTRICT a1, size_t rsa) {
  // assume dimensions are > 0
  const size_t kShiftCol = 24;
  const size_t kInc = 1;

  size_t tss0 |= (size_t)1 << kShiftCol;
  size_t tss0_end = tss0 + tn0 * kInc;

  __asm__ volatile(
      "sf.vsettnt x0, x0, e32, w1\n"

      "0:\n"
      "sf.vsettn x0, %[tm0]\n"
      "sf.vste32 %[tss0], (%[a0t])\n"
      "sf.vsettn x0, %[tn1]\n"
      "sf.vlte32 %[tss1], (%[a1])\n"
      "add %[a0t], %[a0t], %[rsat]\n"
      "add %[a1], %[a1], %[rsa]\n"
      "add %[tss0], %[tss0], %[kInc]\n"
      "add %[tss1], %[tss1], %[kInc]\n"
      "bltu %[tss0], %[tss0_end], 0b\n"
      : [tss0] "+&r"(tss0), [tss1] "+&r"(tss1), [a0t] "+&r"(a0t), [a1] "+&r"(a1)
      : [tm0] "r"(tm0), [tn1] "r"(tn1), [tss0_end] "r"(tss0_end),
        [rsa] "r"(rsa * sizeof(uint32_t)), [rsat] "r"(rsat * sizeof(uint32_t)),
        [kInc] "rI"(kInc)
      : "vtype", "vl", "memory");
}
