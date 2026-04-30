// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfofp4min)
#error This file requires the Zvfofp4min extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_cvt_f4e2m1_f8e4m3_zvfofp4min(
    uint8_t *pDst, // NOLINT(readability-non-const-parameter)
    const uint8_t *pSrc, size_t n) {

  size_t vl_pairs = 0;
  size_t vl_n = 0;
  size_t pairs_remain = (n + 1) / 2;
  size_t n_remain = n;

  for (; pairs_remain > 0; pairs_remain -= vl_pairs, n_remain -= vl_n) {
    __asm__ volatile(
        "vsetvli %[vl_pairs], %[pairs_remain], e8, m4, ta, ma\n\t"
        "vle8.v v28, (%[src])\n\t"
        "add %[src], %[src], %[vl_pairs]\n\t"
        "slli    %[vl_n], %[vl_pairs], 1\n\t"
        "minu     %[vl_n], %[vl_n], %[n_remain]\n\t"
        "vsetvli %[vl_n], %[vl_n], e8, m8, ta, ma\n\t"
        "vfext.vf2 v16, v28\n\t"
        "vse8.v v16, (%[dst])\n\t"
        "add %[dst], %[dst], %[vl_n]\n\t"
        : [vl_n] "=&r"(vl_n), [vl_pairs] "=&r"(vl_pairs), [src] "+&r"(pSrc),
          [dst] "+r"(pDst)
        : [n_remain] "r"(n_remain), [pairs_remain] "r"(pairs_remain)
        : "v28", "v29", "v30", "v31", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "vtype", "vl", "memory");
  }
}
