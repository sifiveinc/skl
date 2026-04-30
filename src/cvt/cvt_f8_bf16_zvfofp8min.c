// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_cvt_f8e4m3_bf16_zvfofp8min(
    __bf16 *pDst, // NOLINT(readability-non-const-parameter)
    const uint8_t *pSrc, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  for (; avl > 0; avl -= vl) {
    __asm__ volatile("vsetvli %[vl], %[avl], e8, m4, ta, ma\n\t"
                     "vle8.v	v28, (%[src])\n\t"
                     "add %[src],%[src],%[vl]\n\t"
                     "vfwcvtbf16.f.f.v v16, v28\n\t"
                     "vse16.v	v16, (%[dst])\n\t"
                     "sh1add %[dst],%[vl],%[dst]\n\t"
                     : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                     : [avl] "r"(avl)
                     : "v28", "v29", "v30", "v31", "v16", "v17", "v18", "v19",
                       "v20", "v21", "v22", "v23", "vtype", "vl", "memory");
  }
}

SKL_FUNC void skl_cvt_f8e5m2_bf16_zvfofp8min(
    __bf16 *pDst, // NOLINT(readability-non-const-parameter)
    const uint8_t *pSrc, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  for (; avl > 0; avl -= vl) {
    __asm__ volatile("vsetvli %[vl], %[avl], e8alt, m4, ta, ma\n\t"
                     "vle8.v	v28, (%[src])\n\t"
                     "add %[src],%[src],%[vl]\n\t"
                     "vfwcvtbf16.f.f.v v16, v28\n\t"
                     "vse16.v	v16, (%[dst])\n\t"
                     "sh1add %[dst],%[vl],%[dst]\n\t"
                     : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                     : [avl] "r"(avl)
                     : "v28", "v29", "v30", "v31", "v16", "v17", "v18", "v19",
                       "v20", "v21", "v22", "v23", "vtype", "vl", "memory");
  }
}
