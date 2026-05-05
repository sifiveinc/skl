// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

#if !defined(__riscv_zvfbfmin)
#error This file requires the Zvfbfmin extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_cvt_bf16_f8e4m3_zvfofp8min_zvfbfmin(
    uint8_t *pDst, // NOLINT(readability-non-const-parameter)
    const __bf16 *pSrc, float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m8, ta, ma\n\t"
                       "vle16.v v24, (%[src])\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8, m4, ta, ma\n\t"
                       "vfncvtbf16.f.f.w	v16, v24\n\t"
                       "vse8.v	v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "vtype", "vl", "memory");
    }

  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m4, ta, ma\n\t"
                       "vle16.v v28, (%[src])\n\t"
                       "vfwcvtbf16.f.f.v v16, v28\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e32, m8, ta, ma\n\t"
                       "vfmul.vf	v24, v16, %[scale]\n\t"
                       "vsetvli x0, x0, e8, m2, ta, ma\n\t"
                       "vfncvt.f.f.q v12, v24\n\t"
                       "vse8.v	v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_cvt_sat_bf16_f8e4m3_zvfofp8min_zvfbfmin(
    uint8_t *pDst, // NOLINT(readability-non-const-parameter)
    const __bf16 *pSrc, float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m8, ta, ma\n\t"
                       "vle16.v v24, (%[src])\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8, m4, ta, ma\n\t"
                       "vfncvtbf16.sat.f.f.w	v16, v24\n\t"
                       "vse8.v	v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "vtype", "vl", "memory");
    }

  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m4, ta, ma\n\t"
                       "vle16.v v28, (%[src])\n\t"
                       "vfwcvtbf16.f.f.v v16, v28\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e32, m8, ta, ma\n\t"
                       "vfmul.vf	v24, v16, %[scale]\n\t"
                       "vsetvli x0, x0, e8, m2, ta, ma\n\t"
                       "vfncvt.sat.f.f.q v12, v24\n\t"
                       "vse8.v	v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_cvt_bf16_f8e5m2_zvfofp8min_zvfbfmin(
    uint8_t *pDst, // NOLINT(readability-non-const-parameter)
    const __bf16 *pSrc, float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m8, ta, ma\n\t"
                       "vle16.v v24, (%[src])\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8alt, m4, ta, ma\n\t"
                       "vfncvtbf16.f.f.w	v16, v24\n\t"
                       "vse8.v	v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "vtype", "vl", "memory");
    }

  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m4, ta, ma\n\t"
                       "vle16.v v28, (%[src])\n\t"
                       "vfwcvtbf16.f.f.v v16, v28\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e32, m8, ta, ma\n\t"
                       "vfmul.vf	v24, v16, %[scale]\n\t"
                       "vsetvli x0, x0, e8alt, m2, ta, ma\n\t"
                       "vfncvt.f.f.q v12, v24\n\t"
                       "vse8.v	v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_cvt_sat_bf16_f8e5m2_zvfofp8min_zvfbfmin(
    uint8_t *pDst, // NOLINT(readability-non-const-parameter)
    const __bf16 *pSrc, float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m8, ta, ma\n\t"
                       "vle16.v v24, (%[src])\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8alt, m4, ta, ma\n\t"
                       "vfncvtbf16.sat.f.f.w	v16, v24\n\t"
                       "vse8.v	v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "vtype", "vl", "memory");
    }

  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e16, m4, ta, ma\n\t"
                       "vle16.v v28, (%[src])\n\t"
                       "vfwcvtbf16.f.f.v v16, v28\n\t"
                       "sh1add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e32, m8, ta, ma\n\t"
                       "vfmul.vf	v24, v16, %[scale]\n\t"
                       "vsetvli x0, x0, e8alt, m2, ta, ma\n\t"
                       "vfncvt.sat.f.f.q v12, v24\n\t"
                       "vse8.v	v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}
