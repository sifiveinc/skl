// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_zvfofp8min)
#error This file requires the Zvfofp8min extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_cvt_f32_f8e4m3_zvfofp8min(
    uint8_t *pDst, const float *pSrc, // NOLINT(readability-non-const-parameter)
    float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8, m2, ta, ma\n\t"
                       "vfncvt.f.f.q v16, v24\n\t"
                       "vse8.v v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "vtype", "vl", "memory");
    }
  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "vfmul.vf v16, v24, %[scale]\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8, m2, ta, ma\n\t"
                       "vfncvt.f.f.q v12, v16\n\t"
                       "vse8.v v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_cvt_sat_f32_f8e4m3_zvfofp8min(
    uint8_t *pDst, const float *pSrc, // NOLINT(readability-non-const-parameter)
    float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8, m2, ta, ma\n\t"
                       "vfncvt.sat.f.f.q v16, v24\n\t"
                       "vse8.v v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "vtype", "vl", "memory");
    }
  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "vfmul.vf v16, v24, %[scale]\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8, m2, ta, ma\n\t"
                       "vfncvt.sat.f.f.q v12, v16\n\t"
                       "vse8.v v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_cvt_f32_f8e5m2_zvfofp8min(
    uint8_t *pDst, const float *pSrc, // NOLINT(readability-non-const-parameter)
    float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8alt, m2, ta, ma\n\t"
                       "vfncvt.f.f.q v16, v24\n\t"
                       "vse8.v v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "vtype", "vl", "memory");
    }
  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "vfmul.vf v16, v24, %[scale]\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8alt, m2, ta, ma\n\t"
                       "vfncvt.f.f.q v12, v16\n\t"
                       "vse8.v v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}

SKL_FUNC void skl_cvt_sat_f32_f8e5m2_zvfofp8min(
    uint8_t *pDst, const float *pSrc, // NOLINT(readability-non-const-parameter)
    float scaling_factor, size_t n) {
  size_t vl = 0;
  size_t avl = n;

  if (scaling_factor == 1.0f) {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8alt, m2, ta, ma\n\t"
                       "vfncvt.sat.f.f.q v16, v24\n\t"
                       "vse8.v v16, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "vtype", "vl", "memory");
    }
  } else {
    for (; avl > 0; avl -= vl) {
      __asm__ volatile("vsetvli %[vl], %[avl], e32, m8, ta, ma\n\t"
                       "vle32.v v24, (%[src])\n\t"
                       "vfmul.vf v16, v24, %[scale]\n\t"
                       "sh2add %[src], %[vl], %[src]\n\t"
                       "vsetvli x0, x0, e8alt, m2, ta, ma\n\t"
                       "vfncvt.sat.f.f.q v12, v16\n\t"
                       "vse8.v v12, (%[dst])\n\t"
                       "add %[dst], %[dst], %[vl]\n\t"
                       : [vl] "=&r"(vl), [src] "+&r"(pSrc), [dst] "+r"(pDst)
                       : [avl] "r"(avl), [scale] "f"(scaling_factor)
                       : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                         "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                         "v12", "v13", "vtype", "vl", "memory");
    }
  }
}
