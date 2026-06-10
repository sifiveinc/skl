// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC_PRIVATE void RMSNormRemain(float *pDst, const float *pSrc,
                                    const float *pWeight, size_t rsc,
                                    float epsilon, size_t n) {
  __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"((size_t)~0) : "cc");
  __asm__ volatile("vmv.v.i v8, 0"
               :
               :
               : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");

  float *iptr = (float*)pSrc;
  float *iptr2 = iptr + rsc;

  float *iptr_norm = (float*)pSrc;
  float *iptr2_norm = iptr_norm + rsc;

  float *optr = pDst;
  float *optr2 = optr + rsc;

  float f_one = 1.0f;
  const float half = (float)0x1.000206p-1;

  size_t remaining_rows = n / rsc;
  float reciprocal_ncols = 1.0f / (float)rsc;

  while (remaining_rows >= 2) {

    __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma"
                 :
                 : "r"((size_t)~0)
                 : "cc");
    __asm__ volatile("vmv.v.i v0, 0"
                 :
                 :
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    __asm__ volatile("vmv.v.i v24, 0"
                 :
                 :
                 : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

    size_t avl = rsc;
    while (avl) {
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e32, m8, tu, ma"
                   : "=r"(vl)
                   : "r"(avl)
                   : "cc");

      __asm__ volatile("vle32.v v8, (%0)"
                   :
                   : "r"(iptr)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
      __asm__ volatile("vfmacc.vv v0, v8, v8"
                   :
                   :
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");

      __asm__ volatile("vle32.v v8, (%0)"
                   :
                   : "r"(iptr2)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
      __asm__ volatile("vfmacc.vv v24, v8, v8"
                   :
                   :
                   : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

      avl -= vl;
      iptr += vl;  
      iptr2 += vl; 
    }

    // Two rows reduction in turn
    __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma"
                 :
                 : "r"((size_t)~0)
                 : "cc");
    __asm__ volatile("vfadd.vv v0, v0, v1" : : : "v0");
    __asm__ volatile("vfadd.vv v24, v24, v25" : : : "v24");
    __asm__ volatile("vfadd.vv v2, v2, v3" : : : "v2");
    __asm__ volatile("vfadd.vv v26, v26, v27" : : : "v26");
    __asm__ volatile("vfadd.vv v4, v4, v5" : : : "v4");
    __asm__ volatile("vfadd.vv v28, v28, v29" : : : "v28");
    __asm__ volatile("vfadd.vv v6, v6, v7" : : : "v6");
    __asm__ volatile("vfadd.vv v30, v30, v31" : : : "v30");

    __asm__ volatile("vmv.v.i v8, 0" : : : "v8");

    __asm__ volatile("vfadd.vv v0, v0, v2" : : : "v0");
    __asm__ volatile("vfadd.vv v24, v24, v26" : : : "v24");
    __asm__ volatile("vfadd.vv v4, v4, v6" : : : "v4");
    __asm__ volatile("vfadd.vv v28, v28, v30" : : : "v28");
    __asm__ volatile("vfadd.vv v0, v0, v4" : : : "v0");
    __asm__ volatile("vfadd.vv v24, v24, v28" : : : "v24");

    iptr += rsc;
    iptr2 += rsc;

    remaining_rows -= 2;
    __asm__ volatile("vfredusum.vs v17, v0, v8" : : : "v17");
    __asm__ volatile("vfredusum.vs v18, v24, v8" : : : "v18");

    __asm__ volatile("vfmv.v.f v1, %0" : : "f"(epsilon) : "v1");
    __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma" : : "r"((size_t)2) : "cc");
    __asm__ volatile("vslideup.vx v17, v18, %0" : : "r"((size_t)1) : "v17");
    __asm__ volatile("vfmadd.vf v17, %0, v1" : : "f"(reciprocal_ncols) : "v17");

    float rsqrt1, rsqrt2;

    __asm__ volatile("vfmv.v.f v9, %0" : : "f"(f_one) : "v9");
    __asm__ volatile("vfrsqrt7.v   v11, v17" : : : "v11");
    __asm__ volatile("vfmul.vv     v13, v11, v17" : : : "v13");
    __asm__ volatile("vmfeq.vv     v0, v13, v13" : : : "v0");
    __asm__ volatile("vfmsub.vv    v13, v11, v9" : : : "v13");
    __asm__ volatile("vfmul.vf     v14, v11, %0" : : "f"(half) : "v14");
    __asm__ volatile("vfnmsac.vv   v11, v14, v13, v0.t" : : : "v11");

    __asm__ volatile("vfmul.vv     v15, v11, v17 " : : : "v15");
    __asm__ volatile("vfmsub.vv    v15, v11, v9" : : : "v15");
    __asm__ volatile("vfmul.vf     v16, v11, %0" : : "f"(half) : "v16");
    __asm__ volatile("vfnmsac.vv   v11, v16, v15, v0.t" : : : "v11");

    __asm__ volatile("vfmv.f.s %0, v11" : "=f"(rsqrt1));
    __asm__ volatile("vfslide1down.vf v11, v11, %0" : : "f"(0.0f) : "v11");
    __asm__ volatile("vfmv.f.s %0, v11" : "=f"(rsqrt2));

    for (size_t avl = rsc, vl = 0, col_offset = 0; avl > 0;
         avl -= vl, col_offset += vl) {
      __asm__ volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(avl)
                   : "cc");

      __asm__ volatile("vle32.v v8, (%0)"
                   :
                   : "r"(iptr_norm + col_offset)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
      __asm__ volatile("vfmul.vf v8, v8, %0"
                   :
                   : "f"(rsqrt1)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
      __asm__ volatile("vle32.v v0, (%0)"
                   :
                   : "r"(pWeight + col_offset)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
      __asm__ volatile("vfmul.vv v8, v8, v0"
                   :
                   :
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
      __asm__ volatile("vse32.v v8, (%0)" : : "r"(optr + col_offset) : "memory");

      __asm__ volatile("vle32.v v16, (%0)"
                   :
                   : "r"(iptr2_norm + col_offset)
                   : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                     "memory");
      __asm__ volatile("vfmul.vf v16, v16, %0"
                   :
                   : "f"(rsqrt2)
                   : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
      __asm__ volatile("vfmul.vv v16, v16, v0"
                   :
                   :
                   : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
      __asm__ volatile("vse32.v v16, (%0)" : : "r"(optr2 + col_offset) : "memory");
    }

    iptr_norm += 2 * rsc;
    iptr2_norm += 2 * rsc;

    optr += 2 * rsc;
    optr2 += 2 * rsc;
  }

  if (remaining_rows) {
    __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma"
                 :
                 : "r"((size_t)~0)
                 : "cc");
    __asm__ volatile("vmv.v.i v0, 0"
                 :
                 :
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");

    size_t avl = rsc;
    while (avl) {
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e32, m8, tu, ma"
                   : "=r"(vl)
                   : "r"(avl)
                   : "cc");
      __asm__ volatile("vle32.v v8, (%0)"
                   :
                   : "r"(iptr)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
      __asm__ volatile("vfmacc.vv v0, v8, v8"
                   :
                   :
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
      avl -= vl;
      iptr += vl;
    }

    __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma"
                 :
                 : "r"((size_t)~0)
                 : "cc");
    __asm__ volatile("vfadd.vv v0, v0, v1" : : : "v0");
    __asm__ volatile("vfadd.vv v4, v4, v5" : : : "v4");
    __asm__ volatile("vfadd.vv v2, v2, v3" : : : "v2");
    __asm__ volatile("vfadd.vv v6, v6, v7" : : : "v6");

    __asm__ volatile("vmv.v.i v8, 0" : : : "v8");

    __asm__ volatile("vfadd.vv v0, v0, v2" : : : "v0");
    __asm__ volatile("vfadd.vv v4, v4, v6" : : : "v4");
    __asm__ volatile("vfadd.vv v0, v0, v4" : : : "v0");

    __asm__ volatile("vfredusum.vs v17, v0, v8" : : : "v17");

    __asm__ volatile("vfmv.v.f v1, %0" : : "f"(epsilon) : "v1");
    __asm__ volatile("vfmadd.vf v17, %0, v1" : : "f"(reciprocal_ncols) : "v17");

    float rsqrt;

    __asm__ volatile("vfmv.v.f v9, %0" : : "f"(f_one) : "v9");
    __asm__ volatile("vfrsqrt7.v   v11, v17" : : : "v11");
    __asm__ volatile("vfmul.vv     v13, v11, v17" : : : "v13");
    __asm__ volatile("vmfeq.vv     v0, v13, v13" : : : "v0");
    __asm__ volatile("vfmsub.vv    v13, v11, v9" : : : "v13");
    __asm__ volatile("vfmul.vf     v14, v11, %0" : : "f"(half) : "v14");
    __asm__ volatile("vfnmsac.vv   v11, v14, v13, v0.t" : : : "v11");

    __asm__ volatile("vfmul.vv     v15, v11, v17 " : : : "v15");
    __asm__ volatile("vfmsub.vv    v15, v11, v9" : : : "v15");
    __asm__ volatile("vfmul.vf     v16, v11, %0" : : "f"(half) : "v16");
    __asm__ volatile("vfnmsac.vv   v11, v16, v15, v0.t" : : : "v11");

    __asm__ volatile("vfmv.f.s %0, v11" : "=f"(rsqrt));

    for (size_t avl = rsc, vl = 0, col_offset = 0; avl > 0;
         avl -= vl, col_offset += vl) {
      __asm__ volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(avl)
                   : "cc");

      __asm__ volatile("vle32.v v8, (%0)"
                   :
                   : "r"(iptr_norm + col_offset)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
      __asm__ volatile("vfmul.vf v8, v8, %0"
                   :
                   : "f"(rsqrt)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
      __asm__ volatile("vle32.v v0, (%0)"
                   :
                   : "r"(pWeight + col_offset)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
      __asm__ volatile("vfmul.vv v8, v8, v0"
                   :
                   :
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
      __asm__ volatile("vse32.v v8, (%0)" : : "r"(optr + col_offset) : "memory");
    }
  }
}

SKL_FUNC void skl_rmsnorm_f32_zve32f(float *pDst, const float *pSrc,
                                     const float *pWeight, size_t rsc,
                                     float epsilon, size_t n) {
  // Hello world - placeholder implementation
  // TODO: Implement RMS normalization using RISC-V vector intrinsics
  RMSNormRemain(pDst, pSrc, pWeight, rsc, epsilon, n);
}
