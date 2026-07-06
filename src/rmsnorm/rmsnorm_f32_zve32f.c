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




SKL_FUNC_PRIVATE void RMSNormAioRows(const float *input, const float *scale, float *output,
                      size_t nrows, size_t ncols, float reciprocal_ncols,
                      float epsilon) {

  size_t reg_vl_m4 = __riscv_v_min_vlen >> 3;
  size_t reg_vl_m1 = __riscv_v_min_vlen >> 5;

  __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(~0));

  float *iptr = (float *)input;
  float *iptr2 = iptr + ncols;

  float *iptr_norm = (float *)input;


  float *optr = output;

  size_t remaining_rows = nrows;

  while(remaining_rows >= 4){

    while(reg_vl_m4 > remaining_rows) reg_vl_m4 /= 2;

    while (remaining_rows >= reg_vl_m4) {

      // Process 4 rows at a time
      for(size_t i = 0 ; i < (reg_vl_m4/4) ; i++)
      {
        size_t avl;

        // 2 Rows Square Sum 
        __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(~0));
        __asm__ volatile("vmv.v.i v0, 0");
        __asm__ volatile("vmv.v.i v24, 0");
        avl = ncols;
        while (avl) {
          size_t vl;
          __asm__ volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(avl));
          __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr));
          __asm__ volatile("vfmacc.vv v0, v8, v8");
          __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr2));
          __asm__ volatile("vfmacc.vv v24, v8, v8");
          avl -= vl;
          iptr += vl;
          iptr2 += vl;
        }
        __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma" : : "r"(~0));
        __asm__ volatile("vfadd.vv v0, v0, v1");
        __asm__ volatile("vfadd.vv v24, v24, v25");
        __asm__ volatile("vfadd.vv v2, v2, v3");
        __asm__ volatile("vfadd.vv v26, v26, v27");
        __asm__ volatile("vfadd.vv v4, v4, v5");
        __asm__ volatile("vfadd.vv v28, v28, v29");
        __asm__ volatile("vfadd.vv v6, v6, v7");
        __asm__ volatile("vfadd.vv v30, v30, v31");
        __asm__ volatile("vmv.v.i v8, 0");
        __asm__ volatile("vfadd.vv v0, v0, v2");
        __asm__ volatile("vfadd.vv v24, v24, v26");
        __asm__ volatile("vfadd.vv v4, v4, v6");
        __asm__ volatile("vfadd.vv v28, v28, v30");
        __asm__ volatile("vfadd.vv v0, v0, v4");
        __asm__ volatile("vfadd.vv v24, v24, v28");
        iptr += ncols;
        iptr2 += ncols;

        remaining_rows -= 2;
        __asm__ volatile("vfredusum.vs v20, v0, v8");
        __asm__ volatile("vfredusum.vs v21, v24, v8");

        // 2 Rows Square Sum 
        __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(~0));
        __asm__ volatile("vmv.v.i v0, 0");
        __asm__ volatile("vmv.v.i v24, 0");
        avl = ncols;
        while (avl) {
          size_t vl;
          __asm__ volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(avl));
          __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr));
          __asm__ volatile("vfmacc.vv v0, v8, v8");
          __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr2));
          __asm__ volatile("vfmacc.vv v24, v8, v8");
          avl -= vl;
          iptr += vl;
          iptr2 += vl;
        }
        __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma" : : "r"(~0));
        __asm__ volatile("vfadd.vv v0, v0, v1");
        __asm__ volatile("vfadd.vv v24, v24, v25");
        __asm__ volatile("vfadd.vv v2, v2, v3");
        __asm__ volatile("vfadd.vv v26, v26, v27");
        __asm__ volatile("vfadd.vv v4, v4, v5");
        __asm__ volatile("vfadd.vv v28, v28, v29");
        __asm__ volatile("vfadd.vv v6, v6, v7");
        __asm__ volatile("vfadd.vv v30, v30, v31");
        __asm__ volatile("vmv.v.i v8, 0");
        __asm__ volatile("vfadd.vv v0, v0, v2");
        __asm__ volatile("vfadd.vv v24, v24, v26");
        __asm__ volatile("vfadd.vv v4, v4, v6");
        __asm__ volatile("vfadd.vv v28, v28, v30");
        __asm__ volatile("vfadd.vv v0, v0, v4");
        __asm__ volatile("vfadd.vv v24, v24, v28");
        iptr += ncols;
        iptr2 += ncols;


        remaining_rows -= 2;
        __asm__ volatile("vfredusum.vs v22, v0, v8");
        __asm__ volatile("vfredusum.vs v23, v24, v8");

        __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma": : "r"(~0));

        // collect the square sum in v16~v19
        size_t four_row_cap = reg_vl_m1 / 4;
        if(i >= (four_row_cap*3))
        {
          size_t idx = (i - four_row_cap*3)*4;
          __asm__ volatile("vslideup.vx v19, v20, %0" : : "r"((idx+0)) : "v19");
          __asm__ volatile("vslideup.vx v19, v21, %0" : : "r"((idx+1)) : "v19");
          __asm__ volatile("vslideup.vx v19, v22, %0" : : "r"((idx+2)) : "v19");
          __asm__ volatile("vslideup.vx v19, v23, %0" : : "r"((idx+3)) : "v19");
        } else if(i >= (four_row_cap*2)) {
          size_t idx = (i - four_row_cap*2)*4;
          __asm__ volatile("vslideup.vx v18, v20, %0" : : "r"((idx+0)) : "v18");
          __asm__ volatile("vslideup.vx v18, v21, %0" : : "r"((idx+1)) : "v18");
          __asm__ volatile("vslideup.vx v18, v22, %0" : : "r"((idx+2)) : "v18");
          __asm__ volatile("vslideup.vx v18, v23, %0" : : "r"((idx+3)) : "v18");
        } else if(i >= four_row_cap) {
          size_t idx = (i - four_row_cap)*4;
          __asm__ volatile("vslideup.vx v17, v20, %0" : : "r"((idx+0)) : "v17");
          __asm__ volatile("vslideup.vx v17, v21, %0" : : "r"((idx+1)) : "v17");
          __asm__ volatile("vslideup.vx v17, v22, %0" : : "r"((idx+2)) : "v17");
          __asm__ volatile("vslideup.vx v17, v23, %0" : : "r"((idx+3)) : "v17");
        } else {
          size_t idx = i*4;
          __asm__ volatile("vslideup.vx v16, v20, %0" : : "r"((idx+0)) : "v16");
          __asm__ volatile("vslideup.vx v16, v21, %0" : : "r"((idx+1)) : "v16");
          __asm__ volatile("vslideup.vx v16, v22, %0" : : "r"((idx+2)) : "v16");
          __asm__ volatile("vslideup.vx v16, v23, %0" : : "r"((idx+3)) : "v16");
        }
      }

      __asm__ volatile("vsetvli zero, %0, e32, m4, tu, ma" : : "r"(~0));

      __asm__ volatile("vfmv.v.f v0, %0" : : "f"(epsilon));
      __asm__ volatile("vfmadd.vf v16, %0, v0" : : "f"(reciprocal_ncols));

      // Calculate rsqrt
      float f_one = 1.0f;
      const float half = (float)0x1.000206p-1;

      __asm__ volatile("vfmv.v.f v20, %0" : : "f"(f_one));
      __asm__ volatile("vfrsqrt7.v   v28, v16");
      __asm__ volatile("vfmul.vv     v8, v28, v16");
      __asm__ volatile("vmfeq.vv     v0, v8, v8");
      __asm__ volatile("vfmsub.vv    v8, v28, v20");
      __asm__ volatile("vfmul.vf     v12, v28, %0" : : "f"(half));
      __asm__ volatile("vfnmsac.vv   v28, v12, v8, v0.t");
      __asm__ volatile("vfmul.vv     v8, v28, v16");
      __asm__ volatile("vfmsub.vv    v8, v28, v20");
      __asm__ volatile("vfmul.vf     v12, v28, %0" : : "f"(half));
      __asm__ volatile("vfnmsac.vv   v28, v12, v8, v0.t");
      
      __asm__ volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(reg_vl_m4));
      __asm__ volatile("vsse32.v   v28, (%0), %1" : : "r" (optr), "r" (ncols*sizeof(float)));

      for (size_t i = 0, row_offset = 0; i < reg_vl_m4; i+=3, row_offset += 3*ncols) {
        float rms_1 = *(optr + row_offset);
        float rms_2;
        if(i < (reg_vl_m4-1)) rms_2 = *(optr + row_offset + ncols);
        float rms_3;
        if(i < (reg_vl_m4-2)) rms_3 = *(optr + row_offset + 2*ncols);
        for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0;
          avl -= vl, col_offset += vl) {
              __asm__ volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
              __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr_norm + row_offset + col_offset));
              __asm__ volatile("vfmul.vf v8, v8, %0" : : "f"(rms_1));
              __asm__ volatile("vle32.v v0, (%0)" : : "r"(scale + col_offset));
              __asm__ volatile("vfmul.vv v8, v8, v0" :);
              __asm__ volatile("vse32.v v8, (%0)" : : "r"(optr + row_offset + col_offset));

              if(i >= (reg_vl_m4-1)) continue;

              __asm__ volatile("vle32.v v16, (%0)" : : "r"(iptr_norm + row_offset + col_offset + ncols));
              __asm__ volatile("vfmul.vf v16, v16, %0" : : "f"(rms_2));
              __asm__ volatile("vfmul.vv v16, v16, v0" :);
              __asm__ volatile("vse32.v v16, (%0)" : : "r"(optr + row_offset + col_offset + ncols));

              if(i >= (reg_vl_m4-2)) continue;

              __asm__ volatile("vle32.v v24, (%0)" : : "r"(iptr_norm + row_offset + col_offset + ncols*2));
              __asm__ volatile("vfmul.vf v24, v24, %0" : : "f"(rms_3));
              __asm__ volatile("vfmul.vv v24, v24, v0" :);
              __asm__ volatile("vse32.v v24, (%0)" : : "r"(optr + row_offset + col_offset + ncols*2));
          }
      }

      iptr_norm += reg_vl_m4 * ncols;
      optr += reg_vl_m4 * ncols;

    }
  }
}

SKL_FUNC_PRIVATE void RMSNormAioRem(const float *input, const float *scale, float *output,
                      size_t nrows, size_t ncols, float reciprocal_ncols,
                      float epsilon) {
  __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(~0));
  __asm__ volatile("vmv.v.i v8, 0");

  float *iptr = (float *)input;
  float *iptr2 = iptr + ncols;

  float *iptr_norm = (float *)input;
  float *iptr2_norm = iptr_norm + ncols;

  float *optr = output;
  float *optr2 = optr + ncols;

  float f_one = 1.0f;
  const float half = (float)0x1.000206p-1;

  size_t remaining_rows = nrows;
  while (remaining_rows >= 2) {

      // 2 Rows Square Sum 
      __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(~0));
      __asm__ volatile("vmv.v.i v0, 0");
      __asm__ volatile("vmv.v.i v24, 0");
      size_t avl = ncols;
      while (avl) {
        size_t vl;
        __asm__ volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(avl));
        __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr));
        __asm__ volatile("vfmacc.vv v0, v8, v8");
        __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr2));
        __asm__ volatile("vfmacc.vv v24, v8, v8");
        avl -= vl;
        iptr += vl;
        iptr2 += vl;
      }
      __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma" : : "r"(~0));
      __asm__ volatile("vfadd.vv v0, v0, v1");
      __asm__ volatile("vfadd.vv v24, v24, v25");
      __asm__ volatile("vfadd.vv v2, v2, v3");
      __asm__ volatile("vfadd.vv v26, v26, v27");
      __asm__ volatile("vfadd.vv v4, v4, v5");
      __asm__ volatile("vfadd.vv v28, v28, v29");
      __asm__ volatile("vfadd.vv v6, v6, v7");
      __asm__ volatile("vfadd.vv v30, v30, v31");
      __asm__ volatile("vmv.v.i v8, 0");
      __asm__ volatile("vfadd.vv v0, v0, v2");
      __asm__ volatile("vfadd.vv v24, v24, v26");
      __asm__ volatile("vfadd.vv v4, v4, v6");
      __asm__ volatile("vfadd.vv v28, v28, v30");
      __asm__ volatile("vfadd.vv v0, v0, v4");
      __asm__ volatile("vfadd.vv v24, v24, v28");
      iptr += ncols;
      iptr2 += ncols;
    
    remaining_rows -= 2;
    __asm__ volatile("vfredusum.vs v17, v0, v8");
    __asm__ volatile("vfredusum.vs v18, v24, v8");

    __asm__ volatile("vfmv.v.f v1, %0" : : "f"(epsilon));
    __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma": : "r"(2));
    __asm__ volatile("vslideup.vx v17, v18, %0" : : "r"(1) : "v16");
    __asm__ volatile("vfmadd.vf v17, %0, v1" : : "f"(reciprocal_ncols));

    float rsqrt1, rsqrt2;

    __asm__ volatile("vfmv.v.f v9, %0" : : "f"(f_one));
    __asm__ volatile("vfrsqrt7.v  v11, v17");
    __asm__ volatile("vfmul.vv    v13, v11, v17");
    __asm__ volatile("vmfeq.vv    v0, v13, v13");
    __asm__ volatile("vfmsub.vv   v13, v11, v9");
    __asm__ volatile("vfmul.vf    v14, v11, %0" : : "f"(half));
    __asm__ volatile("vfnmsac.vv  v11, v14, v13, v0.t");

    __asm__ volatile("vfmul.vv    v15, v11, v17 ");
    __asm__ volatile("vfmsub.vv   v15, v11, v9");
    __asm__ volatile("vfmul.vf    v16, v11, %0" : : "f"(half));
    __asm__ volatile("vfnmsac.vv  v11, v16, v15, v0.t");

    __asm__ volatile("vfmv.f.s %0, v11" : "=f"(rsqrt1));
    __asm__ volatile("vfslide1down.vf v11, v11, %0" : : "f"(0.0f));
    __asm__ volatile("vfmv.f.s %0, v11" : "=f"(rsqrt2));


    for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0;
        avl -= vl, col_offset += vl) {
      __asm__ volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
      
      __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr_norm + col_offset));
      __asm__ volatile("vfmul.vf v8, v8, %0" : : "f"(rsqrt1));
      __asm__ volatile("vle32.v v0, (%0)" : : "r"(scale + col_offset));
      __asm__ volatile("vfmul.vv v8, v8, v0" :);
      __asm__ volatile("vse32.v v8, (%0)" : : "r"(optr + col_offset));
      
      __asm__ volatile("vle32.v v16, (%0)" : : "r"(iptr2_norm + col_offset));
      __asm__ volatile("vfmul.vf v16, v16, %0" : : "f"(rsqrt2));
      __asm__ volatile("vfmul.vv v16, v16, v0" :);
      __asm__ volatile("vse32.v v16, (%0)" : : "r"(optr2 + col_offset));
    }

    iptr_norm += 2 * ncols;
    iptr2_norm += 2 * ncols;

    optr += 2 * ncols;
    optr2 += 2 * ncols;
  }

  if (remaining_rows) {
    __asm__ volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(~0));
    __asm__ volatile("vmv.v.i v0, 0");

    size_t avl = ncols;
    while (avl) {
      size_t vl;
      __asm__ volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(avl));
      __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr));
      __asm__ volatile("vfmacc.vv v0, v8, v8");
      avl -= vl;
      iptr += vl;
    }

    __asm__ volatile("vsetvli zero, %0, e32, m1, tu, ma" : : "r"(~0));
    __asm__ volatile("vfadd.vv v0, v0, v1");
    __asm__ volatile("vfadd.vv v4, v4, v5");
    __asm__ volatile("vfadd.vv v2, v2, v3");
    __asm__ volatile("vfadd.vv v6, v6, v7");

    __asm__ volatile("vmv.v.i v8, 0");

    __asm__ volatile("vfadd.vv v0, v0, v2");
    __asm__ volatile("vfadd.vv v4, v4, v6");
    __asm__ volatile("vfadd.vv v0, v0, v4");

    __asm__ volatile("vfredusum.vs v17, v0, v8");

    __asm__ volatile("vfmv.v.f v1, %0" : : "f"(epsilon));
    __asm__ volatile("vfmadd.vf v17, %0, v1" : : "f"(reciprocal_ncols));

    float rsqrt;

    __asm__ volatile("vfmv.v.f v9, %0" : : "f"(f_one));
    __asm__ volatile("vfrsqrt7.v  v11, v17");
    __asm__ volatile("vfmul.vv    v13, v11, v17");
    __asm__ volatile("vmfeq.vv    v0, v13, v13");
    __asm__ volatile("vfmsub.vv   v13, v11, v9");
    __asm__ volatile("vfmul.vf    v14, v11, %0" : : "f"(half));
    __asm__ volatile("vfnmsac.vv  v11, v14, v13, v0.t");

    __asm__ volatile("vfmul.vv    v15, v11, v17 ");
    __asm__ volatile("vfmsub.vv   v15, v11, v9");
    __asm__ volatile("vfmul.vf    v16, v11, %0" : : "f"(half));
    __asm__ volatile("vfnmsac.vv  v11, v16, v15, v0.t");

    __asm__ volatile("vfmv.f.s %0, v11" : "=f"(rsqrt));

    for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0;
        avl -= vl, col_offset += vl) {

      __asm__ volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));

      __asm__ volatile("vle32.v v8, (%0)" : : "r"(iptr_norm + col_offset));
      __asm__ volatile("vfmul.vf v8, v8, %0" : : "f"(rsqrt));
      __asm__ volatile("vle32.v v0, (%0)" : : "r"(scale + col_offset));
      __asm__ volatile("vfmul.vv v8, v8, v0" :);
      __asm__ volatile("vse32.v v8, (%0)" : : "r"(optr + col_offset));

    }

  }
      
}



SKL_FUNC_PRIVATE void RmsNormAio(float *pDst, const float *pSrc,
                                    const float *pWeight, size_t rsc,
                                    float epsilon, size_t n) {

  size_t nrows = n / rsc;
  float reciprocal_ncols = 1.0f / (float)rsc;

  size_t nrows_remains = nrows - (nrows % 4);  
  const float *ipt = pSrc;
  float *opt = pDst;

  if(nrows_remains > 0) {
    RMSNormAioRows(ipt, pWeight, opt, nrows_remains, rsc, reciprocal_ncols,
                                epsilon);

    ipt = pSrc + nrows_remains*rsc;
    opt = pDst + nrows_remains*rsc;
  }

  if((nrows % 4) > 0)
  {
      RMSNormAioRem(ipt, pWeight, opt, (nrows % 4), rsc, reciprocal_ncols,
                          epsilon);
  }
}


SKL_FUNC void skl_rmsnorm_f32_zve32f(float *pDst, const float *pSrc,
                                     const float *pWeight, size_t rsc,
                                     float epsilon, size_t n) {
  RmsNormAio(pDst, pSrc, pWeight, rsc, epsilon, n);
}
