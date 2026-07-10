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


SKL_FUNC_PRIVATE void skl_rmsnorm_rows_f32_zve32f(const float *input, const float *scale, float *output,
                      size_t nrows, size_t ncols, float reciprocal_ncols,
                      float epsilon) {

  size_t reg_vl_m4 = __riscv_v_min_vlen >> 3;
  size_t reg_vl_m1 = __riscv_v_min_vlen >> 5;

  float *iptr = (float *)input;
  float *iptr2 = iptr + ncols;
  float *iptr_norm = (float *)input;
  float *optr = output;
  size_t remaining_rows = nrows;

  // Initial vsetvli configuration
  __asm__ volatile(
    "vsetvli zero, %[max_vl], e32, m8, tu, ma\n\t"
    :
    : [max_vl] "r"(~0UL)
    : "vl", "vtype"
  );

  while(remaining_rows >= 4){
    while(reg_vl_m4 > remaining_rows) reg_vl_m4 /= 2;

    while (remaining_rows >= reg_vl_m4) {
      // Process 4 rows at a time
      for(size_t i = 0 ; i < (reg_vl_m4/4) ; i++)
      {
        size_t avl;

        /* ========== First 2 Rows Square Sum ========== */
        // Register allocation:
        // v0[m8]: accumulator for row 1 square sum
        // v24[m8]: accumulator for row 2 square sum
        // v8[m8]: temporary for loaded data
        // v20, v21: reduction results
        __asm__ volatile(
          "vsetvli zero, %[max_vl], e32, m8, tu, ma\n\t"
          "vmv.v.i v0, 0\n\t"
          "vmv.v.i v24, 0\n\t"
          :
          : [max_vl] "r"(~0UL)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
            "vl", "vtype"
        );

        avl = ncols;
        while (avl) {
          size_t vl;
          __asm__ volatile(
            "vsetvli %[vl_out], %[avl_in], e32, m8, tu, ma\n\t"
            "vle32.v v8, (%[ptr1])\n\t"
            "vfmacc.vv v0, v8, v8\n\t"
            "vle32.v v8, (%[ptr2])\n\t"
            "vfmacc.vv v24, v8, v8\n\t"
            : [vl_out] "=&r"(vl)
            : [avl_in] "r"(avl),
              [ptr1] "r"(iptr),
              [ptr2] "r"(iptr2)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "vl", "vtype", "memory"
          );
          avl -= vl;
          iptr += vl;
          iptr2 += vl;
        }

        // Tree reduction m8->m1
        __asm__ volatile(
          "vsetvli zero, %[max_vl], e32, m1, tu, ma\n\t"
          "vfadd.vv v0, v0, v1\n\t"
          "vfadd.vv v24, v24, v25\n\t"
          "vfadd.vv v2, v2, v3\n\t"
          "vfadd.vv v26, v26, v27\n\t"
          "vfadd.vv v4, v4, v5\n\t"
          "vfadd.vv v28, v28, v29\n\t"
          "vfadd.vv v6, v6, v7\n\t"
          "vfadd.vv v30, v30, v31\n\t"
          "vmv.v.i v8, 0\n\t"
          "vfadd.vv v0, v0, v2\n\t"
          "vfadd.vv v24, v24, v26\n\t"
          "vfadd.vv v4, v4, v6\n\t"
          "vfadd.vv v28, v28, v30\n\t"
          "vfadd.vv v0, v0, v4\n\t"
          "vfadd.vv v24, v24, v28\n\t"
          :
          : [max_vl] "r"(~0UL)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
            "vl", "vtype"
        );

        iptr += ncols;
        iptr2 += ncols;
        remaining_rows -= 2;

        // Horizontal reduction
        __asm__ volatile(
          "vfredusum.vs v20, v0, v8\n\t"
          "vfredusum.vs v21, v24, v8\n\t"
          :
          :
          : "v20", "v21"
        );

        /* ========== Second 2 Rows Square Sum ========== */
        __asm__ volatile(
          "vsetvli zero, %[max_vl], e32, m8, tu, ma\n\t"
          "vmv.v.i v0, 0\n\t"
          "vmv.v.i v24, 0\n\t"
          :
          : [max_vl] "r"(~0UL)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
            "vl", "vtype"
        );

        avl = ncols;
        while (avl) {
          size_t vl;
          __asm__ volatile(
            "vsetvli %[vl_out], %[avl_in], e32, m8, tu, ma\n\t"
            "vle32.v v8, (%[ptr1])\n\t"
            "vfmacc.vv v0, v8, v8\n\t"
            "vle32.v v8, (%[ptr2])\n\t"
            "vfmacc.vv v24, v8, v8\n\t"
            : [vl_out] "=&r"(vl)
            : [avl_in] "r"(avl),
              [ptr1] "r"(iptr),
              [ptr2] "r"(iptr2)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "vl", "vtype", "memory"
          );
          avl -= vl;
          iptr += vl;
          iptr2 += vl;
        }

        __asm__ volatile(
          "vsetvli zero, %[max_vl], e32, m1, tu, ma\n\t"
          "vfadd.vv v0, v0, v1\n\t"
          "vfadd.vv v24, v24, v25\n\t"
          "vfadd.vv v2, v2, v3\n\t"
          "vfadd.vv v26, v26, v27\n\t"
          "vfadd.vv v4, v4, v5\n\t"
          "vfadd.vv v28, v28, v29\n\t"
          "vfadd.vv v6, v6, v7\n\t"
          "vfadd.vv v30, v30, v31\n\t"
          "vmv.v.i v8, 0\n\t"
          "vfadd.vv v0, v0, v2\n\t"
          "vfadd.vv v24, v24, v26\n\t"
          "vfadd.vv v4, v4, v6\n\t"
          "vfadd.vv v28, v28, v30\n\t"
          "vfadd.vv v0, v0, v4\n\t"
          "vfadd.vv v24, v24, v28\n\t"
          :
          : [max_vl] "r"(~0UL)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
            "vl", "vtype"
        );

        iptr += ncols;
        iptr2 += ncols;
        remaining_rows -= 2;

        __asm__ volatile(
          "vfredusum.vs v22, v0, v8\n\t"
          "vfredusum.vs v23, v24, v8\n\t"
          :
          :
          : "v22", "v23"
        );

        /* ========== Collect square sums into v16~v19 ========== */
        __asm__ volatile(
          "vsetvli zero, %[max_vl], e32, m1, tu, ma\n\t"
          :
          : [max_vl] "r"(~0UL)
          : "vl", "vtype"
        );

        size_t four_row_cap = reg_vl_m1 / 4;
        if(i >= (four_row_cap*3))
        {
          size_t idx = (i - four_row_cap*3)*4;
          __asm__ volatile(
            "vslideup.vx v19, v20, %[i0]\n\t"
            "vslideup.vx v19, v21, %[i1]\n\t"
            "vslideup.vx v19, v22, %[i2]\n\t"
            "vslideup.vx v19, v23, %[i3]\n\t"
            :
            : [i0] "r"(idx+0), [i1] "r"(idx+1), [i2] "r"(idx+2), [i3] "r"(idx+3)
            : "v19"
          );
        } else if(i >= (four_row_cap*2)) {
          size_t idx = (i - four_row_cap*2)*4;
          __asm__ volatile(
            "vslideup.vx v18, v20, %[i0]\n\t"
            "vslideup.vx v18, v21, %[i1]\n\t"
            "vslideup.vx v18, v22, %[i2]\n\t"
            "vslideup.vx v18, v23, %[i3]\n\t"
            :
            : [i0] "r"(idx+0), [i1] "r"(idx+1), [i2] "r"(idx+2), [i3] "r"(idx+3)
            : "v18"
          );
        } else if(i >= four_row_cap) {
          size_t idx = (i - four_row_cap)*4;
          __asm__ volatile(
            "vslideup.vx v17, v20, %[i0]\n\t"
            "vslideup.vx v17, v21, %[i1]\n\t"
            "vslideup.vx v17, v22, %[i2]\n\t"
            "vslideup.vx v17, v23, %[i3]\n\t"
            :
            : [i0] "r"(idx+0), [i1] "r"(idx+1), [i2] "r"(idx+2), [i3] "r"(idx+3)
            : "v17"
          );
        } else {
          size_t idx = i*4;
          __asm__ volatile(
            "vslideup.vx v16, v20, %[i0]\n\t"
            "vslideup.vx v16, v21, %[i1]\n\t"
            "vslideup.vx v16, v22, %[i2]\n\t"
            "vslideup.vx v16, v23, %[i3]\n\t"
            :
            : [i0] "r"(idx+0), [i1] "r"(idx+1), [i2] "r"(idx+2), [i3] "r"(idx+3)
            : "v16"
          );
        }
      } // end for loop

      /* ========== Calculate rsqrt for all rows ========== */
      // Register allocation:
      // v16[m4]: mean square values -> rsqrt output
      // v0[m4]: temp (epsilon vector, then mask)
      // v28[m4]: rsqrt approximation
      // v8[m4], v12[m4], v20[m4]: temporaries
      float f_one = 1.0f;
      const float half = (float)0x1.000206p-1;

      __asm__ volatile(
        "vsetvli zero, %[max_vl], e32, m4, tu, ma\n\t"
        "vfmv.v.f v0, %[eps]\n\t"
        "vfmadd.vf v16, %[recip], v0\n\t"
        "vfmv.v.f v20, %[one]\n\t"
        "vfrsqrt7.v v28, v16\n\t"
        "vfmul.vv v8, v28, v16\n\t"
        "vmfeq.vv v0, v8, v8\n\t"
        "vfmsub.vv v8, v28, v20\n\t"
        "vfmul.vf v12, v28, %[half]\n\t"
        "vfnmsac.vv v28, v12, v8, v0.t\n\t"
        "vfmul.vv v8, v28, v16\n\t"
        "vfmsub.vv v8, v28, v20\n\t"
        "vfmul.vf v12, v28, %[half]\n\t"
        "vfnmsac.vv v28, v12, v8, v0.t\n\t"
        :
        : [max_vl] "r"(~0UL),
          [eps] "f"(epsilon),
          [recip] "f"(reciprocal_ncols),
          [one] "f"(f_one),
          [half] "f"(half)
        : "v0", "v1", "v2", "v3",
          "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23",
          "v28", "v29", "v30", "v31",
          "vl", "vtype", "frm", "fflags"
      );

      // Store rsqrt values with stride
      __asm__ volatile(
        "vsetvli zero, %[vl_m4], e32, m4, ta, ma\n\t"
        "vsse32.v v28, (%[optr]), %[stride]\n\t"
        :
        : [vl_m4] "r"(reg_vl_m4),
          [optr] "r"(optr),
          [stride] "r"(ncols*sizeof(float))
        : "vl", "vtype", "memory"
      );

      /* ========== Normalize and scale ========== */
      if(scale) {
        for (size_t i = 0, row_offset = 0; i < reg_vl_m4; i+=3, row_offset += 3*ncols) {
          float rms_1 = *(optr + row_offset);
          float rms_2;
          if(i < (reg_vl_m4-1)) rms_2 = *(optr + row_offset + ncols);
          float rms_3;
          if(i < (reg_vl_m4-2)) rms_3 = *(optr + row_offset + 2*ncols);

          for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0; avl -= vl, col_offset += vl) {
            // Process row 1
            __asm__ volatile(
              "vsetvli %[vl_out], %[avl_in], e32, m8, ta, ma\n\t"
              "vle32.v v8, (%[in_ptr])\n\t"
              "vfmul.vf v8, v8, %[rms1]\n\t"
              "vle32.v v0, (%[scale_ptr])\n\t"
              "vfmul.vv v8, v8, v0\n\t"
              "vse32.v v8, (%[out_ptr])\n\t"
              : [vl_out] "=&r"(vl)
              : [avl_in] "r"(avl),
                [in_ptr] "r"(iptr_norm + row_offset + col_offset),
                [scale_ptr] "r"(scale + col_offset),
                [out_ptr] "r"(optr + row_offset + col_offset),
                [rms1] "f"(rms_1)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "vl", "vtype", "memory", "frm", "fflags"
            );

            if(i >= (reg_vl_m4-1)) continue;

            // Process row 2
            __asm__ volatile(
              "vle32.v v16, (%[in_ptr])\n\t"
              "vfmul.vf v16, v16, %[rms2]\n\t"
              "vfmul.vv v16, v16, v0\n\t"
              "vse32.v v16, (%[out_ptr])\n\t"
              :
              : [in_ptr] "r"(iptr_norm + row_offset + col_offset + ncols),
                [out_ptr] "r"(optr + row_offset + col_offset + ncols),
                [rms2] "f"(rms_2)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                "memory", "frm", "fflags"
            );

            if(i >= (reg_vl_m4-2)) continue;

            // Process row 3
            __asm__ volatile(
              "vle32.v v24, (%[in_ptr])\n\t"
              "vfmul.vf v24, v24, %[rms3]\n\t"
              "vfmul.vv v24, v24, v0\n\t"
              "vse32.v v24, (%[out_ptr])\n\t"
              :
              : [in_ptr] "r"(iptr_norm + row_offset + col_offset + ncols*2),
                [out_ptr] "r"(optr + row_offset + col_offset + ncols*2),
                [rms3] "f"(rms_3)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                "memory", "frm", "fflags"
            );
          }
        }
      } else {
        for (size_t i = 0, row_offset = 0; i < reg_vl_m4; i+=3, row_offset += 3*ncols) {
          float rms_1 = *(optr + row_offset);
          float rms_2;
          if(i < (reg_vl_m4-1)) rms_2 = *(optr + row_offset + ncols);
          float rms_3;
          if(i < (reg_vl_m4-2)) rms_3 = *(optr + row_offset + 2*ncols);

          for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0; avl -= vl, col_offset += vl) {
            // Process row 1
            __asm__ volatile(
              "vsetvli %[vl_out], %[avl_in], e32, m8, ta, ma\n\t"
              "vle32.v v8, (%[in_ptr])\n\t"
              "vfmul.vf v8, v8, %[rms1]\n\t"
              "vse32.v v8, (%[out_ptr])\n\t"
              : [vl_out] "=&r"(vl)
              : [avl_in] "r"(avl),
                [in_ptr] "r"(iptr_norm + row_offset + col_offset),
                [out_ptr] "r"(optr + row_offset + col_offset),
                [rms1] "f"(rms_1)
              : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "vl", "vtype", "memory", "frm", "fflags"
            );

            if(i >= (reg_vl_m4-1)) continue;

            // Process row 2
            __asm__ volatile(
              "vle32.v v16, (%[in_ptr])\n\t"
              "vfmul.vf v16, v16, %[rms2]\n\t"
              "vse32.v v16, (%[out_ptr])\n\t"
              :
              : [in_ptr] "r"(iptr_norm + row_offset + col_offset + ncols),
                [out_ptr] "r"(optr + row_offset + col_offset + ncols),
                [rms2] "f"(rms_2)
              : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                "memory", "frm", "fflags"
            );

            if(i >= (reg_vl_m4-2)) continue;

            // Process row 3
            __asm__ volatile(
              "vle32.v v24, (%[in_ptr])\n\t"
              "vfmul.vf v24, v24, %[rms3]\n\t"
              "vse32.v v24, (%[out_ptr])\n\t"
              :
              : [in_ptr] "r"(iptr_norm + row_offset + col_offset + ncols*2),
                [out_ptr] "r"(optr + row_offset + col_offset + ncols*2),
                [rms3] "f"(rms_3)
              : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                "memory", "frm", "fflags"
            );
          }
        }


      }

      iptr_norm += reg_vl_m4 * ncols;
      optr += reg_vl_m4 * ncols;
    }
  }
}

SKL_FUNC_PRIVATE void skl_rmsnorm_rem_f32_zve32f(const float *input, const float *scale, float *output,
                      size_t nrows, size_t ncols, float reciprocal_ncols,
                      float epsilon) {
  float *iptr = (float *)input;
  float *iptr2 = iptr + ncols;
  float *iptr_norm = (float *)input;
  float *iptr2_norm = iptr_norm + ncols;
  float *optr = output;
  float *optr2 = optr + ncols;

  float f_one = 1.0f;
  const float half = (float)0x1.000206p-1;
  size_t remaining_rows = nrows;

  // Initial setup
  __asm__ volatile(
    "vsetvli zero, %[max_vl], e32, m8, tu, ma\n\t"
    "vmv.v.i v8, 0\n\t"
    :
    : [max_vl] "r"(~0UL)
    : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "vl", "vtype"
  );

  while (remaining_rows >= 2) {
    /* ========== 2 Rows Square Sum ========== */
    // Register allocation:
    // v0[m8]: row 1 accumulator
    // v24[m8]: row 2 accumulator
    // v8[m8]: temp for loads
    // v17, v18: reduction results
    __asm__ volatile(
      "vsetvli zero, %[max_vl], e32, m8, tu, ma\n\t"
      "vmv.v.i v0, 0\n\t"
      "vmv.v.i v24, 0\n\t"
      :
      : [max_vl] "r"(~0UL)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        "vl", "vtype"
    );

    size_t avl = ncols;
    while (avl) {
      size_t vl;
      __asm__ volatile(
        "vsetvli %[vl_out], %[avl_in], e32, m8, tu, ma\n\t"
        "vle32.v v8, (%[ptr1])\n\t"
        "vfmacc.vv v0, v8, v8\n\t"
        "vle32.v v8, (%[ptr2])\n\t"
        "vfmacc.vv v24, v8, v8\n\t"
        : [vl_out] "=&r"(vl)
        : [avl_in] "r"(avl),
          [ptr1] "r"(iptr),
          [ptr2] "r"(iptr2)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "vl", "vtype", "memory"
      );
      avl -= vl;
      iptr += vl;
      iptr2 += vl;
    }

    // Tree reduction m8->m1
    __asm__ volatile(
      "vsetvli zero, %[max_vl], e32, m1, tu, ma\n\t"
      "vfadd.vv v0, v0, v1\n\t"
      "vfadd.vv v24, v24, v25\n\t"
      "vfadd.vv v2, v2, v3\n\t"
      "vfadd.vv v26, v26, v27\n\t"
      "vfadd.vv v4, v4, v5\n\t"
      "vfadd.vv v28, v28, v29\n\t"
      "vfadd.vv v6, v6, v7\n\t"
      "vfadd.vv v30, v30, v31\n\t"
      "vmv.v.i v8, 0\n\t"
      "vfadd.vv v0, v0, v2\n\t"
      "vfadd.vv v24, v24, v26\n\t"
      "vfadd.vv v4, v4, v6\n\t"
      "vfadd.vv v28, v28, v30\n\t"
      "vfadd.vv v0, v0, v4\n\t"
      "vfadd.vv v24, v24, v28\n\t"
      :
      : [max_vl] "r"(~0UL)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        "vl", "vtype"
    );

    iptr += ncols;
    iptr2 += ncols;
    remaining_rows -= 2;

    // Horizontal reduction
    __asm__ volatile(
      "vfredusum.vs v17, v0, v8\n\t"
      "vfredusum.vs v18, v24, v8\n\t"
      :
      :
      : "v17", "v18"
    );

    /* ========== Calculate mean and prepare for rsqrt ========== */
    // Register allocation:
    // v17[m1]: contains [sum1, ?, ?, ...] -> [mean1, mean2, ...]
    // v1[m1]: epsilon vector
    // v9[m1]: ones vector
    // v11[m1]: rsqrt approximation -> final rsqrt
    float rsqrt1, rsqrt2;

    __asm__ volatile(
      "vfmv.v.f v1, %[eps]\n\t"
      "vsetvli zero, %[vl2], e32, m1, tu, ma\n\t"
      "vslideup.vx v17, v18, %[one_idx]\n\t"
      "vfmadd.vf v17, %[recip], v1\n\t"
      "vfmv.v.f v9, %[f_one]\n\t"
      "vfrsqrt7.v v11, v17\n\t"
      "vfmul.vv v13, v11, v17\n\t"
      "vmfeq.vv v0, v13, v13\n\t"
      "vfmsub.vv v13, v11, v9\n\t"
      "vfmul.vf v14, v11, %[half]\n\t"
      "vfnmsac.vv v11, v14, v13, v0.t\n\t"
      "vfmul.vv v15, v11, v17\n\t"
      "vfmsub.vv v15, v11, v9\n\t"
      "vfmul.vf v16, v11, %[half]\n\t"
      "vfnmsac.vv v11, v16, v15, v0.t\n\t"
      :
      : [eps] "f"(epsilon),
        [vl2] "r"(2UL),
        [one_idx] "r"(1UL),
        [recip] "f"(reciprocal_ncols),
        [f_one] "f"(f_one),
        [half] "f"(half)
      : "v0", "v1", "v9", "v11", "v13", "v14", "v15", "v16", "v17", "v18",
        "vl", "vtype", "frm", "fflags"
    );

    // Extract rsqrt values
    __asm__ volatile(
      "vfmv.f.s %[rs1], v11\n\t"
      "vfslide1down.vf v11, v11, %[zero]\n\t"
      "vfmv.f.s %[rs2], v11\n\t"
      : [rs1] "=&f"(rsqrt1),
        [rs2] "=&f"(rsqrt2)
      : [zero] "f"(0.0f)
      : "v11"
    );

    /* ========== Normalize and scale both rows ========== */
    if(scale) {
      for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0; avl -= vl, col_offset += vl) {
        __asm__ volatile(
          "vsetvli %[vl_out], %[avl_in], e32, m8, ta, ma\n\t"
          // Row 1
          "vle32.v v8, (%[in1])\n\t"
          "vfmul.vf v8, v8, %[rs1]\n\t"
          "vle32.v v0, (%[scale_ptr])\n\t"
          "vfmul.vv v8, v8, v0\n\t"
          "vse32.v v8, (%[out1])\n\t"
          // Row 2
          "vle32.v v16, (%[in2])\n\t"
          "vfmul.vf v16, v16, %[rs2]\n\t"
          "vfmul.vv v16, v16, v0\n\t"
          "vse32.v v16, (%[out2])\n\t"
          : [vl_out] "=&r"(vl)
          : [avl_in] "r"(avl),
            [in1] "r"(iptr_norm + col_offset),
            [in2] "r"(iptr2_norm + col_offset),
            [scale_ptr] "r"(scale + col_offset),
            [out1] "r"(optr + col_offset),
            [out2] "r"(optr2 + col_offset),
            [rs1] "f"(rsqrt1),
            [rs2] "f"(rsqrt2)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "vl", "vtype", "memory", "frm", "fflags"
        );
      }
    } else {
      for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0; avl -= vl, col_offset += vl) {
        __asm__ volatile(
          "vsetvli %[vl_out], %[avl_in], e32, m8, ta, ma\n\t"
          // Row 1
          "vle32.v v8, (%[in1])\n\t"
          "vfmul.vf v8, v8, %[rs1]\n\t"
          "vse32.v v8, (%[out1])\n\t"
          // Row 2
          "vle32.v v16, (%[in2])\n\t"
          "vfmul.vf v16, v16, %[rs2]\n\t"
          "vse32.v v16, (%[out2])\n\t"
          : [vl_out] "=&r"(vl)
          : [avl_in] "r"(avl),
            [in1] "r"(iptr_norm + col_offset),
            [in2] "r"(iptr2_norm + col_offset),
            [out1] "r"(optr + col_offset),
            [out2] "r"(optr2 + col_offset),
            [rs1] "f"(rsqrt1),
            [rs2] "f"(rsqrt2)
          : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "vl", "vtype", "memory", "frm", "fflags"
        );
      }
    }

    iptr_norm += 2 * ncols;
    iptr2_norm += 2 * ncols;
    optr += 2 * ncols;
    optr2 += 2 * ncols;
  }

  /* ========== Handle remaining single row ========== */
  if (remaining_rows) {
    __asm__ volatile(
      "vsetvli zero, %[max_vl], e32, m8, tu, ma\n\t"
      "vmv.v.i v0, 0\n\t"
      :
      : [max_vl] "r"(~0UL)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "vl", "vtype"
    );

    size_t avl = ncols;
    while (avl) {
      size_t vl;
      __asm__ volatile(
        "vsetvli %[vl_out], %[avl_in], e32, m8, tu, ma\n\t"
        "vle32.v v8, (%[ptr])\n\t"
        "vfmacc.vv v0, v8, v8\n\t"
        : [vl_out] "=&r"(vl)
        : [avl_in] "r"(avl),
          [ptr] "r"(iptr)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "vl", "vtype", "memory"
      );
      avl -= vl;
      iptr += vl;
    }

    // Tree reduction
    __asm__ volatile(
      "vsetvli zero, %[max_vl], e32, m1, tu, ma\n\t"
      "vfadd.vv v0, v0, v1\n\t"
      "vfadd.vv v4, v4, v5\n\t"
      "vfadd.vv v2, v2, v3\n\t"
      "vfadd.vv v6, v6, v7\n\t"
      "vmv.v.i v8, 0\n\t"
      "vfadd.vv v0, v0, v2\n\t"
      "vfadd.vv v4, v4, v6\n\t"
      "vfadd.vv v0, v0, v4\n\t"
      :
      : [max_vl] "r"(~0UL)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "vl", "vtype"
    );

    float rsqrt;
    __asm__ volatile(
      "vfredusum.vs v17, v0, v8\n\t"
      "vfmv.v.f v1, %[eps]\n\t"
      "vfmadd.vf v17, %[recip], v1\n\t"
      "vfmv.v.f v9, %[f_one]\n\t"
      "vfrsqrt7.v v11, v17\n\t"
      "vfmul.vv v13, v11, v17\n\t"
      "vmfeq.vv v0, v13, v13\n\t"
      "vfmsub.vv v13, v11, v9\n\t"
      "vfmul.vf v14, v11, %[half]\n\t"
      "vfnmsac.vv v11, v14, v13, v0.t\n\t"
      "vfmul.vv v15, v11, v17\n\t"
      "vfmsub.vv v15, v11, v9\n\t"
      "vfmul.vf v16, v11, %[half]\n\t"
      "vfnmsac.vv v11, v16, v15, v0.t\n\t"
      "vfmv.f.s %[rs], v11\n\t"
      : [rs] "=&f"(rsqrt)
      : [eps] "f"(epsilon),
        [recip] "f"(reciprocal_ncols),
        [f_one] "f"(f_one),
        [half] "f"(half)
      : "v0", "v1", "v9", "v11", "v13", "v14", "v15", "v16", "v17",
        "frm", "fflags"
    );

    // Normalize
    if(scale) {
      for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0; avl -= vl, col_offset += vl) {
        __asm__ volatile(
          "vsetvli %[vl_out], %[avl_in], e32, m8, ta, ma\n\t"
          "vle32.v v8, (%[in_ptr])\n\t"
          "vfmul.vf v8, v8, %[rs]\n\t"
          "vle32.v v0, (%[scale_ptr])\n\t"
          "vfmul.vv v8, v8, v0\n\t"
          "vse32.v v8, (%[out_ptr])\n\t"
          : [vl_out] "=&r"(vl)
          : [avl_in] "r"(avl),
            [in_ptr] "r"(iptr_norm + col_offset),
            [scale_ptr] "r"(scale + col_offset),
            [out_ptr] "r"(optr + col_offset),
            [rs] "f"(rsqrt)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "vl", "vtype", "memory", "frm", "fflags"
        );
      }
    } else {
      for (size_t avl = ncols, vl = 0, col_offset = 0; avl > 0; avl -= vl, col_offset += vl) {
        __asm__ volatile(
          "vsetvli %[vl_out], %[avl_in], e32, m8, ta, ma\n\t"
          "vle32.v v8, (%[in_ptr])\n\t"
          "vfmul.vf v8, v8, %[rs]\n\t"
          "vse32.v v8, (%[out_ptr])\n\t"
          : [vl_out] "=&r"(vl)
          : [avl_in] "r"(avl),
            [in_ptr] "r"(iptr_norm + col_offset),
            [out_ptr] "r"(optr + col_offset),
            [rs] "f"(rsqrt)
          : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "vl", "vtype", "memory", "frm", "fflags"
        );
      }
    }
  }
}


SKL_FUNC void skl_rmsnorm_f32_zve32f(float *pDst, const float *pSrc,
                                     const float *pWeight, size_t rsc,
                                     float epsilon, size_t n) {
  size_t nrows = n / rsc;
  float reciprocal_ncols = 1.0f / (float)rsc;

  size_t nrows_remains = nrows - (nrows % 4);  
  const float *ipt = pSrc;
  float *opt = pDst;

  if(nrows_remains > 0) {
    skl_rmsnorm_rows_f32_zve32f(ipt, pWeight, opt, nrows_remains, rsc, reciprocal_ncols,
                                epsilon);

    ipt = pSrc + nrows_remains*rsc;
    opt = pDst + nrows_remains*rsc;
  }

  if((nrows % 4) > 0)
  {
      skl_rmsnorm_rem_f32_zve32f(ipt, pWeight, opt, (nrows % 4), rsc, reciprocal_ncols,
                          epsilon);
  }
}
