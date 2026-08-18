// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfmm32a32f)
#error This file requires the Xsfmm32a32f extension
#endif

#include <riscv_vector.h>
#include <stddef.h>

#include "skl-common.h"

typedef void (*skl_gemm_tile_zero_f32_f32_xsfmmbase_t)(size_t tm,
                                                       size_t tn) SKL_XSFMM_OUT;

typedef void (*skl_gemm_tile_load_f32_f32rcptexterc_f32_xsfmmbase_t)(
    size_t m, size_t n, const void *c, size_t rsc0, size_t csc0, size_t rsc1,
    size_t csc1, size_t row1, size_t col1, size_t tss, size_t rstss,
    size_t cstss) SKL_XSFMM_OUT;

typedef void (*skl_gemm_inner_loop_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f_t)(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1) SKL_XSFMM_INOUT;

typedef void (*skl_gemm_fused_kernel_f32_f32_f32rcptexterc_xsfmmbase_t)(
    size_t tm, size_t tn, size_t tss, void *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1,
    void *params) SKL_XSFMM_IN;

/* params type for the alpha/beta scaling kernel */
typedef struct {
  float alpha;
  float beta;
} skl_gemm_alpha_beta_scaling_params_f32_f32_f32rcptexterc_xsfmmbase_t;

SKL_FUNC_PRIVATE
void skl_gemm_tile_zero_mt0_f32_f32_xsfmmbase(size_t tm,
                                              size_t tn) SKL_XSFMM_OUT {
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"
                   "sf.vtzero.t mt0\n"
                   :
                   : [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl");
}

SKL_FUNC_PRIVATE
void skl_gemm_tile_zero_mt4_f32_f32_xsfmmbase(size_t tm,
                                              size_t tn) SKL_XSFMM_OUT {
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"
                   "sf.vtzero.t mt4\n"
                   :
                   : [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl");
}

SKL_FUNC_PRIVATE
void skl_gemm_tile_zero_mt8_f32_f32_xsfmmbase(size_t tm,
                                              size_t tn) SKL_XSFMM_OUT {
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"
                   "sf.vtzero.t mt8\n"
                   :
                   : [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl");
}

SKL_FUNC_PRIVATE
void skl_gemm_tile_zero_mt12_f32_f32_xsfmmbase(size_t tm,
                                               size_t tn) SKL_XSFMM_OUT {
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"
                   "sf.vtzero.t mt12\n"
                   :
                   : [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl");
}

/* Zero out the leading m x n portion of the tile state.
 *  - tss: tile subset specifier for first tile (pattern, index ignored)
 *  - rstss, cstss: tile index strides
 */
SKL_FUNC_PRIVATE
void skl_gemm_tile_zero_f32_f32_xsfmmbase(size_t m, size_t n, size_t tss,
                                          size_t rstss,
                                          size_t cstss) SKL_XSFMM_OUT {
  if (m == 0 || n == 0) {
    return;
  }

  skl_gemm_tile_zero_f32_f32_xsfmmbase_t tile_zero_functions[] = {
      &skl_gemm_tile_zero_mt0_f32_f32_xsfmmbase,
      &skl_gemm_tile_zero_mt4_f32_f32_xsfmmbase,
      &skl_gemm_tile_zero_mt8_f32_f32_xsfmmbase,
      &skl_gemm_tile_zero_mt12_f32_f32_xsfmmbase,
  };

  const size_t kShiftTile = 27;
  size_t tzf0 = (tss >> kShiftTile) / 4;
  size_t rstzf = rstss / 4;
  size_t cstzf = cstss / 4;

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");
  size_t m1 = (m + ete - 1) / ete;
  size_t n1 = (n + ete - 1) / ete;
  size_t m_avl = m;
  for (size_t i1 = 0; i1 < m1; ++i1) {
    size_t tm = m_avl >= ete ? ete : m_avl;
    size_t n_avl = n;
    for (size_t j1 = 0; j1 < n1; ++j1) {
      size_t tn = n_avl >= ete ? ete : n_avl;
      tile_zero_functions[tzf0 + i1 * rstzf + j1 * cstzf](tm, tn);
      n_avl -= tn;
    }
    m_avl -= tm;
  }
}

/* Load the leading m x n portion of packed matrix C starting at block
 * (row1, col1) into the tile state. */
SKL_FUNC_PRIVATE
void skl_gemm_tile_load_f32_f32rcptexterc_f32_xsfmmbase(
    size_t m, size_t n, const void *c, size_t rsc0, size_t csc0, size_t rsc1,
    size_t csc1, size_t row1, size_t col1, size_t tss, size_t rstss,
    size_t cstss) SKL_XSFMM_OUT {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  const size_t kShiftTile = 27;
  const size_t kRowInc = 1;
  rstss <<= kShiftTile;
  cstss <<= kShiftTile;

  size_t m1 = (m + ete - 1) / ete;
  size_t n1 = (n + ete - 1) / ete;
  size_t m_avl = m;
  for (size_t i1 = 0; i1 < m1; ++i1) {
    size_t tm = m_avl >= ete ? ete : m_avl;
    size_t n_avl = n;
    for (size_t j1 = 0; j1 < n1; ++j1) {
      size_t tn = n_avl >= ete ? ete : n_avl;
      size_t tss_tile = tss + i1 * rstss + j1 * cstss;
      const float *c_block =
          (float *)c + (row1 + i1) * rsc1 + (col1 + j1) * csc1;
      size_t i0 = 0;
      if (csc0 == 1) {
        __asm__ volatile(
            "sf.vsettnt x0, %[tn], e32, w1\n"

            "0:\n"
            "addi %[i0], %[i0], 1\n"
            "sf.vlte32 %[tss_tile], (%[c_block])\n"
            "add %[tss_tile], %[tss_tile], %[kRowInc]\n"
            "add %[c_block], %[c_block], %[rsc0]\n"
            "bltu %[i0], %[tm], 0b\n"
            :
            [tss_tile] "+&r"(tss_tile), [c_block] "+&r"(c_block), [i0] "+&r"(i0)
            : [kRowInc] "rI"(kRowInc), [rsc0] "r"(rsc0 * sizeof(float)),
              [tm] "r"(tm), [tn] "r"(tn)
            : "vtype", "vl", "memory");
      } else {
        vfloat32m8_t cvec = __riscv_vundefined_f32m8();
        __asm__ volatile(
            "sf.vsettnt x0, %[tn], e32, w1\n"

            "0:\n"
            "addi %[i0], %[i0], 1\n"
            "vlse32.v %[cvec], (%[c_block]), %[csc0]\n"
            "sf.vtmv.t.v %[tss_tile], %[cvec]\n"
            "add %[tss_tile], %[tss_tile], %[kRowInc]\n"
            "add %[c_block], %[c_block], %[rsc0]\n"
            "bltu %[i0], %[tm], 0b\n"
            : [cvec] "=&vr"(cvec), [tss_tile] "+&r"(tss_tile),
              [c_block] "+&r"(c_block), [i0] "+&r"(i0)
            : [kRowInc] "rI"(kRowInc), [rsc0] "r"(rsc0 * sizeof(float)),
              [csc0] "r"(csc0 * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn)
            : "vtype", "vl", "memory");
      }
      n_avl -= tn;
    }
    m_avl -= tm;
  }
}

/* Apply a fused kernel to the leading m x n portion of the tile state and
 * packed matrix C.
 */
SKL_FUNC_PRIVATE
void skl_gemm_apply_fused_f32_f32_f32rcptexterc_xsfmmbase(
    size_t m, size_t n, size_t tss, size_t rstss, size_t cstss, void *c,
    size_t rsc0, size_t csc0, size_t rsc1, size_t csc1, size_t row1,
    size_t col1, skl_gemm_fused_kernel_f32_f32_f32rcptexterc_xsfmmbase_t kernel,
    void *params) SKL_XSFMM_IN {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  const size_t kShiftTile = 27;
  rstss <<= kShiftTile;
  cstss <<= kShiftTile;

  size_t m1 = (m + ete - 1) / ete;
  size_t n1 = (n + ete - 1) / ete;
  size_t m_avl = m;
  for (size_t i1 = 0; i1 < m1; ++i1) {
    size_t tm = m_avl >= ete ? ete : m_avl;
    size_t n_avl = n;
    for (size_t j1 = 0; j1 < n1; ++j1) {
      size_t tn = n_avl >= ete ? ete : n_avl;
      (*kernel)(tm, tn, tss + i1 * rstss + j1 * cstss, c, rsc0, csc0, rsc1,
                csc1, row1 + i1, col1 + j1, params);
      n_avl -= tn;
    }
    m_avl -= tm;
  }
}

/* Computes C := alpha * tile + beta * C, where:
 *  - tile is the tm x tn tile specified by tss.
 *  - tm and tn must be <= ETE.
 *
 * This is a general implementation that will work on all Xsfmm machines.
 */
SKL_FUNC_PRIVATE void skl_gemm_alpha_beta_scaling_m8_f32_f32_f32rc_xsfmmbase(
    size_t tm, size_t tn, float alpha, size_t tss, float beta,
    float *c, // NOLINT(readability-non-const-parameter)
    size_t rsc0, size_t csc0) SKL_XSFMM_IN {
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kRowInc = 1;

  size_t i = tm;
  if (csc0 == 1) {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e32, w1\n"

        "0:\n"
        "sf.vtmv.v.t v0, %[tss]\n"
        "add %[tss], %[tss], %[kRowInc]\n"
        "vle32.v v16, (%[c])\n"
        "vfmul.vf v16, v16, %[beta]\n"
        "vfmacc.vf v16, %[alpha], v0\n"
        "vse32.v v16, (%[c])\n"
        "add %[c], %[c], %[rsc0]\n"
        "addi %[i], %[i], -1\n"
        "bnez %[i], 0b"
        : [tss] "+&r"(tss), [c] "+&r"(c), [i] "+&r"(i)
        : [alpha] "f"(alpha), [beta] "f"(beta), [kRowInc] "rI"(kRowInc),
          [rsc0] "r"(rsc0 * sizeof(float)), [tn] "r"(tn)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "vtype", "vl", "memory");
  } else {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e32, w1\n"

        "0:\n"
        "sf.vtmv.v.t v0, %[tss]\n"
        "add %[tss], %[tss], %[kRowInc]\n"
        "vlse32.v v16, (%[c]), %[csc0]\n"
        "vfmul.vf v16, v16, %[beta]\n"
        "vfmacc.vf v16, %[alpha], v0\n"
        "vsse32.v v16, (%[c]), %[csc0]\n"
        "add %[c], %[c], %[rsc0]\n"
        "addi %[i], %[i], -1\n"
        "bnez %[i], 0b"
        : [tss] "+&r"(tss), [c] "+&r"(c), [i] "+&r"(i)
        : [alpha] "f"(alpha), [beta] "f"(beta), [kRowInc] "rI"(kRowInc),
          [rsc0] "r"(rsc0 * sizeof(float)), [csc0] "r"(csc0 * sizeof(float)),
          [tn] "r"(tn)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "vtype", "vl", "memory");
  }
}

/* Computes C := alpha * tile + beta * C, where:
 *  - tile is the tm x tn tile specified by tss
 *  - tm and tn must be <= ETE
 *  - csc0 == 1
 *  - tile rows fit into LMUL = 2 vector register groups
 *
 * The last condition must be checked by the caller.
 *
 * In this kernel, the availability of 16 register groups allows unrolling the
 * scaling loop by a factor of 8 and software pipelining the tile row transfers
 * to overlap their latency with the scaling computation.
 */
SKL_FUNC_PRIVATE void skl_gemm_alpha_beta_scaling_m2_f32_f32_f32_xsfmmbase(
    size_t tm, size_t tn, float alpha, size_t tss, float beta, float *c,
    size_t rsc0) SKL_XSFMM_IN {
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kRowInc = 1;

  size_t tss_0 = tss;
  size_t tss_1 = tss_0 + kRowInc;
  float *c_load_0 = c;
  float *c_load_1 = c_load_0 + rsc0;
  float *c_store_0 = c;
  float *c_store_1 = c_store_0 + rsc0;

  /* Process (tm / 8) * 8 rows. */
  size_t i = (tm / 8) * 8;
  __asm__ volatile(
      "sf.vsettnt x0, %[tn], e32, w1\n"

      "beqz %[i], 2f\n"

      "sf.vtmv.v.t v0, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vle32.v v16, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"

      "sf.vtmv.v.t v2, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vle32.v v18, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"

      "sf.vtmv.v.t v4, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vle32.v v20, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"

      "sf.vtmv.v.t v6, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vle32.v v22, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"

      "sf.vtmv.v.t v8, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vfmul.vf v16, v16, %[beta]\n"

      "sf.vtmv.v.t v10, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vfmul.vf v18, v18, %[beta]\n"

      "sf.vtmv.v.t v12, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vle32.v v24, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"
      "vfmul.vf v20, v20, %[beta]\n"

      "sf.vtmv.v.t v14, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vle32.v v26, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"
      "vfmul.vf v22, v22, %[beta]\n"

      "vfmacc.vf v16, %[alpha], v0\n"
      "addi %[i], %[i], -8\n"
      "vle32.v v28, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"
      "vfmacc.vf v18, %[alpha], v2\n"

      "vfmacc.vf v20, %[alpha], v4\n"
      "vle32.v v30, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"
      "vfmacc.vf v22, %[alpha], v6\n"
      "bltu %[i], %[i8], 1f\n"

      "0:\n" // loop body
      "vse32.v v16, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "sf.vtmv.v.t v0, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vfmul.vf v24, v24, %[beta]\n"

      "vse32.v v18, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "sf.vtmv.v.t v2, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vfmul.vf v26, v26, %[beta]\n"

      "vse32.v v20, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "sf.vtmv.v.t v4, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vle32.v v16, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"
      "vfmul.vf v28, v28, %[beta]\n"

      "vse32.v v22, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "sf.vtmv.v.t v6, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vle32.v v18, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"
      "vfmul.vf v30, v30, %[beta]\n"

      "vfmacc.vf v24, %[alpha], v8\n"
      "addi %[i], %[i], -8\n"
      "vle32.v v20, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"
      "vfmacc.vf v26, %[alpha], v10\n"

      "vfmacc.vf v28, %[alpha], v12\n"
      "vle32.v v22, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"
      "vfmacc.vf v30, %[alpha], v14\n"

      "vse32.v v24, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "sf.vtmv.v.t v8, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vfmul.vf v16, v16, %[beta]\n"

      "vse32.v v26, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "sf.vtmv.v.t v10, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vfmul.vf v18, v18, %[beta]\n"

      "vse32.v v28, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "sf.vtmv.v.t v12, %[tss_0]\n"
      "add %[tss_0], %[tss_0], %[kRowInc]\n"
      "vle32.v v24, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"
      "vfmul.vf v20, v20, %[beta]\n"

      "vse32.v v30, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "sf.vtmv.v.t v14, %[tss_1]\n"
      "add %[tss_1], %[tss_1], %[kRowInc]\n"
      "vle32.v v26, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"
      "vfmul.vf v22, v22, %[beta]\n"

      "vfmacc.vf v16, %[alpha], v0\n"
      "vle32.v v28, (%[c_load_0])\n"
      "add %[c_load_0], %[c_load_0], %[sc]\n"
      "vfmacc.vf v18, %[alpha], v2\n"

      "vfmacc.vf v20, %[alpha], v4\n"
      "vle32.v v30, (%[c_load_1])\n"
      "add %[c_load_1], %[c_load_1], %[sc]\n"
      "vfmacc.vf v22, %[alpha], v6\n"
      "bgeu %[i], %[i8], 0b\n"

      "1:\n"
      "vse32.v v16, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "vfmul.vf v24, v24, %[beta]\n"

      "vse32.v v18, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "vfmul.vf v26, v26, %[beta]\n"

      "vse32.v v20, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "vfmul.vf v28, v28, %[beta]\n"

      "vse32.v v22, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "vfmul.vf v30, v30, %[beta]\n"

      "vfmacc.vf v24, %[alpha], v8\n"
      "vfmacc.vf v26, %[alpha], v10\n"

      "vfmacc.vf v28, %[alpha], v12\n"
      "vfmacc.vf v30, %[alpha], v14\n"

      "vse32.v v24, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "vse32.v v26, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"
      "vse32.v v28, (%[c_store_0])\n"
      "add %[c_store_0], %[c_store_0], %[sc]\n"
      "vse32.v v30, (%[c_store_1])\n"
      "add %[c_store_1], %[c_store_1], %[sc]\n"

      "2:\n"
      : [tss_0] "+&r"(tss_0), [tss_1] "+&r"(tss_1), [c_load_0] "+&r"(c_load_0),
        [c_load_1] "+&r"(c_load_1), [c_store_0] "+&r"(c_store_0),
        [c_store_1] "+&r"(c_store_1), [i] "+&r"(i)
      : [alpha] "f"(alpha), [beta] "f"(beta), [kRowInc] "rI"(2 * kRowInc),
        [sc] "r"(2 * rsc0 * sizeof(float)), [tn] "r"(tn), [i8] "r"(8)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "vtype", "vl", "memory");

  /* Process the remaining rows. */
  i = tm % 8;
  if (i >= 4) {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e32, w1\n"

        "sf.vtmv.v.t v0, %[tss_0]\n"
        "add %[tss_0], %[tss_0], %[kRowInc]\n"
        "vle32.v v16, (%[c_load_0])\n"
        "add %[c_load_0], %[c_load_0], %[sc]\n"

        "sf.vtmv.v.t v2, %[tss_1]\n"
        "add %[tss_1], %[tss_1], %[kRowInc]\n"
        "vle32.v v18, (%[c_load_1])\n"
        "add %[c_load_1], %[c_load_1], %[sc]\n"

        "sf.vtmv.v.t v4, %[tss_0]\n"
        "add %[tss_0], %[tss_0], %[kRowInc]\n"
        "vle32.v v20, (%[c_load_0])\n"
        "add %[c_load_0], %[c_load_0], %[sc]\n"

        "sf.vtmv.v.t v6, %[tss_1]\n"
        "add %[tss_1], %[tss_1], %[kRowInc]\n"
        "vle32.v v22, (%[c_load_1])\n"
        "add %[c_load_1], %[c_load_1], %[sc]\n"

        "vfmul.vf v16, v16, %[beta]\n"
        "vfmul.vf v18, v18, %[beta]\n"
        "vfmul.vf v20, v20, %[beta]\n"
        "vfmul.vf v22, v22, %[beta]\n"

        "vfmacc.vf v16, %[alpha], v0\n"
        "vfmacc.vf v18, %[alpha], v2\n"
        "vfmacc.vf v20, %[alpha], v4\n"
        "vfmacc.vf v22, %[alpha], v6\n"

        "vse32.v v16, (%[c_store_0])\n"
        "add %[c_store_0], %[c_store_0], %[sc]\n"
        "vse32.v v18, (%[c_store_1])\n"
        "add %[c_store_1], %[c_store_1], %[sc]\n"
        "vse32.v v20, (%[c_store_0])\n"
        "add %[c_store_0], %[c_store_0], %[sc]\n"
        "vse32.v v22, (%[c_store_1])\n"
        "add %[c_store_1], %[c_store_1], %[sc]\n"
        : [tss_0] "+&r"(tss_0), [tss_1] "+&r"(tss_1),
          [c_load_0] "+&r"(c_load_0), [c_load_1] "+&r"(c_load_1),
          [c_store_0] "+&r"(c_store_0), [c_store_1] "+&r"(c_store_1)
        : [alpha] "f"(alpha), [beta] "f"(beta), [kRowInc] "rI"(2 * kRowInc),
          [sc] "r"(2 * rsc0 * sizeof(float)), [tn] "r"(tn)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "vtype", "vl", "memory");
    i -= 4;
  }

  if (i >= 2) {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e32, w1\n"

        "sf.vtmv.v.t v0, %[tss_0]\n"
        "add %[tss_0], %[tss_0], %[kRowInc]\n"
        "vle32.v v16, (%[c_load_0])\n"
        "add %[c_load_0], %[c_load_0], %[sc]\n"

        "sf.vtmv.v.t v2, %[tss_1]\n"
        "add %[tss_1], %[tss_1], %[kRowInc]\n"
        "vle32.v v18, (%[c_load_1])\n"
        "add %[c_load_1], %[c_load_1], %[sc]\n"

        "vfmul.vf v16, v16, %[beta]\n"
        "vfmul.vf v18, v18, %[beta]\n"

        "vfmacc.vf v16, %[alpha], v0\n"
        "vfmacc.vf v18, %[alpha], v2\n"

        "vse32.v v16, (%[c_store_0])\n"
        "add %[c_store_0], %[c_store_0], %[sc]\n"
        "vse32.v v18, (%[c_store_1])\n"
        "add %[c_store_1], %[c_store_1], %[sc]\n"
        : [tss_0] "+&r"(tss_0), [tss_1] "+&r"(tss_1),
          [c_load_0] "+&r"(c_load_0), [c_load_1] "+&r"(c_load_1),
          [c_store_0] "+&r"(c_store_0), [c_store_1] "+&r"(c_store_1)
        : [alpha] "f"(alpha), [beta] "f"(beta), [kRowInc] "rI"(2 * kRowInc),
          [sc] "r"(2 * rsc0 * sizeof(float)), [tn] "r"(tn)
        : "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "vtype", "vl",
          "memory");
    i -= 2;
  }

  if (i >= 1) {
    __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"

                     "sf.vtmv.v.t v0, %[tss_0]\n"
                     "vle32.v v16, (%[c_load_0])\n"

                     "vfmul.vf v16, v16, %[beta]\n"

                     "vfmacc.vf v16, %[alpha], v0\n"

                     "vse32.v v16, (%[c_store_0])\n"
                     :
                     : [tss_0] "r"(tss_0), [c_load_0] "r"(c_load_0),
                       [c_store_0] "r"(c_store_0), [alpha] "f"(alpha),
                       [beta] "f"(beta), [tn] "r"(tn)
                     : "v0", "v1", "v16", "v17", "vtype", "vl", "memory");
  }
}

/* Computes C := alpha * tile + beta * C, where:
 *  - tile is the tm x tn tile specified by tss
 *  - tm and tn must be <= TE
 *
 * This kernel dispatches to the optimized m2 version when possible.
 */
SKL_FUNC_PRIVATE
void skl_gemm_alpha_beta_scaling_f32_f32_f32rcptexterc_xsfmmbase(
    size_t tm, size_t tn, size_t tss, void *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1,
    void *params) SKL_XSFMM_IN {
  /* Use row-major code when rsc0 == 1 if possible. */
  // Check that tss specifies the first row or column of a tile (bits 23:0).
  if (rsc0 == 1 && csc0 != 1 && ((tss & (size_t)0xFFFFFF) == 0)) {
    // Swap tm and tn.
    size_t temp = tm;
    tm = tn;
    tn = temp;
    // Toggle the pattern (bit 24).
    tss ^= ((size_t)1 << 24);
    // Swap rsc0 and csc0.
    temp = rsc0;
    rsc0 = csc0;
    csc0 = temp;
  }

  skl_gemm_alpha_beta_scaling_params_f32_f32_f32rcptexterc_xsfmmbase_t
      *params_cast =
          (skl_gemm_alpha_beta_scaling_params_f32_f32_f32rcptexterc_xsfmmbase_t
               *)params;
  float alpha = params_cast->alpha;
  float beta = params_cast->beta;
  float *c_block = (float *)c + row1 * rsc1 + col1 * csc1;

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  if (ete <= (size_t)__riscv_vsetvlmax_e32m2() && csc0 == 1) {
    skl_gemm_alpha_beta_scaling_m2_f32_f32_f32_xsfmmbase(tm, tn, alpha, tss,
                                                         beta, c_block, rsc0);
  } else {
    skl_gemm_alpha_beta_scaling_m8_f32_f32_f32rc_xsfmmbase(
        tm, tn, alpha, tss, beta, c_block, rsc0, csc0);
  }
}

/* Computes an m x n x k matrix product A * B for packed A and B.
 * m and n must be <= ETE.
 */
SKL_FUNC_PRIVATE void
skl_gemm_inner_loop_1x1_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a,
    __attribute__((unused)) size_t rsa1, size_t csa1, const float *b,
    size_t rsb1, __attribute__((unused)) size_t csb1) SKL_XSFMM_INOUT {
  if (m == 0 || n == 0) {
    return;
  }

  const float *a0 = a;
  const float *b0 = b;
  __asm__ volatile("beqz %[k], 1f\n"

                   "sf.vsettnt x0, %[n], e32, w1\n"
                   "sf.vsettm x0, %[m]\n"
                   "sf.vsettk x0, %[k]\n"

                   "0:\n"
                   "addi %[k], %[k], -1\n"
                   "sf.vsettn x0, %[m]\n"
                   "vle32.v v0, (%[a0])\n"
                   "add %[a0], %[a0], %[csa1]\n"

                   "sf.vsettn x0, %[n]\n"
                   "vle32.v v8, (%[b0])\n"
                   "add %[b0], %[b0], %[rsb1]\n"

                   "sf.mm.f.f mt0, v0, v8\n"
                   "bnez %[k], 0b\n"

                   "1:\n"
                   : [a0] "+&r"(a0), [b0] "+&r"(b0), [k] "+&r"(k)
                   : [csa1] "r"(csa1 * sizeof(float)),
                     [rsb1] "r"(rsb1 * sizeof(float)), [m] "r"(m), [n] "r"(n)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15", "vtype", "vl",
                     "memory");
}

/* m <= ETE, n <= 2 * ETE. */
SKL_FUNC_PRIVATE void
skl_gemm_inner_loop_1x2_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a,
    __attribute__((unused)) size_t rsa1, size_t csa1, const float *b,
    size_t rsb1, size_t csb1) SKL_XSFMM_INOUT {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");
  size_t tn0 = n >= ete ? ete : n;
  size_t tn1 = n - tn0;

  const float *a0 = a;
  const float *b0 = b;
  const float *b1 = b0 + csb1;
  __asm__ volatile(
      "beqz %[k], 2f\n"

      "sf.vsettnt x0, %[tn0], e32, w1\n"
      "sf.vsettm x0, %[m]\n"
      "sf.vsettk x0, %[k]\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -1\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"
      "add %[b1], %[b1], %[rsb1]\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "bgeu %[k], %[i2], 0b\n"

      "1:\n"
      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"
      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "2:\n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [b1] "+&r"(b1), [k] "+&r"(k)
      : [csa1] "r"(csa1 * sizeof(float)), [rsb1] "r"(rsb1 * sizeof(float)),
        [m] "r"(m), [tn0] "r"(tn0), [tn1] "r"(tn1), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "vtype", "vl", "memory");
}

/* m <= ETE, n <= 3 * ETE. */
SKL_FUNC_PRIVATE void
skl_gemm_inner_loop_1x3_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a,
    __attribute__((unused)) size_t rsa1, size_t csa1, const float *b,
    size_t rsb1, size_t csb1) SKL_XSFMM_INOUT {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");
  size_t tn0 = n >= ete ? ete : n;
  n -= tn0;
  size_t tn1 = n >= ete ? ete : n;
  size_t tn2 = n - tn1;

  const float *a0 = a;
  const float *b0 = b;
  const float *b1 = b0 + csb1;
  const float *b2 = b0 + 2 * csb1;
  __asm__ volatile(
      "beqz %[k], 2f\n"

      "sf.vsettnt x0, %[tn0], e32, w1\n"
      "sf.vsettm x0, %[m]\n"
      "sf.vsettk x0, %[k]\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -1\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"
      "add %[b1], %[b1], %[rsb1]\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "sf.vsettn x0, %[tn2]\n"
      "vle32.v v24, (%[b2])\n"
      "add %[b2], %[b2], %[rsb1]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "sf.vsettn x0, %[tn2]\n"
      "sf.mm.f.f mt8, v0, v24\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "bgeu %[k], %[i2], 0b\n"
      "1:\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "sf.vsettn x0, %[tn2]\n"
      "vle32.v v24, (%[b2])\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"
      "sf.vsettn x0, %[tn2]\n"
      "sf.mm.f.f mt8, v0, v24\n"

      "2:\n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [b1] "+&r"(b1), [b2] "+&r"(b2),
        [k] "+&r"(k)
      : [csa1] "r"(csa1 * sizeof(float)), [rsb1] "r"(rsb1 * sizeof(float)),
        [m] "r"(m), [tn0] "r"(tn0), [tn1] "r"(tn1), [tn2] "r"(tn2), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "vtype", "vl", "memory");
}

/* m <= ETE, n <= 4 * ETE. */
SKL_FUNC_PRIVATE void
skl_gemm_inner_loop_1x4_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a,
    __attribute__((unused)) size_t rsa1, size_t csa1, const float *b,
    size_t rsb1, size_t csb1) SKL_XSFMM_INOUT {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");
  size_t tn0 = n >= ete ? ete : n;
  n -= tn0;
  size_t tn1 = n >= ete ? ete : n;
  n -= tn1;
  size_t tn2 = n >= ete ? ete : n;
  size_t tn3 = n - tn2;

  const float *a0 = a;
  const float *b0 = b;
  const float *b1 = b0 + csb1;
  const float *b2 = b0 + 2 * csb1;
  const float *b3 = b0 + 3 * csb1;
  __asm__ volatile(
      "beqz %[k], 2f\n"

      "sf.vsettnt x0, %[tn0], e32, w1\n"
      "sf.vsettm x0, %[m]\n"
      "sf.vsettk x0, %[k]\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -1\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"
      "add %[b1], %[b1], %[rsb1]\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "sf.vsettn x0, %[tn2]\n"
      "vle32.v v8, (%[b2])\n"
      "add %[b2], %[b2], %[rsb1]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "sf.vsettn x0, %[tn3]\n"
      "vle32.v v16, (%[b3])\n"
      "add %[b3], %[b3], %[rsb1]\n"

      "sf.vsettn x0, %[tn2]\n"
      "sf.mm.f.f mt8, v0, v8\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "sf.vsettn x0, %[tn3]\n"
      "sf.mm.f.f mt12, v0, v16\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "bgeu %[k], %[i2], 0b\n"
      "1:\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "sf.vsettn x0, %[tn2]\n"
      "vle32.v v8, (%[b2])\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "sf.vsettn x0, %[tn3]\n"
      "vle32.v v16, (%[b3])\n"

      "sf.vsettn x0, %[tn2]\n"
      "sf.mm.f.f mt8, v0, v8\n"
      "sf.vsettn x0, %[tn3]\n"
      "sf.mm.f.f mt12, v0, v16\n"

      "2:\n"
      : [a0] "+&r"(a0), [b0] "+&r"(b0), [b1] "+&r"(b1), [b2] "+&r"(b2),
        [b3] "+&r"(b3), [k] "+&r"(k)
      : [csa1] "r"(csa1 * sizeof(float)), [rsb1] "r"(rsb1 * sizeof(float)),
        [m] "r"(m), [tn0] "r"(tn0), [tn1] "r"(tn1), [tn2] "r"(tn2),
        [tn3] "r"(tn3), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "vtype", "vl", "memory");
}

/* m <= 2 * ETE, n <= 2 * ETE. */
SKL_FUNC_PRIVATE void
skl_gemm_inner_loop_2x2_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1) SKL_XSFMM_INOUT {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");
  size_t tm0 = m >= ete ? ete : m;
  size_t tm1 = m - tm0;
  size_t tn0 = n >= ete ? ete : n;
  size_t tn1 = n - tn0;

  const float *a0 = a;
  const float *a1 = a0 + rsa1;
  const float *b0 = b;
  const float *b1 = b0 + csb1;
  __asm__ volatile(
      "beqz %[k], 2f\n"

      "sf.vsettnt x0, %[tn0], e32, w1\n"
      "sf.vsettm x0, %[tm0]\n"
      "sf.vsettk x0, %[k]\n"

      "vle32.v v16, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "sf.vsettn x0, %[tm0]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "sf.vsettn x0, %[tn1]\n"
      "bltu %[k], %[i2], 2f\n"

      // dispatch to general inner loop if tm0, tm1, tn0, tn1 are not all equal
      "bne %[tm0], %[tm1], 1f\n"
      "bne %[tn0], %[tn1], 1f\n"
      "bne %[tm0], %[tn0], 1f\n"

      // specialized inner loop (tm0 == tm1 == tn0 == tn1)
      "0:\n"
      "addi %[k], %[k], -1\n"
      "vle32.v v24, (%[b1])\n"
      "add %[b1], %[b1], %[rsb1]\n"

      "sf.mm.f.f mt0, v0, v16\n"

      "vle32.v v8, (%[a1])\n"
      "add %[a1], %[a1], %[csa1]\n"

      "sf.mm.f.f mt4, v0, v24\n"

      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "sf.mm.f.f mt8, v8, v16\n"

      "vle32.v v16, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "sf.mm.f.f mt12, v8, v24\n"
      "bgeu %[k], %[i2], 0b\n"
      "j 2f\n"

      // general inner loop
      "1:\n"
      "addi %[k], %[k], -1\n"
      "vle32.v v24, (%[b1])\n"
      "add %[b1], %[b1], %[rsb1]\n"

      "sf.vsettm x0, %[tm0]\n"
      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v16\n"

      "sf.vsettn x0, %[tm1]\n"
      "vle32.v v8, (%[a1])\n"
      "add %[a1], %[a1], %[csa1]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v24\n"

      "sf.vsettn x0, %[tm0]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[csa1]\n"

      "sf.vsettm x0, %[tm1]\n"
      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt8, v8, v16\n"

      "vle32.v v16, (%[b0])\n"
      "add %[b0], %[b0], %[rsb1]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt12, v8, v24\n"
      "bgeu %[k], %[i2], 1b\n"

      "2:\n"
      "vle32.v v24, (%[b1])\n"

      "sf.vsettm x0, %[tm0]\n"
      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v16\n"

      "sf.vsettn x0, %[tm1]\n"
      "vle32.v v8, (%[a1])\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v24\n"
      "sf.vsettm x0, %[tm1]\n"
      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt8, v8, v16\n"
      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt12, v8, v24\n"

      "3:\n"
      : [a0] "+&r"(a0), [a1] "+&r"(a1), [b0] "+&r"(b0), [b1] "+&r"(b1),
        [k] "+&r"(k)
      : [csa1] "r"(csa1 * sizeof(float)), [rsb1] "r"(rsb1 * sizeof(float)),
        [tm0] "r"(tm0), [tm1] "r"(tm1), [tn0] "r"(tn0), [tn1] "r"(tn1),
        [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "vtype", "vl", "memory");
}

/* Computes A * B (accum == false) or C + A * B (accum == true) and applies a
 * fused kernel.
 *  - m1, n1: tiling dimensions
 *  - inner_loop: inner loop function used to compute A * B (or A^T * B^T if
 *    m1 > n1)
 *  - m, n, k: matrix dimensions. m must be <= m1 * ETE, n <= n1 * ETE.
 *  - a, rsa1, csa1: packed matrix A and its strides
 *  - b, rsb1, csb1: packed matrix B and its strides
 *  - c. rsc0, csc0, rsc1, csc1: packed matrix C and its strides
 *  - row1, col1: block-row and -column indices for first block of C
 *  - kernel: pointer to fused kernel
 *  - params: pointer to params struct for fused kernel
 *
 * This function will compute partial blocks of C if m or n is not multiple of
 * ETE. If m1 > n1, then the n1 x m1 inner loop function should be called.
 */
SKL_FUNC_PRIVATE void
skl_gemm_apply_tiling_f32rcptex1c_f32rcp1xte_f32_f32rcptexterc_xsfmm32a32f(
    skl_gemm_tile_load_f32_f32rcptexterc_f32_xsfmmbase_t tile_load, size_t m1,
    size_t n1,
    skl_gemm_inner_loop_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f_t inner_loop,
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1, void *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, bool accum,
    skl_gemm_fused_kernel_f32_f32_f32rcptexterc_xsfmmbase_t kernel,
    void *params) SKL_XSFMM_NEW {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t kShiftPattern = 24;
  const size_t mt0 = 0;
  const size_t mt0c = 0 | 1 << kShiftPattern;

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  size_t rstss = 0;
  size_t cstss = 0;
  bool trans = m1 > n1;
  if (m1 == 1) {
    cstss = 4;
  } else if (n1 == 1) {
    rstss = 4;
  } else { // 2 x 2
    rstss = 8;
    cstss = 4;
  }

  if (accum) {
    tile_load(m, n, c, rsc0, csc0, rsc1, csc1, row1, col1, trans ? mt0c : mt0,
              rstss, cstss);
  } else {
    if (trans) {
      // NOLINTBEGIN(readability-suspicious-call-argument)
      skl_gemm_tile_zero_f32_f32_xsfmmbase(n, m, mt0, cstss, rstss);
      // NOLINTEND(readability-suspicious-call-argument)
    } else {
      skl_gemm_tile_zero_f32_f32_xsfmmbase(m, n, mt0, rstss, cstss);
    }
  }

  if (trans) {
    inner_loop(n, m, k, b, csb1, rsb1, a, csa1, rsa1);
  } else {
    inner_loop(m, n, k, a, rsa1, csa1, b, rsb1, csb1);
  }

  skl_gemm_apply_fused_f32_f32_f32rcptexterc_xsfmmbase(
      m, n, trans ? mt0c : mt0, rstss, cstss, c, rsc0, csc0, rsc1, csc1, row1,
      col1, kernel, params);

  __asm__ volatile("sf.vtdiscard");
}

#define SKL_GEMM_TILE(M1, N1, M1XN1, M, N, ROW1, COL1)                          \
  do {                                                                          \
    skl_gemm_apply_tiling_f32rcptex1c_f32rcp1xte_f32_f32rcptexterc_xsfmm32a32f( \
        tile_load, M1, N1,                                                      \
        skl_gemm_inner_loop_##M1XN1##_f32rcptex1c_f32rcp1xte_f32_xsfmm32a32f,   \
        M, N, k, a + (ROW1) * rsa1, rsa1, csa1, b + (COL1) * csb1, rsb1, csb1,  \
        c, rsc0, csc0, rsc1, csc1, ROW1, COL1, accum, kernel, params);          \
  } while (0)

/* Computes A * B (accum == false) or C + A * B (accum == true) and applies a
 * fused kernel.
 */
SKL_FUNC_PRIVATE void
skl_gemm_fused_f32rcptex1c_f32rcp1xte_f32rcptexterc_xsfmm32a32f(
    skl_gemm_tile_load_f32_f32rcptexterc_f32_xsfmmbase_t tile_load, size_t m,
    size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1, void *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, bool accum,
    skl_gemm_fused_kernel_f32_f32_f32rcptexterc_xsfmmbase_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  size_t m1 = (m + ete - 1) / ete;
  size_t n1 = (n + ete - 1) / ete;
  size_t i1 = 0;
  size_t m_avl = m;
  for (; i1 + 3 < m1; i1 += 4) {
    size_t m_vl = m_avl >= 4 * ete ? 4 * ete : m_avl;
    size_t n_avl = n;
    size_t j1 = 0;
    for (; j1 + 1 < n1; j1 += 2) {
      size_t n_vl = n_avl >= 2 * ete ? 2 * ete : n_avl;
      SKL_GEMM_TILE(2, 2, 2x2, 2 * ete, n_vl, i1, j1);
      SKL_GEMM_TILE(2, 2, 2x2, m_vl - 2 * ete, n_vl, i1 + 2, j1);
      n_avl -= n_vl;
    }
    if (j1 < n1) {
      SKL_GEMM_TILE(4, 1, 1x4, m_vl, n_avl, i1, j1);
    }
    m_avl -= m_vl;
  }

  for (; i1 + 1 < m1; i1 += 2) {
    size_t m_vl = m_avl >= 2 * ete ? 2 * ete : m_avl;
    size_t n_avl = n;
    size_t j1 = 0;
    for (; j1 + 1 < n1; j1 += 2) {
      size_t n_vl = n_avl >= 2 * ete ? 2 * ete : n_avl;
      SKL_GEMM_TILE(2, 2, 2x2, m_vl, n_vl, i1, j1);
      n_avl -= n_vl;
    }
    if (j1 < n1) {
      SKL_GEMM_TILE(2, 1, 1x2, m_vl, n_avl, i1, j1);
    }
    m_avl -= m_vl;
  }

  if (i1 < m1) {
    size_t n_avl = n;
    size_t j1 = 0;
    for (; j1 + 3 < n1; j1 += 4) {
      size_t n_vl = n_avl >= 4 * ete ? 4 * ete : n_avl;
      SKL_GEMM_TILE(1, 4, 1x4, m_avl, n_vl, i1, j1);
      n_avl -= n_vl;
    }
    switch (n1 - j1) {
    case 3:
      SKL_GEMM_TILE(1, 3, 1x3, m_avl, n_avl, i1, j1);
      break;
    case 2:
      SKL_GEMM_TILE(1, 2, 1x2, m_avl, n_avl, i1, j1);
      break;
    case 1:
      SKL_GEMM_TILE(1, 1, 1x1, m_avl, n_avl, i1, j1);
      break;
    default:
      break;
    }
  }
}
#undef SKL_GEMM_TILE

/* Computes C = alpha * A * B + beta * C when A is column-major. */
SKL_FUNC void skl_gemm_f32c_f32_f32_xsfmm32a32f(size_t m, size_t n, size_t k,
                                                float alpha, const float *a,
                                                size_t csa, const float *b,
                                                size_t rsb, float beta,
                                                float *c, size_t rsc) {
  skl_gemm_alpha_beta_scaling_params_f32_f32_f32rcptexterc_xsfmmbase_t params =
      {.alpha = alpha, .beta = beta};

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  skl_gemm_fused_f32rcptex1c_f32rcp1xte_f32rcptexterc_xsfmm32a32f(
      skl_gemm_tile_load_f32_f32rcptexterc_f32_xsfmmbase, m, n, k, a, ete, csa,
      b, rsb, ete, c, rsc, 1, ete * rsc, ete, false,
      skl_gemm_alpha_beta_scaling_f32_f32_f32rcptexterc_xsfmmbase, &params);
}

/* Computes C = alpha * A * B + beta * C for packed matrices A, B, and C. */
SKL_FUNC void skl_gemm_f32rcptex1c_f32rcp1xte_f32rcptexterc_xsfmm32a32f(
    size_t m1, size_t n1, size_t k, float alpha, const float *a, size_t rsa1,
    size_t csa1, const float *b, size_t rsb1, size_t csb1, float beta, float *c,
    size_t rsc0, size_t csc0, size_t rsc1, size_t csc1) {
  skl_gemm_alpha_beta_scaling_params_f32_f32_f32rcptexterc_xsfmmbase_t params =
      {.alpha = alpha, .beta = beta};

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  skl_gemm_fused_f32rcptex1c_f32rcp1xte_f32rcptexterc_xsfmm32a32f(
      skl_gemm_tile_load_f32_f32rcptexterc_f32_xsfmmbase, m1 * ete, n1 * ete, k,
      a, rsa1, csa1, b, rsb1, csb1, c, rsc0, csc0, rsc1, csc1, false,
      skl_gemm_alpha_beta_scaling_f32_f32_f32rcptexterc_xsfmmbase, &params);
}
