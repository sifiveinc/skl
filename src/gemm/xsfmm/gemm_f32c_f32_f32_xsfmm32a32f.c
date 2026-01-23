// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfmm32a32f)
#error This file requires the Xsfmm32a32f extension
#endif

#include <stdbool.h>
#include <stddef.h>

#include "skl-common.h"

// params type for the fused alpha/beta scaling kernel
typedef struct {
  float alpha;
  float beta;
} alpha_beta_f32_f32;

typedef void (*fused_ker_f32_f32_t)(size_t tm, size_t tn, size_t tss, float *c,
                                    size_t rsc0, size_t csc0, size_t rsc1,
                                    size_t csc1, size_t row, size_t col,
                                    void *params);

/* Computes C := alpha * tile + beta * C, where tile is the tm x tn tile
 * specified by tss. tm and tn must be <= TE. This is a general implementation
 * that will work on all Xsfmm machines.
 */
SKL_XSFMM_IN
SKL_FUNC_PRIVATE void skl_gemm_alpha_beta_scaling_m8_f32_f32_xsfmmbase(
    size_t tm, size_t tn, float alpha, size_t tss, float beta,
    float *c, // NOLINT(readability-non-const-parameter)
    size_t rsc0, size_t csc0) {
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

/* Computes C := alpha * tile + beta * C, where tile is the tm x tn tile
 * specified by tss. tm and tn must be <= TE. This is an optimized
 * implementation when tile rows fit into m2 register groups and csc0 == 1.
 * In this case, the availability of 16 register groups allows unrolling the
 * scaling loop by a factor of 8 and software pipelining the tile row transfers
 * to overlap their latency with the scaling computation.
 */
SKL_XSFMM_IN
SKL_FUNC_PRIVATE void skl_gemm_alpha_beta_scaling_m2_f32_f32_xsfmmbase(
    size_t tm, size_t tn, float alpha, size_t tss, float beta, float *c,
    size_t rsc0) {
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

/* Computes C := alpha * tile + beta * C, where tile is the tm x tn tile
 * specified by tss. tm and tn must be <= TE.
 * To avoid redefinition errors when SKL_FUNC_PRIVATE is not set to "static",
 * the names of alpha-beta scaling functions include the input type of the
 * corresponding GEMM even though they do not depend on it.
 */
SKL_XSFMM_IN
SKL_FUNC_PRIVATE void skl_gemm_alpha_beta_scaling_f32_f32_xsfmmbase(
    size_t tm, size_t tn, size_t tss, float *c, size_t rsc0, size_t csc0,
    size_t rsc1, size_t csc1, size_t row, size_t col, void *params) {
  alpha_beta_f32_f32 *params_cast = (alpha_beta_f32_f32 *)params;
  float alpha = params_cast->alpha;
  float beta = params_cast->beta;
  float *c_tile = c + row * rsc1 + col * csc1;

  size_t te = 0;
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(te) : : "vtype", "vl");

  if (te <= __riscv_vsetvlmax_e32m2() && csc0 == 1) {
    skl_gemm_alpha_beta_scaling_m2_f32_f32_xsfmmbase(tm, tn, alpha, tss, beta,
                                                     c_tile, rsc0);
  } else {
    skl_gemm_alpha_beta_scaling_m8_f32_f32_xsfmmbase(tm, tn, alpha, tss, beta,
                                                     c_tile, rsc0, csc0);
  }
}

/* Process a tm x tn tile of c. tm and tn must be <= TE. */
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
    size_t tm, size_t tn, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb, float *c, size_t rsc, fused_ker_f32_f32_t kernel,
    void *params) {
  if (tm == 0 || tn == 0) {
    return;
  }

  /* Zero-initialize tile. */
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"

                   "sf.vtzero.t mt0\n"
                   :
                   : [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl");

  /* Accumulate matrix product into tile. */
  const float *a0 = a;
  const float *b0 = b;
  __asm__ volatile("beqz %[k], 1f\n"

                   "sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"
                   "sf.vsettk x0, %[k]\n"

                   "0:\n"
                   "addi %[k], %[k], -1\n"
                   "sf.vsettn x0, %[tm]\n"
                   "vle32.v v0, (%[a0])\n"
                   "add %[a0], %[a0], %[sa]\n"

                   "sf.vsettn x0, %[tn]\n"
                   "vle32.v v8, (%[b0])\n"
                   "add %[b0], %[b0], %[sb]\n"

                   "sf.mm.f.f mt0, v0, v8\n"
                   "bnez %[k], 0b\n"

                   "1:\n"
                   : [a0] "+&r"(a0), [b0] "+&r"(b0), [k] "+&r"(k)
                   : [sa] "r"(csa * sizeof(float)),
                     [sb] "r"(rsb * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15", "vtype", "vl",
                     "memory");

  /* Apply fused kernel. */
  const size_t mt0 = 0;

  kernel(tm, tn, mt0, c, rsc, 1, 0, 0, 0, 0, params);

  __asm__ volatile("sf.vtdiscard");
}

/* Process 4 (= 2 x 2) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_gemm_fused_2tm2tn_f32pc_f32cp_f32rcp_xsfmm32a32f(
    size_t tm, size_t tn, size_t k, const float *a, size_t csa0, size_t rsa1,
    const float *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, fused_ker_f32_f32_t kernel, void *params) {
  if (tm == 0 || tn == 0) {
    return;
  }

  /* Zero-initialize tiles. */
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"
                   "sf.vsettm x0, %[tm]\n"

                   "sf.vtzero.t mt0\n"
                   "sf.vtzero.t mt4\n"
                   "sf.vtzero.t mt8\n"
                   "sf.vtzero.t mt12\n"
                   :
                   : [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl");

  /* Accumulate matrix product into tiles. */
  const float *a0 = a;
  const float *a1 = a0 + rsa1;
  const float *b0 = b;
  const float *b1 = b0 + csb1;
  __asm__ volatile(
      "beqz %[k], 2f\n"

      "sf.vsettnt x0, %[tn], e32, w1\n"
      "sf.vsettm x0, %[tm]\n"
      "sf.vsettk x0, %[k]\n"

      "vle32.v v16, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"

      "sf.vsettn x0, %[tm]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[sa]\n"
      "sf.vsettn x0, %[tn]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -1\n"
      "vle32.v v24, (%[b1])\n"
      "add %[b1], %[b1], %[sb]\n"

      "sf.mm.f.f mt0, v0, v16\n"

      "sf.vsettn x0, %[tm]\n"
      "vle32.v v8, (%[a1])\n"
      "add %[a1], %[a1], %[sa]\n"
      "sf.vsettn x0, %[tn]\n"

      "sf.mm.f.f mt4, v0, v24\n"

      "sf.vsettn x0, %[tm]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[sa]\n"
      "sf.vsettn x0, %[tn]\n"

      "sf.mm.f.f mt8, v8, v16\n"

      "vle32.v v16, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"

      "sf.mm.f.f mt12, v8, v24\n"
      "bgeu %[k], %[i2], 0b\n"

      "1:\n"
      "vle32.v v24, (%[b1])\n"

      "sf.mm.f.f mt0, v0, v16\n"

      "sf.vsettn x0, %[tm]\n"
      "vle32.v v8, (%[a1])\n"
      "sf.vsettn x0, %[tn]\n"

      "sf.mm.f.f mt4, v0, v24\n"
      "sf.mm.f.f mt8, v8, v16\n"
      "sf.mm.f.f mt12, v8, v24\n"

      "2:\n"
      : [a0] "+&r"(a0), [a1] "+&r"(a1), [b0] "+&r"(b0), [b1] "+&r"(b1),
        [k] "+&r"(k)
      : [sa] "r"(csa0 * sizeof(float)), [sb] "r"(rsb0 * sizeof(float)),
        [tm] "r"(tm), [tn] "r"(tn), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "vtype", "vl", "memory");

  /* Apply fused kernel. */
  const size_t kShiftTile = 27;

  const size_t mt0 = 0;
  const size_t mt4 = (size_t)(4) << kShiftTile;
  const size_t mt8 = (size_t)(8) << kShiftTile;
  const size_t mt12 = (size_t)(12) << kShiftTile;

  kernel(tm, tn, mt0, c, rsc0, 1, rsc1, csc1, 0, 0, params);
  kernel(tm, tn, mt4, c, rsc0, 1, rsc1, csc1, 0, 1, params);
  kernel(tm, tn, mt8, c, rsc0, 1, rsc1, csc1, 1, 0, params);
  kernel(tm, tn, mt12, c, rsc0, 1, rsc1, csc1, 1, 1, params);

  __asm__ volatile("sf.vtdiscard");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_gemm_fused_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb, float *c, size_t rsc, fused_ker_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW=32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  size_t num_processed_by_2tm2tn_m = (m / (2 * ete)) * 2 * ete;
  size_t num_processed_by_2tm2tn_n = (n / (2 * ete)) * 2 * ete;

  size_t i = 0;
  for (; i < num_processed_by_2tm2tn_m; i += 2 * ete) {
    size_t j = 0;
    for (; j < num_processed_by_2tm2tn_n; j += 2 * ete) {
      skl_gemm_fused_2tm2tn_f32pc_f32cp_f32rcp_xsfmm32a32f(
          ete, ete, k, a + i, csa, ete, b + j, rsb, ete, c + i * rsc + j, rsc,
          ete * rsc, ete, kernel, params);
    }
    while (j < n) {
      size_t tn = 0;
      __asm__ volatile("sf.vsettnt %0, %1, e32, w1"
                       : "=r"(tn)
                       : "r"(n - j)
                       : "vtype", "vl");
      skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
          ete, tn, k, a + i, csa, b + j, rsb, c + i * rsc + j, rsc, kernel,
          params);
      skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
          ete, tn, k, a + i + ete, csa, b + j, rsb, c + (i + ete) * rsc + j,
          rsc, kernel, params);
      j += tn;
    }
  }

  while (i < m) {
    size_t tm = 0;
    __asm__ volatile("sf.vsettnt x0, x0, e32, w1\n"
                     "sf.vsettm %0, %1\n"
                     : "=r"(tm)
                     : "r"(m - i)
                     : "vtype", "vl");
    size_t j = 0;
    while (j < n) {
      size_t tn = 0;
      __asm__ volatile("sf.vsettnt %0, %1, e32, w1"
                       : "=r"(tn)
                       : "r"(n - j)
                       : "vtype", "vl");
      skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
          tm, tn, k, a + i, csa, b + j, rsb, c + i * rsc + j, rsc, kernel,
          params);
      j += tn;
    }
    i += tm;
  }
  __asm__ volatile("sf.vtdiscard");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_gemm_fused_f32pc_f32cp_f32rcp_xsfmm32a32f(
    size_t m1, size_t n1, size_t k, const float *a_pack, size_t rsa1,
    const float *b_pack, size_t csb1, float *c_pack, size_t rsc1, size_t csc1,
    fused_ker_f32_f32_t kernel, void *params) {
  if (m1 == 0 || n1 == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW=32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  size_t num_processed_by_2tm2tn_m1 = (m1 / 2) * 2;
  size_t num_processed_by_2tm2tn_n1 = (n1 / 2) * 2;

  size_t i = 0;
  for (; i < num_processed_by_2tm2tn_m1; i += 2) {
    size_t j = 0;
    for (; j < num_processed_by_2tm2tn_n1; j += 2) {
      skl_gemm_fused_2tm2tn_f32pc_f32cp_f32rcp_xsfmm32a32f(
          ete, ete, k, a_pack + i * rsa1, ete, rsa1, b_pack + j * csb1, ete,
          csb1, c_pack + i * rsc1 + j * csc1, ete, rsc1, csc1, kernel, params);
    }
    while (j < n1) {
      skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
          ete, ete, k, a_pack + i * rsa1, ete, b_pack + j * csb1, ete,
          c_pack + i * rsc1 + j * csc1, ete, kernel, params);
      skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
          ete, ete, k, a_pack + (i + 1) * rsa1, ete, b_pack + j * csb1, ete,
          c_pack + (i + 1) * rsc1 + j * csc1, ete, kernel, params);
      j += 1;
    }
  }

  while (i < m1) {
    size_t j = 0;
    while (j < n1) {
      skl_gemm_fused_1tm1tn_f32c_f32_f32_xsfmm32a32f(
          ete, ete, k, a_pack + i * rsa1, ete, b_pack + j * csb1, ete,
          c_pack + i * rsc1 + j * csc1, ete, kernel, params);
      j += 1;
    }
    i += 1;
  }
  __asm__ volatile("sf.vtdiscard");
}

SKL_XSFMM_NEW
SKL_FUNC void skl_gemm_f32c_f32_f32_xsfmm32a32f(size_t m, size_t n, size_t k,
                                                float alpha, const float *a,
                                                size_t csa, const float *b,
                                                size_t rsb, float beta,
                                                float *c, size_t rsc) {
  alpha_beta_f32_f32 params = {.alpha = alpha, .beta = beta};
  skl_gemm_fused_f32c_f32_f32_xsfmm32a32f(
      m, n, k, a, csa, b, rsb, c, rsc,
      &skl_gemm_alpha_beta_scaling_f32_f32_xsfmmbase, &params);
}

SKL_XSFMM_NEW
SKL_FUNC void skl_gemm_f32pc_f32cp_f32rcp_xsfmm32a32f(
    size_t m1, size_t n1, size_t k, float alpha, const float *a_pack,
    size_t rsa1, const float *b_pack, size_t csb1, float beta, float *c_pack,
    size_t rsc1, size_t csc1) {
  alpha_beta_f32_f32 params = {.alpha = alpha, .beta = beta};
  skl_gemm_fused_f32pc_f32cp_f32rcp_xsfmm32a32f(
      m1, n1, k, a_pack, rsa1, b_pack, csb1, c_pack, rsc1, csc1,
      &skl_gemm_alpha_beta_scaling_f32_f32_xsfmmbase, &params);
}
