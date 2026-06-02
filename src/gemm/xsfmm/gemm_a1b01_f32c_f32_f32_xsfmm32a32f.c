// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfmm32a32f)
#error This file requires the Xsfmm32a32f extension
#endif

#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

typedef void (*fused_f32_f32_t)(size_t m, size_t n, size_t tss, float *c,
                                size_t rsc0, size_t csc0, size_t rsc1,
                                size_t csc1, size_t row1, size_t col1,
                                void *params);

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_mt0_e32_xsfmmbase(size_t m, size_t n) {
  __asm__ volatile("sf.vsettnt x0, %[n], e32, w1\n"
                   "sf.vsettm x0, %[m]\n"
                   "sf.vtzero.t mt0\n"
                   :
                   : [m] "r"(m), [n] "r"(n)
                   : "vtype", "vl");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_mt4_e32_xsfmmbase(size_t m, size_t n) {
  __asm__ volatile("sf.vsettnt x0, %[n], e32, w1\n"
                   "sf.vsettm x0, %[m]\n"
                   "sf.vtzero.t mt4\n"
                   :
                   : [m] "r"(m), [n] "r"(n)
                   : "vtype", "vl");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_mt8_e32_xsfmmbase(size_t m, size_t n) {
  __asm__ volatile("sf.vsettnt x0, %[n], e32, w1\n"
                   "sf.vsettm x0, %[m]\n"
                   "sf.vtzero.t mt8\n"
                   :
                   : [m] "r"(m), [n] "r"(n)
                   : "vtype", "vl");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_mt12_e32_xsfmmbase(size_t m, size_t n) {
  __asm__ volatile("sf.vsettnt x0, %[n], e32, w1\n"
                   "sf.vsettm x0, %[m]\n"
                   "sf.vtzero.t mt12\n"
                   :
                   : [m] "r"(m), [n] "r"(n)
                   : "vtype", "vl");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_zero_e32_xsfmmbase(size_t m, size_t n, size_t tss, size_t rstss,
                                 size_t cstss) {
  if (m == 0 || n == 0) {
    return;
  }

  void (*tile_zero_functions[])(size_t, size_t) = {
      skl_tile_zero_mt0_e32_xsfmmbase,
      skl_tile_zero_mt4_e32_xsfmmbase,
      skl_tile_zero_mt8_e32_xsfmmbase,
      skl_tile_zero_mt12_e32_xsfmmbase,
  };

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
      tile_zero_functions[tss + i1 * rstzf + j1 * cstzf](tm, tn);
      n_avl -= tn;
    }
    m_avl -= tm;
  }
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE
void skl_tile_load_e32rcp_xsfmmbase(size_t m, size_t n, uint32_t *c,
                                    size_t rsc0, size_t rsc1, size_t csc1,
                                    size_t tss, size_t rstss, size_t cstss) {
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
      size_t tss_block = tss + i1 * rstss + j1 * cstss;
      uint32_t *c_block = c + i1 * rsc1 + j1 * csc1;
      size_t i0 = 0;
      __asm__ volatile(
          "sf.vsettnt x0, %[tn], e32, w1\n"

          "0:\n"
          "addi %[i0], %[i0], 1\n"
          "sf.vlte32 %[tss_block], (%[c_block])\n"
          "add %[tss_block], %[tss_block], %[kRowInc]\n"
          "add %[c_block], %[c_block], %[sc]\n"
          "bltu %[i0], %[tm], 0b\n"
          :
          [tss_block] "+&r"(tss_block), [c_block] "+&r"(c_block), [i0] "+&r"(i0)
          : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc0 * sizeof(float)),
            [tm] "r"(tm), [tn] "r"(tn)
          : "vtype", "vl", "memory");
      n_avl -= tn;
    }
    m_avl -= tm;
  }
}

SKL_XSFMM_IN
SKL_FUNC_PRIVATE
void skl_tile_store_f32rcp_xsfmmbase(size_t tm, size_t tn, size_t tss, float *c,
                                     size_t rsc0, size_t csc0, size_t rsc1,
                                     size_t csc1, size_t row1, size_t col1,
                                     void *params) {
  (void)params;

  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kRowInc = 1;

  float *c_block = c + row1 * rsc1 + col1 * csc1;

  if (csc0 == 1) {
    size_t i0 = 0;
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e32, w1\n"

        "0:\n"
        "addi %[i0], %[i0], 1\n"
        "sf.vste32 %[tss], (%[c_block])\n"
        "add %[tss], %[tss], %[kRowInc]\n"
        "add %[c_block], %[c_block], %[sc]\n"
        "bltu %[i0], %[tm], 0b\n"
        : [tss] "+&r"(tss), [c_block] "+&r"(c_block), [i0] "+&r"(i0)
        : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm),
          [tn] "r"(tn)
        : "vtype", "vl", "memory");
  } else {
    vuint32m8_t vec = __riscv_vundefined_u32m8();
    size_t i0 = 0;
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e32, w1\n"

        "0:\n"
        "addi %[i0], %[i0], 1\n"
        "sf.vtmv.v.t %[vec], %[tss]\n"
        "add %[tss], %[tss], %[kRowInc]\n"
        "vsse32.v %[vec], (%[c_block]), %[csc0]\n"
        "add %[c_block], %[c_block], %[rsc0]\n"
        "bltu %[i0], %[tm], 0b\n"
        : [vec] "=&vr"(vec), [tss] "+&r"(tss), [c_block] "+&r"(c_block),
          [i0] "+&r"(i0)
        : [kRowInc] "rI"(kRowInc), [rsc0] "r"(rsc0 * sizeof(float)),
          [csc0] "r"(csc0 * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn)
        : "vtype", "vl", "memory");
  }
}

SKL_XSFMM_IN
SKL_FUNC_PRIVATE
void skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(
    size_t m, size_t n, size_t tss, size_t rstss, size_t cstss, float *c,
    size_t rsc0, size_t csc0, size_t rsc1, size_t csc1, size_t row1,
    size_t col1, fused_f32_f32_t kernel, void *params) {
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

/* Process a tm x tn tile of c. tm and tn must be <= TE. */
SKL_XSFMM_INOUT
SKL_FUNC_PRIVATE void
skl_gemm_1x1_f32c_f32_f32_xsfmm32a32f(size_t m, size_t n, size_t k,
                                      const float *a, size_t csa,
                                      const float *b, size_t rsb) {
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
                   "add %[a0], %[a0], %[sa]\n"

                   "sf.vsettn x0, %[n]\n"
                   "vle32.v v8, (%[b0])\n"
                   "add %[b0], %[b0], %[sb]\n"

                   "sf.mm.f.f mt0, v0, v8\n"
                   "bnez %[k], 0b\n"

                   "1:\n"
                   : [a0] "+&r"(a0), [b0] "+&r"(b0), [k] "+&r"(k)
                   : [sa] "r"(csa * sizeof(float)),
                     [sb] "r"(rsb * sizeof(float)), [m] "r"(m), [n] "r"(n)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15", "vtype", "vl",
                     "memory");
}

SKL_FUNC_PRIVATE void skl_gemm_1x2_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb1, size_t csb1) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");
  size_t tn0 = n >= ete ? ete : n;
  size_t tn1 = n - tn0;

  /* Accumulate matrix product into tiles. */
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
      "add %[a0], %[a0], %[sa]\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -1\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"
      "add %[b1], %[b1], %[sb]\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[sa]\n"

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
      : [sa] "r"(csa * sizeof(float)), [sb] "r"(rsb1 * sizeof(float)),
        [m] "r"(m), [tn0] "r"(tn0), [tn1] "r"(tn1), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "vtype", "vl", "memory");
}

SKL_FUNC_PRIVATE void skl_gemm_1x3_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb1, size_t csb1) {
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
      "add %[a0], %[a0], %[sa]\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -1\n"

      "sf.vsettn x0, %[tn1]\n"
      "vle32.v v16, (%[b1])\n"
      "add %[b1], %[b1], %[sb]\n"

      "sf.vsettn x0, %[tn0]\n"
      "sf.mm.f.f mt0, v0, v8\n"

      "sf.vsettn x0, %[tn2]\n"
      "vle32.v v24, (%[b2])\n"
      "add %[b2], %[b2], %[sb]\n"

      "sf.vsettn x0, %[tn1]\n"
      "sf.mm.f.f mt4, v0, v16\n"

      "sf.vsettn x0, %[tn0]\n"
      "vle32.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"

      "sf.vsettn x0, %[tn2]\n"
      "sf.mm.f.f mt8, v0, v24\n"

      "sf.vsettn x0, %[m]\n"
      "vle32.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[sa]\n"

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
      : [sa] "r"(csa * sizeof(float)), [sb] "r"(rsb1 * sizeof(float)),
        [m] "r"(m), [tn0] "r"(tn0), [tn1] "r"(tn1), [tn2] "r"(tn2), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "vtype", "vl", "memory");
}

SKL_FUNC_PRIVATE void skl_gemm_1x4_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb1, size_t csb1) {
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
  __asm__ volatile("beqz %[k], 2f\n"

                   "sf.vsettnt x0, %[tn0], e32, w1\n"
                   "sf.vsettm x0, %[m]\n"
                   "sf.vsettk x0, %[k]\n"

                   "sf.vsettn x0, %[m]\n"
                   "vle32.v v0, (%[a0])\n"
                   "add %[a0], %[a0], %[sa]\n"

                   "sf.vsettn x0, %[tn0]\n"
                   "vle32.v v8, (%[b0])\n"
                   "add %[b0], %[b0], %[sb]\n"

                   "bltu %[k], %[i2], 1f\n"

                   "0:\n"
                   "addi %[k], %[k], -1\n"

                   "sf.vsettn x0, %[tn1]\n"
                   "vle32.v v16, (%[b1])\n"
                   "add %[b1], %[b1], %[sb]\n"

                   "sf.vsettn x0, %[tn0]\n"
                   "sf.mm.f.f mt0, v0, v8\n"

                   "sf.vsettn x0, %[tn2]\n"
                   "vle32.v v8, (%[b2])\n"
                   "add %[b2], %[b2], %[sb]\n"

                   "sf.vsettn x0, %[tn1]\n"
                   "sf.mm.f.f mt4, v0, v16\n"

                   "sf.vsettn x0, %[tn3]\n"
                   "vle32.v v16, (%[b3])\n"
                   "add %[b3], %[b3], %[sb]\n"

                   "sf.vsettn x0, %[tn2]\n"
                   "sf.mm.f.f mt8, v0, v8\n"

                   "sf.vsettn x0, %[tn0]\n"
                   "vle32.v v8, (%[b0])\n"
                   "add %[b0], %[b0], %[sb]\n"

                   "sf.vsettn x0, %[tn3]\n"
                   "sf.mm.f.f mt12, v0, v16\n"

                   "sf.vsettn x0, %[m]\n"
                   "vle32.v v0, (%[a0])\n"
                   "add %[a0], %[a0], %[sa]\n"

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
                   : [a0] "+&r"(a0), [b0] "+&r"(b0), [b1] "+&r"(b1),
                     [b2] "+&r"(b2), [b3] "+&r"(b3), [k] "+&r"(k)
                   : [sa] "r"(csa * sizeof(float)),
                     [sb] "r"(rsb1 * sizeof(float)), [m] "r"(m), [tn0] "r"(tn0),
                     [tn1] "r"(tn1), [tn2] "r"(tn2), [tn3] "r"(tn3), [i2] "r"(2)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                     "v18", "v19", "v20", "v21", "v22", "v23", "vtype", "vl",
                     "memory");
}

/* Process 4 (= 2 x 2) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_2x2_f32rcp_f32rcp_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1) {
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
  __asm__ volatile("beqz %[k], 2f\n"

                   "sf.vsettnt x0, %[tn0], e32, w1\n"
                   "sf.vsettm x0, %[tm0]\n"
                   "sf.vsettk x0, %[k]\n"

                   "vle32.v v16, (%[b0])\n"
                   "add %[b0], %[b0], %[sb]\n"

                   "sf.vsettn x0, %[tm0]\n"
                   "vle32.v v0, (%[a0])\n"
                   "add %[a0], %[a0], %[sa]\n"

                   "bltu %[k], %[i2], 1f\n"

                   "0:\n"
                   "addi %[k], %[k], -1\n"
                   "sf.vsettn x0, %[tn1]\n"
                   "vle32.v v24, (%[b1])\n"
                   "add %[b1], %[b1], %[sb]\n"

                   "sf.vsettm x0, %[tm0]\n"
                   "sf.vsettn x0, %[tn0]\n"
                   "sf.mm.f.f mt0, v0, v16\n"

                   "sf.vsettn x0, %[tm1]\n"
                   "vle32.v v8, (%[a1])\n"
                   "add %[a1], %[a1], %[sa]\n"

                   "sf.vsettn x0, %[tn1]\n"
                   "sf.mm.f.f mt4, v0, v24\n"

                   "sf.vsettn x0, %[tm0]\n"
                   "vle32.v v0, (%[a0])\n"
                   "add %[a0], %[a0], %[sa]\n"

                   "sf.vsettm x0, %[tm1]\n"
                   "sf.vsettn x0, %[tn0]\n"
                   "sf.mm.f.f mt8, v8, v16\n"

                   "vle32.v v16, (%[b0])\n"
                   "add %[b0], %[b0], %[sb]\n"

                   "sf.vsettn x0, %[tn1]\n"
                   "sf.mm.f.f mt12, v8, v24\n"
                   "bgeu %[k], %[i2], 0b\n"

                   "1:\n"
                   "sf.vsettn x0, %[tn1]\n"
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

                   "2:\n"
                   : [a0] "+&r"(a0), [a1] "+&r"(a1), [b0] "+&r"(b0),
                     [b1] "+&r"(b1), [k] "+&r"(k)
                   : [sa] "r"(csa1 * sizeof(float)),
                     [sb] "r"(rsb1 * sizeof(float)), [tm0] "r"(tm0),
                     [tm1] "r"(tm1), [tn0] "r"(tn0), [tn1] "r"(tn1), [i2] "r"(2)
                   : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                     "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                     "v26", "v27", "v28", "v29", "v30", "v31", "vtype", "vl",
                     "memory");
}

/* Process a tm x tn tile of c. tm and tn must be <= TE. */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm1tn_a1b01_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb, float *c, size_t rsc0, size_t rsc1, size_t csc1, size_t row1,
    size_t col1, bool accum, fused_f32_f32_t kernel, void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t mt0 = 0;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0, 0, 0);
  } else {
    skl_tile_zero_e32_xsfmmbase(m, n, mt0, 0, 0);
  }

  skl_gemm_1x1_f32c_f32_f32_xsfmm32a32f(m, n, k, a, csa, b, rsb);

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0, 0, 0, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

/* Process 2 (= 1 x 2) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm2tn_a1b01_f32c_f32cp_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb1, size_t csb1, float *c, size_t rsc0, size_t rsc1, size_t csc1,
    size_t row1, size_t col1, bool accum, fused_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t mt0 = 0;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0, 0, 4);
  } else {
    skl_tile_zero_e32_xsfmmbase(m, n, mt0, 0, 4);
  }

  skl_gemm_1x2_f32c_f32_f32_xsfmm32a32f(m, n, k, a, csa, b, rsb1, csb1);

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0, 0, 4, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

/* Process 2 (= 2 x 1) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_2tm1tn_a1b01_f32pc_f32_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb, float *c, size_t rsc0, size_t rsc1, size_t csc1,
    size_t row1, size_t col1, bool accum, fused_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t kShiftPattern = 24;
  const size_t mt0 = 0;
  const size_t mt0c = mt0 | 1 << kShiftPattern;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0c, 4, 0);
  } else {
    skl_tile_zero_e32_xsfmmbase(n, m, mt0, 0, 4);
  }

  // NOLINTBEGIN(readability-suspicious-call-argument)
  skl_gemm_1x2_f32c_f32_f32_xsfmm32a32f(n, m, k, b, rsb, a, csa1, rsa1);
  // NOLINTEND(readability-suspicious-call-argument)

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0c, 4, 0, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

/* Process 3 (= 1 x 3) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm3tn_a1b01_f32c_f32cp_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb1, size_t csb1, float *c, size_t rsc0, size_t rsc1, size_t csc1,
    size_t row1, size_t col1, bool accum, fused_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t mt0 = 0;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0, 0, 4);
  } else {
    skl_tile_zero_e32_xsfmmbase(m, n, mt0, 0, 4);
  }

  skl_gemm_1x3_f32c_f32_f32_xsfmm32a32f(m, n, k, a, csa, b, rsb1, csb1);

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0, 0, 4, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

/* Process 3 (= 3 x 1) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_3tm1tn_a1b01_f32pc_f32_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb, float *c, size_t rsc0, size_t rsc1, size_t csc1,
    size_t row1, size_t col1, bool accum, fused_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t kShiftPattern = 24;
  const size_t mt0 = 0;
  const size_t mt0c = mt0 | 1 << kShiftPattern;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0c, 4, 0);
  } else {
    skl_tile_zero_e32_xsfmmbase(n, m, mt0, 0, 4);
  }

  // NOLINTBEGIN(readability-suspicious-call-argument)
  skl_gemm_1x3_f32c_f32_f32_xsfmm32a32f(n, m, k, b, rsb, a, csa1, rsa1);
  // NOLINTEND(readability-suspicious-call-argument)

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0c, 4, 0, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

/* Process 4 (= 1 x 4) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm4tn_a1b01_f32c_f32cp_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb1, size_t csb1, float *c, size_t rsc0, size_t rsc1, size_t csc1,
    size_t row1, size_t col1, bool accum, fused_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t mt0 = 0;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0, 0, 4);
  } else {
    skl_tile_zero_e32_xsfmmbase(m, n, mt0, 0, 4);
  }

  skl_gemm_1x4_f32c_f32_f32_xsfmm32a32f(m, n, k, a, csa, b, rsb1, csb1);

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0, 0, 4, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

/* Process 4 (= 4 x 1) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_4tm1tn_a1b01_f32pc_f32_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb, float *c, size_t rsc0, size_t rsc1, size_t csc1,
    size_t row1, size_t col1, bool accum, fused_f32_f32_t kernel,
    void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t kShiftPattern = 24;
  const size_t mt0 = 0;
  const size_t mt0c = mt0 | 1 << kShiftPattern;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0c, 4, 0);
  } else {
    skl_tile_zero_e32_xsfmmbase(n, m, mt0, 0, 4);
  }

  // NOLINTBEGIN(readability-suspicious-call-argument)
  skl_gemm_1x4_f32c_f32_f32_xsfmm32a32f(n, m, k, b, rsb, a, csa1, rsa1);
  // NOLINTEND(readability-suspicious-call-argument)

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0c, 4, 0, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_2tm2tn_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, size_t row1, size_t col1, bool accum,
    fused_f32_f32_t kernel, void *params) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t mt0 = 0;
  if (accum) {
    skl_tile_load_e32rcp_xsfmmbase(m, n,
                                   (uint32_t *)c + row1 * rsc1 + col1 * csc1,
                                   rsc0, rsc1, csc1, mt0, 8, 4);
  } else {
    skl_tile_zero_e32_xsfmmbase(m, n, mt0, 8, 4);
  }

  skl_gemm_2x2_f32rcp_f32rcp_f32_xsfmm32a32f(m, n, k, a, rsa1, csa1, b, rsb1,
                                             csb1);

  skl_gemm_fused_apply_f32rcprc_xsfmm32a32f(m, n, mt0, 8, 4, c, rsc0, 1, rsc1,
                                            csc1, row1, col1, kernel, params);
}

SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_dispatch_a1b01_f32rcpc_f32rcp_f32rcp_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t rsa1, size_t csa1,
    const float *b, size_t rsb1, size_t csb1, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, bool accum, fused_f32_f32_t kernel,
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
      skl_gemm_2tm2tn_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f(
          2 * ete, n_vl, k, a + i1 * rsa1, rsa1, csa1, b + j1 * csb1, rsb1,
          csb1, c, rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
      skl_gemm_2tm2tn_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f(
          m_vl - 2 * ete, n_vl, k, a + (i1 + 2) * rsa1, rsa1, csa1,
          b + j1 * csb1, rsb1, csb1, c, rsc0, rsc1, csc1, i1 + 2, j1, accum,
          kernel, params);
      n_avl -= n_vl;
    }
    if (j1 < n1) {
      skl_gemm_4tm1tn_a1b01_f32pc_f32_f32rcp_xsfmm32a32f(
          m_vl, n_avl, k, a + i1 * rsa1, rsa1, csa1, b + j1 * csb1, rsb1, c,
          rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
    }
    m_avl -= m_vl;
  }

  for (; i1 + 1 < m1; i1 += 2) {
    size_t m_vl = m_avl >= 2 * ete ? 2 * ete : m_avl;
    size_t n_avl = n;
    size_t j1 = 0;
    for (; j1 + 1 < n1; j1 += 2) {
      size_t n_vl = n_avl >= 2 * ete ? 2 * ete : n_avl;
      skl_gemm_2tm2tn_a1b01_f32pc_f32cp_f32rcp_xsfmm32a32f(
          m_vl, n_vl, k, a + i1 * rsa1, rsa1, csa1, b + j1 * csb1, rsb1, csb1,
          c, rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
      n_avl -= n_vl;
    }
    if (j1 < n1) {
      skl_gemm_4tm1tn_a1b01_f32pc_f32_f32rcp_xsfmm32a32f(
          m_vl, n_avl, k, a + i1 * rsa1, rsa1, csa1, b + j1 * csb1, rsb1, c,
          rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
    }
    m_avl -= m_vl;
  }

  if (i1 < m1) {
    size_t n_avl = n;
    size_t j1 = 0;
    for (; j1 + 3 < n1; j1 += 4) {
      size_t n_vl = n_avl >= 4 * ete ? 4 * ete : n_avl;
      skl_gemm_1tm4tn_a1b01_f32c_f32cp_f32rcp_xsfmm32a32f(
          m_avl, n_vl, k, a + i1 * rsa1, csa1, b + j1 * csb1, rsb1, csb1, c,
          rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
      n_avl -= n_vl;
    }
    switch (n1 - j1) {
    case 3:
      skl_gemm_1tm3tn_a1b01_f32c_f32cp_f32rcp_xsfmm32a32f(
          m_avl, n_avl, k, a + i1 * rsa1, csa1, b + j1 * csb1, rsb1, csb1, c,
          rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
      break;
    case 2:
      skl_gemm_1tm2tn_a1b01_f32c_f32cp_f32rcp_xsfmm32a32f(
          m_avl, n_avl, k, a + i1 * rsa1, csa1, b + j1 * csb1, rsb1, csb1, c,
          rsc0, rsc1, csc1, i1, j1, accum, kernel, params);
      break;
    case 1:
      skl_gemm_1tm1tn_a1b01_f32c_f32_f32_xsfmm32a32f(
          m_avl, n_avl, k, a + i1 * rsa1, csa1, b + j1 * csb1, rsb1, c, rsc0,
          rsc1, csc1, i1, j1, accum, kernel, params);
      break;
    default:
      break;
    }
  }
  __asm__ volatile("sf.vtdiscard");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_gemm_fused_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb, float *c, size_t rsc, bool accum, fused_f32_f32_t kernel,
    void *params) {
  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  skl_gemm_dispatch_a1b01_f32rcpc_f32rcp_f32rcp_xsfmm32a32f(
      m, n, k, a, ete, csa, b, rsb, ete, c, rsc, ete * rsc, ete, accum, kernel,
      params);
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_gemm_fused_f32pc_f32cp_f32rcp_xsfmm32a32f(
    size_t m1, size_t n1, size_t k, const float *a, size_t rsa1, const float *b,
    size_t csb1, float *c, size_t rsc1, size_t csc1, bool accum,
    fused_f32_f32_t kernel, void *params) {
  size_t ete = 0; // Effective tile edge length (always TE for TEW = 32).
  __asm__ volatile("sf.vsettnt %0, x0, e32, w1" : "=r"(ete) : : "vtype", "vl");

  skl_gemm_dispatch_a1b01_f32rcpc_f32rcp_f32rcp_xsfmm32a32f(
      m1 * ete, n1 * ete, k, a, rsa1, ete, b, ete, csb1, c, ete, rsc1, csc1,
      accum, kernel, params);
}

SKL_XSFMM_NEW
SKL_FUNC void skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f(
    size_t m, size_t n, size_t k, const float *a, size_t csa, const float *b,
    size_t rsb, float *c, size_t rsc, bool accum) {
  skl_gemm_fused_f32c_f32_f32_xsfmm32a32f(
      m, n, k, a, csa, b, rsb, c, rsc, accum, skl_tile_store_f32rcp_xsfmmbase,
      NULL);
}

SKL_XSFMM_NEW
SKL_FUNC void skl_gemm_a1b01_f32ptex1c_f32cp1xte_f32rcptexte_xsfmm32a32f(
    size_t m1, size_t n1, size_t k, const float *a, size_t rsa1, const float *b,
    size_t csb1, float *c, size_t rsc1, size_t csc1, bool accum) {
  skl_gemm_fused_f32pc_f32cp_f32rcp_xsfmm32a32f(
      m1, n1, k, a, rsa1, b, csb1, c, rsc1, csc1, accum,
      skl_tile_store_f32rcp_xsfmmbase, NULL);
}
