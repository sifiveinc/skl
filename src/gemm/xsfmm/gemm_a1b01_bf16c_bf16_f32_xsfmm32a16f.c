// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfmm32a16f)
#error This file requires the Xsfmm32a16f extension
#endif

#include <stdbool.h>
#include <stddef.h>

#include "skl-common.h"

/* Process a tm x tn tile of c. tm and tn must be <= TE. */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa,
    const __bf16 *b, size_t rsb, float *c, size_t rsc, bool accum) {
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kRowInc = 1;

  const size_t mt0 = 0;
  float *c00 = c;

  /* Load or zero-initialize tile. */
  size_t mt0_load = mt0;
  float *c00_load = c00;
  size_t i = 0;
  __asm__ volatile("sf.vsettnt x0, %[tn], e16alt, w2\n"
                   "sf.vsettm x0, %[tm]\n"

                   "bnez %[accum], 0f\n"
                   "sf.vtzero.t mt0\n"
                   "j 1f\n"
                   "0:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vlte32 %[mt0], (%[c00])\n"
                   "add %[mt0], %[mt0], %[kRowInc]\n"
                   "add %[c00], %[c00], %[sc]\n"
                   "bltu %[i], %[tm], 0b\n"
                   "1:\n"
                   : [mt0] "+&r"(mt0_load), [c00] "+&r"(c00_load), [i] "+&r"(i)
                   : [accum] "r"(accum), [kRowInc] "rI"(kRowInc),
                     [sc] "r"(rsc * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl", "memory");

  /* Accumulate matrix product into tile. */
  const __bf16 *a0 = a;
  const __bf16 *a1 = a0 + csa;
  const __bf16 *b0 = b;
  const __bf16 *b1 = b0 + rsb;
  __asm__ volatile(
      "beqz %[k], 2f\n"

      "sf.vsettnt x0, %[tn], e16alt, w2\n"
      "sf.vsettm x0, %[tm]\n"
      "sf.vsettk x0, %[k]\n"

      "bltu %[k], %[i2], 1f\n"

      "0:\n"
      "addi %[k], %[k], -2\n"
      "sf.vsettn x0, %[tm]\n"
      "vle16.v v0, (%[a0])\n"
      "add %[a0], %[a0], %[sa]\n"
      "vle16.v v4, (%[a1])\n"
      "add %[a1], %[a1], %[sa]\n"

      "sf.vsettn x0, %[tn]\n"
      "vle16.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[sb]\n"
      "vle16.v v12, (%[b1])\n"
      "add %[b1], %[b1], %[sb]\n"

      "sf.mm.f.f mt0, v0, v8\n"
      "bgeu %[k], %[i2], 0b\n"
      "sf.vsettk x0, %[k]\n"

      "1:\n"
      "beqz %[k], 2f\n"

      // k % 2 == 1
      "sf.vsettn x0, %[tm]\n"
      "vle16.v v0, (%[a0])\n"

      "sf.vsettn x0, %[tn]\n"
      "vle16.v v8, (%[b0])\n"

      "sf.mm.f.f mt0, v0, v8\n"

      "2:\n"
      : [a0] "+&r"(a0), [a1] "+&r"(a1), [b0] "+&r"(b0), [b1] "+&r"(b1),
        [k] "+&r"(k)
      : [sa] "r"(2 * csa * sizeof(__bf16)), [sb] "r"(2 * rsb * sizeof(__bf16)),
        [tm] "r"(tm), [tn] "r"(tn), [i2] "r"(2)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "vtype", "vl", "memory");

  /* Store tile to memory. */
  size_t mt0_store = mt0;
  float *c00_store = c00;
  i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, %[tn], e16alt, w2\n"

      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vste32 %[mt0], (%[c00])\n"
      "add %[mt0], %[mt0], %[kRowInc]\n"
      "add %[c00], %[c00], %[sc]\n"
      "bltu %[i], %[tm], 0b\n"

      "sf.vtdiscard"
      : [mt0] "+&r"(mt0_store), [c00] "+&r"(c00_store), [i] "+&r"(i)
      : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc * sizeof(float)), [tm] "r"(tm),
        [tn] "r"(tn)
      : "vtype", "vl", "memory");
}

/* C consists of two tm x tn row major tiles C0 and C1. Each tile's row stride
 * is rsc0, and the distance from C0 to C1 in elements is tsc1. A is a row major
 * matrix with k rows and row stride rsa. B consists of two row major tiles B0
 * and B1. Each tile has k rows and row stride rsb0, and the distance from B0 to
 * B1 in elements is csb1. If bta == true, this routine computes C0 = B0^T * A,
 * C1 = B1^T * A. Otherwise, it computes C0 = A^T * B0, C1 = A^T * B1.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm2tn_2tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t rsa,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t tsc1, bool accum, bool bta) {
  // We can use the same inner loop code for both values of bta by computing A^T
  // * B0 and A^T * B1 and then transposing the result if bta == true.
  // The active tile state has shape tm0 x tn0.
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kShiftTile = 27;
  const size_t kShiftPattern = 24;
  const size_t kRowInc = 1;

  const size_t tm0 = bta ? tn : tm;
  const size_t tn0 = bta ? tm : tn;
  const size_t tss_pattern = bta ? (size_t)(1) << kShiftPattern : 0;
  const size_t mt0 = 0 | tss_pattern;
  const size_t mt4 = (size_t)(4) << kShiftTile | tss_pattern;

  float *c0 = c;
  float *c1 = c0 + tsc1;

  /* Load or zero-initialize tiles. */
  size_t mt0_load = mt0;
  size_t mt4_load = mt4;

  float *c0_load = c0;
  float *c1_load = c1;
  size_t i = 0;
  __asm__ volatile("bnez %[accum], 0f\n"
                   "sf.vsettnt x0, %[tn0], e32, w1\n"
                   "sf.vsettm x0, %[tm0]\n"

                   "sf.vtzero.t mt0\n"
                   "sf.vtzero.t mt4\n"
                   "j 2f\n"
                   "0:\n"
                   "sf.vsettnt x0, %[tn], e32, w1\n"
                   "1:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vlte32 %[mt0], (%[c0])\n"
                   "add %[mt0], %[mt0], %[kRowInc]\n"
                   "add %[c0], %[c0], %[sc]\n"
                   "sf.vlte32 %[mt4], (%[c1])\n"
                   "add %[mt4], %[mt4], %[kRowInc]\n"
                   "add %[c1], %[c1], %[sc]\n"
                   "bltu %[i], %[tm], 1b\n"
                   "2:\n"
                   : [mt0] "+&r"(mt0_load), [mt4] "+&r"(mt4_load),
                     [c0] "+&r"(c0_load), [c1] "+&r"(c1_load), [i] "+&r"(i)
                   : [accum] "r"(accum), [kRowInc] "rI"(kRowInc),
                     [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn),
                     [tm0] "r"(tm0), [tn0] "r"(tn0)
                   : "vtype", "vl", "memory");

  /* Accumulate matrix product into tiles. */
  // a0_{0,1} are used to load a 2 x tm submatrix of A,
  // b0_{0,1} are used to load a 2 x tn submatrix of B,
  // and similarly for a1_{0,1} and b1_{0,1}.
  const __bf16 *a0_0 = a;
  const __bf16 *a0_1 = a0_0 + rsa;
  const __bf16 *b0_0 = b;
  const __bf16 *b0_1 = b0_0 + rsb0;
  const __bf16 *b1_0 = b0_0 + csb1;
  const __bf16 *b1_1 = b1_0 + rsb0;

  if (k < 2) {
    __asm__ volatile("beqz %[k], 0f\n"

                     "sf.vsettnt x0, %[tn0], e16alt, w2\n"
                     "sf.vsettm x0, %[tm0]\n"
                     "sf.vsettk x0, %[k]\n"

                     // k == 1
                     "sf.vsettn x0, %[tm0]\n"
                     "vle16.v v0, (%[a0_0])\n"
                     "sf.vsettn x0, %[tn0]\n"

                     "vle16.v v8, (%[b0_0])\n"

                     "sf.mm.f.f mt0, v0, v8\n"

                     "vle16.v v16, (%[b1_0])\n"

                     "sf.mm.f.f mt4, v0, v16\n"

                     "0:\n"
                     :
                     : [a0_0] "r"(a0_0), [b0_0] "r"(b0_0), [b1_0] "r"(b1_0),
                       [tm0] "r"(tm0), [tn0] "r"(tn0), [k] "r"(k)
                     : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16",
                       "v17", "v18", "v19", "vtype", "vl", "memory");
  } else {
    __asm__ volatile("sf.vsettnt x0, %[tn0], e16alt, w2\n"
                     "sf.vsettm x0, %[tm0]\n"
                     "sf.vsettk x0, %[k]\n"

                     "sf.vsettn x0, %[tm0]\n"
                     "vle16.v v0, (%[a0_0])\n"
                     "add %[a0_0], %[a0_0], %[sa]\n"
                     "vle16.v v4, (%[a0_1])\n"
                     "add %[a0_1], %[a0_1], %[sa]\n"
                     "sf.vsettn x0, %[tn0]\n"

                     "vle16.v v8, (%[b0_0])\n"
                     "add %[b0_0], %[b0_0], %[sb]\n"
                     "vle16.v v12, (%[b0_1])\n"
                     "add %[b0_1], %[b0_1], %[sb]\n"

                     "bltu %[k], %[i4], 1f\n"

                     "0:\n"
                     "addi %[k], %[k], -2\n"
                     "vle16.v v16, (%[b1_0])\n"
                     "add %[b1_0], %[b1_0], %[sb]\n"
                     "vle16.v v20, (%[b1_1])\n"
                     "add %[b1_1], %[b1_1], %[sb]\n"

                     "sf.mm.f.f mt0, v0, v8\n"

                     "vle16.v v8, (%[b0_0])\n"
                     "add %[b0_0], %[b0_0], %[sb]\n"
                     "vle16.v v12, (%[b0_1])\n"
                     "add %[b0_1], %[b0_1], %[sb]\n"

                     "sf.mm.f.f mt4, v0, v16\n"

                     "sf.vsettn x0, %[tm0]\n"
                     "vle16.v v0, (%[a0_0])\n"
                     "add %[a0_0], %[a0_0], %[sa]\n"
                     "vle16.v v4, (%[a0_1])\n"
                     "add %[a0_1], %[a0_1], %[sa]\n"
                     "sf.vsettn x0, %[tn0]\n"

                     "bgeu %[k], %[i4], 0b\n"

                     "1:\n"
                     "addi %[k], %[k], -2\n"
                     "vle16.v v16, (%[b1_0])\n"
                     "add %[b1_0], %[b1_0], %[sb]\n"
                     "vle16.v v20, (%[b1_1])\n"
                     "add %[b1_1], %[b1_1], %[sb]\n"

                     "sf.mm.f.f mt0, v0, v8\n"

                     "beqz %[k], 2f\n"

                     // k % 2 == 1
                     "vle16.v v8, (%[b0_0])\n"

                     "sf.mm.f.f mt4, v0, v16\n"

                     "sf.vsettn x0, %[tm0]\n"
                     "vle16.v v0, (%[a0_0])\n"
                     "sf.vsettn x0, %[tn0]\n"

                     "vle16.v v16, (%[b1_0])\n"

                     "sf.vsettk x0, %[k]\n"
                     "sf.mm.f.f mt0, v0, v8\n"
                     "sf.mm.f.f mt4, v0, v16\n"
                     "j 3f\n"

                     "2:\n" // k % 2 == 0
                     "sf.mm.f.f mt4, v0, v16\n"

                     "3:\n"
                     : [a0_0] "+&r"(a0_0), [a0_1] "+&r"(a0_1),
                       [b0_0] "+&r"(b0_0), [b0_1] "+&r"(b0_1),
                       [b1_0] "+&r"(b1_0), [b1_1] "+&r"(b1_1), [k] "+&r"(k)
                     : [sa] "r"(2 * rsa * sizeof(__bf16)),
                       [sb] "r"(2 * rsb0 * sizeof(__bf16)), [tm0] "r"(tm0),
                       [tn0] "r"(tn0), [i4] "r"(4)
                     : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                       "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                       "v17", "v18", "v19", "v20", "v21", "v22", "v23", "vtype",
                       "vl", "memory");
  }

  /* Store tiles to memory. */
  size_t mt0_store = mt0;
  size_t mt4_store = mt4;

  float *c0_store = c0;
  float *c1_store = c1;
  i = 0;
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"

                   "0:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vste32 %[mt0], (%[c0])\n"
                   "add %[mt0], %[mt0], %[kRowInc]\n"
                   "add %[c0], %[c0], %[sc]\n"
                   "sf.vste32 %[mt4], (%[c1])\n"
                   "add %[mt4], %[mt4], %[kRowInc]\n"
                   "add %[c1], %[c1], %[sc]\n"
                   "bltu %[i], %[tm], 0b\n"

                   "sf.vtdiscard"
                   : [mt0] "+&r"(mt0_store), [mt4] "+&r"(mt4_store),
                     [c0] "+&r"(c0_store), [c1] "+&r"(c1_store), [i] "+&r"(i)
                   : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc0 * sizeof(float)),
                     [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl", "memory");
}

SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm3tn_3tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t rsa,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t tsc1, bool accum, bool bta) {
  // We can use the same inner loop code for both values of bta by computing A^T
  // * B0 and A^T * B1 and then transposing the result if bta == true.
  // The active tile state has shape tm0 x tn0.
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kShiftTile = 27;
  const size_t kShiftPattern = 24;
  const size_t kRowInc = 1;

  const size_t tm0 = bta ? tn : tm;
  const size_t tn0 = bta ? tm : tn;
  const size_t tss_pattern = bta ? (size_t)(1) << kShiftPattern : 0;
  const size_t mt0 = 0 | tss_pattern;
  const size_t mt4 = (size_t)(4) << kShiftTile | tss_pattern;
  const size_t mt8 = (size_t)(8) << kShiftTile | tss_pattern;

  float *c0 = c;
  float *c1 = c0 + tsc1;
  float *c2 = c0 + 2 * tsc1;

  /* Load or zero-initialize tiles. */
  size_t mt0_load = mt0;
  size_t mt4_load = mt4;
  size_t mt8_load = mt8;

  float *c0_load = c0;
  float *c1_load = c1;
  float *c2_load = c2;
  size_t i = 0;
  __asm__ volatile("bnez %[accum], 0f\n"
                   "sf.vsettnt x0, %[tn0], e32, w1\n"
                   "sf.vsettm x0, %[tm0]\n"

                   "sf.vtzero.t mt0\n"
                   "sf.vtzero.t mt4\n"
                   "sf.vtzero.t mt8\n"
                   "j 2f\n"
                   "0:\n"
                   "sf.vsettnt x0, %[tn], e32, w1\n"
                   "1:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vlte32 %[mt0], (%[c0])\n"
                   "add %[mt0], %[mt0], %[kRowInc]\n"
                   "add %[c0], %[c0], %[sc]\n"
                   "sf.vlte32 %[mt4], (%[c1])\n"
                   "add %[mt4], %[mt4], %[kRowInc]\n"
                   "add %[c1], %[c1], %[sc]\n"
                   "sf.vlte32 %[mt8], (%[c2])\n"
                   "add %[mt8], %[mt8], %[kRowInc]\n"
                   "add %[c2], %[c2], %[sc]\n"
                   "bltu %[i], %[tm], 1b\n"
                   "2:\n"
                   : [mt0] "+&r"(mt0_load), [mt4] "+&r"(mt4_load),
                     [mt8] "+&r"(mt8_load), [c0] "+&r"(c0_load),
                     [c1] "+&r"(c1_load), [c2] "+&r"(c2_load), [i] "+&r"(i)
                   : [accum] "r"(accum), [kRowInc] "rI"(kRowInc),
                     [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn),
                     [tm0] "r"(tm0), [tn0] "r"(tn0)
                   : "vtype", "vl", "memory");

  /* Accumulate matrix product into tiles. */
  // a0_{0,1} are used to load a 2 x tm submatrix of A,
  // b0_{0,1} are used to load a 2 x tn submatrix of B,
  // and similarly for a1_{0,1} and b1_{0,1}.
  const __bf16 *a0_0 = a;
  const __bf16 *a0_1 = a0_0 + rsa;
  const __bf16 *b0_0 = b;
  const __bf16 *b0_1 = b0_0 + rsb0;
  const __bf16 *b1_0 = b0_0 + csb1;
  const __bf16 *b1_1 = b1_0 + rsb0;
  const __bf16 *b2_0 = b0_0 + 2 * csb1;
  const __bf16 *b2_1 = b2_0 + rsb0;

  if (k < 2) {
    __asm__ volatile(
        "beqz %[k], 0f\n"

        "sf.vsettnt x0, %[tn0], e16alt, w2\n"
        "sf.vsettm x0, %[tm0]\n"
        "sf.vsettk x0, %[k]\n"

        // k == 1
        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "sf.vsettn x0, %[tn0]\n"

        "vle16.v v8, (%[b0_0])\n"

        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v16, (%[b1_0])\n"

        "sf.mm.f.f mt4, v0, v16\n"

        "vle16.v v24, (%[b2_0])\n"

        "sf.mm.f.f mt8, v0, v24\n"

        "0:\n"
        :
        : [a0_0] "r"(a0_0), [b0_0] "r"(b0_0), [b1_0] "r"(b1_0),
          [b2_0] "r"(b2_0), [tm0] "r"(tm0), [tn0] "r"(tn0), [k] "r"(k)
        : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
          "v19", "v24", "v25", "v26", "v27", "vtype", "vl", "memory");
  } else {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn0], e16alt, w2\n"
        "sf.vsettm x0, %[tm0]\n"
        "sf.vsettk x0, %[k]\n"

        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "add %[a0_0], %[a0_0], %[sa]\n"
        "vle16.v v4, (%[a0_1])\n"
        "add %[a0_1], %[a0_1], %[sa]\n"
        "sf.vsettn x0, %[tn0]\n"

        "vle16.v v8, (%[b0_0])\n"
        "add %[b0_0], %[b0_0], %[sb]\n"
        "vle16.v v12, (%[b0_1])\n"
        "add %[b0_1], %[b0_1], %[sb]\n"

        "bltu %[k], %[i4], 1f\n"

        "0:\n"
        "addi %[k], %[k], -2\n"
        "vle16.v v16, (%[b1_0])\n"
        "add %[b1_0], %[b1_0], %[sb]\n"
        "vle16.v v20, (%[b1_1])\n"
        "add %[b1_1], %[b1_1], %[sb]\n"

        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v24, (%[b2_0])\n"
        "add %[b2_0], %[b2_0], %[sb]\n"
        "vle16.v v28, (%[b2_1])\n"
        "add %[b2_1], %[b2_1], %[sb]\n"

        "sf.mm.f.f mt4, v0, v16\n"

        "vle16.v v8, (%[b0_0])\n"
        "add %[b0_0], %[b0_0], %[sb]\n"
        "vle16.v v12, (%[b0_1])\n"
        "add %[b0_1], %[b0_1], %[sb]\n"

        "sf.mm.f.f mt8, v0, v24\n"

        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "add %[a0_0], %[a0_0], %[sa]\n"
        "vle16.v v4, (%[a0_1])\n"
        "add %[a0_1], %[a0_1], %[sa]\n"
        "sf.vsettn x0, %[tn0]\n"

        "bgeu %[k], %[i4], 0b\n"

        "1:\n"
        "addi %[k], %[k], -2\n"
        "vle16.v v16, (%[b1_0])\n"
        "add %[b1_0], %[b1_0], %[sb]\n"
        "vle16.v v20, (%[b1_1])\n"
        "add %[b1_1], %[b1_1], %[sb]\n"

        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v24, (%[b2_0])\n"
        "add %[b2_0], %[b2_0], %[sb]\n"
        "vle16.v v28, (%[b2_1])\n"
        "add %[b2_1], %[b2_1], %[sb]\n"

        "sf.mm.f.f mt4, v0, v16\n"

        "beqz %[k], 2f\n"

        // k % 2 == 1
        "vle16.v v8, (%[b0_0])\n"

        "sf.mm.f.f mt8, v0, v24\n"

        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "sf.vsettn x0, %[tn0]\n"

        "vle16.v v16, (%[b1_0])\n"

        "sf.vsettk x0, %[k]\n"
        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v24, (%[b2_0])\n"

        "sf.mm.f.f mt4, v0, v16\n"
        "sf.mm.f.f mt8, v0, v24\n"
        "j 3f\n"

        "2:\n" // k % 2 == 0
        "sf.mm.f.f mt8, v0, v24\n"

        "3:\n"
        : [a0_0] "+&r"(a0_0), [a0_1] "+&r"(a0_1), [b0_0] "+&r"(b0_0),
          [b0_1] "+&r"(b0_1), [b1_0] "+&r"(b1_0), [b1_1] "+&r"(b1_1),
          [b2_0] "+&r"(b2_0), [b2_1] "+&r"(b2_1), [k] "+&r"(k)
        : [sa] "r"(2 * rsa * sizeof(__bf16)),
          [sb] "r"(2 * rsb0 * sizeof(__bf16)), [tm0] "r"(tm0), [tn0] "r"(tn0),
          [i4] "r"(4)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31", "vtype", "vl", "memory");
  }

  /* Store tiles to memory. */
  size_t mt0_store = mt0;
  size_t mt4_store = mt4;
  size_t mt8_store = mt8;

  float *c0_store = c0;
  float *c1_store = c1;
  float *c2_store = c2;
  i = 0;
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"

                   "0:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vste32 %[mt0], (%[c0])\n"
                   "add %[mt0], %[mt0], %[kRowInc]\n"
                   "add %[c0], %[c0], %[sc]\n"
                   "sf.vste32 %[mt4], (%[c1])\n"
                   "add %[mt4], %[mt4], %[kRowInc]\n"
                   "add %[c1], %[c1], %[sc]\n"
                   "sf.vste32 %[mt8], (%[c2])\n"
                   "add %[mt8], %[mt8], %[kRowInc]\n"
                   "add %[c2], %[c2], %[sc]\n"
                   "bltu %[i], %[tm], 0b\n"

                   "sf.vtdiscard"
                   : [mt0] "+&r"(mt0_store), [mt4] "+&r"(mt4_store),
                     [mt8] "+&r"(mt8_store), [c0] "+&r"(c0_store),
                     [c1] "+&r"(c1_store), [c2] "+&r"(c2_store), [i] "+&r"(i)
                   : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc0 * sizeof(float)),
                     [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl", "memory");
}

SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm4tn_4tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t rsa,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t tsc1, bool accum, bool bta) {
  // We can use the same inner loop code for both values of bta by computing A^T
  // * B0 and A^T * B1 and then transposing the result if bta == true.
  // The active tile state has shape tm0 x tn0.
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kShiftTile = 27;
  const size_t kShiftPattern = 24;
  const size_t kRowInc = 1;

  const size_t tm0 = bta ? tn : tm;
  const size_t tn0 = bta ? tm : tn;
  const size_t tss_pattern = bta ? (size_t)(1) << kShiftPattern : 0;
  const size_t mt0 = 0 | tss_pattern;
  const size_t mt4 = (size_t)(4) << kShiftTile | tss_pattern;
  const size_t mt8 = (size_t)(8) << kShiftTile | tss_pattern;
  const size_t mt12 = (size_t)(12) << kShiftTile | tss_pattern;

  float *c0 = c;
  float *c1 = c0 + tsc1;
  float *c2 = c0 + 2 * tsc1;
  float *c3 = c0 + 3 * tsc1;

  /* Load or zero-initialize tiles. */
  size_t mt0_load = mt0;
  size_t mt4_load = mt4;
  size_t mt8_load = mt8;
  size_t mt12_load = mt12;

  float *c0_load = c0;
  float *c1_load = c1;
  float *c2_load = c2;
  float *c3_load = c3;
  size_t i = 0;
  __asm__ volatile(
      "bnez %[accum], 0f\n"
      "sf.vsettnt x0, %[tn0], e32, w1\n"
      "sf.vsettm x0, %[tm0]\n"

      "sf.vtzero.t mt0\n"
      "sf.vtzero.t mt4\n"
      "sf.vtzero.t mt8\n"
      "sf.vtzero.t mt12\n"
      "j 2f\n"
      "0:\n"
      "sf.vsettnt x0, %[tn], e32, w1\n"
      "1:\n"
      "addi %[i], %[i], 1\n"
      "sf.vlte32 %[mt0], (%[c0])\n"
      "add %[mt0], %[mt0], %[kRowInc]\n"
      "add %[c0], %[c0], %[sc]\n"
      "sf.vlte32 %[mt4], (%[c1])\n"
      "add %[mt4], %[mt4], %[kRowInc]\n"
      "add %[c1], %[c1], %[sc]\n"
      "sf.vlte32 %[mt8], (%[c2])\n"
      "add %[mt8], %[mt8], %[kRowInc]\n"
      "add %[c2], %[c2], %[sc]\n"
      "sf.vlte32 %[mt12], (%[c3])\n"
      "add %[mt12], %[mt12], %[kRowInc]\n"
      "add %[c3], %[c3], %[sc]\n"
      "bltu %[i], %[tm], 1b\n"
      "2:\n"
      : [mt0] "+&r"(mt0_load), [mt4] "+&r"(mt4_load), [mt8] "+&r"(mt8_load),
        [mt12] "+&r"(mt12_load), [c0] "+&r"(c0_load), [c1] "+&r"(c1_load),
        [c2] "+&r"(c2_load), [c3] "+&r"(c3_load), [i] "+&r"(i)
      : [accum] "r"(accum), [kRowInc] "rI"(kRowInc),
        [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn),
        [tm0] "r"(tm0), [tn0] "r"(tn0)
      : "vtype", "vl", "memory");

  /* Accumulate matrix product into tiles. */
  // a0_{0,1} are used to load a 2 x tm submatrix of A,
  // b0_{0,1} are used to load a 2 x tn submatrix of B,
  // and similarly for a1_{0,1} and b1_{0,1}.
  const __bf16 *a0_0 = a;
  const __bf16 *a0_1 = a0_0 + rsa;
  const __bf16 *b0_0 = b;
  const __bf16 *b0_1 = b0_0 + rsb0;
  const __bf16 *b1_0 = b0_0 + csb1;
  const __bf16 *b1_1 = b1_0 + rsb0;
  const __bf16 *b2_0 = b0_0 + 2 * csb1;
  const __bf16 *b2_1 = b2_0 + rsb0;
  const __bf16 *b3_0 = b0_0 + 3 * csb1;
  const __bf16 *b3_1 = b3_0 + rsb0;

  if (k < 2) {
    __asm__ volatile("beqz %[k], 0f\n"

                     "sf.vsettnt x0, %[tn0], e16alt, w2\n"
                     "sf.vsettm x0, %[tm0]\n"
                     "sf.vsettk x0, %[k]\n"

                     // k == 1
                     "sf.vsettn x0, %[tm0]\n"
                     "vle16.v v0, (%[a0_0])\n"
                     "sf.vsettn x0, %[tn0]\n"

                     "vle16.v v8, (%[b0_0])\n"

                     "sf.mm.f.f mt0, v0, v8\n"

                     "vle16.v v16, (%[b1_0])\n"

                     "sf.mm.f.f mt4, v0, v16\n"

                     "vle16.v v8, (%[b2_0])\n"

                     "sf.mm.f.f mt8, v0, v8\n"

                     "vle16.v v16, (%[b3_0])\n"

                     "sf.mm.f.f mt12, v0, v16\n"

                     "0:\n"
                     :
                     : [a0_0] "r"(a0_0), [b0_0] "r"(b0_0), [b1_0] "r"(b1_0),
                       [b2_0] "r"(b2_0), [b3_0] "r"(b3_0), [tm0] "r"(tm0),
                       [tn0] "r"(tn0), [k] "r"(k)
                     : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16",
                       "v17", "v18", "v19", "vtype", "vl", "memory");
  } else {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn0], e16alt, w2\n"
        "sf.vsettm x0, %[tm0]\n"
        "sf.vsettk x0, %[k]\n"

        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "add %[a0_0], %[a0_0], %[sa]\n"
        "vle16.v v4, (%[a0_1])\n"
        "add %[a0_1], %[a0_1], %[sa]\n"
        "sf.vsettn x0, %[tn0]\n"

        "vle16.v v8, (%[b0_0])\n"
        "add %[b0_0], %[b0_0], %[sb]\n"
        "vle16.v v12, (%[b0_1])\n"
        "add %[b0_1], %[b0_1], %[sb]\n"

        "bltu %[k], %[i4], 1f\n"

        "0:\n"
        "addi %[k], %[k], -2\n"
        "vle16.v v16, (%[b1_0])\n"
        "add %[b1_0], %[b1_0], %[sb]\n"
        "vle16.v v20, (%[b1_1])\n"
        "add %[b1_1], %[b1_1], %[sb]\n"

        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v8, (%[b2_0])\n"
        "add %[b2_0], %[b2_0], %[sb]\n"
        "vle16.v v12, (%[b2_1])\n"
        "add %[b2_1], %[b2_1], %[sb]\n"

        "sf.mm.f.f mt4, v0, v16\n"

        "vle16.v v16, (%[b3_0])\n"
        "add %[b3_0], %[b3_0], %[sb]\n"
        "vle16.v v20, (%[b3_1])\n"
        "add %[b3_1], %[b3_1], %[sb]\n"

        "sf.mm.f.f mt8, v0, v8\n"

        "vle16.v v8, (%[b0_0])\n"
        "add %[b0_0], %[b0_0], %[sb]\n"
        "vle16.v v12, (%[b0_1])\n"
        "add %[b0_1], %[b0_1], %[sb]\n"

        "sf.mm.f.f mt12, v0, v16\n"

        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "add %[a0_0], %[a0_0], %[sa]\n"
        "vle16.v v4, (%[a0_1])\n"
        "add %[a0_1], %[a0_1], %[sa]\n"
        "sf.vsettn x0, %[tn0]\n"

        "bgeu %[k], %[i4], 0b\n"

        "1:\n"
        "addi %[k], %[k], -2\n"
        "vle16.v v16, (%[b1_0])\n"
        "add %[b1_0], %[b1_0], %[sb]\n"
        "vle16.v v20, (%[b1_1])\n"
        "add %[b1_1], %[b1_1], %[sb]\n"

        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v8, (%[b2_0])\n"
        "add %[b2_0], %[b2_0], %[sb]\n"
        "vle16.v v12, (%[b2_1])\n"
        "add %[b2_1], %[b2_1], %[sb]\n"

        "sf.mm.f.f mt4, v0, v16\n"

        "vle16.v v16, (%[b3_0])\n"
        "add %[b3_0], %[b3_0], %[sb]\n"
        "vle16.v v20, (%[b3_1])\n"
        "add %[b3_1], %[b3_1], %[sb]\n"

        "sf.mm.f.f mt8, v0, v8\n"

        "beqz %[k], 2f\n"

        // k % 2 == 1
        "vle16.v v8, (%[b0_0])\n"

        "sf.mm.f.f mt12, v0, v16\n"

        "sf.vsettn x0, %[tm0]\n"
        "vle16.v v0, (%[a0_0])\n"
        "sf.vsettn x0, %[tn0]\n"

        "vle16.v v16, (%[b1_0])\n"

        "sf.vsettk x0, %[k]\n"
        "sf.mm.f.f mt0, v0, v8\n"

        "vle16.v v8, (%[b2_0])\n"

        "sf.mm.f.f mt4, v0, v16\n"

        "vle16.v v16, (%[b3_0])\n"

        "sf.mm.f.f mt8, v0, v8\n"
        "sf.mm.f.f mt12, v0, v16\n"
        "j 3f\n"

        "2:\n" // k % 2 == 0
        "sf.mm.f.f mt12, v0, v16\n"

        "3:\n"
        : [a0_0] "+&r"(a0_0), [a0_1] "+&r"(a0_1), [b0_0] "+&r"(b0_0),
          [b0_1] "+&r"(b0_1), [b1_0] "+&r"(b1_0), [b1_1] "+&r"(b1_1),
          [b2_0] "+&r"(b2_0), [b2_1] "+&r"(b2_1), [b3_0] "+&r"(b3_0),
          [b3_1] "+&r"(b3_1), [k] "+&r"(k)
        : [sa] "r"(2 * rsa * sizeof(__bf16)),
          [sb] "r"(2 * rsb0 * sizeof(__bf16)), [tm0] "r"(tm0), [tn0] "r"(tn0),
          [i4] "r"(4)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "vtype", "vl", "memory");
  }

  /* Store tiles to memory. */
  size_t mt0_store = mt0;
  size_t mt4_store = mt4;
  size_t mt8_store = mt8;
  size_t mt12_store = mt12;

  float *c0_store = c0;
  float *c1_store = c1;
  float *c2_store = c2;
  float *c3_store = c3;
  i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, %[tn], e32, w1\n"

      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vste32 %[mt0], (%[c0])\n"
      "add %[mt0], %[mt0], %[kRowInc]\n"
      "add %[c0], %[c0], %[sc]\n"
      "sf.vste32 %[mt4], (%[c1])\n"
      "add %[mt4], %[mt4], %[kRowInc]\n"
      "add %[c1], %[c1], %[sc]\n"
      "sf.vste32 %[mt8], (%[c2])\n"
      "add %[mt8], %[mt8], %[kRowInc]\n"
      "add %[c2], %[c2], %[sc]\n"
      "sf.vste32 %[mt12], (%[c3])\n"
      "add %[mt12], %[mt12], %[kRowInc]\n"
      "add %[c3], %[c3], %[sc]\n"
      "bltu %[i], %[tm], 0b\n"

      "sf.vtdiscard"
      : [mt0] "+&r"(mt0_store), [mt4] "+&r"(mt4_store), [mt8] "+&r"(mt8_store),
        [mt12] "+&r"(mt12_store), [c0] "+&r"(c0_store), [c1] "+&r"(c1_store),
        [c2] "+&r"(c2_store), [c3] "+&r"(c3_store), [i] "+&r"(i)
      : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm),
        [tn] "r"(tn)
      : "vtype", "vl", "memory");
}

/* Process 2 (= 1 x 2) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm2tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t csc1, bool accum) {
  skl_gemm_1tm2tn_2tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
      tm, tn, k, a, csa, b, rsb0, csb1, c, rsc0, csc1, accum, false);
}

/* Process 2 (= 2 x 1) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_2tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa0, size_t rsa1,
    const __bf16 *b, size_t rsb, float *c, size_t rsc0, size_t rsc1,
    bool accum) {
  skl_gemm_1tm2tn_2tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
      tm, tn, k, b, rsb, a, csa0, rsa1, c, rsc0, rsc1, accum, true);
}

/* Process 3 (= 1 x 3) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm3tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t csc1, bool accum) {
  skl_gemm_1tm3tn_3tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
      tm, tn, k, a, csa, b, rsb0, csb1, c, rsc0, csc1, accum, false);
}

/* Process 3 (= 3 x 1) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_3tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa0, size_t rsa1,
    const __bf16 *b, size_t rsb, float *c, size_t rsc0, size_t rsc1,
    bool accum) {
  skl_gemm_1tm3tn_3tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
      tm, tn, k, b, rsb, a, csa0, rsa1, c, rsc0, rsc1, accum, true);
}

/* Process 4 (= 1 x 4) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_1tm4tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t csc1, bool accum) {
  skl_gemm_1tm4tn_4tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
      tm, tn, k, a, csa, b, rsb0, csb1, c, rsc0, csc1, accum, false);
}

/* Process 4 (= 4 x 1) contiguous tm x tn tiles of c.
 * tm and tn must be <= TE.
 */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_4tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
    size_t tm, size_t tn, size_t k, const __bf16 *a, size_t csa0, size_t rsa1,
    const __bf16 *b, size_t rsb, float *c, size_t rsc0, size_t rsc1,
    bool accum) {
  skl_gemm_1tm4tn_4tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
      tm, tn, k, b, rsb, a, csa0, rsa1, c, rsc0, rsc1, accum, true);
}

/* Process 4 (= 2 x 2) contiguous tn x tn tiles of c. tn must be <= TE. */
SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
    size_t tn, size_t k, const __bf16 *a, size_t csa0, size_t rsa1,
    const __bf16 *b, size_t rsb0, size_t csb1, float *c, size_t rsc0,
    size_t rsc1, size_t csc1, bool accum) {
  /* Avoiding sf.vsettn instructions in the inner loop slightly improves
   * performance but requires tm == tn. */
  size_t tm = tn;

  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kShiftTile = 27;
  const size_t kRowInc = 1;

  const size_t mt0 = 0;
  const size_t mt4 = (size_t)(4) << kShiftTile;
  const size_t mt8 = (size_t)(8) << kShiftTile;
  const size_t mt12 = (size_t)(12) << kShiftTile;

  float *c00 = c;
  float *c01 = c00 + csc1;
  float *c10 = c00 + rsc1;
  float *c11 = c10 + csc1;

  /* Load or zero-initialize tiles. */
  size_t mt0_load = mt0;
  size_t mt4_load = mt4;
  size_t mt8_load = mt8;
  size_t mt12_load = mt12;

  float *c00_load = c00;
  float *c01_load = c01;
  float *c10_load = c10;
  float *c11_load = c11;
  size_t i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, %[tn], e16alt, w2\n"
      "sf.vsettm x0, %[tm]\n"

      "bnez %[accum], 0f\n"
      "sf.vtzero.t mt0\n"
      "sf.vtzero.t mt4\n"
      "sf.vtzero.t mt8\n"
      "sf.vtzero.t mt12\n"
      "j 1f\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vlte32 %[mt0], (%[c00])\n"
      "add %[mt0], %[mt0], %[kRowInc]\n"
      "add %[c00], %[c00], %[sc]\n"
      "sf.vlte32 %[mt4], (%[c01])\n"
      "add %[mt4], %[mt4], %[kRowInc]\n"
      "add %[c01], %[c01], %[sc]\n"
      "sf.vlte32 %[mt8], (%[c10])\n"
      "add %[mt8], %[mt8], %[kRowInc]\n"
      "add %[c10], %[c10], %[sc]\n"
      "sf.vlte32 %[mt12], (%[c11])\n"
      "add %[mt12], %[mt12], %[kRowInc]\n"
      "add %[c11], %[c11], %[sc]\n"
      "bltu %[i], %[tm], 0b\n"
      "1:\n"
      : [mt0] "+&r"(mt0_load), [mt4] "+&r"(mt4_load), [mt8] "+&r"(mt8_load),
        [mt12] "+&r"(mt12_load), [c00] "+&r"(c00_load), [c01] "+&r"(c01_load),
        [c10] "+&r"(c10_load), [c11] "+&r"(c11_load), [i] "+&r"(i)
      : [accum] "r"(accum), [kRowInc] "rI"(kRowInc),
        [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm), [tn] "r"(tn)
      : "vtype", "vl", "memory");

  /* Accumulate matrix product into tiles. */
  // a0_{0,1} are used to load a tm x 2 submatrix of A,
  // b0_{0,1} are used to load a 2 x tn submatrix of B,
  // and similarly for a1_{0,1} and b1_{0,1}.
  const __bf16 *a0_0 = a;
  const __bf16 *a0_1 = a0_0 + csa0;
  const __bf16 *a1_0 = a0_0 + rsa1;
  const __bf16 *a1_1 = a1_0 + csa0;
  const __bf16 *b0_0 = b;
  const __bf16 *b0_1 = b0_0 + rsb0;
  const __bf16 *b1_0 = b0_0 + csb1;
  const __bf16 *b1_1 = b1_0 + rsb0;
  if (k < 2) {
    __asm__ volatile("beqz %[k], 0f\n"

                     "sf.vsettnt x0, %[tn], e16alt, w2\n"
                     "sf.vsettm x0, %[tm]\n"
                     "sf.vsettk x0, %[k]\n"

                     // k == 1
                     "vle16.v v0, (%[a0_0])\n"

                     "vle16.v v16, (%[b0_0])\n"

                     "sf.mm.f.f mt0, v0, v16\n"

                     "vle16.v v24, (%[b1_0])\n"

                     "sf.mm.f.f mt4, v0, v24\n"

                     "vle16.v v8, (%[a1_0])\n"

                     "sf.mm.f.f mt8, v8, v16\n"
                     "sf.mm.f.f mt12, v8, v24\n"

                     "0:\n"
                     :
                     : [a0_0] "r"(a0_0), [a1_0] "r"(a1_0), [b0_0] "r"(b0_0),
                       [b1_0] "r"(b1_0), [tm] "r"(tm), [tn] "r"(tn), [k] "r"(k)
                     : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16",
                       "v17", "v18", "v19", "v24", "v25", "v26", "v27", "vtype",
                       "vl", "memory");
  } else {
    __asm__ volatile(
        "sf.vsettnt x0, %[tn], e16alt, w2\n"
        "sf.vsettm x0, %[tm]\n"
        "sf.vsettk x0, %[k]\n"

        "vle16.v v16, (%[b0_0])\n"
        "add %[b0_0], %[b0_0], %[sb]\n"
        "vle16.v v20, (%[b0_1])\n"
        "add %[b0_1], %[b0_1], %[sb]\n"

        "vle16.v v0, (%[a0_0])\n"
        "add %[a0_0], %[a0_0], %[sa]\n"
        "vle16.v v4, (%[a0_1])\n"
        "add %[a0_1], %[a0_1], %[sa]\n"

        "bltu %[k], %[i4], 1f\n"

        "0:\n"
        "addi %[k], %[k], -2\n"
        "vle16.v v24, (%[b1_0])\n"
        "add %[b1_0], %[b1_0], %[sb]\n"
        "vle16.v v28, (%[b1_1])\n"
        "add %[b1_1], %[b1_1], %[sb]\n"

        "sf.mm.f.f mt0, v0, v16\n"

        "vle16.v v8, (%[a1_0])\n"
        "add %[a1_0], %[a1_0], %[sa]\n"
        "vle16.v v12, (%[a1_1])\n"
        "add %[a1_1], %[a1_1], %[sa]\n"

        "sf.mm.f.f mt4, v0, v24\n"

        "vle16.v v0, (%[a0_0])\n"
        "add %[a0_0], %[a0_0], %[sa]\n"
        "vle16.v v4, (%[a0_1])\n"
        "add %[a0_1], %[a0_1], %[sa]\n"

        "sf.mm.f.f mt8, v8, v16\n"

        "vle16.v v16, (%[b0_0])\n"
        "add %[b0_0], %[b0_0], %[sb]\n"
        "vle16.v v20, (%[b0_1])\n"
        "add %[b0_1], %[b0_1], %[sb]\n"

        "sf.mm.f.f mt12, v8, v24\n"
        "bgeu %[k], %[i4], 0b\n"

        "1:\n"
        "addi %[k], %[k], -2\n"
        "vle16.v v24, (%[b1_0])\n"
        "add %[b1_0], %[b1_0], %[sb]\n"
        "vle16.v v28, (%[b1_1])\n"

        "sf.mm.f.f mt0, v0, v16\n"

        "vle16.v v8, (%[a1_0])\n"
        "add %[a1_0], %[a1_0], %[sa]\n"
        "vle16.v v12, (%[a1_1])\n"

        "sf.mm.f.f mt4, v0, v24\n"

        "beqz %[k], 2f\n"

        // k % 2 == 1
        "vle16.v v0, (%[a0_0])\n"

        "sf.mm.f.f mt8, v8, v16\n"

        "vle16.v v16, (%[b0_0])\n"

        "sf.mm.f.f mt12, v8, v24\n"

        "vle16.v v24, (%[b1_0])\n"

        "sf.vsettk x0, %[k]\n"
        "sf.mm.f.f mt0, v0, v16\n"

        "vle16.v v8, (%[a1_0])\n"

        "sf.mm.f.f mt4, v0, v24\n"
        "sf.mm.f.f mt8, v8, v16\n"
        "sf.mm.f.f mt12, v8, v24\n"
        "j 3f\n"

        "2:\n" // k % 2 == 0
        "sf.mm.f.f mt8, v8, v16\n"
        "sf.mm.f.f mt12, v8, v24\n"

        "3:\n"
        : [a0_0] "+&r"(a0_0), [a0_1] "+&r"(a0_1), [a1_0] "+&r"(a1_0),
          [a1_1] "+&r"(a1_1), [b0_0] "+&r"(b0_0), [b0_1] "+&r"(b0_1),
          [b1_0] "+&r"(b1_0), [b1_1] "+&r"(b1_1), [k] "+&r"(k)
        : [sa] "r"(2 * csa0 * sizeof(__bf16)),
          [sb] "r"(2 * rsb0 * sizeof(__bf16)), [tm] "r"(tm), [tn] "r"(tn),
          [i4] "r"(4)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31", "vtype", "vl", "memory");
  }

  /* Store tiles to memory. */
  size_t mt0_store = mt0;
  size_t mt4_store = mt4;
  size_t mt8_store = mt8;
  size_t mt12_store = mt12;

  float *c00_store = c00;
  float *c01_store = c01;
  float *c10_store = c10;
  float *c11_store = c11;
  i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, %[tn], e32, w1\n"

      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vste32 %[mt0], (%[c00])\n"
      "add %[mt0], %[mt0], %[kRowInc]\n"
      "add %[c00], %[c00], %[sc]\n"
      "sf.vste32 %[mt4], (%[c01])\n"
      "add %[mt4], %[mt4], %[kRowInc]\n"
      "add %[c01], %[c01], %[sc]\n"
      "sf.vste32 %[mt8], (%[c10])\n"
      "add %[mt8], %[mt8], %[kRowInc]\n"
      "add %[c10], %[c10], %[sc]\n"
      "sf.vste32 %[mt12], (%[c11])\n"
      "add %[mt12], %[mt12], %[kRowInc]\n"
      "add %[c11], %[c11], %[sc]\n"
      "bltu %[i], %[tm], 0b\n"

      "sf.vtdiscard"
      : [mt0] "+&r"(mt0_store), [mt4] "+&r"(mt4_store), [mt8] "+&r"(mt8_store),
        [mt12] "+&r"(mt12_store), [c00] "+&r"(c00_store),
        [c01] "+&r"(c01_store), [c10] "+&r"(c10_store), [c11] "+&r"(c11_store),
        [i] "+&r"(i)
      : [kRowInc] "rI"(kRowInc), [sc] "r"(rsc0 * sizeof(float)), [tm] "r"(tm),
        [tn] "r"(tn)
      : "vtype", "vl", "memory");
}

SKL_XSFMM_NEW
SKL_FUNC void skl_gemm_a1b01_bf16c_bf16_f32_xsfmm32a16f(
    size_t m, size_t n, size_t k, const __bf16 *a, size_t csa, const __bf16 *b,
    size_t rsb, float *c, size_t rsc, bool accum) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW=32).
  __asm__ volatile("sf.vsettnt %0, x0, e16alt, w2"
                   : "=r"(ete)
                   :
                   : "vtype", "vl");

  size_t i = 0;
  for (; i + 4 * ete <= m; i += 4 * ete) {
    size_t j = 0;
    for (; j + 2 * ete <= n; j += 2 * ete) {
      skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
          ete, k, a + i, csa, ete, b + j, rsb, ete, c + i * rsc + j, rsc,
          ete * rsc, ete, accum);
      skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
          ete, k, a + i + 2 * ete, csa, ete, b + j, rsb, ete,
          c + (i + 2 * ete) * rsc + j, rsc, ete * rsc, ete, accum);
    }
    while (j < n) {
      size_t tn = 0;
      __asm__ volatile("sf.vsettnt %0, %1, e16alt, w2"
                       : "=r"(tn)
                       : "r"(n - j)
                       : "vtype", "vl");
      skl_gemm_4tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
          ete, tn, k, a + i, csa, ete, b + j, rsb, c + i * rsc + j, rsc,
          ete * rsc, accum);
      j += tn;
    }
  }

  if (i + 2 * ete <= m) {
    size_t j = 0;
    for (; j + 2 * ete <= n; j += 2 * ete) {
      skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
          ete, k, a + i, csa, ete, b + j, rsb, ete, c + i * rsc + j, rsc,
          ete * rsc, ete, accum);
    }
    while (j < n) {
      size_t tn = 0;
      __asm__ volatile("sf.vsettnt %0, %1, e16alt, w2"
                       : "=r"(tn)
                       : "r"(n - j)
                       : "vtype", "vl");
      skl_gemm_2tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
          ete, tn, k, a + i, csa, ete, b + j, rsb, c + i * rsc + j, rsc,
          ete * rsc, accum);
      j += tn;
    }
    i += 2 * ete;
  }

  while (i < m) {
    size_t tm = 0;
    __asm__ volatile("sf.vsettnt x0, x0, e16alt, w2\n"
                     "sf.vsettm %0, %1\n"
                     : "=r"(tm)
                     : "r"(m - i)
                     : "vtype", "vl");
    size_t j = 0;
    for (; j + 4 * ete <= n; j += 4 * ete) {
      skl_gemm_1tm4tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
          tm, ete, k, a + i, csa, b + j, rsb, ete, c + i * rsc + j, rsc, ete,
          accum);
    }
    if (j + 3 * ete <= n) {
      skl_gemm_1tm3tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
          tm, ete, k, a + i, csa, b + j, rsb, ete, c + i * rsc + j, rsc, ete,
          accum);
      j += 3 * ete;
    }
    if (j + 2 * ete <= n) {
      skl_gemm_1tm2tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
          tm, ete, k, a + i, csa, b + j, rsb, ete, c + i * rsc + j, rsc, ete,
          accum);
      j += 2 * ete;
    }
    while (j < n) {
      size_t tn = 0;
      __asm__ volatile("sf.vsettnt %0, %1, e16alt, w2"
                       : "=r"(tn)
                       : "r"(n - j)
                       : "vtype", "vl");
      skl_gemm_1tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
          tm, tn, k, a + i, csa, b + j, rsb, c + i * rsc + j, rsc, accum);
      j += tn;
    }
    i += tm;
  }

  __asm__ volatile("sf.vtdiscard");
}

SKL_XSFMM_NEW
SKL_FUNC void skl_gemm_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
    size_t m1, size_t n1, size_t k, const __bf16 *a_pack, size_t rsa1,
    const __bf16 *b_pack, size_t csb1, float *c_pack, size_t rsc1, size_t csc1,
    bool accum) {
  if (m1 == 0 || n1 == 0) {
    return;
  }

  size_t ete = 0; // Effective tile edge length (always TE for TEW=32).
  __asm__ volatile("sf.vsettnt %0, x0, e16alt, w2"
                   : "=r"(ete)
                   :
                   : "vtype", "vl");

  size_t i = 0;
  for (; i + 4 <= m1; i += 4) {
    size_t j = 0;
    for (; j + 2 <= n1; j += 2) {
      skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
          ete, k, a_pack + i * rsa1, ete, rsa1, b_pack + j * csb1, ete, csb1,
          c_pack + i * rsc1 + j * csc1, ete, rsc1, csc1, accum);
      skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
          ete, k, a_pack + (i + 2) * rsa1, ete, rsa1, b_pack + j * csb1, ete,
          csb1, c_pack + (i + 2) * rsc1 + j * csc1, ete, rsc1, csc1, accum);
    }
    if (j < n1) {
      skl_gemm_4tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
          ete, ete, k, a_pack + i * rsa1, ete, rsa1, b_pack + j * csb1, ete,
          c_pack + i * rsc1 + j * csc1, ete, rsc1, accum);
    }
  }

  if (i + 2 <= m1) {
    size_t j = 0;
    for (; j + 2 <= n1; j += 2) {
      skl_gemm_2tm2tn_a1b01_bf16pc_bf16cp_f32rcp_xsfmm32a16f(
          ete, k, a_pack + i * rsa1, ete, rsa1, b_pack + j * csb1, ete, csb1,
          c_pack + i * rsc1 + j * csc1, ete, rsc1, csc1, accum);
    }
    if (j < n1) {
      skl_gemm_2tm1tn_a1b01_bf16pc_bf16_f32rcp_xsfmm32a16f(
          ete, ete, k, a_pack + i * rsa1, ete, rsa1, b_pack + j * csb1, ete,
          c_pack + i * rsc1 + j * csc1, ete, rsc1, accum);
    }
    i += 2;
  }

  if (i < m1) {
    size_t j = 0;
    for (; j + 4 <= n1; j += 4) {
      skl_gemm_1tm4tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
          ete, ete, k, a_pack + i * rsa1, ete, b_pack + j * csb1, ete, csb1,
          c_pack + i * rsc1 + j * csc1, ete, csc1, accum);
    }
    switch (n1 - j) {
    case 3:
      skl_gemm_1tm3tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
          ete, ete, k, a_pack + i * rsa1, ete, b_pack + j * csb1, ete, csb1,
          c_pack + i * rsc1 + j * csc1, ete, csc1, accum);
      break;
    case 2:
      skl_gemm_1tm2tn_a1b01_bf16c_bf16cp_f32rcp_xsfmm32a16f(
          ete, ete, k, a_pack + i * rsa1, ete, b_pack + j * csb1, ete, csb1,
          c_pack + i * rsc1 + j * csc1, ete, csc1, accum);
      break;
    case 1:
      skl_gemm_1tm1tn_a1b01_bf16c_bf16_f32_xsfmm32a16f(
          ete, ete, k, a_pack + i * rsa1, ete, b_pack + j * csb1, ete,
          c_pack + i * rsc1 + j * csc1, ete, accum);
      break;
    default:
      break;
    }
  }

  __asm__ volatile("sf.vtdiscard");
}
