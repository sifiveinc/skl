// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_load_tile_e32_xsfmmbase(size_t tm, size_t tn,
                                                  const uint32_t *a, size_t rsa,
                                                  size_t tss) {
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kRowInc = 1;

  size_t i = 0;
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"

                   "0:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vlte32 %[tss], (%[a])\n"
                   "add %[tss], %[tss], %[kRowInc]\n"
                   "add %[a], %[a], %[sa]\n"
                   "bltu %[i], %[tm], 0b\n"
                   : [tss] "+&r"(tss), [a] "+&r"(a), [i] "+&r"(i)
                   : [kRowInc] "rI"(kRowInc), [sa] "r"(rsa * sizeof(uint32_t)),
                     [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl", "memory");
}

// *0: store tss0 to a0
// *1: load a1 into tss1
SKL_XSFMM_INOUT
SKL_FUNC_PRIVATE void skl_load_store_tile_full_full_e32_xsfmmbase(
    size_t m0, size_t tm0, size_t tn0, size_t tss0, uint32_t *a0, size_t rsa0,
    size_t tm1, size_t tn1, const uint32_t *a1, size_t rsa1, size_t tss1) {
  if ((m0 == 0 || tn0 == 0) && (tm1 == 0 || tn1 == 0)) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(0, tn0);
  const size_t kRowInc = 1;

  size_t k0 = tm0 <= tm1 ? tm0 : tm1;
  size_t k1 = m0 <= tm1 ? m0 : tm1;
  size_t i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, x0, e32, w1\n"

      "bgeu %[i], %[k0], 1f\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vsettn x0, %[tn0]\n"
      "sf.vste32 %[tss0], (%[a0])\n"
      "add %[tss0], %[tss0], %[kRowInc]\n"
      "add %[a0], %[a0], %[sa0]\n"
      "sf.vsettn x0, %[tn1]\n"
      "sf.vlte32 %[tss1], (%[a1])\n"
      "add %[tss1], %[tss1], %[kRowInc]\n"
      "add %[a1], %[a1], %[sa1]\n"
      "bltu %[i], %[k0], 0b\n"

      "1:\n"
      "bgeu %[i], %[k1], 2f\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vsettn x0, %[tn0]\n"
      "vse32.v %[pad], (%[a0])\n"
      "add %[a0], %[a0], %[sa0]\n"
      "sf.vsettn x0, %[tn1]\n"
      "sf.vlte32 %[tss1], (%[a1])\n"
      "add %[tss1], %[tss1], %[kRowInc]\n"
      "add %[a1], %[a1], %[sa1]\n"
      "bltu %[i], %[k1], 0b\n"

      "2:\n"
      "bgeu %[i], %[tm0], 3f\n"
      "sf.vsettn x0, %[tn0]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vste32 %[tss0], (%[a0])\n"
      "add %[tss0], %[tss0], %[kRowInc]\n"
      "add %[a0], %[a0], %[sa0]\n"
      "bltu %[i], %[tm0], 0b\n"

      "3:\n"
      "bgeu %[i], %[m0], 4f\n"
      "sf.vsettn x0, %[tn0]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "vse32.v %[pad], (%[a0])\n"
      "add %[a0], %[a0], %[sa0]\n"
      "bltu %[i], %[m0], 0b\n"

      "4:\n"
      "bgeu %[i], %[tm1], 5f\n"
      "sf.vsettn x0, %[tn1]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vlte32 %[tss1], (%[a1])\n"
      "add %[tss1], %[tss1], %[kRowInc]\n"
      "add %[a1], %[a1], %[sa1]\n"
      "bltu %[i], %[tm1], 0b\n"

      "5:\n"
      : [tss0] "+&r"(tss0), [tss1] "+&r"(tss1), [a0] "+&r"(a0), [a1] "+&r"(a1),
        [i] "+&r"(i)
      : [pad] "vr"(pad), [kRowInc] "rI"(kRowInc),
        [sa0] "r"(rsa0 * sizeof(float)), [sa1] "r"(rsa1 * sizeof(float)),
        [k0] "r"(k0), [k1] "r"(k1), [m0] "r"(m0), [tm0] "r"(tm0),
        [tn0] "r"(tn0), [tm1] "r"(tm1), [tn1] "r"(tn1)
      : "vtype", "vl", "memory");
}

SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_store_tile_e32_xsfmmbase(size_t m0, size_t tm,
                                                   size_t tn, const uint32_t *a,
                                                   size_t rsa, size_t tss) {
  if (m0 == 0 || tn == 0) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(0, tn);
  const size_t kRowInc = 1;

  size_t i = 0;
  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"

                   "bgeu %[i], %[tm], 1f\n"
                   "0:\n"
                   "addi %[i], %[i], 1\n"
                   "sf.vste32 %[tss], (%[a])\n"
                   "add %[tss], %[tss], %[kRowInc]\n"
                   "add %[a], %[a], %[sa]\n"
                   "bltu %[i], %[tm], 0b\n"

                   "1:\n"
                   "bgeu %[i], %[m0], 2f\n"
                   "0:\n"
                   "addi %[i], %[i], 1\n"
                   "vse32.v %[pad], (%[a])\n"
                   "add %[a], %[a], %[sa]\n"
                   "bltu %[i], %[m0], 0b\n"

                   "2:\n"
                   : [tss] "+&r"(tss), [a] "+&r"(a), [i] "+&r"(i)
                   : [pad] "vr"(pad), [kRowInc] "rI"(kRowInc),
                     [sa] "r"(rsa * sizeof(uint32_t)), [m0] "r"(m0),
                     [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl", "memory");
}

SKL_XSFMM_NEW
SKL_FUNC void skl_pack_transpose_e32_inner_transpose_xsfmmbase(
    size_t m, size_t n, const uint32_t *a, size_t rsa, size_t m0, size_t n0,
    uint32_t *b, size_t rsb0, size_t csb0, size_t rsb1, size_t csb1) {
  // for now, assume no bottom padding required (m % m0 == 0)
  size_t m1 = (m + m0 - 1) / m0;
  size_t n1 = (n + n0 - 1) / n0;
  if (m1 == 0 || n1 == 0) {
    return;
  }

  const size_t mt0 = 0;
  const size_t mt0c = mt0 | (size_t)1 << 24;
  const size_t mt4 = (size_t)4 << 27;
  const size_t mt4c = mt4 | (size_t)1 << 24;
  size_t nb0 = n0 <= n ? n0 : n;
  size_t nb1 = 0;
  skl_load_tile_e32_xsfmmbase(m0, nb0, a, rsa, mt0);
  size_t i1 = 0;
  size_t j1 = 0;
  for (; i1 + 1 < m1; i1 += 2) {
    size_t avl = n;
    for (; j1 + 1 < n1; ++j1) {
      avl -= nb0;
      nb1 = n0 <= avl ? n0 : avl;
      skl_load_store_tile_full_full_e32_xsfmmbase(
          n0, nb0, m0, mt0c, b + i1 * rsb1 + j1 * csb1, csb0, m0, nb0,
          a + (i1 + 1) * m0 * rsa + j1 * n0, rsa, mt4);
      skl_load_store_tile_full_full_e32_xsfmmbase(
          n0, nb0, m0, mt4c, b + (i1 + 1) * rsb1 + j1 * csb1, csb0, m0, nb1,
          a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt0);
      nb0 = nb1;
    }
    skl_load_store_tile_full_full_e32_xsfmmbase(
        n0, nb0, m0, mt0c, b + i1 * rsb1 + j1 * csb1, csb0, m0, nb0,
        a + (i1 + 1) * m0 * rsa + j1 * n0, rsa, mt4);
    if (i1 + 2 < m1) {
      nb1 = n0 <= n ? n0 : n;
      skl_load_store_tile_full_full_e32_xsfmmbase(
          n0, nb0, m0, mt4c, b + (i1 + 1) * rsb1 + j1 * csb1, csb0, m0, nb1,
          a + (i1 + 2) * m0 * rsa + 0 * n0, rsa, mt0);
      nb0 = nb1;
    } else {
      skl_store_tile_e32_xsfmmbase(n0, nb0, m0, b + (i1 + 1) * rsb1 + j1 * csb1,
                                   csb0, mt4c);
      return;
    }
  }

  // m1 % 2 == 1
  size_t avl = n;
  size_t nb2 = 0;
  for (size_t j1 = 0; j1 + 2 < n1; j1 += 2) {
    avl -= nb0;
    nb1 = n0 <= avl ? n0 : avl;
    avl -= nb1;
    nb2 = n0 <= avl ? n0 : avl;
    skl_load_store_tile_full_full_e32_xsfmmbase(
        n0, nb0, m0, mt0c, b + i1 * rsb1 + j1 * csb1, csb0, m0, nb1,
        a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt4);
    skl_load_store_tile_full_full_e32_xsfmmbase(
        n0, nb1, m0, mt4c, b + i1 * rsb1 + (j1 + 1) * csb1, csb0, m0, nb2,
        a + i1 * m0 * rsa + (j1 + 2) * n0, rsa, mt0);
    nb0 = nb2;
  }
  if (n1 % 2) {
    skl_store_tile_e32_xsfmmbase(n0, nb0, m0, b + i1 * rsb1 + j1 * csb1, csb0,
                                 mt0c);
  } else {
    avl -= nb0;
    nb1 = n0 <= avl ? n0 : avl;
    skl_load_store_tile_full_full_e32_xsfmmbase(
        n0, nb0, m0, mt0c, b + i1 * rsb1 + j1 * csb1, csb0, m0, nb1,
        a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt4);
    skl_store_tile_e32_xsfmmbase(n0, nb1, m0, b + i1 * rsb1 + (j1 + 1) * csb1,
                                 csb0, mt4c);
  }
}

SKL_XSFMM_NEW
SKL_FUNC void skl_transpose_e32_xsfmmbase(size_t m, size_t n,
                                          const uint32_t *SKL_RESTRICT a,
                                          size_t rsa, uint32_t *SKL_RESTRICT at,
                                          size_t rsat) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t kShiftCol = 24;
  const size_t kInc = 1;
  size_t tm = 0;
  size_t tn = 0;
  for (size_t i = 0; i < m; i += tm) {
    for (size_t j = 0; j < n; j += tn) {
      size_t mt0_load = 0;
      size_t mt0_store = 1 << kShiftCol;

      const uint32_t *SKL_RESTRICT a_load = a + i * rsa + j;
      uint32_t *SKL_RESTRICT at_store = at + j * rsat + i;

      size_t k = 0;
      __asm__ volatile(
          "sf.vsettnt %[tn], %[avl_n], e32, w1\n"
          "sf.vsettm %[tm], %[avl_m]\n"

          "0:\n"
          "sf.vlte32 %[mt0_load], (%[a])\n"
          "add %[a], %[a], %[sa]\n"
          "add %[mt0_load], %[mt0_load], %[kInc]\n"
          "addi %[k], %[k], 1\n"
          "bltu %[k], %[tm], 0b\n"

          "li %[k], 0\n"
          "sf.vsettn x0, %[tm]\n"
          "1:\n"
          "sf.vste32 %[mt0_store], (%[at])\n"
          "add %[at], %[at], %[sat]\n"
          "add %[mt0_store], %[mt0_store], %[kInc]\n"
          "addi %[k], %[k], 1\n"
          "bltu %[k], %[tn], 1b\n"
          : [tm] "=&r"(tm), [tn] "=&r"(tn), [mt0_load] "+&r"(mt0_load),
            [mt0_store] "+&r"(mt0_store), [a] "+&r"(a_load),
            [at] "+&r"(at_store), [k] "+&r"(k)
          : [avl_m] "r"(m - i), [avl_n] "r"(n - j),
            [sa] "r"(rsa * sizeof(uint32_t)),
            [sat] "r"(rsat * sizeof(uint32_t)), [kInc] "rI"(kInc)
          : "vtype", "vl", "memory");
    }
  }
  __asm__ volatile("sf.vtdiscard");
}
