// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

#include <stdbool.h>
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

// Interleaves store of tss0 to a0 with load of a1 into tss1.
// Tile size is m0 x n0.
// tss0 is tm0 x tn0 tile with column pattern.
// tss1 is tm1 x tn1 tile with row pattern.
// Note: this function will only be called with tm0 == m0 or tn0 == n0 since the
// bottom right corner tile is stored last.
SKL_XSFMM_INOUT
SKL_FUNC_PRIVATE void skl_store_load_tile_e32c_e32_xsfmmbase(
    size_t tm0, size_t tn0, size_t tss0, size_t m0, size_t n0, uint32_t *a0,
    size_t csa0, size_t tm1, size_t tn1, const uint32_t *a1, size_t rsa1,
    size_t tss1, uint32_t padding_value) {
  if ((m0 == 0 || n0 == 0) && (tm1 == 0 || tn1 == 0)) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(padding_value, m0);

  const size_t kRowInc = 1;

  if (tm0 == m0) {
    size_t k0 = tn0 <= tm1 ? tn0 : tm1;
    size_t k1 = n0 <= tm1 ? n0 : tm1;
    size_t i = 0;
    __asm__ volatile("sf.vsettnt x0, x0, e32, w1\n"

                     "bgeu %[i], %[k0], 1f\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "sf.vsettn x0, %[tm0]\n"
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
                     "sf.vsettn x0, %[tm0]\n"
                     "vse32.v %[pad], (%[a0])\n"
                     "add %[a0], %[a0], %[sa0]\n"
                     "sf.vsettn x0, %[tn1]\n"
                     "sf.vlte32 %[tss1], (%[a1])\n"
                     "add %[tss1], %[tss1], %[kRowInc]\n"
                     "add %[a1], %[a1], %[sa1]\n"
                     "bltu %[i], %[k1], 0b\n"

                     "2:\n"
                     "bgeu %[i], %[tn0], 3f\n"
                     "sf.vsettn x0, %[tm0]\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "sf.vste32 %[tss0], (%[a0])\n"
                     "add %[tss0], %[tss0], %[kRowInc]\n"
                     "add %[a0], %[a0], %[sa0]\n"
                     "bltu %[i], %[tn0], 0b\n"

                     "3:\n"
                     "bgeu %[i], %[n0], 4f\n"
                     "sf.vsettn x0, %[tm0]\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "vse32.v %[pad], (%[a0])\n"
                     "add %[a0], %[a0], %[sa0]\n"
                     "bltu %[i], %[n0], 0b\n"

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
                     : [tss0] "+&r"(tss0), [tss1] "+&r"(tss1), [a0] "+&r"(a0),
                       [a1] "+&r"(a1), [i] "+&r"(i)
                     : [pad] "vr"(pad), [kRowInc] "rI"(kRowInc),
                       [sa0] "r"(csa0 * sizeof(uint32_t)),
                       [sa1] "r"(rsa1 * sizeof(uint32_t)), [k0] "r"(k0),
                       [k1] "r"(k1), [m0] "r"(m0), [n0] "r"(n0), [tm0] "r"(tm0),
                       [tn0] "r"(tn0), [tm1] "r"(tm1), [tn1] "r"(tn1)
                     : "vtype", "vl", "memory");
  } else { // tn0 == n0
    uint32_t *a0_pad = a0 + tm0;
    size_t k0 = tn0 <= tm1 ? tn0 : tm1;
    size_t i = 0;
    __asm__ volatile("sf.vsettnt x0, x0, e32, w1\n"

                     "bgeu %[i], %[k0], 1f\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "sf.vsettn x0, %[tm0]\n"
                     "sf.vste32 %[tss0], (%[a0])\n"
                     "add %[tss0], %[tss0], %[kRowInc]\n"
                     "add %[a0], %[a0], %[sa0]\n"
                     "sf.vsettn x0, %[tn1]\n"
                     "sf.vlte32 %[tss1], (%[a1])\n"
                     "add %[tss1], %[tss1], %[kRowInc]\n"
                     "add %[a1], %[a1], %[sa1]\n"
                     "bltu %[i], %[k0], 0b\n"

                     "1:\n"
                     "bgeu %[i], %[tn0], 2f\n"
                     "sf.vsettn x0, %[tm0]\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "sf.vste32 %[tss0], (%[a0])\n"
                     "add %[tss0], %[tss0], %[kRowInc]\n"
                     "add %[a0], %[a0], %[sa0]\n"
                     "bltu %[i], %[tn0], 0b\n"

                     "2:\n"
                     "bgeu %[i], %[tm1], 3f\n"
                     "sf.vsettn x0, %[tn1]\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "sf.vlte32 %[tss1], (%[a1])\n"
                     "add %[tss1], %[tss1], %[kRowInc]\n"
                     "add %[a1], %[a1], %[sa1]\n"
                     "bltu %[i], %[tm1], 0b\n"

                     "3:\n"
                     "beqz %[tn0], 4f\n"
                     "beqz %[tm_pad], 4f\n"
                     "li %[i], 0\n"
                     "sf.vsettn x0, %[tm_pad]\n"
                     "0:\n"
                     "addi %[i], %[i], 1\n"
                     "vse32.v %[pad], (%[a0_pad])\n"
                     "add %[a0_pad], %[a0_pad], %[sa0]\n"
                     "bltu %[i], %[tn0], 0b\n"

                     "4:\n"
                     : [tss0] "+&r"(tss0), [tss1] "+&r"(tss1), [a0] "+&r"(a0),
                       [a1] "+&r"(a1), [a0_pad] "+&r"(a0_pad), [i] "+&r"(i)
                     : [pad] "vr"(pad), [kRowInc] "rI"(kRowInc),
                       [sa0] "r"(csa0 * sizeof(uint32_t)),
                       [sa1] "r"(rsa1 * sizeof(uint32_t)), [k0] "r"(k0),
                       [m0] "r"(m0), [tm0] "r"(tm0), [tn0] "r"(tn0),
                       [tm1] "r"(tm1), [tn1] "r"(tn1), [tm_pad] "r"(m0 - tm0)
                     : "vtype", "vl", "memory");
  }
}

// tss is a tm x tn tile with column pattern
SKL_XSFMM_IN
SKL_FUNC_PRIVATE void skl_store_tile_e32c_xsfmmbase(size_t tm, size_t tn,
                                                    size_t tss, size_t m0,
                                                    size_t n0, uint32_t *a,
                                                    size_t csa,
                                                    uint32_t padding_value) {
  if (m0 == 0 || n0 == 0) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(padding_value, m0);
  const size_t kRowInc = 1;

  uint32_t *a_pad = a + tm;

  size_t i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, x0, e32, w1\n"

      "bgeu %[i], %[tn], 1f\n"
      "sf.vsettn x0, %[tm]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vste32 %[tss], (%[a])\n"
      "add %[tss], %[tss], %[kRowInc]\n"
      "add %[a], %[a], %[sa]\n"
      "bltu %[i], %[tn], 0b\n"

      "1:\n"
      "bgeu %[i], %[n0], 2f\n"
      "sf.vsettn x0, %[m0]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "vse32.v %[pad], (%[a])\n"
      "add %[a], %[a], %[sa]\n"
      "bltu %[i], %[n0], 0b\n"

      "2:\n"
      "beqz %[tn], 3f\n"
      "beqz %[tm_pad], 3f\n"
      "li %[i], 0\n"
      "sf.vsettn x0, %[tm_pad]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "vse32.v %[pad], (%[a_pad])\n"
      "add %[a_pad], %[a_pad], %[sa]\n"
      "bltu %[i], %[tn], 0b\n"

      "3:\n"
      : [tss] "+&r"(tss), [a] "+&r"(a), [a_pad] "+&r"(a_pad), [i] "+&r"(i)
      : [pad] "vr"(pad), [kRowInc] "rI"(kRowInc),
        [sa] "r"(csa * sizeof(uint32_t)), [m0] "r"(m0), [n0] "r"(n0),
        [tm] "r"(tm), [tn] "r"(tn), [tm_pad] "r"(m0 - tm)
      : "vtype", "vl", "memory");
}

SKL_XSFMM_NEW
SKL_FUNC_PRIVATE void skl_pack_e32_e32rcpc_xsfmmbase(
    size_t m, size_t n, const uint32_t *a, size_t rsa, size_t m0, size_t n0,
    uint32_t *a_pack, size_t csa0, size_t rsa1, size_t csa1,
    bool pad_right = true, bool pad_bottom = true, uint32_t padding_value = 0) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t m1 = (m + m0 - 1) / m0;
  const size_t n1 = (n + n0 - 1) / n0;

  const size_t mt0 = 0;
  const size_t mt0c = mt0 | (size_t)1 << 24;
  const size_t mt4 = (size_t)4 << 27;
  const size_t mt4c = mt4 | (size_t)1 << 24;

  size_t avl_m = m;
  size_t mb0 = m0 <= avl_m ? m0 : avl_m;
  size_t mb1 = 0;
  size_t nb0 = n0 <= n ? n0 : n;
  size_t nb1 = 0;

  size_t m0_bottom = pad_bottom ? m0 : (m - 1) % m0 + 1;
  size_t n0_right = pad_right ? n0 : (n - 1) % n0 + 1;

  skl_load_tile_e32_xsfmmbase(mb0, nb0, a, rsa, mt0);

  size_t i1 = 0;
  size_t j1 = 0;
  for (; i1 + 1 < m1; i1 += 2) {
    avl_m -= mb0;
    mb1 = m0 <= avl_m ? m0 : avl_m;
    size_t m0_2nd_row = i1 == m1 - 2 ? m0_bottom : m0;
    size_t avl_n = n;
    for (j1 = 0; j1 + 1 < n1; ++j1) {
      avl_n -= nb0;
      nb1 = n0 <= avl_n ? n0 : avl_n;
      skl_store_load_tile_e32c_e32_xsfmmbase(
          mb0, nb0, mt0c, m0, n0, a_pack + i1 * rsa1 + j1 * csa1, csa0, mb1,
          nb0, a + (i1 + 1) * m0 * rsa + j1 * n0, rsa, mt4, padding_value);
      skl_store_load_tile_e32c_e32_xsfmmbase(
          mb1, nb0, mt4c, m0_2nd_row, n0, a_pack + (i1 + 1) * rsa1 + j1 * csa1,
          csa0, mb0, nb1, a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt0,
          padding_value);
      nb0 = nb1;
    }
    skl_store_load_tile_e32c_e32_xsfmmbase(
        mb0, nb0, mt0c, m0, n0_right, a_pack + i1 * rsa1 + j1 * csa1, csa0, mb1,
        nb0, a + (i1 + 1) * m0 * rsa + j1 * n0, rsa, mt4, padding_value);
    if (i1 + 2 < m1) {
      avl_m -= mb1;
      mb0 = m0 <= avl_m ? m0 : avl_m;
      nb1 = n0 <= n ? n0 : n;
      skl_store_load_tile_e32c_e32_xsfmmbase(
          mb1, nb0, mt4c, m0, n0_right, a_pack + (i1 + 1) * rsa1 + j1 * csa1,
          csa0, mb0, nb1, a + (i1 + 2) * m0 * rsa + 0 * n0, rsa, mt0,
          padding_value);
      nb0 = nb1;
    } else {
      skl_store_tile_e32c_xsfmmbase(mb1, nb0, mt4c, m0_2nd_row, n0_right,
                                    a_pack + (i1 + 1) * rsa1 + j1 * csa1, csa0,
                                    padding_value);
      return;
    }
  }

  // m1 % 2 == 1
  size_t avl_n = n;
  size_t nb2 = 0;
  for (j1 = 0; j1 + 2 < n1; j1 += 2) {
    avl_n -= nb0;
    nb1 = n0 <= avl_n ? n0 : avl_n;
    avl_n -= nb1;
    nb2 = n0 <= avl_n ? n0 : avl_n;
    skl_store_load_tile_e32c_e32_xsfmmbase(
        mb0, nb0, mt0c, m0_bottom, n0, a_pack + i1 * rsa1 + j1 * csa1, csa0,
        mb0, nb1, a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt4, padding_value);
    skl_store_load_tile_e32c_e32_xsfmmbase(
        mb0, nb1, mt4c, m0_bottom, n0, a_pack + i1 * rsa1 + (j1 + 1) * csa1,
        csa0, mb0, nb2, a + i1 * m0 * rsa + (j1 + 2) * n0, rsa, mt0,
        padding_value);
    nb0 = nb2;
  }
  if (n1 % 2) {
    skl_store_tile_e32c_xsfmmbase(mb0, nb0, mt0c, m0_bottom, n0_right,
                                  a_pack + i1 * rsa1 + j1 * csa1, csa0,
                                  padding_value);
  } else {
    avl_n -= nb0;
    nb1 = n0 <= avl_n ? n0 : avl_n;
    skl_store_load_tile_e32c_e32_xsfmmbase(
        mb0, nb0, mt0c, m0_bottom, n0, a_pack + i1 * rsa1 + j1 * csa1, csa0, m0,
        nb1, a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt4, padding_value);
    skl_store_tile_e32c_xsfmmbase(mb0, nb1, mt4c, m0_bottom, n0_right,
                                  a_pack + i1 * rsa1 + (j1 + 1) * csa1, csa0,
                                  padding_value);
  }
}

SKL_FUNC_PRIVATE void
skl_pack_tex1_e32_e32pc_xsfmmbase(size_t m, size_t n, const uint32_t *a,
                                  size_t rsa, uint32_t *a_pack, size_t csa0,
                                  size_t rsa1, uint32_t padding_value = 0) {
  size_t te = 0;
  __asm__ volatile("sf.vsettnt %[te], x0, e32, w1"
                   : [te] "=r"(te)
                   :
                   : "vtype", "vl");
  skl_pack_e32_e32rcpc_xsfmmbase(m, n, a, rsa, te, te, a_pack, csa0, rsa1,
                                 te * te, false, true, padding_value);
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
