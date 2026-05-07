// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfmmbase)
#error This file requires the Xsfmmbase extension
#endif

#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

/* Load a tm x tn row-major matrix a into the tile specified by tss.
 * tm and tn must be <= ETE.
 */
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

/* Store the tm x n0 tile specified by tss into an m0 x n0 row-major matrix a.
 *
 * This function requires tm <= m0 <= ETE and tn <= n0 <= ETE. If tss has the
 * column pattern bit set, tm is the number of columns and tn is the column
 * length. If tm < m0, bottom padding is added. The padding value is set by
 * padding_value.
 */
SKL_XSFMM_IN
SKL_FUNC_PRIVATE void skl_store_tile_no_right_pad_e32_xsfmmbase(
    size_t tm, size_t tss, size_t m0, size_t n0,
    uint32_t *a, // NOLINT(readability-non-const-parameter)
    size_t rsa, uint32_t padding_value) {
  if (m0 == 0 || n0 == 0) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(padding_value, n0);
  const size_t kRowInc = 1;

  size_t i = 0;
  __asm__ volatile("sf.vsettnt x0, %[n0], e32, w1\n"

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
                     [n0] "r"(n0), [tm] "r"(tm)
                   : "vtype", "vl", "memory");
}

/*
 * Equivalent to:
 * skl_store_tile_e32_xsfmmbase(tm0, tn0, tss0, m0, n0, a0, rsa0,
 *                              padding_value);
 * skl_load_tile_e32_xsfmmbase(tm1, tn1, a1, rsa1, tss1);
 *
 * This kernel overlaps store and load execution for improved performance over
 * storing tss0 first and then loading a1.
 */
SKL_XSFMM_INOUT
SKL_FUNC_PRIVATE void skl_store_load_tile_e32_e32_xsfmmbase(
    size_t tm0, size_t tss0, size_t m0, size_t n0,
    uint32_t *a0, // NOLINT(readability-non-const-parameter)
    size_t rsa0, size_t tm1, size_t tn1, const uint32_t *a1, size_t rsa1,
    size_t tss1, uint32_t padding_value) {
  if ((m0 == 0 || n0 == 0) && (tm1 == 0 || tn1 == 0)) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(padding_value, n0);

  const size_t kRowInc = 1;

  size_t k0 = tm0 <= tm1 ? tm0 : tm1;
  size_t k1 = m0 <= tm1 ? m0 : tm1;
  size_t i = 0;
  __asm__ volatile(
      "sf.vsettnt x0, x0, e32, w1\n"

      "bgeu %[i], %[k0], 1f\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vsettn x0, %[n0]\n"
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
      "sf.vsettn x0, %[n0]\n"
      "vse32.v %[pad], (%[a0])\n"
      "add %[a0], %[a0], %[sa0]\n"
      "sf.vsettn x0, %[tn1]\n"
      "sf.vlte32 %[tss1], (%[a1])\n"
      "add %[tss1], %[tss1], %[kRowInc]\n"
      "add %[a1], %[a1], %[sa1]\n"
      "bltu %[i], %[k1], 0b\n"

      "2:\n"
      "bgeu %[i], %[tm0], 3f\n"
      "sf.vsettn x0, %[n0]\n"
      "0:\n"
      "addi %[i], %[i], 1\n"
      "sf.vste32 %[tss0], (%[a0])\n"
      "add %[tss0], %[tss0], %[kRowInc]\n"
      "add %[a0], %[a0], %[sa0]\n"
      "bltu %[i], %[tm0], 0b\n"

      "3:\n"
      "bgeu %[i], %[m0], 4f\n"
      "sf.vsettn x0, %[n0]\n"
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
        [sa0] "r"(rsa0 * sizeof(uint32_t)), [sa1] "r"(rsa1 * sizeof(uint32_t)),
        [k0] "r"(k0), [k1] "r"(k1), [m0] "r"(m0), [n0] "r"(n0), [tm0] "r"(tm0),
        [tm1] "r"(tm1), [tn1] "r"(tn1)
      : "vtype", "vl", "memory");
}

/* Pack a into packed matrix a_pack with m0 x n0 column-major blocks.
 * pad_right and pad_bottom determine whether right and bottom padding,
 * respectively, are written. m0 and n0 must be > 0 and <= ETE.
 */
SKL_XSFMM_OUT
SKL_FUNC_PRIVATE void skl_pack_e32_e32rcpc_xsfmmbase(
    size_t m, size_t n, const uint32_t *a, size_t rsa, size_t m0, size_t n0,
    uint32_t *a_pack, size_t csa0, size_t rsa1, size_t csa1, bool pad_right,
    bool pad_bottom, uint32_t padding_value) {
  if (m == 0 || n == 0) {
    return;
  }

  vuint32m8_t pad = __riscv_vmv_v_x_u32m8(padding_value, n0);

  const size_t kRowInc = 1;
  const size_t mt0 = 0;
  const size_t mt0c = mt0 | (size_t)1 << 24;
  const size_t mt4 = (size_t)4 << 27;
  const size_t mt4c = mt4 | (size_t)1 << 24;

  const size_t m1 = (m + m0 - 1) / m0;
  const size_t n1 = (n + n0 - 1) / n0;

  size_t tm_bottom = (m - 1) % m0 + 1;
  size_t tn_right = (n - 1) % n0 + 1;
  size_t m0_bottom = pad_bottom ? m0 : tm_bottom;
  size_t n0_right = pad_right ? n0 : tn_right;

  size_t i1 = 0;
  size_t j1 = 0;

  size_t tm0 = m1 == 1 ? tm_bottom : m0;
  size_t tm1 = 0;
  size_t tn0 = n1 == 1 ? tn_right : n0;
  size_t tn1 = 0;

  skl_load_tile_e32_xsfmmbase(tm0, tn0, a, rsa, mt0);

  // NOLINTBEGIN(readability-suspicious-call-argument)
  for (; i1 + 1 < m1; i1 += 2) {
    tm1 = m0;
    size_t m0_2nd_row = m0;
    if (i1 == m1 - 2) {
      tm1 = tm_bottom;
      m0_2nd_row = m0_bottom;
      size_t tss4 = mt4 + tm_bottom * kRowInc;
      size_t tss4_end = mt4 + m0_bottom * kRowInc;
      __asm__ volatile("bgeu %[tss4], %[tss4_end], 1f\n"
                       "sf.vsettnt x0, %[n0], e32, w1\n"
                       "0:\n"
                       "sf.vtmv.t.v %[tss4], %[pad]\n"
                       "add %[tss4], %[tss4], %[kRowInc]\n"
                       "bltu %[tss4], %[tss4_end], 0b\n"
                       "1:\n"
                       : [tss4] "+&r"(tss4)
                       : [pad] "vr"(pad), [n0] "r"(n0),
                         [tss4_end] "r"(tss4_end), [kRowInc] "rI"(kRowInc)
                       : "vtype", "vl");
    }
    for (j1 = 0; j1 + 1 < n1; ++j1) {
      tn1 = j1 == n1 - 2 ? tn_right : n0;
      skl_store_load_tile_e32_e32_xsfmmbase(
          tn0, mt0c, n0, m0, a_pack + i1 * rsa1 + j1 * csa1, csa0, tm1, tn0,
          a + (i1 + 1) * m0 * rsa + j1 * n0, rsa, mt4, padding_value);
      skl_store_load_tile_e32_e32_xsfmmbase(
          tn0, mt4c, n0, m0_2nd_row, a_pack + (i1 + 1) * rsa1 + j1 * csa1, csa0,
          tm0, tn1, a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt0, padding_value);
      tn0 = tn1;
    }
    skl_store_load_tile_e32_e32_xsfmmbase(
        tn0, mt0c, n0_right, m0, a_pack + i1 * rsa1 + j1 * csa1, csa0, tm1, tn0,
        a + (i1 + 1) * m0 * rsa + j1 * n0, rsa, mt4, padding_value);
    if (i1 < m1 - 2) {
      tm0 = i1 == m1 - 3 ? tm_bottom : m0;
      tn1 = n1 == 1 ? tn_right : n0;
      skl_store_load_tile_e32_e32_xsfmmbase(
          tn0, mt4c, n0_right, m0_2nd_row, a_pack + (i1 + 1) * rsa1 + j1 * csa1,
          csa0, tm0, tn1, a + (i1 + 2) * m0 * rsa + 0 * n0, rsa, mt0,
          padding_value);
      tn0 = tn1;
    } else {
      skl_store_tile_no_right_pad_e32_xsfmmbase(
          tn0, mt4c, n0_right, m0_2nd_row, a_pack + (i1 + 1) * rsa1 + j1 * csa1,
          csa0, padding_value);
      return;
    }
  }

  // m1 % 2 == 1
  size_t tss0 = mt0 + tm_bottom * kRowInc;
  size_t tss4 = mt4 + tm_bottom * kRowInc;
  size_t tss4_end = mt4 + m0_bottom * kRowInc;
  __asm__ volatile("bgeu %[tss4], %[tss4_end], 1f\n"
                   "sf.vsettnt x0, %[n0], e32, w1\n"
                   "0:\n"
                   "sf.vtmv.t.v %[tss0], %[pad]\n"
                   "sf.vtmv.t.v %[tss4], %[pad]\n"
                   "add %[tss0], %[tss0], %[kRowInc]\n"
                   "add %[tss4], %[tss4], %[kRowInc]\n"
                   "bltu %[tss4], %[tss4_end], 0b\n"
                   "1:\n"
                   : [tss0] "+&r"(tss0), [tss4] "+&r"(tss4)
                   : [pad] "vr"(pad), [n0] "r"(n0), [tss4_end] "r"(tss4_end),
                     [kRowInc] "rI"(kRowInc)
                   : "vtype", "vl");

  size_t tn2 = 0;
  for (j1 = 0; j1 + 2 < n1; j1 += 2) {
    tn1 = n0;
    tn2 = j1 == n1 - 3 ? tn_right : n0;
    skl_store_load_tile_e32_e32_xsfmmbase(
        tn0, mt0c, n0, m0_bottom, a_pack + i1 * rsa1 + j1 * csa1, csa0, tm0,
        tn1, a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt4, padding_value);
    skl_store_load_tile_e32_e32_xsfmmbase(
        tn1, mt4c, n0, m0_bottom, a_pack + i1 * rsa1 + (j1 + 1) * csa1, csa0,
        tm0, tn2, a + i1 * m0 * rsa + (j1 + 2) * n0, rsa, mt0, padding_value);
    tn0 = tn2;
  }
  if (n1 % 2) {
    skl_store_tile_no_right_pad_e32_xsfmmbase(tn0, mt0c, n0_right, m0_bottom,
                                              a_pack + i1 * rsa1 + j1 * csa1,
                                              csa0, padding_value);
  } else {
    tn1 = tn_right;
    skl_store_load_tile_e32_e32_xsfmmbase(
        tn0, mt0c, n0, m0_bottom, a_pack + i1 * rsa1 + j1 * csa1, csa0, tm0,
        tn1, a + i1 * m0 * rsa + (j1 + 1) * n0, rsa, mt4, padding_value);
    skl_store_tile_no_right_pad_e32_xsfmmbase(
        tn1, mt4c, n0_right, m0_bottom, a_pack + i1 * rsa1 + (j1 + 1) * csa1,
        csa0, padding_value);
  }
  // NOLINTEND(readability-suspicious-call-argument)
}

SKL_XSFMM_NEW
SKL_FUNC void skl_pack_tex1_e32_e32pc_xsfmmbase(size_t m, size_t n,
                                                const uint32_t *a, size_t rsa,
                                                uint32_t *a_pack, size_t rsa1,
                                                uint32_t padding_value) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t te = 0;
  __asm__ volatile("sf.vsettnt %[te], x0, e32, w1"
                   : [te] "=r"(te)
                   :
                   : "vtype", "vl");

  skl_pack_e32_e32rcpc_xsfmmbase(m, n, a, rsa, te, te, a_pack, te, rsa1,
                                 te * te, false, true, padding_value);

  __asm__ volatile("sf.vtdiscard");
}

SKL_XSFMM_NEW
SKL_FUNC void skl_transpose_e32_xsfmmbase(size_t m, size_t n,
                                          const uint32_t *SKL_RESTRICT a,
                                          size_t rsa, uint32_t *SKL_RESTRICT at,
                                          size_t rsat) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t te = 0;
  __asm__ volatile("sf.vsettnt %[te], x0, e32, w1"
                   : [te] "=r"(te)
                   :
                   : "vtype", "vl");

  skl_pack_e32_e32rcpc_xsfmmbase(m, n, a, rsa, te, te, at, rsat, te, te * rsat,
                                 false, false, 0);

  __asm__ volatile("sf.vtdiscard");
}
