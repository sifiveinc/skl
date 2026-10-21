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

/* Load a tm x tn row-major matrix into the tile specified by tss.
 * tm and tn must be <= ETE.
 *
 * Tiles may have padding, so marked as INOUT.
 */
SKL_FUNC_PRIVATE void
skl_pack_tile_load_e32_e32_xsfmmbase(size_t tm, size_t tn, const uint32_t *src,
                                     size_t rs, size_t tss)
    __riscv_inout("xsfmm") {
  if (tm == 0 || tn == 0) {
    return;
  }

  const size_t kRowInc = 1;
  const uint32_t *src_end = src + tm * rs;

  __asm__ volatile("sf.vsettnt x0, %[tn], e32, w1\n"

                   "0:\n"
                   "sf.vlte32 %[tss], (%[src])\n"
                   "add %[tss], %[tss], %[kRowInc]\n"
                   "add %[src], %[src], %[rs]\n"
                   "bltu %[src], %[src_end], 0b\n"
                   : [tss] "+&r"(tss), [src] "+&r"(src)
                   : [src_end] "r"(src_end), [kRowInc] "rI"(kRowInc),
                     [rs] "r"(rs * sizeof(uint32_t)), [tm] "r"(tm), [tn] "r"(tn)
                   : "vtype", "vl", "memory");
}

/* Store tm rows (columns) of length n0 from the tile specified by tss into the
 * rows of an m0 x n0 row-major matrix.
 *  - If tss has a column pattern, columns are stored.
 *  - m0 and n0 must be <= ETE.
 *  - tm must be <= m0. If tm < m0, the bottom rows of the output are padded.
 */
SKL_FUNC_PRIVATE void skl_pack_tile_store_pad_bottom_e32_e32_xsfmmbase(
    size_t tm, size_t tss, size_t m0, size_t n0,
    uint32_t *dst, // NOLINT(readability-non-const-parameter)
    size_t rs, uint32_t pad) __riscv_in("xsfmm") {
  if (m0 == 0 || n0 == 0) {
    return;
  }

  vuint32m8_t pad_vec = __riscv_vmv_v_x_u32m8(pad, n0);
  const size_t kRowInc = 1;
  const uint32_t *dst_store_end = dst + tm * rs;
  const uint32_t *dst_pad_end = dst + m0 * rs;

  __asm__ volatile("sf.vsettnt x0, %[n0], e32, w1\n"

                   "bgeu %[dst], %[dst_store_end], 1f\n"
                   "0:\n"
                   "sf.vste32 %[tss], (%[dst])\n"
                   "add %[tss], %[tss], %[kRowInc]\n"
                   "add %[dst], %[dst], %[rs]\n"
                   "bltu %[dst], %[dst_store_end], 0b\n"

                   "1:\n"
                   "bgeu %[dst], %[dst_pad_end], 2f\n"
                   "0:\n"
                   "vse32.v %[pad_vec], (%[dst])\n"
                   "add %[dst], %[dst], %[rs]\n"
                   "bltu %[dst], %[dst_pad_end], 0b\n"

                   "2:\n"
                   : [tss] "+&r"(tss), [dst] "+&r"(dst)
                   : [dst_store_end] "r"(dst_store_end),
                     [dst_pad_end] "r"(dst_pad_end), [pad_vec] "vr"(pad_vec),
                     [kRowInc] "rI"(kRowInc), [rs] "r"(rs * sizeof(uint32_t)),
                     [m0] "r"(m0), [n0] "r"(n0), [tm] "r"(tm)
                   : "vtype", "vl", "memory");
}

/*
 * Equivalent to:
 * skl_pack_tile_store_pad_bottom_e32_e32_xsfmmbase(tm_store, tss_store, m0,
                                                    n0, dst, rsdst, pad);
 * skl_pack_tile_load_e32_e32_xsfmmbase(tm_load, tn_load, src, rssrc,
                                        tss_load);
 *
 * This kernel overlaps store and load execution for improved performance over
 * storing tss_store first and then loading src.
 */
SKL_FUNC_PRIVATE void skl_pack_tile_store_load_e32_e32_xsfmmbase(
    size_t tm_store, size_t tss_store, size_t m0, size_t n0,
    uint32_t *dst, // NOLINT(readability-non-const-parameter)
    size_t rsdst, size_t tm_load, size_t tn_load, const uint32_t *src,
    size_t rssrc, size_t tss_load, uint32_t pad) __riscv_inout("xsfmm") {
  if ((m0 == 0 || n0 == 0) && (tm_load == 0 || tn_load == 0)) {
    return;
  }

  vuint32m8_t pad_vec = __riscv_vmv_v_x_u32m8(pad, n0);

  const size_t kRowInc = 1;
  size_t k0 = tm_store <= tm_load ? tm_store : tm_load;
  size_t k1 = m0 <= tm_load ? m0 : tm_load;
  const uint32_t *src_end = src + tm_load * rssrc;
  const uint32_t *dst_store_load_end = dst + k0 * rsdst;
  const uint32_t *dst_pad_load_end = dst + k1 * rsdst;
  const uint32_t *dst_store_end = dst + tm_store * rsdst;
  const uint32_t *dst_pad_end = dst + m0 * rsdst;
  __asm__ volatile(
      "sf.vsettnt x0, x0, e32, w1\n"

      "bgeu %[dst], %[dst_store_load_end], 2f\n"
      "beq %[tn_load], %[n0], 1f\n"
      "0:\n"
      "sf.vsettn x0, %[tn_load]\n"
      "sf.vlte32 %[tss_load], (%[src])\n"
      "add %[tss_load], %[tss_load], %[kRowInc]\n"
      "add %[src], %[src], %[rssrc]\n"
      "sf.vsettn x0, %[n0]\n"
      "sf.vste32 %[tss_store], (%[dst])\n"
      "add %[tss_store], %[tss_store], %[kRowInc]\n"
      "add %[dst], %[dst], %[rsdst]\n"
      "bltu %[dst], %[dst_store_load_end], 0b\n"
      "j 2f\n"

      "1:\n"
      "sf.vsettn x0, %[tn_load]\n"
      "0:\n"
      "sf.vlte32 %[tss_load], (%[src])\n"
      "add %[tss_load], %[tss_load], %[kRowInc]\n"
      "add %[src], %[src], %[rssrc]\n"
      "sf.vste32 %[tss_store], (%[dst])\n"
      "add %[tss_store], %[tss_store], %[kRowInc]\n"
      "add %[dst], %[dst], %[rsdst]\n"
      "bltu %[dst], %[dst_store_load_end], 0b\n"

      "2:\n"
      "bgeu %[dst], %[dst_pad_load_end], 3f\n"
      "0:\n"
      "sf.vsettn x0, %[tn_load]\n"
      "sf.vlte32 %[tss_load], (%[src])\n"
      "add %[tss_load], %[tss_load], %[kRowInc]\n"
      "add %[src], %[src], %[rssrc]\n"
      "sf.vsettn x0, %[n0]\n"
      "vse32.v %[pad_vec], (%[dst])\n"
      "add %[dst], %[dst], %[rsdst]\n"
      "bltu %[dst], %[dst_pad_load_end], 0b\n"

      "3:\n"
      "bgeu %[dst], %[dst_store_end], 4f\n"
      "sf.vsettn x0, %[n0]\n"
      "0:\n"
      "sf.vste32 %[tss_store], (%[dst])\n"
      "add %[tss_store], %[tss_store], %[kRowInc]\n"
      "add %[dst], %[dst], %[rsdst]\n"
      "bltu %[dst], %[dst_store_end], 0b\n"

      "4:\n"
      "bgeu %[dst], %[dst_pad_end], 5f\n"
      "sf.vsettn x0, %[n0]\n"
      "0:\n"
      "vse32.v %[pad_vec], (%[dst])\n"
      "add %[dst], %[dst], %[rsdst]\n"
      "bltu %[dst], %[dst_pad_end], 0b\n"

      "5:\n"
      "bgeu %[src], %[src_end], 6f\n"
      "sf.vsettn x0, %[tn_load]\n"
      "0:\n"
      "sf.vlte32 %[tss_load], (%[src])\n"
      "add %[tss_load], %[tss_load], %[kRowInc]\n"
      "add %[src], %[src], %[rssrc]\n"
      "bltu %[src], %[src_end], 0b\n"

      "6:\n"
      : [tss_store] "+&r"(tss_store), [tss_load] "+&r"(tss_load),
        [dst] "+&r"(dst), [src] "+&r"(src)
      : [dst_store_load_end] "r"(dst_store_load_end),
        [dst_pad_load_end] "r"(dst_pad_load_end),
        [dst_store_end] "r"(dst_store_end), [dst_pad_end] "r"(dst_pad_end),
        [src_end] "r"(src_end), [pad_vec] "vr"(pad_vec),
        [kRowInc] "rI"(kRowInc), [rsdst] "r"(rsdst * sizeof(uint32_t)),
        [rssrc] "r"(rssrc * sizeof(uint32_t)), [k0] "r"(k0), [k1] "r"(k1),
        [m0] "r"(m0), [n0] "r"(n0), [tm_store] "r"(tm_store),
        [tm_load] "r"(tm_load), [tn_load] "r"(tn_load)
      : "vtype", "vl", "memory");
}

/* Set the entries of the tm x tn tile specified by tss to pad.
 * Tiles may have loaded data, so marked as INOUT.
 */
SKL_FUNC_PRIVATE void skl_pack_tile_set_e32_xsfmmbase(size_t tm, size_t tn,
                                                      size_t tss, uint32_t pad)
    __riscv_inout("xsfmm") {
  if (tm == 0 || tn == 0) {
    return;
  }

  vuint32m8_t pad_vec = __riscv_vmv_v_x_u32m8(pad, tn);

  const size_t kRowInc = 1;
  size_t tss_end = tss + kRowInc * tm;
  __asm__ volatile("bgeu %[tss], %[tss_end], 1f\n"
                   "sf.vsettnt x0, %[tn], e32, w1\n"
                   "0:\n"
                   "sf.vtmv.t.v %[tss], %[pad_vec]\n"
                   "add %[tss], %[tss], %[kRowInc]\n"
                   "bltu %[tss], %[tss_end], 0b\n"
                   "1:\n"
                   : [tss] "+&r"(tss)
                   : [pad_vec] "vr"(pad_vec), [tn] "r"(tn),
                     [tss_end] "r"(tss_end), [kRowInc] "rI"(kRowInc)
                   : "vtype", "vl");
}

/* Pack src into ETE x ETE column-major blocks. pad_right and pad_bottom
 * determine whether right and bottom padding are written.
 */
SKL_FUNC_PRIVATE void skl_pack_padding_optional_e32_e32rcptextec_xsfmmbase(
    size_t m, size_t n, const uint32_t *src, size_t rs, uint32_t *dst,
    size_t cs0, size_t rs1, size_t cs1, bool pad_right, bool pad_bottom,
    uint32_t pad) __riscv_new("xsfmm") {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %[ete], x0, e32, w1"
                   : [ete] "=r"(ete)
                   :
                   : "vtype", "vl");
  const size_t m0 = ete;
  const size_t n0 = ete;

  const size_t kRowInc = 1;
  const size_t kShiftTile = 27;
  const size_t kShiftPattern = 24;
  const size_t mt0 = 0;
  const size_t mt0c = mt0 | (size_t)1 << kShiftPattern;
  const size_t mt4 = (size_t)4 << kShiftTile;
  const size_t mt4c = mt4 | (size_t)1 << kShiftPattern;

  const size_t m1 = (m + m0 - 1) / m0;
  const size_t n1 = (n + n0 - 1) / n0;

  size_t tm_bottom = (m - 1) % m0 + 1;
  size_t tn_right = (n - 1) % n0 + 1;
  size_t m0_bottom = pad_bottom ? m0 : tm_bottom;
  size_t n0_right = pad_right ? n0 : tn_right;

  size_t i1 = 0;
  size_t j1 = 0;

  size_t tm_load = m1 == 1 ? tm_bottom : m0;
  size_t tn_load = n1 == 1 ? tn_right : n0;
  skl_pack_tile_load_e32_e32_xsfmmbase(tm_load, tn_load, src, rs, mt0);

  // NOLINTBEGIN(readability-suspicious-call-argument)
  for (; i1 + 1 < m1; i1 += 2) {
    size_t m0_store_bottom = m0;
    size_t tm_load_bottom = m0;
    if (i1 == m1 - 2) {
      m0_store_bottom = m0_bottom;
      tm_load_bottom = tm_bottom;
      skl_pack_tile_set_e32_xsfmmbase(m0_bottom - tm_bottom, n0,
                                      mt4 + tm_bottom * kRowInc, pad);
    }
    for (j1 = 0; j1 + 1 < n1; ++j1) {
      size_t tn_load_right = j1 == n1 - 2 ? tn_right : n0;
      skl_pack_tile_store_load_e32_e32_xsfmmbase(
          n0, mt0c, n0, m0, dst + i1 * rs1 + j1 * cs1, cs0, tm_load_bottom, n0,
          src + (i1 + 1) * m0 * rs + j1 * n0, rs, mt4, pad);
      skl_pack_tile_store_load_e32_e32_xsfmmbase(
          n0, mt4c, n0, m0_store_bottom, dst + (i1 + 1) * rs1 + j1 * cs1, cs0,
          m0, tn_load_right, src + i1 * m0 * rs + (j1 + 1) * n0, rs, mt0, pad);
    }
    skl_pack_tile_store_load_e32_e32_xsfmmbase(
        tn_right, mt0c, n0_right, m0, dst + i1 * rs1 + j1 * cs1, cs0,
        tm_load_bottom, tn_right, src + (i1 + 1) * m0 * rs + j1 * n0, rs, mt4,
        pad);
    if (i1 < m1 - 2) {
      tm_load = i1 == m1 - 3 ? tm_bottom : m0;
      tn_load = n1 == 1 ? tn_right : n0;
      skl_pack_tile_store_load_e32_e32_xsfmmbase(
          tn_right, mt4c, n0_right, m0, dst + (i1 + 1) * rs1 + j1 * cs1, cs0,
          tm_load, tn_load, src + (i1 + 2) * m0 * rs + 0 * n0, rs, mt0, pad);
    } else {
      skl_pack_tile_store_pad_bottom_e32_e32_xsfmmbase(
          tn_right, mt4c, n0_right, m0_bottom, dst + (i1 + 1) * rs1 + j1 * cs1,
          cs0, pad);
      return;
    }
  }

  // m1 % 2 == 1
  skl_pack_tile_set_e32_xsfmmbase(m0_bottom - tm_bottom, n0,
                                  mt0 + tm_bottom * kRowInc, pad);
  skl_pack_tile_set_e32_xsfmmbase(m0_bottom - tm_bottom, n0,
                                  mt4 + tm_bottom * kRowInc, pad);
  for (j1 = 0; j1 + 2 < n1; j1 += 2) {
    size_t tn_load_right = j1 == n1 - 3 ? tn_right : n0;
    skl_pack_tile_store_load_e32_e32_xsfmmbase(
        n0, mt0c, n0, m0_bottom, dst + i1 * rs1 + j1 * cs1, cs0, tm_bottom, n0,
        src + i1 * m0 * rs + (j1 + 1) * n0, rs, mt4, pad);
    skl_pack_tile_store_load_e32_e32_xsfmmbase(
        n0, mt4c, n0, m0_bottom, dst + i1 * rs1 + (j1 + 1) * cs1, cs0,
        tm_bottom, tn_load_right, src + i1 * m0 * rs + (j1 + 2) * n0, rs, mt0,
        pad);
  }
  if (n1 % 2) {
    skl_pack_tile_store_pad_bottom_e32_e32_xsfmmbase(
        tn_right, mt0c, n0_right, m0_bottom, dst + i1 * rs1 + j1 * cs1, cs0,
        pad);
  } else {
    skl_pack_tile_store_load_e32_e32_xsfmmbase(
        n0, mt0c, n0, m0_bottom, dst + i1 * rs1 + j1 * cs1, cs0, tm_bottom,
        tn_right, src + i1 * m0 * rs + (j1 + 1) * n0, rs, mt4, pad);
    skl_pack_tile_store_pad_bottom_e32_e32_xsfmmbase(
        tn_right, mt4c, n0_right, m0_bottom, dst + i1 * rs1 + (j1 + 1) * cs1,
        cs0, pad);
  }
  // NOLINTEND(readability-suspicious-call-argument)

  __asm__ volatile("sf.vtdiscard");
}

SKL_FUNC void skl_pack_e32_e32rcptex1c_xsfmmbase(size_t m, size_t n,
                                                 const uint32_t *src, size_t rs,
                                                 uint32_t *dst, size_t rs1,
                                                 size_t cs1, uint32_t pad) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %[ete], x0, e32, w1"
                   : [ete] "=r"(ete)
                   :
                   : "vtype", "vl");

  skl_pack_padding_optional_e32_e32rcptextec_xsfmmbase(
      m, n, src, rs, dst, cs1, rs1, ete * cs1, false, true, pad);
}

SKL_FUNC void skl_transpose_e32_xsfmmbase(size_t m, size_t n,
                                          const uint32_t *SKL_RESTRICT a,
                                          size_t rsa, uint32_t *SKL_RESTRICT at,
                                          size_t rsat) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t ete = 0;
  __asm__ volatile("sf.vsettnt %[ete], x0, e32, w1"
                   : [ete] "=r"(ete)
                   :
                   : "vtype", "vl");

  skl_pack_padding_optional_e32_e32rcptextec_xsfmmbase(
      m, n, a, rsa, at, rsat, ete, ete * rsat, false, false, 0);
}
