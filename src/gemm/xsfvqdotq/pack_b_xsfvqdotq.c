// Copyright (c) 2025-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_pack_b_i8_xsfvqdotq(size_t k, size_t n, const int8_t *b,
                                      size_t rsb, int8_t *b_pack, size_t rsb1) {
  size_t k_rem = k % 4;
  size_t k_idx = 0;

  for (; k_idx + 3 < k; k_idx += 4) {
    const int8_t *b_read = b + k_idx * rsb;
    int8_t *b_pack_write = b_pack + (k_idx / 4) * rsb1;
    for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
      vl = __riscv_vsetvl_e8m1(avl);
      vint8m1_t bvec0 = __riscv_vle8_v_i8m1(b_read, vl);
      vint8m1_t bvec1 = __riscv_vle8_v_i8m1(b_read + 1 * rsb, vl);
      vint8m1_t bvec2 = __riscv_vle8_v_i8m1(b_read + 2 * rsb, vl);
      vint8m1_t bvec3 = __riscv_vle8_v_i8m1(b_read + 3 * rsb, vl);
      vint8m1x4_t bvec_group =
          __riscv_vcreate_v_i8m1x4(bvec0, bvec1, bvec2, bvec3);
      __riscv_vsseg4e8_v_i8m1x4(b_pack_write, bvec_group, vl);
      b_pack_write += 4 * vl;
      b_read += vl;
    }
  }
  if (k_rem) {
    const int8_t *b_read = b + (k_idx + k_rem - 1) * rsb;
    int8_t *b_pack_write = b_pack + (k_idx / 4) * rsb1;
    vint8m1_t bvec0 = __riscv_vmv_v_x_i8m1(0, __riscv_vsetvlmax_e8m1());
    vint8m1_t bvec1 = __riscv_vmv_v_x_i8m1(0, __riscv_vsetvlmax_e8m1());
    vint8m1_t bvec2 = __riscv_vmv_v_x_i8m1(0, __riscv_vsetvlmax_e8m1());
    vint8m1_t bvec3 = __riscv_vmv_v_x_i8m1(0, __riscv_vsetvlmax_e8m1());
    for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
      vl = __riscv_vsetvl_e8m1(avl);
      const int8_t *b_read_rev = b_read;
      switch (k_rem) {
      case 3:
        bvec2 = __riscv_vle8_v_i8m1(b_read_rev, vl);
        b_read_rev -= rsb;
        __attribute__((fallthrough));
      case 2:
        bvec1 = __riscv_vle8_v_i8m1(b_read_rev, vl);
        b_read_rev -= rsb;
        __attribute__((fallthrough));
      case 1:
        bvec0 = __riscv_vle8_v_i8m1(b_read_rev, vl);
        __attribute__((fallthrough));
      default:
        break;
      }
      vint8m1x4_t bvec_group =
          __riscv_vcreate_v_i8m1x4(bvec0, bvec1, bvec2, bvec3);
      __riscv_vsseg4e8_v_i8m1x4(b_pack_write, bvec_group, vl);
      b_pack_write += 4 * vl;
      b_read += vl;
    }
  }
}
