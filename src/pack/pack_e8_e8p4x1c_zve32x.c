// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_pack_e8_e8p4x1c_zve32x(size_t m, size_t n, const uint8_t *src,
                                         size_t rs, uint8_t *dst, size_t rs1,
                                         uint8_t pad) {
  size_t m_rem = m % 4;
  size_t m_idx = 0;

  for (; m_idx + 3 < m; m_idx += 4) {
    const uint8_t *src_read = src + m_idx * rs;
    uint8_t *dst_write = dst + (m_idx / 4) * rs1;
    for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
      vl = __riscv_vsetvl_e8m2(avl);
      vuint8m2_t vec0 = __riscv_vle8_v_u8m2(src_read, vl);
      vuint8m2_t vec1 = __riscv_vle8_v_u8m2(src_read + 1 * rs, vl);
      vuint8m2_t vec2 = __riscv_vle8_v_u8m2(src_read + 2 * rs, vl);
      vuint8m2_t vec3 = __riscv_vle8_v_u8m2(src_read + 3 * rs, vl);
      vuint8m2x4_t vec_group = __riscv_vcreate_v_u8m2x4(vec0, vec1, vec2, vec3);
      __riscv_vsseg4e8_v_u8m2x4(dst_write, vec_group, vl);
      dst_write += 4 * vl;
      src_read += vl;
    }
  }
  if (m_rem) {
    const uint8_t *src_read = src + (m_idx + m_rem - 1) * rs;
    uint8_t *dst_write = dst + (m_idx / 4) * rs1;
    vuint8m2_t vec0 = __riscv_vmv_v_x_u8m2(pad, __riscv_vsetvlmax_e8m2());
    vuint8m2_t vec1 = __riscv_vmv_v_x_u8m2(pad, __riscv_vsetvlmax_e8m2());
    vuint8m2_t vec2 = __riscv_vmv_v_x_u8m2(pad, __riscv_vsetvlmax_e8m2());
    vuint8m2_t vec3 = __riscv_vmv_v_x_u8m2(pad, __riscv_vsetvlmax_e8m2());
    for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
      vl = __riscv_vsetvl_e8m2(avl);
      const uint8_t *src_read_rev = src_read;
      switch (m_rem) {
      case 3:
        vec2 = __riscv_vle8_v_u8m2(src_read_rev, vl);
        src_read_rev -= rs;
        __attribute__((fallthrough));
      case 2:
        vec1 = __riscv_vle8_v_u8m2(src_read_rev, vl);
        src_read_rev -= rs;
        __attribute__((fallthrough));
      case 1:
        vec0 = __riscv_vle8_v_u8m2(src_read_rev, vl);
        __attribute__((fallthrough));
      default:
        break;
      }
      vuint8m2x4_t vec_group = __riscv_vcreate_v_u8m2x4(vec0, vec1, vec2, vec3);
      __riscv_vsseg4e8_v_u8m2x4(dst_write, vec_group, vl);
      dst_write += 4 * vl;
      src_read += vl;
    }
  }
}
