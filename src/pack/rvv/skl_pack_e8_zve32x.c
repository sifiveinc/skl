// Copyright (c) 2026 SiFive, Inc. All rights reserved.
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

SKL_FUNC_PRIVATE void skl_set_e8_zve32x(uint8_t *dst, uint8_t value, size_t n) {
  size_t vl = 0;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m8(n - i);
    vuint8m8_t v_value = __riscv_vmv_v_x_u8m8(value, vl);
    __riscv_vse8_v_u8m8(dst + i, v_value, vl);
  }
}

SKL_FUNC_PRIVATE void skl_copy_e8_zve32x(uint8_t *dst, const uint8_t *src,
                                         size_t n) {
  size_t vl = 0;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m8(n - i);
    vuint8m8_t v_src = __riscv_vle8_v_u8m8(src + i, vl);
    __riscv_vse8_v_u8m8(dst + i, v_src, vl);
  }
}

SKL_FUNC_PRIVATE void skl_copy2d_e8_zve32x(size_t m, size_t n,
                                           const uint8_t *SKL_RESTRICT a,
                                           size_t rsa, uint8_t *SKL_RESTRICT b,
                                           size_t rsb) {
  if (rsa == n && rsb == n) {
    skl_copy_e8_zve32x(b, a, m * n);
    return;
  }
  for (size_t i = 0; i < m; ++i) {
    skl_copy_e8_zve32x(b + i * rsb, a + i * rsa, n);
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) with vectorization along the M dimension.
 */
SKL_FUNC_PRIVATE void
skl_transpose_mvec_e8_zve32x(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                             size_t rsa, uint8_t *SKL_RESTRICT at,
                             size_t rsat) {

  size_t vl = 0;
  size_t n_multiple_8 = n / 8 * 8;
  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint8_t));

  if (n_multiple_8) {
    for (size_t ii = 0; ii + 7 < n_multiple_8; ii += 8) {
      // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
      vuint8m1x8_t vcols = __riscv_vundefined_u8m1x8();
      // NOLINTEND(clang-analyzer-deadcode.DeadStores)

      const uint8_t *in_tile = a + ii;
      uint8_t *out_tile = at + ii * rsat;
      for (size_t jj = 0; jj < m; jj += vl) {
        vl = __riscv_vsetvl_e8m1(m - jj);

        vcols = __riscv_vlsseg8e8_v_u8m1x8(in_tile, input_seg_bstride, vl);

        uint8_t *write_ptr = out_tile;
        const size_t output_seg_stride = rsat;
        // store each segment continuously along m
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 0),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 1),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 2),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 3),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 4),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 5),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 6),
                            vl);
        write_ptr += output_seg_stride;
        __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 7),
                            vl);

        in_tile += vl * rsa;
        out_tile += vl;
      }
    }
  }

  const uint8_t *in_tile = a + n_multiple_8;
  uint8_t *out_tile = at + n_multiple_8 * rsat;

  switch (n % 8) {
  case 1: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m8_t vcol = __riscv_vundefined_u8m8();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m8(m - jj);

      vcol = __riscv_vlse8_v_u8m8(in_tile, input_seg_bstride, vl);

      // store extently along m
      __riscv_vse8_v_u8m8(out_tile, vcol, vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  case 2: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m4x2_t vcols = __riscv_vundefined_u8m4x2();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m4(m - jj);

      vcols = __riscv_vlsseg2e8_v_u8m4x2(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment extently along m
      __riscv_vse8_v_u8m4(write_ptr, __riscv_vget_v_u8m4x2_u8m4(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m4(write_ptr, __riscv_vget_v_u8m4x2_u8m4(vcols, 1), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  case 3: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m2x3_t vcols = __riscv_vundefined_u8m2x3();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m2(m - jj);

      vcols = __riscv_vlsseg3e8_v_u8m2x3(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment extently along m
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x3_u8m2(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x3_u8m2(vcols, 1), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x3_u8m2(vcols, 2), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  case 4: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m2x4_t vcols = __riscv_vundefined_u8m2x4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m2(m - jj);

      vcols = __riscv_vlsseg4e8_v_u8m2x4(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment extently along m
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 1), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 2), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 3), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  case 5: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1x5_t vcols = __riscv_vundefined_u8m1x5();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m1(m - jj);

      vcols = __riscv_vlsseg5e8_v_u8m1x5(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment extently along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 1), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 2), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 3), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 4), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  case 6: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1x6_t vcols = __riscv_vundefined_u8m1x6();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m1(m - jj);

      vcols = __riscv_vlsseg6e8_v_u8m1x6(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment extently along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 1), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 2), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 3), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 4), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 5), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  case 7: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1x7_t vcols = __riscv_vundefined_u8m1x7();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m1(m - jj);

      vcols = __riscv_vlsseg7e8_v_u8m1x7(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment extently along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 1), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 2), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 3), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 4), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 5), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 6), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  default:
    break;
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) with vectorization along the N dimension.
 */
SKL_FUNC_PRIVATE void
skl_transpose_nvec_e8_zve32x(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                             size_t rsa, uint8_t *SKL_RESTRICT at,
                             size_t rsat) {

  size_t vl = 0;
  size_t m_multiple_8 = m / 8 * 8;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint8_t));

  if (m_multiple_8) {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1_t vrow0 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow1 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow2 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow3 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow4 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow5 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow6 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow7 = __riscv_vundefined_u8m1();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t ii = 0; ii + 7 < m_multiple_8; ii += 8) {
      for (size_t jj = 0; jj < n; jj += vl) {
        const uint8_t *read_ptr = a + ii * rsa + jj;
        vl = __riscv_vsetvl_e8m1(n - jj);

        vrow0 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow1 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow2 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow3 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow4 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow5 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow6 = __riscv_vle8_v_u8m1(read_ptr, vl);
        read_ptr += rsa;
        vrow7 = __riscv_vle8_v_u8m1(read_ptr, vl);

        vuint8m1x8_t vrows = __riscv_vcreate_v_u8m1x8(
            vrow0, vrow1, vrow2, vrow3, vrow4, vrow5, vrow6, vrow7);

        __riscv_vssseg8e8_v_u8m1x8(at + (jj * rsat) + ii, output_seg_bstride,
                                   vrows, vl);
      }
    }
  }

  const uint8_t *in_tile = a + m_multiple_8 * rsa;
  uint8_t *out_tile = at + m_multiple_8;

  switch (m % 8) {
  case 1: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m8_t vrow = __riscv_vundefined_u8m8();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m8(n - jj);
      vrow = __riscv_vle8_v_u8m8(in_tile, vl);
      __riscv_vsse8_v_u8m8(out_tile, output_seg_bstride, vrow, vl);

      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  case 2: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m4_t vrow0 = __riscv_vundefined_u8m4();
    vuint8m4_t vrow1 = __riscv_vundefined_u8m4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m4(n - jj);

      vrow0 = __riscv_vle8_v_u8m4(in_tile, vl);
      vrow1 = __riscv_vle8_v_u8m4(in_tile + rsa, vl);
      vuint8m4x2_t vrows = __riscv_vcreate_v_u8m4x2(vrow0, vrow1);
      __riscv_vssseg2e8_v_u8m4x2(out_tile, output_seg_bstride, vrows, vl);
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  case 3: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m2_t vrow0 = __riscv_vundefined_u8m2();
    vuint8m2_t vrow1 = __riscv_vundefined_u8m2();
    vuint8m2_t vrow2 = __riscv_vundefined_u8m2();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m2(n - jj);

      vrow0 = __riscv_vle8_v_u8m2(in_tile, vl);
      vrow1 = __riscv_vle8_v_u8m2(in_tile + rsa, vl);
      vrow2 = __riscv_vle8_v_u8m2(in_tile + 2 * rsa, vl);
      vuint8m2x3_t vrows = __riscv_vcreate_v_u8m2x3(vrow0, vrow1, vrow2);
      __riscv_vssseg3e8_v_u8m2x3(out_tile, output_seg_bstride, vrows, vl);
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  case 4: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m2_t vrow0 = __riscv_vundefined_u8m2();
    vuint8m2_t vrow1 = __riscv_vundefined_u8m2();
    vuint8m2_t vrow2 = __riscv_vundefined_u8m2();
    vuint8m2_t vrow3 = __riscv_vundefined_u8m2();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m2(n - jj);

      vrow0 = __riscv_vle8_v_u8m2(in_tile, vl);
      vrow1 = __riscv_vle8_v_u8m2(in_tile + rsa, vl);
      vrow2 = __riscv_vle8_v_u8m2(in_tile + 2 * rsa, vl);
      vrow3 = __riscv_vle8_v_u8m2(in_tile + 3 * rsa, vl);
      vuint8m2x4_t vrows = __riscv_vcreate_v_u8m2x4(vrow0, vrow1, vrow2, vrow3);
      __riscv_vssseg4e8_v_u8m2x4(out_tile, output_seg_bstride, vrows, vl);
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  case 5: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1_t vrow0 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow1 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow2 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow3 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow4 = __riscv_vundefined_u8m1();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m1(n - jj);

      vrow0 = __riscv_vle8_v_u8m1(in_tile, vl);
      vrow1 = __riscv_vle8_v_u8m1(in_tile + rsa, vl);
      vrow2 = __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl);
      vrow3 = __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl);
      vrow4 = __riscv_vle8_v_u8m1(in_tile + 4 * rsa, vl);
      vuint8m1x5_t vrows =
          __riscv_vcreate_v_u8m1x5(vrow0, vrow1, vrow2, vrow3, vrow4);
      __riscv_vssseg5e8_v_u8m1x5(out_tile, output_seg_bstride, vrows, vl);
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  case 6: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1_t vrow0 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow1 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow2 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow3 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow4 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow5 = __riscv_vundefined_u8m1();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m1(n - jj);

      vrow0 = __riscv_vle8_v_u8m1(in_tile, vl);
      vrow1 = __riscv_vle8_v_u8m1(in_tile + rsa, vl);
      vrow2 = __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl);
      vrow3 = __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl);
      vrow4 = __riscv_vle8_v_u8m1(in_tile + 4 * rsa, vl);
      vrow5 = __riscv_vle8_v_u8m1(in_tile + 5 * rsa, vl);
      vuint8m1x6_t vrows =
          __riscv_vcreate_v_u8m1x6(vrow0, vrow1, vrow2, vrow3, vrow4, vrow5);
      __riscv_vssseg6e8_v_u8m1x6(out_tile, output_seg_bstride, vrows, vl);
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  case 7: {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1_t vrow0 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow1 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow2 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow3 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow4 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow5 = __riscv_vundefined_u8m1();
    vuint8m1_t vrow6 = __riscv_vundefined_u8m1();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m1(n - jj);

      vrow0 = __riscv_vle8_v_u8m1(in_tile, vl);
      vrow1 = __riscv_vle8_v_u8m1(in_tile + rsa, vl);
      vrow2 = __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl);
      vrow3 = __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl);
      vrow4 = __riscv_vle8_v_u8m1(in_tile + 4 * rsa, vl);
      vrow5 = __riscv_vle8_v_u8m1(in_tile + 5 * rsa, vl);
      vrow6 = __riscv_vle8_v_u8m1(in_tile + 6 * rsa, vl);
      vuint8m1x7_t vrows = __riscv_vcreate_v_u8m1x7(vrow0, vrow1, vrow2, vrow3,
                                                    vrow4, vrow5, vrow6);
      __riscv_vssseg7e8_v_u8m1x7(out_tile, output_seg_bstride, vrows, vl);
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  default:
    break;
  }
}

SKL_FUNC_PRIVATE bool skl_transpose_e8_is_mvec(size_t m, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e8m1();
  if (m > n || m >= vlmax) {
    return true;
  }
  return false;
}

SKL_FUNC_PRIVATE void
skl_transpose_e8_zve32x(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                        size_t rsa, uint8_t *SKL_RESTRICT at, size_t rsat) {
  if (skl_transpose_e8_is_mvec(m, n)) {
    skl_transpose_mvec_e8_zve32x(m, n, a, rsa, at, rsat);
  } else {
    skl_transpose_nvec_e8_zve32x(m, n, a, rsa, at, rsat);
  }
}

SKL_FUNC_PRIVATE void skl_pad_e8_zve32x(uint8_t *dst, size_t extent,
                                        size_t stride, uint8_t padding_value,
                                        size_t n_segs) {
  if (extent == 0 || n_segs == 0) {
    return;
  }
  if (extent == stride) {
    skl_set_e8_zve32x(dst, padding_value, extent * n_segs);
    return;
  }
  if (extent <= 8) {

    vuint8m1_t pad_vec0 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec1 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec2 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec3 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec4 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec5 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec6 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec7 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());

    switch (extent) {
    case 1: {
      vuint8m8_t pad_vec_group1 =
          __riscv_vcreate_v_u8m1_u8m8(pad_vec0, pad_vec1, pad_vec2, pad_vec3,
                                      pad_vec4, pad_vec5, pad_vec6, pad_vec7);
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m8(avl);
        __riscv_vsse8_v_u8m8(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                             pad_vec_group1, vl);
        dst += vl * stride;
      }
      break;
    }
    case 2: {
      vuint8m4x2_t pad_vec_group2 = __riscv_vcreate_v_u8m4x2(
          __riscv_vcreate_v_u8m1_u8m4(pad_vec0, pad_vec1, pad_vec2, pad_vec3),
          __riscv_vcreate_v_u8m1_u8m4(pad_vec4, pad_vec5, pad_vec6, pad_vec7));
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m4(avl);
        __riscv_vssseg2e8_v_u8m4x2(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group2, vl);
        dst += vl * stride;
      }
      break;
    }
    case 3: {
      vuint8m2x3_t pad_vec_group3 = __riscv_vcreate_v_u8m2x3(
          __riscv_vcreate_v_u8m1_u8m2(pad_vec0, pad_vec1),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec2, pad_vec3),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec4, pad_vec5));
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        __riscv_vssseg3e8_v_u8m2x3(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group3, vl);
        dst += vl * stride;
      }
      break;
    }
    case 4: {
      vuint8m2x4_t pad_vec_group4 = __riscv_vcreate_v_u8m2x4(
          __riscv_vcreate_v_u8m1_u8m2(pad_vec0, pad_vec1),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec2, pad_vec3),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec4, pad_vec5),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec6, pad_vec7));
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        __riscv_vssseg4e8_v_u8m2x4(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group4, vl);
        dst += vl * stride;
      }
      break;
    }
    case 5: {
      vuint8m1x5_t pad_vec_group5 = __riscv_vcreate_v_u8m1x5(
          pad_vec0, pad_vec1, pad_vec2, pad_vec3, pad_vec4);
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg5e8_v_u8m1x5(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group5, vl);
        dst += vl * stride;
      }
      break;
    }
    case 6: {
      vuint8m1x6_t pad_vec_group6 = __riscv_vcreate_v_u8m1x6(
          pad_vec0, pad_vec1, pad_vec2, pad_vec3, pad_vec4, pad_vec5);
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg6e8_v_u8m1x6(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group6, vl);
        dst += vl * stride;
      }
      break;
    }
    case 7: {
      vuint8m1x7_t pad_vec_group7 = __riscv_vcreate_v_u8m1x7(
          pad_vec0, pad_vec1, pad_vec2, pad_vec3, pad_vec4, pad_vec5, pad_vec6);
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg7e8_v_u8m1x7(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group7, vl);
        dst += vl * stride;
      }
      break;
    }
    case 8: {
      vuint8m1x8_t pad_vec_group8 =
          __riscv_vcreate_v_u8m1x8(pad_vec0, pad_vec1, pad_vec2, pad_vec3,
                                   pad_vec4, pad_vec5, pad_vec6, pad_vec7);
      for (size_t vl = 0, avl = n_segs; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg8e8_v_u8m1x8(dst, (ptrdiff_t)(stride * sizeof(uint8_t)),
                                   pad_vec_group8, vl);
        dst += vl * stride;
      }
      break;
    }
    default:
      break;
    }
  } else {
    for (size_t i = 0; i < n_segs; ++i) {
      skl_set_e8_zve32x(dst, padding_value, extent);
      dst += stride;
    }
  }
}

SKL_FUNC_PRIVATE void skl_transpose_padded_m_e8_zve32x(
    size_t m_padded, size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
    size_t rsa, uint8_t *SKL_RESTRICT at, size_t rsat, uint8_t padding_value) {

  if (m_padded == m) {
    skl_transpose_e8_zve32x(m, n, a, rsa, at, rsat);
    return;
  }

  if (m / 8) {
    skl_transpose_e8_zve32x(m / 8 * 8, n, a, rsa, at, rsat);
  }
  const uint8_t *in_tile = a + m / 8 * 8 * rsa;
  uint8_t *out_tile = at + m / 8 * 8;
  if (m % 8) {
    size_t m_transition = m_padded - m / 8 * 8 > 8 ? 8 : m_padded - m / 8 * 8;
    vuint8m1_t pad_vec0 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec1 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec2 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec3 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec4 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec5 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec6 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());
    vuint8m1_t pad_vec7 =
        __riscv_vmv_v_x_u8m1(padding_value, __riscv_vsetvlmax_e8m1());

    switch (m_transition) {
    case 2: {
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m4(avl);
        vuint8m4x2_t pad_vec_group2 = __riscv_vcreate_v_u8m4x2(
            __riscv_vle8_v_u8m4(in_tile, vl),
            __riscv_vcreate_v_u8m1_u8m4(pad_vec4, pad_vec5, pad_vec6,
                                        pad_vec7));
        __riscv_vssseg2e8_v_u8m4x2(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group2, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 3: {
      vuint8m2x3_t pad_vec_group3 = __riscv_vcreate_v_u8m2x3(
          __riscv_vcreate_v_u8m1_u8m2(pad_vec0, pad_vec1),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec2, pad_vec3),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec4, pad_vec5));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        switch (m % 8) {
        case 2:
          pad_vec_group3 = __riscv_vset_v_u8m2_u8m2x3(
              pad_vec_group3, 1, __riscv_vle8_v_u8m2(in_tile + 1 * rsa, vl));
          __attribute__((fallthrough));
        case 1:
          pad_vec_group3 = __riscv_vset_v_u8m2_u8m2x3(
              pad_vec_group3, 0, __riscv_vle8_v_u8m2(in_tile, vl));
          __attribute__((fallthrough));
        default:
          break;
        }
        __riscv_vssseg3e8_v_u8m2x3(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group3, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 4: {
      vuint8m2x4_t pad_vec_group4 = __riscv_vcreate_v_u8m2x4(
          __riscv_vcreate_v_u8m1_u8m2(pad_vec0, pad_vec1),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec2, pad_vec3),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec4, pad_vec5),
          __riscv_vcreate_v_u8m1_u8m2(pad_vec6, pad_vec7));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        switch (m % 8) {
        case 3:
          pad_vec_group4 = __riscv_vset_v_u8m2_u8m2x4(
              pad_vec_group4, 2, __riscv_vle8_v_u8m2(in_tile + 2 * rsa, vl));
          __attribute__((fallthrough));
        case 2:
          pad_vec_group4 = __riscv_vset_v_u8m2_u8m2x4(
              pad_vec_group4, 1, __riscv_vle8_v_u8m2(in_tile + 1 * rsa, vl));
          __attribute__((fallthrough));
        case 1:
          pad_vec_group4 = __riscv_vset_v_u8m2_u8m2x4(
              pad_vec_group4, 0, __riscv_vle8_v_u8m2(in_tile, vl));
          __attribute__((fallthrough));
        default:
          break;
        }
        __riscv_vssseg4e8_v_u8m2x4(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group4, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 5: {
      vuint8m1x5_t pad_vec_group5 = __riscv_vcreate_v_u8m1x5(
          pad_vec0, pad_vec1, pad_vec2, pad_vec3, pad_vec4);
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m % 8) {
        case 4:
          pad_vec_group5 = __riscv_vset_v_u8m1_u8m1x5(
              pad_vec_group5, 3, __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl));
          __attribute__((fallthrough));
        case 3:
          pad_vec_group5 = __riscv_vset_v_u8m1_u8m1x5(
              pad_vec_group5, 2, __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl));
          __attribute__((fallthrough));
        case 2:
          pad_vec_group5 = __riscv_vset_v_u8m1_u8m1x5(
              pad_vec_group5, 1, __riscv_vle8_v_u8m1(in_tile + 1 * rsa, vl));
          __attribute__((fallthrough));
        case 1:
          pad_vec_group5 = __riscv_vset_v_u8m1_u8m1x5(
              pad_vec_group5, 0, __riscv_vle8_v_u8m1(in_tile, vl));
          __attribute__((fallthrough));
        default:
          break;
        }
        __riscv_vssseg5e8_v_u8m1x5(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group5, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 6: {
      vuint8m1x6_t pad_vec_group6 = __riscv_vcreate_v_u8m1x6(
          pad_vec0, pad_vec1, pad_vec2, pad_vec3, pad_vec4, pad_vec5);
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m % 8) {
        case 5:
          pad_vec_group6 = __riscv_vset_v_u8m1_u8m1x6(
              pad_vec_group6, 4, __riscv_vle8_v_u8m1(in_tile + 4 * rsa, vl));
          __attribute__((fallthrough));
        case 4:
          pad_vec_group6 = __riscv_vset_v_u8m1_u8m1x6(
              pad_vec_group6, 3, __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl));
          __attribute__((fallthrough));
        case 3:
          pad_vec_group6 = __riscv_vset_v_u8m1_u8m1x6(
              pad_vec_group6, 2, __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl));
          __attribute__((fallthrough));
        case 2:
          pad_vec_group6 = __riscv_vset_v_u8m1_u8m1x6(
              pad_vec_group6, 1, __riscv_vle8_v_u8m1(in_tile + 1 * rsa, vl));
          __attribute__((fallthrough));
        case 1:
          pad_vec_group6 = __riscv_vset_v_u8m1_u8m1x6(
              pad_vec_group6, 0, __riscv_vle8_v_u8m1(in_tile, vl));
          __attribute__((fallthrough));
        default:
          break;
        }
        __riscv_vssseg6e8_v_u8m1x6(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group6, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 7: {
      vuint8m1x7_t pad_vec_group7 = __riscv_vcreate_v_u8m1x7(
          pad_vec0, pad_vec1, pad_vec2, pad_vec3, pad_vec4, pad_vec5, pad_vec6);
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m % 8) {
        case 6:
          pad_vec_group7 = __riscv_vset_v_u8m1_u8m1x7(
              pad_vec_group7, 5, __riscv_vle8_v_u8m1(in_tile + 5 * rsa, vl));
          __attribute__((fallthrough));
        case 5:
          pad_vec_group7 = __riscv_vset_v_u8m1_u8m1x7(
              pad_vec_group7, 4, __riscv_vle8_v_u8m1(in_tile + 4 * rsa, vl));
          __attribute__((fallthrough));
        case 4:
          pad_vec_group7 = __riscv_vset_v_u8m1_u8m1x7(
              pad_vec_group7, 3, __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl));
          __attribute__((fallthrough));
        case 3:
          pad_vec_group7 = __riscv_vset_v_u8m1_u8m1x7(
              pad_vec_group7, 2, __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl));
          __attribute__((fallthrough));
        case 2:
          pad_vec_group7 = __riscv_vset_v_u8m1_u8m1x7(
              pad_vec_group7, 1, __riscv_vle8_v_u8m1(in_tile + 1 * rsa, vl));
          __attribute__((fallthrough));
        case 1:
          pad_vec_group7 = __riscv_vset_v_u8m1_u8m1x7(
              pad_vec_group7, 0, __riscv_vle8_v_u8m1(in_tile, vl));
          __attribute__((fallthrough));
        default:
          break;
        }
        __riscv_vssseg7e8_v_u8m1x7(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group7, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 8: {
      vuint8m1x8_t pad_vec_group8 =
          __riscv_vcreate_v_u8m1x8(pad_vec0, pad_vec1, pad_vec2, pad_vec3,
                                   pad_vec4, pad_vec5, pad_vec6, pad_vec7);
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m % 8) {
        case 7:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 6, __riscv_vle8_v_u8m1(in_tile + 6 * rsa, vl));
          __attribute__((fallthrough));
        case 6:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 5, __riscv_vle8_v_u8m1(in_tile + 5 * rsa, vl));
          __attribute__((fallthrough));
        case 5:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 4, __riscv_vle8_v_u8m1(in_tile + 4 * rsa, vl));
          __attribute__((fallthrough));
        case 4:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 3, __riscv_vle8_v_u8m1(in_tile + 3 * rsa, vl));
          __attribute__((fallthrough));
        case 3:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 2, __riscv_vle8_v_u8m1(in_tile + 2 * rsa, vl));
          __attribute__((fallthrough));
        case 2:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 1, __riscv_vle8_v_u8m1(in_tile + 1 * rsa, vl));
          __attribute__((fallthrough));
        case 1:
          pad_vec_group8 = __riscv_vset_v_u8m1_u8m1x8(
              pad_vec_group8, 0, __riscv_vle8_v_u8m1(in_tile, vl));
          __attribute__((fallthrough));
        default:
          break;
        }
        __riscv_vssseg8e8_v_u8m1x8(
            out_tile, (ptrdiff_t)(rsat * sizeof(uint8_t)), pad_vec_group8, vl);
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    default:
      break;
    }
  } // End if (m % 8)
  if (m_padded > (m + 7) / 8 * 8) {
    out_tile = at + (m + 7) / 8 * 8;
    skl_pad_e8_zve32x(out_tile, m_padded - (m + 7) / 8 * 8, rsat, padding_value,
                      n);
  }
}

SKL_FUNC_PRIVATE void skl_pack_e8_e8rcprc_zve32x(
    size_t m,             // Num. rows in input matrix
    size_t n,             // Num. columns in input matrix
    const uint8_t *src,   // Input matrix
    size_t rs,            // Row stride of input matrix
    size_t m0,            // Num. rows in a block of the input matrix
    size_t n0,            // Num. columns in a block of the input matrix
    uint8_t *dst,         // Output packed matrix [m1 x n1]
    size_t rs0,           // Row stride within a block of the output matrix
    size_t cs0,           // Column stride within a block of the output matrix
    size_t rs1,           // Row stride between blocks of the output matrix
    size_t cs1,           // Column stride between blocks of the output matrix
    uint8_t padding_value // Value to use for padding
) {

  // Handle division by zero
  if (m0 == 0 || n0 == 0) {
    return;
  }

  size_t m1 = (m + m0 - 1) / m0; // Num. row blocks in the input matrix
  size_t n1 = (n + n0 - 1) / n0; // Num. column blocks in the input matrix

  if ((cs0 * n0 == cs1) && rs0 == 1) {
    for (size_t ii1 = 0; ii1 < m1; ++ii1) {
      const uint8_t *src_block = src + ii1 * m0 * rs;
      uint8_t *dst_block = dst + ii1 * rs1;
      size_t m_length = m0 < m - ii1 * m0 ? m0 : m - ii1 * m0;
      skl_transpose_padded_m_e8_zve32x(m0, m_length, n, src_block, rs,
                                       dst_block, cs0, padding_value);
      if (n % n0) {
        // pad right
        skl_pad_e8_zve32x(dst_block + cs1 * (n1 - 1) + cs0 * (n % n0), m0, cs0,
                          padding_value, n0 - n % n0);
      }
    }
    return;
  }

  if ((rs0 * m0 == rs1) && cs0 == 1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const uint8_t *src_block = src + jj1 * n0;
      uint8_t *dst_block = dst + jj1 * cs1;
      size_t n_length = n0 < n - jj1 * n0 ? n0 : n - jj1 * n0;
      skl_copy2d_e8_zve32x(m, n_length, src_block, rs, dst_block, rs0);
      if (m % m0) {
        // pad bottom
        skl_pad_e8_zve32x(dst_block + m * rs0, n0, rs0, padding_value,
                          m0 - m % m0);
      }
      // pad right: pad (n0 - n_length) elements in each of m rows
      skl_pad_e8_zve32x(dst_block + n_length, n0 - n_length, rs0, padding_value,
                        m);
    }
    return;
  }

  if (rs0 == 1 && n0 == 1 && m0 == rs1) {
    skl_transpose_padded_m_e8_zve32x(m0 * m1, m, n, src, rs, dst, cs1,
                                     padding_value);
    return;
  }

  if (cs0 == 1 && m0 == 1 && n0 == cs1) {
    skl_copy2d_e8_zve32x(m, n, src, rs, dst, rs1);
    // pad right
    if (n % n0) {
      skl_pad_e8_zve32x(dst + n, n0 - n % n0, rs1, padding_value, m);
    }
    return;
  }

  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const uint8_t *src_block = src + ii1 * m0 * rs + jj1 * n0;
      uint8_t *dst_block = dst + ii1 * rs1 + jj1 * cs1;

      if (rs0 == 1) {
        size_t m_length = m0 < m - ii1 * m0 ? m0 : m - ii1 * m0;
        size_t n_length = n0 < n - jj1 * n0 ? n0 : n - jj1 * n0;
        skl_transpose_padded_m_e8_zve32x(m0, m_length, n_length, src_block, rs,
                                         dst_block, cs0, padding_value);
        // pad right
        skl_pad_e8_zve32x(dst_block + cs0 * n_length, m0, cs0, padding_value,
                          n0 - n_length);
      } else if (cs0 == 1) {
        size_t m_length = m0 < m - ii1 * m0 ? m0 : m - ii1 * m0;
        size_t n_length = n0 < n - jj1 * n0 ? n0 : n - jj1 * n0;
        skl_copy2d_e8_zve32x(m_length, n_length, src_block, rs, dst_block, rs0);
        // pad right: pad (n0 - n_length) elements in each of m_length rows
        skl_pad_e8_zve32x(dst_block + n_length, n0 - n_length, rs0,
                          padding_value, m_length);
        // pad bottom: pad (m0 - m_length) rows, each with n0 elements
        skl_pad_e8_zve32x(dst_block + m_length * rs0, n0, rs0, padding_value,
                          m0 - m_length);
      } else {
        for (size_t ii0 = 0; ii0 < m0; ++ii0) {
          for (size_t jj0 = 0; jj0 < n0; ++jj0) {
            if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
              dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0];
            } else {
              // Pad with zeros
              dst_block[ii0 * rs0 + jj0 * cs0] = padding_value;
            }
          }
        }
      }
    }
  }
}

SKL_FUNC void skl_pack_e8rc_e8rcprc_zve32x(
    size_t m,             // Num. rows in input matrix
    size_t n,             // Num. columns in input matrix
    const uint8_t *src,   // Input matrix
    size_t rs,            // Row stride of input matrix
    size_t cs,            // Column stride of input matrix
    size_t m0,            // Num. rows in a block of the input matrix
    size_t n0,            // Num. columns in a block of the input matrix
    uint8_t *dst,         // Output packed matrix [m1 x n1]
    size_t rs0,           // Row stride within a block of the output matrix
    size_t cs0,           // Column stride within a block of the output matrix
    size_t rs1,           // Row stride between blocks of the output matrix
    size_t cs1,           // Column stride between blocks of the output matrix
    uint8_t padding_value // Value to use for padding
) {

  if (cs == 1) {
    skl_pack_e8_e8rcprc_zve32x(m, n, src, rs, m0, n0, dst, rs0, cs0, rs1, cs1,
                               padding_value);
  } else if (rs == 1) {
    // When input is column-major (rs==1), transpose both dimensions and strides
    skl_pack_e8_e8rcprc_zve32x(n, m, src, cs, n0, m0, dst, cs0, rs0, cs1, rs1,
                               padding_value);
  } else {
    size_t m1 = (m + m0 - 1) / m0; // Num. row blocks in the input matrix
    size_t n1 = (n + n0 - 1) / n0; // Num. column blocks in the input matrix
    for (size_t ii1 = 0; ii1 < m1; ++ii1) {
      for (size_t jj1 = 0; jj1 < n1; ++jj1) {
        const uint8_t *src_block = src + ii1 * m0 * rs + jj1 * n0 * cs;
        uint8_t *dst_block = dst + ii1 * rs1 + jj1 * cs1;
        for (size_t ii0 = 0; ii0 < m0; ++ii0) {
          for (size_t jj0 = 0; jj0 < n0; ++jj0) {
            if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
              dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0 * cs];
            } else {
              // Pad with zeros
              dst_block[ii0 * rs0 + jj0 * cs0] = padding_value;
            }
          }
        }
      }
    }
  }
}
