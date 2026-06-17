// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC_PRIVATE void skl_set_e8_zve32x(uint8_t *dst, uint8_t value, size_t n) {
  size_t vl = __riscv_vsetvl_e8m8(n);
  vuint8m8_t v_value = __riscv_vmv_v_x_u8m8(value, vl);

  while (n) {
    vl = __riscv_vsetvl_e8m8(n);
    __riscv_vse8_v_u8m8(dst, v_value, vl);
    n -= vl;
    dst += vl;
  }
}

SKL_FUNC_PRIVATE void skl_copy_e8_zve32x(uint8_t *SKL_RESTRICT dst,
                                         const uint8_t *SKL_RESTRICT src,
                                         size_t n) {
  size_t vl = 0;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8m8(n - i);
    vuint8m8_t v_src = __riscv_vle8_v_u8m8(src + i, vl);
    __riscv_vse8_v_u8m8(dst + i, v_src, vl);
  }
}

/**
 * @brief Set a 2D region of elements to a constant value.
 *
 * @param dst - Pointer to the start of the 2D region.
 * @param rs - Row stride of the 2D region in elements.
 * @param value - Value to set each element to.
 * @param m - Number of rows in the 2D region.
 * @param n - Number of columns in the 2D region.
 *
 * This function sets each element in the 2D region to the specified value.
 */
SKL_FUNC_PRIVATE void skl_set_2d_e8_zve32x(uint8_t *dst, size_t rs,
                                           uint8_t value, size_t m, size_t n) {
  if (n == 0 || m == 0) {
    return;
  }
  if (n == rs) {
    skl_set_e8_zve32x(dst, value, n * m);
    return;
  }
  if (n <= 8) {

    vuint8m8_t pad_vec = __riscv_vmv_v_x_u8m8(value, __riscv_vsetvlmax_e8m8());

    switch (n) {
    case 1: {
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m8(avl);
        __riscv_vsse8_v_u8m8(dst, (ptrdiff_t)(rs * sizeof(uint8_t)), pad_vec,
                             vl);
        dst += vl * rs;
      }
      break;
    }
    case 2: {
      vuint8m4x2_t pad_vec_group2 =
          __riscv_vcreate_v_u8m4x2(__riscv_vget_v_u8m8_u8m4(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m4(pad_vec, 1));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m4(avl);
        __riscv_vssseg2e8_v_u8m4x2(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group2, vl);
        dst += vl * rs;
      }
      break;
    }
    case 3: {
      vuint8m2x3_t pad_vec_group3 =
          __riscv_vcreate_v_u8m2x3(__riscv_vget_v_u8m8_u8m2(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 2));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        __riscv_vssseg3e8_v_u8m2x3(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group3, vl);
        dst += vl * rs;
      }
      break;
    }
    case 4: {
      vuint8m2x4_t pad_vec_group4 =
          __riscv_vcreate_v_u8m2x4(__riscv_vget_v_u8m8_u8m2(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 3));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        __riscv_vssseg4e8_v_u8m2x4(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group4, vl);
        dst += vl * rs;
      }
      break;
    }
    case 5: {
      vuint8m1x5_t pad_vec_group5 =
          __riscv_vcreate_v_u8m1x5(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg5e8_v_u8m1x5(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group5, vl);
        dst += vl * rs;
      }
      break;
    }
    case 6: {
      vuint8m1x6_t pad_vec_group6 =
          __riscv_vcreate_v_u8m1x6(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 5));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg6e8_v_u8m1x6(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group6, vl);
        dst += vl * rs;
      }
      break;
    }
    case 7: {
      vuint8m1x7_t pad_vec_group7 =
          __riscv_vcreate_v_u8m1x7(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 5),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 6));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg7e8_v_u8m1x7(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group7, vl);
        dst += vl * rs;
      }
      break;
    }
    case 8: {
      vuint8m1x8_t pad_vec_group8 =
          __riscv_vcreate_v_u8m1x8(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 5),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 6),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 7));
      for (size_t vl = 0, avl = m; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg8e8_v_u8m1x8(dst, (ptrdiff_t)(rs * sizeof(uint8_t)),
                                   pad_vec_group8, vl);
        dst += vl * rs;
      }
      break;
    }
    default:
      break;
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      skl_set_e8_zve32x(dst, value, n);
      dst += rs;
    }
  }
}

SKL_FUNC_PRIVATE void skl_copy_2d_e8_zve32x(size_t m, size_t n,
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

/**
 * @brief Copy a 2D matrix and pad the output matrix with a constant value in
 * the N dimension.
 *
 * @param n_padded - Number of columns in output matrix
 * @param m - Number of rows in input matrix and output matrix
 * @param n - Number of columns in input matrix
 * @param a - Pointer to input matrix
 * @param rsa - Row stride of input matrix in elements.
 * @param b - Pointer to output matrix.
 * @param rsb - Row stride of output matrix in elements.
 * @param pad - Value to use for padding.
 *
 */
SKL_FUNC_PRIVATE void skl_copy_2d_padded_n_e8_zve32x(
    size_t n_padded, size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
    size_t rsa, uint8_t *SKL_RESTRICT b, size_t rsb, uint8_t pad) {
  if (n_padded == n) {
    skl_copy_2d_e8_zve32x(m, n, a, rsa, b, rsb);
    return;
  }
  /* Process N dimension in 3 parts:
   * - head: Full vlmax-element chunks (direct copy)
   * - transition: Partial vlmax chunk with data + padding
   * - tail: Pure padding beyond transition
   */
  size_t vlmax = __riscv_vsetvlmax_e8m8();
  size_t n_head = n / vlmax * vlmax;
  if (n_head) {
    skl_copy_2d_e8_zve32x(m, n_head, a, rsa, b, rsb);
    a += n_head;
    b += n_head;
  }

  size_t n_transition = 0;
  if (n % vlmax) {
    n_transition = n_padded - n_head > vlmax ? vlmax : n_padded - n_head;
    vuint8m8_t pad_vec = __riscv_vmv_v_x_u8m8(pad, n_transition);
    size_t n_load = n - n_head;
    for (size_t i = 0; i < m; ++i) {
      pad_vec = __riscv_vle8_v_u8m8_tu(pad_vec, a + i * rsa, n_load);
      __riscv_vse8_v_u8m8(b + i * rsb, pad_vec, n_transition);
    }
    b += n_transition;
  }

  size_t n_tail = n_padded - n_head - n_transition;
  if (n_tail) {
    skl_set_2d_e8_zve32x(b, rsb, pad, m, n_tail);
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
  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint8_t));

  const uint8_t *in_tile = a;
  uint8_t *out_tile = at;
  size_t n_mod8 = n % 8;

  switch (n_mod8) {
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

      if (rsa == 2) {
        vcols = __riscv_vlseg2e8_v_u8m4x2(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg2e8_v_u8m4x2(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment extently along m
      __riscv_vse8_v_u8m4(write_ptr, __riscv_vget_v_u8m4x2_u8m4(vcols, 0), vl);
      write_ptr += rsat;
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

      if (rsa == 3) {
        vcols = __riscv_vlseg3e8_v_u8m2x3(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg3e8_v_u8m2x3(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment extently along m
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x3_u8m2(vcols, 0), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x3_u8m2(vcols, 1), vl);
      write_ptr += rsat;
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

      if (rsa == 4) {
        vcols = __riscv_vlseg4e8_v_u8m2x4(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg4e8_v_u8m2x4(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment extently along m
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 0), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 1), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m2(write_ptr, __riscv_vget_v_u8m2x4_u8m2(vcols, 2), vl);
      write_ptr += rsat;
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

      if (rsa == 5) {
        vcols = __riscv_vlseg5e8_v_u8m1x5(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg5e8_v_u8m1x5(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment extently along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 0), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 1), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 2), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x5_u8m1(vcols, 3), vl);
      write_ptr += rsat;
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

      if (rsa == 6) {
        vcols = __riscv_vlseg6e8_v_u8m1x6(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg6e8_v_u8m1x6(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment extently along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 0), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 1), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 2), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 3), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x6_u8m1(vcols, 4), vl);
      write_ptr += rsat;
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

      if (rsa == 7) {
        vcols = __riscv_vlseg7e8_v_u8m1x7(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg7e8_v_u8m1x7(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment extently along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 0), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 1), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 2), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 3), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 4), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 5), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x7_u8m1(vcols, 6), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
    break;
  }
  default:
    break;
  }

  n -= n_mod8;
  a += n_mod8;
  at += n_mod8 * rsat;

  for (size_t ii = 0; ii < n; ii += 8) {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m1x8_t vcols = __riscv_vundefined_u8m1x8();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    const uint8_t *in_tile = a + ii;
    uint8_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m1(m - jj);

      if (rsa == 8) {
        vcols = __riscv_vlseg8e8_v_u8m1x8(in_tile, vl);
      } else {
        vcols = __riscv_vlsseg8e8_v_u8m1x8(in_tile, input_seg_bstride, vl);
      }

      uint8_t *write_ptr = out_tile;
      // store each segment continuously along m
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 0), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 1), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 2), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 3), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 4), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 5), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 6), vl);
      write_ptr += rsat;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 7), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
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
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint8_t));

  size_t m_mod8 = m % 8;
  const uint8_t *in_tile = a;
  uint8_t *out_tile = at;

  switch (m_mod8) {
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
      const uint8_t *read_ptr = in_tile;
      vrow0 = __riscv_vle8_v_u8m4(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle8_v_u8m4(read_ptr, vl);
      vuint8m4x2_t vrows = __riscv_vcreate_v_u8m4x2(vrow0, vrow1);
      if (rsat == 2) {
        __riscv_vsseg2e8_v_u8m4x2(out_tile, vrows, vl);
      } else {
        __riscv_vssseg2e8_v_u8m4x2(out_tile, output_seg_bstride, vrows, vl);
      }
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
      const uint8_t *read_ptr = in_tile;
      vrow0 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow2 = __riscv_vle8_v_u8m2(read_ptr, vl);
      vuint8m2x3_t vrows = __riscv_vcreate_v_u8m2x3(vrow0, vrow1, vrow2);
      if (rsat == 3) {
        __riscv_vsseg3e8_v_u8m2x3(out_tile, vrows, vl);
      } else {
        __riscv_vssseg3e8_v_u8m2x3(out_tile, output_seg_bstride, vrows, vl);
      }
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
      const uint8_t *read_ptr = in_tile;
      vrow0 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow2 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow3 = __riscv_vle8_v_u8m2(read_ptr, vl);
      vuint8m2x4_t vrows = __riscv_vcreate_v_u8m2x4(vrow0, vrow1, vrow2, vrow3);
      if (rsat == 4) {
        __riscv_vsseg4e8_v_u8m2x4(out_tile, vrows, vl);
      } else {
        __riscv_vssseg4e8_v_u8m2x4(out_tile, output_seg_bstride, vrows, vl);
      }
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
      const uint8_t *read_ptr = in_tile;
      vrow0 = __riscv_vle8_v_u8m1(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle8_v_u8m1(read_ptr, vl);
      read_ptr += rsa;
      vrow2 = __riscv_vle8_v_u8m1(read_ptr, vl);
      read_ptr += rsa;
      vrow3 = __riscv_vle8_v_u8m1(read_ptr, vl);
      read_ptr += rsa;
      vrow4 = __riscv_vle8_v_u8m1(read_ptr, vl);
      vuint8m1x5_t vrows =
          __riscv_vcreate_v_u8m1x5(vrow0, vrow1, vrow2, vrow3, vrow4);
      if (rsat == 5) {
        __riscv_vsseg5e8_v_u8m1x5(out_tile, vrows, vl);
      } else {
        __riscv_vssseg5e8_v_u8m1x5(out_tile, output_seg_bstride, vrows, vl);
      }
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
      const uint8_t *read_ptr = in_tile;
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
      vuint8m1x6_t vrows =
          __riscv_vcreate_v_u8m1x6(vrow0, vrow1, vrow2, vrow3, vrow4, vrow5);
      if (rsat == 6) {
        __riscv_vsseg6e8_v_u8m1x6(out_tile, vrows, vl);
      } else {
        __riscv_vssseg6e8_v_u8m1x6(out_tile, output_seg_bstride, vrows, vl);
      }
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
      const uint8_t *read_ptr = in_tile;
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
      vuint8m1x7_t vrows = __riscv_vcreate_v_u8m1x7(vrow0, vrow1, vrow2, vrow3,
                                                    vrow4, vrow5, vrow6);
      if (rsat == 7) {
        __riscv_vsseg7e8_v_u8m1x7(out_tile, vrows, vl);
      } else {
        __riscv_vssseg7e8_v_u8m1x7(out_tile, output_seg_bstride, vrows, vl);
      }
      in_tile += vl;
      out_tile += vl * rsat;
    }
    break;
  }
  default:
    break;
  }

  m -= m_mod8;
  a += m_mod8 * rsa;
  at += m_mod8;

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

  for (size_t ii = 0; ii < m; ii += 8) {
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

      vuint8m1x8_t vrows = __riscv_vcreate_v_u8m1x8(vrow0, vrow1, vrow2, vrow3,
                                                    vrow4, vrow5, vrow6, vrow7);
      if (rsat == 8) {
        __riscv_vsseg8e8_v_u8m1x8(at + (jj * rsat) + ii, vrows, vl);
      } else {
        __riscv_vssseg8e8_v_u8m1x8(at + (jj * rsat) + ii, output_seg_bstride,
                                   vrows, vl);
      }
    }
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

/**
 * @brief Transpose a matrix and pad the output matrix with a constant value in
 * the M dimension.
 *
 * @param m_padded - Number of columns in output matrix (Must be greater than or
 * equal to m)
 * @param m - Number of rows in input matrix
 * @param n - Number of columns in input matrix and rows in output matrix.
 * @param a - Pointer to input matrix
 * @param rsa - Row stride of input matrix in elements.
 * @param at - Pointer to output matrix.
 * @param rsat - Row stride of output matrix in elements.
 * @param pad - Value to use for padding.
 *
 * Transposes an m×n matrix and pads the transposed output in the M dimension.
 *
 * Transformation:
 *   Input:  m×n matrix (m rows, n columns)
 *   Output: n×m_padded matrix (n rows, m_padded columns), where m_padded >= m
 */
SKL_FUNC_PRIVATE void skl_transpose_padded_m_e8_zve32x(
    size_t m_padded, size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
    size_t rsa, uint8_t *SKL_RESTRICT at, size_t rsat, uint8_t pad) {

  if (m_padded == m) {
    skl_transpose_e8_zve32x(m, n, a, rsa, at, rsat);
    return;
  }
  size_t m_align8 = m / 8 * 8;
  size_t m_ceil8 = (m + 7) / 8 * 8;
  size_t m_mod8 = m % 8;

  if (m_align8) {
    skl_transpose_e8_zve32x(m_align8, n, a, rsa, at, rsat);
  }
  const uint8_t *in_tile = a + m_align8 * rsa;
  uint8_t *out_tile = at + m_align8;
  if (m_mod8) {
    /* m_transition: Number of rows to process in the current tile (transpose +
     * pad). Range: [2, 8]
     *   - At least 2: minimum 1 remaining data row + 1 padding row
     *   - At most 8: limited by tile size
     */
    size_t m_transition = m_padded - m_align8 > 8 ? 8 : m_padded - m_align8;
    vuint8m8_t pad_vec = __riscv_vmv_v_x_u8m8(pad, __riscv_vsetvlmax_e8m8());

    switch (m_transition) {
    case 2: {
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m4(avl);
        vuint8m4x2_t pad_vec_group2 =
            __riscv_vcreate_v_u8m4x2(__riscv_vle8_v_u8m4(in_tile, vl),
                                     __riscv_vget_v_u8m8_u8m4(pad_vec, 1));
        if (rsat == 2) {
          __riscv_vsseg2e8_v_u8m4x2(out_tile, pad_vec_group2, vl);
        } else {
          __riscv_vssseg2e8_v_u8m4x2(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group2, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 3: {
      vuint8m2x3_t pad_vec_group3 =
          __riscv_vcreate_v_u8m2x3(__riscv_vget_v_u8m8_u8m2(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 2));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        switch (m_mod8) {
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
        if (rsat == 3) {
          __riscv_vsseg3e8_v_u8m2x3(out_tile, pad_vec_group3, vl);
        } else {
          __riscv_vssseg3e8_v_u8m2x3(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group3, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 4: {
      vuint8m2x4_t pad_vec_group4 =
          __riscv_vcreate_v_u8m2x4(__riscv_vget_v_u8m8_u8m2(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m2(pad_vec, 3));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        switch (m_mod8) {
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
        if (rsat == 4) {
          __riscv_vsseg4e8_v_u8m2x4(out_tile, pad_vec_group4, vl);
        } else {
          __riscv_vssseg4e8_v_u8m2x4(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group4, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 5: {
      vuint8m1x5_t pad_vec_group5 =
          __riscv_vcreate_v_u8m1x5(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m_mod8) {
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
        if (rsat == 5) {
          __riscv_vsseg5e8_v_u8m1x5(out_tile, pad_vec_group5, vl);
        } else {
          __riscv_vssseg5e8_v_u8m1x5(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group5, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 6: {
      vuint8m1x6_t pad_vec_group6 =
          __riscv_vcreate_v_u8m1x6(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 5));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m_mod8) {
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
        if (rsat == 6) {
          __riscv_vsseg6e8_v_u8m1x6(out_tile, pad_vec_group6, vl);
        } else {
          __riscv_vssseg6e8_v_u8m1x6(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group6, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 7: {
      vuint8m1x7_t pad_vec_group7 =
          __riscv_vcreate_v_u8m1x7(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 5),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 6));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m_mod8) {
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
        if (rsat == 7) {
          __riscv_vsseg7e8_v_u8m1x7(out_tile, pad_vec_group7, vl);
        } else {
          __riscv_vssseg7e8_v_u8m1x7(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group7, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    case 8: {
      vuint8m1x8_t pad_vec_group8 =
          __riscv_vcreate_v_u8m1x8(__riscv_vget_v_u8m8_u8m1(pad_vec, 0),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 1),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 2),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 3),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 4),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 5),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 6),
                                   __riscv_vget_v_u8m8_u8m1(pad_vec, 7));
      for (size_t vl = 0, avl = n; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        switch (m_mod8) {
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
        if (rsat == 8) {
          __riscv_vsseg8e8_v_u8m1x8(out_tile, pad_vec_group8, vl);
        } else {
          __riscv_vssseg8e8_v_u8m1x8(out_tile,
                                     (ptrdiff_t)(rsat * sizeof(uint8_t)),
                                     pad_vec_group8, vl);
        }
        in_tile += vl;
        out_tile += vl * rsat;
      }
      break;
    }
    default:
      break;
    }
  } // if (m_mod8)

  if (m_padded > m_ceil8) {
    out_tile = at + m_ceil8;
    skl_set_2d_e8_zve32x(out_tile, rsat, pad, n, m_padded - m_ceil8);
  }
}

SKL_FUNC_PRIVATE void
skl_pack_e8_e8rcprc_zve32x(size_t m, size_t n, const uint8_t *SKL_RESTRICT src,
                           size_t rs, size_t m0, size_t n0,
                           uint8_t *SKL_RESTRICT dst, size_t rs0, size_t cs0,
                           size_t rs1, size_t cs1, uint8_t pad) {

  // Handle division by zero
  if (m0 == 0 || n0 == 0) {
    return;
  }

  size_t m1 = (m + m0 - 1) / m0; // Num. block-rows in output matrix
  size_t n1 = (n + n0 - 1) / n0; // Num. block-columns in output matrix

  /* Column panels */
  if (rs0 == 1 && n0 == 1) {
    if (m0 == rs1) {
      /* column-major block ordering */
      skl_transpose_padded_m_e8_zve32x(m0 * m1, m, n, src, rs, dst, cs1, pad);
    } else {
      /* row-major block ordering */
      const uint8_t *src_block = src;
      uint8_t *dst_block = dst;
      for (size_t ii1 = 0; ii1 < m1 - 1; ++ii1) {
        skl_transpose_e8_zve32x(m0, n, src_block, rs, dst_block, cs1);
        src_block += m0 * rs;
        dst_block += rs1;
      }
      size_t m_tail = m - m0 * (m1 - 1);
      skl_transpose_padded_m_e8_zve32x(m0, m_tail, n, src_block, rs, dst_block,
                                       cs1, pad);
    }
    return;
  }

  /* intra-block: column-major, inter-block: row-major */
  if ((cs0 * n0 == cs1) && rs0 == 1) {
    const uint8_t *src_block = src;
    uint8_t *dst_block = dst;
    for (size_t ii1 = 0; ii1 < m1 - 1; ++ii1) {
      skl_transpose_e8_zve32x(m0, n, src_block, rs, dst_block, cs0);
      if (n % n0) {
        // pad right
        skl_set_2d_e8_zve32x(dst_block + cs0 * n, cs0, pad, n0 - n % n0, m0);
      }
      src_block += m0 * rs;
      dst_block += rs1;
    }
    size_t m_tail = m - m0 * (m1 - 1);
    skl_transpose_padded_m_e8_zve32x(m0, m_tail, n, src_block, rs, dst_block,
                                     cs0, pad);
    if (n % n0) {
      // pad right
      skl_set_2d_e8_zve32x(dst_block + cs0 * n, cs0, pad, n0 - n % n0, m0);
    }
    return;
  }

  /* Row panels */
  if (cs0 == 1 && m0 == 1) {
    if (n0 == cs1) {
      /* row-major block ordering */
      skl_copy_2d_padded_n_e8_zve32x(n0 * n1, m, n, src, rs, dst, rs1, pad);
    } else {
      /* column-major block ordering */
      const uint8_t *src_block = src;
      uint8_t *dst_block = dst;

      for (size_t jj1 = 0; jj1 < n1 - 1; ++jj1) {
        skl_copy_2d_e8_zve32x(m, n0, src_block, rs, dst_block, rs1);
        src_block += n0;
        dst_block += cs1;
      }
      size_t n_tail = n - n0 * (n1 - 1);
      skl_copy_2d_padded_n_e8_zve32x(n0, m, n_tail, src_block, rs, dst_block,
                                     rs1, pad);
    }
    return;
  }

  /* intra-block: row-major, inter-block: column-major */
  if ((rs0 * m0 == rs1) && cs0 == 1) {
    const uint8_t *src_block = src;
    uint8_t *dst_block = dst;
    for (size_t jj1 = 0; jj1 < n1 - 1; ++jj1) {
      skl_copy_2d_e8_zve32x(m, n0, src_block, rs, dst_block, rs0);
      if (m % m0) {
        // pad bottom
        skl_set_2d_e8_zve32x(dst_block + m * rs0, rs0, pad, m0 - m % m0, n0);
      }
      src_block += n0;
      dst_block += cs1;
    }
    size_t n_tail = n - n0 * (n1 - 1);
    skl_copy_2d_padded_n_e8_zve32x(n0, m, n_tail, src_block, rs, dst_block, rs0,
                                   pad);
    if (m % m0) {
      // pad bottom
      skl_set_2d_e8_zve32x(dst_block + m * rs0, rs0, pad, m0 - m % m0, n0);
    }
    return;
  }

  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    size_t m_length = m0 < m - ii1 * m0 ? m0 : m - ii1 * m0;
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const uint8_t *src_block = src + ii1 * m0 * rs + jj1 * n0;
      uint8_t *dst_block = dst + ii1 * rs1 + jj1 * cs1;
      size_t n_length = n0 < n - jj1 * n0 ? n0 : n - jj1 * n0;
      if (rs0 == 1) {
        skl_transpose_padded_m_e8_zve32x(m0, m_length, n_length, src_block, rs,
                                         dst_block, cs0, pad);
        // pad right
        skl_set_2d_e8_zve32x(dst_block + cs0 * n_length, cs0, pad,
                             n0 - n_length, m0);
      } else if (cs0 == 1) {
        skl_copy_2d_padded_n_e8_zve32x(n0, m_length, n_length, src_block, rs,
                                       dst_block, rs0, pad);
        // pad bottom: pad (m0 - m_length) rows, each with n0 elements
        skl_set_2d_e8_zve32x(dst_block + m_length * rs0, rs0, pad,
                             m0 - m_length, n0);
      } else {
        for (size_t ii0 = 0; ii0 < m0; ++ii0) {
          for (size_t jj0 = 0; jj0 < n0; ++jj0) {
            if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
              dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0];
            } else {
              dst_block[ii0 * rs0 + jj0 * cs0] = pad;
            }
          }
        }
      }
    }
  }
}

SKL_FUNC void skl_pack_e8rc_e8rcprc_zve32x(size_t m, size_t n,
                                           const uint8_t *SKL_RESTRICT src,
                                           size_t rs, size_t cs, size_t m0,
                                           size_t n0, uint8_t *SKL_RESTRICT dst,
                                           size_t rs0, size_t cs0, size_t rs1,
                                           size_t cs1, uint8_t pad) {

  if (cs == 1) {
    skl_pack_e8_e8rcprc_zve32x(m, n, src, rs, m0, n0, dst, rs0, cs0, rs1, cs1,
                               pad);
  } else if (rs == 1) {
    // When input is column-major (rs==1), transpose both dimensions and strides
    skl_pack_e8_e8rcprc_zve32x(n, m, src, cs, n0, m0, dst, cs0, rs0, cs1, rs1,
                               pad);
  } else {
    size_t m1 = (m + m0 - 1) / m0; // Num. block-rows in output matrix
    size_t n1 = (n + n0 - 1) / n0; // Num. block-columns in output matrix
    for (size_t ii1 = 0; ii1 < m1; ++ii1) {
      for (size_t jj1 = 0; jj1 < n1; ++jj1) {
        const uint8_t *src_block = src + ii1 * m0 * rs + jj1 * n0 * cs;
        uint8_t *dst_block = dst + ii1 * rs1 + jj1 * cs1;
        for (size_t ii0 = 0; ii0 < m0; ++ii0) {
          for (size_t jj0 = 0; jj0 < n0; ++jj0) {
            if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
              dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0 * cs];
            } else {
              dst_block[ii0 * rs0 + jj0 * cs0] = pad;
            }
          }
        }
      }
    }
  }
}
