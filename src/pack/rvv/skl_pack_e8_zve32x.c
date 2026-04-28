// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC_PRIVATE bool skl_transpose_e8_is_mvec(size_t m, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e8m1();
  if (m > n || m >= vlmax) {
    return true;
  }
  return false;
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided segment-8 loads and unit-strided
 * stores. Vectorization is performed along the M dimension, and N is expected
 * to be a multiple of 8.
 */
SKL_FUNC_PRIVATE void
skl_transpose_mvec_x8_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint8_t));

  size_t vl = 0;
  for (size_t ii = 0; ii + 7 < n; ii += 8) {
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
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 1), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 2), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 3), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 4), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 5), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 6), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m1(write_ptr, __riscv_vget_v_u8m1x8_u8m1(vcols, 7), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided segment-4 loads and unit-strided
 * stores. Vectorization is performed along the M dimension, and N is expected
 * to be a multiple of 4.
 */
SKL_FUNC_PRIVATE void
skl_transpose_mvec_x4_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint8_t));

  size_t vl = 0;

  for (size_t ii = 0; ii + 3 < n; ii += 4) {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m2x4_t vcols = __riscv_vundefined_u8m2x4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    const uint8_t *in_tile = a + ii;
    uint8_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m2(m - jj);

      vcols = __riscv_vlsseg4e8_v_u8m2x4(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment continuously along m
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
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided segment-2 loads and unit-strided
 * stores. Vectorization is performed along the M dimension, and N is expected
 * to be a multiple of 2.
 */
SKL_FUNC_PRIVATE void
skl_transpose_mvec_x2_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint8_t));

  size_t vl = 0;

  for (size_t ii = 0; ii + 1 < n; ii += 2) {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint8m4x2_t vcols = __riscv_vundefined_u8m4x2();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    const uint8_t *in_tile = a + ii;
    uint8_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m4(m - jj);

      vcols = __riscv_vlsseg2e8_v_u8m4x2(in_tile, input_seg_bstride, vl);

      uint8_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment continuously along m
      __riscv_vse8_v_u8m4(write_ptr, __riscv_vget_v_u8m4x2_u8m4(vcols, 0), vl);
      write_ptr += output_seg_stride;
      __riscv_vse8_v_u8m4(write_ptr, __riscv_vget_v_u8m4x2_u8m4(vcols, 1), vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided loads and unit-strided stores.
 * Vectorization is performed along the M dimension.
 */
SKL_FUNC_PRIVATE void
skl_transpose_mvec_x1_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_bstride = (ptrdiff_t)(rsa * sizeof(uint8_t));

  size_t vl = 0;

  for (size_t ii = 0; ii < n; ii += 1) {
    const uint8_t *in_tile = a + ii;
    uint8_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e8m8(m - jj);
      vuint8m8_t v_data = __riscv_vlse8_v_u8m8(in_tile, input_bstride, vl);
      __riscv_vse8_v_u8m8(out_tile, v_data, vl);
      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) with vectorization along the M dimension.
 */
SKL_FUNC_PRIVATE void
skl_transpose_mvec_e8_zve32x(size_t m, size_t n, const uint8_t *SKL_RESTRICT a,
                             size_t rsa, uint8_t *SKL_RESTRICT at,
                             size_t rsat) {

  size_t n_begin = 0;
  size_t n_end = n_begin + ((n - n_begin) / 8) * 8;

  skl_transpose_mvec_x8_e8_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                  at + n_begin * rsat, rsat);

  n_begin = n_end;
  n_end = n_begin + ((n - n_begin) / 4) * 4;

  skl_transpose_mvec_x4_e8_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                  at + n_begin * rsat, rsat);

  n_begin = n_end;
  n_end = n_begin + ((n - n_begin) / 2) * 2;

  skl_transpose_mvec_x2_e8_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                  at + n_begin * rsat, rsat);

  n_begin = n_end;
  n_end = n;

  skl_transpose_mvec_x1_e8_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                  at + n_begin * rsat, rsat);
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided segment-8
 * stores. Vectorization is performed along the N dimension, and M is expected
 * to be a multiple of 8.
 */
SKL_FUNC_PRIVATE void
skl_transpose_nvec_x8_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint8_t));

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

  for (size_t ii = 0; ii + 7 < m; ii += 8) {
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

      __riscv_vssseg8e8_v_u8m1x8(at + (jj * rsat) + ii, output_seg_bstride,
                                 vrows, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided segment-4
 * stores. Vectorization is performed along the N dimension, and M is expected
 * to be a multiple of 4.
 */
SKL_FUNC_PRIVATE void
skl_transpose_nvec_x4_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint8_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint8m2_t vrow0 = __riscv_vundefined_u8m2();
  vuint8m2_t vrow1 = __riscv_vundefined_u8m2();
  vuint8m2_t vrow2 = __riscv_vundefined_u8m2();
  vuint8m2_t vrow3 = __riscv_vundefined_u8m2();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii + 3 < m; ii += 4) {
    for (size_t jj = 0; jj < n; jj += vl) {
      const uint8_t *read_ptr = a + ii * rsa + jj;
      vl = __riscv_vsetvl_e8m2(n - jj);
      vrow0 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow2 = __riscv_vle8_v_u8m2(read_ptr, vl);
      read_ptr += rsa;
      vrow3 = __riscv_vle8_v_u8m2(read_ptr, vl);

      vuint8m2x4_t vrows = __riscv_vcreate_v_u8m2x4(vrow0, vrow1, vrow2, vrow3);
      __riscv_vssseg4e8_v_u8m2x4(at + (jj * rsat) + ii, output_seg_bstride,
                                 vrows, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided segment-2
 * stores. Vectorization is performed along the N dimension, and M is expected
 * to be a multiple of 2.
 */
SKL_FUNC_PRIVATE void
skl_transpose_nvec_x2_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint8_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint8m4_t vrow0 = __riscv_vundefined_u8m4();
  vuint8m4_t vrow1 = __riscv_vundefined_u8m4();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii + 1 < m; ii += 2) {
    for (size_t jj = 0; jj < n; jj += vl) {
      const uint8_t *read_ptr = a + ii * rsa + jj;
      vl = __riscv_vsetvl_e8m4(n - jj);
      vrow0 = __riscv_vle8_v_u8m4(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle8_v_u8m4(read_ptr, vl);

      vuint8m4x2_t vrows = __riscv_vcreate_v_u8m4x2(vrow0, vrow1);
      __riscv_vssseg2e8_v_u8m4x2(at + (jj * rsat) + ii, output_seg_bstride,
                                 vrows, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided stores.
 * Vectorization is performed along the N dimension.
 */
SKL_FUNC_PRIVATE void
skl_transpose_nvec_x1_e8_zve32x(size_t m, size_t n,
                                const uint8_t *SKL_RESTRICT a, size_t rsa,
                                uint8_t *SKL_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_bstride = (ptrdiff_t)(rsat * sizeof(uint8_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint8m8_t vrow0 = __riscv_vundefined_u8m8();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii < m; ii++) {
    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e8m8(n - jj);
      vrow0 = __riscv_vle8_v_u8m8(a + (ii + 0) * rsa + jj, vl);
      __riscv_vsse8_v_u8m8(at + (jj * rsat) + ii, output_bstride, vrow0, vl);
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

  size_t m_begin = 0;
  size_t m_end = m_begin + ((m - m_begin) / 8) * 8;

  skl_transpose_nvec_x8_e8_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                  at + m_begin, rsat);

  m_begin = m_end;
  m_end = m_begin + ((m - m_begin) / 4) * 4;

  skl_transpose_nvec_x4_e8_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                  at + m_begin, rsat);

  m_begin = m_end;
  m_end = m_begin + ((m - m_begin) / 2) * 2;

  skl_transpose_nvec_x2_e8_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                  at + m_begin, rsat);

  m_begin = m_end;
  m_end = m;

  skl_transpose_nvec_x1_e8_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                  at + m_begin, rsat);
}

SKL_FUNC void skl_transpose_e8_zve32x(size_t m, size_t n,
                                      const uint8_t *SKL_RESTRICT a, size_t rsa,
                                      uint8_t *SKL_RESTRICT at, size_t rsat) {
  if (skl_transpose_e8_is_mvec(m, n)) {
    skl_transpose_mvec_e8_zve32x(m, n, a, rsa, at, rsat);
  } else {
    skl_transpose_nvec_e8_zve32x(m, n, a, rsa, at, rsat);
  }
}

SKL_FUNC void skl_padd_e8_zve32x(uint8_t *dst, size_t continuous, size_t stride,
                                 uint8_t pad_val, size_t n_section) {
  if (continuous == stride) {
    memset(dst, pad_val, continuous * n_section * sizeof(uint8_t));
  } else if (continuous <= 8) {

    vuint8m1_t padd_vec0 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec1 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec2 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec3 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec4 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec5 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec6 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());
    vuint8m1_t padd_vec7 =
        __riscv_vmv_v_x_u8m1(pad_val, __riscv_vsetvlmax_e8m1());

    switch (continuous) {
    case 1:
      vuint8m8_t padd_vec_group1 = __riscv_vcreate_v_u8m1_u8m8(
          padd_vec0, padd_vec1, padd_vec2, padd_vec3, padd_vec4, padd_vec5,
          padd_vec6, padd_vec7);
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m8(avl);
        __riscv_vsse8_v_u8m8(dst, stride * sizeof(uint8_t), padd_vec_group1,
                             vl);
        dst += vl * stride;
      }
      break;
    case 2:
      vuint8m4x2_t padd_vec_group2 = __riscv_vcreate_v_u8m4x2(
          __riscv_vcreate_v_u8m1_u8m4(padd_vec0, padd_vec1, padd_vec2,
                                      padd_vec3),
          __riscv_vcreate_v_u8m1_u8m4(padd_vec4, padd_vec5, padd_vec6,
                                      padd_vec7));
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m4(avl);
        __riscv_vssseg2e8_v_u8m4x2(dst, stride * sizeof(uint8_t),
                                   padd_vec_group2, vl);
        dst += vl * stride;
      }
      break;
    case 3:
      vuint8m2x3_t padd_vec_group3 = __riscv_vcreate_v_u8m2x3(
          __riscv_vcreate_v_u8m1_u8m2(padd_vec0, padd_vec1),
          __riscv_vcreate_v_u8m1_u8m2(padd_vec2, padd_vec3),
          __riscv_vcreate_v_u8m1_u8m2(padd_vec4, padd_vec5));
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        __riscv_vssseg3e8_v_u8m2x3(dst, stride * sizeof(uint8_t),
                                   padd_vec_group3, vl);
        dst += vl * stride;
      }
      break;
    case 4:
      vuint8m2x4_t padd_vec_group4 = __riscv_vcreate_v_u8m2x4(
          __riscv_vcreate_v_u8m1_u8m2(padd_vec0, padd_vec1),
          __riscv_vcreate_v_u8m1_u8m2(padd_vec2, padd_vec3),
          __riscv_vcreate_v_u8m1_u8m2(padd_vec4, padd_vec5),
          __riscv_vcreate_v_u8m1_u8m2(padd_vec6, padd_vec7));
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m2(avl);
        __riscv_vssseg4e8_v_u8m2x4(dst, stride * sizeof(uint8_t),
                                   padd_vec_group4, vl);
        dst += vl * stride;
      }
      break;
    case 5:
      vuint8m1x5_t padd_vec_group5 = __riscv_vcreate_v_u8m1x5(
          padd_vec0, padd_vec1, padd_vec2, padd_vec3, padd_vec4);
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg5e8_v_u8m1x5(dst, stride * sizeof(uint8_t),
                                   padd_vec_group5, vl);
        dst += vl * stride;
      }
      break;
    case 6:
      vuint8m1x6_t padd_vec_group6 = __riscv_vcreate_v_u8m1x6(
          padd_vec0, padd_vec1, padd_vec2, padd_vec3, padd_vec4, padd_vec5);
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg6e8_v_u8m1x6(dst, stride * sizeof(uint8_t),
                                   padd_vec_group6, vl);
        dst += vl * stride;
      }
      break;
    case 7:
      vuint8m1x7_t padd_vec_group7 =
          __riscv_vcreate_v_u8m1x7(padd_vec0, padd_vec1, padd_vec2, padd_vec3,
                                   padd_vec4, padd_vec5, padd_vec6);
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg7e8_v_u8m1x7(dst, stride * sizeof(uint8_t),
                                   padd_vec_group7, vl);
        dst += vl * stride;
      }
      break;
    case 8:
      vuint8m1x8_t padd_vec_group8 =
          __riscv_vcreate_v_u8m1x8(padd_vec0, padd_vec1, padd_vec2, padd_vec3,
                                   padd_vec4, padd_vec5, padd_vec6, padd_vec7);
      for (size_t vl = 0, avl = n_section; avl > 0; avl -= vl) {
        vl = __riscv_vsetvl_e8m1(avl);
        __riscv_vssseg8e8_v_u8m1x8(dst, stride * sizeof(uint8_t),
                                   padd_vec_group8, vl);
        dst += vl * stride;
      }
      break;
    default:
      break;
    }
  } else {
    for (size_t i = 0; i < n_section; ++i) {
      memset(dst, pad_val, continuous * sizeof(uint8_t));
      dst += stride;
    }
  }
}

SKL_FUNC void skl_pack_rcbrc_e8_zve32x(
    size_t m,           // Num. rows in input matrix
    size_t n,           // Num. columns in input matrix
    const uint8_t *src, // Input matrix
    size_t rs,          // Row stride of input matrix
    size_t cs,          // Column stride of input matrix
    size_t m0,          // Num. rows in a block of the input matrix
    size_t n0,          // Num. columns in a block of the input matrix
    uint8_t *dst,       // Output packed matrix [m1 x n1]
    size_t rs0,         // Row stride within a block of the output matrix
    size_t cs0,         // Column stride within a block of the output matrix
    size_t rs1,         // Row stride between blocks of the output matrix
    size_t cs1          // Column stride between blocks of the output matrix
) {

  size_t m1 = (m + m0 - 1) / m0; // Num. row blocks in the input matrix
  size_t n1 = (n + n0 - 1) / n0; // Num. column blocks in the input matrix

  if ((cs0 * n0 == cs1) && (cs == 1 && rs0 == 1)) {
    for (size_t ii1 = 0; ii1 < m1; ++ii1) {
      const uint8_t *src_block = src + ii1 * m0 * rs;
      uint8_t *dst_block = dst + ii1 * rs1;
      if (ii1 != m1 - 1) {
        skl_transpose_e8_zve32x(m0, n, src_block, rs, dst_block, cs0);
        if (n % n0) {
          // padd right
          skl_padd_e8_zve32x(dst_block + cs1 * (n1 - 1) + cs0 * (n % n0), m0,
                             cs0, 0, n0 - n % n0);
        }
      } else {
        skl_transpose_e8_zve32x(m - ii1 * m0, n, src_block, rs, dst_block, cs0);
        if (n % n0) {
          // padd right (bottom included)
          skl_padd_e8_zve32x(dst_block + cs1 * (n1 - 1) + cs0 * (n % n0), m0,
                             cs0, 0, n0 - n % n0);
        }
        // padd bottom
        skl_padd_e8_zve32x(dst_block + m % m0, m0 - m % m0, cs0, 0, n);
      }
    }
    return;
  }

  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      const uint8_t *src_block = src + ii1 * m0 * rs + jj1 * n0 * cs;
      uint8_t *dst_block = dst + ii1 * rs1 + jj1 * cs1;

      if (rs == 1 && cs0 == 1) {
        if (not fringe)
          skl_transpose_e8_zve32x(n0, m0, src_block, cs, dst_block, rs0);
        else
          skl_transpose_e8_zve32x(remain_n0, remain_m0, src_block, cs,
                                  dst_block, rs0);
      } else if (cs == 1 && rs0 == 1) {
        if (not fringe)
          skl_transpose_e8_zve32x(m0, n0, src_block, rs, dst_block, cs0);
        else
          skl_transpose_e8_zve32x(remain_m0, remain_n0, src_block, rs,
                                  dst_block, cs0);
      } else {

        for (size_t ii0 = 0; ii0 < m0; ++ii0) {
          for (size_t jj0 = 0; jj0 < n0; ++jj0) {
            if (ii1 * m0 + ii0 < m && jj1 * n0 + jj0 < n) {
              dst_block[ii0 * rs0 + jj0 * cs0] = src_block[ii0 * rs + jj0 * cs];
            } else {
              // Pad with zeros
              dst_block[ii0 * rs0 + jj0 * cs0] = 0;
            }
          }
        }
      }
    }
  }
}
