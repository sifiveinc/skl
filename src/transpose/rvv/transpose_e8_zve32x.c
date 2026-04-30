// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_zve32x)
#error This file requires the Zve32x extension
#endif

#include <riscv_vector.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

/* Determines the optimal vectorization strategy for 8-bit element transpose.
 * Returns true for M-dimension vectorization, false for N-dimension
 * vectorization.
 */
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
