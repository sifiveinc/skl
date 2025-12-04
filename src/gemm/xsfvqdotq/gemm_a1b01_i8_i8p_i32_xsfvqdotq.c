// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_zve32x)
#error This source file requires compiler support for the RISC-V Zve32x extension.
#endif

#if !defined(__riscv_xsfvqdotq)
#error This source file requires compiler support for the Xsfvqdotq extension.
#endif

#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC_PRIVATE void skl_mm_6xe32m4_k_multiple_of_4_i8i32_vqdotvx(
    size_t k, size_t n, size_t lda, size_t a_tile_offset, size_t b_tile_offset,
    size_t ldc, const int8_t *a, const int8_t *b, int32_t *c, bool accum) {
  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vint32m4_t cvec0 = __riscv_vundefined_i32m4();
  vint32m4_t cvec1 = __riscv_vundefined_i32m4();
  vint32m4_t cvec2 = __riscv_vundefined_i32m4();
  vint32m4_t cvec3 = __riscv_vundefined_i32m4();
  vint32m4_t cvec4 = __riscv_vundefined_i32m4();
  vint32m4_t cvec5 = __riscv_vundefined_i32m4();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)
  if (!accum) {
    cvec0 = __riscv_vmv_v_x_i32m4(0, n);
    cvec1 = __riscv_vmv_v_x_i32m4(0, n);
    cvec2 = __riscv_vmv_v_x_i32m4(0, n);
    cvec3 = __riscv_vmv_v_x_i32m4(0, n);
    cvec4 = __riscv_vmv_v_x_i32m4(0, n);
    cvec5 = __riscv_vmv_v_x_i32m4(0, n);
  } else {
    // load 6 rows from C.
    int32_t *c_read = c;
    cvec0 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += ldc;
    cvec1 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += ldc;
    cvec2 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += ldc;
    cvec3 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += ldc;
    cvec4 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += ldc;
    cvec5 = __riscv_vle32_v_i32m4(c_read, n);
  }

  size_t k_rem = k % 4;
  size_t k_idx = 0;
  // compute 1 tile at once.
  const int8_t *a_tile_ptr = a;
  for (; k_idx + 3 < k; k_idx += 4) {
    // load one B tile (4xn).
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a_tile_ptr, n);
    a_tile_ptr += lda;
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, *(uint32_t *)a_tile_ptr, n);
    a_tile_ptr += lda;
    cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, *(uint32_t *)a_tile_ptr, n);
    a_tile_ptr += lda;
    cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, *(uint32_t *)a_tile_ptr, n);
    a_tile_ptr += lda;
    cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, *(uint32_t *)a_tile_ptr, n);
    a_tile_ptr += lda;
    cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec, *(uint32_t *)a_tile_ptr, n);

    a += a_tile_offset;
    a_tile_ptr = a;
  }

  // handle k dimension of `a` is not the multiple of 4.
  if (k_rem) {
    uint32_t mask = 0xffffffff >> (4 - k_rem) * 8;
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec,
                                      *(uint32_t *)a_tile_ptr & mask, n);
    a_tile_ptr += lda;
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec,
                                      *(uint32_t *)a_tile_ptr & mask, n);
    a_tile_ptr += lda;
    cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec,
                                      *(uint32_t *)a_tile_ptr & mask, n);
    a_tile_ptr += lda;
    cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec,
                                      *(uint32_t *)a_tile_ptr & mask, n);
    a_tile_ptr += lda;
    cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec,
                                      *(uint32_t *)a_tile_ptr & mask, n);
    a_tile_ptr += lda;
    cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec,
                                      *(uint32_t *)a_tile_ptr & mask, n);
  }

  // write result back to c tile.
  __riscv_vse32_v_i32m4(c, cvec0, n);
  c += ldc;
  __riscv_vse32_v_i32m4(c, cvec1, n);
  c += ldc;
  __riscv_vse32_v_i32m4(c, cvec2, n);
  c += ldc;
  __riscv_vse32_v_i32m4(c, cvec3, n);
  c += ldc;
  __riscv_vse32_v_i32m4(c, cvec4, n);
  c += ldc;
  __riscv_vse32_v_i32m4(c, cvec5, n);
}

SKL_FUNC_PRIVATE void skl_mm_lt6xe32m4_k_multiple_of_4_i8i32_vqdotvx(
    size_t m, size_t k, size_t n, size_t lda, size_t a_tile_offset,
    size_t b_tile_offset, size_t ldc, const int8_t *a, const int8_t *b,
    int32_t *c, bool accum) {
  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vint32m4_t cvec0 = __riscv_vundefined_i32m4();
  vint32m4_t cvec1 = __riscv_vundefined_i32m4();
  vint32m4_t cvec2 = __riscv_vundefined_i32m4();
  vint32m4_t cvec3 = __riscv_vundefined_i32m4();
  vint32m4_t cvec4 = __riscv_vundefined_i32m4();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  if (!accum) {
    cvec0 = __riscv_vmv_v_x_i32m4(0, n);
    cvec1 = __riscv_vmv_v_x_i32m4(0, n);
    cvec2 = __riscv_vmv_v_x_i32m4(0, n);
    cvec3 = __riscv_vmv_v_x_i32m4(0, n);
    cvec4 = __riscv_vmv_v_x_i32m4(0, n);
  } else {
    int32_t *c_read = c + (m - 1) * ldc;
    switch (m) {
    case 5:
      cvec4 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= ldc;
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= ldc;
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= ldc;
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= ldc;
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_vle32_v_i32m4(c_read, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  size_t k_rem = k % 4;
  size_t k_idx = 0;
  // compute 1 micro-tile at once.
  for (const int8_t *a_tile_ptr = a + (m - 1) * lda; k_idx + 3 < k;
       k_idx += 4) {
    // load one B tile (4xn).
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;

    // compute micro-tile.
    switch (m) {
    case 5:
      cvec4 =
          __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, *(uint32_t *)a_tile_ptr, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 4:
      cvec3 =
          __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, *(uint32_t *)a_tile_ptr, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 3:
      cvec2 =
          __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, *(uint32_t *)a_tile_ptr, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 2:
      cvec1 =
          __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, *(uint32_t *)a_tile_ptr, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 1:
      cvec0 =
          __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a_tile_ptr, n);
      __attribute__((fallthrough));
    default:
      break;
    }

    a += a_tile_offset;
    a_tile_ptr = a + (m - 1) * lda;
  }

  // handle k dimension of `a` is not the multiple of 4.
  if (k_rem) {
    const int8_t *a_tile_ptr = a + (m - 1) * lda;
    uint32_t mask = 0xffffffff >> (4 - k_rem) * 8;
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);

    // compute micro-tile.
    switch (m) {
    case 5:
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec,
                                        *(uint32_t *)a_tile_ptr & mask, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec,
                                        *(uint32_t *)a_tile_ptr & mask, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec,
                                        *(uint32_t *)a_tile_ptr & mask, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec,
                                        *(uint32_t *)a_tile_ptr & mask, n);
      a_tile_ptr -= lda;
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec,
                                        *(uint32_t *)a_tile_ptr & mask, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  // write result back to c tile.
  c += (m - 1) * ldc;
  switch (m) {
  case 5:
    __riscv_vse32_v_i32m4(c, cvec4, n);
    c -= ldc;
    __attribute__((fallthrough));
  case 4:
    __riscv_vse32_v_i32m4(c, cvec3, n);
    c -= ldc;
    __attribute__((fallthrough));
  case 3:
    __riscv_vse32_v_i32m4(c, cvec2, n);
    c -= ldc;
    __attribute__((fallthrough));
  case 2:
    __riscv_vse32_v_i32m4(c, cvec1, n);
    c -= ldc;
    __attribute__((fallthrough));
  case 1:
    __riscv_vse32_v_i32m4(c, cvec0, n);
    __attribute__((fallthrough));
  default:
    break;
  }
}

SKL_FUNC_PRIVATE void skl_vm_1xe32m4_k_multiple_of_4_i8i32_vqdotvx(
    size_t k, size_t n, size_t b_tile_offset, const int8_t *a, const int8_t *b,
    int32_t *c, bool accum) {
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  vint32m4_t cvec = __riscv_vundefined_i32m4();
  if (!accum) {
    cvec = __riscv_vmv_v_x_i32m4(0, n);
  } else {
    // load 1 row from C.
    cvec = __riscv_vle32_v_i32m4(c, n);
  }
  // init 2 int32m4_t accumulators.
  vint32m4_t cvec0 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t cvec1 = __riscv_vmv_v_x_i32m4(0, n);

  size_t k_rem = k % 4;
  size_t k_idx = 0;
  for (; k_idx + 15 < k; k_idx += 16) {
    // load 4 words from A.
    uint32_t a0 = *(uint32_t *)a;
    a += 4;
    uint32_t a1 = *(uint32_t *)a;
    a += 4;
    uint32_t a2 = *(uint32_t *)a;
    a += 4;
    uint32_t a3 = *(uint32_t *)a;
    a += 4;

    // load 4 B tiles (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;
    vint8m4_t bvec2 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;
    vint8m4_t bvec3 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1, n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec2, a2, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec3, a3, n);
  }

  for (; k_idx + 7 < k; k_idx += 8) {
    // load 2 words from A.
    uint32_t a0 = *(uint32_t *)a;
    a += 4;
    uint32_t a1 = *(uint32_t *)a;
    a += 4;

    // load 2 B tiles (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1, n);
  }

  for (; k_idx + 3 < k; k_idx += 4) {
    // load 1 word from A.
    uint32_t a0 = *(uint32_t *)a;
    a += 4;

    // load 1 B tile (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += b_tile_offset;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
  }

  if (k_rem) {
    uint32_t mask = 0xffffffff >> (4 - k_rem) * 8;
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a & mask, n);
  }

  cvec0 = __riscv_vadd_vv_i32m4(cvec0, cvec1, n);
  cvec = __riscv_vadd_vv_i32m4(cvec, cvec0, n);
  __riscv_vse32_v_i32m4(c, cvec, n);
}

SKL_FUNC void skl_gemm_a1b01_i8_i8p_i32_xsfvqdotq(size_t m, size_t n, size_t k,
                                                  const int8_t *a, size_t rsa,
                                                  const int8_t *b_pack,
                                                  size_t rsb_pack, int32_t *c,
                                                  size_t rsc, bool accum) {
  const size_t m0 = 6;
  const size_t n0 = __riscv_vsetvlmax_e32m4();
  const size_t k0 = 4;

  if (m == 1) {
    // dispatch to GEMV kernel for better performance.
    for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
      int32_t *c_write = c + n_idx;
      const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
      const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
      skl_vm_1xe32m4_k_multiple_of_4_i8i32_vqdotvx(k, n_tile, k0 * rsb_pack, a,
                                                   b_tile_ptr, c_write, accum);
    }
  } else {
    size_t m_idx = 0;
    for (; m_idx + m0 - 1 < m; m_idx += m0) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a + m_idx * rsa;
      for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
        const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
        const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
        skl_mm_6xe32m4_k_multiple_of_4_i8i32_vqdotvx(
            k, n_tile, rsa, k0, k0 * rsb_pack, rsc, a_tile_ptr, b_tile_ptr,
            c_write, accum);
        c_write += n_tile;
      }
    }

    size_t m_left = m - m_idx;
    if (m_left) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a + m_idx * rsa;
      for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
        const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
        const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
        skl_mm_lt6xe32m4_k_multiple_of_4_i8i32_vqdotvx(
            m_left, k, n_tile, rsa, k0, k0 * rsb_pack, rsc, a_tile_ptr,
            b_tile_ptr, c_write, accum);
        c_write += n_tile;
      }
    }
  }
}
