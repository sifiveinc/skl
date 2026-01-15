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
#include <string.h>

#include "skl-common.h"

SKL_FUNC_PRIVATE void skl_mm_6xe32m4_i8i32_vqdotvx(size_t n, size_t k,
                                                   const int8_t *a, size_t rsa1,
                                                   size_t csa1, const int8_t *b,
                                                   size_t rsb1, int32_t *c,
                                                   size_t rsc, bool accum) {
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
    c_read += rsc;
    cvec1 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec2 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec3 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec4 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec5 = __riscv_vle32_v_i32m4(c_read, n);
  }

  const int8_t *a0 = a + 0 * rsa1;
  const int8_t *a1 = a + 1 * rsa1;
  const int8_t *a2 = a + 2 * rsa1;
  const int8_t *a3 = a + 3 * rsa1;
  const int8_t *a4 = a + 4 * rsa1;
  const int8_t *a5 = a + 5 * rsa1;

  uint32_t a0_0;
  uint32_t a0_1;
  uint32_t a0_2;
  uint32_t a0_3;
  uint32_t a0_4;
  uint32_t a0_5;
  uint32_t a1_0;
  uint32_t a1_1;
  uint32_t a1_2;
  uint32_t a1_3;
  uint32_t a1_4;
  uint32_t a1_5;

  // process the first (k / 4) * 4 elements of the inner dimension
  if (k >= 4) {
    memcpy(&a0_0, a0, 4);
    a0 += csa1;
    memcpy(&a0_1, a1, 4);
    a1 += csa1;
    memcpy(&a0_2, a2, 4);
    a2 += csa1;
    memcpy(&a0_3, a3, 4);
    a3 += csa1;
    memcpy(&a0_4, a4, 4);
    a4 += csa1;
    memcpy(&a0_5, a5, 4);
    a5 += csa1;

    // load first B tile (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    while (k >= 12) {
      vint8m4_t bvec1;
      __asm__ volatile(
          // compute first tile
          "vsetvli x0, %[n], e32, m4, ta, ma\n"
          "sf.vqdot.vx %[cvec0], %[bvec0], %[a0_0]\n"
          "sf.vqdot.vx %[cvec1], %[bvec0], %[a0_1]\n"
          "sf.vqdot.vx %[cvec2], %[bvec0], %[a0_2]\n"
          "sf.vqdot.vx %[cvec3], %[bvec0], %[a0_3]\n"

          // load second B tile (4xn)
          "vsetvli x0, %[n4], e8, m4, ta, ma\n"
          "vle8.v %[bvec1], (%[b])\n"
          "add %[b], %[b], %[rsb1]\n"
          : [cvec0] "+&vr"(cvec0), [cvec1] "+&vr"(cvec1), [cvec2] "+&vr"(cvec2),
            [cvec3] "+&vr"(cvec3), [bvec1] "=&vr"(bvec1), [b] "+&r"(b)
          : [bvec0] "vr"(bvec0), [a0_0] "r"(a0_0), [a0_1] "r"(a0_1),
            [a0_2] "r"(a0_2), [a0_3] "r"(a0_3),
            [rsb1] "rI"(rsb1 * sizeof(int8_t)), [n] "r"(n), [n4] "r"(4 * n)
          : "vl", "vtype", "memory");

      memcpy(&a1_0, a0, 4);
      a0 += csa1;
      memcpy(&a1_1, a1, 4);
      a1 += csa1;
      memcpy(&a1_2, a2, 4);
      a2 += csa1;
      memcpy(&a1_3, a3, 4);
      a3 += csa1;
      memcpy(&a1_4, a4, 4);
      a4 += csa1;
      memcpy(&a1_5, a5, 4);
      a5 += csa1;

      __asm__ volatile(
          "vsetvli x0, %[n], e32, m4, ta, ma\n"
          "sf.vqdot.vx %[cvec4], %[bvec0], %[a0_4]\n"
          "sf.vqdot.vx %[cvec5], %[bvec0], %[a0_5]\n"

          // compute second tile
          "sf.vqdot.vx %[cvec0], %[bvec1], %[a1_0]\n"
          "sf.vqdot.vx %[cvec1], %[bvec1], %[a1_1]\n"
          "sf.vqdot.vx %[cvec2], %[bvec1], %[a1_2]\n"
          "sf.vqdot.vx %[cvec3], %[bvec1], %[a1_3]\n"
          "sf.vqdot.vx %[cvec4], %[bvec1], %[a1_4]\n"
          "sf.vqdot.vx %[cvec5], %[bvec1], %[a1_5]\n"

          // pre-load first B tile (4xn) of next iteration
          "vsetvli x0, %[n4], e8, m4, ta, ma\n"
          "vle8.v %[bvec0], (%[b])\n"
          "add %[b], %[b], %[rsb1]\n"
          : [cvec0] "+&vr"(cvec0), [cvec1] "+&vr"(cvec1), [cvec2] "+&vr"(cvec2),
            [cvec3] "+&vr"(cvec3), [cvec4] "+&vr"(cvec4), [cvec5] "+&vr"(cvec5),
            [bvec0] "+&vr"(bvec0), [b] "+&r"(b)
          : [bvec1] "vr"(bvec1), [a0_4] "r"(a0_4), [a0_5] "r"(a0_5),
            [a1_0] "r"(a1_0), [a1_1] "r"(a1_1), [a1_2] "r"(a1_2),
            [a1_3] "r"(a1_3), [a1_4] "r"(a1_4), [a1_5] "r"(a1_5),
            [rsb1] "rI"(rsb1 * sizeof(int8_t)), [n] "r"(n), [n4] "r"(4 * n)
          : "vl", "vtype", "memory");

      memcpy(&a0_0, a0, 4);
      a0 += csa1;
      memcpy(&a0_1, a1, 4);
      a1 += csa1;
      memcpy(&a0_2, a2, 4);
      a2 += csa1;
      memcpy(&a0_3, a3, 4);
      a3 += csa1;
      memcpy(&a0_4, a4, 4);
      a4 += csa1;
      memcpy(&a0_5, a5, 4);
      a5 += csa1;

      k -= 8;
    }

    if (k >= 8) { // 8 <= k < 11
      // compute first tile
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0_0, n);
      memcpy(&a1_0, a0, 4);
      a0 += csa1;
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec0, a0_1, n);
      memcpy(&a1_1, a1, 4);
      a1 += csa1;

      // load second B tile (4xn)
      vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
      b += rsb1;

      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec0, a0_2, n);
      memcpy(&a1_2, a2, 4);
      a2 += csa1;
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec0, a0_3, n);
      memcpy(&a1_3, a3, 4);
      a3 += csa1;
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec0, a0_4, n);
      memcpy(&a1_4, a4, 4);
      a4 += csa1;
      cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec0, a0_5, n);
      memcpy(&a1_5, a5, 4);
      a5 += csa1;

      // compute second tile
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec1, a1_0, n);
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1_1, n);
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec1, a1_2, n);
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec1, a1_3, n);
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec1, a1_4, n);
      cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec1, a1_5, n);

      k -= 8;
    } else { // 4 <= k < 8
      // compute first tile
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0_0, n);
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec0, a0_1, n);
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec0, a0_2, n);
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec0, a0_3, n);
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec0, a0_4, n);
      cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec0, a0_5, n);

      k -= 4;
    }
  }

  // process the last k % 4 elements of the inner dimension
  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));

    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);

    memcpy(&a0_0, a0, k);
    memcpy(&a0_1, a1, k);
    memcpy(&a0_2, a2, k);
    memcpy(&a0_3, a3, k);
    memcpy(&a0_4, a4, k);
    memcpy(&a0_5, a5, k);

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, a0_0 & mask, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, a0_1 & mask, n);
    cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, a0_2 & mask, n);
    cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, a0_3 & mask, n);
    cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, a0_4 & mask, n);
    cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec, a0_5 & mask, n);
  }

  // write result back to c tile.
  __riscv_vse32_v_i32m4(c, cvec0, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec1, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec2, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec3, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec4, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec5, n);
}

// a is 4-byte aligned and rsa1 is a multiple of 4
SKL_FUNC_PRIVATE void skl_mm_6xe32m4_aligned_i8i32_vqdotvx(
    size_t n, size_t k, const int8_t *a, size_t rsa1, size_t csa1,
    const int8_t *b, size_t rsb1, int32_t *c, size_t rsc, bool accum) {
  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vint32m4_t cvec0 = __riscv_vundefined_i32m4();
  vint32m4_t cvec1 = __riscv_vundefined_i32m4();
  vint32m4_t cvec2 = __riscv_vundefined_i32m4();
  vint32m4_t cvec3 = __riscv_vundefined_i32m4();
  // vint32m4_t cvec4 = __riscv_vundefined_i32m4();
  // vint32m4_t cvec5 = __riscv_vundefined_i32m4();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)
  if (!accum) {
    cvec0 = __riscv_vmv_v_x_i32m4(0, n);
    cvec1 = __riscv_vmv_v_x_i32m4(0, n);
    cvec2 = __riscv_vmv_v_x_i32m4(0, n);
    cvec3 = __riscv_vmv_v_x_i32m4(0, n);
    // cvec4 = __riscv_vmv_v_x_i32m4(0, n);
    // cvec5 = __riscv_vmv_v_x_i32m4(0, n);
  } else {
    // load 6 rows from C.
    int32_t *c_read = c;
    cvec0 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec1 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec2 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    cvec3 = __riscv_vle32_v_i32m4(c_read, n);
    // c_read += rsc;
    // cvec4 = __riscv_vle32_v_i32m4(c_read, n);
    // c_read += rsc;
    // cvec5 = __riscv_vle32_v_i32m4(c_read, n);
  }

  const int8_t *a0 = a + 0 * rsa1;
  const int8_t *a1 = a + 1 * rsa1;
  const int8_t *a2 = a + 2 * rsa1;
  const int8_t *a3 = a + 3 * rsa1;
  // const int8_t *a4 = a + 4 * rsa1;
  // const int8_t *a5 = a + 5 * rsa1;

  // process the first (k / 4) * 4 elements of the inner dimension
  if (k >= 8) {
    uint32_t a0_0;
    uint32_t a0_1;
    uint32_t a0_2;
    uint32_t a0_3;
    // uint32_t a0_4;
    // uint32_t a0_5;
    uint32_t a1_0;
    uint32_t a1_1;
    uint32_t a1_2;
    uint32_t a1_3;
    // uint32_t a1_4;
    // uint32_t a1_5;
    // a0_0 = *(uint32_t *)a0;
    // a0 += csa1;
    // a0_1 = *(uint32_t *)a1;
    // a1 += csa1;
    // a0_2 = *(uint32_t *)a2;
    // a2 += csa1;
    // a0_3 = *(uint32_t *)a3;
    // a3 += csa1;
    // a0_4 = *(uint32_t *)a4;
    // a4 += csa1;
    // a0_5 = *(uint32_t *)a5;
    // a5 += csa1;

    // load first B tile (4xn)
    // vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    // b += rsb1;

    while (k >= 8) {
      vint8m4_t bvec0;
      vint8m4_t bvec1;
      __asm__ volatile(
          // pre-load first B tile (4xn) of next iteration
          "vsetvli x0, %[n4], e8, m4, ta, ma\n"
          "vle8.v %[bvec0], (%[b])\n"
          "add %[b], %[b], %[rsb1]\n"

          // load second B tile (4xn)
          // "vsetvli x0, %[n4], e8, m4, ta, ma\n"
          "vle8.v %[bvec1], (%[b])\n"
          "add %[b], %[b], %[rsb1]\n"

          "lw %[a0_0], 0(%[a0])\n"
          "add %[a0], %[a0], %[csa1]\n"
          "lw %[a0_1], 0(%[a1])\n"
          "add %[a1], %[a1], %[csa1]\n"
          "lw %[a0_2], 0(%[a2])\n"
          "add %[a2], %[a2], %[csa1]\n"
          "lw %[a0_3], 0(%[a3])\n"
          "add %[a3], %[a3], %[csa1]\n"
          // "lw %[a0_4], 0(%[a4])\n"
          // "add %[a4], %[a4], %[csa1]\n"
          // "lw %[a0_5], 0(%[a5])\n"
          // "add %[a5], %[a5], %[csa1]\n"
          "lw %[a1_0], 0(%[a0])\n"
          "add %[a0], %[a0], %[csa1]\n"
          "lw %[a1_1], 0(%[a1])\n"
          "add %[a1], %[a1], %[csa1]\n"
          "lw %[a1_2], 0(%[a2])\n"
          "add %[a2], %[a2], %[csa1]\n"
          "lw %[a1_3], 0(%[a3])\n"
          "add %[a3], %[a3], %[csa1]\n"
          // "lw %[a1_4], 0(%[a4])\n"
          // "add %[a4], %[a4], %[csa1]\n"
          // "lw %[a1_5], 0(%[a5])\n"
          // "add %[a5], %[a5], %[csa1]\n"

          // compute first tile
          "vsetvli x0, %[n], e32, m4, ta, ma\n"
          "sf.vqdot.vx %[cvec0], %[bvec0], %[a0_0]\n"
          "sf.vqdot.vx %[cvec1], %[bvec0], %[a0_1]\n"
          "sf.vqdot.vx %[cvec2], %[bvec0], %[a0_2]\n"
          "sf.vqdot.vx %[cvec3], %[bvec0], %[a0_3]\n"


          // "vsetvli x0, %[n], e32, m4, ta, ma\n"
          // "sf.vqdot.vx %[cvec4], %[bvec0], %[a0_4]\n"
          // "sf.vqdot.vx %[cvec5], %[bvec0], %[a0_5]\n"

          // compute second tile
          "sf.vqdot.vx %[cvec0], %[bvec1], %[a1_0]\n"
          "sf.vqdot.vx %[cvec1], %[bvec1], %[a1_1]\n"
          "sf.vqdot.vx %[cvec2], %[bvec1], %[a1_2]\n"
          "sf.vqdot.vx %[cvec3], %[bvec1], %[a1_3]\n"
          // "sf.vqdot.vx %[cvec4], %[bvec1], %[a1_4]\n"
          // "sf.vqdot.vx %[cvec5], %[bvec1], %[a1_5]\n"

          : [cvec0] "+&vr"(cvec0), [cvec1] "+&vr"(cvec1), [cvec2] "+&vr"(cvec2),
            [cvec3] "+&vr"(cvec3),
            // [cvec4] "+&vr"(cvec4), [cvec5] "+&vr"(cvec5),
            [bvec0] "+&vr"(bvec0), [bvec1] "=&vr"(bvec1), [a0_0] "+&r"(a0_0),
            [a0_1] "+&r"(a0_1), [a0_2] "+&r"(a0_2), [a0_3] "+&r"(a0_3),
            // [a0_4] "+&r"(a0_4), [a0_5] "+&r"(a0_5),
            [a1_0] "=&r"(a1_0),
            [a1_1] "=&r"(a1_1), [a1_2] "=&r"(a1_2), [a1_3] "=&r"(a1_3),
            // [a1_4] "=&r"(a1_4), [a1_5] "=&r"(a1_5),
            [a0] "+&r"(a0),
            [a1] "+&r"(a1), [a2] "+&r"(a2), [a3] "+&r"(a3),
            // [a4] "+&r"(a4), [a5] "+&r"(a5),
            [b] "+&r"(b)
          : [csa1] "rI"(csa1 * sizeof(int8_t)),
            [rsb1] "rI"(rsb1 * sizeof(int8_t)), [n] "r"(n), [n4] "r"(4 * n)
          : "vl", "vtype", "memory");

      k -= 8;
    }
  }

    /*
    if (k >= 8) { // 8 <= k < 11
      vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
      b += rsb1;
      // load second B tile (4xn)
      vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
      b += rsb1;

      // compute first tile
      a0_0 = *(uint32_t *)a0;
      a0 += csa1;
      a0_1 = *(uint32_t *)a1;
      a0 += csa1;
      a0_2 = *(uint32_t *)a2;
      a2 += csa1;
      a0_3 = *(uint32_t *)a3;
      a3 += csa1;

      a1_0 = *(uint32_t *)a0;
      a0 += csa1;
      a1_1 = *(uint32_t *)a1;
      a1 += csa1;
      a1_2 = *(uint32_t *)a2;
      a2 += csa1;
      a1_3 = *(uint32_t *)a3;
      a3 += csa1;
      // cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec0, a0_4, n);
      // a1_4 = *(uint32_t *)a4;
      // a4 += csa1;
      // cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec0, a0_5, n);
      // a1_5 = *(uint32_t *)a5;
      // a5 += csa1;

      // compute second tile
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0_0, n);
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec0, a0_1, n);
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec0, a0_2, n);
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec0, a0_3, n);
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec1, a1_0, n);
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1_1, n);
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec1, a1_2, n);
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec1, a1_3, n);
      // cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec1, a1_4, n);
      // cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec1, a1_5, n);

      k -= 8;
    } else { // 4 <= k < 8
      // compute first tile
      // cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0_0, n);
      // cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec0, a0_1, n);
      // cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec0, a0_2, n);
      // cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec0, a0_3, n);
      // cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec0, a0_4, n);
      // cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec0, a0_5, n);

      k -= 4;
    }
  }

  // process the last k % 4 elements of the inner dimension
  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));

    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a0 & mask, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, *(uint32_t *)a1 & mask, n);
    cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, *(uint32_t *)a2 & mask, n);
    cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, *(uint32_t *)a3 & mask, n);
    // cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, *(uint32_t *)a4 & mask, n);
    // cvec5 = __riscv_sf_vqdot_vx_i32m4(cvec5, bvec, *(uint32_t *)a5 & mask, n);
  }
    */

  // write result back to c tile.
  __riscv_vse32_v_i32m4(c, cvec0, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec1, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec2, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, cvec3, n);
  // c += rsc;
  // __riscv_vse32_v_i32m4(c, cvec4, n);
  // c += rsc;
  // __riscv_vse32_v_i32m4(c, cvec5, n);
}

SKL_FUNC_PRIVATE void skl_mm_lt6xe32m4_i8i32_vqdotvx(
    size_t m, size_t n, size_t k, const int8_t *a, size_t rsa1, size_t csa1,
    const int8_t *b, size_t rsb1, int32_t *c, size_t rsc, bool accum) {
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
    int32_t *c_read = c + (m - 1) * rsc;
    switch (m) {
    case 5:
      cvec4 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_vle32_v_i32m4(c_read, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  const int8_t *a0 = a + 0 * rsa1;
  const int8_t *a1 = a + 1 * rsa1;
  const int8_t *a2 = a + 2 * rsa1;
  const int8_t *a3 = a + 3 * rsa1;
  const int8_t *a4 = a + 4 * rsa1;

  uint32_t a0_0;
  uint32_t a0_1;
  uint32_t a0_2;
  uint32_t a0_3;
  uint32_t a0_4;

  // compute 1 micro-tile at once.
  while (k >= 4) {
    // load one B tile (4xn).
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    switch (m) {
    case 5:
      memcpy(&a0_4, a4, 4);
      a4 += csa1;
      __attribute__((fallthrough));
    case 4:
      memcpy(&a0_3, a3, 4);
      a3 += csa1;
      __attribute__((fallthrough));
    case 3:
      memcpy(&a0_2, a2, 4);
      a2 += csa1;
      __attribute__((fallthrough));
    case 2:
      memcpy(&a0_1, a1, 4);
      a1 += csa1;
      __attribute__((fallthrough));
    case 1:
      memcpy(&a0_0, a0, 4);
      a0 += csa1;
      __attribute__((fallthrough));
    default:
      break;
    }

    // compute micro-tile.
    switch (m) {
    case 5:
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, a0_4, n);
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, a0_3, n);
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, a0_2, n);
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, a0_1, n);
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, a0_0, n);
      __attribute__((fallthrough));
    default:
      break;
    }

    k -= 4;
  }

  // handle k dimension of `a` is not the multiple of 4.
  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);

    switch (m) {
    case 5:
      memcpy(&a0_4, a4, k);
      __attribute__((fallthrough));
    case 4:
      memcpy(&a0_3, a3, k);
      __attribute__((fallthrough));
    case 3:
      memcpy(&a0_2, a2, k);
      __attribute__((fallthrough));
    case 2:
      memcpy(&a0_1, a1, k);
      __attribute__((fallthrough));
    case 1:
      memcpy(&a0_0, a0, k);
      __attribute__((fallthrough));
    default:
      break;
    }

    // compute micro-tile.
    switch (m) {
    case 5:
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, a0_4 & mask, n);
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, a0_3 & mask, n);
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, a0_2 & mask, n);
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, a0_1 & mask, n);
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, a0_0 & mask, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  // write result back to c tile.
  c += (m - 1) * rsc;
  switch (m) {
  case 5:
    __riscv_vse32_v_i32m4(c, cvec4, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 4:
    __riscv_vse32_v_i32m4(c, cvec3, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 3:
    __riscv_vse32_v_i32m4(c, cvec2, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 2:
    __riscv_vse32_v_i32m4(c, cvec1, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 1:
    __riscv_vse32_v_i32m4(c, cvec0, n);
    __attribute__((fallthrough));
  default:
    break;
  }
}

// a is 4-byte aligned and rsa1 is a multiple of 4
SKL_FUNC_PRIVATE void skl_mm_lt6xe32m4_aligned_i8i32_vqdotvx(
    size_t m, size_t n, size_t k, const int8_t *a, size_t rsa1, size_t csa1,
    const int8_t *b, size_t rsb1, int32_t *c, size_t rsc, bool accum) {
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
    int32_t *c_read = c + (m - 1) * rsc;
    switch (m) {
    case 5:
      cvec4 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_vle32_v_i32m4(c_read, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  const int8_t *a0 = a + 0 * rsa1;
  const int8_t *a1 = a + 1 * rsa1;
  const int8_t *a2 = a + 2 * rsa1;
  const int8_t *a3 = a + 3 * rsa1;
  const int8_t *a4 = a + 4 * rsa1;

  // compute 1 micro-tile at once.
  while (k >= 4) {
    // load one B tile (4xn).
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    // compute micro-tile.
    switch (m) {
    case 5:
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, *(uint32_t *)a4, n);
      a4 += csa1;
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, *(uint32_t *)a3, n);
      a3 += csa1;
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, *(uint32_t *)a2, n);
      a2 += csa1;
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, *(uint32_t *)a1, n);
      a1 += csa1;
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a0, n);
      a0 += csa1;
      __attribute__((fallthrough));
    default:
      break;
    }

    k -= 4;
  }

  // handle k dimension of `a` is not the multiple of 4.
  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);

    // compute micro-tile.
    switch (m) {
    case 5:
      cvec4 = __riscv_sf_vqdot_vx_i32m4(cvec4, bvec, *(uint32_t *)a4 & mask, n);
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_sf_vqdot_vx_i32m4(cvec3, bvec, *(uint32_t *)a3 & mask, n);
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_sf_vqdot_vx_i32m4(cvec2, bvec, *(uint32_t *)a2 & mask, n);
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec, *(uint32_t *)a1 & mask, n);
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a0 & mask, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  // write result back to c tile.
  c += (m - 1) * rsc;
  switch (m) {
  case 5:
    __riscv_vse32_v_i32m4(c, cvec4, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 4:
    __riscv_vse32_v_i32m4(c, cvec3, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 3:
    __riscv_vse32_v_i32m4(c, cvec2, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 2:
    __riscv_vse32_v_i32m4(c, cvec1, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 1:
    __riscv_vse32_v_i32m4(c, cvec0, n);
    __attribute__((fallthrough));
  default:
    break;
  }
}

SKL_FUNC_PRIVATE void skl_vm_1xe32m4_i8i32_vqdotvx(size_t n, size_t k,
                                                   const int8_t *a,
                                                   const int8_t *b, size_t rsb1,
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

  uint32_t a0;
  uint32_t a1;
  uint32_t a2;
  uint32_t a3;

  while (k >= 16) {
    // load 4 words from A.
    memcpy(&a0, a, 4);
    a += 4;
    memcpy(&a1, a, 4);
    a += 4;
    memcpy(&a2, a, 4);
    a += 4;
    memcpy(&a3, a, 4);
    a += 4;

    // load 4 B tiles (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec2 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec3 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1, n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec2, a2, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec3, a3, n);

    k -= 16;
  }

  while (k >= 8) {
    // load 2 words from A.
    memcpy(&a0, a, 4);
    a += 4;
    memcpy(&a1, a, 4);
    a += 4;

    // load 2 B tiles (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1, n);

    k -= 8;
  }

  while (k >= 4) {
    // load 1 word from A.
    memcpy(&a0, a, 4);
    a += 4;

    // load 1 B tile (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);

    k -= 4;
  }

  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    memcpy(&a0, a, k);
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, a0 & mask, n);
  }

  cvec0 = __riscv_vadd_vv_i32m4(cvec0, cvec1, n);
  cvec = __riscv_vadd_vv_i32m4(cvec, cvec0, n);
  __riscv_vse32_v_i32m4(c, cvec, n);
}

SKL_FUNC_PRIVATE void
skl_vm_1xe32m4_aligned_i8i32_vqdotvx(size_t n, size_t k, const int8_t *a,
                                     const int8_t *b, size_t rsb1, int32_t *c,
                                     bool accum) {
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

  while (k >= 16) {
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
    b += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec2 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec3 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1, n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec2, a2, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec3, a3, n);

    k -= 16;
  }

  while (k >= 8) {
    // load 2 words from A.
    uint32_t a0 = *(uint32_t *)a;
    a += 4;
    uint32_t a1 = *(uint32_t *)a;
    a += 4;

    // load 2 B tiles (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);
    cvec1 = __riscv_sf_vqdot_vx_i32m4(cvec1, bvec1, a1, n);

    k -= 8;
  }

  while (k >= 4) {
    // load 1 word from A.
    uint32_t a0 = *(uint32_t *)a;
    a += 4;

    // load 1 B tile (4xn).
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec0, a0, n);

    k -= 4;
  }

  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    cvec0 = __riscv_sf_vqdot_vx_i32m4(cvec0, bvec, *(uint32_t *)a & mask, n);
  }

  cvec0 = __riscv_vadd_vv_i32m4(cvec0, cvec1, n);
  cvec = __riscv_vadd_vv_i32m4(cvec, cvec0, n);
  __riscv_vse32_v_i32m4(c, cvec, n);
}

SKL_FUNC void skl_gemm_a1b01_i8_i8pc_i32_xsfvqdotq(size_t m, size_t n, size_t k,
                                                   const int8_t *a, size_t rsa,
                                                   const int8_t *b_pack,
                                                   size_t rsb1, int32_t *c,
                                                   size_t rsc, bool accum) {
  const size_t m0 = 6;
  const size_t n0 = __riscv_vsetvlmax_e32m4();
  const size_t k0 = 4;

  void (*skl_mm_6xe32m4_i8i32_vqdotvx_kernel)(
      size_t n, size_t k, const int8_t *a, size_t rsa1, size_t csa1,
      const int8_t *b, size_t rsb1, int32_t *c, size_t rsc, bool accum);
  void (*skl_mm_lt6xe32m4_i8i32_vqdotvx_kernel)(
      size_t m, size_t n, size_t k, const int8_t *a, size_t rsa1, size_t csa1,
      const int8_t *b, size_t rsb1, int32_t *c, size_t rsc, bool accum);
  void (*skl_vm_1xe32m4_i8i32_vqdotvx_kernel)(
      size_t n, size_t k, const int8_t *a, const int8_t *b, size_t rsb1,
      int32_t *c, bool accum);

  if ((uintptr_t)a % (k0 * sizeof(int8_t)) == 0 && rsa % k0 == 0) {
    skl_mm_6xe32m4_i8i32_vqdotvx_kernel = &skl_mm_6xe32m4_aligned_i8i32_vqdotvx;
    skl_mm_lt6xe32m4_i8i32_vqdotvx_kernel =
        &skl_mm_lt6xe32m4_aligned_i8i32_vqdotvx;
    skl_vm_1xe32m4_i8i32_vqdotvx_kernel = &skl_vm_1xe32m4_aligned_i8i32_vqdotvx;
  } else {
    skl_mm_6xe32m4_i8i32_vqdotvx_kernel = &skl_mm_6xe32m4_i8i32_vqdotvx;
    skl_mm_lt6xe32m4_i8i32_vqdotvx_kernel = &skl_mm_lt6xe32m4_i8i32_vqdotvx;
    skl_vm_1xe32m4_i8i32_vqdotvx_kernel = &skl_vm_1xe32m4_i8i32_vqdotvx;
  }

  if (m == 1) {
    // dispatch to GEMV kernel for better performance.
    for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
      int32_t *c_write = c + n_idx;
      const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
      const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
      (*skl_vm_1xe32m4_i8i32_vqdotvx_kernel)(n_tile, k, a, b_tile_ptr, rsb1,
                                             c_write, accum);
    }
  } else {
    size_t m_idx = 0;
    for (; m_idx + m0 - 1 < m; m_idx += m0) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a + m_idx * rsa;
      for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
        const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
        const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
        (*skl_mm_6xe32m4_i8i32_vqdotvx_kernel)(n_tile, k, a_tile_ptr, rsa, k0,
                                               b_tile_ptr, rsb1, c_write, rsc,
                                               accum);
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
        (*skl_mm_lt6xe32m4_i8i32_vqdotvx_kernel)(m_left, n_tile, k, a_tile_ptr,
                                                 rsa, k0, b_tile_ptr, rsb1,
                                                 c_write, rsc, accum);
        c_write += n_tile;
      }
    }
  }
}
