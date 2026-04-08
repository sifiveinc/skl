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

#if defined(__riscv_zihintntl)
#define NTL_P1 "ntl.p1\n"
#else
#define NTL_P1
#endif

// a_pack is 4-byte aligned and rsa1 and csa1 are multiples of 4
SKL_FUNC_PRIVATE void skl_gemm_6xm4_aligned_i8rcp_i8pc_i32_xsfvqdotq(
    size_t n, size_t k, int32_t alpha, const int8_t *a_pack, size_t rsa1,
    size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta, int32_t *c,
    size_t rsc) {
  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec1 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec2 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec3 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec4 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec5 = __riscv_vmv_v_x_i32m4(0, n);

  const int8_t *a0 = a_pack + 0 * rsa1;
  const int8_t *a1 = a_pack + 1 * rsa1;

  // process the first (k / 4) * 4 elements of the inner dimension
  if (k >= 4) {
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
    a0_0 = *(uint32_t *)a0;
    a0 += 2 * rsa1;
    a0_1 = *(uint32_t *)a1;
    a1 += 2 * rsa1;
    a0_2 = *(uint32_t *)a0;
    a0 += 2 * rsa1;
    a0_3 = *(uint32_t *)a1;
    a1 += 2 * rsa1;
    a0_4 = *(uint32_t *)a0;
    a0 += csa1 - 4 * rsa1;
    a0_5 = *(uint32_t *)a1;
    a1 += csa1 - 4 * rsa1;

    // load first B tile (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;

    while (k >= 12) {
      vint8m4_t bvec1;
      __asm__ volatile(
          // clang-format off
          NTL_P1
          "lw %[a1_0], 0(%[a0])\n"
          "add %[a0], %[a0], %[rsa1_0]\n"

          NTL_P1
          "lw %[a1_1], 0(%[a1])\n"
          "add %[a1], %[a1], %[rsa1_0]\n"

          NTL_P1
          "lw %[a1_2], 0(%[a0])\n"
          "add %[a0], %[a0], %[rsa1_0]\n"

          NTL_P1
          "lw %[a1_3], 0(%[a1])\n"
          "add %[a1], %[a1], %[rsa1_0]\n"

          NTL_P1
          "lw %[a1_4], 0(%[a0])\n"
          "add %[a0], %[a0], %[rsa1_1]\n"

          NTL_P1
          "lw %[a1_5], 0(%[a1])\n"
          "add %[a1], %[a1], %[rsa1_1]\n"

          // load second B tile (4xn)
          "vsetvli x0, %[n4], e8, m4, ta, ma\n"
          "vle8.v %[bvec1], (%[b_pack])\n"
          "add %[b_pack], %[b_pack], %[rsb1]\n"

          // compute first tile
          "vsetvli x0, %[n], e32, m4, ta, ma\n"
          "sf.vqdot.vx %[vec0], %[bvec0], %[a0_0]\n"
          "sf.vqdot.vx %[vec1], %[bvec0], %[a0_1]\n"
          "sf.vqdot.vx %[vec2], %[bvec0], %[a0_2]\n"
          "sf.vqdot.vx %[vec3], %[bvec0], %[a0_3]\n"
          "sf.vqdot.vx %[vec4], %[bvec0], %[a0_4]\n"
          "sf.vqdot.vx %[vec5], %[bvec0], %[a0_5]\n"

          NTL_P1
          "lw %[a0_0], 0(%[a0])\n"
          "add %[a0], %[a0], %[rsa1_0]\n"

          NTL_P1
          "lw %[a0_1], 0(%[a1])\n"
          "add %[a1], %[a1], %[rsa1_0]\n"

          NTL_P1
          "lw %[a0_2], 0(%[a0])\n"
          "add %[a0], %[a0], %[rsa1_0]\n"

          NTL_P1
          "lw %[a0_3], 0(%[a1])\n"
          "add %[a1], %[a1], %[rsa1_0]\n"

          NTL_P1
          "lw %[a0_4], 0(%[a0])\n"
          "add %[a0], %[a0], %[rsa1_1]\n"

          NTL_P1
          "lw %[a0_5], 0(%[a1])\n"
          "add %[a1], %[a1], %[rsa1_1]\n"

          // pre-load first B tile (4xn) of next iteration
          "vsetvli x0, %[n4], e8, m4, ta, ma\n"
          "vle8.v %[bvec0], (%[b_pack])\n"
          "add %[b_pack], %[b_pack], %[rsb1]\n"

          // compute second tile
          "vsetvli x0, %[n], e32, m4, ta, ma\n"
          "sf.vqdot.vx %[vec0], %[bvec1], %[a1_0]\n"
          "sf.vqdot.vx %[vec1], %[bvec1], %[a1_1]\n"
          "sf.vqdot.vx %[vec2], %[bvec1], %[a1_2]\n"
          "sf.vqdot.vx %[vec3], %[bvec1], %[a1_3]\n"
          "sf.vqdot.vx %[vec4], %[bvec1], %[a1_4]\n"
          "sf.vqdot.vx %[vec5], %[bvec1], %[a1_5]\n"
          // clang-format on
          : [vec0] "+&vr"(vec0), [vec1] "+&vr"(vec1), [vec2] "+&vr"(vec2),
            [vec3] "+&vr"(vec3), [vec4] "+&vr"(vec4), [vec5] "+&vr"(vec5),
            [bvec0] "+&vr"(bvec0), [bvec1] "=&vr"(bvec1), [a0_0] "+&r"(a0_0),
            [a0_1] "+&r"(a0_1), [a0_2] "+&r"(a0_2), [a0_3] "+&r"(a0_3),
            [a0_4] "+&r"(a0_4), [a0_5] "+&r"(a0_5), [a1_0] "=&r"(a1_0),
            [a1_1] "=&r"(a1_1), [a1_2] "=&r"(a1_2), [a1_3] "=&r"(a1_3),
            [a1_4] "=&r"(a1_4), [a1_5] "=&r"(a1_5), [a0] "+&r"(a0),
            [a1] "+&r"(a1), [b_pack] "+&r"(b_pack)
          : [rsa1_0] "rI"(2 * rsa1 * sizeof(int8_t)),
            [rsa1_1] "rI"(csa1 - 4 * rsa1 * sizeof(int8_t)),
            [rsb1] "rI"(rsb1 * sizeof(int8_t)), [n] "r"(n), [n4] "r"(4 * n)
          : "vl", "vtype", "memory");

      k -= 8;
    }

    if (k >= 8) { // 8 <= k < 11
      // compute first tile
      vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0_0, n);
      a1_0 = *(uint32_t *)a0;
      a0 += 2 * rsa1;
      vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec0, a0_1, n);
      a1_1 = *(uint32_t *)a1;
      a1 += 2 * rsa1;

      // load second B tile (4xn)
      vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
      b_pack += rsb1;

      vec2 = __riscv_sf_vqdot_vx_i32m4(vec2, bvec0, a0_2, n);
      a1_2 = *(uint32_t *)a0;
      a0 += 2 * rsa1;
      vec3 = __riscv_sf_vqdot_vx_i32m4(vec3, bvec0, a0_3, n);
      a1_3 = *(uint32_t *)a1;
      a1 += 2 * rsa1;
      vec4 = __riscv_sf_vqdot_vx_i32m4(vec4, bvec0, a0_4, n);
      a1_4 = *(uint32_t *)a0;
      a0 += csa1 - 4 * rsa1;
      vec5 = __riscv_sf_vqdot_vx_i32m4(vec5, bvec0, a0_5, n);
      a1_5 = *(uint32_t *)a1;
      a1 += csa1 - 4 * rsa1;

      // compute second tile
      vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec1, a1_0, n);
      vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec1, a1_1, n);
      vec2 = __riscv_sf_vqdot_vx_i32m4(vec2, bvec1, a1_2, n);
      vec3 = __riscv_sf_vqdot_vx_i32m4(vec3, bvec1, a1_3, n);
      vec4 = __riscv_sf_vqdot_vx_i32m4(vec4, bvec1, a1_4, n);
      vec5 = __riscv_sf_vqdot_vx_i32m4(vec5, bvec1, a1_5, n);

      k -= 8;
    } else { // 4 <= k < 8
      // compute first tile
      vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0_0, n);
      vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec0, a0_1, n);
      vec2 = __riscv_sf_vqdot_vx_i32m4(vec2, bvec0, a0_2, n);
      vec3 = __riscv_sf_vqdot_vx_i32m4(vec3, bvec0, a0_3, n);
      vec4 = __riscv_sf_vqdot_vx_i32m4(vec4, bvec0, a0_4, n);
      vec5 = __riscv_sf_vqdot_vx_i32m4(vec5, bvec0, a0_5, n);

      k -= 4;
    }
  }

  // process the last k % 4 elements of the inner dimension
  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));

    vint8m4_t bvec = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec, *(uint32_t *)a0 & mask, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec, *(uint32_t *)a1 & mask, n);
    vec2 = __riscv_sf_vqdot_vx_i32m4(vec2, bvec,
                                     *(uint32_t *)(a0 + 2 * rsa1) & mask, n);
    vec3 = __riscv_sf_vqdot_vx_i32m4(vec3, bvec,
                                     *(uint32_t *)(a1 + 2 * rsa1) & mask, n);
    vec4 = __riscv_sf_vqdot_vx_i32m4(vec4, bvec,
                                     *(uint32_t *)(a0 + 4 * rsa1) & mask, n);
    vec5 = __riscv_sf_vqdot_vx_i32m4(vec5, bvec,
                                     *(uint32_t *)(a1 + 4 * rsa1) & mask, n);
  }

  if (beta != 0) {
    if (alpha != 1) {
      vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
      vec1 = __riscv_vmul_vx_i32m4(vec1, alpha, n);
      vec2 = __riscv_vmul_vx_i32m4(vec2, alpha, n);
      vec3 = __riscv_vmul_vx_i32m4(vec3, alpha, n);
      vec4 = __riscv_vmul_vx_i32m4(vec4, alpha, n);
      vec5 = __riscv_vmul_vx_i32m4(vec5, alpha, n);
    }
    int32_t *c_read = c;
    vint32m4_t cvec0 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    vec0 = __riscv_vmacc_vx_i32m4(vec0, beta, cvec0, n);

    vint32m4_t cvec1 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    vec1 = __riscv_vmacc_vx_i32m4(vec1, beta, cvec1, n);

    vint32m4_t cvec2 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    vec2 = __riscv_vmacc_vx_i32m4(vec2, beta, cvec2, n);

    vint32m4_t cvec3 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    vec3 = __riscv_vmacc_vx_i32m4(vec3, beta, cvec3, n);

    vint32m4_t cvec4 = __riscv_vle32_v_i32m4(c_read, n);
    c_read += rsc;
    vec4 = __riscv_vmacc_vx_i32m4(vec4, beta, cvec4, n);

    vint32m4_t cvec5 = __riscv_vle32_v_i32m4(c_read, n);
    vec5 = __riscv_vmacc_vx_i32m4(vec5, beta, cvec5, n);
  } else {
    if (alpha != 1) {
      vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
      vec1 = __riscv_vmul_vx_i32m4(vec1, alpha, n);
      vec2 = __riscv_vmul_vx_i32m4(vec2, alpha, n);
      vec3 = __riscv_vmul_vx_i32m4(vec3, alpha, n);
      vec4 = __riscv_vmul_vx_i32m4(vec4, alpha, n);
      vec5 = __riscv_vmul_vx_i32m4(vec5, alpha, n);
    }
  }

  // write result back to C tile
  __riscv_vse32_v_i32m4(c, vec0, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec1, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec2, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec3, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec4, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec5, n);
}

// a_pack is 4-byte aligned and rsa1 and csa1 are multiples of 4
SKL_FUNC_PRIVATE void skl_gemm_lt6xm4_aligned_i8rcp_i8pc_i32_xsfvqdotq(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a_pack,
    size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta,
    int32_t *c, size_t rsc) {
  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec1 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec2 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec3 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec4 = __riscv_vmv_v_x_i32m4(0, n);

  const int8_t *a0 = a_pack + 0 * rsa1;
  const int8_t *a1 = a_pack + 1 * rsa1;
  const int8_t *a2 = a_pack + 2 * rsa1;
  const int8_t *a3 = a_pack + 3 * rsa1;
  const int8_t *a4 = a_pack + 4 * rsa1;

  // compute 1 micro-tile at once
  while (k >= 4) {
    // load one B tile (4xn)
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;

    // compute micro-tile
    switch (m) {
    case 5:
      vec4 = __riscv_sf_vqdot_vx_i32m4(vec4, bvec, *(uint32_t *)a4, n);
      a4 += csa1;
      __attribute__((fallthrough));
    case 4:
      vec3 = __riscv_sf_vqdot_vx_i32m4(vec3, bvec, *(uint32_t *)a3, n);
      a3 += csa1;
      __attribute__((fallthrough));
    case 3:
      vec2 = __riscv_sf_vqdot_vx_i32m4(vec2, bvec, *(uint32_t *)a2, n);
      a2 += csa1;
      __attribute__((fallthrough));
    case 2:
      vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec, *(uint32_t *)a1, n);
      a1 += csa1;
      __attribute__((fallthrough));
    case 1:
      vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec, *(uint32_t *)a0, n);
      a0 += csa1;
      __attribute__((fallthrough));
    default:
      break;
    }

    k -= 4;
  }

  // handle k not a multiple of 4
  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b_pack, 4 * n);

    // compute micro-tile
    switch (m) {
    case 5:
      vec4 = __riscv_sf_vqdot_vx_i32m4(vec4, bvec, *(uint32_t *)a4 & mask, n);
      __attribute__((fallthrough));
    case 4:
      vec3 = __riscv_sf_vqdot_vx_i32m4(vec3, bvec, *(uint32_t *)a3 & mask, n);
      __attribute__((fallthrough));
    case 3:
      vec2 = __riscv_sf_vqdot_vx_i32m4(vec2, bvec, *(uint32_t *)a2 & mask, n);
      __attribute__((fallthrough));
    case 2:
      vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec, *(uint32_t *)a1 & mask, n);
      __attribute__((fallthrough));
    case 1:
      vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec, *(uint32_t *)a0 & mask, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  }

  if (beta != 0) {
    if (alpha != 1) {
      switch (m) {
      case 5:
        vec4 = __riscv_vmul_vx_i32m4(vec4, alpha, n);
        __attribute__((fallthrough));
      case 4:
        vec3 = __riscv_vmul_vx_i32m4(vec3, alpha, n);
        __attribute__((fallthrough));
      case 3:
        vec2 = __riscv_vmul_vx_i32m4(vec2, alpha, n);
        __attribute__((fallthrough));
      case 2:
        vec1 = __riscv_vmul_vx_i32m4(vec1, alpha, n);
        __attribute__((fallthrough));
      case 1:
        vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
        __attribute__((fallthrough));
      default:
        break;
      }
    }

    int32_t *c_read = c + (m - 1) * rsc;
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vint32m4_t cvec0 = __riscv_vundefined_i32m4();
    vint32m4_t cvec1 = __riscv_vundefined_i32m4();
    vint32m4_t cvec2 = __riscv_vundefined_i32m4();
    vint32m4_t cvec3 = __riscv_vundefined_i32m4();
    vint32m4_t cvec4 = __riscv_vundefined_i32m4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    switch (m) {
    case 5:
      cvec4 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      vec4 = __riscv_vmacc_vx_i32m4(vec4, beta, cvec4, n);
      __attribute__((fallthrough));
    case 4:
      cvec3 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      vec3 = __riscv_vmacc_vx_i32m4(vec3, beta, cvec3, n);
      __attribute__((fallthrough));
    case 3:
      cvec2 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      vec2 = __riscv_vmacc_vx_i32m4(vec2, beta, cvec2, n);
      __attribute__((fallthrough));
    case 2:
      cvec1 = __riscv_vle32_v_i32m4(c_read, n);
      c_read -= rsc;
      vec1 = __riscv_vmacc_vx_i32m4(vec1, beta, cvec1, n);
      __attribute__((fallthrough));
    case 1:
      cvec0 = __riscv_vle32_v_i32m4(c_read, n);
      vec0 = __riscv_vmacc_vx_i32m4(vec0, beta, cvec0, n);
      __attribute__((fallthrough));
    default:
      break;
    }
  } else {
    if (alpha != 1) {
      switch (m) {
      case 5:
        vec4 = __riscv_vmul_vx_i32m4(vec4, alpha, n);
        __attribute__((fallthrough));
      case 4:
        vec3 = __riscv_vmul_vx_i32m4(vec3, alpha, n);
        __attribute__((fallthrough));
      case 3:
        vec2 = __riscv_vmul_vx_i32m4(vec2, alpha, n);
        __attribute__((fallthrough));
      case 2:
        vec1 = __riscv_vmul_vx_i32m4(vec1, alpha, n);
        __attribute__((fallthrough));
      case 1:
        vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
        __attribute__((fallthrough));
      default:
        break;
      }
    }
  }

  // write result back to C tile
  c += (m - 1) * rsc;
  switch (m) {
  case 5:
    __riscv_vse32_v_i32m4(c, vec4, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 4:
    __riscv_vse32_v_i32m4(c, vec3, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 3:
    __riscv_vse32_v_i32m4(c, vec2, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 2:
    __riscv_vse32_v_i32m4(c, vec1, n);
    c -= rsc;
    __attribute__((fallthrough));
  case 1:
    __riscv_vse32_v_i32m4(c, vec0, n);
    __attribute__((fallthrough));
  default:
    break;
  }
}

// a_pack need not be 4-byte-aligned nor csa1 a multiple of 4
SKL_FUNC_PRIVATE void skl_gemm_1xm4_unaligned_i8rcp_i8pc_i32_xsfvqdotq(
    size_t n, size_t k, int32_t alpha, const int8_t *a_pack, size_t csa1,
    const int8_t *b_pack, size_t rsb1, int32_t beta, int32_t *c) {
  // init 1 int32m4_t accumulator
  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);

  uint32_t a0;

  while (k >= 4) {
    // load 1 word from A
    skl_memcpy(&a0, a_pack, 4);
    a_pack += csa1;

    // load 1 B tile (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);

    k -= 4;
  }

  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    skl_memcpy(&a0, a_pack, k);
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0 & mask, n);
  }

  if (beta != 0) {
    if (alpha != 1) {
      vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
    }
    int32_t *c_read = c;
    vint32m4_t cvec = __riscv_vle32_v_i32m4(c_read, n);
    vec0 = __riscv_vmacc_vx_i32m4(vec0, beta, cvec, n);
  } else {
    if (alpha != 1) {
      vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
    }
  }

  __riscv_vse32_v_i32m4(c, vec0, n);
}

// a_pack is 4-byte aligned and csa1 is a multiple of 4
SKL_FUNC_PRIVATE void skl_gemm_1xm4_aligned_i8rcp_i8pc_i32_xsfvqdotq(
    size_t n, size_t k, int32_t alpha, const int8_t *a_pack, size_t csa1,
    const int8_t *b_pack, size_t rsb1, int32_t beta, int32_t *c) {
  // init 2 int32m4_t accumulators
  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec1 = __riscv_vmv_v_x_i32m4(0, n);

  while (k >= 16) {
    // load 4 words from A
    uint32_t a0 = *(uint32_t *)a_pack;
    a_pack += csa1;
    uint32_t a1 = *(uint32_t *)a_pack;
    a_pack += csa1;
    uint32_t a2 = *(uint32_t *)a_pack;
    a_pack += csa1;
    uint32_t a3 = *(uint32_t *)a_pack;
    a_pack += csa1;

    // load 4 B tiles (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;
    vint8m4_t bvec2 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;
    vint8m4_t bvec3 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec1, a1, n);
    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec2, a2, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec3, a3, n);

    k -= 16;
  }

  while (k >= 8) {
    // load 2 words from A
    uint32_t a0 = *(uint32_t *)a_pack;
    a_pack += csa1;
    uint32_t a1 = *(uint32_t *)a_pack;
    a_pack += csa1;

    // load 2 B tiles (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec1, a1, n);

    k -= 8;
  }

  while (k >= 4) {
    // load 1 word from A
    uint32_t a0 = *(uint32_t *)a_pack;
    a_pack += csa1;

    // load 1 B tile (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    b_pack += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);

    k -= 4;
  }

  if (k) {
    uint32_t mask = 0xFFFFFFFF >> (8 * (4 - k));
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b_pack, 4 * n);
    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec, *(uint32_t *)a_pack & mask, n);
  }

  vec0 = __riscv_vadd_vv_i32m4(vec0, vec1, n);

  if (beta != 0) {
    if (alpha != 1) {
      vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
    }
    int32_t *c_read = c;
    vint32m4_t cvec = __riscv_vle32_v_i32m4(c_read, n);
    vec0 = __riscv_vmacc_vx_i32m4(vec0, beta, cvec, n);
  } else {
    if (alpha != 1) {
      vec0 = __riscv_vmul_vx_i32m4(vec0, alpha, n);
    }
  }

  __riscv_vse32_v_i32m4(c, vec0, n);
}

SKL_FUNC_PRIVATE void skl_gemm_unaligned_i8rcp_i8pc_i32_xsfvqdotq(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a_pack,
    size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta,
    int32_t *c, size_t rsc) {
  const size_t m0 = 1;
  const size_t n0 = __riscv_vsetvlmax_e32m4();
  const size_t k0 = 4;

  size_t m_idx = 0;
  for (; m_idx + m0 - 1 < m; m_idx += m0) {
    int32_t *c_write = c + m_idx * rsc;
    const int8_t *a_tile_ptr = a_pack + m_idx * rsa1;
    for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
      const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
      const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
      skl_gemm_1xm4_unaligned_i8rcp_i8pc_i32_xsfvqdotq(
          n_tile, k, alpha, a_tile_ptr, csa1, b_tile_ptr, rsb1, beta, c_write);
      c_write += n_tile;
    }
  }
}

SKL_FUNC_PRIVATE void skl_gemm_aligned_i8rcp_i8pc_i32_xsfvqdotq(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a_pack,
    size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta,
    int32_t *c, size_t rsc) {
  const size_t m0 = 6;
  const size_t n0 = __riscv_vsetvlmax_e32m4();
  const size_t k0 = 4;

  if (m == 1) {
    // dispatch to GEMV kernel for better performance
    for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
      int32_t *c_write = c + n_idx;
      const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
      const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
      skl_gemm_1xm4_aligned_i8rcp_i8pc_i32_xsfvqdotq(
          n_tile, k, alpha, a_pack, csa1, b_tile_ptr, rsb1, beta, c_write);
    }
  } else {
    size_t m_idx = 0;
    for (; m_idx + m0 - 1 < m; m_idx += m0) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a_pack + m_idx * rsa1;
      for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
        const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
        const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
        skl_gemm_6xm4_aligned_i8rcp_i8pc_i32_xsfvqdotq(
            n_tile, k, alpha, a_tile_ptr, rsa1, csa1, b_tile_ptr, rsb1, beta,
            c_write, rsc);
        c_write += n_tile;
      }
    }

    size_t m_left = m - m_idx;
    if (m_left) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a_pack + m_idx * rsa1;
      for (size_t n_idx = 0; n_idx < n; n_idx += n0) {
        const int8_t *b_tile_ptr = b_pack + k0 * n_idx;
        const size_t n_tile = n0 <= n - n_idx ? n0 : n - n_idx;
        skl_gemm_lt6xm4_aligned_i8rcp_i8pc_i32_xsfvqdotq(
            m_left, n_tile, k, alpha, a_tile_ptr, rsa1, csa1, b_tile_ptr, rsb1,
            beta, c_write, rsc);
        c_write += n_tile;
      }
    }
  }
}

SKL_FUNC void skl_gemm_i8rcp_i8pc_i32_xsfvqdotq(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a_pack,
    size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta,
    int32_t *c, size_t rsc) {
  const size_t k0 = 4;

  if ((uintptr_t)a_pack % (k0 * sizeof(int8_t)) == 0 && rsa1 % k0 == 0 &&
      csa1 % k0 == 0) {
    skl_gemm_aligned_i8rcp_i8pc_i32_xsfvqdotq(m, n, k, alpha, a_pack, rsa1,
                                              csa1, b_pack, rsb1, beta, c, rsc);
  } else {
    skl_gemm_unaligned_i8rcp_i8pc_i32_xsfvqdotq(
        m, n, k, alpha, a_pack, rsa1, csa1, b_pack, rsb1, beta, c, rsc);
  }
}
