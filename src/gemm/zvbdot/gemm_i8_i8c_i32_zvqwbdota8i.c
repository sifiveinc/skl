// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvqwbdota8i)
#error This source file requires compiler support for the Zvqwbdota8i extension.
#endif

#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC_PRIVATE void skl_gemm_a1b0_vlen512_15x16_i8_i8c_i32_zvqwbdota8i(
    size_t k, const int8_t *a, size_t rsa, const int8_t *b, size_t csb,
    int32_t *c, size_t rsc) {
  vint32m1_t cvec0 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec1 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec2 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec3 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec4 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec5 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec6 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec7 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec8 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec9 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec10 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec11 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec12 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec13 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec14 = __riscv_vmv_v_x_i32m1(0, 16);
  vint32m1_t cvec15 = __riscv_vmv_v_x_i32m1(0, 16);

  size_t avl = k;
  while (avl) {
    vint8m1_t avec;
    vint8m1_t bvec0;
    vint8m1_t bvec1;
    vint8m1_t bvec2;
    vint8m1_t bvec3;
    vint8m1_t bvec4;
    vint8m1_t bvec5;
    vint8m1_t bvec6;
    vint8m1_t bvec7;
    vint8m1_t bvec8;
    vint8m1_t bvec9;
    vint8m1_t bvec10;
    vint8m1_t bvec11;
    vint8m1_t bvec12;
    vint8m1_t bvec13;
    vint8m1_t bvec14;
    vint8m1_t bvec15;

    const int8_t *a0 = a;
    const int8_t *b0 = b;
    __asm__ volatile(
        "vsetvli %[vl], %[avl], e8alt, m1, ta, ma\n"
        "vle8.v %[bvec0], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec1], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec2], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec3], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec4], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec5], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec6], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec7], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec8], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec9], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec10], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec11], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec12], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec13], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec14], %[b0]\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v %[bvec15], %[b0]\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec0], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec0], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec1], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec1], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec2], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec2], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec3], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec3], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec4], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec4], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec5], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec5], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec6], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec6], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec7], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec7], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec8], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec8], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec9], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec9], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec10], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec10], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec11], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec11], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec12], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec12], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec13], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec13], %[bvec8], %[avec], 8\n"

        "vle8.v %[avec], %[a0]\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec14], %[bvec0], %[avec], 0\n"
        "vqwbdotas.vv %[cvec14], %[bvec8], %[avec], 8\n"

        : [bvec0] "=&vr"(bvec0), [bvec1] "=&vr"(bvec1), [bvec2] "=&vr"(bvec2),
          [bvec3] "=&vr"(bvec3), [bvec4] "=&vr"(bvec4), [bvec5] "=&vr"(bvec5),
          [bvec6] "=&vr"(bvec6), [bvec7] "=&vr"(bvec7), [bvec8] "=&vr"(bvec8),
          [bvec9] "=&vr"(bvec9), [bvec10] "=&vr"(bvec10),
          [bvec11] "=&vr"(bvec11), [bvec12] "=&vr"(bvec12),
          [bvec13] "=&vr"(bvec13), [bvec14] "=&vr"(bvec14),
          [bvec15] "=&vr"(bvec15), [avec] "=&vr"(avec), [cvec0] "+&vr"(cvec0),
          [cvec1] "+&vr"(cvec1), [cvec2] "+&vr"(cvec2), [cvec3] "+&vr"(cvec3),
          [cvec4] "+&vr"(cvec4), [cvec5] "+&vr"(cvec5), [cvec6] "+&vr"(cvec6),
          [cvec7] "+&vr"(cvec7), [cvec8] "+&vr"(cvec8), [cvec9] "+&vr"(cvec9),
          [cvec10] "+&vr"(cvec10), [cvec11] "+&vr"(cvec11),
          [cvec12] "+&vr"(cvec12), [cvec13] "+&vr"(cvec13),
          [cvec14] "+&vr"(cvec14), [vl] "=&r"(vl), [avl] "+&r"(avl),
          [a0] "+&r"(a0), [a1] "+&r"(a1), [b0] "+&r"(b0)
        : [rsa] "rI"(rsa * sizeof(int8_t)), [rsb] "rI"(rsb * sizeof(int8_t))
        : "vl", "vtype", "memory");
    a += vl;
    b += vl;
    avl -= vl;
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
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec6, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec7, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec8, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec9, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec10, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec11, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec12, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec13, n);
  c += rsc;
  __riscv_vse32_v_i32m4(c, vec14, n);
}

// a is 4-byte aligned and rsa1 and csa1 are multiples of 4
SKL_FUNC_PRIVATE void skl_gemm_lt6xm4_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
    size_t m, size_t n, size_t k1, int32_t alpha, const int8_t *a, size_t rsa1,
    size_t csa1, const int8_t *b, size_t rsb1, int32_t beta, int32_t *c,
    size_t rsc) {
  if (m == 0 || n == 0) {
    return;
  }

  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec1 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec2 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec3 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec4 = __riscv_vmv_v_x_i32m4(0, n);

  const int8_t *a0 = a + 0 * rsa1;
  const int8_t *a1 = a + 1 * rsa1;
  const int8_t *a2 = a + 2 * rsa1;
  const int8_t *a3 = a + 3 * rsa1;
  const int8_t *a4 = a + 4 * rsa1;

  // compute 1 micro-tile at once
  while (k1) {
    // load one B tile (4xn)
    vint8m4_t bvec = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

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

    k1 -= 1;
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

// a need not be 4-byte-aligned nor csa1 a multiple of 4
SKL_FUNC_PRIVATE void skl_gemm_1xm4_unaligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
    size_t n, size_t k1, int32_t alpha, const int8_t *a, size_t csa1,
    const int8_t *b, size_t rsb1, int32_t beta, int32_t *c) {
  if (n == 0) {
    return;
  }

  // init 1 int32m4_t accumulator
  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);

  uint32_t a0;

  while (k1) {
    // load 1 word from A
    skl_memcpy(&a0, a, 4);
    a += csa1;

    // load 1 B tile (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);

    k1 -= 1;
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

// a is 4-byte aligned and csa1 is a multiple of 4
SKL_FUNC_PRIVATE void skl_gemm_1xm4_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
    size_t n, size_t k1, int32_t alpha, const int8_t *a, size_t csa1,
    const int8_t *b, size_t rsb1, int32_t beta, int32_t *c) {
  if (n == 0) {
    return;
  }

  // init 2 int32m4_t accumulators
  vint32m4_t vec0 = __riscv_vmv_v_x_i32m4(0, n);
  vint32m4_t vec1 = __riscv_vmv_v_x_i32m4(0, n);

  while (k1 >= 4) {
    // load 4 words from A
    uint32_t a0 = *(uint32_t *)a;
    a += csa1;
    uint32_t a1 = *(uint32_t *)a;
    a += csa1;
    uint32_t a2 = *(uint32_t *)a;
    a += csa1;
    uint32_t a3 = *(uint32_t *)a;
    a += csa1;

    // load 4 B tiles (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec2 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec3 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec1, a1, n);
    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec2, a2, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec3, a3, n);

    k1 -= 4;
  }

  while (k1 >= 2) {
    // load 2 words from A
    uint32_t a0 = *(uint32_t *)a;
    a += csa1;
    uint32_t a1 = *(uint32_t *)a;
    a += csa1;

    // load 2 B tiles (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;
    vint8m4_t bvec1 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);
    vec1 = __riscv_sf_vqdot_vx_i32m4(vec1, bvec1, a1, n);

    k1 -= 2;
  }

  while (k1) {
    // load 1 word from A
    uint32_t a0 = *(uint32_t *)a;
    a += csa1;

    // load 1 B tile (4xn)
    vint8m4_t bvec0 = __riscv_vle8_v_i8m4(b, 4 * n);
    b += rsb1;

    vec0 = __riscv_sf_vqdot_vx_i32m4(vec0, bvec0, a0, n);

    k1 -= 1;
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

SKL_FUNC_PRIVATE void skl_gemm_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
    size_t m, size_t n, size_t k1, int32_t alpha, const int8_t *a, size_t rsa1,
    size_t csa1, const int8_t *b, size_t rsb1, int32_t beta, int32_t *c,
    size_t rsc) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t m0 = 6;
  const size_t k0 = 4;

  if (m == 1) {
    // dispatch to GEMV kernel for better performance
    int32_t *c_write = c;
    const int8_t *a_tile_ptr = a;
    const int8_t *b_tile_ptr = b;
    size_t n_avl = n;
    while (n_avl) {
      size_t vl = __riscv_vsetvl_e32m4(n_avl);
      skl_gemm_1xm4_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
          vl, k1, alpha, a_tile_ptr, csa1, b_tile_ptr, rsb1, beta, c_write);
      b_tile_ptr += k0 * vl;
      c_write += vl;
      n_avl -= vl;
    }
  } else {
    size_t m_idx = 0;
    for (; m_idx + m0 - 1 < m; m_idx += m0) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a + m_idx * rsa1;
      const int8_t *b_tile_ptr = b;
      size_t n_avl = n;
      while (n_avl) {
        size_t vl = __riscv_vsetvl_e32m4(n_avl);
        skl_gemm_6xm4_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
            vl, k1, alpha, a_tile_ptr, rsa1, csa1, b_tile_ptr, rsb1, beta,
            c_write, rsc);
        b_tile_ptr += k0 * vl;
        c_write += vl;
        n_avl -= vl;
      }
    }

    size_t m_left = m - m_idx;
    if (m_left) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a + m_idx * rsa1;
      const int8_t *b_tile_ptr = b;
      size_t n_avl = n;
      while (n_avl) {
        size_t vl = __riscv_vsetvl_e32m4(n_avl);
        skl_gemm_lt6xm4_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
            m_left, vl, k1, alpha, a_tile_ptr, rsa1, csa1, b_tile_ptr, rsb1,
            beta, c_write, rsc);
        b_tile_ptr += k0 * vl;
        c_write += vl;
        n_avl -= vl;
      }
    }
  }
}

SKL_FUNC void skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
    size_t m, size_t n, size_t k1, int32_t alpha, const int8_t *a, size_t rsa1,
    size_t csa1, const int8_t *b, size_t rsb1, int32_t beta, int32_t *c,
    size_t rsc) {
  if (m == 0 || n == 0) {
    return;
  }

  const size_t k0 = 4;

  if ((uintptr_t)a % (k0 * sizeof(int8_t)) == 0 && rsa1 % k0 == 0 &&
      csa1 % k0 == 0) {
    skl_gemm_aligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
        m, n, k1, alpha, a, rsa1, csa1, b, rsb1, beta, c, rsc);
  } else {
    size_t m_idx = 0;
    for (; m_idx < m; ++m_idx) {
      int32_t *c_write = c + m_idx * rsc;
      const int8_t *a_tile_ptr = a + m_idx * rsa1;
      const int8_t *b_tile_ptr = b;
      size_t n_avl = n;
      while (n_avl) {
        size_t vl = __riscv_vsetvl_e32m4(n_avl);
        skl_gemm_1xm4_unaligned_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
            vl, k1, alpha, a_tile_ptr, csa1, b_tile_ptr, rsb1, beta, c_write);
        b_tile_ptr += k0 * vl;
        c_write += vl;
        n_avl -= vl;
      }
    }
  }
}
