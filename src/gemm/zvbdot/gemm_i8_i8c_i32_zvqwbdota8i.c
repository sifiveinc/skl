// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvqwbdota8i)
#error This source file requires compiler support for the Zvqwbdota8i extension.
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC void skl_gemm_a1b0_2x8_i8_i8c_i32_zvqwbdota8i(
    size_t k, const int8_t *a, size_t rsa, const int8_t *b, size_t csb,
    int32_t *c, size_t rsc) {
  vint32m8_t cvec0 = __riscv_vmv_v_x_i32m8(0, 8);
  vint32m8_t cvec1 = __riscv_vmv_v_x_i32m8(0, 8);

  size_t avl = k;
  vint8m1_t avec;

  const int8_t *a0 = a;
  const int8_t *b0 = b;
  size_t vl = 0;
  __asm__ volatile("0:\n"
                   "mv %[a0], %[a]\n"
                   "mv %[b0], %[b]\n"

                   "vsetvli %[vl], %[avl], e8alt, m1, ta, ma\n"
                   "vle8.v %[avec], (%[a0])\n"
                   "add %[a0], %[a0], %[rsa]\n"
                   "vle8.v v0, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v1, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v2, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v3, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v4, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v5, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v6, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "vle8.v v7, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"

                   "vqwbdotas.vv %[cvec0], v0, %[avec], 0\n"
                   "vle8.v %[avec], (%[a0])\n"
                   "vqwbdotas.vv %[cvec1], v0, %[avec], 0\n"

                   "add %[a], %[a], %[vl]\n"
                   "add %[b], %[b], %[vl]\n"
                   "sub %[avl], %[avl], %[vl]\n"

                   "bnez %[avl], 0b\n"
                   : [avec] "=&vr"(avec), [cvec0] "+&vr"(cvec0),
                     [cvec1] "+&vr"(cvec1), [a0] "=&r"(a0), [b0] "=&r"(b0),
                     [a] "+&r"(a), [b] "+&r"(b), [vl] "=&r"(vl)
                   : [rsa] "rI"(rsa),
                     [csb] "rI"(csb), [avl] "r"(avl)
                   : "vl", "vtype", "memory", "v0", "v1", "v2", "v3", "v4",
                     "v5", "v6", "v7");

  __riscv_vse32_v_i32m8(c, cvec0, 8);
  c += rsc;
  __riscv_vse32_v_i32m8(c, cvec1, 8);
}

/*
SKL_FUNC void skl_gemm_a1b0_vlen512_15x16_i8_i8c_i32_zvqwbdota8i(
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
  // vint32m1_t cvec14 = __riscv_vmv_v_x_i32m1(0, 16);

  size_t avl = k;
  while (avl) {
    vint8m1_t avec;

    const int8_t *a0 = a;
    const int8_t *b0 = b;
    size_t vl = 0;
    __asm__ volatile(
        "vsetvli %[vl], %[avl], e8alt, m1, ta, ma\n"
        "vle8.v v0, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v1, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v2, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v3, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v4, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v5, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v6, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v7, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v8, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v9, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v10, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v11, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v12, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v13, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v14, (%[b0])\n"
        "add %[b0], %[b0], %[csb]\n"
        "vle8.v v15, (%[b0])\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec0], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec0], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec1], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec1], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec2], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec2], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec3], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec3], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec4], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec4], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec5], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec5], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec6], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec6], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec7], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec7], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec8], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec8], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec9], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec9], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec10], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec10], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec11], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec11], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec12], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec12], v8, %[avec], 1\n"

        "vle8.v %[avec], (%[a0])\n"
        "add %[a0], %[a0], %[rsa]\n"
        "vqwbdotas.vv %[cvec13], v0, %[avec], 0\n"
        "vqwbdotas.vv %[cvec13], v8, %[avec], 1\n"

        // "vle8.v %[avec], (%[a0])\n"
        // "add %[a0], %[a0], %[rsa]\n"
        // "vqwbdotas.vv %[cvec14], v0, %[avec], 1\n"
        // "vqwbdotas.vv %[cvec14], v8, %[avec], 0\n"
        : [avec] "=&vr"(avec), [cvec0] "+&vr"(cvec0), [cvec1] "+&vr"(cvec1),
          [cvec2] "+&vr"(cvec2), [cvec3] "+&vr"(cvec3), [cvec4] "+&vr"(cvec4),
          [cvec5] "+&vr"(cvec5), [cvec6] "+&vr"(cvec6), [cvec7] "+&vr"(cvec7),
          [cvec8] "+&vr"(cvec8), [cvec9] "+&vr"(cvec9), [cvec10] "+&vr"(cvec10),
          [cvec11] "+&vr"(cvec11), [cvec12] "+&vr"(cvec12),
          [cvec13] "+&vr"(cvec13), [cvec14] "+&vr"(cvec14),[a0] "+&r"(a0),
          [b0] "+&r"(b0), [vl] "=&r"(vl)
        : [rsa] "rI"(rsa * sizeof(int8_t)), [csb] "rI"(csb * sizeof(int8_t)),
          [avl] "r"(avl)
        : "vl", "vtype", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
          "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
    a += vl;
    b += vl;
    avl -= vl;
  }

  // write result back to C tile
  __riscv_vse32_v_i32m1(c, cvec0, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec1, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec2, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec3, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec4, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec5, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec6, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec7, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec8, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec9, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec10, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec11, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec12, 16);
  c += rsc;
  __riscv_vse32_v_i32m1(c, cvec13, 16);
  // c += rsc;
  // __riscv_vse32_v_i32m1(c, cvec14, 16);
}
*/
