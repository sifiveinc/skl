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

SKL_FUNC_PRIVATE void
skl_gemm_a1b0_2xle8_i8_i8c_i32_zvqwbdota8i(size_t n, size_t k, const int8_t *a,
                                           size_t rsa, const int8_t *b,
                                           size_t csb, int32_t *c, size_t rsc) {
  if (n == 0) {
    return;
  }
  vint32m8_t cvec0 = __riscv_vmv_v_x_i32m8(0, 8);
  vint32m8_t cvec1 = __riscv_vmv_v_x_i32m8(0, 8);
  vint8m1_t avec = __riscv_vundefined_i8m1();

  const int8_t *a0 = a;
  const int8_t *b0 = b;
  size_t avl = k;
  size_t vl = 0;
  uint8_t mask = 0xFFU >> (8 - n);
  __asm__ volatile(
      "beqz %[avl], 2f\n"
      "vsetvli x0, x0, e8, m1, ta, ma\n"
      "vmv.s.x v0, %[mask]\n"

      "0:\n"
      "mv %[a0], %[a]\n"
      "mv %[b0], %[b]\n"

      "vsetvli %[vl], %[avl], e8alt, m1, ta, ma\n"
      "vle8.v %[avec], (%[a0])\n"
      "add %[a0], %[a0], %[rsa]\n"

      "vle8.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i1], 1f\n"
      "vle8.v v9, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i2], 1f\n"
      "vle8.v v10, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i3], 1f\n"
      "vle8.v v11, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i4], 1f\n"
      "vle8.v v12, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i5], 1f\n"
      "vle8.v v13, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i6], 1f\n"
      "vle8.v v14, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i7], 1f\n"
      "vle8.v v15, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"

      "1:\n"
      "vqwbdotas.vv %[cvec0], v8, %[avec], 0, v0.t\n"
      "vle8.v %[avec], (%[a0])\n"
      "vqwbdotas.vv %[cvec1], v8, %[avec], 0, v0.t\n"

      "add %[a], %[a], %[vl]\n"
      "add %[b], %[b], %[vl]\n"
      "sub %[avl], %[avl], %[vl]\n"

      "bnez %[avl], 0b\n"
      "2:\n"
      : [avec] "=&vr"(avec), [cvec0] "+&vr"(cvec0), [cvec1] "+&vr"(cvec1),
        [a0] "=&r"(a0), [b0] "=&r"(b0), [a] "+&r"(a), [b] "+&r"(b),
        [vl] "=&r"(vl), [avl] "+&r"(avl)
      : [mask] "r"(mask), [rsa] "rI"(rsa), [csb] "rI"(csb), [n] "r"(n),
        [i1] "r"(1), [i2] "r"(2), [i3] "r"(3), [i4] "r"(4), [i5] "r"(5),
        [i6] "r"(6), [i7] "r"(7)
      : "vl", "vtype", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15");

  __riscv_vse32_v_i32m8(c, cvec0, n);
  c += rsc;
  __riscv_vse32_v_i32m8(c, cvec1, n);
}

SKL_FUNC_PRIVATE void
skl_gemm_a1b0_1xle8_i8_i8c_i32_zvqwbdota8i(size_t n, size_t k, const int8_t *a,
                                           size_t rsa, const int8_t *b,
                                           size_t csb, int32_t *c) {
  if (n == 0) {
    return;
  }
  vint32m8_t cvec0 = __riscv_vmv_v_x_i32m8(0, 8);
  vint8m1_t avec = __riscv_vundefined_i8m1();

  const int8_t *a0 = a;
  const int8_t *b0 = b;
  size_t avl = k;
  size_t vl = 0;
  uint8_t mask = 0xFFU >> (8 - n);
  __asm__ volatile("beqz %[avl], 2f\n"

                   "vsetvli x0, x0, e8, m1, ta, ma\n"
                   "vmv.s.x v0, %[mask]\n"

                   "0:\n"
                   "mv %[a0], %[a]\n"
                   "mv %[b0], %[b]\n"

                   "vsetvli %[vl], %[avl], e8alt, m1, ta, ma\n"
                   "vle8.v %[avec], (%[a0])\n"

                   "vle8.v v8, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i1], 1f\n"
                   "vle8.v v9, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i2], 1f\n"
                   "vle8.v v10, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i3], 1f\n"
                   "vle8.v v11, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i4], 1f\n"
                   "vle8.v v12, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i5], 1f\n"
                   "vle8.v v13, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i6], 1f\n"
                   "vle8.v v14, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"
                   "beq %[n], %[i7], 1f\n"
                   "vle8.v v15, (%[b0])\n"
                   "add %[b0], %[b0], %[csb]\n"

                   "1:\n"
                   "vqwbdotas.vv %[cvec0], v8, %[avec], 0, v0.t\n"

                   "add %[a], %[a], %[vl]\n"
                   "add %[b], %[b], %[vl]\n"
                   "sub %[avl], %[avl], %[vl]\n"

                   "bnez %[avl], 0b\n"
                   "2:\n"
                   : [avec] "=&vr"(avec), [cvec0] "+&vr"(cvec0), [a0] "=&r"(a0),
                     [b0] "=&r"(b0), [a] "+&r"(a), [b] "+&r"(b), [vl] "=&r"(vl),
                     [avl] "+&r"(avl)
                   : [mask] "r"(mask), [rsa] "rI"(rsa), [csb] "rI"(csb),
                     [n] "r"(n), [i1] "r"(1), [i2] "r"(2), [i3] "r"(3),
                     [i4] "r"(4), [i5] "r"(5), [i6] "r"(6), [i7] "r"(7)
                   : "vl", "vtype", "memory", "v0", "v8", "v9", "v10", "v11",
                     "v12", "v13", "v14", "v15");

  __riscv_vse32_v_i32m8(c, cvec0, n);
}

SKL_FUNC void skl_gemm_a1b0_i8_i8c_i32_zvqwbdota8i(size_t m, size_t n, size_t k,
                                                   const int8_t *a, size_t rsa,
                                                   const int8_t *b, size_t csb,
                                                   int32_t *c, size_t rsc) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t i = 0;
  for (; m - i >= 2; i += 2) {
    size_t n_vl = 0;
    for (size_t j = 0; j < n; j += n_vl) {
      n_vl = n - j >= 8 ? 8 : n - j;
      skl_gemm_a1b0_2xle8_i8_i8c_i32_zvqwbdota8i(
          n_vl, k, a + i * rsa, rsa, b + j * csb, csb, c + i * rsc + j, rsc);
    }
  }

  if (i < m) {
    size_t n_vl = 0;
    for (size_t j = 0; j < n; j += n_vl) {
      n_vl = n - j >= 8 ? 8 : n - j;
      skl_gemm_a1b0_1xle8_i8_i8c_i32_zvqwbdota8i(
          n_vl, k, a + i * rsa, rsa, b + j * csb, csb, c + i * rsc + j);
    }
  }
}
