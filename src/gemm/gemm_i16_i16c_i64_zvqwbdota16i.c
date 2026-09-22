// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zvqwbdota16i)
#error This source file requires compiler support for the Zvqwbdota16i extension.
#endif

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "skl-common.h"

SKL_FUNC_PRIVATE void skl_gemm_2xle8_i16_i16c_i64_zvqwbdota16i(
    size_t n, size_t k, int64_t alpha, const int16_t *a, size_t rsa,
    const int16_t *b, size_t csb, int64_t beta, int64_t *c, size_t rsc) {
  if (n == 0) {
    return;
  }

  vint64m8_t vec0 = __riscv_vundefined_i64m8();
  vint64m8_t vec1 = __riscv_vundefined_i64m8();
  vint16m1_t avec0 = __riscv_vundefined_i16m1();
  vint16m1_t avec1 = __riscv_vundefined_i16m1();

  const int16_t *a0 = a;
  const int16_t *b0 = b;
  uint8_t mask = 0xFFU >> (8 - n);
  size_t avl = k;
  size_t vl = 0;
  __asm__ volatile(
      "vsetivli x0, 8, e64, m8, ta, ma\n"
      "vmv.v.i %[vec0], 0\n"
      "vmv.v.i %[vec1], 0\n"

      "beqz %[avl], 2f\n"

      "vsetivli x0, 1, e8, m1, ta, ma\n"
      "vmv.s.x v0, %[mask]\n"

      "0:\n"
      "mv %[a0], %[a]\n"
      "mv %[b0], %[b]\n"

      "vsetvli %[vl], %[avl], e16alt, m1, ta, ma\n"
      "vle16.v %[avec0], (%[a0])\n"
      "add %[a0], %[a0], %[rsa]\n"
      "vle16.v %[avec1], (%[a0])\n"
      "sh1add %[a], %[vl], %[a]\n"
      "sh1add %[b], %[vl], %[b]\n"

      "vle16.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i1], 1f\n"
      "vle16.v v9, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i2], 1f\n"
      "vle16.v v10, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i3], 1f\n"
      "vle16.v v11, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i4], 1f\n"
      "vle16.v v12, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i5], 1f\n"
      "vle16.v v13, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i6], 1f\n"
      "vle16.v v14, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i7], 1f\n"
      "vle16.v v15, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"

      "1:\n"
      "vqwbdotas.vv %[vec0], v8, %[avec0], 0, v0.t\n"
      "vqwbdotas.vv %[vec1], v8, %[avec1], 0, v0.t\n"

      "sub %[avl], %[avl], %[vl]\n"

      "bnez %[avl], 0b\n"
      "2:\n"
      : [avec0] "=&vr"(avec0), [avec1] "=&vr"(avec1), [vec0] "=&vr"(vec0),
        [vec1] "=&vr"(vec1), [a0] "=&r"(a0), [b0] "=&r"(b0), [a] "+&r"(a),
        [b] "+&r"(b), [vl] "=&r"(vl), [avl] "+&r"(avl)
      : [mask] "r"(mask), [rsa] "rI"(rsa * sizeof(*a)),
        [csb] "rI"(csb * sizeof(*b)), [n] "r"(n), [i1] "r"(1), [i2] "r"(2),
        [i3] "r"(3), [i4] "r"(4), [i5] "r"(5), [i6] "r"(6), [i7] "r"(7)
      : "vl", "vtype", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15");

  if (alpha != 1) {
    vec0 = __riscv_vmul_vx_i64m8(vec0, alpha, n);
    vec1 = __riscv_vmul_vx_i64m8(vec1, alpha, n);
  }
  if (beta != 0) {
    int64_t *c0 = c;
    vint64m8_t cvec0 = __riscv_vle64_v_i64m8(c0, n);
    c0 += rsc;
    vint64m8_t cvec1 = __riscv_vle64_v_i64m8(c0, n);

    vec0 = __riscv_vmacc_vx_i64m8(vec0, beta, cvec0, n);
    vec1 = __riscv_vmacc_vx_i64m8(vec1, beta, cvec1, n);
  }

  __riscv_vse64_v_i64m8(c, vec0, n);
  c += rsc;
  __riscv_vse64_v_i64m8(c, vec1, n);
}

SKL_FUNC_PRIVATE void
skl_gemm_1xle8_i16_i16c_i64_zvqwbdota16i(size_t n, size_t k, int64_t alpha,
                                         const int16_t *a, const int16_t *b,
                                         size_t csb, int64_t beta, int64_t *c) {
  if (n == 0) {
    return;
  }

  vint64m8_t vec0 = __riscv_vundefined_i64m8();
  vint16m1_t avec = __riscv_vundefined_i16m1();

  const int16_t *b0 = NULL;
  uint8_t mask = 0xFFU >> (8 - n);
  size_t avl = k;
  size_t vl = 0;
  __asm__ volatile(
      "vsetivli x0, 8, e64, m8, ta, ma\n"
      "vmv.v.i %[vec0], 0\n"

      "beqz %[avl], 2f\n"

      "vsetivli x0, 1, e8, m1, ta, ma\n"
      "vmv.s.x v0, %[mask]\n"

      "0:\n"
      "mv %[b0], %[b]\n"

      "vsetvli %[vl], %[avl], e16alt, m1, ta, ma\n"
      "vle16.v %[avec], (%[a])\n"
      "sh1add %[a], %[vl], %[a]\n"
      "sh1add %[b], %[vl], %[b]\n"

      "vle16.v v8, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i1], 1f\n"
      "vle16.v v9, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i2], 1f\n"
      "vle16.v v10, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i3], 1f\n"
      "vle16.v v11, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i4], 1f\n"
      "vle16.v v12, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i5], 1f\n"
      "vle16.v v13, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i6], 1f\n"
      "vle16.v v14, (%[b0])\n"
      "add %[b0], %[b0], %[csb]\n"
      "beq %[n], %[i7], 1f\n"
      "vle16.v v15, (%[b0])\n"

      "1:\n"
      "vqwbdotas.vv %[vec0], v8, %[avec], 0, v0.t\n"

      "sub %[avl], %[avl], %[vl]\n"

      "bnez %[avl], 0b\n"
      "2:\n"
      : [avec] "=&vr"(avec), [vec0] "=&vr"(vec0), [b0] "=&r"(b0), [a] "+&r"(a),
        [b] "+&r"(b), [vl] "=&r"(vl), [avl] "+&r"(avl)
      : [mask] "r"(mask), [csb] "rI"(csb * sizeof(*b)), [n] "r"(n), [i1] "r"(1),
        [i2] "r"(2), [i3] "r"(3), [i4] "r"(4), [i5] "r"(5), [i6] "r"(6),
        [i7] "r"(7)
      : "vl", "vtype", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15");

  if (alpha != 1) {
    vec0 = __riscv_vmul_vx_i64m8(vec0, alpha, n);
  }
  if (beta != 0) {
    int64_t *c0 = c;
    vint64m8_t cvec0 = __riscv_vle64_v_i64m8(c0, n);
    vec0 = __riscv_vmacc_vx_i64m8(vec0, beta, cvec0, n);
  }

  __riscv_vse64_v_i64m8(c, vec0, n);
}

SKL_FUNC void skl_gemm_i16_i16c_i64_zvqwbdota16i(
    size_t m, size_t n, size_t k, int64_t alpha, const int16_t *a, size_t rsa,
    const int16_t *b, size_t csb, int64_t beta, int64_t *c, size_t rsc) {
  if (m == 0 || n == 0) {
    return;
  }

  size_t i = 0;
  for (; i + 2 <= m; i += 2) {
    size_t n_vl = 0;
    for (size_t j = 0; j < n; j += n_vl) {
      n_vl = n - j >= 8 ? 8 : n - j;
      skl_gemm_2xle8_i16_i16c_i64_zvqwbdota16i(n_vl, k, alpha, a + i * rsa, rsa,
                                               b + j * csb, csb, beta,
                                               c + i * rsc + j, rsc);
    }
  }

  if (i < m) {
    size_t n_vl = 0;
    for (size_t j = 0; j < n; j += n_vl) {
      n_vl = n - j >= 8 ? 8 : n - j;
      skl_gemm_1xle8_i16_i16c_i64_zvqwbdota16i(
          n_vl, k, alpha, a + i * rsa, b + j * csb, csb, beta, c + i * rsc + j);
    }
  }
}
