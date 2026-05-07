// Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>

SKL_FUNC void skl_gelu_p9_f32_zve32f(float *dst, const float *src, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t x = __riscv_vle32_v_f32m8(src + i, vl);
    // 0. Clamp
    x = __riscv_vfmin_vf_f32m8(x, +0x1.7efc0ep+1f, vl);
    x = __riscv_vfmax_vf_f32m8(x, -0x1.7efc0ep+1f, vl);
    // 1. Evaluate polynomial
    vfloat32m8_t X = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(-0x1.6f1244p-11f, vl);
    vfloat32m8_t p = __riscv_vfmacc_vf_f32m8(c, 0x1.f78348p-17f, X, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.a193dp-7f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-0x1.ebac44p-4f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.96410cp-1f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    vfloat32m8_t e = __riscv_vfmul_vv_f32m8(p, x, vl);
    // 2. Reconstruct
    vfloat32m8_t h = __riscv_vle32_v_f32m8(src + i, vl);
    h = __riscv_vfmul_vf_f32m8(h, 0.5f, vl);
    vfloat32m8_t g = __riscv_vfmadd_vv_f32m8(e, h, h, vl);
    // 3. Store
    __riscv_vse32_v_f32m8(dst + i, g, vl);
  }
}

SKL_FUNC void skl_gelu_p13_f32_zve32f(float *dst, const float *src, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t x = __riscv_vle32_v_f32m8(src + i, vl);
    // 0. Clamp
    x = __riscv_vfmin_vf_f32m8(x, +0x1.ce62f2p1f, vl);
    x = __riscv_vfmax_vf_f32m8(x, -0x1.ce62f2p1f, vl);
    // 1. Evaluate polynomial
    vfloat32m8_t X = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(-0x1.13d444p-18f, vl);
    vfloat32m8_t p = __riscv_vfmacc_vf_f32m8(c, 0x1.0b318ap-24f, X, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.e3fa7p-14f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-0x1.dffbap-10f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.30c73ep-6f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-0x1.0d9532p-3f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.98648ap-1f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    vfloat32m8_t e = __riscv_vfmul_vv_f32m8(p, x, vl);
    // 2. Reconstruct
    vfloat32m8_t h = __riscv_vle32_v_f32m8(src + i, vl);
    h = __riscv_vfmul_vf_f32m8(h, 0.5f, vl);
    vfloat32m8_t g = __riscv_vfmadd_vv_f32m8(e, h, h, vl);
    // 3. Store
    __riscv_vse32_v_f32m8(dst + i, g, vl);
  }
}

SKL_FUNC void skl_gelu_p17_f32_zve32f(float *dst, const float *src, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t x = __riscv_vle32_v_f32m8(src + i, vl);
    // 0. Clamp
    x = __riscv_vfmin_vf_f32m8(x, +0x1.01fe1cp2f, vl);
    x = __riscv_vfmax_vf_f32m8(x, -0x1.01fe1cp2f, vl);
    // 1. Evaluate polynomial
    vfloat32m8_t X = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(-0x1.043dcep-26f, vl);
    vfloat32m8_t p = __riscv_vfmacc_vf_f32m8(c, 0x1.869362p-33f, X, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.35572p-21f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-0x1.b78176p-17f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.a84a6cp-13f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-0x1.2c3fc6p-9f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.44b128p-6f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-0x1.102e06p-3f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m8(0x1.98832ep-1f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, X, c, vl);
    vfloat32m8_t e = __riscv_vfmul_vv_f32m8(p, x, vl);
    // 2. Reconstruct
    vfloat32m8_t h = __riscv_vle32_v_f32m8(src + i, vl);
    h = __riscv_vfmul_vf_f32m8(h, 0.5f, vl);
    vfloat32m8_t g = __riscv_vfmadd_vv_f32m8(e, h, h, vl);
    // 3. Store
    __riscv_vse32_v_f32m8(dst + i, g, vl);
  }
}

SKL_FUNC void skl_gelu_rat_f32_zve32f(float *dst, const float *src, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    vfloat32m4_t x = __riscv_vle32_v_f32m4(src + i, vl);
    // 0. Clamp
    x = __riscv_vfmin_vf_f32m4(x, +0x1.4a7c2ap2f, vl);
    x = __riscv_vfmax_vf_f32m4(x, -0x1.4a7c2ap2f, vl);
    // 1. Evaluate rational numerator
    vfloat32m4_t X = __riscv_vfmul_vv_f32m4(x, x, vl);
    vfloat32m4_t c = __riscv_vfmv_v_f_f32m4(0x1.a71d38p-17f, vl);
    vfloat32m4_t p = __riscv_vfmacc_vf_f32m4(c, 0x1.89b8fp-25f, X, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.59bf88p-12f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.30f4b2p-7f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.12516ap-4f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, X, c, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.988452p-1f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, X, c, vl);
    p = __riscv_vfmul_vv_f32m4(p, x, vl);
    // 2. Evaluate rational denominator
    c = __riscv_vfmv_v_f_f32m4(0x1.2f1ccep-14f, vl);
    vfloat32m4_t q = __riscv_vfmacc_vf_f32m4(c, 0x1.41d272p-20f, X, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.e82fc8p-10f, vl);
    q = __riscv_vfmadd_vv_f32m4(q, X, c, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.d1d45cp-6f, vl);
    q = __riscv_vfmadd_vv_f32m4(q, X, c, vl);
    c = __riscv_vfmv_v_f_f32m4(0x1.009e36p-2f, vl);
    q = __riscv_vfmadd_vv_f32m4(q, X, c, vl);
    c = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    q = __riscv_vfmadd_vv_f32m4(q, X, c, vl);
    // 3. Divide
    vfloat32m4_t r = __riscv_vfrec7_v_f32m4(q, vl);
    vfloat32m4_t t = __riscv_vfnmsub_vv_f32m4(q, r, c, vl);
    r = __riscv_vfmadd_vv_f32m4(r, t, r, vl);
    t = __riscv_vfnmsub_vv_f32m4(q, r, c, vl);
    r = __riscv_vfmadd_vv_f32m4(r, t, r, vl);
    vfloat32m4_t e = __riscv_vfmul_vv_f32m4(p, r, vl);
    // 4. Reconstruct
    vfloat32m4_t h = __riscv_vle32_v_f32m4(src + i, vl);
    h = __riscv_vfmul_vf_f32m4(h, 0.5f, vl);
    vfloat32m4_t g = __riscv_vfmadd_vv_f32m4(e, h, h, vl);
    // 5. Store
    __riscv_vse32_v_f32m4(dst + i, g, vl);
  }
}
