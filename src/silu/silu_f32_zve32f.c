// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

SKL_FUNC_PRIVATE vfloat32m8_t skl_leftexp_zve32f_f32m8(vfloat32m8_t x,
                                                       size_t vl);

SKL_FUNC void skl_silu_52u_f32_zve32f(float *out, const float *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);

    /* Approximate exp(-|vx|) */
    const vfloat32m8_t n =
        skl_leftexp_zve32f_f32m8(__riscv_vfsgnj_vf_f32m8(vx, -1, vl), vl);

    /* Approximate 1 / (1 + exp(-|vx|)) */
    const vfloat32m8_t d = __riscv_vfadd_vf_f32m8(n, 1, vl);
    const vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1, vl);
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);
    vfloat32m8_t t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl);
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);
    t = __riscv_vfnmsub_vv_f32m8(d, r, one, vl); // 1 - x * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - x * r)

    /* Calculate quotient */
    const vbool4_t m = __riscv_vmflt_vf_f32m8_b4(vx, 0, vl);
    vfloat32m8_t vy = __riscv_vfmul_vv_f32m8_mu(m, r, r, n, vl);

    /* Multiply */
    vy = __riscv_vfmul_vv_f32m8(vy, vx, vl);

    __riscv_vse32_v_f32m8(out + i, vy, vl);
  }
}

/**
 * Approximate the exponential function on a vector of f32 floating-point values
 * with a 1-ULP error bound in [-inf; 0].
 *
 * @note NaNs are not propagated.
 */
SKL_FUNC_PRIVATE vfloat32m8_t skl_leftexp_zve32f_f32m8(vfloat32m8_t x,
                                                       size_t vl) {
  /* 0. Clamp inputs to lower bound */
  x = __riscv_vfmax_vf_f32m8(x, -0x1.9fe36ap6f, vl);

  /* 1. Reduction */
  const float r_ln2 = 0x1.715476p0f; // single(1/log(2));
  const vfloat32m8_t v = __riscv_vfmul_vf_f32m8(x, r_ln2, vl);
  const vint16m4_t q = __riscv_vfncvt_x_f_w_i16m4(v, vl);
  const vfloat32m8_t z = __riscv_vfwcvt_f_x_v_f32m8(q, vl);

  const float l2u = 0x1.62e4p-1f;    // round(log(2), 24-8, RN);
  const float l2l = 0x1.7f7d1cp-20f; // round(log(2) - l2u, single, RN);
  vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(x, l2u, z, vl); // s = x - l2u * z
  const vfloat32m8_t c5 = __riscv_vfmv_v_f_f32m8(0x1.123bccp-7f, vl);
  s = __riscv_vfnmsac_vf_f32m8(s, l2l, z, vl); // s = s - l2l * z

  /* 2. Approximation: e^s. */
  const float c6 = 0x1.6850e4p-10f;
  vfloat32m8_t u = __riscv_vfmacc_vf_f32m8(c5, c6, s, vl);
  vfloat32m8_t c4 = __riscv_vfmv_v_f_f32m8(0x1.555b98p-5f, vl);
  u = __riscv_vfmadd_vv_f32m8(u, s, c4, vl);
  vfloat32m8_t c3 = __riscv_vfmv_v_f_f32m8(0x1.55548ep-3f, vl);
  u = __riscv_vfmadd_vv_f32m8(u, s, c3, vl);
  vfloat32m8_t c2 = __riscv_vfmv_v_f_f32m8(0x1.fffff8p-2f, vl);
  u = __riscv_vfmadd_vv_f32m8(u, s, c2, vl);
  vfloat32m8_t c01 = __riscv_vfmv_v_f_f32m8(1.0f, vl);
  u = __riscv_vfmadd_vv_f32m8(u, s, c01, vl);
  u = __riscv_vfmadd_vv_f32m8(u, s, c01, vl);

  /* 3. Reconstruction: compute e = u*2^(q+24) */
  vint32m8_t i = __riscv_vreinterpret_v_f32m8_i32m8(u);
  const int16_t exp_ = 24 << 8;
  vint16m4_t vexp = __riscv_vlse16_v_i16m4(&exp_, 0, vl);
  vexp = __riscv_vmadd_vx_i16m4(q, 1 << 8, vexp, vl);
  i = __riscv_vwmaccus_vx_i32m8(i, 1 << 15, vexp, vl);
  vfloat32m8_t e = __riscv_vreinterpret_v_i32m8_f32m8(i);

  /* 4. Re-scaling */
  return __riscv_vfmul_vf_f32m8(e, 0x1p-24f, vl); // UF happens here
}
