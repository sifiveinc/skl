// Copyright (c) 2025 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC_PRIVATE vfloat32m8_t skl_leftexp_xsfvfexpa_f32m8(vfloat32m8_t x,
                                                          size_t vl);

SKL_FUNC void skl_logistic_4u_f32_xsfvfexpa(float *out, const float *in,
                                            size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);

    /* Approximate exp(-|vx|) */
    const vfloat32m8_t n =
        skl_leftexp_xsfvfexpa_f32m8(__riscv_vfsgnj_vf_f32m8(vx, -1, vl), vl);

    /* Approximate 1 / (1 + exp(-|vx|)) */
    const vfloat32m8_t d = __riscv_vfadd_vf_f32m8(n, 1, vl);
    // With __riscv_xsfvfexpa, our leftexp(x) is slighty less accurate.  To
    // retain some additional accuracy, calculate n / d using a more accurate
    // division algorithm.  This requires an additional two instructions, but
    // lowers maximum error by about 0.5ulp.  Since we know that n is in [0;1]
    // and d is in [1;2], we can skip the usual scaling and refinement checks
    // that are required in `div`.
    vfloat32m8_t r = __riscv_vfrec7_v_f32m8(d, vl);
    vfloat32m8_t q = __riscv_vfmul_vv_f32m8(n, r, vl);
    /* Refine r */
    const vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1, vl);
    vfloat32m8_t t = __riscv_vfnmsac_vv_f32m8(one, d, r, vl); // 1 - d * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - d * r)
    t = __riscv_vfnmsac_vv_f32m8(one, d, r, vl); // 1 - d * r
    r = __riscv_vfmadd_vv_f32m8(r, t, r, vl);    // r + r * (1 - d * r)
    /* Refine q */
    t = __riscv_vfnmsub_vv_f32m8(d, q, n, vl);
    q = __riscv_vfmacc_vv_f32m8(q, r, t, vl);

    const vbool4_t m = __riscv_vmfgt_vf_f32m8_b4(vx, 0, vl);
    vfloat32m8_t vy = __riscv_vfrsub_vf_f32m8_mu(m, q, q, 1, vl);
    __riscv_vse32_v_f32m8(out + i, vy, vl);
  }
}

/**
 * @brief Approximate the exponential function on vector of f32 floating-point
 * values with a 1-ULP error bound in [-inf; 0].
 */
SKL_FUNC_PRIVATE vfloat32m8_t skl_leftexp_xsfvfexpa_f32m8(vfloat32m8_t x,
                                                          size_t vl) {
  /* 0. Clamp inputs to lower bound */
  vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(x, x, vl); // Propagate NaN inputs
  x = __riscv_vfmax_vf_f32m8_mu(nn, x, x, -0x1.9fe36ap6f, vl);

  /* 1. Reduction */
  const float r_ln2 = 0x1.715476p0f; // 1/log(2)
  const float Q = 0x1.004b80p17f;    // 2^(24-6-1) + B + p
  const vfloat32m8_t v =
      __riscv_vfmacc_vf_f32m8(__riscv_vfmv_v_f_f32m8(Q, vl), r_ln2, x, vl);
  const vfloat32m8_t f = __riscv_sf_vfexpa_v_f32m8(v, vl);
  const vfloat32m8_t z = __riscv_vfsub_vf_f32m8(v, Q, vl);

  const float l2u = 0x1.63p-1f; // round(log(2),24-6-7,RN)
  const float l2l = -0x1.bd0106p-13f;
  const vfloat32m8_t s = // x - log(2) z
      __riscv_vfnmsac_vf_f32m8(__riscv_vfnmsac_vf_f32m8(x, l2u, z, vl), l2l, z,
                               vl);

  /* 2. Approximation */
  const float a3 = 0x1.5558f4p-3f;
  const float a2 = 0x1.fff92cp-2f;
  vfloat32m8_t p =
      __riscv_vfmacc_vf_f32m8(__riscv_vfmv_v_f_f32m8(a2, vl), a3, s, vl);
  p = __riscv_vfmadd_vv_f32m8(p, __riscv_vfmul_vv_f32m8(s, s, vl), s, vl);

  /* 3. Reconstruction */
  const vfloat32m8_t e = __riscv_vfmadd_vv_f32m8(p, f, f, vl);

  /* 4. Re-scaling */
  return __riscv_vfmul_vf_f32m8(e, 0x1p-24f, vl); // UF happens here
}
