// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexpa) || !defined(__riscv_zvfh)
#error This file requires the Xsfvfexpa and Zvfh extensions
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC_PRIVATE vfloat16m8_t skl_leftexp_xsfvfexpa_zvfh_f16m8(vfloat16m8_t x,
                                                               size_t vl);

SKL_FUNC void skl_sigmoid_f16_xsfvfexpa_zvfh(_Float16 *out, _Float16 beta,
                                             const _Float16 *x,
                                             const _Float16 *y,
                                             const _Float16 *up, _Float16 delta,
                                             size_t n) {
  if (!x)
    return;

  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e16m8(n - i);
    /* 0. Load and scale by beta */
    vfloat16m8_t vx = __riscv_vle16_v_f16m8(x + i, vl);
    if (beta != 1)
      vx = __riscv_vfmul_vf_f16m8(vx, beta, vl);

    /* Approximate exp(-|beta x|) */
    const vfloat16m8_t e = skl_leftexp_xsfvfexpa_zvfh_f16m8(
        __riscv_vfsgnj_vf_f16m8(vx, -1, vl), vl);

    /* Approximate 1 / (1 + exp(-|beta x|)) */
    const vfloat16m8_t d = __riscv_vfadd_vf_f16m8(e, 1, vl);
    // Since we know that e is in [0;1] and d is in [1;2], we can skip the usual
    // scaling and refinement checks that are required in `div`.
    vfloat16m8_t r = __riscv_vfrec7_v_f16m8(d, vl);
    /* Refine r */
    const vfloat16m8_t one = __riscv_vfmv_v_f_f16m8(1, vl);
    vfloat16m8_t t = __riscv_vfnmsac_vv_f16m8(one, d, r, vl); // 1 - d * r
    r = __riscv_vfmadd_vv_f16m8(r, t, r, vl); // r + r * (1 - d * r)

    /* Calculate quotient */
    const vbool2_t m = __riscv_vmflt_vf_f16m8_b2(vx, 0, vl);
    vfloat16m8_t res = __riscv_vfmul_vv_f16m8_mu(m, r, r, e, vl);

    /* Optionally multiply by y */
    if (y) {
      const vfloat16m8_t vy = __riscv_vle16_v_f16m8(y + i, vl);
      res = __riscv_vfmul_vv_f16m8(res, vy, vl);
    }

    /* Optionally multiply by (up + delta) */
    if (up) {
      vfloat16m8_t vu = __riscv_vle16_v_f16m8(up + i, vl);
      vu = __riscv_vfadd_vf_f16m8(vu, delta, vl);
      res = __riscv_vfmul_vv_f16m8(res, vu, vl);
    }

    __riscv_vse16_v_f16m8(out + i, res, vl);
  }
}

/**
 * @brief Approximate the exponential function on vector of f16 floating-point
 * values with a 1-ULP error bound in [-inf; 0].
 *
 * @note NaNs are not propagated.
 */
SKL_FUNC_PRIVATE vfloat16m8_t skl_leftexp_xsfvfexpa_zvfh_f16m8(vfloat16m8_t x,
                                                               size_t vl) {
  /* 0. Clamp inputs to lower bound */
  x = __riscv_vfmax_vf_f16m8(x, -0x1.158p4f16, vl);

  /* 1. Reduction */
  const _Float16 r_ln2 = 0x1.714p0f16; // 1/log(2)
  const _Float16 Q = 0x1.dp5f16;       // 2^(11-5-1) + B + p
  const vfloat16m8_t v =
      __riscv_vfmacc_vf_f16m8(__riscv_vfmv_v_f_f16m8(Q, vl), r_ln2, x, vl);
  const vfloat16m8_t f = __riscv_sf_vfexpa_v_f16m8(v, vl);
  const vfloat16m8_t z = __riscv_vfsub_vf_f16m8(v, Q, vl);

  const _Float16 l2u = 0x1.8p-1f16; // round(log(2),11-3-5,RN)
  const _Float16 l2l = -0x1.d1cp-5f16;
  const vfloat16m8_t s = // x - log(2) z
      __riscv_vfnmsac_vf_f16m8(__riscv_vfnmsac_vf_f16m8(x, l2u, z, vl), l2l, z,
                               vl);

  /* 2. Approximation */
  const vfloat16m8_t e = __riscv_vfmadd_vv_f16m8(s, f, f, vl);

  /* 3. Re-scaling */
  return __riscv_vfmul_vf_f16m8(e, 0x1p-11f16, vl); // UF happens here
}

SKL_FUNC void skl_logistic_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *x,
                                              size_t n) {
  skl_sigmoid_f16_xsfvfexpa_zvfh(out, (_Float16)1, x, NULL, NULL, (_Float16)0,
                                 n);
}

SKL_FUNC void skl_silu_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *x,
                                          size_t n) {
  skl_sigmoid_f16_xsfvfexpa_zvfh(out, (_Float16)1, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swish_f16_xsfvfexpa_zvfh(_Float16 *out, _Float16 beta,
                                           const _Float16 *x, size_t n) {
  skl_sigmoid_f16_xsfvfexpa_zvfh(out, beta, x, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_glu_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *x,
                                         const _Float16 *y, size_t n) {
  skl_sigmoid_f16_xsfvfexpa_zvfh(out, (_Float16)1, y, x, NULL, (_Float16)0, n);
}

SKL_FUNC void skl_swiglu_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *gate,
                                            const _Float16 *up, _Float16 delta,
                                            size_t n) {
  skl_sigmoid_f16_xsfvfexpa_zvfh(out, (_Float16)1, gate, gate, up, delta, n);
}
