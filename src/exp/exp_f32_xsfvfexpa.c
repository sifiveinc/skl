// Copyright 2025 SiFive, Inc.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#if !defined(__riscv_xsfvfexpa)
#error This file requires the Xsfvfexpa extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

SKL_FUNC void skl_exp_1u_f32_xsfvfexpa(float *out, const float *in, size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);
    // 0. Clamp
    vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(vx, vx, vl);
    const float LB = -0x1.9fe36ap6f; // round_down(log(0x1p-150))
    const float UB = +0x1.62e430p6f; // round_up(log(0x1.fffffffp127))
    vx = __riscv_vfmax_vf_f32m8_mu(nn, vx, vx, LB, vl);
    vx = __riscv_vfmin_vf_f32m8_mu(nn, vx, vx, UB, vl);
    // 1. Reduce
    vfloat32m8_t R = __riscv_vfmv_v_f_f32m8(0x1.715476p0f, vl);
    vfloat32m8_t Q = __riscv_vfmv_v_f_f32m8(0x1.004b8p17f, vl);
    vbool4_t m = __riscv_vmfgt_vf_f32m8_b4(vx, 0.0f, vl);
    Q = __riscv_vfmerge_vfm_f32m8(Q, 0x1.00338p17f, m, vl);
    vfloat32m8_t v = __riscv_vfmadd_vv_f32m8(R, vx, Q, vl);
    vfloat32m8_t z = __riscv_vfsub_vv_f32m8(v, Q, vl);
    const float C1 = +0x1.62e800p-1f;  // round(log(2),24-6-4)
    const float C2 = -0x1.e8082ep-16f; // log(2) - C1
    vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f32m8(s, C2, z, vl);
    // 2. Approximate
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(0x1.fff832p-2f, vl);
    vfloat32m8_t p = __riscv_vfmacc_vf_f32m8(c, 0x1.551ec2p-3f, s, vl);
    c = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    p = __riscv_vfmadd_vv_f32m8(p, s, c, vl);
    p = __riscv_vfmul_vv_f32m8(p, s, vl);
    // 3. Assemble
    vfloat32m8_t f = __riscv_sf_vfexpa_v_f32m8(v, vl);
    vfloat32m8_t e = __riscv_vfmadd_vv_f32m8(p, f, f, vl);
    // 4. Rescale
    vfloat32m8_t r = __riscv_vfmv_v_f_f32m8(0x1p-24f, vl);
    r = __riscv_vfmerge_vfm_f32m8(r, 0x1p+24f, m, vl);
    e = __riscv_vfmul_vv_f32m8(e, r, vl);
    // 5. Store
    __riscv_vse32_v_f32m8(out + i, e, vl);
  }
}

SKL_FUNC void skl_exp_1p0002ugen5d639eP6s0_f32_xsfvfexpa(float *out,
                                                         const float *in,
                                                         size_t n) {
  size_t vl;
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(in + i, vl);
    // 0. Clamp
    vbool4_t nn = __riscv_vmfeq_vv_f32m8_b4(vx, vx, vl);
    const float LB = -0x1.6023e8p6f;
    const float UB = +0x1.62e430p6f; // round_up(log(0x1.fffffffp127))
    vx = __riscv_vfmax_vf_f32m8_mu(nn, vx, vx, LB, vl);
    vx = __riscv_vfmin_vf_f32m8_mu(nn, vx, vx, UB, vl);
    // 1. Reduce
    const float R = 0x1.714eb2p0f;  // ~1/ln(2), attenuate for large x
    const float Q = 0x1.003f80p17f; // 2^(p-6-1) + B
    vfloat32m8_t q = __riscv_vfmv_v_f_f32m8(Q, vl);
    vfloat32m8_t v = __riscv_vfmacc_vf_f32m8(q, R, vx, vl);
    vfloat32m8_t z = __riscv_vfsub_vf_f32m8(v, Q, vl);
    const float C1 = +0x1.62e800p-1f;  // round(log(2),24-6-4)
    const float C2 = -0x1.e8082ep-16f; // log(2) - C1
    vfloat32m8_t s = __riscv_vfnmsac_vf_f32m8(vx, C1, z, vl);
    s = __riscv_vfnmsac_vf_f32m8(s, C2, z, vl);
    // 2. Approximate
    vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(0x1.fffa74p-2f, vl);
    vfloat32m8_t r = __riscv_vfmul_vv_f32m8(s, s, vl);
    vfloat32m8_t p = __riscv_vfmacc_vf_f32m8(c, 0x1.561e70p-3f, s, vl);
    p = __riscv_vfmadd_vv_f32m8(p, r, s, vl);
    // 3. Assemble
    vfloat32m8_t f = __riscv_sf_vfexpa_v_f32m8(v, vl);
    vfloat32m8_t e = __riscv_vfmadd_vv_f32m8(p, f, f, vl);
    // 4. Store
    __riscv_vse32_v_f32m8(out + i, e, vl);
  }
}
