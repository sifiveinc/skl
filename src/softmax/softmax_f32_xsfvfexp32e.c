// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>
#include <stdint.h>

SKL_FUNC void skl_softmax_f32_xsfvfexp32e(float *pDst, const float *pSrc,
                                          const float beta, const size_t n) {
  size_t vl = __riscv_vsetvl_e32m8(n);
  vfloat32m8_t vmax = __riscv_vle32_v_f32m8(pSrc, vl);
  for (size_t i = vl; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc + i, vl);
    vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, vx, vl);
  }
  vfloat32m1_t vfirst = __riscv_vlmul_trunc_v_f32m8_f32m1(vmax);
  vfirst = __riscv_vfredmax_vs_f32m8_f32m1(vmax, vfirst, n);
  float max = __riscv_vfmv_f_s_f32m1_f32(vfirst);

  vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, n);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc + i, vl);
    vx = __riscv_vfsub_vf_f32m8(vx, max, vl);
    if (beta != 1.0f)
      vx = __riscv_vfmul_vf_f32m8(vx, beta, vl);
    vx = __riscv_sf_vfexp_v_f32m8(vx, vl);
    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, vx, vl);
    __riscv_vse32_v_f32m8(pDst + i, vx, vl);
  }
  vfloat32m1_t ssum = __riscv_vfmv_s_f_f32m1(0.0f, n);
  ssum = __riscv_vfredusum_vs_f32m8_f32m1(vsum, ssum, n);
  float sum = __riscv_vfmv_f_s_f32m1_f32(ssum);
  float recip_sum = 1.0f / sum;

  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(pDst + i, vl);
    vx = __riscv_vfmul_vf_f32m8(vx, recip_sum, vl);
    __riscv_vse32_v_f32m8(pDst + i, vx, vl);
  }
}

/* 1D vector softmax. */
SKL_FUNC_PRIVATE vfloat32m8_t skl_softmax_vf_f32m8_xsfvfexp32e(
    vfloat32m8_t x, const float beta, const size_t vl) {
  vfloat32m1_t vfirst = __riscv_vlmul_trunc_v_f32m8_f32m1(x);
  vfirst = __riscv_vfredmax_vs_f32m8_f32m1(x, vfirst, vl);
  float max = __riscv_vfmv_f_s_f32m1_f32(vfirst);

  x = __riscv_vfsub_vf_f32m8(x, max, vl);
  if (beta != 1.0f)
    x = __riscv_vfmul_vf_f32m8(x, beta, vl);
  x = __riscv_sf_vfexp_v_f32m8(x, vl);

  vfloat32m1_t ssum = __riscv_vfmv_s_f_f32m1(0.0f, vl);
  ssum = __riscv_vfredusum_vs_f32m8_f32m1(x, ssum, vl);
  float sum = __riscv_vfmv_f_s_f32m1_f32(ssum);
  float recip_sum = 1.0f / sum;

  return __riscv_vfmul_vf_f32m8(x, recip_sum, vl);
}

SKL_FUNC_PRIVATE void
skl_softmax_2d_f32m8_xsfvfexp32e(float *s, const size_t rss, const float *a,
                                 const size_t rsa, const float beta,
                                 const size_t m, const size_t vl) {
  /* Rows fit in a single f32m8 register group. */
  for (size_t i = 0; i < m; ++i) {
    vfloat32m8_t x = __riscv_vle32_v_f32m8(a + i * rsa, vl);
    x = skl_softmax_vf_f32m8_xsfvfexp32e(x, beta, vl);
    __riscv_vse32_v_f32m8(s + i * rss, x, vl);
  }
}

SKL_FUNC_PRIVATE void
skl_softmax_2d_f32m8_x2_xsfvfexp32e(float *s, const size_t rss, const float *a,
                                    const size_t rsa, const float beta,
                                    const size_t m, const size_t vl) {
  /* Rows fit in a single f32m8 register group, processes two rows at
     a time. */
  size_t i;
  for (i = 0; i + 2 <= m; i += 2) {
    vfloat32m8_t x0 = __riscv_vle32_v_f32m8(a + (i + 0) * rsa, vl);
    vfloat32m8_t x1 = __riscv_vle32_v_f32m8(a + (i + 1) * rsa, vl);

    vfloat32m1_t f0 = __riscv_vlmul_trunc_v_f32m8_f32m1(x0);
    vfloat32m1_t f1 = __riscv_vlmul_trunc_v_f32m8_f32m1(x1);
    f0 = __riscv_vfredmax_vs_f32m8_f32m1(x0, f0, vl);
    f1 = __riscv_vfredmax_vs_f32m8_f32m1(x1, f1, vl);
    float max0 = __riscv_vfmv_f_s_f32m1_f32(f0);
    float max1 = __riscv_vfmv_f_s_f32m1_f32(f1);

    x0 = __riscv_vfsub_vf_f32m8(x0, max0, vl);
    x1 = __riscv_vfsub_vf_f32m8(x1, max1, vl);
    if (beta != 1.0f) {
      x0 = __riscv_vfmul_vf_f32m8(x0, beta, vl);
      x1 = __riscv_vfmul_vf_f32m8(x1, beta, vl);
    }
    x0 = __riscv_sf_vfexp_v_f32m8(x0, vl);
    x1 = __riscv_sf_vfexp_v_f32m8(x1, vl);

    vfloat32m1_t s0 = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t s1 = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    s0 = __riscv_vfredusum_vs_f32m8_f32m1(x0, s0, vl);
    s1 = __riscv_vfredusum_vs_f32m8_f32m1(x1, s1, vl);
    float r0 = 1.0f / __riscv_vfmv_f_s_f32m1_f32(s0);
    float r1 = 1.0f / __riscv_vfmv_f_s_f32m1_f32(s1);

    x0 = __riscv_vfmul_vf_f32m8(x0, r0, vl);
    x1 = __riscv_vfmul_vf_f32m8(x1, r1, vl);
    __riscv_vse32_v_f32m8(s + (i + 0) * rss, x0, vl);
    __riscv_vse32_v_f32m8(s + (i + 1) * rss, x1, vl);
  }
  if (i != m) { /* Cleanup */
    skl_softmax_2d_f32m8_xsfvfexp32e(s + i * rss, rss, a + i * rsa, rsa, beta,
                                     1, vl);
  }
}

SKL_FUNC_PRIVATE void
skl_softmax_2d_f32m8_x3_xsfvfexp32e(float *s, const size_t rss, const float *a,
                                    const size_t rsa, const float beta,
                                    const size_t m, const size_t vl) {
  /* Rows fit in a single f32m8 register group, processes three rows
     at a time. */
  size_t i;
  for (i = 0; i + 3 <= m; i += 3) {
    vfloat32m8_t x0 = __riscv_vle32_v_f32m8(a + (i + 0) * rsa, vl);
    vfloat32m8_t x1 = __riscv_vle32_v_f32m8(a + (i + 1) * rsa, vl);
    vfloat32m8_t x2 = __riscv_vle32_v_f32m8(a + (i + 2) * rsa, vl);

    vfloat32m1_t f0 = __riscv_vlmul_trunc_v_f32m8_f32m1(x0);
    vfloat32m1_t f1 = __riscv_vlmul_trunc_v_f32m8_f32m1(x1);
    vfloat32m1_t f2 = __riscv_vlmul_trunc_v_f32m8_f32m1(x2);
    f0 = __riscv_vfredmax_vs_f32m8_f32m1(x0, f0, vl);
    f1 = __riscv_vfredmax_vs_f32m8_f32m1(x1, f1, vl);
    f2 = __riscv_vfredmax_vs_f32m8_f32m1(x2, f2, vl);
    float max0 = __riscv_vfmv_f_s_f32m1_f32(f0);
    float max1 = __riscv_vfmv_f_s_f32m1_f32(f1);
    float max2 = __riscv_vfmv_f_s_f32m1_f32(f2);

    x0 = __riscv_vfsub_vf_f32m8(x0, max0, vl);
    x1 = __riscv_vfsub_vf_f32m8(x1, max1, vl);
    x2 = __riscv_vfsub_vf_f32m8(x2, max2, vl);
    if (beta != 1.0f) {
      x0 = __riscv_vfmul_vf_f32m8(x0, beta, vl);
      x1 = __riscv_vfmul_vf_f32m8(x1, beta, vl);
      x2 = __riscv_vfmul_vf_f32m8(x2, beta, vl);
    }
    x0 = __riscv_sf_vfexp_v_f32m8(x0, vl);
    x1 = __riscv_sf_vfexp_v_f32m8(x1, vl);
    x2 = __riscv_sf_vfexp_v_f32m8(x2, vl);

    vfloat32m1_t s0 = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t s1 = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    vfloat32m1_t s2 = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    s0 = __riscv_vfredusum_vs_f32m8_f32m1(x0, s0, vl);
    s1 = __riscv_vfredusum_vs_f32m8_f32m1(x1, s1, vl);
    s2 = __riscv_vfredusum_vs_f32m8_f32m1(x2, s2, vl);
    float r0 = 1.0f / __riscv_vfmv_f_s_f32m1_f32(s0);
    float r1 = 1.0f / __riscv_vfmv_f_s_f32m1_f32(s1);
    float r2 = 1.0f / __riscv_vfmv_f_s_f32m1_f32(s2);

    x0 = __riscv_vfmul_vf_f32m8(x0, r0, vl);
    x1 = __riscv_vfmul_vf_f32m8(x1, r1, vl);
    x2 = __riscv_vfmul_vf_f32m8(x2, r2, vl);
    __riscv_vse32_v_f32m8(s + (i + 0) * rss, x0, vl);
    __riscv_vse32_v_f32m8(s + (i + 1) * rss, x1, vl);
    __riscv_vse32_v_f32m8(s + (i + 2) * rss, x2, vl);
  }
  if (i != m) { /* Cleanup */
    skl_softmax_2d_f32m8_x2_xsfvfexp32e(s + i * rss, rss, a + i * rsa, rsa,
                                        beta, m - i, vl);
  }
}

/* 2D Softmax vectorizing along rows of S and A. */
SKL_FUNC_PRIVATE void
skl_softmax_2d_nvec_f32_xsfvfexp32e(float *s, const size_t rss, const float *a,
                                    const size_t rsa, const float beta,
                                    const size_t m, const size_t n) {
  size_t vl = __riscv_vsetvl_e32m8(n);
  if (vl == n) {
    skl_softmax_2d_f32m8_x3_xsfvfexp32e(s, rss, a, rsa, beta, m, vl);
  } else {
    for (size_t i = 0; i < m; ++i) {
      skl_softmax_f32_xsfvfexp32e(s + i * rss, a + i * rsa, beta, n);
    }
  }
}

// NOLINTBEGIN(*-confusable-identifiers)
SKL_FUNC_PRIVATE vfloat32m1x4_t
skl_softmax_vget_v_f32m1x8_f32m1x4(vfloat32m1x8_t src, size_t index) {
  vfloat32m1_t v0, v1, v2, v3;
  if (index == 0) {
    v0 = __riscv_vget_v_f32m1x8_f32m1(src, 0);
    v1 = __riscv_vget_v_f32m1x8_f32m1(src, 1);
    v2 = __riscv_vget_v_f32m1x8_f32m1(src, 2);
    v3 = __riscv_vget_v_f32m1x8_f32m1(src, 3);
  } else {
    v0 = __riscv_vget_v_f32m1x8_f32m1(src, 4);
    v1 = __riscv_vget_v_f32m1x8_f32m1(src, 5);
    v2 = __riscv_vget_v_f32m1x8_f32m1(src, 6);
    v3 = __riscv_vget_v_f32m1x8_f32m1(src, 7);
  }
  return __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
}

SKL_FUNC_PRIVATE vfloat32m1x8_t
skl_softmax_vcreate_v_f32m1x4_f32m1x8(vfloat32m1x4_t vs1, vfloat32m1x4_t vs2) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs1, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs1, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs1, 3);
  vfloat32m1_t v4 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v5 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v6 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v7 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  return __riscv_vcreate_v_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfmax_vv_f32m1x4(vfloat32m1x4_t vs1,
                                                             vfloat32m1x4_t vs2,
                                                             size_t vl) {
  vfloat32m1_t v10 = __riscv_vget_v_f32m1x4_f32m1(vs1, 0);
  vfloat32m1_t v11 = __riscv_vget_v_f32m1x4_f32m1(vs1, 1);
  vfloat32m1_t v12 = __riscv_vget_v_f32m1x4_f32m1(vs1, 2);
  vfloat32m1_t v13 = __riscv_vget_v_f32m1x4_f32m1(vs1, 3);
  vfloat32m1_t v20 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v21 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v22 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v23 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  vfloat32m1_t vm0 = __riscv_vfmax_vv_f32m1(v10, v20, vl);
  vfloat32m1_t vm1 = __riscv_vfmax_vv_f32m1(v11, v21, vl);
  vfloat32m1_t vm2 = __riscv_vfmax_vv_f32m1(v12, v22, vl);
  vfloat32m1_t vm3 = __riscv_vfmax_vv_f32m1(v13, v23, vl);
  return __riscv_vcreate_v_f32m1x4(vm0, vm1, vm2, vm3);
}

SKL_FUNC_PRIVATE vfloat32m1_t
skl_softmax_vfredmax_v_f32m1x2_f32m1(vfloat32m1x2_t vs1, size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs1, 1);
  return __riscv_vfmax_vv_f32m1(v0, v1, vl);
}

SKL_FUNC_PRIVATE vfloat32m1x2_t
skl_softmax_vfredmax_v_f32m1x4_f32m1x2(vfloat32m1x4_t vs1, size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs1, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs1, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs1, 3);

  vfloat32m1_t vm02 = __riscv_vfmax_vv_f32m1(v0, v2, vl);
  vfloat32m1_t vm13 = __riscv_vfmax_vv_f32m1(v1, v3, vl);
  return __riscv_vcreate_v_f32m1x2(vm02, vm13);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t
skl_softmax_vfredmax_v_f32m1x8_f32m1x4(vfloat32m1x8_t vs1, size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs1, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs1, 1);
  return skl_softmax_vfmax_vv_f32m1x4(v0, v1, vl);
}

SKL_FUNC_PRIVATE vfloat32m1_t
skl_softmax_vfredmax_v_f32m1x4_f32m1(vfloat32m1x4_t vs1, size_t vl) {
  vfloat32m1x2_t vp0 = skl_softmax_vfredmax_v_f32m1x4_f32m1x2(vs1, vl);
  return skl_softmax_vfredmax_v_f32m1x2_f32m1(vp0, vl);
}

SKL_FUNC_PRIVATE vfloat32m1_t
skl_softmax_vfredmax_v_f32m1x8_f32m1(vfloat32m1x8_t vs1, size_t vl) {
  vfloat32m1x4_t vp0 = skl_softmax_vfredmax_v_f32m1x8_f32m1x4(vs1, vl);
  vfloat32m1x2_t vp1 = skl_softmax_vfredmax_v_f32m1x4_f32m1x2(vp0, vl);
  return skl_softmax_vfredmax_v_f32m1x2_f32m1(vp1, vl);
}

SKL_FUNC_PRIVATE vfloat32m1_t skl_softmax_vfredmax_v_f32m1x16_f32m1(
    vfloat32m1x8_t vs1, vfloat32m1x8_t vs2, size_t vl) {
  vfloat32m1x4_t v20 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 0);
  vfloat32m1x4_t v21 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 1);

  vfloat32m1x4_t vm = skl_softmax_vfredmax_v_f32m1x8_f32m1x4(vs1, vl);
  vm = skl_softmax_vfmax_vv_f32m1x4(vm, v20, vl);
  vm = skl_softmax_vfmax_vv_f32m1x4(vm, v21, vl);
  return skl_softmax_vfredmax_v_f32m1x4_f32m1(vm, vl);
}

/* Addition */
SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfadd_vv_f32m1x4(vfloat32m1x4_t vs1,
                                                             vfloat32m1x4_t vs2,
                                                             size_t vl) {
  vfloat32m1_t v10 = __riscv_vget_v_f32m1x4_f32m1(vs1, 0);
  vfloat32m1_t v11 = __riscv_vget_v_f32m1x4_f32m1(vs1, 1);
  vfloat32m1_t v12 = __riscv_vget_v_f32m1x4_f32m1(vs1, 2);
  vfloat32m1_t v13 = __riscv_vget_v_f32m1x4_f32m1(vs1, 3);
  vfloat32m1_t v20 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v21 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v22 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v23 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  vfloat32m1_t vd0 = __riscv_vfadd_vv_f32m1(v10, v20, vl);
  vfloat32m1_t vd1 = __riscv_vfadd_vv_f32m1(v11, v21, vl);
  vfloat32m1_t vd2 = __riscv_vfadd_vv_f32m1(v12, v22, vl);
  vfloat32m1_t vd3 = __riscv_vfadd_vv_f32m1(v13, v23, vl);
  return __riscv_vcreate_v_f32m1x4(vd0, vd1, vd2, vd3);
}

/* Reduction sums */
SKL_FUNC_PRIVATE vfloat32m1_t
skl_softmax_vfredsum_v_f32m1x2_f32m1(vfloat32m1x2_t vs1, size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs1, 1);
  return __riscv_vfadd_vv_f32m1(v0, v1, vl);
}

SKL_FUNC_PRIVATE vfloat32m1x2_t
skl_softmax_vfredsum_v_f32m1x4_f32m1x2(vfloat32m1x4_t vs1, size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs1, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs1, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs1, 3);

  vfloat32m1_t vs02 = __riscv_vfadd_vv_f32m1(v0, v1, vl);
  vfloat32m1_t vs13 = __riscv_vfadd_vv_f32m1(v2, v3, vl);
  return __riscv_vcreate_v_f32m1x2(vs02, vs13);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t
skl_softmax_vfredsum_v_f32m1x8_f32m1x4(vfloat32m1x8_t vs1, size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs1, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs1, 1);
  return skl_softmax_vfadd_vv_f32m1x4(v0, v1, vl);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfredsum_v_f32m1x16_f32m1x4(
    vfloat32m1x8_t vs1, vfloat32m1x8_t vs2, size_t vl) {
  vfloat32m1x4_t v20 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 0);
  vfloat32m1x4_t v21 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 1);

  vfloat32m1x4_t vs = skl_softmax_vfredsum_v_f32m1x8_f32m1x4(vs1, vl);
  vs = skl_softmax_vfadd_vv_f32m1x4(vs, v20, vl);
  vs = skl_softmax_vfadd_vv_f32m1x4(vs, v21, vl);
  return vs;
}

/* Subtraction */
SKL_FUNC_PRIVATE vfloat32m1x2_t skl_softmax_vfsub_vv_f32m1x2(vfloat32m1x2_t vs2,
                                                             vfloat32m1_t vs1,
                                                             size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs2, 1);
  v0 = __riscv_vfsub_vv_f32m1(v0, vs1, vl);
  v1 = __riscv_vfsub_vv_f32m1(v1, vs1, vl);
  return __riscv_vcreate_v_f32m1x2(v0, v1);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfsub_vv_f32m1x4(vfloat32m1x4_t vs2,
                                                             vfloat32m1_t vs1,
                                                             size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  v0 = __riscv_vfsub_vv_f32m1(v0, vs1, vl);
  v1 = __riscv_vfsub_vv_f32m1(v1, vs1, vl);
  v2 = __riscv_vfsub_vv_f32m1(v2, vs1, vl);
  v3 = __riscv_vfsub_vv_f32m1(v3, vs1, vl);
  return __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
}

SKL_FUNC_PRIVATE vfloat32m1x8_t skl_softmax_vfsub_vv_f32m1x8(vfloat32m1x8_t vs2,
                                                             vfloat32m1_t vs1,
                                                             size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 1);
  v0 = skl_softmax_vfsub_vv_f32m1x4(v0, vs1, vl);
  v1 = skl_softmax_vfsub_vv_f32m1x4(v1, vs1, vl);
  return skl_softmax_vcreate_v_f32m1x4_f32m1x8(v0, v1);
}

/* Multiplication */
SKL_FUNC_PRIVATE vfloat32m1x2_t skl_softmax_vfmul_vf_f32m1x2(vfloat32m1x2_t vs2,
                                                             float rs1,
                                                             size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs2, 1);
  v0 = __riscv_vfmul_vf_f32m1(v0, rs1, vl);
  v1 = __riscv_vfmul_vf_f32m1(v1, rs1, vl);
  return __riscv_vcreate_v_f32m1x2(v0, v1);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfmul_vf_f32m1x4(vfloat32m1x4_t vs2,
                                                             float rs1,
                                                             size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  v0 = __riscv_vfmul_vf_f32m1(v0, rs1, vl);
  v1 = __riscv_vfmul_vf_f32m1(v1, rs1, vl);
  v2 = __riscv_vfmul_vf_f32m1(v2, rs1, vl);
  v3 = __riscv_vfmul_vf_f32m1(v3, rs1, vl);
  return __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
}

SKL_FUNC_PRIVATE vfloat32m1x8_t skl_softmax_vfmul_vf_f32m1x8(vfloat32m1x8_t vs2,
                                                             float rs1,
                                                             size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 1);
  v0 = skl_softmax_vfmul_vf_f32m1x4(v0, rs1, vl);
  v1 = skl_softmax_vfmul_vf_f32m1x4(v1, rs1, vl);
  return skl_softmax_vcreate_v_f32m1x4_f32m1x8(v0, v1);
}

SKL_FUNC_PRIVATE vfloat32m1x2_t skl_softmax_vfmul_vv_f32m1x2(vfloat32m1x2_t vs2,
                                                             vfloat32m1_t vs1,
                                                             size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs2, 1);
  v0 = __riscv_vfmul_vv_f32m1(v0, vs1, vl);
  v1 = __riscv_vfmul_vv_f32m1(v1, vs1, vl);
  return __riscv_vcreate_v_f32m1x2(v0, v1);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfmul_vv_f32m1x4(vfloat32m1x4_t vs2,
                                                             vfloat32m1_t vs1,
                                                             size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  v0 = __riscv_vfmul_vv_f32m1(v0, vs1, vl);
  v1 = __riscv_vfmul_vv_f32m1(v1, vs1, vl);
  v2 = __riscv_vfmul_vv_f32m1(v2, vs1, vl);
  v3 = __riscv_vfmul_vv_f32m1(v3, vs1, vl);
  return __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
}

SKL_FUNC_PRIVATE vfloat32m1x8_t skl_softmax_vfmul_vv_f32m1x8(vfloat32m1x8_t vs2,
                                                             vfloat32m1_t vs1,
                                                             size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs2, 1);
  v0 = skl_softmax_vfmul_vv_f32m1x4(v0, vs1, vl);
  v1 = skl_softmax_vfmul_vv_f32m1x4(v1, vs1, vl);
  return skl_softmax_vcreate_v_f32m1x4_f32m1x8(v0, v1);
}

/* Exponential */
SKL_FUNC_PRIVATE vfloat32m1x2_t skl_softmax_vfexp_v_f32m1x2(vfloat32m1x2_t vs1,
                                                            size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs1, 1);
  v0 = __riscv_sf_vfexp_v_f32m1(v0, vl);
  v1 = __riscv_sf_vfexp_v_f32m1(v1, vl);
  return __riscv_vcreate_v_f32m1x2(v0, v1);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfexp_v_f32m1x4(vfloat32m1x4_t vs1,
                                                            size_t vl) {
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs1, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs1, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs1, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs1, 3);
  v0 = __riscv_sf_vfexp_v_f32m1(v0, vl);
  v1 = __riscv_sf_vfexp_v_f32m1(v1, vl);
  v2 = __riscv_sf_vfexp_v_f32m1(v2, vl);
  v3 = __riscv_sf_vfexp_v_f32m1(v3, vl);
  return __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
}

SKL_FUNC_PRIVATE vfloat32m1x8_t skl_softmax_vfexp_v_f32m1x8(vfloat32m1x8_t vs1,
                                                            size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs1, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vs1, 1);
  v0 = skl_softmax_vfexp_v_f32m1x4(v0, vl);
  v1 = skl_softmax_vfexp_v_f32m1x4(v1, vl);
  return skl_softmax_vcreate_v_f32m1x4_f32m1x8(v0, v1);
}

/* Fused multiply-add */
SKL_FUNC_PRIVATE vfloat32m1x2_t skl_softmax_vfmadd_vv_f32m1x2(
    vfloat32m1x2_t vd, vfloat32m1_t vc, vfloat32m1x2_t vs2, size_t vl) {
  vfloat32m1_t vd0 = __riscv_vget_v_f32m1x2_f32m1(vd, 0);
  vfloat32m1_t vd1 = __riscv_vget_v_f32m1x2_f32m1(vd, 1);
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x2_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x2_f32m1(vs2, 1);
  vd0 = __riscv_vfmadd_vv_f32m1(vd0, vc, v0, vl);
  vd1 = __riscv_vfmadd_vv_f32m1(vd1, vc, v1, vl);
  return __riscv_vcreate_v_f32m1x2(vd0, vd1);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfmadd_vv_f32m1x4(
    vfloat32m1x4_t vd, vfloat32m1_t vc, vfloat32m1x4_t vs2, size_t vl) {
  vfloat32m1_t vd0 = __riscv_vget_v_f32m1x4_f32m1(vd, 0);
  vfloat32m1_t vd1 = __riscv_vget_v_f32m1x4_f32m1(vd, 1);
  vfloat32m1_t vd2 = __riscv_vget_v_f32m1x4_f32m1(vd, 2);
  vfloat32m1_t vd3 = __riscv_vget_v_f32m1x4_f32m1(vd, 3);
  vfloat32m1_t v0 = __riscv_vget_v_f32m1x4_f32m1(vs2, 0);
  vfloat32m1_t v1 = __riscv_vget_v_f32m1x4_f32m1(vs2, 1);
  vfloat32m1_t v2 = __riscv_vget_v_f32m1x4_f32m1(vs2, 2);
  vfloat32m1_t v3 = __riscv_vget_v_f32m1x4_f32m1(vs2, 3);
  vd0 = __riscv_vfmadd_vv_f32m1(vd0, vc, v0, vl);
  vd1 = __riscv_vfmadd_vv_f32m1(vd1, vc, v1, vl);
  vd2 = __riscv_vfmadd_vv_f32m1(vd2, vc, v2, vl);
  vd3 = __riscv_vfmadd_vv_f32m1(vd3, vc, v3, vl);
  return __riscv_vcreate_v_f32m1x4(vd0, vd1, vd2, vd3);
}

/* Fused multiply-subtract */
SKL_FUNC_PRIVATE vfloat32m1x2_t skl_softmax_vfmsub_vf_f32m1x2(vfloat32m1x2_t vd,
                                                              float rs1,
                                                              vfloat32m1_t vs2,
                                                              size_t vl) {
  vfloat32m1_t vd0 = __riscv_vget_v_f32m1x2_f32m1(vd, 0);
  vfloat32m1_t vd1 = __riscv_vget_v_f32m1x2_f32m1(vd, 1);
  vd0 = __riscv_vfmsub_vf_f32m1(vd0, rs1, vs2, vl);
  vd1 = __riscv_vfmsub_vf_f32m1(vd1, rs1, vs2, vl);
  return __riscv_vcreate_v_f32m1x2(vd0, vd1);
}

SKL_FUNC_PRIVATE vfloat32m1x4_t skl_softmax_vfmsub_vf_f32m1x4(vfloat32m1x4_t vd,
                                                              float rs1,
                                                              vfloat32m1_t vs2,
                                                              size_t vl) {
  vfloat32m1_t vd0 = __riscv_vget_v_f32m1x4_f32m1(vd, 0);
  vfloat32m1_t vd1 = __riscv_vget_v_f32m1x4_f32m1(vd, 1);
  vfloat32m1_t vd2 = __riscv_vget_v_f32m1x4_f32m1(vd, 2);
  vfloat32m1_t vd3 = __riscv_vget_v_f32m1x4_f32m1(vd, 3);
  vd0 = __riscv_vfmsub_vf_f32m1(vd0, rs1, vs2, vl);
  vd1 = __riscv_vfmsub_vf_f32m1(vd1, rs1, vs2, vl);
  vd2 = __riscv_vfmsub_vf_f32m1(vd2, rs1, vs2, vl);
  vd3 = __riscv_vfmsub_vf_f32m1(vd3, rs1, vs2, vl);
  return __riscv_vcreate_v_f32m1x4(vd0, vd1, vd2, vd3);
}

SKL_FUNC_PRIVATE vfloat32m1x8_t skl_softmax_vfmsub_vf_f32m1x8(vfloat32m1x8_t vd,
                                                              float rs1,
                                                              vfloat32m1_t vs2,
                                                              size_t vl) {
  vfloat32m1x4_t v0 = skl_softmax_vget_v_f32m1x8_f32m1x4(vd, 0);
  vfloat32m1x4_t v1 = skl_softmax_vget_v_f32m1x8_f32m1x4(vd, 1);
  v0 = skl_softmax_vfmsub_vf_f32m1x4(v0, rs1, vs2, vl);
  v1 = skl_softmax_vfmsub_vf_f32m1x4(v1, rs1, vs2, vl);
  return skl_softmax_vcreate_v_f32m1x4_f32m1x8(v0, v1);
}
// NOLINTEND(*-confusable-identifiers)

/* Element-wise reciprocal approximation where inputs are known to be
   non-zero and infinite inputs are safe to become NaN. */
SKL_FUNC_PRIVATE vfloat32m1_t skl_softmax_vfrcp_v_f32m1(vfloat32m1_t x,
                                                        size_t vl) {
  vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  vfloat32m1_t r = __riscv_vfrec7_v_f32m1(x, vl);
  vfloat32m1_t t = __riscv_vfnmsub_vv_f32m1(x, r, one, vl);
  r = __riscv_vfmadd_vv_f32m1(r, t, r, vl);
  t = __riscv_vfnmsub_vv_f32m1(x, r, one, vl);
  return __riscv_vfmadd_vv_f32m1(r, t, r, vl);
}

/* Element-wise logarithm, suitable for use in softmax:
   - input domain [1;∞],
   - no special handling for NaN, x<0, x==∞ (returns 128*ln2),
   - result can be slightly inaccurate (1.306 ulp). */
SKL_FUNC_PRIVATE vfloat32m1_t skl_softmax_vflog_v_f32m1(vfloat32m1_t x,
                                                        size_t vl) {
  /* Reduce argument */
  vint32m1_t i = __riscv_vreinterpret_v_f32m1_i32m1(x);
  vint32m1_t j = i;
  i = __riscv_vand_vx_i32m1(i, 0x7f800000, vl); /* isolate exponent */
  i = __riscv_vsub_vx_i32m1(i, 0x3f800000, vl); /* unbias */
  j = __riscv_vsub_vv_i32m1(j, i, vl);          /* reset exponent */
  vfloat32m1_t E = __riscv_vfcvt_f_x_v_f32m1(i, vl);
  vfloat32m1_t m = __riscv_vreinterpret_v_i32m1_f32m1(j);
  vfloat32m1_t r = __riscv_vfsub_vf_f32m1(m, 1.0f, vl); /* exact */

  /* Approximate log(1+r) on [1;2) */
  const float ca = -0x1.afc06ep-9f;
  vfloat32m1_t c9 = __riscv_vfmv_v_f_f32m1(+0x1.492c90p-6f, vl);
  vfloat32m1_t c8 = __riscv_vfmv_v_f_f32m1(-0x1.d695d8p-5f, vl);
  const float c7 = +0x1.b4872ap-4f;
  vfloat32m1_t c6 = __riscv_vfmv_v_f_f32m1(-0x1.3a2fa4p-3f, vl);
  const float c5 = +0x1.934142p-3f;
  vfloat32m1_t c4 = __riscv_vfmv_v_f_f32m1(-0x1.ff2136p-3f, vl);
  const float c3 = +0x1.554da4p-2f;
  vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(-0x1.ffffd0p-2f, vl);

  vfloat32m1_t p9a = __riscv_vfmacc_vf_f32m1(c9, ca, r, vl);
  vfloat32m1_t p67 = __riscv_vfmacc_vf_f32m1(c6, c7, r, vl);
  vfloat32m1_t p45 = __riscv_vfmacc_vf_f32m1(c4, c5, r, vl);
  vfloat32m1_t p23 = __riscv_vfmacc_vf_f32m1(c2, c3, r, vl);

  vfloat32m1_t R = __riscv_vfmul_vv_f32m1(r, r, vl);
  vfloat32m1_t p89a = __riscv_vfmacc_vv_f32m1(c8, r, p9a, vl);

  vfloat32m1_t S = __riscv_vfmul_vv_f32m1(R, R, vl);
  vfloat32m1_t p2345 = __riscv_vfmacc_vv_f32m1(p23, R, p45, vl);
  vfloat32m1_t p6789a = __riscv_vfmacc_vv_f32m1(p67, R, p89a, vl);

  vfloat32m1_t p = __riscv_vfmacc_vv_f32m1(p2345, S, p6789a, vl);
  p = __riscv_vfmacc_vv_f32m1(r, R, p, vl);

  /* Reconstruct result from log(x) = E * ln2 + log(1+r) */
  const float ln2 = 0x1.62e43p-24f; /* log(2) * 0x1p-23 */
  return __riscv_vfmadd_vf_f32m1(E, ln2, p, vl);
}

/* On-line scaling using segmented IO and reciprocal sums. */
SKL_FUNC_PRIVATE void skl_softmax_2d_olscal_sr_f32m1_xsfvfexp32e(
    float *s, const size_t rss, const float *a, const size_t rsa,
    const float beta, vfloat32m1_t vsum, vfloat32m1_t vmax, const size_t m,
    const size_t n) {
  const ptrdiff_t bsa = (ptrdiff_t)(rsa * sizeof(float));
  const ptrdiff_t bss = (ptrdiff_t)(rss * sizeof(float));
  vfloat32m1_t vrcp = skl_softmax_vfrcp_v_f32m1(vsum, m);
  /* Exponentiate and normalize: */
  size_t j = 0;
  for (; j + 8 <= n; j += 8) {
    vfloat32m1x8_t vx = __riscv_vlsseg8e32_v_f32m1x8(a + j, bsa, m);
    vx = skl_softmax_vfsub_vv_f32m1x8(vx, vmax, m);
    if (beta != 1.f)
      vx = skl_softmax_vfmul_vf_f32m1x8(vx, beta, m);
    vx = skl_softmax_vfexp_v_f32m1x8(vx, m);
    vx = skl_softmax_vfmul_vv_f32m1x8(vx, vrcp, m);
    __riscv_vssseg8e32_v_f32m1x8(s + j, bss, vx, m);
  }
  for (; j + 4 <= n; j += 4) {
    vfloat32m1x4_t vx = __riscv_vlsseg4e32_v_f32m1x4(a + j, bsa, m);
    vx = skl_softmax_vfsub_vv_f32m1x4(vx, vmax, m);
    if (beta != 1.f)
      vx = skl_softmax_vfmul_vf_f32m1x4(vx, beta, m);
    vx = skl_softmax_vfexp_v_f32m1x4(vx, m);
    vx = skl_softmax_vfmul_vv_f32m1x4(vx, vrcp, m);
    __riscv_vssseg4e32_v_f32m1x4(s + j, bss, vx, m);
  }
  for (; j + 2 <= n; j += 2) {
    vfloat32m1x2_t vx = __riscv_vlsseg2e32_v_f32m1x2(a + j, bsa, m);
    vx = skl_softmax_vfsub_vv_f32m1x2(vx, vmax, m);
    if (beta != 1.f)
      vx = skl_softmax_vfmul_vf_f32m1x2(vx, beta, m);
    vx = skl_softmax_vfexp_v_f32m1x2(vx, m);
    vx = skl_softmax_vfmul_vv_f32m1x2(vx, vrcp, m);
    __riscv_vssseg2e32_v_f32m1x2(s + j, bss, vx, m);
  }
  for (; j + 1 <= n; ++j) {
    vfloat32m1_t vx = __riscv_vlse32_v_f32m1(a + j, bsa, m);
    vx = __riscv_vfsub_vv_f32m1(vx, vmax, m);
    if (beta != 1.f)
      vx = __riscv_vfmul_vf_f32m1(vx, beta, m);
    vx = __riscv_sf_vfexp_v_f32m1(vx, m);
    vx = __riscv_vfmul_vv_f32m1(vx, vrcp, m);
    __riscv_vsse32_v_f32m1(s + j, bss, vx, m);
  }
}

/* On-line scaling using segmented IO and logarithmic sums.  Some very
   small results may vanish to zero here, when they might not have
   with `skl_softmax_2d_olscal_sr_f32m1_xsfvfexp32e`. */
SKL_FUNC_PRIVATE void skl_softmax_2d_olscal_sl_f32m1_xsfvfexp32e(
    float *s, const size_t rss, const float *a, const size_t rsa,
    const float beta, vfloat32m1_t vsum, vfloat32m1_t vmax, const size_t m,
    const size_t n) {
  const ptrdiff_t bsa = (ptrdiff_t)(rsa * sizeof(float));
  const ptrdiff_t bss = (ptrdiff_t)(rss * sizeof(float));
  vfloat32m1_t vlog = skl_softmax_vflog_v_f32m1(vsum, m);
  /* Exponentiate and normalize */
  size_t j = 0;
  for (; j + 8 <= n; j += 8) {
    vfloat32m1x8_t vx = __riscv_vlsseg8e32_v_f32m1x8(a + j, bsa, m);
    vx = skl_softmax_vfsub_vv_f32m1x8(vx, vmax, m);
    vx = skl_softmax_vfmsub_vf_f32m1x8(vx, beta, vlog, m);
    vx = skl_softmax_vfexp_v_f32m1x8(vx, m);
    __riscv_vssseg8e32_v_f32m1x8(s + j, bss, vx, m);
  }
  for (; j + 4 <= n; j += 4) {
    vfloat32m1x4_t vx = __riscv_vlsseg4e32_v_f32m1x4(a + j, bsa, m);
    vx = skl_softmax_vfsub_vv_f32m1x4(vx, vmax, m);
    vx = skl_softmax_vfmsub_vf_f32m1x4(vx, beta, vlog, m);
    vx = skl_softmax_vfexp_v_f32m1x4(vx, m);
    __riscv_vssseg4e32_v_f32m1x4(s + j, bss, vx, m);
  }
  for (; j + 2 <= n; j += 2) {
    vfloat32m1x2_t vx = __riscv_vlsseg2e32_v_f32m1x2(a + j, bsa, m);
    vx = skl_softmax_vfsub_vv_f32m1x2(vx, vmax, m);
    vx = skl_softmax_vfmsub_vf_f32m1x2(vx, beta, vlog, m);
    vx = skl_softmax_vfexp_v_f32m1x2(vx, m);
    __riscv_vssseg2e32_v_f32m1x2(s + j, bss, vx, m);
  }
  for (; j + 1 <= n; ++j) {
    vfloat32m1_t vx = __riscv_vlse32_v_f32m1(a + j, bsa, m);
    vx = __riscv_vfsub_vv_f32m1(vx, vmax, m);
    vx = __riscv_vfmsub_vf_f32m1(vx, beta, vlog, m);
    vx = __riscv_sf_vfexp_v_f32m1(vx, m);
    __riscv_vsse32_v_f32m1(s + j, bss, vx, m);
  }
}

/* On-line scaling using segmented IO.  Dispatch based on BETA. */
SKL_FUNC_PRIVATE void skl_softmax_2d_olscal_s_f32m1_xsfvfexp32e(
    float *s, const size_t rss, const float *a, const size_t rsa,
    const float beta, vfloat32m1_t vsum, vfloat32m1_t vmax, const size_t m,
    const size_t n) {
  if (beta != 1.f && n > 16) /* enough work to offset cost of logarithm */
    skl_softmax_2d_olscal_sl_f32m1_xsfvfexp32e(s, rss, a, rsa, beta, vsum, vmax,
                                               m, n);
  else /* beta == 1.f || n <= 16 */
    skl_softmax_2d_olscal_sr_f32m1_xsfvfexp32e(s, rss, a, rsa, beta, vsum, vmax,
                                               m, n);
}

/* On-line scaling using loops over rows and reciprocal sums */
SKL_FUNC_PRIVATE void skl_softmax_2d_olscal_lr_f32m1_xsfvfexp32e(
    float *s, const size_t rss, const float *a, const size_t rsa,
    const float beta, vfloat32m1_t vsum, vfloat32m1_t vmax, const size_t m,
    const size_t n) {
  vfloat32m1_t vrcp = skl_softmax_vfrcp_v_f32m1(vsum, m);
  /* Exponentiate and normalize by rows */
  for (size_t k = 0; k < m; ++k) {
    float max = __riscv_vfmv_f_s_f32m1_f32(vmax);
    vmax = __riscv_vfslide1down_vf_f32m1(vmax, beta, m);
    float rcp = __riscv_vfmv_f_s_f32m1_f32(vrcp);
    vrcp = __riscv_vfslide1down_vf_f32m1(vrcp, beta, m);

    size_t nl = 0;
    for (size_t b = 0; b < n; b += nl) {
      nl = __riscv_vsetvl_e32m4(n - b);
      vfloat32m4_t vx = __riscv_vle32_v_f32m4(a + k * rsa + b, nl);
      vx = __riscv_vfsub_vf_f32m4(vx, max, nl);
      if (beta != 1.f)
        vx = __riscv_vfmul_vf_f32m4(vx, beta, nl);
      vx = __riscv_sf_vfexp_v_f32m4(vx, nl);
      vx = __riscv_vfmul_vf_f32m4(vx, rcp, nl);
      __riscv_vse32_v_f32m4(s + k * rss + b, vx, nl);
    }
  }
}

/* Calculate exponential term and scale the N elements in each row of
   A by the reciprocal of corresponding sum in VSUM, of vector length
   M, and store the results in S.  Intended for use after on-line scan
   of maximum inputs and their exponential sum. */
SKL_FUNC_PRIVATE void skl_softmax_2d_olscal_f32m1_xsfvfexp32e(
    float *s, const size_t rss, const float *a, const size_t rsa,
    const float beta, vfloat32m1_t vsum, vfloat32m1_t vmax, const size_t m,
    const size_t n) {
  size_t vlm1 = __riscv_vsetvlmax_e32m1();
  if (n <= vlm1 + (vlm1 >> 1)) {
    skl_softmax_2d_olscal_s_f32m1_xsfvfexp32e(s, rss, a, rsa, beta, vsum, vmax,
                                              m, n);
  } else {
    skl_softmax_2d_olscal_lr_f32m1_xsfvfexp32e(s, rss, a, rsa, beta, vsum, vmax,
                                               m, n);
  }
}

/* Row-storage, row-wise reduction.  Intended for skinny-ish matrices:
   vectorizes along columns, processing 16 columns at a time using
   strided segment loads/store, with on-line block stabilization to
   minimize strided memory access.  Normalization is adaptive. */
SKL_FUNC_PRIVATE void skl_softmax_2d_mvec_f32_xsfvfexp32e(float *s, size_t rss,
                                                          const float *a,
                                                          size_t rsa,
                                                          float beta, size_t m,
                                                          size_t n) {
  size_t vl = 0;
  const ptrdiff_t bsa = (ptrdiff_t)(rsa * sizeof(float));

  for (size_t i = 0; i < m; i += vl) {
    vl = __riscv_vsetvl_e32m1(m - i);

    /* Attempt to peel first iteration. */
    vfloat32m1x4_t vss; /* Partial sums */
    vfloat32m1_t vmax;  /* Global maximum */
    size_t j = 0;       /* column index */

    if (n >= 16) {
      vfloat32m1x8_t vx0 =
          __riscv_vlsseg8e32_v_f32m1x8(a + i * rsa + 0, bsa, vl);
      vfloat32m1x8_t vx1 =
          __riscv_vlsseg8e32_v_f32m1x8(a + i * rsa + 8, bsa, vl);
      vmax = skl_softmax_vfredmax_v_f32m1x16_f32m1(vx0, vx1, vl);
      vx0 = skl_softmax_vfsub_vv_f32m1x8(vx0, vmax, vl);
      vx1 = skl_softmax_vfsub_vv_f32m1x8(vx1, vmax, vl);
      if (beta != 1.0f) {
        vx0 = skl_softmax_vfmul_vf_f32m1x8(vx0, beta, vl);
        vx1 = skl_softmax_vfmul_vf_f32m1x8(vx1, beta, vl);
      }
      vx0 = skl_softmax_vfexp_v_f32m1x8(vx0, vl);
      vx1 = skl_softmax_vfexp_v_f32m1x8(vx1, vl);
      vss = skl_softmax_vfredsum_v_f32m1x16_f32m1x4(vx0, vx1, vl);
      j += 16;
    } else {
      vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.f, vl);
      vss = __riscv_vcreate_v_f32m1x4(z, z, z, z);
      vmax = __riscv_vfmv_v_f_f32m1(-__builtin_inff(), vl);
    }

    for (; j + 16 <= n; j += 16) {
      vfloat32m1x8_t vx0 =
          __riscv_vlsseg8e32_v_f32m1x8(a + i * rsa + j + 0, bsa, vl);
      __asm__ volatile("" ::"vr"(vss)); // Prevent reordering, which
                                        // may cause spills
      vfloat32m1x8_t vx1 =
          __riscv_vlsseg8e32_v_f32m1x8(a + i * rsa + j + 8, bsa, vl);

      /* Determine row-wise block maximum */
      vfloat32m1_t vm = skl_softmax_vfredmax_v_f32m1x16_f32m1(vx0, vx1, vl);
      /* New global row-wise maximum */
      vm = __riscv_vfmax_vv_f32m1(vmax, vm, vl);

      vfloat32m1_t vd = __riscv_vfsub_vv_f32m1(vmax, vm, vl);
      vx0 = skl_softmax_vfsub_vv_f32m1x8(vx0, vm, vl);
      vx1 = skl_softmax_vfsub_vv_f32m1x8(vx1, vm, vl);
      if (beta != 1.0f) {
        vd = __riscv_vfmul_vf_f32m1(vd, beta, vl);
        vx0 = skl_softmax_vfmul_vf_f32m1x8(vx0, beta, vl);
        vx1 = skl_softmax_vfmul_vf_f32m1x8(vx1, beta, vl);
      }
      /* Calculate correction and exponential terms */
      vfloat32m1_t vc = __riscv_sf_vfexp_v_f32m1(vd, vl);
      vx0 = skl_softmax_vfexp_v_f32m1x8(vx0, vl);
      vx1 = skl_softmax_vfexp_v_f32m1x8(vx1, vl);

      /* Correct and accumulate partial sums */
      vfloat32m1x4_t ves =
          skl_softmax_vfredsum_v_f32m1x16_f32m1x4(vx0, vx1, vl);
      vss = skl_softmax_vfmadd_vv_f32m1x4(vss, vc, ves, vl);
      vmax = vm;
    }
    for (; j + 8 <= n; j += 8) {
      vfloat32m1x8_t vx =
          __riscv_vlsseg8e32_v_f32m1x8(a + i * rsa + j, bsa, vl);

      vfloat32m1_t vm = skl_softmax_vfredmax_v_f32m1x8_f32m1(vx, vl);
      vm = __riscv_vfmax_vv_f32m1(vmax, vm, vl);

      vfloat32m1_t vd = __riscv_vfsub_vv_f32m1(vmax, vm, vl);
      vx = skl_softmax_vfsub_vv_f32m1x8(vx, vm, vl);
      if (beta != 1.0f) {
        vd = __riscv_vfmul_vf_f32m1(vd, beta, vl);
        vx = skl_softmax_vfmul_vf_f32m1x8(vx, beta, vl);
      }
      vfloat32m1_t vc = __riscv_sf_vfexp_v_f32m1(vd, vl);
      vx = skl_softmax_vfexp_v_f32m1x8(vx, vl);

      vfloat32m1x4_t ves = skl_softmax_vfredsum_v_f32m1x8_f32m1x4(vx, vl);
      vss = skl_softmax_vfmadd_vv_f32m1x4(vss, vc, ves, vl);
      vmax = vm;
    }
    for (; j + 4 <= n; j += 4) {
      vfloat32m1x4_t vx =
          __riscv_vlsseg4e32_v_f32m1x4(a + i * rsa + j, bsa, vl);

      vfloat32m1_t vm = skl_softmax_vfredmax_v_f32m1x4_f32m1(vx, vl);
      vm = __riscv_vfmax_vv_f32m1(vmax, vm, vl);

      vfloat32m1_t vd = __riscv_vfsub_vv_f32m1(vmax, vm, vl);
      vx = skl_softmax_vfsub_vv_f32m1x4(vx, vm, vl);
      if (beta != 1.0f) {
        vd = __riscv_vfmul_vf_f32m1(vd, beta, vl);
        vx = skl_softmax_vfmul_vf_f32m1x4(vx, beta, vl);
      }
      vfloat32m1_t vc = __riscv_sf_vfexp_v_f32m1(vd, vl);
      vfloat32m1x4_t ves = skl_softmax_vfexp_v_f32m1x4(vx, vl);

      vss = skl_softmax_vfmadd_vv_f32m1x4(vss, vc, ves, vl);
      vmax = vm;
    }
    vfloat32m1x2_t vs2 = skl_softmax_vfredsum_v_f32m1x4_f32m1x2(vss, vl);
    for (; j + 2 <= n; j += 2) {
      vfloat32m1x2_t vx =
          __riscv_vlsseg2e32_v_f32m1x2(a + i * rsa + j, bsa, vl);

      vfloat32m1_t vm = skl_softmax_vfredmax_v_f32m1x2_f32m1(vx, vl);
      vm = __riscv_vfmax_vv_f32m1(vmax, vm, vl);

      vfloat32m1_t vd = __riscv_vfsub_vv_f32m1(vmax, vm, vl);
      vx = skl_softmax_vfsub_vv_f32m1x2(vx, vm, vl);
      if (beta != 1.0f) {
        vd = __riscv_vfmul_vf_f32m1(vd, beta, vl);
        vx = skl_softmax_vfmul_vf_f32m1x2(vx, beta, vl);
      }
      vfloat32m1_t vc = __riscv_sf_vfexp_v_f32m1(vd, vl);
      vfloat32m1x2_t ves = skl_softmax_vfexp_v_f32m1x2(vx, vl);

      vs2 = skl_softmax_vfmadd_vv_f32m1x2(vs2, vc, ves, vl);
      vmax = vm;
    }
    vfloat32m1_t vsum = skl_softmax_vfredsum_v_f32m1x2_f32m1(vs2, vl);
    for (; j + 1 <= n; ++j) {
      vfloat32m1_t vx = __riscv_vlse32_v_f32m1(a + i * rsa + j, bsa, vl);
      vfloat32m1_t vm = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
      vfloat32m1_t vd = __riscv_vfsub_vv_f32m1(vmax, vm, vl);
      vx = __riscv_vfsub_vv_f32m1(vx, vm, vl);
      if (beta != 1.0f) {
        vd = __riscv_vfmul_vf_f32m1(vd, beta, vl);
        vx = __riscv_vfmul_vf_f32m1(vx, beta, vl);
      }
      vfloat32m1_t vc = __riscv_sf_vfexp_v_f32m1(vd, vl);
      vfloat32m1_t ve = __riscv_sf_vfexp_v_f32m1(vx, vl);
      vsum = __riscv_vfmadd_vv_f32m1(vsum, vc, ve, vl);
      vmax = vm;
    }
    /* Row-wise global maximums are in `vmax`, and row-wise global
       exponential sums are in `vsum`. */

    /* Finalize results */
    skl_softmax_2d_olscal_f32m1_xsfvfexp32e(s + i * rss, rss, a + i * rsa, rsa,
                                            beta, vsum, vmax, vl, n);
  }
}

/* 2D Softmax. */
SKL_FUNC void skl_softmax_2d_f32_xsfvfexp32e(float *s, size_t rss,
                                             const float *a, size_t rsa,
                                             float beta, size_t m, size_t n) {
  size_t vlm1 = __riscv_vsetvlmax_e32m1();
  if ((n >= 6 * vlm1 && m < vlm1) || n >= 16 * vlm1)
    skl_softmax_2d_nvec_f32_xsfvfexp32e(s, rss, a, rsa, beta, m, n);
  else
    skl_softmax_2d_mvec_f32_xsfvfexp32e(s, rss, a, rsa, beta, m, n);
}
