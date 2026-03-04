// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#if !defined(__riscv_xsfvfexp32e)
#error This file requires the Xsfvfexp32e extension
#endif

#include "skl-common.h"
#include <riscv_vector.h>
#include <sifive_vector.h>
#include <stddef.h>

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
SKL_FUNC void skl_softmax_2d_f32_xsfvfexp32e(float *s, const size_t rss,
                                             const float *a, const size_t rsa,
                                             const float beta, const size_t m,
                                             const size_t n) {
  size_t vl = __riscv_vsetvl_e32m8(n);
  if (vl == n) {
    skl_softmax_2d_f32m8_x3_xsfvfexp32e(s, rss, a, rsa, beta, m, vl);
  } else {
    for (size_t i = 0; i < m; ++i) {
      skl_softmax_f32_xsfvfexp32e(s + i * rss, a + i * rsa, beta, n);
    }
  }
}
