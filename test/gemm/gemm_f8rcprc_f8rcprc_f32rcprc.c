// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the gemm_f8rcprc_f8rcprc_f32rcprc test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g.
 * xsfmm/skl_gemm_f8e4m3c_f8e4m3_f32_xsfmm32a8f.c).
 */

#include "gemm_f8rcprc_f8rcprc_f32rcprc.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl_test_gemm.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void gemm_f8rcprc_f8rcprc_f32rcprc_init(skl_test_t *t) {
  gemm_f8rcprc_f8rcprc_f32rcprc_t *h =
      (gemm_f8rcprc_f8rcprc_f32rcprc_t *)t->harness;

  size_t m0 = h->m0;
  size_t n0 = h->n0;
  size_t k0 = h->k0;
  size_t m1 = h->m1;
  size_t n1 = h->n1;
  size_t k1 = h->k1;
  size_t rsa0 = h->rsa0;
  size_t csa0 = h->csa0;
  size_t rsa1 = h->rsa1;
  size_t csa1 = h->csa1;
  size_t rsb0 = h->rsb0;
  size_t csb0 = h->csb0;
  size_t rsb1 = h->rsb1;
  size_t csb1 = h->csb1;
  size_t rsc0 = h->rsc0;
  size_t csc0 = h->csc0;
  size_t rsc1 = h->rsc1;
  size_t csc1 = h->csc1;

  skl_test_check_matrix_params_rcprc(t, m0, k0, m1, k1, rsa0, csa0, rsa1, csa1);
  skl_test_check_matrix_params_rcprc(t, k0, n0, k1, n1, rsb0, csb0, rsb1, csb1);
  skl_test_check_matrix_params_rcprc(t, m0, n0, m1, n1, rsc0, csc0, rsc1, csc1);

  if (t->status.init_status != SKL_TEST_PASS) {
    return;
  }

  if (m1 == 0 || k1 == 0) {
    h->a.len = 0;
  } else {
    h->a.len = (m1 - 1) * rsa1 + (k1 - 1) * csa1 + (m0 - 1) * rsa0 +
               (k0 - 1) * csa0 + 1;
  }

  if (k1 == 0 || n1 == 0) {
    h->b.len = 0;
  } else {
    h->b.len = (k1 - 1) * rsb1 + (n1 - 1) * csb1 + (k0 - 1) * rsb0 +
               (n0 - 1) * csb0 + 1;
  }

  if (m1 == 0 || n1 == 0) {
    h->c.len = 0;
  } else {
    h->c.len = (m1 - 1) * rsc1 + (n1 - 1) * csc1 + (m0 - 1) * rsc0 +
               (n0 - 1) * csc0 + 1;
  }

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, uint8_t, &h->a);
  SKL_TEST_BUF_CREATE(t, uint8_t, &h->b);
  SKL_TEST_BUF_CREATE(t, float, &h->c);
  if (h->steps.verify) {
    h->ctx.a_wide =
        h->a.len ? malloc(h->a.len * sizeof(*(h->ctx.a_wide))) : NULL;
    h->ctx.b_wide =
        h->b.len ? malloc(h->b.len * sizeof(*(h->ctx.b_wide))) : NULL;
    h->ctx.ref_c = h->c.len ? malloc(h->c.len * sizeof(*(h->ctx.ref_c))) : NULL;
    h->ctx.bound = h->c.len ? malloc(h->c.len * sizeof(*(h->ctx.bound))) : NULL;
    if (h->c.len) {
      memcpy(h->ctx.ref_c, h->c.data, h->c.len * sizeof(*(h->c.data)));
    }
  }
}

void gemm_f8rcprc_f8rcprc_f32rcprc_verify(skl_test_t *t) {
  gemm_f8rcprc_f8rcprc_f32rcprc_t *h =
      (gemm_f8rcprc_f8rcprc_f32rcprc_t *)t->harness;

  size_t m0 = h->m0;
  size_t n0 = h->n0;
  size_t k0 = h->k0;
  size_t m1 = h->m1;
  size_t n1 = h->n1;
  size_t k1 = h->k1;
  size_t rsa0 = h->rsa0;
  size_t csa0 = h->csa0;
  size_t rsa1 = h->rsa1;
  size_t csa1 = h->csa1;
  size_t rsb0 = h->rsb0;
  size_t csb0 = h->csb0;
  size_t rsb1 = h->rsb1;
  size_t csb1 = h->csb1;
  size_t rsc0 = h->rsc0;
  size_t csc0 = h->csc0;
  size_t rsc1 = h->rsc1;
  size_t csc1 = h->csc1;
  float alpha = h->alpha;
  float beta = h->beta;
  uint8_t *a = h->a.data;
  uint8_t *b = h->b.data;
  float *c = h->c.data;
  size_t a_len = h->a.len;
  size_t b_len = h->b.len;
  size_t c_len = h->c.len;
  double *a_wide = h->ctx.a_wide;
  double *b_wide = h->ctx.b_wide;
  float *ref_c = h->ctx.ref_c;
  double *bound = h->ctx.bound;

  //
  // Compute the error bound array for comparing test vs reference results.
  //
  // For any matrix M, define |M| as the matrix of absolute values where:
  //     |M|(i,j) = |M(i,j)|
  //
  // Let u = 2^-P be the maximum relative roundoff error for a floating-point
  // type with P-1 mantissa bits.
  //
  // For GEMM operations, the error between computed and exact results is
  // bounded by:
  //     ((1 + u)^(K + 2) - 1) * (|alpha| * |A| * |B| + |beta| * |C|)
  //
  // Since both test and reference results have roundoff errors, we double this
  // bound using the triangle inequality to get the final comparison threshold.
  //
  // Note: h->ctx.ref_c contains the original C values (copied in
  // skl_test_init) which are needed for the |beta| * |C| term in the error
  // bound.
  //
  float (*skl_cvt_f8_f32)(uint8_t);
  void (*skl_gemm_f8rcprc_f8rcprc_f32rcprc_ref)(
      size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
      float alpha, const uint8_t *a, size_t rsa0, size_t csa0, size_t rsa1,
      size_t csa1, const uint8_t *b, size_t rsb0, size_t csb0, size_t rsb1,
      size_t csb1, float beta, float *c, size_t rsc0, size_t csc0, size_t rsc1,
      size_t csc1);
  switch (h->input_type) {
  case F8E4M3:
    skl_cvt_f8_f32 = &skl_cvt_f8e4m3_f32;
    skl_gemm_f8rcprc_f8rcprc_f32rcprc_ref =
        &skl_gemm_f8e4m3rcprc_f8e4m3rcprc_f32rcprc_ref;
    break;
  case F8E5M2:
    skl_cvt_f8_f32 = &skl_cvt_f8e5m2_f32;
    skl_gemm_f8rcprc_f8rcprc_f32rcprc_ref =
        &skl_gemm_f8e5m2rcprc_f8e5m2rcprc_f32rcprc_ref;
    break;
  default:
    skl_cvt_f8_f32 = NULL;
    skl_gemm_f8rcprc_f8rcprc_f32rcprc_ref = NULL;
    break;
  }
  for (size_t i = 0; i < a_len; ++i) {
    a_wide[i] = fabsf((*skl_cvt_f8_f32)(a[i]));
  }
  for (size_t i = 0; i < b_len; ++i) {
    b_wide[i] = fabsf((*skl_cvt_f8_f32)(b[i]));
  }
  for (size_t i = 0; i < c_len; ++i) {
    bound[i] = fabsf(ref_c[i]);
  }
  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K1 * K0 + 2) - 1) by change of base formula:
  const double roundoff_scaling = 2 * expm1((double)(k1 * k0 + 2) * log1p(u));
  skl_gemm_f64rcprc_f64rcprc_f64rcprc_ref(
      m0, n0, k0, m1, n1, k1, roundoff_scaling * fabs((double)alpha), a_wide,
      rsa0, csa0, rsa1, csa1, b_wide, rsb0, csb0, rsb1, csb1,
      roundoff_scaling * fabs((double)beta), bound, rsc0, csc0, rsc1, csc1);

  // Compute the reference result using h->ctx.ref_c
  (*skl_gemm_f8rcprc_f8rcprc_f32rcprc_ref)(
      m0, n0, k0, m1, n1, k1, alpha, a, rsa0, csa0, rsa1, csa1, b, rsb0, csb0,
      rsb1, csb1, beta, ref_c, rsc0, csc0, rsc1, csc1);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < m1; ++i1) {
    for (size_t j1 = 0; j1 < n1; ++j1) {
      for (size_t i0 = 0; i0 < m0; ++i0) {
        for (size_t j0 = 0; j0 < n0; ++j0) {
          size_t idx = i1 * rsc1 + j1 * csc1 + i0 * rsc0 + j0 * csc0;
          if (fabs((double)c[idx] - (double)ref_c[idx]) > bound[idx]) {
            SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                         "result [%zu, %zu, %zu, %zu] (%f) != reference (%f) "
                         "[bound = %f]\n",
                         i1, j1, i0, j0, c[idx], ref_c[idx], bound[idx]);
            t->status.verify_status = SKL_TEST_FAIL;
            return;
          }
        }
      }
    }
  }

  /* Check for clobbered elements. */
  skl_test_check_matrix_clobbered_rcprc(t, sizeof(*c), c_len, m0, n0, m1, n1, c,
                                        ref_c, rsc0, csc0, rsc1, csc1);
}

void gemm_f8rcprc_f8rcprc_f32rcprc_test_report(skl_test_t *t) {
  gemm_f8rcprc_f8rcprc_f32rcprc_t *h =
      (gemm_f8rcprc_f8rcprc_f32rcprc_t *)t->harness;

  const char *input_type_str;
  switch (h->input_type) {
  case F8E4M3:
    input_type_str = "F8E4M3";
    break;
  case F8E5M2:
    input_type_str = "F8E5M2";
    break;
  default:
    input_type_str = "";
    break;
  }
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Input type: %s\n", input_type_str);
  gemm_rcprc_rcprc_rcprc_report_matrix_params(
      t, h->m0, h->n0, h->k0, h->m1, h->n1, h->k1, h->rsa0, h->csa0, h->rsa1,
      h->csa1, h->rsb0, h->csb0, h->rsb1, h->csb1, h->rsc0, h->csc0, h->rsc1,
      h->csc1);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Alpha: %f, Beta: %f\n", h->alpha,
               h->beta);
}

void gemm_f8rcprc_f8rcprc_f32rcprc_benchmark_report(skl_test_t *t) {
  gemm_f8rcprc_f8rcprc_f32rcprc_t *h =
      (gemm_f8rcprc_f8rcprc_f32rcprc_t *)t->harness;

  gemm_f8rcprc_f8rcprc_f32rcprc_test_report(t);

  size_t maccs = h->m1 * h->n1 * h->k1 * h->m0 * h->n0 * h->k0;
  gemm_rcprc_rcprc_rcprc_report_perf(t, h->steps.warmup, maccs);
}

void gemm_f8rcprc_f8rcprc_f32rcprc_cleanup(skl_test_t *t) {
  gemm_f8rcprc_f8rcprc_f32rcprc_t *h =
      (gemm_f8rcprc_f8rcprc_f32rcprc_t *)t->harness;

  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a);
  SKL_TEST_BUF_FREE(t, &h->b);
  SKL_TEST_BUF_FREE(t, &h->c);
  if (h->steps.verify) {
    free(h->ctx.a_wide);
    free(h->ctx.b_wide);
    free(h->ctx.ref_c);
    free(h->ctx.bound);
  }
}
