// Copyright 2025 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the gemm_f16rc_f16rc_f32rc test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. rvv/skl_gemm_f16_f16_f32_zvfh_x390.c).
 */
#include "gemm_f16rc_f16rc_f32rc.h"
#include "skl-test-driver.h"
#include "skl.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

int gemm_f16rc_f16rc_f32rc_init(skl_test_t *t) {
  gemm_f16rc_f16rc_f32rc_t *h = (gemm_f16rc_f16rc_f32rc_t *)t->harness;

  // Set buffer lengths.
  // Handle zero dimensions to avoid size_t underflow in (dim - 1) expressions
  if (h->m == 0 || h->n == 0 || h->k == 0) {
    h->a.len = 0;
    h->b.len = 0;
  } else {
    h->a.len = (h->m - 1) * h->rsa + (h->k - 1) * h->csa + 1;
    h->b.len = (h->k - 1) * h->rsb + (h->n - 1) * h->csb + 1;
  }
  if (h->m == 0 || h->n == 0) {
    h->c.len = 0;
  } else {
    h->c.len = (h->m - 1) * h->rsc + (h->n - 1) * h->csc + 1;
  }

  SKL_TEST_BUF_CREATE(t, _Float16, &h->a);
  SKL_TEST_BUF_CREATE(t, _Float16, &h->b);
  SKL_TEST_BUF_CREATE(t, float, &h->c);

  if (h->verify) {
    // Only allocate if lengths are non-zero to avoid malloc(0)
    h->ctx.a_wide = h->a.len > 0 ? malloc(h->a.len * sizeof(double)) : NULL;
    h->ctx.b_wide = h->b.len > 0 ? malloc(h->b.len * sizeof(double)) : NULL;
    h->ctx.ref_c = h->c.len > 0 ? malloc(h->c.len * sizeof(float)) : NULL;
    h->ctx.bound = h->c.len > 0 ? malloc(h->c.len * sizeof(double)) : NULL;
    // Copy initial C values to ref_c for the beta*C term in reference GEMM
    memcpy(h->ctx.ref_c, h->c.data, h->c.len * sizeof(float));
  }

  return 0;
}

int gemm_f16rc_f16rc_f32rc_warmup(skl_test_t *t) {
  gemm_f16rc_f16rc_f32rc_t *h = (gemm_f16rc_f16rc_f32rc_t *)t->harness;
  if (h->warmup) {
    return t->suite->execute(t);
  }
  return 0;
}

int gemm_f16rc_f16rc_f32rc_verify(skl_test_t *t) {
  gemm_f16rc_f16rc_f32rc_t *h = (gemm_f16rc_f16rc_f32rc_t *)t->harness;

  if (!h->verify) {
    return 0;
  }

  for (size_t i = 0; i < h->a.len; ++i) {
    h->ctx.a_wide[i] = fabs((double)h->a.data[i]);
  }
  for (size_t i = 0; i < h->b.len; ++i) {
    h->ctx.b_wide[i] = fabs((double)h->b.data[i]);
  }
  for (size_t i = 0; i < h->c.len; ++i) {
    h->ctx.bound[i] = fabs((double)h->c.data[i]);
  }

  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K + 2) - 1) by change of base formula:
  const double roundoff_scaling = 2 * expm1((double)(h->k + 2) * log1p(u));
  skl_gemm_f64rc_f64rc_f64rc_scalar(
      h->m, h->n, h->k, roundoff_scaling * fabs((double)h->alpha),
      h->ctx.a_wide, h->rsa, h->csa, h->ctx.b_wide, h->rsb, h->csb,
      roundoff_scaling * fabs((double)h->beta), h->ctx.bound, h->rsc, h->csc);

  skl_gemm_f16rc_f16rc_f32rc_scalar(h->m, h->n, h->k, h->alpha, h->a.data,
                                    h->rsa, h->csa, h->b.data, h->rsb, h->csb,
                                    h->beta, h->ctx.ref_c, h->rsc, h->csc);

  for (size_t i = 0; i < h->m; ++i) {
    for (size_t j = 0; j < h->n; ++j) {
      size_t idx = i * h->rsc + j * h->csc;
      float res = h->c.data[idx];
      float ref = h->ctx.ref_c[idx];
      double bnd = h->ctx.bound[idx];
      if (fabs((double)res - (double)ref) > bnd) {
        skl_test_driver_error(
            t, "result [%zu, %zu] (%f) != reference (%f) [bound = %f]\n", i, j,
            res, ref, bnd);
        return 1; // Error
      }
    }
  }

  return 0;
}

int gemm_f16rc_f16rc_f32rc_report(skl_test_t *t) {
  gemm_f16rc_f16rc_f32rc_t *h = (gemm_f16rc_f16rc_f32rc_t *)t->harness;

  size_t maccs = h->m * h->n * h->k;
  float mpc = (float)maccs / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("M: %zd, N: %zd, K: %zd\n", h->m, h->n, h->k);
  INFO("RSA: %zd, CSA: %zd\n", h->rsa, h->csa);
  INFO("RSB: %zd, CSB: %zd\n", h->rsb, h->csb);
  INFO("RSC: %zd, CSC: %zd\n", h->rsc, h->csc);
  INFO("Alpha: %f, Beta: %f\n", (float)h->alpha, (float)h->beta);
  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("MACs/Cycle: %f\n", mpc);
#undef INFO

  return 0;
}

int gemm_f16rc_f16rc_f32rc_cleanup(skl_test_t *t) {
  gemm_f16rc_f16rc_f32rc_t *h = (gemm_f16rc_f16rc_f32rc_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->a);
  SKL_TEST_BUF_FREE(t, &h->b);
  SKL_TEST_BUF_FREE(t, &h->c);
  if (h->verify) {
    free(h->ctx.a_wide);
    free(h->ctx.b_wide);
    free(h->ctx.ref_c);
    free(h->ctx.bound);
  }
  return 0;
}
