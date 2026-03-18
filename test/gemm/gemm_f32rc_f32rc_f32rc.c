// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the gemm_f32rc_f32rc_f32rc test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. rvv/skl_gemm_f32_f32_f32_zve32f_x390.c).
 */

#include "gemm_f32rc_f32rc_f32rc.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int gemm_f32rc_f32rc_f32rc_init(skl_test_t *t) {
  gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;

  // Allow default strides:
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

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, float, &h->a);
  SKL_TEST_BUF_CREATE(t, float, &h->b);
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

int gemm_f32rc_f32rc_f32rc_warmup(skl_test_t *t) {
  gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;
  if (h->warmup) {
    return t->suite->execute(t);
  }
  return 0;
}

int gemm_f32rc_f32rc_f32rc_verify(skl_test_t *t) {
  /* Compute the reference (scalar) matrix output. */
  gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;
  if (!h->verify) {
    return 0;
  }

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
  for (size_t i = 0; i < h->a.len; ++i) {
    h->ctx.a_wide[i] = fabsf(h->a.data[i]);
  }
  for (size_t i = 0; i < h->b.len; ++i) {
    h->ctx.b_wide[i] = fabsf(h->b.data[i]);
  }
  for (size_t i = 0; i < h->c.len; ++i) {
    h->ctx.bound[i] = fabsf(h->ctx.ref_c[i]);
  }
  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K + 2) - 1) by change of base formula:
  const double roundoff_scaling = 2 * expm1((double)(h->k + 2) * log1p(u));
  skl_gemm_f64rc_f64rc_f64rc_scalar(
      h->m, h->n, h->k, roundoff_scaling * fabs((double)h->alpha),
      h->ctx.a_wide, h->rsa, h->csa, h->ctx.b_wide, h->rsb, h->csb,
      roundoff_scaling * fabs((double)h->beta), h->ctx.bound, h->rsc, h->csc);

  // Compute the reference result using h->ctx.ref_c
  // h->ctx.ref_c contains the original C values (copied in skl_test_init)
  // After this call, h->ctx.ref_c will contain the reference result
  skl_gemm_f32rc_f32rc_f32rc_scalar(h->m, h->n, h->k, h->alpha, h->a.data,
                                    h->rsa, h->csa, h->b.data, h->rsb, h->csb,
                                    h->beta, h->ctx.ref_c, h->rsc, h->csc);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < h->m; ++i) {
    for (size_t j = 0; j < h->n; ++j) {
      size_t idx = i * h->rsc + j * h->csc;
      if (fabs((double)h->c.data[idx] - (double)h->ctx.ref_c[idx]) >
          h->ctx.bound[idx]) {
        skl_test_driver_error(
            t, "result [%zu, %zu] (%f) != reference (%f) [bound = %f]\n", i, j,
            h->c.data[idx], h->ctx.ref_c[idx], h->ctx.bound[idx]);
        return 1; // Error
      }
    }
  }

  return 0; // Success
}

int gemm_f32rc_f32rc_f32rc_report(skl_test_t *t) {
  gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;

  size_t maccs = h->m * h->n * h->k;
  float mpc = (float)maccs / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("M: %zd, N: %zd, K: %zd\n", h->m, h->n, h->k);
  INFO("CSA: %zd, RSB: %zd, RSC: %zd\n", h->csa, h->rsb, h->rsc);
  INFO("Alpha: %f, Beta: %f\n", h->alpha, h->beta);
  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("MACs/Cycle: %f\n", mpc);

  return 0;
}

int gemm_f32rc_f32rc_f32rc_cleanup(skl_test_t *t) {
  gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;

  // Free buffers
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
