// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Generic test harness for f32 unary element-wise functions.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. skl_exp_f32_zve32f.c).
 */

#include "math/unary_f32.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void unary_f32_init(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;
  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, float, &h->a);
  h->ctx.b = malloc(h->a.len * sizeof(float));
  if (!h->ctx.b)
    t->status.init_status = SKL_TEST_FAIL;
  if (h->steps.verify) {
    h->ctx.ref = malloc(h->a.len * sizeof(float));
    if (!h->ctx.ref)
      t->status.init_status = SKL_TEST_FAIL;
  }
}

void unary_f32_execute(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;
  unary_func_t func = h->func;
  func(h->ctx.b, h->a.data, h->a.len);
}

void unary_f32_verify(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;

  // The ulp tolerance for this test.
  float max = h->ctx.max_err;

  // Compute the reference result
  skl_exp_f32_ref(h->ctx.ref, h->a.data, h->a.len);

  float ulp = 0.f;
  for (size_t i = 0; i < h->a.len; ++i) {
    float Y = h->ctx.b[i];
    float y = h->ctx.ref[i];
    float err = skl_abs_error_ulp_f32(Y, y);
    if (err > max)
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "[%4d]: %15.6a %15.6a : %.4g ulp [bound = %.4g ulp]\n", i, Y,
                   y, err, max);
    ulp = fmaxf(err, ulp);
  }
  h->ctx.max_err = ulp;
  t->status.verify_status = (ulp <= max) ? SKL_TEST_PASS : SKL_TEST_FAIL;
}

void unary_f32_report(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  if (h->steps.verify) {
    INFO("N: %zd\n", h->a.len);
    INFO("Max ulp: %.3f\n", h->ctx.max_err);
  }

  if (h->steps.warmup) {
    INFO("N: %zd\n", h->a.len);
    INFO("Cycles: %zd\n", t->counters.cycles);
    INFO("Instructions: %zd\n", t->counters.instret);
    INFO("Elements/Cycle: %f\n", (float)h->a.len / t->counters.cycles);
  }

#undef INFO
}

void unary_f32_cleanup(skl_test_t *t) {
  unary_f32_t *h = (unary_f32_t *)t->harness;
  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a);
  free(h->ctx.b);
  if (h->steps.verify)
    free(h->ctx.ref);
}
