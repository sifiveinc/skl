// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Generic test harness for FP16 unary element-wise functions.
 *
 */

#include "elementwise/unary_f16.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void unary_f16_init(skl_test_t *t) {
  unary_f16_t *h = (unary_f16_t *)t->harness;
  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, _Float16, &h->in);
  h->ctx.out = malloc(h->in.len * sizeof(_Float16));
  if (h->steps.verify) {
    h->ctx.ref = malloc(h->in.len * sizeof(_Float16));
  }
  if (!h->ctx.out || (h->steps.verify && !h->ctx.ref))
    t->status.init_status = SKL_TEST_FAIL;
}

void unary_f16_execute(skl_test_t *t) {
  unary_f16_t *h = (unary_f16_t *)t->harness;
  unary_func_f16_t func = h->func;
  func(h->ctx.out, h->in.data, h->in.len);
}

void unary_f16_verify(skl_test_t *t) {
  unary_f16_t *h = (unary_f16_t *)t->harness;

  // The ulp tolerance for this test.
  float tol = h->ctx.max_err;

  // Compute the reference result
  h->ref_func(h->ctx.ref, h->in.data, h->in.len);

  float ulp = 0.f;
  size_t errors = 0;
  for (size_t i = 0; i < h->in.len; ++i) {
    _Float16 x = h->in.data[i];
    _Float16 Y = h->ctx.out[i];
    _Float16 y = h->ctx.ref[i];
    float err = skl_abs_error_ulp_f16(Y, y);
    if (err > tol) {
      if (errors++ >= h->ctx.max_errors)
        break;
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "[%4d]: in %15.3a; out %15.3a; ref %15.3a : %.4g ulp "
                   "[bound = %.4g ulp]\n",
                   i, (double)x, (double)Y, (double)y, err, tol);
    }
    ulp = fmaxf(err, ulp);
  }
  h->ctx.max_err = ulp;
  t->status.verify_status = !errors ? SKL_TEST_PASS : SKL_TEST_FAIL;
}

void unary_f16_report(skl_test_t *t) {
  unary_f16_t *h = (unary_f16_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Function: %s\n", h->func_name);
  INFO("N: %zd\n", h->in.len);
  if (h->steps.verify) {
    INFO("Max ulp: %.3f\n", h->ctx.max_err);
  } else {
    INFO("Cycles: %zd\n", t->counters.cycles);
    INFO("Instructions: %zd\n", t->counters.instret);
    INFO("Elements/Cycle: %f\n", (float)h->in.len / t->counters.cycles);
  }

#undef INFO
}

void unary_f16_cleanup(skl_test_t *t) {
  unary_f16_t *h = (unary_f16_t *)t->harness;
  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->in);
  free(h->ctx.out);
  if (h->steps.verify)
    free(h->ctx.ref);
}
