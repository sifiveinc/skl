// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Generic test harness for BF16 unary element-wise functions.
 *
 */

#include "elementwise/unary_bf16.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void unary_bf16_init(skl_test_t *t) {
  unary_bf16_t *h = (unary_bf16_t *)t->harness;
  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, __bf16, &h->in);
  h->ctx.out = malloc(h->in.len * sizeof(__bf16));
  if (h->steps.verify) {
    h->ctx.ref = malloc(h->in.len * sizeof(__bf16));
  }
  if (!h->ctx.out || (h->steps.verify && !h->ctx.ref))
    t->status.init_status = SKL_TEST_FAIL;
}

void unary_bf16_execute(skl_test_t *t) {
  unary_bf16_t *h = (unary_bf16_t *)t->harness;
  unary_func_bf16_t func = h->func;
  func(h->ctx.out, h->in.data, h->in.len);
}

void unary_bf16_verify(skl_test_t *t) {
  unary_bf16_t *h = (unary_bf16_t *)t->harness;

  // The ulp tolerance for this test.
  float tol = h->ctx.max_err;

  // Compute the reference result
  h->ref_func(h->ctx.ref, h->in.data, h->in.len);

  float ulp = 0.f;
  size_t errors = 0;
  for (size_t i = 0; i < h->in.len; ++i) {
    __bf16 x = h->in.data[i];
    __bf16 Y = h->ctx.out[i];
    __bf16 y = h->ctx.ref[i];
    float err = skl_abs_error_ulp_bf16(Y, y);
    if (err > tol) {
      if (errors++ >= h->ctx.max_errors)
        break;
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "[%4d]: in %15.2a; out %15.2a; ref %15.2a : %.4g ulp "
                   "[bound = %.4g ulp]\n",
                   i, (double)x, (double)Y, (double)y, err, tol);
    }
    ulp = fmaxf(err, ulp);
  }
  h->ctx.max_err = ulp;
  t->status.verify_status = !errors ? SKL_TEST_PASS : SKL_TEST_FAIL;
}

void unary_bf16_report(skl_test_t *t) {
  unary_bf16_t *h = (unary_bf16_t *)t->harness;

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

void unary_bf16_cleanup(skl_test_t *t) {
  unary_bf16_t *h = (unary_bf16_t *)t->harness;
  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->in);
  free(h->ctx.out);
  if (h->steps.verify)
    free(h->ctx.ref);
}
