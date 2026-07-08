// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "rmsnorm/rmsnorm_f32.h"
#include "skl-test-driver.h"
#include <math.h>
#include <stdlib.h>

void rmsnorm_f32_init(skl_test_t *t) {
  rmsnorm_f32_t *h = (rmsnorm_f32_t *)t->harness;

  // Set buffer lengths
  h->src.len = h->n;
  h->weight.len = h->n;
#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  // Allocate and initialize input buffers
  SKL_TEST_BUF_CREATE(t, float, &h->src);
  SKL_TEST_BUF_CREATE(t, float, &h->weight);

#undef INFO

  // Allocate output buffer
  h->ctx.dst = h->n ? malloc(h->n * sizeof(float)) : NULL;

  if (h->steps.verify) {
    // Allocate reference buffer
    h->ctx.ref_dst = h->n ? malloc(h->n * sizeof(double)) : NULL;
  }
}

typedef void (*skl_rmsnorm_f32_t)(float *, const float *, const float *, size_t,
                                  float, size_t);

void rmsnorm_f32_execute(skl_test_t *t) {
  const rmsnorm_f32_t *h = (rmsnorm_f32_t *)t->harness;
  skl_rmsnorm_f32_t fn = (skl_rmsnorm_f32_t)(h->func);
  if(h->do_scale)
    fn(h->ctx.dst, h->src.data, h->weight.data, h->rsc, h->epsilon, h->n);
  else
    fn(h->ctx.dst, h->src.data, NULL, h->rsc, h->epsilon, h->n);
}

void rmsnorm_f32_verify(skl_test_t *t) {
  // Hello world - placeholder verification
  // TODO: Implement verification logic
  rmsnorm_f32_t *h = (rmsnorm_f32_t *)t->harness;

  // The ulp tolerance for this test.
  float tol = h->ctx.max_err;
  skl_rmsnorm_f32_t fn_ref = (skl_rmsnorm_f32_t)(h->ref_func);

  // Compute the reference result
  if(h->do_scale)
    fn_ref(h->ctx.ref_dst, h->src.data, h->weight.data, h->rsc, h->epsilon, h->n);
  else
    fn_ref(h->ctx.ref_dst, h->src.data, NULL, h->rsc, h->epsilon, h->n);

  float ulp = 0.f;
  size_t errors = 0;

  for (size_t i = 0; i < h->n; ++i) {
    float x = h->src.data[i];
    float Y = h->ctx.dst[i];
    float y = h->ctx.ref_dst[i];
    float err = skl_abs_error_ulp_f32(Y, y);
    if (err > tol) {
      errors++;
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "[%4d]: src %15.6a; dst %15.6a; ref_dst %15.6a : %.4g ulp "
                   "[bound = %.4g ulp]\n",
                   i, x, Y, y, err, tol);
    }
    ulp = fmaxf(err, ulp);
  }
  h->ctx.max_err = ulp;
  t->status.verify_status = !errors ? SKL_TEST_PASS : SKL_TEST_FAIL;


  (void)t;
}

void rmsnorm_f32_report(skl_test_t *t) {
  // Hello world - placeholder report
  // TODO: Implement performance reporting

  rmsnorm_f32_t *h = (rmsnorm_f32_t *)t->harness;
#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("Function: %s\n", h->name);
  INFO("N: %zd\n", h->n);
  if (h->steps.verify) {
    INFO("Max ulp: %.3f\n", h->ctx.max_err);
  } else {
    INFO("Cycles: %zd\n", t->counters.cycles);
    INFO("Instructions: %zd\n", t->counters.instret);
    INFO("Elements/Cycle: %f\n", (float)h->n / t->counters.cycles);
  }
#undef INFO

  (void)t;
}

void rmsnorm_f32_cleanup(skl_test_t *t) {
  rmsnorm_f32_t *h = (rmsnorm_f32_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->src);
  SKL_TEST_BUF_FREE(t, &h->weight);

  free(h->ctx.dst);
  if (h->steps.verify) {
    free(h->ctx.ref_dst);
  }
}
