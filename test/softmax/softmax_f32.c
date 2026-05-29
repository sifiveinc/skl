// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "softmax/softmax_f32.h"
#include "skl-test-driver.h"
#include <math.h>
#include <stdlib.h>

void softmax_f32_init(skl_test_t *t) {
  softmax_f32_t *h = (softmax_f32_t *)t->harness;
  size_t slen = 0;

  // Set default row strides, if necessary
  if (h->rsa == 0)
    h->rsa = h->n;
  if (h->rss == 0)
    h->rss = h->rsa;
  if (h->m == 0 || h->n == 0) {
    h->a.len = 0;
  } else {
    h->a.len = (h->m - 1) * h->rsa + h->n;
    slen = (h->m - 1) * h->rss + h->n;
  }

  // Allocate and initialize input buffer
  SKL_TEST_BUF_CREATE(t, float, &h->a);
  // Allocate output buffer, avoiding malloc(0)
  h->ctx.s = slen ? malloc(slen * sizeof(float)) : NULL;

  if (h->steps.verify) {
    // Allocate reference buffer
    h->ctx.S = h->n ? malloc(h->n * sizeof(double)) : NULL;
  }
}

typedef void (*skl_softmax_f32_t)(float *, const float *, const float,
                                  const size_t);

void softmax_f32_execute(skl_test_t *t) {
  const softmax_f32_t *h = (softmax_f32_t *)t->harness;
  skl_softmax_f32_t fn = (skl_softmax_f32_t)(h->func);
  fn(h->ctx.s, h->a.data, h->beta, h->n);
}

typedef void (*skl_softmax_2d_f32_t)(float *, size_t, const float *, size_t,
                                     const float, size_t, size_t);

void softmax_2d_f32_execute(skl_test_t *t) {
  const softmax_f32_t *h = (softmax_f32_t *)t->harness;
  skl_softmax_2d_f32_t fn = (skl_softmax_2d_f32_t)(h->func);
  fn(h->ctx.s, h->rss, h->a.data, h->rsa, h->beta, h->m, h->n);
}

/** Reference softmax producing FP64 results for high accuracy */
static void softmax_f32_f64(double *out, const float *in, float beta,
                            size_t n) {
  if (n < 1)
    return;

  float max = in[0];
  for (size_t i = 1; i < n; i++) {
    max = fmaxf(in[i], max);
  }

  double sum = 0;
  for (size_t i = 0; i < n; i++) {
    out[i] = exp(beta * ((double)in[i] - max));
    sum += out[i];
  }

  for (size_t i = 0; i < n; i++) {
    out[i] /= sum;
  }
}

void softmax_f32_verify(skl_test_t *t) {
  softmax_f32_t *h = (softmax_f32_t *)t->harness;
  const size_t M = h->m, N = h->n;
  const float min = h->a.min, max = h->a.max;
  const float beta = h->beta;
  size_t errs = 0;

  // Expected error is error of an N-element summation, i.e. `N u`,
  // plus error introduced by stabilization and beta-scaling.  The
  // latter subtraction and multiplication would alone introduce just
  // `3 u` error, but this is inflated to `3β(min-max)` when passed
  // through the exponential.
  double u = 0x1p-24; // "unit round-off"
  double tol = (3 * fmax(beta * fabsf(max - min), 1) + (double)N) * u;
  tol = tol / (1 + tol); // relative to reference

  for (size_t m = 0; m < M; m++) {
    // Calculate reference result
    softmax_f32_f64(h->ctx.S, h->a.data + m * h->rsa, beta, N);

    for (size_t n = 0; n < N; n++) {
      float inp = h->a.data[m * h->rsa + n];
      float val = h->ctx.s[m * h->rss + n];
      double ref = h->ctx.S[n];
      double err = 0;

      if (isnan(val) != isnan(ref)) {
        SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                     "[%4zd,%4zd : %-16.6a]: %16.6a != ref %a (NaN mismatch)\n",
                     m, n, inp, val, ref);
        h->ctx.max_err = INFINITY;
        errs++;
      } else if (isnan(ref)) {
        continue; // both are NaN, no error
      } else if (ref > 0x1p-126) {
        err = fabs(val - ref); // absolute error
        err /= ref;            // relative error, ref always unsigned
        if (err > tol) {
          SKL_TEST_LOG(
              t, SKL_TEST_LOG_ERROR,
              "[%4zd,%4zd : %-16.6a]: %16.6a !~ ref %a (%.2a rel err > %.2a)\n",
              m, n, inp, val, ref, err, tol);
          errs++;
        }
        h->ctx.max_err = (float)fmax(h->ctx.max_err, err);
      } // else don't bother.  Many Softmax flush tiny results to
        // zero, and relative error is problematic in the tiny realm.
      if (errs >= 10)
        break;
    }
    if (errs >= 10)
      break;
  }
  t->status.verify_status =
      (h->ctx.max_err <= tol) ? SKL_TEST_PASS : SKL_TEST_FAIL;
}

void softmax_f32_report(skl_test_t *t) {
  softmax_f32_t *h = (softmax_f32_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Function: %s\n", h->name);
  INFO("M: %zd, N: %zd\n", h->m, h->n);
  if (h->rss != h->n || h->rsa != h->n)
    INFO("RSS: %zd, RSA: %zd\n", h->rss, h->rsa);
  INFO("Beta: %g\n", h->beta);
  if (h->steps.verify != NULL) {
    INFO("Domain: [%g ; %g]\n", h->a.min, h->a.max);
    INFO("Max rel: %.3a\n", h->ctx.max_err);
  } else {
    INFO("Cycles: %zd\n", t->counters.cycles);
    INFO("Instructions: %zd\n", t->counters.instret);
    INFO("Elements/Cycle: %f\n", (float)(h->m * h->n) / t->counters.cycles);
  }
#undef INFO
}

void softmax_f32_cleanup(skl_test_t *t) {
  softmax_f32_t *h = (softmax_f32_t *)t->harness;

  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a);
  free(h->ctx.s);
  if (h->steps.verify)
    free(h->ctx.S);
}
