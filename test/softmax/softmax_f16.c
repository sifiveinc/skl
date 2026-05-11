// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "softmax/softmax_f16.h"
#include "skl-test-driver.h"
#include <math.h>
#include <stdlib.h>

void softmax_f16_init(skl_test_t *t) {
  softmax_f16_t *h = (softmax_f16_t *)t->harness;

  // Allocate and initialize input buffer
  h->a.len = h->n;
  SKL_TEST_BUF_CREATE(t, _Float16, &h->a);
  // Allocate output buffer, avoiding malloc(0)
  h->ctx.s = h->n ? malloc(h->n * sizeof(_Float16)) : NULL;

  if (h->steps.verify) {
    // Allocate reference buffer
    h->ctx.S = h->n ? malloc(h->n * sizeof(double)) : NULL;
  }
}

typedef void (*skl_softmax_f16_t)(_Float16 *, const _Float16 *, const _Float16,
                                  const size_t);

void softmax_f16_execute(skl_test_t *t) {
  const softmax_f16_t *h = (softmax_f16_t *)t->harness;
  skl_softmax_f16_t fn = (skl_softmax_f16_t)(h->func);
  fn(h->ctx.s, h->a.data, h->beta, h->n);
}

/** Reference softmax producing FP64 results for high accuracy */
static void softmax_f16_f64(double *out, const _Float16 *in, _Float16 beta,
                            size_t n) {
  if (n < 1)
    return;

  _Float16 max = in[0];
  for (size_t i = 1; i < n; i++) {
    max = (_Float16)fmaxf(in[i], max);
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

void softmax_f16_verify(skl_test_t *t) {
  softmax_f16_t *h = (softmax_f16_t *)t->harness;
  const size_t N = h->n;
  const _Float16 min = h->a.min, max = h->a.max;
  const _Float16 beta = h->beta;
  size_t errs = 0;

  // Expected error is error of an N-element summation, i.e. `N u`,
  // plus error introduced by stabilization and beta-scaling.  The
  // latter subtraction and multiplication would alone introduce just
  // `3 u` error, but this is inflated to `3β(min-max)` when passed
  // through the exponential.
  double u = 0x1p-11; // "unit round-off"
  double tol = (3 * fmax(beta * fabsf(max - min), 1) + (double)N) * u;
  tol = tol / (1 + tol); // relative to reference

  // Calculate reference result
  softmax_f16_f64(h->ctx.S, h->a.data, beta, N);

  for (size_t n = 0; n < N; n++) {
    _Float16 inp = h->a.data[n];
    _Float16 val = h->ctx.s[n];
    double ref = h->ctx.S[n];
    double err = 0;

    if (isnan(val) != isnan(ref)) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "[%4zd : %-16.6a]: %16.6a != ref %a (NaN mismatch)\n", n,
                   (float)inp, (float)val, ref);
      h->ctx.max_err = INFINITY;
      errs++;
    } else if (isnan(ref)) {
      continue; // both are NaN, no error
    } else if (ref > 0x1p-14) {
      err = fabs(val - ref); // absolute error
      err /= ref;            // relative error, ref always unsigned
      if (err > tol) {
        SKL_TEST_LOG(
            t, SKL_TEST_LOG_ERROR,
            "[%4zd : %-16.6a]: %16.6a !~ ref %a (%.2a rel err > %.2a)\n", n,
            (float)inp, (float)val, ref, err, tol);
        errs++;
      }
      h->ctx.max_err = (float)fmax(h->ctx.max_err, err);
    } // else don't bother.  Many Softmax flush tiny results to
    // zero, and relative error is problematic in the tiny realm.
    if (errs >= 10)
      break;
  }
  t->status.verify_status =
      (h->ctx.max_err <= tol) ? SKL_TEST_PASS : SKL_TEST_FAIL;
}

void softmax_f16_report(skl_test_t *t) {
  softmax_f16_t *h = (softmax_f16_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Function: %s\n", h->name);
  INFO("N: %zd\n", h->n);
  INFO("Beta: %g\n", (float)h->beta);
  if (h->steps.verify != NULL) {
    INFO("Domain: [%g ; %g]\n", (float)h->a.min, (float)h->a.max);
    INFO("Max rel: %.3a\n", h->ctx.max_err);
  } else {
    INFO("Cycles: %zd\n", t->counters.cycles);
    INFO("Instructions: %zd\n", t->counters.instret);
    INFO("Elements/Cycle: %f\n", (double)h->n / t->counters.cycles);
  }
#undef INFO
}

void softmax_f16_cleanup(skl_test_t *t) {
  softmax_f16_t *h = (softmax_f16_t *)t->harness;

  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a);
  free(h->ctx.s);
  if (h->steps.verify)
    free(h->ctx.S);
}
