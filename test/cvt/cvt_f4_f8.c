// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the cvt_f4_f8 test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. cvt_f4_f8_zvfofp4min.c).
 */

#include "cvt_f4_f8.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cvt_f4_f8_init(skl_test_t *t) {
  cvt_f4_f8_t *h = (cvt_f4_f8_t *)t->harness;
  h->in.len = (h->len + 1) / 2;
  h->out.len = h->len;

  if (h->len > 0) {
    SKL_TEST_BUF_CREATE(t, uint8_t, &h->in);
    SKL_TEST_BUF_CREATE(t, uint8_t, &h->out);
  }
  if (h->steps.verify) {
    h->ref = h->len > 0 ? malloc(h->len * sizeof(uint8_t)) : NULL;
  }
}

void cvt_f4_f8_verify(skl_test_t *t) {
  cvt_f4_f8_t *h = (cvt_f4_f8_t *)t->harness;
  uint8_t *out = h->out.data;
  uint8_t *ref = h->ref;

  // Generate reference output using reference implementation
  skl_cvt_f4e2m1_f8e4m3_ref(ref, h->in.data, h->len);

  // Compare results
  for (size_t i = 0; i < h->len; ++i) {
    if (out[i] != ref[i]) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "result [%zu] (%d) != reference (%d)\n", i, out[i], ref[i]);
      t->status.verify_status = SKL_TEST_FAIL;
    }
  }
}

void cvt_f4_f8_report(skl_test_t *t) {
  cvt_f4_f8_t *h = (cvt_f4_f8_t *)t->harness;

  size_t elements = h->len;
  float elements_per_cycle = (float)elements / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Input type: %s\n", "F4E2M1");
  INFO("Output type: %s\n", "F8E4M3");
  INFO("Length: %zd\n", h->len);
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("Elements/Cycle: %f\n", elements_per_cycle);
#undef INFO
}

void cvt_f4_f8_cleanup(skl_test_t *t) {
  cvt_f4_f8_t *h = (cvt_f4_f8_t *)t->harness;
  if (h->len > 0) {
    SKL_TEST_BUF_FREE(t, &h->in);
    SKL_TEST_BUF_FREE(t, &h->out);
    if (h->steps.verify)
      free(h->ref);
  }
}
