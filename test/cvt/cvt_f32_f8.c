// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the cvt_f32_f8 test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. cvt_f32_f8_zvfofp8min.c).
 */

#include "cvt_f32_f8.h"
#include "skl-test-driver.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "skl-ref.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cvt_f32_f8_init(skl_test_t *t) {
  cvt_f32_f8_t *h = (cvt_f32_f8_t *)t->harness;
  h->in.len = h->len;
  h->out.len = h->len;
  SKL_TEST_BUF_CREATE(t, float, &h->in);
  size_t count = h->len < 3 ? h->len : 3;
  if (count > 0)
    h->in.data[0] = nanf("");
  if (count > 1)
    h->in.data[1] = INFINITY;
  if (count > 2)
    h->in.data[2] = -INFINITY;

  if (h->out.len > 0) {
    SKL_TEST_BUF_CREATE(t, uint8_t, &h->out);
  }
  if (h->steps.verify) {
    h->ref = h->len > 0 ? malloc(h->len * sizeof(uint8_t)) : NULL;
  }
}

void cvt_f32_f8_verify(skl_test_t *t) {
  cvt_f32_f8_t *h = (cvt_f32_f8_t *)t->harness;
  uint8_t *out = h->out.data;
  uint8_t *ref = h->ref;

  for (size_t i = 0; i < h->len; ++i) {
    ref[i] = (h->out_type == F8E4M3)
                 ? skl_cvt_f32_f8e4m3(h->in.data[i] * h->scale, h->saturation)
                 : skl_cvt_f32_f8e5m2(h->in.data[i] * h->scale, h->saturation);
  }

  for (size_t i = 0; i < h->len; ++i) {
    if (out[i] != ref[i]) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "result [%zu] (%d) != reference (%d)\n", i, out[i], ref[i]);
      t->status.verify_status = SKL_TEST_FAIL;
    }
  }
}

void cvt_f32_f8_report(skl_test_t *t) {
  cvt_f32_f8_t *h = (cvt_f32_f8_t *)t->harness;

  size_t elements = h->len;
  float elements_per_cycle = (float)elements / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Input type: %s\n", "F32");
  INFO("Output type: %s\n", h->out_type == F8E4M3 ? "F8E4M3" : "F8E5M2");
  INFO("Saturation: %s\n", h->saturation ? "yes" : "no");
  INFO("Scaling factor: %s\n", h->scale != 1.0f ? "!= 1.0f" : "1.0f");
  INFO("Length: %zd\n", h->len);
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("Elements/Cycle: %f\n", elements_per_cycle);
#undef INFO
}

void cvt_f32_f8_cleanup(skl_test_t *t) {
  cvt_f32_f8_t *h = (cvt_f32_f8_t *)t->harness;
  SKL_TEST_BUF_FREE(t, &h->in);
  if (h->len > 0) {
    SKL_TEST_BUF_FREE(t, &h->out);
    if (h->steps.verify)
      free(h->ref);
  }
}
