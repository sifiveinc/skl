// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.

/**
 * @brief Implementation of the cvt_f16_f8 test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. cvt_bf16_f8_zvfofp8min_zvfbfmin.c).
 */

#include "cvt_bf16_f8.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cvt_bf16_f8_init(skl_test_t *t) {
  cvt_bf16_f8_t *h = (cvt_bf16_f8_t *)t->harness;
  h->in.len = h->len;
  h->out.len = h->len;
  SKL_TEST_BUF_CREATE(t, __bf16, &h->in);

  size_t count = h->len < 3 ? h->len : 3;
  if (count > 0)
    h->in.data[0] = (__bf16)nanf("");
  if (count > 1)
    h->in.data[1] = (__bf16)INFINITY;
  if (count > 2)
    h->in.data[2] = (__bf16)-INFINITY;

  if (h->out.len > 0) {
    SKL_TEST_BUF_CREATE(t, uint8_t, &h->out);
  }
  if (h->steps.verify) {
    h->ref = h->len > 0 ? malloc(h->len * sizeof(uint8_t)) : NULL;
  }
}

void cvt_bf16_f8_verify(skl_test_t *t) {
  cvt_bf16_f8_t *h = (cvt_bf16_f8_t *)t->harness;
  uint8_t *out = h->out.data;
  uint8_t *ref = h->ref;

  // Generate reference output using reference implementation
  if (h->out_type == F8E4M3) {
    if (h->saturation) {
      skl_cvt_sat_bf16_f8e4m3_ref(ref, h->in.data, h->scale, h->len);
    } else {
      skl_cvt_bf16_f8e4m3_ref(ref, h->in.data, h->scale, h->len);
    }
  } else { // F8E5M2
    if (h->saturation) {
      skl_cvt_sat_bf16_f8e5m2_ref(ref, h->in.data, h->scale, h->len);
    } else {
      skl_cvt_bf16_f8e5m2_ref(ref, h->in.data, h->scale, h->len);
    }
  }

  // Compare results
  for (size_t i = 0; i < h->len; ++i) {
    if (out[i] != ref[i]) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "result [%zu] (%d) != reference (%d)\n", i, out[i], ref[i]);
      t->status.verify_status = SKL_TEST_FAIL;
    }
  }
}

void cvt_bf16_f8_report(skl_test_t *t) {
  cvt_bf16_f8_t *h = (cvt_bf16_f8_t *)t->harness;

  size_t elements = h->len;
  float elements_per_cycle = (float)elements / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Input type: %s\n", "BF16");
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

void cvt_bf16_f8_cleanup(skl_test_t *t) {
  cvt_bf16_f8_t *h = (cvt_bf16_f8_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->in);
  if (h->len > 0) {
    SKL_TEST_BUF_FREE(t, &h->out);
    if (h->steps.verify)
      free(h->ref);
  }
}
