// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the cvt_f8_bf16 test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. cvt_f8_bf16_zvfofp8min.c).
 */

#include "cvt_f8_bf16.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cvt_f8_bf16_init(skl_test_t *t) {
  cvt_f8_bf16_t *h = (cvt_f8_bf16_t *)t->harness;
  h->in.len = h->len;
  h->out.len = h->len;

  if (h->len > 0) {
    SKL_TEST_BUF_CREATE(t, uint8_t, &h->in);
    SKL_TEST_BUF_CREATE(t, __bf16, &h->out);
  }
  if (h->steps.verify) {
    h->ref = h->len > 0 ? malloc(h->len * sizeof(__bf16)) : NULL;
  }
}

void cvt_f8_bf16_verify(skl_test_t *t) {
  cvt_f8_bf16_t *h = (cvt_f8_bf16_t *)t->harness;
  __bf16 *out = h->out.data;
  __bf16 *ref = h->ref;

  // Generate reference output using reference implementation
  if (h->in_type == F8E4M3) {
    skl_cvt_f8e4m3_bf16_ref(ref, h->in.data, h->len);
  } else { // F8E5M2
    skl_cvt_f8e5m2_bf16_ref(ref, h->in.data, h->len);
  }

  // Compare results
  for (size_t i = 0; i < h->len; ++i) {
    uint16_t out_bits;
    uint16_t ref_bits;
    memcpy(&out_bits, &out[i], sizeof(uint16_t));
    memcpy(&ref_bits, &ref[i], sizeof(uint16_t));

    if (out_bits != ref_bits) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "result [%zu] (%A) != reference (%A)\n", i, (float)out[i],
                   (float)ref[i]);
      t->status.verify_status = SKL_TEST_FAIL;
    }
  }
}

void cvt_f8_bf16_report(skl_test_t *t) {
  cvt_f8_bf16_t *h = (cvt_f8_bf16_t *)t->harness;

  size_t elements = h->len;
  float elements_per_cycle = (float)elements / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("Input type: %s\n", h->in_type == F8E4M3 ? "F8E4M3" : "F8E5M2");
  INFO("Output type: %s\n", "BF16");
  INFO("Length: %zd\n", h->len);
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("Elements/Cycle: %f\n", elements_per_cycle);
#undef INFO
}

void cvt_f8_bf16_cleanup(skl_test_t *t) {
  cvt_f8_bf16_t *h = (cvt_f8_bf16_t *)t->harness;

  if (h->len > 0) {
    SKL_TEST_BUF_FREE(t, &h->in);
    SKL_TEST_BUF_FREE(t, &h->out);
    if (h->steps.verify)
      free(h->ref);
  }
}
