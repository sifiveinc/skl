// Copyright (c) 2026-Present SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the transpose_e8 test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/skl_transpose_e8_zve32x.c).
 */

#include "transpose_e8.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void transpose_e8_init(skl_test_t *t) {
  transpose_e8_t *h = (transpose_e8_t *)t->harness;

  h->a.len = h->m * h->rsa;
  h->at.len = h->n * h->rsat;

  SKL_TEST_REQUIRE(t, init_status, h->rsa >= h->n);
  SKL_TEST_REQUIRE(t, init_status, h->rsat >= h->m);
  if (t->status.init_status != SKL_TEST_PASS)
    return;

  SKL_TEST_BUF_CREATE(t, uint8_t, &h->a);
  SKL_TEST_BUF_CREATE(t, uint8_t, &h->at);
  if (h->steps.verify && h->at.len) {
    h->ctx.ref_at = malloc(h->at.len * sizeof(uint8_t));

    // Copy original `at` contents into ref to check for clobbered data later.
    memcpy(h->ctx.ref_at, h->at.data, h->at.len * sizeof(uint8_t));
  }
}

void transpose_e8_verify(skl_test_t *t) {
  transpose_e8_t *h = (transpose_e8_t *)t->harness;

  size_t m = h->m;
  size_t n = h->n;
  size_t rsa = h->rsa;
  size_t rsat = h->rsat;
  const uint8_t *a = h->a.data;
  const uint8_t *at = h->at.data;
  uint8_t *ref_at = h->ctx.ref_at;

  // Compute reference value
  skl_transpose_e8_ref(m, n, a, rsa, ref_at, rsat);

  // Verify result
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < rsat; ++j) {
      size_t idx = i * rsat + j;
      if (at[idx] != ref_at[idx]) {
        SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                     "position [%zu, %zu]: %hhu != ref %hhu%s\n", i, j, at[idx],
                     ref_at[idx], j > m ? " (clobbered)" : "");
        t->status.verify_status = SKL_TEST_FAIL;
        return;
      }
    }
  }
}

void transpose_e8_report(skl_test_t *t) {
  transpose_e8_t *h = (transpose_e8_t *)t->harness;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)
  INFO("M: %zd, N: %zd\n", h->m, h->n);
  INFO("RSA: %zd, RSAT: %zd\n", h->rsa, h->rsat);
  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
#undef INFO
}

void transpose_e8_cleanup(skl_test_t *t) {
  transpose_e8_t *h = (transpose_e8_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->a);
  SKL_TEST_BUF_FREE(t, &h->at);
  if (h->steps.verify && h->at.len) {
    free(h->ctx.ref_at);
  }
}
