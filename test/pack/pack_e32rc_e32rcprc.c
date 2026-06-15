// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the pack_e32rc_e32rcprc test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/skl_pack_e32_zve32x.c).
 */

#include "pack_e32rc_e32rcprc.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl_test_pack.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void pack_e32rc_e32rcprc_init(skl_test_t *t) {
  pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;

  size_t m = h->m;
  size_t n = h->n;
  size_t rs = h->rs;
  size_t cs = h->cs;
  size_t m0 = h->m0;
  size_t n0 = h->n0;
  size_t m1 = (m + m0 - 1) / m0;
  size_t n1 = (n + n0 - 1) / n0;
  size_t rs0 = h->rs0;
  size_t cs0 = h->cs0;
  size_t rs1 = h->rs1;
  size_t cs1 = h->cs1;

  skl_test_check_matrix_params_rcprc(t, 1, 1, m, n, 1, 1, rs, cs);
  skl_test_check_matrix_params_rcprc(t, m0, n0, m1, n1, rs0, cs0, rs1, cs1);

  if (t->status.init_status != SKL_TEST_PASS) {
    return;
  }

  if (m == 0 || n == 0) {
    h->src.len = 0;
  } else {
    h->src.len = (m - 1) * rs + (n - 1) * cs + 1;
  }

  if (m1 == 0 || n1 == 0) {
    h->dst.len = 0;
  } else {
    h->dst.len =
        (m1 - 1) * rs1 + (n1 - 1) * cs1 + (m0 - 1) * rs0 + (n0 - 1) * cs0 + 1;
  }

  SKL_TEST_BUF_CREATE(t, uint32_t, &h->src);
  SKL_TEST_BUF_CREATE(t, uint32_t, &h->dst);
  if (h->steps.verify && h->dst.len) {
    h->ctx.ref_dst = malloc(h->dst.len * sizeof(uint32_t));

    // Copy original `dst` contents into ref to check for clobbered data later.
    memcpy(h->ctx.ref_dst, h->dst.data, h->dst.len * sizeof(uint32_t));
  }
}

void pack_e32rc_e32rcprc_verify(skl_test_t *t) {
  pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;

  const uint32_t *dst = h->dst.data;
  uint32_t *ref_dst = h->ctx.ref_dst;

  // Compute reference value
  skl_pack_e32rc_e32rcprc_ref(h->m, h->n, h->src.data, h->rs, h->cs, h->m0,
                              h->n0, ref_dst, h->rs0, h->cs0, h->rs1, h->cs1,
                              h->pad);

  // Verify result
  for (size_t i = 0; i < h->dst.len; ++i) {
    if (dst[i] != ref_dst[i]) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "position [%zu]: %u != ref %u\n", i,
                   dst[i], ref_dst[i]);
      t->status.verify_status = SKL_TEST_FAIL;
      return;
    }
  }
}

void pack_e32rc_e32rcprc_test_report(skl_test_t *t) {
  pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;
  pack_rc_rcprc_report_param(t, h->m, h->n, h->rs, h->cs, h->m0, h->n0, h->rs0,
                             h->cs0, h->rs1, h->cs1);
}

void pack_e32rc_e32rcprc_benchmark_report(skl_test_t *t) {
  pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;
  pack_e32rc_e32rcprc_test_report(t);
  size_t m1 = (h->m + h->m0 - 1) / h->m0;
  size_t n1 = (h->n + h->n0 - 1) / h->n0;
  size_t output_elements = m1 * n1 * h->m0 * h->n0;
  pack_rc_rcprc_report_perf(t, h->steps.warmup, output_elements);
}

void pack_e32rc_e32rcprc_cleanup(skl_test_t *t) {
  pack_e32rc_e32rcprc_t *h = (pack_e32rc_e32rcprc_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->src);
  SKL_TEST_BUF_FREE(t, &h->dst);
  if (h->steps.verify && h->dst.len) {
    free(h->ctx.ref_dst);
  }
}
