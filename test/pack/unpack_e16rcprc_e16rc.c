// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the unpack_e16rcprc_e16rc test harness.
 *
 * This file defines all harness functions _except_ `execute`, which is
 * defined in the test file (e.g. rvv/skl_unpack_e16rcprc_e16rc_zve32x.c).
 */

#include "unpack_e16rcprc_e16rc.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl_test_pack.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void unpack_e16rcprc_e16rc_init(skl_test_t *t) {
  unpack_e16rcprc_e16rc_t *h = (unpack_e16rcprc_e16rc_t *)t->harness;

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

  skl_test_check_matrix_params_rcprc(t, m0, n0, m1, n1, rs0, cs0, rs1, cs1);
  skl_test_check_matrix_params_rcprc(t, 1, 1, m, n, 1, 1, rs, cs);

  if (t->status.init_status != SKL_TEST_PASS) {
    return;
  }

  if (m == 0 || n == 0) {
    h->src.len = 0;
  } else {
    h->src.len =
        (m1 - 1) * rs1 + (n1 - 1) * cs1 + (m0 - 1) * rs0 + (n0 - 1) * cs0 + 1;
  }

  if (m1 == 0 || n1 == 0) {
    h->dst.len = 0;
  } else {
    h->dst.len = (m - 1) * rs + (n - 1) * cs + 1;
  }

  SKL_TEST_BUF_CREATE(t, uint16_t, &h->src);
  SKL_TEST_BUF_CREATE(t, uint16_t, &h->dst);
  if (h->steps.verify && h->dst.len) {
    h->ctx.ref_dst = malloc(h->dst.len * sizeof(uint16_t));

    // Copy original `dst` contents into ref to check for clobbered data later.
    memcpy(h->ctx.ref_dst, h->dst.data, h->dst.len * sizeof(uint16_t));
  }
}
void unpack_e16rcprc_e16rc_verify(skl_test_t *t) {
  unpack_e16rcprc_e16rc_t *h = (unpack_e16rcprc_e16rc_t *)t->harness;

  const uint16_t *dst = h->dst.data;
  uint16_t *ref_dst = h->ctx.ref_dst;

  // Compute reference value
  skl_unpack_e16rcprc_e16rc_ref(h->m0, h->n0, h->src.data, h->rs0, h->cs0,
                                h->rs1, h->cs1, h->m, h->n, ref_dst, h->rs,
                                h->cs);

  // Verify result
  for (size_t i = 0; i < h->dst.len; ++i) {
    if (dst[i] != ref_dst[i]) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR, "position [%zu]: %hu != ref %hu\n", i,
                   dst[i], ref_dst[i]);
      t->status.verify_status = SKL_TEST_FAIL;
      return;
    }
  }
}

void unpack_e16rcprc_e16rc_test_report(skl_test_t *t) {
  unpack_e16rcprc_e16rc_t *h = (unpack_e16rcprc_e16rc_t *)t->harness;
  pack_rc_rcprc_report_param(t, h->m, h->n, h->rs, h->cs, h->m0, h->n0, h->rs0,
                             h->cs0, h->rs1, h->cs1);
}

void unpack_e16rcprc_e16rc_benchmark_report(skl_test_t *t) {

  unpack_e16rcprc_e16rc_t *h = (unpack_e16rcprc_e16rc_t *)t->harness;
  unpack_e16rcprc_e16rc_test_report(t);
  size_t output_elements = h->m * h->n;

  pack_rc_rcprc_report_perf(t, h->steps.warmup, output_elements);
}

void unpack_e16rcprc_e16rc_cleanup(skl_test_t *t) {
  unpack_e16rcprc_e16rc_t *h = (unpack_e16rcprc_e16rc_t *)t->harness;

  SKL_TEST_BUF_FREE(t, &h->src);
  SKL_TEST_BUF_FREE(t, &h->dst);
  if (h->steps.verify && h->dst.len) {
    free(h->ctx.ref_dst);
  }
}
