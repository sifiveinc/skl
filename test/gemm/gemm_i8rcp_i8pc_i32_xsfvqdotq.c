// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the gemm_i8rcp_i8pc_i32_xsfvqdotq test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g.
 * xsfvqdotq/skl_gemm_i8rcp_i8pc_i32_xsfvqdotq.c).
 */

#include "gemm_i8rcp_i8pc_i32_xsfvqdotq.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void skl_gemm_i8rcp_i8pc_i32_xsfvqdotq_ref(
    size_t m, size_t n, size_t k, int32_t alpha, const int8_t *a_pack,
    size_t rsa1, size_t csa1, const int8_t *b_pack, size_t rsb1, int32_t beta,
    int32_t *c, size_t rsc) {
  for (size_t ii1 = 0; ii1 < m; ++ii1) {
    for (size_t jj1 = 0; jj1 < n; ++jj1) {
      int32_t acc = 0;
      for (size_t kk1 = 0; kk1 < (k + 3) / 4; ++kk1) {
        const int8_t *ap_block = a_pack + ii1 * rsa1 + kk1 * csa1;
        const int8_t *bp_block = b_pack + kk1 * rsb1 + jj1 * 4;
        size_t k0 = (k - kk1 * 4) >= 4 ? 4 : (k - kk1 * 4);
        for (size_t kk0 = 0; kk0 < k0; ++kk0) {
          acc += ap_block[kk0] * bp_block[kk0];
        }
      }
      c[ii1 * rsc + jj1] = beta * c[ii1 * rsc + jj1] + alpha * acc;
    }
  }
}

void gemm_i8rcp_i8pc_i32_xsfvqdotq_init(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  size_t k1 = (h->k + 3) / 4;

  if (h->rsa1 > h->csa1) {
    h->a_pack.len = h->m * h->rsa1;
  } else if (h->rsa1 < h->csa1) {
    h->a_pack.len = k1 * h->csa1;
  } else {
    h->a_pack.len = (h->m >= k1 ? h->m : k1) * h->rsa1;
  }
  h->b_pack.len = k1 * h->rsb1;
  h->c.len = h->m * h->rsc;

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, int8_t, &h->a_pack);
  SKL_TEST_BUF_CREATE(t, int8_t, &h->b_pack);
  SKL_TEST_BUF_CREATE(t, int32_t, &h->c);
  if (h->steps.verify) {
    h->ctx.ref_c = malloc(h->c.len * sizeof(int32_t));
    memcpy(h->ctx.ref_c, h->c.data, h->c.len * sizeof(int32_t));
  }
}

void gemm_i8rcp_i8pc_i32_xsfvqdotq_verify(skl_test_t *t) {
  /* Compute the reference matrix output. */
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  skl_gemm_i8rcp_i8pc_i32_xsfvqdotq_ref(
      h->m, h->n, h->k, h->alpha, h->a_pack.data, h->rsa1, h->csa1,
      h->b_pack.data, h->rsb1, h->beta, h->ctx.ref_c, h->rsc);

  /* Compare the reference and test outputs. */
  for (size_t i = 0; i < h->m; ++i) {
    for (size_t j = 0; j < h->rsc; ++j) {
      size_t idx = i * h->rsc + j;
      if (h->c.data[idx] != h->ctx.ref_c[idx]) {
        SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                     "result [%zu, %zu] (%d) != reference (%d)\n", i, j,
                     h->c.data[idx], h->ctx.ref_c[idx]);
        t->status.verify_status = SKL_TEST_FAIL;
        return;
      }
    }
  }
}

void gemm_i8rcp_i8pc_i32_xsfvqdotq_report(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  size_t maccs = h->m * h->n * h->k;
  float mpc = (float)maccs / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("M: %zd, N: %zd, K: %zd\n", h->m, h->n, h->k);
  INFO("RSA1: %zd, CSA1: %zd, RSB1: %zd, RSC: %zd\n", h->rsa1, h->csa1, h->rsb1,
       h->rsc);
  INFO("Alpha: %d, Beta: %d\n", h->alpha, h->beta);
  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("MACs/Cycle: %f\n", mpc);
#undef INFO
}

void gemm_i8rcp_i8pc_i32_xsfvqdotq_cleanup(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a_pack);
  SKL_TEST_BUF_FREE(t, &h->b_pack);
  SKL_TEST_BUF_FREE(t, &h->c);
  if (h->steps.verify) {
    free(h->ctx.ref_c);
  }
}
