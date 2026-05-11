// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the gemm_i8rcp_i8pc_i32_xsfvqdotq test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g.
 * xsfvqdotq/skl_gemm_i8rcp_i8pc_i32_xsfvqdotq.c).
 */

#include "gemm_i8rcp_i8pc_i32_xsfvqdotq.h"
#include "skl-test-driver.h"
#include "skl_test_gemm.h"
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

// NOLINTBEGIN(readability-function-cognitive-complexity)
void gemm_i8rcp_i8pc_i32_xsfvqdotq_init(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  size_t m = h->m;
  size_t n = h->n;
  size_t k = h->k;
  size_t m0 = 1;
  size_t n0 = 1;
  size_t k0 = 4;
  size_t m1 = m;
  size_t n1 = n;
  size_t k1 = (k + 3) / 4;
  size_t rsa0 = 4;
  size_t csa0 = 1;
  size_t rsa1 = h->rsa1;
  size_t csa1 = h->csa1;
  size_t rsb0 = 1;
  size_t csb0 = 4;
  size_t rsb1 = h->rsb1;
  size_t csb1 = 4;
  size_t rsc0 = 1;
  size_t csc0 = 1;
  size_t rsc1 = h->rsc;
  size_t csc1 = 1;

  size_t a_block_min_len = (m0 - 1) * rsa0 + (k0 - 1) * csa0 + 1;
  size_t a_right_block_min_len = (m0 - 1) * rsa0 + ((k - 1) % k0) * csa0 + 1;
  if (m1 > 1 && k1 > 0) {
    SKL_TEST_REQUIRE(t, init_status, rsa1 >= a_right_block_min_len);
  }
  if (m1 > 0 && k1 > 1) {
    SKL_TEST_REQUIRE(t, init_status, csa1 >= a_block_min_len);
  }
  if (m1 > 1 && k1 > 1) {
    if (rsa1 >= csa1) {
      SKL_TEST_REQUIRE(t, init_status,
                       rsa1 >= (k1 - 1) * csa1 + a_right_block_min_len);
    } else {
      SKL_TEST_REQUIRE(t, init_status, rsa1 >= a_block_min_len);
      SKL_TEST_REQUIRE(t, init_status,
                       csa1 >= (m1 - 1) * rsa1 + a_block_min_len);
    }
  }
  skl_test_check_matrix_params_rcprc(t, k0, n0, k1, n1, rsb0, csb0, rsb1, csb1);
  skl_test_check_matrix_params_rcprc(t, m0, n0, m1, n1, rsc0, csc0, rsc1, csc1);

  if (t->status.init_status != SKL_TEST_PASS) {
    return;
  }

  if (m1 == 0 || k1 == 0) {
    h->a_pack.len = 0;
  } else {
    h->a_pack.len = (m1 - 1) * rsa1 + (k1 - 1) * csa1 + (m0 - 1) * rsa0 +
                    ((k - 1) % k0) * csa0 + 1;
  }

  if (k1 == 0 || n1 == 0) {
    h->b_pack.len = 0;
  } else {
    h->b_pack.len = (k1 - 1) * rsb1 + (n1 - 1) * csb1 + (k0 - 1) * rsb0 +
                    (n0 - 1) * csb0 + 1;
  }

  if (m1 == 0 || n1 == 0) {
    h->c.len = 0;
  } else {
    h->c.len = (m1 - 1) * rsc1 + (n1 - 1) * csc1 + (m0 - 1) * rsc0 +
               (n0 - 1) * csc0 + 1;
  }

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, int8_t, &h->a_pack);
  SKL_TEST_BUF_CREATE(t, int8_t, &h->b_pack);
  SKL_TEST_BUF_CREATE(t, int32_t, &h->c);
  if (h->steps.verify) {
    h->ctx.ref_c = h->c.len ? malloc(h->c.len * sizeof(*(h->ctx.ref_c))) : NULL;
    if (h->c.len) {
      memcpy(h->ctx.ref_c, h->c.data, h->c.len * sizeof(*(h->c.data)));
    }
  }
}
// NOLINTEND(readability-function-cognitive-complexity)

void gemm_i8rcp_i8pc_i32_xsfvqdotq_verify(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  size_t m = h->m;
  size_t n = h->n;
  size_t k = h->k;
  size_t m0 = 1;
  size_t n0 = 1;
  size_t m1 = m;
  size_t n1 = n;
  size_t rsa1 = h->rsa1;
  size_t csa1 = h->csa1;
  size_t rsb1 = h->rsb1;
  size_t rsc0 = 1;
  size_t csc0 = 1;
  size_t rsc1 = h->rsc;
  size_t csc1 = 1;
  int32_t alpha = h->alpha;
  int32_t beta = h->beta;
  int8_t *a_pack = h->a_pack.data;
  int8_t *b_pack = h->b_pack.data;
  int32_t *c = h->c.data;
  size_t c_len = h->c.len;
  int32_t *ref_c = h->ctx.ref_c;

  /* Compute the reference matrix output. */
  skl_gemm_i8rcp_i8pc_i32_xsfvqdotq_ref(m, n, k, alpha, a_pack, rsa1, csa1,
                                        b_pack, rsb1, beta, ref_c, rsc1);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < m1; ++i1) {
    for (size_t j1 = 0; j1 < n1; ++j1) {
      for (size_t i0 = 0; i0 < m0; ++i0) {
        for (size_t j0 = 0; j0 < n0; ++j0) {
          size_t idx = i1 * rsc1 + j1 * csc1 + i0 * rsc0 + j0 * csc0;
          if (c[idx] != ref_c[idx]) {
            SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                         "result [%zu, %zu, %zu, %zu] (%d) != reference (%d)\n",
                         i1, j1, i0, j0, c[idx], ref_c[idx]);
            t->status.verify_status = SKL_TEST_FAIL;
            return;
          }
        }
      }
    }
  }

  /* Check for clobbered elements. */
  skl_test_check_matrix_clobbered_rcprc(t, sizeof(*c), c_len, m0, n0, m1, n1, c,
                                        ref_c, rsc0, csc0, rsc1, csc1);
}

void gemm_i8rcp_i8pc_i32_xsfvqdotq_test_report(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "M: %zu, N: %zu, K: %zu\n", h->m, h->n,
               h->k);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO,
               "RSA1: %zu, CSA1: %zu, RSB1: %zu, RSC: %zu\n", h->rsa1, h->csa1,
               h->rsb1, h->rsc);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Alpha: %d, Beta: %d\n", h->alpha,
               h->beta);
}

void gemm_i8rcp_i8pc_i32_xsfvqdotq_benchmark_report(skl_test_t *t) {
  gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  gemm_i8rcp_i8pc_i32_xsfvqdotq_test_report(t);

  size_t maccs = h->m * h->n * h->k;
  gemm_rcprc_rcprc_rcprc_report_perf(t, h->steps.warmup, maccs);
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
