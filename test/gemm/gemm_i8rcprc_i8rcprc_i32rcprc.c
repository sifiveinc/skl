// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the gemm_i8rcprc_i8rcprc_i32rcprc test
 * harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. xsfmm/skl_gemm_i8c_i8_i32_xsfmm32a8f.c).
 */

#include "gemm_i8rcprc_i8rcprc_i32rcprc.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl_test_gemm.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void gemm_i8rcprc_i8rcprc_i32rcprc_init(skl_test_t *t) {
  gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  size_t m0 = h->m0;
  size_t n0 = h->n0;
  size_t k0 = h->k0;
  size_t m1 = h->m1;
  size_t n1 = h->n1;
  size_t k1 = h->k1;
  size_t rsa0 = h->rsa0;
  size_t csa0 = h->csa0;
  size_t rsa1 = h->rsa1;
  size_t csa1 = h->csa1;
  size_t rsb0 = h->rsb0;
  size_t csb0 = h->csb0;
  size_t rsb1 = h->rsb1;
  size_t csb1 = h->csb1;
  size_t rsc0 = h->rsc0;
  size_t csc0 = h->csc0;
  size_t rsc1 = h->rsc1;
  size_t csc1 = h->csc1;

  skl_test_check_matrix_params_rcprc(t, m0, k0, m1, k1, rsa0, csa0, rsa1, csa1);
  skl_test_check_matrix_params_rcprc(t, k0, n0, k1, n1, rsb0, csb0, rsb1, csb1);
  skl_test_check_matrix_params_rcprc(t, m0, n0, m1, n1, rsc0, csc0, rsc1, csc1);

  if (t->status.init_status != SKL_TEST_PASS) {
    return;
  }

  if (m1 == 0 || k1 == 0) {
    h->a.len = 0;
  } else {
    h->a.len = (m1 - 1) * rsa1 + (k1 - 1) * csa1 + (m0 - 1) * rsa0 +
               (k0 - 1) * csa0 + 1;
  }

  if (k1 == 0 || n1 == 0) {
    h->b.len = 0;
  } else {
    h->b.len = (k1 - 1) * rsb1 + (n1 - 1) * csb1 + (k0 - 1) * rsb0 +
               (n0 - 1) * csb0 + 1;
  }

  if (m1 == 0 || n1 == 0) {
    h->c.len = 0;
  } else {
    h->c.len = (m1 - 1) * rsc1 + (n1 - 1) * csc1 + (m0 - 1) * rsc0 +
               (n0 - 1) * csc0 + 1;
  }

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, int8_t, &h->a);
  SKL_TEST_BUF_CREATE(t, int8_t, &h->b);
  SKL_TEST_BUF_CREATE(t, int32_t, &h->c);
  if (h->steps.verify) {
    h->ctx.ref_c = h->c.len ? malloc(h->c.len * sizeof(*(h->ctx.ref_c))) : NULL;
    if (h->c.len) {
      memcpy(h->ctx.ref_c, h->c.data, h->c.len * sizeof(*(h->c.data)));
    }
  }
}

void gemm_i8rcprc_i8rcprc_i32rcprc_verify(skl_test_t *t) {
  gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  size_t m0 = h->m0;
  size_t n0 = h->n0;
  size_t k0 = h->k0;
  size_t m1 = h->m1;
  size_t n1 = h->n1;
  size_t k1 = h->k1;
  size_t rsa0 = h->rsa0;
  size_t csa0 = h->csa0;
  size_t rsa1 = h->rsa1;
  size_t csa1 = h->csa1;
  size_t rsb0 = h->rsb0;
  size_t csb0 = h->csb0;
  size_t rsb1 = h->rsb1;
  size_t csb1 = h->csb1;
  size_t rsc0 = h->rsc0;
  size_t csc0 = h->csc0;
  size_t rsc1 = h->rsc1;
  size_t csc1 = h->csc1;
  int32_t alpha = h->alpha;
  int32_t beta = h->beta;
  int8_t *a = h->a.data;
  int8_t *b = h->b.data;
  int32_t *c = h->c.data;
  size_t c_len = h->c.len;
  int32_t *ref_c = h->ctx.ref_c;

  // Compute the reference result using h->ctx.ref_c
  // h->ctx.ref_c contains the original C values (copied in skl_test_init)
  skl_gemm_i8rcprc_i8rcprc_i32rcprc_ref(
      m0, n0, k0, m1, n1, k1, alpha, a, rsa0, csa0, rsa1, csa1, b, rsb0, csb0,
      rsb1, csb1, beta, ref_c, rsc0, csc0, rsc1, csc1);

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

void gemm_i8rcprc_i8rcprc_i32rcprc_test_report(skl_test_t *t) {
  gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  gemm_rcprc_rcprc_rcprc_report_matrix_params(
      t, h->m0, h->n0, h->k0, h->m1, h->n1, h->k1, h->rsa0, h->csa0, h->rsa1,
      h->csa1, h->rsb0, h->csb0, h->rsb1, h->csb1, h->rsc0, h->csc0, h->rsc1,
      h->csc1);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Alpha: %d, Beta: %d\n", h->alpha,
               h->beta);
}

void gemm_i8rcprc_i8rcprc_i32rcprc_benchmark_report(skl_test_t *t) {
  gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  gemm_i8rcprc_i8rcprc_i32rcprc_test_report(t);

  size_t maccs = h->m1 * h->n1 * h->k1 * h->m0 * h->n0 * h->k0;
  gemm_rcprc_rcprc_rcprc_report_perf(t, h->steps.warmup, maccs);
}

void gemm_i8rcprc_i8rcprc_i32rcprc_cleanup(skl_test_t *t) {
  gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a);
  SKL_TEST_BUF_FREE(t, &h->b);
  SKL_TEST_BUF_FREE(t, &h->c);
  if (h->steps.verify) {
    free(h->ctx.ref_c);
  }
}
