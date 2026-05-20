// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

/**
 * @brief Implementation of the gemm_f64rcprc_f64rcprc_f64rcprc test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. rvv/skl_gemm_f64_f64_f64_zve64d_x390.c).
 */

#include "gemm_f64rcprc_f64rcprc_f64rcprc.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include "skl_test_gemm.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void gemm_f64rcprc_f64rcprc_f64rcprc_init(skl_test_t *t) {
  gemm_f64rcprc_f64rcprc_f64rcprc_t *h =
      (gemm_f64rcprc_f64rcprc_f64rcprc_t *)t->harness;

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
    h->a_pack.len = 0;
  } else {
    h->a_pack.len = (m1 - 1) * rsa1 + (k1 - 1) * csa1 + (m0 - 1) * rsa0 +
                    (k0 - 1) * csa0 + 1;
  }

  if (k1 == 0 || n1 == 0) {
    h->b_pack.len = 0;
  } else {
    h->b_pack.len = (k1 - 1) * rsb1 + (n1 - 1) * csb1 + (k0 - 1) * rsb0 +
                    (n0 - 1) * csb0 + 1;
  }

  if (m1 == 0 || n1 == 0) {
    h->c_pack.len = 0;
  } else {
    h->c_pack.len = (m1 - 1) * rsc1 + (n1 - 1) * csc1 + (m0 - 1) * rsc0 +
                    (n0 - 1) * csc0 + 1;
  }

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, double, &h->a_pack);
  SKL_TEST_BUF_CREATE(t, double, &h->b_pack);
  SKL_TEST_BUF_CREATE(t, double, &h->c_pack);
  if (h->steps.verify) {
    h->ctx.a_wide =
        h->a_pack.len ? malloc(h->a_pack.len * sizeof(*(h->ctx.a_wide))) : NULL;
    h->ctx.b_wide =
        h->b_pack.len ? malloc(h->b_pack.len * sizeof(*(h->ctx.b_wide))) : NULL;
    h->ctx.ref_c =
        h->c_pack.len ? malloc(h->c_pack.len * sizeof(*(h->ctx.ref_c))) : NULL;
    h->ctx.bound =
        h->c_pack.len ? malloc(h->c_pack.len * sizeof(*(h->ctx.bound))) : NULL;
    if (h->c_pack.len) {
      memcpy(h->ctx.ref_c, h->c_pack.data,
             h->c_pack.len * sizeof(*(h->c_pack.data)));
    }
  }
}

// Higher precision reference GEMM for computing error bounds for GEMMs with
// float64 accumulators. This function is placed here instead of skl-ref because
// it doesn't correspond to any skl library function.
static void skl_gemm_f128rcprc_f128rcprc_f128rcprc_ref(
    size_t m0, size_t n0, size_t k0, size_t m1, size_t n1, size_t k1,
    long double alpha, const long double *a_pack, size_t rsa0, size_t csa0,
    size_t rsa1, size_t csa1, const long double *b_pack, size_t rsb0,
    size_t csb0, size_t rsb1, size_t csb1, long double beta,
    long double *c_pack, size_t rsc0, size_t csc0, size_t rsc1, size_t csc1) {
  for (size_t ii1 = 0; ii1 < m1; ++ii1) {
    for (size_t jj1 = 0; jj1 < n1; ++jj1) {
      long double *cp_block = c_pack + ii1 * rsc1 + jj1 * csc1;
      for (size_t ii0 = 0; ii0 < m0; ++ii0) {
        for (size_t jj0 = 0; jj0 < n0; ++jj0) {
          long double acc = 0;
          for (size_t kk1 = 0; kk1 < k1; ++kk1) {
            const long double *ap_block = a_pack + ii1 * rsa1 + kk1 * csa1;
            const long double *bp_block = b_pack + kk1 * rsb1 + jj1 * csb1;
            for (size_t kk0 = 0; kk0 < k0; ++kk0) {
              long double a_val = ap_block[ii0 * rsa0 + kk0 * csa0];
              long double b_val = bp_block[kk0 * rsb0 + jj0 * csb0];
              acc += a_val * b_val;
            }
          }
          cp_block[ii0 * rsc0 + jj0 * csc0] =
              beta * cp_block[ii0 * rsc0 + jj0 * csc0] + alpha * acc;
        }
      }
    }
  }
}

void gemm_f64rcprc_f64rcprc_f64rcprc_verify(skl_test_t *t) {
  /* Compute the reference matrix output. */
  gemm_f64rcprc_f64rcprc_f64rcprc_t *h =
      (gemm_f64rcprc_f64rcprc_f64rcprc_t *)t->harness;

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
  double alpha = h->alpha;
  double beta = h->beta;
  double *a_pack = h->a_pack.data;
  double *b_pack = h->b_pack.data;
  double *c_pack = h->c_pack.data;
  size_t a_pack_len = h->a_pack.len;
  size_t b_pack_len = h->b_pack.len;
  size_t c_pack_len = h->c_pack.len;
  long double *a_wide = h->ctx.a_wide;
  long double *b_wide = h->ctx.b_wide;
  double *ref_c = h->ctx.ref_c;
  long double *bound = h->ctx.bound;

  //
  // Compute the error bound array for comparing test vs reference results.
  //
  // For any matrix M, define |M| as the matrix of absolute values where:
  //     |M|(i,j) = |M(i,j)|
  //
  // Let u = 2^-P be the maximum relative roundoff error for a floating-point
  // type with P-1 mantissa bits.
  //
  // For GEMM operations, the error between computed and exact results is
  // bounded by:
  //     ((1 + u)^(K + 2) - 1) * (|alpha| * |A| * |B| + |beta| * |C|)
  //
  // Since both test and reference results have roundoff errors, we double this
  // bound using the triangle inequality to get the final comparison threshold.
  //
  // Note: h->ctx.ref_c contains the original C values (copied in
  // skl_test_init) which are needed for the |beta| * |C| term in the error
  // bound.
  //
  for (size_t i = 0; i < a_pack_len; ++i) {
    a_wide[i] = fabsl(a_pack[i]);
  }
  for (size_t i = 0; i < b_pack_len; ++i) {
    b_wide[i] = fabsl(b_pack[i]);
  }
  for (size_t i = 0; i < c_pack_len; ++i) {
    bound[i] = fabsl(ref_c[i]);
  }
  const int P = 53; // 52 bits of mantissa for float64 accumulator
  const long double u =
      ldexpl((long double)1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K1 * K0 + 2) - 1) by change of base formula:
  const long double roundoff_scaling =
      2 * expm1l((long double)(k1 * k0 + 2) * log1pl(u));
  skl_gemm_f128rcprc_f128rcprc_f128rcprc_ref(
      m0, n0, k0, m1, n1, k1, roundoff_scaling * fabsl((long double)alpha),
      a_wide, rsa0, csa0, rsa1, csa1, b_wide, rsb0, csb0, rsb1, csb1,
      roundoff_scaling * fabsl((long double)beta), bound, rsc0, csc0, rsc1,
      csc1);

  // Compute the reference result using h->ctx.ref_c
  skl_gemm_f64rcprc_f64rcprc_f64rcprc_ref(
      m0, n0, k0, m1, n1, k1, alpha, a_pack, rsa0, csa0, rsa1, csa1, b_pack,
      rsb0, csb0, rsb1, csb1, beta, ref_c, rsc0, csc0, rsc1, csc1);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < m1; ++i1) {
    for (size_t j1 = 0; j1 < n1; ++j1) {
      for (size_t i0 = 0; i0 < m0; ++i0) {
        for (size_t j0 = 0; j0 < n0; ++j0) {
          size_t idx = i1 * rsc1 + j1 * csc1 + i0 * rsc0 + j0 * csc0;
          if (fabsl((long double)c_pack[idx] - (long double)ref_c[idx]) >
              bound[idx]) {
            SKL_TEST_LOG(
                t, SKL_TEST_LOG_ERROR,
                "result [%zu, %zu, %zu, %zu] (%.10f) != reference (%.10f) "
                "[bound = %.20Lf]\n",
                i1, j1, i0, j0, c_pack[idx], ref_c[idx], bound[idx]);
            t->status.verify_status = SKL_TEST_FAIL;
            return;
          }
        }
      }
    }
  }

  /* Check for clobbered elements. */
  skl_test_check_matrix_clobbered_rcprc(t, sizeof(*c_pack), c_pack_len, m0, n0,
                                        m1, n1, c_pack, ref_c, rsc0, csc0, rsc1,
                                        csc1);
}

void gemm_f64rcprc_f64rcprc_f64rcprc_test_report(skl_test_t *t) {
  gemm_f64rcprc_f64rcprc_f64rcprc_t *h =
      (gemm_f64rcprc_f64rcprc_f64rcprc_t *)t->harness;

  gemm_rcprc_rcprc_rcprc_report_matrix_params(
      t, h->m0, h->n0, h->k0, h->m1, h->n1, h->k1, h->rsa0, h->csa0, h->rsa1,
      h->csa1, h->rsb0, h->csb0, h->rsb1, h->csb1, h->rsc0, h->csc0, h->rsc1,
      h->csc1);
  SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, "Alpha: %f, Beta: %f\n", h->alpha,
               h->beta);
}

void gemm_f64rcprc_f64rcprc_f64rcprc_benchmark_report(skl_test_t *t) {
  gemm_f64rcprc_f64rcprc_f64rcprc_t *h =
      (gemm_f64rcprc_f64rcprc_f64rcprc_t *)t->harness;

  gemm_f64rcprc_f64rcprc_f64rcprc_test_report(t);

  size_t maccs = h->m1 * h->n1 * h->k1 * h->m0 * h->n0 * h->k0;
  gemm_rcprc_rcprc_rcprc_report_perf(t, h->steps.warmup, maccs);
}

void gemm_f64rcprc_f64rcprc_f64rcprc_cleanup(skl_test_t *t) {
  gemm_f64rcprc_f64rcprc_f64rcprc_t *h =
      (gemm_f64rcprc_f64rcprc_f64rcprc_t *)t->harness;

  // Free buffers
  SKL_TEST_BUF_FREE(t, &h->a_pack);
  SKL_TEST_BUF_FREE(t, &h->b_pack);
  SKL_TEST_BUF_FREE(t, &h->c_pack);
  if (h->steps.verify) {
    free(h->ctx.a_wide);
    free(h->ctx.b_wide);
    free(h->ctx.ref_c);
    free(h->ctx.bound);
  }
}
