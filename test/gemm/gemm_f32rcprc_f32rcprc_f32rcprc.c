// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Implementation of the gemm_f32rcprc_f32rcprc_f32rcprc test harness.
 *
 * This file defines all harness functions _except_ `skl_test_execute`, which is
 * defined in the test file (e.g. rvv/skl_gemm_f32_f32_f32_zve32f_x390.c).
 */

#include "gemm_f32rcprc_f32rcprc_f32rcprc.h"
#include "skl-ref.h"
#include "skl-test-driver.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gemm_f32rcprc_f32rcprc_f32rcprc_init(skl_test_t *t) {
  gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

  if (h->rsa1 > h->csa1) {
    h->a_pack.len = h->m1 * h->rsa1;
  } else if (h->rsa1 < h->csa1) {
    h->a_pack.len = h->k1 * h->csa1;
  } else {
    h->a_pack.len = (h->m1 >= h->k1 ? h->m1 : h->k1) * h->rsa1;
  }

  if (h->rsb1 > h->csb1) {
    h->b_pack.len = h->k1 * h->rsb1;
  } else if (h->rsb1 < h->csb1) {
    h->b_pack.len = h->n1 * h->csb1;
  } else {
    h->b_pack.len = (h->k1 >= h->n1 ? h->k1 : h->n1) * h->rsb1;
  }

  if (h->rsc1 > h->csc1) {
    h->c_pack.len = h->m1 * h->rsc1;
  } else if (h->rsc1 < h->csc1) {
    h->c_pack.len = h->n1 * h->csc1;
  } else {
    h->c_pack.len = (h->m1 >= h->n1 ? h->m1 : h->n1) * h->rsc1;
  }

  // Allocate buffers
  SKL_TEST_BUF_CREATE(t, float, &h->a_pack);
  SKL_TEST_BUF_CREATE(t, float, &h->b_pack);
  SKL_TEST_BUF_CREATE(t, float, &h->c_pack);
  if (h->steps.verify) {
    h->ctx.a_wide = malloc(h->a_pack.len * sizeof(*(h->ctx.a_wide)));
    h->ctx.b_wide = malloc(h->b_pack.len * sizeof(*(h->ctx.b_wide)));
    h->ctx.ref_c = malloc(h->c_pack.len * sizeof(*(h->ctx.ref_c)));
    h->ctx.c_pack_copy = malloc(h->c_pack.len * sizeof(*(h->ctx.c_pack_copy)));
    h->ctx.bound = malloc(h->c_pack.len * sizeof(*(h->ctx.bound)));
    // Copy initial C values to ref_c for the beta * C term in reference GEMM
    memcpy(h->ctx.ref_c, h->c_pack.data,
           h->c_pack.len * sizeof(*(h->c_pack.data)));
    // Copy initial C values to c_pack_copy to check for clobbered elements
    memcpy(h->ctx.c_pack_copy, h->c_pack.data,
           h->c_pack.len * sizeof(*(h->c_pack.data)));
  }
}

void gemm_f32rcprc_f32rcprc_f32rcprc_verify(skl_test_t *t) {
  /* Compute the reference matrix output. */
  gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

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
  for (size_t i = 0; i < h->a_pack.len; ++i) {
    h->ctx.a_wide[i] = fabsf(h->a_pack.data[i]);
  }
  for (size_t i = 0; i < h->b_pack.len; ++i) {
    h->ctx.b_wide[i] = fabsf(h->b_pack.data[i]);
  }
  for (size_t i = 0; i < h->c_pack.len; ++i) {
    h->ctx.bound[i] = fabsf(h->ctx.ref_c[i]);
  }
  const int P = 24; // 23 bits of mantissa for float32 accumulator
  const double u = ldexp(1.0, -P); // Maximum relative roundoff error
  // Compute 2 * ((1 + u)^(K1 * K0 + 2) - 1) by change of base formula:
  const double roundoff_scaling =
      2 * expm1((double)(h->k1 * h->k0 + 2) * log1p(u));
  skl_gemm_f64rcprc_f64rcprc_f64rcprc_ref(
      h->m0, h->n0, h->k0, h->m1, h->n1, h->k1,
      roundoff_scaling * fabs((double)h->alpha), h->ctx.a_wide, h->rsa0,
      h->csa0, h->rsa1, h->csa1, h->ctx.b_wide, h->rsb0, h->csb0, h->rsb1,
      h->csb1, roundoff_scaling * fabs((double)h->beta), h->ctx.bound, h->rsc0,
      h->csc0, h->rsc1, h->csc1);

  // Compute the reference result using h->ctx.ref_c
  // h->ctx.ref_c contains the original C values (copied in skl_test_init)
  // After this call, h->ctx.ref_c will contain the reference result
  skl_gemm_f32rcprc_f32rcprc_f32rcprc_ref(
      h->m0, h->n0, h->k0, h->m1, h->n1, h->k1, h->alpha, h->a_pack.data,
      h->rsa0, h->csa0, h->rsa1, h->csa1, h->b_pack.data, h->rsb0, h->csb0,
      h->rsb1, h->csb1, h->beta, h->ctx.ref_c, h->rsc0, h->csc0, h->rsc1,
      h->csc1);

  /* Compare the reference and test outputs. */
  for (size_t i1 = 0; i1 < h->m1; ++i1) {
    for (size_t j1 = 0; j1 < h->n1; ++j1) {
      for (size_t i0 = 0; i0 < h->m0; ++i0) {
        for (size_t j0 = 0; j0 < h->n0; ++j0) {
          size_t idx =
              i1 * h->rsc1 + j1 * h->csc1 + i0 * h->rsc0 + j0 * h->csc0;
          if (fabs((double)h->c_pack.data[idx] - (double)h->ctx.ref_c[idx]) >
              h->ctx.bound[idx]) {
            SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                         "result [%zu, %zu, %zu, %zu] (%f) != reference (%f) "
                         "[bound = %f]\n",
                         i1, j1, i0, j0, h->c_pack.data[idx], h->ctx.ref_c[idx],
                         h->ctx.bound[idx]);
            t->status.verify_status = SKL_TEST_FAIL;
            return;
          }
        }
      }
    }
  }

  /* Check for clobbered elements. */
  // When m0 == 1, rsc0 can be arbitrary. We adjust its value in this case to
  // reflect the length of a row. csc0, rsc1, and csc1 are adjusted similarly.
  size_t rsc0_adj = h->m0 == 1 ? (h->n0 - 1) * h->csc0 + 1 : h->rsc0;
  size_t csc0_adj = h->n0 == 1 ? (h->m0 - 1) * h->rsc0 + 1 : h->csc0;
  size_t c_block_min_len = (h->m0 - 1) * rsc0_adj + (h->n0 - 1) * csc0_adj + 1;
  size_t rsc1_adj =
      h->m1 == 1 ? (h->n1 - 1) * h->csc1 + c_block_min_len : h->rsc1;
  size_t csc1_adj =
      h->n1 == 1 ? (h->m1 - 1) * h->rsc1 + c_block_min_len : h->csc1;
  // Each block of C_pack lies in an ambient m0_amb x n0_amb row- or
  // column-major matrix with row and column strides rsc0_amb and csc0_amb.
  // Let bsc1 be the block stride of C_pack, i.e. the distance from one block to
  // the next in memory, in elements; bsc1 varies from block to block.
  // Lastly, set layoutc0_row_major to true if blocks of C_pack are row major
  // and false otherwise. Similarly, set layoutc1_row_major to true if C_pack is
  // block-row-major and false otherwise.
  size_t m0_amb = 0;
  size_t n0_amb = 0;
  size_t rsc0_amb = 0;
  size_t csc0_amb = 0;
  size_t bsc1 = 0;
  bool layoutc0_row_major = rsc0_adj >= csc0_adj;
  bool layoutc1_row_major = rsc1_adj >= csc1_adj;
  if (layoutc0_row_major) {
    rsc0_amb = rsc0_adj;
    csc0_amb = 1;
    m0_amb = h->m0;
    n0_amb = rsc0_amb;
  } else {
    rsc0_amb = 1;
    csc0_amb = csc0_adj;
    m0_amb = csc0_amb;
    n0_amb = h->n0;
  }

  // Check each block.
  for (size_t i1 = 0; i1 < h->m1; ++i1) {
    for (size_t j1 = 0; j1 < h->n1; ++j1) {
      // Check the first c_block_min_len elements.
      for (size_t i0 = 0; i0 < m0_amb; ++i0) {
        for (size_t j0 = 0; j0 < n0_amb; ++j0) {
          if (layoutc0_row_major && j0 % csc0_adj == 0 &&
              j0 < h->n0 * csc0_adj) {
            continue;
          }
          if (!layoutc0_row_major && i0 % rsc0_adj == 0 &&
              i0 < h->m0 * rsc0_adj) {
            continue;
          }
          if (i0 * rsc0_amb + j0 * csc0_amb >= c_block_min_len) {
            continue;
          }

          size_t idx =
              i1 * rsc1_adj + j1 * csc1_adj + i0 * rsc0_amb + j0 * csc0_amb;
          // NaNs always compare as false, so do a bitwise comparison.
          int64_t initial = 0;
          int64_t result = 0;
          memcpy(&initial, &(h->ctx.c_pack_copy[idx]),
                 sizeof(*(h->ctx.c_pack_copy)));
          memcpy(&result, &(h->c_pack.data[idx]), sizeof(*(h->c_pack.data)));
          if (result != initial) {
            SKL_TEST_LOG(
                t, SKL_TEST_LOG_ERROR,
                "result [%zu, %zu, %zu, %zu] (%f) vs. clobbered (%f)\n", i1, j1,
                i0, j0, h->c_pack.data[idx], h->ctx.c_pack_copy[idx]);
            t->status.verify_status = SKL_TEST_FAIL;
            return;
          }
        }
      }

      // Check until the next block.
      if (layoutc1_row_major) {
        bsc1 = j1 < h->n1 - 1 ? csc1_adj : rsc1_adj - (h->n1 - 1) * csc1_adj;
      } else {
        bsc1 = i1 < h->m1 - 1 ? rsc1_adj : csc1_adj - (h->m1 - 1) * rsc1_adj;
      }
      if (i1 == h->m1 - 1 && j1 == h->n1 - 1) {
        bsc1 = c_block_min_len;
      }
      for (size_t i = c_block_min_len; i < bsc1; ++i) {
        size_t idx = i1 * rsc1_adj + j1 * csc1_adj + i;
        // NaNs always compare as false, so do a bitwise comparison.
        int64_t initial = 0;
        int64_t result = 0;
        memcpy(&initial, &(h->ctx.c_pack_copy[idx]),
               sizeof(*(h->ctx.c_pack_copy)));
        memcpy(&result, &(h->c_pack.data[idx]), sizeof(*(h->c_pack.data)));
        if (result != initial) {
          SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                       "result [%zu, %zu, %zu] (%f) vs. clobbered (%f)\n", i1,
                       j1, i, h->c_pack.data[idx], h->ctx.c_pack_copy[idx]);
          t->status.verify_status = SKL_TEST_FAIL;
          return;
        }
      }
    }
  }

  // Check until the end of the array.
  size_t c_pack_min_len =
      h->m1 == 0 || h->n1 == 0
          ? 0
          : (h->m1 - 1) * rsc1_adj + (h->n1 - 1) * csc1_adj + c_block_min_len;
  for (size_t idx = c_pack_min_len; idx < h->c_pack.len; ++idx) {
    // NaNs always compare as false, so do a bitwise comparison.
    int64_t initial = 0;
    int64_t result = 0;
    memcpy(&initial, &(h->ctx.c_pack_copy[idx]), sizeof(*(h->ctx.c_pack_copy)));
    memcpy(&result, &(h->c_pack.data[idx]), sizeof(*(h->c_pack.data)));
    if (result != initial) {
      SKL_TEST_LOG(t, SKL_TEST_LOG_ERROR,
                   "result [%zu] (%f) vs. clobbered (%f)\n", idx,
                   h->c_pack.data[idx], h->ctx.c_pack_copy[idx]);
      t->status.verify_status = SKL_TEST_FAIL;
      return;
    }
  }
}

void gemm_f32rcprc_f32rcprc_f32rcprc_report(skl_test_t *t) {
  gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

  size_t maccs = h->m1 * h->n1 * h->k1 * h->m0 * h->n0 * h->k0;
  float mpc = (float)maccs / (float)t->counters.cycles;

#define INFO(fmt, ...) SKL_TEST_LOG(t, SKL_TEST_LOG_INFO, fmt, __VA_ARGS__)

  INFO("M0: %zd, N0: %zd, K0: %zd\n", h->m0, h->n0, h->k0);
  INFO("M1: %zd, N1: %zd, K1: %zd\n", h->m1, h->n1, h->k1);
  INFO("RSA0: %zd, CSA0: %zd, RSA1: %zd, CSA1: %zd\n", h->rsa0, h->csa0,
       h->rsa1, h->csa1);
  INFO("RSB0: %zd, CSB0: %zd, RSB1: %zd, CSB1: %zd\n", h->rsb0, h->csb0,
       h->rsb1, h->csb1);
  INFO("RSC0: %zd, CSC0: %zd, RSC1: %zd, CSC1: %zd\n", h->rsc0, h->csc0,
       h->rsc1, h->csc1);
  INFO("Alpha: %f, Beta: %f\n", h->alpha, h->beta);
  INFO("%s", "\n");
  INFO("Warmup: %s\n", h->steps.warmup ? "yes" : "no");
  INFO("Cycles: %zd\n", t->counters.cycles);
  INFO("Instructions: %zd\n", t->counters.instret);
  INFO("MACs/Cycle: %f\n", mpc);
#undef INFO
}

void gemm_f32rcprc_f32rcprc_f32rcprc_cleanup(skl_test_t *t) {
  gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

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
