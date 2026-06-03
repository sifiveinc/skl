// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_f32rcprc_f32rcprc_f32rcprc.h"
#include "gemm/skl_test_gemm.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_zve32f)
#error This file requires the Zve32f extension
#endif

/**
 * @brief Test cases for GEMM with Zve32f extension.
 *
 * This test uses the gemm_f32rcprc_f32rcprc_f32rcprc harness with the following
 * restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 1
 *  - All matrices are row-major (csa1 == 1, csb1 == 1, csc1 == 1)
 */

#define TEST(M1, N1, K1)                                                       \
  GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps =                                                                 \
          {                                                                    \
              .init = gemm_f32rcprc_f32rcprc_f32rcprc_init,                    \
              .warmup = NULL,                                                  \
              .execute = execute,                                              \
              .verify = gemm_f32rcprc_f32rcprc_f32rcprc_verify,                \
              .report = gemm_f32rcprc_f32rcprc_f32rcprc_test_report,           \
              .cleanup = gemm_f32rcprc_f32rcprc_f32rcprc_cleanup,              \
  },                                                                           \
      .assert_dims_f = skl_test_assert_gemm_dims_r_r_r,                        \
      GEMM_PARAMS_R_R_R(M1, N1, K1)

#define BENCH(M1, N1, K1)                                                      \
  GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps =                                                                 \
          {                                                                    \
              .init = gemm_f32rcprc_f32rcprc_f32rcprc_init,                    \
              .warmup = execute,                                               \
              .execute = execute,                                              \
              .verify = NULL,                                                  \
              .report = gemm_f32rcprc_f32rcprc_f32rcprc_benchmark_report,      \
              .cleanup = gemm_f32rcprc_f32rcprc_f32rcprc_cleanup,              \
  },                                                                           \
      .assert_dims_f = skl_test_assert_gemm_dims_r_r_r,                        \
      GEMM_PARAMS_R_R_R(M1, N1, K1)

static void execute(skl_test_t *t);

// clang-format off
gemm_f32rcprc_f32rcprc_f32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH(64, 128, 128), .alpha = 1.f},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for RVV GEMM
    /* Edge cases: minimal dimensions */
    {TEST(1, 1, 0),  .alpha = 1.f},
    {TEST(1, 1,1 ),  .alpha = 1.f},
    /* Small odd dimensions for remainder handling */
    {TEST(7, 7, 7),  .alpha = 1.f},
    {TEST(17, 17, 17), .alpha = 1.f},
    /* Skinny matrices (one dimension = 1) */
    {TEST(1, 33, 31), .alpha = 1.f},
    {TEST(33, 1, 31), .alpha = 1.f},
    /* k=0 edge case (C = beta*C, no A*B contribution) */
    {TEST(33, 33, 0),  .alpha = 1.f},
    {TEST(16, 16, 0),  .alpha = 1.f,  .beta = 1.f},
    /* Vector length boundary tests (multiples of 4, 8, 16, 32) */
    {TEST(16, 16, 16), .alpha = 1.f},
    {TEST(32, 32, 32), .alpha = 1.f},
    /* Near-boundary tests (±1 from vector length multiples) */
    {TEST(16, 17, 31), .alpha = 1.f},
    {TEST(31, 33, 15), .alpha = 1.f},
    /* Wide and tall matrices */
    {TEST(33, 129, 32), .alpha = 1.f},
    {TEST(129, 33, 32), .alpha = 1.f},
    {TEST(31, 133, 32), .alpha = 1.f},
    /* Beta scaling test (beta=1, accumulate into existing C) */
    {TEST(32, 32, 32), .alpha = 1.f,  .beta = 1.f},
    /* Nontrivial leading dimensions */
    {TEST(32, 32, 32), .params.rsa1 = 64, .params.rsb1 = 64, .params.rsc1 = 64, .alpha = 1.f},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static void execute(skl_test_t *t) {
  const gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

  skl_gemm_f32_f32_f32_zve32f_x390(h->params.m1, h->params.n1, h->params.k1,
                                   h->alpha, h->a_pack.data, h->params.rsa1,
                                   h->b_pack.data, h->params.rsb1, h->beta,
                                   h->c_pack.data, h->params.rsc1);
}

static skl_test_suite_t suite = {.name = "skl_gemm_f32_f32_f32_zve32f_x390",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size =
                                     sizeof(gemm_f32rcprc_f32rcprc_f32rcprc_t),
                                 .tests = tests};

int main(void) { return skl_test_driver_run_suite(&suite); }
