// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_f16rcprc_f16rcprc_f32rcprc.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_zvfh)
#error This file requires the Zvfh extension
#endif

/**
 * @brief Test cases for GEMM with Zvfh extension.
 *
 * This test uses the gemm_f16rcprc_f16rcprc_f32rcprc harness with the following
 * restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 1
 *  - All matrices are row-major (csa1 == 1, csb1 == 1, csc1 == 1)
 */

#define TEST                                                                   \
  GEMM_F16RCPRC_F16RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_f16rcprc_f16rcprc_f32rcprc_verify,                    \
          .report = gemm_f16rcprc_f16rcprc_f32rcprc_test_report,               \
          .cleanup = gemm_f16rcprc_f16rcprc_f32rcprc_cleanup,                  \
  }

#define BENCH                                                                  \
  GEMM_F16RCPRC_F16RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_f16rcprc_f16rcprc_f32rcprc_benchmark_report,          \
          .cleanup = gemm_f16rcprc_f16rcprc_f32rcprc_cleanup,                  \
  }

static void init(skl_test_t *t);
static void execute(skl_test_t *t);

// clang-format off
gemm_f16rcprc_f16rcprc_f32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH, .m1 =  64, .n1 = 128, .k1 = 128, .alpha = 1.f},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for RVV GEMM
    /* Edge cases: minimal dimensions */
    {TEST, .m1 = 1,   .n1 = 1,   .k1 = 0,  .alpha = 1.f},
    {TEST, .m1 = 1,   .n1 = 1,   .k1 = 1,  .alpha = 1.f},
    /* Small odd dimensions for remainder handling */
    {TEST, .m1 = 7,   .n1 = 7,   .k1 = 7,  .alpha = 1.f},
    {TEST, .m1 = 17,  .n1 = 17,  .k1 = 17, .alpha = 1.f},
    /* Skinny matrices (one dimension = 1) */
    {TEST, .m1 = 1,   .n1 = 33,  .k1 = 31, .alpha = 1.f},
    {TEST, .m1 = 33,  .n1 = 1,   .k1 = 31, .alpha = 1.f},
    /* k=0 edge case (C = beta*C, no A*B contribution) */
    {TEST, .m1 = 33,  .n1 = 33,  .k1 = 0,  .alpha = 1.f},
    {TEST, .m1 = 16,  .n1 = 16,  .k1 = 0,  .alpha = 1.f,  .beta = 1.f},
    /* Vector length boundary tests (multiples of 4, 8, 16, 32) */
    {TEST, .m1 = 16,  .n1 = 16,  .k1 = 16, .alpha = 1.f},
    {TEST, .m1 = 32,  .n1 = 32,  .k1 = 32, .alpha = 1.f},
    /* Near-boundary tests (±1 from vector length multiples) */
    {TEST, .m1 = 15,  .n1 = 17,  .k1 = 31, .alpha = 1.f},
    {TEST, .m1 = 31,  .n1 = 33,  .k1 = 15, .alpha = 1.f},
    /* Wide and tall matrices */
    {TEST, .m1 = 33,  .n1 = 129, .k1 = 32, .alpha = 1.f},
    {TEST, .m1 = 129, .n1 = 33,  .k1 = 32, .alpha = 1.f},
    {TEST, .m1 = 31,  .n1 = 133, .k1 = 32, .alpha = 1.f},
    /* Beta scaling test (beta=1, accumulate into existing C) */
    {TEST, .m1 = 32,  .n1 = 32,  .k1 = 32, .alpha = 1.f,  .beta = 1.f},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_gemm_f16_f16_f32_zvfh_x390",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size =
                                     sizeof(gemm_f16rcprc_f16rcprc_f32rcprc_t),
                                 .tests = tests};

static void init(skl_test_t *t) {
  const gemm_f16rcprc_f16rcprc_f32rcprc_t *h =
      (gemm_f16rcprc_f16rcprc_f32rcprc_t *)t->harness;

  SKL_TEST_REQUIRE(t, init_status, h->m0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->k0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->csa1 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->csb1 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->csc1 == 1);

  gemm_f16rcprc_f16rcprc_f32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_f16rcprc_f16rcprc_f32rcprc_t *h =
      (gemm_f16rcprc_f16rcprc_f32rcprc_t *)t->harness;

  skl_gemm_f16_f16_f32_zvfh_x390(h->m1, h->n1, h->k1, h->alpha, h->a_pack.data,
                                 h->rsa1, h->b_pack.data, h->rsb1, h->beta,
                                 h->c_pack.data, h->rsc1);
}

int main(void) {
  // Set default strides: all matrices are row-major
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = 1;
    tests[i].n0 = 1;
    tests[i].k0 = 1;

    tests[i].rsa0 = 1;
    tests[i].csa0 = 1;
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : tests[i].k1;
    tests[i].csa1 = tests[i].csa1 ? tests[i].csa1 : 1;

    tests[i].rsb0 = 1;
    tests[i].csb0 = 1;
    tests[i].rsb1 = tests[i].rsb1 ? tests[i].rsb1 : tests[i].n1;
    tests[i].csb1 = tests[i].csb1 ? tests[i].csb1 : 1;

    tests[i].rsc0 = 1;
    tests[i].csc0 = 1;
    tests[i].rsc1 = tests[i].rsc1 ? tests[i].rsc1 : tests[i].n1;
    tests[i].csc1 = tests[i].csc1 ? tests[i].csc1 : 1;
  }

  return skl_test_driver_run_suite(&suite);
}
