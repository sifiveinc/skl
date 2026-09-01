// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_i8rcprc_i8rcprc_i32rcprc.h"
#include "gemm/skl_test_gemm.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_xsfmm32a8i)
#error This file requires the Xsfmm32a8i extension
#endif

/**
 * @brief Test cases for the skl_gemm_i8c_i8_i32_xsfmm32a8i kernel.
 *
 * This test uses the gemm_i8rcprc_i8rcprc_i32rcprc harness with the following
 * restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 1
 *  - Matrix A is column-major (rsa1 == 1)
 *  - Matrix B is row-major (csb1 == 1)
 *  - Matrix C is row-major (csc1 == 1)
 */

#define TEST                                                                   \
  GEMM_I8RCPRC_I8RCPRC_I32RCPRC_DEFAULTS,                                      \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_i8rcprc_i8rcprc_i32rcprc_verify,                      \
          .report = gemm_i8rcprc_i8rcprc_i32rcprc_test_report,                 \
          .cleanup = gemm_i8rcprc_i8rcprc_i32rcprc_cleanup,                    \
  }

#define BENCH                                                                  \
  GEMM_I8RCPRC_I8RCPRC_I32RCPRC_DEFAULTS,                                      \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_i8rcprc_i8rcprc_i32rcprc_benchmark_report,            \
          .cleanup = gemm_i8rcprc_i8rcprc_i32rcprc_cleanup,                    \
  }

static void init(skl_test_t *t);
static void execute(skl_test_t *t);

// clang-format off
gemm_i8rcprc_i8rcprc_i32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH, .m1 = 2 * SKL_XSFMM_TE, .n1 = 2 * SKL_XSFMM_TE, .k1 = 8192, .alpha = 1, .beta = 0},
    {BENCH, .m1 = 2 * SKL_XSFMM_TE, .n1 = 2 * SKL_XSFMM_TE, .k1 = 8192, .alpha = 1, .beta = 1},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfmm
    /* Edge case: 1x1 matrix with k=0 (no computation, C = beta * C) */
    {TEST, .m1 = 1, .n1 = 1, .k1 = 0, .alpha = 1},
    /* Edge case: 1x1 matrix with k=1 (minimal computation) */
    {TEST, .m1 = 1, .n1 = 1, .k1 = 1, .alpha = 1},
    /* ETE-1 boundary: tests just below tile edge */
    {TEST, .m1 = 1 * SKL_XSFMM_TE - 1, .n1 = 1 * SKL_XSFMM_TE - 1, .k1 = 1, .alpha = 1},
    {TEST, .m1 = 1 * SKL_XSFMM_TE - 1, .n1 = 1 * SKL_XSFMM_TE - 1, .k1 = 2, .alpha = 1},
    {TEST, .m1 = 1 * SKL_XSFMM_TE - 1, .n1 = 1 * SKL_XSFMM_TE - 1, .k1 = 5, .alpha = 1},
    /* Exact ETE boundary: ETExETE tiles */
    {TEST, .m1 = 1 * SKL_XSFMM_TE,     .n1 = 1 * SKL_XSFMM_TE,     .k1 = 1, .alpha = 1},
    {TEST, .m1 = 1 * SKL_XSFMM_TE,     .n1 = 1 * SKL_XSFMM_TE,     .k1 = 5, .alpha = 1},
    /* ETE+1 boundary: tests just past tile edge */
    {TEST, .m1 = 1 * SKL_XSFMM_TE + 1, .n1 = 1 * SKL_XSFMM_TE + 1, .k1 = 1, .alpha = 1},
    {TEST, .m1 = 1 * SKL_XSFMM_TE + 1, .n1 = 1 * SKL_XSFMM_TE + 1, .k1 = 5, .alpha = 1},
    /* Multi-tile: 2*ETE */
    {TEST, .m1 = 2 * SKL_XSFMM_TE,     .n1 = 2 * SKL_XSFMM_TE,     .k1 = 1, .alpha = 1},
    {TEST, .m1 = 2 * SKL_XSFMM_TE,     .n1 = 2 * SKL_XSFMM_TE,     .k1 = 5, .alpha = 1},
    /* Multi-tile+1: 2*ETE+1 */
    {TEST, .m1 = 2 * SKL_XSFMM_TE + 1, .n1 = 2 * SKL_XSFMM_TE + 1, .k1 = 1, .alpha = 1},
    {TEST, .m1 = 2 * SKL_XSFMM_TE + 1, .n1 = 2 * SKL_XSFMM_TE + 1, .k1 = 5, .alpha = 1},
    /* Non-square matrices (rectangular tiles) */
    {TEST, .m1 = 2 * SKL_XSFMM_TE + 1, .n1 = 1 * SKL_XSFMM_TE - 1, .k1 = 1, .alpha = 1},
    {TEST, .m1 = 1 * SKL_XSFMM_TE + 1, .n1 = 2 * SKL_XSFMM_TE + 1, .k1 = 2, .alpha = 1},
    {TEST, .m1 = 1 * SKL_XSFMM_TE,     .n1 = 2 * SKL_XSFMM_TE,     .k1 = 5, .alpha = 1},
    /* General Alpha and Beta tests */
    {TEST, .m1 = 1 * SKL_XSFMM_TE + 1, .n1 = 1 * SKL_XSFMM_TE + 1, .k1 = 0,  .alpha = 2, .beta = 3},
    {TEST, .m1 = 1 * SKL_XSFMM_TE + 1, .n1 = 1 * SKL_XSFMM_TE + 1, .k1 = 1,  .alpha = 2, .beta = 3},
    {TEST, .m1 = 1 * SKL_XSFMM_TE + 1, .n1 = 1 * SKL_XSFMM_TE + 1, .k1 = 33, .alpha = 2, .beta = 3},
    {TEST, .m1 = 2 * SKL_XSFMM_TE,     .n1 = 2 * SKL_XSFMM_TE,     .k1 = 33, .alpha = 2, .beta = 3},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_gemm_i8c_i8_i32_xsfmm32a8i",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size =
                                     sizeof(gemm_i8rcprc_i8rcprc_i32rcprc_t),
                                 .tests = tests};

static void init(skl_test_t *t) {
  const gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  SKL_TEST_REQUIRE(t, init_status, h->m0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->k0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rsa1 == 1); // Note: column-major
  SKL_TEST_REQUIRE(t, init_status, h->csb1 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->csc1 == 1);

  gemm_i8rcprc_i8rcprc_i32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  skl_gemm_i8c_i8_i32_xsfmm32a8i(h->m1, h->n1, h->k1, h->alpha, h->a.data,
                                 h->csa1, h->b.data, h->rsb1, h->beta,
                                 h->c.data, h->rsc1);
}

int main(void) {
  // Set default strides: A is column-major, B and C are row-major
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = 1;
    tests[i].n0 = 1;
    tests[i].k0 = 1;

    tests[i].rsa0 = 1;
    tests[i].csa0 = 1;
    tests[i].rsa1 = 1;
    tests[i].csa1 = tests[i].csa1 ? tests[i].csa1 : tests[i].m1;

    tests[i].rsb0 = 1;
    tests[i].csb0 = 1;
    tests[i].rsb1 = tests[i].rsb1 ? tests[i].rsb1 : tests[i].n1;
    tests[i].csb1 = 1;

    tests[i].rsc0 = 1;
    tests[i].csc0 = 1;
    tests[i].rsc1 = tests[i].rsc1 ? tests[i].rsc1 : tests[i].n1;
    tests[i].csc1 = 1;
  }

  return skl_test_driver_run_suite(&suite);
}
