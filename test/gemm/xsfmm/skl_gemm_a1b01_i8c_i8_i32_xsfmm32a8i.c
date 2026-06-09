// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_i8rcprc_i8rcprc_i32rcprc.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_xsfmm32a8i)
#error This file requires the Xsfmm32a8i extension
#endif

/**
 * @brief Test cases for GEMM with Xsfmm32a8i extension.
 *
 * This test uses the gemm_i8rcprc_i8rcprc_i32rcprc harness with the
 * following restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 1
 *  - Matrix A is column-major (rsa1 == 1)
 *  - Matrix B is row-major (csb1 == 1)
 *  - Matrix C is row-major (csc1 == 1)
 *  - Alpha must be 1
 *  - Beta must be 0 or 1
 *
 * The kernel computes C = A * B (beta = 0) or C += A * B (beta = 1).
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
    {BENCH, .m1 = 128, .n1 = 128, .k1 = 8192, .alpha = 1, .beta = 0},
    {BENCH, .m1 = 128, .n1 = 128, .k1 = 8192, .alpha = 1, .beta = 1},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfmm A1B01 layout (ETE=64)
    /* Edge case: 1x1 matrix with k=0 (no computation, C = beta * C) */
    {TEST, .m1 = 1,   .n1 = 1,   .k1 = 0, .alpha = 1},
    /* Edge case: 1x1 matrix with k=1 (minimal computation) */
    {TEST, .m1 = 1,   .n1 = 1,   .k1 = 1, .alpha = 1},
    /* ETE-1 boundary: 63 = 64-1, tests just below tile edge */
    {TEST, .m1 = 63,  .n1 = 63,  .k1 = 1, .alpha = 1},
    {TEST, .m1 = 63,  .n1 = 63,  .k1 = 2, .alpha = 1},
    {TEST, .m1 = 63,  .n1 = 63,  .k1 = 5, .alpha = 1},
    /* Exact ETE boundary: 64x64 tiles */
    {TEST, .m1 = 64,  .n1 = 64,  .k1 = 1, .alpha = 1},
    {TEST, .m1 = 64,  .n1 = 64,  .k1 = 5, .alpha = 1},
    /* ETE+1 boundary: 65 = 64+1, tests just past tile edge */
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 1, .alpha = 1},
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 5, .alpha = 1},
    /* Multi-tile: 2*ETE = 128 */
    {TEST, .m1 = 128, .n1 = 128, .k1 = 1, .alpha = 1},
    {TEST, .m1 = 128, .n1 = 128, .k1 = 5, .alpha = 1},
    /* Multi-tile+1: 2*ETE+1 = 129 */
    {TEST, .m1 = 129, .n1 = 129, .k1 = 1, .alpha = 1},
    {TEST, .m1 = 129, .n1 = 129, .k1 = 5, .alpha = 1},
    /* Non-square matrices (rectangular tiles) */
    {TEST, .m1 = 129, .n1 = 63,  .k1 = 1, .alpha = 1},
    {TEST, .m1 = 65,  .n1 = 129, .k1 = 2, .alpha = 1},
    {TEST, .m1 = 64,  .n1 = 128, .k1 = 5, .alpha = 1},
    /* Beta=1 tests (accumulate into existing C) */
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 0,  .alpha = 1, .beta = 1},
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 1,  .alpha = 1, .beta = 1},
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 33, .alpha = 1, .beta = 1},
    {TEST, .m1 = 128, .n1 = 128, .k1 = 33, .alpha = 1, .beta = 1},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_gemm_a1b01_i8c_i8_i32_xsfmm32a8i",
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
  SKL_TEST_REQUIRE(t, init_status, h->alpha == 1);
  SKL_TEST_REQUIRE(t, init_status, h->beta == 0 || h->beta == 1);

  gemm_i8rcprc_i8rcprc_i32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  // Call the kernel with the appropriate parameters
  // The kernel signature is: (m, n, k, a, csa, b, rsb, c, rsc, accum)
  // where accum = (beta != 0)
  skl_gemm_a1b01_i8c_i8_i32_xsfmm32a8i(h->m1, h->n1, h->k1, h->a_pack.data,
                                       h->csa1, h->b_pack.data, h->rsb1,
                                       h->c_pack.data, h->rsc1, h->beta != 0);
}

int main(void) {
  // Set default strides: A is column-major, B is row-major, C is row-major
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = 1;
    tests[i].n0 = 1;
    tests[i].k0 = 1;

    tests[i].rsa0 = 1;
    tests[i].csa0 = 1;
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : 1;
    tests[i].csa1 = tests[i].csa1 ? tests[i].csa1 : tests[i].m1;

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
