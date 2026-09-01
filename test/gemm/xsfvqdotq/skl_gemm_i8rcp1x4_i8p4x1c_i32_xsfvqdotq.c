// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_i8rcprc_i8rcprc_i32rcprc.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_xsfvqdotq)
#error This file requires the Xsfvqdotq extension
#endif

/**
 * @brief Test cases for the skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq kernel.
 *
 * This test uses the gemm_i8rcprc_i8rcprc_i32rcprc harness with the following
 * restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 4
 *  - Matrix A has row-major blocks (csa0 = 1)
 *  - Matrix B is block-row major with column-major blocks (rsb0 = 1, csb1
 *    = k0 * n0)
 *  - Matrix C is row-major (csc1 = 1)
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
    {BENCH, .m1 =  126, .n1 = 128, .k1 = 64, .alpha = 1},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfvqdotq GEMM
    /* Zero-dimension edge cases */
    {TEST, .m1 = 0,   .n1 = 0,     .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 0,   .n1 = 0,     .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 0,   .n1 = 128,   .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 0,     .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 0,     .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 0,     .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 0,   .n1 = 128,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 0,     .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 0,     .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 0,     .k1 = 4,  .alpha = 1},
    /* Test tile functions (m == 1, m < 6, m == 6) */
    {TEST, .m1 = 1,   .n1 = 128,   .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 128,   .k1 = 1,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 128,   .k1 = 2,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 128,   .k1 = 3,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 128,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 1,   .n1 = 128,   .k1 = 4,  .alpha = 2,  .beta = 3},

    {TEST, .m1 = 5,   .n1 = 128,   .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 128,   .k1 = 1,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 128,   .k1 = 2,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 128,   .k1 = 3,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 128,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 5,   .n1 = 128,   .k1 = 4,  .alpha = 2,  .beta = 3},

    {TEST, .m1 = 6,   .n1 = 128,   .k1 = 0,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 128,   .k1 = 1,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 128,   .k1 = 2,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 128,   .k1 = 3,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 128,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 6,   .n1 = 128,   .k1 = 4,  .alpha = 2,  .beta = 3},
    /* Multiple tiles */
    {TEST, .m1 = 29,  .n1 = 64,    .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 29,  .n1 = 127,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 29,  .n1 = 129,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 29,  .n1 = 192,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 29,  .n1 = 255,   .k1 = 4,  .alpha = 1},
    {TEST, .m1 = 29,  .n1 = 255,   .k1 = 4,  .alpha = 2,  .beta = 3},
    /* Block-column-major A */
    {TEST, .m1 = 29,  .n1 = 128,   .k1 = 4,  .alpha = 1,  .rsa1 = 4,
           .csa1 = (size_t)(29 * 4)},
    /* A with misaligned rows */
    {TEST, .m1 = 29,  .n1 = 128,   .k1 = 4,  .alpha = 1,
           .rsa1 = (size_t)(4 * 4 + 1),  .csa1 = 4},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {
    .name = "skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq",
    .num_tests = sizeof(tests) / sizeof(tests[0]),
    .test_size = sizeof(gemm_i8rcprc_i8rcprc_i32rcprc_t),
    .tests = tests};

static void init(skl_test_t *t) {
  const gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  SKL_TEST_REQUIRE(t, init_status, h->m0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->k0 == 4);
  SKL_TEST_REQUIRE(t, init_status, h->csa0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rsb0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->csb1 == h->k0 * h->n0);
  SKL_TEST_REQUIRE(t, init_status, h->csc1 == 1);

  gemm_i8rcprc_i8rcprc_i32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  skl_gemm_i8rcp1x4_i8p4x1c_i32_xsfvqdotq(
      h->m1, h->n1, h->k1, h->alpha, h->a.data, h->rsa1, h->csa1, h->b.data,
      h->rsb1, h->beta, h->c.data, h->rsc1);
}

int main(void) {
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = 1;
    tests[i].n0 = 1;
    tests[i].k0 = 4;

    tests[i].rsa0 = 4;
    tests[i].csa0 = 1;
    tests[i].csa1 = tests[i].csa1 ? tests[i].csa1 : tests[i].m0 * tests[i].k0;
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : tests[i].k1 * tests[i].csa1;

    tests[i].rsb0 = 1;
    tests[i].csb0 = 4;
    tests[i].csb1 = tests[i].k0 * tests[i].n0;
    tests[i].rsb1 = tests[i].rsb1 ? tests[i].rsb1 : tests[i].n1 * tests[i].csb1;

    tests[i].rsc0 = 1;
    tests[i].csc0 = 1;
    tests[i].rsc1 = tests[i].rsc1 ? tests[i].rsc1 : tests[i].n1;
    tests[i].csc1 = 1;
  }

  return skl_test_driver_run_suite(&suite);
}
