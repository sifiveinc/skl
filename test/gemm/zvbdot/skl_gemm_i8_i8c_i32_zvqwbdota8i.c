// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_i8rcprc_i8rcprc_i32rcprc.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_zvqwbdota8i)
#error This file requires the Zvqwbdota8i extension
#endif

/**
 * @brief Test cases for GEMM with Zvqwbdota8i extension.
 *
 * This test uses the gemm_i8rcprc_i8rcprc_i32rcprc harness with the following
 * restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 1
 *  - Matrix A is row-major (csa1 = 1)
 *  - Matrix B is column-major (rsb1 = 1)
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
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests
    {TEST, .m1 = 0, .n1 = 0, .k1 = 67},
    {TEST, .m1 = 0, .n1 = 1, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 0, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 8, .k1 = 0},
    {TEST, .m1 = 2, .n1 = 8, .k1 = 0},

    {TEST, .m1 = 1, .n1 = 1, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 2, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 3, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 4, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 5, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 6, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 7, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 8, .k1 = 67},
    {TEST, .m1 = 1, .n1 = 15, .k1 = 67},

    {TEST, .m1 = 2, .n1 = 1, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 2, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 3, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 4, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 5, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 6, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 7, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 8, .k1 = 67},
    {TEST, .m1 = 2, .n1 = 15, .k1 = 67},

    {TEST, .m1 = 3, .n1 = 1, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 2, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 3, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 4, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 5, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 6, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 7, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 8, .k1 = 67},
    {TEST, .m1 = 3, .n1 = 15, .k1 = 67},

    {TEST, .m1 = 65, .n1 = 65, .k1 = 255},
    {TEST, .m1 = 65, .n1 = 65, .k1 = 255, .alpha = 2, .beta = 3},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_gemm_i8_i8c_i32_zvqwbdota8i",
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
  SKL_TEST_REQUIRE(t, init_status, h->csa1 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rsb1 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->csc1 == 1);

  gemm_i8rcprc_i8rcprc_i32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_i8rcprc_i8rcprc_i32rcprc_t *h =
      (gemm_i8rcprc_i8rcprc_i32rcprc_t *)t->harness;

  skl_gemm_i8_i8c_i32_zvqwbdota8i(h->m1, h->n1, h->k1, h->alpha, h->a_pack.data,
                                  h->rsa1, h->b_pack.data, h->csb1, h->beta,
                                  h->c_pack.data, h->rsc1);
}

int main(void) {
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = 1;
    tests[i].n0 = 1;
    tests[i].k0 = 1;

    tests[i].rsa0 = 1;
    tests[i].csa0 = 1;
    tests[i].csa1 = 1;
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : tests[i].k1;

    tests[i].rsb0 = 1;
    tests[i].csb0 = 1;
    tests[i].rsb1 = 1;
    tests[i].csb1 = tests[i].csb1 ? tests[i].csb1 : tests[i].k1;

    tests[i].rsc0 = 1;
    tests[i].csc0 = 1;
    tests[i].csc1 = 1;
    tests[i].rsc1 = tests[i].rsc1 ? tests[i].rsc1 : tests[i].n1;
  }

  return skl_test_driver_run_suite(&suite);
}
