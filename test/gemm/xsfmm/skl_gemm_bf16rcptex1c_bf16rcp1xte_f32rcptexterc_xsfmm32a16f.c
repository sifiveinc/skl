// Copyright (c) 2026 SiFive, Inc. All rights reserved.
// Licensed under the MIT License.
// See LICENSE file in the project root for full license information.
// SPDX-License-Identifier: MIT

#include "gemm/gemm_bf16rcprc_bf16rcprc_f32rcprc.h"
#include "gemm/skl_test_gemm.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_xsfmm32a16f)
#error This file requires the Xsfmm32a16f extension
#endif

/**
 * @brief Test cases for the
 * skl_gemm_bf16rcptex1c_bf16rcp1xte_f32rcptexterc_xsfmm32a16f kernel.
 *
 * This test uses the gemm_bf16rcprc_bf16rcprc_f32rcprc harness with the
 * following restrictions on the input parameters:
 *  - The block dimensions are m0 = TE, n0 = TE, and k0 = 1
 *  - Matrix A has column-major blocks (rsa0 == 1)
 *  - Matrix B has row-major blocks (csb0 == 1)
 */

#define TEST                                                                   \
  GEMM_BF16RCPRC_BF16RCPRC_F32RCPRC_DEFAULTS,                                  \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_bf16rcprc_bf16rcprc_f32rcprc_verify,                  \
          .report = gemm_bf16rcprc_bf16rcprc_f32rcprc_test_report,             \
          .cleanup = gemm_bf16rcprc_bf16rcprc_f32rcprc_cleanup,                \
  }

#define BENCH                                                                  \
  GEMM_BF16RCPRC_BF16RCPRC_F32RCPRC_DEFAULTS,                                  \
      .steps = {                                                               \
          .init = init,                                                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_bf16rcprc_bf16rcprc_f32rcprc_benchmark_report,        \
          .cleanup = gemm_bf16rcprc_bf16rcprc_f32rcprc_cleanup,                \
  }

static void init(skl_test_t *t);
static void execute(skl_test_t *t);

// clang-format off
gemm_bf16rcprc_bf16rcprc_f32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH, .m1 = 2, .n1 = 2, .k1 = 4096, .alpha = 1.f, .beta = 0.f},
    {BENCH, .m1 = 2, .n1 = 2, .k1 = 4096, .alpha = 1.f, .beta = 1.f},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfmm
    {TEST, .m1 = 7, .n1 = 1, .k1 = 0, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 0, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 0, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 0, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 0, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 0, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 0, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 1, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 1, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 1, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 1, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 1, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 1, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 1, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 2, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 2, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 2, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 2, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 2, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 2, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 2, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 3, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 3, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 3, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 3, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 3, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 3, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 3, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 4, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 4, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 4, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 4, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 4, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 4, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 4, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 5, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 5, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 5, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 5, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 5, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 5, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 5, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 1, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 1, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 1, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 1, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 1, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 1, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 1, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 2, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 2, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 2, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 2, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 2, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 2, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 2, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 3, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 3, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 3, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 3, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 3, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 3, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 3, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 4, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 4, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 4, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 4, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 4, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 4, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 4, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 5, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 5, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 5, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 5, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 5, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 5, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 5, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 6, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 6, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 6, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 6, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 6, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 6, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 6, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 15, .alpha = 1.f, .beta = 0.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 15, .alpha = 1.f, .beta = 0.f},

    {TEST, .m1 = 7, .n1 = 1, .k1 = 15, .alpha = 2.f, .beta = 3.f},
    {TEST, .m1 = 7, .n1 = 2, .k1 = 15, .alpha = 2.f, .beta = 3.f},
    {TEST, .m1 = 7, .n1 = 3, .k1 = 15, .alpha = 2.f, .beta = 3.f},
    {TEST, .m1 = 7, .n1 = 4, .k1 = 15, .alpha = 2.f, .beta = 3.f},
    {TEST, .m1 = 7, .n1 = 5, .k1 = 15, .alpha = 2.f, .beta = 3.f},
    {TEST, .m1 = 7, .n1 = 6, .k1 = 15, .alpha = 2.f, .beta = 3.f},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 15, .alpha = 2.f, .beta = 3.f},

    {TEST, .m1 = 7, .n1 = 7, .k1 = 15, .alpha = 2.f, .beta = 3.f, .csc0 = 2},
    {TEST, .m1 = 7, .n1 = 7, .k1 = 15, .alpha = 2.f, .beta = 3.f, .rsc0 = 1,
           .csc0 = SKL_XSFMM_TE},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {
    .name = "skl_gemm_bf16rcptex1c_bf16rcp1xte_f32rcptexterc_xsfmm32a16f",
    .num_tests = sizeof(tests) / sizeof(tests[0]),
    .test_size = sizeof(gemm_bf16rcprc_bf16rcprc_f32rcprc_t),
    .tests = tests};

static void init(skl_test_t *t) {
  const gemm_bf16rcprc_bf16rcprc_f32rcprc_t *h =
      (gemm_bf16rcprc_bf16rcprc_f32rcprc_t *)t->harness;

  SKL_TEST_REQUIRE(t, init_status, h->m0 == SKL_XSFMM_TE);
  SKL_TEST_REQUIRE(t, init_status, h->n0 == SKL_XSFMM_TE);
  SKL_TEST_REQUIRE(t, init_status, h->k0 == 1);
  SKL_TEST_REQUIRE(t, init_status, h->rsa0 == 1); // Note: column-major
  SKL_TEST_REQUIRE(t, init_status, h->csb0 == 1);

  gemm_bf16rcprc_bf16rcprc_f32rcprc_init(t);
}

static void execute(skl_test_t *t) {
  const gemm_bf16rcprc_bf16rcprc_f32rcprc_t *h =
      (gemm_bf16rcprc_bf16rcprc_f32rcprc_t *)t->harness;

  skl_gemm_bf16rcptex1c_bf16rcp1xte_f32rcptexterc_xsfmm32a16f(
      h->m1, h->n1, h->k1, h->alpha, h->a.data, h->rsa1, h->csa1, h->b.data,
      h->rsb1, h->csb1, h->beta, h->c.data, h->rsc0, h->csc0, h->rsc1, h->csc1);
}

int main(void) {
  // Set default strides: A has column-major blocks, B has row-major blocks
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].m0 = SKL_XSFMM_TE;
    tests[i].n0 = SKL_XSFMM_TE;
    tests[i].k0 = 1;

    tests[i].rsa0 = 1;
    tests[i].csa0 = tests[i].m0;
    tests[i].csa1 = tests[i].csa1 ? tests[i].csa1 : tests[i].m0 * tests[i].k0;
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : tests[i].k1 * tests[i].csa1;

    tests[i].rsb0 = tests[i].n0;
    tests[i].csb0 = 1;
    tests[i].rsb1 = tests[i].rsb1 ? tests[i].rsb1 : tests[i].k0 * tests[i].n0;
    tests[i].csb1 = tests[i].csb1 ? tests[i].csb1 : tests[i].k1 * tests[i].rsb1;

    tests[i].csc0 = tests[i].csc0 ? tests[i].csc0 : 1;
    tests[i].rsc0 = tests[i].rsc0 ? tests[i].rsc0 : tests[i].n0 * tests[i].csc0;
    tests[i].csc1 = tests[i].csc1 ? tests[i].csc1
                                  : (tests[i].m0 - 1) * tests[i].rsc0 +
                                        (tests[i].n0 - 1) * tests[i].csc0 + 1;
    tests[i].rsc1 = tests[i].rsc1 ? tests[i].rsc1 : tests[i].n1 * tests[i].csc1;
  }

  return skl_test_driver_run_suite(&suite);
}
