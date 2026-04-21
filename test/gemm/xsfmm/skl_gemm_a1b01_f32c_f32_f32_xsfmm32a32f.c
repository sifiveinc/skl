// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gemm/gemm_f32rcprc_f32rcprc_f32rcprc.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stdbool.h>
#include <stddef.h>

#if !defined(__riscv_xsfmm32a32f)
#error This file requires the Xsfmm32a32f extension
#endif

/**
 * @brief Test cases for GEMM with Xsfmm32a32f extension.
 *
 * This test uses the gemm_f32rcprc_f32rcprc_f32rcprc harness with the following
 * restrictions on the input parameters:
 *  - The block dimensions are m0 = 1, n0 = 1, and k0 = 1
 *  - Matrix A is column-major (rsa1 == 1)
 *  - Matrix B is row-major (csb1 == 1)
 *  - Matrix C is row-major (csc1 == 1)
 *  - Alpha must be 1.0
 *  - Beta must be 0.0 or 1.0
 *
 * The kernel computes C = A * B (beta = 0) or C += A * B (beta = 1).
 */

#define TEST                                                                   \
  GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps = {                                                               \
          .init = gemm_f32rcprc_f32rcprc_f32rcprc_init,                        \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_f32rcprc_f32rcprc_f32rcprc_verify,                    \
          .report = NULL,                                                      \
          .cleanup = gemm_f32rcprc_f32rcprc_f32rcprc_cleanup,                  \
  }
#define BENCH                                                                  \
  GEMM_F32RCPRC_F32RCPRC_F32RCPRC_DEFAULTS,                                    \
      .steps = {                                                               \
          .init = gemm_f32rcprc_f32rcprc_f32rcprc_init,                        \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_f32rcprc_f32rcprc_f32rcprc_report,                    \
          .cleanup = gemm_f32rcprc_f32rcprc_f32rcprc_cleanup,                  \
  }
static void execute(skl_test_t *t);

// clang-format off
gemm_f32rcprc_f32rcprc_f32rcprc_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH, .m1 = 128, .n1 = 128, .k1 = 2048, .alpha = 1.f, .beta = 0.f},
    {BENCH, .m1 = 128, .n1 = 128, .k1 = 2048, .alpha = 1.f, .beta = 1.f},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfmm A1B01 layout (ETE=64)
    /* Edge case: 1x1 matrix with k=0 (no computation, C = beta * C) */
    {TEST, .m1 = 1,   .n1 = 1,   .k1 = 0, .alpha = 1.f},
    /* Edge case: 1x1 matrix with k=1 (minimal computation) */
    {TEST, .m1 = 1,   .n1 = 1,   .k1 = 1, .alpha = 1.f},
    /* ETE-1 boundary: 63 = 64-1, tests just below tile edge */
    {TEST, .m1 = 63,  .n1 = 63,  .k1 = 1, .alpha = 1.f},
    {TEST, .m1 = 63,  .n1 = 63,  .k1 = 2, .alpha = 1.f},
    {TEST, .m1 = 63,  .n1 = 63,  .k1 = 5, .alpha = 1.f},
    /* Exact ETE boundary: 64x64 tiles */
    {TEST, .m1 = 64,  .n1 = 64,  .k1 = 1, .alpha = 1.f},
    {TEST, .m1 = 64,  .n1 = 64,  .k1 = 5, .alpha = 1.f},
    /* ETE+1 boundary: 65 = 64+1, tests just past tile edge */
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 1, .alpha = 1.f},
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 5, .alpha = 1.f},
    /* Multi-tile: 2*ETE = 128 */
    {TEST, .m1 = 128, .n1 = 128, .k1 = 1, .alpha = 1.f},
    {TEST, .m1 = 128, .n1 = 128, .k1 = 5, .alpha = 1.f},
    /* Multi-tile+1: 2*ETE+1 = 129 */
    {TEST, .m1 = 129, .n1 = 129, .k1 = 1, .alpha = 1.f},
    {TEST, .m1 = 129, .n1 = 129, .k1 = 5, .alpha = 1.f},
    /* Non-square matrices (rectangular tiles) */
    {TEST, .m1 = 129, .n1 = 63,  .k1 = 1, .alpha = 1.f},
    {TEST, .m1 = 65,  .n1 = 129, .k1 = 2, .alpha = 1.f},
    {TEST, .m1 = 64,  .n1 = 128, .k1 = 5, .alpha = 1.f},
    /* Beta=1 tests (accumulate into existing C) */
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 0,  .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 1,  .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 65,  .n1 = 65,  .k1 = 33, .alpha = 1.f, .beta = 1.f},
    {TEST, .m1 = 128, .n1 = 128, .k1 = 33, .alpha = 1.f, .beta = 1.f},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {
    .name = "skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f",
    .num_tests = sizeof(tests) / sizeof(tests[0]),
    .test_size = sizeof(gemm_f32rcprc_f32rcprc_f32rcprc_t),
    .tests = tests};

static void execute(skl_test_t *t) {
  const gemm_f32rcprc_f32rcprc_f32rcprc_t *h =
      (gemm_f32rcprc_f32rcprc_f32rcprc_t *)t->harness;

  SKL_TEST_REQUIRE(t, execute_status, h->rsa1 == 1); // Note: column-major
  SKL_TEST_REQUIRE(t, execute_status, h->csb1 == 1);
  SKL_TEST_REQUIRE(t, execute_status, h->csc1 == 1);
  SKL_TEST_REQUIRE(t, execute_status, h->alpha == 1.f);
  SKL_TEST_REQUIRE(t, execute_status, h->beta == 0.f || h->beta == 1.f);

  // Call the kernel with the appropriate parameters
  // The kernel signature is: (m, n, k, a, csa, b, rsb, c, rsc, accum)
  // where accum = (beta != 0)
  skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f(
      h->m1, h->n1, h->k1, h->a_pack.data, h->csa1, h->b_pack.data, h->rsb1,
      h->c_pack.data, h->rsc1, h->beta != 0.f);
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
