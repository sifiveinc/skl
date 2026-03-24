// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gemm/gemm_f32rc_f32rc_f32rc.h"
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
 * This test uses the gemm_f32rc_f32rc_f32rc harness with the following
 * restrictions on the input parameters:
 *  - Matrix A is column-major (rsa == 1)
 *  - Matrix B is row-major (csb == 1)
 *  - Matrix C is row-major (csc == 1)
 *  - Alpha must be 1.0
 *  - Beta must be 0.0 or 1.0
 *
 * The kernel computes C = A * B (beta=0) or C += A * B (beta=1).
 */

#define TEST                                                                   \
  GEMM_F32RC_F32RC_F32RC_DEFAULTS,                                             \
      .steps = {                                                               \
          .init = gemm_f32rc_f32rc_f32rc_init,                                 \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_f32rc_f32rc_f32rc_verify,                             \
          .report = NULL,                                                      \
          .cleanup = gemm_f32rc_f32rc_f32rc_cleanup,                           \
  }
#define BENCH                                                                  \
  GEMM_F32RC_F32RC_F32RC_DEFAULTS,                                             \
      .steps = {                                                               \
          .init = gemm_f32rc_f32rc_f32rc_init,                                 \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_f32rc_f32rc_f32rc_report,                             \
          .cleanup = gemm_f32rc_f32rc_f32rc_cleanup,                           \
  }
static int execute(skl_test_t *t);
static int steps(skl_test_t *t);

// clang-format off
gemm_f32rc_f32rc_f32rc_t tests[] = {
    // Benchmark tests
    {BENCH, .m = 128, .n = 128, .k = 2048, .beta = 0.f},
    {BENCH, .m = 128, .n = 128, .k = 2048, .beta = 1.f},

    // Verification tests - comprehensive coverage for Xsfmm A1B01 layout (TE=64)
    /* Edge case: 1x1 matrix with k=0 (no computation, C = beta*C) */
    {TEST, .rsa = 1, .m = 1,   .n = 1,   .k = 0},
    /* Edge case: 1x1 matrix with k=1 (minimal computation) */
    {TEST, .rsa = 1, .m = 1,   .n = 1,   .k = 1},
    /* TE-1 boundary: 63 = 64-1, tests just below tile edge */
    {TEST, .rsa = 1, .m = 63,  .n = 63,  .k = 1},
    {TEST, .rsa = 1, .m = 63,  .n = 63,  .k = 2},
    {TEST, .rsa = 1, .m = 63,  .n = 63,  .k = 5},
    /* Exact TE boundary: 64x64 tiles */
    {TEST, .rsa = 1, .m = 64,  .n = 64,  .k = 1},
    {TEST, .rsa = 1, .m = 64,  .n = 64,  .k = 5},
    /* TE+1 boundary: 65 = 64+1, tests just past tile edge */
    {TEST, .rsa = 1, .m = 65,  .n = 65,  .k = 1},
    {TEST, .rsa = 1, .m = 65,  .n = 65,  .k = 5},
    /* Multi-tile: 2*TE = 128 */
    {TEST, .rsa = 1, .m = 128, .n = 128, .k = 1},
    {TEST, .rsa = 1, .m = 128, .n = 128, .k = 5},
    /* Multi-tile+1: 2*TE+1 = 129 */
    {TEST, .rsa = 1, .m = 129, .n = 129, .k = 1},
    {TEST, .rsa = 1, .m = 129, .n = 129, .k = 5},
    /* Non-square matrices (rectangular tiles) */
    {TEST, .rsa = 1, .m = 129, .n = 63,  .k = 1},
    {TEST, .rsa = 1, .m = 65,  .n = 129, .k = 2},
    {TEST, .rsa = 1, .m = 64,  .n = 128, .k = 5},
    /* Beta=1 tests (accumulate into existing C) */
    {TEST, .rsa = 1, .m = 65,  .n = 65,  .k = 0,  .beta = 1.f},
    {TEST, .rsa = 1, .m = 65,  .n = 65,  .k = 1,  .beta = 1.f},
    {TEST, .rsa = 1, .m = 65,  .n = 65,  .k = 33, .beta = 1.f},
    {TEST, .rsa = 1, .m = 128, .n = 128, .k = 33, .beta = 1.f},
};
// clang-format on

static skl_test_suite_t suite = {.name =
                                     "skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size = sizeof(gemm_f32rc_f32rc_f32rc_t),
                                 .tests = tests,
                                 .steps = steps};

static int steps(skl_test_t *t) {
  const gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;
  t->steps = &h->steps;
  return 0; // Success
}

static int execute(skl_test_t *t) {
  const gemm_f32rc_f32rc_f32rc_t *h = (gemm_f32rc_f32rc_f32rc_t *)t->harness;

  SKL_TEST_REQUIRE(t, h->rsa == 1); // Note: column-major
  SKL_TEST_REQUIRE(t, h->csb == 1);
  SKL_TEST_REQUIRE(t, h->csc == 1);
  SKL_TEST_REQUIRE(t, h->alpha == 1.f);
  SKL_TEST_REQUIRE(t, h->beta == 0.f || h->beta == 1.f);

  // Call the kernel with the appropriate parameters
  // The kernel signature is: (m, n, k, a, csa, b, rsb, c, rsc, accum)
  // where accum = (beta != 0)
  skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f(h->m, h->n, h->k, h->a.data, h->csa,
                                          h->b.data, h->rsb, h->c.data, h->rsc,
                                          h->beta != 0.f);
  return (t->status == SKL_TEST_PASS) ? 0 : 1;
}

int main(void) {
  // Set default strides: A is column-major, B is row-major, C is row-major
  for (size_t i = 0; i < suite.num_tests; i++) {
    tests[i].csa = tests[i].csa ? tests[i].csa : tests[i].m;
    tests[i].rsb = tests[i].rsb ? tests[i].rsb : tests[i].n;
    tests[i].rsc = tests[i].rsc ? tests[i].rsc : tests[i].n;
    // Make sure we still generate an error if a csc/csb/rsa is set accidentally
    tests[i].csc = tests[i].csc ? tests[i].csc : 1;
    tests[i].csb = tests[i].csb ? tests[i].csb : 1;
    tests[i].rsa = tests[i].rsa ? tests[i].rsa : 1;
  }

  return skl_test_driver_run_suite(&suite);
}
