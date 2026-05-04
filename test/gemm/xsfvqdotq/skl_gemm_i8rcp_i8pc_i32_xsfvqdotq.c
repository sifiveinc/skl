// Copyright 2026 SiFive, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gemm/gemm_i8rcp_i8pc_i32_xsfvqdotq.h"
#include "skl-test-driver.h"
#include "skl.h"
#include <stddef.h>

#if !defined(__riscv_xsfvqdotq)
#error This file requires the Xsfvqdotq extension
#endif

/**
 * @brief Test cases for GEMM with Xsfvqdotq extension.
 *
 * This test uses the gemm_i8rcp_i8pc_i32_xsfvqdotq harness.
 */

#define TEST                                                                   \
  GEMM_I8RCP_I8PC_I32_XSFVQDOTQ_DEFAULTS,                                      \
      .steps = {                                                               \
          .init = gemm_i8rcp_i8pc_i32_xsfvqdotq_init,                          \
          .warmup = NULL,                                                      \
          .execute = execute,                                                  \
          .verify = gemm_i8rcp_i8pc_i32_xsfvqdotq_verify,                      \
          .report = gemm_i8rcp_i8pc_i32_xsfvqdotq_test_report,                 \
          .cleanup = gemm_i8rcp_i8pc_i32_xsfvqdotq_cleanup,                    \
  }
#define BENCH                                                                  \
  GEMM_I8RCP_I8PC_I32_XSFVQDOTQ_DEFAULTS,                                      \
      .steps = {                                                               \
          .init = gemm_i8rcp_i8pc_i32_xsfvqdotq_init,                          \
          .warmup = execute,                                                   \
          .execute = execute,                                                  \
          .verify = NULL,                                                      \
          .report = gemm_i8rcp_i8pc_i32_xsfvqdotq_benchmark_report,            \
          .cleanup = gemm_i8rcp_i8pc_i32_xsfvqdotq_cleanup,                    \
  }

static void execute(skl_test_t *t);

// clang-format off
gemm_i8rcp_i8pc_i32_xsfvqdotq_t tests[] = {
#ifdef SKL_ENABLE_BENCHMARKS
    // Benchmark tests
    {BENCH, .m =  126, .n = 128, .k = 256, .alpha = 1},
#endif // SKL_ENABLE_BENCHMARKS

#ifdef SKL_ENABLE_TESTS
    // Verification tests - comprehensive coverage for Xsfvqdotq GEMM
    {TEST, .m =  1,  .n =  32,  .k = 31,  .alpha = 1},
    {TEST, .m =  1,  .n = 128,  .k = 31,  .alpha = 1},
    {TEST, .m =  1,  .n = 192,  .k = 31,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k =  0,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k =  1,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k =  2,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k =  3,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k =  4,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k =  8,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k = 12,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k = 16,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k = 20,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k = 21,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k = 22,  .alpha = 1},
    {TEST, .m = 32,  .n = 128,  .k = 23,  .alpha = 1},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 1},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 1,  .beta = 1},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 1,  .beta = 2},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 0},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 0,  .beta = 1},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 0,  .beta = 2},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 2},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 2,  .beta = 1},
    {TEST, .m = 32,  .n = 192,  .k = 32,  .alpha = 2,  .beta = 2},
    {TEST, .m = 32,  .n = 192,  .k = 31,  .rsa1 = 31,  .csa1 = 4,  .alpha = 1,  .beta = 1},
    {TEST, .m = 32,  .n = 192,  .k = 31,  .rsa1 = 4,  .csa1 = (size_t)32 * (size_t)4,  .alpha = 1,  .beta = 1},
#endif // SKL_ENABLE_TESTS
};
// clang-format on

static skl_test_suite_t suite = {.name = "skl_gemm_i8rcp_i8pc_i32_xsfvqdotq",
                                 .num_tests = sizeof(tests) / sizeof(tests[0]),
                                 .test_size =
                                     sizeof(gemm_i8rcp_i8pc_i32_xsfvqdotq_t),
                                 .tests = tests};

static void execute(skl_test_t *t) {
  const gemm_i8rcp_i8pc_i32_xsfvqdotq_t *h =
      (gemm_i8rcp_i8pc_i32_xsfvqdotq_t *)t->harness;

  skl_gemm_i8rcp_i8pc_i32_xsfvqdotq(h->m, h->n, h->k, h->alpha, h->a_pack.data,
                                    h->rsa1, h->csa1, h->b_pack.data, h->rsb1,
                                    h->beta, h->c.data, h->rsc);
}

int main(void) {
  // Set default strides
  for (size_t i = 0; i < suite.num_tests; ++i) {
    tests[i].rsa1 = tests[i].rsa1 ? tests[i].rsa1 : (tests[i].k + 3) / 4 * 4;
    tests[i].csa1 = tests[i].csa1 ? tests[i].csa1 : 4;
    tests[i].rsb1 = tests[i].rsb1 ? tests[i].rsb1 : tests[i].n * 4;
    tests[i].rsc = tests[i].rsc ? tests[i].rsc : tests[i].n;
  }

  return skl_test_driver_run_suite(&suite);
}
